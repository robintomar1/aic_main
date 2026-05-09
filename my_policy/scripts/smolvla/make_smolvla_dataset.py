#!/usr/bin/env python3
"""Build a SmolVLA-friendly variant of the port-local dataset by dropping
the 6-dim tcp_error block AND the 12-dim task one-hot from observation.state
(44 → 26 channels).

Why drop tcp_error: it is auto-regressive on the policy's own past output
(`target_pose - current_pose` from the live controller), and the
phase-trigger probe identified state[15] = tcp_error.z as the channel the
trained model used as a hover-vs-commit shortcut. Removing it forces the
model to rely on the visual / spatial / wrench channels.

Why drop the task one-hot: SmolVLA gets task identity via the
natural-language `tasks` string per episode.

Source 44-dim port-local layout (from make_port_local_dataset.py):
  [ 0..6 ] tcp_pose       — port frame                 (kept)
  [ 7..12] tcp_velocity   — port frame                 (kept)
  [13..18] tcp_error      — base_link frame            (DROPPED)
  [19..25] joint_positions                             (kept)
  [26..31] wrench         — port frame                 (kept)
  [32..43] task one-hot                                (DROPPED)

Resulting 26-dim layout: tcp_pose ++ tcp_velocity ++ joints ++ wrench.

What this script does:
  * Read v9_port_local_merged_clean (or any 44-dim port-local dataset).
  * Slice observation.state down to the 26 retained channels.
  * Recompute per-episode stats and aggregate stats.json for state
    (action / videos unchanged).
  * Update info.json: observation.state.shape = [26], names filtered,
    frame_transform = "port_local_smolvla_no_err".
  * Copy meta/tasks.parquet, train/val splits, source_episode_map.json
    (if present) verbatim. Symlink videos/.

Pure pyarrow + numpy. No torch / lerobot dependency.

Usage:
    python3 my_policy/scripts/smolvla/make_smolvla_dataset.py \\
        --src /root/aic_data/v9_act_build/v9_port_local_merged_clean \\
        --dst /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SRC_DIM = 44                # 44-dim port-local source
KEPT_CHANNELS = 26          # 7 tcp_pose + 6 tcp_velocity + 7 joints + 6 wrench
KEEP_SRC_INDICES = list(range(0, 13)) + list(range(19, 32))  # drop [13:19]+[32:44]
assert len(KEEP_SRC_INDICES) == KEPT_CHANNELS


def per_episode_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "min": arr.min(axis=0).astype(np.float32),
        "max": arr.max(axis=0).astype(np.float32),
        "mean": arr.mean(axis=0).astype(np.float32),
        "std": arr.std(axis=0).astype(np.float32),
        "count": np.array([arr.shape[0]], dtype=np.int64),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float32),
        "q10": np.quantile(arr, 0.10, axis=0).astype(np.float32),
        "q50": np.quantile(arr, 0.50, axis=0).astype(np.float32),
        "q90": np.quantile(arr, 0.90, axis=0).astype(np.float32),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float32),
    }


def aggregate_stats_from_per_ep(per_ep: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for feat in per_ep[0].keys():
        feat_stats_list = [p[feat] for p in per_ep]
        counts = np.array([s["count"] for s in feat_stats_list]).flatten()
        total = counts.sum()
        mins = np.stack([s["min"] for s in feat_stats_list])
        maxs = np.stack([s["max"] for s in feat_stats_list])
        means = np.stack([s["mean"] for s in feat_stats_list])
        agg_mean = (means * counts[:, None]).sum(axis=0) / total
        stds = np.stack([s["std"] for s in feat_stats_list])
        var_within = (counts[:, None] * stds**2).sum(axis=0) / total
        var_between = (counts[:, None] * (means - agg_mean)**2).sum(axis=0) / total
        agg_std = np.sqrt(var_within + var_between)
        agg_q = {}
        for qk in ("q01", "q10", "q50", "q90", "q99"):
            qs = np.stack([s[qk] for s in feat_stats_list])
            agg_q[qk] = (qs * counts[:, None]).sum(axis=0) / total
        out[feat] = {
            "min": mins.min(axis=0).astype(np.float32),
            "max": maxs.max(axis=0).astype(np.float32),
            "mean": agg_mean.astype(np.float32),
            "std": agg_std.astype(np.float32),
            "count": np.array([int(total)], dtype=np.int64),
            **{k: v.astype(np.float32) for k, v in agg_q.items()},
        }
    return out


def stats_to_json(stats: dict[str, dict]) -> dict:
    return {
        feat: {k: v.tolist() for k, v in feat_stats.items()}
        for feat, feat_stats in stats.items()
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="Source dataset root (44-dim port-local).")
    p.add_argument("--dst", type=Path, required=True,
                   help="Destination dataset root (will be created).")
    p.add_argument("--force", action="store_true",
                   help="Delete dst if it exists.")
    args = p.parse_args()

    if not args.src.exists():
        print(f"error: source {args.src} does not exist", file=sys.stderr)
        return 1
    if args.dst.exists():
        if args.force:
            print(f"--force: removing existing {args.dst}")
            shutil.rmtree(args.dst)
        else:
            print(f"error: destination {args.dst} already exists "
                  f"(pass --force to overwrite)", file=sys.stderr)
            return 1

    print(f"=== {args.src} -> {args.dst} (drop tcp_error + task one-hot, 44 → {KEPT_CHANNELS}) ===")
    args.dst.mkdir(parents=True)
    (args.dst / "data" / "chunk-000").mkdir(parents=True)
    (args.dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    # --- Read source data parquet ------------------------------------------
    src_data_path = args.src / "data" / "chunk-000" / "file-000.parquet"
    src_table = pq.read_table(src_data_path)
    n_rows = src_table.num_rows
    print(f"  rows: {n_rows}, cols: {len(src_table.column_names)}")

    state = np.stack(src_table.column("observation.state").to_pylist()).astype(np.float32)
    if state.shape[1] != SRC_DIM:
        print(f"error: source state has {state.shape[1]} channels, "
              f"expected {SRC_DIM} (port-local with task one-hot). "
              f"Run make_port_local_dataset.py first.", file=sys.stderr)
        return 1
    print(f"  selecting {KEPT_CHANNELS} of {SRC_DIM} channels — "
          f"drops [13:19] tcp_error and [32:44] task one-hot")
    new_state = state[:, KEEP_SRC_INDICES].copy()

    ep_idx = np.array(src_table.column("episode_index").to_pylist(), dtype=np.int64)
    ep_starts = np.r_[0, np.where(np.diff(ep_idx) != 0)[0] + 1]
    ep_ends = np.r_[ep_starts[1:], n_rows]
    n_eps = len(ep_starts)
    print(f"  episodes: {n_eps}")

    # --- Write modified data parquet ---------------------------------------
    new_state_col = pa.array(
        new_state.tolist(), type=pa.list_(pa.float32(), KEPT_CHANNELS),
    )
    cols = {}
    for c in src_table.column_names:
        if c == "observation.state":
            cols[c] = new_state_col
        else:
            cols[c] = src_table.column(c)
    new_table = pa.Table.from_pydict(cols)
    pq.write_table(new_table, args.dst / "data" / "chunk-000" / "file-000.parquet")
    print(f"  wrote data/chunk-000/file-000.parquet")

    # --- Per-episode stats: state recomputed; action carried -------------
    src_eps_path = args.src / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    src_eps_table = pq.read_table(src_eps_path)
    n_eps_meta = src_eps_table.num_rows
    if n_eps_meta != n_eps:
        print(f"warning: data parquet has {n_eps} episodes but episode-meta "
              f"parquet has {n_eps_meta}", file=sys.stderr)

    new_state_per_ep: list[dict] = []
    for s, e in zip(ep_starts, ep_ends):
        new_state_per_ep.append(per_episode_stats(new_state[s:e]))

    new_eps_cols = {}
    for c in src_eps_table.column_names:
        if c.startswith("stats/observation.state/"):
            stat_name = c.split("/")[-1]
            new_eps_cols[c] = pa.array(
                [d[stat_name].tolist() for d in new_state_per_ep])
        else:
            new_eps_cols[c] = src_eps_table.column(c)
    new_eps_table = pa.Table.from_pydict(new_eps_cols)
    pq.write_table(new_eps_table,
                   args.dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    print(f"  wrote meta/episodes/chunk-000/file-000.parquet ({n_eps_meta} eps)")

    # --- Aggregate stats.json: state from new per-ep; everything else carried
    src_stats_path = args.src / "meta" / "stats.json"
    if not src_stats_path.exists():
        print(f"error: source meta/stats.json missing at {src_stats_path}; "
              f"required by lerobot make_dataset.", file=sys.stderr)
        return 1
    src_stats = json.loads(src_stats_path.read_text())

    state_agg = aggregate_stats_from_per_ep(
        [{"observation.state": d} for d in new_state_per_ep]
    )["observation.state"]
    new_stats = dict(src_stats)
    new_stats["observation.state"] = stats_to_json({"x": state_agg})["x"]
    (args.dst / "meta" / "stats.json").write_text(json.dumps(new_stats, indent=2))
    print(f"  wrote meta/stats.json ({len(new_stats)} feature keys)")

    # --- info.json ---------------------------------------------------------
    info = json.loads((args.src / "meta" / "info.json").read_text())
    new_info = dict(info)
    feats = dict(new_info["features"])
    state_feat = dict(feats["observation.state"])
    state_feat["shape"] = [KEPT_CHANNELS]
    src_names = state_feat["names"]
    if len(src_names) != SRC_DIM:
        print(f"warning: info.json names list has {len(src_names)} entries, "
              f"expected {SRC_DIM}", file=sys.stderr)
    state_feat["names"] = [src_names[i] for i in KEEP_SRC_INDICES
                           if i < len(src_names)]
    feats["observation.state"] = state_feat
    new_info["features"] = feats
    new_info["frame_transform"] = "port_local_smolvla_no_err"
    (args.dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))
    print(f"  wrote meta/info.json (state shape=[{KEPT_CHANNELS}], "
          f"frame_transform=port_local_smolvla_no_err)")

    # --- Copy small files; symlink videos ---------------------------------
    shutil.copy(args.src / "meta" / "tasks.parquet",
                args.dst / "meta" / "tasks.parquet")
    for sidecar in ("train_episodes.json", "val_episodes.json",
                    "source_episode_map.json"):
        sp = args.src / sidecar
        if sp.exists():
            shutil.copy(sp, args.dst / sidecar)
            print(f"  copied {sidecar}")
    src_videos = args.src / "videos"
    if src_videos.exists():
        target = src_videos.resolve() if src_videos.is_symlink() else src_videos
        (args.dst / "videos").symlink_to(target)
        print(f"  symlinked videos/ -> {target}")

    print()
    print(f"=== done: {args.dst} ===")
    print(f"  state shape: {SRC_DIM} -> {KEPT_CHANNELS}")
    print(f"  next: train_smolvla.py --dataset-root {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
