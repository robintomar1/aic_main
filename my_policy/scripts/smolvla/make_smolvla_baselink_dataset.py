#!/usr/bin/env python3
"""Build a 26-dim base_link SmolVLA dataset from the 44-dim cleaned ACT dataset.

Source 44-dim ACT base_link layout:
  [ 0.. 6] tcp_pose       — base_link frame (kept)
  [ 7..12] tcp_velocity   — base_link frame (kept)
  [13..18] tcp_error      — base_link frame (DROPPED)
  [19..25] joint_positions                  (kept)
  [26..31] wrench         — sensor frame    (kept)
  [32..43] task one-hot                     (DROPPED)

Resulting 26-dim layout: tcp_pose ++ tcp_velocity ++ joint_positions ++ wrench.
Actions stay as 7-dim absolute TCP poses in base_link (unchanged).

No port pose or TF lookup needed at inference — RunSmolVLA reads tcp_pose/vel/
joints/wrench directly from controller_state without any coordinate transform.

Usage:
    python3 my_policy/scripts/smolvla/make_smolvla_baselink_dataset.py \\
        --src /path/to/datasets/v9_act_merged_clean \\
        --dst /path/to/datasets/v9_smolvla_baselink \\
        [--val-fraction 0.2] [--seed 42] [--force]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SRC_DIM = 44
KEPT_CHANNELS = 26

# Drop tcp_error [13:19] and task one-hot [32:44].
KEEP_SRC_INDICES = list(range(0, 13)) + list(range(19, 32))
assert len(KEEP_SRC_INDICES) == KEPT_CHANNELS

KEPT_CHANNEL_NAMES = [
    "tcp_pose.position.x", "tcp_pose.position.y", "tcp_pose.position.z",
    "tcp_pose.orientation.x", "tcp_pose.orientation.y",
    "tcp_pose.orientation.z", "tcp_pose.orientation.w",
    "tcp_velocity.linear.x", "tcp_velocity.linear.y", "tcp_velocity.linear.z",
    "tcp_velocity.angular.x", "tcp_velocity.angular.y", "tcp_velocity.angular.z",
    "joint_positions.0", "joint_positions.1", "joint_positions.2",
    "joint_positions.3", "joint_positions.4", "joint_positions.5",
    "joint_positions.6",
    "wrench.fx", "wrench.fy", "wrench.fz",
    "wrench.tx", "wrench.ty", "wrench.tz",
]
assert len(KEPT_CHANNEL_NAMES) == KEPT_CHANNELS


# ---------------------------------------------------------------------------
# Stats helpers (mirrors make_smolvla_dataset.py verbatim)
# ---------------------------------------------------------------------------

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
        var_within = (counts[:, None] * stds ** 2).sum(axis=0) / total
        var_between = (counts[:, None] * (means - agg_mean) ** 2).sum(axis=0) / total
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="Source dataset root (44-dim ACT base_link, cleaned).")
    p.add_argument("--dst", type=Path, required=True,
                   help="Destination dataset root (will be created).")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Fraction of episodes to hold out for val (default 0.2).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/val split (default 42).")
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

    print(f"=== {args.src} -> {args.dst} "
          f"(drop tcp_error + task one-hot, 44 → {KEPT_CHANNELS}, base_link) ===")
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
              f"expected {SRC_DIM}. Point --src at a 44-dim ACT base_link dataset.",
              file=sys.stderr)
        return 1
    print(f"  keeping {KEPT_CHANNELS}/{SRC_DIM} channels "
          f"(drop tcp_error[13:19] and task_vec[32:44])")
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

    # --- Per-episode stats -------------------------------------------------
    src_eps_path = (args.src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    src_eps_table = pq.read_table(src_eps_path)

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
    pq.write_table(
        new_eps_table,
        args.dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    print(f"  wrote meta/episodes/chunk-000/file-000.parquet ({src_eps_table.num_rows} eps)")

    # --- Aggregate stats.json ----------------------------------------------
    src_stats_path = args.src / "meta" / "stats.json"
    if not src_stats_path.exists():
        print(f"error: source meta/stats.json missing at {src_stats_path}",
              file=sys.stderr)
        return 1
    src_stats = json.loads(src_stats_path.read_text())

    state_agg = aggregate_stats_from_per_ep(
        [{"observation.state": d} for d in new_state_per_ep]
    )["observation.state"]
    new_stats = dict(src_stats)
    new_stats["observation.state"] = stats_to_json({"x": state_agg})["x"]
    (args.dst / "meta" / "stats.json").write_text(json.dumps(new_stats, indent=2))
    print(f"  wrote meta/stats.json")

    # --- info.json ---------------------------------------------------------
    info = json.loads((args.src / "meta" / "info.json").read_text())
    new_info = dict(info)
    feats = dict(new_info["features"])
    state_feat = dict(feats["observation.state"])
    state_feat["shape"] = [KEPT_CHANNELS]
    state_feat["names"] = KEPT_CHANNEL_NAMES
    feats["observation.state"] = state_feat
    new_info["features"] = feats
    new_info["frame_transform"] = "baselink_smolvla"
    (args.dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))
    print(f"  wrote meta/info.json (state shape=[{KEPT_CHANNELS}], "
          f"frame_transform=baselink_smolvla)")

    # --- tasks.parquet (language strings already correct) ------------------
    shutil.copy(args.src / "meta" / "tasks.parquet",
                args.dst / "meta" / "tasks.parquet")
    print(f"  copied meta/tasks.parquet")

    # --- Train/val split ---------------------------------------------------
    all_ep_indices = list(range(n_eps))
    rng = random.Random(args.seed)
    rng.shuffle(all_ep_indices)
    n_val = max(1, int(n_eps * args.val_fraction))
    n_train = n_eps - n_val
    train_eps = sorted(all_ep_indices[:n_train])
    val_eps = sorted(all_ep_indices[n_train:])
    (args.dst / "train_episodes.json").write_text(json.dumps(train_eps))
    (args.dst / "val_episodes.json").write_text(json.dumps(val_eps))
    print(f"  wrote train_episodes.json ({n_train} eps) "
          f"and val_episodes.json ({n_val} eps) [seed={args.seed}]")

    # --- Videos: symlink ---------------------------------------------------
    src_videos = args.src / "videos"
    if src_videos.exists():
        target = src_videos.resolve() if src_videos.is_symlink() else src_videos
        (args.dst / "videos").symlink_to(target)
        print(f"  symlinked videos/ -> {target}")

    print()
    print(f"=== done: {args.dst} ===")
    print(f"  state: {SRC_DIM}-dim base_link -> {KEPT_CHANNELS}-dim base_link")
    print(f"  action: 7-dim absolute TCP pose (base_link, unchanged)")
    print(f"  next: train_smolvla.py --dataset-root {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
