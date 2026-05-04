#!/usr/bin/env python3
"""Clean v9-act dataset: patch stale leading action frames and canonicalize
action quaternion signs to match the state quaternion at the same frame.

Two transforms applied to the `action` column ONLY (state, images, episode
boundaries, video timestamps are all preserved):

1. **Stale leading-frame patching** — rare (~0.06% of frames). When the
   recorder captures the first ROS msg for a new episode, it sometimes
   pulls the previous trial's pose_command before the new policy has had
   a chance to publish. Detection: episodes where `action.position` differs
   from the synchronized `state.tcp_pose.position` by >50 mm at episode
   start. Fix: overwrite each such leading frame with the first "good"
   action of that episode (the one whose position agrees with TCP).

2. **Quaternion sign canonicalization** — pervasive (~41% of frames).
   `q` and `-q` represent the same rotation, but CheatCodeRobust's pose-
   target math doesn't preserve a hemisphere convention, so the recorded
   action quat sometimes lands in the opposite hemisphere from the
   synchronized state quat. The model has no input signal that can
   predict which hemisphere — the optimal L1 solution under random sign
   flips is to predict a smoothed average that's neither, which is what
   manifested as the "wrist locked" behavior in v1. Fix: per frame, if
   `dot(action_quat, state_quat) < 0`, negate the entire action quat.

After both transforms:
- Per-episode stats (in meta/episodes/) are recomputed from the new action.
- Aggregate stats.json is recomputed from per-episode stats.
- Everything else is symlinked from the source dataset (videos/) or
  copied (small metadata files).

Pure pyarrow + numpy. No torch / lerobot dependency. Run anywhere.

Usage:
    python3 my_policy/scripts/clean_act_dataset.py \\
        --src /root/aic_data/v9_act_build/v9_act_merged \\
        --dst /root/aic_data/v9_act_build/v9_act_merged_clean
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


STALE_POS_THRESHOLD_M = 0.05   # 50 mm
STATE_POS = slice(0, 3)
STATE_QUAT = slice(3, 7)
ACTION_POS = slice(0, 3)
ACTION_QUAT = slice(3, 7)


def per_episode_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the same stats columns that build_act_dataset.py writes
    into the per-episode parquet (min/max/mean/std/count + quantiles)."""
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
    """Aggregate per-episode stats into dataset-level stats.

    For min/max: pointwise min/max across episodes.
    For mean: count-weighted average.
    For std: count-weighted variance combined with between-episode variance,
             then sqrt. (Standard pooled-variance formula.)
    For quantiles: count-weighted average (an approximation; lerobot's
                   aggregate_stats does the same).
    """
    out: dict[str, dict] = {}
    for feat in per_ep[0].keys():
        feat_stats_list = [p[feat] for p in per_ep]
        counts = np.array([s["count"] for s in feat_stats_list]).flatten()
        total = counts.sum()
        # min / max
        mins = np.stack([s["min"] for s in feat_stats_list])
        maxs = np.stack([s["max"] for s in feat_stats_list])
        # mean: count-weighted
        means = np.stack([s["mean"] for s in feat_stats_list])
        agg_mean = (means * counts[:, None]).sum(axis=0) / total
        # std via pooled variance: sigma^2 = sum(n_i*(s_i^2 + (mu_i-mu)^2)) / N
        stds = np.stack([s["std"] for s in feat_stats_list])
        var_within = (counts[:, None] * stds**2).sum(axis=0) / total
        var_between = (counts[:, None] * (means - agg_mean)**2).sum(axis=0) / total
        agg_std = np.sqrt(var_within + var_between)
        # quantiles: count-weighted average (approximation)
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
    """Serialize stats dict for stats.json (lists, not numpy)."""
    return {
        feat: {k: v.tolist() for k, v in feat_stats.items()}
        for feat, feat_stats in stats.items()
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="Source dataset root (e.g. v9_act_merged/).")
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

    print(f"=== Cleaning {args.src} -> {args.dst} ===")
    args.dst.mkdir(parents=True)
    (args.dst / "data" / "chunk-000").mkdir(parents=True)
    (args.dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    # --- Read source data parquet ------------------------------------------
    src_data_path = args.src / "data" / "chunk-000" / "file-000.parquet"
    print(f"  reading {src_data_path}")
    src_table = pq.read_table(src_data_path)
    n_rows = src_table.num_rows
    print(f"  {n_rows} rows, {len(src_table.column_names)} columns")

    action = np.stack(src_table.column("action").to_pylist()).astype(np.float32)
    state = np.stack(src_table.column("observation.state").to_pylist()).astype(np.float32)
    ep_idx = np.array(src_table.column("episode_index").to_pylist(), dtype=np.int64)

    ep_starts = np.r_[0, np.where(np.diff(ep_idx) != 0)[0] + 1]
    ep_ends = np.r_[ep_starts[1:], n_rows]
    n_eps = len(ep_starts)
    print(f"  {n_eps} episodes")

    # --- Fix 1: patch stale leading action frames --------------------------
    n_patched_frames = 0
    n_patched_episodes = 0
    for ep_i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        # Find first frame where action.position matches state.position.
        first_good = None
        for j in range(min(20, e - s)):
            a_pos = action[s + j, ACTION_POS]
            st_pos = state[s + j, STATE_POS]
            if np.linalg.norm(a_pos) > 1e-3 \
                    and np.linalg.norm(a_pos - st_pos) <= STALE_POS_THRESHOLD_M:
                first_good = j
                break
        if first_good is None or first_good == 0:
            continue
        # Overwrite frames [s, s+first_good) with the first good action.
        replacement = action[s + first_good].copy()
        action[s:s + first_good] = replacement
        n_patched_frames += first_good
        n_patched_episodes += 1
    print(f"  Fix 1: patched {n_patched_frames} stale frames in "
          f"{n_patched_episodes} episodes")

    # --- Fix 2: canonicalize action quat sign ------------------------------
    # For each frame: if dot(action_quat, state_quat) < 0, negate action quat.
    a_quat = action[:, ACTION_QUAT]
    st_quat = state[:, ACTION_QUAT]
    a_norms = np.linalg.norm(a_quat, axis=1)
    valid = a_norms > 0.5
    dots = (a_quat * st_quat).sum(axis=1)
    flip_mask = (dots < 0) & valid
    n_flipped = int(flip_mask.sum())
    action[flip_mask, 3:7] = -action[flip_mask, 3:7]
    print(f"  Fix 2: canonicalized {n_flipped} action quaternions "
          f"({100 * n_flipped / n_rows:.1f}% of frames)")

    # Sanity: after Fix 2, no frame should have negative dot.
    new_dots = (action[:, ACTION_QUAT] * state[:, STATE_QUAT]).sum(axis=1)
    bad_post = ((new_dots < 0) & valid).sum()
    assert bad_post == 0, f"sanity failed: {bad_post} frames still have neg dot"

    # --- Write modified data parquet ---------------------------------------
    print(f"  writing modified data parquet...")
    new_action_col = pa.array(action.tolist(), type=pa.list_(pa.float32(), 7))
    cols = {}
    for c in src_table.column_names:
        if c == "action":
            cols[c] = new_action_col
        else:
            cols[c] = src_table.column(c)
    new_table = pa.Table.from_pydict(cols)
    pq.write_table(new_table, args.dst / "data" / "chunk-000" / "file-000.parquet")
    print(f"  wrote data/chunk-000/file-000.parquet")

    # --- Recompute per-episode stats with new action -----------------------
    print(f"  recomputing per-episode stats...")
    src_eps_path = args.src / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    src_eps_table = pq.read_table(src_eps_path)
    n_eps_meta = src_eps_table.num_rows

    # Build new stat columns for action and observation.state. State is
    # unchanged but we recompute for consistency (cheap).
    new_action_per_ep: list[dict] = []
    new_state_per_ep: list[dict] = []
    for ep_i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        new_action_per_ep.append(per_episode_stats(action[s:e]))
        new_state_per_ep.append(per_episode_stats(state[s:e]))

    # Replace stats/action/* and stats/observation.state/* columns.
    new_eps_cols = {}
    for c in src_eps_table.column_names:
        if c.startswith("stats/action/"):
            stat_name = c.split("/")[-1]
            new_eps_cols[c] = pa.array(
                [d[stat_name].tolist() for d in new_action_per_ep])
        elif c.startswith("stats/observation.state/"):
            stat_name = c.split("/")[-1]
            new_eps_cols[c] = pa.array(
                [d[stat_name].tolist() for d in new_state_per_ep])
        else:
            new_eps_cols[c] = src_eps_table.column(c)
    new_eps_table = pa.Table.from_pydict(new_eps_cols)
    pq.write_table(new_eps_table,
                   args.dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    print(f"  wrote meta/episodes/chunk-000/file-000.parquet "
          f"({n_eps_meta} episodes)")

    # --- Recompute aggregate stats.json ------------------------------------
    print(f"  computing aggregate stats.json...")
    # Aggregate from the per-episode action and state stats we just computed.
    agg = {}
    agg["action"] = aggregate_stats_from_per_ep(new_action_per_ep)["action_dummy"] \
        if False else aggregate_stats_from_per_ep(
            [{"action": d} for d in new_action_per_ep])["action"]
    agg["observation.state"] = aggregate_stats_from_per_ep(
        [{"observation.state": d} for d in new_state_per_ep])["observation.state"]

    info = json.loads((args.src / "meta" / "info.json").read_text())
    stats_json = stats_to_json(agg)
    # Camera image stats: carry from source stats.json. Empty {} placeholders
    # don't survive lerobot's load_stats round-trip (cast_stats_to_numpy uses
    # flatten_dict which drops empty dicts), so make_dataset would crash with
    # KeyError when overlaying imagenet stats. We didn't modify any images,
    # so the source camera stats are still valid.
    src_stats_path = args.src / "meta" / "stats.json"
    if src_stats_path.exists():
        src_stats = json.loads(src_stats_path.read_text())
        for k, ft in info["features"].items():
            if ft.get("dtype") == "video" and k in src_stats:
                stats_json[k] = src_stats[k]
                print(f"  carried camera stats for {k} from source stats.json")
    else:
        # Source has no stats.json — fall back to imagenet (lerobot overrides
        # to imagenet anyway when use_imagenet_stats=True).
        IMAGENET = {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std":  [[[0.229]], [[0.224]], [[0.225]]],
            "min":  [[[0.0]], [[0.0]], [[0.0]]],
            "max":  [[[1.0]], [[1.0]], [[1.0]]],
        }
        for k, ft in info["features"].items():
            if ft.get("dtype") == "video":
                stats_json[k] = IMAGENET
        print(f"  source stats.json missing — using ImageNet for camera keys")
    (args.dst / "meta" / "stats.json").write_text(
        json.dumps(stats_json, indent=2))
    print(f"  wrote meta/stats.json ({len(stats_json)} feature keys)")

    # --- Copy / symlink everything else ------------------------------------
    print(f"  copying small metadata files...")
    shutil.copy(args.src / "meta" / "info.json", args.dst / "meta" / "info.json")
    shutil.copy(args.src / "meta" / "tasks.parquet",
                args.dst / "meta" / "tasks.parquet")
    for split in ("train_episodes.json", "val_episodes.json"):
        sp = args.src / split
        if sp.exists():
            shutil.copy(sp, args.dst / split)

    # videos/ is large — symlink the source tree rather than copying.
    src_videos = args.src / "videos"
    if src_videos.exists():
        (args.dst / "videos").symlink_to(src_videos.resolve())
        print(f"  symlinked videos/ -> {src_videos.resolve()}")

    print()
    print(f"=== done ===")
    print(f"  cleaned dataset at: {args.dst}")
    print(f"  next: train v2 with --dataset-root {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
