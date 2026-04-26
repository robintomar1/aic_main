#!/usr/bin/env python3
"""Quick health check for a LeRobot v3 dataset produced by collect_lerobot.py.

Usage (inside the dev container):
    pixi run python my_policy/scripts/inspect_dataset.py /root/aic_data/smoke_d4b_dataset

Reads the dataset directly via the parquet files under meta/episodes/ and
data/ — no LeRobotDataset API call needed (its surface has shifted across
versions; the on-disk layout has been stable since v3.0).

What it reports:
  - meta/info.json top-line stats (fps, episode count, total frames, features)
  - Cross-check vs. summary.json from the recorder logs dir (if present)
  - Per-episode summary table (frames, duration, instruction)
  - Sanity checks via precomputed stats:
      * episode shorter than 0.5 s        → partial recording
      * action std ≈ 0                    → robot didn't move
      * groundtruth.port_pose all-zero    → TF lookup wasn't wired
      * wrench_compensated constant       → tare math broken
      * camera video file missing         → encoder failed silently
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_episode_meta(root: Path):
    """Concatenate all meta/episodes/chunk-*/file-*.parquet into one DataFrame."""
    import pandas as pd
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        return None
    parts = sorted(meta_dir.rglob("file-*.parquet"))
    if not parts:
        return None
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _load_data_episode(root: Path, ep_row) -> "pd.DataFrame":
    """Read just the rows for one episode from the data parquet."""
    import pandas as pd
    chunk = int(ep_row["data/chunk_index"])
    file_idx = int(ep_row["data/file_index"])
    path = root / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    from_ = int(ep_row["dataset_from_index"])
    to_ = int(ep_row["dataset_to_index"])
    # The parquet file may pack multiple episodes; slice by index col if
    # present, else assume row order matches.
    if "index" in df.columns:
        return df[(df["index"] >= from_) & (df["index"] < to_)]
    return df.iloc[from_:to_]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root", type=Path, help="Path to the LeRobot dataset directory.")
    p.add_argument("--episode", type=int, default=None,
                   help="Deep-dive on a single episode index (lists feature shapes, "
                        "first/last action, etc).")
    args = p.parse_args()

    if not args.root.is_dir():
        print(f"ERROR: {args.root} is not a directory", file=sys.stderr)
        return 2
    info_path = args.root / "meta" / "info.json"
    if not info_path.exists():
        print(f"ERROR: not a LeRobot dataset (missing {info_path})", file=sys.stderr)
        return 2

    info = json.loads(info_path.read_text())
    fps = info.get("fps", 20)
    feature_keys = list(info.get("features", {}).keys())

    print(f"=== Dataset: {args.root} ===")
    print(f"  fps:              {fps}")
    print(f"  total_episodes:   {info.get('total_episodes')}")
    print(f"  total_frames:     {info.get('total_frames')}")
    print(f"  features:         {len(feature_keys)} keys")

    # Recorder summary cross-check.
    summary = None
    summary_paths = list(args.root.parent.glob(f"{args.root.name}_logs/summary.json"))
    if summary_paths:
        try:
            summary = json.loads(summary_paths[0].read_text())
        except Exception as ex:
            print(f"  (could not parse {summary_paths[0]}: {ex})")
    if summary is not None:
        save_stats = summary.get("save_stats", {})
        n_disk = info.get("total_episodes", 0)
        print()
        print(f"=== Recorder summary.json ===")
        print(f"  trials run:       {len(summary.get('trials', []))}")
        print(f"  saves submitted:  {save_stats.get('submitted', '?')}")
        print(f"  saves succeeded:  {save_stats.get('succeeded', '?')}")
        print(f"  saves failed:     {save_stats.get('failed', '?')}")
        if save_stats.get("errors"):
            print(f"  save errors:")
            for e in save_stats["errors"]:
                print(f"    - {e}")
        if (s := save_stats.get("succeeded")) is not None and s != n_disk:
            print(f"  ⚠ MISMATCH: {s} succeeded saves vs. {n_disk} episodes on disk")
        # Per-trial outcome counts
        outcomes = {}
        for t in summary.get("trials", []):
            outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1
        if outcomes:
            print(f"  trial outcomes:   {outcomes}")

    # Per-episode metadata.
    try:
        eps = _load_episode_meta(args.root)
    except ImportError as ex:
        print(f"\nERROR: pandas/pyarrow not importable ({ex}). "
              f"Run inside the pixi env.", file=sys.stderr)
        return 2
    if eps is None or len(eps) == 0:
        print(f"\nERROR: no episode metadata found under {args.root}/meta/episodes/",
              file=sys.stderr)
        return 2

    eps = eps.sort_values("episode_index").reset_index(drop=True)
    print()
    print(f"=== Per-episode summary ===")
    print(f"  {'idx':>4} {'frames':>7} {'dur_s':>7}  {'instruction':<70}")
    print(f"  {'-' * 4} {'-' * 7} {'-' * 7}  {'-' * 70}")
    issues: list[str] = []
    for _, row in eps.iterrows():
        ep_idx = int(row["episode_index"])
        length = int(row["length"])
        duration = length / fps if fps else 0.0
        tasks = row.get("tasks", [])
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        instr = (tasks[0] if tasks else "") if isinstance(tasks, list) else str(tasks)
        instr_disp = instr[:67] + "..." if len(instr) > 70 else instr
        print(f"  {ep_idx:>4} {length:>7} {duration:>7.1f}  {instr_disp:<70}")
        if length < int(0.5 * fps):
            issues.append(
                f"episode {ep_idx}: only {length} frames (~{duration:.2f}s) — partial?"
            )

    # Sanity checks via precomputed stats.
    print()
    print(f"=== Sanity checks (from precomputed per-episode stats) ===")

    def _stat_col(feature: str, stat: str) -> str:
        return f"stats/{feature}/{stat}"

    # action variance
    if _stat_col("action", "std") in eps.columns:
        action_stds = []
        for _, row in eps.iterrows():
            arr = np.asarray(row[_stat_col("action", "std")], dtype=float).flatten()
            action_stds.append(float(arr.max()) if arr.size else 0.0)
        if action_stds:
            print(f"  action.std (max-component, across episodes): "
                  f"min={min(action_stds):.4f} mean={np.mean(action_stds):.4f} "
                  f"max={max(action_stds):.4f}")
            if max(action_stds) < 1e-4:
                issues.append(
                    "action stream is ~constant in every episode "
                    "(robot didn't move, or wrong action topic)"
                )
    else:
        issues.append("no `action` stats column — action wasn't recorded")

    # groundtruth.port_pose all-zero
    gt_port_keys = [k for k in feature_keys if "groundtruth.port_pose" in k]
    if gt_port_keys:
        gt_key = gt_port_keys[0]
        col_max = _stat_col(gt_key, "max")
        col_min = _stat_col(gt_key, "min")
        if col_max in eps.columns:
            zero_eps = []
            for _, row in eps.iterrows():
                hi = np.asarray(row[col_max], dtype=float).flatten()
                lo = np.asarray(row[col_min], dtype=float).flatten()
                if np.max(np.abs(np.concatenate([hi, lo]))) < 1e-9:
                    zero_eps.append(int(row["episode_index"]))
            if zero_eps:
                issues.append(
                    f"{gt_key} is all-zero in episodes {zero_eps} "
                    "(TF wasn't wired or ground_truth:=true was off)"
                )
            else:
                print(f"  {gt_key}: non-zero in all episodes ✓")
    else:
        issues.append("no groundtruth.port_pose feature — TF wasn't logged")

    # wrench compensated variance
    wrench_keys = [k for k in feature_keys if "wrench_compensated" in k]
    if wrench_keys:
        w_key = wrench_keys[0]
        col_std = _stat_col(w_key, "std")
        if col_std in eps.columns:
            const_eps = []
            for _, row in eps.iterrows():
                arr = np.asarray(row[col_std], dtype=float).flatten()
                if arr.size and float(arr.max()) < 1e-9:
                    const_eps.append(int(row["episode_index"]))
            if const_eps:
                issues.append(
                    f"{w_key} is constant in episodes {const_eps} "
                    "(tare math broken or fts topic stalled)"
                )
            else:
                print(f"  {w_key}: variance >0 in all episodes ✓")
    else:
        issues.append("no wrench_compensated feature — F/T wasn't logged")

    # video files
    videos_dir = args.root / "videos"
    if videos_dir.exists():
        n_videos = sum(1 for _ in videos_dir.rglob("*.mp4"))
        cam_keys = [k for k in feature_keys if "image" in k.lower()]
        expected = len(eps) * len(cam_keys)
        print(f"  video files:      {n_videos} found "
              f"(expected ≥{expected}: {len(eps)} ep × {len(cam_keys)} cams)")
        if expected and n_videos < expected:
            issues.append(
                f"only {n_videos} mp4 files (expected {expected}) — "
                "encoder may have failed silently"
            )
    else:
        issues.append("no videos/ directory — videos weren't written")

    # Optional deep-dive on one episode.
    if args.episode is not None:
        sel = eps[eps["episode_index"] == args.episode]
        if len(sel) == 0:
            print(f"\n--episode {args.episode} not found")
        else:
            row = sel.iloc[0]
            print()
            print(f"=== Episode {args.episode} deep-dive ===")
            length = int(row["length"])
            print(f"  length: {length}  duration: {length / fps:.2f}s")
            tasks = row.get("tasks", [])
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            print(f"  tasks: {tasks}")
            try:
                df = _load_data_episode(args.root, row)
            except Exception as ex:
                df = None
                print(f"  (could not load data parquet: {ex})")
            if df is not None and len(df) > 0:
                print(f"  data parquet columns: {len(df.columns)} cols, "
                      f"{len(df)} rows")
                # Print first/last action and key signals
                for col in ("action", "observation.wrench_compensated",
                            "observation.groundtruth.port_pose_base_link"):
                    if col in df.columns:
                        first = df[col].iloc[0]
                        last = df[col].iloc[-1]
                        print(f"  {col}:")
                        print(f"    first: {first}")
                        print(f"    last:  {last}")

    print()
    print(f"=== Verdict ===")
    if not issues:
        print("  ✓ No issues detected.")
    else:
        for i in issues:
            print(f"  ⚠ {i}")

    print()
    print(f"=== Visualize (lerobot 0.5.x: lerobot_dataset_viz, uses rerun) ===")
    repo_id = info.get("repo_id", "local/inspect")
    logs_dir = f"{args.root.parent}/{args.root.name}_logs"
    print(f"  Local GUI:   pixi run python -m lerobot.scripts.lerobot_dataset_viz --repo-id {repo_id} --root {args.root} --episode-index 0")
    print(f"  Save .rrd:   pixi run python -m lerobot.scripts.lerobot_dataset_viz --repo-id {repo_id} --root {args.root} --episode-index 0 --save 1 --output-dir {logs_dir}")
    print(f"  Distant:     pixi run python -m lerobot.scripts.lerobot_dataset_viz --repo-id {repo_id} --root {args.root} --episode-index 0 --mode distant")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
