#!/usr/bin/env python3
"""Quick health check for a LeRobot dataset produced by collect_lerobot.py.

Usage (inside the dev container):
    pixi run python my_policy/scripts/inspect_dataset.py \
        /root/aic_data/smoke_d4b_dataset

What it reports:
  - Top-level stats from meta/info.json (fps, episode/frame counts, feature list)
  - Per-episode summary: length, duration, instruction, success metadata
  - Sanity checks designed to catch known recorder failure modes:
      * Episode shorter than 0.5 s  → likely partial recording (loop-blocked save)
      * Action variance ≈ 0          → policy never moved (or wrong action topic)
      * groundtruth port_pose all-zero → TF lookup wasn't actually wired
      * wrench_compensated all-zero  → tare math broken or fts topic stalled
      * Camera video file missing   → encoder failed silently
  - Comparison of episode count on disk vs. summary.json (if present)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _fmt(x, w=8, p=3):
    if x is None:
        return " " * w + "—"
    return f"{x:>{w}.{p}f}"


def _check_nonzero(arr: np.ndarray, name: str, atol: float = 1e-6) -> str | None:
    """Return an issue string if the array is constant/zero, else None."""
    if arr.size == 0:
        return f"{name}: empty"
    std = float(np.std(arr))
    if std < atol:
        return f"{name}: variance ≈ 0 (std={std:.2e}, mean={float(np.mean(arr)):.3f})"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path,
                   help="Path to the LeRobot dataset directory.")
    p.add_argument("--episode", type=int, default=None,
                   help="Deep-dive into a single episode index "
                        "(prints first/last frame, action/wrench stats).")
    p.add_argument("--show-summary", action="store_true",
                   help="Also print the recorder's summary.json (if present).")
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

    print(f"=== Dataset: {args.root} ===")
    print(f"  repo_id:          {info.get('repo_id', '?')}")
    print(f"  fps:              {fps}")
    print(f"  total_episodes:   {info.get('total_episodes')}")
    print(f"  total_frames:     {info.get('total_frames')}")
    print(f"  total_videos:     {info.get('total_videos')}")
    print(f"  features:         {len(info.get('features', {}))} keys")

    # Recorder summary side-by-side (if recorder ran in same root)
    summary_paths = list(args.root.parent.glob(f"{args.root.name}_logs/summary.json"))
    summary = None
    if summary_paths:
        summary = json.loads(summary_paths[0].read_text())
        save_stats = summary.get("save_stats", {})
        print()
        print(f"=== Recorder summary.json ===")
        print(f"  trials run:       {len(summary.get('trials', []))}")
        print(f"  saves submitted:  {save_stats.get('submitted', '?')}")
        print(f"  saves succeeded:  {save_stats.get('succeeded', '?')}")
        print(f"  saves failed:     {save_stats.get('failed', '?')}")
        if save_stats.get("errors"):
            print(f"  save errors:")
            for err in save_stats["errors"]:
                print(f"    - {err}")
        # Cross-check
        if (s := save_stats.get("succeeded")) is not None:
            disk = info.get("total_episodes", 0)
            if s != disk:
                print(f"  ⚠ MISMATCH: {s} succeeded saves vs. {disk} on disk")
        if args.show_summary:
            print(json.dumps(summary, indent=2))

    # Load via LeRobotDataset for richer access.
    print()
    print(f"=== Loading dataset ... ===")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as ex:
        print(f"ERROR: lerobot not importable: {ex}", file=sys.stderr)
        print("Run inside `pixi shell` or via `pixi run python ...`.", file=sys.stderr)
        return 2

    repo_id = info.get("repo_id", "local/inspect")
    ds = LeRobotDataset(repo_id, root=str(args.root))

    # Per-episode summary.
    print()
    print(f"=== Per-episode summary ===")
    print(f"  {'idx':>4} {'frames':>7} {'dur_s':>7} {'instruction':<70}")
    print(f"  {'-' * 4} {'-' * 7} {'-' * 7} {'-' * 70}")

    issues: list[str] = []
    fps_eff = fps

    for ep_idx in range(ds.num_episodes):
        try:
            from_ = int(ds.episode_data_index["from"][ep_idx])
            to_ = int(ds.episode_data_index["to"][ep_idx])
        except Exception:
            issues.append(f"episode {ep_idx}: failed to read index range")
            continue
        length = to_ - from_
        duration = length / fps_eff if fps_eff else 0.0

        instr = ""
        try:
            first = ds[from_]
            instr = str(first.get("task", ""))
        except Exception as ex:
            issues.append(f"episode {ep_idx}: failed to read first frame: {ex}")

        if len(instr) > 70:
            instr = instr[:67] + "..."
        print(f"  {ep_idx:>4} {length:>7} {duration:>7.1f} {instr:<70}")

        if length < int(0.5 * fps_eff):
            issues.append(
                f"episode {ep_idx}: only {length} frames (~{duration:.2f}s) — "
                "partial recording? (was the save loop blocked?)"
            )

    print()
    print(f"=== Sanity checks ===")
    # Aggregate sanity stats over all episodes (sampled at first/middle/last
    # frame of each to keep this fast even for big datasets).
    feature_keys = list(info.get("features", {}).keys())
    has_action = any(k.startswith("action") for k in feature_keys)
    has_gt_port = any("groundtruth.port_pose" in k for k in feature_keys)
    has_wrench = any("wrench_compensated" in k or "wrench" in k for k in feature_keys)

    if not has_action:
        issues.append("no `action` feature in dataset")
    if not has_gt_port:
        issues.append("no `groundtruth.port_pose*` feature — TF wasn't logged")
    if not has_wrench:
        issues.append("no `wrench*` feature — F/T wasn't logged")

    # For each episode, sample a few frames and check the recorded values
    # actually look like data.
    sample_n = min(10, ds.num_episodes)
    print(f"  sampling first {sample_n} episode(s) for value sanity ...")
    action_stds: list[float] = []
    port_pose_zero_eps: list[int] = []
    wrench_zero_eps: list[int] = []

    for ep_idx in range(sample_n):
        from_ = int(ds.episode_data_index["from"][ep_idx])
        to_ = int(ds.episode_data_index["to"][ep_idx])
        idxs = np.linspace(from_, to_ - 1, num=min(50, to_ - from_), dtype=int)

        actions, port_poses, wrenches = [], [], []
        for i in idxs:
            try:
                f = ds[int(i)]
            except Exception:
                continue
            if "action" in f:
                a = np.asarray(f["action"], dtype=float).flatten()
                if a.size:
                    actions.append(a)
            for k in feature_keys:
                if "groundtruth.port_pose" in k and k in f:
                    port_poses.append(np.asarray(f[k], dtype=float).flatten())
                    break
            for k in feature_keys:
                if "wrench_compensated" in k and k in f:
                    wrenches.append(np.asarray(f[k], dtype=float).flatten())
                    break

        if actions:
            stack = np.stack(actions)
            action_stds.append(float(np.std(stack)))
        if port_poses:
            stack = np.stack(port_poses)
            if float(np.max(np.abs(stack))) < 1e-9:
                port_pose_zero_eps.append(ep_idx)
        if wrenches:
            stack = np.stack(wrenches)
            if float(np.std(stack)) < 1e-9:
                wrench_zero_eps.append(ep_idx)

    if action_stds:
        print(f"  action std (across episodes): "
              f"min={min(action_stds):.4f} mean={np.mean(action_stds):.4f} "
              f"max={max(action_stds):.4f}")
        if max(action_stds) < 1e-4:
            issues.append("action stream is ~constant in every sampled episode "
                          "(robot didn't move, or wrong action topic)")
    if port_pose_zero_eps:
        issues.append(
            f"groundtruth.port_pose is all-zero in episodes: {port_pose_zero_eps} "
            "(TF lookup not wired or ground_truth:=true was off)"
        )
    if wrench_zero_eps:
        issues.append(
            f"wrench_compensated is constant in episodes: {wrench_zero_eps} "
            "(tare math broken, or fts topic stalled)"
        )

    # Video files
    videos_dir = args.root / "videos"
    if videos_dir.exists():
        n_videos = sum(1 for _ in videos_dir.rglob("*.mp4"))
        expected = (info.get("total_episodes", 0) *
                    sum(1 for k in feature_keys if "image" in k.lower()))
        print(f"  videos found:  {n_videos} files under {videos_dir}")
        if expected and n_videos < expected:
            issues.append(
                f"expected ≥{expected} video files (3 cams × {info.get('total_episodes')} episodes), "
                f"found {n_videos} — encoder may have failed"
            )
    else:
        issues.append("no `videos/` directory — videos weren't written")

    # Optional deep-dive on one episode.
    if args.episode is not None:
        ep = args.episode
        if ep < 0 or ep >= ds.num_episodes:
            print(f"\n--episode {ep} out of range (0..{ds.num_episodes - 1})")
        else:
            print()
            print(f"=== Episode {ep} deep-dive ===")
            from_ = int(ds.episode_data_index["from"][ep])
            to_ = int(ds.episode_data_index["to"][ep])
            length = to_ - from_
            print(f"  range: [{from_}, {to_})  length={length}  "
                  f"duration={length / fps_eff:.2f}s")
            print()
            print(f"  features in first frame:")
            f0 = ds[from_]
            for k in sorted(f0.keys()):
                v = f0[k]
                if hasattr(v, "shape"):
                    desc = f"shape={tuple(v.shape)} dtype={v.dtype}"
                elif isinstance(v, str):
                    s = v[:50] + ("..." if len(v) > 50 else "")
                    desc = f"str  {s!r}"
                else:
                    desc = repr(v)[:60]
                print(f"    {k:<45} {desc}")

    print()
    print(f"=== Verdict ===")
    if not issues:
        print("  ✓ No issues detected.")
    else:
        for i in issues:
            print(f"  ⚠ {i}")
    print()
    print(f"=== Visualize (lerobot 0.5.x: lerobot_dataset_viz, uses rerun) ===")
    logs_dir = f"{args.root.parent}/{args.root.name}_logs"
    print(f"  Local GUI (needs X11; container has it via xhost +local:docker):")
    print(f"    pixi run python -m lerobot.scripts.lerobot_dataset_viz --repo-id {repo_id} --root {args.root} --episode-index 0")
    print()
    print(f"  Save to .rrd (no GUI; open later on host with `rerun <file.rrd>`):")
    print(f"    pixi run python -m lerobot.scripts.lerobot_dataset_viz --repo-id {repo_id} --root {args.root} --episode-index 0 --save 1 --output-dir {logs_dir}")
    print()
    print(f"  Distant (SSH-friendly: container serves, host views):")
    print(f"    pixi run python -m lerobot.scripts.lerobot_dataset_viz --repo-id {repo_id} --root {args.root} --episode-index 0 --mode distant")
    print(f"    # then on host (install: pip install rerun-sdk):")
    print(f"    rerun rerun+http://localhost:<grpc-port>/proxy")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
