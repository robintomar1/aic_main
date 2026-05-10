#!/usr/bin/env python3
"""Smooth the SC oracle's hover->descend setpoint step in action[xyz].

Source datasets (e.g. v9_port_local_smolvla_dataset_wo_tcp_error) contain
two task families:

  * SFP (task_index 0..9, 296 eps): action z descends as a continuous
    micro-stair-step approach. No macro discontinuity. SmolVLA learns
    these episodes fine.

  * SC  (task_index 10..11, 130 eps): the oracle holds a "hover"
    setpoint at z ~= -0.179 m for ~125 frames, then snaps the target to
    the descend setpoint at z ~= -0.080 m in one frame (~10 cm step).
    The 100% prevalence of this step exactly matches SmolVLA's failure
    mode on SC inserts: the chunk straddling the jump must predict a
    10 cm action step from observation that did not change, and there
    is no visual / state cue for the phase transition.

Fix: replace action[K-W+1 : K+1, 0:3] with a 5th-order smoothstep ramp
from action[K-W] (still at the hover setpoint) to action[K+1] (the
descent setpoint). K = argmax|dz_action| in z; threshold |dz| > 5 cm
isolates the macro jump from 4-10 mm approach micro-steps. Quaternion
is left untouched (max delta < 1e-4 across the boundary in original).

The ramp is applied BACKWARD (ending at K, not starting at K+1) on
purpose. During the original "hover" phase the TCP was already drifting
toward the port at ~3 mm/frame due to admittance + cable load -- the
hover action target was being ignored by physics. A forward ramp
(replacing K+1..K+W) would produce action labels in the WRONG
direction relative to obs: TCP visibly near the port, action label
saying "back up to hover". A backward ramp keeps the hover action
constant through frame K-W, then transitions smoothly down so that by
frame K the action has reached the descent setpoint -- action stays
slightly ahead of obs in the toward-port direction throughout, matching
the lead pattern that SFP episodes already exhibit.

W = 25 frames @ 20 Hz (1.25 s) -- comfortably inside SmolVLA
chunk_size=50, gentler than SFP's natural ~5 mm/frame descent.

SFP rows and observation.state are NOT modified -- observation is
sensor data and SFP needs no fix.

Per-episode action stats and aggregate stats.json[action] are
recomputed; everything else (obs / image stats, info.json features,
tasks.parquet, splits) is carried verbatim. videos/ is symlinked.

Usage:
    python3 my_policy/scripts/smolvla/smooth_action_z_jumps.py \\
        --src /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset_wo_tcp_error \\
        --dst /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset_z_smoothed
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SC_TASK_INDICES = {10, 11}
JUMP_THRESHOLD_M = 0.05
RAMP_WINDOW = 25
ACTION_DIM = 7
STATS_KEYS = ("min", "max", "mean", "std", "count",
              "q01", "q10", "q50", "q90", "q99")


def smoothstep5(t: np.ndarray) -> np.ndarray:
    """Perlin's 5th-order smoothstep: 6t^5 - 15t^4 + 10t^3.

    C^2 continuous: zero 1st and 2nd derivatives at t=0 and t=1.
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def smooth_episode(action: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """Smooth one SC episode's action[xyz] step (BACKWARD ramp).

    Replaces action[K-W+1 : K+1, 0:3] with a smoothstep ramp from
    action[K-W] (last hover) to action[K+1] (first descent target).
    The ramp ends at frame K so that frame K+1 onward is bit-identical
    to source.

    Returns (new_action, info_dict). If no qualifying jump or the
    episode is too short to fit the ramp before K, returns (a, None).
    """
    a = action.copy()
    dz = np.diff(a[:, 2])
    K = int(np.argmax(np.abs(dz)))
    dz_K = float(dz[K])
    if abs(dz_K) < JUMP_THRESHOLD_M:
        return a, None

    start = max(K - RAMP_WINDOW + 1, 1)   # leave at least frame 0 unchanged
    actual_W = K - start + 1              # number of ramp samples (replaces frames start..K)
    if actual_W < 2:
        return a, None

    anchor_before = a[start - 1, :3].copy()  # last unchanged hover sample
    anchor_after = a[K + 1, :3].copy()       # first unchanged descent sample
    t = (np.arange(1, actual_W + 1, dtype=np.float64) /
         (actual_W + 1))                     # excludes 0 and 1
    s = smoothstep5(t)                       # (actual_W,)
    ramp = (anchor_before[None, :]
            + s[:, None] * (anchor_after - anchor_before)[None, :])
    a[start:K + 1, :3] = ramp.astype(a.dtype)
    return a, {
        "K": K, "dz_K": dz_K, "ramp_W": actual_W,
        "ramp_start": start, "ramp_end": K,
        "z_anchor_before": float(anchor_before[2]),
        "z_anchor_after": float(anchor_after[2]),
    }


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


def aggregate_action_stats(per_ep: list[dict]) -> dict[str, np.ndarray]:
    counts = np.array([s["count"] for s in per_ep]).flatten()
    total = int(counts.sum())
    mins = np.stack([s["min"] for s in per_ep])
    maxs = np.stack([s["max"] for s in per_ep])
    means = np.stack([s["mean"] for s in per_ep])
    agg_mean = (means * counts[:, None]).sum(axis=0) / total
    stds = np.stack([s["std"] for s in per_ep])
    var_within = (counts[:, None] * stds**2).sum(axis=0) / total
    var_between = (counts[:, None] * (means - agg_mean)**2).sum(axis=0) / total
    agg_std = np.sqrt(var_within + var_between)
    out = {
        "min": mins.min(axis=0).astype(np.float32),
        "max": maxs.max(axis=0).astype(np.float32),
        "mean": agg_mean.astype(np.float32),
        "std": agg_std.astype(np.float32),
        "count": np.array([total], dtype=np.int64),
    }
    for qk in ("q01", "q10", "q50", "q90", "q99"):
        qs = np.stack([s[qk] for s in per_ep])
        out[qk] = ((qs * counts[:, None]).sum(axis=0) / total).astype(np.float32)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--force", action="store_true",
                   help="Delete dst if it exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Detect jumps and report ramp params; write nothing.")
    args = p.parse_args()

    if not args.src.exists():
        print(f"error: source {args.src} does not exist", file=sys.stderr)
        return 1
    if args.dst.exists() and not args.dry_run:
        if args.force:
            print(f"--force: removing existing {args.dst}")
            shutil.rmtree(args.dst)
        else:
            print(f"error: destination {args.dst} already exists "
                  f"(pass --force to overwrite)", file=sys.stderr)
            return 1

    print(f"=== smooth action[xyz] step in SC episodes ===")
    print(f"  src: {args.src}")
    print(f"  dst: {args.dst}{' (DRY RUN)' if args.dry_run else ''}")
    print(f"  ramp: smoothstep5, W={RAMP_WINDOW} frames, threshold |dz|>{JUMP_THRESHOLD_M} m")

    # --- Read source data parquet ----------------------------------------
    src_data_path = args.src / "data" / "chunk-000" / "file-000.parquet"
    src_table = pq.read_table(src_data_path)
    n_rows = src_table.num_rows
    print(f"  rows: {n_rows}")

    action = np.stack(src_table.column("action").to_pylist()).astype(np.float32)
    if action.shape[1] != ACTION_DIM:
        print(f"error: action has {action.shape[1]} dims, expected {ACTION_DIM}",
              file=sys.stderr)
        return 1
    obs_state = np.stack(src_table.column("observation.state").to_pylist()).astype(np.float32)
    obs_z = obs_state[:, 2]  # tcp_pose.position.z

    ep_idx = np.array(src_table.column("episode_index").to_pylist(), dtype=np.int64)
    task_idx = np.array(src_table.column("task_index").to_pylist(), dtype=np.int64)
    ep_starts = np.r_[0, np.where(np.diff(ep_idx) != 0)[0] + 1]
    ep_ends = np.r_[ep_starts[1:], n_rows]
    n_eps = len(ep_starts)
    print(f"  episodes: {n_eps}")

    # --- Smooth SC episodes ----------------------------------------------
    new_action = action.copy()
    sc_modified = 0
    sc_seen = 0
    sc_skipped_no_jump = []
    sample_log = []
    for s, e in zip(ep_starts, ep_ends):
        ep_task = int(task_idx[s])
        ep_id = int(ep_idx[s])
        if ep_task not in SC_TASK_INDICES:
            continue
        sc_seen += 1
        smoothed, info = smooth_episode(action[s:e])
        if info is None:
            sc_skipped_no_jump.append(ep_id)
            continue
        new_action[s:e] = smoothed
        sc_modified += 1
        if len(sample_log) < 5:
            sample_log.append((ep_id, info))
    print(f"  SC episodes: seen={sc_seen}, modified={sc_modified}, "
          f"no-qualifying-jump={len(sc_skipped_no_jump)}")
    if sc_skipped_no_jump:
        print(f"  WARNING skipped SC eps (no |dz|>{JUMP_THRESHOLD_M}m): "
              f"{sc_skipped_no_jump[:10]}{' ...' if len(sc_skipped_no_jump) > 10 else ''}")
    print(f"  sample modified episodes:")
    for ep_id, info in sample_log:
        print(f"    ep {ep_id}: K={info['K']} dz={info['dz_K']:+.4f} "
              f"W={info['ramp_W']} ramp[{info['ramp_start']}..{info['ramp_end']}] "
              f"z {info['z_anchor_before']:+.4f} -> {info['z_anchor_after']:+.4f}")

    # Verify SFP rows bit-identical
    sfp_mask = ~np.isin(task_idx, list(SC_TASK_INDICES))
    sfp_diff = np.abs(new_action[sfp_mask] - action[sfp_mask]).max()
    print(f"  max |diff| on SFP rows (must be 0): {sfp_diff}")
    if sfp_diff != 0.0:
        print("error: SFP rows changed", file=sys.stderr)
        return 1

    # Quaternion untouched on SC?
    sc_mask = ~sfp_mask
    quat_diff = np.abs(new_action[sc_mask, 3:] - action[sc_mask, 3:]).max()
    print(f"  max |diff| on SC quaternion (must be 0): {quat_diff}")
    if quat_diff != 0.0:
        print("error: SC quaternion changed", file=sys.stderr)
        return 1

    # Frames K+1.. in SC episodes must be bit-identical (only K-W+1..K changed)
    sc_post_jump_diffs = []
    for s, e in zip(ep_starts, ep_ends):
        if int(task_idx[s]) not in SC_TASK_INDICES:
            continue
        a_orig = action[s:e]
        dz = np.diff(a_orig[:, 2])
        K = int(np.argmax(np.abs(dz)))
        if abs(dz[K]) < JUMP_THRESHOLD_M:
            continue
        # Frame K+1 onward should be unchanged
        d = np.abs(new_action[s + K + 1:e] - action[s + K + 1:e]).max()
        sc_post_jump_diffs.append(d)
    print(f"  max |diff| on SC frames K+1.. (must be 0): {max(sc_post_jump_diffs):.6g}")
    if max(sc_post_jump_diffs) != 0.0:
        print("error: SC post-jump frames changed", file=sys.stderr)
        return 1

    # Lead-direction sanity vs ORIGINAL: at any frame in the new ramp
    # window, the smoothed action must not have a more-negative lead
    # (obs - action) than the original action at the same frame.
    # In port-local frame: less-negative z = closer to port; lead toward
    # port = action_z >= obs_z. So degradation = max((obs - new_action)
    # - (obs - orig_action)) = max(orig_action - new_action). If this is
    # <= 0 we've only made the lead better or kept it the same.
    degraded = []
    max_new_overshoot = 0.0
    max_orig_overshoot = 0.0
    for s, e in zip(ep_starts, ep_ends):
        if int(task_idx[s]) not in SC_TASK_INDICES:
            continue
        a_orig = action[s:e]
        dz = np.diff(a_orig[:, 2])
        K = int(np.argmax(np.abs(dz)))
        if abs(dz[K]) < JUMP_THRESHOLD_M:
            continue
        ramp_start = max(K - RAMP_WINDOW + 1, 1)
        new_az = new_action[s + ramp_start:s + K + 1, 2]
        orig_az = action[s + ramp_start:s + K + 1, 2]
        oz = obs_z[s + ramp_start:s + K + 1]
        new_overshoot = (oz - new_az).max()    # positive = obs ahead of new action
        orig_overshoot = (oz - orig_az).max()  # positive = obs ahead of orig action
        max_new_overshoot = max(max_new_overshoot, float(new_overshoot))
        max_orig_overshoot = max(max_orig_overshoot, float(orig_overshoot))
        # We degrade if new_action is more negative than orig_action anywhere.
        deg = float((orig_az - new_az).max())
        if deg > 1e-4:
            degraded.append((int(ep_idx[s]), deg))
    print(f"  lead overshoot (max obs - action) in ramp window:")
    print(f"    original: {max_orig_overshoot*1000:.1f} mm  smoothed: {max_new_overshoot*1000:.1f} mm")
    if degraded:
        print(f"  WARNING: {len(degraded)} SC eps where new action is MORE negative than original "
              f"in ramp (sample: {degraded[:3]})")
    else:
        print(f"  no SC ep has new action more negative than original in ramp -- OK")

    if args.dry_run:
        print("\n=== dry-run complete (no files written) ===")
        return 0

    # --- Write dataset directories ---------------------------------------
    args.dst.mkdir(parents=True)
    (args.dst / "data" / "chunk-000").mkdir(parents=True)
    (args.dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    # --- Write modified data parquet -------------------------------------
    new_action_col = pa.array(
        new_action.tolist(), type=pa.list_(pa.float32(), ACTION_DIM),
    )
    cols = {}
    for c in src_table.column_names:
        if c == "action":
            cols[c] = new_action_col
        else:
            cols[c] = src_table.column(c)
    new_table = pa.Table.from_pydict(cols)
    pq.write_table(new_table, args.dst / "data" / "chunk-000" / "file-000.parquet")
    print(f"  wrote data/chunk-000/file-000.parquet")

    # --- Recompute per-episode action stats ------------------------------
    src_eps_path = args.src / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    src_eps_table = pq.read_table(src_eps_path)
    n_eps_meta = src_eps_table.num_rows
    if n_eps_meta != n_eps:
        print(f"warning: data has {n_eps} eps but episode-meta has {n_eps_meta}",
              file=sys.stderr)

    new_action_per_ep: list[dict] = []
    for s, e in zip(ep_starts, ep_ends):
        new_action_per_ep.append(per_episode_stats(new_action[s:e]))

    new_eps_cols = {}
    for c in src_eps_table.column_names:
        if c.startswith("stats/action/"):
            stat_name = c.split("/")[-1]
            new_eps_cols[c] = pa.array(
                [d[stat_name].tolist() for d in new_action_per_ep])
        else:
            new_eps_cols[c] = src_eps_table.column(c)
    new_eps_table = pa.Table.from_pydict(new_eps_cols)
    pq.write_table(new_eps_table,
                   args.dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    print(f"  wrote meta/episodes/chunk-000/file-000.parquet ({n_eps_meta} eps)")

    # --- Aggregate stats.json: action recomputed; rest carried -----------
    src_stats_path = args.src / "meta" / "stats.json"
    if not src_stats_path.exists():
        print(f"error: source meta/stats.json missing", file=sys.stderr)
        return 1
    src_stats = json.loads(src_stats_path.read_text())

    action_agg = aggregate_action_stats(new_action_per_ep)
    new_stats = dict(src_stats)
    new_stats["action"] = {k: v.tolist() for k, v in action_agg.items()}
    (args.dst / "meta" / "stats.json").write_text(json.dumps(new_stats, indent=2))
    print(f"  wrote meta/stats.json ({len(new_stats)} feature keys)")

    # --- info.json: bump frame_transform tag -----------------------------
    info = json.loads((args.src / "meta" / "info.json").read_text())
    new_info = dict(info)
    src_tag = new_info.get("frame_transform", "")
    new_info["frame_transform"] = src_tag + "_z_smoothed" if src_tag else "z_smoothed"
    (args.dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))
    print(f"  wrote meta/info.json (frame_transform={new_info['frame_transform']})")

    # --- Copy small files; symlink videos --------------------------------
    shutil.copy(args.src / "meta" / "tasks.parquet",
                args.dst / "meta" / "tasks.parquet")
    for sidecar in ("train_episodes.json", "val_episodes.json",
                    "source_episode_map.json"):
        sp = args.src / sidecar
        if sp.exists():
            shutil.copy(sp, args.dst / sidecar)
            print(f"  copied {sidecar}")
    src_videos = args.src / "videos"
    if src_videos.is_symlink():
        # Preserve the original symlink target verbatim. Source datasets
        # built inside the docker container often point at container-internal
        # paths like /root/aic_data/... which we cannot resolve from the host
        # but are valid when training runs inside the container.
        target = os.readlink(src_videos)
        (args.dst / "videos").symlink_to(target)
        print(f"  copied videos symlink -> {target}")
    elif src_videos.exists():
        # Real directory: relative symlink for portability across host/container.
        target_rel = os.path.relpath(src_videos, args.dst)
        (args.dst / "videos").symlink_to(target_rel)
        print(f"  symlinked videos/ -> {target_rel}")
    else:
        print(f"  WARNING source has no videos/ entry")

    print()
    print(f"=== done: {args.dst} ===")
    print(f"  SC episodes smoothed: {sc_modified}/{sc_seen}")
    print(f"  next: train_smolvla.py --dataset-root {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
