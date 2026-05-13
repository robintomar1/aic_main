#!/usr/bin/env python3
"""Augment a SmolVLA-ready dataset with causal conditioning channels.

Adds two optional things to `observation.state`:

  1. **prev_action conditioning** — append the K most recent dispatched
     actions (in port-local frame, matching the action dim of the
     dataset). Most recent action first. For the first K frames of an
     episode, missing slots are zero-padded. K controllable via --prev-K.

  2. **Phase indicator** — append a 4-class one-hot describing the
     trial phase, derived heuristically from port-local TCP_z and
     wrench magnitude. Classes:
       [0] APPROACH    : TCP_z < z_approach_thresh, |F| < f_contact_thresh
       [1] DESCEND     : z_approach_thresh ≤ TCP_z < z_chamfer_thresh, low F
       [2] CONTACT     : |F| ≥ f_contact_thresh (anywhere) — spiral/seat
       [3] INSERTED    : TCP_z ≥ z_inserted_thresh, low F

  Thresholds parameterizable. Single class active per frame.

Outputs a new LeRobotDataset directory mirroring the source layout
(data + meta + tasks + train/val split + video symlinks). State dim grows
from 26 → 26 + K*7 + 4 by default (K=1 → 37 dims; K=4 → 58 dims).

Usage:
    pixi run python my_policy/scripts/smolvla/build_conditioned_dataset.py \\
        --src /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset_with_corrections \\
        --dst /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset_with_corrections_cond \\
        --prev-K 1 \\
        --phase-indicator

To skip a feature pass --prev-K 0 or --no-phase-indicator.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Phase classifier
# ---------------------------------------------------------------------------

# Channel indices in the source 26-dim state.
SRC_TCP_Z_IDX = 2
SRC_WRENCH_F_START = 20  # wrench.fx,fy,fz are at [20:23]
SRC_WRENCH_F_END = 23

PHASE_APPROACH = 0
PHASE_DESCEND = 1
PHASE_CONTACT = 2
PHASE_INSERTED = 3
PHASE_NAMES = ["approach", "descend", "contact", "inserted"]


def classify_phase(
    state_26: np.ndarray,
    z_approach_thresh: float,
    z_chamfer_thresh: float,
    z_inserted_thresh: float,
    f_contact_thresh: float,
) -> np.ndarray:
    """Return integer phase id per frame, shape (N,).

    Decision tree (first match wins):
      * |F| ≥ f_contact_thresh           → CONTACT
      * tcp_z ≥ z_inserted_thresh        → INSERTED
      * tcp_z ≥ z_chamfer_thresh         → DESCEND
      * else                             → APPROACH
    """
    tcp_z = state_26[:, SRC_TCP_Z_IDX]
    f = state_26[:, SRC_WRENCH_F_START:SRC_WRENCH_F_END]
    fmag = np.linalg.norm(f, axis=1)
    phase = np.full(tcp_z.shape, PHASE_APPROACH, dtype=np.int32)
    phase = np.where(tcp_z >= z_chamfer_thresh, PHASE_DESCEND, phase)
    phase = np.where(tcp_z >= z_inserted_thresh, PHASE_INSERTED, phase)
    # Contact overrides z-based phase because force is the more reliable signal
    # for "we're seating into the chamfer."
    phase = np.where(fmag >= f_contact_thresh, PHASE_CONTACT, phase)
    return phase


def phase_to_onehot(phase: np.ndarray) -> np.ndarray:
    """(N,) int → (N, 4) float32 one-hot."""
    onehot = np.zeros((phase.shape[0], 4), dtype=np.float32)
    onehot[np.arange(phase.shape[0]), phase] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# Prev-action history
# ---------------------------------------------------------------------------

def build_prev_action_history(
    actions: np.ndarray,
    episode_indices: np.ndarray,
    k: int,
) -> np.ndarray:
    """For each row, return the previous K actions concatenated.

    Most-recent first. For the first K frames of each episode, missing
    slots are zero-padded.

    Args:
        actions: (N, 7) array of actions per frame.
        episode_indices: (N,) int episode id per frame.
        k: how many previous actions to include.

    Returns:
        (N, k * 7) float32 array, where columns [0:7] are action[t-1],
        [7:14] are action[t-2], ..., [(k-1)*7:k*7] are action[t-k].
    """
    n, a_dim = actions.shape
    out = np.zeros((n, k * a_dim), dtype=np.float32)
    for offset in range(1, k + 1):
        # action at t - offset.
        shifted = np.zeros_like(actions)
        shifted[offset:] = actions[:-offset]
        # Zero out where the previous row belongs to a different episode.
        if offset < n:
            same_ep = np.zeros(n, dtype=bool)
            same_ep[offset:] = episode_indices[offset:] == episode_indices[:-offset]
            shifted = shifted * same_ep[:, None].astype(np.float32)
        else:
            shifted = np.zeros_like(actions)
        col_start = (offset - 1) * a_dim
        out[:, col_start : col_start + a_dim] = shifted.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Dataset I/O helpers
# ---------------------------------------------------------------------------

def build_state_names(prev_k: int, add_phase: bool) -> list[str]:
    names: list[str] = [
        "tcp_pose.position.x", "tcp_pose.position.y", "tcp_pose.position.z",
        "tcp_pose.orientation.x", "tcp_pose.orientation.y",
        "tcp_pose.orientation.z", "tcp_pose.orientation.w",
        "tcp_velocity.linear.x", "tcp_velocity.linear.y", "tcp_velocity.linear.z",
        "tcp_velocity.angular.x", "tcp_velocity.angular.y", "tcp_velocity.angular.z",
        *[f"joint_positions.{i}" for i in range(7)],
        "wrench.fx", "wrench.fy", "wrench.fz",
        "wrench.tx", "wrench.ty", "wrench.tz",
    ]
    assert len(names) == 26
    for k in range(1, prev_k + 1):
        names.extend([
            f"prev_action[{k}].position.x", f"prev_action[{k}].position.y",
            f"prev_action[{k}].position.z",
            f"prev_action[{k}].orientation.x", f"prev_action[{k}].orientation.y",
            f"prev_action[{k}].orientation.z", f"prev_action[{k}].orientation.w",
        ])
    if add_phase:
        names.extend([f"phase.{p}" for p in PHASE_NAMES])
    return names


def per_array_stats(arr: np.ndarray) -> dict:
    """Statistics matching lerobot's stats.json convention."""
    return {
        "min": arr.min(axis=0).astype(np.float32),
        "max": arr.max(axis=0).astype(np.float32),
        "mean": arr.mean(axis=0).astype(np.float32),
        "std": arr.std(axis=0).astype(np.float32),
        "count": np.array([arr.shape[0]], dtype=np.float32),
    }


def stats_to_jsonable(d: dict) -> dict:
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="Source SmolVLA dataset root (26-dim state).")
    p.add_argument("--dst", type=Path, required=True,
                   help="Destination dataset root. Created fresh.")
    p.add_argument("--prev-K", type=int, default=1,
                   help="Number of previous actions to concatenate into state. "
                        "0 disables. Default 1 (most recent action only).")
    p.add_argument("--phase-indicator", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Append 4-class phase one-hot (approach/descend/contact/inserted). "
                        "Default ON. Pass --no-phase-indicator to disable.")
    p.add_argument("--z-approach-thresh", type=float, default=-0.15,
                   help="port-local TCP_z below this (more negative) = APPROACH.")
    p.add_argument("--z-chamfer-thresh", type=float, default=-0.05,
                   help="port-local TCP_z above this = DESCEND (else APPROACH).")
    p.add_argument("--z-inserted-thresh", type=float, default=-0.01,
                   help="port-local TCP_z above this = INSERTED (overrides DESCEND).")
    p.add_argument("--f-contact-thresh", type=float, default=8.0,
                   help="|F| above this (N) = CONTACT (overrides z-based phase).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite --dst if it exists.")
    args = p.parse_args()

    if not args.src.is_dir():
        sys.exit(f"--src does not exist: {args.src}")
    if args.prev_K < 0:
        sys.exit("--prev-K must be >= 0")

    if args.dst.exists():
        if args.force:
            print(f"--force: removing existing {args.dst}")
            shutil.rmtree(args.dst)
        else:
            sys.exit(f"--dst exists: {args.dst} (use --force to overwrite)")
    args.dst.mkdir(parents=True)

    # --- 1. Read source ---------------------------------------------------
    print(f"reading source: {args.src}")
    src_info = json.loads((args.src / "meta" / "info.json").read_text())
    src_state_names = src_info["features"]["observation.state"]["names"]
    if len(src_state_names) != 26:
        sys.exit(
            f"expected 26 state channels in source, got {len(src_state_names)}. "
            f"Use a SmolVLA-ready dataset (post make_smolvla_dataset.py)."
        )

    src_data = pq.read_table(str(args.src / "data" / "chunk-000" / "file-000.parquet"))
    n_rows = src_data.num_rows
    print(f"  source frames: {n_rows}")
    print(f"  source state dim: 26")

    state_26 = np.stack([
        np.asarray(r, dtype=np.float32)
        for r in src_data["observation.state"].to_pylist()
    ])
    actions = np.stack([
        np.asarray(r, dtype=np.float32)
        for r in src_data["action"].to_pylist()
    ])
    episode_index = src_data["episode_index"].to_numpy().astype(np.int64)

    # --- 2. Compute new state pieces -------------------------------------
    pieces = [state_26]
    if args.prev_K > 0:
        prev = build_prev_action_history(actions, episode_index, args.prev_K)
        pieces.append(prev)
        print(f"  prev_action history: K={args.prev_K} → +{args.prev_K * 7} channels")

    phase_int = phase_distribution = None
    if args.phase_indicator:
        phase_int = classify_phase(
            state_26,
            z_approach_thresh=args.z_approach_thresh,
            z_chamfer_thresh=args.z_chamfer_thresh,
            z_inserted_thresh=args.z_inserted_thresh,
            f_contact_thresh=args.f_contact_thresh,
        )
        pieces.append(phase_to_onehot(phase_int))
        counts = np.bincount(phase_int, minlength=4)
        phase_distribution = dict(zip(PHASE_NAMES, counts.tolist()))
        print(f"  phase indicator: +4 channels")
        for name, c in phase_distribution.items():
            print(f"    {name:>10}: {c:>7} frames ({100 * c / n_rows:.1f}%)")

    new_state = np.concatenate(pieces, axis=1).astype(np.float32)
    new_state_dim = new_state.shape[1]
    new_names = build_state_names(args.prev_K, args.phase_indicator)
    assert len(new_names) == new_state_dim, \
        f"name/value count mismatch: {len(new_names)} vs {new_state_dim}"
    print(f"  new state dim: {new_state_dim}")

    # --- 3. Write data parquet -------------------------------------------
    out_data_dir = args.dst / "data" / "chunk-000"
    out_data_dir.mkdir(parents=True)

    cols = {}
    for c in src_data.column_names:
        if c == "observation.state":
            continue
        cols[c] = src_data[c]
    cols["observation.state"] = pa.array(
        new_state.tolist(),
        type=pa.list_(pa.float32(), new_state_dim),
    )
    new_table = pa.table(cols)
    pq.write_table(new_table, str(out_data_dir / "file-000.parquet"))
    print(f"  wrote {out_data_dir / 'file-000.parquet'}")

    # --- 4. info.json -----------------------------------------------------
    new_info = dict(src_info)
    new_features = dict(new_info["features"])
    state_feat = dict(new_features["observation.state"])
    state_feat["shape"] = [new_state_dim]
    state_feat["names"] = new_names
    new_features["observation.state"] = state_feat
    new_info["features"] = new_features
    # Mark provenance.
    new_info["conditioning"] = {
        "source": str(args.src),
        "prev_K": args.prev_K,
        "phase_indicator": args.phase_indicator,
        "phase_thresholds": {
            "z_approach": args.z_approach_thresh,
            "z_chamfer": args.z_chamfer_thresh,
            "z_inserted": args.z_inserted_thresh,
            "f_contact": args.f_contact_thresh,
        } if args.phase_indicator else None,
        "phase_distribution": phase_distribution,
    }
    (args.dst / "meta").mkdir(parents=True, exist_ok=True)
    (args.dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))
    print(f"  wrote meta/info.json (new state shape={new_state_dim})")

    # --- 5. Stats.json ----------------------------------------------------
    src_stats = json.loads((args.src / "meta" / "stats.json").read_text())
    new_stats = dict(src_stats)
    new_stats["observation.state"] = stats_to_jsonable(per_array_stats(new_state))
    (args.dst / "meta" / "stats.json").write_text(json.dumps(new_stats, indent=2))
    print(f"  wrote meta/stats.json (recomputed observation.state)")

    # --- 6. Per-episode meta (rewrite observation.state stats) -----------
    src_eps_parquet = args.src / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if src_eps_parquet.exists():
        eps_table = pq.read_table(str(src_eps_parquet)).to_pylist()
        # For each episode, recompute per-episode state stats from new_state.
        for rec in eps_table:
            ep = int(rec["episode_index"])
            mask = episode_index == ep
            ep_state = new_state[mask]
            stats = per_array_stats(ep_state)
            for stat_key in ("min", "max", "mean", "std"):
                rec[f"stats/observation.state/{stat_key}"] = stats[stat_key].tolist()
            rec["stats/observation.state/count"] = [int(stats["count"][0])]
        out_eps_dir = args.dst / "meta" / "episodes" / "chunk-000"
        out_eps_dir.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(eps_table),
            str(out_eps_dir / "file-000.parquet"),
        )
        print(f"  wrote per-episode meta ({len(eps_table)} episodes)")

    # --- 7. Copy or symlink everything else ------------------------------
    # tasks.parquet — copy (small).
    src_tasks = args.src / "meta" / "tasks.parquet"
    if src_tasks.exists():
        shutil.copy(src_tasks, args.dst / "meta" / "tasks.parquet")
        print(f"  copied meta/tasks.parquet")

    # train/val splits — copy.
    for f in ("train_episodes.json", "val_episodes.json",
              "source_episode_map.json"):
        sp = args.src / f
        if sp.exists():
            shutil.copy(sp, args.dst / f)

    # videos — symlink to source (avoid duplicating GB of video data).
    src_videos = args.src / "videos"
    if src_videos.exists():
        target = src_videos.resolve() if src_videos.is_symlink() else src_videos
        (args.dst / "videos").symlink_to(target)
        print(f"  symlinked videos -> {target}")

    print()
    print(f"DONE: {args.dst}")
    print(f"  state dim {26} -> {new_state_dim} "
          f"(prev_K={args.prev_K}, phase={'yes' if args.phase_indicator else 'no'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
