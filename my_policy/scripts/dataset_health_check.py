#!/usr/bin/env python3
"""Health check for the merged v9-act dataset.

Reads ONLY the per-frame data parquet (no video decode, no lerobot
dependency) so it runs fast on 200k frames. Catches the kinds of
issues that silently corrupt IL training:

  * zero-action prefix frames (recorder hadn't received pose command yet)
  * action discontinuities (recorder dropped frames / jumps)
  * TCP-vs-action divergence (controller didn't follow command — bad
    supervision)
  * non-unit quaternions in either state or action
  * episode length outliers (truncated demos that snuck through)
  * per-target-vector coverage skew (model under-trains rare classes)
  * action-z range per episode (tells us which demos descend)

Run:
    pixi run python my_policy/scripts/dataset_health_check.py
    # or, since this needs no lerobot:
    python3 my_policy/scripts/dataset_health_check.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# State channel layout — matches build_act_dataset.py:KEEP_CHANNEL_GROUPS.
STATE_TCP_POS = slice(0, 3)
STATE_TCP_QUAT = slice(3, 7)            # x, y, z, w
ACTION_POS = slice(0, 3)
ACTION_QUAT = slice(3, 7)               # x, y, z, w


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/root/aic_data/v9_act_build/v9_act_merged"))
    args = p.parse_args()

    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    print(f"=== {args.dataset_root} ===")
    print(f"  total_episodes: {info['total_episodes']}")
    print(f"  total_frames  : {info['total_frames']}")
    print(f"  fps           : {info['fps']}")
    print()

    data_files = sorted((args.dataset_root / "data").rglob("*.parquet"))
    print(f"Reading {len(data_files)} data parquet file(s)...")
    tables = [pq.read_table(f) for f in data_files]
    if len(tables) > 1:
        import pyarrow as pa
        t = pa.concat_tables(tables)
    else:
        t = tables[0]
    print(f"  loaded {t.num_rows} rows")

    action = np.stack(t.column("action").to_pylist()).astype(np.float32)        # [N, 7]
    state = np.stack(t.column("observation.state").to_pylist()).astype(np.float32)  # [N, 44]
    ep_idx = np.array(t.column("episode_index").to_pylist(), dtype=np.int64)
    frame_idx = np.array(t.column("frame_index").to_pylist(), dtype=np.int64)
    task_idx = np.array(t.column("task_index").to_pylist(), dtype=np.int64)

    print(f"  action shape: {action.shape}, state shape: {state.shape}")

    # Episode bounds derived from episode_index changes.
    ep_starts = np.r_[0, np.where(np.diff(ep_idx) != 0)[0] + 1]
    ep_ends = np.r_[ep_starts[1:], len(ep_idx)]
    n_eps = len(ep_starts)
    print(f"  derived episodes: {n_eps}")
    print()

    # --- Check 1: zero-action prefix frames -------------------------------
    print("--- Check 1: zero-action prefix frames "
          "(recorder hadn't received pose command yet) ---")
    zero_prefix_lens = []
    for s, e in zip(ep_starts, ep_ends):
        # Count leading frames with ||action[pos]|| ≈ 0.
        ep_act = action[s:e, ACTION_POS]
        norms = np.linalg.norm(ep_act, axis=1)
        prefix = 0
        while prefix < len(norms) and norms[prefix] < 1e-3:
            prefix += 1
        zero_prefix_lens.append(prefix)
    zp = np.array(zero_prefix_lens)
    print(f"  episodes with zero-action prefix: {(zp > 0).sum()} / {n_eps}")
    if (zp > 0).any():
        print(f"  prefix length: max={zp.max()} frames, "
              f"median (over those affected)={int(np.median(zp[zp>0]))}, "
              f"total wasted frames={zp.sum()}")
    print()

    # --- Check 2: action discontinuities (large per-step jumps) -----------
    # Threshold: 50 mm position step OR ||delta_quat|| > 0.5 (large rot jump).
    print("--- Check 2: action discontinuities (frame-to-frame jumps) ---")
    POS_JUMP_M = 0.05
    QUAT_JUMP = 0.5
    big_jumps_per_ep: dict[int, int] = {}
    max_pos_jump = 0.0
    max_quat_jump = 0.0
    for ep_i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        if e - s < 2:
            continue
        ep_act = action[s:e]
        # Skip the leading zero-action frames so a 0→first-real-action jump
        # doesn't count.
        skip = zero_prefix_lens[ep_i]
        if e - (s + skip) < 2:
            continue
        a = ep_act[skip:]
        d_pos = np.linalg.norm(np.diff(a[:, ACTION_POS], axis=0), axis=1)
        d_quat = np.linalg.norm(np.diff(a[:, ACTION_QUAT], axis=0), axis=1)
        big = (d_pos > POS_JUMP_M) | (d_quat > QUAT_JUMP)
        if big.any():
            big_jumps_per_ep[ep_i] = int(big.sum())
        max_pos_jump = max(max_pos_jump, float(d_pos.max()))
        max_quat_jump = max(max_quat_jump, float(d_quat.max()))
    print(f"  episodes with any frame-to-frame action jump > "
          f"{POS_JUMP_M*1000:.0f}mm pos or {QUAT_JUMP} quat: "
          f"{len(big_jumps_per_ep)} / {n_eps}")
    if big_jumps_per_ep:
        worst = sorted(big_jumps_per_ep.items(), key=lambda x: -x[1])[:5]
        print(f"  worst 5 episodes (by jump count): {worst}")
    print(f"  global max pos jump : {max_pos_jump*1000:.1f} mm")
    print(f"  global max quat jump: {max_quat_jump:.3f}")
    print()

    # --- Check 3: TCP-vs-action divergence (controller tracking quality) --
    print("--- Check 3: TCP vs action divergence per episode "
          "(median ||action-tcp|| in mm, ignoring zero-prefix) ---")
    div_per_ep = []
    for ep_i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        skip = zero_prefix_lens[ep_i]
        if e - (s + skip) < 1:
            continue
        a = action[s+skip:e, ACTION_POS]
        c = state[s+skip:e, STATE_TCP_POS]
        d = np.linalg.norm(a - c, axis=1)
        div_per_ep.append(float(np.median(d)))
    div = np.array(div_per_ep)
    print(f"  median per-episode tracking error (mm): "
          f"p50={np.percentile(div, 50)*1000:.1f}, "
          f"p90={np.percentile(div, 90)*1000:.1f}, "
          f"p99={np.percentile(div, 99)*1000:.1f}, "
          f"max={div.max()*1000:.1f}")
    print(f"  (small = controller tracked well; >50mm sustained "
          f"= action signal diverges from observed motion)")
    print()

    # --- Check 4: quaternion sanity ---------------------------------------
    print("--- Check 4: quaternion unit-norm violations ---")
    state_quat = state[:, STATE_TCP_QUAT]
    action_quat = action[:, ACTION_QUAT]
    state_norms = np.linalg.norm(state_quat, axis=1)
    action_norms = np.linalg.norm(action_quat, axis=1)
    # Skip zero-prefix action frames (norms are 0 there by construction).
    nonzero_action_mask = np.linalg.norm(action[:, ACTION_POS], axis=1) > 1e-3
    print(f"  state quat: range [{state_norms.min():.4f}, {state_norms.max():.4f}], "
          f"mean dev from 1.0 = {abs(state_norms - 1.0).mean():.6f}")
    print(f"  action quat (excl zero prefix): "
          f"range [{action_norms[nonzero_action_mask].min():.4f}, "
          f"{action_norms[nonzero_action_mask].max():.4f}], "
          f"mean dev from 1.0 = "
          f"{abs(action_norms[nonzero_action_mask] - 1.0).mean():.6f}")
    print()

    # --- Check 5: episode length distribution -----------------------------
    print("--- Check 5: episode length distribution ---")
    lengths = (ep_ends - ep_starts).astype(np.int64)
    print(f"  frames per episode: min={lengths.min()}, "
          f"p10={np.percentile(lengths, 10):.0f}, "
          f"p50={np.percentile(lengths, 50):.0f}, "
          f"p90={np.percentile(lengths, 90):.0f}, "
          f"max={lengths.max()}")
    print(f"  duration (s): min={lengths.min()/info['fps']:.1f}, "
          f"max={lengths.max()/info['fps']:.1f}")
    short = (lengths < 5 * info['fps']).sum()
    if short:
        print(f"  WARN: {short} episodes shorter than 5 s — likely "
              f"truncated/failed demos that slipped through clean filter")
    print()

    # --- Check 6: per-task coverage ---------------------------------------
    print("--- Check 6: per-task coverage (from task_index) ---")
    # task_index is per-frame, but constant per episode → use first frame.
    ep_task = task_idx[ep_starts]
    counter = Counter(ep_task.tolist())
    tasks_table = pq.read_table(args.dataset_root / "meta" / "tasks.parquet")
    # tasks.parquet has columns: task_index (int), task (str description).
    ti_to_str = {int(ti): s for ti, s in zip(
        tasks_table.column("task_index").to_pylist(),
        tasks_table.column("task").to_pylist(),
    )}
    print(f"  tasks defined: {len(ti_to_str)}")
    sfp_total = sc_total = 0
    for ti, count in sorted(counter.items()):
        ts = ti_to_str.get(ti, "?")
        marker = ""
        if "sfp" in ts:
            sfp_total += count
        if "sc " in ts or " sc_" in ts or "sc plug" in ts.lower() or "sc_port_base" in ts:
            sc_total += count
        # Highlight outliers (more than 2x deviation from mean count)
        mean_count = sum(counter.values()) / len(counter)
        if count < 0.5 * mean_count:
            marker = "  <-- LOW"
        elif count > 2.0 * mean_count:
            marker = "  <-- HIGH"
        print(f"  task {ti:2d}: {count:4d} eps  {ts}{marker}")
    print(f"  sfp episodes total: {sfp_total}, sc episodes total: {sc_total}")
    print()

    # --- Check 7: action-z trajectory per episode -------------------------
    print("--- Check 7: action z-drop per episode "
          "(start z minus end z; positive = descent happened) ---")
    drops = []
    end_zs = []
    for ep_i, (s, e) in enumerate(zip(ep_starts, ep_ends)):
        skip = zero_prefix_lens[ep_i]
        if e - (s + skip) < 2:
            continue
        z = action[s+skip:e, 2]
        drops.append(z[0] - z[-1])
        end_zs.append(z[-1])
    drops = np.array(drops)
    end_zs = np.array(end_zs)
    print(f"  z-drop per episode (m): "
          f"min={drops.min():.3f}, "
          f"p10={np.percentile(drops, 10):.3f}, "
          f"p50={np.percentile(drops, 50):.3f}, "
          f"p90={np.percentile(drops, 90):.3f}, "
          f"max={drops.max():.3f}")
    print(f"  end-of-episode action z (m): "
          f"min={end_zs.min():.3f}, p50={np.percentile(end_zs, 50):.3f}, "
          f"max={end_zs.max():.3f}")
    weak_descent = (drops < 0.05).sum()
    if weak_descent:
        print(f"  WARN: {weak_descent} episodes with <50mm z-descent — "
              f"may not show full insertion")
    print()

    print("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
