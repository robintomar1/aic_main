#!/usr/bin/env python3
"""Build a port-local-frame ACT dataset from one raw recorder batch.

This is the *port-local* sibling of `build_act_dataset.py`. The two scripts
produce the same 44-channel observation.state and 7-dim action shape, but
this one re-expresses every spatial quantity in the target port's coordinate
frame (using the recorded `groundtruth.port_pose`) before writing.

What gets transformed (per frame, before channel selection):
  * `tcp_pose.{position,orientation}` — TCP pose, base_link → port-local.
  * `tcp_velocity.{linear,angular}` — TCP twist, rotated into port axes.
  * `wrench.{f,t}{x,y,z}` — F/T sensor reading, rotated into port axes
    (assumes sensor frame ≈ TCP frame; documented in transforms.py header).
  * `action.pose.{position,orientation}` — TCP pose target, base_link → port-local.

What stays the same (frame-invariant):
  * `tcp_error.*` — controller error, already TCP-relative.
  * `joint_positions.*` — joint space, no frame.
  * `task_*` — one-hot vectors.

What gets dropped (leakage, same as build_act_dataset.py):
  * `groundtruth.port_pose.*` — the model must NOT see the answer.
  * `groundtruth.plug_pose.*`, `meta.insertion_success`.

Episodes with corrupt `groundtruth.port_pose` (e.g. memory cites batch_500_a
where x≤0 and qw≈0) are detected and skipped — see `is_port_pose_valid`.

Usage (mirrors build_act_dataset.py):
    python3 my_policy/scripts/make_port_local_dataset.py \\
        --collection-dir /root/aic_data \\
        --batch batch_100_a \\
        --out-root /root/aic_data/v9_act_build \\
        --clean-episodes-json /root/aic_data/v9_act_build/batch_100_a_act_clean_episodes.json

Pure pyarrow + numpy + yaml. No torch, no lerobot at preprocess time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.act.labels import (  # noqa: E402
    ACT_TASK_VECTOR_DIM,
    ACT_VALID_TARGETS,
    encode_task_vector,
    task_channel_names,
    task_string_for,
)
from my_policy.localizer.labels import match_episodes_to_trials  # noqa: E402
from my_policy.port_local.dataset_io import (  # noqa: E402
    SRC_PORT_POSE_SLICE,
    SRC_TCP_POSE_SLICE,
    SRC_TCP_VEL_SLICE,
    SRC_WRENCH_SLICE,
    EXPECTED_RAW_STATE_DIM,
    is_action_valid,
    is_port_pose_valid,
)
from my_policy.port_local.transforms import (  # noqa: E402
    FrameInputs,
    transform_frame,
)


# ---------------------------------------------------------------------------
# Helpers copied from `build_act_dataset.py` (verified 2026-05-08 to match
# the raw recorder layout). Copied rather than imported because
# `my_policy/scripts/` is not a Python package — see commit history.
# Keep these in sync with build_act_dataset.py if either changes.
# ---------------------------------------------------------------------------


KEEP_CHANNEL_GROUPS: list[tuple[str, list[str]]] = [
    ("tcp_pose", [
        "tcp_pose.position.x", "tcp_pose.position.y", "tcp_pose.position.z",
        "tcp_pose.orientation.x", "tcp_pose.orientation.y",
        "tcp_pose.orientation.z", "tcp_pose.orientation.w",
    ]),
    ("tcp_velocity", [
        "tcp_velocity.linear.x", "tcp_velocity.linear.y", "tcp_velocity.linear.z",
        "tcp_velocity.angular.x", "tcp_velocity.angular.y", "tcp_velocity.angular.z",
    ]),
    ("tcp_error", [
        "tcp_error.x", "tcp_error.y", "tcp_error.z",
        "tcp_error.rx", "tcp_error.ry", "tcp_error.rz",
    ]),
    ("joint_positions", [f"joint_positions.{i}" for i in range(7)]),
    ("wrench", [f"wrench.f{c}" for c in "xyz"] + [f"wrench.t{c}" for c in "xyz"]),
]


def build_new_state_names() -> list[str]:
    out: list[str] = []
    for _, group in KEEP_CHANNEL_GROUPS:
        out.extend(group)
    out.extend(task_channel_names())
    assert len(out) == 44, f"expected 44 channels, got {len(out)}"
    return out


def slice_indices(state_names: list[str], wanted: list[str]) -> list[int]:
    out: list[int] = []
    for n in wanted:
        if n not in state_names:
            raise KeyError(f"channel {n!r} not in source observation.state.names")
        out.append(state_names.index(n))
    return out


def per_episode_stats(values: np.ndarray) -> dict:
    """Min/max/mean/std + count for a (n, d) or (n,) array. Matches the
    shape lerobot expects in episode meta records.
    """
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [arr.shape[0]],
    }


def episode_split(
    episodes: list[int], val_fraction: float = 0.2, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Episode-level split. Matches build_act_dataset.episode_split exactly
    so the same seed produces the same split (apples-to-apples training
    comparison vs. the world-frame baseline).
    """
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(sorted(episodes)))
    n_val = max(1, int(round(len(perm) * val_fraction)))
    val_eps = sorted(int(e) for e in perm[:n_val])
    train_eps = sorted(int(e) for e in perm[n_val:])
    return train_eps, val_eps


def _port_type_for(port_name: str) -> str:
    """Convention from build_act_dataset.py: port_type is "sc" iff
    port_name == "sc_port_base", else "sfp"."""
    return "sc" if port_name == "sc_port_base" else "sfp"


def build_tasks_parquet(out_meta_dir: Path) -> dict[tuple[str, str], int]:
    """Write a deterministic 12-row tasks.parquet keyed by ACT_VALID_TARGETS
    order. Returns {(target_module, port_name) → task_index}.

    `ACT_VALID_TARGETS` is a tuple of `(mount, port_name)` pairs (verified
    2026-05-08 in act/labels.py — 12 entries, no port_type in the tuple).
    """
    rows: list[dict] = []
    out: dict[tuple[str, str], int] = {}
    for i, (mod, port) in enumerate(ACT_VALID_TARGETS):
        ptype = _port_type_for(port)
        rows.append({"task_index": i, "task": task_string_for(mod, port, ptype)})
        out[(mod, port)] = i
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(out_meta_dir / "tasks.parquet"))
    return out


def _transform_state_and_action(
    state_47: np.ndarray, action_7: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply port-local transform to one (state, action) pair.

    Returns (state_47_port_local, action_7_port_local, is_valid).

    The output state shape stays at 47 — the frame transform doesn't
    change shape, only values. Channel selection (47 → 44) and task-vec
    append happen in `main()` after this returns.

    Wrench is rewritten in-place at indices [26..31].
    Groundtruth slots [32..45] and meta [46] are left untouched
    (they get dropped at channel-selection time).

    `is_valid=False` if the port pose OR action is corrupt; caller should
    skip the frame.
    """
    port_pose = state_47[SRC_PORT_POSE_SLICE]
    if not is_port_pose_valid(port_pose):
        return state_47, action_7, False
    if not is_action_valid(action_7):
        return state_47, action_7, False

    inp = FrameInputs(
        tcp_pose_baselink=state_47[SRC_TCP_POSE_SLICE].astype(np.float64),
        tcp_velocity_baselink=state_47[SRC_TCP_VEL_SLICE].astype(np.float64),
        wrench_sensorframe=state_47[SRC_WRENCH_SLICE].astype(np.float64),
        action_baselink=action_7.astype(np.float64),
        port_pose_baselink=port_pose.astype(np.float64),
    )
    out = transform_frame(inp)

    new_state = state_47.copy()
    new_state[SRC_TCP_POSE_SLICE] = out.tcp_pose_portframe.astype(np.float32)
    new_state[SRC_TCP_VEL_SLICE] = out.tcp_velocity_portframe.astype(np.float32)
    new_state[SRC_WRENCH_SLICE] = out.wrench_portframe.astype(np.float32)
    new_action = out.action_portframe.astype(np.float32)
    return new_state, new_action, True


# ---------------------------------------------------------------------------
# Main — structurally similar to build_act_dataset.main, with the transform
# layer injected before channel selection. Episode-renumbering, task-vec
# append, parquet write, video symlink, episode-meta, info.json all
# reuse build_act_dataset's patterns / helpers.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--collection-dir", type=Path, required=True)
    p.add_argument("--batch", type=str, required=True)
    p.add_argument(
        "--out-root", type=Path, required=True,
        help="Parent dir for the output dataset; "
             "creates <out-root>/<batch>_port_local_dataset/.",
    )
    p.add_argument("--clean-episodes-json", type=Path, default=None)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    args = p.parse_args()

    src_root = args.collection_dir / args.batch
    src_yaml = args.collection_dir / f"{args.batch}.yaml"
    src_summary = args.collection_dir / f"{args.batch}_logs" / "summary.json"
    for p_, label in [
        (src_root, "dataset_root"),
        (src_yaml, "batch_yaml"),
        (src_summary, "summary_json"),
    ]:
        if not p_.exists():
            print(f"error: {label} not found at {p_}", file=sys.stderr)
            return 1

    out_root = args.out_root / f"{args.batch}_port_local_dataset"
    out_data_dir = out_root / "data" / "chunk-000"
    out_meta_dir = out_root / "meta"
    out_meta_eps_dir = out_meta_dir / "episodes" / "chunk-000"
    out_videos_dir = out_root / "videos"
    for d in (out_data_dir, out_meta_dir, out_meta_eps_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. Read source ------------------------------------------------
    print(f"reading source: {src_root}")
    src_info = json.loads((src_root / "meta" / "info.json").read_text())
    src_state_names = src_info["features"]["observation.state"]["names"]
    if len(src_state_names) != 47:
        print(
            f"error: source observation.state has {len(src_state_names)} "
            f"channels; expected 47 (raw recorder layout). "
            f"Are you pointing at a *_act_dataset (post-preprocess)? "
            f"This script needs the RAW batch dataset.",
            file=sys.stderr,
        )
        return 1
    keep_idx = slice_indices(
        src_state_names,
        [n for _, group in KEEP_CHANNEL_GROUPS for n in group],
    )

    src_data = pq.read_table(
        str(src_root / "data" / "chunk-000" / "file-000.parquet")
    )
    print(f"  source frames: {src_data.num_rows}")

    # --- 2. Episode + task lookup -------------------------------------
    cfg = yaml.safe_load(src_yaml.read_text())
    summary = json.loads(src_summary.read_text())
    ep_to_trial = match_episodes_to_trials(summary, cfg["trials"])

    keep_episodes: set[int] | None = None
    if args.clean_episodes_json is not None:
        keep_episodes = set(json.loads(args.clean_episodes_json.read_text()))
        print(f"  clean filter: keeping {len(keep_episodes)} episodes")

    task_index_for_pair = build_tasks_parquet(out_meta_dir)
    print(f"  wrote meta/tasks.parquet ({len(task_index_for_pair)} entries)")

    # --- 3. Apply per-frame port-local transform ---------------------
    eps_col = src_data["episode_index"].to_numpy().astype(np.int64)
    raw_states = np.stack(
        [np.asarray(r, dtype=np.float32) for r in src_data["observation.state"].to_pylist()]
    )  # (n_frames, 47)
    raw_actions = np.stack(
        [np.asarray(r, dtype=np.float32) for r in src_data["action"].to_pylist()]
    )  # (n_frames, 7)

    n_frames = src_data.num_rows

    # --- Apply Fix 1 from clean_act_dataset.py: patch stale leading action
    # frames per episode. The recorder occasionally captures the previous
    # trial's /aic_controller/pose_commands at the very start of a new
    # episode, before the new policy has published. Detection: action.position
    # disagrees with state.tcp_pose.position by >50 mm. Fix: overwrite
    # leading bad frames with the first "good" frame's action.
    # Per memory `project_aic_act_dataset.md`, ~0.06% of frames are affected.
    # This fix MUST run before transform_frame, otherwise the port-local
    # action inherits the stale-frame error.
    STALE_POS_THRESHOLD_M = 0.05
    n_stale_frames = 0
    n_stale_episodes = 0
    for ep in sorted(np.unique(eps_col).tolist()):
        ep_global_idx = np.where(eps_col == ep)[0]
        if len(ep_global_idx) == 0:
            continue
        s = int(ep_global_idx[0])
        e = int(ep_global_idx[-1]) + 1
        first_good = None
        for j in range(min(20, e - s)):
            a_pos = raw_actions[s + j, :3]
            st_pos = raw_states[s + j, :3]
            if (np.linalg.norm(a_pos) > 1e-3
                    and np.linalg.norm(a_pos - st_pos) <= STALE_POS_THRESHOLD_M):
                first_good = j
                break
        if first_good is None or first_good == 0:
            continue
        replacement = raw_actions[s + first_good].copy()
        raw_actions[s:s + first_good] = replacement
        n_stale_frames += first_good
        n_stale_episodes += 1
    if n_stale_frames:
        print(f"  Fix 1 (stale-leading): patched {n_stale_frames} frames "
              f"in {n_stale_episodes} episodes")

    transformed_states = np.zeros_like(raw_states)
    transformed_actions = np.zeros_like(raw_actions)
    transform_valid = np.zeros(n_frames, dtype=bool)

    n_invalid_port = 0
    n_invalid_action = 0
    for i in range(n_frames):
        # Inline the validity checks here too so we can attribute the failure mode.
        port_ok = is_port_pose_valid(raw_states[i][SRC_PORT_POSE_SLICE])
        action_ok = is_action_valid(raw_actions[i])
        if not port_ok:
            n_invalid_port += 1
            transformed_states[i] = raw_states[i]
            transformed_actions[i] = raw_actions[i]
            transform_valid[i] = False
            continue
        if not action_ok:
            n_invalid_action += 1
            transformed_states[i] = raw_states[i]
            transformed_actions[i] = raw_actions[i]
            transform_valid[i] = False
            continue
        new_state, new_action, ok = _transform_state_and_action(
            raw_states[i], raw_actions[i]
        )
        transformed_states[i] = new_state
        transformed_actions[i] = new_action
        transform_valid[i] = ok
    if n_invalid_port or n_invalid_action:
        print(f"  WARNING: dropping invalid frames — "
              f"{n_invalid_port} bad port_pose, "
              f"{n_invalid_action} bad action (typically pre-first-command "
              f"frames at episode 0)")

    # --- Apply Fix 2 from clean_act_dataset.py: action quaternion sign
    # canonicalization. Per-frame: if dot(action_quat_port, state_quat_port) < 0,
    # negate the action quat. q and -q are the same rotation, so this doesn't
    # change the rotation; it only forces a consistent hemisphere convention
    # across the dataset. Without this, the model can't predict hemisphere from
    # the input (no signal) — caused the v1 "wrist locked" failure.
    #
    # Note: my rotmat_to_quat_xyzw picks w>=0 for both state and action, which
    # implicitly canonicalizes most frames (raw 45% mismatch → ~0.2% post-
    # transform). But w-positive alone doesn't imply same hemisphere for
    # *pairs* of quats (e.g. qx=+0.9,w=+0.4 vs qx=-0.9,w=+0.4 both have w>=0
    # but dot<0). Empirically validated 2026-05-08: ~68/37178 frames in
    # batch_100_a are still dot<0 after transform; this fix takes them to 0.
    valid_mask_for_quat_fix = transform_valid
    if valid_mask_for_quat_fix.any():
        s_q = transformed_states[valid_mask_for_quat_fix, 3:7]
        a_q = transformed_actions[valid_mask_for_quat_fix, 3:7]
        dots = (s_q * a_q).sum(axis=1)
        flip_mask_local = dots < 0
        n_canonicalized = int(flip_mask_local.sum())
        if n_canonicalized:
            global_idx = np.where(valid_mask_for_quat_fix)[0]
            flip_global = global_idx[flip_mask_local]
            transformed_actions[flip_global, 3:7] = -transformed_actions[flip_global, 3:7]
            print(f"  Fix 2 (quat-sign): canonicalized {n_canonicalized} action quats "
                  f"({100*n_canonicalized/int(valid_mask_for_quat_fix.sum()):.2f}% of valid frames)")
        # Sanity: post-fix, no negative dots remain.
        new_dots = (transformed_states[valid_mask_for_quat_fix, 3:7]
                    * transformed_actions[valid_mask_for_quat_fix, 3:7]).sum(axis=1)
        n_still_bad = int((new_dots < -1e-9).sum())
        assert n_still_bad == 0, (
            f"Fix 2 sanity failed: {n_still_bad} frames still have negative dot"
        )

    # --- 4. Channel selection + task-vec append ----------------------
    n_kept_channels = sum(len(g) for _, g in KEEP_CHANNEL_GROUPS)
    assert n_kept_channels == 32, f"expected 32 source channels kept, got {n_kept_channels}"

    new_states = np.zeros((n_frames, 44), dtype=np.float32)
    new_task_indices = np.zeros(n_frames, dtype=np.int64)
    keep_mask = np.zeros(n_frames, dtype=bool)

    for ep in sorted(np.unique(eps_col).tolist()):
        if keep_episodes is not None and int(ep) not in keep_episodes:
            continue
        trial_key = ep_to_trial.get(int(ep))
        if trial_key is None:
            print(f"warning: no trial_key for ep {ep}; skipping")
            continue
        task = cfg["trials"][trial_key]["tasks"]["task_1"]
        task_vec = encode_task_vector(
            task["target_module_name"], task["port_name"], task["port_type"],
        )
        ti = task_index_for_pair[(task["target_module_name"], task["port_name"])]
        ep_mask = (eps_col == ep) & transform_valid
        if int(ep_mask.sum()) == 0:
            print(f"  episode {ep}: 0 valid frames after port_pose check; skipping")
            continue
        new_states[ep_mask, :n_kept_channels] = transformed_states[ep_mask][:, keep_idx]
        new_states[ep_mask, n_kept_channels:] = task_vec
        new_task_indices[ep_mask] = ti
        keep_mask |= ep_mask

    n_kept_frames = int(keep_mask.sum())
    n_kept_eps = len(np.unique(eps_col[keep_mask]))
    print(f"  kept {n_kept_eps} episodes / {n_kept_frames} frames "
          f"(after clean-filter + port_pose validity)")

    if n_kept_frames == 0:
        print("error: no frames survived filtering", file=sys.stderr)
        return 1

    # --- 5. Renumber episodes 0..N-1 ---------------------------------
    old_eps = sorted(np.unique(eps_col[keep_mask]).tolist())
    ep_remap: dict[int, int] = {int(e): i for i, e in enumerate(old_eps)}

    sub = src_data.filter(pa.array(keep_mask))
    new_eps_col = np.array(
        [ep_remap[int(e)] for e in sub["episode_index"].to_pylist()],
        dtype=np.int64,
    )

    new_frame_idx = np.zeros(n_kept_frames, dtype=np.int64)
    new_idx = np.arange(n_kept_frames, dtype=np.int64)
    counter: dict[int, int] = {}
    for i, ne in enumerate(new_eps_col):
        ne_i = int(ne)
        new_frame_idx[i] = counter.get(ne_i, 0)
        counter[ne_i] = counter.get(ne_i, 0) + 1
    new_task_indices_filt = new_task_indices[keep_mask]

    # --- 6. Build new data parquet -----------------------------------
    cols_to_keep = [c for c in src_data.column_names
                    if c not in {"observation.state", "action",
                                 "episode_index", "frame_index", "index",
                                 "task_index"}]
    new_table_cols = {}
    for c in cols_to_keep:
        new_table_cols[c] = sub[c]
    new_table_cols["observation.state"] = pa.array(
        new_states[keep_mask].tolist(),
        type=pa.list_(pa.float32(), 44),
    )
    new_table_cols["action"] = pa.array(
        transformed_actions[keep_mask].tolist(),
        type=pa.list_(pa.float32(), 7),
    )
    new_table_cols["episode_index"] = pa.array(new_eps_col, type=pa.int64())
    new_table_cols["frame_index"] = pa.array(new_frame_idx, type=pa.int64())
    new_table_cols["index"] = pa.array(new_idx, type=pa.int64())
    new_table_cols["task_index"] = pa.array(new_task_indices_filt, type=pa.int64())
    new_table = pa.table(new_table_cols)
    pq.write_table(new_table, str(out_data_dir / "file-000.parquet"))
    print(f"  wrote data parquet: {n_kept_frames} frames")

    # --- 7. Symlink videos -------------------------------------------
    src_videos = src_root / "videos"
    if src_videos.exists():
        if out_videos_dir.exists() or out_videos_dir.is_symlink():
            out_videos_dir.unlink()
        out_videos_dir.symlink_to(src_videos.resolve())
        print(f"  symlinked videos: {out_videos_dir} → {src_videos}")

    # --- 8. Per-episode metadata + stats -----------------------------
    src_eps_table = pq.read_table(
        str(src_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"),
    )
    src_eps_records = src_eps_table.to_pylist()
    src_eps_by_index = {int(r["episode_index"]): r for r in src_eps_records}

    new_eps_records: list[dict] = []
    cumulative_offset = 0
    for new_ep, old_ep in enumerate(old_eps):
        ep_mask_new = new_eps_col == new_ep
        ep_len = int(ep_mask_new.sum())
        ep_state = new_states[keep_mask][ep_mask_new]
        ep_action = transformed_actions[keep_mask][ep_mask_new]
        ep_timestamp = np.asarray(
            new_table["timestamp"].filter(pa.array(ep_mask_new)).to_pylist(),
            dtype=np.float64,
        )
        ti = int(new_task_indices_filt[ep_mask_new][0])

        rec = dict(src_eps_by_index[int(old_ep)])
        rec["episode_index"] = new_ep
        rec["length"] = ep_len
        rec["dataset_from_index"] = cumulative_offset
        rec["dataset_to_index"] = cumulative_offset + ep_len
        rec["tasks"] = [
            task_string_for(*next(
                (
                    (mod, port, ("sc" if port == "sc_port_base" else "sfp"))
                    for (mod, port), idx in task_index_for_pair.items()
                    if idx == ti
                ),
            ))
        ]

        for k, v in per_episode_stats(ep_state).items():
            rec[f"stats/observation.state/{k}"] = v
        for k, v in per_episode_stats(ep_action).items():
            rec[f"stats/action/{k}"] = v
        for k, v in per_episode_stats(ep_timestamp).items():
            rec[f"stats/timestamp/{k}"] = v
        for col_name, arr in (
            ("frame_index", np.arange(ep_len, dtype=np.int64)),
            ("episode_index", np.full(ep_len, new_ep, dtype=np.int64)),
            ("index", np.arange(cumulative_offset, cumulative_offset + ep_len,
                                dtype=np.int64)),
            ("task_index", np.full(ep_len, ti, dtype=np.int64)),
        ):
            for k, v in per_episode_stats(arr.astype(np.float64)).items():
                rec[f"stats/{col_name}/{k}"] = v

        new_eps_records.append(rec)
        cumulative_offset += ep_len

    new_eps_table = pa.Table.from_pylist(new_eps_records)
    pq.write_table(
        new_eps_table,
        str(out_meta_eps_dir / "file-000.parquet"),
    )
    print(f"  wrote per-episode meta: {len(new_eps_records)} episodes")

    # --- 9. Updated info.json ----------------------------------------
    new_info = dict(src_info)
    new_info["total_episodes"] = len(new_eps_records)
    new_info["total_frames"] = n_kept_frames
    new_info["total_tasks"] = len(task_index_for_pair)

    new_features = dict(new_info["features"])
    new_features["observation.state"] = {
        **new_features["observation.state"],
        "shape": [44],
        "names": build_new_state_names(),
    }
    new_info["features"] = new_features
    new_info["frame_transform"] = "port_local"  # marker for downstream consumers
    (out_meta_dir / "info.json").write_text(json.dumps(new_info, indent=2))
    print(f"  wrote info.json (frame_transform=port_local)")

    # --- 10. Train/val episode split ---------------------------------
    train_eps, val_eps = episode_split(
        list(range(n_kept_eps)), args.val_fraction, args.split_seed,
    )
    (out_root / "train_episodes.json").write_text(
        json.dumps(sorted(train_eps), indent=2)
    )
    (out_root / "val_episodes.json").write_text(
        json.dumps(sorted(val_eps), indent=2)
    )
    print(f"  train/val split: {len(train_eps)}/{len(val_eps)} "
          f"(seed={args.split_seed})")

    print(f"\nDone: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
