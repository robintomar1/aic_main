#!/usr/bin/env python3
"""Phase A1 — preprocess a recorder dataset into ACT-ready format.

Reads ONE recorder-output batch (e.g. /root/aic_data/batch_100_a/), produces
a NEW LeRobotDataset at <out_root>/<batch>_act_dataset/ with:

  - data parquet: same frames, but observation.state rebuilt to 44-dim
    layout from §A1 of the v9-act plan:
      TCP position (3) + TCP orientation xyzw (4)
      + TCP linear vel (3) + TCP angular vel (3)
      + TCP error (6)
      + wrench tare-comp (6)
      + joint positions (7)
      + task vector (12)  ← NEW, from my_policy.act.labels
    Excluded: groundtruth.port_pose.*, meta.insertion_success (label leakage).
  - meta/info.json: updated state schema with 44 channel names.
  - meta/tasks.parquet: 12 deterministic task entries (one per
    (target_module, port_name) pair), indexed by ACT_VALID_TARGETS order so
    task_index is identical across batches when this script is run repeatedly.
  - meta/episodes/...: per-episode metadata + recomputed per-episode stats
    for the NEW state schema (lerobot uses these for normalization).
  - videos/: symlinked from the source batch (no re-encode, no copy).
  - train_episodes.json / val_episodes.json: 80/20 episode-level split with
    seed=42 (matches the localizer's split for apples-to-apples comparison
    at val time).

Optional clean filter: if --clean-episodes-json is passed, only those
episode indices are carried into the new dataset.

Pure pyarrow + numpy + json + yaml; no torch / lerobot needed at preprocess
time. Run inside or outside pixi.

Usage:
    python3 my_policy/scripts/build_act_dataset.py \\
        --collection-dir /root/aic_data \\
        --batch batch_100_a \\
        --out-root /root/aic_data \\
        --clean-episodes-json /root/aic_data/batch_100_a_act_clean_episodes.json
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


# Channel groups extracted from the recorder's 47-dim observation.state.
# The recorder layout (verified from batch_100_a/meta/info.json) is exactly
# these prefixes followed by the leakage channels we drop.
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
    """Compose the 44-channel name list for the new observation.state."""
    out: list[str] = []
    for _, group in KEEP_CHANNEL_GROUPS:
        out.extend(group)
    out.extend(task_channel_names())
    assert len(out) == sum(len(g) for _, g in KEEP_CHANNEL_GROUPS) + ACT_TASK_VECTOR_DIM
    assert len(out) == 44, f"expected 44 channels, got {len(out)}"
    return out


def slice_indices(state_names: list[str], wanted: list[str]) -> list[int]:
    """Look up indices for `wanted` inside `state_names` (raises if any missing)."""
    out: list[int] = []
    for n in wanted:
        if n not in state_names:
            raise KeyError(f"channel {n!r} not in source observation.state.names")
        out.append(state_names.index(n))
    return out


def per_episode_stats(values: np.ndarray) -> dict:
    """Compute the stats dict lerobot's per-episode metadata expects.

    `values` shape: (n_frames, n_channels). Returns floats (or scalar arrays)
    matching the keys the existing parquet's `stats/<feature>/<key>` columns
    use: min, max, mean, std, count, q01, q10, q50, q90, q99.
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "min": values.min(axis=0).astype(np.float64).tolist(),
        "max": values.max(axis=0).astype(np.float64).tolist(),
        "mean": values.mean(axis=0).astype(np.float64).tolist(),
        "std": values.std(axis=0).astype(np.float64).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float64).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).astype(np.float64).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).astype(np.float64).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float64).tolist(),
    }


def episode_split(
    episodes: list[int], val_fraction: float = 0.2, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Episode-level split — same logic as the localizer's episode_split for
    apples-to-apples comparison at val time."""
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(sorted(episodes)))
    n_val = max(1, int(round(len(perm) * val_fraction)))
    val_eps = sorted(int(e) for e in perm[:n_val])
    train_eps = sorted(int(e) for e in perm[n_val:])
    return train_eps, val_eps


def build_tasks_parquet(out_meta_dir: Path) -> dict[tuple[str, str], int]:
    """Write meta/tasks.parquet with 12 deterministic entries (one per
    valid (target_module, port_name) pair), and return the lookup map.

    task_index is the index of the pair in ACT_VALID_TARGETS, so it's
    identical across batches whenever this script is re-run.
    """
    rows: list[dict] = []
    lookup: dict[tuple[str, str], int] = {}
    for i, (mod, port) in enumerate(ACT_VALID_TARGETS):
        ptype = "sc" if port == "sc_port_base" else "sfp"
        rows.append({
            "task_index": i,
            "task": task_string_for(mod, port, ptype),
        })
        lookup[(mod, port)] = i
    table = pa.Table.from_pylist(rows)
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_meta_dir / "tasks.parquet"))
    return lookup


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection-dir", type=Path, required=True)
    p.add_argument("--batch", type=str, required=True)
    p.add_argument("--out-root", type=Path, required=True,
                   help="Parent dir for the output dataset; "
                        "creates <out-root>/<batch>_act_dataset/.")
    p.add_argument("--clean-episodes-json", type=Path, default=None,
                   help="Optional: filter to these episode indices only.")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    args = p.parse_args()

    src_root = args.collection_dir / args.batch
    src_yaml = args.collection_dir / f"{args.batch}.yaml"
    src_summary = args.collection_dir / f"{args.batch}_logs" / "summary.json"
    for p_, label in [(src_root, "dataset_root"),
                      (src_yaml, "batch_yaml"),
                      (src_summary, "summary_json")]:
        if not p_.exists():
            print(f"error: {label} not found at {p_}", file=sys.stderr)
            return 1

    out_root = args.out_root / f"{args.batch}_act_dataset"
    out_data_dir = out_root / "data" / "chunk-000"
    out_meta_dir = out_root / "meta"
    out_meta_eps_dir = out_meta_dir / "episodes" / "chunk-000"
    out_videos_dir = out_root / "videos"
    for d in (out_data_dir, out_meta_dir, out_meta_eps_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. Read source dataset ----------------------------------------
    print(f"reading source: {src_root}")
    src_info = json.loads((src_root / "meta" / "info.json").read_text())
    src_state_names = src_info["features"]["observation.state"]["names"]
    keep_idx = slice_indices(
        src_state_names,
        [n for _, group in KEEP_CHANNEL_GROUPS for n in group],
    )
    src_data = pq.read_table(str(src_root / "data" / "chunk-000" / "file-000.parquet"))
    print(f"  source frames: {src_data.num_rows}")

    # --- 2. Episode + task lookup --------------------------------------
    cfg = yaml.safe_load(src_yaml.read_text())
    summary = json.loads(src_summary.read_text())
    ep_to_trial = match_episodes_to_trials(summary, cfg["trials"])

    keep_episodes: set[int] | None = None
    if args.clean_episodes_json is not None:
        keep_episodes = set(json.loads(args.clean_episodes_json.read_text()))
        print(f"  clean filter: keeping {len(keep_episodes)} episodes")

    # Build per-episode task-index assignment from the deterministic tasks.parquet.
    task_index_for_pair = build_tasks_parquet(out_meta_dir)
    print(f"  wrote meta/tasks.parquet ({len(task_index_for_pair)} entries)")

    # --- 3. Augment state per frame ------------------------------------
    eps_col = src_data["episode_index"].to_numpy().astype(np.int64)
    states = np.stack([
        np.asarray(r, dtype=np.float32)
        for r in src_data["observation.state"].to_pylist()
    ])  # (n_frames, 47)

    # Rebuild observation.state per frame: keep selected channels + append task vec.
    new_states = np.zeros((src_data.num_rows, 44), dtype=np.float32)
    new_task_indices = np.zeros(src_data.num_rows, dtype=np.int64)
    keep_mask = np.zeros(src_data.num_rows, dtype=bool)

    n_kept_channels = sum(len(g) for _, g in KEEP_CHANNEL_GROUPS)
    assert n_kept_channels == 32, f"expected 32 source channels kept, got {n_kept_channels}"

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
        ep_mask = eps_col == ep
        new_states[ep_mask, :n_kept_channels] = states[ep_mask][:, keep_idx]
        new_states[ep_mask, n_kept_channels:] = task_vec
        new_task_indices[ep_mask] = ti
        keep_mask |= ep_mask

    n_kept_frames = int(keep_mask.sum())
    n_kept_eps = len(np.unique(eps_col[keep_mask]))
    print(f"  kept {n_kept_eps} episodes / {n_kept_frames} frames")

    # --- 4. Renumber (episode_index, frame_index, index) post-filter ---
    # Old episode indices may be sparse after filtering; lerobot expects a
    # dense 0..N-1 numbering. Build the remap and apply to all relevant cols.
    old_eps = np.unique(eps_col[keep_mask]).tolist()
    ep_remap: dict[int, int] = {int(e): i for i, e in enumerate(old_eps)}

    sub = src_data.filter(pa.array(keep_mask))
    new_eps_col = np.array(
        [ep_remap[int(e)] for e in sub["episode_index"].to_pylist()],
        dtype=np.int64,
    )

    # frame_index re-derived per new episode (0..len-1 per episode), index global.
    new_frame_idx = np.zeros(n_kept_frames, dtype=np.int64)
    new_idx = np.arange(n_kept_frames, dtype=np.int64)
    counter: dict[int, int] = {}
    for i, ne in enumerate(new_eps_col):
        ne_i = int(ne)
        new_frame_idx[i] = counter.get(ne_i, 0)
        counter[ne_i] = counter.get(ne_i, 0) + 1

    # task_index column post-filter.
    new_task_indices_filt = new_task_indices[keep_mask]

    # --- 5. Build the new data parquet ---------------------------------
    # Carry through every column from the source EXCEPT observation.state,
    # episode_index, frame_index, index, task_index — those we replace.
    cols_to_keep = [c for c in src_data.column_names
                    if c not in {"observation.state", "episode_index",
                                 "frame_index", "index", "task_index"}]
    new_table_cols = {}
    for c in cols_to_keep:
        new_table_cols[c] = sub[c]
    new_table_cols["observation.state"] = pa.array(
        new_states[keep_mask].tolist(),
        type=pa.list_(pa.float32(), 44),
    )
    new_table_cols["episode_index"] = pa.array(new_eps_col, type=pa.int64())
    new_table_cols["frame_index"] = pa.array(new_frame_idx, type=pa.int64())
    new_table_cols["index"] = pa.array(new_idx, type=pa.int64())
    new_table_cols["task_index"] = pa.array(new_task_indices_filt, type=pa.int64())
    new_table = pa.table(new_table_cols)
    pq.write_table(new_table, str(out_data_dir / "file-000.parquet"))
    print(f"  wrote data parquet: {n_kept_frames} frames")

    # --- 6. Symlink videos --------------------------------------------
    src_videos = src_root / "videos"
    if src_videos.exists():
        if out_videos_dir.exists() or out_videos_dir.is_symlink():
            out_videos_dir.unlink()
        out_videos_dir.symlink_to(src_videos.resolve())
        print(f"  symlinked videos: {out_videos_dir} → {src_videos}")
    # Note: if filtering removed any episodes, the symlinked videos still
    # reference all original episodes. lerobot reads videos via the
    # per-episode meta `videos/.../from_timestamp`/`to_timestamp` ranges,
    # which we recompute below — so unused video segments are simply not
    # accessed. No need to re-encode.

    # --- 7. Per-episode metadata + stats -------------------------------
    # Read the source episode meta table — we mostly want to keep the
    # video timestamp pointers and recompute the stats columns.
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
        ep_state = new_states[keep_mask][ep_mask_new]    # (ep_len, 44)
        ep_action = np.stack([
            np.asarray(r, dtype=np.float32)
            for r in new_table["action"].filter(pa.array(ep_mask_new)).to_pylist()
        ])
        ep_timestamp = np.asarray(
            new_table["timestamp"].filter(pa.array(ep_mask_new)).to_pylist(),
            dtype=np.float64,
        )
        ti = int(new_task_indices_filt[ep_mask_new][0])

        rec = dict(src_eps_by_index[int(old_ep)])
        # Update fields that change post-filter / post-renumber.
        rec["episode_index"] = new_ep
        rec["length"] = ep_len
        rec["dataset_from_index"] = cumulative_offset
        rec["dataset_to_index"] = cumulative_offset + ep_len
        rec["tasks"] = [task_string_for(
            *next(((mod, port, ("sc" if port == "sc_port_base" else "sfp"))
                   for (mod, port), idx in task_index_for_pair.items() if idx == ti)),
        )]

        # Recompute stats/observation.state/* (since the schema changed).
        for k, v in per_episode_stats(ep_state).items():
            rec[f"stats/observation.state/{k}"] = v
        # Action stats — schema unchanged but values are now per-filtered-episode.
        for k, v in per_episode_stats(ep_action).items():
            rec[f"stats/action/{k}"] = v
        # timestamp stats too (rest re-derive trivially below).
        for k, v in per_episode_stats(ep_timestamp).items():
            rec[f"stats/timestamp/{k}"] = v
        # frame_index / episode_index / index / task_index stats — recomputed
        # against the new ranges.
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
    pq.write_table(new_eps_table,
                   str(out_meta_eps_dir / "file-000.parquet"))
    print(f"  wrote per-episode meta: {len(new_eps_records)} episodes")

    # --- 8. Updated info.json -----------------------------------------
    new_info = dict(src_info)
    new_info["total_episodes"] = len(new_eps_records)
    new_info["total_frames"] = n_kept_frames
    new_info["total_tasks"] = len(ACT_VALID_TARGETS)
    new_info["features"] = dict(src_info["features"])
    new_info["features"]["observation.state"] = {
        "dtype": "float32",
        "names": build_new_state_names(),
        "shape": [44],
    }
    (out_meta_dir / "info.json").write_text(json.dumps(new_info, indent=2))
    print(f"  wrote meta/info.json (state schema: 47 → 44 dims, "
          f"+12-dim task vec, −18 leakage channels)")

    # --- 9. Episode-level train/val split -----------------------------
    train_eps, val_eps = episode_split(
        list(range(len(new_eps_records))),
        val_fraction=args.val_fraction, seed=args.split_seed,
    )
    (out_root / "train_episodes.json").write_text(json.dumps(train_eps))
    (out_root / "val_episodes.json").write_text(json.dumps(val_eps))
    print(f"  wrote train_episodes.json ({len(train_eps)}) "
          f"+ val_episodes.json ({len(val_eps)})  seed={args.split_seed}")

    print(f"\nDONE: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
