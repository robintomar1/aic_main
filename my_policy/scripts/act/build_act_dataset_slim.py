#!/usr/bin/env python3
"""Slim a v9-act ACT-ready dataset down to a thinner observation space.

Input: a dataset produced by `build_act_dataset.py` or `merge_act_datasets.py`.
Must have 44-dim `observation.state` (channels 0..6 = tcp_pose, 32..43 = task
vec) and the three cameras `observation.images.{left_camera,center_camera,right_camera}`.

Output: a lerobot-compatible dataset with:
  - `observation.state`  shape [19] = tcp_pose (7) + task_vec (12)
       (drops: tcp_velocity, tcp_error, joint_positions, wrench)
  - `observation.images.center_camera`, `observation.images.right_camera`
       (drops: `observation.images.left_camera`)
  - `action` [7]  — unchanged
  - All episodes kept; episode_index / global index unchanged.
  - `meta/tasks.parquet`, `train_episodes.json`, `val_episodes.json`
    carried over verbatim if present.

The output is drop-in for `train_act.py`:

    pixi run python my_policy/scripts/act/build_act_dataset_slim.py \
        --src-root /root/aic_data/v9_act_build/my_50ep_act_dataset \
        --out-root /root/aic_data/v9_act_build/my_50ep_act_slim

    pixi run python my_policy/scripts/act/train_act.py \
        --name v9_act_slim_v1 \
        --dataset-root /root/aic_data/v9_act_build/my_50ep_act_slim

If your 50-episode dataset is still in raw recorder form (47-dim state, no
task vec), run `build_act_dataset.py` first to produce the v9-act-ready
input this script expects.
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


# Slim layout: tcp_pose [0:7] + task_vec [32:44] = 19 dims.
# Matches the channel order locked in by build_act_dataset.py:KEEP_CHANNEL_GROUPS
# and my_policy.act.labels.task_channel_names().
KEEP_STATE_INDICES: list[int] = list(range(0, 7)) + list(range(32, 44))
SLIM_STATE_DIM = len(KEEP_STATE_INDICES)
assert SLIM_STATE_DIM == 19

DROP_CAMERAS: set[str] = {"observation.images.left_camera"}


def per_episode_stats(values: np.ndarray) -> dict:
    """Same stats schema as build_act_dataset.py — lerobot expects these
    exact keys in per-episode metadata."""
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


def is_dropped_camera_column(col: str) -> bool:
    """Returns True if a per-episode meta column name is scoped to a dropped
    camera. Covers both `stats/<cam>/...` and `videos/<cam>/...` prefixes."""
    for cam in DROP_CAMERAS:
        if col.startswith(f"stats/{cam}/") or col.startswith(f"videos/{cam}/"):
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src-root", type=Path, required=True,
                   help="v9-act-ready source dataset (44-dim state, 3 cams).")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Output dir for the slim variant. Created if missing.")
    p.add_argument("--force", action="store_true",
                   help="Wipe --out-root if it exists.")
    args = p.parse_args()

    src: Path = args.src_root
    out: Path = args.out_root

    if not (src / "meta" / "info.json").exists():
        print(f"error: {src} is not a lerobot dataset (no meta/info.json).",
              file=sys.stderr)
        return 1
    if out.exists():
        if args.force:
            print(f"--force: removing {out}")
            shutil.rmtree(out)
        else:
            print(f"error: {out} already exists. Pass --force or pick another path.",
                  file=sys.stderr)
            return 1

    out_data_dir = out / "data" / "chunk-000"
    out_meta_dir = out / "meta"
    out_meta_eps_dir = out_meta_dir / "episodes" / "chunk-000"
    out_videos_dir = out / "videos"
    for d in (out_data_dir, out_meta_dir, out_meta_eps_dir, out_videos_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. info.json -------------------------------------------------------
    src_info = json.loads((src / "meta" / "info.json").read_text())
    src_state_names = src_info["features"]["observation.state"]["names"]
    if len(src_state_names) != 44:
        print(f"error: expected 44-dim source state, got {len(src_state_names)}. "
              f"This script only operates on v9-act-ready datasets.",
              file=sys.stderr)
        return 1
    slim_state_names = [src_state_names[i] for i in KEEP_STATE_INDICES]
    assert len(slim_state_names) == SLIM_STATE_DIM

    src_image_keys = [k for k in src_info["features"] if k.startswith("observation.images.")]
    missing = DROP_CAMERAS - set(src_image_keys)
    if missing:
        print(f"error: source is missing camera(s) we expected to drop: {missing}. "
              f"Found image keys: {src_image_keys}", file=sys.stderr)
        return 1

    new_features: dict = {}
    for k, v in src_info["features"].items():
        if k in DROP_CAMERAS:
            continue
        if k == "observation.state":
            new_features[k] = {
                "dtype": "float32",
                "names": slim_state_names,
                "shape": [SLIM_STATE_DIM],
            }
        else:
            new_features[k] = v
    new_info = dict(src_info)
    new_info["features"] = new_features
    (out_meta_dir / "info.json").write_text(json.dumps(new_info, indent=2))
    kept_cams = [k for k in new_features if k.startswith("observation.images.")]
    print(f"wrote meta/info.json: state {len(src_state_names)} → {SLIM_STATE_DIM} dims, "
          f"dropped {sorted(DROP_CAMERAS)}, kept {sorted(kept_cams)}.")

    # --- 2. data parquet ----------------------------------------------------
    src_data_path = src / "data" / "chunk-000" / "file-000.parquet"
    if not src_data_path.exists():
        print(f"error: missing {src_data_path}", file=sys.stderr)
        return 1
    src_data = pq.read_table(str(src_data_path))
    states = np.stack([
        np.asarray(r, dtype=np.float32)
        for r in src_data["observation.state"].to_pylist()
    ])
    if states.shape[1] != 44:
        print(f"error: source state width {states.shape[1]} ≠ 44.", file=sys.stderr)
        return 1
    slim_states = states[:, KEEP_STATE_INDICES]

    cols: dict = {}
    for c in src_data.column_names:
        if c == "observation.state":
            cols[c] = pa.array(
                slim_states.tolist(),
                type=pa.list_(pa.float32(), SLIM_STATE_DIM),
            )
        else:
            cols[c] = src_data[c]
    new_table = pa.table(cols)
    pq.write_table(new_table, str(out_data_dir / "file-000.parquet"))
    print(f"wrote data parquet: {new_table.num_rows} frames, "
          f"state column reshaped to {SLIM_STATE_DIM} dims.")

    # --- 3. episodes parquet -----------------------------------------------
    src_eps_path = src / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not src_eps_path.exists():
        print(f"error: missing {src_eps_path}", file=sys.stderr)
        return 1
    src_eps_table = pq.read_table(str(src_eps_path))
    src_eps_records = src_eps_table.to_pylist()

    eps_col = src_data["episode_index"].to_numpy().astype(np.int64)

    new_eps_records: list[dict] = []
    for rec in src_eps_records:
        ep = int(rec["episode_index"])
        ep_mask = eps_col == ep
        ep_state = slim_states[ep_mask]

        new_rec = {k: v for k, v in rec.items() if not is_dropped_camera_column(k)}

        # Recompute stats/observation.state/* from the slim state — schema
        # changed (19 dims), so stale per-channel arrays would be wrong shape.
        for k, v in per_episode_stats(ep_state).items():
            new_rec[f"stats/observation.state/{k}"] = v

        new_eps_records.append(new_rec)

    new_eps_table = pa.Table.from_pylist(new_eps_records)
    pq.write_table(new_eps_table, str(out_meta_eps_dir / "file-000.parquet"))
    print(f"wrote per-episode meta: {len(new_eps_records)} episodes, "
          f"camera-left columns dropped, state stats recomputed.")

    # --- 4. tasks.parquet (copy as-is) -------------------------------------
    src_tasks = src / "meta" / "tasks.parquet"
    if src_tasks.exists():
        shutil.copy(src_tasks, out_meta_dir / "tasks.parquet")
        print(f"copied meta/tasks.parquet")

    # --- 5. videos: symlink kept cameras only ------------------------------
    # Layout: videos/<camera>/<chunk-N>/<file>.mp4 — symlink at the camera
    # level so all chunks under each kept camera flow through unchanged.
    src_videos_root = src / "videos"
    if src_videos_root.exists():
        for cam_dir in sorted(src_videos_root.iterdir()):
            if not cam_dir.is_dir():
                continue
            if cam_dir.name in DROP_CAMERAS:
                continue
            link = out_videos_dir / cam_dir.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(cam_dir.resolve())
        print(f"symlinked videos for cameras: "
              f"{[c.name for c in sorted(out_videos_dir.iterdir())]}")

    # --- 6. train/val splits ----------------------------------------------
    for fname in ("train_episodes.json", "val_episodes.json"):
        src_split = src / fname
        if src_split.exists():
            shutil.copy(src_split, out / fname)
            print(f"copied {fname}")

    print(f"\nslim dataset ready at: {out}")
    print(f"  state dim     : {SLIM_STATE_DIM}")
    print(f"  state names   : {slim_state_names}")
    print(f"  cameras kept  : {sorted(kept_cams)}")
    print(f"  action dim    : unchanged (7)")
    print()
    print(f"train with:")
    print(f"  pixi run python my_policy/scripts/act/train_act.py \\")
    print(f"      --name <run_name> \\")
    print(f"      --dataset-root {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
