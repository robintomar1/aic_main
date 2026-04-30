#!/usr/bin/env python3
"""Phase A2 — merge multiple ACT-built datasets into one trainable dataset.

Each input dataset is the output of `build_act_dataset.py` for one batch
(e.g. /tmp/v9_act_build/batch_500_a_act_dataset/). They share the SAME
schema (44-dim observation.state, identical 12-entry tasks.parquet) but
have independent (episode_index, frame_index, index) numbering and their
own videos/{cam}/chunk-000/file-XXX.mp4 trees.

This script writes ONE merged dataset at <out-root>/<out-name>/ with:
  - data/chunk-000/file-000.parquet : all frames concatenated, with
    episode_index/frame_index/index renumbered globally.
  - meta/info.json : totals updated; everything else inherited from
    source 0 (schema is identical across sources).
  - meta/tasks.parquet : copied from source 0 (verified identical).
  - meta/episodes/chunk-000/file-000.parquet : per-episode records
    concatenated, episode_index + dataset_from/to_index + videos/{cam}/
    file_index updated for the merged numbering.
  - videos/{cam}/chunk-000/file-{global_idx:03d}.mp4 : symlinks to each
    source's video files, with file_index assigned in source-order.
  - train_episodes.json / val_episodes.json : concatenation of each
    source's split, with episode-index offsets applied.

Pure pyarrow + numpy + json. No torch / lerobot. Run inside or outside
pixi.

Usage:
    python3 my_policy/scripts/merge_act_datasets.py \\
        --sources /tmp/v9_act_build/batch_100_a_act_dataset \\
                  /tmp/v9_act_build/batch_100_b_act_dataset \\
                  /tmp/v9_act_build/batch_500_a_act_dataset \\
        --out-root /tmp/v9_act_build \\
        --out-name v9_act_merged
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


CAMERA_KEYS = (
    "observation.images.left_camera",
    "observation.images.center_camera",
    "observation.images.right_camera",
)


def _verify_schema_compatibility(sources: list[Path]) -> dict:
    """Sanity check — every source must agree on observation.state shape,
    feature schema, and tasks.parquet contents. Returns source 0's info dict."""
    base_info = json.loads((sources[0] / "meta" / "info.json").read_text())
    base_state_names = base_info["features"]["observation.state"]["names"]
    base_tasks = pq.read_table(str(sources[0] / "meta" / "tasks.parquet")).to_pylist()

    for s in sources[1:]:
        info = json.loads((s / "meta" / "info.json").read_text())
        names = info["features"]["observation.state"]["names"]
        if names != base_state_names:
            raise ValueError(
                f"observation.state schema mismatch between\n  {sources[0]}\n  {s}\n"
                f"first differing channel index: "
                f"{next((i for i, (a, b) in enumerate(zip(names, base_state_names)) if a != b), 'len-mismatch')}"
            )
        tasks = pq.read_table(str(s / "meta" / "tasks.parquet")).to_pylist()
        if tasks != base_tasks:
            raise ValueError(
                f"tasks.parquet mismatch between\n  {sources[0]}\n  {s}\n"
                f"this is a build_act_dataset.py bug — tasks should be deterministic"
            )
    return base_info


def _count_video_files_per_camera(source: Path) -> dict[str, list[Path]]:
    """Returns {cam_key: [file-000.mp4, file-001.mp4, ...]} sorted by file_index."""
    out: dict[str, list[Path]] = {}
    for cam in CAMERA_KEYS:
        chunk_dir = source / "videos" / cam / "chunk-000"
        if not chunk_dir.exists():
            raise FileNotFoundError(f"missing video dir: {chunk_dir}")
        files = sorted(chunk_dir.glob("file-*.mp4"))
        if not files:
            raise FileNotFoundError(f"no video files in {chunk_dir}")
        out[cam] = files
    # All cameras for one source MUST agree on file count (recorder writes them in lockstep).
    counts = {cam: len(v) for cam, v in out.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(
            f"camera video file count mismatch in {source}: {counts} — "
            f"the recorder should write all cameras in lockstep"
        )
    return out


def _resolve_through_symlink(p: Path) -> Path:
    """For a path that may itself be a symlink, return the absolute final target."""
    return p.resolve()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sources", type=Path, nargs="+", required=True,
                   help="One or more *_act_dataset/ roots produced by build_act_dataset.py.")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Parent dir for the merged dataset; "
                        "creates <out-root>/<out-name>/.")
    p.add_argument("--out-name", type=str, default="v9_act_merged",
                   help="Subdir name inside --out-root (default: v9_act_merged).")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Used only if a source is missing train/val_episodes.json.")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Delete <out-root>/<out-name>/ if it already exists.")
    args = p.parse_args()

    sources: list[Path] = [s.resolve() for s in args.sources]
    for s in sources:
        if not (s / "meta" / "info.json").exists():
            print(f"error: {s} doesn't look like a built ACT dataset "
                  f"(missing meta/info.json)", file=sys.stderr)
            return 1

    print(f"merging {len(sources)} sources:")
    for s in sources:
        print(f"  {s}")
    base_info = _verify_schema_compatibility(sources)
    print(f"schema OK: {base_info['features']['observation.state']['shape']} state dim, "
          f"{len(CAMERA_KEYS)} cameras")

    out_root = args.out_root / args.out_name
    if out_root.exists():
        if args.force:
            print(f"removing existing {out_root}")
            shutil.rmtree(out_root)
        else:
            print(f"error: {out_root} exists; pass --force to overwrite", file=sys.stderr)
            return 1
    (out_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for cam in CAMERA_KEYS:
        (out_root / "videos" / cam / "chunk-000").mkdir(parents=True, exist_ok=True)

    # --- 1. Symlink videos with global file_index per camera ----------
    # Walk sources in order; each source contributes its files in order.
    # Track (source_idx, source_local_file_idx) → global_file_idx mapping per cam.
    video_remap: dict[tuple[int, str, int], int] = {}
    global_file_counter: dict[str, int] = {cam: 0 for cam in CAMERA_KEYS}
    for src_idx, src in enumerate(sources):
        files_per_cam = _count_video_files_per_camera(src)
        for cam, files in files_per_cam.items():
            for local_idx, src_file in enumerate(files):
                global_idx = global_file_counter[cam]
                global_file_counter[cam] += 1
                target = _resolve_through_symlink(src_file)
                link = (out_root / "videos" / cam / "chunk-000" /
                        f"file-{global_idx:03d}.mp4")
                link.symlink_to(target)
                video_remap[(src_idx, cam, local_idx)] = global_idx
    for cam, n in global_file_counter.items():
        print(f"  videos/{cam}: {n} files symlinked")

    # --- 2. Concatenate data parquets, renumber indices --------------
    all_data_tables: list[pa.Table] = []
    episode_offset = 0
    frame_offset = 0
    per_source_episode_offset: list[int] = []
    for src_idx, src in enumerate(sources):
        per_source_episode_offset.append(episode_offset)
        t = pq.read_table(str(src / "data" / "chunk-000" / "file-000.parquet"))
        n_frames = t.num_rows
        old_eps = np.asarray(t["episode_index"].to_pylist(), dtype=np.int64)
        n_eps = int(old_eps.max()) + 1 if n_frames else 0

        new_eps = old_eps + episode_offset
        # frame_index already 0..len-1 per episode in source — keep as-is.
        new_idx = np.arange(n_frames, dtype=np.int64) + frame_offset

        # Replace the affected columns; keep the rest intact.
        cols = {c: t[c] for c in t.column_names}
        cols["episode_index"] = pa.array(new_eps, type=pa.int64())
        cols["index"] = pa.array(new_idx, type=pa.int64())
        # task_index is per-episode constant; values copy through unchanged.
        new_table = pa.table(cols)
        all_data_tables.append(new_table)

        episode_offset += n_eps
        frame_offset += n_frames
        print(f"  source {src_idx}: {n_eps} eps, {n_frames} frames "
              f"(global eps {per_source_episode_offset[-1]}..{episode_offset - 1})")

    merged_data = pa.concat_tables(all_data_tables, promote_options="default")
    pq.write_table(merged_data,
                   str(out_root / "data" / "chunk-000" / "file-000.parquet"))
    print(f"  wrote merged data parquet: {merged_data.num_rows} frames, "
          f"{episode_offset} episodes")

    # --- 3. Per-episode meta: concatenate + remap --------------------
    all_eps_records: list[dict] = []
    for src_idx, src in enumerate(sources):
        ep_offset = per_source_episode_offset[src_idx]
        # Need the source's data parquet to recompute dataset_from/to_index in
        # the GLOBAL indexing scheme. The source's records still use that
        # source's local from/to.
        src_data = pq.read_table(
            str(src / "data" / "chunk-000" / "file-000.parquet"),
            columns=["episode_index"],
        )
        src_eps_col = np.asarray(src_data["episode_index"].to_pylist(), dtype=np.int64)
        # Cumulative frame offset for THIS source within the merged dataset.
        # = sum of frame counts in all PRIOR sources.
        global_frame_offset = sum(t.num_rows for t in all_data_tables[:src_idx])

        eps_records = pq.read_table(
            str(src / "meta" / "episodes" / "chunk-000" / "file-000.parquet"),
        ).to_pylist()
        for rec in eps_records:
            old_ep = int(rec["episode_index"])
            new_ep = old_ep + ep_offset
            rec["episode_index"] = new_ep
            # dataset_from/to_index — shift by the global frame offset.
            rec["dataset_from_index"] = int(rec["dataset_from_index"]) + global_frame_offset
            rec["dataset_to_index"] = int(rec["dataset_to_index"]) + global_frame_offset
            # Update video file_index per camera using the symlink remap.
            for cam in CAMERA_KEYS:
                local_file_idx = int(rec[f"videos/{cam}/file_index"])
                rec[f"videos/{cam}/file_index"] = video_remap[
                    (src_idx, cam, local_file_idx)
                ]
                # chunk_index stays at 0 (all symlinks live under chunk-000).
                rec[f"videos/{cam}/chunk_index"] = 0
            # Re-derive episode_index stats since the value changed.
            n = int(rec["length"])
            rec["stats/episode_index/min"] = [float(new_ep)]
            rec["stats/episode_index/max"] = [float(new_ep)]
            rec["stats/episode_index/mean"] = [float(new_ep)]
            rec["stats/episode_index/std"] = [0.0]
            rec["stats/episode_index/count"] = [n]
            for q in ("q01", "q10", "q50", "q90", "q99"):
                rec[f"stats/episode_index/{q}"] = [float(new_ep)]
            # Re-derive index stats too.
            new_from = rec["dataset_from_index"]
            new_to = rec["dataset_to_index"]
            idx_arr = np.arange(new_from, new_to, dtype=np.float64)
            rec["stats/index/min"] = [float(idx_arr.min())]
            rec["stats/index/max"] = [float(idx_arr.max())]
            rec["stats/index/mean"] = [float(idx_arr.mean())]
            rec["stats/index/std"] = [float(idx_arr.std())]
            rec["stats/index/count"] = [n]
            rec["stats/index/q01"] = [float(np.quantile(idx_arr, 0.01))]
            rec["stats/index/q10"] = [float(np.quantile(idx_arr, 0.10))]
            rec["stats/index/q50"] = [float(np.quantile(idx_arr, 0.50))]
            rec["stats/index/q90"] = [float(np.quantile(idx_arr, 0.90))]
            rec["stats/index/q99"] = [float(np.quantile(idx_arr, 0.99))]
            all_eps_records.append(rec)

    merged_eps_table = pa.Table.from_pylist(all_eps_records)
    pq.write_table(
        merged_eps_table,
        str(out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"),
    )
    print(f"  wrote merged per-episode meta: {len(all_eps_records)} episodes")

    # --- 4. tasks.parquet (deterministic, copy from source 0) ---------
    shutil.copy(
        sources[0] / "meta" / "tasks.parquet",
        out_root / "meta" / "tasks.parquet",
    )
    print(f"  copied tasks.parquet (12 deterministic entries)")

    # --- 5. info.json: totals updated --------------------------------
    new_info = dict(base_info)
    new_info["total_episodes"] = episode_offset
    new_info["total_frames"] = merged_data.num_rows
    # total_videos = total mp4 files across all cameras.
    new_info["total_videos"] = sum(global_file_counter.values())
    (out_root / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))
    print(f"  wrote meta/info.json (total_episodes={episode_offset}, "
          f"total_frames={merged_data.num_rows}, total_videos={new_info['total_videos']})")

    # --- 6. Concatenate train/val splits (with offset) ---------------
    merged_train: list[int] = []
    merged_val: list[int] = []
    for src_idx, src in enumerate(sources):
        ep_offset = per_source_episode_offset[src_idx]
        train_path = src / "train_episodes.json"
        val_path = src / "val_episodes.json"
        if train_path.exists() and val_path.exists():
            train_eps = json.loads(train_path.read_text())
            val_eps = json.loads(val_path.read_text())
        else:
            # Fallback — re-derive split for this source's episodes.
            print(f"  source {src_idx}: missing train/val json; regenerating "
                  f"with seed={args.split_seed}")
            src_data = pq.read_table(
                str(src / "data" / "chunk-000" / "file-000.parquet"),
                columns=["episode_index"],
            )
            src_eps = sorted(set(src_data["episode_index"].to_pylist()))
            rng = np.random.default_rng(args.split_seed + src_idx)
            perm = list(rng.permutation(src_eps))
            n_val = max(1, int(round(len(perm) * args.val_fraction)))
            val_eps = sorted(int(e) for e in perm[:n_val])
            train_eps = sorted(int(e) for e in perm[n_val:])
        merged_train.extend(int(e) + ep_offset for e in train_eps)
        merged_val.extend(int(e) + ep_offset for e in val_eps)
    merged_train.sort()
    merged_val.sort()
    overlap = set(merged_train) & set(merged_val)
    if overlap:
        raise RuntimeError(f"train/val overlap after merge: {sorted(overlap)[:5]}...")
    (out_root / "train_episodes.json").write_text(json.dumps(merged_train))
    (out_root / "val_episodes.json").write_text(json.dumps(merged_val))
    print(f"  wrote train_episodes.json ({len(merged_train)}) "
          f"+ val_episodes.json ({len(merged_val)})")

    print(f"\nDONE: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
