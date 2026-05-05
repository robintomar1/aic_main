#!/usr/bin/env python3
"""Generate filtered train/val episode-index lists for a single task type.

For training a per-task-type ACT (e.g. SC-only), we re-use the existing
cleaned merged dataset (v9_act_merged_clean/) but supply different train/val
splits that include ONLY the desired episodes. The dataset itself is not
duplicated — lerobot's `--dataset.episodes` arg filters at load time, so
the trainer will only see the filtered subset.

Inputs:
  --dataset-root  : the cleaned dataset (e.g. v9_act_merged_clean/)
  --task-contains : substring matched against tasks.parquet `task` column
                    (e.g. "sc plug" picks both sc tasks; "sfp" picks all sfp)
  --out-dir       : where to write the filtered split JSONs

Outputs:
  <out-dir>/<tag>_train_episodes.json
  <out-dir>/<tag>_val_episodes.json
  where <tag> is derived from --task-contains (sanitized)

The script preserves the same per-episode train/val membership the merged
splits already had (i.e. an episode that was in train remains in train) —
only filters out episodes whose task doesn't match.

Run:
    python3 my_policy/scripts/make_task_filtered_splits.py \\
        --dataset-root /root/aic_data/v9_act_build/v9_act_merged_clean \\
        --task-contains "sc plug" \\
        --out-dir /root/aic_data/v9_act_build/v9_act_merged_clean
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


def sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--task-contains", type=str, required=True,
                   help="Substring matched against `task` column of tasks.parquet.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to dataset root.")
    p.add_argument("--tag", type=str, default=None,
                   help="Prefix for output filenames (default: derived from filter).")
    args = p.parse_args()

    out_dir = args.out_dir or args.dataset_root
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or sanitize(args.task_contains)

    # Find which task_index values match the substring.
    tasks_table = pq.read_table(args.dataset_root / "meta" / "tasks.parquet")
    matching_ti: set[int] = set()
    print(f"Matching tasks containing {args.task_contains!r}:")
    for row in tasks_table.to_pylist():
        if args.task_contains in row["task"]:
            matching_ti.add(int(row["task_index"]))
            print(f"  task_index {row['task_index']:2d}: {row['task']}")
    if not matching_ti:
        print(f"error: no tasks matched {args.task_contains!r}")
        return 1
    print(f"  -> {len(matching_ti)} task_index values matched")

    # Build episode_index -> task_index map by reading per-episode parquet
    # (one row per episode, with task_index column).
    ep_files = sorted((args.dataset_root / "meta" / "episodes").rglob("*.parquet"))
    ep_to_task: dict[int, int] = {}
    for f in ep_files:
        et = pq.read_table(f, columns=["episode_index", "tasks"])
        # The `tasks` column is a list of task strings (lerobot v3 supports
        # multi-task episodes). We map it back to indices via the tasks table.
        # Easier path: use the data parquet's task_index directly.
    # Re-do via data parquet (one per-frame, but task_index is constant per ep).
    data_table = pq.read_table(
        args.dataset_root / "data" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "task_index"],
    )
    seen: dict[int, int] = {}
    for ei, ti in zip(data_table.column("episode_index").to_pylist(),
                       data_table.column("task_index").to_pylist()):
        ei, ti = int(ei), int(ti)
        if ei not in seen:
            seen[ei] = ti
    ep_to_task = seen
    print(f"Built episode->task map for {len(ep_to_task)} episodes")

    # Load original train/val splits and intersect with matching tasks.
    train_path = args.dataset_root / "train_episodes.json"
    val_path = args.dataset_root / "val_episodes.json"
    if not train_path.exists() or not val_path.exists():
        print(f"error: missing {train_path} or {val_path}")
        return 1
    train_orig = set(json.loads(train_path.read_text()))
    val_orig = set(json.loads(val_path.read_text()))

    train_filtered = sorted(
        ei for ei in train_orig if ep_to_task.get(ei) in matching_ti)
    val_filtered = sorted(
        ei for ei in val_orig if ep_to_task.get(ei) in matching_ti)

    out_train = out_dir / f"{tag}_train_episodes.json"
    out_val = out_dir / f"{tag}_val_episodes.json"
    out_train.write_text(json.dumps(train_filtered))
    out_val.write_text(json.dumps(val_filtered))

    print()
    print(f"=== filtered splits ===")
    print(f"  train: {len(train_orig)} -> {len(train_filtered)} episodes  "
          f"-> {out_train}")
    print(f"  val  : {len(val_orig)} -> {len(val_filtered)} episodes  "
          f"-> {out_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
