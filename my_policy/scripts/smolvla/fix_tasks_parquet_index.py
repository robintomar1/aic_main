#!/usr/bin/env python3
"""One-shot fixer: rewrite tasks.parquet so it's INDEXED by the task string
(canonical lerobot format), not the default RangeIndex.

Why: see memory `feedback_lerobot_tasks_parquet.md`. SmolVLA's tokenizer
needs `meta.tasks.iloc[task_idx].name` to return the task string — that
only happens when the parquet's index VALUES are the task strings.
Datasets built with `pa.Table.from_pylist([{...}])` have a default
RangeIndex and break this contract.

Usage:
    pixi run python my_policy/scripts/smolvla/fix_tasks_parquet_index.py \\
        /root/aic_data/v9_act_build/v9_port_local_merged_clean/meta/tasks.parquet \\
        /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset/meta/tasks.parquet

After running, validate end-to-end with:
    pixi run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \\
        ds = LeRobotDataset(repo_id='local/probe', \\
                            root='<dataset_root>', video_backend='pyav'); \\
        item = ds[0]; print('task:', type(item['task']).__name__, repr(item['task']))"
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def fix(path: Path) -> None:
    df = pd.read_parquet(path)
    if "task" not in df.columns or "task_index" not in df.columns:
        # Already fixed (canonical writer omits the redundant `task` column;
        # only `task_index` survives, with task strings as the index).
        if df.index.name == "task" and df.index.dtype == object:
            print(f"  {path} — already canonical (skipping)")
            return
        sys.exit(f"{path}: unexpected columns {df.columns.tolist()} / index {df.index}")
    fixed = pd.DataFrame(
        {"task_index": df["task_index"].tolist()},
        index=pd.Index(df["task"].tolist(), name="task"),
    )
    fixed.to_parquet(path)
    print(f"  {path}")
    print(f"    index[0] = {fixed.index[0]!r}")
    print(f"    rows     = {len(fixed)}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("usage: fix_tasks_parquet_index.py <tasks.parquet> [<tasks.parquet> ...]")
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            sys.exit(f"missing: {p}")
        fix(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
