#!/usr/bin/env python3
"""Build meta/stats.json for an ACT dataset by aggregating per-episode stats.

`build_act_dataset.py` / `merge_act_datasets.py` write per-episode stats
into the episode meta parquet but never produce the dataset-level
`meta/stats.json`. lerobot's `make_dataset` needs that file (it indexes
`dataset.meta.stats[key][stat_type]` to overlay imagenet stats on camera
keys), so without it training crashes with
`'NoneType' object is not subscriptable`.

This is a thin CLI around `my_policy.act.stats.aggregate_stats_to_json`,
which is also called automatically by `build_act_dataset_slim.py` as its
final step.

Run:
    pixi run python my_policy/scripts/act/aggregate_act_stats.py \\
        --root /root/aic_data/v9_act_build/v9_act_merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

from my_policy.act.stats import aggregate_stats_to_json


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--root", type=Path, required=True,
        help="Dataset root (contains meta/episodes/, meta/info.json, etc.)",
    )
    args = p.parse_args()

    print(f"aggregating per-episode stats under {args.root}...")
    agg = aggregate_stats_to_json(args.root)
    print(f"wrote {args.root/'meta'/'stats.json'}")
    print(f"aggregated keys: {sorted(agg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
