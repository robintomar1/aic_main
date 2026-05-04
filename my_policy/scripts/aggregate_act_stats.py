#!/usr/bin/env python3
"""Build meta/stats.json for an ACT dataset by aggregating per-episode stats.

`build_act_dataset.py` writes per-episode stats into the episode meta parquet
(columns `stats/<feature>/<min|max|mean|std|count|q01..q99>`) but never
produces the dataset-level `meta/stats.json`. lerobot's `make_dataset` needs
that file (it indexes `dataset.meta.stats[key][stat_type]` to overlay
imagenet stats on camera keys), so without it training crashes with
`'NoneType' object is not subscriptable`.

This script:
  1. Reads every per-episode parquet under `<root>/meta/episodes/`.
  2. Unflattens the `stats/...` columns to per-episode dicts.
  3. Calls lerobot's `aggregate_stats` to produce dataset-level stats.
  4. Adds empty dicts for camera keys (the imagenet override populates them).
  5. Writes `meta/stats.json` via lerobot's `write_stats`.

Run:
    pixi run python my_policy/scripts/aggregate_act_stats.py \\
        --root /root/aic_data/v9_act_build/v9_act_merged
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _per_episode_stats(ep_table) -> list[dict]:
    """Return list of per-episode stats dicts from an episodes parquet table."""
    cols = [c for c in ep_table.column_names if c.startswith("stats/")]
    out = []
    for row_i in range(ep_table.num_rows):
        d: dict[str, dict[str, np.ndarray]] = {}
        for c in cols:
            _, feat, stat = c.split("/", 2)
            v = ep_table.column(c)[row_i].as_py()
            d.setdefault(feat, {})[stat] = np.asarray(v)
        out.append(d)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="Dataset root (contains meta/episodes/, meta/info.json, etc.)")
    args = p.parse_args()

    info = json.loads((args.root / "meta" / "info.json").read_text())
    camera_keys = [k for k, ft in info["features"].items()
                   if ft.get("dtype") == "video"]

    ep_files = sorted((args.root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        print(f"error: no episode parquets under {args.root/'meta/episodes'}")
        return 1

    per_ep: list[dict] = []
    for f in ep_files:
        per_ep.extend(_per_episode_stats(pq.read_table(f)))
    print(f"loaded per-episode stats for {len(per_ep)} episodes from "
          f"{len(ep_files)} parquet file(s)")

    # Defer lerobot import until after CLI parse (slow).
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats

    agg = aggregate_stats(per_ep)
    print(f"aggregated stats keys: {sorted(agg)}")

    # make_dataset overlays imagenet stats onto camera keys; it indexes
    # stats[key][stat_type] so the key must exist as a dict.
    for cam in camera_keys:
        agg.setdefault(cam, {})
    print(f"added empty placeholder for {len(camera_keys)} camera key(s): "
          f"{camera_keys}")

    write_stats(agg, args.root)
    print(f"wrote {args.root/'meta'/'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
