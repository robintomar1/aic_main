"""Aggregate per-episode stats into the dataset-level meta/stats.json.

`build_act_dataset.py`, `build_act_dataset_slim.py`, and
`merge_act_datasets.py` write per-episode stats into
`meta/episodes/.../file-*.parquet` but skip the dataset-level
`meta/stats.json`. lerobot's `make_dataset` then crashes when overlaying
imagenet stats onto camera keys (`'NoneType' object is not subscriptable`
because `dataset.meta.stats[key]` is None). This module computes that
missing file from the per-episode parquets.

Importable from both the CLI script (`scripts/act/aggregate_act_stats.py`)
and from the slim builder, which now calls it as its final step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _per_episode_stats(ep_table) -> list[dict]:
    """Unflatten `stats/<feature>/<stat>` columns of an episode parquet into
    a list of `{feature: {stat: np.ndarray}}` dicts — one per row."""
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


def aggregate_stats_to_json(root: Path) -> dict:
    """Aggregate per-episode stats under `<root>/meta/episodes/` into a
    dataset-level dict, add empty placeholders for video features (so
    lerobot's imagenet overlay can populate them), and write
    `<root>/meta/stats.json` via lerobot's `write_stats`.

    Returns the aggregated dict for caller inspection.

    Raises FileNotFoundError if no per-episode parquets exist.
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    camera_keys = [
        k for k, ft in info["features"].items() if ft.get("dtype") == "video"
    ]

    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(
            f"no episode parquets under {root/'meta/episodes'} — "
            f"is this a fully-built lerobot dataset?"
        )

    per_ep: list[dict] = []
    for f in ep_files:
        per_ep.extend(_per_episode_stats(pq.read_table(f)))

    # Defer lerobot import — keeps the module cheap to load when only
    # the helper `_per_episode_stats` is needed.
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats

    agg = aggregate_stats(per_ep)
    # make_dataset overlays imagenet stats onto camera keys; it indexes
    # stats[key][stat_type] so the key must exist as a dict, not None.
    for cam in camera_keys:
        agg.setdefault(cam, {})

    write_stats(agg, root)
    return agg
