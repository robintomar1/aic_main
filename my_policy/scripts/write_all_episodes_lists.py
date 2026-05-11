#!/usr/bin/env python3
"""Write `<batch>_act_all_episodes.json` per batch as the union of
existing `<batch>_act_clean_episodes.json` and `<batch>_act_messy_episodes.json`.

Used to rebuild the SmolVLA training dataset with the force-gate "correction"
episodes included (no force filtering, just the data-quality fixes that
`make_port_local_dataset.py` already applies).

Usage (inside pixi):
    pixi run python my_policy/scripts/write_all_episodes_lists.py \\
        --out-root /root/aic_data/v9_act_build

Leaves existing clean/messy JSONs untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_BATCHES = [
    "batch_100_a",
    "batch_100_b",
    "batch_100_c",
    "batch_100_d",
    "batch_100_e",
    "batch_500_a",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, required=True,
                   help="Directory holding `<batch>_act_clean_episodes.json` "
                        "and `<batch>_act_messy_episodes.json` (e.g. "
                        "/root/aic_data/v9_act_build).")
    p.add_argument("--batches", nargs="+", default=DEFAULT_BATCHES)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing *_act_all_episodes.json files.")
    args = p.parse_args()

    totals = {"clean": 0, "messy": 0, "all": 0}
    for batch in args.batches:
        clean_path = args.out_root / f"{batch}_act_clean_episodes.json"
        messy_path = args.out_root / f"{batch}_act_messy_episodes.json"
        all_path = args.out_root / f"{batch}_act_all_episodes.json"
        if not clean_path.exists():
            sys.exit(f"missing {clean_path}")
        if not messy_path.exists():
            sys.exit(f"missing {messy_path}")
        if all_path.exists() and not args.overwrite:
            print(f"[skip] {all_path} exists (pass --overwrite to replace)")
            continue
        clean = set(json.loads(clean_path.read_text()))
        messy = set(json.loads(messy_path.read_text()))
        overlap = clean & messy
        if overlap:
            sys.exit(f"clean/messy overlap in {batch}: {sorted(overlap)}")
        union = sorted(clean | messy)
        all_path.write_text(json.dumps(union))
        totals["clean"] += len(clean)
        totals["messy"] += len(messy)
        totals["all"] += len(union)
        print(f"{batch}: clean={len(clean)} + messy={len(messy)} -> all={len(union)} -> {all_path}")
    print(f"\nTOTAL: clean={totals['clean']} + messy={totals['messy']} -> all={totals['all']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
