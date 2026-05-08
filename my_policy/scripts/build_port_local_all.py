#!/usr/bin/env python3
"""Driver — build per-batch port-local datasets for all 6 oracle batches,
then merge into a single trainable dataset.

Calls `make_port_local_dataset.py` once per batch (skipping batches whose
output already exists, unless --force is given), then `merge_act_datasets.py`
with all 6 outputs as sources.

Why a single driver: keeps the GPU-side command paste-friendly and chains
the build + merge so a partial failure is visible at one place.

Usage (from repo root, inside pixi):
    pixi run python my_policy/scripts/build_port_local_all.py \\
        --collection-dir /root/aic_data \\
        --out-root /root/aic_data/v9_act_build

Defaults are sized to the actual batch layout on disk (verified 2026-05-08).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_BATCHES = [
    "batch_100_a",
    "batch_100_b",
    "batch_100_c",
    "batch_100_d",
    "batch_100_e",
    "batch_500_a",  # usable; only 6/86k frames are TF-lookup failures (see memory)
]

SCRIPTS_DIR = Path(__file__).resolve().parent
BUILDER = SCRIPTS_DIR / "make_port_local_dataset.py"
MERGER = SCRIPTS_DIR / "merge_act_datasets.py"
CLEANER = SCRIPTS_DIR / "clean_act_dataset.py"


def _run(cmd: list[str], step_name: str) -> None:
    """Run a subprocess; raise on non-zero. Streams stdout/stderr to console."""
    print(f"\n=== {step_name} ===")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(f"FAILED at step '{step_name}' (exit code {result.returncode})")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--collection-dir", type=Path, required=True,
                   help="Where the raw batch_*/ dirs live (e.g. /root/aic_data).")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Where to write per-batch + merged datasets "
                        "(e.g. /root/aic_data/v9_act_build).")
    p.add_argument("--out-name", type=str, default="v9_port_local_merged",
                   help="Subdir name for the merged dataset (default v9_port_local_merged).")
    p.add_argument("--batches", nargs="+", default=DEFAULT_BATCHES,
                   help=f"Batches to include. Default: {DEFAULT_BATCHES}")
    p.add_argument("--force", action="store_true",
                   help="Re-build batches whose output already exists, "
                        "and overwrite the merged dataset if it exists.")
    args = p.parse_args()

    if not BUILDER.exists():
        sys.exit(f"missing builder: {BUILDER}")
    if not MERGER.exists():
        sys.exit(f"missing merger: {MERGER}")
    if not CLEANER.exists():
        sys.exit(f"missing cleaner: {CLEANER}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    # Step 1: build per-batch port-local datasets.
    source_paths: list[Path] = []
    for batch in args.batches:
        out_dir = args.out_root / f"{batch}_port_local_dataset"
        clean_json = args.out_root / f"{batch}_act_clean_episodes.json"
        if not clean_json.exists():
            sys.exit(
                f"missing clean-episodes filter: {clean_json}\n"
                f"(produced by inspect_act_demos.py — re-run if needed)"
            )

        if out_dir.exists() and not args.force:
            print(f"\n[skip] {out_dir} already exists (pass --force to rebuild)")
            source_paths.append(out_dir)
            continue

        if out_dir.exists() and args.force:
            print(f"\n[force] removing {out_dir}")
            import shutil
            shutil.rmtree(out_dir)

        _run(
            [
                sys.executable, str(BUILDER),
                "--collection-dir", str(args.collection_dir),
                "--batch", batch,
                "--out-root", str(args.out_root),
                "--clean-episodes-json", str(clean_json),
            ],
            f"build {batch}",
        )
        source_paths.append(out_dir)

    # Step 2: merge all 6 into one dataset.
    merged_dir = args.out_root / args.out_name
    merge_cmd = [
        sys.executable, str(MERGER),
        "--sources", *(str(s) for s in source_paths),
        "--out-root", str(args.out_root),
        "--out-name", args.out_name,
    ]
    if args.force:
        merge_cmd.append("--force")
    _run(merge_cmd, f"merge {len(source_paths)} sources")

    # Step 3: run clean_act_dataset to write meta/stats.json.
    # We already applied Fix 1 + Fix 2 per-batch in make_port_local_dataset,
    # so the cleaner's action mods will be 0. The only useful side effect is
    # writing the aggregate stats.json that lerobot's `make_dataset` requires.
    # Without it, training fails with `'NoneType' object is not subscriptable`
    # in factory.make_dataset.
    clean_dir = args.out_root / f"{args.out_name}_clean"
    clean_cmd = [
        sys.executable, str(CLEANER),
        "--src", str(merged_dir),
        "--dst", str(clean_dir),
    ]
    if args.force:
        clean_cmd.append("--force")
    _run(clean_cmd, f"clean (write stats.json)")

    print(f"\n=== DONE ===")
    print(f"  per-batch outputs: {args.out_root}/<batch>_port_local_dataset/")
    print(f"  merged output:     {merged_dir}/")
    print(f"  trainable output:  {clean_dir}/  ← USE THIS FOR train_act.py --dataset-root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
