#!/usr/bin/env python3
"""Tally success rates from a series of noise-bench recorder runs.

Each bench run produces a `summary.json` in <root>_logs/. This script walks
those summaries and prints success rate per noise level — the data needed to
set the localizer accuracy target.

Usage:
    pixi run python my_policy/scripts/analyze_noise_bench.py \\
        /root/aic_data/noise_bench_xy00mm \\
        /root/aic_data/noise_bench_xy10mm \\
        /root/aic_data/noise_bench_xy20mm \\
        /root/aic_data/noise_bench_xy50mm

    # Or supply a glob: pixi run python ... /root/aic_data/noise_bench_*

Prints per-level: trials, saved (full insertion), discarded breakdown,
success rate. Bench passes if success rate stays high (≥80%) up to the
expected localizer error budget.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def label_for(root: Path) -> str:
    """Pretty label derived from dataset dir name. Falls back to dir name."""
    name = root.name
    # Heuristic: strip leading "noise_bench_" if present.
    return name.replace("noise_bench_", "") if name.startswith("noise_bench_") else name


def analyze_one(root: Path) -> dict:
    """Return a small dict of stats for one run."""
    logs = root.parent / (root.name + "_logs") / "summary.json"
    if not logs.exists():
        return {"label": label_for(root), "error": f"summary.json not found at {logs}"}
    summary = json.loads(logs.read_text())
    trials = summary.get("trials", [])
    outcomes = Counter(t.get("outcome", "?") for t in trials)
    n = len(trials)
    saved = outcomes.get("saved_inserted", 0)
    return {
        "label": label_for(root),
        "n": n,
        "saved": saved,
        "rate": saved / n if n else 0.0,
        "outcomes": dict(outcomes),
        "events_observed": summary.get("events_observed", 0),
        "save_stats": summary.get("save_stats", {}),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", type=Path,
                   help="Bench dataset directories (one per noise level).")
    args = p.parse_args()

    print(f"{'level':<14s} {'trials':>6s} {'saved':>6s} {'rate':>6s}  outcomes")
    print("-" * 80)
    rows = [analyze_one(r) for r in args.roots]
    for r in rows:
        if "error" in r:
            print(f"{r['label']:<14s}  {r['error']}")
            continue
        outcomes = " ".join(f"{k}={v}" for k, v in sorted(r["outcomes"].items()))
        rate_pct = 100.0 * r["rate"]
        print(f"{r['label']:<14s} {r['n']:>6d} {r['saved']:>6d} {rate_pct:>5.0f}%  {outcomes}")

    # Find the elbow: highest noise level where rate is still ≥ 80%.
    okay = [r for r in rows if "error" not in r and r["rate"] >= 0.8]
    if okay:
        print()
        # Order by label so the user can eyeball the elbow themselves.
        labels = ", ".join(r["label"] for r in okay)
        print(f"≥80% success at: {labels}")
    else:
        print("\nNo level reached 80% success.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
