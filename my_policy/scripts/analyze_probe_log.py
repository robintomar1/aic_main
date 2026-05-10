#!/usr/bin/env python3
"""Offline analyzer for probe.jsonl logs (CHEATCODE_PROBE_LOG output).

Computes Stage 1 pass criteria for the tilt-probe physics check:

  1. ΔF_range = max(F) − min(F) across 8 probe directions:
     ≥ 3N typical (median), ≥ 1N minimum (5th percentile).
     Confirms the probe physics gives a discriminating force signal in
     this sim despite the cable-tension noise floor (~5–10N drift, see
     reference_aic_ft_sensor.md).

  2. Direction-decode accuracy: probe-identified angle within ±60° of
     the true GT-TF (port − plug) angle in plug-local frame.
     Pass if ≥ 70% of probes pass.

  3. Outcome distribution: how many translated, retried, exhausted.

Usage:
  python my_policy/scripts/analyze_probe_log.py /root/aic_data/probe_smoke_logs/probe.jsonl

Exits 0 on PASS, 1 on FAIL (any criterion missed).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def angle_diff(a: float, b: float) -> float:
    """Minimum-magnitude angular difference, in [-π, π]."""
    return ((a - b) + math.pi) % (2.0 * math.pi) - math.pi


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log_path", type=Path, help="Path to probe.jsonl")
    p.add_argument("--df-min-typical-n", type=float, default=3.0,
                   help="Median ΔF must exceed this. Default 3.0N.")
    p.add_argument("--df-min-floor-n", type=float, default=1.0,
                   help="5th-percentile ΔF must exceed this. Default 1.0N.")
    p.add_argument("--accuracy-tolerance-deg", type=float, default=60.0,
                   help="Probe-identified direction must be within this many "
                        "degrees of GT for the probe to count as 'correct'. "
                        "Default 60° (matches plan; ±360°/n_directions = ±45° "
                        "is the resolution floor of the 8-direction sweep).")
    p.add_argument("--accuracy-pass-rate", type=float, default=0.7,
                   help="Fraction of probes that must be 'correct' to pass. "
                        "Default 0.7.")
    args = p.parse_args()

    if not args.log_path.exists():
        print(f"ERROR: log path does not exist: {args.log_path}", file=sys.stderr)
        return 1

    records = []
    with args.log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as ex:
                print(f"WARN: skipping malformed line: {ex}", file=sys.stderr)

    if not records:
        print(f"ERROR: no records in {args.log_path}", file=sys.stderr)
        return 1

    print(f"Loaded {len(records)} probe records from {args.log_path}")
    print()

    # Outcome distribution
    outcomes = {}
    for r in records:
        outcomes[r.get("outcome", "")] = outcomes.get(r.get("outcome", ""), 0) + 1
    print("Outcomes:")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome:20s}  {count:4d}  ({count / len(records) * 100:5.1f}%)")
    print()

    # ΔF range distribution (only over translated probes — others may have widened
    # tilt and we want each tilt's discrimination)
    dfs = [r.get("force_range_n", 0.0) for r in records]
    df_median = statistics.median(dfs)
    df_p05 = statistics.quantiles(dfs, n=20)[0] if len(dfs) >= 20 else min(dfs)
    df_p95 = statistics.quantiles(dfs, n=20)[18] if len(dfs) >= 20 else max(dfs)
    df_max = max(dfs)
    df_min = min(dfs)
    print("ΔF range (max(F) − min(F) across 8 probe directions):")
    print(f"  median:    {df_median:.2f}N")
    print(f"  p05:       {df_p05:.2f}N")
    print(f"  p95:       {df_p95:.2f}N")
    print(f"  min/max:   {df_min:.2f}N / {df_max:.2f}N")
    df_typical_pass = df_median >= args.df_min_typical_n
    df_floor_pass = df_p05 >= args.df_min_floor_n
    print(f"  PASS (median ≥ {args.df_min_typical_n}N)?   {df_typical_pass}")
    print(f"  PASS (p05 ≥ {args.df_min_floor_n}N)?      {df_floor_pass}")
    print()

    # Direction-decode accuracy: chosen vs GT angle
    accuracy_records = [
        r for r in records
        if r.get("outcome") == "translated"
        and r.get("chosen_angle_local_rad") is not None
    ]
    if not accuracy_records:
        print("WARN: no translated records to evaluate decode accuracy")
        accuracy_pass = False
        n_correct = 0
        n_total = 0
    else:
        n_total = len(accuracy_records)
        n_correct = 0
        max_err_deg = 0.0
        errs_deg = []
        for r in accuracy_records:
            err_rad = angle_diff(
                r["chosen_angle_local_rad"], r["gt_angle_local_rad"])
            err_deg = abs(math.degrees(err_rad))
            errs_deg.append(err_deg)
            if err_deg <= args.accuracy_tolerance_deg:
                n_correct += 1
            max_err_deg = max(max_err_deg, err_deg)
        rate = n_correct / n_total
        accuracy_pass = rate >= args.accuracy_pass_rate
        print(f"Direction decode (chosen angle vs GT angle in plug-local frame):")
        print(f"  total translated:    {n_total}")
        print(f"  correct (≤{args.accuracy_tolerance_deg:.0f}°):   {n_correct}  "
              f"({rate * 100:.1f}%)")
        print(f"  median |err|:        {statistics.median(errs_deg):.1f}°")
        print(f"  p95 |err|:           "
              f"{(statistics.quantiles(errs_deg, n=20)[18] if len(errs_deg) >= 20 else max(errs_deg)):.1f}°")
        print(f"  max |err|:           {max_err_deg:.1f}°")
        print(f"  PASS (rate ≥ {args.accuracy_pass_rate * 100:.0f}%)?   {accuracy_pass}")
    print()

    # Retry distribution
    retries = [r.get("retry_count", 0) for r in records]
    retry_dist = {}
    for r in retries:
        retry_dist[r] = retry_dist.get(r, 0) + 1
    print("Retry count distribution:")
    for k, v in sorted(retry_dist.items()):
        print(f"  retries={k}: {v} ({v / len(records) * 100:.1f}%)")
    print()

    # Overall verdict
    all_pass = df_typical_pass and df_floor_pass and accuracy_pass
    print("=" * 60)
    print(f"OVERALL: {'PASS — Stage 1 physics check OK' if all_pass else 'FAIL — see criteria above'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
