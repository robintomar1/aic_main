#!/usr/bin/env python3
"""Compare two `eval_offline_action_mae.py` JSON outputs and apply the
decision rule from the v9-port-local plan.

Decision rule (from `/home/robin/.claude/plans/check-out-the-plan-quizzical-donut.md`,
"Day 1 Immediate Next Action"):

  If port-local val pos-MAE ≤ 0.5× world-frame val pos-MAE
    → port-local works as expected; commit the next 5 days to it.
  If 0.5× < ratio ≤ 0.8×
    → marginal improvement; pivot probably still helps but the gain is
      small. Decide based on remaining budget.
  If ratio > 0.8×
    → architectural assumption was wrong; don't pivot, triage RunACT
      with the world-frame baseline instead.

Why pos-MAE in raw meters is the fair comparison: both port-local and
world-frame transforms are RIGID, so 1 mm of action error in either
frame = 1 mm of physical world error. Comparing MAE in raw units is
apples-to-apples.

Usage:
    python3 my_policy/scripts/compare_eval_runs.py \\
        --port-local /tmp/v9_pl_v1_step10k_val_mae.json \\
        --world-frame /tmp/v9_act_v1_step10k_val_mae.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _row(label: str, pl_val: float, wf_val: float, fmt: str = "{:.4f}") -> str:
    """Format one comparison row. Highlights the lower (better) value."""
    pl_s = fmt.format(pl_val)
    wf_s = fmt.format(wf_val)
    if pl_val < wf_val:
        marker = "  ← port-local lower"
    elif wf_val < pl_val:
        marker = "  ← world-frame lower"
    else:
        marker = "  (equal)"
    return f"  {label:<22} {pl_s:>12}   {wf_s:>12}{marker}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--port-local", type=Path, required=True,
                   help="JSON output from eval_offline_action_mae.py for the "
                        "port-local checkpoint+dataset.")
    p.add_argument("--world-frame", type=Path, required=True,
                   help="JSON output from eval_offline_action_mae.py for the "
                        "world-frame checkpoint+dataset.")
    args = p.parse_args()

    pl = json.loads(args.port_local.read_text())
    wf = json.loads(args.world_frame.read_text())

    print(f"port-local : {pl['checkpoint_dir']}")
    print(f"             {pl['dataset_root']}")
    print(f"             {pl['n_episodes']} eps / {pl['n_frames']} frames "
          f"(wall {pl['wall_time_s']:.0f}s)")
    print()
    print(f"world-frame: {wf['checkpoint_dir']}")
    print(f"             {wf['dataset_root']}")
    print(f"             {wf['n_episodes']} eps / {wf['n_frames']} frames "
          f"(wall {wf['wall_time_s']:.0f}s)")
    print()

    if pl["n_episodes"] != wf["n_episodes"]:
        print(f"WARNING: episode count mismatch ({pl['n_episodes']} vs "
              f"{wf['n_episodes']}); val splits should be identical with "
              f"seed=42. Comparison may not be apples-to-apples.")
        print()

    print(f"{'Metric':<24} {'port-local':>12}   {'world-frame':>12}")
    print(f"  {'-' * 80}")
    for stat in ("mean", "median", "p95", "p99", "max"):
        print(_row(f"pos {stat} (m)",
                   pl["pos_err_m"][stat], wf["pos_err_m"][stat]))
    print()
    for stat in ("mean", "median", "p95", "p99", "max"):
        print(_row(f"rot {stat} (deg)",
                   pl["rot_err_deg"][stat], wf["rot_err_deg"][stat], "{:.2f}"))
    print()
    pl_pd = pl["per_dim_mae"]
    wf_pd = wf["per_dim_mae"]
    dim_names = ["px", "py", "pz", "qx", "qy", "qz", "qw"]
    print(f"  per-dim L1 MAE:")
    for n, (a, b) in zip(dim_names, zip(pl_pd, wf_pd)):
        print(_row(f"  {n}", a, b))

    # ------------------------------------------------------------------
    # Decision rule.
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    pl_pos = pl["pos_err_m"]["mean"]
    wf_pos = wf["pos_err_m"]["mean"]
    if wf_pos <= 0:
        ratio = float("inf")
    else:
        ratio = pl_pos / wf_pos
    print(f"Position-MAE ratio  port-local / world-frame  =  {ratio:.3f}")
    print()
    if ratio <= 0.5:
        verdict = (
            "STRONG WIN: port-local pos-MAE is ≤ 0.5× world-frame.\n"
            "  → Commit the remaining days to port-local. Train to convergence,\n"
            "    benchmark vs the existing world-frame submission."
        )
    elif ratio <= 0.8:
        verdict = (
            "MARGINAL: port-local helps but the gain is < 2×.\n"
            "  → Pivot probably still helps, but factor in the cost of\n"
            "    integrating port-local at inference (localizer + transform).\n"
            "    With ~5 days left to deadline, decide based on whether\n"
            "    that integration risk is worth the moderate MAE drop."
        )
    elif ratio < 1.0:
        verdict = (
            "WEAK: port-local barely helps.\n"
            "  → Architectural assumption was less load-bearing than expected.\n"
            "    Don't sink more time here; triage RunACT with the world-frame\n"
            "    baseline instead (workstream A from the master plan)."
        )
    else:
        verdict = (
            "REGRESSION: port-local is WORSE than world-frame.\n"
            "  → Something went wrong in the pivot. Possible causes:\n"
            "    1. Distribution-shift effect: port-local val episodes look\n"
            "       different from train episodes in some way world-frame\n"
            "       handles by accident.\n"
            "    2. Bug in the transform we missed despite Tier 1+2 being\n"
            "       green (e.g. an asymmetry that round-trip tests don't catch).\n"
            "    3. Port-local distribution is so tight that the model\n"
            "       overfits in 10k steps.\n"
            "  → Investigate before committing further. Don't pivot."
        )
    print(verdict)
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
