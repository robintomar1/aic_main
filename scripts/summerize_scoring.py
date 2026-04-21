#!/usr/bin/env python3
"""
Summarize AIC scoring.yaml files.

Usage:
    # Single run (path can be a scoring.yaml or a dir containing one)
    python summarize_scoring.py ~/aic_results/cheatcode

    # Compare multiple runs
    python summarize_scoring.py ~/aic_results/cheatcode ~/aic_results/runact ~/aic_results/wavearm

    # Shell glob also works
    python summarize_scoring.py ~/aic_results/*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "pyyaml not installed. `pip install pyyaml` or run inside pixi.\n"
    )
    sys.exit(1)


# Category keys inside tier_2. Source YAML uses spaces; we normalize.
CATEGORY_ALIASES = {
    "trajectory smoothness": "smooth",
    "trajectory_smoothness": "smooth",
    "duration": "dur",
    "trajectory efficiency": "eff",
    "trajectory_efficiency": "eff",
    "insertion force": "force",
    "insertion_force": "force",
    "contacts": "contact",
}

COLS = ["T1", "smooth", "dur", "eff", "force", "contact", "T3", "trial_total"]


def load_scoring(path: Path) -> tuple[str, dict] | None:
    """Return (run_label, parsed_yaml) or None on failure."""
    if path.is_dir():
        yaml_path = path / "scoring.yaml"
        label = path.name
    else:
        yaml_path = path
        label = path.parent.name if path.parent.name else path.stem
    if not yaml_path.exists():
        sys.stderr.write(f"[skip] no scoring.yaml at {yaml_path}\n")
        return None
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except yaml.YAMLError as e:
        sys.stderr.write(f"[skip] parse error in {yaml_path}: {e}\n")
        return None
    if not isinstance(data, dict):
        sys.stderr.write(f"[skip] unexpected top-level type in {yaml_path}\n")
        return None
    return label, data


def extract_trial_row(trial: dict) -> dict:
    """Pull the seven numeric fields out of one trial dict."""
    row = {c: "-" for c in COLS}
    row["T1"] = _safe_score(trial.get("tier_1"))
    row["T3"] = _safe_score(trial.get("tier_3"))

    t2 = trial.get("tier_2") or {}
    t2_score = _safe_score(t2)
    cats = t2.get("categories") or {}
    for raw_key, val in cats.items():
        norm = CATEGORY_ALIASES.get(raw_key.strip().lower())
        if norm is not None:
            row[norm] = _safe_score(val)

    # Per-trial total: tier_1 + tier_2 + tier_3
    try:
        row["trial_total"] = _num(row["T1"]) + _num(t2_score) + _num(row["T3"])
    except Exception:
        row["trial_total"] = "-"
    return row


def extract_trial_messages(trial: dict) -> list[str]:
    """Return short human-readable notes for failures/penalties worth surfacing."""
    notes = []
    t3 = trial.get("tier_3") or {}
    if _num(t3.get("score", 0)) <= 0 and t3.get("message"):
        notes.append(f"T3: {t3['message']}")
    cats = (trial.get("tier_2") or {}).get("categories") or {}
    contacts = cats.get("contacts") or {}
    if _num(contacts.get("score", 0)) < 0 and contacts.get("message"):
        msg = contacts["message"]
        if len(msg) > 140:
            msg = msg[:137] + "..."
        notes.append(f"contact: {msg}")
    force = cats.get("insertion force") or cats.get("insertion_force") or {}
    if _num(force.get("score", 0)) < 0 and force.get("message"):
        notes.append(f"force: {force['message']}")
    return notes


def _safe_score(obj):
    if isinstance(obj, dict) and "score" in obj:
        return obj["score"]
    return "-"


def _num(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def fmt(v) -> str:
    if isinstance(v, float):
        if v == 0:
            return "0"
        if v.is_integer():
            return f"{int(v):+d}"
        return f"{v:+.1f}"
    if isinstance(v, int):
        return f"{v:+d}" if v != 0 else "0"
    return str(v)


def print_run(label: str, data: dict) -> None:
    trials = sorted(
        [(k, v) for k, v in data.items() if k.startswith("trial_") and isinstance(v, dict)],
        key=lambda kv: kv[0],
    )
    total = data.get("total", "-")

    print(f"\n=== {label}  (total: {total}) ===")
    header = f"{'trial':<9} " + " ".join(f"{c:>11}" for c in COLS)
    print(header)
    print("-" * len(header))
    for name, trial in trials:
        row = extract_trial_row(trial)
        cells = " ".join(f"{fmt(row[c]):>11}" for c in COLS)
        print(f"{name:<9} {cells}")

    for name, trial in trials:
        for note in extract_trial_messages(trial):
            print(f"  [{name}] {note}")


def print_comparison(runs: list[tuple[str, dict]]) -> None:
    if len(runs) < 2:
        return
    print("\n=== comparison ===")
    name_w = max(len(label) for label, _ in runs)
    name_w = max(name_w, len("run"))
    header = f"{'run':<{name_w}} {'total':>8}  " + " ".join(
        f"{'t' + str(i):>7}" for i in range(1, 4)
    )
    print(header)
    print("-" * len(header))
    for label, data in runs:
        total = data.get("total", "-")
        trial_totals = []
        for i in (1, 2, 3):
            t = data.get(f"trial_{i}")
            if isinstance(t, dict):
                r = extract_trial_row(t)
                trial_totals.append(fmt(r["trial_total"]))
            else:
                trial_totals.append("-")
        print(
            f"{label:<{name_w}} {fmt(total):>8}  "
            + " ".join(f"{v:>7}" for v in trial_totals)
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize AIC scoring.yaml files.")
    ap.add_argument("paths", nargs="+", type=Path, help="scoring.yaml files or dirs")
    args = ap.parse_args()

    runs = []
    for p in args.paths:
        result = load_scoring(p)
        if result is not None:
            runs.append(result)

    if not runs:
        sys.stderr.write("No valid scoring.yaml files found.\n")
        return 1

    for label, data in runs:
        print_run(label, data)
    print_comparison(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())