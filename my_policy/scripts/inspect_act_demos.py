#!/usr/bin/env python3
"""Phase A0 — classify saved oracle demos as clean vs messy for ACT training.

ACT can only learn what's in the demos. Of the 354 saved trials, some
inserted via a clean approach-align-descend sequence; others recovered via
the force-gate retreat cycle and/or the spiral search. The messy ones teach
the model to wiggle arbitrarily during INSERT — a real failure mode.

This script walks each saved episode's force history (wrench.fx/fy/fz from
observation.state) and tags it as:

  clean : peak force magnitude stayed below FORCE_GATE_N throughout AND
          fewer than CHAMFER_BAND_FRAME_BUDGET frames spent in the
          chamfer-contact band [SPIRAL_LO_N, FORCE_GATE_N).
  messy : at least one force-gate engagement OR sustained chamfer-band
          contact (proxy for spiral search).

Output:
  <collection-dir>/<batch>_act_clean_episodes.json    list[int]
  <collection-dir>/<batch>_act_messy_episodes.json    list[int]
  stdout: summary table per batch + per-task-type breakdown.

Run:
    python3 my_policy/scripts/inspect_act_demos.py \\
        --collection-dir /root/aic_data \\
        --batches batch_100_a batch_100_b batch_100_c batch_100_d batch_100_e

Pure numpy + pyarrow + yaml — no torch / lerobot / rclpy needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.localizer.labels import match_episodes_to_trials  # noqa: E402


# Mirrors CheatCodeRobust constants — keep in sync with my_policy/ros/CheatCodeRobust.py.
FORCE_GATE_N = 18.0          # CheatCodeRobust.FORCE_STOP_N
SPIRAL_LO_N = 8.0            # CheatCodeRobust.SPIRAL_FORCE_LO_N
# How many frames in the chamfer-contact band before we call the trial "messy".
# At 20Hz, 10 frames = 0.5s of contact. Spiral search runs at 0.5Hz so a single
# revolution = 2s = 40 frames, but transient contact during ALIGN is normal.
# 10 is a budget that catches sustained contact without flagging brief brushes.
CHAMFER_BAND_FRAME_BUDGET = 10


def _load_state_names(dataset_root: Path) -> list[str]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    return list(info["features"]["observation.state"]["names"])


def _wrench_force_indices(state_names: list[str]) -> list[int]:
    """Slice indices for wrench.fx, wrench.fy, wrench.fz inside observation.state."""
    return [state_names.index(f"wrench.f{c}") for c in "xyz"]


def classify_episode(force_mag: np.ndarray) -> tuple[str, dict]:
    """Returns (label, stats) for a single episode's force-magnitude trace."""
    peak = float(force_mag.max()) if len(force_mag) else 0.0
    frames_above_lo = int((force_mag >= SPIRAL_LO_N).sum())
    frames_above_gate = int((force_mag >= FORCE_GATE_N).sum())
    if frames_above_gate > 0:
        label = "messy"  # any force-gate engagement is messy by definition
    elif frames_above_lo > CHAMFER_BAND_FRAME_BUDGET:
        label = "messy"  # sustained chamfer-band contact = spiral search likely
    else:
        label = "clean"
    return label, {
        "peak_force_n": peak,
        "frames_above_lo": frames_above_lo,
        "frames_above_gate": frames_above_gate,
        "n_frames": int(len(force_mag)),
    }


def process_batch(
    collection_dir: Path, batch_name: str,
) -> tuple[list[int], list[int], list[dict]]:
    """Returns (clean_eps, messy_eps, per_episode_records) for one batch."""
    dataset_root = collection_dir / batch_name
    batch_yaml = collection_dir / f"{batch_name}.yaml"
    summary_json = collection_dir / f"{batch_name}_logs" / "summary.json"

    for path, label in [(dataset_root, "dataset_root"),
                        (batch_yaml, "batch_yaml"),
                        (summary_json, "summary_json")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found at {path}")

    state_names = _load_state_names(dataset_root)
    f_idx = _wrench_force_indices(state_names)

    parquet_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(
        str(parquet_path),
        columns=["observation.state", "episode_index"],
    )
    eps = table["episode_index"].to_numpy().astype(np.int64)
    states = np.stack([
        np.asarray(r, dtype=np.float32)
        for r in table["observation.state"].to_pylist()
    ])

    cfg = yaml.safe_load(batch_yaml.read_text())
    summary = json.loads(summary_json.read_text())
    ep_to_trial = match_episodes_to_trials(summary, cfg["trials"])

    unique_eps = sorted(np.unique(eps).tolist())
    clean_eps: list[int] = []
    messy_eps: list[int] = []
    records: list[dict] = []

    for ep in unique_eps:
        mask = eps == ep
        ep_states = states[mask]
        # Force magnitude per frame: |F| = sqrt(fx² + fy² + fz²).
        forces = ep_states[:, f_idx]
        force_mag = np.sqrt((forces * forces).sum(axis=1))
        label, stats = classify_episode(force_mag)

        # Annotate with task identity for the per-task-type summary.
        trial_key = ep_to_trial.get(int(ep))
        task = (cfg["trials"][trial_key]["tasks"]["task_1"] if trial_key
                else {"port_type": "?", "target_module_name": "?", "port_name": "?"})

        records.append({
            "batch": batch_name,
            "episode_index": int(ep),
            "label": label,
            "trial_key": trial_key,
            "port_type": task.get("port_type", "?"),
            "target_module_name": task.get("target_module_name", "?"),
            "port_name": task.get("port_name", "?"),
            **stats,
        })
        (clean_eps if label == "clean" else messy_eps).append(int(ep))

    return clean_eps, messy_eps, records


def print_summary(all_records: list[dict]) -> None:
    """Pretty-print the per-batch and per-task-type breakdowns."""
    by_batch: dict[str, dict[str, int]] = {}
    by_port_type: dict[str, dict[str, int]] = {}
    by_pair: dict[tuple[str, str], dict[str, int]] = {}
    for r in all_records:
        by_batch.setdefault(r["batch"], {"clean": 0, "messy": 0})[r["label"]] += 1
        by_port_type.setdefault(r["port_type"], {"clean": 0, "messy": 0})[r["label"]] += 1
        key = (r["target_module_name"], r["port_name"])
        by_pair.setdefault(key, {"clean": 0, "messy": 0})[r["label"]] += 1

    total_clean = sum(b["clean"] for b in by_batch.values())
    total_messy = sum(b["messy"] for b in by_batch.values())
    total = total_clean + total_messy

    print("\n=== Per-batch summary ===")
    print(f"{'batch':<20}  {'clean':>5}  {'messy':>5}  {'clean%':>7}")
    for batch, counts in sorted(by_batch.items()):
        n = counts["clean"] + counts["messy"]
        pct = 100.0 * counts["clean"] / max(n, 1)
        print(f"{batch:<20}  {counts['clean']:>5}  {counts['messy']:>5}  {pct:>6.1f}%")
    pct = 100.0 * total_clean / max(total, 1)
    print(f"{'TOTAL':<20}  {total_clean:>5}  {total_messy:>5}  {pct:>6.1f}%")

    print("\n=== By port_type ===")
    for ptype, counts in sorted(by_port_type.items()):
        n = counts["clean"] + counts["messy"]
        pct = 100.0 * counts["clean"] / max(n, 1)
        print(f"{ptype:<10}  clean={counts['clean']:>3}  messy={counts['messy']:>3}  ({pct:>4.1f}%)")

    print("\n=== By (target_module, port_name) ===")
    for (mod, port), counts in sorted(by_pair.items()):
        n = counts["clean"] + counts["messy"]
        pct = 100.0 * counts["clean"] / max(n, 1)
        print(f"  {mod:<22} {port:<14}  clean={counts['clean']:>3}  "
              f"messy={counts['messy']:>3}  ({pct:>4.1f}%)")

    print(f"\n=== Decision gate ===")
    print(f"clean ratio = {pct:.1f}%  (target ≥50% to train on clean only)")
    if pct >= 50.0:
        print("→ recommend training on clean-only filtered dataset.")
    else:
        print("→ recommend training on all 354 episodes (clean ratio too low).")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir", type=Path, required=True,
                   help="Where the recorder wrote its batches (read-only OK).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write the per-batch JSON files (default: same as --collection-dir; "
                        "use this when --collection-dir is a read-only mount).")
    p.add_argument("--batches", type=str, nargs="+", required=True)
    p.add_argument("--out-records-json", type=Path, default=None,
                   help="Optional: write the full per-episode records list as one JSON file (for offline analysis).")
    args = p.parse_args()

    out_dir = args.out_dir or args.collection_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for batch in args.batches:
        print(f"processing {batch}...")
        clean_eps, messy_eps, records = process_batch(args.collection_dir, batch)
        clean_path = out_dir / f"{batch}_act_clean_episodes.json"
        messy_path = out_dir / f"{batch}_act_messy_episodes.json"
        clean_path.write_text(json.dumps(clean_eps))
        messy_path.write_text(json.dumps(messy_eps))
        print(f"  wrote {clean_path.name} ({len(clean_eps)} eps), "
              f"{messy_path.name} ({len(messy_eps)} eps)")
        all_records.extend(records)

    if args.out_records_json is not None:
        args.out_records_json.write_text(json.dumps(all_records, indent=2))
        print(f"wrote full records → {args.out_records_json}")

    print_summary(all_records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
