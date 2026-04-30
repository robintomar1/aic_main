#!/usr/bin/env python3
"""One-shot calibration of port-in-board quaternions for PortLocalizer.

The localizer predicts (board_x, board_y, sin_yaw, cos_yaw, rail_t). The full
port pose adds a static port-in-board orientation that depends on
(port_type, port_name). `labels.calibrate_port_in_board_rotations` extracts
this empirically by reading recorded `groundtruth.port_pose` quaternions
from a training dataset and inverting the per-trial yaw.

Run once after training, dump the result to `<checkpoint>.quats.json`.
PortLocalizer auto-loads that file at construction.

Usage:
    pixi run python my_policy/scripts/calibrate_localizer_quats.py \\
        --checkpoint /root/aic_data/localizer_v7.pt \\
        --collection-dir /root/aic_data \\
        --batch batch_100_a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.localizer.labels import calibrate_port_in_board_rotations  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to trained localizer checkpoint; result is written to "
                        "<checkpoint>.quats.json next to it.")
    p.add_argument("--collection-dir", type=Path, required=True,
                   help="Recorder output dir containing the training batches.")
    p.add_argument("--batch", type=str, required=True,
                   help="Name of the batch to calibrate against (any of the "
                        "training batches works — port-in-board is geometric).")
    p.add_argument("--out", type=Path, default=None,
                   help="Override output JSON path (default: <checkpoint>.quats.json).")
    args = p.parse_args()

    # Recorder layout (matches MultiBatchLocalizerDataset.from_collection_dir):
    #   <collection_dir>/<batch>/                  → dataset_root
    #   <collection_dir>/<batch>.yaml              → batch yaml
    #   <collection_dir>/<batch>_logs/summary.json → recorder summary
    dataset_root = args.collection_dir / args.batch
    batch_yaml = args.collection_dir / f"{args.batch}.yaml"
    summary_json = args.collection_dir / f"{args.batch}_logs" / "summary.json"

    for path, label in [
        (dataset_root, "dataset_root"),
        (batch_yaml, "batch_yaml"),
        (summary_json, "summary_json"),
    ]:
        if not path.exists():
            print(f"error: {label} not found at {path}", file=sys.stderr)
            return 1

    print(f"calibrating from collection_dir={args.collection_dir} batch={args.batch}")
    quats = calibrate_port_in_board_rotations(
        dataset_root=dataset_root,
        batch_yaml=batch_yaml,
        summary_json=summary_json,
    )
    if not quats:
        print("error: calibration produced no entries — empty dataset?", file=sys.stderr)
        return 1

    print(f"calibrated {len(quats)} (port_type, port_name) pairs:")
    for (pt, pn), (qw, qx, qy, qz) in sorted(quats.items()):
        print(f"  ({pt}, {pn}) -> ({qw:+.4f}, {qx:+.4f}, {qy:+.4f}, {qz:+.4f})")

    out_path = args.out or args.checkpoint.with_suffix(args.checkpoint.suffix + ".quats.json")
    serializable = {
        f"{pt}|{pn}": [float(qw), float(qx), float(qy), float(qz)]
        for (pt, pn), (qw, qx, qy, qz) in quats.items()
    }
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
