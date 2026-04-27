#!/usr/bin/env python3
"""Tier 3 manual sanity check for localizer labels.

Walks an existing recorder dataset + batch YAML + summary.json, and prints a
per-episode table:

  episode | trial    | port_type | bx_bl    | by_bl    | yaw_deg | rail_t_mm |
          residual_mm (first..last)

Reading hint:
  - Board pose (x, y) should be in robot reach: typically x ∈ [−0.5, 0],
    y ∈ [−0.5, 0.5] roughly (depends on robot mount).
  - yaw_deg should span the full ±180° across episodes (when the dataset is
    randomized; smoke datasets cover only a subset).
  - rail_t_mm: |t| ≤ ~25 mm for NIC, ~60 mm for SC.
  - residual_mm: should be sub-millimeter; the killer test enforces this but
    the table is the human-readable view.

Usage:
  python3 my_policy/scripts/viz_localizer_labels.py <dataset_root>

  dataset_root = e.g. /root/aic_data/<batch>_dataset
  We auto-locate <batch>.yaml (sibling) and <batch>_dataset_logs/summary.json.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from my_policy.my_policy.localizer.dataset import LocalizerDataset  # noqa: E402
from my_policy.my_policy.localizer.labels import reconstruct_port_in_baselink  # noqa: E402


def _locate_artifacts(dataset_root: Path) -> tuple[Path, Path]:
    """From <dataset_root>, derive the batch YAML and summary.json paths.

    Convention: collect_lerobot.py is invoked with --batch-config /<dir>/<name>.yaml
    and --root /<dir>/<name>_dataset, producing --root + "_logs"/summary.json.
    """
    name = dataset_root.name
    suffix = "_dataset"
    if name.endswith(suffix):
        batch_name = name[: -len(suffix)]
    else:
        batch_name = name
    batch_yaml = dataset_root.parent / f"{batch_name}.yaml"
    summary_json = dataset_root.parent / f"{name}_logs" / "summary.json"
    return batch_yaml, summary_json


def _residual_for_frame(ds: LocalizerDataset, idx: int) -> float:
    sample = ds[idx]
    label = ds.label_for_episode(sample["_meta"]["episode_index"])
    target_module = sample["_meta"]["target_module_name"]
    port_name = sample["_meta"]["port_name"]
    predicted = reconstruct_port_in_baselink(label, target_module, port_name)
    recorded = ds.port_pose_baselink(idx)
    return float(np.linalg.norm(predicted - recorded))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dataset_root", type=Path,
                   help="LeRobot dataset directory (e.g. /root/aic_data/<batch>_dataset)")
    p.add_argument("--batch-yaml", type=Path, default=None,
                   help="Override batch YAML path. Default: derived from dataset_root.")
    p.add_argument("--summary-json", type=Path, default=None,
                   help="Override summary.json path. Default: derived from dataset_root.")
    args = p.parse_args()

    if args.batch_yaml is None or args.summary_json is None:
        derived_yaml, derived_summary = _locate_artifacts(args.dataset_root)
        batch_yaml = args.batch_yaml or derived_yaml
        summary_json = args.summary_json or derived_summary
    else:
        batch_yaml = args.batch_yaml
        summary_json = args.summary_json

    for required, label in [(args.dataset_root, "dataset_root"),
                            (batch_yaml, "batch_yaml"),
                            (summary_json, "summary_json")]:
        if not required.exists():
            print(f"error: {label} not found at {required}", file=sys.stderr)
            return 2

    ds = LocalizerDataset(args.dataset_root, batch_yaml, summary_json, cameras=())

    # Group frame indices by episode (in their dataset order).
    ep_to_frame_indices: dict[int, list[int]] = {}
    for i in range(len(ds)):
        ep = int(ds[i]["_meta"]["episode_index"])
        ep_to_frame_indices.setdefault(ep, []).append(i)

    # Print header
    cols = ("episode", "trial", "type", "bx_bl", "by_bl", "yaw_deg",
            "rail_t_mm", "first_mm", "last_mm", "frames")
    print(f"{cols[0]:>7s} {cols[1]:>10s} {cols[2]:>4s} "
          f"{cols[3]:>8s} {cols[4]:>8s} {cols[5]:>8s} "
          f"{cols[6]:>10s} {cols[7]:>9s} {cols[8]:>9s} {cols[9]:>7s}")
    print("-" * 95)

    max_residual_mm = 0.0
    for ep in sorted(ep_to_frame_indices.keys()):
        frame_indices = ep_to_frame_indices[ep]
        first_idx, last_idx = frame_indices[0], frame_indices[-1]
        first_meta = ds[first_idx]["_meta"]
        label = ds.label_for_episode(ep)
        first_resid_mm = _residual_for_frame(ds, first_idx) * 1000
        last_resid_mm = _residual_for_frame(ds, last_idx) * 1000
        max_residual_mm = max(max_residual_mm, first_resid_mm, last_resid_mm)
        print(
            f"{ep:>7d} {first_meta['trial_key']:>10s} {label.port_type:>4s} "
            f"{label.board_x_baselink:>8.4f} {label.board_y_baselink:>8.4f} "
            f"{math.degrees(label.board_yaw_baselink_rad):>+8.1f} "
            f"{label.target_rail_translation_m * 1000:>+10.1f} "
            f"{first_resid_mm:>9.4f} {last_resid_mm:>9.4f} "
            f"{len(frame_indices):>7d}"
        )

    print()
    print(f"max residual across first/last frames: {max_residual_mm:.4f} mm "
          f"(killer test passes if < 1.0 mm)")
    if max_residual_mm > 1.0:
        print("WARNING: residuals exceed 1 mm — investigate before training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
