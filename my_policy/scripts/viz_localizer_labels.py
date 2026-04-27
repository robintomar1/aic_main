#!/usr/bin/env python3
"""Tier 3 manual sanity check for localizer labels (and optionally model preds).

Two modes:

(default) — host-runnable, validates labels against recorded ground truth.
Walks an existing recorder dataset + batch YAML + summary.json, and prints a
per-episode table:

  episode | trial    | port_type | bx_bl    | by_bl    | yaw_deg | rail_t_mm |
          residual_mm (first..last)

`residual_mm` is `||reconstruct_port_in_baselink(label) - groundtruth.port_pose||`
in the recorded base_link frame — should be sub-millimeter.

(with --checkpoint) — pixi-required (torch). Also runs the trained model on
each frame and reports predicted port pose vs recorded ground truth. This is
the acceptance check for Phase D — answers "did the model learn to localize?"
in physical units (mm and degrees).

Reading hint (label mode):
  - Board pose (x, y) should be in robot reach: typically x ∈ [−0.5, 0],
    y ∈ [−0.5, 0.5] roughly (depends on robot mount).
  - yaw_deg should span the full ±180° across episodes (when the dataset is
    randomized; smoke datasets cover only a subset).
  - rail_t_mm: |t| ≤ ~25 mm for NIC, ~60 mm for SC.
  - residual_mm: should be sub-millimeter; the killer test enforces this but
    the table is the human-readable view.

Usage:
  python3 my_policy/scripts/viz_localizer_labels.py <dataset_root>
  pixi run python3 my_policy/scripts/viz_localizer_labels.py <dataset_root> \\
      --checkpoint /root/aic_data/<batch>_localizer.pt

  dataset_root = e.g. /root/aic_data/<batch>_dataset
  We auto-locate <batch>.yaml (sibling) and <batch>_dataset_logs/summary.json.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from my_policy.localizer.dataset import LocalizerDataset  # noqa: E402
from my_policy.localizer.labels import reconstruct_port_in_baselink  # noqa: E402


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


def _eval_with_checkpoint(
    dataset_root: Path, batch_yaml: Path, summary_json: Path,
    checkpoint_path: Path, max_frames_per_episode: int = 50,
) -> dict[int, dict[str, float]]:
    """Per-episode predicted-vs-recorded port pose stats from a trained model.

    Runs the model on up to `max_frames_per_episode` frames per episode (uniform
    stride) so we don't decode 1000 video frames per episode. Returns a dict
    keyed by episode_index with 'pred_xy_mm', 'pred_yaw_deg', 'pred_rail_mm',
    'pred_port_mm' (end-to-end port translation error in mm).
    """
    import torch
    from my_policy.localizer.model import (
        BoardPoseRegressor,
        BoardPoseRegressorConfig,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    config = BoardPoseRegressorConfig(**ckpt.get("config", {}))
    config.backbone_pretrained = False
    model = BoardPoseRegressor(config).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    ds_with_imgs = LocalizerDataset(
        dataset_root, batch_yaml, summary_json,
        cameras=("center_camera",),
    )

    # Lazy-import the inference helper (does the same image preprocessing
    # train_localizer.py uses).
    from my_policy.localizer.inference import _preprocess_image
    from my_policy.localizer.labels import (
        LocalizerLabel, reconstruct_port_in_baselink, task_one_hot,
    )

    # Group frames by episode.
    ep_to_indices: dict[int, list[int]] = {}
    for i in range(len(ds_with_imgs)):
        ep = int(ds_with_imgs._episode_index[i])
        ep_to_indices.setdefault(ep, []).append(i)

    out: dict[int, dict[str, float]] = {}
    with torch.no_grad():
        for ep, all_idx in ep_to_indices.items():
            stride = max(1, len(all_idx) // max_frames_per_episode)
            sample_idx = all_idx[::stride][:max_frames_per_episode]
            label_gt = ds_with_imgs.label_for_episode(ep)
            target_module, port_name = ds_with_imgs.target_for_episode(ep)
            xy_errs, yaw_errs, rail_errs, port_errs = [], [], [], []
            for i in sample_idx:
                sample = ds_with_imgs[i]
                image = _preprocess_image(sample["images"]["center_camera"]).unsqueeze(0).to(device)
                tcp = torch.from_numpy(sample["tcp_pose"]).float().unsqueeze(0).to(device)
                oh = torch.from_numpy(sample["task_one_hot"]).float().unsqueeze(0).to(device)
                pred = model(image, tcp, oh).squeeze(0).cpu().numpy()
                bx, by, sin_yaw, cos_yaw, rail_t = pred.tolist()
                pred_yaw = float(np.arctan2(sin_yaw, cos_yaw))
                xy_errs.append(np.hypot(bx - label_gt.board_x_baselink,
                                          by - label_gt.board_y_baselink) * 1000)
                yaw_diff = (pred_yaw - label_gt.board_yaw_baselink_rad + np.pi) % (2 * np.pi) - np.pi
                yaw_errs.append(abs(yaw_diff) * 180 / np.pi)
                rail_errs.append(abs(rail_t - label_gt.target_rail_translation_m) * 1000)
                # End-to-end port translation error.
                pred_label = LocalizerLabel(
                    board_x_baselink=bx, board_y_baselink=by,
                    board_yaw_baselink_rad=pred_yaw,
                    target_rail_translation_m=rail_t,
                    port_type=label_gt.port_type,
                )
                pred_port = reconstruct_port_in_baselink(pred_label, target_module, port_name)
                recorded_port = ds_with_imgs.port_pose_baselink(i)
                port_errs.append(float(np.linalg.norm(pred_port - recorded_port)) * 1000)
            out[ep] = {
                "pred_xy_mm": float(np.mean(xy_errs)),
                "pred_yaw_deg": float(np.mean(yaw_errs)),
                "pred_rail_mm": float(np.mean(rail_errs)),
                "pred_port_mm": float(np.mean(port_errs)),
            }
    return out


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
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Optional: trained model checkpoint (.pt). If provided, "
                        "runs the model on a subset of frames per episode and "
                        "reports predicted-vs-recorded port-pose error.")
    p.add_argument("--max-frames-per-episode", type=int, default=50,
                   help="When --checkpoint is set, run the model on this many "
                        "evenly-spaced frames per episode (default: 50).")
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

    # Optional: run trained model and report predicted-vs-recorded port pose.
    pred_stats: dict[int, dict[str, float]] = {}
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            print(f"error: checkpoint not found at {args.checkpoint}", file=sys.stderr)
            return 2
        print(f"running model {args.checkpoint} on up to "
              f"{args.max_frames_per_episode} frames per episode...")
        pred_stats = _eval_with_checkpoint(
            args.dataset_root, batch_yaml, summary_json,
            args.checkpoint, args.max_frames_per_episode,
        )

    # Group frame indices by episode (in their dataset order).
    ep_to_frame_indices: dict[int, list[int]] = {}
    for i in range(len(ds)):
        ep = int(ds[i]["_meta"]["episode_index"])
        ep_to_frame_indices.setdefault(ep, []).append(i)

    # Print header
    if pred_stats:
        cols = ("ep", "trial", "type", "bx_bl", "by_bl", "yaw", "rail_mm",
                "lbl_mm", "p_xy_mm", "p_yaw_deg", "p_rail_mm", "p_port_mm",
                "frames")
        print(
            f"{cols[0]:>3s} {cols[1]:>9s} {cols[2]:>4s} "
            f"{cols[3]:>8s} {cols[4]:>8s} {cols[5]:>8s} {cols[6]:>9s} "
            f"{cols[7]:>8s} {cols[8]:>9s} {cols[9]:>9s} {cols[10]:>9s} {cols[11]:>9s} "
            f"{cols[12]:>7s}"
        )
        print("-" * 130)
    else:
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
        if pred_stats:
            stats = pred_stats.get(ep, {})
            print(
                f"{ep:>3d} {first_meta['trial_key']:>9s} {label.port_type:>4s} "
                f"{label.board_x_baselink:>8.4f} {label.board_y_baselink:>8.4f} "
                f"{math.degrees(label.board_yaw_baselink_rad):>+8.1f} "
                f"{label.target_rail_translation_m * 1000:>+9.1f} "
                f"{first_resid_mm:>8.3f} "
                f"{stats.get('pred_xy_mm', float('nan')):>9.2f} "
                f"{stats.get('pred_yaw_deg', float('nan')):>9.2f} "
                f"{stats.get('pred_rail_mm', float('nan')):>9.2f} "
                f"{stats.get('pred_port_mm', float('nan')):>9.2f} "
                f"{len(frame_indices):>7d}"
            )
        else:
            print(
                f"{ep:>7d} {first_meta['trial_key']:>10s} {label.port_type:>4s} "
                f"{label.board_x_baselink:>8.4f} {label.board_y_baselink:>8.4f} "
                f"{math.degrees(label.board_yaw_baselink_rad):>+8.1f} "
                f"{label.target_rail_translation_m * 1000:>+10.1f} "
                f"{first_resid_mm:>9.4f} {last_resid_mm:>9.4f} "
                f"{len(frame_indices):>7d}"
            )

    print()
    print(f"max label residual across first/last frames: {max_residual_mm:.4f} mm "
          f"(killer test passes if < 1.0 mm)")
    if max_residual_mm > 1.0:
        print("WARNING: label residuals exceed 1 mm — investigate before training.")
        return 1
    if pred_stats:
        # Aggregate over episodes for a one-line model summary.
        all_xy = [s["pred_xy_mm"] for s in pred_stats.values()]
        all_yaw = [s["pred_yaw_deg"] for s in pred_stats.values()]
        all_rail = [s["pred_rail_mm"] for s in pred_stats.values()]
        all_port = [s["pred_port_mm"] for s in pred_stats.values()]
        print(
            f"\nmodel summary across {len(pred_stats)} episodes (per-episode means):\n"
            f"  pred_xy_mm:   mean={np.mean(all_xy):.2f}   max={np.max(all_xy):.2f}\n"
            f"  pred_yaw_deg: mean={np.mean(all_yaw):.2f}  max={np.max(all_yaw):.2f}\n"
            f"  pred_rail_mm: mean={np.mean(all_rail):.2f} max={np.max(all_rail):.2f}\n"
            f"  pred_port_mm: mean={np.mean(all_port):.2f} max={np.max(all_port):.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
