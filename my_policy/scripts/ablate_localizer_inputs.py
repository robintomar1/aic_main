#!/usr/bin/env python3
"""Diagnostic: zero each input in turn and see which one the model uses.

Loads a checkpoint and runs val inference four ways:
  baseline (real inputs), image=0, tcp=0, task=0.
If a metric doesn't change vs baseline when input X is zeroed, the model
isn't using X. That tells us where the architecture is broken.

Usage:
    pixi run python my_policy/scripts/ablate_localizer_inputs.py \\
        --collection-dir /root/aic_data \\
        --batches batch_100_a batch_100_b batch_100_c batch_100_d batch_100_e batch_500_a \\
        --checkpoint /root/aic_data/localizer_v1_latest.pt \\
        --frame-stride 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.localizer.dataset import MultiBatchLocalizerDataset  # noqa: E402
from my_policy.localizer.model import (  # noqa: E402
    BoardPoseRegressor, BoardPoseRegressorConfig, reconstruct_metric_errors,
)
from train_localizer import (  # noqa: E402
    TrainSampleWrapper, episode_split, apply_stride,
)
from torch.utils.data import DataLoader
from my_policy.localizer.model import denormalize_pred  # noqa: E402


def evaluate_with_mask(
    model, loader, device, *,
    zero_image: bool = False,
    zero_tcp: bool = False,
    zero_task: bool = False,
) -> dict[str, float]:
    model.eval()
    buckets = {"board_xy_mm": [], "yaw_deg": [], "rail_t_mm": []}
    with torch.no_grad():
        for images, tcp, oh, target in loader:
            images = images.to(device)
            tcp = tcp.to(device)
            oh = oh.to(device)
            target = target.to(device)
            if zero_image:
                images = torch.zeros_like(images)
            if zero_tcp:
                tcp = torch.zeros_like(tcp)
            if zero_task:
                oh = torch.zeros_like(oh)
            pred = model(images, tcp, oh)
            errs = reconstruct_metric_errors(pred, target)
            for k, v in errs.items():
                buckets[k].append(v.detach().cpu())
    out = {}
    for k, vs in buckets.items():
        cat = torch.cat(vs).numpy()
        out[f"{k}_mean"] = float(cat.mean())
        out[f"{k}_p95"] = float(np.percentile(cat, 95))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collection-dir", type=Path, required=True)
    p.add_argument("--batches", nargs="+", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cameras", type=str, nargs="+",
                   default=["left_camera", "center_camera", "right_camera"])
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}; checkpoint: {args.checkpoint}")

    cameras = tuple(args.cameras)
    base = MultiBatchLocalizerDataset.from_collection_dir(
        args.collection_dir, args.batches, cameras=cameras,
    )
    print(f"  episodes: {base.num_episodes}; frames: {len(base)}")

    train_idx, val_idx = episode_split(
        base, val_fraction=args.val_fraction, seed=args.split_seed
    )
    train_idx = apply_stride(train_idx, args.frame_stride)
    val_idx = apply_stride(val_idx, args.frame_stride)
    # Subsample train to roughly match val size — we want to know how the
    # model does on TRAINING data with no augmentation, so the diagnostic
    # discriminates "can't fit" from "overfits but doesn't generalize".
    train_eval_n = min(len(val_idx), len(train_idx))
    rng_np = np.random.default_rng(args.split_seed)
    train_eval_idx = list(rng_np.choice(train_idx, size=train_eval_n, replace=False))
    train_eval_ds = TrainSampleWrapper(base, train_eval_idx, cameras=cameras, augment=False)
    val_ds = TrainSampleWrapper(base, val_idx, cameras=cameras, augment=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    loader = val_loader  # alias for the existing ablation block below
    print(f"  train_eval frames: {len(train_eval_idx)}; val frames: {len(val_idx)}")

    config = BoardPoseRegressorConfig(backbone_pretrained=False, num_cameras=len(cameras))
    model = BoardPoseRegressor(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  loaded epoch {ckpt.get('epoch', '?')}, best_val={ckpt.get('best_val', '?')}")

    # --- Train-vs-val diagnostic: distinguishes "can't fit" from "overfit".
    print("\ntrain-vs-val (no input ablation):")
    train_m = evaluate_with_mask(model, train_eval_loader, device)
    val_m = evaluate_with_mask(model, val_loader, device)
    for name, m in (("train_eval", train_m), ("val       ", val_m)):
        print(f"  {name}  xy_mm={m['board_xy_mm_mean']:.2f}/{m['board_xy_mm_p95']:.2f}  "
              f"yaw_deg={m['yaw_deg_mean']:.2f}  rail_mm={m['rail_t_mm_mean']:.2f}")

    # --- Distribution check: does the model emit varied predictions or cluster
    # at the marginal mean? std(pred) close to 0 => mean-predicting.
    model.eval()
    preds_phys = []
    targets_phys = []
    with torch.no_grad():
        for images, tcp, oh, target in val_loader:
            pred = model(images.to(device), tcp.to(device), oh.to(device))
            preds_phys.append(denormalize_pred(pred).cpu())
            targets_phys.append(target.cpu())
    P = torch.cat(preds_phys).numpy()
    T = torch.cat(targets_phys).numpy()
    print("\nval prediction distribution (physical units):")
    print(f"  pred xy mean=({P[:,0].mean():.4f}, {P[:,1].mean():.4f})  "
          f"std=({P[:,0].std():.4f}, {P[:,1].std():.4f})")
    print(f"  tgt  xy mean=({T[:,0].mean():.4f}, {T[:,1].mean():.4f})  "
          f"std=({T[:,0].std():.4f}, {T[:,1].std():.4f})")
    print(f"  pred rail_t mean={P[:,4].mean():.4f}  std={P[:,4].std():.4f}")
    print(f"  tgt  rail_t mean={T[:,4].mean():.4f}  std={T[:,4].std():.4f}")
    pred_std_ratio_xy = (P[:, :2].std(axis=0) / T[:, :2].std(axis=0)).mean()
    print(f"  pred_std / target_std (xy avg) = {pred_std_ratio_xy:.3f}  "
          f"(<0.3 => mean-predicting; ~1.0 => varied predictions)")

    print("\nrunning input ablations on val...")
    cases = [
        ("baseline    ", dict()),
        ("zero_image  ", dict(zero_image=True)),
        ("zero_tcp    ", dict(zero_tcp=True)),
        ("zero_task   ", dict(zero_task=True)),
    ]
    rows = []
    for name, kw in cases:
        m = evaluate_with_mask(model, loader, device, **kw)
        rows.append((name, m))
        print(f"  {name}  xy_mm={m['board_xy_mm_mean']:.2f}/{m['board_xy_mm_p95']:.2f}  "
              f"yaw_deg={m['yaw_deg_mean']:.2f}/{m['yaw_deg_p95']:.2f}  "
              f"rail_mm={m['rail_t_mm_mean']:.2f}/{m['rail_t_mm_p95']:.2f}")

    base_m = rows[0][1]
    print("\ndelta vs baseline (mean only):")
    for name, m in rows[1:]:
        dxy = m["board_xy_mm_mean"] - base_m["board_xy_mm_mean"]
        dyw = m["yaw_deg_mean"] - base_m["yaw_deg_mean"]
        drt = m["rail_t_mm_mean"] - base_m["rail_t_mm_mean"]
        print(f"  {name}  Δxy_mm={dxy:+.2f}  Δyaw_deg={dyw:+.2f}  Δrail_mm={drt:+.2f}")
    print("\ninterpretation: a near-zero Δ means that input is being ignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
