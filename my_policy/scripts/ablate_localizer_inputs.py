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


def evaluate_with_mask(
    model, loader, device, *,
    zero_image: bool = False,
    zero_tcp: bool = False,
    zero_task: bool = False,
) -> dict[str, float]:
    model.eval()
    buckets = {"board_xy_mm": [], "yaw_deg": [], "rail_t_mm": []}
    with torch.no_grad():
        for image, tcp, oh, target in loader:
            image = image.to(device)
            tcp = tcp.to(device)
            oh = oh.to(device)
            target = target.to(device)
            if zero_image:
                image = torch.zeros_like(image)
            if zero_tcp:
                tcp = torch.zeros_like(tcp)
            if zero_task:
                oh = torch.zeros_like(oh)
            pred = model(image, tcp, oh)
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
    p.add_argument("--camera", type=str, default="center_camera")
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

    base = MultiBatchLocalizerDataset.from_collection_dir(
        args.collection_dir, args.batches, cameras=(args.camera,),
    )
    print(f"  episodes: {base.num_episodes}; frames: {len(base)}")

    _, val_idx = episode_split(
        base, val_fraction=args.val_fraction, seed=args.split_seed
    )
    val_idx = apply_stride(val_idx, args.frame_stride)
    val_ds = TrainSampleWrapper(base, val_idx, camera=args.camera, augment=False)
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"  val frames: {len(val_idx)}")

    config = BoardPoseRegressorConfig(backbone_pretrained=False)
    model = BoardPoseRegressor(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  loaded epoch {ckpt.get('epoch', '?')}, best_val={ckpt.get('best_val', '?')}")

    print("\nrunning ablations...")
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
