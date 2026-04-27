#!/usr/bin/env python3
"""Train the port-localizer board-pose regressor on a LocalizerDataset.

Usage:
    pixi run python my_policy/scripts/train_localizer.py \\
        --dataset /root/aic_data/<batch>_dataset \\
        --batch-yaml /root/aic_data/<batch>.yaml \\
        --summary /root/aic_data/<batch>_dataset_logs/summary.json \\
        --output /root/aic_data/<batch>_localizer.pt \\
        --epochs 30 --batch-size 64 --frame-stride 10

Required env: pixi (torch, torchvision, lerobot, pyarrow, yaml).

Train/val split is over EPISODES, not frames — frames within the same episode
share a label (the board doesn't move during insertion), so a frame-level split
would leak labels into validation. With episodes split, validation truly tests
generalization to unseen board poses.

Frame stride: subsamples within each episode (every Kth frame). Adjacent frames
at 20 Hz are nearly identical; stride=10 ≈ 2 Hz still captures viewpoint
diversity from wrist motion and cuts training time 10×. Set stride=1 to use
every frame.

Logs per epoch: train/val loss components + per-axis metric errors (board_xy_mm,
yaw_deg, rail_t_mm). Saves the best (lowest val loss) checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from torchvision.transforms import functional as TF

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.localizer.dataset import LocalizerDataset  # noqa: E402
from my_policy.localizer.model import (  # noqa: E402
    BoardPoseRegressor,
    BoardPoseRegressorConfig,
    loss_fn,
    reconstruct_metric_errors,
)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_MODEL_INPUT_SIZE = 224  # ResNet18 default


def _to_chw_float01(image) -> torch.Tensor:
    """Normalize the image format coming out of LeRobotDataset.

    LeRobot may return:
      - torch.Tensor (C, H, W) float in [0, 1]   — pass through.
      - torch.Tensor (C, H, W) uint8 in [0, 255] — divide by 255.
      - torch.Tensor (H, W, C) uint8/float       — transpose, divide if uint8.
      - numpy ndarray (H, W, C) uint8            — convert + transpose + divide.
    Normalizes to (3, H, W) float32 in [0, 1].
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"image must be 3D (got {image.ndim}D shape {image.shape})")
    # Heuristic: channel axis is whichever has 3.
    if image.shape[0] == 3:
        chw = image
    elif image.shape[-1] == 3:
        chw = image.permute(2, 0, 1)
    else:
        raise ValueError(f"image has no length-3 axis: shape {image.shape}")
    if chw.dtype == torch.uint8:
        chw = chw.float() / 255.0
    elif chw.dtype != torch.float32:
        chw = chw.float()
        if chw.max() > 1.5:  # heuristic: still in [0, 255] range
            chw = chw / 255.0
    return chw


def _preprocess_image(image) -> torch.Tensor:
    """Resize to 224×224 and normalize for ImageNet (matches ResNet18 weights)."""
    chw = _to_chw_float01(image)
    chw = TF.resize(chw, [_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE], antialias=True)
    chw = TF.normalize(chw, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))
    return chw


class TrainSampleWrapper(Dataset):
    """Wraps a LocalizerDataset for PyTorch training.

    Returns torch tensors only (DataLoader auto-batches), drops _meta to keep
    the collation simple. Applies image preprocessing here so the underlying
    dataset stays raw and other tools can consume it as-is.
    """

    def __init__(self, base: LocalizerDataset, indices: list[int],
                 camera: str = "center_camera"):
        self.base = base
        self.indices = list(indices)
        self.camera = camera

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        sample = self.base[self.indices[i]]
        if self.camera not in sample["images"]:
            raise KeyError(
                f"camera {self.camera!r} not in images dict; got "
                f"{list(sample['images'].keys())}"
            )
        image = _preprocess_image(sample["images"][self.camera])
        tcp = torch.from_numpy(sample["tcp_pose"]).float()
        oh = torch.from_numpy(sample["task_one_hot"]).float()
        target = torch.from_numpy(sample["target"]).float()
        return image, tcp, oh, target


def episode_split(
    base: LocalizerDataset, *, val_fraction: float = 0.2, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Returns (train_indices, val_indices) where indices are positions into
    `base` and the episode-level split is honored — no episode appears in both
    sets. Held-out episodes test pose generalization.
    """
    rng = np.random.default_rng(seed)
    all_eps = sorted(base._ep_to_label.keys())
    perm = list(rng.permutation(all_eps))
    n_val = max(1, int(round(len(perm) * val_fraction)))
    val_eps = set(perm[:n_val])
    train_eps = set(perm[n_val:])
    train_idx, val_idx = [], []
    for i in range(len(base)):
        ep = int(base._episode_index[i])
        (val_idx if ep in val_eps else train_idx).append(i)
    return train_idx, val_idx


def apply_stride(indices: list[int], stride: int) -> list[int]:
    """Subsample every `stride`-th index. stride=1 returns the input unchanged."""
    if stride <= 1:
        return list(indices)
    return list(indices[::stride])


def _aggregate_metrics(metric_lists: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    out = {}
    for k, v in metric_lists.items():
        cat = torch.cat(v).cpu().numpy()
        out[f"{k}_mean"] = float(cat.mean())
        out[f"{k}_p50"] = float(np.percentile(cat, 50))
        out[f"{k}_p95"] = float(np.percentile(cat, 95))
        out[f"{k}_max"] = float(cat.max())
    return out


def evaluate(
    model: BoardPoseRegressor, loader: DataLoader, device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    metric_buckets: dict[str, list[torch.Tensor]] = {
        "board_xy_mm": [], "yaw_deg": [], "rail_t_mm": [],
    }
    with torch.no_grad():
        for image, tcp, oh, target in loader:
            image = image.to(device, non_blocking=True)
            tcp = tcp.to(device, non_blocking=True)
            oh = oh.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(image, tcp, oh)
            losses.append(loss_fn(pred, target).item())
            errs = reconstruct_metric_errors(pred, target)
            for k, v in errs.items():
                metric_buckets[k].append(v.detach())
    out = _aggregate_metrics(metric_buckets)
    out["loss"] = float(np.mean(losses)) if losses else float("nan")
    return out


def train(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  CUDA name: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Dataset
    print(f"opening dataset at {args.dataset}")
    base = LocalizerDataset(
        args.dataset, args.batch_yaml, args.summary,
        cameras=(args.camera,),
        repo_id=args.repo_id,
    )
    print(f"  episodes: {base.num_episodes}; frames: {len(base)}")
    train_idx, val_idx = episode_split(
        base, val_fraction=args.val_fraction, seed=args.split_seed
    )
    print(f"  episode split: {len(set(int(base._episode_index[i]) for i in train_idx))} train, "
          f"{len(set(int(base._episode_index[i]) for i in val_idx))} val")
    train_idx = apply_stride(train_idx, args.frame_stride)
    val_idx = apply_stride(val_idx, args.frame_stride)
    print(f"  after stride={args.frame_stride}: {len(train_idx)} train frames, "
          f"{len(val_idx)} val frames")
    if not train_idx or not val_idx:
        print("error: empty train or val set after split+stride", file=sys.stderr)
        return 2

    train_ds = TrainSampleWrapper(base, train_idx, camera=args.camera)
    val_ds = TrainSampleWrapper(base, val_idx, camera=args.camera)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False, persistent_workers=(args.num_workers > 0),
    )

    # --- Model
    config = BoardPoseRegressorConfig(backbone_pretrained=args.pretrained)
    model = BoardPoseRegressor(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params / 1e6:.2f} M")

    # --- Optim
    # Conservative LR for the pretrained backbone, higher LR for head/film.
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs
    )

    # --- Train
    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for step, (image, tcp, oh, target) in enumerate(train_loader):
            image = image.to(device, non_blocking=True)
            tcp = tcp.to(device, non_blocking=True)
            oh = oh.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(image, tcp, oh)
            loss = loss_fn(pred, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(train_losses))
        val = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        msg = (
            f"epoch {epoch + 1:>3d}/{args.epochs}  train_loss={train_loss:.5f}  "
            f"val_loss={val['loss']:.5f}  "
            f"xy_mm={val['board_xy_mm_mean']:.2f}/{val['board_xy_mm_p95']:.2f}/{val['board_xy_mm_max']:.2f}  "
            f"yaw_deg={val['yaw_deg_mean']:.2f}/{val['yaw_deg_p95']:.2f}/{val['yaw_deg_max']:.2f}  "
            f"rail_mm={val['rail_t_mm_mean']:.2f}/{val['rail_t_mm_p95']:.2f}/{val['rail_t_mm_max']:.2f}  "
            f"({elapsed:.0f}s)"
        )
        print(msg)
        log.append({"epoch": epoch + 1, "train_loss": train_loss, **val,
                    "elapsed_s": elapsed})

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(config),
                "args": vars(args),
                "epoch": epoch + 1,
                "val": val,
            }, args.output)
            print(f"  saved best checkpoint to {args.output} (val_loss={best_val:.5f})")

    log_path = args.output.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\ntraining done. best_val_loss={best_val:.5f}; log at {log_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, required=True,
                   help="LeRobot dataset directory (e.g. /root/aic_data/batch_500_dataset)")
    p.add_argument("--batch-yaml", type=Path, required=True,
                   help="The batch config YAML used by the eval container.")
    p.add_argument("--summary", type=Path, required=True,
                   help="The recorder's summary.json (in <dataset>_logs/).")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to save the best checkpoint (.pt).")
    p.add_argument("--repo-id", type=str, default="local/localizer")
    p.add_argument("--camera", type=str, default="center_camera",
                   help="Which camera to use (center_camera | left_camera | right_camera).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--frame-stride", type=int, default=10,
                   help="Sample every Nth frame within each episode (1 = all frames).")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Fraction of EPISODES (not frames) held out for validation.")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="Use ImageNet-pretrained ResNet18 backbone (default: on).")
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    args = p.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
