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

# Avoid /dev/shm exhaustion when running in the dev container (default shm=64M
# is far too small for multi-worker DataLoaders carrying batched images). The
# 'file_system' strategy uses tmpfile fds instead of POSIX shared memory.
torch.multiprocessing.set_sharing_strategy("file_system")
from torchvision.models import ResNet18_Weights
from torchvision.transforms import functional as TF

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.localizer.dataset import (  # noqa: E402
    LocalizerDataset,
    MultiBatchLocalizerDataset,
)
from my_policy.localizer.labels import TASK_ONE_HOT_ORDER  # noqa: E402
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


def _augment_image(chw_float01: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Photometric + geometric + JPEG augmentation. Input/output: (3, H, W).

    Geometric (random resized crop + translation): the dominant failure mode
    seen in v1/v2 was the model memorizing per-episode visual signatures
    rather than learning pose-from-pixels. Photometric aug alone left the
    geometric layout of each episode pixel-stable, so memorization still
    worked. A random crop per frame breaks that — every frame from the same
    episode now lands at a different pixel offset, forcing the network to
    learn pose-invariant features.
    No rotation (would confuse perceived board yaw with image rotation).
    No horizontal flip (board layout has chirality — sc_port_0 vs _1 etc.).
    """
    img = chw_float01

    # --- Geometric: aggressive random resized crop. v3 used scale 0.80-1.0
    # which left enough per-episode pixels intact for the model to memorize
    # via episode-specific features. v4 widens to 0.5-1.0 so episode mates
    # rarely share crop windows; the model has to learn pose from generic
    # board features.
    if rng.random() < 0.95:
        _, h, w = img.shape
        scale = float(rng.uniform(0.5, 1.0))
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        top = int(rng.integers(0, h - new_h + 1))
        left = int(rng.integers(0, w - new_w + 1))
        img = img[:, top:top + new_h, left:left + new_w]

    # Brightness ±0.4 (was ±0.2 — episode-stable lighting was a memorization
    # foothold; widen to break it).
    if rng.random() < 0.9:
        img = TF.adjust_brightness(img, float(1.0 + rng.uniform(-0.4, 0.4)))
    if rng.random() < 0.9:
        img = TF.adjust_contrast(img, float(1.0 + rng.uniform(-0.4, 0.4)))
    if rng.random() < 0.7:
        img = TF.adjust_saturation(img, float(1.0 + rng.uniform(-0.4, 0.4)))
    # Hue still kept narrow — cable color is informative for some tasks.
    if rng.random() < 0.4:
        img = TF.adjust_hue(img, float(rng.uniform(-0.05, 0.05)))
    img = img.clamp(0.0, 1.0)

    # Gaussian noise σ=0.02 (was 0.01).
    if rng.random() < 0.7:
        img = (img + torch.randn_like(img) * 0.02).clamp(0.0, 1.0)

    # Random erasing (cutout): zeroes a small rectangle. Forces the model not
    # to rely on a single visual landmark. Done AFTER photometric so the
    # erased region is unambiguous black.
    if rng.random() < 0.5:
        _, h, w = img.shape
        eh = int(rng.integers(int(h * 0.05), int(h * 0.20) + 1))
        ew = int(rng.integers(int(w * 0.05), int(w * 0.20) + 1))
        et = int(rng.integers(0, h - eh + 1))
        el = int(rng.integers(0, w - ew + 1))
        img[:, et:et + eh, el:el + ew] = 0.0

    return img


def _augment_tcp(tcp: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Add ±2 mm position noise, ±0.5° orientation noise (small-angle approx).

    Input: (7,) float — xyz + xyzw quat. Output: same shape, normalized quat.
    """
    out = tcp.clone()
    out[0:3] += torch.from_numpy(rng.normal(0.0, 0.002, size=3).astype(np.float32))
    # Small-angle quaternion perturbation: tangent vector with σ ≈ 0.5° (radians).
    sigma = math.radians(0.5)
    omega = rng.normal(0.0, sigma, size=3)
    half = 0.5 * omega
    dq = np.array([half[0], half[1], half[2], 1.0], dtype=np.float32)  # xyzw
    dq /= np.linalg.norm(dq)
    # Quat multiply (xyzw): q_new = q ⊗ dq.
    qx, qy, qz, qw = out[3].item(), out[4].item(), out[5].item(), out[6].item()
    dx, dy, dz, dw = dq
    nx = qw*dx + qx*dw + qy*dz - qz*dy
    ny = qw*dy - qx*dz + qy*dw + qz*dx
    nz = qw*dz + qx*dy - qy*dx + qz*dw
    nw = qw*dw - qx*dx - qy*dy - qz*dz
    n = math.sqrt(nx*nx + ny*ny + nz*nz + nw*nw) or 1.0
    out[3:7] = torch.tensor([nx/n, ny/n, nz/n, nw/n], dtype=torch.float32)
    return out


class TrainSampleWrapper(Dataset):
    """Wraps a LocalizerDataset for PyTorch training.

    Returns torch tensors only (DataLoader auto-batches), drops _meta to keep
    the collation simple. Applies image preprocessing here so the underlying
    dataset stays raw and other tools can consume it as-is.

    `augment=True` enables D3 photometric + JPEG augmentations on the image
    and (if `tcp_noise=True`) small TCP pose perturbations. Should be off for
    val so the metrics measure real generalization, not noise robustness.
    """

    def __init__(self, base, indices: list[int],
                 camera: str = "center_camera",
                 *, augment: bool = False, tcp_noise: bool = False,
                 augment_seed: int = 0):
        self.base = base
        self.indices = list(indices)
        self.camera = camera
        self.augment = augment
        self.tcp_noise = tcp_noise
        # Per-worker RNG seeded by (augment_seed, index) inside __getitem__ so
        # that DataLoader workers don't share state and reseeding across
        # epochs keeps the augmentation distribution non-degenerate.
        self._augment_seed = augment_seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        sample = self.base[self.indices[i]]
        if self.camera not in sample["images"]:
            raise KeyError(
                f"camera {self.camera!r} not in images dict; got "
                f"{list(sample['images'].keys())}"
            )
        image = _to_chw_float01(sample["images"][self.camera])
        tcp = torch.from_numpy(sample["tcp_pose"]).float()
        if self.augment:
            rng = np.random.default_rng(
                (self._augment_seed, self.indices[i], int(torch.empty(()).uniform_(0, 1e9).item()))
            )
            image = _augment_image(image, rng)
            if self.tcp_noise:
                tcp = _augment_tcp(tcp, rng)
        # Resize + ImageNet normalize after augmentation.
        image = TF.resize(image, [_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE], antialias=True)
        image = TF.normalize(image, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))
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

    # --- Dataset (single-batch or multi-batch)
    if args.collection_dir is not None:
        if not args.batches:
            print("error: --collection-dir requires --batches", file=sys.stderr)
            return 2
        print(f"opening multi-batch dataset at {args.collection_dir}; "
              f"batches: {args.batches}")
        base = MultiBatchLocalizerDataset.from_collection_dir(
            args.collection_dir, args.batches,
            cameras=(args.camera,),
        )
    else:
        if args.dataset is None or args.batch_yaml is None or args.summary is None:
            print("error: need either --collection-dir + --batches OR "
                  "--dataset + --batch-yaml + --summary", file=sys.stderr)
            return 2
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

    train_ds = TrainSampleWrapper(
        base, train_idx, camera=args.camera,
        augment=args.augment, tcp_noise=args.tcp_noise,
        augment_seed=args.split_seed,
    )
    val_ds = TrainSampleWrapper(base, val_idx, camera=args.camera, augment=False)
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

    # --- Resume support
    best_val = float("inf")
    start_epoch = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latest_path = args.output.with_name(args.output.stem + "_latest.pt")
    log: list = []
    log_path = args.output.with_suffix(".log.json")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optim_state_dict" in ckpt:
            optim.load_state_dict(ckpt["optim_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        if log_path.exists():
            log = json.loads(log_path.read_text())
        print(f"resumed from {args.resume} at epoch {start_epoch}, "
              f"best_val={best_val:.5f}")

    def _ckpt_payload(epoch: int, val: dict | None) -> dict:
        return {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": vars(config),
            "args": vars(args),
            "epoch": epoch,
            "best_val": best_val,
            "val": val,
            # Inference-side metadata: pin invariants so a future code change
            # can't silently break the saved checkpoint.
            "task_one_hot_order": list(TASK_ONE_HOT_ORDER),
            "image_size": _MODEL_INPUT_SIZE,
            "imagenet_mean": list(_IMAGENET_MEAN),
            "imagenet_std": list(_IMAGENET_STD),
            "cameras": [args.camera],
        }

    # --- Train
    for epoch in range(start_epoch, args.epochs):
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

        # Always save latest (for resume); save best (for inference) when val improves.
        is_best = val["loss"] < best_val
        if is_best:
            best_val = val["loss"]
        payload = _ckpt_payload(epoch + 1, val)
        torch.save(payload, latest_path)
        if is_best:
            torch.save(payload, args.output)
            print(f"  saved best checkpoint to {args.output} (val_loss={best_val:.5f})")
        # Flush log every epoch so progress survives a crash.
        log_path.write_text(json.dumps(log, indent=2))
    print(f"\ntraining done. best_val_loss={best_val:.5f}; log at {log_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=None,
                   help="LeRobot dataset directory (single-batch mode).")
    p.add_argument("--batch-yaml", type=Path, default=None,
                   help="Batch config YAML (single-batch mode).")
    p.add_argument("--summary", type=Path, default=None,
                   help="Recorder summary.json (single-batch mode).")
    p.add_argument("--collection-dir", type=Path, default=None,
                   help="Multi-batch mode: parent dir holding <name>/, "
                        "<name>.yaml, <name>_logs/summary.json triplets.")
    p.add_argument("--batches", type=str, nargs="+", default=None,
                   help="Multi-batch mode: list of batch names under "
                        "--collection-dir (e.g. batch_100_a batch_100_b).")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to save the best checkpoint (.pt).")
    p.add_argument("--repo-id", type=str, default="local/localizer")
    p.add_argument("--camera", type=str, default="center_camera",
                   help="Which camera to use (center_camera | left_camera | right_camera).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
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
    p.add_argument("--augment", action="store_true", default=True,
                   help="Enable photometric + JPEG image augmentation on train split (default: on).")
    p.add_argument("--no-augment", action="store_false", dest="augment")
    p.add_argument("--tcp-noise", action="store_true", default=False,
                   help="Add ±2mm / ±0.5° noise to the TCP pose input (default: off).")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to <output>_latest.pt to resume training from.")
    args = p.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
