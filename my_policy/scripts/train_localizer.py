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
    AUX_PIXEL_WEIGHT,
    BoardPoseRegressor,
    BoardPoseRegressorConfig,
    aux_pixel_loss,
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


def _augment_image(
    chw_float01: torch.Tensor, rng: np.random.Generator,
) -> tuple[torch.Tensor, tuple | None]:
    """Photometric + geometric + JPEG augmentation. Input: (3, H, W).
    Returns (augmented_image, crop_info) where crop_info is None or
    (top_norm, left_norm, h_norm, w_norm) in [0,1] of the input image.

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

    # --- Geometric: random resized crop. Mild crops (0.65-1.0) — with
    # corrected labels (the v1-v5 dataset bug fixed in 2026-04-29), heavy
    # aug isn't needed and makes fitting slower without much generalization
    # gain. Keep enough geometric variation to break per-episode pixel
    # memorization without strangling the fit.
    crop = None  # (top_norm, left_norm, h_norm, w_norm) in [0,1] of input img
    if rng.random() < 0.9:
        _, h, w = img.shape
        scale = float(rng.uniform(0.65, 1.0))
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        top = int(rng.integers(0, h - new_h + 1))
        left = int(rng.integers(0, w - new_w + 1))
        img = img[:, top:top + new_h, left:left + new_w]
        crop = (top / h, left / w, new_h / h, new_w / w)

    # Photometric: ±0.25 (between v3's 0.2 and v4's 0.4).
    if rng.random() < 0.85:
        img = TF.adjust_brightness(img, float(1.0 + rng.uniform(-0.25, 0.25)))
    if rng.random() < 0.85:
        img = TF.adjust_contrast(img, float(1.0 + rng.uniform(-0.25, 0.25)))
    if rng.random() < 0.6:
        img = TF.adjust_saturation(img, float(1.0 + rng.uniform(-0.25, 0.25)))
    if rng.random() < 0.3:
        img = TF.adjust_hue(img, float(rng.uniform(-0.04, 0.04)))
    img = img.clamp(0.0, 1.0)

    # Gaussian noise σ=0.015.
    if rng.random() < 0.6:
        img = (img + torch.randn_like(img) * 0.015).clamp(0.0, 1.0)

    # Random erasing: zeroes a small rectangle. Cheap regularizer.
    if rng.random() < 0.4:
        _, h, w = img.shape
        eh = int(rng.integers(int(h * 0.05), int(h * 0.18) + 1))
        ew = int(rng.integers(int(w * 0.05), int(w * 0.18) + 1))
        et = int(rng.integers(0, h - eh + 1))
        el = int(rng.integers(0, w - ew + 1))
        img[:, et:et + eh, el:el + ew] = 0.0

    return img, crop


def _adjust_pixel_target_for_crop(
    uv_norm_with_valid: np.ndarray,  # shape (3,): [u_norm, v_norm, valid]
    crop: tuple | None,              # (top_norm, left_norm, h_norm, w_norm)
) -> np.ndarray:
    """Adjust a port-pixel target after a random crop. The pixel positions
    in the crop are remapped to [0, 1] of the cropped region; samples that
    fall outside the crop are marked invalid (valid=0)."""
    out = uv_norm_with_valid.copy()
    if crop is None:
        return out
    top, left, h_n, w_n = crop
    u, v, valid = float(out[0]), float(out[1]), float(out[2])
    if valid <= 0:
        return out
    if u < left or u >= left + w_n or v < top or v >= top + h_n:
        out[2] = 0.0
        return out
    out[0] = (u - left) / w_n
    out[1] = (v - top) / h_n
    return out


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
    """Wraps a LocalizerDataset for PyTorch training. v7: multi-cam.

    Returns `(images, tcp, task_one_hot, target)` where `images` is a
    stacked tensor of shape `(num_cameras, 3, H, W)` ordered by `cameras`.
    Per-camera augmentation is applied independently — different crops/
    photometrics per cam — which gives more diverse training samples than
    sharing transforms across views, and at test time all 3 views are
    seen unaugmented anyway.
    """

    def __init__(self, base, indices: list[int],
                 cameras: tuple[str, ...] = ("left_camera", "center_camera", "right_camera"),
                 *, augment: bool = False, tcp_noise: bool = False,
                 augment_seed: int = 0):
        self.base = base
        self.indices = list(indices)
        self.cameras = tuple(cameras)
        self.augment = augment
        self.tcp_noise = tcp_noise
        self._augment_seed = augment_seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        sample = self.base[self.indices[i]]
        for cam in self.cameras:
            if cam not in sample["images"]:
                raise KeyError(
                    f"camera {cam!r} not in images dict; got "
                    f"{list(sample['images'].keys())}"
                )

        rng_for_tcp = None
        if self.augment:
            rng_for_tcp = np.random.default_rng(
                (self._augment_seed, self.indices[i],
                 int(torch.empty(()).uniform_(0, 1e9).item()))
            )

        # Process each camera independently. Different crops per cam = more
        # aug diversity. Adjust the v8 pixel target by the crop applied to
        # that camera so the aux supervision lines up with what the model
        # actually sees.
        port_pixels_in = sample["port_pixels"]   # (num_cams, 3) [u_norm, v_norm, valid]
        per_cam = []
        adjusted_pixels = np.zeros_like(port_pixels_in)
        for j, cam in enumerate(self.cameras):
            img = _to_chw_float01(sample["images"][cam])
            crop = None
            if self.augment:
                rng_cam = np.random.default_rng(
                    (self._augment_seed, self.indices[i], j,
                     int(torch.empty(()).uniform_(0, 1e9).item()))
                )
                img, crop = _augment_image(img, rng_cam)
            adjusted_pixels[j] = _adjust_pixel_target_for_crop(port_pixels_in[j], crop)
            img = TF.resize(img, [_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE], antialias=True)
            img = TF.normalize(img, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))
            per_cam.append(img)
        images = torch.stack(per_cam, dim=0)  # (num_cameras, 3, H, W)

        tcp = torch.from_numpy(sample["tcp_pose"]).float()
        if self.augment and self.tcp_noise:
            tcp = _augment_tcp(tcp, rng_for_tcp)

        oh = torch.from_numpy(sample["task_one_hot"]).float()
        target = torch.from_numpy(sample["target"]).float()
        port_pixels = torch.from_numpy(adjusted_pixels).float()
        return images, tcp, oh, target, port_pixels


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
    relative = model.config.tcp_relative_target
    losses = []
    metric_buckets: dict[str, list[torch.Tensor]] = {
        "board_xy_mm": [], "yaw_deg": [], "rail_t_mm": [],
    }
    with torch.no_grad():
        for images, tcp, oh, target, _port_pix in loader:
            images = images.to(device, non_blocking=True)
            tcp = tcp.to(device, non_blocking=True)
            oh = oh.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if relative:
                # In-place subtract is safe because target is fresh per-batch.
                target = target.clone()
                target[:, :2] -= tcp[:, :2]
            pred = model(images, tcp, oh)
            losses.append(loss_fn(pred, target, relative=relative).item())
            errs = reconstruct_metric_errors(pred, target, relative=relative)
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

    cameras = tuple(args.cameras)
    if not cameras:
        print("error: --cameras must list at least one camera", file=sys.stderr)
        return 2

    # --- Dataset (single-batch or multi-batch)
    if args.collection_dir is not None:
        if not args.batches:
            print("error: --collection-dir requires --batches", file=sys.stderr)
            return 2
        print(f"opening multi-batch dataset at {args.collection_dir}; "
              f"batches: {args.batches}; cameras: {cameras}")
        base = MultiBatchLocalizerDataset.from_collection_dir(
            args.collection_dir, args.batches,
            cameras=cameras,
        )
    else:
        if args.dataset is None or args.batch_yaml is None or args.summary is None:
            print("error: need either --collection-dir + --batches OR "
                  "--dataset + --batch-yaml + --summary", file=sys.stderr)
            return 2
        print(f"opening dataset at {args.dataset}; cameras: {cameras}")
        base = LocalizerDataset(
            args.dataset, args.batch_yaml, args.summary,
            cameras=cameras,
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
        base, train_idx, cameras=cameras,
        augment=args.augment, tcp_noise=args.tcp_noise,
        augment_seed=args.split_seed,
    )
    val_ds = TrainSampleWrapper(base, val_idx, cameras=cameras, augment=False)
    _loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=False, **_loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, **_loader_kw,
    )

    # --- Empirical relative-target stats. Printed once so we can sanity-check
    # the hardcoded TARGET_MEAN_RELATIVE / TARGET_STD_RELATIVE in model.py
    # before training. If the printed values differ from the hardcoded ones
    # by more than ~2×, paste them in and re-run; normalization fighting the
    # data is a slow-convergence trap that's hard to spot mid-training.
    if args.tcp_relative_target:
        sample_n = min(2000, len(train_ds))
        rel_xy = []
        for i in range(sample_n):
            s = train_ds[i]
            target_i = s[3].numpy() if hasattr(s[3], "numpy") else np.asarray(s[3])
            tcp_i = s[1].numpy() if hasattr(s[1], "numpy") else np.asarray(s[1])
            rel_xy.append([target_i[0] - tcp_i[0], target_i[1] - tcp_i[1]])
        rel_xy = np.asarray(rel_xy)
        from my_policy.localizer.model import (  # noqa: WPS433
            TARGET_MEAN_RELATIVE, TARGET_STD_RELATIVE,
        )
        print(
            "  empirical (board − tcp) xy over "
            f"{sample_n} train samples:\n"
            f"    mean=({rel_xy[:, 0].mean():+.4f}, {rel_xy[:, 1].mean():+.4f})  "
            f"std =({rel_xy[:, 0].std():.4f}, {rel_xy[:, 1].std():.4f})\n"
            "  hardcoded (model.py):\n"
            f"    mean=({TARGET_MEAN_RELATIVE[0]:+.4f}, {TARGET_MEAN_RELATIVE[1]:+.4f})  "
            f"std =({TARGET_STD_RELATIVE[0]:.4f}, {TARGET_STD_RELATIVE[1]:.4f})\n"
            "  → if these differ by more than ~2×, paste empirical values "
            "into TARGET_MEAN_RELATIVE / TARGET_STD_RELATIVE and re-run."
        )

    # --- Model
    # Resume-path safety: when continuing from a checkpoint trained with a
    # different aux mode or backbone, the saved state_dict won't match a
    # freshly-built model. Peek at the state_dict + saved config and override
    # the CLI flags (with warnings) so resume "just works".
    if args.resume is not None and args.resume.exists():
        peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = peek.get("model_state_dict", {})
        peek_cfg = peek.get("config", {})

        has_pooled = any(k.startswith("aux_head.") for k in sd)
        has_spatial = any(
            k.startswith("aux_conv.") or k.startswith("aux_pathway_head.")
            for k in sd
        )
        resume_aux_mode = (
            "pooled" if has_pooled else "spatial" if has_spatial else "none"
        )
        if resume_aux_mode != args.aux_mode:
            print(
                f"  --resume override: ckpt has aux_mode={resume_aux_mode!r} "
                f"but --aux-mode={args.aux_mode!r}; using ckpt's mode."
            )
            args.aux_mode = resume_aux_mode

        has_dinov2 = any(k.startswith("backbone_dinov2.") for k in sd)
        resume_backbone = (
            "dinov2_vits14" if has_dinov2
            else peek_cfg.get("backbone", "resnet18")
        )
        if resume_backbone != args.backbone:
            print(
                f"  --resume override: ckpt has backbone={resume_backbone!r} "
                f"but --backbone={args.backbone!r}; using ckpt's backbone."
            )
            args.backbone = resume_backbone

        # tcp_relative_target has no state-dict signature (it's a target/loss
        # change, not architectural), so we have to read it from saved config
        # if present. Pre-tcp-relative checkpoints (v7/v8/v9-pathway/early
        # v9-dino) lack the field — those were trained with absolute targets
        # so the right default is False.
        resume_relative = peek_cfg.get("tcp_relative_target", False)
        if resume_relative != args.tcp_relative_target:
            print(
                f"  --resume override: ckpt has tcp_relative_target="
                f"{resume_relative!r} but CLI is {args.tcp_relative_target!r}; "
                f"using ckpt's value."
            )
            args.tcp_relative_target = resume_relative
        del peek  # release the peek copy before the actual load below

    aux_pixel_head = (args.aux_mode == "pooled")
    aux_pathway = (args.aux_mode == "spatial")
    config = BoardPoseRegressorConfig(
        backbone=args.backbone,
        backbone_pretrained=args.pretrained,
        backbone_freeze=args.freeze_backbone,
        num_cameras=len(cameras),
        aux_pixel_head=aux_pixel_head,
        aux_pathway=aux_pathway,
        tcp_relative_target=args.tcp_relative_target,
    )
    model = BoardPoseRegressor(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  model params: {n_params / 1e6:.2f} M total, "
        f"{n_trainable / 1e6:.2f} M trainable  "
        f"backbone={args.backbone} freeze={args.freeze_backbone} "
        f"aux_mode={args.aux_mode} "
        f"tcp_relative={args.tcp_relative_target}"
    )

    # --- Optim
    # Conservative LR for the pretrained backbone, higher LR for head/film.
    # NOTE on prefix: ResNet18 path uses `backbone_trunk` / `backbone_avgpool`
    # which do NOT match `backbone.` (with the dot) — historical behavior is
    # therefore that ALL ResNet18 params got lr_head; preserve that here so
    # we don't silently change v7-era training dynamics. DINOv2 params live
    # under `backbone_dinov2.` and DO get lr_backbone, which matters when
    # --no-freeze-backbone is set (DINOv2 at lr_head=1e-3 would explode).
    # Frozen params (requires_grad=False) are excluded from the optimizer
    # entirely.
    backbone_params, head_params = [], []
    if args.backbone == "dinov2_vits14":
        backbone_prefix = "backbone_dinov2."
    else:
        backbone_prefix = "backbone."  # legacy non-match for resnet18
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(backbone_prefix):
            backbone_params.append(p)
        else:
            head_params.append(p)
    # Build only non-empty param groups — AdamW tolerates empty groups but the
    # state dict gets confusing, and frozen-DINO + ResNet18-with-legacy-prefix
    # both produce empty backbone groups.
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})
    if not param_groups:
        raise RuntimeError("no trainable parameters — did everything get frozen?")
    optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
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
            "cameras": list(cameras),
        }

    # --- Train
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        train_aux_losses = []
        relative = model.config.tcp_relative_target
        for step, (images, tcp, oh, target, port_pix) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            tcp = tcp.to(device, non_blocking=True)
            oh = oh.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            port_pix = port_pix.to(device, non_blocking=True)
            if relative:
                target = target.clone()
                target[:, :2] -= tcp[:, :2]
            pred, aux_pred = model(images, tcp, oh, return_aux=True)
            pose_loss = loss_fn(pred, target, relative=relative)
            if aux_pred is not None:
                aux_loss = aux_pixel_loss(aux_pred, port_pix)
                loss = pose_loss + AUX_PIXEL_WEIGHT * aux_loss
                train_aux_losses.append(aux_loss.item())
            else:
                loss = pose_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(train_losses))
        train_aux = float(np.mean(train_aux_losses)) if train_aux_losses else 0.0
        val = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        aux_str = f"  aux_pix={train_aux:.5f}" if train_aux_losses else ""
        msg = (
            f"epoch {epoch + 1:>3d}/{args.epochs}  train_loss={train_loss:.5f}{aux_str}  "
            f"val_loss={val['loss']:.5f}  "
            f"xy_mm={val['board_xy_mm_mean']:.2f}/{val['board_xy_mm_p95']:.2f}/{val['board_xy_mm_max']:.2f}  "
            f"yaw_deg={val['yaw_deg_mean']:.2f}/{val['yaw_deg_p95']:.2f}/{val['yaw_deg_max']:.2f}  "
            f"rail_mm={val['rail_t_mm_mean']:.2f}/{val['rail_t_mm_p95']:.2f}/{val['rail_t_mm_max']:.2f}  "
            f"({elapsed:.0f}s)"
        )
        print(msg)
        log.append({"epoch": epoch + 1, "train_loss": train_loss,
                    "train_aux_pixel_loss": train_aux, **val,
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
    p.add_argument("--cameras", type=str, nargs="+",
                   default=["left_camera", "center_camera", "right_camera"],
                   help="Cameras to feed the model (multi-cam concat). Default: 3 wrist cams.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=3e-4)
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
    p.add_argument(
        "--aux-mode", type=str, default="none",
        choices=["none", "pooled", "spatial"],
        help=(
            "Auxiliary per-camera port-pixel supervision integration point.\n"
            "  none    - no aux loss; pose-only training (this branch's\n"
            "            default — matches v7's recipe).\n"
            "  pooled  - aux head reads the shared pooled feature (v8).\n"
            "  spatial - aux head reads spatial features via its own conv\n"
            "            pathway (v9-pathway). Code path stays available for\n"
            "            cross-run comparison."
        ),
    )
    p.add_argument(
        "--backbone", type=str, default="dinov2_vits14",
        choices=["resnet18", "dinov2_vits14"],
        help=(
            "Visual backbone. resnet18 matches v6/v7/v8/v9-pathway training.\n"
            "dinov2_vits14 (this branch's default) swaps in DINOv2 ViT-S/14:\n"
            "stronger self-supervised priors, 16x16 patch grid (vs 7x7),\n"
            "feature dim 384 (vs 512). Fine-tuned by default on this branch\n"
            "to match v7's training recipe — see --freeze-backbone."
        ),
    )
    p.add_argument(
        "--freeze-backbone", action="store_true", default=False,
        help="Freeze the backbone (default off on this branch — v7 fine-tuned). "
             "Pass to switch DINOv2 into linear-probe mode instead.",
    )
    p.add_argument(
        "--no-freeze-backbone", action="store_false", dest="freeze_backbone",
        help="(Default — kept as an explicit alias for clarity.)",
    )
    p.add_argument(
        "--tcp-relative-target", action="store_true", default=False,
        help=(
            "Predict (board_xy − tcp_xy) instead of absolute board_xy in "
            "base_link. Default off on this branch — v7 used absolute targets, "
            "and this branch tests the v7 recipe with DINOv2 swapped in."
        ),
    )
    p.add_argument(
        "--no-tcp-relative-target", action="store_false",
        dest="tcp_relative_target",
        help="Predict absolute board_xy in base_link (legacy v6/v7/v8 form).",
    )
    args = p.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
