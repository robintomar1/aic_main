"""Port-localizer regression model.

Architecture:
  - ResNet18 backbone (pretrained on ImageNet), final FC stripped.
  - FiLM modulation on the 7-dim task one-hot — applies element-wise
    gamma/beta to the backbone's flat feature vector. Lets the network
    condition its prediction on which target the engine has named, without
    the heavyweight machinery of cross-attention.
  - TCP pose (7 floats) concatenated to the modulated features. Cameras are
    wrist-mounted, so TCP pose is a strong viewpoint hint that resolves the
    "where am I looking from?" ambiguity in the image alone.
  - 2-layer MLP head outputting 5 floats: (board_x_baselink, board_y_baselink,
    sin_yaw_baselink, cos_yaw_baselink, target_rail_translation_m).

Loss: MSE on all 5 outputs. The (sin, cos) pair makes yaw cyclic-friendly
(no discontinuity at ±π) and is automatically self-consistent under MSE
(both components shrink toward the unit circle together).

Submission constraint reminder: weights must be bundled in the eval container
since there's no internet at submission time. ResNet18 ImageNet weights are
~46 MB. We pull them once from torchvision; the training script writes the
final state_dict into the checkpoint, which the inference shim loads cold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


# Approximate per-dim target means and stds for the 5-vector
# (board_x, board_y, sin_yaw, cos_yaw, rail_t). Derived from the
# `gen_trial_config.py` randomization bounds:
#   board_x ∈ [0.13, 0.20], board_y ∈ [-0.10, 0.10], yaw ∈ ±35°,
#   rail_t ∈ [0, 0.05].
# Without normalization, sin_yaw/cos_yaw (magnitude ~1) dominated the MSE
# gradient and the 0.04-magnitude xy targets got ~12× less signal. Scaling to
# unit-std equalizes the per-dim gradient and lets each output learn at the
# same rate.
TARGET_MEAN = (0.165, 0.000, 0.000, 0.940, 0.025)
TARGET_STD = (0.022, 0.058, 0.340, 0.050, 0.014)


def _stats_tensors(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(TARGET_MEAN, dtype=torch.float32)
    std = torch.tensor(TARGET_STD, dtype=torch.float32)
    if device is not None:
        mean = mean.to(device)
        std = std.to(device)
    return mean, std


def normalize_target(t: torch.Tensor) -> torch.Tensor:
    """Convert physical-units target (5,) or (B,5) to z-scored space."""
    mean, std = _stats_tensors(t.device)
    return (t - mean) / std


def denormalize_pred(pred: torch.Tensor) -> torch.Tensor:
    """Convert z-scored model output back to physical units."""
    mean, std = _stats_tensors(pred.device)
    return pred * std + mean


@dataclass
class BoardPoseRegressorConfig:
    task_one_hot_dim: int = 7
    tcp_pose_dim: int = 7
    tcp_embed_dim: int = 64
    head_hidden_dim: int = 256
    output_dim: int = 5
    backbone_pretrained: bool = True
    head_dropout: float = 0.1


class FiLM(nn.Module):
    """FiLM modulation: x ← x * (1 + gamma(c)) + beta(c).

    The +1 in the gamma scaling means an all-zero conditioning vector leaves
    features unchanged, so an untrained FiLM layer doesn't destroy the
    backbone's pretrained features at init time.
    """

    def __init__(self, conditioning_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(conditioning_dim, feature_dim)
        self.beta = nn.Linear(conditioning_dim, feature_dim)
        # Initialize so init output is approximately identity.
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(conditioning)
        beta = self.beta(conditioning)
        return features * (1.0 + gamma) + beta


class BoardPoseRegressor(nn.Module):
    """Image + TCP pose + task one-hot → 5-dim board-pose / rail-translation."""

    def __init__(self, config: BoardPoseRegressorConfig | None = None):
        super().__init__()
        self.config = config or BoardPoseRegressorConfig()

        weights = ResNet18_Weights.IMAGENET1K_V1 if self.config.backbone_pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features  # 512 for resnet18
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = feature_dim

        self.film_task = FiLM(
            conditioning_dim=self.config.task_one_hot_dim,
            feature_dim=feature_dim,
        )

        # Project TCP to a richer embedding before FiLM-modulating the image
        # feature. Concat alone (previous arch) let the 512-dim image feature
        # dominate the head by fan-in, so TCP got near-zero gradient and was
        # ignored — confirmed empirically by ablation. FiLM gives TCP equal
        # multiplicative say in every feature channel.
        self.tcp_proj = nn.Sequential(
            nn.Linear(self.config.tcp_pose_dim, self.config.tcp_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.tcp_embed_dim, self.config.tcp_embed_dim),
        )
        self.film_tcp = FiLM(
            conditioning_dim=self.config.tcp_embed_dim,
            feature_dim=feature_dim,
        )

        p = self.config.head_dropout
        self.head = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(feature_dim, self.config.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(self.config.head_hidden_dim, self.config.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(self.config.head_hidden_dim, self.config.output_dim),
        )

    def forward(
        self,
        image: torch.Tensor,        # (B, 3, H, W) float32 normalized for ImageNet
        tcp_pose: torch.Tensor,     # (B, 7) float32, base_link xyz + xyzw
        task_one_hot: torch.Tensor, # (B, 7) float32
    ) -> torch.Tensor:
        feat = self.backbone(image)                  # (B, 512)
        feat = self.film_task(feat, task_one_hot)    # (B, 512)
        tcp_emb = self.tcp_proj(tcp_pose)            # (B, tcp_embed_dim)
        feat = self.film_tcp(feat, tcp_emb)          # (B, 512)
        return self.head(feat)                        # (B, 5)


def loss_fn(
    pred: torch.Tensor, target: torch.Tensor, *, return_components: bool = False
) -> torch.Tensor | dict[str, torch.Tensor]:
    """MSE between model output (z-scored) and z-scored target.

    `target` is in physical units; we normalize it here so callers don't have
    to remember the convention. Per-dim contributions are equalized because
    each dim has ~unit std after normalization.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    target_n = normalize_target(target)
    diff = pred - target_n
    sq = diff ** 2
    total = sq.mean()
    if not return_components:
        return total
    return {
        "total": total,
        "board_x": sq[:, 0].mean(),
        "board_y": sq[:, 1].mean(),
        "yaw_sincos": sq[:, 2:4].mean(),
        "rail_t": sq[:, 4].mean(),
    }


def predicted_yaw_rad(pred_physical: torch.Tensor) -> torch.Tensor:
    """Recover yaw from (sin, cos) outputs via atan2. Returns radians in (-π, π].
    Input must already be in physical units (denormalized).
    """
    return torch.atan2(pred_physical[..., 2], pred_physical[..., 3])


def reconstruct_metric_errors(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-axis metric errors in physical units.

    `pred` is the raw model output (z-scored); `target` is in physical units.
    Both are converted to physical units for error computation so the metrics
    stay interpretable (mm, degrees) regardless of what the loss space is.
    """
    pred_p = denormalize_pred(pred)
    diff_xy = pred_p[..., :2] - target[..., :2]
    err_xy_mm = diff_xy.norm(dim=-1) * 1000.0
    pred_yaw = predicted_yaw_rad(pred_p)
    target_yaw = predicted_yaw_rad(target)
    yaw_diff = (pred_yaw - target_yaw + torch.pi) % (2.0 * torch.pi) - torch.pi
    err_yaw_deg = yaw_diff.abs() * (180.0 / torch.pi)
    err_rail_mm = (pred_p[..., 4] - target[..., 4]).abs() * 1000.0
    return {
        "board_xy_mm": err_xy_mm,
        "yaw_deg": err_yaw_deg,
        "rail_t_mm": err_rail_mm,
    }
