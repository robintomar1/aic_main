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


@dataclass
class BoardPoseRegressorConfig:
    task_one_hot_dim: int = 7
    tcp_pose_dim: int = 7
    head_hidden_dim: int = 256
    output_dim: int = 5
    backbone_pretrained: bool = True


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

        self.film = FiLM(
            conditioning_dim=self.config.task_one_hot_dim,
            feature_dim=feature_dim,
        )

        self.head = nn.Sequential(
            nn.Linear(feature_dim + self.config.tcp_pose_dim, self.config.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.head_hidden_dim, self.config.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.head_hidden_dim, self.config.output_dim),
        )

    def forward(
        self,
        image: torch.Tensor,        # (B, 3, H, W) float32 normalized for ImageNet
        tcp_pose: torch.Tensor,     # (B, 7) float32, base_link xyz + xyzw
        task_one_hot: torch.Tensor, # (B, 7) float32
    ) -> torch.Tensor:
        feat = self.backbone(image)             # (B, 512)
        feat = self.film(feat, task_one_hot)    # (B, 512)
        x = torch.cat([feat, tcp_pose], dim=1)  # (B, 519)
        return self.head(x)                      # (B, 5)


def loss_fn(
    pred: torch.Tensor, target: torch.Tensor, *, return_components: bool = False
) -> torch.Tensor | dict[str, torch.Tensor]:
    """MSE on the 5-vector. Components reported separately for diagnostics —
    yaw and rail_t scales differ from board_xy by ~10×, so a single scalar
    loss can hide which axis is failing to converge.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    diff = pred - target
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


def predicted_yaw_rad(pred: torch.Tensor) -> torch.Tensor:
    """Recover yaw from (sin, cos) outputs via atan2. Returns radians in (-π, π].

    pred can be the raw 5-vector output; we slice [2:4]. The (sin, cos) pair
    isn't constrained to the unit circle by training (just toward it via
    MSE), so atan2 is correct without renormalization.
    """
    return torch.atan2(pred[..., 2], pred[..., 3])


def reconstruct_metric_errors(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-axis metric errors for monitoring (mean abs error, in physical units).

    Returns a dict with: board_xy_mm, yaw_deg, rail_t_mm. These are the values
    we actually care about — sub-mm board pose, sub-degree yaw, sub-mm rail
    translation are the targets.
    """
    diff_xy = pred[..., :2] - target[..., :2]
    err_xy_mm = diff_xy.norm(dim=-1) * 1000.0  # per-sample 2D xy error
    pred_yaw = predicted_yaw_rad(pred)
    target_yaw = predicted_yaw_rad(target)
    yaw_diff = (pred_yaw - target_yaw + torch.pi) % (2.0 * torch.pi) - torch.pi
    err_yaw_deg = yaw_diff.abs() * (180.0 / torch.pi)
    err_rail_mm = (pred[..., 4] - target[..., 4]).abs() * 1000.0
    return {
        "board_xy_mm": err_xy_mm,
        "yaw_deg": err_yaw_deg,
        "rail_t_mm": err_rail_mm,
    }
