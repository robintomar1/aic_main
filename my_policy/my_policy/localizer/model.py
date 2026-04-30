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
    num_cameras: int = 3  # v7: 3-cam concat for absolute-position parallax
    # Visual backbone choice. "resnet18" matches v6/v7/v8/v9-pathway training.
    # "dinov2_vits14" (v9-dino) swaps in DINOv2's small ViT, which has stronger
    # self-supervised priors and preserves spatial info via patch tokens
    # (16x16 grid for 224 input, vs ResNet18's 7x7). Different feature dim
    # (384 vs 512) so cam_fuse / FiLM / head dimensions all key off
    # `self.feature_dim` set in __init__ rather than a hardcoded 512.
    backbone: str = "resnet18"
    # When using dinov2, freeze its parameters by default. DINOv2 features are
    # designed to be useful out of the box (the canonical recipe is "frozen
    # features + linear probe"), and 21M params unfrozen on 354 episodes is a
    # real overfit risk. Set to False (--no-freeze-backbone) for full fine-
    # tuning if frozen features stagnate. Ignored when backbone="resnet18".
    backbone_freeze: bool = True
    # Auxiliary supervision modes — at most one may be True. Both False = pose-only
    # (v7). Defaults below ship the v9-pathway architecture for new training while
    # preserving the ability to load older v7/v8 checkpoints (PortLocalizer auto-
    # detects the saved mode from state-dict key prefixes).
    aux_pixel_head: bool = False  # v8: per-cam pixel head reading the POOLED
                                  # backbone feature; competed with cam_fuse
                                  # for use of that 512-d bottleneck and
                                  # destabilized pose val (acknowledged 2026-04-30)
    aux_pathway: bool = True       # v9: separate conv pathway from spatial
                                   # features, so the pose-path pooled feature
                                   # is no longer constrained to encode pixels

    def __post_init__(self) -> None:
        if self.aux_pixel_head and self.aux_pathway:
            raise ValueError(
                "aux_pixel_head (v8) and aux_pathway (v9) are mutually "
                "exclusive — they're alternative integration points for the "
                "per-camera port-pixel auxiliary supervision."
            )
        if self.backbone not in ("resnet18", "dinov2_vits14"):
            raise ValueError(
                f"unknown backbone {self.backbone!r}; "
                f"supported: resnet18 | dinov2_vits14"
            )


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
    """Multi-cam images + TCP pose + task one-hot → 5-dim board-pose / rail-t.

    v7: takes `num_cameras` images (default 3 for left/center/right wrist
    cams). All cameras share the ResNet18 backbone — same physical sensor,
    same visual statistics, 3× separate weights would just overfit. Per-cam
    pooled features concat to (B, num_cameras*512) and project back to 512
    via cam_fuse so FiLM dimensions stay constant.

    The shared-backbone forward processes all camera images in one batched
    pass `view(B*num_cameras, 3, H, W)` for GPU efficiency.

    Spatial feature maps (pre-pool) are NOT yet used but the architecture
    keeps them around so v8's heatmap aux loss can layer on without a
    rewrite.
    """

    def __init__(self, config: BoardPoseRegressorConfig | None = None):
        super().__init__()
        self.config = config or BoardPoseRegressorConfig()

        if self.config.backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if self.config.backbone_pretrained else None
            backbone = resnet18(weights=weights)
            feature_dim = backbone.fc.in_features  # 512 for resnet18
            backbone.fc = nn.Identity()
            # Split into trunk (spatial (B,512,7,7)) and avgpool ((B,512)) so
            # the v9 aux pathway can read the spatial maps directly.
            self.backbone_trunk = nn.Sequential(*list(backbone.children())[:-2])
            self.backbone_avgpool = backbone.avgpool
            self.backbone_dinov2 = None
        elif self.config.backbone == "dinov2_vits14":
            # DINOv2 ViT-S/14: 21M params, 384-dim features. patch_size=14, so
            # 224x224 input → 16x16 patch grid (256 tokens) + 1 CLS token. We
            # use the CLS token as the pooled summary (its design role) and
            # reshape patch tokens to a (B, 384, 16, 16) spatial map for the
            # v9 aux pathway. Freezing on by default per config.backbone_freeze.
            self.backbone_dinov2 = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=self.config.backbone_pretrained,
            )
            feature_dim = 384  # ViT-S/14 embedding dim
            if self.config.backbone_freeze:
                for p in self.backbone_dinov2.parameters():
                    p.requires_grad = False
                self.backbone_dinov2.eval()
            self.backbone_trunk = None
            self.backbone_avgpool = None
        else:
            raise ValueError(f"unsupported backbone {self.config.backbone!r}")
        self.feature_dim = feature_dim

        # Multi-cam fusion: concat per-cam pooled features and project back.
        nc = self.config.num_cameras
        if nc > 1:
            self.cam_fuse = nn.Sequential(
                nn.Linear(feature_dim * nc, feature_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.cam_fuse = nn.Identity()

        self.film_task = FiLM(
            conditioning_dim=self.config.task_one_hot_dim,
            feature_dim=feature_dim,
        )
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

        # v8: auxiliary per-camera port-pixel regression head. Sigmoid output
        # so predictions stay in [0, 1] image-normalized space. Operates on
        # the per-camera POOLED feature (the same 512-d bottleneck cam_fuse
        # consumes), so the two heads end up competing for that vector.
        # Disabled by default — kept under config flag for backward compat
        # when loading v8 checkpoints; new training defaults to aux_pathway.
        if self.config.aux_pixel_head:
            self.aux_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2),
                nn.Sigmoid(),
            )
        else:
            self.aux_head = None

        # v9: aux pathway reads from SPATIAL features (pre-pool), runs them
        # through its own small conv stack with its own channel pool, and
        # emits the same (u_norm, v_norm) sigmoid pair v8's head did. This
        # gives aux its own bottleneck — independent of the pooled feature
        # cam_fuse uses — so the two losses no longer fight over a shared
        # 512-d summary. Sized to roughly match v8's aux head (~55K params)
        # to avoid overfitting on 354 episodes:
        #   1×1 conv (32K params): channel reduction, no spatial mixing
        #   3×3 conv (18K params): spatial mixing into 32 channels
        #   global avg pool: aux gets to choose its own per-channel summary
        #   Linear(32→2): final pixel coords; Sigmoid clamps to [0,1].
        # If this underfits the aux loss, escalate channel widths or move to
        # spatial soft-argmax (the documented backup option).
        if self.config.aux_pathway:
            self.aux_conv = nn.Sequential(
                nn.Conv2d(feature_dim, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.aux_pathway_head = nn.Sequential(
                nn.Linear(32, 2),
                nn.Sigmoid(),
            )
        else:
            self.aux_conv = None
            self.aux_pathway_head = None

    def train(self, mode: bool = True) -> "BoardPoseRegressor":
        """Override train() so a frozen backbone stays in eval() mode across
        the per-epoch train/eval mode toggles. Without this, `model.train()`
        in the train loop would recursively flip the backbone's submodules
        (e.g. BatchNorm running-stats updates, dropout activation) which
        defeats the point of freezing."""
        super().train(mode)
        if self.backbone_dinov2 is not None and self.config.backbone_freeze:
            self.backbone_dinov2.eval()
        return self

    def _extract_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the visual backbone on (B*num_cams, 3, H, W).
        Returns (spatial, pooled), each shape:
          spatial: (B*num_cams, feature_dim, h, w)
          pooled : (B*num_cams, feature_dim)
        ResNet18: spatial 7×7, pooled = avgpool over spatial.
        DINOv2:   spatial 16×16 (patch tokens reshaped), pooled = CLS token.
        """
        if self.backbone_dinov2 is not None:
            # DINOv2's `forward_features` returns the LayerNorm'd tokens. The
            # CLS token is the global summary the network was self-supervised
            # to produce; the patch tokens preserve per-region info.
            out = self.backbone_dinov2.forward_features(images)
            cls = out["x_norm_clstoken"]                          # (BN, 384)
            patches = out["x_norm_patchtokens"]                   # (BN, N, 384)
            BN, N, D = patches.shape
            side = int(round(N ** 0.5))
            if side * side != N:
                raise RuntimeError(
                    f"DINOv2 patch token count {N} is not a perfect square; "
                    f"input size must be divisible by patch_size (14)."
                )
            spatial = patches.transpose(1, 2).reshape(BN, D, side, side)
            return spatial, cls
        # ResNet18 path (default).
        spatial = self.backbone_trunk(images)
        pooled = self.backbone_avgpool(spatial).flatten(1)
        return spatial, pooled

    def forward(
        self,
        images: torch.Tensor,       # (B, num_cams, 3, H, W) float32, ImageNet-norm
        tcp_pose: torch.Tensor,     # (B, 7) float32, base_link xyz + xyzw
        task_one_hot: torch.Tensor, # (B, 7) float32
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """If `return_aux=True` and aux head is enabled, returns (pred_5,
        aux_pixels) where aux_pixels has shape (B, num_cams, 2) in [0,1].
        At inference (no aux supervision), default `return_aux=False` keeps
        the call site identical to v7.
        """
        # Accept (B, 3, H, W) for legacy single-cam paths and reshape.
        if images.dim() == 4:
            images = images.unsqueeze(1)
        B, num_cams, C, H, W = images.shape
        if num_cams != self.config.num_cameras:
            raise ValueError(
                f"got {num_cams} camera images but model configured for "
                f"num_cameras={self.config.num_cameras}"
            )
        flat = images.view(B * num_cams, C, H, W)
        spatial_flat, pooled_flat = self._extract_features(flat)
        # spatial_flat: (B*num_cams, 512, 7, 7); pooled_flat: (B*num_cams, 512)

        # Per-camera aux pixel regression. Both v8 and v9 emit the same
        # output shape (B, num_cams, 2) in [0, 1] image-normalized space —
        # only the input pathway differs:
        #   v8 (aux_pixel_head): reads the SHARED pooled feature
        #   v9 (aux_pathway):    reads the SPATIAL feature map via its own
        #                        conv stack, leaving the pose pooled feature
        #                        untouched
        # Computed BEFORE cam_fuse so each cam's aux output sees only its
        # own image (matches the per-cam pixel target).
        aux_pixels = None
        if return_aux:
            if self.aux_head is not None:
                aux_flat = self.aux_head(pooled_flat)
                aux_pixels = aux_flat.view(B, num_cams, 2)
            elif self.aux_conv is not None:
                aux_pooled = self.aux_conv(spatial_flat)        # (B*num_cams, 32)
                aux_flat = self.aux_pathway_head(aux_pooled)    # (B*num_cams, 2)
                aux_pixels = aux_flat.view(B, num_cams, 2)

        pooled = pooled_flat.view(B, num_cams * self.feature_dim)
        feat = self.cam_fuse(pooled)                           # (B, 512)
        feat = self.film_task(feat, task_one_hot)
        tcp_emb = self.tcp_proj(tcp_pose)
        feat = self.film_tcp(feat, tcp_emb)
        pred = self.head(feat)
        if return_aux:
            return pred, aux_pixels
        return pred


# Per-dim loss weights. v7: boost xy 3× since yaw/rail were close to target
# at v6's plateau (3.1° and 3.3mm) but xy was 8× over (24mm vs <3mm target).
# Scaling xy components in normalized-MSE space redirects gradient toward
# the bottleneck axis without de-rating the others below useful learning
# signal.
LOSS_WEIGHTS = (3.0, 3.0, 1.0, 1.0, 1.0)

# v8: weight for the auxiliary per-camera port-pixel MSE loss. The aux
# target is in [0, 1] image-normalized space, so MSE is naturally O(0.01).
# AUX_WEIGHT scales it to be a non-trivial fraction of the pose loss
# (which is O(1) in z-space) — large enough to force the spatial features
# to encode "where is the port" info, small enough not to dominate the
# pose head's gradient.
AUX_PIXEL_WEIGHT = 5.0


def aux_pixel_loss(
    pred_pixels: torch.Tensor,    # (B, num_cams, 2)  in [0, 1]
    target_pixels: torch.Tensor,  # (B, num_cams, 3)  [u_norm, v_norm, valid]
) -> torch.Tensor:
    """Masked MSE on per-camera port pixel positions.

    Frames where the projected port is out-of-frame or behind the camera
    have valid=0 and contribute zero to the loss (and to the denominator
    for normalization). Returns the per-valid-sample mean. If no sample is
    valid in the batch, returns 0 (so the optimizer step is a no-op rather
    than a NaN).
    """
    target_uv = target_pixels[..., :2]    # (B, num_cams, 2)
    valid = target_pixels[..., 2:3]       # (B, num_cams, 1)
    sq = (pred_pixels - target_uv) ** 2   # (B, num_cams, 2)
    masked = sq * valid                    # zero where invalid
    n_valid = valid.sum() * 2              # 2 channels per valid sample
    if n_valid.item() == 0:
        return torch.zeros((), device=pred_pixels.device, dtype=pred_pixels.dtype)
    return masked.sum() / n_valid


def loss_fn(
    pred: torch.Tensor, target: torch.Tensor, *, return_components: bool = False
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Weighted MSE between model output (z-scored) and z-scored target.

    `target` is in physical units; we normalize it here so callers don't
    have to remember the convention. Per-dim weights `LOSS_WEIGHTS` shape
    the gradient — see module-level note.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    target_n = normalize_target(target)
    diff = pred - target_n
    sq = diff ** 2
    w = torch.tensor(LOSS_WEIGHTS, dtype=sq.dtype, device=sq.device)
    weighted = sq * w
    total = weighted.mean()
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
