"""Runtime localizer wrapper.

Loads a trained BoardPoseRegressor checkpoint and provides a single method
`predict_port_pose(images, tcp_pose, target_module_name, port_name)` that
returns a 7-DoF port pose `(x, y, z, qx, qy, qz, qw)` in base_link.

Used at submission time as a drop-in replacement for the TF lookup in
CheatCodeRobust. Composes:
  - Network output (board_x, board_y, sin_yaw, cos_yaw, target_rail_translation)
  - URDF formula for port_in_board (via reconstruct_port_in_baselink)
  - Static port-in-board rotation per port type (Phase B2 calibration)

Image preprocessing matches train_localizer.py exactly — same resize, same
ImageNet normalization. The inference path runs at policy tick rate (~50 ms
budget); a single ResNet18 forward at 224×224 on L4 is ~5-10 ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.transforms import functional as TF

from .labels import LocalizerLabel, reconstruct_port_in_baselink, task_one_hot
from .model import BoardPoseRegressor, BoardPoseRegressorConfig, predicted_yaw_rad


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_MODEL_INPUT_SIZE = 224


# Static port-in-board quaternion per (port_type, port_name).
# Populated empirically from a recorder dataset by
# calibrate_port_in_board_rotation(); the caller fills these in once and
# reuses across trials. Quaternion convention: (qw, qx, qy, qz).
DEFAULT_PORT_IN_BOARD_QUAT: dict[tuple[str, str], tuple[float, float, float, float]] = {}


def _preprocess_image(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Same as train_localizer.py — keep aligned with training distribution."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"image must be 3D (got shape {image.shape})")
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
        if chw.max() > 1.5:
            chw = chw / 255.0
    chw = TF.resize(chw, [_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE], antialias=True)
    chw = TF.normalize(chw, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))
    return chw


def _quat_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product, (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    """R_z(yaw) as (w, x, y, z)."""
    half = yaw_rad * 0.5
    return (np.cos(half), 0.0, 0.0, np.sin(half))


@dataclass
class PortPose:
    """7-DoF port pose in base_link. Matches the TF transform shape that
    CheatCodeRobust currently consumes."""

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


class PortLocalizer:
    """Trained-model wrapper for runtime use.

    Construction is heavy (loads a 46 MB ResNet18). Do it once at
    aic_model.activate() and reuse the instance across trials.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "cuda",
        port_in_board_quat: dict[tuple[str, str], tuple[float, float, float, float]] | None = None,
    ):
        self.device = torch.device(device)
        ckpt = torch.load(str(checkpoint_path), map_location=self.device)
        config = BoardPoseRegressorConfig(**ckpt.get("config", {}))
        # backbone_pretrained=False at inference: we'll load weights from the
        # checkpoint, no need to download from torchvision (and no internet
        # at submission anyway).
        config.backbone_pretrained = False
        self.model = BoardPoseRegressor(config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._port_in_board_quat = dict(
            port_in_board_quat if port_in_board_quat is not None
            else DEFAULT_PORT_IN_BOARD_QUAT
        )

    def set_port_in_board_quat(
        self, port_type: str, port_name: str,
        quat: tuple[float, float, float, float],
    ) -> None:
        """Register the static (qw, qx, qy, qz) for one port. Call this before
        `predict_port_pose` for each port type/name you'll query."""
        self._port_in_board_quat[(port_type, port_name)] = tuple(quat)

    @torch.no_grad()
    def predict_port_pose(
        self,
        image: np.ndarray | torch.Tensor,
        tcp_pose: np.ndarray | torch.Tensor,
        target_module_name: str,
        port_name: str,
        port_type: str,
    ) -> PortPose:
        """Run one forward pass + URDF composition. Returns 7-DoF port pose."""
        img = _preprocess_image(image).unsqueeze(0).to(self.device)
        if isinstance(tcp_pose, np.ndarray):
            tcp = torch.from_numpy(tcp_pose).float()
        else:
            tcp = tcp_pose.float()
        tcp = tcp.unsqueeze(0).to(self.device)
        oh = torch.from_numpy(task_one_hot(target_module_name)).float().unsqueeze(0).to(self.device)

        pred = self.model(img, tcp, oh).squeeze(0).cpu().numpy()
        bx, by, sin_yaw, cos_yaw, rail_t = pred.tolist()
        yaw = float(np.arctan2(sin_yaw, cos_yaw))

        label = LocalizerLabel(
            board_x_baselink=bx, board_y_baselink=by,
            board_yaw_baselink_rad=yaw,
            target_rail_translation_m=rail_t,
            port_type=port_type,
        )
        port_xyz = reconstruct_port_in_baselink(label, target_module_name, port_name)

        # Compose port quaternion: Q_port_baselink = R_z(yaw) ⊗ Q_port_in_board
        key = (port_type, port_name)
        if key not in self._port_in_board_quat:
            raise KeyError(
                f"no port-in-board quaternion registered for {key}. "
                f"Call set_port_in_board_quat() or pass port_in_board_quat "
                f"dict at construction. Run calibrate_port_in_board_rotation "
                f"on the training dataset to populate."
            )
        q_pb = self._port_in_board_quat[key]
        q_yaw = _yaw_to_quat(yaw)
        q_port_baselink = _quat_multiply(q_yaw, q_pb)
        qw, qx, qy, qz = q_port_baselink

        return PortPose(
            x=float(port_xyz[0]),
            y=float(port_xyz[1]),
            z=float(port_xyz[2]),
            qx=float(qx),
            qy=float(qy),
            qz=float(qz),
            qw=float(qw),
        )
