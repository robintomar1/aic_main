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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.transforms import functional as TF

from .labels import LocalizerLabel, reconstruct_port_in_baselink, task_one_hot
from .model import (
    BoardPoseRegressor,
    BoardPoseRegressorConfig,
    denormalize_pred,
    predicted_yaw_rad,
)


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


def _load_quats_json(
    path: Path,
) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """Load port-in-board quats produced by calibrate_localizer_quats.py.

    File format: `{"<port_type>|<port_name>": [qw, qx, qy, qz], ...}`.
    """
    raw = json.loads(Path(path).read_text())
    out: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for k, v in raw.items():
        port_type, port_name = k.split("|", 1)
        out[(port_type, port_name)] = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return out


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
        quats_json_path: Path | None = None,
    ):
        self.device = torch.device(device)
        # weights_only=False: the checkpoint payload includes non-tensor
        # objects (PosixPath in the saved args, plus the cameras list etc.)
        # that PyTorch 2.6+'s default safe-loader rejects. We trust our own
        # checkpoint files so this is safe; matches train_localizer.py's
        # resume path which uses the same flag.
        ckpt = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False,
        )
        config = BoardPoseRegressorConfig(**ckpt.get("config", {}))
        # backbone_pretrained=False at inference: we'll load weights from the
        # checkpoint, no need to download from torchvision (and no internet
        # at submission anyway).
        config.backbone_pretrained = False
        # Auto-detect aux head presence from the saved state_dict — older
        # checkpoints (v7 pre-aux, v8 pooled-aux) were saved before the
        # aux_pathway field existed, so their saved config either doesn't
        # carry the right defaults or carries values incompatible with the
        # current dataclass defaults. Trust the weights over the saved config.
        # Anchor checks at startswith() so v8's "aux_head.*" doesn't match
        # any v9 prefix and vice versa.
        sd = ckpt["model_state_dict"]
        has_aux_head = any(k.startswith("aux_head.") for k in sd)
        has_aux_pathway = (
            any(k.startswith("aux_conv.") for k in sd)
            or any(k.startswith("aux_pathway_head.") for k in sd)
        )
        config.aux_pixel_head = has_aux_head
        config.aux_pathway = has_aux_pathway
        self.model = BoardPoseRegressor(config).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()
        # Resolve port-in-board quaternions in priority order:
        # 1. explicit dict argument (test/dev override),
        # 2. explicit JSON path,
        # 3. <checkpoint>.quats.json next to the checkpoint (default),
        # 4. DEFAULT_PORT_IN_BOARD_QUAT (empty unless set externally).
        if port_in_board_quat is not None:
            self._port_in_board_quat = dict(port_in_board_quat)
        else:
            ckpt_path = Path(checkpoint_path)
            default_json = ckpt_path.with_suffix(ckpt_path.suffix + ".quats.json")
            json_path = quats_json_path or (default_json if default_json.exists() else None)
            if json_path is not None and Path(json_path).exists():
                self._port_in_board_quat = _load_quats_json(Path(json_path))
            else:
                self._port_in_board_quat = dict(DEFAULT_PORT_IN_BOARD_QUAT)
        # Camera ordering bound to the checkpoint. The model was trained with
        # this exact order (cam_fuse Linear weights are positional); a runtime
        # caller passing a dict is safer than a list because we look up by
        # name and reorder here, rather than trusting the caller to remember
        # which slot is which.
        self._cameras: list[str] = list(ckpt.get("cameras", []))
        if not self._cameras:
            raise ValueError(
                f"checkpoint {checkpoint_path} has no 'cameras' metadata. "
                f"This checkpoint was saved before v7's multi-cam contract; "
                f"retrain or set self._cameras manually."
            )
        if len(self._cameras) != self.model.config.num_cameras:
            raise ValueError(
                f"checkpoint cameras={self._cameras} (len {len(self._cameras)}) "
                f"disagrees with model num_cameras={self.model.config.num_cameras}"
            )

    @property
    def cameras(self) -> list[str]:
        """Camera names the model expects, in the order it was trained on."""
        return list(self._cameras)

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
        images: dict | list | np.ndarray | torch.Tensor,
        tcp_pose: np.ndarray | torch.Tensor,
        target_module_name: str,
        port_name: str,
        port_type: str,
    ) -> PortPose:
        """Run one forward pass + URDF composition. Returns 7-DoF port pose.

        `images` accepts:
          - dict[str, image]: keyed by camera name (left_camera/center_camera/
            right_camera). Reordered by `self.cameras` automatically — this is
            the SAFE form, ordering can't go wrong by accident.
          - list[image]: assumed already ordered to match self.cameras. Used
            only when the caller has explicitly verified the order.
          - single ndarray/Tensor: only valid for a 1-cam model.
        """
        # Normalize to an ordered list matching self._cameras.
        if isinstance(images, dict):
            missing = [c for c in self._cameras if c not in images]
            if missing:
                raise KeyError(
                    f"images dict missing camera keys {missing!r}; "
                    f"have {list(images.keys())!r}, need {self._cameras!r}"
                )
            ordered = [images[c] for c in self._cameras]
        elif isinstance(images, list):
            ordered = images
        else:
            # single image — only valid for 1-cam model.
            ordered = [images]

        if len(ordered) != self.model.config.num_cameras:
            raise ValueError(
                f"got {len(ordered)} images but model expects "
                f"{self.model.config.num_cameras} cameras "
                f"({self._cameras!r})"
            )
        per_cam = [_preprocess_image(im) for im in ordered]
        img_tensor = torch.stack(per_cam, dim=0).unsqueeze(0).to(self.device)  # (1, num_cams, 3, H, W)
        if isinstance(tcp_pose, np.ndarray):
            tcp = torch.from_numpy(tcp_pose).float()
        else:
            tcp = tcp_pose.float()
        tcp = tcp.unsqueeze(0).to(self.device)
        oh = torch.from_numpy(task_one_hot(target_module_name)).float().unsqueeze(0).to(self.device)

        pred_n = self.model(img_tensor, tcp, oh)         # z-scored model output
        pred = denormalize_pred(pred_n).squeeze(0).cpu().numpy()
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
