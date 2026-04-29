"""Project port pose into camera pixels — used for v8 heatmap aux supervision.

The cameras are wrist-mounted, so `base_link → camera_optical` varies per frame
with TCP. To project a port location (in base_link) to pixels at any frame,
we need the static transform `tcp → camera_optical` per camera and compose
with the per-frame `base_link → tcp` (i.e. the recorded TCP pose).

This module:
  - Computes `tcp → camera_optical (static)` from a single HOME-pose data point.
    `gen_trial_config.CAMERA_OPTICAL_IN_BASE_LINK` already gives `base_link →
    camera_optical at HOME`. Reading TCP at frame 0 of any episode (where the
    robot is at HOME by recorder convention) gives the missing piece.
  - Projects a port pose at any later frame to per-camera (u, v) by composing
    the per-frame TCP with the static rigid extrinsic.

No torch — pure numpy. Pixel coords are in image frame [0, IMG_W) × [0, IMG_H).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gen_trial_config as gtc  # noqa: E402


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw) → 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _se3_from_xyz_quat(xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transform from translation + xyzw quaternion."""
    T = np.eye(4)
    T[:3, :3] = _quat_xyzw_to_rotmat(quat_xyzw)
    T[:3, 3] = xyz
    return T


def _se3_inverse(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4×4 transform (R^T, -R^T t)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def compute_static_tcp_to_camera_optical(
    home_tcp_xyz: np.ndarray,
    home_tcp_quat_xyzw: np.ndarray,
) -> dict[str, np.ndarray]:
    """Given the recorded TCP pose at HOME, compute the rigid pose of each
    camera_optical frame **in TCP frame** (`T_cam_in_tcp`), one per camera.

    SE(3) convention: a 4×4 matrix `T_X_in_Y` built from a pose
    "X expressed in Y" (xyz, quat) transforms X-frame coords → Y-frame coords:
        p_Y = T_X_in_Y @ p_X.

    Composition we use here:
        T_cam_in_tcp = T_bl_in_tcp @ T_cam_in_bl
                     = inv(T_tcp_in_bl) @ T_cam_in_bl
    where `T_tcp_in_bl` is built from the TCP pose at HOME (recorded in
    base_link), and `T_cam_in_bl` is the static HOME-pose camera extrinsic
    from gen_trial_config.CAMERA_OPTICAL_IN_BASE_LINK.

    Returns dict[cam_short_name, T_cam_in_tcp]. Camera-mount-to-wrist is
    rigid, so this transform is constant across all frames after init.
    """
    T_tcp_in_bl = _se3_from_xyz_quat(home_tcp_xyz, home_tcp_quat_xyzw)
    T_bl_in_tcp = _se3_inverse(T_tcp_in_bl)
    out: dict[str, np.ndarray] = {}
    for short_name, ext in gtc.CAMERA_OPTICAL_IN_BASE_LINK.items():
        T_cam_in_bl = _se3_from_xyz_quat(
            np.array(ext["xyz"]), np.array(ext["quat"]),
        )
        out[short_name] = T_bl_in_tcp @ T_cam_in_bl
    return out


def project_port_to_pixels(
    port_baselink: np.ndarray,
    tcp_baselink_xyz: np.ndarray,
    tcp_baselink_quat_xyzw: np.ndarray,
    static_tcp_to_camera_optical: dict[str, np.ndarray],
) -> dict[str, tuple[float, float, float]]:
    """Project a 3D port location (base_link, meters) to (u, v, depth) per
    camera at the current frame's TCP pose.

    Composes the per-frame TCP pose with the static `T_cam_in_tcp`:
        T_cam_in_bl_now = T_tcp_in_bl_now @ T_cam_in_tcp
        p_cam = inv(T_cam_in_bl_now) @ p_bl
    Then applies pinhole intrinsics (FX, CX, FY, CY from gen_trial_config).

    Returns: dict[cam_short_name, (u, v, depth)]. depth > 0 means in front
    of the camera; depth ≤ 0 means behind (NaN for u,v). Depth is reported
    so downstream consumers can mask out-of-frame / behind-camera samples.
    """
    T_tcp_in_bl = _se3_from_xyz_quat(tcp_baselink_xyz, tcp_baselink_quat_xyzw)
    out: dict[str, tuple[float, float, float]] = {}
    port_h = np.append(port_baselink, 1.0)
    for short_name, T_cam_in_tcp in static_tcp_to_camera_optical.items():
        T_cam_in_bl = T_tcp_in_bl @ T_cam_in_tcp
        T_bl_in_cam = _se3_inverse(T_cam_in_bl)
        port_in_cam = (T_bl_in_cam @ port_h)[:3]
        z = float(port_in_cam[2])
        if z <= 0:
            out[short_name] = (float("nan"), float("nan"), z)
            continue
        u = gtc.FX * port_in_cam[0] / z + gtc.CX
        v = gtc.FY * port_in_cam[1] / z + gtc.CY
        out[short_name] = (float(u), float(v), z)
    return out


# Map from LeRobot camera key to the short name used in gen_trial_config.
LEROBOT_CAM_TO_SHORT = {
    "left_camera": "left",
    "center_camera": "center",
    "right_camera": "right",
}
