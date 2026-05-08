"""Port-local frame transformations — convert recorded base_link-frame
quantities (TCP pose, action, velocity, wrench) into the target port's
coordinate frame.

The motivation: the IL dataset records actions as absolute TCP pose targets
in `base_link`, which forces the model to learn `(image, state) → world-frame`
mappings spanning 360° of board yaw + ±15 cm of board XY. Re-expressing
everything in port-local frame collapses that variation. EasyInsert
(arxiv/2505.16187) and CFVS (arxiv/2209.08864) both validate this
representation choice for peg-in-hole insertion.

This module is pure numpy. No torch, no lerobot, no rclpy. Every public
function takes plain arrays and returns plain arrays so it can be unit-
tested on a host without GPU/sim/ROS plumbing.

Conventions baked in (verified 2026-05-08 against the dataset and recorder):
  * **Quaternions are XYZW.** Confirmed in:
      - dataset `meta/info.json` channel names (qx, qy, qz, qw)
      - `aic_robot_aic_controller.py:_lookup_pose` returns (..., qx, qy, qz, qw)
      - action `pose.orientation.{x,y,z,w}` ordering in recorder
  * **`groundtruth.port_pose`** is `T_port_in_baselink`: the pose of the
    target port frame expressed in `base_link`. Verified from
    `aic_robot_aic_controller.py:_lookup_pose` docstring
    "Returns (x, y, z, qx, qy, qz, qw) for `base_link -> frame_id`".
  * **`tcp_pose`** is `T_tcp_in_baselink`. Same convention, recorded
    in the same way.
  * **`action`** (7-d) is an absolute TCP pose target in `base_link`.
    The recorder subscribes to `/aic_controller/pose_commands` (a
    MotionUpdate in MODE_POSITION) and stores `pose.position` +
    `pose.orientation` directly.
  * **`tcp_velocity`** (linear+angular twist) is the TCP velocity in
    `base_link`. We rotate the linear and angular components
    independently by `R_port_in_baselink^T` to express them in port
    frame.
  * **`wrench`** is the F/T sensor reading in the F/T sensor's body frame
    (which co-rotates with TCP, modulo a small rigid offset). For the
    transform we use the rotation of the TCP relative to the port and
    apply that to wrench. This is an *approximation* — it ignores the
    sensor-to-TCP rigid offset, but the offset is small (~few cm) and
    rotation-only (no translation in the wrench transform).
    See `feedback_no_assumed_numbers.md`: this is documented assumption,
    not verified. Test 6 in `test_port_local_transforms.py` checks
    rotational invariance; if the sensor-frame assumption is wrong,
    the per-task slice test (Tier 2 #8) will catch a mismatch.

Frame notation throughout this module:
  * `T_X_in_Y` is a 4×4 homogeneous transform such that `p_Y = T_X_in_Y @ p_X`.
    "X expressed in Y" is the verbose reading.
  * `R_X_in_Y` is the 3×3 rotation from `T_X_in_Y[:3, :3]`.

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# SE(3) primitives — re-implemented here rather than importing from the
# localizer module so this module's tests establish their own correctness
# without trust-by-import. Test 0 cross-checks our primitives against the
# localizer's so an invariant linking them is documented (and would fail
# loudly if either side drifts).
# ---------------------------------------------------------------------------


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw) → 3×3 rotation matrix.

    Standard right-handed convention. The rotation matrix maps body-frame
    vectors to world-frame vectors when the quaternion describes the body
    orientation in the world frame: `v_world = R @ v_body`.
    """
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    x, y, z, w = q
    # Defensive: normalize. Recorded quaternions can drift slightly from
    # unit norm due to float32 storage (~1e-7 error); without normalization
    # the resulting R has determinant slightly off 1, which compounds when
    # composed.
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError(f"quaternion has near-zero norm: {q.tolist()}")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → (qx, qy, qz, qw). Inverse of quat_xyzw_to_rotmat.

    Uses Shepperd's method (numerically stable across all rotation regimes).
    Returns w-positive convention (q and -q are the same rotation; we pick
    the half-sphere with w >= 0 so consecutive frames don't flip sign for
    near-identity rotations).
    """
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    if q[3] < 0:
        q = -q
    return q


def make_se3(xyz: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """Build T_X_in_Y from translation `xyz` and rotation `q_xyzw`."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotmat(q_xyzw)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Invert a rigid transform: T_Y_in_X = (T_X_in_Y)^-1.

    Uses the closed form `(R^T, -R^T t)` rather than `np.linalg.inv` —
    faster and exact for rigid transforms (no float drift from a general
    matrix inverse).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def split_se3(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose T into (xyz, qxyzw)."""
    xyz = np.asarray(T[:3, 3], dtype=np.float64).copy()
    quat = rotmat_to_quat_xyzw(T[:3, :3])
    return xyz, quat


# ---------------------------------------------------------------------------
# The actual port-local transforms.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameInputs:
    """Per-frame raw quantities pulled out of the dataset row.

    All in `base_link`. Quaternions are xyzw. Velocity/wrench are 6-vectors
    (linear/force first, then angular/torque).

    `port_pose_baselink` and `tcp_pose_baselink` are 7-vectors: xyz + xyzw.
    """
    tcp_pose_baselink: np.ndarray      # (7,) xyz + xyzw
    tcp_velocity_baselink: np.ndarray  # (6,) linear, angular
    wrench_sensorframe: np.ndarray     # (6,) force, torque
    action_baselink: np.ndarray        # (7,) xyz + xyzw — pose target
    port_pose_baselink: np.ndarray     # (7,) xyz + xyzw


@dataclass(frozen=True)
class FrameOutputs:
    """Per-frame port-local quantities — same shapes as FrameInputs minus
    `port_pose_baselink` (the new origin is implicit) and minus
    `wrench_sensorframe` (reported as `wrench_portframe` instead).
    """
    tcp_pose_portframe: np.ndarray      # (7,)
    tcp_velocity_portframe: np.ndarray  # (6,)
    wrench_portframe: np.ndarray        # (6,)
    action_portframe: np.ndarray        # (7,)


def _transform_pose_into_port(
    pose_baselink_7d: np.ndarray, T_port_in_bl: np.ndarray
) -> np.ndarray:
    """Re-express a `T_X_in_baselink`-style 7-vector as `T_X_in_port`.

    Uses `T_X_in_port = T_baselink_in_port @ T_X_in_baselink
                      = inv(T_port_in_baselink) @ T_X_in_baselink`.
    """
    xyz = pose_baselink_7d[:3]
    quat = pose_baselink_7d[3:7]
    T_X_in_bl = make_se3(xyz, quat)
    T_X_in_port = se3_inverse(T_port_in_bl) @ T_X_in_bl
    new_xyz, new_quat = split_se3(T_X_in_port)
    return np.concatenate([new_xyz, new_quat]).astype(np.float64)


def _rotate_twist_into_port(
    twist_baselink_6d: np.ndarray, R_port_in_bl: np.ndarray
) -> np.ndarray:
    """Rotate a (linear, angular) twist from base_link to port frame.

    Twist components rotate independently — there is no translation
    coupling for a velocity expressed at the same physical point.
    The TCP-velocity in base_link gives "TCP point's velocity in
    base_link"; rotating both linear and angular by `R_port_in_bl^T`
    gives the same physical velocity expressed in port-frame axes.

    NOTE: this assumes the velocity is expressed at the TCP and we're
    only changing the *axes* it's reported in, not the reference point.
    The recorder publishes `controller_state.tcp_velocity` which is
    exactly that.
    """
    R = R_port_in_bl.T  # base_link → port
    linear = twist_baselink_6d[:3]
    angular = twist_baselink_6d[3:6]
    return np.concatenate([R @ linear, R @ angular]).astype(np.float64)


def _rotate_wrench_into_port(
    wrench_sensor_6d: np.ndarray,
    R_tcp_in_bl: np.ndarray,
    R_port_in_bl: np.ndarray,
) -> np.ndarray:
    """Rotate a (force, torque) wrench from F/T sensor frame to port frame.

    Assumption (documented in module header): F/T sensor frame ≈ TCP frame
    (a small rigid offset exists in the URDF but the rotation is identity).
    Under this assumption:
        R_sensor_in_port = R_baselink_in_port @ R_sensor_in_baselink
                         ≈ R_port_in_bl^T @ R_tcp_in_bl

    Force and torque rotate independently (the ATI sensor publishes both
    at the same physical point, so there's no force-torque coupling under
    a pure rotation). If the sensor-to-TCP offset matters in some future
    application, this is the place to add it: `wrench_sensor` would need
    a small adjoint correction `[[I, 0], [t̂, I]]` where `t̂` is the
    skew-symmetric of the sensor-to-TCP translation.
    """
    R = R_port_in_bl.T @ R_tcp_in_bl  # sensor (~tcp) → port
    force = wrench_sensor_6d[:3]
    torque = wrench_sensor_6d[3:6]
    return np.concatenate([R @ force, R @ torque]).astype(np.float64)


def transform_frame(inp: FrameInputs) -> FrameOutputs:
    """Transform one frame's worth of base_link quantities to port-local.

    This is the workhorse function — the dataset builder calls it once
    per frame and writes the results to the new dataset.
    """
    T_port_in_bl = make_se3(
        inp.port_pose_baselink[:3], inp.port_pose_baselink[3:7]
    )
    R_port_in_bl = T_port_in_bl[:3, :3]
    R_tcp_in_bl = quat_xyzw_to_rotmat(inp.tcp_pose_baselink[3:7])

    return FrameOutputs(
        tcp_pose_portframe=_transform_pose_into_port(
            inp.tcp_pose_baselink, T_port_in_bl
        ),
        tcp_velocity_portframe=_rotate_twist_into_port(
            inp.tcp_velocity_baselink, R_port_in_bl
        ),
        wrench_portframe=_rotate_wrench_into_port(
            inp.wrench_sensorframe, R_tcp_in_bl, R_port_in_bl
        ),
        action_portframe=_transform_pose_into_port(
            inp.action_baselink, T_port_in_bl
        ),
    )


def transform_pose_back_to_baselink(
    pose_portframe_7d: np.ndarray, port_pose_baselink_7d: np.ndarray
) -> np.ndarray:
    """Inverse of `_transform_pose_into_port` — the inference-time op.

    Given a port-local pose target (e.g. policy output) and the localizer's
    estimated port pose in base_link, returns the equivalent base_link pose
    that we hand to `set_pose_target`.

    Used both at inference time AND in Tier 1 / Tier 2 round-trip tests
    that verify our port→baselink reconstruction matches the original
    base_link recording exactly.
    """
    T_port_in_bl = make_se3(
        port_pose_baselink_7d[:3], port_pose_baselink_7d[3:7]
    )
    T_X_in_port = make_se3(
        pose_portframe_7d[:3], pose_portframe_7d[3:7]
    )
    T_X_in_bl = T_port_in_bl @ T_X_in_port
    new_xyz, new_quat = split_se3(T_X_in_bl)
    return np.concatenate([new_xyz, new_quat]).astype(np.float64)
