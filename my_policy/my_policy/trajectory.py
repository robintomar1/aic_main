"""Pure-helper module for trajectory math used by CheatCodeContinuous.

Quaternion ops + pose interpolation + minimum-jerk shape function. Self-contained
(no transforms3d dependency) so the module is host-testable without the pixi env.
Convention for quaternions: (w, x, y, z), matches transforms3d._gohlketransforms
so values are interchangeable with the rest of the codebase.

This module evolved from `probe.py` (the failed Stage-1 tilt-probe attempt). The
probe-specific symbols (TiltProbeStateMachine, ProbeConfig, compute_tilt_axis,
decide_probe_direction) were removed when the tilt-probe approach was scrapped.
The remaining helpers are general-purpose trajectory primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Quaternion ops (w, x, y, z)
# ----------------------------------------------------------------------------

def quaternion_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product, (w, x, y, z) convention. Same result as
    transforms3d._gohlketransforms.quaternion_multiply."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quaternion_slerp(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    """Spherical linear interpolation between q1 and q2 by `fraction` ∈ [0, 1].
    Always takes the shorter arc (negates q2 if dot(q1, q2) < 0).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    dot = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    if dot < 0.0:
        w2, x2, y2, z2 = -w2, -x2, -y2, -z2
        dot = -dot
    if dot > 0.9995:
        # Quats nearly parallel — linear interpolation suffices, then normalize.
        w = w1 + (w2 - w1) * fraction
        x = x1 + (x2 - x1) * fraction
        y = y1 + (y2 - y1) * fraction
        z = z1 + (z2 - z1) * fraction
        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        return (w / n, x / n, y / n, z / n)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * fraction
    sin_theta = math.sin(theta)
    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return (
        s1 * w1 + s2 * w2,
        s1 * x1 + s2 * x2,
        s1 * y1 + s2 * y2,
        s1 * z1 + s2 * z2,
    )


def axis_angle_to_quat(
    axis_xyz: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float, float]:
    """Axis-angle → quaternion (w, x, y, z). Matches transforms3d convention.
    Axis is taken as a 3-tuple (or anything indexable [0..2]) so this works
    for both numpy arrays and plain tuples."""
    half = angle_rad * 0.5
    s = math.sin(half)
    return (
        math.cos(half),
        float(axis_xyz[0]) * s,
        float(axis_xyz[1]) * s,
        float(axis_xyz[2]) * s,
    )


def rotate_vec_by_quat(
    v: tuple[float, float, float],
    q: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Rotate 3-vector v by quaternion q (w, x, y, z). Returns a new 3-vector
    as a (vx, vy, vz) tuple."""
    qv = (0.0, float(v[0]), float(v[1]), float(v[2]))
    qc = (q[0], -q[1], -q[2], -q[3])
    rotated = quaternion_multiply(quaternion_multiply(q, qv), qc)
    return (rotated[1], rotated[2], rotated[3])


# ----------------------------------------------------------------------------
# Minimum-jerk shape function
# ----------------------------------------------------------------------------

def minjerk_s(t_norm: float) -> float:
    """Minimum-jerk shape function on t_norm ∈ [0, 1]:
    s(t) = t³(10 − 15t + 6t²). s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0.

    Used for smooth ramps in trajectory phases (Phase 1 XY interpolation,
    Phase 2 orientation slerp, spiral-amplitude on/off ramps, LATCH blend).
    """
    t = max(0.0, min(1.0, t_norm))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


# ----------------------------------------------------------------------------
# PoseSnapshot — pose dataclass usable in tests AND production
# ----------------------------------------------------------------------------

@dataclass
class PoseSnapshot:
    """A frozen (position, orientation) pair. Avoids depending on
    geometry_msgs.Pose construction in host tests; convertible from/to
    Pose / Transform via the from_ros_* classmethods.
    """
    px: float
    py: float
    pz: float
    qw: float
    qx: float
    qy: float
    qz: float

    @classmethod
    def from_ros_pose(cls, pose) -> "PoseSnapshot":
        return cls(
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z,
        )

    @classmethod
    def from_ros_transform(cls, transform) -> "PoseSnapshot":
        return cls(
            transform.translation.x, transform.translation.y, transform.translation.z,
            transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z,
        )

    def quat(self) -> tuple[float, float, float, float]:
        return (self.qw, self.qx, self.qy, self.qz)

    def position(self) -> tuple[float, float, float]:
        return (self.px, self.py, self.pz)


def lerp_pose(start: PoseSnapshot, end: PoseSnapshot, s: float) -> PoseSnapshot:
    """Lerp position, slerp orientation. s ∈ [0, 1]; clamped if outside."""
    s = max(0.0, min(1.0, s))
    qs = quaternion_slerp(start.quat(), end.quat(), s)
    return PoseSnapshot(
        px=start.px + (end.px - start.px) * s,
        py=start.py + (end.py - start.py) * s,
        pz=start.pz + (end.pz - start.pz) * s,
        qw=qs[0], qx=qs[1], qy=qs[2], qz=qs[3],
    )
