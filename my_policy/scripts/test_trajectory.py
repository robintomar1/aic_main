#!/usr/bin/env python3
"""Host-runnable tests for my_policy.trajectory.

Tests pure helpers: quaternion ops, axis-angle conversion, vector rotation,
PoseSnapshot lerp, minjerk shape function. No rclpy/transforms3d dependency.

Runs WITHOUT pixi env: numpy is the only runtime dep.
"""

from __future__ import annotations

import math
import sys

import numpy as np

# Add my_policy package to path so we can import without ROS.
sys.path.insert(
    0, "/home/robin/ssd/aic_workspace/aic_code_robin/aic_main/my_policy"
)

from my_policy.trajectory import (  # noqa: E402
    PoseSnapshot,
    axis_angle_to_quat,
    lerp_pose,
    minjerk_s,
    quaternion_multiply,
    quaternion_slerp,
    rotate_vec_by_quat,
)


# ============================================================================
# Quaternion ops
# ============================================================================

def test_quaternion_multiply_identity_left():
    """1 ⊗ q == q for any q."""
    q = (math.cos(0.3), math.sin(0.3), 0.0, 0.0)  # 0.6 rad about X
    out = quaternion_multiply((1.0, 0.0, 0.0, 0.0), q)
    assert np.allclose(out, q, atol=1e-12)


def test_quaternion_multiply_identity_right():
    """q ⊗ 1 == q for any q."""
    q = (math.cos(0.3), 0.0, math.sin(0.3), 0.0)  # 0.6 rad about Y
    out = quaternion_multiply(q, (1.0, 0.0, 0.0, 0.0))
    assert np.allclose(out, q, atol=1e-12)


def test_quaternion_multiply_compose_two_z_rotations():
    """Z-rotation by π/4 ⊗ Z-rotation by π/4 == Z-rotation by π/2."""
    q_quarter = axis_angle_to_quat((0.0, 0.0, 1.0), math.pi / 4)
    q_half = quaternion_multiply(q_quarter, q_quarter)
    expected = axis_angle_to_quat((0.0, 0.0, 1.0), math.pi / 2)
    assert np.allclose(q_half, expected, atol=1e-12)


def test_quaternion_slerp_endpoints():
    """slerp(q1, q2, 0) == q1; slerp(q1, q2, 1) == q2."""
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = axis_angle_to_quat((1.0, 0.0, 0.0), math.pi / 3)
    out0 = quaternion_slerp(q1, q2, 0.0)
    out1 = quaternion_slerp(q1, q2, 1.0)
    assert np.allclose(out0, q1, atol=1e-9)
    assert np.allclose(out1, q2, atol=1e-9)


def test_quaternion_slerp_takes_short_arc():
    """slerp must take the shorter arc — even if q2 has flipped sign."""
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2_flipped = (-1.0, 0.0, 0.0, 0.0)  # same orientation, flipped sign
    out = quaternion_slerp(q1, q2_flipped, 0.5)
    assert abs(abs(out[0]) - 1.0) < 1e-9, f"got {out}"
    assert abs(out[1]) < 1e-9
    assert abs(out[2]) < 1e-9
    assert abs(out[3]) < 1e-9


# ============================================================================
# Axis-angle / vector rotation
# ============================================================================

def test_axis_angle_to_quat_identity():
    q = axis_angle_to_quat((1.0, 0.0, 0.0), 0.0)
    assert abs(q[0] - 1.0) < 1e-9
    assert abs(q[1]) < 1e-9
    assert abs(q[2]) < 1e-9
    assert abs(q[3]) < 1e-9


def test_axis_angle_to_quat_x90():
    q = axis_angle_to_quat((1.0, 0.0, 0.0), math.pi / 2)
    expected_w = math.cos(math.pi / 4)
    expected_x = math.sin(math.pi / 4)
    assert abs(q[0] - expected_w) < 1e-9
    assert abs(q[1] - expected_x) < 1e-9
    assert abs(q[2]) < 1e-9
    assert abs(q[3]) < 1e-9


def test_rotate_vec_by_quat_identity():
    v = (0.5, 0.3, -0.1)
    q = (1.0, 0.0, 0.0, 0.0)
    rotated = rotate_vec_by_quat(v, q)
    assert np.allclose(rotated, v, atol=1e-9)


def test_rotate_vec_by_quat_z90():
    """R_z(90°) on (1, 0, 0) → (0, 1, 0)."""
    v = (1.0, 0.0, 0.0)
    q = axis_angle_to_quat((0.0, 0.0, 1.0), math.pi / 2)
    rotated = rotate_vec_by_quat(v, q)
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9), f"got {rotated}"


# ============================================================================
# Min-jerk shape function
# ============================================================================

def test_minjerk_s_endpoints_and_monotonic():
    assert minjerk_s(0.0) == 0.0
    assert abs(minjerk_s(1.0) - 1.0) < 1e-9
    assert minjerk_s(0.5) == 0.5  # symmetric
    last = -1.0
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        v = minjerk_s(t)
        assert v >= last, f"non-monotonic at t={t}: {v} < {last}"
        last = v


def test_minjerk_s_clamps_outside_unit_interval():
    assert minjerk_s(-0.5) == 0.0
    assert minjerk_s(2.0) == 1.0


def test_minjerk_s_zero_derivative_at_endpoints():
    """s'(0) = s'(1) = 0 — what makes phase transitions C¹-smooth."""
    h = 1e-4
    deriv_at_0 = (minjerk_s(h) - minjerk_s(0.0)) / h
    deriv_at_1 = (minjerk_s(1.0) - minjerk_s(1.0 - h)) / h
    # Polynomial t³(10-15t+6t²) has derivative 30t²(1-t)², exactly zero at
    # t=0 and t=1. Finite-difference has O(h) error.
    assert abs(deriv_at_0) < 1e-6, f"derivative at 0 = {deriv_at_0}"
    assert abs(deriv_at_1) < 1e-6, f"derivative at 1 = {deriv_at_1}"


# ============================================================================
# PoseSnapshot + lerp_pose
# ============================================================================

def test_lerp_pose_endpoints():
    a = PoseSnapshot(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    b = PoseSnapshot(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    out_start = lerp_pose(a, b, 0.0)
    assert abs(out_start.px) < 1e-9
    out_end = lerp_pose(a, b, 1.0)
    assert abs(out_end.px - 1.0) < 1e-9
    assert abs(out_end.py - 2.0) < 1e-9
    assert abs(out_end.pz - 3.0) < 1e-9


def test_lerp_pose_position_lerps_linearly():
    """Position is linear in s; caller applies minjerk_s explicitly when
    smooth phases are wanted."""
    a = PoseSnapshot(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    b = PoseSnapshot(10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    out_mid = lerp_pose(a, b, 0.5)
    assert abs(out_mid.px - 5.0) < 1e-9


def test_lerp_pose_clamps_s_outside_unit_interval():
    a = PoseSnapshot(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    b = PoseSnapshot(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    out_neg = lerp_pose(a, b, -1.0)
    assert abs(out_neg.px) < 1e-9
    out_big = lerp_pose(a, b, 5.0)
    assert abs(out_big.px - 1.0) < 1e-9


def test_lerp_pose_orientation_slerps():
    """At s=0.5, orientation should be a 45° Z rotation when end is 90°."""
    a = PoseSnapshot(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    b_q = axis_angle_to_quat((0.0, 0.0, 1.0), math.pi / 2)
    b = PoseSnapshot(0.0, 0.0, 0.0, b_q[0], b_q[1], b_q[2], b_q[3])
    out_mid = lerp_pose(a, b, 0.5)
    expected = axis_angle_to_quat((0.0, 0.0, 1.0), math.pi / 4)
    assert abs(out_mid.qw - expected[0]) < 1e-9
    assert abs(out_mid.qz - expected[3]) < 1e-9


# ============================================================================
# PoseSnapshot adapter from ROS-style messages
# ============================================================================

def test_pose_snapshot_from_ros_transform_duck_typed():
    """from_ros_transform reads .translation and .rotation; works with any
    duck-typed object."""
    class _Vec3:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Quat:
        def __init__(self, w, x, y, z):
            self.w, self.x, self.y, self.z = w, x, y, z

    class _Transform:
        def __init__(self):
            self.translation = _Vec3(0.5, -0.2, 1.14)
            self.rotation = _Quat(0.7, 0.1, 0.2, 0.3)

    snap = PoseSnapshot.from_ros_transform(_Transform())
    assert snap.px == 0.5 and snap.py == -0.2 and snap.pz == 1.14
    assert snap.qw == 0.7 and snap.qx == 0.1 and snap.qy == 0.2 and snap.qz == 0.3


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_quaternion_multiply_identity_left,
        test_quaternion_multiply_identity_right,
        test_quaternion_multiply_compose_two_z_rotations,
        test_quaternion_slerp_endpoints,
        test_quaternion_slerp_takes_short_arc,
        test_axis_angle_to_quat_identity,
        test_axis_angle_to_quat_x90,
        test_rotate_vec_by_quat_identity,
        test_rotate_vec_by_quat_z90,
        test_minjerk_s_endpoints_and_monotonic,
        test_minjerk_s_clamps_outside_unit_interval,
        test_minjerk_s_zero_derivative_at_endpoints,
        test_lerp_pose_endpoints,
        test_lerp_pose_position_lerps_linearly,
        test_lerp_pose_clamps_s_outside_unit_interval,
        test_lerp_pose_orientation_slerps,
        test_pose_snapshot_from_ros_transform_duck_typed,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as ex:
            failures += 1
            import traceback
            print(f"FAIL  {t.__name__}: {type(ex).__name__}: {ex}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(0 if failures == 0 else 1)
