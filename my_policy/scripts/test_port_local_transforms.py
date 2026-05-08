#!/usr/bin/env python3
"""Tier 1 — pure-math tests for `my_policy.port_local.transforms`.

Host-runnable. Requires only numpy. No torch, no lerobot, no dataset access.

Tests are designed so a wrong sign / order / convention fails LOUDLY (residual
of cm-rad scale), not silently. The localizer pipeline got hit by silent
label corruption (memory `feedback_audit_labels_first.md`) — these tests are
the safeguard that prevents the same trap here.

Usage:
    python3 my_policy/scripts/test_port_local_transforms.py

Tests, ranked by what they catch:

  Test 0   — SE(3) primitives self-consistency:
                quat→R→quat round-trip, R→quat→R round-trip, identity, det=1.
                Catches: numerical stability bugs in our SE(3) math.

  Test 0b  — Cross-check against localizer/projection.py:
                Same input → same output for `quat_xyzw_to_rotmat` and
                `se3_inverse`. Establishes that if these tests pass AND
                our code matches the localizer's, the localizer's primitives
                are also trustworthy (or both fail together — visible).
                Catches: convention drift between modules.

  Test 1   — Round-trip identity (the killer test):
                pose_baselink → port_local → baselink. Residual must be
                < 1e-9 m position / < 1e-9 rad rotation. A bug anywhere
                in the transform stack produces residuals at cm-rad scale.

  Test 2   — Invariance under board rotation:
                Synthesize TCP+port+action; rotate the entire scene by
                arbitrary R; re-run transform. Port-local outputs must
                be bitwise unchanged (within float64 noise ~1e-12).
                Catches: rotating two components but not the third.

  Test 3   — Velocity rotation correctness:
                yaw=π/2 port, linear=(1,0,0) base_link → linear=(0,-1,0)
                port. Specific named-axis check — catches off-by-90°
                errors that random tests can mask.

  Test 4   — Wrench rotation correctness:
                Same shape as Test 3 but for force/torque. Critically
                tests the chain `R_port^T @ R_tcp` (the wrench rotation
                composition).

  Test 5   — Identity edge case:
                When `T_port_in_bl = identity`, port-local == baselink.
                Catches: "I forgot to multiply by inverse" bugs.

  Test 6   — Magnitude invariance under rotation:
                After rotation, linear+angular velocity magnitudes and
                force+torque magnitudes are preserved within float64
                precision. A pure rotation cannot change a Euclidean
                norm; if it does, the rotation matrix isn't proper.

  Test 7   — Action target consistency:
                Action transforms identically to TCP pose (both are
                `T_X_in_baselink`-style). If we transform a synthetic
                "action = current TCP" frame, port-local action equals
                port-local TCP. Catches: divergent transform logic between
                pose and action paths.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Make `my_policy.*` importable when run from anywhere inside the repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "my_policy"))

from my_policy.port_local.transforms import (
    FrameInputs,
    quat_xyzw_to_rotmat,
    rotmat_to_quat_xyzw,
    make_se3,
    se3_inverse,
    split_se3,
    transform_frame,
    transform_pose_back_to_baselink,
)


# ---------------------------------------------------------------------------
# Helpers — used across tests, also pure numpy.
# ---------------------------------------------------------------------------


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit quaternion (xyzw). Marsaglia's method via normal."""
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def _random_pose7(rng: np.random.Generator, max_xyz: float = 0.5) -> np.ndarray:
    return np.concatenate(
        [rng.uniform(-max_xyz, max_xyz, 3), _random_unit_quat(rng)]
    )


def _quat_residual(q1: np.ndarray, q2: np.ndarray) -> float:
    """Sign-invariant residual: 0 iff q1 and q2 represent the same rotation.

    Returns `1 - |q1·q2|`. This is numerically stable near identity, unlike
    `2 * arccos(|q1·q2|)` which has a vertical asymptote at d=1: a 1e-15
    error in d (float64 ulp) leaks out as a 4e-8 rad "angle", masking the
    actual precision.

    For small mismatches: residual ≈ (angle/2)² / 2, so residual=1e-14
    corresponds to ~1.4e-7 rad. Most "exact" tests should hit residual
    in the 1e-15 to 1e-12 range; pick a tolerance accordingly.
    """
    d = abs(float(np.dot(q1, q2)))
    return 1.0 - min(1.0, d)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_0_se3_primitives_self_consistency():
    """SE(3) primitives produce mutually-inverse mappings."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = _random_unit_quat(rng)
        R = quat_xyzw_to_rotmat(q)
        # R is proper orthonormal.
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12), "R is not orthonormal"
        assert abs(np.linalg.det(R) - 1.0) < 1e-12, f"det(R) = {np.linalg.det(R)}"
        # quat → R → quat round-trips (sign-invariant).
        q_back = rotmat_to_quat_xyzw(R)
        assert _quat_residual(q, q_back) < 1e-14, (
            f"quat round-trip residual: {_quat_residual(q, q_back):.2e}"
        )
        # SE(3) inverse is left + right inverse.
        T = make_se3(rng.uniform(-1, 1, 3), q)
        Tinv = se3_inverse(T)
        assert np.allclose(T @ Tinv, np.eye(4), atol=1e-12), "T @ Tinv != I"
        assert np.allclose(Tinv @ T, np.eye(4), atol=1e-12), "Tinv @ T != I"
    # Identity edge case.
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    R_id = quat_xyzw_to_rotmat(q_id)
    assert np.allclose(R_id, np.eye(3), atol=1e-15)


def test_0b_cross_check_against_localizer_projection():
    """Our SE(3) primitives produce the same numerical output as
    `localizer/projection.py`. If both this test and Test 0 pass, the
    localizer's primitives are also trustworthy. If this test fails, one
    of the two is wrong (look at Test 0 to decide which).
    """
    from my_policy.localizer.projection import (
        _quat_xyzw_to_rotmat as loc_quat,
        _se3_from_xyz_quat as loc_se3,
        _se3_inverse as loc_inv,
    )

    rng = np.random.default_rng(7)
    for _ in range(50):
        xyz = rng.uniform(-1, 1, 3)
        q = _random_unit_quat(rng)

        # Note: localizer's quat fn doesn't normalize; ours does. So pass
        # already-unit quats to keep the comparison fair.
        R_ours = quat_xyzw_to_rotmat(q)
        R_loc = loc_quat(np.array(q))
        assert np.allclose(R_ours, R_loc, atol=1e-12), (
            f"R mismatch: max_abs={np.max(np.abs(R_ours - R_loc))}"
        )

        T_ours = make_se3(xyz, q)
        T_loc = loc_se3(np.array(xyz), np.array(q))
        assert np.allclose(T_ours, T_loc, atol=1e-12), "T mismatch"

        Tinv_ours = se3_inverse(T_ours)
        Tinv_loc = loc_inv(T_loc)
        assert np.allclose(Tinv_ours, Tinv_loc, atol=1e-12), "Tinv mismatch"


def test_1_round_trip_identity():
    """Killer test — pose_baselink → port_local → baselink reproduces
    original within numerical precision. A bug anywhere in the transform
    fails this loudly.
    """
    rng = np.random.default_rng(1)
    max_pos_err = 0.0
    max_rot_err_rad = 0.0
    for _ in range(500):
        port = _random_pose7(rng, max_xyz=2.0)
        tcp = _random_pose7(rng, max_xyz=2.0)
        action = _random_pose7(rng, max_xyz=2.0)
        velocity = rng.uniform(-1, 1, 6)
        wrench = rng.uniform(-50, 50, 6)
        inp = FrameInputs(
            tcp_pose_baselink=tcp,
            tcp_velocity_baselink=velocity,
            wrench_sensorframe=wrench,
            action_baselink=action,
            port_pose_baselink=port,
        )
        out = transform_frame(inp)
        # Round-trip TCP and action.
        tcp_back = transform_pose_back_to_baselink(out.tcp_pose_portframe, port)
        action_back = transform_pose_back_to_baselink(out.action_portframe, port)
        for original, back in [(tcp, tcp_back), (action, action_back)]:
            pos_err = float(np.linalg.norm(original[:3] - back[:3]))
            rot_residual = _quat_residual(original[3:7], back[3:7])
            max_pos_err = max(max_pos_err, pos_err)
            max_rot_err_rad = max(max_rot_err_rad, rot_residual)
            assert pos_err < 1e-9, f"position drift: {pos_err} m"
            assert rot_residual < 1e-12, f"rotation residual: {rot_residual:.2e}"
    print(f"  max position drift: {max_pos_err:.2e} m")
    print(f"  max rotation residual: {max_rot_err_rad:.2e}")


def test_2_invariance_under_world_rotation():
    """Rotate the entire scene (TCP + port + action + velocity + wrench
    transforms together). Port-local outputs must not change.

    This catches: rotating two components but not the third, mismatched
    quaternion conventions across components, sign errors that cancel
    in round-trip but show up under rotation.
    """
    rng = np.random.default_rng(2)
    for _ in range(50):
        # Original scene.
        port_xyz = rng.uniform(-1, 1, 3)
        port_quat = _random_unit_quat(rng)
        tcp_xyz = rng.uniform(-1, 1, 3)
        tcp_quat = _random_unit_quat(rng)
        action_xyz = rng.uniform(-1, 1, 3)
        action_quat = _random_unit_quat(rng)
        velocity = rng.uniform(-1, 1, 6)
        wrench = rng.uniform(-50, 50, 6)

        inp_a = FrameInputs(
            tcp_pose_baselink=np.concatenate([tcp_xyz, tcp_quat]),
            tcp_velocity_baselink=velocity,
            wrench_sensorframe=wrench,
            action_baselink=np.concatenate([action_xyz, action_quat]),
            port_pose_baselink=np.concatenate([port_xyz, port_quat]),
        )
        out_a = transform_frame(inp_a)

        # Now rotate the entire world by R_world. Every "_in_baselink"
        # quantity gets pre-multiplied: T_X_in_bl' = R_world @ T_X_in_bl.
        # The wrench is in SENSOR frame (not base_link), so it does NOT
        # rotate when the world rotates — it's a sensor-local reading.
        # Same for tcp_velocity in this test: tcp_velocity_baselink IS
        # in base_link, so it DOES rotate.
        R_world_q = _random_unit_quat(rng)
        R_world = quat_xyzw_to_rotmat(R_world_q)
        T_world_rotation = np.eye(4)
        T_world_rotation[:3, :3] = R_world

        def rotate_pose7(p7):
            T = make_se3(p7[:3], p7[3:7])
            T_rot = T_world_rotation @ T
            xyz, q = split_se3(T_rot)
            return np.concatenate([xyz, q])

        def rotate_twist6(t6):
            return np.concatenate([R_world @ t6[:3], R_world @ t6[3:6]])

        inp_b = FrameInputs(
            tcp_pose_baselink=rotate_pose7(inp_a.tcp_pose_baselink),
            tcp_velocity_baselink=rotate_twist6(velocity),
            wrench_sensorframe=wrench,  # sensor-frame, untouched
            action_baselink=rotate_pose7(inp_a.action_baselink),
            port_pose_baselink=rotate_pose7(inp_a.port_pose_baselink),
        )
        out_b = transform_frame(inp_b)

        # All port-local outputs must be bitwise identical (within float64).
        assert np.allclose(
            out_a.tcp_pose_portframe[:3], out_b.tcp_pose_portframe[:3], atol=1e-10
        ), "tcp position varies under world rotation"
        assert _quat_residual(
            out_a.tcp_pose_portframe[3:], out_b.tcp_pose_portframe[3:]
        ) < 1e-14, "tcp rotation varies under world rotation"
        assert np.allclose(
            out_a.action_portframe[:3], out_b.action_portframe[:3], atol=1e-10
        ), "action position varies under world rotation"
        assert _quat_residual(
            out_a.action_portframe[3:], out_b.action_portframe[3:]
        ) < 1e-14, "action rotation varies under world rotation"
        assert np.allclose(
            out_a.tcp_velocity_portframe, out_b.tcp_velocity_portframe, atol=1e-10
        ), "velocity varies under world rotation"
        # Wrench: TCP-in-port rotation is unchanged by world rotation
        # (both R_tcp_in_bl and R_port_in_bl rotated the same way → cancels
        # in the composition), so port-local wrench is also unchanged.
        assert np.allclose(
            out_a.wrench_portframe, out_b.wrench_portframe, atol=1e-10
        ), "wrench varies under world rotation"


def test_3_velocity_rotation_named_axes():
    """Specific check: port at yaw=+90° around z, TCP velocity = (1, 0, 0)
    in base_link → port-local velocity = (0, -1, 0).

    Reasoning: a +90° rotation around z maps base_link's x-axis to port's
    y-axis. So a velocity along base_link's x-axis points along port's
    +y-axis... wait, let me re-derive. R_port_in_bl rotates port-frame
    vectors to base_link-frame vectors. So R_port_in_bl @ (1,0,0)_port =
    base_link velocity along port's x-axis (in base_link coords). The
    inverse rotation R_port_in_bl.T maps base_link vectors to port-frame
    vectors. So velocity (1,0,0) in base_link expressed in port frame is
    R^T @ (1,0,0). With R = Rz(+90°) = [[0,-1,0],[1,0,0],[0,0,1]], R^T =
    [[0,1,0],[-1,0,0],[0,0,1]]. So R^T @ (1,0,0) = (0,-1,0). ✓
    """
    yaw = np.pi / 2
    cy, sy = np.cos(yaw), np.sin(yaw)
    # Quaternion for yaw rotation around z: (0, 0, sin(y/2), cos(y/2)).
    port_quat = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
    inp = FrameInputs(
        tcp_pose_baselink=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        tcp_velocity_baselink=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        wrench_sensorframe=np.zeros(6),
        action_baselink=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        port_pose_baselink=np.concatenate([np.zeros(3), port_quat]),
    )
    out = transform_frame(inp)
    expected_linear = np.array([0.0, -1.0, 0.0])
    assert np.allclose(out.tcp_velocity_portframe[:3], expected_linear, atol=1e-12), (
        f"got {out.tcp_velocity_portframe[:3]}, expected {expected_linear}"
    )


def test_4_wrench_rotation_named_axes():
    """Same shape as test 3 but for wrench. Critically tests the
    composition `R_port^T @ R_tcp`.

    Setup: TCP and port both at the origin, port at yaw=+90° around z,
    TCP at identity rotation. Sensor force = (1, 0, 0) — should appear
    as port-local force = (0, -1, 0), same as the velocity case (since
    R_tcp = I, the chain R_port^T @ R_tcp = R_port^T).
    """
    yaw = np.pi / 2
    port_quat = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
    inp = FrameInputs(
        tcp_pose_baselink=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        tcp_velocity_baselink=np.zeros(6),
        wrench_sensorframe=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        action_baselink=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        port_pose_baselink=np.concatenate([np.zeros(3), port_quat]),
    )
    out = transform_frame(inp)
    expected_force = np.array([0.0, -1.0, 0.0])
    assert np.allclose(out.wrench_portframe[:3], expected_force, atol=1e-12), (
        f"got {out.wrench_portframe[:3]}, expected {expected_force}"
    )

    # Also test: when TCP rotates +90° around z AND port rotates +90° around z,
    # the chain R_port^T @ R_tcp = identity → sensor wrench passes through.
    inp2 = FrameInputs(
        tcp_pose_baselink=np.concatenate([np.zeros(3), port_quat]),  # TCP also rotated
        tcp_velocity_baselink=np.zeros(6),
        wrench_sensorframe=np.array([1.0, 2.0, 3.0, 0.5, 0.6, 0.7]),
        action_baselink=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        port_pose_baselink=np.concatenate([np.zeros(3), port_quat]),
    )
    out2 = transform_frame(inp2)
    assert np.allclose(out2.wrench_portframe, inp2.wrench_sensorframe, atol=1e-12), (
        "wrench should pass through unchanged when TCP and port rotation are equal"
    )


def test_5_identity_edge_case():
    """When port pose is identity, port-local quantities equal base_link
    quantities."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        tcp = _random_pose7(rng, max_xyz=0.5)
        action = _random_pose7(rng, max_xyz=0.5)
        velocity = rng.uniform(-1, 1, 6)
        wrench = rng.uniform(-50, 50, 6)
        inp = FrameInputs(
            tcp_pose_baselink=tcp,
            tcp_velocity_baselink=velocity,
            wrench_sensorframe=wrench,
            action_baselink=action,
            port_pose_baselink=np.array([0, 0, 0, 0, 0, 0, 1.0]),
        )
        out = transform_frame(inp)
        # Position and velocity pass through.
        assert np.allclose(out.tcp_pose_portframe[:3], tcp[:3], atol=1e-12)
        assert _quat_residual(out.tcp_pose_portframe[3:], tcp[3:]) < 1e-14
        assert np.allclose(out.action_portframe[:3], action[:3], atol=1e-12)
        assert _quat_residual(out.action_portframe[3:], action[3:]) < 1e-14
        assert np.allclose(out.tcp_velocity_portframe, velocity, atol=1e-12)
        # Wrench passes through ONLY because R_port = I makes the chain
        # R_port^T @ R_tcp = R_tcp, NOT identity. So wrench will be
        # rotated by R_tcp_in_bl. Verify that's what we get.
        R_tcp = quat_xyzw_to_rotmat(tcp[3:7])
        expected_wrench = np.concatenate([R_tcp @ wrench[:3], R_tcp @ wrench[3:6]])
        assert np.allclose(out.wrench_portframe, expected_wrench, atol=1e-12), (
            "wrench rotation under port=identity should be R_tcp_in_bl @ wrench_sensor"
        )


def test_6_magnitude_invariance():
    """A pure rotation cannot change Euclidean norms. Catches: rotation
    matrices that aren't orthonormal."""
    rng = np.random.default_rng(6)
    for _ in range(100):
        port = _random_pose7(rng, max_xyz=1.0)
        tcp = _random_pose7(rng, max_xyz=1.0)
        velocity = rng.uniform(-10, 10, 6)
        wrench = rng.uniform(-100, 100, 6)
        inp = FrameInputs(
            tcp_pose_baselink=tcp,
            tcp_velocity_baselink=velocity,
            wrench_sensorframe=wrench,
            action_baselink=np.array([0, 0, 0, 0, 0, 0, 1.0]),
            port_pose_baselink=port,
        )
        out = transform_frame(inp)
        # Linear velocity magnitude preserved.
        assert abs(
            np.linalg.norm(out.tcp_velocity_portframe[:3])
            - np.linalg.norm(velocity[:3])
        ) < 1e-10
        # Angular velocity magnitude preserved.
        assert abs(
            np.linalg.norm(out.tcp_velocity_portframe[3:6])
            - np.linalg.norm(velocity[3:6])
        ) < 1e-10
        # Force magnitude preserved.
        assert abs(
            np.linalg.norm(out.wrench_portframe[:3])
            - np.linalg.norm(wrench[:3])
        ) < 1e-10
        # Torque magnitude preserved.
        assert abs(
            np.linalg.norm(out.wrench_portframe[3:6])
            - np.linalg.norm(wrench[3:6])
        ) < 1e-10


def test_7_action_eq_tcp_when_action_targets_current_tcp():
    """If action_baselink is set equal to tcp_baselink (a "stay put"
    command), then action_portframe must equal tcp_portframe. Catches:
    divergent transform logic between the pose and action paths.
    """
    rng = np.random.default_rng(7)
    for _ in range(20):
        port = _random_pose7(rng)
        tcp = _random_pose7(rng)
        inp = FrameInputs(
            tcp_pose_baselink=tcp,
            tcp_velocity_baselink=np.zeros(6),
            wrench_sensorframe=np.zeros(6),
            action_baselink=tcp.copy(),  # action = current TCP
            port_pose_baselink=port,
        )
        out = transform_frame(inp)
        assert np.allclose(
            out.tcp_pose_portframe[:3], out.action_portframe[:3], atol=1e-12
        )
        assert _quat_residual(
            out.tcp_pose_portframe[3:], out.action_portframe[3:]
        ) < 1e-14


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


TESTS = [
    ("0  SE(3) primitives self-consistency", test_0_se3_primitives_self_consistency),
    ("0b SE(3) cross-check vs localizer/projection.py",
     test_0b_cross_check_against_localizer_projection),
    ("1  Round-trip identity (killer test)", test_1_round_trip_identity),
    ("2  Invariance under world rotation", test_2_invariance_under_world_rotation),
    ("3  Velocity rotation named-axes", test_3_velocity_rotation_named_axes),
    ("4  Wrench rotation named-axes", test_4_wrench_rotation_named_axes),
    ("5  Identity edge case (port = I)", test_5_identity_edge_case),
    ("6  Magnitude invariance", test_6_magnitude_invariance),
    ("7  Action == TCP when action targets current TCP",
     test_7_action_eq_tcp_when_action_targets_current_tcp),
]


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            print(f"[ run] {name}")
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}")
            print(f"       {e}")
        except Exception as e:
            failed += 1
            print(f"[ERR ] {name}: {type(e).__name__}: {e}")
    print()
    if failed == 0:
        print(f"All {len(TESTS)} tests passed.")
        return 0
    print(f"{failed}/{len(TESTS)} tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
