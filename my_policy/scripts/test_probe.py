#!/usr/bin/env python3
"""Host-runnable tests for my_policy.probe.

Tests pure helpers (compute_tilt_axis, axis_angle_to_quat, rotate_vec_by_quat,
decide_probe_direction, minjerk_s, lerp_pose) and the TiltProbeStateMachine
end-to-end with synthetic force inputs.

Runs WITHOUT pixi env: numpy + transforms3d are the only runtime deps; both
ship with the dev container's python.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# Add my_policy package to path so we can import from it without ROS.
sys.path.insert(
    0, "/home/robin/ssd/aic_workspace/aic_code_robin/aic_main/my_policy"
)

from my_policy.probe import (  # noqa: E402
    PoseSnapshot,
    ProbeConfig,
    TiltProbeStateMachine,
    axis_angle_to_quat,
    compute_tilt_axis,
    decide_probe_direction,
    lerp_pose,
    minjerk_s,
    rotate_vec_by_quat,
)


# ============================================================================
# Pure helpers
# ============================================================================

def test_compute_tilt_axis_for_plug_straight_down_x_dir():
    """Plug at (0, 0, -L), direction (+X). Expected tilt axis = (0, -1, 0)
    (rotating about -Y by positive angle moves plug tip in +X)."""
    v = np.array([0.0, 0.0, -0.05])  # plug 5 cm below gripper
    d = np.array([1.0, 0.0, 0.0])    # want +X arc
    omega = compute_tilt_axis(v, d)
    assert abs(omega[0] - 0.0) < 1e-9, f"expected ω=(0,-1,0), got {omega}"
    assert abs(omega[1] - (-1.0)) < 1e-9, f"expected ω=(0,-1,0), got {omega}"
    assert abs(omega[2] - 0.0) < 1e-9, f"expected ω=(0,-1,0), got {omega}"


def test_compute_tilt_axis_verifies_plug_arcs_in_direction():
    """Apply the computed tilt and check the plug tip moved in the requested
    direction. Cross-check between the math and the rotation it commands."""
    v = np.array([0.0, 0.0, -0.05])
    for angle_target in [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]:
        d = np.array([math.cos(angle_target), math.sin(angle_target), 0.0])
        omega = compute_tilt_axis(v, d)
        # Apply small tilt about omega and check direction
        tilt_rad = math.radians(2.0)
        q_tilt = axis_angle_to_quat(omega, tilt_rad)
        v_new = rotate_vec_by_quat(v, q_tilt)
        delta = v_new - v
        # Delta should be in the (d) direction in the XY plane.
        delta_xy = delta[:2]
        delta_xy_norm = float(np.linalg.norm(delta_xy))
        assert delta_xy_norm > 1e-5, (
            f"angle={angle_target}: tilt produced no XY motion ({delta_xy})"
        )
        delta_xy_unit = delta_xy / delta_xy_norm
        d_xy = d[:2]
        cosang = float(np.dot(delta_xy_unit, d_xy))
        assert cosang > 0.99, (
            f"angle={angle_target}: plug tip moved in direction {delta_xy_unit} "
            f"but expected {d_xy} (cos={cosang:.4f})"
        )


def test_compute_tilt_axis_degenerate_returns_z():
    """v parallel to d → fallback to z-axis (won't crash)."""
    v = np.array([1.0, 0.0, 0.0])
    d = np.array([1.0, 0.0, 0.0])
    omega = compute_tilt_axis(v, d)
    assert np.allclose(omega, [0.0, 0.0, 1.0])


def test_axis_angle_to_quat_identity():
    q = axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), 0.0)
    assert abs(q[0] - 1.0) < 1e-9
    assert abs(q[1]) < 1e-9
    assert abs(q[2]) < 1e-9
    assert abs(q[3]) < 1e-9


def test_axis_angle_to_quat_x90():
    q = axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), math.pi / 2)
    expected_w = math.cos(math.pi / 4)
    expected_x = math.sin(math.pi / 4)
    assert abs(q[0] - expected_w) < 1e-9
    assert abs(q[1] - expected_x) < 1e-9
    assert abs(q[2]) < 1e-9
    assert abs(q[3]) < 1e-9


def test_rotate_vec_by_quat_identity():
    v = np.array([0.5, 0.3, -0.1])
    q = (1.0, 0.0, 0.0, 0.0)
    rotated = rotate_vec_by_quat(v, q)
    assert np.allclose(rotated, v, atol=1e-9)


def test_rotate_vec_by_quat_z90():
    """R_z(90°) on (1, 0, 0) → (0, 1, 0)."""
    v = np.array([1.0, 0.0, 0.0])
    q = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), math.pi / 2)
    rotated = rotate_vec_by_quat(v, q)
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9), f"got {rotated}"


def test_decide_probe_direction_picks_minimum_force():
    """Synthetic samples with min force at angle = 3 × 2π/8 = 3π/4.
    Expect chosen angle near 3π/4."""
    n = 8
    samples = []
    for i in range(n):
        angle = i * 2.0 * math.pi / n
        # Force minimum at i=3, increases with circular distance from i=3
        circ_dist = min((i - 3) % n, (3 - i) % n)
        force = 5.0 + circ_dist * 1.0  # 5N at i=3, 6N at i=2/4, ..., 9N at i=7
        samples.append((angle, force))
    chosen, df = decide_probe_direction(samples, temperature_n=1.0)
    expected = 3 * 2.0 * math.pi / n  # 3π/4 ≈ 2.36
    # With temperature=1 and ΔF up to 4N, the weighting gives ~e^4≈55x weight to min.
    # Should be very close to 3π/4.
    angle_err = abs((chosen - expected + math.pi) % (2.0 * math.pi) - math.pi)
    assert angle_err < math.radians(15), (
        f"chosen {math.degrees(chosen):.1f}° but expected "
        f"{math.degrees(expected):.1f}° (err {math.degrees(angle_err):.1f}°)"
    )
    assert df == 4.0, f"expected force_range=4.0, got {df}"


def test_decide_probe_direction_ambiguous_uniform_force():
    """All forces equal → force_range=0 → caller should treat as ambiguous."""
    n = 8
    samples = [(i * 2.0 * math.pi / n, 5.0) for i in range(n)]
    chosen, df = decide_probe_direction(samples)
    assert df == 0.0
    # Chosen angle could be anything; we just don't crash.


def test_minjerk_s_endpoints_and_monotonic():
    assert minjerk_s(0.0) == 0.0
    assert abs(minjerk_s(1.0) - 1.0) < 1e-9
    assert minjerk_s(0.5) == 0.5  # symmetric
    # Monotone increasing
    last = -1.0
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        v = minjerk_s(t)
        assert v >= last, f"non-monotonic at t={t}: {v} < {last}"
        last = v


def test_minjerk_s_clamps_outside_unit_interval():
    assert minjerk_s(-0.5) == 0.0
    assert minjerk_s(2.0) == 1.0


def test_lerp_pose_endpoints():
    a = PoseSnapshot(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    b = PoseSnapshot(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    out_start = lerp_pose(a, b, 0.0)
    assert abs(out_start.px) < 1e-9
    out_end = lerp_pose(a, b, 1.0)
    assert abs(out_end.px - 1.0) < 1e-9
    assert abs(out_end.py - 2.0) < 1e-9
    assert abs(out_end.pz - 3.0) < 1e-9


# ============================================================================
# State machine
# ============================================================================

def _entry_setup():
    """Standard probe-entry conditions: gripper at (0, 0, 0.5), plug 5 cm
    below at (0, 0, 0.45), identity quaternion (plug aligned with base)."""
    cfg = ProbeConfig(
        tilt_deg=5.0, n_directions=8, sample_s=0.10, settle_s=0.10,
        translate_m=0.0015, translate_duration_s=0.20,
        ambiguous_df_n=1.0, retry_tilt_deg=[5.0, 8.0, 12.0],
    )
    sm = TiltProbeStateMachine(cfg)
    entry_pose = PoseSnapshot(0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    plug_pos = np.array([0.0, 0.0, 0.45])
    plug_q = (1.0, 0.0, 0.0, 0.0)
    return cfg, sm, entry_pose, plug_pos, plug_q


def _drive(sm, force_fn, max_s=10.0, dt_s=0.020):
    """Drive the state machine through completion. force_fn(t_s, sm) -> force_mag.

    Returns the list of (t_s, state, pose) tuples observed."""
    history = []
    t_ns = 0
    dt_ns = int(dt_s * 1e9)
    while sm.is_active() and t_ns / 1e9 < max_s:
        t_s = t_ns / 1e9
        f = force_fn(t_s, sm)
        pose = sm.step(t_ns, f)
        history.append((t_s, sm.state, pose))
        t_ns += dt_ns
    return history


def test_probe_settle_holds_entry_pose():
    """During SETTLE the commanded pose must equal the entry pose."""
    cfg, sm, entry, plug_pos, plug_q = _entry_setup()
    sm.enter(0, entry, plug_pos, plug_q)
    pose = sm.step(int(0.05 * 1e9), force_mag=10.0)
    assert sm.state == sm.STATE_SETTLE
    assert abs(pose.px - entry.px) < 1e-9
    assert abs(pose.py - entry.py) < 1e-9
    assert abs(pose.pz - entry.pz) < 1e-9
    assert abs(pose.qw - entry.qw) < 1e-9


def test_probe_advances_through_states():
    """Run with constant low force; should: SETTLE → SWEEP → exhausted_retries
    (uniform force is ambiguous → exhaust retries → IDLE)."""
    cfg, sm, entry, plug_pos, plug_q = _entry_setup()
    sm.enter(0, entry, plug_pos, plug_q)
    history = _drive(sm, force_fn=lambda t, _sm: 10.0, max_s=20.0)
    states_seen = {st for _, st, _ in history}
    assert sm.STATE_SETTLE in states_seen
    assert sm.STATE_SWEEP in states_seen
    assert sm.last_run_outcome == "exhausted_retries", (
        f"expected exhausted_retries, got {sm.last_run_outcome}"
    )
    assert sm.last_run_retry_count == 2, (
        f"expected to retry until index 2 (last entry of [5,8,12]), got "
        f"retry_count={sm.last_run_retry_count}"
    )


def test_probe_recognizes_low_force_direction_and_translates():
    """Force minimum at angle 3 (135°). After sweep, expect TRANSLATE state
    and final XY position offset toward 135° in plug-local frame.

    Plug q is identity, so plug-local 135° = base 135° = (-√2/2, +√2/2, 0).
    """
    cfg, sm, entry, plug_pos, plug_q = _entry_setup()
    sm.enter(0, entry, plug_pos, plug_q)

    def force_fn(_t, sm_):
        # Synthesize force based on current sweep direction.
        if sm_.state != sm_.STATE_SWEEP:
            return 5.0
        i = sm_._current_dir_i
        # Min at i=3: force=2N; others increase with circular distance.
        n = sm_.cfg.n_directions
        circ_dist = min((i - 3) % n, (3 - i) % n)
        return 2.0 + circ_dist * 2.0  # 2 → 4 → 6 → 8 → 10

    history = _drive(sm, force_fn=force_fn, max_s=10.0)
    states_seen = {st for _, st, _ in history}
    assert sm.STATE_TRANSLATE in states_seen, (
        f"expected to reach TRANSLATE state, observed: {states_seen}"
    )
    # Last pose is the final translate target
    final_pose = history[-1][2]
    chosen_deg = math.degrees(sm.last_run_chosen_angle or 0.0)
    # Expected translate direction is 135° in plug-local frame, which = base-frame for identity plug_q
    # Final XY offset should be in the chosen direction
    dx = final_pose.px - entry.px
    dy = final_pose.py - entry.py
    final_offset_mag = math.hypot(dx, dy)
    expected_mag = cfg.translate_m
    assert abs(final_offset_mag - expected_mag) / expected_mag < 0.01, (
        f"expected translate magnitude {expected_mag * 1000:.2f}mm, "
        f"got {final_offset_mag * 1000:.2f}mm"
    )
    final_angle = math.atan2(dy, dx)
    expected_angle = 3 * 2 * math.pi / 8  # 135°
    angle_err = abs((final_angle - expected_angle + math.pi) % (2 * math.pi) - math.pi)
    assert angle_err < math.radians(15), (
        f"final translate angle {math.degrees(final_angle):.1f}° but probe chose "
        f"{chosen_deg:.1f}°, expected ~{math.degrees(expected_angle):.1f}°"
    )
    assert sm.last_run_outcome == "translated"


def test_probe_widens_tilt_on_ambiguous_signal():
    """Force is ALMOST uniform — small variation < ambiguous_df_n. Expect retry
    to widen tilt to next entry in retry_tilt_deg, eventually exhausting."""
    cfg, sm, entry, plug_pos, plug_q = _entry_setup()
    sm.enter(0, entry, plug_pos, plug_q)
    # Force varies by 0.5N → < ambiguous_df_n=1.0
    history = _drive(sm, force_fn=lambda t, _sm: 5.0 + 0.5 * math.sin(t * 7.0), max_s=15.0)
    assert sm.last_run_outcome == "exhausted_retries"
    assert sm.last_run_retry_count >= 2  # tried at least the third retry magnitude


def test_probe_does_not_increase_z():
    """Across the entire probe (settle + sweep + translate), commanded gripper
    z must equal entry z to within float precision."""
    cfg, sm, entry, plug_pos, plug_q = _entry_setup()
    sm.enter(0, entry, plug_pos, plug_q)

    def force_fn(_t, sm_):
        if sm_.state != sm_.STATE_SWEEP:
            return 5.0
        # Strong gradient so probe finds a direction quickly.
        i = sm_._current_dir_i
        return 2.0 + i * 1.0

    history = _drive(sm, force_fn=force_fn, max_s=10.0)
    z_max = max(p.pz for _, _, p in history)
    z_min = min(p.pz for _, _, p in history)
    assert abs(z_max - entry.pz) < 1e-9, f"max z {z_max} > entry z {entry.pz}"
    assert abs(z_min - entry.pz) < 1e-9, f"min z {z_min} != entry z {entry.pz}"


def test_probe_tilt_axis_is_plug_local():
    """With non-identity plug orientation, the probe's tilt direction must
    follow the plug's frame, not the base frame.

    Setup: plug rotated 90° about base z. A plug-local +X probe direction
    should now arc the plug tip in base +Y."""
    cfg = ProbeConfig(
        tilt_deg=5.0, n_directions=8, sample_s=0.10, settle_s=0.10,
        translate_m=0.0015, translate_duration_s=0.20,
        ambiguous_df_n=1.0, retry_tilt_deg=[5.0, 8.0, 12.0],
    )
    sm = TiltProbeStateMachine(cfg)
    entry_pose = PoseSnapshot(0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
    plug_pos = np.array([0.0, 0.0, 0.45])
    # Plug rotated 90° about Z: q = (cos45, 0, 0, sin45)
    plug_q = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
    sm.enter(0, entry_pose, plug_pos, plug_q)
    # Force minimum at i=0 (plug-local +X)
    history = _drive(sm, force_fn=lambda t, sm_: (
        2.0 if sm_.state == sm_.STATE_SWEEP and sm_._current_dir_i == 0
        else 5.0 + sm_._current_dir_i * 0.5
    ), max_s=10.0)
    final_pose = history[-1][2]
    dx = final_pose.px - entry_pose.px
    dy = final_pose.py - entry_pose.py
    # plug-local +X rotated 90° about z = base +Y. So dx ≈ 0, dy ≈ +translate_m.
    assert abs(dx) < cfg.translate_m * 0.2, (
        f"expected translate along base +Y, but base +X component is {dx * 1000:.2f}mm"
    )
    assert dy > cfg.translate_m * 0.8, (
        f"expected translate along base +Y, got dy={dy * 1000:.2f}mm "
        f"(translate_m={cfg.translate_m * 1000:.2f}mm)"
    )


def test_probe_translate_target_magnitude():
    """Translate phase moves XY by exactly translate_m in identified direction
    (no drift due to slerp / lerp interpolation)."""
    cfg, sm, entry, plug_pos, plug_q = _entry_setup()
    sm.enter(0, entry, plug_pos, plug_q)

    def force_fn(_t, sm_):
        if sm_.state != sm_.STATE_SWEEP:
            return 5.0
        i = sm_._current_dir_i
        return 2.0 if i == 0 else 6.0  # min at 0 (+X)

    history = _drive(sm, force_fn=force_fn, max_s=10.0)
    final_pose = history[-1][2]
    mag = math.hypot(final_pose.px - entry.px, final_pose.py - entry.py)
    assert abs(mag - cfg.translate_m) < 1e-6, (
        f"final magnitude {mag * 1000:.4f}mm vs expected {cfg.translate_m * 1000:.4f}mm"
    )


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    tests = [
        # Pure helpers
        test_compute_tilt_axis_for_plug_straight_down_x_dir,
        test_compute_tilt_axis_verifies_plug_arcs_in_direction,
        test_compute_tilt_axis_degenerate_returns_z,
        test_axis_angle_to_quat_identity,
        test_axis_angle_to_quat_x90,
        test_rotate_vec_by_quat_identity,
        test_rotate_vec_by_quat_z90,
        test_decide_probe_direction_picks_minimum_force,
        test_decide_probe_direction_ambiguous_uniform_force,
        test_minjerk_s_endpoints_and_monotonic,
        test_minjerk_s_clamps_outside_unit_interval,
        test_lerp_pose_endpoints,
        # State machine
        test_probe_settle_holds_entry_pose,
        test_probe_advances_through_states,
        test_probe_recognizes_low_force_direction_and_translates,
        test_probe_widens_tilt_on_ambiguous_signal,
        test_probe_does_not_increase_z,
        test_probe_tilt_axis_is_plug_local,
        test_probe_translate_target_magnitude,
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
