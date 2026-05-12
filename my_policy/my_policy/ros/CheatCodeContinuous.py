"""CheatCodeContinuous — three-phase smooth oracle for IL training.

Trajectory factors into THREE sequential, smoothly-transitioning phases:

  Phase 1 — XY APPROACH: smoothly translate gripper to put plug above port
            (current orientation kept; min-jerk).
  Phase 2 — ORIENT: position frozen; slerp gripper orientation so plug aligns
            with port.
  Phase 3 — DESCEND: PI on XY + Z descent at 4 mm/s + force-feedback freeze
            (no commanded lift) + spiral search on chamfer contact (smoothly
            ramped) + LATCH smooth blend when plug confirmed inside port.

Properties (verified by test_cheatcode_continuous.py):
  * Z is monotonically non-increasing tick-over-tick (NO commanded lift).
  * Phase boundaries preserve pose continuity to ~1e-6 m / 1e-6 quat.
  * Spiral on/off is ramped over 5 ticks (250 ms) — no instantaneous ±2 mm jump.
  * LATCH is blended over 5 ticks — no instantaneous XY freeze.
  * Phase 1 is strictly XY-only — Z stays at the initial gripper Z. Phase 3
    owns the entire Z descent. Eliminates the 100 mm SC z-step that the prior
    CheatCodeRobust had at the end of APPROACH (it forced Z to a hover height
    before descent; we just don't touch Z until descent).

Hybrid recovery-scenario injection (default 30 % of trials get a deterministic
1–3 mm XY offset in plug-local frame, capped at 0.5 mm on the SC tight axis)
spreads the arrival-pose distribution so the IL model sees varied trajectories.

Training-oracle only. Uses ground_truth:=true TF and is NOT submission-safe.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, TransformStamped
from rclpy.duration import Duration
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from my_policy.trajectory import (
    PoseSnapshot,
    lerp_pose,
    minjerk_s,
    quaternion_multiply as tj_quat_mul,
    quaternion_slerp as tj_quat_slerp,
    rotate_vec_by_quat,
)


# ---------------------------------------------------------------------------
# Tunable parameters (all env-var overridable)
# ---------------------------------------------------------------------------

@dataclass
class ContinuousParams:
    """All env-overridable knobs in one place. Built once in __init__ from
    the policy's class constants + os.environ."""
    # Phase 1 / Phase 2 durations are COMPUTED dynamically per-trial based on
    # the actual displacement / angular distance, so the peak min-jerk velocity
    # stays bounded for smoothness regardless of where the port is. Longer
    # distance → longer phase. Caps below set the peak velocity ceiling.
    max_linear_velocity_m_s: float = 0.025     # 25 mm/s peak Phase 1
    max_angular_velocity_rad_s: float = 0.5    # ~28°/s peak Phase 2
    min_phase_duration_s: float = 1.0          # floor — even tiny moves take this long
    # Descent kinematics — constant rate, with a min-jerk ramp-in at Phase 3 start.
    descent_rate_m_s: float = 0.010    # 10 mm/s steady-state descent
    descent_ramp_s: float = 1.0        # ramp descent rate 0 → full over 1 s using min-jerk
    tick_period_s: float = 0.05
    insert_z_offset: float = -0.015
    # Safety floor for the plug's Z above port_z before any lateral motion.
    # Phase 0 (RAISE) fires only if the initial plug Z is below this — moves
    # straight up to clear the NIC card / SFP port / SC port body before
    # Phase 1's XY approach engages. 5 cm covers SFP port height (~46 mm)
    # plus a small buffer; bump higher via env if extra clearance is wanted.
    min_safe_plug_z_above_port_m: float = 0.05
    # Force-gate (no retreat — Z just freezes)
    force_stop_n: float = 18.0
    force_resume_n: float = 12.0
    hold_timeout_s: float = 3.0
    # Spiral
    spiral_lo_n: float = 8.0
    spiral_radius_m: float = 0.002
    spiral_freq_hz: float = 0.5
    spiral_ramp_ticks: int = 5
    spiral_mode_by_plug: dict = None
    spiral_mode_default: str = "circular"
    # LATCH
    latch_blend_ticks: int = 5
    inside_xy_threshold_m: float = 0.002
    inside_depth_by_plug: dict = None
    inside_depth_default: float = -0.002
    inside_tight_threshold_m: float = 0.0005
    inside_chamfer_threshold_m: float = 0.003
    plug_tight_axis_by_plug: dict = None
    # PI gains (same as CheatCodeRobust — known good)
    proportional_gain: float = 0.25
    integrator_gain: float = 0.1
    derivative_gain: float = 0.08
    max_integrator_windup: float = 0.30
    # Injection
    inject_rate: float = 0.3
    inject_min_m: float = 0.001
    inject_max_m: float = 0.003

    def __post_init__(self):
        if self.spiral_mode_by_plug is None:
            self.spiral_mode_by_plug = {"sc": "x_only", "sfp": "circular"}
        if self.inside_depth_by_plug is None:
            self.inside_depth_by_plug = {"sfp": -0.003, "sc": -0.0015}
        if self.plug_tight_axis_by_plug is None:
            self.plug_tight_axis_by_plug = {"sc": "y"}


# ---------------------------------------------------------------------------
# 3-phase trajectory generator
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """Stateful per-trial trajectory generator. step() returns the next
    commanded gripper pose."""

    PHASE_0_RAISE = "PHASE_0_RAISE"
    PHASE_1_XY = "PHASE_1_XY_APPROACH"
    PHASE_2_ORIENT = "PHASE_2_ORIENT"
    PHASE_3_DESCEND = "PHASE_3_DESCEND"
    PHASE_DONE = "DONE"

    SPIRAL_OFF = "OFF"
    SPIRAL_RAMPING_UP = "RAMPING_UP"
    SPIRAL_ON = "ON"
    SPIRAL_RAMPING_DOWN = "RAMPING_DOWN"

    def __init__(
        self,
        task,
        port_transform: Transform,
        plug_tf_stamped: TransformStamped,
        gripper_tf_stamped: TransformStamped,
        params: ContinuousParams,
    ):
        self.task = task
        self.params = params
        self._port_xyz = (
            float(port_transform.translation.x),
            float(port_transform.translation.y),
            float(port_transform.translation.z),
        )
        self._port_quat = (
            float(port_transform.rotation.w),
            float(port_transform.rotation.x),
            float(port_transform.rotation.y),
            float(port_transform.rotation.z),
        )
        self._initial_gripper_pose = PoseSnapshot.from_ros_transform(
            gripper_tf_stamped.transform)
        initial_plug_xyz = (
            float(plug_tf_stamped.transform.translation.x),
            float(plug_tf_stamped.transform.translation.y),
            float(plug_tf_stamped.transform.translation.z),
        )
        initial_plug_quat = (
            float(plug_tf_stamped.transform.rotation.w),
            float(plug_tf_stamped.transform.rotation.x),
            float(plug_tf_stamped.transform.rotation.y),
            float(plug_tf_stamped.transform.rotation.z),
        )

        # Sample per-trial XY injection (deterministic in task identity).
        self._injected_xy_local = self._sample_injected_offset(task, initial_plug_quat)

        # gripper-to-plug XYZ offset in BASE FRAME at trial start.
        gp_offset_base = (
            self._initial_gripper_pose.px - initial_plug_xyz[0],
            self._initial_gripper_pose.py - initial_plug_xyz[1],
            self._initial_gripper_pose.pz - initial_plug_xyz[2],
        )

        # Safety floor: plug must be at least min_safe_plug_z_above_port_m above
        # port_z BEFORE any lateral motion (Phase 1) — otherwise the plug body
        # could clip the NIC card / port body while sweeping over.
        # plug_z = gripper_z - gp_offset_z, so:
        #   safe gripper_z = port_z + min_safe_plug_z_above_port_m + gp_offset_z
        safe_gripper_pz = (
            self._port_xyz[2]
            + self.params.min_safe_plug_z_above_port_m
            + gp_offset_base[2]
        )
        self._needs_raise = self._initial_gripper_pose.pz < safe_gripper_pz
        # Phase 0 raise target (only used if needs_raise).
        self._phase0_target_pz = (
            safe_gripper_pz if self._needs_raise else self._initial_gripper_pose.pz
        )
        # Phase 1 starts at whatever Z Phase 0 leaves us at (= safe_z if raised,
        # else initial_z). Phase 1 itself doesn't touch Z.
        phase1_pz = self._phase0_target_pz

        # Compute the port-matching gripper orientation FIRST, so Phase 1's
        # XY target can anticipate Phase 2's rotation effect (look-ahead).
        # Without look-ahead, Phase 2's rotation arcs the plug off port,
        # causing a visible XY "jump" at the start of Phase 3 when PI
        # immediately drives gripper back. With look-ahead, plug ends up
        # exactly at port_xy AFTER Phase 2 — no jump needed.
        # q_target_gripper = q_diff(q_port, q_plug) ⊗ q_gripper_current
        q_plug_inv = (
            -initial_plug_quat[0], initial_plug_quat[1],
            initial_plug_quat[2], initial_plug_quat[3],
        )
        q_diff = quaternion_multiply(self._port_quat, q_plug_inv)
        q_gripper_target_tup = quaternion_multiply(
            q_diff,
            (self._initial_gripper_pose.qw, self._initial_gripper_pose.qx,
             self._initial_gripper_pose.qy, self._initial_gripper_pose.qz),
        )

        # Compute the plug position in the GRIPPER's local frame at trial start:
        #   plug_local = R(q_initial_gripper)⁻¹ × (plug_init − gripper_init)
        # This vector is rigidly attached to the gripper; rotating the gripper
        # rotates this vector in base frame too.
        q_init_grip_inv = (
            self._initial_gripper_pose.qw,
            -self._initial_gripper_pose.qx,
            -self._initial_gripper_pose.qy,
            -self._initial_gripper_pose.qz,
        )
        plug_minus_gripper = (
            initial_plug_xyz[0] - self._initial_gripper_pose.px,
            initial_plug_xyz[1] - self._initial_gripper_pose.py,
            initial_plug_xyz[2] - self._initial_gripper_pose.pz,
        )
        plug_local_in_gripper = rotate_vec_by_quat(
            plug_minus_gripper, q_init_grip_inv)

        # Plug position relative to gripper AFTER the Phase 2 rotation (in base
        # frame): R(q_target_gripper) × plug_local_in_gripper.
        plug_offset_after_rotation = rotate_vec_by_quat(
            plug_local_in_gripper, q_gripper_target_tup)

        # Phase 1 target XY: where the gripper must be so that, after Phase 2
        # rotation, the plug ends EXACTLY at port_xy + injected_xy. Solving
        # plug_after = gripper_after + plug_offset_after_rotation = port + inj
        # gives gripper_after.xy = (port + inj).xy − plug_offset_after_rotation.xy.
        self._phase1_target = PoseSnapshot(
            px=(self._port_xyz[0] + self._injected_xy_local[0]
                - float(plug_offset_after_rotation[0])),
            py=(self._port_xyz[1] + self._injected_xy_local[1]
                - float(plug_offset_after_rotation[1])),
            pz=phase1_pz,
            qw=self._initial_gripper_pose.qw,
            qx=self._initial_gripper_pose.qx,
            qy=self._initial_gripper_pose.qy,
            qz=self._initial_gripper_pose.qz,
        )

        # Phase 2 target: same XYZ as Phase 1; orientation = port-matching.
        self._phase2_target = PoseSnapshot(
            px=self._phase1_target.px,
            py=self._phase1_target.py,
            pz=self._phase1_target.pz,
            qw=q_gripper_target_tup[0], qx=q_gripper_target_tup[1],
            qy=q_gripper_target_tup[2], qz=q_gripper_target_tup[3],
        )

        # Compute Phase 0 / Phase 1 / Phase 2 durations dynamically based on
        # actual displacement so the peak min-jerk velocity stays bounded.
        # min-jerk peak velocity = 1.875 × distance / duration
        # → duration = 1.875 × distance / max_velocity
        # Phase 0 (RAISE) only runs if needs_raise; pure Z translation.
        raise_distance = abs(
            self._phase0_target_pz - self._initial_gripper_pose.pz)
        self._phase0_duration_s = (
            max(
                self.params.min_phase_duration_s,
                1.875 * raise_distance / max(1e-6, self.params.max_linear_velocity_m_s),
            )
            if self._needs_raise else 0.0
        )
        # Phase 1 displacement: from Phase 0 end (XY=initial, Z=phase0_target_pz)
        # to phase1_target (XY=over-port, Z=phase0_target_pz). Pure XY motion.
        disp_xyz = (
            self._phase1_target.px - self._initial_gripper_pose.px,
            self._phase1_target.py - self._initial_gripper_pose.py,
            self._phase1_target.pz - self._phase0_target_pz,  # 0 by construction
        )
        disp_norm = math.sqrt(sum(d * d for d in disp_xyz))
        self._phase1_duration_s = max(
            self.params.min_phase_duration_s,
            1.875 * disp_norm / max(1e-6, self.params.max_linear_velocity_m_s),
        )
        # Angular distance between initial gripper quat and port-matching quat
        # = 2·arccos(|dot|).
        q_a = (
            self._initial_gripper_pose.qw, self._initial_gripper_pose.qx,
            self._initial_gripper_pose.qy, self._initial_gripper_pose.qz,
        )
        q_b = (
            self._phase2_target.qw, self._phase2_target.qx,
            self._phase2_target.qy, self._phase2_target.qz,
        )
        dot = abs(q_a[0] * q_b[0] + q_a[1] * q_b[1]
                  + q_a[2] * q_b[2] + q_a[3] * q_b[3])
        ang_dist = 2.0 * math.acos(min(1.0, dot))
        self._phase2_duration_s = max(
            self.params.min_phase_duration_s,
            1.875 * ang_dist / max(1e-6, self.params.max_angular_velocity_rad_s),
        )

        # Phase 3 state. z_offset = current commanded plug Z relative to port Z.
        # Starts wherever Phase 2 leaves us (= phase0_target_pz, since Phase 1/2
        # don't touch Z), decreases monotonically until insert_z_offset.
        self._gp_offset_z = gp_offset_base[2]
        self._z_offset = (
            self._phase0_target_pz
            - self._port_xyz[2]
            - gp_offset_base[2]
        )
        self._integrator = (0.0, 0.0)
        self._prev_err = (0.0, 0.0)
        self._hold_start_ns: Optional[int] = None
        self._hold_freeze_z: Optional[float] = None
        # Spiral
        self._spiral_state = self.SPIRAL_OFF
        self._spiral_amplitude = 0.0
        self._spiral_t_anchor_ns: Optional[int] = None
        self._spiral_ramp_progress = 0  # 0..spiral_ramp_ticks
        # LATCH
        self._latched = False
        self._latch_blend_t = 0
        self._locked_pose: Optional[PoseSnapshot] = None
        # Phase 3 ramp-in (smooth handoff Phase 2 → Phase 3).
        self._phase3_handoff_t = 0
        self._phase3_handoff_ticks = self.params.latch_blend_ticks  # reuse 5 ticks
        # Phase tracking. Start in PHASE_0_RAISE iff initial Z is below the
        # safe-clearance floor; otherwise skip straight to PHASE_1_XY.
        self._phase = (
            self.PHASE_0_RAISE if self._needs_raise else self.PHASE_1_XY
        )
        self._t_phase_start_ns: Optional[int] = None
        self._last_pose: Optional[PoseSnapshot] = None
        # Counters for end-of-trial logging.
        self.spiral_engagements = 0
        self.hold_engagements = 0

    # ---------------- Per-trial setup helpers ----------------

    def _sample_injected_offset(
        self, task, plug_quat: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """Per-trial deterministic XY offset (in base frame) sampled in
        plug-local frame, capped on the tight axis.

        With probability `inject_rate`, sample magnitude in [min, max]; cap
        plug-local Y at 0.5 mm for SC (tight axis); rotate to base via plug quat.
        Otherwise return (0, 0).
        """
        if self.params.inject_rate <= 0.0:
            return (0.0, 0.0)
        seed_str = (
            f"inject|{getattr(task, 'cable_name', '') or ''}"
            f"|{getattr(task, 'target_module_name', '') or ''}"
            f"|{getattr(task, 'port_name', '') or ''}"
            f"|{getattr(task, 'plug_name', '') or ''}"
        )
        digest = hashlib.sha1(seed_str.encode()).digest()
        seed = int.from_bytes(digest[:4], "little")
        rng = np.random.default_rng(seed)
        # Inject decision
        if rng.random() >= self.params.inject_rate:
            return (0.0, 0.0)
        mag = rng.uniform(self.params.inject_min_m, self.params.inject_max_m)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        local_dx = mag * math.cos(theta)
        local_dy = mag * math.sin(theta)
        # Plug-aware tight-axis cap.
        tight_axis = self.params.plug_tight_axis_by_plug.get(
            getattr(task, "plug_type", ""))
        if tight_axis == "y":
            local_dy = max(-0.0005, min(0.0005, local_dy))
        elif tight_axis == "x":
            local_dx = max(-0.0005, min(0.0005, local_dx))
        # Rotate plug-local (dx, dy, 0) into base via plug quat.
        d_base = rotate_vec_by_quat((local_dx, local_dy, 0.0), plug_quat)
        return (float(d_base[0]), float(d_base[1]))

    # ---------------- Public step ----------------

    def is_done(self) -> bool:
        return self._phase == self.PHASE_DONE

    def step(
        self,
        now_ns: int,
        port_transform: Transform,
        plug_tf_stamped: Optional[TransformStamped],
        gripper_tf_stamped: Optional[TransformStamped],
        force_mag: float,
    ) -> PoseSnapshot:
        """Advance the trajectory by one tick. Returns the commanded pose."""
        if self._t_phase_start_ns is None:
            self._t_phase_start_ns = now_ns
        if self._phase == self.PHASE_0_RAISE:
            pose = self._step_phase0(now_ns)
        elif self._phase == self.PHASE_1_XY:
            pose = self._step_phase1(now_ns)
        elif self._phase == self.PHASE_2_ORIENT:
            pose = self._step_phase2(now_ns)
        elif self._phase == self.PHASE_3_DESCEND:
            pose = self._step_phase3(
                now_ns, port_transform, plug_tf_stamped,
                gripper_tf_stamped, force_mag,
            )
        else:
            pose = self._last_pose if self._last_pose is not None else self._phase2_target
        self._last_pose = pose
        return pose

    # ---------------- Phase 0: RAISE Z (safety pre-clearance) ----------------

    def _step_phase0(self, now_ns: int) -> PoseSnapshot:
        """Pure Z translation from initial Z to phase0_target_pz so the plug
        clears NIC card / port body height before lateral motion. XY and
        orientation frozen at initial values."""
        elapsed_s = (now_ns - self._t_phase_start_ns) / 1e9
        s = minjerk_s(elapsed_s / max(1e-6, self._phase0_duration_s))
        new_z = (
            self._initial_gripper_pose.pz
            + s * (self._phase0_target_pz - self._initial_gripper_pose.pz)
        )
        if elapsed_s >= self._phase0_duration_s:
            # Continuity: final tick returns the exact target Z.
            self._phase = self.PHASE_1_XY
            self._t_phase_start_ns = now_ns
            return PoseSnapshot(
                px=self._initial_gripper_pose.px,
                py=self._initial_gripper_pose.py,
                pz=self._phase0_target_pz,
                qw=self._initial_gripper_pose.qw,
                qx=self._initial_gripper_pose.qx,
                qy=self._initial_gripper_pose.qy,
                qz=self._initial_gripper_pose.qz,
            )
        return PoseSnapshot(
            px=self._initial_gripper_pose.px,
            py=self._initial_gripper_pose.py,
            pz=new_z,
            qw=self._initial_gripper_pose.qw,
            qx=self._initial_gripper_pose.qx,
            qy=self._initial_gripper_pose.qy,
            qz=self._initial_gripper_pose.qz,
        )

    # ---------------- Phase 1: XY APPROACH ----------------

    def _step_phase1(self, now_ns: int) -> PoseSnapshot:
        elapsed_s = (now_ns - self._t_phase_start_ns) / 1e9
        s = minjerk_s(elapsed_s / max(1e-6, self._phase1_duration_s))
        # Interpolate from Phase 0 end pose (XY=initial, Z=phase0_target_pz,
        # orientation=initial) to phase1_target (XY=over-port, Z=same,
        # orientation=initial). Build the start snapshot inline so that
        # whether or not Phase 0 ran, Phase 1 starts at the right pose.
        phase1_start = PoseSnapshot(
            px=self._initial_gripper_pose.px,
            py=self._initial_gripper_pose.py,
            pz=self._phase0_target_pz,
            qw=self._initial_gripper_pose.qw,
            qx=self._initial_gripper_pose.qx,
            qy=self._initial_gripper_pose.qy,
            qz=self._initial_gripper_pose.qz,
        )
        out = lerp_pose(phase1_start, self._phase1_target, s)
        if elapsed_s >= self._phase1_duration_s:
            # Continuity: final tick of Phase 1 returns the exact target pose.
            self._phase = self.PHASE_2_ORIENT
            self._t_phase_start_ns = now_ns
            return self._phase1_target
        return out

    # ---------------- Phase 2: ORIENT ALIGN ----------------

    def _step_phase2(self, now_ns: int) -> PoseSnapshot:
        elapsed_s = (now_ns - self._t_phase_start_ns) / 1e9
        s = minjerk_s(elapsed_s / max(1e-6, self._phase2_duration_s))
        # Position frozen at phase1_target; orientation slerps.
        q = tj_quat_slerp(
            self._phase1_target.quat(), self._phase2_target.quat(), s)
        pose = PoseSnapshot(
            px=self._phase1_target.px,
            py=self._phase1_target.py,
            pz=self._phase1_target.pz,
            qw=q[0], qx=q[1], qy=q[2], qz=q[3],
        )
        if elapsed_s >= self._phase2_duration_s:
            self._phase = self.PHASE_3_DESCEND
            self._t_phase_start_ns = now_ns
            self._phase3_handoff_t = 0
            return self._phase2_target
        return pose

    # ---------------- Phase 3: DESCEND ----------------

    def _step_phase3(
        self,
        now_ns: int,
        port_transform: Transform,
        plug_tf_stamped: Optional[TransformStamped],
        gripper_tf_stamped: Optional[TransformStamped],
        force_mag: float,
    ) -> PoseSnapshot:
        # Decide if we're holding (force-gate freeze, no retreat).
        in_hold = self._hold_start_ns is not None
        # Force-gate state machine (no retreat — Z just freezes).
        if not in_hold and force_mag > self.params.force_stop_n:
            self._hold_start_ns = now_ns
            self._hold_freeze_z = self._z_offset
            self.hold_engagements += 1
            in_hold = True
        elif in_hold:
            elapsed = (now_ns - self._hold_start_ns) / 1e9
            if force_mag < self.params.force_resume_n or elapsed > self.params.hold_timeout_s:
                # Resume descent from the same z (no retreat).
                self._z_offset = self._hold_freeze_z
                self._hold_start_ns = None
                self._hold_freeze_z = None
                in_hold = False

        # Spiral state machine — engagement gate based on force band.
        engagement_ok = (
            self.params.spiral_lo_n <= force_mag < self.params.force_stop_n
            and not in_hold
            and not self._latched
        )
        self._update_spiral_state(now_ns, engagement_ok)

        # Advance z_offset (descent step) only if not holding AND not latched-with-z-frozen.
        # Ramp descent rate from 0 to full over `descent_ramp_s` at start of Phase 3
        # using min-jerk shape, so Z velocity has zero derivative at the Phase 2 → 3
        # boundary (matches Phase 2's zero rotation velocity at its own boundary).
        if not in_hold:
            elapsed_p3_s = (now_ns - self._t_phase_start_ns) / 1e9
            if self.params.descent_ramp_s > 0.0:
                ramp = minjerk_s(elapsed_p3_s / self.params.descent_ramp_s)
            else:
                ramp = 1.0
            effective_rate = self.params.descent_rate_m_s * ramp
            self._z_offset = max(
                self.params.insert_z_offset,
                self._z_offset - effective_rate * self.params.tick_period_s,
            )

        # Effective z offset for pose computation.
        effective_z = self._hold_freeze_z if in_hold else self._z_offset

        # Compute the live-PI pose. Requires plug_tf and gripper_tf.
        live_pi_pose = self._compute_pi_pose(
            port_transform, plug_tf_stamped, gripper_tf_stamped, effective_z,
        )

        # Apply spiral perturbation (smoothly ramped amplitude, computed in
        # plug-local then rotated to base).
        spiral_dx, spiral_dy = self._compute_spiral_offset(
            now_ns, plug_tf_stamped)
        live_pi_pose = PoseSnapshot(
            px=live_pi_pose.px + spiral_dx,
            py=live_pi_pose.py + spiral_dy,
            pz=live_pi_pose.pz,
            qw=live_pi_pose.qw, qx=live_pi_pose.qx,
            qy=live_pi_pose.qy, qz=live_pi_pose.qz,
        )

        # Phase 2 → Phase 3 smooth handoff: blend over `phase3_handoff_ticks`
        # from phase2_target to live_pi_pose. Eliminates the discontinuity that
        # would otherwise appear when PI's natural output differs from the
        # phase-2-end pose (because the rotation in phase 2 moved plug XY).
        if self._phase3_handoff_t < self._phase3_handoff_ticks:
            blend = minjerk_s(
                (self._phase3_handoff_t + 1) / float(self._phase3_handoff_ticks))
            handoff_pose = lerp_pose(self._phase2_target, live_pi_pose, blend)
            self._phase3_handoff_t += 1
            commanded = handoff_pose
        else:
            commanded = live_pi_pose

        # LATCH check + smooth blend.
        commanded = self._maybe_latch_and_blend(
            commanded, port_transform, plug_tf_stamped, effective_z,
        )

        # Don't auto-terminate Phase 3 when descent reaches insert_z_offset.
        # z_offset is already clamped at insert_z_offset by the descent step
        # above (max() floor), so Z stops decreasing — but PI on XY, spiral
        # search, and LATCH stay live so the policy keeps trying to find the
        # port. The loop exits only on (a) /scoring/insertion_event firing
        # (handled in insert_cable) or (b) engine cancellation at time_limit
        # (handled by _should_abort). This way every trial uses its full
        # time_limit budget instead of giving up at the moment the commanded
        # depth target is met.

        return commanded

    # ---------------- Phase 3 helpers ----------------

    def _compute_pi_pose(
        self,
        port_transform: Transform,
        plug_tf_stamped: Optional[TransformStamped],
        gripper_tf_stamped: Optional[TransformStamped],
        z_offset: float,
    ) -> PoseSnapshot:
        """Compute the PI-corrected gripper pose for descent. Mirrors
        CheatCodeRobust._calc_gripper_pose math, with orientation locked at
        phase2_target (the port-matching orientation)."""
        # Fall-back: if any TF unavailable, command gripper Z based on stored
        # plug-grip offset; XY/orientation frozen at phase2_target.
        if plug_tf_stamped is None or gripper_tf_stamped is None:
            return PoseSnapshot(
                px=self._phase2_target.px,
                py=self._phase2_target.py,
                pz=self._port_xyz[2] + z_offset + self._gp_offset_z,
                qw=self._phase2_target.qw, qx=self._phase2_target.qx,
                qy=self._phase2_target.qy, qz=self._phase2_target.qz,
            )

        port_xy = (port_transform.translation.x, port_transform.translation.y)
        port_z = port_transform.translation.z
        target_xy = (
            port_xy[0] + self._injected_xy_local[0],
            port_xy[1] + self._injected_xy_local[1],
        )
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        gp_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_err = target_xy[0] - plug_xyz[0]
        tip_y_err = target_xy[1] - plug_xyz[1]

        # PI integrator (clamped).
        new_int_x = max(-self.params.max_integrator_windup,
                        min(self.params.max_integrator_windup,
                            self._integrator[0] + tip_x_err))
        new_int_y = max(-self.params.max_integrator_windup,
                        min(self.params.max_integrator_windup,
                            self._integrator[1] + tip_y_err))
        # Derivative.
        d_x = tip_x_err - self._prev_err[0]
        d_y = tip_y_err - self._prev_err[1]
        self._integrator = (new_int_x, new_int_y)
        self._prev_err = (tip_x_err, tip_y_err)

        target_x = (
            target_xy[0]
            + self.params.proportional_gain * tip_x_err
            + self.params.integrator_gain * new_int_x
            + self.params.derivative_gain * d_x
        )
        target_y = (
            target_xy[1]
            + self.params.proportional_gain * tip_y_err
            + self.params.integrator_gain * new_int_y
            + self.params.derivative_gain * d_y
        )
        target_z = port_z + z_offset + gp_offset[2]

        return PoseSnapshot(
            px=target_x, py=target_y, pz=target_z,
            qw=self._phase2_target.qw, qx=self._phase2_target.qx,
            qy=self._phase2_target.qy, qz=self._phase2_target.qz,
        )

    def _update_spiral_state(self, now_ns: int, engagement_ok: bool) -> None:
        """Advance the spiral state machine. amplitude ∈ [0, 1] ramps via
        minjerk_s over `spiral_ramp_ticks` between OFF and ON."""
        if self._spiral_state == self.SPIRAL_OFF:
            if engagement_ok:
                self._spiral_state = self.SPIRAL_RAMPING_UP
                self._spiral_ramp_progress = 0
                self._spiral_t_anchor_ns = now_ns
                self.spiral_engagements += 1
        elif self._spiral_state == self.SPIRAL_RAMPING_UP:
            if not engagement_ok:
                self._spiral_state = self.SPIRAL_RAMPING_DOWN
            else:
                self._spiral_ramp_progress += 1
                if self._spiral_ramp_progress >= self.params.spiral_ramp_ticks:
                    self._spiral_state = self.SPIRAL_ON
                    self._spiral_amplitude = 1.0
                else:
                    self._spiral_amplitude = minjerk_s(
                        self._spiral_ramp_progress / float(self.params.spiral_ramp_ticks))
        elif self._spiral_state == self.SPIRAL_ON:
            if not engagement_ok:
                self._spiral_state = self.SPIRAL_RAMPING_DOWN
                self._spiral_ramp_progress = self.params.spiral_ramp_ticks
        elif self._spiral_state == self.SPIRAL_RAMPING_DOWN:
            self._spiral_ramp_progress -= 1
            if self._spiral_ramp_progress <= 0:
                self._spiral_state = self.SPIRAL_OFF
                self._spiral_amplitude = 0.0
                self._spiral_t_anchor_ns = None
            else:
                self._spiral_amplitude = minjerk_s(
                    self._spiral_ramp_progress / float(self.params.spiral_ramp_ticks))
                if engagement_ok:
                    self._spiral_state = self.SPIRAL_RAMPING_UP

    def _compute_spiral_offset(
        self,
        now_ns: int,
        plug_tf_stamped: Optional[TransformStamped],
    ) -> tuple[float, float]:
        """Compute the spiral XY offset in base frame (already scaled by current
        amplitude). Plug-local mode (x_only or circular) determined per plug type.
        """
        if self._spiral_amplitude <= 1e-9 or self._spiral_t_anchor_ns is None:
            return (0.0, 0.0)
        elapsed_s = (now_ns - self._spiral_t_anchor_ns) / 1e9
        ang = 2.0 * math.pi * self.params.spiral_freq_hz * elapsed_s
        mode = self.params.spiral_mode_by_plug.get(
            getattr(self.task, "plug_type", ""), self.params.spiral_mode_default)
        if mode == "x_only":
            local_dx = self.params.spiral_radius_m * math.cos(ang)
            local_dy = 0.0
        else:
            local_dx = self.params.spiral_radius_m * math.cos(ang)
            local_dy = self.params.spiral_radius_m * math.sin(ang)
        # Rotate plug-local (dx, dy, 0) into base via plug quat.
        if plug_tf_stamped is None:
            base_dx = local_dx
            base_dy = local_dy
        else:
            qp = (
                plug_tf_stamped.transform.rotation.w,
                plug_tf_stamped.transform.rotation.x,
                plug_tf_stamped.transform.rotation.y,
                plug_tf_stamped.transform.rotation.z,
            )
            base = rotate_vec_by_quat((local_dx, local_dy, 0.0), qp)
            base_dx, base_dy = float(base[0]), float(base[1])
        return (
            self._spiral_amplitude * base_dx,
            self._spiral_amplitude * base_dy,
        )

    def _maybe_latch_and_blend(
        self,
        commanded: PoseSnapshot,
        port_transform: Transform,
        plug_tf_stamped: Optional[TransformStamped],
        effective_z: float,
    ) -> PoseSnapshot:
        """If LATCH conditions met, freeze XY/orientation and blend over
        latch_blend_ticks. Z continues to descend 1:1 from latch."""
        if not self._latched and plug_tf_stamped is not None:
            dx = port_transform.translation.x - plug_tf_stamped.transform.translation.x
            dy = port_transform.translation.y - plug_tf_stamped.transform.translation.y
            dz_plug = (plug_tf_stamped.transform.translation.z
                       - port_transform.translation.z)
            xy_err = (dx * dx + dy * dy) ** 0.5
            inside_depth = self.params.inside_depth_by_plug.get(
                getattr(self.task, "plug_type", ""), self.params.inside_depth_default)
            tight_axis = self.params.plug_tight_axis_by_plug.get(
                getattr(self.task, "plug_type", ""))
            xy_ok = self._xy_aligned(
                tight_axis, port_transform, plug_tf_stamped,
                self.params.inside_xy_threshold_m,
                self.params.inside_tight_threshold_m,
                self.params.inside_chamfer_threshold_m,
            )
            if xy_ok and dz_plug < inside_depth:
                self._latched = True
                self._latch_blend_t = 0
                self._locked_pose = commanded  # freeze at this pose

        if self._latched and self._locked_pose is not None:
            self._latch_blend_t += 1
            blend = minjerk_s(
                self._latch_blend_t / float(self.params.latch_blend_ticks))
            blended = lerp_pose(commanded, self._locked_pose, blend)
            # Z continues to descend 1:1 from latch — replace blended.z with
            # the live commanded z (which already tracks the descent).
            return PoseSnapshot(
                px=blended.px, py=blended.py, pz=commanded.pz,
                qw=blended.qw, qx=blended.qx, qy=blended.qy, qz=blended.qz,
            )
        return commanded

    def _xy_aligned(
        self,
        tight_axis: Optional[str],
        port_transform: Transform,
        plug_tf_stamped: TransformStamped,
        magnitude_threshold: float,
        tight_threshold: float,
        chamfer_threshold: float,
    ) -> bool:
        """Compute plug-local error, apply axis-aware threshold."""
        dx = port_transform.translation.x - plug_tf_stamped.transform.translation.x
        dy = port_transform.translation.y - plug_tf_stamped.transform.translation.y
        dz = port_transform.translation.z - plug_tf_stamped.transform.translation.z
        q = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        # rotate (dx, dy, dz) into plug-local frame: v_local = q^-1 * v * q
        q_inv = (q[0], -q[1], -q[2], -q[3])
        qv = (0.0, dx, dy, dz)
        tmp = quaternion_multiply(q_inv, qv)
        res = quaternion_multiply(tmp, q)
        ex_l, ey_l = res[1], res[2]
        if tight_axis is None:
            return (ex_l * ex_l + ey_l * ey_l) ** 0.5 < magnitude_threshold
        if tight_axis == "y":
            e_tight, e_chamfer = ey_l, ex_l
        else:
            e_tight, e_chamfer = ex_l, ey_l
        return abs(e_tight) < tight_threshold and abs(e_chamfer) < chamfer_threshold


# ---------------------------------------------------------------------------
# Policy class — wraps TrajectoryGenerator inside the framework's lifecycle.
# ---------------------------------------------------------------------------

class CheatCodeContinuous(Policy):
    """3-phase smooth oracle for IL-friendly data collection."""

    # Env-var names (defaults live in ContinuousParams; env vars override).
    # Phase 1 / Phase 2 durations are dynamic — these velocity caps determine
    # the duration relative to the actual displacement / angular distance.
    MAX_LINEAR_VEL_M_S_ENV = "CHEATCODE_MAX_LINEAR_VEL_M_S"
    MAX_ANGULAR_VEL_RAD_S_ENV = "CHEATCODE_MAX_ANGULAR_VEL_RAD_S"
    MIN_PHASE_DURATION_S_ENV = "CHEATCODE_MIN_PHASE_DURATION_S"
    DESCENT_RATE_M_S_ENV = "CHEATCODE_DESCENT_RATE_M_S"
    DESCENT_RAMP_S_ENV = "CHEATCODE_DESCENT_RAMP_S"
    MIN_SAFE_PLUG_Z_ABOVE_PORT_M_ENV = "CHEATCODE_MIN_SAFE_PLUG_Z_ABOVE_PORT_M"
    FORCE_STOP_N_ENV = "CHEATCODE_FORCE_STOP_N"
    FORCE_RESUME_N_ENV = "CHEATCODE_FORCE_RESUME_N"
    HOLD_TIMEOUT_S_ENV = "CHEATCODE_HOLD_TIMEOUT_S"
    SPIRAL_LO_N_ENV = "CHEATCODE_SPIRAL_LO_N"
    SPIRAL_RADIUS_M_ENV = "CHEATCODE_SPIRAL_RADIUS_M"
    SPIRAL_FREQ_HZ_ENV = "CHEATCODE_SPIRAL_FREQ_HZ"
    SPIRAL_RAMP_TICKS_ENV = "CHEATCODE_SPIRAL_RAMP_TICKS"
    LATCH_BLEND_TICKS_ENV = "CHEATCODE_LATCH_BLEND_TICKS"
    INJECT_RATE_ENV = "CHEATCODE_INJECT_RATE"
    INJECT_MIN_M_ENV = "CHEATCODE_INJECT_MIN_M"
    INJECT_MAX_M_ENV = "CHEATCODE_INJECT_MAX_M"

    TICK_PERIOD_S = 0.05  # 20 Hz

    def __init__(self, parent_node):
        self._task = None
        self._inserted_flag = False
        self._traj: Optional[TrajectoryGenerator] = None
        self._params = self._build_params()
        super().__init__(parent_node)
        # Match the QoS the aic_scoring node advertises.
        event_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._parent_node.create_subscription(
            String, "/scoring/insertion_event",
            self._on_insertion_event, event_qos,
        )

    # ---------------- Param builder ----------------

    def _build_params(self) -> ContinuousParams:
        p = ContinuousParams()

        def _f(name, default):
            try:
                return float(os.environ.get(name, "").strip() or default)
            except ValueError:
                return default

        def _i(name, default):
            try:
                return int(float(os.environ.get(name, "").strip() or default))
            except ValueError:
                return default

        p.max_linear_velocity_m_s = _f(
            self.MAX_LINEAR_VEL_M_S_ENV, p.max_linear_velocity_m_s)
        p.max_angular_velocity_rad_s = _f(
            self.MAX_ANGULAR_VEL_RAD_S_ENV, p.max_angular_velocity_rad_s)
        p.min_phase_duration_s = _f(
            self.MIN_PHASE_DURATION_S_ENV, p.min_phase_duration_s)
        p.descent_rate_m_s = _f(self.DESCENT_RATE_M_S_ENV, p.descent_rate_m_s)
        p.descent_ramp_s = _f(self.DESCENT_RAMP_S_ENV, p.descent_ramp_s)
        p.min_safe_plug_z_above_port_m = _f(
            self.MIN_SAFE_PLUG_Z_ABOVE_PORT_M_ENV,
            p.min_safe_plug_z_above_port_m,
        )
        p.force_stop_n = _f(self.FORCE_STOP_N_ENV, p.force_stop_n)
        p.force_resume_n = _f(self.FORCE_RESUME_N_ENV, p.force_resume_n)
        p.hold_timeout_s = _f(self.HOLD_TIMEOUT_S_ENV, p.hold_timeout_s)
        p.spiral_lo_n = _f(self.SPIRAL_LO_N_ENV, p.spiral_lo_n)
        p.spiral_radius_m = _f(self.SPIRAL_RADIUS_M_ENV, p.spiral_radius_m)
        p.spiral_freq_hz = _f(self.SPIRAL_FREQ_HZ_ENV, p.spiral_freq_hz)
        p.spiral_ramp_ticks = _i(self.SPIRAL_RAMP_TICKS_ENV, p.spiral_ramp_ticks)
        p.latch_blend_ticks = _i(self.LATCH_BLEND_TICKS_ENV, p.latch_blend_ticks)
        p.inject_rate = _f(self.INJECT_RATE_ENV, p.inject_rate)
        p.inject_min_m = _f(self.INJECT_MIN_M_ENV, p.inject_min_m)
        p.inject_max_m = _f(self.INJECT_MAX_M_ENV, p.inject_max_m)
        p.tick_period_s = self.TICK_PERIOD_S
        return p

    def _on_insertion_event(self, msg: String) -> None:
        self.get_logger().info(f"/scoring/insertion_event: {msg.data!r}")
        self._inserted_flag = True

    # ---------------- Lifecycle / abort ----------------

    def _should_abort(self) -> bool:
        parent = self._parent_node
        if not getattr(parent, "is_active", False):
            return True
        gh = getattr(parent, "goal_handle", None)
        if gh is None:
            return True
        if not getattr(gh, "is_active", True):
            return True
        if getattr(gh, "is_cancel_requested", False):
            return True
        return False

    # ---------------- TF helpers ----------------

    def _wait_for_tf(self, target_frame: str, source_frame: str,
                     timeout_sec: float = 10.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame, source_frame, Time())
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}' "
                        "... -- running with ground_truth:=true?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s")
        return False

    def _lookup(self, source_frame: str) -> Optional[TransformStamped]:
        try:
            return self._parent_node._tf_buffer.lookup_transform(
                "base_link", source_frame, Time())
        except TransformException:
            return None

    # ---------------- Force helper ----------------

    @staticmethod
    def _force_mag_from_obs(obs) -> float:
        if obs is None:
            return 0.0
        try:
            wr = obs.wrist_wrench.wrench
            tare = obs.controller_state.fts_tare_offset.wrench
            fx = wr.force.x - tare.force.x
            fy = wr.force.y - tare.force.y
            fz = wr.force.z - tare.force.z
            return (fx * fx + fy * fy + fz * fz) ** 0.5
        except Exception:
            return 0.0

    # ---------------- insert_cable entry point ----------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"CheatCodeContinuous.insert_cable() task: {task}")
        self._task = task
        self._inserted_flag = False

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        # Wait for required TFs.
        for frame in [port_frame, cable_tip_frame, "gripper/tcp"]:
            if not self._wait_for_tf("base_link", frame):
                return False

        # Snapshot initial TFs to seed the trajectory generator.
        port_stamped = self._lookup(port_frame)
        plug_stamped = self._lookup(cable_tip_frame)
        gripper_stamped = self._lookup("gripper/tcp")
        if port_stamped is None or plug_stamped is None or gripper_stamped is None:
            self.get_logger().error("trial setup TF lookup failed")
            return False
        port_transform: Transform = port_stamped.transform

        self._traj = TrajectoryGenerator(
            task=task,
            port_transform=port_transform,
            plug_tf_stamped=plug_stamped,
            gripper_tf_stamped=gripper_stamped,
            params=self._params,
        )
        self.get_logger().info(
            f"injected_xy = ({self._traj._injected_xy_local[0] * 1000:+.2f}, "
            f"{self._traj._injected_xy_local[1] * 1000:+.2f}) mm"
        )

        # Main loop.
        last_log_t = None
        while not self._traj.is_done():
            if self._should_abort():
                self.get_logger().info("aborting (cancel/deactivate)")
                return False
            if self._inserted_flag:
                self.get_logger().info("insertion_event received, exiting")
                return True
            now = self.time_now()
            # Refresh plug + gripper TFs every tick. Port refresh cheap; do it too.
            plug_stamped = self._lookup(cable_tip_frame) or plug_stamped
            gripper_stamped = self._lookup("gripper/tcp") or gripper_stamped
            new_port = self._lookup(port_frame)
            if new_port is not None:
                port_transform = new_port.transform
            obs = None
            try:
                obs = get_observation()
            except Exception as ex:
                self.get_logger().warn(f"get_observation failed: {ex}")
            force_mag = self._force_mag_from_obs(obs)

            pose_snap = self._traj.step(
                now.nanoseconds, port_transform, plug_stamped, gripper_stamped, force_mag,
            )
            pose = Pose(
                position=Point(x=pose_snap.px, y=pose_snap.py, z=pose_snap.pz),
                orientation=Quaternion(
                    w=pose_snap.qw, x=pose_snap.qx, y=pose_snap.qy, z=pose_snap.qz),
            )
            self.set_pose_target(move_robot=move_robot, pose=pose)

            # Throttled log.
            if last_log_t is None or (now - last_log_t).nanoseconds > int(0.5 * 1e9):
                last_log_t = now
                self.get_logger().info(
                    f"phase={self._traj._phase} z_off={self._traj._z_offset * 1000:6.1f}mm "
                    f"|F|={force_mag:5.2f}N spiral={self._traj._spiral_state} "
                    f"latched={self._traj._latched}"
                )
            self.sleep_for(self.TICK_PERIOD_S)

        self.get_logger().info(
            f"CheatCodeContinuous done — spiral engaged "
            f"{self._traj.spiral_engagements} times, "
            f"hold engaged {self._traj.hold_engagements} times"
        )
        return True
