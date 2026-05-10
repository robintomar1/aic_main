"""Tilt-probe sub-trajectory for chamfer-band recovery.

Used by CheatCodeRobust (feature-flagged via CHEATCODE_USE_PROBE) and by
CheatCodeContinuous (always on). The probe replaces the blind XY spiral
search with a force-feedback haptic search that is reproducible by an IL
model at inference time (no /tf in the recovery loop — only force, TCP
pose, and orientation that the model has via Observation).

Mechanism
---------
When the plug rests on the housing with a small XY offset from the port:

  * Toward the port direction the chamfer/hole is open — tilting the gripper
    that way lets the plug tip descend into the unobstructed region.
  * Toward the housing wall the surface is solid or rising — tilting that
    way pushes the plug edge into the wall and the reaction force grows.

Sweeping a small tilt cone (default ±5°) through 8 directions in plug-local
XY plane and recording mean |F| per direction yields a force-vs-angle curve.
The argmin of that curve points roughly at the port. We then translate XY
(default 1.5mm) toward the identified direction, return orientation to
nominal, and let DESCEND resume.

Pivot convention
----------------
We pivot at the *gripper TCP*, holding gripper TCP position constant and
rotating only orientation. The plug tip (rigidly attached to the gripper)
arcs through 3D — moves both horizontally and vertically. Force discriminates
by where the tip can physically descend.

This is the simpler interpretation of "tilt the plug at the same z" — gripper
z is the commanded z and stays fixed; plug tip's z varies along the arc.

State machine
-------------
  IDLE -> SETTLE -> SWEEP -> TRANSLATE -> IDLE

  SETTLE   (default 250ms): hold entry pose, let force ring-down.
  SWEEP    (~2s @ 250ms × 8 directions): rotate gripper through 8 directions,
           record mean |F| per direction.
  TRANSLATE (default 500ms): minimum-jerk XY by translate_m toward identified
           low-force direction; orientation slerps back to nominal in parallel.

On ambiguous signal (max(F) - min(F) < ambiguous_threshold; default 1N), the
probe widens tilt magnitude to the next entry in retry_tilt_deg (default
[5, 8, 12]) and re-sweeps. After exhausting retries the probe exits to IDLE
without translating; the caller (CheatCodeRobust / CheatCodeContinuous)
resumes descent and the trial may fail (recorder discards).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Quaternion ops inlined (rather than importing transforms3d._gohlketransforms)
# so this module is host-testable without the pixi env. Convention: (w, x, y, z),
# matches transforms3d._gohlketransforms — values are interchangeable with the
# rest of the codebase.

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


# ----------------------------------------------------------------------------
# Pure helpers — testable without rclpy
# ----------------------------------------------------------------------------

def compute_tilt_axis(
    v_gripper_to_plug: np.ndarray,
    d_base: np.ndarray,
) -> np.ndarray:
    """Return unit axis vector ω such that rotating the gripper by small angle
    θ about ω moves the plug tip in direction `d_base`.

    Math: for small θ, Δ_plug ≈ θ × (ω × v). Solving ω × v = d (unit vec)
    when v ⊥ d gives ω = (v × d) / |v|². Normalize for unit axis.

    Args:
      v_gripper_to_plug: vector from gripper TCP to plug tip in base frame.
      d_base: unit direction vector (in base frame) the plug should arc toward.

    Returns:
      Unit-length 3-vector axis (in base frame) for the tilt rotation.

    Edge case: if v and d are parallel (degenerate, e.g. plug points exactly
    along d), returns the global +Z axis as a fallback. The resulting tilt
    won't move the plug but won't crash either.
    """
    cross = np.cross(v_gripper_to_plug, d_base)
    norm = float(np.linalg.norm(cross))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return cross / norm


def axis_angle_to_quat(
    axis: np.ndarray,
    angle_rad: float,
) -> tuple[float, float, float, float]:
    """Axis-angle → quaternion (w, x, y, z) using the (w, x, y, z) convention
    matching transforms3d._gohlketransforms.quaternion_multiply."""
    half = angle_rad * 0.5
    s = math.sin(half)
    return (math.cos(half), float(axis[0]) * s, float(axis[1]) * s, float(axis[2]) * s)


def rotate_vec_by_quat(
    v: np.ndarray,
    q: tuple[float, float, float, float],
) -> np.ndarray:
    """Rotate 3-vector v by quaternion q (w, x, y, z). Returns a new 3-vector."""
    qv = (0.0, float(v[0]), float(v[1]), float(v[2]))
    qc = (q[0], -q[1], -q[2], -q[3])
    rotated = quaternion_multiply(quaternion_multiply(q, qv), qc)
    return np.array([rotated[1], rotated[2], rotated[3]])


def decide_probe_direction(
    samples: list[tuple[float, float]],
    temperature_n: float = 1.0,
) -> tuple[float, float]:
    """Given (angle_rad, mean_force_N) samples, return (chosen_angle_rad, force_range_N).

    Uses circular weighted average with weights `exp(-force / T)`:
    low force → high weight → result pulled toward that angle.

    Returns:
      chosen_angle_rad: argmin direction (smoothed across all samples).
      force_range_N:    max(force) − min(force). Caller uses this to detect
                        ambiguous signal (force_range < threshold means no
                        clear winner).
    """
    if not samples:
        return 0.0, 0.0
    forces = [f for _, f in samples]
    fmin = min(forces)
    weights = [math.exp(-(f - fmin) / temperature_n) for f in forces]
    sum_w = sum(weights)
    if sum_w <= 0.0:
        return 0.0, max(forces) - min(forces)
    mean_x = sum(w * math.cos(a) for w, (a, _) in zip(weights, samples)) / sum_w
    mean_y = sum(w * math.sin(a) for w, (a, _) in zip(weights, samples)) / sum_w
    chosen_angle = math.atan2(mean_y, mean_x)
    return chosen_angle, max(forces) - min(forces)


def minjerk_s(t_norm: float) -> float:
    """Minimum-jerk shape function on t_norm ∈ [0, 1]:
    s(t) = t³(10 − 15t + 6t²). s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0.
    """
    t = max(0.0, min(1.0, t_norm))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


# ----------------------------------------------------------------------------
# Pose adapter — dataclass stand-in usable in tests AND production
# ----------------------------------------------------------------------------

@dataclass
class PoseSnapshot:
    """A frozen (position, orientation) pair. Used by the probe state machine
    to capture entry pose / interpolate translate target without depending on
    geometry_msgs.Pose construction in pure tests."""
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


def lerp_pose(start: PoseSnapshot, end: PoseSnapshot, s: float) -> PoseSnapshot:
    """Lerp position, slerp orientation. s ∈ [0, 1]."""
    qs = quaternion_slerp(start.quat(), end.quat(), s)
    return PoseSnapshot(
        px=start.px + (end.px - start.px) * s,
        py=start.py + (end.py - start.py) * s,
        pz=start.pz + (end.pz - start.pz) * s,
        qw=qs[0], qx=qs[1], qy=qs[2], qz=qs[3],
    )


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    """Tunable parameters. All exposable via env vars (see CheatCodeRobust /
    CheatCodeContinuous wiring)."""
    tilt_deg: float = 5.0
    n_directions: int = 8
    sample_s: float = 0.25
    settle_s: float = 0.25
    translate_m: float = 0.0015
    translate_duration_s: float = 0.5
    trigger_f_n: float = 8.0
    ambiguous_df_n: float = 1.0
    retry_tilt_deg: list[float] = field(default_factory=lambda: [5.0, 8.0, 12.0])
    decide_temperature_n: float = 1.0


# ----------------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------------

class TiltProbeStateMachine:
    """Force-feedback tilt-probe state machine. Caller drives via .step(now,
    force_mag); pose returned per tick replaces the normal commanded pose."""

    STATE_IDLE = "IDLE"
    STATE_SETTLE = "SETTLE"
    STATE_SWEEP = "SWEEP"
    STATE_TRANSLATE = "TRANSLATE"

    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.state = self.STATE_IDLE
        # Set on enter(); cleared on transition to IDLE.
        self._t_state_start_ns: Optional[int] = None
        self._entry_pose: Optional[PoseSnapshot] = None
        self._plug_pos: Optional[np.ndarray] = None  # base-frame
        self._plug_q: Optional[tuple[float, float, float, float]] = None
        # Sweep accounting.
        self._current_dir_i: int = 0
        self._current_tilt_deg: float = cfg.tilt_deg
        self._retry_idx: int = 0
        self._force_acc: float = 0.0
        self._force_count: int = 0
        self._samples: list[tuple[float, float]] = []
        # Translate target.
        self._chosen_angle_local: Optional[float] = None
        self._translate_start_pose: Optional[PoseSnapshot] = None
        self._translate_target_pose: Optional[PoseSnapshot] = None
        # Diagnostics for outer loop / logging.
        self.last_run_samples: list[tuple[float, float]] = []
        self.last_run_chosen_angle: Optional[float] = None
        self.last_run_force_range: float = 0.0
        self.last_run_retry_count: int = 0
        self.last_run_outcome: str = ""  # "translated" | "exhausted_retries" | ""

    def is_active(self) -> bool:
        return self.state != self.STATE_IDLE

    def enter(
        self,
        now_ns: int,
        gripper_pose: PoseSnapshot,
        plug_pos: np.ndarray,
        plug_q: tuple[float, float, float, float],
    ) -> None:
        """Begin a probe sub-trajectory. Captures the entry pose so the probe
        can return XY translation relative to it and slerp orientation back to it.
        """
        self.state = self.STATE_SETTLE
        self._t_state_start_ns = now_ns
        self._entry_pose = gripper_pose
        self._plug_pos = np.asarray(plug_pos, dtype=np.float64)
        self._plug_q = plug_q
        self._current_dir_i = 0
        self._retry_idx = 0
        self._current_tilt_deg = (
            self.cfg.retry_tilt_deg[0] if self.cfg.retry_tilt_deg else self.cfg.tilt_deg
        )
        self._force_acc = 0.0
        self._force_count = 0
        self._samples = []
        self._chosen_angle_local = None
        self._translate_start_pose = None
        self._translate_target_pose = None
        self.last_run_outcome = ""
        self.last_run_chosen_angle = None
        self.last_run_force_range = 0.0
        self.last_run_retry_count = 0

    def step(self, now_ns: int, force_mag: float) -> PoseSnapshot:
        """Advance the state machine one tick; return the commanded pose for
        this tick. Caller checks is_active() after the call to detect probe
        completion."""
        if self.state == self.STATE_SETTLE:
            return self._step_settle(now_ns)
        if self.state == self.STATE_SWEEP:
            return self._step_sweep(now_ns, force_mag)
        if self.state == self.STATE_TRANSLATE:
            return self._step_translate(now_ns)
        # IDLE: caller shouldn't call us here, but return entry pose as a safe fallback.
        return self._entry_pose if self._entry_pose is not None else PoseSnapshot(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    def _elapsed_s(self, now_ns: int) -> float:
        if self._t_state_start_ns is None:
            return 0.0
        return (now_ns - self._t_state_start_ns) / 1e9

    def _step_settle(self, now_ns: int) -> PoseSnapshot:
        if self._elapsed_s(now_ns) >= self.cfg.settle_s:
            self.state = self.STATE_SWEEP
            self._t_state_start_ns = now_ns
            self._current_dir_i = 0
            self._force_acc = 0.0
            self._force_count = 0
        # During settle, hold entry pose unchanged.
        return self._entry_pose  # type: ignore[return-value]

    def _step_sweep(self, now_ns: int, force_mag: float) -> PoseSnapshot:
        # Compute tilted pose for current direction.
        angle_local = self._current_dir_i * 2.0 * math.pi / self.cfg.n_directions
        pose = self._compute_tilt_pose(
            angle_local, math.radians(self._current_tilt_deg))
        # Accumulate force.
        self._force_acc += float(force_mag)
        self._force_count += 1
        # Advance after sample window.
        if self._elapsed_s(now_ns) >= self.cfg.sample_s:
            mean_force = self._force_acc / max(1, self._force_count)
            self._samples.append((angle_local, mean_force))
            self._current_dir_i += 1
            self._t_state_start_ns = now_ns
            self._force_acc = 0.0
            self._force_count = 0
            if self._current_dir_i >= self.cfg.n_directions:
                self._end_of_sweep()
        return pose

    def _end_of_sweep(self) -> None:
        chosen, df_range = decide_probe_direction(
            self._samples, temperature_n=self.cfg.decide_temperature_n)
        self.last_run_chosen_angle = chosen
        self.last_run_force_range = df_range
        self.last_run_retry_count = self._retry_idx
        self.last_run_samples = list(self._samples)
        if df_range < self.cfg.ambiguous_df_n:
            # Ambiguous → widen tilt and retry (if budget remains).
            self._retry_idx += 1
            if self._retry_idx < len(self.cfg.retry_tilt_deg):
                self._current_tilt_deg = self.cfg.retry_tilt_deg[self._retry_idx]
                self._current_dir_i = 0
                self._samples = []
                # Stay in SWEEP state.
                return
            # Exhausted retries — give up, exit to IDLE.
            self.state = self.STATE_IDLE
            self.last_run_outcome = "exhausted_retries"
            return
        # Got a direction. Compute translate target and proceed.
        self._chosen_angle_local = chosen
        last_tilt_pose = self._compute_tilt_pose(
            self._samples[-1][0],
            math.radians(self._current_tilt_deg),
        )
        self._translate_start_pose = last_tilt_pose
        self._translate_target_pose = self._compute_translate_target(chosen)
        self.state = self.STATE_TRANSLATE
        self.last_run_outcome = "translated"

    def _step_translate(self, now_ns: int) -> PoseSnapshot:
        elapsed = self._elapsed_s(now_ns)
        s = minjerk_s(elapsed / max(1e-6, self.cfg.translate_duration_s))
        out = lerp_pose(
            self._translate_start_pose,  # type: ignore[arg-type]
            self._translate_target_pose,  # type: ignore[arg-type]
            s,
        )
        if elapsed >= self.cfg.translate_duration_s:
            self.state = self.STATE_IDLE
        return out

    def _compute_tilt_pose(self, angle_local: float, tilt_rad: float) -> PoseSnapshot:
        """Compute the gripper pose corresponding to a tilt of `tilt_rad` in
        plug-local direction `angle_local`. Position unchanged; orientation
        rotated."""
        d_local = np.array([math.cos(angle_local), math.sin(angle_local), 0.0])
        d_base = rotate_vec_by_quat(d_local, self._plug_q)  # type: ignore[arg-type]
        v_base = self._plug_pos - np.array([
            self._entry_pose.px, self._entry_pose.py, self._entry_pose.pz,  # type: ignore[union-attr]
        ])
        omega = compute_tilt_axis(v_base, d_base)
        q_tilt = axis_angle_to_quat(omega, tilt_rad)
        q_g = self._entry_pose.quat()  # type: ignore[union-attr]
        q_new = quaternion_multiply(q_tilt, q_g)
        return PoseSnapshot(
            px=self._entry_pose.px,  # type: ignore[union-attr]
            py=self._entry_pose.py,  # type: ignore[union-attr]
            pz=self._entry_pose.pz,  # type: ignore[union-attr]
            qw=q_new[0], qx=q_new[1], qy=q_new[2], qz=q_new[3],
        )

    def _compute_translate_target(self, chosen_angle_local: float) -> PoseSnapshot:
        """Translate target: gripper XY moves by translate_m in plug-local
        direction `chosen_angle_local`; Z unchanged; orientation back to entry."""
        d_local = np.array([math.cos(chosen_angle_local), math.sin(chosen_angle_local), 0.0])
        d_base = rotate_vec_by_quat(d_local, self._plug_q)  # type: ignore[arg-type]
        return PoseSnapshot(
            px=self._entry_pose.px + self.cfg.translate_m * float(d_base[0]),  # type: ignore[union-attr]
            py=self._entry_pose.py + self.cfg.translate_m * float(d_base[1]),  # type: ignore[union-attr]
            pz=self._entry_pose.pz,  # type: ignore[union-attr]
            qw=self._entry_pose.qw,  # type: ignore[union-attr]
            qx=self._entry_pose.qx,  # type: ignore[union-attr]
            qy=self._entry_pose.qy,  # type: ignore[union-attr]
            qz=self._entry_pose.qz,  # type: ignore[union-attr]
        )
