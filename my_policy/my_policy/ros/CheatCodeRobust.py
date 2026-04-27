"""CheatCodeRobust — oracle policy tuned for randomized data collection.

Based on aic_example_policies.ros.CheatCode with modifications that come from
(a) Rocky Shao's public writeup on the ROS Discourse AIC thread (t/53155) and
(b) direct observation that vanilla CheatCode fails 2/3 trials under fully
randomized board pose because its XY integrator has too little correction
authority to compensate for cable swing.

Changes vs. upstream CheatCode:
  - Integrator gain 0.15 -> 0.2; max windup 0.05 -> 0.15 m (wider authority).
  - Added ALIGN phase: 1 s dwell at 5 cm hover so the cable settles before descent.
  - Descent speed 10 mm/s -> 4 mm/s (matches Rocky's 0.02 m/s).
  - Re-look up plug-tip TF on every descent iteration. Vanilla CheatCode's
    XY target is port_xy, which ignores the cable swing-induced offset
    between the gripper and the actual plug tip. Re-reading TF every step
    feeds the current real error into the integrator.

Training-oracle only. Uses ground_truth:=true TF and is NOT submission-safe.
"""

import hashlib
import math
import os

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


class CheatCodeRobust(Policy):
    """Ground-truth-TF insertion oracle tuned for randomized configs."""

    # XY correction: PI controller on (port_xy - plug_xy). The P term
    # reacts immediately to the current error (not accumulated), so the
    # gripper pulls toward the position that puts the plug at the port.
    # The I term handles steady-state offset from cable hang.
    # Keep P < 1 so we don't over-react (at P=1 this becomes v3-style
    # full feedforward, which was unstable because cable lag fed back on
    # itself). P=0.5 gives useful immediate correction with stable damping.
    # P=0.3 (down from 0.5): observed XY oscillation at hover with 0.5; the
    # cable's lateral inertia + admittance compliance gives the loop enough
    # phase lag that 0.5 over-reacts. 0.3 still drives the steady-state
    # error fast enough that the I term doesn't have to work alone.
    PROPORTIONAL_GAIN = 0.3
    INTEGRATOR_GAIN = 0.15
    MAX_INTEGRATOR_WINDUP = 0.30     # 15 mm max I correction; typical
                                     # steady-state cable offset after settling

    # Timing / kinematics.
    APPROACH_Z_OFFSET = 0.2          # start this far above the port along port Z
    # Hover height depends on plug length: SFP module is longer than SC plug,
    # needs more clearance above port rim so body doesn't pre-contact.
    HOVER_Z_OFFSET_BY_PLUG = {
        "sfp": 0.20,
        "sc": 0.10,
    }
    HOVER_Z_OFFSET_DEFAULT = 0.10
    INSERT_Z_OFFSET = -0.015         # final descent depth (upstream value)
    APPROACH_STEPS = 100
    APPROACH_SLEEP = 0.06            # 100 * 0.06 = 6 s approach
    DESCENT_STEP = 0.0003            # 0.15 mm per tick
    DESCENT_SLEEP = 0.05             # -> 3 mm/s (graceful insertion descent)

    # ALIGN phase: gate descent on actual XY convergence. Tightened to
    # 2.5 mm so we enter the chamfer with margin, not at the edge.
    # Timeout 5 s — with the wider P/I gains the loop converges in ~1-2 s
    # when it's going to converge at all, so 5 s is plenty and bails out
    # of stuck-misaligned cases faster.
    ALIGN_XY_THRESHOLD_M = 0.0025
    ALIGN_STABLE_S = 1.0
    ALIGN_TIMEOUT_S = 10.0
    ALIGN_POLL_S = 0.05
    # ALIGN_BAIL_XY_M removed 2026-04-29: policy-side trial termination is no
    # longer used. Authoritative terminator is the eval container's per-task
    # time_limit. Bad demos are filtered downstream by the recorder's
    # strict-discard predicate (only trials firing /scoring/insertion_event
    # are saved).

    # INSERT phase force gate. Thresholds sit just below the scoring
    # penalty cutoff (>20 N for >1 s = -12 pts). Setting STOP at 18 N
    # means normal chamfer-entry forces of 10-15 N don't trigger a hold
    # at all — we only react to genuine smashing. Once engaged, we need
    # force to drop back near the noise floor to resume, so FORCE_RESUME_N
    # is kept tight relative to STOP (hysteresis prevents chatter but the
    # gate stays usable — too high a resume threshold and we never exit).
    FORCE_STOP_N = 18.0
    FORCE_RESUME_N = 12.0
    FORCE_HOLD_MAX_S = 1.0

    # When the force gate engages we not only freeze Z — we also slowly
    # *retreat* upward during the hold. This disengages the plug from
    # whatever it's stuck on (typically the port rim), letting force
    # drop below FORCE_RESUME_N so the gate can release cleanly. On
    # resume, z_offset is set to the retreated value so the commanded
    # gripper pose doesn't snap back down.
    HOLD_RETREAT_STEP = 0.0001       # 2 mm/s retreat at 50 ms tick
    HOLD_RETREAT_MAX = 0.005         # cap retreat per single hold at 5 mm

    # Inside-port XY lock. Once the plug body is past the port's chamfer the
    # mechanical walls constrain XY — continued P/I corrections fight that
    # constraint, generating lateral forces that bind the plug or amplify
    # force-hold engagements. Latch on (xy_err small AND plug_tip below port
    # plane by depth-threshold), freeze the commanded XY+orientation, and let
    # only Z continue to descend. Per-plug-type depth: SFP body is longer so
    # the tip needs to be deeper before we trust "inside"; SC is shorter.
    INSIDE_XY_THRESHOLD_M = 0.002    # tighter than ALIGN (2.5 mm) by design
    INSIDE_DEPTH_BY_PLUG = {
        "sfp": -0.003,
        "sc": -0.0015,
    }
    INSIDE_DEPTH_DEFAULT = -0.002

    # Plug-type anisotropy. SFP plug has chamfers along both local X and Y so
    # any small XY error is forgiven by the chamfer — magnitude check is
    # appropriate. SC plug has a chamfer along local X only; local Y is the
    # tight direction (jams if eY > ~0.5 mm at chamfer entry, even when
    # |xy_err| is sub-mm). Empirically validated against 7-trial run
    # 2026-04-28: failures clustered at eY > 0.5 mm regardless of eX magnitude
    # (eX up to ±1.4 mm in successes); successes had eY < 0.5 mm. Trial E
    # was the proof: eY=1.96 mm jammed → force-gate retreat cycle walked the
    # plug along the chamfer until eY=0.07 → insertion fired.
    # Map value None means "symmetric, use magnitude check".
    PLUG_TIGHT_AXIS_BY_PLUG = {"sc": "y"}
    ALIGN_TIGHT_THRESHOLD_M = 0.0005     # 0.5 mm on tight axis
    ALIGN_CHAMFER_THRESHOLD_M = 0.003    # 3 mm on chamfer axis (forgiving)
    INSIDE_TIGHT_THRESHOLD_M = 0.0005    # match ALIGN: tight before latching
    INSIDE_CHAMFER_THRESHOLD_M = 0.003

    # Status print throttle. Periodic 1Hz line during APPROACH/ALIGN/INSERT
    # with the metrics we actually want to see (xy err, z err, traj time,
    # Fz). Replaces the framework's bare "insert_cable execute loop" heartbeat
    # in aic_model.py — that's policy-agnostic and can't see these values.
    STATUS_LOG_PERIOD_S = 1.0

    # Noise injection for the localizer-accuracy bench. When the env vars
    # are set, every port-pose lookup from TF is perturbed by a deterministic
    # per-trial offset before downstream use. This mimics what a learned
    # board-pose localizer at submission time would feed CheatCodeRobust:
    # localizer error in (board_x, board_y, board_yaw) translates 1:1 to
    # port-pose error in base_link, so injecting at the port level is exactly
    # equivalent to injecting at the board level. Used to measure how much
    # error the PI controller + chamfer search + retreat-during-hold can
    # mechanically absorb. NOT submission-safe (this is a bench knob).
    NOISE_XY_M_ENV = "CHEATCODE_NOISE_XY_M"
    NOISE_YAW_RAD_ENV = "CHEATCODE_NOISE_YAW_RAD"

    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._task = None
        self._last_status_log_t = None
        # Latched on first "inserted" event; reset at the start of each
        # insert_cable call so stale events don't bleed across trials.
        self._inserted_flag = False
        # Inside-port XY lock state; reset per trial in insert_cable().
        self._inside_latched = False
        # Per-trial port-pose noise offsets; populated at insert_cable() start
        # from env vars. (0,0)/0.0 is a no-op (the default).
        self._noise_xy_offset: tuple[float, float] = (0.0, 0.0)
        self._noise_yaw_offset: float = 0.0
        super().__init__(parent_node)
        # Match the QoS the aic_scoring node advertises
        # (reliable, volatile, keep_last, depth 10) — same profile we use
        # in collect_episode.py's BatchMonitor.
        event_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._parent_node.create_subscription(
            String,
            "/scoring/insertion_event",
            self._on_insertion_event,
            event_qos,
        )

    def _on_insertion_event(self, msg: String) -> None:
        # Any event on this topic during a trial means the scoring system
        # has detected insertion. Log the exact string so we can confirm.
        self.get_logger().info(f"/scoring/insertion_event: {msg.data!r}")
        self._inserted_flag = True

    # ------------------------------------------------------------------
    # Cancellation / lifecycle abort check
    # ------------------------------------------------------------------
    def _should_abort(self) -> bool:
        """True iff insert_cable() must stop running ASAP.

        The framework's `insert_cable_execute_callback` in aic_model.py returns
        on cancel, but it does NOT terminate the action_thread that is running
        this policy method. If we don't poll for the abort condition ourselves,
        the cancelled trial's insert_cable keeps publishing pose commands while
        the framework starts a NEW action_thread for the next trial — both
        threads then interleave move_robot calls. Observed in the 2026-04-26
        log: trial 2 cancel at sim t=39.4s, trial 3 starts at sim t≈+74s, both
        running concurrently for ~70 wall sec; then on lifecycle cleanup the
        publisher was destroyed and trial 2's stale loop spammed
        `move_robot exception: 'NoneType' object has no attribute 'publish'`
        for ~25 wall sec.
        """
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

    # ------------------------------------------------------------------
    # TF helpers
    # ------------------------------------------------------------------
    def _refresh_port_transform(
        self, port_frame: str, last_good: Transform
    ) -> Transform:
        """Re-read the port transform from TF so alignment follows any pose
        changes (board jitter, residual orientation drift). Falls back to
        last_good on transient TF failure so the descent loop keeps running.
        Applies bench noise (no-op when env vars unset).
        """
        try:
            stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time())
            return self._apply_pose_noise(stamped.transform)
        except TransformException:
            return last_good

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
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

    # ------------------------------------------------------------------
    # Port-pose noise injection (localizer-accuracy bench)
    # ------------------------------------------------------------------
    def _read_noise_config(self) -> tuple[float, float]:
        """Returns (xy_radius_m, yaw_radius_rad) from env vars; zeros = no-op."""
        try:
            xy = float(os.environ.get(self.NOISE_XY_M_ENV, "0.0"))
        except ValueError:
            xy = 0.0
        try:
            yaw = float(os.environ.get(self.NOISE_YAW_RAD_ENV, "0.0"))
        except ValueError:
            yaw = 0.0
        return xy, yaw

    def _sample_trial_noise(
        self, task
    ) -> tuple[tuple[float, float], float]:
        """Per-trial deterministic noise offset for port-pose perturbation.

        Same task identity → same offset across re-runs (so a re-bench at the
        same noise magnitude produces the same trajectory). Different tasks
        within a batch get different offsets (random angle in XY, random sign
        for yaw) so the bench measures robustness to noise in any direction,
        not just one. Magnitudes from env vars; both zero → returns no-op.
        """
        xy_radius, yaw_radius = self._read_noise_config()
        if xy_radius == 0.0 and yaw_radius == 0.0:
            return (0.0, 0.0), 0.0
        seed_str = (
            f"{task.cable_name}|{task.target_module_name}|"
            f"{task.port_name}|{task.plug_name}"
        )
        # Stable 32-bit seed from the task-identity bytes. SHA-1 spreads the
        # whole string across the bits — int.from_bytes(...) % 2^32 would only
        # see the bottom 4 bytes (little-endian truncation), causing two task
        # identities that share their first 4 bytes (e.g. "cable_0…", "cable_1…")
        # to collide. Python's built-in hash() is salted per-process so it
        # can't be used for cross-run determinism.
        digest = hashlib.sha1(seed_str.encode()).digest()
        seed = int.from_bytes(digest[:4], "little")
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        xy = (xy_radius * math.cos(theta), xy_radius * math.sin(theta))
        yaw_sign = 1.0 if rng.random() < 0.5 else -1.0
        yaw = yaw_sign * yaw_radius
        return xy, yaw

    def _apply_pose_noise(self, transform: Transform) -> Transform:
        """Return a NEW Transform with the trial-fixed noise offsets applied.

        XY translation is shifted by self._noise_xy_offset; rotation is
        left-multiplied by R_z(self._noise_yaw_offset) so the port "looks"
        rotated about base_link Z. Z, roll, pitch unchanged. Does not mutate
        the input. Returns the input unchanged when both offsets are zero.
        """
        xy = self._noise_xy_offset
        yaw = self._noise_yaw_offset
        if xy == (0.0, 0.0) and yaw == 0.0:
            return transform
        out = Transform()
        out.translation.x = transform.translation.x + xy[0]
        out.translation.y = transform.translation.y + xy[1]
        out.translation.z = transform.translation.z
        if yaw != 0.0:
            half = yaw * 0.5
            qn = (math.cos(half), 0.0, 0.0, math.sin(half))
            qo = (
                transform.rotation.w, transform.rotation.x,
                transform.rotation.y, transform.rotation.z,
            )
            qr = quaternion_multiply(qn, qo)
            out.rotation.w = qr[0]
            out.rotation.x = qr[1]
            out.rotation.y = qr[2]
            out.rotation.z = qr[3]
        else:
            out.rotation.w = transform.rotation.w
            out.rotation.x = transform.rotation.x
            out.rotation.y = transform.rotation.y
            out.rotation.z = transform.rotation.z
        return out

    # ------------------------------------------------------------------
    # Plug-local-frame error decomposition + axis-aware alignment gate
    # ------------------------------------------------------------------
    def _local_axis_err(
        self, port_transform: Transform, plug_tf_stamped
    ) -> tuple[float, float, float]:
        """Project (port − plug) into the plug's local frame.

        Returns (e_x_local, e_y_local, e_z_local) in metres. Used both for
        diagnostic logging and the axis-aware ALIGN/INSIDE-latch gates.
        Quaternion convention matches transforms3d._gohlketransforms (w,x,y,z):
        v_local = q_plug^-1 * v_world * q_plug, with v as a pure quaternion.
        """
        dx = port_transform.translation.x - plug_tf_stamped.transform.translation.x
        dy = port_transform.translation.y - plug_tf_stamped.transform.translation.y
        dz = port_transform.translation.z - plug_tf_stamped.transform.translation.z
        q = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_inv = (q[0], -q[1], -q[2], -q[3])
        qv = (0.0, dx, dy, dz)
        tmp = quaternion_multiply(q_inv, qv)
        res = quaternion_multiply(tmp, q)
        return (res[1], res[2], res[3])

    def _xy_aligned(
        self,
        task,
        ex_local: float,
        ey_local: float,
        magnitude_threshold: float,
        tight_threshold: float,
        chamfer_threshold: float,
    ) -> bool:
        """True iff the plug-local XY error meets alignment criteria.

        Symmetric plugs (no entry in PLUG_TIGHT_AXIS_BY_PLUG, e.g. SFP):
        magnitude check `sqrt(eX² + eY²) < magnitude_threshold`.

        Anisotropic plugs (e.g. SC): per-axis check
        `|e_tight| < tight_threshold AND |e_chamfer| < chamfer_threshold`,
        where the tight/chamfer split is given by PLUG_TIGHT_AXIS_BY_PLUG.
        """
        tight_axis = self.PLUG_TIGHT_AXIS_BY_PLUG.get(task.plug_type)
        if tight_axis is None:
            mag = (ex_local * ex_local + ey_local * ey_local) ** 0.5
            return mag < magnitude_threshold
        if tight_axis == "y":
            e_tight, e_chamfer = ey_local, ex_local
        elif tight_axis == "x":
            e_tight, e_chamfer = ex_local, ey_local
        else:
            raise ValueError(f"unknown tight_axis {tight_axis!r} for plug "
                             f"{task.plug_type!r}")
        return (abs(e_tight) < tight_threshold
                and abs(e_chamfer) < chamfer_threshold)

    # ------------------------------------------------------------------
    # Periodic status line
    # ------------------------------------------------------------------
    def _log_status(
        self,
        phase: str,
        port_transform: Transform,
        traj_start: Time,
        z_offset: float | None = None,
        get_observation=None,
    ) -> None:
        """1 Hz status print: traj time, xy err, z err, Fz, current z_offset."""
        now = self.time_now()
        if self._last_status_log_t is not None:
            if (now - self._last_status_log_t).nanoseconds < int(
                self.STATUS_LOG_PERIOD_S * 1e9
            ):
                return
        self._last_status_log_t = now

        t_traj = (now - traj_start).nanoseconds / 1e9

        xy_err_mm = float("nan")
        z_err_mm = float("nan")
        err_local_x_mm = float("nan")
        err_local_y_mm = float("nan")
        try:
            plug_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                f"{self._task.cable_name}/{self._task.plug_name}_link",
                Time(),
            )
            dx = port_transform.translation.x - plug_tf.transform.translation.x
            dy = port_transform.translation.y - plug_tf.transform.translation.y
            dz = port_transform.translation.z - plug_tf.transform.translation.z
            xy_err_mm = (dx * dx + dy * dy) ** 0.5 * 1000.0
            z_err_mm = dz * 1000.0
            ex_l, ey_l, _ = self._local_axis_err(port_transform, plug_tf)
            err_local_x_mm = ex_l * 1000.0
            err_local_y_mm = ey_l * 1000.0
        except TransformException:
            pass

        fz = float("nan")
        if get_observation is not None:
            try:
                obs = get_observation()
                wr = obs.wrist_wrench.wrench
                tare = obs.controller_state.fts_tare_offset.wrench
                fz = wr.force.z - tare.force.z
            except Exception:
                pass

        z_str = f" z_off={z_offset * 1000:6.1f}mm" if z_offset is not None else ""
        self.get_logger().info(
            f"[{phase} t={t_traj:5.1f}s xy_err={xy_err_mm:5.1f}mm "
            f"(eX={err_local_x_mm:+6.2f}mm eY={err_local_y_mm:+6.2f}mm) "
            f"z_err={z_err_mm:6.1f}mm Fz={fz:6.2f}N{z_str}]"
        )

    # ------------------------------------------------------------------
    # Gripper pose computation — mirrors upstream but with widened integrator.
    # ------------------------------------------------------------------
    def _calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff = quaternion_multiply(q_port, q_plug_inv)

        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link", "gripper/tcp", Time())
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self.MAX_INTEGRATOR_WINDUP,
                self.MAX_INTEGRATOR_WINDUP,
            )
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self.MAX_INTEGRATOR_WINDUP,
                self.MAX_INTEGRATOR_WINDUP,
            )

        # Z feedforward: static offset (gravity-dominated), safe to apply.
        # XY: PI controller on (port - plug) error. P reacts to current
        # error immediately so the gripper pulls toward a position that
        # puts the plug at the port; I handles steady-state offset from
        # cable hang. See PROPORTIONAL_GAIN / MAX_INTEGRATOR_WINDUP above.
        target_x = (
            port_xy[0]
            + self.PROPORTIONAL_GAIN * tip_x_error
            + self.INTEGRATOR_GAIN * self._tip_x_error_integrator
        )
        target_y = (
            port_xy[1]
            + self.PROPORTIONAL_GAIN * tip_y_error
            + self.INTEGRATOR_GAIN * self._tip_y_error_integrator
        )
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(x=blend_xyz[0], y=blend_xyz[1], z=blend_xyz[2]),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    # ------------------------------------------------------------------
    # insert_cable — APPROACH -> ALIGN -> INSERT -> STABILIZE
    # ------------------------------------------------------------------
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"CheatCodeRobust.insert_cable() task: {task}")
        self._task = task
        # Reset insertion-event latch for this trial.
        self._inserted_flag = False
        self._inside_latched = False
        self._last_status_log_t = None
        # Sample per-trial port-pose noise (no-op when env vars unset).
        self._noise_xy_offset, self._noise_yaw_offset = self._sample_trial_noise(task)
        if self._noise_xy_offset != (0.0, 0.0) or self._noise_yaw_offset != 0.0:
            self.get_logger().warn(
                f"NOISE INJECTED for bench: "
                f"xy=({self._noise_xy_offset[0] * 1000:+.1f}, "
                f"{self._noise_xy_offset[1] * 1000:+.1f})mm "
                f"yaw={math.degrees(self._noise_yaw_offset):+.1f}° "
                f"(env: {self.NOISE_XY_M_ENV}, {self.NOISE_YAW_RAD_ENV})"
            )
        traj_start = self.time_now()

        # Pick the plug-type-specific hover distance.
        hover_z = self.HOVER_Z_OFFSET_BY_PLUG.get(
            task.plug_type, self.HOVER_Z_OFFSET_DEFAULT)
        self.get_logger().info(
            f"plug_type={task.plug_type!r}, HOVER_Z_OFFSET={hover_z:.3f}m")

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time())
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False
        port_transform = self._apply_pose_noise(port_tf_stamped.transform)

        # --- APPROACH: smooth interpolation to the hover pose -----------
        self.get_logger().info("Phase: APPROACH")
        for t in range(0, self.APPROACH_STEPS):
            if self._should_abort():
                self.get_logger().info("APPROACH: aborting (cancel/deactivate)")
                return False
            interp_fraction = t / float(self.APPROACH_STEPS)
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self._calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=self.APPROACH_Z_OFFSET,
                        reset_xy_integrator=True,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during approach: {ex}")
            self._log_status("APPROACH", port_transform, traj_start)
            self.sleep_for(self.APPROACH_SLEEP)

        # --- ALIGN: hold at hover and wait for the plug to converge to
        # within ALIGN_XY_THRESHOLD_M of the port in XY. Descent only
        # starts once the error stays below threshold for ALIGN_STABLE_S
        # consecutive seconds. Times out at ALIGN_TIMEOUT_S.
        self.get_logger().info("Phase: ALIGN (gate on plug-port XY error)")
        align_start = self.time_now()
        align_timeout = align_start + Duration(seconds=self.ALIGN_TIMEOUT_S)
        stable_since = None
        converged = False
        # Per-plug-type alignment criterion. Anisotropic plugs (SC) gate
        # on |e_tight| AND |e_chamfer| in plug-local frame; symmetric plugs
        # (SFP) gate on |xy_err| magnitude. The latched-tight-axis label
        # below is for the convergence log only.
        tight_axis = self.PLUG_TIGHT_AXIS_BY_PLUG.get(task.plug_type)
        while self.time_now() < align_timeout:
            if self._should_abort():
                self.get_logger().info("ALIGN: aborting (cancel/deactivate)")
                return False
            port_transform = self._refresh_port_transform(port_frame, port_transform)
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self._calc_gripper_pose(
                        port_transform, z_offset=hover_z),
                )
                plug_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link",
                    f"{self._task.cable_name}/{self._task.plug_name}_link",
                    Time(),
                )
                dx = port_transform.translation.x - plug_tf.transform.translation.x
                dy = port_transform.translation.y - plug_tf.transform.translation.y
                xy_err = (dx * dx + dy * dy) ** 0.5
                ex_l, ey_l, _ = self._local_axis_err(port_transform, plug_tf)
                aligned = self._xy_aligned(
                    task, ex_l, ey_l,
                    magnitude_threshold=self.ALIGN_XY_THRESHOLD_M,
                    tight_threshold=self.ALIGN_TIGHT_THRESHOLD_M,
                    chamfer_threshold=self.ALIGN_CHAMFER_THRESHOLD_M,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during align: {ex}")
                xy_err = float("inf")
                aligned = False

            if aligned:
                if stable_since is None:
                    stable_since = self.time_now()
                elif (self.time_now() - stable_since) >= Duration(seconds=self.ALIGN_STABLE_S):
                    converged = True
                    break
            else:
                stable_since = None

            self._log_status("ALIGN   ", port_transform, traj_start)
            self.sleep_for(self.ALIGN_POLL_S)

        if converged:
            if tight_axis is not None:
                self.get_logger().info(
                    f"ALIGN converged (|e_tight| < {self.ALIGN_TIGHT_THRESHOLD_M * 1000:.1f}mm, "
                    f"|e_chamfer| < {self.ALIGN_CHAMFER_THRESHOLD_M * 1000:.1f}mm; tight_axis={tight_axis})")
            else:
                self.get_logger().info(f"ALIGN converged (xy_err < {self.ALIGN_XY_THRESHOLD_M * 1000:.1f} mm)")
        else:
            # Policy-side trial termination removed 2026-04-29: previously we
            # returned False on xy_err > ALIGN_BAIL_XY_M to prevent bad demos,
            # but the eval container's per-task time_limit is the authoritative
            # terminator at submission time anyway. Always descend after ALIGN
            # timeout — the recorder's strict-discard predicate filters out
            # trials that don't fire /scoring/insertion_event, so bad attempts
            # don't pollute the dataset.
            xy_err_mm = xy_err * 1000 if (xy_err is not None and xy_err < float("inf")) else float("inf")
            self.get_logger().warn(
                f"ALIGN timeout after {self.ALIGN_TIMEOUT_S}s — descending anyway. "
                f"Last xy_err: {xy_err_mm:.1f} mm")

        # --- INSERT: slow descent with force gate and insertion_event exit.
        # PRIMARY exit: the scoring system publishes on /scoring/insertion_event
        # when it detects full insertion — the callback latches
        # self._inserted_flag. When set, stop descent.
        # Force gate: when contact force exceeds FORCE_STOP_N, freeze the
        # Z target (don't decrement z_offset) but LET THE XY INTEGRATOR
        # KEEP RUNNING. This is the key fix for the stuck-at-rim case:
        # while we wait for force to drop, the integrator keeps correcting
        # XY alignment, so the plug drifts toward the port axis instead of
        # sitting misaligned against the rim forever.
        self.get_logger().info("Phase: INSERT")
        z_offset = hover_z
        hold_start = None
        hold_z_offset = None             # commanded Z during hold (retreats upward)
        hold_entry_z_offset = None       # z_offset at the moment of hold entry
        force_stopped_count = 0
        inside_depth_threshold = self.INSIDE_DEPTH_BY_PLUG.get(
            task.plug_type, self.INSIDE_DEPTH_DEFAULT)
        locked_pose: Pose | None = None
        locked_z_offset: float | None = None

        while z_offset > self.INSERT_Z_OFFSET:
            if self._should_abort():
                self.get_logger().info("INSERT: aborting (cancel/deactivate)")
                return False
            if self._inserted_flag:
                self.get_logger().info(
                    f"INSERT: insertion_event received, exiting at "
                    f"z_offset={z_offset * 1000:.1f}mm")
                break

            now = self.time_now()

            # Read compensated wrist wrench.
            force_mag = 0.0
            try:
                obs = get_observation()
                wr = obs.wrist_wrench.wrench
                tare = obs.controller_state.fts_tare_offset.wrench
                fx = wr.force.x - tare.force.x
                fy = wr.force.y - tare.force.y
                fz = wr.force.z - tare.force.z
                force_mag = (fx * fx + fy * fy + fz * fz) ** 0.5
            except Exception as ex:
                self.get_logger().warn(f"Obs read failed during insert: {ex}")

            # Force gate state machine.
            if hold_start is None and force_mag > self.FORCE_STOP_N:
                # Enter HOLD. Freeze z_offset at its current value but
                # retreat the commanded Z upward during the hold so the
                # plug disengages from whatever it's jammed on. XY
                # integrator keeps running via _calc_gripper_pose.
                hold_start = now
                hold_entry_z_offset = z_offset
                hold_z_offset = z_offset
                force_stopped_count += 1
                self.get_logger().info(
                    f"INSERT: force gate engaged at |F|={force_mag:.1f}N, "
                    f"z_offset={z_offset * 1000:.1f}mm — retreating Z, XY free")
            elif hold_start is not None:
                # Retreat upward (increase z_offset) to disengage the plug,
                # capped at HOLD_RETREAT_MAX from the entry point.
                if (hold_z_offset - hold_entry_z_offset) < self.HOLD_RETREAT_MAX:
                    hold_z_offset += self.HOLD_RETREAT_STEP

                elapsed = (now - hold_start).nanoseconds / 1e9
                if force_mag < self.FORCE_RESUME_N or elapsed > self.FORCE_HOLD_MAX_S:
                    # Resume descent from the retreated position so the
                    # commanded pose doesn't snap back down on release.
                    retreat_mm = (hold_z_offset - hold_entry_z_offset) * 1000
                    self.get_logger().info(
                        f"INSERT: resuming (|F|={force_mag:.1f}N, held {elapsed:.2f}s, "
                        f"retreated {retreat_mm:.1f}mm)")
                    z_offset = hold_z_offset
                    hold_start = None
                    hold_z_offset = None
                    hold_entry_z_offset = None

            # Advance z_offset when not holding.
            if hold_start is None:
                z_offset -= self.DESCENT_STEP

            # Command the gripper. Pre-latch: recompute via _calc_gripper_pose
            # so the XY integrator keeps correcting. Z is pinned to the frozen
            # hold_z_offset during a hold. Post-latch: pin XY/orientation to
            # the locked pose; gripper Z tracks effective_z 1:1 from the latch
            # point so descent and force-hold retreat both work transparently.
            effective_z = hold_z_offset if hold_start is not None else z_offset
            port_transform = self._refresh_port_transform(
                port_frame, port_transform)

            # Inside-port latch check (only while not yet latched). Uses the
            # same axis-aware criterion as ALIGN: SC requires |e_tight| AND
            # |e_chamfer| sub-threshold; SFP uses magnitude.
            if not self._inside_latched:
                try:
                    plug_tf_latch = self._parent_node._tf_buffer.lookup_transform(
                        "base_link", cable_tip_frame, Time())
                    dx = (port_transform.translation.x
                          - plug_tf_latch.transform.translation.x)
                    dy = (port_transform.translation.y
                          - plug_tf_latch.transform.translation.y)
                    dz_plug = (plug_tf_latch.transform.translation.z
                               - port_transform.translation.z)
                    xy_err_latch = (dx * dx + dy * dy) ** 0.5
                    ex_l, ey_l, _ = self._local_axis_err(
                        port_transform, plug_tf_latch)
                    xy_ok = self._xy_aligned(
                        task, ex_l, ey_l,
                        magnitude_threshold=self.INSIDE_XY_THRESHOLD_M,
                        tight_threshold=self.INSIDE_TIGHT_THRESHOLD_M,
                        chamfer_threshold=self.INSIDE_CHAMFER_THRESHOLD_M,
                    )
                    if xy_ok and dz_plug < inside_depth_threshold:
                        locked_pose = self._calc_gripper_pose(
                            port_transform, z_offset=effective_z)
                        locked_z_offset = effective_z
                        self._inside_latched = True
                        self.get_logger().info(
                            f"INSERT: inside-port latch engaged "
                            f"(xy_err={xy_err_latch * 1000:.1f}mm "
                            f"eX={ex_l * 1000:+.2f}mm eY={ey_l * 1000:+.2f}mm "
                            f"plug_dz={dz_plug * 1000:.1f}mm) — "
                            f"freezing XY/orientation, Z-only from here")
                except TransformException:
                    pass

            try:
                if self._inside_latched and locked_pose is not None:
                    target_z = (locked_pose.position.z
                                + (effective_z - locked_z_offset))
                    pose = Pose(
                        position=Point(
                            x=locked_pose.position.x,
                            y=locked_pose.position.y,
                            z=target_z,
                        ),
                        orientation=locked_pose.orientation,
                    )
                else:
                    pose = self._calc_gripper_pose(
                        port_transform, z_offset=effective_z)
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insert: {ex}")

            self._log_status(
                "INSERT  ", port_transform, traj_start,
                z_offset=effective_z, get_observation=get_observation,
            )
            self.sleep_for(self.DESCENT_SLEEP)

        self.get_logger().info(
            f"INSERT done — force gate engaged {force_stopped_count} times, "
            f"final z_offset={z_offset * 1000:.1f}mm")

        # No STABILIZE dwell — once /scoring/insertion_event has fired the
        # trial is scored, and on a non-insertion exit there's nothing useful
        # to settle. Return immediately so the engine can deactivate.
        self.get_logger().info("CheatCodeRobust.insert_cable() exiting...")
        return True
