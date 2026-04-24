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
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


class CheatCodeRobust(Policy):
    """Ground-truth-TF insertion oracle tuned for randomized configs."""

    # Primary XY correction is now feedforward from the measured
    # plug-to-gripper offset (mirrors how Z has always worked in upstream
    # CheatCode). Integrator only handles small residuals from cable
    # settling dynamics.
    INTEGRATOR_GAIN = 0.15
    MAX_INTEGRATOR_WINDUP = 0.02     # ~3 mm max correction — residuals only

    # Timing / kinematics.
    APPROACH_Z_OFFSET = 0.2          # start this far above the port along port Z
    HOVER_Z_OFFSET = 0.05            # hover height during ALIGN phase
    INSERT_Z_OFFSET = -0.015         # final descent depth (upstream value)
    APPROACH_STEPS = 100
    APPROACH_SLEEP = 0.05            # 100 * 0.05 = 5 s approach
    DESCENT_STEP = 0.0002            # 0.2 mm per tick
    DESCENT_SLEEP = 0.05             # -> 4 mm/s

    # ALIGN phase: gate descent on actual XY convergence, not a fixed dwell.
    # Scoring tolerance for "inside port" is 5 mm (see docs/scoring.md); we
    # require plug within 3 mm so the 5 mm chamfer is comfortably inside
    # reach. The error must stay under the threshold for ALIGN_STABLE_S
    # consecutive seconds (guards against a transient dip during cable
    # oscillation). If convergence doesn't happen by ALIGN_TIMEOUT_S we
    # descend anyway — better a partial insertion than a stall.
    ALIGN_XY_THRESHOLD_M = 0.003
    ALIGN_STABLE_S = 0.5
    ALIGN_TIMEOUT_S = 6.0
    ALIGN_POLL_S = 0.05

    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._task = None
        super().__init__(parent_node)

    # ------------------------------------------------------------------
    # TF helpers
    # ------------------------------------------------------------------
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

        # Feedforward: cancel the gripper->plug lever so the plug lands at
        # the port. Same principle as target_z has always used. Integrator
        # handles residuals only (cable settling / motion lag).
        target_x = (
            port_xy[0]
            + plug_tip_gripper_offset[0]
            + self.INTEGRATOR_GAIN * self._tip_x_error_integrator
        )
        target_y = (
            port_xy[1]
            + plug_tip_gripper_offset[1]
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
        port_transform = port_tf_stamped.transform

        # --- APPROACH: smooth interpolation to the hover pose -----------
        self.get_logger().info("Phase: APPROACH")
        for t in range(0, self.APPROACH_STEPS):
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
        while self.time_now() < align_timeout:
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self._calc_gripper_pose(
                        port_transform, z_offset=self.HOVER_Z_OFFSET),
                )
                plug_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link",
                    f"{self._task.cable_name}/{self._task.plug_name}_link",
                    Time(),
                )
                dx = port_transform.translation.x - plug_tf.transform.translation.x
                dy = port_transform.translation.y - plug_tf.transform.translation.y
                xy_err = (dx * dx + dy * dy) ** 0.5
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during align: {ex}")
                xy_err = float("inf")

            if xy_err < self.ALIGN_XY_THRESHOLD_M:
                if stable_since is None:
                    stable_since = self.time_now()
                elif (self.time_now() - stable_since) >= Duration(seconds=self.ALIGN_STABLE_S):
                    converged = True
                    break
            else:
                stable_since = None

            self.sleep_for(self.ALIGN_POLL_S)

        if converged:
            self.get_logger().info(f"ALIGN converged (xy_err < {self.ALIGN_XY_THRESHOLD_M * 1000:.1f} mm)")
        else:
            self.get_logger().warn(
                f"ALIGN timeout after {self.ALIGN_TIMEOUT_S}s — descending anyway. "
                f"Last xy_err: {xy_err * 1000:.1f} mm")

        # --- INSERT: slow descent with continuous plug-tip re-read ------
        # By calling _calc_gripper_pose every tick, the plug-tip TF is
        # re-read, so plug_tip_gripper_offset reflects the *current* cable
        # swing state, and the integrator sees the real error.
        self.get_logger().info("Phase: INSERT")
        z_offset = self.HOVER_Z_OFFSET
        while z_offset > self.INSERT_Z_OFFSET:
            z_offset -= self.DESCENT_STEP
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self._calc_gripper_pose(
                        port_transform, z_offset=z_offset),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insert: {ex}")
            self.sleep_for(self.DESCENT_SLEEP)

        self.get_logger().info("Phase: STABILIZE")
        self.sleep_for(5.0)

        self.get_logger().info("CheatCodeRobust.insert_cable() exiting...")
        return True
