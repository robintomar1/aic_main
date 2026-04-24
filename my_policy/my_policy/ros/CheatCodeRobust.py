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

    # Integrator authority — keep wide range for reliability but moderate
    # gain so the correction trajectory is smooth. Without force-feedback
    # stop, a high gain causes overshoot and the controller holds the
    # commanded pose against contact for seconds, producing force penalties.
    INTEGRATOR_GAIN = 0.15            # upstream value; conservative
    MAX_INTEGRATOR_WINDUP = 0.15      # ours: widened for large swing errors

    # Timing / kinematics.
    APPROACH_Z_OFFSET = 0.2          # start this far above the port along port Z
    HOVER_Z_OFFSET = 0.05            # hover height during ALIGN phase
    INSERT_Z_OFFSET = -0.015         # final descent depth (upstream value)
    APPROACH_STEPS = 100
    APPROACH_SLEEP = 0.05            # 100 * 0.05 = 5 s approach
    ALIGN_DWELL_S = 2.5              # longer dwell so the integrator converges
    DESCENT_STEP = 0.0001            # 0.1 mm per tick
    DESCENT_SLEEP = 0.05             # -> 2 mm/s

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

        target_x = port_xy[0] + self.INTEGRATOR_GAIN * self._tip_x_error_integrator
        target_y = port_xy[1] + self.INTEGRATOR_GAIN * self._tip_y_error_integrator
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

        # --- ALIGN: dwell at hover so the cable stops swinging ----------
        self.get_logger().info("Phase: ALIGN (dwell)")
        dwell_start = self.time_now()
        dwell_end = dwell_start + Duration(seconds=self.ALIGN_DWELL_S)
        while self.time_now() < dwell_end:
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self._calc_gripper_pose(
                        port_transform, z_offset=self.HOVER_Z_OFFSET),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during align: {ex}")
            self.sleep_for(0.05)

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
