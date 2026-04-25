#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""LeRobot driver for AIC robot: derived from https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/robot.py"""

import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from threading import Thread
from typing import Any, Callable, TypedDict, cast

import cv2
import numpy as np
import rclpy
from aic_control_interfaces.msg import (
    ControllerState,
    JointMotionUpdate,
    MotionUpdate,
    TargetMode,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Twist, Vector3, Wrench, WrenchStamped
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from numpy.typing import NDArray
from rclpy.client import Client
from rclpy.executors import SingleThreadedExecutor
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
from rclpy.subscription import Subscription
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .aic_robot import aic_cameras, arm_joint_names
from .types import JointMotionUpdateActionDict, MotionUpdateActionDict

logger = logging.getLogger(__name__)


ObservationState = TypedDict(
    "ObservationState",
    {
        "tcp_pose.position.x": float,
        "tcp_pose.position.y": float,
        "tcp_pose.position.z": float,
        "tcp_pose.orientation.x": float,
        "tcp_pose.orientation.y": float,
        "tcp_pose.orientation.z": float,
        "tcp_pose.orientation.w": float,
        "tcp_velocity.linear.x": float,
        "tcp_velocity.linear.y": float,
        "tcp_velocity.linear.z": float,
        "tcp_velocity.angular.x": float,
        "tcp_velocity.angular.y": float,
        "tcp_velocity.angular.z": float,
        "tcp_error.x": float,
        "tcp_error.y": float,
        "tcp_error.z": float,
        "tcp_error.rx": float,
        "tcp_error.ry": float,
        "tcp_error.rz": float,
        "joint_positions.0": float,
        "joint_positions.1": float,
        "joint_positions.2": float,
        "joint_positions.3": float,
        "joint_positions.4": float,
        "joint_positions.5": float,
        "joint_positions.6": float,
    },
)


# Tare-compensated wrench (raw F/T minus controller's tare offset, which is
# orientation-aware — see reference_aic_ft_sensor.md). 6 floats.
ObservationWrench = TypedDict(
    "ObservationWrench",
    {
        "wrench.fx": float,
        "wrench.fy": float,
        "wrench.fz": float,
        "wrench.tx": float,
        "wrench.ty": float,
        "wrench.tz": float,
    },
)


# Ground-truth port and plug poses in base_link via /tf (only published with
# ground_truth:=true; zeros at eval). 7 floats each: position + quaternion.
# Frame names are configured per-trial via set_active_trial().
ObservationGroundTruth = TypedDict(
    "ObservationGroundTruth",
    {
        "groundtruth.port_pose.x": float,
        "groundtruth.port_pose.y": float,
        "groundtruth.port_pose.z": float,
        "groundtruth.port_pose.qx": float,
        "groundtruth.port_pose.qy": float,
        "groundtruth.port_pose.qz": float,
        "groundtruth.port_pose.qw": float,
        "groundtruth.plug_pose.x": float,
        "groundtruth.plug_pose.y": float,
        "groundtruth.plug_pose.z": float,
        "groundtruth.plug_pose.qx": float,
        "groundtruth.plug_pose.qy": float,
        "groundtruth.plug_pose.qz": float,
        "groundtruth.plug_pose.qw": float,
        # Latched: 1.0 once /scoring/insertion_event "inserted" fires for the
        # active trial; reset on set_active_trial(). Useful for episode-success
        # filtering and as the supervisory label for the data collection run.
        "meta.insertion_success": float,
    },
)


class CameraImageScaling(TypedDict):
    left_camera: float
    center_camera: float
    right_camera: float


@RobotConfig.register_subclass("aic_controller")
@dataclass(kw_only=True)
class AICRobotAICControllerConfig(RobotConfig):
    teleop_target_mode: str = "cartesian"  # "cartesian" or "joint"
    teleop_frame_id: str = "gripper/tcp"  # "gripper/tcp" or "base_link"

    arm_joint_names: list[str] = field(default_factory=arm_joint_names.copy)

    cameras: dict[str, CameraConfig] = field(default_factory=aic_cameras.copy)
    camera_image_scaling: CameraImageScaling = field(
        default_factory=lambda: {
            "left_camera": 0.25,
            "center_camera": 0.25,
            "right_camera": 0.25,
        }
    )


@dataclass(kw_only=True)
class AICRos2Interface:
    node: Node
    executor: SingleThreadedExecutor
    executor_thread: Thread
    change_target_mode_client: Client[
        ChangeTargetMode.Request, ChangeTargetMode.Response
    ]
    motion_update_pub: Publisher[MotionUpdate]
    joint_motion_update_pub: Publisher[JointMotionUpdate]
    controller_state_sub: Subscription[ControllerState]
    joint_states_sub: Subscription[JointState]
    wrench_sub: Subscription[WrenchStamped]
    insertion_event_sub: Subscription[String]
    tf_buffer: Buffer
    tf_listener: TransformListener
    logger: RcutilsLogger

    @staticmethod
    def connect(
        controller_state_cb: Callable[[ControllerState], None],
        joint_states_cb: Callable[[JointState], None],
        wrench_cb: Callable[[WrenchStamped], None],
        insertion_event_cb: Callable[[String], None],
    ) -> "AICRos2Interface":
        if not rclpy.ok():
            rclpy.init()

        node = Node("aic_robot_node")
        logger = node.get_logger()
        logger.set_level(logging.DEBUG)

        change_target_mode_client = node.create_client(
            ChangeTargetMode, f"/aic_controller/change_target_mode"
        )

        while not change_target_mode_client.wait_for_service():
            node.get_logger().info(
                f"Waiting for service 'aic_controller/change_target_mode'..."
            )
            time.sleep(1.0)

        motion_update_pub = node.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10
        )

        joint_motion_update_pub = node.create_publisher(
            JointMotionUpdate, "/aic_controller/joint_commands", 10
        )

        controller_state_sub = node.create_subscription(
            ControllerState, "/aic_controller/controller_state", controller_state_cb, 10
        )

        joint_states_sub = node.create_subscription(
            JointState, "/joint_states", joint_states_cb, qos_profile_sensor_data
        )

        wrench_sub = node.create_subscription(
            WrenchStamped, "/fts_broadcaster/wrench", wrench_cb, qos_profile_sensor_data
        )

        # Latched insertion-success signal published by aic_scoring on full insertion.
        insertion_event_sub = node.create_subscription(
            String, "/scoring/insertion_event", insertion_event_cb, 10
        )

        # TF buffer + listener for ground-truth port/plug pose lookups (only
        # populated when ground_truth:=true; lookups silently fail at eval).
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, node)

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor_thread = Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        time.sleep(3)  # Give some time to connect to services and receive messages

        return AICRos2Interface(
            node=node,
            executor=executor,
            executor_thread=executor_thread,
            change_target_mode_client=change_target_mode_client,
            motion_update_pub=motion_update_pub,
            joint_motion_update_pub=joint_motion_update_pub,
            controller_state_sub=controller_state_sub,
            joint_states_sub=joint_states_sub,
            wrench_sub=wrench_sub,
            insertion_event_sub=insertion_event_sub,
            tf_buffer=tf_buffer,
            tf_listener=tf_listener,
            logger=logger,
        )


class AICRobotAICController(Robot):
    name = "ur5e_aic"

    def __init__(self, config: AICRobotAICControllerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.ros2_interface: AICRos2Interface | None = None
        self.last_controller_state: ControllerState | None = None
        self.last_joint_states: JointState | None = None
        self.last_wrench: WrenchStamped | None = None

        # Per-trial active state (set by recorder via set_active_trial()).
        self._active_port_frame: str | None = None
        self._active_plug_frame: str | None = None
        self._inserted_flag: bool = False

        self._is_connected = False

        if config.teleop_frame_id not in ["gripper/tcp", "base_link"]:
            raise ValueError(
                f"Invalid teleop_frame_id: '{config.teleop_frame_id}'. "
                "Supported frames are 'gripper/tcp' or 'base_link'."
            )
        self.frame_id = config.teleop_frame_id

        if config.teleop_target_mode not in ["cartesian", "joint"]:
            raise ValueError(
                f"Invalid teleop_target_mode: '{config.teleop_target_mode}'. "
                "Supported modes are 'cartesian' or 'joint'."
            )
        self.teleop_target_mode = config.teleop_target_mode

        print(f"Teleop frame id: {self.frame_id}")
        print(f"Teleop target mode: {self.teleop_target_mode}")

    def send_change_control_mode_req(self, mode: int):
        if not self.ros2_interface:
            raise DeviceNotConnectedError()

        req = ChangeTargetMode.Request()
        req.target_mode.mode = mode

        self.ros2_interface.logger.info(
            f"Sending request to change control mode to {mode}"
        )

        response = self.ros2_interface.change_target_mode_client.call(req)

        if not response or not response.success:
            self.ros2_interface.logger.info(f"Failed to change control mode to {mode}")
        else:
            self.ros2_interface.logger.info(
                f"Successfully changed control mode to {mode}"
            )

        time.sleep(0.5)

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (
                # assuming that opencv2 rounds down when being asked to scale without perfect ratio
                int(
                    self.config.cameras[cam].height
                    * self.config.camera_image_scaling[cam]
                ),
                int(
                    self.config.cameras[cam].width
                    * self.config.camera_image_scaling[cam]
                ),
                3,
            )
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict:
        return {
            **ObservationState.__annotations__,
            **ObservationWrench.__annotations__,
            **ObservationGroundTruth.__annotations__,
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return (
            MotionUpdateActionDict.__annotations__
            if self.teleop_target_mode == "cartesian"
            else JointMotionUpdateActionDict.__annotations__
        )

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if calibrate is True:
            print(
                "Warning: Calibration is not supported, ensure the robot is already calibrated before running lerobot."
            )

        def controller_state_cb(msg: ControllerState):
            self.last_controller_state = msg

        def joint_states_cb(msg: JointState):
            self.last_joint_states = msg

        def wrench_cb(msg: WrenchStamped):
            self.last_wrench = msg

        def insertion_event_cb(msg: String):
            # Latch on success; reset is via set_active_trial().
            if msg.data == "inserted":
                self._inserted_flag = True

        self.ros2_interface = AICRos2Interface.connect(
            controller_state_cb, joint_states_cb, wrench_cb, insertion_event_cb
        )

        change_mode_req = (
            TargetMode.MODE_JOINT
            if self.teleop_target_mode == "joint"
            else TargetMode.MODE_CARTESIAN
        )
        self.send_change_control_mode_req(change_mode_req)

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass  # robot must be calibrated before running LeRobot

    def configure(self) -> None:
        pass  # robot must be configured before running LeRobot

    def set_active_trial(
        self,
        port_frame: str | None,
        plug_frame: str | None,
    ) -> None:
        """Configure the active TF frames for the current trial.

        The recorder calls this at trial boundaries. Frames are looked up by
        name from the TF tree on every observation; pass None to disable.
        Resets the latched insertion-success flag for the new trial.
        """
        self._active_port_frame = port_frame
        self._active_plug_frame = plug_frame
        self._inserted_flag = False

    def _compensated_wrench(self) -> tuple:
        """Returns (fx, fy, fz, tx, ty, tz). Zeros if data not yet available.

        Tare uses the controller's orientation-aware fts_tare_offset, matching
        the approach in CheatCodeRobust and the recommendation from Rocky's
        thread on Discourse.
        """
        if self.last_wrench is None or self.last_controller_state is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        raw = self.last_wrench.wrench
        tare = self.last_controller_state.fts_tare_offset.wrench
        return (
            raw.force.x - tare.force.x,
            raw.force.y - tare.force.y,
            raw.force.z - tare.force.z,
            raw.torque.x - tare.torque.x,
            raw.torque.y - tare.torque.y,
            raw.torque.z - tare.torque.z,
        )

    def _lookup_pose(self, frame_id: str | None) -> tuple:
        """Returns (x, y, z, qx, qy, qz, qw) for `base_link -> frame_id`.

        Returns all-zeros if the frame is unset, the lookup fails (TF not
        ready yet, or frame not in tree at eval time), or the interface is
        not connected.
        """
        if frame_id is None or self.ros2_interface is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        try:
            t = self.ros2_interface.tf_buffer.lookup_transform(
                "base_link", frame_id, Time()
            ).transform
        except TransformException:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            t.translation.x, t.translation.y, t.translation.z,
            t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w,
        )

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.last_controller_state or not self.last_joint_states:
            return {}

        tcp_pose = self.last_controller_state.tcp_pose
        tcp_velocity = self.last_controller_state.tcp_velocity
        tcp_error = self.last_controller_state.tcp_error
        joint_positions = self.last_joint_states.position
        controller_state_obs: ObservationState = {
            "tcp_pose.position.x": tcp_pose.position.x,
            "tcp_pose.position.y": tcp_pose.position.y,
            "tcp_pose.position.z": tcp_pose.position.z,
            "tcp_pose.orientation.x": tcp_pose.orientation.x,
            "tcp_pose.orientation.y": tcp_pose.orientation.y,
            "tcp_pose.orientation.z": tcp_pose.orientation.z,
            "tcp_pose.orientation.w": tcp_pose.orientation.w,
            "tcp_velocity.linear.x": tcp_velocity.linear.x,
            "tcp_velocity.linear.y": tcp_velocity.linear.y,
            "tcp_velocity.linear.z": tcp_velocity.linear.z,
            "tcp_velocity.angular.x": tcp_velocity.angular.x,
            "tcp_velocity.angular.y": tcp_velocity.angular.y,
            "tcp_velocity.angular.z": tcp_velocity.angular.z,
            "tcp_error.x": tcp_error[0],
            "tcp_error.y": tcp_error[1],
            "tcp_error.z": tcp_error[2],
            "tcp_error.rx": tcp_error[3],
            "tcp_error.ry": tcp_error[4],
            "tcp_error.rz": tcp_error[5],
            "joint_positions.0": joint_positions[0],
            "joint_positions.1": joint_positions[1],
            "joint_positions.2": joint_positions[2],
            "joint_positions.3": joint_positions[3],
            "joint_positions.4": joint_positions[4],
            "joint_positions.5": joint_positions[5],
            "joint_positions.6": joint_positions[6],
        }

        # Capture images from cameras
        cam_obs: dict[str, NDArray[Any]] = {}
        for cam_key, cam in self.cameras.items():
            try:
                data = cam.async_read(timeout_ms=2000)
                if data is not None and data.size > 0:
                    image_scale = self.config.camera_image_scaling[cam_key]
                    if image_scale != 1:
                        cam_obs[cam_key] = cv2.resize(
                            data,
                            None,
                            fx=image_scale,
                            fy=image_scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        cam_obs[cam_key] = data
                else:
                    logger.debug(
                        f"Camera {cam_key} data is empty (camera not ready yet?), using all-black placeholder."
                    )
                    cam_obs[cam_key] = np.zeros(
                        self._cameras_ft[cam_key], dtype=np.uint8
                    )
            except Exception as e:
                logger.error(f"Failed to read camera {cam_key}: {e}")

        # Tare-compensated wrench (zeros if not yet received).
        fx, fy, fz, tx, ty, tz = self._compensated_wrench()
        wrench_obs: ObservationWrench = {
            "wrench.fx": fx, "wrench.fy": fy, "wrench.fz": fz,
            "wrench.tx": tx, "wrench.ty": ty, "wrench.tz": tz,
        }

        # Ground-truth port and plug poses + insertion-success latch.
        px, py, pz, pqx, pqy, pqz, pqw = self._lookup_pose(self._active_port_frame)
        gx, gy, gz, gqx, gqy, gqz, gqw = self._lookup_pose(self._active_plug_frame)
        gt_obs: ObservationGroundTruth = {
            "groundtruth.port_pose.x": px, "groundtruth.port_pose.y": py,
            "groundtruth.port_pose.z": pz,
            "groundtruth.port_pose.qx": pqx, "groundtruth.port_pose.qy": pqy,
            "groundtruth.port_pose.qz": pqz, "groundtruth.port_pose.qw": pqw,
            "groundtruth.plug_pose.x": gx, "groundtruth.plug_pose.y": gy,
            "groundtruth.plug_pose.z": gz,
            "groundtruth.plug_pose.qx": gqx, "groundtruth.plug_pose.qy": gqy,
            "groundtruth.plug_pose.qz": gqz, "groundtruth.plug_pose.qw": gqw,
            "meta.insertion_success": float(self._inserted_flag),
        }

        obs = {**cam_obs, **controller_state_obs, **wrench_obs, **gt_obs}
        return obs

    def send_action_cartesian(self, action: dict[str, Any]) -> None:
        if not self._is_connected or not self.ros2_interface:
            raise DeviceNotConnectedError()

        motion_update_action = cast(MotionUpdateActionDict, action)

        twist_msg = Twist()

        try:
            twist_msg.linear.x = float(motion_update_action["linear.x"])
        except KeyError:
            raise KeyError(
                "Missing key 'linear.x'. If using `--teleop.type=aic_keyboard_joint`, have you set `--robot.teleop_target_mode=joint`?"
            ) from None
        twist_msg.linear.y = float(motion_update_action["linear.y"])
        twist_msg.linear.z = float(motion_update_action["linear.z"])
        twist_msg.angular.x = float(motion_update_action["angular.x"])
        twist_msg.angular.y = float(motion_update_action["angular.y"])
        twist_msg.angular.z = float(motion_update_action["angular.z"])

        msg = MotionUpdate()
        msg.header.stamp = self.ros2_interface.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.velocity = twist_msg
        msg.target_stiffness = np.diag([85.0, 85.0, 85.0, 85.0, 85.0, 85.0]).flatten()
        msg.target_damping = np.diag([75.0, 75.0, 75.0, 75.0, 75.0, 75.0]).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        self.ros2_interface.motion_update_pub.publish(msg)

    def send_action_joint(self, action: dict[str, Any]) -> None:
        if not self._is_connected or not self.ros2_interface:
            raise DeviceNotConnectedError()

        joint_motion_update_action = cast(JointMotionUpdateActionDict, action)
        msg = JointMotionUpdate()

        if "shoulder_pan_joint" not in joint_motion_update_action:
            raise KeyError(
                "Missing key 'shoulder_pan_joint'. If using `--teleop.type=aic_keyboard_ee` or `--teleop.type=aic_spacemouse`, have you set `--robot.teleop_target_mode=cartesian`?"
            )

        msg.target_state.velocities = list(action.values())

        msg.target_stiffness = [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
        msg.target_damping = [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        self.ros2_interface.joint_motion_update_pub.publish(msg)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.teleop_target_mode == "cartesian":
            self.send_action_cartesian(action)
            return action
        elif self.teleop_target_mode == "joint":
            self.send_action_joint(action)
            return action
        else:
            raise ValueError("Invalid teleop_target_mode")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()

        if self.ros2_interface:
            self.ros2_interface.node.destroy_node()
            self.ros2_interface.executor.shutdown()
            self.ros2_interface.executor_thread.join()
            self.ros2_interface = None

        self._is_connected = False
