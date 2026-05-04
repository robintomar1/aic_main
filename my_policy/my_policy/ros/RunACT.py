"""v9-act inference shim — loads our locally-trained ACT checkpoint and
runs it as an aic_model Policy against the eval container.

Differs from `aic_example_policies/ros/RunACT.py` (the reference shim) in:

  * Loads from a LOCAL checkpoint dir (`AIC_ACT_CHECKPOINT` env var, no
    HF download). Submission packaging will COPY the checkpoint into
    the model image.
  * 44-dim observation.state composed in the exact order produced by
    `build_act_dataset.py:KEEP_CHANNEL_GROUPS`:
        tcp_pose(7) ‖ tcp_velocity(6) ‖ tcp_error(6) ‖ joint_positions(7)
        ‖ wrench(6) ‖ task_vec(12)
    The wrench is tare-compensated against `controller_state.fts_tare_offset`
    (orientation-aware) — same formula the recorder used.
    The 12-dim task vector is one-hot module/port_in_module/port_type, built
    via `my_policy.act.labels.encode_task_vector` so the inference-side
    encoding stays in lockstep with the training-side labels.
  * 7-dim action interpreted as ABSOLUTE TCP pose target
    (pose.position.{x,y,z} + pose.orientation.{x,y,z,w}). The example
    treats it as a Twist; ours sends a Pose via `Policy.set_pose_target`
    (MotionUpdate / MODE_POSITION). The recorder verified action format
    is absolute pose: it subscribes to `/aic_controller/pose_commands` and
    stores the Pose fields directly (collect_lerobot.py:97-104).
  * Image scaling 0.25: the live cameras publish 1152W×1024H (verified in
    aic_assets/.../basler_camera_macro.xacro), and the dataset stores
    288W×256H — exactly 0.25 × native.
  * Loop at 20 Hz to match training fps; uses sim-time-aware sleep_for
    (NOT time.sleep — sim time can run faster or slower than wall).
  * Uses lerobot v0.5.1's `DataProcessorPipeline.from_pretrained()` for
    BOTH the input pipeline (rename → batch → device → normalize) and
    output pipeline (unnormalize → to-cpu). The example RunACT loads the
    normalizer .safetensors by hand and applies (x-mean)/std manually,
    which works but couples the shim to the saved pipeline composition;
    using the saved pipeline directly avoids that coupling.

Run-time configuration (env vars):
    AIC_ACT_CHECKPOINT  Path to the `.../checkpoints/last/pretrained_model/`
                        dir produced by lerobot training. Default points
                        at `/root/aic_data/v9_act_build/runs/v9_act_v1/...`.
    AIC_ACT_TIMEOUT_S   Per-trial inference budget. Default 30 s.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np
import torch
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, Quaternion
from safetensors.torch import load_file

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor.pipeline import DataProcessorPipeline

from my_policy.act.labels import encode_task_vector


DEFAULT_CHECKPOINT = (
    "/root/aic_data/v9_act_build/runs/v9_act_v1/checkpoints/last/pretrained_model"
)
DEFAULT_TIMEOUT_S = 30.0
LOOP_HZ = 20.0
IMAGE_SCALING = 0.25  # 1152x1024 native → 288x256 (matches dataset)


def _load_act_policy(ckpt_dir: Path, device: torch.device) -> ACTPolicy:
    """Load the ACT model weights + config from a checkpoint directory."""
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text())
    # `type` is a draccus-incompatible discriminator that lerobot adds for
    # the choice-class registry but draccus.decode rejects on the concrete
    # ACTConfig dataclass.
    cfg_dict.pop("type", None)
    config = draccus.decode(ACTConfig, cfg_dict)

    policy = ACTPolicy(config)
    policy.load_state_dict(load_file(str(ckpt_dir / "model.safetensors")))
    policy.eval()
    policy.to(device)
    return policy


def _ros_image_to_chw_float(ros_img, scaling: float) -> torch.Tensor:
    """ROS sensor_msgs/Image (uint8 RGB) → float32 [3,H,W] in [0,1].

    Returns UN-batched, UN-normalized — the preprocessor pipeline adds
    the batch dim and applies the saved per-camera normalization.
    """
    img_np = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(
        ros_img.height, ros_img.width, 3
    )
    if scaling != 1.0:
        img_np = cv2.resize(
            img_np, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA
        )
    return (
        torch.from_numpy(img_np.copy())  # copy: numpy buffer is read-only
        .permute(2, 0, 1)                 # HWC → CHW
        .float()
        .div_(255.0)
    )


def _compensated_wrench(obs_msg: Observation) -> tuple[float, float, float, float, float, float]:
    """Tare-compensated F/T (Fx,Fy,Fz,Tx,Ty,Tz) — same formula the
    recorder used (lerobot_robot_aic.aic_robot_aic_controller._compensated_wrench).

    The tare is orientation-aware (the controller updates fts_tare_offset
    as the wrist rotates), so we MUST subtract the live tare snapshot
    rather than a fixed startup zero.
    """
    raw = obs_msg.wrist_wrench.wrench
    tare = obs_msg.controller_state.fts_tare_offset.wrench
    return (
        raw.force.x - tare.force.x,
        raw.force.y - tare.force.y,
        raw.force.z - tare.force.z,
        raw.torque.x - tare.torque.x,
        raw.torque.y - tare.torque.y,
        raw.torque.z - tare.torque.z,
    )


def _build_state(obs_msg: Observation, task_vec: np.ndarray) -> torch.Tensor:
    """Compose the 44-dim observation.state vector in the EXACT order
    produced by build_act_dataset.py:KEEP_CHANNEL_GROUPS + task_vec.

    Returns float32 [44], un-batched, un-normalized.
    """
    if task_vec.shape != (12,):
        raise ValueError(f"task_vec must be shape (12,), got {task_vec.shape}")
    cs = obs_msg.controller_state
    tcp_pose = cs.tcp_pose
    tcp_vel = cs.tcp_velocity
    js = obs_msg.joint_states
    fx, fy, fz, tx, ty, tz = _compensated_wrench(obs_msg)

    state = np.array(
        [
            # tcp_pose (7)
            tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z,
            tcp_pose.orientation.x, tcp_pose.orientation.y,
            tcp_pose.orientation.z, tcp_pose.orientation.w,
            # tcp_velocity (6)
            tcp_vel.linear.x, tcp_vel.linear.y, tcp_vel.linear.z,
            tcp_vel.angular.x, tcp_vel.angular.y, tcp_vel.angular.z,
            # tcp_error (6)
            cs.tcp_error[0], cs.tcp_error[1], cs.tcp_error[2],
            cs.tcp_error[3], cs.tcp_error[4], cs.tcp_error[5],
            # joint_positions (7)
            *js.position[:7],
            # wrench (6) — compensated
            fx, fy, fz, tx, ty, tz,
            # task_vec (12)
            *task_vec.tolist(),
        ],
        dtype=np.float32,
    )
    assert state.shape == (44,), f"state must be 44-dim, got {state.shape}"
    return torch.from_numpy(state)


def _action_to_pose(action7: np.ndarray) -> Pose:
    """7-dim ACT output → geometry_msgs/Pose, with quaternion normalized.

    Network outputs are continuous-valued; the orientation components
    won't be unit-norm without explicit projection. Sending an
    un-normalized quaternion is undefined behavior for downstream IK.
    """
    if action7.shape != (7,):
        raise ValueError(f"expected (7,), got {action7.shape}")
    px, py, pz, qx, qy, qz, qw = (float(v) for v in action7)
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-8:
        # Fall back to identity rotation rather than producing NaNs;
        # this should never happen with a trained model.
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return Pose(
        position=Point(x=px, y=py, z=pz),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )


class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_dir = Path(os.environ.get("AIC_ACT_CHECKPOINT", DEFAULT_CHECKPOINT))
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"AIC_ACT_CHECKPOINT not found: {ckpt_dir}. "
                "Set the env var or place the checkpoint at the default path."
            )
        self.ckpt_dir = ckpt_dir

        self.policy = _load_act_policy(ckpt_dir, self.device)

        # Saved processor pipelines: pre = rename → batch → device → normalize;
        # post = unnormalize → to-cpu. Loaded straight from the train output;
        # the shim never touches stats by hand.
        self.preprocessor = DataProcessorPipeline.from_pretrained(
            str(ckpt_dir), config_filename="policy_preprocessor.json"
        )
        self.postprocessor = DataProcessorPipeline.from_pretrained(
            str(ckpt_dir), config_filename="policy_postprocessor.json"
        )

        self.timeout_s = float(os.environ.get("AIC_ACT_TIMEOUT_S", DEFAULT_TIMEOUT_S))
        self.loop_period_s = 1.0 / LOOP_HZ

        self.get_logger().info(
            f"RunACT loaded checkpoint={ckpt_dir} device={self.device} "
            f"loop={LOOP_HZ}Hz timeout={self.timeout_s}s"
        )

    def _build_obs_dict(
        self, obs_msg: Observation, task_vec: np.ndarray
    ) -> dict[str, torch.Tensor]:
        return {
            "observation.images.left_camera":
                _ros_image_to_chw_float(obs_msg.left_image, IMAGE_SCALING),
            "observation.images.center_camera":
                _ros_image_to_chw_float(obs_msg.center_image, IMAGE_SCALING),
            "observation.images.right_camera":
                _ros_image_to_chw_float(obs_msg.right_image, IMAGE_SCALING),
            "observation.state": _build_state(obs_msg, task_vec),
        }

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs: Any,
    ) -> bool:
        self.get_logger().info(
            f"RunACT.insert_cable: target_module={task.target_module_name} "
            f"port={task.port_name} port_type={task.port_type} "
            f"cable={task.cable_name}"
        )

        # Validates inputs and crashes loudly if Task fields don't match
        # the trained one-hot vocabulary — better than silently feeding
        # an all-zero task vector to the policy.
        task_vec = encode_task_vector(
            task.target_module_name, task.port_name, task.port_type
        )

        self.policy.reset()  # clears the n_action_steps queue from any prior trial

        start_t = self.time_now()
        ticks = 0
        while (self.time_now() - start_t).nanoseconds / 1e9 < self.timeout_s:
            obs_msg = get_observation()
            if obs_msg is None:
                self.sleep_for(self.loop_period_s)
                continue

            obs = self._build_obs_dict(obs_msg, task_vec)
            obs = self.preprocessor(obs)
            with torch.inference_mode():
                action = self.policy.select_action(obs)
            action = self.postprocessor(action)

            # Postprocessor moves to cpu; action shape [1, 7].
            a = action[0].numpy() if isinstance(action, torch.Tensor) \
                else action["action"][0].numpy()
            pose = _action_to_pose(a)
            self.set_pose_target(move_robot, pose, frame_id="base_link")

            send_feedback("running")
            ticks += 1
            self.sleep_for(self.loop_period_s)

        self.get_logger().info(f"RunACT.insert_cable: exit after {ticks} ticks")
        return True
