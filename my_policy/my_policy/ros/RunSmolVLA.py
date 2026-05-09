"""v9-port-local-smolvla inference shim — loads a port-local-trained
SmolVLA checkpoint and runs it as an aic_model Policy against the eval
container.

Architectural cousin of `RunPortLocalACT.py`. Differences:

  * **State is 32-dim (not 44).** SmolVLA gets task identity from the
    natural-language `task` string per call, not a one-hot in state.
    `make_smolvla_dataset.py` strips the trailing 12-dim task vector so
    state matches SmolVLA's default `max_state_dim=32`. The shim must
    compose the same 32-dim layout (no task-vec append) at inference.

  * **Language input.** SmolVLAPolicy expects a `task` string per call.
    The DataProcessorPipeline runs the SmolVLANewLineProcessor +
    TokenizerProcessorStep over `complementary_data["task"]`; the
    pipeline converter (`batch_to_transition`) lifts the `"task"` key
    from the obs dict into complementary_data. So we put the task
    string directly into the obs dict.

  * **SmolVLAPolicy in place of ACTPolicy** — same `select_action`
    contract, but the queue is filled by the flow-matching action head
    rather than the ACT decoder.

Identical to RunPortLocalACT for everything else: port-pose acquisition
(TF or localizer), per-tick port-local frame transform, action
round-trip back to base_link, normalized quaternion, controller
dispatch.

Run-time configuration (env vars):
    AIC_PL_SMOLVLA_CHECKPOINT   Path to .../checkpoints/<step>/pretrained_model/. Required.
    AIC_PL_SMOLVLA_TIMEOUT_S    Per-trial inference budget. Default 30 s.
    AIC_PL_LOCALIZER_CHECKPOINT If set, replaces /tf port lookup.
    AIC_PL_LOCALIZER_QUATS_JSON Optional sidecar quats for localizer.
    AIC_PL_LOCALIZER_DEVICE     cuda or cpu (default cuda).
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
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from safetensors.torch import load_file
from tf2_ros import TransformException

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)

from my_policy.act.labels import task_string_for
from my_policy.port_local.transforms import (
    FrameInputs,
    transform_frame,
    transform_pose_back_to_baselink,
)


# ---------------------------------------------------------------------------
# Constants — must match make_smolvla_dataset.py output (32-dim state).
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_S = 30.0
LOOP_HZ = 20.0
LOOP_PERIOD_S = 1.0 / LOOP_HZ

IMAGE_SCALING = 0.25  # 1152x1024 → 288x256 (matches dataset).

TF_LOOKUP_TIMEOUT_S = 5.0

STATE_DIM = 32  # 7 + 6 + 6 + 7 + 6, no task one-hot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_smolvla_policy(ckpt_dir: Path, device: torch.device) -> SmolVLAPolicy:
    """Load SmolVLA model + config from a checkpoint dir."""
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text())
    cfg_dict.pop("type", None)
    config = draccus.decode(SmolVLAConfig, cfg_dict)
    policy = SmolVLAPolicy(config)
    policy.load_state_dict(load_file(str(ckpt_dir / "model.safetensors")))
    policy.eval()
    policy.to(device)
    return policy


def _ros_image_to_chw_float(ros_img, scaling: float) -> torch.Tensor:
    img_np = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(
        ros_img.height, ros_img.width, 3
    )
    if scaling != 1.0:
        img_np = cv2.resize(
            img_np, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA
        )
    return (
        torch.from_numpy(img_np.copy())
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
    )


def _compensated_wrench(obs_msg: Observation) -> tuple[float, float, float, float, float, float]:
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


def _build_state_32(
    obs_msg: Observation,
    port_pose_baselink: np.ndarray,
    err_z_override: float | None = None,
) -> torch.Tensor:
    """Compose the 32-channel observation.state in the same layout as
    `make_smolvla_dataset.py` produces (port-local frame, no task vec).

    Layout:
      [ 0..6 ] tcp_pose      — port frame
      [ 7..12] tcp_velocity  — port frame
      [13..18] tcp_error     — TCP-relative; pass-through (or err_z override)
      [19..25] joint_pos     — frame-invariant; pass-through
      [26..31] wrench        — port frame (sensor frame ≈ TCP frame)

    `err_z_override`: if not None, overrides state[15] (tcp_error.err_z) to
    this value. Identified empirically via the phase-trigger probe as the
    sole channel gating the model's HOVER → COMMIT regime transition.
    Hover-regime training frames had err_z ≈ +5mm; commit-regime ≈ -3mm.
    Live policy gets stuck because its actual err_z stays at hover-pattern
    (target slightly above tcp), which keeps the model emitting hover-pattern
    commands — chicken-and-egg. Forcing err_z to a commit-pattern value
    breaks the cycle.

    Returns float32 [32], un-batched, un-normalized.
    """
    if port_pose_baselink.shape != (7,):
        raise ValueError(f"port_pose must be shape (7,), got {port_pose_baselink.shape}")

    cs = obs_msg.controller_state
    tcp_pose = cs.tcp_pose
    tcp_vel = cs.tcp_velocity
    js = obs_msg.joint_states

    tcp_pose_bl = np.array(
        [
            tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z,
            tcp_pose.orientation.x, tcp_pose.orientation.y,
            tcp_pose.orientation.z, tcp_pose.orientation.w,
        ],
        dtype=np.float64,
    )
    tcp_vel_bl = np.array(
        [
            tcp_vel.linear.x, tcp_vel.linear.y, tcp_vel.linear.z,
            tcp_vel.angular.x, tcp_vel.angular.y, tcp_vel.angular.z,
        ],
        dtype=np.float64,
    )
    wrench_sensor = np.array(_compensated_wrench(obs_msg), dtype=np.float64)

    inp = FrameInputs(
        tcp_pose_baselink=tcp_pose_bl,
        tcp_velocity_baselink=tcp_vel_bl,
        wrench_sensorframe=wrench_sensor,
        action_baselink=tcp_pose_bl,  # placeholder; we don't read action_portframe here
        port_pose_baselink=port_pose_baselink,
    )
    out = transform_frame(inp)

    err_z_value = float(cs.tcp_error[2]) if err_z_override is None else float(err_z_override)
    state = np.array(
        [
            *out.tcp_pose_portframe,         # 7
            *out.tcp_velocity_portframe,     # 6
            cs.tcp_error[0], cs.tcp_error[1], err_z_value,
            cs.tcp_error[3], cs.tcp_error[4], cs.tcp_error[5],  # 6
            *js.position[:7],                # 7
            *out.wrench_portframe,           # 6
        ],
        dtype=np.float32,
    )
    assert state.shape == (STATE_DIM,), f"state must be {STATE_DIM}-dim, got {state.shape}"
    return torch.from_numpy(state)


def _action_port_to_baselink_pose(
    action_port_7d: np.ndarray, port_pose_baselink: np.ndarray,
) -> Pose:
    if action_port_7d.shape != (7,):
        raise ValueError(f"expected (7,), got {action_port_7d.shape}")
    # Flow-matching's iterative denoising can emit near-zero quats early in
    # training or under OOD inputs; transform_pose_back_to_baselink → make_se3
    # would raise. Sub the identity quat instead so the trial degrades to
    # "hold orientation" rather than crashing mid-trial.
    in_q = action_port_7d[3:7]
    in_qnorm = float(np.linalg.norm(in_q))
    if in_qnorm < 1e-6:
        action_port_7d = action_port_7d.copy()
        action_port_7d[3:7] = [0.0, 0.0, 0.0, 1.0]
    recon = transform_pose_back_to_baselink(
        action_port_7d.astype(np.float64),
        port_pose_baselink.astype(np.float64),
    )
    px, py, pz, qx, qy, qz, qw = (float(v) for v in recon)
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm < 1e-8:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return Pose(
        position=Point(x=px, y=py, z=pz),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )


# ---------------------------------------------------------------------------
# RunSmolVLA — the Policy class loaded by aic_model.
# ---------------------------------------------------------------------------


class RunSmolVLA(Policy):
    CHECKPOINT_ENV = "AIC_PL_SMOLVLA_CHECKPOINT"
    TIMEOUT_ENV = "AIC_PL_SMOLVLA_TIMEOUT_S"

    # Phase-trigger override (see _build_state_32 docstring). If set, replaces
    # state[15] (tcp_error.err_z) with this value at every tick. Empirically
    # identified as the SOLE channel gating HOVER → COMMIT regime in trained
    # SmolVLA. Hover ≈ +0.005, commit ≈ -0.003. Default unset (use live err_z).
    ERR_Z_OVERRIDE_ENV = "AIC_PL_SMOLVLA_ERR_Z_OVERRIDE"

    LOCALIZER_CKPT_ENV = "AIC_PL_LOCALIZER_CHECKPOINT"
    LOCALIZER_QUATS_ENV = "AIC_PL_LOCALIZER_QUATS_JSON"
    LOCALIZER_DEVICE_ENV = "AIC_PL_LOCALIZER_DEVICE"

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        ckpt_dir = os.environ.get(self.CHECKPOINT_ENV, "").strip()
        if not ckpt_dir:
            raise ValueError(
                f"{self.CHECKPOINT_ENV} env var is required — set it to "
                f"the .../checkpoints/<step>/pretrained_model/ dir."
            )
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint dir does not exist: {ckpt_path}")
        self.ckpt_dir = ckpt_path

        self.policy = _load_smolvla_policy(ckpt_path, self.device)

        self.preprocessor = DataProcessorPipeline.from_pretrained(
            str(ckpt_path), config_filename="policy_preprocessor.json"
        )
        self.postprocessor = DataProcessorPipeline.from_pretrained(
            str(ckpt_path),
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        self.timeout_s = float(
            os.environ.get(self.TIMEOUT_ENV, str(DEFAULT_TIMEOUT_S))
        )
        self.loop_period_s = LOOP_PERIOD_S

        err_z_str = os.environ.get(self.ERR_Z_OVERRIDE_ENV, "").strip()
        self.err_z_override = float(err_z_str) if err_z_str else None

        self._localizer = None

        self.get_logger().info(
            f"RunSmolVLA loaded checkpoint={ckpt_path} "
            f"device={self.device} loop={LOOP_HZ}Hz "
            f"timeout={self.timeout_s}s "
            f"localizer_mode={self._localizer_mode()} "
            f"err_z_override={self.err_z_override}"
        )

    # ------------------------------------------------------------------
    # Port-pose acquisition (mirrors RunPortLocalACT).
    # ------------------------------------------------------------------

    def _localizer_mode(self) -> bool:
        return bool(os.environ.get(self.LOCALIZER_CKPT_ENV, "").strip())

    def _ensure_localizer_loaded(self) -> None:
        if self._localizer is not None:
            return
        ckpt = os.environ[self.LOCALIZER_CKPT_ENV]
        quats = os.environ.get(self.LOCALIZER_QUATS_ENV) or None
        device = os.environ.get(self.LOCALIZER_DEVICE_ENV, "cuda")
        from my_policy.localizer.inference import PortLocalizer  # noqa: WPS433
        self._localizer = PortLocalizer(
            checkpoint_path=Path(ckpt),
            device=device,
            quats_json_path=Path(quats) if quats else None,
        )
        self.get_logger().info(
            f"PortLocalizer loaded ckpt={ckpt} device={device} "
            f"cameras={self._localizer.cameras}"
        )

    @staticmethod
    def _ros_image_to_np_uint8(img_msg) -> np.ndarray:
        if img_msg.encoding != "rgb8":
            raise ValueError(
                f"expected rgb8 encoding for localizer input, got "
                f"{img_msg.encoding!r}"
            )
        return np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3,
        )

    def _wait_for_tf(
        self, target_frame: str, source_frame: str,
        timeout_s: float = TF_LOOKUP_TIMEOUT_S,
    ) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_s)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame, source_frame, Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' → "
                        f"'{target_frame}' — running with "
                        f"ground_truth:=true?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"transform '{source_frame}' not available after {timeout_s}s"
        )
        return False

    @staticmethod
    def _compose_port_tf_frame(task: Task) -> str:
        return f"task_board/{task.target_module_name}/{task.port_name}_link"

    def _acquire_port_pose_baselink(
        self, task: Task, get_observation: GetObservationCallback,
    ) -> np.ndarray | None:
        if self._localizer_mode():
            return self._predict_port_pose_via_localizer(task, get_observation)
        return self._lookup_port_pose_via_tf(task)

    def _lookup_port_pose_via_tf(self, task: Task) -> np.ndarray | None:
        port_frame = self._compose_port_tf_frame(task)
        if not self._wait_for_tf("base_link", port_frame):
            return None
        try:
            stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"TF lookup failed: {ex}")
            return None
        t = stamped.transform.translation
        q = stamped.transform.rotation
        pose = np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w], dtype=np.float64)
        self.get_logger().info(
            f"port_pose (TF lookup, {port_frame}): "
            f"xyz=({pose[0]:+.4f},{pose[1]:+.4f},{pose[2]:+.4f}) "
            f"q=({pose[3]:+.3f},{pose[4]:+.3f},{pose[5]:+.3f},{pose[6]:+.3f})"
        )
        return pose

    def _predict_port_pose_via_localizer(
        self, task: Task, get_observation: GetObservationCallback,
    ) -> np.ndarray | None:
        self._ensure_localizer_loaded()
        obs = get_observation()
        if obs is None:
            self.get_logger().error(
                "no observation available for localizer prediction"
            )
            return None
        images = {
            "left_camera": self._ros_image_to_np_uint8(obs.left_image),
            "center_camera": self._ros_image_to_np_uint8(obs.center_image),
            "right_camera": self._ros_image_to_np_uint8(obs.right_image),
        }
        tcp = obs.controller_state.tcp_pose
        tcp_vec = np.array(
            [
                tcp.position.x, tcp.position.y, tcp.position.z,
                tcp.orientation.x, tcp.orientation.y,
                tcp.orientation.z, tcp.orientation.w,
            ],
            dtype=np.float32,
        )
        try:
            pred = self._localizer.predict_port_pose(
                images=images,
                tcp_pose=tcp_vec,
                target_module_name=task.target_module_name,
                port_name=task.port_name,
                port_type=task.port_type,
            )
        except Exception as ex:
            self.get_logger().error(f"PortLocalizer prediction failed: {ex}")
            return None
        pose = np.array(
            [pred.x, pred.y, pred.z, pred.qx, pred.qy, pred.qz, pred.qw],
            dtype=np.float64,
        )
        self.get_logger().info(
            f"port_pose (localizer): "
            f"xyz=({pose[0]:+.4f},{pose[1]:+.4f},{pose[2]:+.4f}) "
            f"q=({pose[3]:+.3f},{pose[4]:+.3f},{pose[5]:+.3f},{pose[6]:+.3f})"
        )
        return pose

    # ------------------------------------------------------------------
    # The trial loop.
    # ------------------------------------------------------------------

    def _build_obs_dict(
        self,
        obs_msg: Observation,
        task_str: str,
        port_pose_baselink: np.ndarray,
    ) -> dict[str, Any]:
        # `task` is lifted into complementary_data by the pipeline's
        # batch_to_transition, then consumed by SmolVLANewLineProcessor +
        # TokenizerProcessorStep.
        return {
            "observation.images.left_camera":
                _ros_image_to_chw_float(obs_msg.left_image, IMAGE_SCALING),
            "observation.images.center_camera":
                _ros_image_to_chw_float(obs_msg.center_image, IMAGE_SCALING),
            "observation.images.right_camera":
                _ros_image_to_chw_float(obs_msg.right_image, IMAGE_SCALING),
            "observation.state": _build_state_32(
                obs_msg, port_pose_baselink, err_z_override=self.err_z_override),
            "task": task_str,
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
            f"RunSmolVLA.insert_cable: target_module={task.target_module_name} "
            f"port={task.port_name} port_type={task.port_type} "
            f"cable={task.cable_name}"
        )

        # Build the language instruction once per trial — same convention as
        # the dataset's per-episode `tasks` field (task_string_for).
        task_str = task_string_for(
            task.target_module_name, task.port_name, task.port_type,
        )
        self.get_logger().info(f"task_str: {task_str!r}")

        # Step 1: port pose (cached for the whole trial).
        port_pose = self._acquire_port_pose_baselink(task, get_observation)
        if port_pose is None:
            send_feedback("aborted: port pose unavailable")
            return False

        # Step 2: per-trial reset of the action queue.
        self.policy.reset()

        # Step 3: 20 Hz loop.
        start_t = self.time_now()
        ticks = 0
        none_obs_count = 0
        LOG_EVERY_N = 10
        last_action_port: np.ndarray | None = None
        max_action_delta = 0.0

        while (self.time_now() - start_t).nanoseconds / 1e9 < self.timeout_s:
            obs_msg = get_observation()
            if obs_msg is None:
                none_obs_count += 1
                self.sleep_for(self.loop_period_s)
                continue

            obs = self._build_obs_dict(obs_msg, task_str, port_pose)
            obs = self.preprocessor(obs)
            with torch.inference_mode():
                action = self.policy.select_action(obs)
            action = self.postprocessor(action)
            a_port = action[0].cpu().numpy()[:7]  # SmolVLA pads to max_action_dim=32

            pose = _action_port_to_baselink_pose(
                a_port.astype(np.float64), port_pose,
            )
            self.set_pose_target(move_robot, pose, frame_id="base_link")

            tcp = obs_msg.controller_state.tcp_pose.position
            tcp_pred_dist = float(np.linalg.norm(
                np.array([pose.position.x, pose.position.y, pose.position.z])
                - np.array([tcp.x, tcp.y, tcp.z])
            ))
            if last_action_port is not None:
                d = float(np.linalg.norm(a_port - last_action_port))
                max_action_delta = max(max_action_delta, d)
            last_action_port = a_port

            if ticks % LOG_EVERY_N == 0:
                self.get_logger().info(
                    f"tick={ticks:4d} "
                    f"tcp=({tcp.x:.3f},{tcp.y:.3f},{tcp.z:.3f}) "
                    f"pred_bl=({pose.position.x:.3f},"
                    f"{pose.position.y:.3f},{pose.position.z:.3f}) "
                    f"||pred-tcp||={tcp_pred_dist*1000:.1f}mm "
                    f"a_port[xyz]=({a_port[0]:.3f},{a_port[1]:.3f},{a_port[2]:.3f}) "
                    f"max_a_step_Δ={max_action_delta:.4f}"
                )

            send_feedback("running")
            ticks += 1
            self.sleep_for(self.loop_period_s)

        self.get_logger().info(
            f"RunSmolVLA.insert_cable: exit after {ticks} ticks "
            f"(none_obs={none_obs_count}, max_a_step_Δ={max_action_delta:.4f})"
        )
        return True
