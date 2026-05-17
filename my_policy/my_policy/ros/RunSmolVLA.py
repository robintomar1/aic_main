"""v9-port-local-smolvla inference shim — loads a port-local-trained
SmolVLA checkpoint and runs it as an aic_model Policy against the eval
container.

Architectural cousin of `RunPortLocalACT.py`. Differences:

  * **State is 26-dim (not 44).** SmolVLA gets task identity from the
    natural-language `task` string per call, not a one-hot in state.
    AND we drop the 6-dim `tcp_error` block — it is auto-regressive on
    the policy's commanded targets (the controller's tracking residual)
    and the phase-trigger probe identified state[15] = tcp_error.z as
    the channel the previously-trained model used as a hover-vs-commit
    shortcut. `make_smolvla_dataset.py` performs the same drop on the
    training data; the shim composes the matching 26-dim layout here.

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
    AIC_PL_SMOLVLA_N_ACTION_STEPS  If set, overrides cfg.n_action_steps at
                                inference (default = chunk_size = 50). Smaller
                                values force the model to re-run on fresh
                                observations more often. Set to 1 for fully
                                closed-loop control IF SmolVLA inference
                                latency stays under the 50 ms / tick budget.
    AIC_PL_LOCALIZER_CHECKPOINT If set, replaces /tf port lookup.
    AIC_PL_LOCALIZER_QUATS_JSON Optional sidecar quats for localizer.
    AIC_PL_LOCALIZER_DEVICE     cuda or cpu (default cuda).
"""
from __future__ import annotations

import json
import os
import threading
import time as _time
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

STATE_DIM = 26  # 7 tcp_pose + 6 tcp_velocity + 7 joints + 6 wrench (no tcp_error, no task one-hot)

# Joint order the dataset was recorded in (alphabetical, as the sim's
# /joint_states publisher emitted during collection). Inference-time
# /joint_states arrives in URDF-kinematic order
# (shoulder_pan, shoulder_lift, elbow, wrist_1..3, gripper), so we must
# look up by name and reorder before composing the state vector.
RECORDED_JOINT_ORDER = (
    "elbow_joint",
    "gripper",
    "shoulder_lift_joint",
    "shoulder_pan_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _joint_positions_in_recorded_order(js) -> np.ndarray:
    """Map sensor_msgs/JointState (any name order) → 7-vec in dataset order.

    Raises if any expected joint is missing — silent fallback would feed the
    policy a wrong-channel state and produce subtle action errors.
    """
    name_to_pos = dict(zip(js.name, js.position))
    missing = [n for n in RECORDED_JOINT_ORDER if n not in name_to_pos]
    if missing:
        raise ValueError(
            f"/joint_states missing expected joints {missing}; "
            f"got names={list(js.name)}"
        )
    return np.array(
        [name_to_pos[n] for n in RECORDED_JOINT_ORDER], dtype=np.float64,
    )

# RTC (Real-Time Chunking) defaults — see lerobot/policies/rtc.
# RTC overlaps chunk generation with dispatch and inpaints the leading
# steps of each new chunk against the executed tail of the previous chunk.
# Eliminates the ~700 ms dispatch gap that select_action produces every
# n_action_steps ticks.
RTC_DEFAULT_INFERENCE_DELAY = 14         # ≈ 700 ms / 50 ms per tick
RTC_DEFAULT_EXECUTION_HORIZON = 10
RTC_DEFAULT_GUIDANCE_WEIGHT = 10.0
RTC_DEFAULT_ATTENTION_SCHEDULE = "EXP"   # one of LINEAR / EXP / ONES / ZEROS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_smolvla_policy(
    ckpt_dir: Path,
    device: torch.device,
    n_action_steps_override: int | None = None,
    rtc_config: Any | None = None,
) -> SmolVLAPolicy:
    """Load SmolVLA model + config from a checkpoint dir.

    `n_action_steps_override`: if set, overrides cfg.n_action_steps before
    instantiating the policy. SmolVLA's default n_action_steps == chunk_size
    (50 at training time) means the policy runs open-loop on its predicted
    chunk between forward passes — at 20 Hz that's 2.5 s of stale plan.

    `rtc_config`: optional `lerobot.policies.rtc.configuration_rtc.RTCConfig`
    instance. When provided, attached to the SmolVLAConfig before instantiating
    the policy so its `_rtc_enabled()` predicate sees the flag and the
    flow-matching head pulls in the RTCProcessor. Inference must then go
    through `predict_action_chunk` (not `select_action`, which hard-asserts
    RTC off).
    """
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text())
    cfg_dict.pop("type", None)
    if n_action_steps_override is not None:
        cfg_dict["n_action_steps"] = int(n_action_steps_override)
    # rtc_config can't be round-tripped through draccus.decode (it expects
    # a dict, and we have a fully-built instance), so strip any pre-existing
    # entry and attach the live instance after decode.
    cfg_dict.pop("rtc_config", None)
    config = draccus.decode(SmolVLAConfig, cfg_dict)
    if rtc_config is not None:
        config.rtc_config = rtc_config
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


def _build_state_26(
    obs_msg: Observation,
    port_pose_baselink: np.ndarray,
) -> torch.Tensor:
    """Compose the 26-channel observation.state in the same layout as
    `make_smolvla_dataset.py` produces (port-local frame, no tcp_error,
    no task vec).

    Layout:
      [ 0..6 ] tcp_pose      — port frame
      [ 7..12] tcp_velocity  — port frame
      [13..19] joint_pos     — frame-invariant; pass-through
      [20..25] wrench        — port frame (sensor frame ≈ TCP frame)

    tcp_error is intentionally NOT in this layout. See module docstring.

    Returns float32 [26], un-batched, un-normalized.
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

    joint_positions = _joint_positions_in_recorded_order(js)
    state = np.array(
        [
            *out.tcp_pose_portframe,         # 7  → [0..6]
            *out.tcp_velocity_portframe,     # 6  → [7..12]
            *joint_positions,                # 7  → [13..19]
            *out.wrench_portframe,           # 6  → [20..25]
        ],
        dtype=np.float32,
    )
    assert state.shape == (STATE_DIM,), f"state must be {STATE_DIM}-dim, got {state.shape}"
    if not getattr(_build_state_26, "_call_count", 0):
        _build_state_26._call_count = 0  # type: ignore[attr-defined]
    _build_state_26._call_count += 1  # type: ignore[attr-defined]
    if _build_state_26._call_count <= 3 or _build_state_26._call_count % 50 == 0:
        s = state
        print(f"[state_dbg #{_build_state_26._call_count}] "
              f"tcp=({s[0]:+.3f},{s[1]:+.3f},{s[2]:+.3f})  "
              f"vel=({s[7]*1000:+.1f},{s[8]*1000:+.1f},{s[9]*1000:+.1f})mm/s  "
              f"|F|={np.linalg.norm(s[20:23]):.2f}N",
              flush=True)
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

    # If set, overrides cfg.n_action_steps at inference. See
    # `_load_smolvla_policy` docstring for tradeoffs.
    N_ACTION_STEPS_ENV = "AIC_PL_SMOLVLA_N_ACTION_STEPS"

    LOCALIZER_CKPT_ENV = "AIC_PL_LOCALIZER_CHECKPOINT"
    LOCALIZER_QUATS_ENV = "AIC_PL_LOCALIZER_QUATS_JSON"
    LOCALIZER_DEVICE_ENV = "AIC_PL_LOCALIZER_DEVICE"

    # RTC opt-in. Setting "1" routes inference through predict_action_chunk +
    # ActionQueue + a producer thread, with prefix-attention guidance against
    # the previous chunk's executed tail.
    RTC_ENABLED_ENV = "AIC_PL_SMOLVLA_RTC"
    RTC_INFERENCE_DELAY_ENV = "AIC_PL_SMOLVLA_RTC_INFERENCE_DELAY"
    RTC_EXECUTION_HORIZON_ENV = "AIC_PL_SMOLVLA_RTC_EXECUTION_HORIZON"
    RTC_GUIDANCE_WEIGHT_ENV = "AIC_PL_SMOLVLA_RTC_GUIDANCE_WEIGHT"
    RTC_ATTENTION_SCHEDULE_ENV = "AIC_PL_SMOLVLA_RTC_ATTENTION_SCHEDULE"

    # SKIP_CHUNK opt-in. Setting to a positive integer N routes inference
    # through predict_action_chunk and drops the FIRST N actions of every
    # chunk before dispatch. Purely synchronous — no async producer, no RTC
    # guidance, no smoothing. Mutually exclusive with RTC. Useful for
    # treating the leading actions of each chunk as "stale" and only
    # executing the later, more-considered tail.
    SKIP_N_ENV = "AIC_PL_SMOLVLA_SKIP_N"

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

        n_steps_str = os.environ.get(self.N_ACTION_STEPS_ENV, "").strip()
        n_action_steps_override = int(n_steps_str) if n_steps_str else None

        # ---- RTC config (opt-in) --------------------------------------
        self.rtc_enabled = os.environ.get(self.RTC_ENABLED_ENV, "").strip() == "1"
        rtc_config = None
        self.rtc_inference_delay = RTC_DEFAULT_INFERENCE_DELAY
        if self.rtc_enabled:
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            schedule_name = os.environ.get(
                self.RTC_ATTENTION_SCHEDULE_ENV, RTC_DEFAULT_ATTENTION_SCHEDULE,
            ).upper()
            rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=int(os.environ.get(
                    self.RTC_EXECUTION_HORIZON_ENV, RTC_DEFAULT_EXECUTION_HORIZON,
                )),
                max_guidance_weight=float(os.environ.get(
                    self.RTC_GUIDANCE_WEIGHT_ENV, RTC_DEFAULT_GUIDANCE_WEIGHT,
                )),
                prefix_attention_schedule=RTCAttentionSchedule[schedule_name],
            )
            self.rtc_inference_delay = int(os.environ.get(
                self.RTC_INFERENCE_DELAY_ENV, RTC_DEFAULT_INFERENCE_DELAY,
            ))

        # ---- SKIP_CHUNK config (opt-in) -------------------------------
        skip_n_str = os.environ.get(self.SKIP_N_ENV, "").strip()
        self.skip_n_chunks = int(skip_n_str) if skip_n_str else 0
        if self.skip_n_chunks < 0:
            raise ValueError(
                f"{self.SKIP_N_ENV}={self.skip_n_chunks} must be >= 0"
            )
        self.skip_chunk_enabled = self.skip_n_chunks > 0
        if self.skip_chunk_enabled and self.rtc_enabled:
            raise ValueError(
                f"{self.SKIP_N_ENV} and {self.RTC_ENABLED_ENV} are mutually "
                f"exclusive — pick one mode"
            )

        self.policy = _load_smolvla_policy(
            ckpt_path, self.device, n_action_steps_override, rtc_config=rtc_config,
        )
        # Validate skip_n against chunk_size now that the policy is loaded.
        if self.skip_chunk_enabled and self.skip_n_chunks >= self.policy.config.chunk_size:
            raise ValueError(
                f"{self.SKIP_N_ENV}={self.skip_n_chunks} must be < "
                f"chunk_size={self.policy.config.chunk_size}"
            )

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

        self._localizer = None

        self.get_logger().info(
            f"RunSmolVLA loaded checkpoint={ckpt_path} "
            f"device={self.device} loop={LOOP_HZ}Hz "
            f"timeout={self.timeout_s}s "
            f"n_action_steps={self.policy.config.n_action_steps} "
            f"chunk_size={self.policy.config.chunk_size} "
            f"localizer_mode={self._localizer_mode()} "
            f"rtc_enabled={self.rtc_enabled} "
            f"skip_chunk_enabled={self.skip_chunk_enabled}"
            + (
                f" skip_n={self.skip_n_chunks}"
                if self.skip_chunk_enabled else ""
            )
            + (
                f" rtc_inference_delay={self.rtc_inference_delay}"
                f" rtc_execution_horizon={self.policy.config.rtc_config.execution_horizon}"
                f" rtc_guidance_weight={self.policy.config.rtc_config.max_guidance_weight}"
                f" rtc_schedule={self.policy.config.rtc_config.prefix_attention_schedule}"
                if self.rtc_enabled else ""
            )
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
            "observation.state": _build_state_26(
                obs_msg, port_pose_baselink),
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
        if self.rtc_enabled:
            return self._insert_cable_rtc(
                task, get_observation, move_robot, send_feedback,
            )
        if self.skip_chunk_enabled:
            return self._insert_cable_skip_chunk(
                task, get_observation, move_robot, send_feedback,
            )
        return self._insert_cable_sync(
            task, get_observation, move_robot, send_feedback,
        )

    # ------------------------------------------------------------------
    # Synchronous loop (no RTC). Inference runs on the main thread; every
    # n_action_steps ticks the policy blocks ~700 ms producing a fresh chunk.
    # Compensated sleep keeps the cheap-tick cadence at true 20 Hz.
    # ------------------------------------------------------------------

    def _insert_cable_sync(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        task_str = task_string_for(
            task.target_module_name, task.port_name, task.port_type,
        )
        self.get_logger().info(f"task_str: {task_str!r}")

        port_pose = self._acquire_port_pose_baselink(task, get_observation)
        if port_pose is None:
            send_feedback("aborted: port pose unavailable")
            return False

        self.policy.reset()

        start_t = self.time_now()
        period_ns = int(self.loop_period_s * 1e9)
        next_deadline = start_t + Duration(nanoseconds=period_ns)
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
                next_deadline = self.time_now() + Duration(nanoseconds=period_ns)
                continue

            t_obs_built_start = _time.perf_counter()
            obs = self._build_obs_dict(obs_msg, task_str, port_pose)
            obs = self.preprocessor(obs)
            t_pre_end = _time.perf_counter()
            # One-shot dump of post-preprocess obs shapes/dtypes/devices
            # to compare against the offline probe. If image shapes are
            # bigger here than in the probe, SmolVLM2 tiling is producing
            # extra sub-crops and that's the source of the 4× slowdown.
            if ticks == 0:
                shape_lines = []
                for k in sorted(obs.keys()):
                    v = obs[k]
                    if isinstance(v, torch.Tensor):
                        shape_lines.append(
                            f"    {k}: shape={tuple(v.shape)} "
                            f"dtype={v.dtype} device={v.device} "
                            f"min={float(v.float().min()):+.3f} "
                            f"max={float(v.float().max()):+.3f}"
                        )
                    else:
                        shape_lines.append(f"    {k}: type={type(v).__name__} val={v!r}")
                self.get_logger().info(
                    "obs after preprocessor (one-shot dump):\n" + "\n".join(shape_lines)
                )
            # GPU-side timing via CUDA events — measures actual SM/kernel
            # time independently of host-thread wall clock. If gpu_ms ≪ inf_ms
            # the slowdown is host-side (GIL, scheduling), not the model.
            use_cuda_evt = torch.cuda.is_available()
            if use_cuda_evt:
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_start.record()
            with torch.inference_mode():
                action = self.policy.select_action(obs)
            if use_cuda_evt:
                ev_end.record()
                torch.cuda.synchronize()
                gpu_ms = ev_start.elapsed_time(ev_end)
            else:
                gpu_ms = float("nan")
            t_inf_end = _time.perf_counter()
            action = self.postprocessor(action)
            a_port = action[0].cpu().numpy()[:7]  # SmolVLA pads to max_action_dim=32
            t_post_end = _time.perf_counter()

            pose = _action_port_to_baselink_pose(
                a_port.astype(np.float64), port_pose,
            )
            self.set_pose_target(move_robot, pose, frame_id="base_link")
            t_dispatch_end = _time.perf_counter()

            pre_ms = (t_pre_end - t_obs_built_start) * 1000.0
            inf_ms = (t_inf_end - t_pre_end) * 1000.0
            post_ms = (t_post_end - t_inf_end) * 1000.0
            disp_ms = (t_dispatch_end - t_post_end) * 1000.0
            total_ms = (t_dispatch_end - t_obs_built_start) * 1000.0

            tcp = obs_msg.controller_state.tcp_pose.position
            tcp_pred_dist = float(np.linalg.norm(
                np.array([pose.position.x, pose.position.y, pose.position.z])
                - np.array([tcp.x, tcp.y, tcp.z])
            ))
            if last_action_port is not None:
                d = float(np.linalg.norm(a_port - last_action_port))
                max_action_delta = max(max_action_delta, d)
            last_action_port = a_port

            # Log timing on every tick where inference exceeded a tick budget
            # so the inference ticks always show up; otherwise log periodically.
            timing_tag = "INF" if inf_ms > 60.0 else "   "
            if ticks % LOG_EVERY_N == 0 or inf_ms > 60.0:
                self.get_logger().info(
                    f"tick={ticks:4d} {timing_tag} "
                    f"t[pre={pre_ms:5.1f} inf={inf_ms:6.1f}(gpu={gpu_ms:5.1f}) "
                    f"post={post_ms:4.1f} disp={disp_ms:4.1f} tot={total_ms:6.1f}]ms "
                    f"tcp=({tcp.x:.3f},{tcp.y:.3f},{tcp.z:.3f}) "
                    f"pred_bl=({pose.position.x:.3f},"
                    f"{pose.position.y:.3f},{pose.position.z:.3f}) "
                    f"||pred-tcp||={tcp_pred_dist*1000:.1f}mm "
                    f"a_port[xyz]=({a_port[0]:.3f},{a_port[1]:.3f},{a_port[2]:.3f}) "
                    f"max_a_step_Δ={max_action_delta:.4f}"
                )

            send_feedback("running")
            ticks += 1

            # Compensated sleep — schedule next tick at fixed period offset
            # from start, not period offset from "now after work." Without
            # this the cheap ticks drift to ~70 ms each (50 ms sleep + 20 ms
            # of image/preproc/dispatch work), turning 50 ticks into 3.5 s
            # instead of 2.5 s.
            now = self.time_now()
            remaining_ns = (next_deadline - now).nanoseconds
            if remaining_ns > 0:
                self.sleep_for(remaining_ns / 1e9)
            next_deadline = next_deadline + Duration(nanoseconds=period_ns)

        self.get_logger().info(
            f"RunSmolVLA.insert_cable[sync]: exit after {ticks} ticks "
            f"(none_obs={none_obs_count}, max_a_step_Δ={max_action_delta:.4f})"
        )
        return True

    # ------------------------------------------------------------------
    # RTC loop. Producer thread runs predict_action_chunk in a tight loop,
    # feeding an ActionQueue; the main 20 Hz consumer just pops + dispatches.
    # RTCProcessor inpaints the leading `inference_delay` steps of each new
    # chunk against the executed tail of the previous chunk so the seam is
    # smooth — no command gap, no plan discontinuity.
    # ------------------------------------------------------------------

    def _insert_cable_rtc(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        from lerobot.policies.rtc.action_queue import ActionQueue  # noqa: WPS433

        task_str = task_string_for(
            task.target_module_name, task.port_name, task.port_type,
        )
        self.get_logger().info(f"task_str: {task_str!r}  [RTC mode]")

        port_pose = self._acquire_port_pose_baselink(task, get_observation)
        if port_pose is None:
            send_feedback("aborted: port pose unavailable")
            return False

        self.policy.reset()
        action_queue = ActionQueue(self.policy.config.rtc_config)
        inference_delay = self.rtc_inference_delay

        stop_evt = threading.Event()
        err_box: list[BaseException] = []
        stats = {"chunks": 0, "producer_iters": 0}

        def _postproc_chunk(actions_btd: torch.Tensor) -> torch.Tensor:
            """Apply the per-action postprocessor (un-normalize) across an
            entire chunk. Input (1, T, A); output (T, A) on CPU."""
            out = []
            for t in range(actions_btd.shape[1]):
                out.append(self.postprocessor(actions_btd[:, t, :]))
            return torch.stack(out, dim=1).squeeze(0).detach().cpu()

        def producer():
            try:
                while not stop_evt.is_set():
                    obs_msg = get_observation()
                    if obs_msg is None:
                        if stop_evt.wait(0.005):
                            return
                        continue
                    prev_left = action_queue.get_left_over()  # (T_left, A) or None
                    obs = self._build_obs_dict(obs_msg, task_str, port_pose)
                    obs = self.preprocessor(obs)
                    prev_left_arg = None
                    if prev_left is not None and prev_left.numel() > 0:
                        prev_left_arg = prev_left.unsqueeze(0).to(self.device)
                    # NB: do NOT wrap in torch.inference_mode() here. RTC's
                    # denoise_step uses torch.enable_grad() to compute its
                    # inpainting correction via torch.autograd.grad; inference_mode
                    # is stricter than no_grad and cannot be overridden by
                    # enable_grad, so wrapping here breaks RTC. predict_action_chunk
                    # itself is @torch.no_grad() decorated (which enable_grad
                    # CAN override), so memory/perf is still bounded.
                    actions = self.policy.predict_action_chunk(
                        obs,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_left_arg,
                    )
                    # actions: (1, T, A_padded). Original kept for RTC merge
                    # math, processed is what the consumer will dispatch.
                    original = actions.squeeze(0).detach().cpu()  # (T, A)
                    processed = _postproc_chunk(actions)         # (T, A)
                    action_queue.merge(original, processed, inference_delay)
                    stats["chunks"] += 1
                    stats["producer_iters"] += 1
            except BaseException as exc:  # noqa: BLE001
                err_box.append(exc)

        producer_thread = threading.Thread(
            target=producer, name="RunSmolVLA-RTC-Producer", daemon=True,
        )
        producer_thread.start()

        start_t = self.time_now()
        period_ns = int(self.loop_period_s * 1e9)
        next_deadline = start_t + Duration(nanoseconds=period_ns)
        ticks = 0
        no_action_ticks = 0
        LOG_EVERY_N = 20
        last_a_port: np.ndarray | None = None
        max_a_step_delta = 0.0

        try:
            while (self.time_now() - start_t).nanoseconds / 1e9 < self.timeout_s:
                if err_box:
                    raise err_box[0]

                action = action_queue.get()  # (A_padded,) or None
                if action is not None:
                    a_port = action.numpy()[:7]
                    pose = _action_port_to_baselink_pose(
                        a_port.astype(np.float64), port_pose,
                    )
                    self.set_pose_target(move_robot, pose, frame_id="base_link")
                    if last_a_port is not None:
                        d = float(np.linalg.norm(a_port - last_a_port))
                        max_a_step_delta = max(max_a_step_delta, d)
                    last_a_port = a_port
                    if ticks % LOG_EVERY_N == 0:
                        self.get_logger().info(
                            f"[RTC] tick={ticks:4d} qsize={action_queue.qsize()} "
                            f"chunks={stats['chunks']} no_action={no_action_ticks} "
                            f"a_port[xyz]=({a_port[0]:+.3f},{a_port[1]:+.3f},{a_port[2]:+.3f}) "
                            f"max_a_step_Δ={max_a_step_delta:.4f}"
                        )
                else:
                    no_action_ticks += 1
                    if no_action_ticks % 20 == 1:
                        self.get_logger().info(
                            f"[RTC] tick={ticks:4d} queue empty "
                            f"(chunks_produced={stats['chunks']})"
                        )

                send_feedback("running")
                ticks += 1

                now = self.time_now()
                remaining_ns = (next_deadline - now).nanoseconds
                if remaining_ns > 0:
                    self.sleep_for(remaining_ns / 1e9)
                next_deadline = next_deadline + Duration(nanoseconds=period_ns)
        finally:
            stop_evt.set()
            producer_thread.join(timeout=2.0)
            if producer_thread.is_alive():
                self.get_logger().warn(
                    "RTC producer thread did not exit within 2s; "
                    "continuing trial cleanup anyway"
                )
            action_queue.clear()

        self.get_logger().info(
            f"RunSmolVLA.insert_cable[RTC]: exit after {ticks} ticks "
            f"(chunks={stats['chunks']}, no_action={no_action_ticks}, "
            f"max_a_step_Δ={max_a_step_delta:.4f})"
        )
        return True

    # ------------------------------------------------------------------
    # SKIP_CHUNK loop. Synchronous (no producer thread, no RTC guidance).
    # Calls predict_action_chunk to get the full chunk, discards the first
    # `skip_n_chunks` actions, dispatches the remaining tail one per tick.
    # When the local buffer drains, fires fresh inference on the latest obs.
    # ------------------------------------------------------------------

    def _insert_cable_skip_chunk(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        task_str = task_string_for(
            task.target_module_name, task.port_name, task.port_type,
        )
        skip_n = self.skip_n_chunks
        chunk_size = self.policy.config.chunk_size
        self.get_logger().info(
            f"task_str: {task_str!r}  [SKIP_CHUNK mode "
            f"skip_n={skip_n} chunk_size={chunk_size} dispatched_per_chunk={chunk_size - skip_n}]"
        )

        port_pose = self._acquire_port_pose_baselink(task, get_observation)
        if port_pose is None:
            send_feedback("aborted: port pose unavailable")
            return False

        self.policy.reset()

        # Local buffer of post-processed actions awaiting dispatch.
        # Each entry is a 1D tensor of shape (action_dim_padded,).
        local_queue: list[torch.Tensor] = []

        start_t = self.time_now()
        period_ns = int(self.loop_period_s * 1e9)
        next_deadline = start_t + Duration(nanoseconds=period_ns)
        ticks = 0
        chunks_produced = 0
        none_obs_count = 0
        LOG_EVERY_N = 10
        last_a_port: np.ndarray | None = None
        max_a_step_delta = 0.0

        while (self.time_now() - start_t).nanoseconds / 1e9 < self.timeout_s:
            # Refill if the local buffer is exhausted.
            if not local_queue:
                obs_msg = get_observation()
                if obs_msg is None:
                    none_obs_count += 1
                    self.sleep_for(self.loop_period_s)
                    next_deadline = self.time_now() + Duration(nanoseconds=period_ns)
                    continue
                obs = self._build_obs_dict(obs_msg, task_str, port_pose)
                obs = self.preprocessor(obs)
                t_inf_start = _time.perf_counter()
                with torch.inference_mode():
                    actions = self.policy.predict_action_chunk(obs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inf_ms = (_time.perf_counter() - t_inf_start) * 1000.0
                # actions: (1, chunk_size, action_dim_padded)
                chunk = actions.squeeze(0)  # (chunk_size, A_padded)
                # Drop the first skip_n and post-process each remaining step.
                for t in range(skip_n, chunk_size):
                    a_post = self.postprocessor(chunk[t : t + 1])
                    # a_post: (1, A_padded); store unbatched on CPU.
                    local_queue.append(a_post[0].detach().cpu())
                chunks_produced += 1
                self.get_logger().info(
                    f"[SKIP_CHUNK] chunk {chunks_produced} inf={inf_ms:.1f}ms "
                    f"dropped={skip_n} buffered={len(local_queue)}"
                )

            action = local_queue.pop(0)
            a_port = action.numpy()[:7]
            pose = _action_port_to_baselink_pose(
                a_port.astype(np.float64), port_pose,
            )
            self.set_pose_target(move_robot, pose, frame_id="base_link")

            if last_a_port is not None:
                d = float(np.linalg.norm(a_port - last_a_port))
                max_a_step_delta = max(max_a_step_delta, d)
            last_a_port = a_port

            if ticks % LOG_EVERY_N == 0:
                self.get_logger().info(
                    f"[SKIP_CHUNK skip_n={skip_n}] tick={ticks:4d} "
                    f"chunks={chunks_produced} buf={len(local_queue)} "
                    f"a_port[xyz]=({a_port[0]:+.3f},{a_port[1]:+.3f},{a_port[2]:+.3f}) "
                    f"max_a_step_Δ={max_a_step_delta:.4f}"
                )

            send_feedback("running")
            ticks += 1

            now = self.time_now()
            remaining_ns = (next_deadline - now).nanoseconds
            if remaining_ns > 0:
                self.sleep_for(remaining_ns / 1e9)
            next_deadline = next_deadline + Duration(nanoseconds=period_ns)

        self.get_logger().info(
            f"RunSmolVLA.insert_cable[SKIP_CHUNK]: exit after {ticks} ticks "
            f"(chunks={chunks_produced}, none_obs={none_obs_count}, "
            f"max_a_step_Δ={max_a_step_delta:.4f})"
        )
        return True
