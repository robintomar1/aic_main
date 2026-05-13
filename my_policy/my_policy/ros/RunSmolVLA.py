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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_smolvla_policy(
    ckpt_dir: Path,
    device: torch.device,
    n_action_steps_override: int | None = None,
) -> SmolVLAPolicy:
    """Load SmolVLA model + config from a checkpoint dir.

    `n_action_steps_override`: if set, overrides cfg.n_action_steps before
    instantiating the policy. SmolVLA's default n_action_steps == chunk_size
    (50 at training time) means the policy runs open-loop on its predicted
    chunk between forward passes — at 20 Hz that's 2.5 s of stale plan.
    Setting this to a smaller value (e.g. 1) forces SmolVLA to re-run its
    flow-matching denoiser on fresh observations more often, at proportional
    cost in inference latency. 1 is fully closed-loop; useful for fine
    alignment where the plan needs to react to small motions. Be aware
    SmolVLA-500M forward passes are not free — verify p99 < 50 ms before
    setting to 1, or stay at 5-10 for a middle ground.
    """
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text())
    cfg_dict.pop("type", None)
    if n_action_steps_override is not None:
        cfg_dict["n_action_steps"] = int(n_action_steps_override)
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

    state = np.array(
        [
            *out.tcp_pose_portframe,         # 7  → [0..6]
            *out.tcp_velocity_portframe,     # 6  → [7..12]
            *js.position[:7],                # 7  → [13..19]
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

    # SKIP_CHUNK opt-in. Setting to a positive integer N routes inference
    # through predict_action_chunk and drops the FIRST N actions of every
    # chunk before dispatch. Purely synchronous — no async producer, no
    # smoothing, no other side effects. Sync path is untouched when this
    # env var is unset or 0.
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
        self.policy = _load_smolvla_policy(
            ckpt_path, self.device, n_action_steps_override,
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

        # ---- SKIP_CHUNK config (opt-in) -------------------------------
        skip_n_str = os.environ.get(self.SKIP_N_ENV, "").strip()
        self.skip_n_chunks = int(skip_n_str) if skip_n_str else 0
        if self.skip_n_chunks < 0:
            raise ValueError(
                f"{self.SKIP_N_ENV}={self.skip_n_chunks} must be >= 0"
            )
        chunk_size_loaded = int(self.policy.config.chunk_size)
        if self.skip_n_chunks >= chunk_size_loaded:
            raise ValueError(
                f"{self.SKIP_N_ENV}={self.skip_n_chunks} must be < "
                f"chunk_size={chunk_size_loaded}"
            )
        self.skip_chunk_enabled = self.skip_n_chunks > 0

        self._localizer = None

        self.get_logger().info(
            f"RunSmolVLA loaded checkpoint={ckpt_path} "
            f"device={self.device} loop={LOOP_HZ}Hz "
            f"timeout={self.timeout_s}s "
            f"n_action_steps={self.policy.config.n_action_steps} "
            f"chunk_size={self.policy.config.chunk_size} "
            f"localizer_mode={self._localizer_mode()} "
            f"skip_chunk_enabled={self.skip_chunk_enabled}"
            + (f" skip_n={self.skip_n_chunks}" if self.skip_chunk_enabled else "")
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

        if self.skip_chunk_enabled:
            return self._insert_cable_skip_chunk(
                task, get_observation, move_robot, send_feedback,
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

    # ------------------------------------------------------------------
    # SKIP_CHUNK loop. Async: a producer thread runs predict_action_chunk
    # in the background and fills a buffer with the post-processed tail
    # (after dropping the first `skip_n_chunks` actions of each chunk).
    # The main 20 Hz consumer pops one per tick; when the buffer is empty
    # (i.e. producer is currently running inference), it RE-DISPATCHES the
    # last commanded action so the robot keeps receiving pose commands at
    # 20 Hz instead of seeing a gap.
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
        chunk_size = int(self.policy.config.chunk_size)
        self.get_logger().info(
            f"task_str: {task_str!r}  [SKIP_CHUNK mode "
            f"skip_n={skip_n} chunk_size={chunk_size} "
            f"dispatched_per_chunk={chunk_size - skip_n}]"
        )

        # Step 1: port pose (cached for the whole trial).
        port_pose = self._acquire_port_pose_baselink(task, get_observation)
        if port_pose is None:
            send_feedback("aborted: port pose unavailable")
            return False

        # Step 2: per-trial reset.
        self.policy.reset()

        # Shared state — protected by queue_lock for thread safety.
        local_queue: list[torch.Tensor] = []
        queue_lock = threading.Lock()
        stop_evt = threading.Event()
        err_box: list[BaseException] = []
        stats = {"chunks": 0, "inference_ticks": 0}

        def producer():
            try:
                while not stop_evt.is_set():
                    # Refill only when the consumer has drained the buffer.
                    with queue_lock:
                        need_refill = len(local_queue) == 0
                    if not need_refill:
                        if stop_evt.wait(0.005):
                            return
                        continue

                    obs_msg = get_observation()
                    if obs_msg is None:
                        if stop_evt.wait(0.01):
                            return
                        continue
                    obs = self._build_obs_dict(obs_msg, task_str, port_pose)
                    # Diagnostic: log the TCP-z we're feeding to inference,
                    # so we can compare against the first commanded z of the
                    # resulting chunk. If obs.tcp_z is low but a_port[2] of
                    # the first dispatched action is high, the model is
                    # ignoring current TCP (not a stale-obs bug).
                    obs_state_port = obs["observation.state"].detach().cpu().numpy().reshape(-1)
                    obs_tcp_port_z = float(obs_state_port[2])
                    obs_wrench_mag = float(np.linalg.norm(obs_state_port[20:23]))
                    obs = self.preprocessor(obs)
                    t_inf_start = _time.perf_counter()
                    with torch.inference_mode():
                        actions = self.policy.predict_action_chunk(obs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inf_ms = (_time.perf_counter() - t_inf_start) * 1000.0
                    # actions: (1, chunk_size, action_dim_padded)
                    chunk = actions.squeeze(0)
                    # Drop the first skip_n; post-process each remaining step.
                    new_actions: list[torch.Tensor] = []
                    for t in range(skip_n, chunk_size):
                        a_post = self.postprocessor(chunk[t : t + 1])
                        new_actions.append(a_post[0].detach().cpu())
                    with queue_lock:
                        local_queue.extend(new_actions)
                    stats["chunks"] += 1
                    # Full-chunk diagnostic: dump every action's port-frame
                    # z so we can see the shape of the model's chunk plan.
                    # Includes the dropped prefix too, so we can compare
                    # "what we threw away" vs "what we'll dispatch."
                    chunk_np = chunk.detach().cpu().numpy()  # (chunk_size, A_padded)
                    z_pre = chunk_np[:skip_n, 2] if skip_n > 0 else np.array([])
                    z_post = chunk_np[skip_n:, 2]
                    first_a_port_z = float(new_actions[0].numpy()[2]) if new_actions else float("nan")
                    self.get_logger().info(
                        f"[SKIP_CHUNK] chunk {stats['chunks']} "
                        f"inf={inf_ms:.1f}ms dropped={skip_n} "
                        f"buffered={len(new_actions)} "
                        f"obs_tcp_z={obs_tcp_port_z:+.3f} "
                        f"|F|_obs={obs_wrench_mag:.2f}N "
                        f"first_a_z={first_a_port_z:+.3f} "
                        f"Δ(a-obs)={first_a_port_z - obs_tcp_port_z:+.3f}"
                    )
                    # Z-trajectory sampled every 10 ticks; spans the FULL
                    # chunk (raw, before skip) so we can spot internal cycles.
                    z_sample = chunk_np[::10, 2]
                    sample_str = " ".join(f"{z:+.3f}" for z in z_sample)
                    self.get_logger().info(
                        f"[SKIP_CHUNK] chunk {stats['chunks']} "
                        f"raw_a_z (every 10 ticks, full chunk): {sample_str}"
                    )
                    # Per-segment stats: pre-skip and post-skip ranges.
                    if skip_n > 0:
                        self.get_logger().info(
                            f"[SKIP_CHUNK] chunk {stats['chunks']} "
                            f"pre-skip z[min/max/Δ]="
                            f"{z_pre.min():+.3f}/{z_pre.max():+.3f}/"
                            f"{z_pre.max() - z_pre.min():+.3f} | "
                            f"post-skip z[min/max/Δ]="
                            f"{z_post.min():+.3f}/{z_post.max():+.3f}/"
                            f"{z_post.max() - z_post.min():+.3f}"
                        )
            except BaseException as exc:  # noqa: BLE001
                err_box.append(exc)

        producer_thread = threading.Thread(
            target=producer, name="RunSmolVLA-SkipChunk-Producer", daemon=True,
        )
        producer_thread.start()

        start_t = self.time_now()
        ticks = 0
        none_obs_count = 0
        LOG_EVERY_N = 20
        last_a_port: np.ndarray | None = None
        last_pose: Pose | None = None
        max_a_step_delta = 0.0

        try:
            while (self.time_now() - start_t).nanoseconds / 1e9 < self.timeout_s:
                if err_box:
                    raise err_box[0]

                with queue_lock:
                    action = local_queue.pop(0) if local_queue else None

                if action is not None:
                    # Fresh action from the producer.
                    a_port = action.numpy()[:7]
                    pose = _action_port_to_baselink_pose(
                        a_port.astype(np.float64), port_pose,
                    )
                    self.set_pose_target(move_robot, pose, frame_id="base_link")
                    if last_a_port is not None:
                        d = float(np.linalg.norm(a_port - last_a_port))
                        max_a_step_delta = max(max_a_step_delta, d)
                    last_a_port = a_port
                    last_pose = pose
                    if ticks % LOG_EVERY_N == 0:
                        with queue_lock:
                            qlen = len(local_queue)
                        self.get_logger().info(
                            f"[SKIP_CHUNK skip_n={skip_n}] tick={ticks:4d} "
                            f"chunks={stats['chunks']} buf={qlen} "
                            f"a_port[xyz]=({a_port[0]:+.3f},{a_port[1]:+.3f},"
                            f"{a_port[2]:+.3f}) max_a_step_Δ={max_a_step_delta:.4f}"
                        )
                else:
                    # Producer is currently running inference — re-dispatch
                    # the last commanded pose so the controller keeps
                    # receiving 20 Hz updates.
                    stats["inference_ticks"] += 1
                    if last_pose is not None:
                        self.set_pose_target(move_robot, last_pose, frame_id="base_link")
                        if stats["inference_ticks"] % 20 == 1:
                            self.get_logger().info(
                                f"[SKIP_CHUNK] tick={ticks:4d} re-dispatching "
                                f"last pose (inference in flight, "
                                f"total inf_ticks={stats['inference_ticks']})"
                            )
                    else:
                        # Cold start — no last action yet; producer hasn't
                        # delivered the first chunk. Just wait.
                        none_obs_count += 1

                send_feedback("running")
                ticks += 1
                self.sleep_for(self.loop_period_s)
        finally:
            stop_evt.set()
            producer_thread.join(timeout=2.0)
            if producer_thread.is_alive():
                self.get_logger().warn(
                    "SKIP_CHUNK producer thread did not exit within 2 s; "
                    "continuing trial cleanup anyway"
                )

        self.get_logger().info(
            f"RunSmolVLA.insert_cable[SKIP_CHUNK]: exit after {ticks} ticks "
            f"(chunks={stats['chunks']}, inf_ticks={stats['inference_ticks']}, "
            f"cold_ticks={none_obs_count}, max_a_step_Δ={max_a_step_delta:.4f})"
        )
        return True
