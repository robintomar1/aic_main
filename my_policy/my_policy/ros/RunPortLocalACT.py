"""v9-port-local inference shim — loads a port-local-trained ACT
checkpoint and runs it as an aic_model Policy against the eval container.

Differs from `RunACT.py` (the world-frame shim) in:

  * **Port pose acquisition** at trial start. The policy was trained on
    state/action expressed in the target port's frame, so we need a
    `T_port_in_baselink` estimate to do the live frame transform.
    Two modes (env-var-gated, mirrors `CheatCodeRobust.py`):

      AIC_PL_LOCALIZER_CHECKPOINT unset (DEFAULT — dev mode):
          /tf lookup of `base_link → <task.port_name>`. Requires the
          eval to be launched with `ground_truth:=true`.
      AIC_PL_LOCALIZER_CHECKPOINT=<path>:
          Load `my_policy.localizer.inference.PortLocalizer`, predict
          port pose from camera images + TCP pose + task one-hot.
          Required at submission (no /tf available).

    Port pose is captured ONCE at trial start (board doesn't move during
    insertion) and cached for the whole trial.

  * **Per-tick state assembly in port-local frame**. Pulls raw TCP pose,
    velocity, tare-compensated wrench, joint positions from the live
    `Observation`, applies `port_local.transforms.transform_frame` to
    get port-frame versions, then composes the 44-channel state in the
    EXACT order produced by `build_act_dataset.py:KEEP_CHANNEL_GROUPS`
    (so the trained model's input layout matches its training data).

  * **Per-tick action transform** back to base_link via
    `transform_pose_back_to_baselink`, then sent to the controller as
    a `MotionUpdate / MODE_POSITION` Pose target.

  * **Quaternion sign canonicalization at training time** (Fix 2 in
    `make_port_local_dataset.py`) means the model's output quat may be
    in either hemisphere relative to the recorded base_link orientation
    after round-trip. This is fine — q and -q are the same rotation —
    but the live controller's IK still expects unit-norm. We normalize
    after the round-trip.

Run-time configuration (env vars):
    AIC_PL_ACT_CHECKPOINT
        Path to .../checkpoints/<step>/pretrained_model/. Required.
    AIC_PL_ACT_TIMEOUT_S
        Per-trial inference budget. Default 30 s.
    AIC_PL_ACT_TEMPORAL_ENSEMBLE_COEFF
        If set (e.g. 0.01), enables ACT's inference-time temporal
        ensembling. Smaller = stronger smoothing. Forces n_action_steps=1.
    AIC_PL_INJECT_VELOCITY_BIAS_Z
        If set (e.g. 0.01), overrides the port-local TCP linear-z
        velocity in observation.state[9] with this value at every tick.
        Tests the velocity-OOD hypothesis: the live policy stalls when
        TCP velocity → 0 because training data never had stationary
        hovers (oracle was always descending). Injecting a small
        positive value (port-local +z = INTO the port) tricks the model
        into thinking it's still descending, potentially breaking the
        fixed-point lock at hover height. Suggested: 0.005-0.02.
    AIC_PL_LOCALIZER_CHECKPOINT
        If set, replaces /tf port lookup with PortLocalizer prediction.
    AIC_PL_LOCALIZER_QUATS_JSON
        Optional sidecar with calibrated port-in-board quaternions
        (mirrors CheatCodeRobust's CHEATCODE_LOCALIZER_QUATS_JSON).
    AIC_PL_LOCALIZER_DEVICE
        cuda or cpu (default cuda).
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

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)

from my_policy.act.labels import encode_task_vector
from my_policy.port_local.transforms import (
    FrameInputs,
    transform_frame,
    transform_pose_back_to_baselink,
)


# ---------------------------------------------------------------------------
# Constants — verified against the v9_port_local_merged_clean dataset built
# by `make_port_local_dataset.py` (so the inference state layout matches
# what the model was trained on, byte-for-byte).
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_S = 30.0
LOOP_HZ = 20.0
LOOP_PERIOD_S = 1.0 / LOOP_HZ

# Native cameras publish 1152W × 1024H; dataset stores 288W × 256H — exactly
# 0.25× scale. Same as RunACT.py.
IMAGE_SCALING = 0.25

# Per-trial timeout for waiting on /tf when in TF mode.
TF_LOOKUP_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Helpers (mostly mirrored from RunACT.py — kept inline so the two shims
# don't develop a hidden cross-dependency).
# ---------------------------------------------------------------------------


def _load_act_policy(
    ckpt_dir: Path,
    device: torch.device,
    temporal_ensemble_coeff: float | None,
) -> ACTPolicy:
    """Load ACT model + config from a checkpoint dir. Mirrors RunACT."""
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text())
    cfg_dict.pop("type", None)  # draccus rejects this discriminator
    if temporal_ensemble_coeff is not None:
        cfg_dict["temporal_ensemble_coeff"] = temporal_ensemble_coeff
        cfg_dict["n_action_steps"] = 1  # required when ensembling enabled
    config = draccus.decode(ACTConfig, cfg_dict)
    policy = ACTPolicy(config)
    policy.load_state_dict(load_file(str(ckpt_dir / "model.safetensors")))
    policy.eval()
    policy.to(device)
    return policy


def _ros_image_to_chw_float(ros_img, scaling: float) -> torch.Tensor:
    """ROS sensor_msgs/Image (uint8 RGB) → float32 [3, H, W] in [0, 1].

    Returns un-batched, un-normalized — preprocessor pipeline adds the batch
    dim and per-camera normalization.
    """
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
    """Tare-compensated F/T (Fx, Fy, Fz, Tx, Ty, Tz) — same formula as
    the recorder's `_compensated_wrench` and RunACT.py. Sensor frame
    co-rotates with the wrist; the port-local transform handles the
    rotation into port axes."""
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


def _build_state_44(
    obs_msg: Observation,
    task_vec: np.ndarray,
    port_pose_baselink: np.ndarray,
) -> torch.Tensor:
    """Compose the 44-channel observation.state in EXACTLY the layout
    `build_act_dataset.py:KEEP_CHANNEL_GROUPS` produces, but with the
    spatial channels (tcp_pose, tcp_velocity, wrench) re-expressed in
    the target port's frame.

    Layout:
      [ 0..6 ] tcp_pose      — port frame
      [ 7..12] tcp_velocity  — port frame
      [13..18] tcp_error     — TCP-relative; pass through
      [19..25] joint_pos     — frame-invariant; pass through
      [26..31] wrench        — port frame (sensor frame ≈ TCP frame
                               assumption documented in transforms.py)
      [32..43] task_vec      — one-hot, frame-invariant

    Returns float32 [44], un-batched, un-normalized.
    """
    if task_vec.shape != (12,):
        raise ValueError(f"task_vec must be shape (12,), got {task_vec.shape}")
    if port_pose_baselink.shape != (7,):
        raise ValueError(f"port_pose must be shape (7,), got {port_pose_baselink.shape}")

    cs = obs_msg.controller_state
    tcp_pose = cs.tcp_pose
    tcp_vel = cs.tcp_velocity
    js = obs_msg.joint_states

    # Raw base_link TCP pose / velocity / sensor wrench.
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
    wrench_sensor = np.array(
        _compensated_wrench(obs_msg), dtype=np.float64,
    )

    # Apply the port-local transform. action_baselink is unused in the
    # FrameOutputs we read here (we only consume tcp/velocity/wrench), but
    # FrameInputs requires it — pass tcp_pose_bl as a benign placeholder.
    inp = FrameInputs(
        tcp_pose_baselink=tcp_pose_bl,
        tcp_velocity_baselink=tcp_vel_bl,
        wrench_sensorframe=wrench_sensor,
        action_baselink=tcp_pose_bl,
        port_pose_baselink=port_pose_baselink,
    )
    out = transform_frame(inp)

    state = np.array(
        [
            # tcp_pose in port frame (7)
            *out.tcp_pose_portframe,
            # tcp_velocity in port frame (6)
            *out.tcp_velocity_portframe,
            # tcp_error (6) — frame-invariant pass-through
            cs.tcp_error[0], cs.tcp_error[1], cs.tcp_error[2],
            cs.tcp_error[3], cs.tcp_error[4], cs.tcp_error[5],
            # joint_positions (7)
            *js.position[:7],
            # wrench in port frame (6)
            *out.wrench_portframe,
            # task_vec (12)
            *task_vec.tolist(),
        ],
        dtype=np.float32,
    )
    assert state.shape == (44,), f"state must be 44-dim, got {state.shape}"

    # Optional port-local Z-velocity injection. Tests the velocity-OOD
    # hypothesis (see module header). state[9] is port-local
    # tcp_velocity.linear.z; positive = INTO the port = descending.
    inject_z_str = os.environ.get("AIC_PL_INJECT_VELOCITY_BIAS_Z", "").strip()
    if inject_z_str:
        try:
            state[9] = float(inject_z_str)
        except ValueError:
            pass  # silently ignore malformed values; default behavior unchanged

    return torch.from_numpy(state)


def _action_port_to_baselink_pose(
    action_port_7d: np.ndarray, port_pose_baselink: np.ndarray,
) -> Pose:
    """Port-local 7-d action → base_link Pose with normalized quaternion.

    The trained model's output is `(px, py, pz, qx, qy, qz, qw)` in port
    frame. We round-trip through the cached `port_pose_baselink` to get
    the equivalent base_link Pose target, then quat-normalize because
    the network output isn't unit-norm by construction.

    Sign-flips during the round-trip don't change the rotation (q and
    -q are the same), so even after the per-frame Fix 2 sign canon
    that was applied at training time, the resulting Pose is correct.
    """
    if action_port_7d.shape != (7,):
        raise ValueError(f"expected (7,), got {action_port_7d.shape}")
    recon = transform_pose_back_to_baselink(
        action_port_7d.astype(np.float64),
        port_pose_baselink.astype(np.float64),
    )
    px, py, pz, qx, qy, qz, qw = (float(v) for v in recon)
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm < 1e-8:
        # Degenerate — fall back to identity rotation rather than NaNs.
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return Pose(
        position=Point(x=px, y=py, z=pz),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )


# ---------------------------------------------------------------------------
# RunPortLocalACT — the Policy class loaded by aic_model.
# ---------------------------------------------------------------------------


class RunPortLocalACT(Policy):
    # Env vars (named to NOT collide with RunACT's AIC_ACT_* family).
    CHECKPOINT_ENV = "AIC_PL_ACT_CHECKPOINT"
    TIMEOUT_ENV = "AIC_PL_ACT_TIMEOUT_S"
    TEMPORAL_ENSEMBLE_ENV = "AIC_PL_ACT_TEMPORAL_ENSEMBLE_COEFF"

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

        te_str = os.environ.get(self.TEMPORAL_ENSEMBLE_ENV, "").strip()
        temporal_ensemble_coeff = float(te_str) if te_str else None

        self.policy = _load_act_policy(
            ckpt_path, self.device, temporal_ensemble_coeff,
        )

        self.preprocessor = DataProcessorPipeline.from_pretrained(
            str(ckpt_path), config_filename="policy_preprocessor.json"
        )
        # Postprocessor consumes a bare action tensor (mirrors RunACT).
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

        # PortLocalizer — lazily loaded on first trial when env var is set.
        self._localizer = None

        inject_z_env = os.environ.get("AIC_PL_INJECT_VELOCITY_BIAS_Z", "").strip()
        self.get_logger().info(
            f"RunPortLocalACT loaded checkpoint={ckpt_path} "
            f"device={self.device} loop={LOOP_HZ}Hz "
            f"timeout={self.timeout_s}s "
            f"temporal_ensemble_coeff={temporal_ensemble_coeff} "
            f"localizer_mode={self._localizer_mode()} "
            f"inject_velocity_z={inject_z_env or 'OFF'}"
        )

    # ------------------------------------------------------------------
    # Port-pose acquisition (TF or localizer).
    # ------------------------------------------------------------------

    def _localizer_mode(self) -> bool:
        return bool(os.environ.get(self.LOCALIZER_CKPT_ENV, "").strip())

    def _ensure_localizer_loaded(self) -> None:
        """Load PortLocalizer on first use; cache for subsequent trials."""
        if self._localizer is not None:
            return
        ckpt = os.environ[self.LOCALIZER_CKPT_ENV]
        quats = os.environ.get(self.LOCALIZER_QUATS_ENV) or None
        device = os.environ.get(self.LOCALIZER_DEVICE_ENV, "cuda")
        # Lazy import: keep the dependency off the import path when unused.
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
        """sensor_msgs/Image (rgb8) → (H, W, 3) uint8. For PortLocalizer
        which expects pre-resize uint8 (its own preprocessor handles
        scaling). Mirrors CheatCodeRobust._ros_image_to_np."""
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
        """Block until TF for `source_frame → target_frame` is available
        or the timeout expires. Mirrors CheatCodeRobust._wait_for_tf."""
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

    def _acquire_port_pose_baselink(
        self, task: Task, get_observation: GetObservationCallback,
    ) -> np.ndarray | None:
        """Get T_port_in_baselink as a 7-vector (xyz + xyzw). Returns None
        on failure; caller should send a feedback message and return."""
        if self._localizer_mode():
            return self._predict_port_pose_via_localizer(task, get_observation)
        return self._lookup_port_pose_via_tf(task)

    @staticmethod
    def _compose_port_tf_frame(task: Task) -> str:
        """Build the fully-qualified TF frame name for the trial's target
        port, matching the convention `collect_lerobot.py` uses to record
        `groundtruth.port_pose`:

            f"task_board/{target_module_name}/{port_name}_link"

        Verified against `task_board.urdf.xacro` — top-level model name is
        "task_board" (from `<robot name="task_board">`), each port instance
        is a nested Gazebo model with the prefix arg as its name (e.g.
        `<xacro:nic_card_mount prefix="nic_card_mount_0" .../>` →
        instance `nic_card_mount_0`), and the link inside that instance
        is named `<port_name>_link` (e.g. `sfp_port_1_link`).

        Examples:
          (nic_card_mount_4, sfp_port_1) → task_board/nic_card_mount_4/sfp_port_1_link
          (sc_port_0, sc_port_base)      → task_board/sc_port_0/sc_port_base_link
        """
        return f"task_board/{task.target_module_name}/{task.port_name}_link"

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
        """Run PortLocalizer once on a fresh observation. Returns the
        predicted port pose as a 7-vector in base_link, or None on
        failure (which is unrecoverable — without a port pose we can't
        do the live frame transform)."""
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
        task_vec: np.ndarray,
        port_pose_baselink: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        return {
            "observation.images.left_camera":
                _ros_image_to_chw_float(obs_msg.left_image, IMAGE_SCALING),
            "observation.images.center_camera":
                _ros_image_to_chw_float(obs_msg.center_image, IMAGE_SCALING),
            "observation.images.right_camera":
                _ros_image_to_chw_float(obs_msg.right_image, IMAGE_SCALING),
            "observation.state": _build_state_44(
                obs_msg, task_vec, port_pose_baselink,
            ),
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
            f"RunPortLocalACT.insert_cable: target_module={task.target_module_name} "
            f"port={task.port_name} port_type={task.port_type} "
            f"cable={task.cable_name}"
        )

        # Validate task vocabulary BEFORE acquiring port pose so we fail
        # fast on unknown inputs.
        task_vec = encode_task_vector(
            task.target_module_name, task.port_name, task.port_type,
        )

        # Step 1: get the port pose in base_link (board doesn't move
        # during a trial, so we cache once for the whole run).
        port_pose = self._acquire_port_pose_baselink(task, get_observation)
        if port_pose is None:
            send_feedback("aborted: port pose unavailable")
            return False

        # Step 2: per-trial reset of ACT's chunk queue.
        self.policy.reset()

        # Step 3: 20 Hz loop.
        start_t = self.time_now()
        ticks = 0
        none_obs_count = 0
        LOG_EVERY_N = 10  # 0.5 s at 20 Hz
        last_action_port: np.ndarray | None = None
        max_action_delta = 0.0

        while (self.time_now() - start_t).nanoseconds / 1e9 < self.timeout_s:
            obs_msg = get_observation()
            if obs_msg is None:
                none_obs_count += 1
                self.sleep_for(self.loop_period_s)
                continue

            obs = self._build_obs_dict(obs_msg, task_vec, port_pose)
            obs = self.preprocessor(obs)
            with torch.inference_mode():
                action = self.policy.select_action(obs)
            action = self.postprocessor(action)
            a_port = action[0].cpu().numpy()

            pose = _action_port_to_baselink_pose(
                a_port.astype(np.float64), port_pose,
            )
            self.set_pose_target(move_robot, pose, frame_id="base_link")

            # Log per-tick action info to help diagnose "robot stuck"
            # vs "model stable" vs "controller rejecting commands".
            tcp = obs_msg.controller_state.tcp_pose.position
            tcp_pred_dist = float(np.linalg.norm(
                np.array([pose.position.x, pose.position.y, pose.position.z])
                - np.array([tcp.x, tcp.y, tcp.z])
            ))
            if last_action_port is not None:
                d = float(np.linalg.norm(a_port - last_action_port))
                max_action_delta = max(max_action_delta, d)
            last_action_port = a_port

            # Pull port-local velocity + wrench magnitudes from the obs we
            # just built. obs is the post-preprocessor dict (normalized);
            # we need the un-normalized state for human-readable log values.
            # Cheapest: re-derive from the source observation we still have.
            #
            # state[7..12] = tcp_velocity port-frame (linear xyz, angular xyz)
            # state[26..31] = wrench port-frame (force xyz, torque xyz)
            # Since we already called _build_state_44 earlier in this loop,
            # rebuild the un-normalized state here for logging only.
            unnormed_state = _build_state_44(
                obs_msg, task_vec, port_pose,
            ).numpy()
            vel_z_port = float(unnormed_state[9])     # linear.z (into-port descent)
            vel_lin_mag = float(np.linalg.norm(unnormed_state[7:10]))
            force_mag = float(np.linalg.norm(unnormed_state[26:29]))
            torque_mag = float(np.linalg.norm(unnormed_state[29:32]))

            if ticks % LOG_EVERY_N == 0:
                self.get_logger().info(
                    f"tick={ticks:4d} "
                    f"tcp=({tcp.x:.3f},{tcp.y:.3f},{tcp.z:.3f}) "
                    f"pred_bl=({pose.position.x:.3f},"
                    f"{pose.position.y:.3f},{pose.position.z:.3f}) "
                    f"||pred-tcp||={tcp_pred_dist*1000:.1f}mm "
                    f"a_port[xyz]=({a_port[0]:.3f},{a_port[1]:.3f},{a_port[2]:.3f}) "
                    f"vel_z_port={vel_z_port:+.4f} "
                    f"|vel_lin|={vel_lin_mag:.4f} "
                    f"|F|={force_mag:.2f}N |τ|={torque_mag:.3f}Nm "
                    f"max_a_step_Δ={max_action_delta:.4f}"
                )

            send_feedback("running")
            ticks += 1
            self.sleep_for(self.loop_period_s)

        self.get_logger().info(
            f"RunPortLocalACT.insert_cable: exit after {ticks} ticks "
            f"(none_obs={none_obs_count}, max_a_step_Δ={max_action_delta:.4f})"
        )
        return True
