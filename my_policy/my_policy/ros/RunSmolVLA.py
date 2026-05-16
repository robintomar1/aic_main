"""v9-baselink-smolvla inference shim — loads a base_link-frame SmolVLA
checkpoint and runs it as an aic_model Policy against the eval container.

No TF lookup or port-pose estimation required — all state is composed
directly from controller_state in base_link / sensor frame, matching the
dataset built by make_smolvla_baselink_dataset.py.

26-dim observation.state layout:
    tcp_pose(7)  ||  tcp_velocity(6)  ||  joint_positions(7)  ||  wrench(6)

7-dim action: absolute TCP pose in base_link (same as RunACT).

Z-advance filter: in base_link frame the robot descends toward the port
(Z decreases). The filter blocks any action commanding upward retreat
(action Z > current TCP Z + tolerance).

Contact detection: force-only — |F_xyz| >= threshold triggers hold.
Depth-gate removed (no port pose → no entrance depth).

Run-time configuration (env vars):
    AIC_PL_SMOLVLA_CHECKPOINT         Path to .../checkpoints/<step>/pretrained_model/. Required.
    AIC_PL_SMOLVLA_TIMEOUT_S          Per-trial budget. Default 30 s.
    AIC_PL_SMOLVLA_N_ACTION_STEPS     Override cfg.n_action_steps.
    AIC_PL_SMOLVLA_SKIP_N             Async skip-chunk mode (positive int, opt-in).
    AIC_PL_SMOLVLA_PER_CHUNK_SEED     RNG seed before each inference call. Default 0.
    AIC_PL_SMOLVLA_Z_ADVANCE_LIMIT_M  Z-filter tolerance in metres. Default 0.002.
    AIC_PL_SMOLVLA_CONTACT_FORCE_N    Force threshold for contact hold. Default 5.0 N.
    AIC_PL_SMOLVLA_Z_LOCK_ENABLE      Set to "1" to enable Z-lock termination. Default off.
    AIC_PL_SMOLVLA_Z_LOCK_SFP_M       base_link Z floor for SFP insertion (metres). Required if Z_LOCK_ENABLE=1.
                                       Empirical value from v9_smolvla_baselink dataset: 0.192
                                       (insertion occurs at tcp_z ≈ 0.1902; threshold is 2 mm above).
    AIC_PL_SMOLVLA_Z_LOCK_SC_M        base_link Z floor for SC insertion (metres). Required if Z_LOCK_ENABLE=1.
                                       Empirical value from v9_smolvla_baselink dataset: 0.036
                                       (insertion occurs at tcp_z ≈ 0.0350; threshold is 1 mm above).
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
from rclpy.node import Node
from safetensors.torch import load_file

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_S = 30.0
LOOP_HZ = 20.0
LOOP_PERIOD_S = 1.0 / LOOP_HZ

# Z-advance filter: base_link Z decreases as robot descends toward port.
# Blocks any action where action_z > tcp_z + this tolerance (upward retreat).
Z_ADVANCE_LIMIT_DEFAULT_M = 0.002

# Contact detection — force magnitude threshold.
CONTACT_FORCE_DEFAULT_N = 5.0

IMAGE_SCALING = 0.25  # 1152×1024 native → 288×256 (matches dataset)

STATE_DIM = 26  # tcp_pose(7) + tcp_velocity(6) + joint_positions(7) + wrench(6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_smolvla_policy(
    ckpt_dir: Path,
    device: torch.device,
    n_action_steps_override: int | None = None,
) -> SmolVLAPolicy:
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


def _compensated_wrench(
    obs_msg: Observation,
) -> tuple[float, float, float, float, float, float]:
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


def _build_state_26_baselink(obs_msg: Observation) -> torch.Tensor:
    """26-dim state in base_link/sensor frame.

    Layout matches make_smolvla_baselink_dataset.py:
      [0..6]   tcp_pose xyz + quat xyzw  (base_link)
      [7..12]  tcp_velocity linear + angular  (base_link)
      [13..19] joint_positions [0..6]
      [20..25] wrench fx,fy,fz,tx,ty,tz  (tare-compensated, sensor frame)
    """
    cs = obs_msg.controller_state
    tcp = cs.tcp_pose
    vel = cs.tcp_velocity
    js = obs_msg.joint_states
    fx, fy, fz, tx, ty, tz = _compensated_wrench(obs_msg)
    state = np.array(
        [
            tcp.position.x, tcp.position.y, tcp.position.z,
            tcp.orientation.x, tcp.orientation.y,
            tcp.orientation.z, tcp.orientation.w,
            vel.linear.x, vel.linear.y, vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *js.position[:7],
            fx, fy, fz, tx, ty, tz,
        ],
        dtype=np.float32,
    )
    assert state.shape == (STATE_DIM,), f"state must be {STATE_DIM}-dim, got {state.shape}"
    return torch.from_numpy(state)


def _action_baselink_to_pose(action7: np.ndarray) -> Pose:
    """7-dim base_link action → geometry_msgs/Pose with normalized quaternion."""
    if action7.shape != (7,):
        raise ValueError(f"expected (7,), got {action7.shape}")
    px, py, pz, qx, qy, qz, qw = (float(v) for v in action7)
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
# RunSmolVLA — Policy class loaded by aic_model.
# ---------------------------------------------------------------------------

class RunSmolVLA(Policy):
    CHECKPOINT_ENV = "AIC_PL_SMOLVLA_CHECKPOINT"
    TIMEOUT_ENV = "AIC_PL_SMOLVLA_TIMEOUT_S"
    N_ACTION_STEPS_ENV = "AIC_PL_SMOLVLA_N_ACTION_STEPS"
    SKIP_N_ENV = "AIC_PL_SMOLVLA_SKIP_N"
    PER_CHUNK_SEED_ENV = "AIC_PL_SMOLVLA_PER_CHUNK_SEED"
    PER_CHUNK_SEED_DEFAULT = 0
    Z_ADVANCE_LIMIT_ENV = "AIC_PL_SMOLVLA_Z_ADVANCE_LIMIT_M"
    CONTACT_FORCE_ENV = "AIC_PL_SMOLVLA_CONTACT_FORCE_N"
    # Z-lock: terminate as soon as tcp.z (base_link) drops below the known
    # insertion floor for the port type. base_link Z decreases as robot descends,
    # so "below" = tcp.z < threshold.
    Z_LOCK_ENABLE_ENV = "AIC_PL_SMOLVLA_Z_LOCK_ENABLE"   # "1" to enable
    Z_LOCK_SFP_ENV = "AIC_PL_SMOLVLA_Z_LOCK_SFP_M"       # base_link Z floor for SFP (metres)
    Z_LOCK_SC_ENV = "AIC_PL_SMOLVLA_Z_LOCK_SC_M"         # base_link Z floor for SC  (metres)

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.policy = _load_smolvla_policy(ckpt_path, self.device, n_action_steps_override)

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

        skip_n_str = os.environ.get(self.SKIP_N_ENV, "").strip()
        self.skip_n_chunks = int(skip_n_str) if skip_n_str else 0
        if self.skip_n_chunks < 0:
            raise ValueError(f"{self.SKIP_N_ENV}={self.skip_n_chunks} must be >= 0")
        chunk_size_loaded = int(self.policy.config.chunk_size)
        if self.skip_n_chunks >= chunk_size_loaded:
            raise ValueError(
                f"{self.SKIP_N_ENV}={self.skip_n_chunks} must be < "
                f"chunk_size={chunk_size_loaded}"
            )
        self.skip_chunk_enabled = self.skip_n_chunks > 0

        per_chunk_seed_raw = os.environ.get(self.PER_CHUNK_SEED_ENV, "").strip()
        if per_chunk_seed_raw == "":
            self._per_chunk_seed: int | None = self.PER_CHUNK_SEED_DEFAULT
        elif per_chunk_seed_raw.lower() in ("off", "none", "disable"):
            self._per_chunk_seed = None
        else:
            self._per_chunk_seed = int(per_chunk_seed_raw)

        z_lim_str = os.environ.get(self.Z_ADVANCE_LIMIT_ENV, "").strip()
        self._z_advance_limit_m = float(z_lim_str) if z_lim_str else Z_ADVANCE_LIMIT_DEFAULT_M

        cf_str = os.environ.get(self.CONTACT_FORCE_ENV, "").strip()
        self._contact_force_n = float(cf_str) if cf_str else CONTACT_FORCE_DEFAULT_N

        self._z_lock_enabled = os.environ.get(self.Z_LOCK_ENABLE_ENV, "0").strip() == "1"
        sfp_z_str = os.environ.get(self.Z_LOCK_SFP_ENV, "").strip()
        sc_z_str = os.environ.get(self.Z_LOCK_SC_ENV, "").strip()
        self._z_lock_sfp_m: float | None = float(sfp_z_str) if sfp_z_str else None
        self._z_lock_sc_m: float | None = float(sc_z_str) if sc_z_str else None
        if self._z_lock_enabled and (self._z_lock_sfp_m is None or self._z_lock_sc_m is None):
            raise ValueError(
                f"{self.Z_LOCK_ENABLE_ENV}=1 requires both "
                f"{self.Z_LOCK_SFP_ENV} and {self.Z_LOCK_SC_ENV} to be set."
            )

        self.get_logger().info(
            f"RunSmolVLA loaded checkpoint={ckpt_path} "
            f"device={self.device} loop={LOOP_HZ}Hz "
            f"timeout={self.timeout_s}s "
            f"n_action_steps={self.policy.config.n_action_steps} "
            f"chunk_size={self.policy.config.chunk_size} "
            f"skip_chunk_enabled={self.skip_chunk_enabled}"
            + (f" skip_n={self.skip_n_chunks}" if self.skip_chunk_enabled else "")
            + f" per_chunk_seed={self._per_chunk_seed} "
            + f"z_advance_limit={self._z_advance_limit_m * 1000:.1f}mm "
            + f"contact_force>={self._contact_force_n:.1f}N"
            + (
                f" z_lock=ON sfp<{self._z_lock_sfp_m:.4f}m sc<{self._z_lock_sc_m:.4f}m"
                if self._z_lock_enabled else " z_lock=OFF"
            )
        )

    def _z_lock_threshold(self, port_type: str) -> float | None:
        """Returns the base_link Z floor for the given port_type, or None if disabled."""
        if not self._z_lock_enabled:
            return None
        return self._z_lock_sfp_m if port_type == "sfp" else self._z_lock_sc_m

    def _seed_before_inference(self) -> None:
        if self._per_chunk_seed is None:
            return
        torch.manual_seed(self._per_chunk_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._per_chunk_seed)

    def _hard_reset_for_new_trial(self) -> None:
        self.policy.reset()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
            torch.cuda.empty_cache()

    def _publish_hold_current_pose(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
    ) -> bool:
        obs_msg = None
        for _ in range(10):
            obs_msg = get_observation()
            if obs_msg is not None:
                break
            self.sleep_for(0.02)
        if obs_msg is None:
            self.get_logger().warn(
                "no observation available for hold-pose at trial start"
            )
            return False
        cur = obs_msg.controller_state.tcp_pose
        hold = Pose(
            position=Point(x=cur.position.x, y=cur.position.y, z=cur.position.z),
            orientation=Quaternion(
                x=cur.orientation.x, y=cur.orientation.y,
                z=cur.orientation.z, w=cur.orientation.w,
            ),
        )
        self.set_pose_target(move_robot, hold, frame_id="base_link")
        self.get_logger().info(
            f"hold-pose at trial start: "
            f"({hold.position.x:+.3f},{hold.position.y:+.3f},{hold.position.z:+.3f})"
        )
        return True

    def _build_obs_dict(
        self,
        obs_msg: Observation,
        task_str: str,
    ) -> dict[str, Any]:
        return {
            "observation.images.left_camera":
                _ros_image_to_chw_float(obs_msg.left_image, IMAGE_SCALING),
            "observation.images.center_camera":
                _ros_image_to_chw_float(obs_msg.center_image, IMAGE_SCALING),
            "observation.images.right_camera":
                _ros_image_to_chw_float(obs_msg.right_image, IMAGE_SCALING),
            "observation.state": _build_state_26_baselink(obs_msg),
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

        task_str = task_string_for(
            task.target_module_name, task.port_name, task.port_type,
        )

        self._hard_reset_for_new_trial()
        self._publish_hold_current_pose(get_observation, move_robot)

        z_lock_threshold = self._z_lock_threshold(task.port_type)

        start_t = self.time_now()
        ticks = 0
        none_obs_count = 0
        LOG_EVERY_N = 10
        last_pose: Pose | None = None

        while (self.time_now() - start_t).nanoseconds / 1e9 < self.timeout_s:
            obs_msg = get_observation()
            if obs_msg is None:
                none_obs_count += 1
                self.sleep_for(self.loop_period_s)
                continue

            fx, fy, fz, _, _, _ = _compensated_wrench(obs_msg)
            force_n = float(np.linalg.norm([fx, fy, fz]))

            tcp = obs_msg.controller_state.tcp_pose.position
            tcp_z = float(tcp.z)

            # Z-lock: tcp descended past known insertion floor → done.
            if z_lock_threshold is not None and tcp_z < z_lock_threshold:
                self.get_logger().info(
                    f"*** Z-LOCK triggered at tick={ticks} "
                    f"tcp_z={tcp_z:+.4f} < threshold={z_lock_threshold:+.4f} "
                    f"port_type={task.port_type} — exiting as inserted ***"
                )
                return True

            obs = self._build_obs_dict(obs_msg, task_str)
            obs = self.preprocessor(obs)
            self._seed_before_inference()
            with torch.inference_mode():
                action = self.policy.select_action(obs)
            action = self.postprocessor(action)
            a = action[0].cpu().numpy()[:7]

            pose = _action_baselink_to_pose(a.astype(np.float64))
            self.set_pose_target(move_robot, pose, frame_id="base_link")
            last_pose = pose

            if ticks % LOG_EVERY_N == 0:
                self.get_logger().info(
                    f"tick={ticks:4d} "
                    f"tcp_bl=({tcp.x:+.3f},{tcp.y:+.3f},{tcp_z:+.3f}) "
                    f"a_z={a[2]:+.4f} |F|={force_n:.2f}N"
                    + (f" z_lock_thresh={z_lock_threshold:+.4f}" if z_lock_threshold is not None else "")
                )

            send_feedback("running")
            ticks += 1
            self.sleep_for(self.loop_period_s)

        self.get_logger().info(
            f"RunSmolVLA.insert_cable: exit after {ticks} ticks "
            f"(none_obs={none_obs_count})"
        )
        return True

    # ------------------------------------------------------------------
    # SKIP_CHUNK loop — async producer/consumer.
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
            f"[SKIP_CHUNK] task={task_str!r} "
            f"skip_n={skip_n} chunk_size={chunk_size} "
            f"dispatched_per_chunk={chunk_size - skip_n}"
        )

        self._hard_reset_for_new_trial()
        self._publish_hold_current_pose(get_observation, move_robot)

        local_queue: list[torch.Tensor] = []
        queue_lock = threading.Lock()
        stop_evt = threading.Event()
        contact_evt = threading.Event()
        err_box: list[BaseException] = []
        stats = {"chunks": 0, "inference_ticks": 0, "z_filter_hits": 0}

        def producer():
            try:
                while not stop_evt.is_set():
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

                    obs = self._build_obs_dict(obs_msg, task_str)
                    obs_state = obs["observation.state"].detach().cpu().numpy().reshape(-1)
                    obs_tcp_z_bl = float(obs_state[2])
                    obs_force_mag = float(np.linalg.norm(obs_state[20:23]))

                    # Contact detection.
                    if not contact_evt.is_set() and obs_force_mag >= self._contact_force_n:
                        contact_evt.set()
                        stop_evt.set()
                        self.get_logger().info(
                            f"*** [SKIP_CHUNK] CONTACT DETECTED "
                            f"|F|={obs_force_mag:.2f}N "
                            f"tcp_z_bl={obs_tcp_z_bl:+.4f} ***"
                        )
                        return

                    obs = self.preprocessor(obs)
                    t0 = _time.perf_counter()
                    self._seed_before_inference()
                    with torch.inference_mode():
                        actions = self.policy.predict_action_chunk(obs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inf_ms = (_time.perf_counter() - t0) * 1000.0

                    chunk = actions.squeeze(0)
                    new_actions: list[torch.Tensor] = []
                    for t in range(skip_n, chunk_size):
                        a_post = self.postprocessor(chunk[t: t + 1])
                        new_actions.append(a_post[0].detach().cpu())

                    # Z-advance filter.
                    chunk_z_hits = 0
                    for i in range(len(new_actions)):
                        a_np = new_actions[i].numpy()
                        if a_np[2] > obs_tcp_z_bl + self._z_advance_limit_m:
                            a_np = a_np.copy()
                            a_np[2] = obs_tcp_z_bl
                            new_actions[i] = torch.from_numpy(a_np)
                            chunk_z_hits += 1
                    stats["z_filter_hits"] += chunk_z_hits

                    with queue_lock:
                        local_queue.extend(new_actions)
                    stats["chunks"] += 1
                    first_z = float(new_actions[0].numpy()[2]) if new_actions else float("nan")
                    self.get_logger().info(
                        f"[SKIP_CHUNK] chunk={stats['chunks']} "
                        f"inf={inf_ms:.1f}ms "
                        f"obs_tcp_z={obs_tcp_z_bl:+.4f} "
                        f"|F|={obs_force_mag:.2f}N "
                        f"first_a_z={first_z:+.4f} "
                        f"z_filt={chunk_z_hits}/{len(new_actions)} "
                        f"buf={len(new_actions)}"
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
        last_pose: Pose | None = None
        contact_tick_sc = -1

        try:
            while (self.time_now() - start_t).nanoseconds / 1e9 < self.timeout_s:
                if err_box:
                    raise err_box[0]

                if contact_evt.is_set():
                    if contact_tick_sc < 0:
                        contact_tick_sc = ticks
                    if last_pose is not None:
                        self.set_pose_target(move_robot, last_pose, frame_id="base_link")
                    if ticks % LOG_EVERY_N == 0:
                        self.get_logger().info(
                            f"[SKIP_CHUNK] tick={ticks:4d} "
                            f"[HOLDING — contact since tick {contact_tick_sc}]"
                        )
                    send_feedback("contact")
                    ticks += 1
                    self.sleep_for(self.loop_period_s)
                    continue

                with queue_lock:
                    action = local_queue.pop(0) if local_queue else None

                if action is not None:
                    a = action.numpy()[:7]
                    pose = _action_baselink_to_pose(a.astype(np.float64))
                    self.set_pose_target(move_robot, pose, frame_id="base_link")
                    last_pose = pose
                    if ticks % LOG_EVERY_N == 0:
                        with queue_lock:
                            qlen = len(local_queue)
                        self.get_logger().info(
                            f"[SKIP_CHUNK] tick={ticks:4d} "
                            f"chunks={stats['chunks']} buf={qlen} "
                            f"a_z={a[2]:+.4f}"
                        )
                else:
                    stats["inference_ticks"] += 1
                    if last_pose is not None:
                        self.set_pose_target(move_robot, last_pose, frame_id="base_link")
                        if stats["inference_ticks"] % 20 == 1:
                            self.get_logger().info(
                                f"[SKIP_CHUNK] tick={ticks:4d} re-dispatching "
                                f"last pose (inf in flight, "
                                f"total inf_ticks={stats['inference_ticks']})"
                            )
                    else:
                        none_obs_count += 1

                send_feedback("running")
                ticks += 1
                self.sleep_for(self.loop_period_s)
        finally:
            stop_evt.set()
            join_timeout_s = 10.0
            producer_thread.join(timeout=join_timeout_s)
            if producer_thread.is_alive():
                self.get_logger().error(
                    f"SKIP_CHUNK producer STILL ALIVE after {join_timeout_s}s — "
                    f"blocking until it exits."
                )
                producer_thread.join()

        self.get_logger().info(
            f"RunSmolVLA.insert_cable[SKIP_CHUNK]: exit after {ticks} ticks "
            f"(chunks={stats['chunks']}, inf_ticks={stats['inference_ticks']}, "
            f"cold_ticks={none_obs_count}, z_filter_hits={stats['z_filter_hits']}, "
            f"contact={'tick ' + str(contact_tick_sc) if contact_evt.is_set() else 'none'})"
        )
        return True
