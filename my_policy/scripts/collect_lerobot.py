#!/usr/bin/env python3
"""Direct-to-LeRobot oracle data collector for AIC.

Replaces the MCAP-based collect_episode.py with a streaming pipeline that
writes a LeRobot HDF5 dataset frame-by-frame, segmented on
/scoring/insertion_event. Skips the lerobot-record CLI (which expects keyboard
teleop) and drives LeRobotDataset directly.

Architecture:
  - Eval container runs the sim + aic_engine (started externally).
  - aic_model with CheatCodeRobust runs in a subprocess inside the dev
    container (same as collect_episode.py).
  - This script instantiates the AICRobotAICController adapter directly,
    subscribes to /aic_controller/pose_commands for action capture, and
    writes one LeRobotDataset episode per trial.

Two-terminal usage:

    # Host terminal — start eval with our batch config
    cd ~/ssd/aic_workspace/aic_docker/aic && docker compose run --rm eval \\
        ground_truth:=true start_aic_engine:=true \\
        aic_engine_config_file:=/root/aic_data/<batch>.yaml

    # Dev container terminal — run the recorder
    pixi run python my_policy/scripts/collect_lerobot.py \\
        --batch-config /root/aic_data/<batch>.yaml \\
        --root /root/aic_data/<batch>_dataset \\
        --repo-id local/aic_oracle_<batch> \\
        --per-trial-timeout-s 180

Outputs a LeRobot dataset at <root>/ that's append-friendly via
LeRobotDataset.resume() across multiple invocations (different batches).
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Import cv2 BEFORE any ROS / aic_* package — pulling in ros2 message types
# triggers libtiff load that's incompatible with cv2's libjpeg expectations.
# Loading cv2 first lets it bring in the version it was built against.
import cv2  # noqa: F401  (used transitively by lerobot_robot_aic)
import numpy as np
import rclpy
import yaml
from aic_control_interfaces.msg import MotionUpdate
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import (
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)

from lerobot_robot_aic.aic_robot_aic_controller import (
    AICRobotAICController,
    AICRobotAICControllerConfig,
)


DEFAULT_POLICY = "my_policy.ros.CheatCodeRobust"
TF_HEARTBEAT_TIMEOUT_S = 30.0
ACTION_DIM = 6
ACTION_NAMES = ["linear.x", "linear.y", "linear.z",
                "angular.x", "angular.y", "angular.z"]
TICK_RATE_HZ = 20
TICK_PERIOD_S = 1.0 / TICK_RATE_HZ


# =============================================================================
# Trial metadata extraction
# =============================================================================

def task_to_frames(task: dict) -> tuple[str, str]:
    """From a task dict (from batch config), derive TF frame names for the
    target port and the cable's plug tip. These are the frames the adapter
    looks up via TF when ground_truth:=true.
    """
    port_frame = f"task_board/{task['target_module_name']}/{task['port_name']}_link"
    plug_frame = f"{task['cable_name']}/{task['plug_name']}_link"
    return port_frame, plug_frame


def task_to_instruction(task: dict) -> str:
    """Synthesize a natural-language instruction. Used as the per-frame `task`
    string in the LeRobot dataset; SmolVLA conditions on it.
    """
    return (
        f"Insert the {task['plug_type']} plug into "
        f"{task['port_name']} on {task['target_module_name']}"
    )


def task_to_metadata_json(task: dict) -> str:
    """Pack non-string-conditioning metadata (config_hash, port_type, etc.)
    into a JSON string. Stored alongside the human-readable instruction.
    """
    return json.dumps({
        "port_type": task["port_type"],
        "port_name": task["port_name"],
        "target_module_name": task["target_module_name"],
        "plug_type": task["plug_type"],
        "cable_name": task["cable_name"],
    }, sort_keys=True)


# =============================================================================
# Side-channel monitor: action capture + episode segmentation signals
# =============================================================================

class CollectorMonitor(Node):
    """Subscribes to topics needed for recording but not exposed by the
    AICRobotAICController adapter:
      - /aic_controller/pose_commands → latest action twist
      - /scoring/insertion_event → trial advancement
      - /tf → eval container heartbeat
    """

    def __init__(self):
        super().__init__("aic_collector_monitor")

        events_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        tf_qos = QoSProfile(
            depth=100,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            MotionUpdate, "/aic_controller/pose_commands", self._on_pose_cmd, 10
        )
        self.create_subscription(
            String, "/scoring/insertion_event", self._on_event, events_qos
        )
        self.create_subscription(TFMessage, "/tf", self._on_tf, tf_qos)

        self._latest_action: np.ndarray = np.zeros(ACTION_DIM, dtype=np.float32)
        self._action_seen = False
        self.event_count = 0
        self.event_log: list[tuple[float, str]] = []
        self.last_tf_time: float | None = None

    def _on_pose_cmd(self, msg: MotionUpdate) -> None:
        v = msg.velocity
        self._latest_action = np.array([
            v.linear.x, v.linear.y, v.linear.z,
            v.angular.x, v.angular.y, v.angular.z,
        ], dtype=np.float32)
        self._action_seen = True

    def _on_event(self, msg: String) -> None:
        self.event_count += 1
        self.event_log.append((time.monotonic(), msg.data))
        self.get_logger().info(
            f"[event {self.event_count}] {msg.data}"
        )

    def _on_tf(self, _msg: TFMessage) -> None:
        self.last_tf_time = time.monotonic()

    def latest_action(self) -> np.ndarray:
        return self._latest_action.copy()


# =============================================================================
# aic_model subprocess management
# =============================================================================

def start_model(log_path: Path, policy: str) -> subprocess.Popen:
    cmd = [
        "ros2", "run", "aic_model", "aic_model",
        "--ros-args", "-p", "use_sim_time:=true", "-p", f"policy:={policy}",
    ]
    log = open(log_path, "w")
    return subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def terminate(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


# =============================================================================
# LeRobot dataset bring-up
# =============================================================================

def make_or_resume_dataset(
    repo_id: str,
    root: Path,
    fps: int,
    robot: AICRobotAICController,
) -> LeRobotDataset:
    """Create a new on-disk dataset, or resume an existing one for append.

    `root` is the local directory; nothing is pushed to HF Hub.
    """
    if root.exists() and (root / "meta" / "info.json").exists():
        # Existing dataset: append.
        return LeRobotDataset.resume(repo_id=repo_id, root=str(root))

    # Build the grouped features schema from the robot's flat dicts. The
    # adapter exposes scalars as `float` and camera images as 3-tuple shapes;
    # hw_to_dataset_features groups scalars into a vector under <prefix>.state
    # (or just <prefix> for actions) and creates one image/video feature per cam.
    features = combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, prefix="action", use_video=True),
        hw_to_dataset_features(robot.observation_features, prefix="observation", use_video=True),
    )

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=str(root),
        robot_type=robot.name,
        use_videos=True,
        # Sequential image writing to keep memory low — we're already at 20 Hz.
        image_writer_processes=0,
        image_writer_threads=4,
    )


# =============================================================================
# Main collection loop
# =============================================================================

def load_trial_sequence(batch_config_path: Path) -> list[dict]:
    """Extract the ordered list of task dicts from a batch config YAML."""
    with batch_config_path.open() as f:
        cfg = yaml.safe_load(f)
    trials = cfg.get("trials", {})
    out = []
    for name in sorted(trials.keys(), key=lambda k: int(k.rsplit("_", 1)[-1])):
        out.append(trials[name]["tasks"]["task_1"])
    return out


def wait_for_first_tf(monitor: CollectorMonitor, executor, warm_up_s: float) -> bool:
    deadline = time.monotonic() + warm_up_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=1.0)
        if monitor.last_tf_time is not None:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--batch-config", type=Path, required=True,
                   help="The batch YAML used by the eval container (defines trial order).")
    p.add_argument("--root", type=Path, required=True,
                   help="Local directory for the LeRobot dataset.")
    p.add_argument("--repo-id", type=str, required=True,
                   help="LeRobot repo_id (no Hub push; just used as identifier).")
    p.add_argument("--policy", type=str, default=DEFAULT_POLICY,
                   help="Dotted import path for the aic_model policy.")
    p.add_argument("--fps", type=int, default=TICK_RATE_HZ)
    p.add_argument("--per-trial-timeout-s", type=float, default=180.0,
                   help="Max wall time per trial before we declare failure and advance.")
    p.add_argument("--inter-trial-gap-s", type=float, default=2.0,
                   help="Pause between trials to let scene reset (frames not written).")
    p.add_argument("--warm-up-s", type=float, default=180.0,
                   help="Max wait for /tf from the eval container before bailing.")
    p.add_argument("--global-timeout-s", type=float, default=14400.0,
                   help="Hard watchdog on total run wall time (default 4h).")
    args = p.parse_args()

    args.root.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = args.root.parent / (args.root.name + "_logs")
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("collect_lerobot")

    # --- Trial sequence
    trials = load_trial_sequence(args.batch_config)
    log.info(f"loaded {len(trials)} trials from {args.batch_config}")

    # --- Bring up rclpy and our side-channel monitor
    rclpy.init()
    monitor = CollectorMonitor()
    monitor_executor = SingleThreadedExecutor()
    monitor_executor.add_node(monitor)

    # --- Wait for /tf so we know the eval container is alive
    log.info(f"waiting up to {args.warm_up_s:.0f}s for /tf...")
    if not wait_for_first_tf(monitor, monitor_executor, args.warm_up_s):
        log.error("timeout waiting for /tf — is the eval container running?")
        monitor.destroy_node()
        rclpy.shutdown()
        return 2
    log.info("/tf seen — eval is up.")

    # --- Connect the LeRobot adapter (this creates its own node/executor/thread)
    robot_cfg = AICRobotAICControllerConfig(id="aic")
    robot = AICRobotAICController(robot_cfg)
    robot.connect(calibrate=False)
    log.info("AICRobotAICController connected.")

    # --- Dataset
    dataset = make_or_resume_dataset(args.repo_id, args.root, args.fps, robot)
    log.info(f"dataset opened at {args.root} (existing episodes: {dataset.num_episodes})")

    # --- Start aic_model subprocess (CheatCodeRobust)
    model = start_model(logs_dir / "model.log", args.policy)
    time.sleep(3.0)
    log.info(f"aic_model started pid={model.pid} policy={args.policy}")

    t_global_start = time.monotonic()
    summary = {
        "trials": [],
        "per_trial_timeout_s": args.per_trial_timeout_s,
        "policy": args.policy,
    }

    def spin_monitor():
        # Pump callbacks so latest_action / event_count stay current.
        monitor_executor.spin_once(timeout_sec=0.0)

    try:
        for trial_idx, task in enumerate(trials):
            port_frame, plug_frame = task_to_frames(task)
            instruction = task_to_instruction(task)
            log.info(f"=== trial {trial_idx + 1}/{len(trials)}: {instruction}")
            log.info(f"    port_frame={port_frame}")
            log.info(f"    plug_frame={plug_frame}")

            robot.set_active_trial(port_frame=port_frame, plug_frame=plug_frame)
            event_count_at_start = monitor.event_count
            t_trial_start = time.monotonic()
            n_frames = 0
            outcome = "timeout"

            while True:
                tick_t = time.monotonic()
                spin_monitor()

                obs = robot.get_observation()
                if obs:  # skip until controller_state + joint_states are populated
                    action_arr = monitor.latest_action()
                    obs_frame = build_dataset_frame(
                        dataset.features, obs, prefix="observation"
                    )
                    action_dict = {name: action_arr[i] for i, name in enumerate(ACTION_NAMES)}
                    action_frame = build_dataset_frame(
                        dataset.features, action_dict, prefix="action"
                    )
                    frame = {**obs_frame, **action_frame, "task": instruction}
                    dataset.add_frame(frame)
                    n_frames += 1

                # Trial completion signals
                if monitor.event_count > event_count_at_start:
                    outcome = "inserted"
                    break
                if (tick_t - t_trial_start) > args.per_trial_timeout_s:
                    outcome = "timeout"
                    break
                if (tick_t - t_global_start) > args.global_timeout_s:
                    outcome = "global_timeout"
                    break
                if model.poll() is not None:
                    outcome = f"model_crash_rc_{model.returncode}"
                    break

                # 20 Hz pacing
                slack = TICK_PERIOD_S - (time.monotonic() - tick_t)
                if slack > 0:
                    time.sleep(slack)

            log.info(
                f"    trial {trial_idx + 1} done: outcome={outcome} "
                f"frames={n_frames} dur={(time.monotonic() - t_trial_start):.1f}s"
            )
            try:
                dataset.save_episode()
            except Exception as ex:
                log.error(f"save_episode failed: {ex}")

            summary["trials"].append({
                "idx": trial_idx,
                "outcome": outcome,
                "frames": n_frames,
                "duration_s": time.monotonic() - t_trial_start,
                "instruction": instruction,
                "task_meta": task_to_metadata_json(task),
            })

            if outcome.startswith("global_timeout") or outcome.startswith("model_crash"):
                log.error(f"aborting batch: {outcome}")
                break

            # Inter-trial gap: don't write frames during scene reset.
            time.sleep(args.inter_trial_gap_s)
    finally:
        log.info("tearing down...")
        terminate(model, timeout=10.0)
        try:
            dataset.finalize()
        except Exception as ex:
            log.error(f"dataset.finalize failed: {ex}")
        try:
            robot.disconnect()
        except Exception as ex:
            log.error(f"robot.disconnect failed: {ex}")
        monitor.destroy_node()
        rclpy.shutdown()

    summary["wall_duration_s"] = time.monotonic() - t_global_start
    summary["events_observed"] = monitor.event_count
    summary["episodes_in_dataset"] = dataset.num_episodes
    (logs_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    n_inserted = sum(1 for t in summary["trials"] if t["outcome"] == "inserted")
    log.info(
        f"done. inserted {n_inserted}/{len(trials)}; "
        f"dataset has {dataset.num_episodes} episodes total"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
