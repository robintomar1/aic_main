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
        --max-episode-s 40

Outputs a LeRobot dataset at <root>/ that's append-friendly via
LeRobotDataset.resume() across multiple invocations (different batches).

Trial advancement is driven by the InsertCable action's status topic
(/insert_cable/_action/status) — every trial start (ACCEPTED) and end
(SUCCEEDED/ABORTED/CANCELED) is observed regardless of insertion success.
Episodes that exceed --max-episode-s are discarded entirely (overlong demos
typically capture stuck/recovering trajectories that hurt IL training quality).
The engine's per-task time_limit in the batch YAML should match.
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
from action_msgs.msg import GoalStatus, GoalStatusArray
from aic_control_interfaces.msg import MotionUpdate
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
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
INSERT_CABLE_ACTION_STATUS_TOPIC = "/insert_cable/_action/status"
# Hard cap per trial. Anything exceeding this is discarded — overlong demos
# represent stuck/recovering trajectories that hurt IL training quality more
# than they help. Engine time_limit in our gen config is set to match.
MAX_EPISODE_DURATION_S = 40.0
TERMINAL_STATUSES = (
    GoalStatus.STATUS_SUCCEEDED,
    GoalStatus.STATUS_CANCELED,
    GoalStatus.STATUS_ABORTED,
)
ACTIVE_STATUSES = (
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
)


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
      - /scoring/insertion_event → success label (NOT used for advancement)
      - /insert_cable/_action/status → authoritative trial start/end
      - /tf → eval container heartbeat
    """

    def __init__(self):
        # Engine's time_limit is enforced in SIM time. The host can run sim
        # at well under 1× RTF (observed ~0.1× on heavier physics scenes), so
        # measuring per-trial duration in wall time would discard everything.
        # Anchor to /clock via use_sim_time so our cap aligns with engine
        # semantics. Must be passed at construction for the clock to switch.
        super().__init__(
            "aic_collector_monitor",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )

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
        # Default action-status QoS per ROS 2 design.
        action_status_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(
            MotionUpdate, "/aic_controller/pose_commands", self._on_pose_cmd, 10
        )
        self.create_subscription(
            String, "/scoring/insertion_event", self._on_event, events_qos
        )
        self.create_subscription(TFMessage, "/tf", self._on_tf, tf_qos)
        self.create_subscription(
            GoalStatusArray, INSERT_CABLE_ACTION_STATUS_TOPIC,
            self._on_action_status, action_status_qos,
        )

        self._latest_action: np.ndarray = np.zeros(ACTION_DIM, dtype=np.float32)
        self._action_seen = False
        self.event_count = 0
        self.event_log: list[tuple[float, str]] = []
        self.last_tf_time: float | None = None

        # Action-status tracking. We count distinct goals that have transitioned
        # into ACTIVE (start of a trial) or TERMINAL (end of a trial). The
        # recorder loop polls these counters to advance.
        self._known_goal_status: dict[str, int] = {}
        self.goal_started_count = 0
        self.goal_terminated_count = 0
        self.last_terminal_status: int | None = None

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

    def _on_action_status(self, msg: GoalStatusArray) -> None:
        for st in msg.status_list:
            uuid = bytes(st.goal_info.goal_id.uuid).hex()
            prev = self._known_goal_status.get(uuid)
            self._known_goal_status[uuid] = st.status
            became_active = (
                st.status in ACTIVE_STATUSES and prev not in ACTIVE_STATUSES
            )
            became_terminal = (
                st.status in TERMINAL_STATUSES and prev not in TERMINAL_STATUSES
            )
            if became_active:
                self.goal_started_count += 1
                self.get_logger().info(
                    f"[goal start] uuid={uuid[:8]} status={st.status}"
                )
            if became_terminal:
                self.goal_terminated_count += 1
                self.last_terminal_status = st.status
                name = {
                    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
                    GoalStatus.STATUS_CANCELED: "CANCELED",
                    GoalStatus.STATUS_ABORTED: "ABORTED",
                }.get(st.status, str(st.status))
                self.get_logger().info(
                    f"[goal end] uuid={uuid[:8]} {name}"
                )

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


def run_collection_loop(
    *,
    monitor,
    robot,
    dataset,
    trials: list[dict],
    log,
    max_episode_s: float,
    global_timeout_s: float,
    is_alive,                       # callable() -> bool
    spin_monitor,                   # callable() -> None (no-op in tests)
    tick_pacer,                     # callable(elapsed_s) -> None (sleep in prod, no-op in tests)
    monotonic_fn=time.monotonic,    # injectable for tests
    t_global_start: float | None = None,
) -> dict:
    """Event-driven recording loop. Extracted for unit-testing.

    Trial advancement is driven by /insert_cable/_action/status transitions
    (ACCEPTED→start, terminal→end). Per-trial duration is measured in SIM time
    via monitor.get_clock(); the engine's time_limit is also sim-time so
    these align even when sim runs well below 1× RTF.

    Per-trial baselines for the goal counters (event_count_at_trial_start,
    goal_terminated_at_trial_start) prevent stale counts from a previously
    overlong-discarded trial from immediately ending the next one.
    """
    if t_global_start is None:
        t_global_start = monotonic_fn()

    def now_sim_s() -> float:
        return monitor.get_clock().now().nanoseconds / 1e9

    summary = {"trials": []}

    trial_idx = 0
    recording = False
    trial_start_sim_s = 0.0
    n_frames = 0
    instruction = ""
    event_count_at_trial_start = 0
    goal_terminated_at_trial_start = monitor.goal_terminated_count
    last_goal_started_count = monitor.goal_started_count

    def _finalize_trial(*, discarded: bool, reason: str) -> None:
        nonlocal recording, trial_idx, n_frames
        duration = now_sim_s() - trial_start_sim_s
        insertion_event_fired = (
            monitor.event_count > event_count_at_trial_start
        )
        if discarded:
            try:
                dataset.clear_episode_buffer()
                outcome = "discarded_overlong"
            except Exception as ex:
                outcome = f"clear_failed:{ex}"
                log.error(f"clear_episode_buffer failed: {ex}")
        else:
            try:
                dataset.save_episode()
                outcome = "saved_inserted" if insertion_event_fired else "saved_no_insertion"
            except Exception as ex:
                outcome = f"save_failed:{ex}"
                log.error(f"save_episode failed: {ex}")
        log.info(
            f"    trial {trial_idx + 1} done: {outcome} ({reason}) "
            f"frames={n_frames} dur={duration:.1f}s"
        )
        task = trials[trial_idx] if trial_idx < len(trials) else {}
        summary["trials"].append({
            "idx": trial_idx,
            "outcome": outcome,
            "reason": reason,
            "frames": n_frames,
            "duration_s": duration,
            "insertion_event_fired": insertion_event_fired,
            "instruction": instruction,
            "task_meta": task_to_metadata_json(task) if task else "",
        })
        trial_idx += 1
        recording = False

    while trial_idx < len(trials):
        tick_t = monotonic_fn()
        spin_monitor()

        # Global watchdogs
        if (tick_t - t_global_start) > global_timeout_s:
            log.error(f"global timeout {global_timeout_s:.0f}s exceeded")
            if recording:
                _finalize_trial(discarded=True, reason="global_timeout")
            break
        if not is_alive():
            log.error("aic_model crashed")
            if recording:
                _finalize_trial(discarded=True, reason="model_crash")
            break

        # New goal accepted by aic_model? Start a trial.
        if monitor.goal_started_count > last_goal_started_count:
            last_goal_started_count = monitor.goal_started_count
            if recording:
                log.warning(
                    f"new goal started before previous terminated; "
                    f"discarding trial {trial_idx + 1}"
                )
                _finalize_trial(discarded=True, reason="overlapping_goals")
            if trial_idx < len(trials):
                task = trials[trial_idx]
                port_frame, plug_frame = task_to_frames(task)
                instruction = task_to_instruction(task)
                log.info(f"=== trial {trial_idx + 1}/{len(trials)}: {instruction}")
                log.info(f"    port_frame={port_frame}")
                log.info(f"    plug_frame={plug_frame}")
                robot.set_active_trial(port_frame=port_frame, plug_frame=plug_frame)
                event_count_at_trial_start = monitor.event_count
                goal_terminated_at_trial_start = monitor.goal_terminated_count
                trial_start_sim_s = now_sim_s()
                n_frames = 0
                recording = True
            else:
                log.warning(
                    f"received goal start beyond expected trial count {len(trials)}"
                )

        # Per-trial termination conditions
        if recording:
            duration_sim = now_sim_s() - trial_start_sim_s
            terminal = monitor.goal_terminated_count > goal_terminated_at_trial_start
            overlong = duration_sim > max_episode_s
            if terminal or overlong:
                if terminal:
                    status_name = {
                        GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
                        GoalStatus.STATUS_CANCELED: "CANCELED",
                        GoalStatus.STATUS_ABORTED: "ABORTED",
                    }.get(monitor.last_terminal_status, str(monitor.last_terminal_status))
                    reason = f"action_status_{status_name}"
                else:
                    reason = f"overlong_{duration_sim:.1f}s_sim>{max_episode_s}s"
                _finalize_trial(discarded=overlong, reason=reason)
            else:
                obs = robot.get_observation()
                if obs:
                    action_arr = monitor.latest_action()
                    obs_frame = build_dataset_frame(
                        dataset.features, obs, prefix="observation"
                    )
                    action_dict = {n: action_arr[i] for i, n in enumerate(ACTION_NAMES)}
                    action_frame = build_dataset_frame(
                        dataset.features, action_dict, prefix="action"
                    )
                    frame = {**obs_frame, **action_frame, "task": instruction}
                    dataset.add_frame(frame)
                    n_frames += 1

        tick_pacer(monotonic_fn() - tick_t)

    return summary


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
    p.add_argument("--max-episode-s", type=float, default=MAX_EPISODE_DURATION_S,
                   help="Hard cap per trial in SIM seconds (matches the engine's "
                        "time_limit, which is also sim-time). Episodes exceeding "
                        "this are DISCARDED. Sim can run well below 1× RTF, so do "
                        "not interpret this as wall time.")
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

    # --- Start aic_model subprocess FIRST, before our own rclpy/Zenoh setup.
    # aic_model has its own ~30s Zenoh discovery; the eval engine has a
    # ~60s timeout searching for the aic_model lifecycle node and times out
    # if discovery + recorder bring-up happens sequentially. Starting it
    # first lets aic_model's discovery overlap with everything else here.
    model = start_model(logs_dir / "model.log", args.policy)
    log.info(f"aic_model started pid={model.pid} policy={args.policy}")

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
        terminate(model, timeout=10.0)
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

    t_global_start = time.monotonic()

    def spin_monitor():
        # Pump callbacks so latest_action / goal counters / event_count stay current.
        monitor_executor.spin_once(timeout_sec=0.0)

    def is_alive() -> bool:
        return model.poll() is None

    def tick_pacer(elapsed_s: float) -> None:
        slack = TICK_PERIOD_S - elapsed_s
        if slack > 0:
            time.sleep(slack)

    summary = {"trials": [], "max_episode_s": args.max_episode_s, "policy": args.policy}
    try:
        loop_summary = run_collection_loop(
            monitor=monitor,
            robot=robot,
            dataset=dataset,
            trials=trials,
            log=log,
            max_episode_s=args.max_episode_s,
            global_timeout_s=args.global_timeout_s,
            is_alive=is_alive,
            spin_monitor=spin_monitor,
            tick_pacer=tick_pacer,
            monotonic_fn=time.monotonic,
            t_global_start=t_global_start,
        )
        summary["trials"] = loop_summary["trials"]
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
