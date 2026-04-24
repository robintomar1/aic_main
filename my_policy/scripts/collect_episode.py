#!/usr/bin/env python3
"""Batch oracle data-collection runner for AIC.

Assumes the eval container is started externally (see run_eval_batch.sh) with
an N-trial config. This script runs in the dev container and:
  1. Starts the aic_model node with CheatCode policy.
  2. Starts a rosbag2 recorder on the training-relevant topics.
  3. Subscribes to /scoring/insertion_event live and counts events.
  4. Exits when (a) expected_trials insertion events have fired, OR (b) the
     hard timeout elapses, OR (c) the eval container stops publishing (heartbeat
     loss on /tf for >30 s).
  5. Tears model + recorder down cleanly so the final insertion_event lands
     in the bag.

Two-terminal usage:

    # Host terminal — start eval with our batch config
    cd aic_docker/aic
    docker compose run --rm eval \\
        ground_truth:=true start_aic_engine:=true \\
        aic_engine_config_file:=/root/aic_results/batch_config.yaml

    # Dev container terminal — run the recorder
    pixi run python my_policy/scripts/collect_episode.py \\
        --out-dir /root/ws_aic/src/aic/results/batch_001 \\
        --expected-trials 20 \\
        --timeout-s 7200

The batch config itself is produced by gen_trial_config.py --n-trials <N> --out
<path>.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


RECORD_TOPICS = [
    "/observations",
    "/aic_controller/pose_commands",
    "/scoring/insertion_event",
    "/tf",
    "/tf_static",
]

DEFAULT_POLICY = "my_policy.ros.CheatCodeRobust"
TF_HEARTBEAT_TIMEOUT_S = 30.0  # if /tf goes silent this long, assume eval died


class BatchMonitor(Node):
    """Subscribes to /scoring/insertion_event + /tf for live progress."""

    def __init__(self):
        super().__init__("batch_monitor")
        # Match whatever QoS aic_scoring publishes with. Transient local is
        # common for latched topics; use reliable volatile as a safer default.
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
            String, "/scoring/insertion_event", self._on_event, events_qos)
        self.create_subscription(
            TFMessage, "/tf", self._on_tf, tf_qos)

        self.event_count = 0
        self.event_log: list[tuple[float, str]] = []
        self.last_tf_time: float | None = None

    def _on_event(self, msg: String) -> None:
        self.event_count += 1
        t = time.monotonic()
        self.event_log.append((t, msg.data))
        self.get_logger().info(
            f"[event {self.event_count}] {msg.data}")

    def _on_tf(self, _msg: TFMessage) -> None:
        self.last_tf_time = time.monotonic()


def start_model(log_path: Path, policy: str) -> subprocess.Popen:
    cmd = [
        "ros2", "run", "aic_model", "aic_model",
        "--ros-args",
        "-p", "use_sim_time:=true",
        "-p", f"policy:={policy}",
    ]
    log = open(log_path, "w")
    return subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def start_recorder(bag_dir: Path, log_path: Path) -> subprocess.Popen:
    cmd = [
        "ros2", "bag", "record",
        "--storage", "mcap",
        "-o", str(bag_dir),
    ] + RECORD_TOPICS
    log = open(log_path, "w")
    return subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
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


def wait_for_first_tf(monitor: BatchMonitor, executor, warm_up_s: float = 120.0) -> bool:
    """Block until /tf starts publishing (sim is up) or timeout."""
    deadline = time.monotonic() + warm_up_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=1.0)
        if monitor.last_tf_time is not None:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write the bag and run metadata.")
    p.add_argument("--expected-trials", type=int, required=True,
                   help="How many insertion events to expect before exiting.")
    p.add_argument("--timeout-s", type=float, default=7200.0,
                   help="Hard watchdog timeout (default 2h).")
    p.add_argument("--warm-up-s", type=float, default=180.0,
                   help="Max time to wait for /tf to appear from the eval container.")
    p.add_argument("--policy", type=str, default=DEFAULT_POLICY,
                   help="Dotted import path for the policy class.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bag_dir = args.out_dir / "bag"
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    rclpy.init()
    monitor = BatchMonitor()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(monitor)

    # Phase 1: wait for the eval container to come up.
    print(f"[orchestrator] Waiting up to {args.warm_up_s:.0f}s for /tf from eval...",
          flush=True)
    if not wait_for_first_tf(monitor, executor, args.warm_up_s):
        print("[orchestrator] TIMEOUT: no /tf observed. Is the eval container running?",
              file=sys.stderr)
        monitor.destroy_node()
        rclpy.shutdown()
        return 2
    print("[orchestrator] /tf seen — eval is up.", flush=True)

    # Phase 2: start model + recorder.
    print(f"[orchestrator] Policy: {args.policy}", flush=True)
    model = start_model(logs_dir / "model.log", args.policy)
    time.sleep(3.0)
    recorder = start_recorder(bag_dir, logs_dir / "recorder.log")
    print(f"[orchestrator] model pid={model.pid} recorder pid={recorder.pid}",
          flush=True)

    t_start = time.monotonic()
    exit_reason = ""

    try:
        while True:
            executor.spin_once(timeout_sec=1.0)

            # Success: all expected trials seen.
            if monitor.event_count >= args.expected_trials:
                exit_reason = "all_trials_complete"
                break

            # Hard timeout.
            elapsed = time.monotonic() - t_start
            if elapsed > args.timeout_s:
                exit_reason = f"timeout_{args.timeout_s:.0f}s"
                break

            # Eval container died (no /tf heartbeat).
            if monitor.last_tf_time is not None:
                silent = time.monotonic() - monitor.last_tf_time
                if silent > TF_HEARTBEAT_TIMEOUT_S:
                    exit_reason = f"eval_heartbeat_lost_{silent:.0f}s"
                    break

            # Subprocess health.
            if model.poll() is not None:
                exit_reason = f"model_crash_rc_{model.returncode}"
                break
            if recorder.poll() is not None:
                exit_reason = f"recorder_crash_rc_{recorder.returncode}"
                break
    finally:
        print(f"[orchestrator] Exit reason: {exit_reason}. Tearing down...", flush=True)
        # Let recorder drain first so the last event lands in the bag.
        time.sleep(2.0)
        terminate(recorder, timeout=20.0)
        terminate(model, timeout=10.0)

    meta = {
        "exit_reason": exit_reason,
        "expected_trials": args.expected_trials,
        "observed_events": monitor.event_count,
        "event_log": [(t - t_start, txt) for t, txt in monitor.event_log],
        "wall_duration_s": time.monotonic() - t_start,
        "bag_path": str(bag_dir),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[orchestrator] Done. events={monitor.event_count}/{args.expected_trials}"
          f" reason={exit_reason}", flush=True)

    monitor.destroy_node()
    rclpy.shutdown()
    return 0 if monitor.event_count >= args.expected_trials else 1


if __name__ == "__main__":
    sys.exit(main())
