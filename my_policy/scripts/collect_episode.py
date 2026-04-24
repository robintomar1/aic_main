#!/usr/bin/env python3
"""Oracle data-collection orchestrator for AIC.

Runs the CheatCode policy against a pre-generated randomized config, records
the data needed to train the 3-stage policy (port localizer + aligner + local
insertion policy), and tags success/failure from /scoring/insertion_event.

Each episode launches three subprocesses:
  1. aic_gz_bringup.launch.py (Gazebo + aic_adapter + aic_engine + gt TF)
  2. aic_model with policy=aic_example_policies.ros.CheatCode
  3. ros2 bag record (MCAP) of the training-relevant topics

The engine is configured with shutdown_on_aic_engine_exit:=true, so when it
finishes all trials in the config it brings the whole launch down — the
orchestrator detects launch exit and tears the other two subprocesses down.

Usage:
    pixi run python collect_episode.py --config-dir /path/to/configs \
        --output-dir /path/to/bags --n-episodes 500

Each episode writes:
    <output-dir>/ep_<idx>_<config-hash>/
        bag/                       # raw mcap bag (deleted after HDF5 build)
        meta.json                  # config_hash, success, duration_s, times
"""

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Topics to record per episode. /observations gives us the full time-synced
# policy-visible state (3 cams + wrench + joint_states + controller_state) at
# 20 Hz. /tf + /tf_static carry ground-truth port poses (training only,
# ground_truth:=true). /aic_controller/pose_commands is the oracle's action
# signal. /scoring/insertion_event tags success.
RECORD_TOPICS = [
    "/observations",
    "/aic_controller/pose_commands",
    "/scoring/insertion_event",
    "/tf",
    "/tf_static",
]

DEFAULT_TIMEOUT_S = 300  # hard watchdog; trial time_limit is usually 180 s

CHEATCODE_POLICY = "aic_example_policies.ros.CheatCode"


@dataclass
class EpisodeResult:
    episode_idx: int
    config_path: str
    config_hash: str
    bag_path: str
    success: bool
    duration_s: float
    insertion_events: int
    aborted_reason: str  # "" on success, else reason


def extract_config_hash(config_path: Path) -> str:
    """gen_trial_config.py embeds a content hash in the filename."""
    stem = config_path.stem
    parts = stem.split("_")
    return parts[-1] if parts else "unknown"


def start_bringup(config_path: Path, log_path: Path) -> subprocess.Popen:
    cmd = [
        "ros2", "launch", "aic_bringup", "aic_gz_bringup.launch.py",
        "ground_truth:=true",
        "start_aic_engine:=true",
        "shutdown_on_aic_engine_exit:=true",
        f"aic_engine_config_file:={config_path}",
    ]
    log = open(log_path, "w")
    return subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,  # isolate signal group so we can SIGINT the tree
    )


def start_model(log_path: Path) -> subprocess.Popen:
    cmd = [
        "ros2", "run", "aic_model", "aic_model",
        "--ros-args",
        "-p", "use_sim_time:=true",
        "-p", f"policy:={CHEATCODE_POLICY}",
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


def count_insertion_events(bag_dir: Path) -> int:
    """Read /scoring/insertion_event from the bag and count events.

    Used only for success labeling. Runs in a helper subprocess so we don't
    pollute this script's rclpy init with a spinning executor.
    """
    # Simple approach: rely on post-processing. For now, return 0 if the bag
    # isn't readable and let the HDF5 builder parse events properly.
    # NOTE: a proper implementation reads the MCAP via rosbag2_py.
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        return 0

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/scoring/insertion_event"]))
    type_map = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    count = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        _ = deserialize_message(data, type_map[topic])
        count += 1
    return count


def run_episode(
    episode_idx: int,
    config_path: Path,
    output_root: Path,
    timeout_s: float,
) -> EpisodeResult:
    config_hash = extract_config_hash(config_path)
    ep_dir = output_root / f"ep_{episode_idx:05d}_{config_hash}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    bag_dir = ep_dir / "bag"
    logs_dir = ep_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    t_start = time.monotonic()
    aborted_reason = ""

    bringup = start_bringup(config_path, logs_dir / "bringup.log")
    # Give Gazebo + controller spawners a head start before model/recorder.
    time.sleep(15.0)
    model = start_model(logs_dir / "model.log")
    # Recorder last — by now the topics exist and QoS handshakes succeed.
    time.sleep(3.0)
    recorder = start_recorder(bag_dir, logs_dir / "recorder.log")

    try:
        # Block until bringup (aic_engine) exits — that is the episode end.
        while True:
            if bringup.poll() is not None:
                break
            if time.monotonic() - t_start > timeout_s:
                aborted_reason = f"timeout_{timeout_s:.0f}s"
                break
            if model.poll() is not None:
                aborted_reason = f"model_crash_rc_{model.returncode}"
                break
            if recorder.poll() is not None:
                aborted_reason = f"recorder_crash_rc_{recorder.returncode}"
                break
            time.sleep(1.0)
    finally:
        # Flush the recorder first so it captures the final insertion_event.
        terminate(recorder, timeout=15.0)
        terminate(model, timeout=10.0)
        terminate(bringup, timeout=15.0)

    duration_s = time.monotonic() - t_start

    insertion_events = 0
    if bag_dir.exists():
        try:
            insertion_events = count_insertion_events(bag_dir)
        except Exception as exc:
            aborted_reason = aborted_reason or f"bag_read_failed: {exc}"

    success = aborted_reason == "" and insertion_events > 0

    return EpisodeResult(
        episode_idx=episode_idx,
        config_path=str(config_path),
        config_hash=config_hash,
        bag_path=str(bag_dir),
        success=success,
        duration_s=duration_s,
        insertion_events=insertion_events,
        aborted_reason=aborted_reason,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", type=Path, required=True,
                   help="Directory of trial YAML configs from gen_trial_config.py.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write per-episode bag + meta.")
    p.add_argument("--n-episodes", type=int, default=1,
                   help="How many episodes to run.")
    p.add_argument("--start-idx", type=int, default=0,
                   help="Starting episode index (useful for resuming).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for config selection order.")
    p.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
                   help="Hard watchdog timeout per episode.")
    p.add_argument("--keep-bag", action="store_true",
                   help="Do not delete raw bag after episode (default: delete).")
    args = p.parse_args()

    configs = sorted(args.config_dir.glob("trial_*.yaml"))
    if not configs:
        print(f"No configs found in {args.config_dir}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(configs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.jsonl"

    n_success = 0
    for i in range(args.n_episodes):
        ep_idx = args.start_idx + i
        cfg = configs[ep_idx % len(configs)]
        print(f"\n==== Episode {ep_idx} / {args.start_idx + args.n_episodes - 1}"
              f" | config {cfg.name} ====", flush=True)

        result = run_episode(ep_idx, cfg, args.output_dir, args.timeout_s)

        with summary_path.open("a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

        tag = "OK" if result.success else f"FAIL({result.aborted_reason or 'no_insertion'})"
        print(f"  -> {tag}  duration={result.duration_s:.1f}s"
              f"  insertions={result.insertion_events}", flush=True)

        if result.success:
            n_success += 1

        # Post-process hook would go here: build_hdf5_episode(result.bag_path)
        # For now, keep the bag so we can iterate on the HDF5 builder separately.
        if not args.keep_bag and not result.success:
            # Clean up failed-episode bags to save disk.
            shutil.rmtree(result.bag_path, ignore_errors=True)

    print(f"\n==== Done. Success: {n_success}/{args.n_episodes} ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
