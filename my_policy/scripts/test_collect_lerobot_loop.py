#!/usr/bin/env python3
"""State-machine tests for run_collection_loop in collect_lerobot.py.

Runs WITHOUT the pixi env: mocks the ROS / LeRobot imports so we can exercise
the loop logic on the host. Each test scripts a sequence of (sim_time, event)
pairs that mutate a fake monitor; the loop's spin_monitor sees the mutations
on the next tick and reacts.

Tests target the bugs we hit in earlier smoke tests:
  - Overlong-discard followed by the previous trial's eventual terminal must
    NOT trigger an empty-buffer save_episode at the start of the next trial.
  - Per-trial duration cap must use SIM time, not wall time.
  - Happy path: goal start → frames → terminal SUCCEEDED → save_episode.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


class _FakeGoalStatus:
    STATUS_UNKNOWN = 0
    STATUS_ACCEPTED = 1
    STATUS_EXECUTING = 2
    STATUS_CANCELING = 3
    STATUS_SUCCEEDED = 4
    STATUS_CANCELED = 5
    STATUS_ABORTED = 6


def _mock_ros_and_lerobot_imports() -> None:
    """Inject minimal mocks so collect_lerobot can be imported on the host."""
    sys.modules.setdefault("cv2", MagicMock())
    sys.modules.setdefault("numpy", __import__("numpy"))

    sys.modules["rclpy"] = MagicMock()
    sys.modules["rclpy.parameter"] = MagicMock()
    sys.modules["rclpy.executors"] = MagicMock()
    sys.modules["rclpy.node"] = MagicMock()
    sys.modules["rclpy.qos"] = MagicMock()

    action_msgs = MagicMock()
    action_msgs_msg = MagicMock()
    action_msgs_msg.GoalStatus = _FakeGoalStatus
    action_msgs_msg.GoalStatusArray = MagicMock()
    action_msgs.msg = action_msgs_msg
    sys.modules["action_msgs"] = action_msgs
    sys.modules["action_msgs.msg"] = action_msgs_msg

    aic_ci = MagicMock()
    aic_ci_msg = MagicMock()
    aic_ci.msg = aic_ci_msg
    sys.modules["aic_control_interfaces"] = aic_ci
    sys.modules["aic_control_interfaces.msg"] = aic_ci_msg

    sys.modules["std_msgs"] = MagicMock()
    sys.modules["std_msgs.msg"] = MagicMock()
    sys.modules["tf2_msgs"] = MagicMock()
    sys.modules["tf2_msgs.msg"] = MagicMock()

    sys.modules["lerobot"] = MagicMock()
    sys.modules["lerobot.datasets"] = MagicMock()
    sys.modules["lerobot.datasets.lerobot_dataset"] = MagicMock()
    feature_utils = MagicMock()
    feature_utils.build_dataset_frame = lambda features, values, prefix: {}
    feature_utils.combine_feature_dicts = MagicMock()
    feature_utils.hw_to_dataset_features = MagicMock()
    sys.modules["lerobot.datasets.feature_utils"] = feature_utils

    sys.modules["lerobot_robot_aic"] = MagicMock()
    sys.modules["lerobot_robot_aic.aic_robot_aic_controller"] = MagicMock()


_mock_ros_and_lerobot_imports()

# Now safe to import the module under test.
sys.path.insert(0, "/home/robin/ssd/aic_workspace/aic_code_robin/aic_main/my_policy/scripts")
import collect_lerobot  # noqa: E402


# ============================================================================
# Test doubles
# ============================================================================

class FakeMonitor:
    """Mimics CollectorMonitor's interface used by run_collection_loop."""

    def __init__(self, sim_clock_ref: dict):
        self._clock_ref = sim_clock_ref  # {'t': float}
        self.goal_started_count = 0
        self.goal_terminated_count = 0
        self.event_count = 0
        self.last_terminal_status: int | None = None
        # Action-callback equivalent: a fixed action vector.
        self._action = [0.0] * 6

    def get_clock(self):
        ns = int(self._clock_ref["t"] * 1e9)
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=ns))

    def latest_action(self):
        return list(self._action)


class FakeRobot:
    def __init__(self):
        self.set_active_trial_calls: list[tuple[str, str]] = []
        self.observation = {"dummy": True}  # truthy so the frame-write path runs

    def set_active_trial(self, *, port_frame, plug_frame):
        self.set_active_trial_calls.append((port_frame, plug_frame))

    def get_observation(self):
        return self.observation


class FakeDataset:
    """Mimics LeRobotDataset behavior: save_episode raises if no frames added."""

    def __init__(self):
        self.features: dict = {}
        self._buffer: list[dict] = []
        self.save_episode_calls = 0
        self.clear_episode_buffer_calls = 0
        self.frames_per_save: list[int] = []

    def add_frame(self, frame: dict) -> None:
        self._buffer.append(frame)

    def save_episode(self) -> None:
        if not self._buffer:
            # Mirror the real "You must add one or several frames" error.
            raise ValueError("no frames in episode buffer")
        self.frames_per_save.append(len(self._buffer))
        self.save_episode_calls += 1
        self._buffer = []

    def clear_episode_buffer(self) -> None:
        self.clear_episode_buffer_calls += 1
        self._buffer = []

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)


# ============================================================================
# Scenario driver
# ============================================================================

class Scenario:
    """Schedules monitor-state mutations at specific sim times.

    The loop calls spin_monitor() each tick (no-op in tests; we apply pending
    events here) and tick_pacer(elapsed) (here: advance sim clock by tick_dt).
    """

    def __init__(
        self,
        monitor: FakeMonitor,
        sim_clock_ref: dict,
        events: list[tuple[float, str, dict]],
        tick_dt: float = 0.05,
        max_ticks: int = 5000,
    ):
        self.monitor = monitor
        self.clock = sim_clock_ref
        self.events = sorted(events, key=lambda e: e[0])
        self.tick_dt = tick_dt
        self.max_ticks = max_ticks
        self.ticks = 0

    def spin(self) -> None:
        """Apply all events whose sim_time <= now (FIFO)."""
        while self.events and self.events[0][0] <= self.clock["t"]:
            _, kind, params = self.events.pop(0)
            if kind == "goal_start":
                self.monitor.goal_started_count += 1
            elif kind == "goal_terminate":
                self.monitor.goal_terminated_count += 1
                self.monitor.last_terminal_status = params["status"]
            elif kind == "insertion_event":
                self.monitor.event_count += 1
            else:
                raise ValueError(f"unknown event kind: {kind}")

    def tick(self, _elapsed_s: float) -> None:
        self.clock["t"] += self.tick_dt
        self.ticks += 1
        if self.ticks > self.max_ticks:
            raise RuntimeError(f"scenario exceeded max_ticks={self.max_ticks}")


def _stub_log() -> MagicMock:
    log = MagicMock()
    log.info = MagicMock()
    log.warning = MagicMock()
    log.error = MagicMock()
    return log


def _make_trial(plug_type: str, port_name: str, target: str, cable: str) -> dict:
    return {
        "plug_type": plug_type,
        "plug_name": f"{plug_type}_tip",
        "cable_name": cable,
        "port_type": plug_type,
        "port_name": port_name,
        "target_module_name": target,
        "time_limit": 40,
    }


def _run(trials, events, *, max_episode_s=40.0, max_ticks=20000, tick_dt=0.05):
    sim_clock = {"t": 0.0}
    monitor = FakeMonitor(sim_clock)
    robot = FakeRobot()
    dataset = FakeDataset()
    log = _stub_log()
    scenario = Scenario(monitor, sim_clock, events, tick_dt=tick_dt, max_ticks=max_ticks)
    monotonic_t = {"t": 0.0}  # monotonic clock for global watchdog (independent of sim)

    def monotonic_fn():
        return monotonic_t["t"]

    def tick_pacer(elapsed_s):
        # Advance both monotonic and sim clocks by tick_dt.
        monotonic_t["t"] += tick_dt
        scenario.tick(elapsed_s)

    summary = collect_lerobot.run_collection_loop(
        monitor=monitor,
        robot=robot,
        dataset=dataset,
        trials=trials,
        log=log,
        max_episode_s=max_episode_s,
        global_timeout_s=10_000.0,
        is_alive=lambda: True,
        spin_monitor=scenario.spin,
        tick_pacer=tick_pacer,
        monotonic_fn=monotonic_fn,
        t_global_start=0.0,
    )
    return summary, monitor, robot, dataset, log


# ============================================================================
# Tests
# ============================================================================

def test_happy_path_two_trials_succeed():
    """Two trials, each insert successfully via SUCCEEDED status."""
    trials = [
        _make_trial("sfp", "sfp_port_0", "nic_card_mount_0", "cable_0"),
        _make_trial("sc", "sc_port_base", "sc_port_1", "cable_1"),
    ]
    events = [
        (0.1, "goal_start", {}),
        (10.0, "insertion_event", {}),
        (12.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_SUCCEEDED}),
        (15.0, "goal_start", {}),
        (25.0, "insertion_event", {}),
        (28.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_SUCCEEDED}),
    ]
    summary, _, robot, dataset, _ = _run(trials, events)
    assert len(summary["trials"]) == 2, summary
    assert summary["trials"][0]["outcome"] == "saved_inserted", summary["trials"][0]
    assert summary["trials"][1]["outcome"] == "saved_inserted", summary["trials"][1]
    assert dataset.save_episode_calls == 2, dataset.save_episode_calls
    assert dataset.clear_episode_buffer_calls == 0
    assert all(n > 0 for n in dataset.frames_per_save), dataset.frames_per_save
    assert len(robot.set_active_trial_calls) == 2


def test_overlong_discard_then_stale_terminal_then_normal():
    """REGRESSION: overlong trial discarded, then its eventual CANCELED arrives,
    then a new goal starts. The stale terminal must NOT immediately end the
    new trial with an empty-buffer save_episode.
    """
    trials = [
        _make_trial("sfp", "sfp_port_0", "nic_card_mount_0", "cable_0"),
        _make_trial("sc", "sc_port_base", "sc_port_0", "cable_1"),
    ]
    events = [
        (0.1, "goal_start", {}),
        # Engine eventually cancels at sim t=45 (well past our 40s cap).
        (45.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_CANCELED}),
        # Next goal at sim t=50.
        (50.0, "goal_start", {}),
        (62.0, "insertion_event", {}),
        (65.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_SUCCEEDED}),
    ]
    summary, _, _, dataset, log = _run(trials, events, max_episode_s=40.0)
    assert len(summary["trials"]) == 2, summary
    assert summary["trials"][0]["outcome"] == "discarded_overlong", summary["trials"][0]
    assert summary["trials"][1]["outcome"] == "saved_inserted", summary["trials"][1]
    # Critical assertions for the bug fix:
    assert dataset.save_episode_calls == 1, (
        f"expected exactly 1 save_episode (trial 1 only); got {dataset.save_episode_calls}"
    )
    assert dataset.clear_episode_buffer_calls == 1, (
        f"expected 1 clear (trial 0 overlong); got {dataset.clear_episode_buffer_calls}"
    )
    # Verify save was called with frames in buffer (would have raised otherwise).
    assert dataset.frames_per_save == [dataset.frames_per_save[0]] and dataset.frames_per_save[0] > 0
    # No save_episode failure should have been logged.
    error_calls = [c for c in log.error.call_args_list if "save_episode" in str(c)]
    assert error_calls == [], f"unexpected save_episode errors: {error_calls}"


def test_succeeded_no_insertion_is_discarded():
    """Trial ends via STATUS_SUCCEEDED but no /scoring/insertion_event fired
    (e.g., policy returned False on ALIGN bail). The action-level success means
    only "the policy method returned without exception", not "the cable was
    inserted". For IL data quality these demos are useless — the gripper traced
    a path that does NOT end in insertion — so the recorder must DISCARD them
    rather than save them as `saved_no_insertion`.
    """
    trials = [_make_trial("sc", "sc_port_base", "sc_port_0", "cable_1")]
    events = [
        (0.1, "goal_start", {}),
        (5.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_SUCCEEDED}),
    ]
    summary, _, _, dataset, _ = _run(trials, events, max_episode_s=40.0)
    assert summary["trials"][0]["outcome"].startswith("discarded"), summary["trials"][0]
    assert summary["trials"][0]["insertion_event_fired"] is False
    assert dataset.save_episode_calls == 0, (
        f"trial without insertion must NOT be saved; got {dataset.save_episode_calls}"
    )
    assert dataset.clear_episode_buffer_calls == 1


def test_canceled_trial_is_discarded():
    """Engine cancels the goal (typically because Task.time_limit elapsed in sim).
    Trial 2 in 2026-04-26 model.log: cancel arrived at sim t=39.4s, JUST under
    the recorder's 40s overlong cap. Currently the recorder saves it as
    `saved_no_insertion`; that polluted the dataset with a 39-second trajectory
    of the policy thrashing against the port rim. Must be discarded.
    """
    trials = [_make_trial("sfp", "sfp_port_0", "nic_card_mount_4", "cable_0")]
    events = [
        (0.1, "goal_start", {}),
        # Cancel at sim t=39.4 — under the 40s cap, but no insertion.
        (39.4, "goal_terminate", {"status": _FakeGoalStatus.STATUS_CANCELED}),
    ]
    summary, _, _, dataset, _ = _run(trials, events, max_episode_s=40.0)
    assert summary["trials"][0]["outcome"].startswith("discarded"), summary["trials"][0]
    assert summary["trials"][0]["insertion_event_fired"] is False
    assert dataset.save_episode_calls == 0
    assert dataset.clear_episode_buffer_calls == 1


def test_aborted_trial_is_discarded():
    """STATUS_ABORTED (action server reported abort, e.g. exception in policy
    thread). Same reasoning as canceled — no useful demo, must discard.
    """
    trials = [_make_trial("sfp", "sfp_port_0", "nic_card_mount_4", "cable_0")]
    events = [
        (0.1, "goal_start", {}),
        (5.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_ABORTED}),
    ]
    summary, _, _, dataset, _ = _run(trials, events, max_episode_s=40.0)
    assert summary["trials"][0]["outcome"].startswith("discarded"), summary["trials"][0]
    assert dataset.save_episode_calls == 0


def test_succeeded_with_insertion_short_trial_is_saved():
    """Positive control: SUCCEEDED + insertion event under the cap → SAVED.
    Distinct from happy_path which has two trials; this isolates the single-
    trial save path so a regression in the new discard predicate doesn't
    accidentally drop good demos.
    """
    trials = [_make_trial("sc", "sc_port_base", "sc_port_1", "cable_1")]
    events = [
        (0.1, "goal_start", {}),
        (15.0, "insertion_event", {}),
        (18.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_SUCCEEDED}),
    ]
    summary, _, _, dataset, _ = _run(trials, events, max_episode_s=40.0)
    assert summary["trials"][0]["outcome"] == "saved_inserted", summary["trials"][0]
    assert dataset.save_episode_calls == 1
    assert dataset.clear_episode_buffer_calls == 0


def test_canceled_overlong_trial_is_discarded():
    """If both overlong AND canceled fire (e.g., engine's time_limit slightly
    exceeds our cap), trial must still be discarded — no double-save, no save.
    """
    trials = [_make_trial("sfp", "sfp_port_0", "nic_card_mount_4", "cable_0")]
    events = [
        (0.1, "goal_start", {}),
        # Cancel arrives at sim t=42 — past our 40s cap.
        (42.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_CANCELED}),
    ]
    summary, _, _, dataset, _ = _run(trials, events, max_episode_s=40.0)
    assert summary["trials"][0]["outcome"].startswith("discarded"), summary["trials"][0]
    assert dataset.save_episode_calls == 0
    assert dataset.clear_episode_buffer_calls == 1


def test_sim_time_used_for_overlong_not_wall_time():
    """If sim runs at ~0.1× RTF, our wall time would be ~10× sim time. The cap
    must be measured against sim time so a normal trial (under 40 sim s) is
    saved even though wall time greatly exceeds 40s.

    We simulate this by having tick_pacer advance both clocks by the same
    tick_dt (so sim and monotonic are 1:1 here), but the sim clock is what
    matters for the cap. The contrast: a trial that runs 35s (under cap)
    must be saved.
    """
    trials = [_make_trial("sfp", "sfp_port_0", "nic_card_mount_0", "cable_0")]
    events = [
        (0.1, "goal_start", {}),
        (35.0, "insertion_event", {}),
        (37.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_SUCCEEDED}),
    ]
    summary, _, _, dataset, _ = _run(trials, events, max_episode_s=40.0)
    assert summary["trials"][0]["outcome"] == "saved_inserted"
    assert summary["trials"][0]["duration_s"] < 40.0
    assert dataset.save_episode_calls == 1


def test_overlapping_goals_discards_previous():
    """If a new goal arrives before the previous one's terminal status, the
    previous trial is discarded (overlapping_goals)."""
    trials = [
        _make_trial("sfp", "sfp_port_0", "nic_card_mount_0", "cable_0"),
        _make_trial("sc", "sc_port_base", "sc_port_0", "cable_1"),
    ]
    events = [
        (0.1, "goal_start", {}),
        # Trial 0 doesn't terminate; another goal arrives at sim t=20.
        (20.0, "goal_start", {}),
        (25.0, "insertion_event", {}),
        (28.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_SUCCEEDED}),
        # Eventually the engine also terminates trial 0 (we'd ignore this).
        (35.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_CANCELED}),
    ]
    summary, _, _, dataset, _ = _run(trials, events, max_episode_s=40.0)
    assert summary["trials"][0]["outcome"] == "discarded_overlong", summary["trials"][0]
    # Note: outcome above is "discarded_overlong" because the FIRST discard fires
    # with reason="overlapping_goals" but the helper labels by the discarded path
    # using "discarded_overlong" — check reason field for the trigger.
    assert summary["trials"][0]["reason"] == "overlapping_goals", summary["trials"][0]
    assert summary["trials"][1]["outcome"] == "saved_inserted"
    assert dataset.save_episode_calls == 1
    assert dataset.clear_episode_buffer_calls == 1


def test_per_trial_baselines_isolate_event_counts():
    """If trial N+1 starts after trial N inserted, event_count_at_trial_start
    must reset so trial N+1's insertion_event_fired isn't True from trial N's
    leftover increment."""
    trials = [
        _make_trial("sfp", "sfp_port_0", "nic_card_mount_0", "cable_0"),
        _make_trial("sc", "sc_port_base", "sc_port_0", "cable_1"),
    ]
    events = [
        (0.1, "goal_start", {}),
        (10.0, "insertion_event", {}),  # event_count → 1
        (12.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_SUCCEEDED}),
        (15.0, "goal_start", {}),
        # Trial 1 has NO insertion_event before terminating.
        (25.0, "goal_terminate", {"status": _FakeGoalStatus.STATUS_SUCCEEDED}),
    ]
    summary, _, _, dataset, _ = _run(trials, events, max_episode_s=40.0)
    assert summary["trials"][0]["insertion_event_fired"] is True
    assert summary["trials"][0]["outcome"] == "saved_inserted"
    # Critical: trial 1 should NOT see trial 0's insertion event.
    assert summary["trials"][1]["insertion_event_fired"] is False, summary["trials"][1]
    # Under the strict discard predicate trial 1 is discarded (no insertion);
    # what we're really verifying here is that it's NOT saved_inserted —
    # i.e., trial 0's leftover insertion_event isn't bleeding through.
    assert summary["trials"][1]["outcome"] != "saved_inserted", summary["trials"][1]


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_happy_path_two_trials_succeed,
        test_overlong_discard_then_stale_terminal_then_normal,
        test_succeeded_no_insertion_is_discarded,
        test_canceled_trial_is_discarded,
        test_aborted_trial_is_discarded,
        test_succeeded_with_insertion_short_trial_is_saved,
        test_canceled_overlong_trial_is_discarded,
        test_sim_time_used_for_overlong_not_wall_time,
        test_overlapping_goals_discards_previous,
        test_per_trial_baselines_isolate_event_counts,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as ex:
            failures += 1
            print(f"FAIL  {t.__name__}: {type(ex).__name__}: {ex}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(0 if failures == 0 else 1)
