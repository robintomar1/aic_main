#!/usr/bin/env python3
"""Cancellation/deactivation tests for CheatCodeRobust.

The policy must abort promptly when:
  - The lifecycle node is deactivated (`parent_node.is_active = False`)
  - The active action goal is cancelled
    (`parent_node.goal_handle.is_cancel_requested = True`)
  - The active action goal is no longer active
    (`parent_node.goal_handle.is_active = False`)

Why this matters (from 2026-04-26 model.log):
  - Trial 2 cancel hit at sim t=39.4s. Framework's
    `insert_cable_execute_callback` returned, but the policy's `insert_cable()`
    Python function (running in `aic_model._action_thread`) was never told to
    stop. It kept looping force-gate retreats.
  - Trial 3 was accepted at sim t≈+74s. A NEW `_action_thread` was started.
    Both trial 2 and trial 3 `insert_cable` invocations ran concurrently,
    interleaving move_robot calls.
  - On lifecycle cleanup the publisher was destroyed; trial 2's stale loop
    spammed `move_robot exception: 'NoneType' object has no attribute
    'publish'` for ~25 wall seconds.

Runs WITHOUT the pixi env: mocks all ROS imports.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


# ============================================================================
# Mock ROS imports BEFORE importing CheatCodeRobust.
#
# The policy imports rclpy/geometry_msgs/aic_*. We replace those modules with
# MagicMock or hand-rolled stubs that match the duck-typed interface the policy
# uses (translation.x/y/z, rotation.w/x/y/z, Time arithmetic, etc.).
# ============================================================================

class _Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w, self.x, self.y, self.z = w, x, y, z


class _Transform:
    def __init__(self, tx=0.0, ty=0.0, tz=0.0):
        self.translation = _Vector3(tx, ty, tz)
        self.rotation = _Quat()


class _TFStamped:
    def __init__(self, transform=None):
        self.transform = transform or _Transform()


class _Pose:
    def __init__(self, position=None, orientation=None):
        self.position = position or _Vector3()
        self.orientation = orientation or _Quat()


class _Duration:
    def __init__(self, seconds=0.0, nanoseconds=None):
        if nanoseconds is None:
            nanoseconds = int(seconds * 1e9)
        self.nanoseconds = nanoseconds

    def __lt__(self, other):
        return self.nanoseconds < other.nanoseconds

    def __le__(self, other):
        return self.nanoseconds <= other.nanoseconds

    def __gt__(self, other):
        return self.nanoseconds > other.nanoseconds

    def __ge__(self, other):
        return self.nanoseconds >= other.nanoseconds

    def __eq__(self, other):
        return isinstance(other, _Duration) and self.nanoseconds == other.nanoseconds


class _Time:
    def __init__(self, ns=0):
        self.nanoseconds = ns

    def __sub__(self, other):
        return _Duration(nanoseconds=self.nanoseconds - other.nanoseconds)

    def __add__(self, other):
        return _Time(self.nanoseconds + other.nanoseconds)

    def __lt__(self, other):
        return self.nanoseconds < other.nanoseconds

    def __le__(self, other):
        return self.nanoseconds <= other.nanoseconds


class _TransformException(Exception):
    pass


def _quaternion_multiply(q1, q2):
    """Hamilton product, (w, x, y, z) convention. Real implementation so
    plug-local-frame error decomposition works in tests.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quaternion_slerp(a, b, frac):
    return (1.0, 0.0, 0.0, 0.0)


def _mock_ros_imports() -> None:
    sys.modules.setdefault("numpy", __import__("numpy"))

    # rclpy
    rclpy = MagicMock()
    sys.modules["rclpy"] = rclpy

    rclpy_duration = MagicMock()
    rclpy_duration.Duration = _Duration
    sys.modules["rclpy.duration"] = rclpy_duration

    rclpy_time = MagicMock()
    rclpy_time.Time = lambda *a, **kw: _Time()
    sys.modules["rclpy.time"] = rclpy_time

    rclpy_qos = MagicMock()
    sys.modules["rclpy.qos"] = rclpy_qos

    # geometry_msgs
    gm = MagicMock()
    gm_msg = MagicMock()
    gm_msg.Point = _Vector3
    gm_msg.Pose = _Pose
    gm_msg.Quaternion = _Quat
    gm_msg.Transform = _Transform
    gm.msg = gm_msg
    sys.modules["geometry_msgs"] = gm
    sys.modules["geometry_msgs.msg"] = gm_msg

    # std_msgs
    std = MagicMock()
    std_msg = MagicMock()
    std.msg = std_msg
    sys.modules["std_msgs"] = std
    sys.modules["std_msgs.msg"] = std_msg

    # tf2_ros
    tf2 = MagicMock()
    tf2.TransformException = _TransformException
    sys.modules["tf2_ros"] = tf2

    # transforms3d
    tf3d = MagicMock()
    tf3d_g = MagicMock()
    tf3d_g.quaternion_multiply = _quaternion_multiply
    tf3d_g.quaternion_slerp = _quaternion_slerp
    tf3d._gohlketransforms = tf3d_g
    sys.modules["transforms3d"] = tf3d
    sys.modules["transforms3d._gohlketransforms"] = tf3d_g

    # aic_model.policy: replace Policy with a thin stub the test controls.
    class _StubPolicy:
        def __init__(self, parent_node):
            self._parent_node = parent_node

        def get_logger(self):
            return self._parent_node.get_logger()

        def get_clock(self):
            return self._parent_node.get_clock()

        def time_now(self):
            return self.get_clock().now()

        def sleep_for(self, duration_sec):
            # no-op in tests; advance the fake clock instead
            self._parent_node._advance_clock(duration_sec)

        def set_pose_target(self, move_robot, pose, **kwargs):
            # Real Policy wraps move_robot in try/except. We mirror that so a
            # failed publish doesn't kill the test mid-loop, but we still want
            # to count calls for assertions.
            try:
                move_robot(motion_update=pose)
            except Exception as ex:
                self.get_logger().info(f"move_robot exception: {ex}")

    aic_model = MagicMock()
    aic_model_policy = MagicMock()
    aic_model_policy.Policy = _StubPolicy
    aic_model_policy.GetObservationCallback = object
    aic_model_policy.MoveRobotCallback = object
    aic_model_policy.SendFeedbackCallback = object
    aic_model.policy = aic_model_policy
    sys.modules["aic_model"] = aic_model
    sys.modules["aic_model.policy"] = aic_model_policy

    # aic_task_interfaces
    aic_ti = MagicMock()
    aic_ti_msg = MagicMock()
    aic_ti.msg = aic_ti_msg
    sys.modules["aic_task_interfaces"] = aic_ti
    sys.modules["aic_task_interfaces.msg"] = aic_ti_msg


_mock_ros_imports()

sys.path.insert(
    0, "/home/robin/ssd/aic_workspace/aic_code_robin/aic_main/my_policy/my_policy/ros"
)
import CheatCodeRobust as ccr_mod  # noqa: E402

CheatCodeRobust = ccr_mod.CheatCodeRobust


# ============================================================================
# Fake parent_node — exposes the lifecycle/goal-handle state surface the
# policy must consult to know when to abort.
# ============================================================================

class FakeParentNode:
    """Minimum surface area used by CheatCodeRobust's insert_cable().

    Mirrors aic_model.AICModel: exposes is_active (lifecycle) and goal_handle
    (current ServerGoalHandle, or None). The policy is expected to consult
    both on every iteration of its phase loops.
    """

    def __init__(self):
        self.is_active = True
        # Goal handle defaults to "active, not cancelled". Tests can swap.
        self.goal_handle = SimpleNamespace(is_active=True, is_cancel_requested=False)

        # Sim clock in nanoseconds.
        self._clock_ns = 0

        # TF buffer mock — return reasonable defaults so APPROACH/ALIGN/INSERT
        # math doesn't divide by zero or raise.
        self._tf_buffer = MagicMock()
        self._tf_buffer.lookup_transform = self._lookup_transform

        # Track move_robot subscriptions (the policy calls
        # _parent_node.create_subscription in __init__ for /scoring/insertion_event).
        self.create_subscription = MagicMock()

    def _advance_clock(self, seconds):
        self._clock_ns += int(seconds * 1e9)

    def _lookup_transform(self, target, source, time):
        # Plug 5 mm offset in X from the port — enough to drive ALIGN, but
        # close enough that ALIGN can converge if it wants to.
        if "tcp" in source:
            return _TFStamped(_Transform(0.0, 0.0, 0.5))
        if "plug" in source.lower() or "tip" in source:
            return _TFStamped(_Transform(0.005, 0.0, 0.0))
        # Port frame.
        return _TFStamped(_Transform(0.0, 0.0, 0.0))

    def get_logger(self):
        return _StubLogger()

    def get_clock(self):
        node = self
        return SimpleNamespace(now=lambda: _Time(node._clock_ns))


class _StubLogger:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass


def _make_task():
    """Minimal Task duck-typed object used by the policy."""
    return SimpleNamespace(
        id="task_1",
        cable_type="sfp_sc",
        cable_name="cable_0",
        plug_type="sfp",
        plug_name="sfp_tip",
        port_type="sfp",
        port_name="sfp_port_0",
        target_module_name="nic_card_mount_0",
        time_limit=40,
    )


def _build_callbacks():
    move_calls = {"n": 0}

    def move_robot(motion_update=None, joint_motion_update=None):
        move_calls["n"] += 1

    def get_observation():
        return SimpleNamespace(
            wrist_wrench=SimpleNamespace(
                wrench=SimpleNamespace(
                    force=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    torque=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                )
            ),
            controller_state=SimpleNamespace(
                fts_tare_offset=SimpleNamespace(
                    wrench=SimpleNamespace(
                        force=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                        torque=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    )
                )
            ),
        )

    def send_feedback(_):
        pass

    return move_robot, get_observation, send_feedback, move_calls


# ============================================================================
# Tests
# ============================================================================

def test_should_abort_returns_true_when_node_deactivated():
    parent = FakeParentNode()
    policy = CheatCodeRobust(parent)
    parent.is_active = False
    assert policy._should_abort() is True, (
        "must abort when lifecycle node is no longer active"
    )


def test_should_abort_returns_true_when_cancel_requested():
    parent = FakeParentNode()
    policy = CheatCodeRobust(parent)
    parent.goal_handle.is_cancel_requested = True
    assert policy._should_abort() is True, "must abort on cancel request"


def test_should_abort_returns_true_when_goal_handle_inactive():
    parent = FakeParentNode()
    policy = CheatCodeRobust(parent)
    parent.goal_handle.is_active = False
    assert policy._should_abort() is True, (
        "must abort when goal handle is no longer active "
        "(canceled/aborted via cancel_task service)"
    )


def test_should_abort_returns_true_when_goal_handle_none():
    """If goal_handle is None (race between cancel and policy thread),
    safest behavior is to abort.
    """
    parent = FakeParentNode()
    policy = CheatCodeRobust(parent)
    parent.goal_handle = None
    assert policy._should_abort() is True


def test_should_abort_returns_false_in_healthy_state():
    parent = FakeParentNode()
    policy = CheatCodeRobust(parent)
    assert policy._should_abort() is False


def test_insert_cable_returns_false_immediately_when_deactivated():
    """If the node is deactivated BEFORE insert_cable starts running, the
    policy should bail before doing any meaningful work — definitely before
    completing all APPROACH_STEPS (100).
    """
    parent = FakeParentNode()
    parent.is_active = False
    policy = CheatCodeRobust(parent)
    move_robot, get_obs, send_fb, move_calls = _build_callbacks()
    result = policy.insert_cable(_make_task(), get_obs, move_robot, send_fb)
    assert result is False, "must return False when deactivated"
    assert move_calls["n"] < CheatCodeRobust.APPROACH_STEPS, (
        f"expected early bailout (<{CheatCodeRobust.APPROACH_STEPS} move_robot "
        f"calls), got {move_calls['n']} — did the phase loops actually check "
        f"_should_abort()?"
    )


def test_insert_cable_returns_false_when_cancel_requested_at_start():
    parent = FakeParentNode()
    parent.goal_handle.is_cancel_requested = True
    policy = CheatCodeRobust(parent)
    move_robot, get_obs, send_fb, move_calls = _build_callbacks()
    result = policy.insert_cable(_make_task(), get_obs, move_robot, send_fb)
    assert result is False
    assert move_calls["n"] < CheatCodeRobust.APPROACH_STEPS, (
        f"expected early bailout, got {move_calls['n']} move_robot calls"
    )


def test_insert_cable_aborts_mid_approach_when_cancel_arrives():
    """Trigger cancel after a few APPROACH iterations. Policy must stop within
    a small number of additional iterations — NOT continue through ALIGN/INSERT
    and certainly not keep running after the action thread should have died.
    """
    parent = FakeParentNode()
    policy = CheatCodeRobust(parent)
    move_robot, get_obs, send_fb, move_calls = _build_callbacks()

    # After 5 move_robot calls, simulate cancel arriving.
    cancel_after = 5

    def move_robot_then_cancel(motion_update=None, joint_motion_update=None):
        move_calls["n"] += 1
        if move_calls["n"] == cancel_after:
            parent.goal_handle.is_cancel_requested = True

    result = policy.insert_cable(_make_task(), get_obs, move_robot_then_cancel, send_fb)
    assert result is False, "must return False when cancel arrives mid-APPROACH"
    # Allow some slack — the loop checks abort at the top of each iteration so
    # we may take 1-2 more move_robot calls before the check fires. But we
    # MUST NOT complete all 100 APPROACH_STEPS, let alone proceed to ALIGN.
    assert move_calls["n"] < cancel_after + 5, (
        f"expected ≤{cancel_after + 5} move_robot calls (cancel after "
        f"{cancel_after}, +slack); got {move_calls['n']}. "
        f"Likely the APPROACH loop isn't checking _should_abort()."
    )


def _make_sc_task():
    return SimpleNamespace(
        id="t", cable_type="sfp_sc", cable_name="cable_1", plug_type="sc",
        plug_name="sc_tip", port_type="sc", port_name="sc_port_base",
        target_module_name="sc_port_0", time_limit=40,
    )


def test_xy_aligned_axis_aware_for_sc():
    """SC plug is anisotropic: chamfer along local X (eX up to ±3 mm
    forgiven), local Y is the tight axis (must be < 0.5 mm).
    Same xy_err magnitude can pass or fail depending on direction.
    """
    parent = FakeParentNode()
    policy = CheatCodeRobust(parent)
    sc_task = _make_sc_task()

    # Chamfer-tolerant: eX=2 mm (in chamfer dir), eY=0.3 mm (within tight).
    assert policy._xy_aligned(
        sc_task, ex_local=0.002, ey_local=0.0003,
        magnitude_threshold=policy.ALIGN_XY_THRESHOLD_M,
        tight_threshold=policy.ALIGN_TIGHT_THRESHOLD_M,
        chamfer_threshold=policy.ALIGN_CHAMFER_THRESHOLD_M,
    ) is True, "SC: eX=2mm (chamfer) + eY=0.3mm (within tight) must align"

    # Tight-axis violation, same magnitude rotated 90°: eY=2 mm.
    assert policy._xy_aligned(
        sc_task, ex_local=0.0003, ey_local=0.002,
        magnitude_threshold=policy.ALIGN_XY_THRESHOLD_M,
        tight_threshold=policy.ALIGN_TIGHT_THRESHOLD_M,
        chamfer_threshold=policy.ALIGN_CHAMFER_THRESHOLD_M,
    ) is False, "SC: eY=2mm violates tight axis even though magnitude is < 2.5mm"


def test_xy_aligned_uses_magnitude_for_symmetric_plug():
    """SFP has chamfers on both axes — magnitude check applies regardless
    of which axis the error lies on. Both directions of the same magnitude
    should give the same answer.
    """
    parent = FakeParentNode()
    policy = CheatCodeRobust(parent)
    sfp_task = SimpleNamespace(
        id="t", cable_type="sfp_sc", cable_name="cable_0", plug_type="sfp",
        plug_name="sfp_tip", port_type="sfp", port_name="sfp_port_0",
        target_module_name="nic_card_mount_0", time_limit=40,
    )

    # eX=2mm, eY=0.3mm: magnitude ≈ 2.02 mm < 2.5 mm → aligned.
    assert policy._xy_aligned(
        sfp_task, ex_local=0.002, ey_local=0.0003,
        magnitude_threshold=policy.ALIGN_XY_THRESHOLD_M,
        tight_threshold=policy.ALIGN_TIGHT_THRESHOLD_M,
        chamfer_threshold=policy.ALIGN_CHAMFER_THRESHOLD_M,
    ) is True

    # Same magnitude in the orthogonal direction — must give same result.
    assert policy._xy_aligned(
        sfp_task, ex_local=0.0003, ey_local=0.002,
        magnitude_threshold=policy.ALIGN_XY_THRESHOLD_M,
        tight_threshold=policy.ALIGN_TIGHT_THRESHOLD_M,
        chamfer_threshold=policy.ALIGN_CHAMFER_THRESHOLD_M,
    ) is True, "SFP must be direction-independent (chamfered on both axes)"


def test_inside_latch_axis_aware_for_sc():
    """Inside-port latch uses the same axis-aware gate. With plug below the
    port plane and eY violating tight axis, the latch must NOT engage even
    though magnitude is below the symmetric INSIDE_XY_THRESHOLD.
    """
    # eX=0 mm, eY=0.8 mm, plug 5 mm below port plane.
    # Magnitude 0.8 mm < INSIDE_XY_THRESHOLD_M (2.0 mm), but eY=0.8 mm
    # exceeds INSIDE_TIGHT_THRESHOLD_M (0.5 mm) — must not latch.
    parent = _make_parent_with_plug_at((0.0, -0.0008, -0.005))
    policy = CheatCodeRobust(parent)
    move_robot, get_obs, send_fb, _ = _build_callbacks()

    policy.insert_cable(_make_sc_task(), get_obs, move_robot, send_fb)

    assert policy._inside_latched is False, (
        "SC: latch must NOT engage with eY=0.8 mm (> tight threshold) "
        "even with plug below port plane and small magnitude xy_err"
    )


def _make_parent_with_plug_at(plug_xyz):
    """FakeParentNode whose plug TF is configurable. Used by inside-latch tests.

    Port stays at (0,0,0); gripper TCP stays at (0,0,0.5); plug at given xyz.
    """
    parent = FakeParentNode()

    def _lookup(target, source, time):
        if "tcp" in source:
            return _TFStamped(_Transform(0.0, 0.0, 0.5))
        if "plug" in source.lower() or "tip" in source:
            return _TFStamped(_Transform(*plug_xyz))
        return _TFStamped(_Transform(0.0, 0.0, 0.0))

    parent._tf_buffer.lookup_transform = _lookup
    return parent


def test_insert_locks_xy_once_plug_inside_port():
    """When plug satisfies (xy_err < INSIDE_XY_THRESHOLD AND
    plug_z - port_z < INSIDE_DEPTH), the policy must freeze its commanded
    XY/orientation. Subsequent commands may only change Z.

    Setup: plug at (0.001, 0, -0.005) — 1 mm xy_err (< 2 mm) and 5 mm below
    port plane (< -3 mm INSIDE_DEPTH for sfp). Latch should fire on the very
    first INSERT iteration.
    """
    parent = _make_parent_with_plug_at((0.001, 0.0, -0.005))
    policy = CheatCodeRobust(parent)
    move_robot, get_obs, send_fb, _ = _build_callbacks()

    captured = []

    def capture(motion_update=None, joint_motion_update=None):
        if motion_update is not None and hasattr(motion_update, "position"):
            captured.append((
                motion_update.position.x,
                motion_update.position.y,
                motion_update.position.z,
            ))

    policy.insert_cable(_make_task(), get_obs, capture, send_fb)

    assert policy._inside_latched is True, (
        "expected inside-port latch to engage with plug at (1mm, 0, -5mm) "
        "vs port at origin"
    )

    # Take the last K commanded poses (deep into INSERT, well past latch).
    # XY must be constant (locked); Z must vary (still descending).
    tail = captured[-200:]
    assert len(tail) > 50, f"expected many INSERT iterations, got {len(tail)}"
    xs = [p[0] for p in tail]
    ys = [p[1] for p in tail]
    zs = [p[2] for p in tail]
    assert max(xs) - min(xs) < 1e-9, (
        f"X must be locked post-latch but varied by "
        f"{(max(xs) - min(xs)) * 1000:.3f} mm"
    )
    assert max(ys) - min(ys) < 1e-9, (
        f"Y must be locked post-latch but varied by "
        f"{(max(ys) - min(ys)) * 1000:.3f} mm"
    )
    assert max(zs) - min(zs) > 1e-4, (
        "Z must continue to descend post-latch but appears frozen"
    )


def test_insert_does_not_lock_xy_on_near_miss():
    """xy_err just above INSIDE_XY_THRESHOLD: latch must NOT engage even
    though the plug is below the port plane. Guards against premature lock
    that would freeze the integrator on a misaligned plug.

    Setup: plug at (0.005, 0, -0.005) — 5 mm xy_err > 2 mm threshold.
    """
    parent = _make_parent_with_plug_at((0.005, 0.0, -0.005))
    policy = CheatCodeRobust(parent)
    move_robot, get_obs, send_fb, _ = _build_callbacks()

    policy.insert_cable(_make_task(), get_obs, move_robot, send_fb)

    assert policy._inside_latched is False, (
        "latch must NOT engage when xy_err (5 mm) exceeds "
        "INSIDE_XY_THRESHOLD_M (2 mm), even with plug below port plane"
    )


def test_insert_cable_aborts_when_node_deactivates_mid_run():
    """Same as above but trip lifecycle deactivation instead of cancel."""
    parent = FakeParentNode()
    policy = CheatCodeRobust(parent)
    move_robot, get_obs, send_fb, move_calls = _build_callbacks()

    deactivate_after = 5

    def move_robot_then_deactivate(motion_update=None, joint_motion_update=None):
        move_calls["n"] += 1
        if move_calls["n"] == deactivate_after:
            parent.is_active = False

    result = policy.insert_cable(_make_task(), get_obs, move_robot_then_deactivate, send_fb)
    assert result is False
    assert move_calls["n"] < deactivate_after + 5, (
        f"expected ≤{deactivate_after + 5} move_robot calls; got {move_calls['n']}"
    )


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_should_abort_returns_true_when_node_deactivated,
        test_should_abort_returns_true_when_cancel_requested,
        test_should_abort_returns_true_when_goal_handle_inactive,
        test_should_abort_returns_true_when_goal_handle_none,
        test_should_abort_returns_false_in_healthy_state,
        test_insert_cable_returns_false_immediately_when_deactivated,
        test_insert_cable_returns_false_when_cancel_requested_at_start,
        test_insert_cable_aborts_mid_approach_when_cancel_arrives,
        test_insert_cable_aborts_when_node_deactivates_mid_run,
        test_insert_locks_xy_once_plug_inside_port,
        test_insert_does_not_lock_xy_on_near_miss,
        test_xy_aligned_axis_aware_for_sc,
        test_xy_aligned_uses_magnitude_for_symmetric_plug,
        test_inside_latch_axis_aware_for_sc,
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
