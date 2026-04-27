#!/usr/bin/env python3
"""Tier 1 unit tests for the port-localizer label generation.

Pure functions only — no LeRobot, no torch, no rclpy. Runs on host (no pixi).

Verifies:
  - task_one_hot encoding round-trips and rejects unknowns.
  - compute_label produces correct values for hand-rolled SFP and SC trials.
  - reconstruct_port_in_baselink is the exact inverse of compute_label modulo
    the URDF static offsets — round-trip residual < 1 µm.
  - yaw composition matches yaw_world − robot_mount_yaw, wrapped to (−π, π].
  - match_episodes_to_trials handles discards and duplicate-key fallback.
  - LocalizerLabel.as_target_5() obeys sin²+cos² = 1.

Run: `python3 my_policy/scripts/test_localizer_labels.py`
"""

import math
import sys
from pathlib import Path

import numpy as np

# Make the package importable from a checkout (no installation needed).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from my_policy.my_policy.localizer.labels import (  # noqa: E402
    SC_PORT_BASE_OFFSET_IN_BOARD,
    SFP_PORT_OFFSET_IN_MOUNT,
    TASK_ONE_HOT_DIM,
    TASK_ONE_HOT_ORDER,
    LocalizerLabel,
    _wrap_to_pi,
    compute_label,
    match_episodes_to_trials,
    reconstruct_port_in_baselink,
    task_one_hot,
)


# ============================================================================
# Trial-dict factories
# ============================================================================

def _make_sfp_trial(
    *,
    board_x_world: float = 0.15,
    board_y_world: float = -0.20,
    board_yaw_world: float = 0.0,
    nic_index: int = 2,
    rail_translation: float = 0.0,
    port_name: str = "sfp_port_0",
) -> dict:
    """Trial dict matching the gen_trial_config format (SFP target)."""
    scene_rails = {f"nic_rail_{i}": {"entity_present": False} for i in range(5)}
    scene_rails[f"nic_rail_{nic_index}"] = {
        "entity_present": True,
        "entity_name": f"nic_card_{nic_index}",
        "entity_pose": {
            "translation": rail_translation,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        },
    }
    for i in range(2):
        scene_rails[f"sc_rail_{i}"] = {"entity_present": False}
    return {
        "scene": {
            "task_board": {
                "pose": {
                    "x": board_x_world, "y": board_y_world, "z": 1.14,
                    "roll": 0.0, "pitch": 0.0, "yaw": board_yaw_world,
                },
                **scene_rails,
            },
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": port_name,
                "target_module_name": f"nic_card_mount_{nic_index}",
                "time_limit": 40,
            },
        },
    }


def _make_sc_trial(
    *,
    board_x_world: float = 0.15,
    board_y_world: float = -0.20,
    board_yaw_world: float = 0.0,
    sc_index: int = 0,
    rail_translation: float = 0.0,
) -> dict:
    """Trial dict matching the gen_trial_config format (SC target)."""
    scene_rails = {f"nic_rail_{i}": {"entity_present": False} for i in range(5)}
    scene_rails[f"sc_rail_{sc_index}"] = {
        "entity_present": True,
        "entity_name": f"sc_mount_{sc_index}",
        "entity_pose": {
            "translation": rail_translation,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        },
    }
    for i in range(2):
        if i != sc_index:
            scene_rails[f"sc_rail_{i}"] = {"entity_present": False}
    return {
        "scene": {
            "task_board": {
                "pose": {
                    "x": board_x_world, "y": board_y_world, "z": 1.14,
                    "roll": 0.0, "pitch": 0.0, "yaw": board_yaw_world,
                },
                **scene_rails,
            },
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": "cable_1",
                "plug_type": "sc",
                "plug_name": "sc_tip",
                "port_type": "sc",
                "port_name": "sc_port_base",
                "target_module_name": f"sc_port_{sc_index}",
                "time_limit": 40,
            },
        },
    }


# ============================================================================
# Tests
# ============================================================================

def test_task_one_hot_orderings():
    """Every entry in TASK_ONE_HOT_ORDER produces a distinct one-hot summing
    to 1, with the correct active index."""
    seen = set()
    for i, name in enumerate(TASK_ONE_HOT_ORDER):
        v = task_one_hot(name)
        assert v.shape == (TASK_ONE_HOT_DIM,), f"{name}: wrong shape {v.shape}"
        assert v.sum() == 1.0, f"{name}: must sum to 1, got {v.sum()}"
        assert v[i] == 1.0, f"{name}: expected hot at idx {i}, got {v}"
        # Distinctness guarantee.
        key = tuple(v.tolist())
        assert key not in seen, f"{name}: duplicate one-hot {key}"
        seen.add(key)


def test_task_one_hot_unknown_raises():
    try:
        task_one_hot("nic_card_mount_99")
    except ValueError as ex:
        assert "unknown" in str(ex)
        return
    raise AssertionError("expected ValueError for unknown module name")


def test_compute_label_sfp_known_trial():
    """Hand-rolled SFP trial with explicit board pose; verify label fields."""
    trial = _make_sfp_trial(
        board_x_world=0.10, board_y_world=-0.30, board_yaw_world=0.0,
        nic_index=3, rail_translation=0.012, port_name="sfp_port_1",
    )
    label = compute_label(trial)
    # board_xy_baselink = world point (0.10, -0.30, 1.14) under
    # _world_to_base_link with robot at xyz=(-0.2, 0.2, 1.14), yaw=-3.141.
    # The transform: R_z(+3.141) @ (point_world - robot_xyz_world).
    # (point_world - robot_xyz_world) = (0.30, -0.50, 0).
    # R_z(+3.141) ≈ R_z(π); for π exactly: (-0.30, +0.50). With −3.141 (close
    # to but not exactly −π) the sign sort-of flips but with a tiny rotation.
    # Compute exactly via the helper to avoid duplicating its math:
    import gen_trial_config as gtc
    expected_xy_baselink = gtc._world_to_base_link(np.array([0.10, -0.30, 1.14]))
    assert abs(label.board_x_baselink - expected_xy_baselink[0]) < 1e-9
    assert abs(label.board_y_baselink - expected_xy_baselink[1]) < 1e-9
    # yaw_baselink = wrap(yaw_world − robot_yaw_world); robot yaw is −3.141.
    expected_yaw = _wrap_to_pi(0.0 - (-3.141))
    assert abs(label.board_yaw_baselink_rad - expected_yaw) < 1e-12
    assert label.target_rail_translation_m == 0.012
    assert label.port_type == "sfp"


def test_compute_label_sc_known_trial():
    """Verify SC label reads from sc_rail_<i>, not a mount rail."""
    trial = _make_sc_trial(
        board_x_world=0.20, board_y_world=0.10, board_yaw_world=1.0,
        sc_index=1, rail_translation=-0.02,
    )
    label = compute_label(trial)
    assert label.port_type == "sc"
    assert label.target_rail_translation_m == -0.02
    expected_yaw = _wrap_to_pi(1.0 - (-3.141))
    assert abs(label.board_yaw_baselink_rad - expected_yaw) < 1e-12


def test_compute_label_sc_unpopulated_target_rail_raises():
    """Defensive: an SC trial whose sc_rail_<i> has entity_present=False
    must raise so a silent zero translation doesn't poison the label."""
    trial = _make_sc_trial(sc_index=0, rail_translation=0.01)
    # Stomp the target rail to look "empty".
    trial["scene"]["task_board"]["sc_rail_0"] = {"entity_present": False}
    try:
        compute_label(trial)
    except (ValueError, KeyError):
        return
    raise AssertionError("expected error when target rail is unpopulated")


def test_reconstruct_port_round_trip_sfp():
    """compute_label → reconstruct_port_in_baselink should match the direct
    composition: world → board_in_baselink + R_yaw_baselink @ port_in_board.

    Pure-math regression — any sign error in the inverse breaks this.
    """
    import gen_trial_config as gtc
    trial = _make_sfp_trial(
        board_x_world=0.20, board_y_world=-0.15, board_yaw_world=0.7,
        nic_index=4, rail_translation=0.02, port_name="sfp_port_0",
    )
    label = compute_label(trial)
    task = trial["tasks"]["task_1"]
    predicted = reconstruct_port_in_baselink(
        label, task["target_module_name"], task["port_name"]
    )
    # Direct path: compute port-in-board manually using the same constants
    # the reconstructor uses, then transform to base_link via the helper.
    mount_x = gtc.NIC_BOARD_X_BASE + 0.02
    mount_y = gtc.NIC_RAIL_Y_BY_INDEX[4]
    mount_z = gtc.NIC_BOARD_Z
    ox, oy, oz = SFP_PORT_OFFSET_IN_MOUNT["sfp_port_0"]
    port_in_board = np.array([mount_x + ox, mount_y + oy, mount_z + oz])
    board_xyz_baselink = gtc._world_to_base_link(np.array([0.20, -0.15, 1.14]))
    yaw_baselink = label.board_yaw_baselink_rad
    expected = board_xyz_baselink + gtc._yaw_to_rotmat(yaw_baselink) @ port_in_board
    assert np.linalg.norm(predicted - expected) < 1e-9, (
        f"round-trip failed: predicted={predicted}, expected={expected}"
    )


def test_reconstruct_port_round_trip_sc():
    import gen_trial_config as gtc
    trial = _make_sc_trial(
        board_x_world=0.18, board_y_world=0.05, board_yaw_world=-1.2,
        sc_index=1, rail_translation=0.04,
    )
    label = compute_label(trial)
    task = trial["tasks"]["task_1"]
    predicted = reconstruct_port_in_baselink(
        label, task["target_module_name"], task["port_name"]
    )
    port_x = gtc.SC_PORT_BOARD_X_BASE + 0.04
    port_y = gtc.SC_PORT_Y_BY_INDEX[1]
    port_z = gtc.SC_PORT_BOARD_Z
    ox, oy, oz = SC_PORT_BASE_OFFSET_IN_BOARD
    port_in_board = np.array([port_x + ox, port_y + oy, port_z + oz])
    board_xyz_baselink = gtc._world_to_base_link(np.array([0.18, 0.05, 1.14]))
    yaw_baselink = label.board_yaw_baselink_rad
    expected = board_xyz_baselink + gtc._yaw_to_rotmat(yaw_baselink) @ port_in_board
    assert np.linalg.norm(predicted - expected) < 1e-9


def test_yaw_baselink_sign():
    """yaw_baselink = wrap(yaw_world − robot_mount_yaw_world)."""
    import gen_trial_config as gtc
    robot_yaw = gtc.ROBOT_BASE_LINK_IN_WORLD["yaw"]
    for yaw_world in [0.0, 0.5, math.pi, -math.pi / 3, 2.0 * math.pi]:
        expected = _wrap_to_pi(yaw_world - robot_yaw)
        trial = _make_sfp_trial(board_yaw_world=yaw_world, rail_translation=0.0)
        label = compute_label(trial)
        assert abs(label.board_yaw_baselink_rad - expected) < 1e-12, (
            f"yaw_world={yaw_world}: got {label.board_yaw_baselink_rad}, "
            f"expected {expected}"
        )


def test_yaw_wrap_to_pi():
    """Wrap to (-π, π]; angles equivalent mod 2π must produce equal values."""
    cases = [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),  # canonical form is +π (the upper-bound case)
        (3.5 * math.pi, _wrap_to_pi(3.5 * math.pi)),
        (-3.5 * math.pi, _wrap_to_pi(-3.5 * math.pi)),
    ]
    for raw, expected in cases:
        got = _wrap_to_pi(raw)
        assert -math.pi < got <= math.pi + 1e-12, f"out of range: {got}"
        assert abs(got - expected) < 1e-12, f"raw={raw}: got {got}, expected {expected}"
    # Equivalence under 2π.
    a = _wrap_to_pi(1.234)
    b = _wrap_to_pi(1.234 + 4.0 * math.pi)
    assert abs(a - b) < 1e-12


def test_label_as_target_5_sincos_identity():
    """target[2]² + target[3]² ≈ 1 always."""
    for yaw in [0.0, 0.5, 2.5, -1.7, math.pi - 0.01]:
        label = LocalizerLabel(
            board_x_baselink=0.1, board_y_baselink=-0.2,
            board_yaw_baselink_rad=yaw, target_rail_translation_m=0.0,
            port_type="sfp",
        )
        t = label.as_target_5()
        assert t.shape == (5,)
        assert abs(t[2] ** 2 + t[3] ** 2 - 1.0) < 1e-7
        # sin and cos in expected positions.
        assert abs(t[2] - math.sin(yaw)) < 1e-7
        assert abs(t[3] - math.cos(yaw)) < 1e-7


def test_match_episodes_to_trials_with_discard():
    """A summary mixing saved + discarded entries — only saved ones get
    saved-episode indices, mapped to the right trial_N."""
    yaml_trials = {
        "trial_1": _make_sfp_trial(nic_index=0),  # cable_0
        "trial_2": _make_sc_trial(sc_index=0),     # cable_1
        "trial_3": _make_sfp_trial(nic_index=2),
    }
    summary = {
        "trials": [
            {"idx": 0, "outcome": "saved_inserted",
             "task_meta": '{"cable_name": "cable_0", "target_module_name": "nic_card_mount_0", "port_name": "sfp_port_0", "port_type": "sfp", "plug_type": "sfp"}'},
            {"idx": 1, "outcome": "discarded_overlong",
             "task_meta": '{"cable_name": "cable_1", "target_module_name": "sc_port_0", "port_name": "sc_port_base", "port_type": "sc", "plug_type": "sc"}'},
            {"idx": 2, "outcome": "saved_inserted",
             "task_meta": '{"cable_name": "cable_0", "target_module_name": "nic_card_mount_2", "port_name": "sfp_port_0", "port_type": "sfp", "plug_type": "sfp"}'},
        ],
    }
    out = match_episodes_to_trials(summary, yaml_trials)
    # Only 2 saved → indices 0 and 1, mapping to trial_1 and trial_3.
    assert out == {0: "trial_1", 1: "trial_3"}, f"got {out}"


def test_match_episodes_disambiguates_duplicate_module():
    """Two trials with the same target_module_name but different cable_name
    must each match correctly."""
    # Both target nic_card_mount_2; one with cable_0 (sfp) one with cable_2 (also sfp,
    # different cable instance).
    t1 = _make_sfp_trial(nic_index=2)
    t2 = _make_sfp_trial(nic_index=2)
    t2["tasks"]["task_1"]["cable_name"] = "cable_2"
    yaml_trials = {"trial_1": t1, "trial_2": t2}
    summary = {
        "trials": [
            {"idx": 0, "outcome": "saved_inserted",
             "task_meta": '{"cable_name": "cable_2", "target_module_name": "nic_card_mount_2", "port_name": "sfp_port_0", "port_type": "sfp", "plug_type": "sfp"}'},
            {"idx": 1, "outcome": "saved_inserted",
             "task_meta": '{"cable_name": "cable_0", "target_module_name": "nic_card_mount_2", "port_name": "sfp_port_0", "port_type": "sfp", "plug_type": "sfp"}'},
        ],
    }
    out = match_episodes_to_trials(summary, yaml_trials)
    assert out == {0: "trial_2", 1: "trial_1"}, f"got {out}"


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_task_one_hot_orderings,
        test_task_one_hot_unknown_raises,
        test_compute_label_sfp_known_trial,
        test_compute_label_sc_known_trial,
        test_compute_label_sc_unpopulated_target_rail_raises,
        test_reconstruct_port_round_trip_sfp,
        test_reconstruct_port_round_trip_sc,
        test_yaw_baselink_sign,
        test_yaw_wrap_to_pi,
        test_label_as_target_5_sincos_identity,
        test_match_episodes_to_trials_with_discard,
        test_match_episodes_disambiguates_duplicate_module,
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
