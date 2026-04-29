"""Pure-function label generation for the port-localizer training pipeline.

The localizer NN learns to map (camera images + TCP pose + task identity) to
a 5-vector `(board_x, board_y, sin_yaw, cos_yaw, target_rail_translation)`
in the robot's base_link frame. At inference time the policy composes this
with the URDF formulas to produce a port pose, replacing CheatCodeRobust's
TF lookup.

This module produces training labels from a batch-config trial dict + a
recorder summary.json. No torch / lerobot / rclpy — host-runnable so the
unit tests don't require pixi.

The reconstruction inverse `reconstruct_port_in_baselink` is used by both
the killer integration test (verifies labels reproduce recorded
`groundtruth.port_pose`) and the inference shim. For SC ports the URDF
chain has a non-trivial rotation between sc_port_link and sc_port_base_link
that gen_trial_config.py's `_target_port_in_board` ignores (it adds the
2 mm sc_port_base offset in board-XY when the offset is actually along
board-Z). We correct that here with a board-frame offset constant.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# gen_trial_config lives in my_policy/scripts/, not in the package. Pulling its
# constants and helpers in via sys.path beats duplicating the URDF math; the
# alternative would be a separate "geometry" module that both consumers depend
# on, which is a larger refactor than this phase needs.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gen_trial_config as gtc  # noqa: E402


# Fixed task-target order — change only when the port inventory changes. NEVER
# read from a YAML; we want the network's output dimensionality to be a
# compile-time constant of the codebase, not a function of the scene config.
TASK_ONE_HOT_ORDER: list[str] = [
    "nic_card_mount_0",
    "nic_card_mount_1",
    "nic_card_mount_2",
    "nic_card_mount_3",
    "nic_card_mount_4",
    "sc_port_0",
    "sc_port_1",
]
TASK_ONE_HOT_DIM: int = len(TASK_ONE_HOT_ORDER)
_TASK_INDEX_BY_NAME: dict[str, int] = {n: i for i, n in enumerate(TASK_ONE_HOT_ORDER)}


# Corrected SC offset in BOARD frame (gen_trial_config.py applies (0,-0.002,0)
# directly which is wrong because sc_port_link is rotated roll=π/2, yaw=π/2
# relative to the board — the SDF child offset (0,-0.002,0) sits in the rotated
# frame, which composes to (0,0,-0.002) in board frame). We use the corrected
# value here so reconstruct_port_in_baselink reproduces the actual TF.
SC_PORT_BASE_OFFSET_IN_BOARD: tuple[float, float, float] = (0.0, 0.0, -0.002)

# Corrected SFP port offsets in NIC_CARD_MOUNT frame.
#
# gen_trial_config.NIC_TO_SFP_PORT lists offsets in nic_card_LINK frame
# (z ≈ 5 mm). But the SFP ports physically sit on the NIC card which is mounted
# vertically — nic_card_link sits at (-0.002, -0.01785, 0.0899) with rotation
# Rx(-π/2) inside nic_card_mount (per NIC Card Mount/model.sdf:71-72). After
# applying that rotation+translation, the port lands ~12 cm above the mount
# base.
#
# Composition (Rx(-π/2) @ link-frame offset + link-frame translation):
#   sfp_port_0: (0.01295, -0.031572, 0.00501) -> (0.01095, -0.01284, 0.121472)
#   sfp_port_1: (-0.01025, -0.031572, 0.00501) -> (-0.01225, -0.01284, 0.121472)
SFP_PORT_OFFSET_IN_MOUNT: dict[str, tuple[float, float, float]] = {
    "sfp_port_0": (0.01095, -0.01284, 0.121472),
    "sfp_port_1": (-0.01225, -0.01284, 0.121472),
}


@dataclass(frozen=True)
class LocalizerLabel:
    """Per-frame training label.

    All values are in robot base_link frame (the frame the NN must predict).
    `target_rail_translation_m` is the entity_pose.translation along the
    target rail (board-X axis for both NIC and SC). `port_type` ∈ {"sfp", "sc"}
    is carried so the inference-side reconstruction picks the right URDF math.
    """

    board_x_baselink: float
    board_y_baselink: float
    board_yaw_baselink_rad: float
    target_rail_translation_m: float
    port_type: str  # "sfp" | "sc"

    def as_target_5(self) -> np.ndarray:
        """Network regression target: (bx, by, sin_yaw, cos_yaw, rail_t)."""
        return np.array([
            self.board_x_baselink,
            self.board_y_baselink,
            math.sin(self.board_yaw_baselink_rad),
            math.cos(self.board_yaw_baselink_rad),
            self.target_rail_translation_m,
        ], dtype=np.float32)


def task_one_hot(target_module_name: str) -> np.ndarray:
    """Return a 7-dim one-hot for the named target. Raises ValueError if unknown.

    The order is fixed by TASK_ONE_HOT_ORDER. New port inventory in the future
    requires explicitly editing that list (and any consumer's input dimension).
    """
    idx = _TASK_INDEX_BY_NAME.get(target_module_name)
    if idx is None:
        raise ValueError(
            f"unknown target_module_name {target_module_name!r}; "
            f"expected one of {TASK_ONE_HOT_ORDER}"
        )
    out = np.zeros(TASK_ONE_HOT_DIM, dtype=np.float32)
    out[idx] = 1.0
    return out


def _wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to (-π, π]."""
    a = (angle_rad + math.pi) % (2.0 * math.pi)
    if a == 0.0:
        return math.pi
    return a - math.pi


def _board_pose_in_baselink(trial: dict) -> tuple[np.ndarray, float]:
    """Returns (board_xyz_baselink, board_yaw_baselink_rad).

    Uses gen_trial_config._world_to_base_link for the position and the
    yaw composition rule yaw_baselink = yaw_world − robot_mount_yaw_world.
    """
    board_xyz_world, board_yaw_world = gtc._board_pose_to_world(trial)
    board_xyz_baselink = gtc._world_to_base_link(board_xyz_world)
    yaw_baselink = _wrap_to_pi(
        board_yaw_world - gtc.ROBOT_BASE_LINK_IN_WORLD["yaw"]
    )
    return board_xyz_baselink, yaw_baselink


def _target_rail_translation(trial: dict) -> float:
    """Read entity_pose.translation for the trial's target rail.

    Maps target_module_name → rail key (`nic_rail_<i>` for SFP,
    `sc_rail_<i>` for SC) and reads the translation value. Raises if the
    rail isn't populated (which shouldn't happen — the target is always
    populated by gen_trial_config — but a missing key would silently give
    wrong labels otherwise).
    """
    task = trial["tasks"]["task_1"]
    port_type = task["port_type"]
    target_module = task["target_module_name"]
    scene_rails = trial["scene"]["task_board"]
    if port_type == "sfp":
        i = int(target_module.rsplit("_", 1)[-1])
        rail_key = f"nic_rail_{i}"
    elif port_type == "sc":
        i = int(target_module.rsplit("_", 1)[-1])
        rail_key = f"sc_rail_{i}"
    else:
        raise ValueError(f"unknown port_type {port_type!r}")

    if rail_key not in scene_rails:
        raise KeyError(
            f"target rail {rail_key!r} missing from trial scene; "
            f"target {target_module!r} has no rail entry"
        )
    rail = scene_rails[rail_key]
    if not rail.get("entity_present", False):
        raise ValueError(
            f"target rail {rail_key!r} is marked entity_present=False; "
            f"target {target_module!r} cannot have its translation read"
        )
    return float(rail["entity_pose"]["translation"])


def compute_label(trial: dict) -> LocalizerLabel:
    """Build a per-trial label from a batch-config trial dict."""
    task = trial["tasks"]["task_1"]
    board_xyz_baselink, board_yaw_baselink = _board_pose_in_baselink(trial)
    rail_t = _target_rail_translation(trial)
    return LocalizerLabel(
        board_x_baselink=float(board_xyz_baselink[0]),
        board_y_baselink=float(board_xyz_baselink[1]),
        board_yaw_baselink_rad=float(board_yaw_baselink),
        target_rail_translation_m=rail_t,
        port_type=task["port_type"],
    )


def _port_in_board(
    port_type: str,
    target_module_name: str,
    port_name: str,
    rail_t: float,
) -> np.ndarray:
    """Return port-link xyz in board frame, given port identity + rail_t.

    Mirrors gen_trial_config._target_port_in_board's translation arithmetic
    but applies the SC sub-offset in board-Z (corrected, see header comment)
    and reads rail_t directly rather than from a trial dict.
    """
    if port_type == "sfp":
        i = int(target_module_name.rsplit("_", 1)[-1])
        mount_x = gtc.NIC_BOARD_X_BASE + rail_t
        mount_y = gtc.NIC_RAIL_Y_BY_INDEX[i]
        mount_z = gtc.NIC_BOARD_Z
        if port_name not in SFP_PORT_OFFSET_IN_MOUNT:
            raise ValueError(
                f"unknown SFP port_name {port_name!r}; "
                f"expected one of {list(SFP_PORT_OFFSET_IN_MOUNT.keys())}"
            )
        ox, oy, oz = SFP_PORT_OFFSET_IN_MOUNT[port_name]
        return np.array([mount_x + ox, mount_y + oy, mount_z + oz])

    if port_type == "sc":
        i = int(target_module_name.rsplit("_", 1)[-1])
        port_x = gtc.SC_PORT_BOARD_X_BASE + rail_t
        port_y = gtc.SC_PORT_Y_BY_INDEX[i]
        port_z = gtc.SC_PORT_BOARD_Z
        ox, oy, oz = SC_PORT_BASE_OFFSET_IN_BOARD
        return np.array([port_x + ox, port_y + oy, port_z + oz])

    raise ValueError(f"unknown port_type {port_type!r}")


def reconstruct_port_in_baselink(
    label: LocalizerLabel,
    target_module_name: str,
    port_name: str,
) -> np.ndarray:
    """Inverse of compute_label: predicts the port xyz in base_link from a label.

    Used by the killer integration test (compares to recorded
    groundtruth.port_pose) and at inference time (composes localizer output
    with task strings to produce a port pose). target_module_name + port_name
    are NOT in the label because they're known from the engine's Task message
    at inference; passing them explicitly avoids a redundant lookup.
    """
    port_in_board = _port_in_board(
        port_type=label.port_type,
        target_module_name=target_module_name,
        port_name=port_name,
        rail_t=label.target_rail_translation_m,
    )
    R = gtc._yaw_to_rotmat(label.board_yaw_baselink_rad)
    board_xyz_baselink = np.array([
        label.board_x_baselink,
        label.board_y_baselink,
        # Board Z is fixed and not in the label; reuse the world-frame Z and
        # apply _world_to_base_link to get its base_link Z (the robot mount has
        # zero Z offset relative to the board so this collapses to BOARD_Z, but
        # keep it general).
        gtc._world_to_base_link(np.array([0.0, 0.0, gtc.BOARD_Z]))[2],
    ])
    return board_xyz_baselink + R @ port_in_board


def _quat_multiply_wxyz(q1: tuple, q2: tuple) -> tuple:
    """Hamilton product, (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _yaw_quat(yaw_rad: float) -> tuple:
    """R_z(yaw) as (w, x, y, z)."""
    half = yaw_rad * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def calibrate_port_in_board_rotations(
    dataset_root: Path,
    batch_yaml: Path,
    summary_json: Path,
) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """Empirically derive Q_port_in_board per (port_type, port_name).

    Inverse of: Q_port_baselink = R_z(board_yaw) ⊗ Q_port_in_board
    Therefore: Q_port_in_board = R_z(-board_yaw) ⊗ Q_port_baselink

    The static value should be identical across all frames of any given trial
    (Gazebo TF noise aside) and across rail indices for the same port_name
    (geometry of the port frame itself doesn't depend on which rail it's on).
    Averages across all frames of all trials sharing a (port_type, port_name).

    Returns a dict suitable for handing to PortLocalizer.set_port_in_board_quat()
    or for hardcoding as a constant.

    Reads parquet directly to avoid needing lerobot/torch.
    """
    import pyarrow.parquet as pq
    import yaml as _yaml

    cfg = _yaml.safe_load(Path(batch_yaml).read_text())
    summary = json.loads(Path(summary_json).read_text())
    ep_to_trial = match_episodes_to_trials(summary, cfg["trials"])

    parquet_path = Path(dataset_root) / "data" / "chunk-000" / "file-000.parquet"
    info = json.loads((Path(dataset_root) / "meta" / "info.json").read_text())
    state_names = info["features"]["observation.state"]["names"]
    quat_idx = [state_names.index(f"groundtruth.port_pose.q{c}") for c in "xyzw"]

    table = pq.read_table(str(parquet_path),
                           columns=["observation.state", "episode_index"])
    eps = table["episode_index"].to_numpy().astype(np.int64)
    states = np.stack([
        np.asarray(r, dtype=np.float64) for r in table["observation.state"].to_pylist()
    ])

    # Group all (qw, qx, qy, qz) of port_in_board by (port_type, port_name).
    samples: dict[tuple[str, str], list[tuple[float, ...]]] = {}
    for ep, trial_key in ep_to_trial.items():
        trial = cfg["trials"][trial_key]
        task = trial["tasks"]["task_1"]
        port_type = task["port_type"]
        port_name = task["port_name"]
        label = compute_label(trial)
        # Take the first frame's quaternion of this episode (constant, but
        # averaging over all frames adds robustness to TF noise).
        mask = eps == ep
        if not mask.any():
            continue
        ep_states = states[mask]
        # Q_baselink reading: parquet stores (qx, qy, qz, qw); convert to (w,x,y,z).
        qx, qy, qz, qw = ep_states[:, quat_idx].T
        for k in range(len(qw)):
            q_port_baselink = (float(qw[k]), float(qx[k]), float(qy[k]), float(qz[k]))
            q_unyaw = _yaw_quat(-label.board_yaw_baselink_rad)
            q_port_in_board = _quat_multiply_wxyz(q_unyaw, q_port_baselink)
            samples.setdefault((port_type, port_name), []).append(q_port_in_board)

    out: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for key, quats in samples.items():
        arr = np.asarray(quats)
        # Quaternion averaging: simple mean is biased but works when the spread
        # is tiny (sub-degree). Re-normalize to unit quaternion.
        # For more spread, would need quaternion-aware averaging (Markley method).
        mean = arr.mean(axis=0)
        mean = mean / np.linalg.norm(mean)
        # Sanity: per-frame std should be ~1e-5 if the static rotation is real.
        spread = float(np.std(arr, axis=0).max())
        if spread > 1e-3:
            print(
                f"WARNING: large spread in port_in_board quat for {key}: "
                f"std_max={spread:.5f}. Calibration may be unreliable."
            )
        out[key] = (float(mean[0]), float(mean[1]), float(mean[2]), float(mean[3]))
    return out


def match_episodes_to_trials(
    summary: dict,
    yaml_trials: dict,
) -> dict[int, str]:
    """Map saved-episode-index → trial_N key.

    The eval engine processes trial_1, trial_2, ... in YAML key order; the
    recorder's summary.json captures every attempt (saved or discarded) in
    that same chronological order. So summary entry at position i ALWAYS
    corresponds to yaml trial_(i+1), regardless of outcome.

    We walk summary in order, count discards, and assign each saved entry to
    the yaml trial at that position. task_meta is cross-checked against the
    yaml's task to catch any mid-run summary corruption (e.g. recorder
    restart that overwrites summary.json while the dataset keeps appending).

    The previous bucket-and-ordinal-fallback implementation was incorrect:
    its key (target_module_name, cable_name) didn't include port_name, so
    SFP trials with the same target module but different sfp_port_0/_1 got
    mixed up. The simpler positional match avoids that class of bug entirely
    AND validates task_meta as a sanity check.
    """
    out: dict[int, str] = {}
    saved_idx = 0
    for i, entry in enumerate(summary.get("trials", [])):
        yaml_key = f"trial_{i + 1}"
        if yaml_key not in yaml_trials:
            raise ValueError(
                f"summary has entry at position {i} but yaml has no {yaml_key!r}; "
                f"summary and yaml are out of sync (mid-run crash + restart "
                f"with overwritten summary.json is a known cause)"
            )
        outcome = entry.get("outcome", "")
        if outcome != "saved_inserted":
            continue

        # Verify task_meta agrees with the positionally-matched yaml trial.
        # Catches the corruption case where summary.json was overwritten while
        # the dataset kept appending across runs.
        meta_str = entry.get("task_meta", "")
        if meta_str:
            meta = json.loads(meta_str)
            yaml_task = yaml_trials[yaml_key]["tasks"]["task_1"]
            for field in ("target_module_name", "port_name", "cable_name"):
                if meta.get(field) != yaml_task.get(field):
                    raise ValueError(
                        f"task_meta mismatch at summary position {i} -> {yaml_key}: "
                        f"summary {field}={meta.get(field)!r} vs yaml "
                        f"{field}={yaml_task.get(field)!r}. Dataset/summary "
                        f"out of sync."
                    )
        out[saved_idx] = yaml_key
        saved_idx += 1
    return out
