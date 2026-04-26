#!/usr/bin/env python3
"""Generate randomized trial configs for AIC oracle data collection.

Reads aic_engine/config/sample_config.yaml as a template for the static
sections (scoring, task_board_limits, robot) and replaces the `trials`
section with a single randomized trial per output file.

Usage:
    gen_trial_config.py --out configs/ --n 500 --seed 42
    gen_trial_config.py --out configs/ --n 50 --task-type sfp
"""

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import yaml


# Rail-limit constants from aic_engine/config/sample_config.yaml task_board_limits.
NIC_RAIL_MIN, NIC_RAIL_MAX = -0.0215, 0.0234
SC_RAIL_MIN, SC_RAIL_MAX = -0.06, 0.055
MOUNT_RAIL_MIN, MOUNT_RAIL_MAX = -0.09425, 0.09425

# Task-board pose randomization. z is fixed at the table height; roll/pitch=0
# per the sample configs. Yaw is unconstrained per the organizer on Discourse.
# xy bounds keep the board inside the UR5e kinematic workspace — derived from
# the two sample poses (0.15,-0.2) and (0.17,0.0), widened by ±0.05m margin.
BOARD_X_MIN, BOARD_X_MAX = 0.10, 0.25
BOARD_Y_MIN, BOARD_Y_MAX = -0.30, 0.30
BOARD_Z = 1.14

# Entity-pose small-angle jitter (radians) for distractor yaw. Keep small to
# avoid mounts colliding with each other on the rails.
MOUNT_YAW_JITTER = 0.3

# Cable gripper offsets, copied from sample_config.yaml (trial_1 for SFP,
# trial_3 for SC). Oracle expects these to match the physical grasp.
SFP_CABLE_OFFSET = {"x": 0.0, "y": 0.015385, "z": 0.04245,
                    "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303}
SC_CABLE_OFFSET = {"x": 0.0, "y": 0.015385, "z": 0.04045,
                   "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303}

NIC_RAILS = [f"nic_rail_{i}" for i in range(5)]
SC_RAILS = [f"sc_rail_{i}" for i in range(2)]
MOUNT_RAILS = [f"{t}_mount_rail_{i}" for t in ("lc", "sfp", "sc") for i in range(2)]


# =============================================================================
# Port-visibility predicate (filters out configs where target port is not in
# any camera frame at the fixed home pose, matching the eval guarantee in
# qualification_phase.md:44/58).
# =============================================================================

# --- Camera intrinsics (Basler acA2440-20gc, from basler_camera_macro.xacro:83-100)
HFOV_RAD = 0.8718
IMG_W, IMG_H = 1152, 1024
FX = (IMG_W / 2.0) / math.tan(HFOV_RAD / 2.0)  # ~1236.4 px
FY = FX  # Gazebo default: square pixels, no separate v-FOV
CX = IMG_W / 2.0
CY = IMG_H / 2.0
VISIBILITY_MARGIN_PX = 50  # require port to be ≥50 px from any image edge

# --- Robot base_link pose in world frame. The trial YAML specifies the task
# board pose in WORLD frame (engine convention), but our visibility predicate
# operates in base_link (where the cameras live). So we need this transform.
# Defaults from aic_bringup/launch/aic_gz_bringup.launch.py:595-625.
ROBOT_BASE_LINK_IN_WORLD = {
    "xyz": (-0.2, 0.2, 1.14),
    "yaw": -3.141,
}

# --- Camera optical-frame poses in base_link at robot home pose.
# Captured via TF on 2026-04-25 with `ros2 run tf2_ros tf2_echo base_link
# <cam>_camera/optical` against a sim spawned with the home joints from
# sample_config.yaml:328-334. The optical-frame convention is REP-103
# (X right, Y down, Z forward out of lens). Each entry: (x,y,z) translation
# in m and (qx, qy, qz, qw) rotation.
CAMERA_OPTICAL_IN_BASE_LINK = {
    "left":   {"xyz": (-0.472, 0.253, 0.534),
               "quat": (-0.859, -0.496, -0.065, -0.113)},
    "center": {"xyz": (-0.371, 0.311, 0.534),
               "quat": (-0.991,  0.000,  0.000, -0.131)},
    "right":  {"xyz": (-0.271, 0.253, 0.534),
               "quat": (-0.859,  0.496,  0.065, -0.113)},
}

# --- Board → port offsets (from task_board.urdf.xacro and component model.sdf).

# nic_card_mount_<i>_link y-position on board (URDF: task_board.urdf.xacro:201-254).
NIC_RAIL_Y_BY_INDEX = [-0.1745, -0.1345, -0.0945, -0.0545, -0.0145]
NIC_BOARD_X_BASE = -0.081418   # x = NIC_BOARD_X_BASE + translation
NIC_BOARD_Z = 0.012

# sfp_port_<j>_link offset within nic_card_mount frame (NIC Card Mount/model.sdf:179-204).
NIC_TO_SFP_PORT = {
    "sfp_port_0": (0.01295, -0.031572, 0.00501),
    "sfp_port_1": (-0.01025, -0.031572, 0.00501),
}

# sc_port_<i>_link y-position on board (task_board.urdf.xacro:181, 192).
SC_PORT_Y_BY_INDEX = [0.0295, 0.0705]
SC_PORT_BOARD_X_BASE = -0.075  # x = SC_PORT_BOARD_X_BASE + translation
SC_PORT_BOARD_Z = 0.0165
# sc_port_base_link offset within sc_port frame (SC Port/model.sdf:118-120). Tiny, but for completeness.
SC_PORT_TO_BASE = (0.0, -0.002, 0.0)


def _quat_to_rotmat(q: tuple) -> np.ndarray:
    """Convert (qx, qy, qz, qw) to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _yaw_to_rotmat(yaw: float) -> np.ndarray:
    """Z-axis rotation only (board placement has roll=pitch=0)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _target_port_in_board(trial: dict) -> np.ndarray:
    """Compute target port position in task_board_base_link frame from a trial dict.

    Handles SFP (nic_card_mount.sfp_port_*) and SC (standalone sc_port_*).
    Returns (x, y, z) in board frame.
    """
    task = trial["tasks"]["task_1"]
    port_type = task["port_type"]
    target_module = task["target_module_name"]  # e.g. "nic_card_mount_2" or "sc_port_1"
    port_name = task["port_name"]               # e.g. "sfp_port_0" or "sc_port_base"
    scene_rails = trial["scene"]["task_board"]

    if port_type == "sfp":
        # target_module is "nic_card_mount_<i>"; translation lives in scene_rails["nic_rail_<i>"]
        i = int(target_module.rsplit("_", 1)[-1])
        rail_key = f"nic_rail_{i}"
        translation = scene_rails[rail_key]["entity_pose"]["translation"]
        mount_x = NIC_BOARD_X_BASE + translation
        mount_y = NIC_RAIL_Y_BY_INDEX[i]
        mount_z = NIC_BOARD_Z
        port_off = NIC_TO_SFP_PORT[port_name]
        return np.array([mount_x + port_off[0], mount_y + port_off[1], mount_z + port_off[2]])

    if port_type == "sc":
        # target_module is "sc_port_<i>"; translation in scene_rails["sc_rail_<i>"] (engine maps it).
        i = int(target_module.rsplit("_", 1)[-1])
        rail_key = f"sc_rail_{i}"
        translation = scene_rails[rail_key]["entity_pose"]["translation"] if rail_key in scene_rails else 0.0
        port_x = SC_PORT_BOARD_X_BASE + translation
        port_y = SC_PORT_Y_BY_INDEX[i]
        port_z = SC_PORT_BOARD_Z
        # Apply tiny sc_port_base offset; ignoring port frame rotation since offset magnitude is ~2mm.
        return np.array([port_x + SC_PORT_TO_BASE[0], port_y + SC_PORT_TO_BASE[1], port_z + SC_PORT_TO_BASE[2]])

    raise ValueError(f"unknown port_type: {port_type}")


def _board_pose_to_world(trial: dict) -> tuple:
    """Returns (translation_xyz, yaw) for the board pose. Roll/pitch are 0 by convention."""
    p = trial["scene"]["task_board"]["pose"]
    return np.array([p["x"], p["y"], p["z"]]), float(p["yaw"])


def _project_to_pixel(point_base_link: np.ndarray, cam_name: str) -> tuple:
    """Project a 3D point in base_link to (u, v) pixel in the named camera.
    Returns (u, v, z) where z is depth in optical frame (z>0 means in front).
    Returns (None, None, None) if camera transform constants are not yet filled in.
    """
    cam = CAMERA_OPTICAL_IN_BASE_LINK[cam_name]
    if cam["xyz"] is None or cam["quat"] is None:
        return None, None, None
    R_b_o = _quat_to_rotmat(cam["quat"])  # base_link -> optical
    t_b_o = np.array(cam["xyz"])
    # Inverse: point in optical frame
    p_optical = R_b_o.T @ (point_base_link - t_b_o)
    z = p_optical[2]
    if z <= 0:
        return None, None, z  # behind camera
    u = FX * p_optical[0] / z + CX
    v = FY * p_optical[1] / z + CY
    return u, v, z


def _world_to_base_link(point_world: np.ndarray) -> np.ndarray:
    """Transform a point from world frame to base_link frame.

    Uses the fixed robot mount pose (translation + yaw-only rotation) from
    aic_bringup/launch/aic_gz_bringup.launch.py defaults.
    """
    t = np.array(ROBOT_BASE_LINK_IN_WORLD["xyz"])
    yaw = ROBOT_BASE_LINK_IN_WORLD["yaw"]
    R_world_to_base = _yaw_to_rotmat(-yaw)  # inverse of yaw rotation = -yaw
    return R_world_to_base @ (point_world - t)


def target_port_visible_at_spawn(trial: dict, margin_px: float = VISIBILITY_MARGIN_PX) -> tuple:
    """Predicate: is the target port projected inside any camera's image (with margin)?

    Returns (visible: bool, per_camera: dict) where per_camera maps cam_name to
    (u, v, depth) tuples (or None if behind camera). Useful for diagnostics.
    """
    if any(c["xyz"] is None for c in CAMERA_OPTICAL_IN_BASE_LINK.values()):
        # Constants not yet captured — skip the check, accept everything.
        return True, {}

    # board pose in YAML is WORLD frame; transform composed result through
    # world → base_link before projecting through camera intrinsics.
    port_in_board = _target_port_in_board(trial)
    board_t_world, board_yaw_world = _board_pose_to_world(trial)
    port_in_world = board_t_world + _yaw_to_rotmat(board_yaw_world) @ port_in_board
    port_in_base_link = _world_to_base_link(port_in_world)

    per_cam = {}
    visible_any = False
    for cam_name in ("left", "center", "right"):
        u, v, z = _project_to_pixel(port_in_base_link, cam_name)
        per_cam[cam_name] = (u, v, z)
        if u is None:
            continue
        if (margin_px <= u < IMG_W - margin_px) and (margin_px <= v < IMG_H - margin_px):
            visible_any = True
    return visible_any, per_cam


def sample_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def sample_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def empty_rail() -> dict:
    return {"entity_present": False}


def nic_card_entity(rng: random.Random, index: int) -> dict:
    return {
        "entity_present": True,
        "entity_name": f"nic_card_{index}",
        "entity_pose": {
            "translation": sample_uniform(rng, NIC_RAIL_MIN, NIC_RAIL_MAX),
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        },
    }


def sc_mount_entity(rng: random.Random, index: int) -> dict:
    return {
        "entity_present": True,
        "entity_name": f"sc_mount_{index}",
        "entity_pose": {
            "translation": sample_uniform(rng, SC_RAIL_MIN, SC_RAIL_MAX),
            "roll": 0.0, "pitch": 0.0,
            "yaw": sample_uniform(rng, -MOUNT_YAW_JITTER, MOUNT_YAW_JITTER),
        },
    }


def distractor_mount_entity(rng: random.Random, rail: str) -> dict:
    """Populate a mount rail with a plausible distractor mount.

    The sample configs use lc_mount_N, sfp_mount_N, sc_mount_N on the
    *_mount_rail rails. We pick an entity type matching the rail's prefix
    since the sample configs always do that (no evidence mismatched entities
    spawn correctly).
    """
    prefix = rail.split("_mount_rail_")[0]  # "lc" | "sfp" | "sc"
    index = rng.randint(0, 2)
    return {
        "entity_present": True,
        "entity_name": f"{prefix}_mount_{index}",
        "entity_pose": {
            "translation": sample_uniform(rng, MOUNT_RAIL_MIN, MOUNT_RAIL_MAX),
            "roll": 0.0, "pitch": 0.0,
            "yaw": sample_uniform(rng, -MOUNT_YAW_JITTER, MOUNT_YAW_JITTER),
        },
    }


def gen_sfp_trial(rng: random.Random, distractor_count: int) -> dict:
    """Randomized SFP-insertion trial.

    Places one NIC card on a random NIC rail; the cable's SFP plug is
    inserted into that card's sfp_port_0. Remaining NIC rails empty.
    Mount rails populated with `distractor_count` random distractors.
    """
    nic_index = rng.randint(0, 4)
    nic_rail_choice = NIC_RAILS[nic_index]

    scene_rails = {}
    for rail in NIC_RAILS:
        scene_rails[rail] = nic_card_entity(rng, nic_index) if rail == nic_rail_choice else empty_rail()
    for rail in SC_RAILS:
        scene_rails[rail] = empty_rail()

    mount_distractors = set(rng.sample(MOUNT_RAILS, distractor_count))
    for rail in MOUNT_RAILS:
        scene_rails[rail] = distractor_mount_entity(rng, rail) if rail in mount_distractors else empty_rail()

    scene = {
        "task_board": {
            "pose": {
                "x": sample_uniform(rng, BOARD_X_MIN, BOARD_X_MAX),
                "y": sample_uniform(rng, BOARD_Y_MIN, BOARD_Y_MAX),
                "z": BOARD_Z,
                "roll": 0.0, "pitch": 0.0,
                "yaw": sample_uniform(rng, 0.0, 2 * 3.14159265),
            },
            **scene_rails,
        },
        "cables": {
            "cable_0": {
                "pose": {
                    "gripper_offset": {k: SFP_CABLE_OFFSET[k] for k in ("x", "y", "z")},
                    "roll": SFP_CABLE_OFFSET["roll"],
                    "pitch": SFP_CABLE_OFFSET["pitch"],
                    "yaw": SFP_CABLE_OFFSET["yaw"],
                },
                "attach_cable_to_gripper": True,
                "cable_type": "sfp_sc_cable",
            },
        },
    }

    tasks = {
        "task_1": {
            "cable_type": "sfp_sc",
            "cable_name": "cable_0",
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": "sfp_port_0",
            "target_module_name": f"nic_card_mount_{nic_index}",
            "time_limit": 40,
        },
    }
    return {"scene": scene, "tasks": tasks}


def gen_sc_trial(rng: random.Random, distractor_count: int) -> dict:
    """Randomized SC-insertion trial (mirrors trial_3 in sample_config.yaml).

    Places one SC mount on a random SC rail; the cable's SC plug goes into
    that mount's sc_port_base. Remaining NIC + SC rails empty.
    """
    sc_index = rng.randint(0, 1)
    sc_rail_choice = SC_RAILS[sc_index]

    scene_rails = {}
    for rail in NIC_RAILS:
        scene_rails[rail] = empty_rail()
    for rail in SC_RAILS:
        scene_rails[rail] = sc_mount_entity(rng, sc_index) if rail == sc_rail_choice else empty_rail()

    mount_distractors = set(rng.sample(MOUNT_RAILS, distractor_count))
    for rail in MOUNT_RAILS:
        scene_rails[rail] = distractor_mount_entity(rng, rail) if rail in mount_distractors else empty_rail()

    scene = {
        "task_board": {
            "pose": {
                "x": sample_uniform(rng, BOARD_X_MIN, BOARD_X_MAX),
                "y": sample_uniform(rng, BOARD_Y_MIN, BOARD_Y_MAX),
                "z": BOARD_Z,
                "roll": 0.0, "pitch": 0.0,
                "yaw": sample_uniform(rng, 0.0, 2 * 3.14159265),
            },
            **scene_rails,
        },
        "cables": {
            "cable_1": {
                "pose": {
                    "gripper_offset": {k: SC_CABLE_OFFSET[k] for k in ("x", "y", "z")},
                    "roll": SC_CABLE_OFFSET["roll"],
                    "pitch": SC_CABLE_OFFSET["pitch"],
                    "yaw": SC_CABLE_OFFSET["yaw"],
                },
                "attach_cable_to_gripper": True,
                "cable_type": "sfp_sc_cable_reversed",
            },
        },
    }

    tasks = {
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
    }
    return {"scene": scene, "tasks": tasks}


def _gen_trial_unchecked(rng: random.Random, task_type: str, distractor_count: int) -> dict:
    if task_type == "sfp":
        return gen_sfp_trial(rng, distractor_count)
    elif task_type == "sc":
        return gen_sc_trial(rng, distractor_count)
    else:
        raise ValueError(f"unknown task_type: {task_type}")


_GEN_REJECTION_STATS = {"attempts": 0, "rejections": 0}


def gen_trial(
    rng: random.Random,
    task_type: str,
    distractor_count: int,
    max_attempts: int = 200,
) -> dict:
    """Sample a trial whose target port projects inside ≥1 camera at home pose.

    If CAMERA_OPTICAL_IN_BASE_LINK constants are not yet captured, this falls
    back to a single unchecked sample (predicate returns True vacuously).
    """
    last_per_cam = None
    for attempt in range(max_attempts):
        trial = _gen_trial_unchecked(rng, task_type, distractor_count)
        visible, per_cam = target_port_visible_at_spawn(trial)
        last_per_cam = per_cam
        _GEN_REJECTION_STATS["attempts"] += 1
        if visible:
            return trial
        _GEN_REJECTION_STATS["rejections"] += 1
    raise RuntimeError(
        f"failed to generate a visible {task_type} trial after {max_attempts} attempts; "
        f"last projection: {last_per_cam}. Tighten BOARD_X/Y_MIN/MAX or check camera constants."
    )


def gen_config(
    template: dict,
    rng: random.Random,
    task_type: str,
    distractor_count: int,
) -> dict:
    config = copy.deepcopy(template)
    config["trials"] = {"trial_1": gen_trial(rng, task_type, distractor_count)}
    return config


def gen_batch_config(
    template: dict,
    rng: random.Random,
    n_trials: int,
    task_type: str,
    distractor_range: tuple,
) -> dict:
    """Pack N randomized trials into one config for batch evaluation.

    aic_engine runs trials sequentially, resetting the scene between each, so
    this lets us record N episodes in a single eval-container lifecycle.
    """
    config = copy.deepcopy(template)
    trials = {}
    dmin, dmax = distractor_range
    for i in range(n_trials):
        tt = rng.choice(["sfp", "sc"]) if task_type == "mixed" else task_type
        trials[f"trial_{i + 1}"] = gen_trial(rng, tt, rng.randint(dmin, dmax))
    config["trials"] = trials
    return config


def config_hash(config: dict) -> str:
    """Stable hash of the config content for reproducibility tagging."""
    blob = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", type=Path,
                   default=Path(__file__).resolve().parents[2] / "aic_engine" / "config" / "sample_config.yaml",
                   help="Template YAML providing scoring/task_board_limits/robot sections.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output path: directory (per-config mode) OR a file ending in .yaml (batch mode).")
    p.add_argument("--n", type=int, default=100,
                   help="Per-config mode: number of single-trial configs to generate.")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Batch mode: pack this many random trials into ONE config written to --out (file).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--task-type", choices=["sfp", "sc", "mixed"], default="mixed",
                   help="Which task type(s) to generate. 'mixed' picks per-trial uniformly.")
    p.add_argument("--distractor-min", type=int, default=0, help="Min distractor mounts per trial.")
    p.add_argument("--distractor-max", type=int, default=4, help="Max distractor mounts per trial.")
    p.add_argument("--validate", type=Path, default=None,
                   help="Validation mode: load a config YAML and report per-trial visibility. No generation.")
    args = p.parse_args()

    if args.validate is not None:
        with args.validate.open() as f:
            cfg = yaml.safe_load(f)
        constants_ready = all(c["xyz"] is not None for c in CAMERA_OPTICAL_IN_BASE_LINK.values())
        if not constants_ready:
            print("WARNING: CAMERA_OPTICAL_IN_BASE_LINK not yet captured — predicate is vacuously True.")
        n_visible = 0
        rows = []
        for tname, trial in cfg.get("trials", {}).items():
            visible, per_cam = target_port_visible_at_spawn(trial)
            n_visible += int(visible)
            rows.append((tname, trial["tasks"]["task_1"]["port_type"], visible, per_cam))
        for tname, ptype, visible, per_cam in rows:
            mark = "OK " if visible else "OUT"
            cams = " ".join(
                f"{k}=" + ("behind" if v[2] is not None and v[2] <= 0
                           else "—" if v[0] is None
                           else f"({v[0]:.0f},{v[1]:.0f},{v[2]:.2f}m)")
                for k, v in per_cam.items()
            ) if per_cam else "(skipped)"
            print(f"[{mark}] {tname:>10s} {ptype:>3s}  {cams}")
        total = len(rows)
        print(f"\nVisible: {n_visible}/{total} = {100.0 * n_visible / max(total, 1):.1f}%")
        return

    with args.template.open() as f:
        template = yaml.safe_load(f)

    rng = random.Random(args.seed)

    if args.n_trials is not None:
        # Batch mode: one file, N trials in it.
        config = gen_batch_config(
            template, rng, args.n_trials, args.task_type,
            (args.distractor_min, args.distractor_max),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        rej = _GEN_REJECTION_STATS
        rej_pct = 100.0 * rej["rejections"] / max(rej["attempts"], 1)
        print(f"Wrote batch config with {args.n_trials} trials to {args.out} "
              f"(hash {config_hash(config)}); visibility rejection rate "
              f"{rej['rejections']}/{rej['attempts']} ({rej_pct:.1f}%)")
        return

    # Per-config mode: N files, one trial each.
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        tt = rng.choice(["sfp", "sc"]) if args.task_type == "mixed" else args.task_type
        distractors = rng.randint(args.distractor_min, args.distractor_max)
        config = gen_config(template, rng, tt, distractors)
        h = config_hash(config)
        name = f"trial_{i:05d}_{tt}_{h}.yaml"
        with (args.out / name).open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    rej = _GEN_REJECTION_STATS
    rej_pct = 100.0 * rej["rejections"] / max(rej["attempts"], 1)
    print(f"Wrote {args.n} configs to {args.out}; visibility rejection rate "
          f"{rej['rejections']}/{rej['attempts']} ({rej_pct:.1f}%)")


if __name__ == "__main__":
    main()
