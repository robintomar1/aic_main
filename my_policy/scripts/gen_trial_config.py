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
import random
from pathlib import Path

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
            "time_limit": 180,
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
            "time_limit": 180,
        },
    }
    return {"scene": scene, "tasks": tasks}


def gen_config(
    template: dict,
    rng: random.Random,
    task_type: str,
    distractor_count: int,
) -> dict:
    config = copy.deepcopy(template)
    if task_type == "sfp":
        trial = gen_sfp_trial(rng, distractor_count)
    elif task_type == "sc":
        trial = gen_sc_trial(rng, distractor_count)
    else:
        raise ValueError(f"unknown task_type: {task_type}")
    config["trials"] = {"trial_1": trial}
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
    p.add_argument("--out", type=Path, required=True, help="Output directory for generated config files.")
    p.add_argument("--n", type=int, default=100, help="Number of configs to generate.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--task-type", choices=["sfp", "sc", "mixed"], default="mixed",
                   help="Which task type(s) to generate. 'mixed' picks per-config uniformly.")
    p.add_argument("--distractor-min", type=int, default=0, help="Min distractor mounts per config.")
    p.add_argument("--distractor-max", type=int, default=4, help="Max distractor mounts per config.")
    args = p.parse_args()

    with args.template.open() as f:
        template = yaml.safe_load(f)

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for i in range(args.n):
        tt = rng.choice(["sfp", "sc"]) if args.task_type == "mixed" else args.task_type
        distractors = rng.randint(args.distractor_min, args.distractor_max)
        config = gen_config(template, rng, tt, distractors)
        h = config_hash(config)
        name = f"trial_{i:05d}_{tt}_{h}.yaml"
        with (args.out / name).open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    print(f"Wrote {args.n} configs to {args.out}")


if __name__ == "__main__":
    main()
