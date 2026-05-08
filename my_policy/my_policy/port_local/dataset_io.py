"""Port-local dataset I/O helpers — shared by `make_port_local_dataset.py`
(builder) and `test_port_local_dataset.py` (validator).

Lives in the `port_local` package so it can be imported by both. Source-state
indexing constants and the port-pose validity check are the contract between
the two scripts; if either drifts, the validator's alignment falls apart.

Verified 2026-05-08 against `/root/aic_data/batch_100_a/meta/info.json`:
the raw recorder schema is exactly the layout reflected in the slice
constants below.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Raw recorder state layout (47 channels). Verified from
# batch_100_a/meta/info.json — see test_port_local_transforms test 0
# and the verification output of the user's plan-mode review.
# ---------------------------------------------------------------------------

SRC_TCP_POSE_SLICE = slice(0, 7)       # xyz + xyzw
SRC_TCP_VEL_SLICE = slice(7, 13)       # linear + angular
SRC_TCP_ERR_SLICE = slice(13, 19)
SRC_JOINT_POS_SLICE = slice(19, 26)
SRC_WRENCH_SLICE = slice(26, 32)       # force + torque
SRC_PORT_POSE_SLICE = slice(32, 39)    # xyz + xyzw
SRC_PLUG_POSE_SLICE = slice(39, 46)
SRC_INSERTION_SUCCESS_INDEX = 46

EXPECTED_RAW_STATE_DIM = 47
EXPECTED_PORT_LOCAL_STATE_DIM = 44


def is_port_pose_valid(port_pose_7d: np.ndarray) -> bool:
    """Sanity-check a recorded `groundtruth.port_pose` (xyz + xyzw).

    Memory `project_aic_act_dataset.md` notes batch_500_a recorded corrupt
    port poses where x≤0 and qw≈0. Valid recordings put the port in front
    of the robot at ~(0.1..0.3, ±0.4, 1.0..1.2) in base_link with a unit
    quaternion.

    Returns True iff:
      * all values finite
      * quaternion within 5% of unit norm
      * x in (-0.1, 0.5)   — port is forward of robot base
      * z in (0.9, 1.4)    — port is at table height

    The bounds are deliberately wide so we drop only obvious garbage,
    not legitimate edge configurations.
    """
    if not np.all(np.isfinite(port_pose_7d)):
        return False
    qnorm = float(np.linalg.norm(port_pose_7d[3:7]))
    if not (0.95 < qnorm < 1.05):
        return False
    x, _, z = port_pose_7d[0], port_pose_7d[1], port_pose_7d[2]
    if not (-0.1 < x < 0.5):
        return False
    if not (0.9 < z < 1.4):
        return False
    return True
