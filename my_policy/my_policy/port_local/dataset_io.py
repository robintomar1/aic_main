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
    """Sanity-check a recorded `groundtruth.port_pose` (xyz + xyzw, in
    base_link).

    Verified on batch_100_a (2026-05-08): real port poses sit at roughly
    x ∈ [-0.36, -0.23], y ∈ [0.05, 0.34], z ∈ [0.01, 0.13] in base_link.
    The quaternion is a ~180° rotation around y (qy ≈ -1, qw ≈ 0) — the
    port frame z-axis points INTO the board, opposite the robot's base
    z-axis. Earlier guidance that "qw≈0 = corrupt" was wrong; qw≈0 is
    normal here.

    The actual corruption signature seen in batch_500_a (per memory
    `project_aic_act_dataset.md`) is `_lookup_pose` returning all-zeros
    when the TF lookup fails — `(0,0,0, 0,0,0,0)` — which has zero
    quaternion norm. The unit-norm check below catches that without
    imposing unverified position bounds.

    Returns True iff:
      * all values finite
      * quaternion has approximately unit norm (the only signal of a
        successful TF lookup; failed lookups produce all-zero records)
    """
    if not np.all(np.isfinite(port_pose_7d)):
        return False
    qnorm = float(np.linalg.norm(port_pose_7d[3:7]))
    if not (0.95 < qnorm < 1.05):
        return False
    return True
