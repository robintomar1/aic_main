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

    Verified across all 6 oracle batches (2026-05-08, 258k frames):
      * batch_100_a..e: 100% pass (172,140 frames)
      * batch_500_a:    99.99% pass (86,096/86,102 — 6 are all-zero
                        TF-lookup failures, correctly rejected)
      * x ∈ [-0.42, -0.16], y ∈ [+0.00, +0.42], z ∈ [+0.01, +0.13]
        across ALL valid frames in all 6 batches.

    The quaternion is a ~180° rotation around y (qy ≈ -1, qw ≈ 0) — the
    port frame z-axis points INTO the board, opposite the robot's base
    z-axis. **Earlier guidance from memory `project_aic_act_dataset.md`
    that "qw≈0 = corrupt batch_500_a" was wrong — qw≈0 is the NORMAL
    state.** That false signal led to batch_500_a being unnecessarily
    excluded from the merged_clean dataset.

    The real corruption signature is `_lookup_pose` returning all-zeros
    when the TF lookup fails momentarily — `(0,0,0, 0,0,0,0)` — which
    has zero quaternion norm. The unit-norm check catches that without
    imposing unverified position bounds.

    Returns True iff:
      * all values finite
      * quaternion has approximately unit norm (the only reliable signal
        of a successful TF lookup)
    """
    if not np.all(np.isfinite(port_pose_7d)):
        return False
    qnorm = float(np.linalg.norm(port_pose_7d[3:7]))
    if not (0.95 < qnorm < 1.05):
        return False
    return True


STALE_LEADING_POS_THRESHOLD_M = 0.05


def patch_stale_leading_actions(
    states_47: np.ndarray,
    actions_7: np.ndarray,
    eps_col: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Apply Fix 1 from `clean_act_dataset.py` (memory: ~0.06% of frames).

    The recorder sometimes captures the previous trial's
    /aic_controller/pose_commands at the start of a new episode, before
    the new policy has published. Detection: action.position disagrees
    with state.tcp_pose.position by > 50 mm. Fix: overwrite leading bad
    frames with the first "good" frame's action (per episode).

    Pure function: returns (patched_actions, n_frames_patched, n_episodes_patched).
    Does NOT mutate the input array.

    Used by both `make_port_local_dataset.py` (builder) and
    `test_port_local_dataset.py` (Tier 2 round-trip target). Sharing this
    avoids the divergence trap where the test silently passes against a
    different patch logic than the builder applies.
    """
    fixed = actions_7.copy()
    n_frames_patched = 0
    n_episodes_patched = 0
    for ep in sorted(np.unique(eps_col).tolist()):
        ep_idx = np.where(eps_col == ep)[0]
        if len(ep_idx) == 0:
            continue
        s = int(ep_idx[0])
        e = int(ep_idx[-1]) + 1
        first_good = None
        for j in range(min(20, e - s)):
            a_pos = fixed[s + j, :3]
            st_pos = states_47[s + j, :3]
            if (np.linalg.norm(a_pos) > 1e-3
                    and np.linalg.norm(a_pos - st_pos)
                        <= STALE_LEADING_POS_THRESHOLD_M):
                first_good = j
                break
        if first_good is None or first_good == 0:
            continue
        replacement = fixed[s + first_good].copy()
        fixed[s:s + first_good] = replacement
        n_frames_patched += first_good
        n_episodes_patched += 1
    return fixed, n_frames_patched, n_episodes_patched


def is_action_valid(action_7d: np.ndarray) -> bool:
    """Sanity-check a recorded action (xyz + xyzw pose target).

    Verified on batch_100_a (2026-05-08): 3/37178 frames have all-zero
    actions, all at frame_index 0..2 of episode 0 (before the policy
    publishes its first /aic_controller/pose_commands message — the
    recorder writes whatever it has, which is the default-zero Pose).
    Other batches likely have the same pattern. Drop these frames.

    Returns True iff:
      * all values finite
      * quaternion has approximately unit norm
    """
    if not np.all(np.isfinite(action_7d)):
        return False
    qnorm = float(np.linalg.norm(action_7d[3:7]))
    if not (0.95 < qnorm < 1.05):
        return False
    return True
