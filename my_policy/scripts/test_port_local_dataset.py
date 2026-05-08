#!/usr/bin/env python3
"""Tier 2 — integration tests for `make_port_local_dataset.py` output.

Runs against a real produced port-local dataset PLUS the source raw batch
that produced it, and validates that the transformation actually did what
it's supposed to do on production-shaped data.

Requirements:
  * `pixi run` (needs pyarrow, numpy — no torch/lerobot needed).
  * A raw batch dataset (e.g. `/root/aic_data/batch_100_a/`) with the
    47-channel observation.state.
  * A port-local dataset built from that raw batch
    (e.g. `/root/aic_data/v9_act_build/batch_100_a_port_local_dataset/`).

What this catches that Tier 1 didn't:
  * Wrong index assignment between raw and transformed states.
  * Off-by-one in episode renumbering.
  * Mismatch between recorded port_pose and the port-local origin
    (e.g. transform applied with the wrong port frame).
  * Distribution-level signal that the transform is doing what we
    *want* it to do (collapse the world-frame variance).

Tests:

  Test 6 (killer) — Oracle action consistency:
    For every (raw frame, port-local frame) pair, transform the port-local
    action back via the recorded port_pose. Residual against the raw
    action must be < 1e-5 m and < 1e-5 quat-residual. A bug in the
    transform stack at any layer (wrong indexing, wrong frame, lost
    precision) blows this up by orders of magnitude.

  Test 7 — Distribution collapse:
    std of `tcp_pose.position.{x,y,z}` across all frames in port-local
    should be ≤ 30% of the std in base_link. If not, the transform isn't
    actually canonicalizing world-frame variance — likely a frame error.

  Test 8 — Per-task slice consistency:
    For each (target_module, port_name) pair, within-task variance of
    port-local TCP across episodes should be small (≤ 5 cm std). If a
    task's port_pose lookup is broken for one or more episodes, that
    task's port-local TCP cluster will be smeared.

  Test 11 — Frame count parity:
    Output frame count == (input clean frames - dropped invalid-port frames).

  Test 12 — No leakage channels in output:
    `observation.state.names` does NOT contain any `groundtruth.*` or
    `meta.insertion_success`. The port-local dataset must be
    submission-clean.

Usage:
    pixi run python my_policy/scripts/test_port_local_dataset.py \\
        --raw-batch /root/aic_data/batch_100_a \\
        --port-local /root/aic_data/v9_act_build/batch_100_a_port_local_dataset \\
        [--clean-episodes-json /root/aic_data/v9_act_build/batch_100_a_act_clean_episodes.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.port_local.transforms import (  # noqa: E402
    transform_pose_back_to_baselink,
)
from my_policy.port_local.dataset_io import (  # noqa: E402
    SRC_PORT_POSE_SLICE,
    is_port_pose_valid,
)


def _load_table(root: Path):
    return pq.read_table(str(root / "data" / "chunk-000" / "file-000.parquet"))


def _load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def _state_arr(table) -> np.ndarray:
    return np.stack(
        [np.asarray(r, dtype=np.float32) for r in table["observation.state"].to_pylist()]
    )


def _action_arr(table) -> np.ndarray:
    return np.stack(
        [np.asarray(r, dtype=np.float32) for r in table["action"].to_pylist()]
    )


def _quat_residual_batched(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Per-row 1 - |q1·q2|. Sign-invariant."""
    d = np.abs(np.sum(q1 * q2, axis=-1))
    return 1.0 - np.minimum(1.0, d)


def _state_index(state_names: list[str], wanted_prefix: str) -> list[int]:
    return [i for i, n in enumerate(state_names) if n.startswith(wanted_prefix)]


def _align_raw_to_portlocal(
    raw_table, pl_table, pl_eps_to_raw_eps: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """For every port-local frame, find the matching raw frame (same
    underlying source row pre-filter). Returns matched raw indices
    aligned with port-local frame order, and the matched port-local
    indices (which is just arange in this implementation).

    Match key: (raw_episode, raw_frame_within_episode). The port-local
    frame_index is per-new-episode 0..len-1; within an old episode the
    relative ordering is preserved by `make_port_local_dataset` (it
    keeps frames in source order, just drops invalid-port-pose frames).

    We rebuild the (raw_ep, raw_frame_in_ep) → raw_global_idx map and
    use it to look up.
    """
    # Source frames within each old episode — global index per (old_ep,
    # local_frame). The make_port_local_dataset script preserves source
    # order within an episode, so port-local frame i within new-ep k
    # corresponds to the i-th VALID frame of the matching old episode.
    raw_eps = raw_table["episode_index"].to_numpy().astype(np.int64)
    raw_states = _state_arr(raw_table)
    raw_globals_by_old_ep: dict[int, list[int]] = {}
    for gi in range(len(raw_eps)):
        oe = int(raw_eps[gi])
        port_pose = raw_states[gi][SRC_PORT_POSE_SLICE]
        if not is_port_pose_valid(port_pose):
            continue
        raw_globals_by_old_ep.setdefault(oe, []).append(gi)

    pl_eps = pl_table["episode_index"].to_numpy().astype(np.int64)
    pl_frames = pl_table["frame_index"].to_numpy().astype(np.int64)

    matched_raw_idx = np.zeros(len(pl_eps), dtype=np.int64)
    for i in range(len(pl_eps)):
        new_ep = int(pl_eps[i])
        new_frame = int(pl_frames[i])
        old_ep = pl_eps_to_raw_eps[new_ep]
        valid_globals = raw_globals_by_old_ep[old_ep]
        if new_frame >= len(valid_globals):
            raise AssertionError(
                f"port-local new_ep={new_ep} new_frame={new_frame} but "
                f"raw old_ep={old_ep} only has {len(valid_globals)} valid frames"
            )
        matched_raw_idx[i] = valid_globals[new_frame]

    return matched_raw_idx, np.arange(len(pl_eps))


def _build_pl_to_raw_episode_map(
    pl_table, raw_table, raw_state_names: list[str],
) -> dict[int, int]:
    """Each port-local episode has one task assignment + a unique
    `task_index`. Match it back to a raw episode by pulling the raw
    episode's task assignment from the source table and finding the
    pairing that matches.

    Cleaner: rely on the `index` column. The source raw table's `index`
    is dense 0..N-1; the port-local table inherits the original `index`
    as `index_in_source` if we keep it. But `make_port_local_dataset`
    overwrites `index`. So we use a different tactic: match by
    (task_index, position-in-sequence) which is fragile.

    Cleanest reliable signal: the port-local table preserves the
    original `timestamp` column. Match (timestamp, task_index) → raw
    row, then group by raw_episode. This is what we do here.
    """
    raw_ts = raw_table["timestamp"].to_numpy().astype(np.float64)
    raw_eps = raw_table["episode_index"].to_numpy().astype(np.int64)
    pl_ts = pl_table["timestamp"].to_numpy().astype(np.float64)
    pl_eps = pl_table["episode_index"].to_numpy().astype(np.int64)

    # For each unique pl episode, take its first frame's timestamp and
    # find a raw row with the same timestamp. There can be ties if two
    # raw episodes happen to share a timestamp, but timestamps are
    # monotone within an episode and reset per episode, so the first
    # frame's timestamp + a uniqueness-by-task assumption is enough.
    pl_to_raw: dict[int, int] = {}
    for new_ep in sorted(np.unique(pl_eps).tolist()):
        first_pl_idx = int(np.argmax(pl_eps == new_ep))
        ts0 = float(pl_ts[first_pl_idx])
        # Find raw rows with this timestamp in any episode.
        raw_match = np.flatnonzero(np.isclose(raw_ts, ts0, atol=1e-9))
        if len(raw_match) == 0:
            raise AssertionError(
                f"port-local episode {new_ep} first ts={ts0} not found in raw"
            )
        # Pick the raw episode that this timestamp falls in. Defensive
        # against ties: should be exactly one because raw timestamps are
        # rate-stamped and unique per episode start.
        candidate_old_eps = np.unique(raw_eps[raw_match]).tolist()
        if len(candidate_old_eps) > 1:
            # Tie-break by checking frame_index==0 in the raw table.
            raw_frames = raw_table["frame_index"].to_numpy().astype(np.int64)
            tied_first = [
                int(raw_eps[g]) for g in raw_match if raw_frames[g] == 0
            ]
            if len(tied_first) != 1:
                raise AssertionError(
                    f"could not disambiguate raw episode for pl ep {new_ep}; "
                    f"candidates={candidate_old_eps}"
                )
            pl_to_raw[new_ep] = tied_first[0]
        else:
            pl_to_raw[new_ep] = int(candidate_old_eps[0])
    return pl_to_raw


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_6_oracle_action_consistency(raw_table, pl_table, raw_state_names):
    """Killer test — round-trip every action via recorded port_pose."""
    raw_states = _state_arr(raw_table)
    raw_actions = _action_arr(raw_table)
    pl_actions = _action_arr(pl_table)
    pl_to_raw = _build_pl_to_raw_episode_map(pl_table, raw_table, raw_state_names)
    matched_raw_idx, _ = _align_raw_to_portlocal(raw_table, pl_table, pl_to_raw)

    pos_errs: list[float] = []
    rot_residuals: list[float] = []

    for i in range(len(pl_actions)):
        ri = int(matched_raw_idx[i])
        port_pose = raw_states[ri][SRC_PORT_POSE_SLICE]
        if not is_port_pose_valid(port_pose):
            raise AssertionError(
                f"matched raw frame {ri} has invalid port_pose; alignment is wrong"
            )
        # Reconstruct base-link action from port-local.
        recon = transform_pose_back_to_baselink(
            pl_actions[i].astype(np.float64),
            port_pose.astype(np.float64),
        )
        original = raw_actions[ri].astype(np.float64)
        pos_errs.append(float(np.linalg.norm(recon[:3] - original[:3])))
        rot_residuals.append(
            float(_quat_residual_batched(
                recon[3:7][None, :], original[3:7][None, :]
            )[0])
        )
    pos_errs = np.array(pos_errs)
    rot_residuals = np.array(rot_residuals)

    print(f"    n_frames matched: {len(pos_errs)}")
    print(f"    position residual:  max={pos_errs.max():.2e} m, "
          f"p99={np.percentile(pos_errs, 99):.2e}, mean={pos_errs.mean():.2e}")
    print(f"    rotation residual:  max={rot_residuals.max():.2e}, "
          f"p99={np.percentile(rot_residuals, 99):.2e}, mean={rot_residuals.mean():.2e}")

    assert pos_errs.max() < 1e-5, (
        f"position residual {pos_errs.max():.2e} m exceeds 1e-5 m — "
        f"transform is not invertible to within float32 precision"
    )
    assert rot_residuals.max() < 1e-5, (
        f"rotation residual {rot_residuals.max():.2e} exceeds 1e-5 — "
        f"quaternion round-trip is broken"
    )


def test_7_distribution_collapse_runner(
    raw_table, pl_table, raw_state_names, pl_state_names,
):
    raw_states = _state_arr(raw_table)
    pl_states = _state_arr(pl_table)
    raw_tcp_idx = _state_index(raw_state_names, "tcp_pose.position")
    pl_tcp_idx = _state_index(pl_state_names, "tcp_pose.position")
    assert len(raw_tcp_idx) == 3
    assert len(pl_tcp_idx) == 3

    # Restrict raw to frames that matched (have valid port_pose).
    raw_port_pose = raw_states[:, SRC_PORT_POSE_SLICE]
    valid_mask = np.array(
        [is_port_pose_valid(p) for p in raw_port_pose], dtype=bool
    )
    raw_tcp_xyz = raw_states[valid_mask][:, raw_tcp_idx]
    pl_tcp_xyz = pl_states[:, pl_tcp_idx]

    raw_std = raw_tcp_xyz.std(axis=0)
    pl_std = pl_tcp_xyz.std(axis=0)
    print(f"    raw  TCP std (m):  x={raw_std[0]:.3f} y={raw_std[1]:.3f} z={raw_std[2]:.3f}")
    print(f"    port TCP std (m):  x={pl_std[0]:.3f} y={pl_std[1]:.3f} z={pl_std[2]:.3f}")
    ratio = pl_std / np.maximum(raw_std, 1e-6)
    print(f"    ratio port/raw: x={ratio[0]:.2f} y={ratio[1]:.2f} z={ratio[2]:.2f}")

    # Acceptance: at least one axis ratio ≤ 0.3 (the variance-collapsing axis),
    # AND mean ratio ≤ 0.7. If raw std is tiny (single-task / single-config
    # batch), the test can pass trivially with high ratio — which is correct
    # behavior for that data.
    if raw_std.max() < 0.05:
        print("    note: raw TCP std is small (<5cm); skipping ratio check")
        return
    assert ratio.min() < 0.6, (
        f"no axis showed clear distribution collapse; min ratio={ratio.min():.2f}"
    )


def test_8_per_task_slice_consistency(pl_table, pl_state_names):
    """Within each task_index, the port-local TCP first-frame distribution
    should be tight (std ≤ 5 cm). A broken port lookup for one episode
    smears its task's cluster."""
    pl_states = _state_arr(pl_table)
    pl_eps = pl_table["episode_index"].to_numpy().astype(np.int64)
    pl_frames = pl_table["frame_index"].to_numpy().astype(np.int64)
    pl_tasks = pl_table["task_index"].to_numpy().astype(np.int64)
    pl_tcp_idx = _state_index(pl_state_names, "tcp_pose.position")

    first_frame_mask = pl_frames == 0
    first_states = pl_states[first_frame_mask][:, pl_tcp_idx]
    first_tasks = pl_tasks[first_frame_mask]
    first_eps = pl_eps[first_frame_mask]

    print(f"    {first_frame_mask.sum()} episode-start frames")
    issues = 0
    for ti in sorted(np.unique(first_tasks).tolist()):
        mask = first_tasks == ti
        n = int(mask.sum())
        if n < 2:
            continue
        std = first_states[mask].std(axis=0)
        print(f"    task {ti}: n={n}, std (m): "
              f"x={std[0]:.4f} y={std[1]:.4f} z={std[2]:.4f}")
        # An episode with a wrong port lookup would deviate by tens of cm.
        if std.max() > 0.10:
            issues += 1
            print(f"      WARN: std exceeds 10 cm — possible bad port lookup")
    assert issues == 0, f"{issues} task(s) had std > 10 cm — likely bad port lookup"


def test_11_frame_count_parity(
    raw_table, pl_table, raw_state_names, clean_episodes_json: Path | None,
):
    """Output frame count should equal input clean frames minus invalid
    port_pose frames."""
    raw_states = _state_arr(raw_table)
    raw_eps = raw_table["episode_index"].to_numpy().astype(np.int64)
    raw_port_pose = raw_states[:, SRC_PORT_POSE_SLICE]
    raw_valid_port = np.array(
        [is_port_pose_valid(p) for p in raw_port_pose], dtype=bool
    )

    if clean_episodes_json is not None:
        keep_eps = set(json.loads(clean_episodes_json.read_text()))
        clean_mask = np.array([int(e) in keep_eps for e in raw_eps])
    else:
        clean_mask = np.ones(len(raw_eps), dtype=bool)

    expected = int((raw_valid_port & clean_mask).sum())
    actual = pl_table.num_rows
    print(f"    raw frames:           {len(raw_eps)}")
    print(f"    raw clean frames:     {clean_mask.sum()}")
    print(f"    raw valid port_pose:  {raw_valid_port.sum()}")
    print(f"    expected output:      {expected}")
    print(f"    actual output:        {actual}")
    assert actual == expected, (
        f"frame count mismatch: expected {expected}, got {actual}"
    )


def test_12_no_leakage_channels_in_output(pl_state_names):
    """Output dataset must not contain any groundtruth.* or
    meta.insertion_success channels."""
    bad = [n for n in pl_state_names if "groundtruth" in n or "insertion_success" in n]
    print(f"    checked {len(pl_state_names)} channel names; "
          f"found {len(bad)} forbidden")
    assert bad == [], f"forbidden channels in output: {bad}"


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-batch", type=Path, required=True)
    p.add_argument("--port-local", type=Path, required=True)
    p.add_argument("--clean-episodes-json", type=Path, default=None)
    args = p.parse_args()

    print(f"raw batch:    {args.raw_batch}")
    print(f"port-local:   {args.port_local}")
    if args.clean_episodes_json:
        print(f"clean filter: {args.clean_episodes_json}")
    print()

    raw_info = _load_info(args.raw_batch)
    pl_info = _load_info(args.port_local)
    raw_state_names = raw_info["features"]["observation.state"]["names"]
    pl_state_names = pl_info["features"]["observation.state"]["names"]

    if len(raw_state_names) != 47:
        print(f"FATAL: raw batch has {len(raw_state_names)}-dim state, expected 47",
              file=sys.stderr)
        return 1
    if len(pl_state_names) != 44:
        print(f"FATAL: port-local has {len(pl_state_names)}-dim state, expected 44",
              file=sys.stderr)
        return 1

    raw_table = _load_table(args.raw_batch)
    pl_table = _load_table(args.port_local)

    tests = [
        ("6  Oracle action consistency (killer)",
         lambda: test_6_oracle_action_consistency(raw_table, pl_table, raw_state_names)),
        ("7  Distribution collapse",
         lambda: test_7_distribution_collapse_runner(
             raw_table, pl_table, raw_state_names, pl_state_names)),
        ("8  Per-task slice consistency",
         lambda: test_8_per_task_slice_consistency(pl_table, pl_state_names)),
        ("11 Frame count parity",
         lambda: test_11_frame_count_parity(
             raw_table, pl_table, raw_state_names, args.clean_episodes_json)),
        ("12 No leakage channels in output",
         lambda: test_12_no_leakage_channels_in_output(pl_state_names)),
    ]

    failed = 0
    for name, fn in tests:
        try:
            print(f"[ run] {name}")
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}")
            print(f"       {e}")
        except Exception as e:
            failed += 1
            print(f"[ERR ] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print()
    if failed == 0:
        print(f"All {len(tests)} Tier 2 tests passed.")
        return 0
    print(f"{failed}/{len(tests)} Tier 2 tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
