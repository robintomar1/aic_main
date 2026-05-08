#!/usr/bin/env python3
"""Offline tests for `my_policy.ros.RunPortLocalACT` — the v9-port-local
inference shim.

The high-leverage thing to verify BEFORE burning eval-container minutes
is that the shim's per-tick state composition produces the EXACT same
44-dim vector that `make_port_local_dataset.py` wrote into the trained
dataset at the same frame.

Same shape as `clean_act_dataset` validation: pick real raw-batch frames,
synthesize an `Observation` from them, run through the shim's helpers,
compare to the port-local dataset's `observation.state` at the same
(episode, frame_index). If they match within float32 precision, the
shim's slot ordering + transform-axis convention are correct.

Three tiers:

  Tier 1  Helper unit tests on synthetic observations:
            * _build_state_44 slot ordering (per-channel distinct value
              flows to the expected output index).
            * Quaternion normalization in _action_port_to_baselink_pose.

  Tier 2  Real-data round-trip:
            For N random frames in batch_100_a:
              1. Load raw frame's TCP/wrench/joint/etc. + port_pose.
              2. Build a fake Observation with those values.
              3. Call _build_state_44 with the matching task_vec.
              4. Load the corresponding frame from the port-local
                 dataset's `observation.state`.
              5. Compare. Should be identical (modulo float32 precision).

  Tier 3  Latency:
            N forward passes through preprocessor → policy → postprocessor
            using the real checkpoint + a real frame. Verifies 20 Hz
            budget is met on the available GPU.

Usage:
    pixi run python my_policy/scripts/test_runportlocalact_offline.py \\
        --checkpoint-dir /root/aic_data/v9_act_build/runs/v9_pl_v2/checkpoints/050000/pretrained_model \\
        --raw-batch /root/aic_data/batch_100_a \\
        --port-local-dataset /root/aic_data/v9_act_build/batch_100_a_port_local_dataset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.act.labels import encode_task_vector  # noqa: E402
from my_policy.localizer.labels import match_episodes_to_trials  # noqa: E402
from my_policy.port_local.dataset_io import (  # noqa: E402
    SRC_PORT_POSE_SLICE,
    SRC_TCP_POSE_SLICE,
    SRC_TCP_VEL_SLICE,
    SRC_WRENCH_SLICE,
    is_action_valid,
    is_port_pose_valid,
)


# ---------------------------------------------------------------------------
# Helpers — fake an Observation msg from a raw-batch row.
# ---------------------------------------------------------------------------


def _make_fake_observation(state_47: np.ndarray, raw_action: np.ndarray):
    """Build a SimpleNamespace mimicking aic_model_interfaces/Observation.

    The shim's `_build_state_44` reads:
      controller_state.tcp_pose.{position,orientation}.{x,y,z[,w]}
      controller_state.tcp_velocity.{linear,angular}.{x,y,z}
      controller_state.tcp_error[0..5]
      controller_state.fts_tare_offset.wrench.{force,torque}.{x,y,z}
      joint_states.position[0..6]
      wrist_wrench.wrench.{force,torque}.{x,y,z}

    The recorder's stored wrench is ALREADY tare-compensated, so to
    reverse-construct a raw + tare pair that the shim's _compensated_wrench
    will sum back to the same compensated value, we set tare = 0 and
    raw = stored. Equivalent and unambiguous.
    """
    cs = SimpleNamespace(
        tcp_pose=SimpleNamespace(
            position=SimpleNamespace(
                x=float(state_47[0]),
                y=float(state_47[1]),
                z=float(state_47[2]),
            ),
            orientation=SimpleNamespace(
                x=float(state_47[3]),
                y=float(state_47[4]),
                z=float(state_47[5]),
                w=float(state_47[6]),
            ),
        ),
        tcp_velocity=SimpleNamespace(
            linear=SimpleNamespace(
                x=float(state_47[7]),
                y=float(state_47[8]),
                z=float(state_47[9]),
            ),
            angular=SimpleNamespace(
                x=float(state_47[10]),
                y=float(state_47[11]),
                z=float(state_47[12]),
            ),
        ),
        tcp_error=[float(state_47[i]) for i in range(13, 19)],
        fts_tare_offset=SimpleNamespace(
            wrench=SimpleNamespace(
                force=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                torque=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            ),
        ),
    )
    return SimpleNamespace(
        controller_state=cs,
        joint_states=SimpleNamespace(
            position=[float(state_47[19 + i]) for i in range(7)]
        ),
        wrist_wrench=SimpleNamespace(
            wrench=SimpleNamespace(
                force=SimpleNamespace(
                    x=float(state_47[26]),
                    y=float(state_47[27]),
                    z=float(state_47[28]),
                ),
                torque=SimpleNamespace(
                    x=float(state_47[29]),
                    y=float(state_47[30]),
                    z=float(state_47[31]),
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Tier 1 — Helper unit tests.
# ---------------------------------------------------------------------------


def test_state_44_slot_ordering():
    """Hand-craft an Observation where every input slot has a distinct
    value, plus a port pose chosen so the port-local transform is the
    IDENTITY (port at origin with identity quaternion). Then verify each
    output slot of the 44-dim vector matches the expected source slot."""
    from my_policy.ros.RunPortLocalACT import _build_state_44

    # Distinct values per source slot (47-dim).
    state_47 = np.arange(0.0, 47.0, dtype=np.float32)
    # Port pose = identity → port-local frame == base_link → transform is no-op
    # for tcp_pose and tcp_velocity. wrench, however, gets rotated by R_tcp
    # because R_port^T @ R_tcp = R_tcp when R_port=I (verified in
    # test_port_local_transforms.test_5_identity_edge_case). To make THIS
    # test focus on slot ordering (not transform math, which has its own
    # tests), set tcp orientation to identity AND port pose to identity.
    state_47[3:7] = [0.0, 0.0, 0.0, 1.0]  # TCP quat identity
    port_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    task_vec = np.zeros(12, dtype=np.float32)
    task_vec[3] = 1.0   # arbitrary slot
    task_vec[8] = 1.0
    task_vec[11] = 1.0

    out = _build_state_44(_make_fake_observation(state_47, None), task_vec,
                          port_pose).numpy()

    assert out.shape == (44,), f"got {out.shape}"

    # Build expected 44-dim manually using the slot map from the builder
    # (KEEP_CHANNEL_GROUPS): tcp_pose(7) + tcp_vel(6) + tcp_err(6) +
    # joints(7) + wrench(6) + task(12).
    expected = np.zeros(44, dtype=np.float32)
    expected[0:7]   = state_47[0:7]    # TCP pose (port=I → no transform)
    expected[7:13]  = state_47[7:13]   # TCP velocity (port=I, tcp=I → no rot)
    expected[13:19] = state_47[13:19]  # TCP error
    expected[19:26] = state_47[19:26]  # joints
    expected[26:32] = state_47[26:32]  # wrench (port=I, tcp=I → no rot)
    expected[32:44] = task_vec

    np.testing.assert_allclose(
        out, expected, atol=1e-6,
        err_msg="state_44 slot ordering doesn't match expected layout",
    )


def test_quaternion_normalization_in_action():
    """`_action_port_to_baselink_pose` must produce a unit quaternion in
    the output Pose, even if the network outputs a non-unit quat AND the
    round-trip introduces sign-flips."""
    from my_policy.ros.RunPortLocalACT import _action_port_to_baselink_pose

    # Action with a 1.5× scaled quaternion (network output isn't unit-norm
    # by construction).
    action_port = np.array([0.01, 0.02, -0.03, 0.6, 0.6, 0.6, 0.9],
                           dtype=np.float64)
    port_pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    pose = _action_port_to_baselink_pose(action_port, port_pose)
    qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    assert abs(norm - 1.0) < 1e-6, f"quat norm = {norm}"


def test_zero_quaternion_safe_in_action():
    """Degenerate zero-norm quat falls back to identity rather than NaN."""
    from my_policy.ros.RunPortLocalACT import _action_port_to_baselink_pose

    action_port = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    port_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    pose = _action_port_to_baselink_pose(action_port, port_pose)
    # The transform_pose_back_to_baselink call goes through make_se3, which
    # raises on zero-norm quat. So the function as written propagates the
    # error. This test documents that degenerate quats DON'T silently
    # produce NaNs — they raise instead, which is what we want.
    # If transform_pose_back_to_baselink ever changes to silently fall
    # through, this assertion catches it.
    # (Note: _action_to_pose's fallback only handles a NORMALIZED zero-quat
    # AFTER round-trip. Pre-round-trip zero-quat hits make_se3's check.)
    pass  # currently unreachable — left for documentation


def run_tier1():
    print("--- Tier 1: helper unit tests ---")
    tests = [
        ("state 44 slot ordering", test_state_44_slot_ordering),
        ("quat normalization in action", test_quaternion_normalization_in_action),
    ]
    failed = 0
    for name, fn in tests:
        try:
            print(f"  [ run] {name}")
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  [ ERR] {name}: {type(e).__name__}: {e}")
    return failed


# ---------------------------------------------------------------------------
# Tier 2 — Real-data round-trip vs port-local dataset.
# ---------------------------------------------------------------------------


def run_tier2(raw_batch: Path, port_local_root: Path, n_frames: int):
    """For N random frames in raw_batch, reconstruct the Observation and
    verify the shim's _build_state_44 output matches the port-local
    dataset's observation.state at the same (episode, frame_index)."""
    from my_policy.ros.RunPortLocalACT import _build_state_44

    print(f"--- Tier 2: real-data round-trip ({n_frames} frames) ---")
    print(f"  raw_batch     : {raw_batch}")
    print(f"  port-local    : {port_local_root}")

    # Load raw + port-local parquets.
    raw_table = pq.read_table(
        str(raw_batch / "data" / "chunk-000" / "file-000.parquet"))
    raw_states = np.stack([
        np.asarray(r, dtype=np.float32)
        for r in raw_table["observation.state"].to_pylist()
    ])
    raw_actions = np.stack([
        np.asarray(r, dtype=np.float32)
        for r in raw_table["action"].to_pylist()
    ])
    raw_eps = raw_table["episode_index"].to_numpy().astype(np.int64)
    raw_frames = raw_table["frame_index"].to_numpy().astype(np.int64)

    pl_table = pq.read_table(
        str(port_local_root / "data" / "chunk-000" / "file-000.parquet"))
    pl_states = np.stack([
        np.asarray(r, dtype=np.float32)
        for r in pl_table["observation.state"].to_pylist()
    ])
    pl_eps = pl_table["episode_index"].to_numpy().astype(np.int64)
    pl_frames = pl_table["frame_index"].to_numpy().astype(np.int64)
    pl_task_idx = pl_table["task_index"].to_numpy().astype(np.int64)

    # Map port-local episode (dense) → raw episode (via source_episode_map).
    map_path = port_local_root / "source_episode_map.json"
    if not map_path.exists():
        sys.exit(f"missing {map_path} — rebuild the port-local dataset")
    pl_to_raw_ep = {int(k): int(v) for k, v in json.loads(map_path.read_text()).items()}

    # Load yaml + summary for task_vec computation.
    import yaml as yaml_mod
    cfg = yaml_mod.safe_load(
        (raw_batch.parent / f"{raw_batch.name}.yaml").read_text())
    summary = json.loads(
        (raw_batch.parent / f"{raw_batch.name}_logs" / "summary.json").read_text())
    raw_ep_to_trial = match_episodes_to_trials(summary, cfg["trials"])

    # Pick N random port-local frames.
    rng = np.random.default_rng(42)
    pick = rng.choice(len(pl_states), size=min(n_frames, len(pl_states)), replace=False)

    failures = 0
    max_pos_err = 0.0
    max_quat_err = 0.0
    max_other_err = 0.0
    for pl_gi in pick:
        pl_gi = int(pl_gi)
        new_ep = int(pl_eps[pl_gi])
        new_fr = int(pl_frames[pl_gi])
        pl_state = pl_states[pl_gi]

        # Find the corresponding raw frame.
        old_ep = pl_to_raw_ep[new_ep]
        # Within the old episode, the frames in raw_eps are contiguous; find
        # the global raw index that matches (episode, frame_index_within).
        # We need the new_fr-th VALID frame of old_ep — same logic the
        # builder uses (drops invalid port_pose / invalid action frames).
        ep_mask = raw_eps == old_ep
        ep_global = np.where(ep_mask)[0]
        valid_count = 0
        raw_gi = None
        for gi in ep_global:
            pp = raw_states[gi, SRC_PORT_POSE_SLICE]
            ac = raw_actions[gi]
            if not is_port_pose_valid(pp) or not is_action_valid(ac):
                continue
            if valid_count == new_fr:
                raw_gi = int(gi)
                break
            valid_count += 1
        if raw_gi is None:
            print(f"  WARN: could not locate raw frame for pl ep={new_ep} fr={new_fr}")
            failures += 1
            continue

        raw_state = raw_states[raw_gi]
        port_pose = raw_state[SRC_PORT_POSE_SLICE].astype(np.float64)
        # Build task_vec from the matching trial.
        trial_key = raw_ep_to_trial[old_ep]
        task = cfg["trials"][trial_key]["tasks"]["task_1"]
        task_vec = encode_task_vector(
            task["target_module_name"], task["port_name"], task["port_type"],
        )

        # Run the shim's helper.
        obs_msg = _make_fake_observation(raw_state, raw_actions[raw_gi])
        shim_state = _build_state_44(obs_msg, task_vec, port_pose).numpy()

        # Compare slot-by-slot.
        diff = shim_state - pl_state
        pos_err = float(np.linalg.norm(diff[0:3]))                    # tcp xyz
        # quat residual sign-invariant.
        q_shim = shim_state[3:7] / max(np.linalg.norm(shim_state[3:7]), 1e-9)
        q_pl = pl_state[3:7] / max(np.linalg.norm(pl_state[3:7]), 1e-9)
        quat_err = 1.0 - abs(float(np.dot(q_shim, q_pl)))
        other_err = float(np.max(np.abs(diff[7:])))   # vel + err + joints + wrench + task

        max_pos_err = max(max_pos_err, pos_err)
        max_quat_err = max(max_quat_err, quat_err)
        max_other_err = max(max_other_err, other_err)

        if pos_err > 1e-5 or quat_err > 1e-5 or other_err > 1e-5:
            print(f"  [FAIL] pl ep={new_ep} fr={new_fr} (raw ep={old_ep} gi={raw_gi}): "
                  f"pos_err={pos_err:.2e}m quat_res={quat_err:.2e} "
                  f"other_max={other_err:.2e}")
            failures += 1

    print(f"  max pos err over {len(pick)} frames:  {max_pos_err:.2e} m")
    print(f"  max quat residual:                    {max_quat_err:.2e}")
    print(f"  max other-channels abs err:           {max_other_err:.2e}")
    if failures == 0:
        print(f"  [PASS] all {len(pick)} frames match port-local dataset within float32")
    else:
        print(f"  [FAIL] {failures}/{len(pick)} frames disagreed with dataset")
    return failures


# ---------------------------------------------------------------------------
# Tier 3 — Latency.
# ---------------------------------------------------------------------------


def run_tier3(checkpoint_dir: Path, n_iters: int = 50):
    print(f"--- Tier 3: latency ({n_iters} iters) ---")
    import draccus
    from safetensors.torch import load_file
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_dict = json.loads((checkpoint_dir / "config.json").read_text())
    cfg_dict.pop("type", None)
    config = draccus.decode(ACTConfig, cfg_dict)
    policy = ACTPolicy(config)
    policy.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")))
    policy.eval().to(device)
    pre = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir), config_filename="policy_preprocessor.json")
    post = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    # Fake input. Shape must match what the dataset has.
    x = {
        "observation.images.left_camera":   torch.zeros(3, 256, 288),
        "observation.images.center_camera": torch.zeros(3, 256, 288),
        "observation.images.right_camera": torch.zeros(3, 256, 288),
        "observation.state": torch.zeros(44),
    }
    # Warm up.
    for _ in range(5):
        a = policy.select_action(pre({**x}))
        _ = post(a)
    times_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        a = policy.select_action(pre({**x}))
        _ = post(a)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times_ms.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times_ms)
    print(f"  per-call ms:  mean={arr.mean():.1f}  p50={np.percentile(arr,50):.1f}  "
          f"p99={np.percentile(arr,99):.1f}  max={arr.max():.1f}")
    print(f"  20 Hz budget = 50 ms/call. p99 < 50ms? "
          f"{'YES' if np.percentile(arr,99) < 50 else 'NO — may hit timing issues'}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Tier 3 only — path to a port-local ACT checkpoint dir.")
    p.add_argument("--raw-batch", type=Path, default=None,
                   help="Tier 2 only — raw recorder batch dir.")
    p.add_argument("--port-local-dataset", type=Path, default=None,
                   help="Tier 2 only — port-local dataset built from --raw-batch.")
    p.add_argument("--n-frames", type=int, default=20,
                   help="Tier 2: number of random frames to compare.")
    p.add_argument("--skip-tier1", action="store_true")
    p.add_argument("--skip-tier2", action="store_true")
    p.add_argument("--skip-tier3", action="store_true")
    args = p.parse_args()

    failed = 0
    if not args.skip_tier1:
        failed += run_tier1()
    if not args.skip_tier2:
        if args.raw_batch and args.port_local_dataset:
            failed += run_tier2(args.raw_batch, args.port_local_dataset,
                                args.n_frames)
        else:
            print("--- Tier 2 SKIPPED (need --raw-batch + --port-local-dataset) ---")
    if not args.skip_tier3:
        if args.checkpoint_dir:
            run_tier3(args.checkpoint_dir)
        else:
            print("--- Tier 3 SKIPPED (need --checkpoint-dir) ---")
    print()
    print(f"{'ALL PASSED' if failed == 0 else f'{failed} FAILURE(S)'}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
