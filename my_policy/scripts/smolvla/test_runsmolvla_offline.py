#!/usr/bin/env python3
"""Offline tests for `my_policy.ros.RunSmolVLA` — the v9-port-local-smolvla
inference shim.

Same shape as `test_runportlocalact_offline.py` but for the 26-dim state
+ language input convention used by SmolVLA. State drops the auto-regressive
tcp_error block; see RunSmolVLA module docstring.

The high-leverage thing to verify BEFORE burning eval-container minutes:
the shim's per-tick state composition produces the EXACT 26-dim vector
that `make_smolvla_dataset.py` wrote, AND the language string SmolVLA's
preprocessor reads matches the dataset's stored `tasks` per-episode.

Three tiers:

  Tier 1  Helper unit tests on synthetic observations:
            * _build_state_26 slot ordering.
            * Quaternion normalization in _action_port_to_baselink_pose.

  Tier 2  Real-data round-trip:
            For N random frames in batch_100_a:
              1. Load raw frame's TCP/wrench/joint/etc. + port_pose.
              2. Build a fake Observation.
              3. Call _build_state_26 with that port_pose.
              4. Load the corresponding frame from the smolvla dataset's
                 `observation.state` (26-dim).
              5. Compare. Should be identical (modulo float32 precision).

  Tier 3  Latency:
            N forward passes through preprocessor → policy → postprocessor
            using the real SmolVLA checkpoint + a real frame. Verifies
            20 Hz budget on the available GPU. SmolVLA is heavier than
            ACT — expect higher latency; the test will surface it.

Usage:
    pixi run python my_policy/scripts/smolvla/test_runsmolvla_offline.py \\
        --checkpoint-dir /root/aic_data/v9_act_build/runs/v9_pl_smolvla_v2/checkpoints/050000/pretrained_model \\
        --raw-batch /root/aic_data/batch_100_a \\
        --smolvla-dataset /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset
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

# Scripts now live at my_policy/scripts/smolvla/<this>.py — go up THREE
# parents to reach the directory containing `my_policy/`.
_PACKAGE_PARENT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.act.labels import task_string_for  # noqa: E402
from my_policy.localizer.labels import match_episodes_to_trials  # noqa: E402
from my_policy.port_local.dataset_io import (  # noqa: E402
    SRC_PORT_POSE_SLICE,
    is_action_valid,
    is_port_pose_valid,
)


# ---------------------------------------------------------------------------
# Helpers — fake an Observation msg from a raw-batch row.
# (Identical to test_runportlocalact_offline.py — kept inline so the two
# tests don't develop a hidden cross-dependency.)
# ---------------------------------------------------------------------------


def _make_fake_observation(state_47: np.ndarray, raw_action: np.ndarray):
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


def test_state_26_slot_ordering():
    """Hand-craft an Observation with distinct values per slot, port pose
    = identity → expect _build_state_26 to copy through with TCP=I no-op
    and tcp_error block dropped."""
    from my_policy.ros.RunSmolVLA import _build_state_26

    state_47 = np.arange(0.0, 47.0, dtype=np.float32)
    state_47[3:7] = [0.0, 0.0, 0.0, 1.0]  # TCP quat identity
    port_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    out = _build_state_26(_make_fake_observation(state_47, None), port_pose).numpy()
    assert out.shape == (26,), f"got {out.shape}"

    expected = np.zeros(26, dtype=np.float32)
    expected[0:7]   = state_47[0:7]    # TCP pose (port=I → no transform)
    expected[7:13]  = state_47[7:13]   # TCP velocity (port=I, tcp=I → no rot)
    expected[13:20] = state_47[19:26]  # joints (source [19:26], tcp_error skipped)
    expected[20:26] = state_47[26:32]  # wrench (port=I, tcp=I → no rot)

    np.testing.assert_allclose(
        out, expected, atol=1e-6,
        err_msg="state_26 slot ordering doesn't match expected layout",
    )


def test_quaternion_normalization_in_action():
    from my_policy.ros.RunSmolVLA import _action_port_to_baselink_pose

    action_port = np.array([0.01, 0.02, -0.03, 0.6, 0.6, 0.6, 0.9],
                           dtype=np.float64)
    port_pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    pose = _action_port_to_baselink_pose(action_port, port_pose)
    qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    assert abs(norm - 1.0) < 1e-6, f"quat norm = {norm}"


def test_task_string_format():
    """The shim builds the language input via task_string_for — verify it
    matches what `make_port_local_dataset.py` wrote into the dataset's
    per-episode `tasks` list."""
    s = task_string_for("nic_card_mount_3", "sfp_port_0", "sfp")
    assert s == "insert sfp plug into sfp_port_0 on nic_card_mount_3", \
        f"unexpected task string: {s!r}"


def run_tier1():
    print("--- Tier 1: helper unit tests ---")
    tests = [
        ("state 26 slot ordering", test_state_26_slot_ordering),
        ("quat normalization in action", test_quaternion_normalization_in_action),
        ("task string format", test_task_string_format),
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
# Tier 2 — Real-data round-trip vs smolvla dataset.
# ---------------------------------------------------------------------------


def run_tier2(raw_batch: Path, smolvla_root: Path, n_frames: int):
    """For N random frames in raw_batch, reconstruct the Observation and
    verify the shim's _build_state_26 output matches the smolvla dataset's
    observation.state at the same (episode, frame_index)."""
    from my_policy.ros.RunSmolVLA import _build_state_26

    print(f"--- Tier 2: real-data round-trip ({n_frames} frames) ---")
    print(f"  raw_batch     : {raw_batch}")
    print(f"  smolvla       : {smolvla_root}")

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

    sv_table = pq.read_table(
        str(smolvla_root / "data" / "chunk-000" / "file-000.parquet"))
    sv_states = np.stack([
        np.asarray(r, dtype=np.float32)
        for r in sv_table["observation.state"].to_pylist()
    ])
    if sv_states.shape[1] != 26:
        sys.exit(f"smolvla dataset state has {sv_states.shape[1]} channels; "
                 "expected 26 — rebuild via make_smolvla_dataset.py")
    sv_eps = sv_table["episode_index"].to_numpy().astype(np.int64)
    sv_frames = sv_table["frame_index"].to_numpy().astype(np.int64)

    map_path = smolvla_root / "source_episode_map.json"
    if not map_path.exists():
        # The merged-clean→smolvla path may have N→N identity remap. Try to
        # default to identity if absent and raw_batch episodes match.
        print(f"  WARN: missing {map_path}; falling back to identity remap")
        sv_to_raw_ep = {int(e): int(e) for e in np.unique(sv_eps).tolist()}
    else:
        sv_to_raw_ep = {
            int(k): int(v) for k, v in json.loads(map_path.read_text()).items()
        }

    # If the smolvla dataset was built from a MERGED clean dataset (not a
    # single batch), the raw_batch passed here may not contain all source
    # episodes. Skip frames whose raw episode is not in raw_batch.
    raw_eps_set = set(int(e) for e in np.unique(raw_eps).tolist())

    rng = np.random.default_rng(42)
    pick = rng.choice(len(sv_states), size=min(n_frames * 4, len(sv_states)),
                      replace=False)

    failures = 0
    checked = 0
    skipped = 0
    max_pos_err = 0.0
    max_quat_err = 0.0
    max_other_err = 0.0
    for sv_gi in pick:
        if checked >= n_frames:
            break
        sv_gi = int(sv_gi)
        new_ep = int(sv_eps[sv_gi])
        new_fr = int(sv_frames[sv_gi])
        sv_state = sv_states[sv_gi]

        old_ep = sv_to_raw_ep.get(new_ep)
        if old_ep is None or old_ep not in raw_eps_set:
            skipped += 1
            continue

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
            skipped += 1
            continue

        raw_state = raw_states[raw_gi]
        port_pose = raw_state[SRC_PORT_POSE_SLICE].astype(np.float64)

        obs_msg = _make_fake_observation(raw_state, raw_actions[raw_gi])
        shim_state = _build_state_26(obs_msg, port_pose).numpy()

        diff = shim_state - sv_state
        pos_err = float(np.linalg.norm(diff[0:3]))
        q_shim = shim_state[3:7] / max(np.linalg.norm(shim_state[3:7]), 1e-9)
        q_pl = sv_state[3:7] / max(np.linalg.norm(sv_state[3:7]), 1e-9)
        quat_err = 1.0 - abs(float(np.dot(q_shim, q_pl)))
        other_err = float(np.max(np.abs(diff[7:])))

        max_pos_err = max(max_pos_err, pos_err)
        max_quat_err = max(max_quat_err, quat_err)
        max_other_err = max(max_other_err, other_err)

        if pos_err > 1e-5 or quat_err > 1e-5 or other_err > 1e-5:
            print(f"  [FAIL] sv ep={new_ep} fr={new_fr} (raw ep={old_ep} gi={raw_gi}): "
                  f"pos_err={pos_err:.2e}m quat_res={quat_err:.2e} "
                  f"other_max={other_err:.2e}")
            failures += 1
        checked += 1

    print(f"  checked={checked}  skipped={skipped}")
    print(f"  max pos err:           {max_pos_err:.2e} m")
    print(f"  max quat residual:     {max_quat_err:.2e}")
    print(f"  max other-channel err: {max_other_err:.2e}")
    if checked == 0:
        print(f"  [WARN] no frames could be aligned to raw_batch — Tier 2 inconclusive")
        return 1
    if failures == 0:
        print(f"  [PASS] all {checked} frames match smolvla dataset within float32")
    else:
        print(f"  [FAIL] {failures}/{checked} frames disagreed with dataset")
    return failures


# ---------------------------------------------------------------------------
# Tier 3 — Latency.
# ---------------------------------------------------------------------------


def run_tier3(checkpoint_dir: Path, n_iters: int = 30):
    print(f"--- Tier 3: latency ({n_iters} iters) ---")
    import draccus
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_dict = json.loads((checkpoint_dir / "config.json").read_text())
    cfg_dict.pop("type", None)
    config = draccus.decode(SmolVLAConfig, cfg_dict)
    policy = SmolVLAPolicy(config)
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

    # Fake input. SmolVLA needs a `task` string for the tokenizer.
    base_x = {
        "observation.images.left_camera":   torch.zeros(3, 256, 288),
        "observation.images.center_camera": torch.zeros(3, 256, 288),
        "observation.images.right_camera": torch.zeros(3, 256, 288),
        "observation.state": torch.zeros(26),
        "task": "insert sfp plug into sfp_port_0 on nic_card_mount_0",
    }
    # Warm up.
    for _ in range(3):
        a = policy.select_action(pre({**base_x}))
        _ = post(a)
    times_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        a = policy.select_action(pre({**base_x}))
        _ = post(a)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times_ms)
    print(f"  per-call ms:  mean={arr.mean():.1f}  p50={np.percentile(arr,50):.1f}  "
          f"p99={np.percentile(arr,99):.1f}  max={arr.max():.1f}")
    print(f"  20 Hz budget = 50 ms/call. p99 < 50ms? "
          f"{'YES' if np.percentile(arr,99) < 50 else 'NO — may need n_action_steps>1 to amortize'}")
    print(f"  (note: SmolVLA queues actions over n_action_steps={config.n_action_steps}; "
          f"only the chunk-boundary tick pays the heavy cost.)")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Tier 3 only — path to a SmolVLA checkpoint dir.")
    p.add_argument("--raw-batch", type=Path, default=None,
                   help="Tier 2 only — raw recorder batch dir.")
    p.add_argument("--smolvla-dataset", type=Path, default=None,
                   help="Tier 2 only — smolvla dataset built by make_smolvla_dataset.py.")
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--skip-tier1", action="store_true")
    p.add_argument("--skip-tier2", action="store_true")
    p.add_argument("--skip-tier3", action="store_true")
    args = p.parse_args()

    failed = 0
    if not args.skip_tier1:
        failed += run_tier1()
    if not args.skip_tier2:
        if args.raw_batch and args.smolvla_dataset:
            failed += run_tier2(args.raw_batch, args.smolvla_dataset, args.n_frames)
        else:
            print("--- Tier 2 SKIPPED (need --raw-batch + --smolvla-dataset) ---")
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
