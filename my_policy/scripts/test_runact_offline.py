#!/usr/bin/env python3
"""Offline smoke test for my_policy.ros.RunACT — runs WITHOUT the eval
container, WITHOUT ROS plumbing.

Three tiers (ALL must pass before bringing up docker compose):

  Tier 1 — Helper unit tests
    Pure logic for _build_state / _ros_image_to_chw_float / _action_to_pose
    against synthetic inputs. Catches ordering, normalization, quaternion
    bugs without loading any model.

  Tier 2 — Model prediction integration
    Loads the trained checkpoint + the merged dataset, picks N frames,
    feeds the dataset's already-prepared tensors directly into
    preprocessor → policy → postprocessor (BYPASSES the helpers, which
    are tested in Tier 1). Compares the predicted action against the
    recorded action.

    The ACT chunk_size is 100, so a single first-of-chunk action need
    not exactly equal the recorded step; we use a loose threshold.
    NOTE: if the model is still mid-training, residuals will be larger
    than they will be at convergence — failure here mid-training is
    expected and not a shim bug.

  Tier 3 — Latency budget
    Times N select_action calls (after warm-up). The 20 Hz loop allows
    50 ms / tick. Local 48 GB GPU is faster than cloud L4 (24 GB);
    treat the result as a lower bound.

Run:
    pixi run python my_policy/scripts/test_runact_offline.py
    pixi run python my_policy/scripts/test_runact_offline.py --skip-tier2 --skip-tier3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


# Make `my_policy.*` importable when run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "my_policy"))


# ---------------------------------------------------------------------------
# Tier 1 — Helper unit tests (no torch model, no rclpy node)
# ---------------------------------------------------------------------------

def _make_fake_observation(state_offset: float = 0.0):
    """Build a SimpleNamespace mimicking aic_model_interfaces/Observation.

    Each field gets a distinct numeric value so a misordering in
    _build_state shows up as a wrong slot rather than a coincidentally
    correct one.
    """
    cs = SimpleNamespace(
        tcp_pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=4.0, y=5.0, z=6.0, w=7.0),
        ),
        tcp_velocity=SimpleNamespace(
            linear=SimpleNamespace(x=8.0, y=9.0, z=10.0),
            angular=SimpleNamespace(x=11.0, y=12.0, z=13.0),
        ),
        tcp_error=[14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        fts_tare_offset=SimpleNamespace(
            wrench=SimpleNamespace(
                force=SimpleNamespace(x=0.5, y=0.5, z=0.5),
                torque=SimpleNamespace(x=0.5, y=0.5, z=0.5),
            )
        ),
    )
    obs = SimpleNamespace(
        controller_state=cs,
        joint_states=SimpleNamespace(
            position=[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 99.0]
        ),
        # raw wrench: compensated = raw - tare = (force values - 0.5)
        wrist_wrench=SimpleNamespace(
            wrench=SimpleNamespace(
                force=SimpleNamespace(x=27.5, y=28.5, z=29.5),
                torque=SimpleNamespace(x=30.5, y=31.5, z=32.5),
            )
        ),
    )
    if state_offset:
        # Allow tests to permute by a constant for "different inputs ⇒
        # different outputs" sanity.
        obs.controller_state.tcp_pose.position.x += state_offset
    return obs


def test_build_state_order():
    from my_policy.ros.RunACT import _build_state

    task_vec = np.zeros(12, dtype=np.float32)
    task_vec[0] = 1.0   # module = nic_card_mount_0
    task_vec[7] = 1.0   # port_in_module = sfp_port_0
    task_vec[10] = 1.0  # port_type = sfp

    obs = _make_fake_observation()
    state = _build_state(obs, task_vec).numpy()
    assert state.shape == (44,), f"got {state.shape}"

    # tcp_pose (7): 1..7
    np.testing.assert_array_equal(state[0:7], [1, 2, 3, 4, 5, 6, 7])
    # tcp_velocity (6): 8..13
    np.testing.assert_array_equal(state[7:13], [8, 9, 10, 11, 12, 13])
    # tcp_error (6): 14..19
    np.testing.assert_array_equal(state[13:19], [14, 15, 16, 17, 18, 19])
    # joint_positions (7): 20..26 — 99.0 (8th joint) MUST be excluded
    np.testing.assert_array_equal(state[19:26], [20, 21, 22, 23, 24, 25, 26])
    # wrench (6): raw - tare = 27..32
    np.testing.assert_array_almost_equal(state[26:32], [27, 28, 29, 30, 31, 32])
    # task_vec (12)
    np.testing.assert_array_equal(state[32:44], task_vec)
    print("  PASS test_build_state_order")


def test_image_helper_shape_and_range():
    from my_policy.ros.RunACT import _ros_image_to_chw_float

    # Fake a 1024x1152 RGB image (the Basler publish size).
    H, W = 1024, 1152
    img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    fake = SimpleNamespace(data=img.tobytes(), height=H, width=W)
    t = _ros_image_to_chw_float(fake, scaling=0.25)
    assert t.shape == (3, 256, 288), f"got {t.shape}"
    assert t.dtype == torch.float32
    assert 0.0 <= t.min().item() and t.max().item() <= 1.0
    print("  PASS test_image_helper_shape_and_range")


def test_quaternion_normalization():
    from my_policy.ros.RunACT import _action_to_pose

    # Non-unit quaternion 5x5x5x5 → ||q|| = 10
    a = np.array([0.1, 0.2, 0.3, 5, 5, 5, 5], dtype=np.float32)
    pose = _action_to_pose(a)
    o = pose.orientation
    norm = (o.x ** 2 + o.y ** 2 + o.z ** 2 + o.w ** 2) ** 0.5
    assert abs(norm - 1.0) < 1e-5, f"quaternion not unit: norm={norm}"
    # position passes through untouched (float32 → float roundoff: ≈, not ==)
    assert abs(pose.position.x - 0.1) < 1e-6
    assert abs(pose.position.y - 0.2) < 1e-6
    assert abs(pose.position.z - 0.3) < 1e-6
    print("  PASS test_quaternion_normalization")


def test_zero_quaternion_safe():
    from my_policy.ros.RunACT import _action_to_pose

    a = np.array([1, 2, 3, 0, 0, 0, 0], dtype=np.float32)
    pose = _action_to_pose(a)
    # Identity rotation fallback (NaN avoidance), not random NaNs.
    assert pose.orientation.w == 1.0
    assert pose.orientation.x == 0.0
    print("  PASS test_zero_quaternion_safe")


def test_task_vector_encodes_canonically():
    from my_policy.act.labels import encode_task_vector

    # SFP: nic_card_mount_3 + sfp_port_1 + sfp
    v = encode_task_vector("nic_card_mount_3", "sfp_port_1", "sfp")
    assert v.shape == (12,)
    # module slot 3, port_in_module slot 1 (offset 7), port_type slot 0 (offset 10)
    expected = np.zeros(12, dtype=np.float32)
    expected[3] = 1.0
    expected[7 + 1] = 1.0
    expected[10 + 0] = 1.0
    np.testing.assert_array_equal(v, expected)

    # SC: sc_port_0 (module slot 5) + sc_port_base + sc
    v = encode_task_vector("sc_port_0", "sc_port_base", "sc")
    expected = np.zeros(12, dtype=np.float32)
    expected[5] = 1.0
    expected[7 + 2] = 1.0
    expected[10 + 1] = 1.0
    np.testing.assert_array_equal(v, expected)
    print("  PASS test_task_vector_encodes_canonically")


def run_tier1():
    print("--- Tier 1 (helpers, no model) ---")
    test_task_vector_encodes_canonically()
    test_build_state_order()
    test_image_helper_shape_and_range()
    test_quaternion_normalization()
    test_zero_quaternion_safe()


# ---------------------------------------------------------------------------
# Tier 2 — Model prediction integration (loads checkpoint + dataset)
# ---------------------------------------------------------------------------

def run_tier2(checkpoint_dir: Path, dataset_root: Path,
              n_frames: int, mae_threshold: float):
    print(f"--- Tier 2 (model prediction, {n_frames} frames) ---")
    print(f"  checkpoint: {checkpoint_dir}")
    print(f"  dataset   : {dataset_root}")

    import json
    import draccus
    from safetensors.torch import load_file
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.processor.pipeline import DataProcessorPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device    : {device}")

    cfg_dict = json.loads((checkpoint_dir / "config.json").read_text())
    cfg_dict.pop("type", None)
    config = draccus.decode(ACTConfig, cfg_dict)
    policy = ACTPolicy(config)
    policy.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")))
    policy.eval().to(device)

    pre = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir), config_filename="policy_preprocessor.json"
    )
    post = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir), config_filename="policy_postprocessor.json"
    )

    ds = LeRobotDataset(
        repo_id="local/test_runact",
        root=str(dataset_root),
        video_backend="pyav",
    )
    print(f"  dataset frames: {len(ds)}")

    # Pick frames from the first episode only — keeps `policy.reset()`
    # semantics simple (one continuous trajectory).
    sample_indices = list(range(min(n_frames, len(ds))))

    policy.reset()
    errs = []
    for i in sample_indices:
        item = ds[i]
        # Already-processed tensors from the dataset:
        #   observation.state        -> [44]
        #   observation.images.X     -> [3, 256, 288] in [0,1]
        #   action                   -> [7]
        obs = {
            "observation.images.left_camera":   item["observation.images.left_camera"],
            "observation.images.center_camera": item["observation.images.center_camera"],
            "observation.images.right_camera": item["observation.images.right_camera"],
            "observation.state": item["observation.state"],
        }
        obs = pre(obs)
        with torch.inference_mode():
            action = policy.select_action(obs)
        action = post(action)
        # action may be a tensor [1,7] or a dict containing it; handle both.
        if isinstance(action, dict):
            action = action.get("action", next(iter(action.values())))
        pred = action[0].cpu().numpy() if action.dim() == 2 else action.cpu().numpy()
        recorded = item["action"].numpy()
        errs.append(np.abs(pred - recorded))

    errs = np.stack(errs)               # [N, 7]
    per_dim_mae = errs.mean(axis=0)
    overall_mae = float(errs.mean())
    print(f"  per-dim MAE: {per_dim_mae}")
    print(f"  overall MAE: {overall_mae:.4f}  (threshold {mae_threshold})")
    if overall_mae > mae_threshold:
        print(f"  WARN MAE above threshold — could be (a) shim bug, "
              f"(b) immature checkpoint, or (c) ACT chunking artifact "
              f"(first action of a chunk doesn't exactly match the "
              f"single recorded step).")
    print("  PASS run_tier2 (no exception); inspect MAE above")


# ---------------------------------------------------------------------------
# Tier 3 — Latency
# ---------------------------------------------------------------------------

def run_tier3(checkpoint_dir: Path, n_iters: int):
    print(f"--- Tier 3 (latency, {n_iters} iters) ---")

    import json
    import draccus
    from safetensors.torch import load_file
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.processor.pipeline import DataProcessorPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_dict = json.loads((checkpoint_dir / "config.json").read_text())
    cfg_dict.pop("type", None)
    config = draccus.decode(ACTConfig, cfg_dict)
    policy = ACTPolicy(config)
    policy.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")))
    policy.eval().to(device)

    pre = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir), config_filename="policy_preprocessor.json"
    )
    post = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir), config_filename="policy_postprocessor.json"
    )

    obs = {
        "observation.images.left_camera":   torch.rand(3, 256, 288),
        "observation.images.center_camera": torch.rand(3, 256, 288),
        "observation.images.right_camera": torch.rand(3, 256, 288),
        "observation.state": torch.zeros(44),
    }

    # Warm-up
    policy.reset()
    for _ in range(5):
        x = pre(obs)
        with torch.inference_mode():
            a = policy.select_action(x)
        _ = post(a)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        x = pre(obs)
        with torch.inference_mode():
            a = policy.select_action(x)
        _ = post(a)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    print(f"  device     : {device}")
    print(f"  mean       : {arr.mean():.2f} ms")
    print(f"  p50        : {np.percentile(arr, 50):.2f} ms")
    print(f"  p99        : {np.percentile(arr, 99):.2f} ms")
    print(f"  budget     : 50.00 ms (= 1/20Hz)")
    print(f"  NOTE: cloud eval is L4 (24 GB), local is faster; treat as "
          f"lower bound.")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=Path(os.environ.get(
                       "AIC_ACT_CHECKPOINT",
                       "/root/aic_data/v9_act_build/runs/v9_act_v1/"
                       "checkpoints/last/pretrained_model")))
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/root/aic_data/v9_act_build/v9_act_merged"))
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--mae-threshold", type=float, default=0.05)
    p.add_argument("--n-latency-iters", type=int, default=100)
    p.add_argument("--skip-tier1", action="store_true")
    p.add_argument("--skip-tier2", action="store_true")
    p.add_argument("--skip-tier3", action="store_true")
    args = p.parse_args()

    if not args.skip_tier1:
        run_tier1()
    if not args.skip_tier2:
        if not args.checkpoint.exists():
            print(f"SKIP Tier 2: checkpoint not found at {args.checkpoint}")
        elif not args.dataset_root.exists():
            print(f"SKIP Tier 2: dataset not found at {args.dataset_root}")
        else:
            run_tier2(args.checkpoint, args.dataset_root,
                      args.n_frames, args.mae_threshold)
    if not args.skip_tier3:
        if not args.checkpoint.exists():
            print(f"SKIP Tier 3: checkpoint not found at {args.checkpoint}")
        else:
            run_tier3(args.checkpoint, args.n_latency_iters)

    print("=== all requested tiers completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
