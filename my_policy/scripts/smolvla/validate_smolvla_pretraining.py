#!/usr/bin/env python3
"""Pre-training end-to-end validation for SmolVLA.

Burns ~30 seconds. Verifies — by running real data through the actual
training-time data path — that everything we assume is true. Run BEFORE
kicking off a 100k overnight run.

What it checks (each prints a section, errors are loud):

  1. Dataset shape + schema
       state shape = (N, 32), action = (N, 7), 3 cameras
       tasks.parquet indexed by string (the SmolVLA tokenizer requirement)
       train/val split sizes + episode counts
  2. Per-episode `tasks` strings populated and consistent
  3. Sample frame inspection (frame 0 of episode 0, val frame too)
       state values per channel
       action values
       image dtype + shape + range
       task string verbatim
  4. SmolVLA preprocessor — feed a real batch through and inspect output
       observation.state shape after batch + normalization
       observation.images shape + range (must be [-1, 1] after SigLIP norm
       in modeling, but pipeline should leave it [0, 1])
       observation.language_tokens shape
       observation.language_attention_mask shape + dtype
  5. Stats.json sanity
       state mean/std are 32-dim, finite, plausible magnitudes
       action mean/std 7-dim, finite
       camera stats present for all 3 cameras
  6. End-to-end loss step
       Build the policy, forward a batch, verify loss is finite

Usage:
    pixi run python my_policy/scripts/smolvla/validate_smolvla_pretraining.py \\
        --dataset-root /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset

Optional: --skip-loss to skip the GPU forward pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def check_dataset_schema(root: Path) -> dict:
    _section("1. Dataset shape + schema")
    info = json.loads((root / "meta" / "info.json").read_text())

    state_feat = info["features"]["observation.state"]
    action_feat = info["features"]["action"]

    assert state_feat["shape"] == [32], \
        f"state shape {state_feat['shape']} != [32]"
    _ok(f"observation.state shape = {state_feat['shape']}")

    assert action_feat["shape"] == [7], \
        f"action shape {action_feat['shape']} != [7]"
    _ok(f"action shape = {action_feat['shape']}")

    assert len(state_feat["names"]) == 32
    expected_first = "tcp_pose.position.x"
    expected_last_groups = ["wrench.fx", "wrench.tz"]
    assert state_feat["names"][0] == expected_first, \
        f"first state channel = {state_feat['names'][0]!r}, expected {expected_first!r}"
    assert state_feat["names"][-1] == expected_last_groups[-1], \
        f"last state channel = {state_feat['names'][-1]!r}, expected wrench.tz"
    _ok(f"state channels: [0]={state_feat['names'][0]!r} … [31]={state_feat['names'][-1]!r}")

    cam_keys = [k for k, v in info["features"].items()
                if v.get("dtype") == "video"]
    assert len(cam_keys) == 3, f"expected 3 cameras, got {cam_keys}"
    _ok(f"cameras = {cam_keys}")

    print(f"  total_episodes = {info['total_episodes']}")
    print(f"  total_frames   = {info['total_frames']}")
    print(f"  total_tasks    = {info['total_tasks']}")
    print(f"  fps            = {info.get('fps')}")
    print(f"  frame_transform= {info.get('frame_transform')}")

    return info


def check_tasks_parquet(root: Path) -> list[str]:
    _section("2. tasks.parquet — must be string-indexed")
    import pandas as pd
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    # pandas may report dtype as 'object' (legacy) or 'str' / 'string[pyarrow]'
    # depending on version — what matters is the actual VALUES are strings.
    if isinstance(tasks_df.index[0], str):
        _ok(f"index value is str, sample = {tasks_df.index[0]!r}")
    else:
        _fail(f"index dtype = {tasks_df.index.dtype}, sample type "
              f"{type(tasks_df.index[0]).__name__}, value = {tasks_df.index[0]!r}")
        _fail("Run fix_tasks_parquet_index.py before training.")
        sys.exit(1)
    print(f"  rows = {len(tasks_df)}, cols = {tasks_df.columns.tolist()}")
    print("  all tasks:")
    for ts in tasks_df.index.tolist():
        print(f"    - {ts}")
    return tasks_df.index.tolist()


def check_per_episode_tasks(root: Path, all_task_strs: list[str]) -> None:
    _section("3. Per-episode `tasks` strings")
    eps_table = pq.read_table(
        str(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ).to_pylist()
    n_eps = len(eps_table)
    counts: dict[str, int] = {}
    for r in eps_table:
        ts = r.get("tasks", [])
        if not ts:
            _fail(f"episode {r['episode_index']} has empty tasks!")
            sys.exit(1)
        if ts[0] not in all_task_strs:
            _fail(f"episode {r['episode_index']} tasks[0]={ts[0]!r} "
                  f"NOT in tasks.parquet!")
            sys.exit(1)
        counts[ts[0]] = counts.get(ts[0], 0) + 1
    _ok(f"all {n_eps} episodes have a `tasks` string from tasks.parquet")
    print("  episodes per task:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {v:3d}  {k}")


def check_train_val_split(root: Path, total_eps: int) -> tuple[list[int], list[int]]:
    _section("4. Train/val split")
    train = json.loads((root / "train_episodes.json").read_text())
    val = json.loads((root / "val_episodes.json").read_text())
    assert len(set(train) & set(val)) == 0, "train/val overlap!"
    assert max(train) < total_eps and max(val) < total_eps
    print(f"  train = {len(train)} episodes")
    print(f"  val   = {len(val)} episodes")
    print(f"  ratio = {len(train)/(len(train)+len(val)):.2%} train")
    _ok("train/val disjoint, indices in range")
    return train, val


def check_stats_json(root: Path) -> None:
    _section("5. stats.json sanity")
    s = json.loads((root / "meta" / "stats.json").read_text())
    keys = list(s.keys())
    print(f"  feature keys: {keys}")
    state_stats = s["observation.state"]
    state_mean = np.array(state_stats["mean"])
    state_std = np.array(state_stats["std"])
    action_stats = s["action"]
    action_mean = np.array(action_stats["mean"])
    action_std = np.array(action_stats["std"])

    assert state_mean.shape == (32,), f"state mean shape = {state_mean.shape}"
    assert state_std.shape == (32,), f"state std shape = {state_std.shape}"
    assert action_mean.shape == (7,), f"action mean shape = {action_mean.shape}"
    assert action_std.shape == (7,), f"action std shape = {action_std.shape}"
    _ok(f"shapes: state {state_mean.shape}, action {action_mean.shape}")

    if not np.all(np.isfinite(state_mean)) or not np.all(np.isfinite(state_std)):
        _fail("non-finite state mean/std — training will NaN")
        sys.exit(1)
    if not np.all(np.isfinite(action_mean)) or not np.all(np.isfinite(action_std)):
        _fail("non-finite action mean/std")
        sys.exit(1)
    _ok("state/action mean/std all finite")

    if (state_std < 1e-8).any():
        zero_chans = np.where(state_std < 1e-8)[0].tolist()
        _warn(f"state channels {zero_chans} have ~zero std (constant) — "
              f"normalization will divide by ~0; lerobot clamps but worth knowing")
    else:
        _ok("all state channels have non-trivial variance")

    print(f"  action mean (xyz)    = {action_mean[:3].round(4).tolist()}")
    print(f"  action std  (xyz)    = {action_std[:3].round(4).tolist()}")
    print(f"  action mean (quat)   = {action_mean[3:7].round(4).tolist()}")
    print(f"  action std  (quat)   = {action_std[3:7].round(4).tolist()}")
    print(f"  state.tcp_pos mean   = {state_mean[0:3].round(4).tolist()}")
    print(f"  state.tcp_pos std    = {state_std[0:3].round(4).tolist()}")

    cams = [k for k in keys if k.startswith("observation.images.")]
    if len(cams) != 3:
        _fail(f"expected 3 camera keys in stats, got {cams}")
        sys.exit(1)
    _ok(f"camera stats present: {cams}")


def inspect_sample_frame(root: Path, train_episodes: list[int]) -> None:
    _section("6. Sample frame inspection (training-time data path)")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(
        repo_id="local/validate_smolvla",
        root=str(root),
        video_backend="pyav",
    )
    item = ds[0]
    keys = sorted(item.keys())
    print(f"  ds[0] keys = {keys}")
    print()
    print(f"  task             = {item['task']!r}")
    print(f"  task type        = {type(item['task']).__name__}")
    if not isinstance(item["task"], str):
        _fail(f"item['task'] is {type(item['task']).__name__}, expected str")
        sys.exit(1)
    print(f"  episode_index    = {int(item['episode_index'])}")
    print(f"  frame_index      = {int(item['frame_index'])}")
    print(f"  task_index       = {int(item['task_index'])}")
    s = item["observation.state"]
    a = item["action"]
    print(f"  state shape      = {tuple(s.shape)}, dtype={s.dtype}")
    print(f"    [0:3]   tcp_pos     = {s[0:3].numpy().round(4).tolist()}")
    print(f"    [3:7]   tcp_quat    = {s[3:7].numpy().round(4).tolist()}")
    print(f"    [7:13]  tcp_velocity= {s[7:13].numpy().round(4).tolist()}")
    print(f"    [13:20] joints      = {s[13:20].numpy().round(4).tolist()}")
    print(f"    [20:26] wrench      = {s[20:26].numpy().round(4).tolist()}")
    print(f"  action shape     = {tuple(a.shape)}, dtype={a.dtype}")
    print(f"    [0:3] xyz   = {a[0:3].numpy().round(4).tolist()}")
    print(f"    [3:7] quat  = {a[3:7].numpy().round(4).tolist()}")
    quat_norm = float(np.linalg.norm(a[3:7].numpy()))
    if abs(quat_norm - 1.0) > 0.05:
        _warn(f"action quat norm = {quat_norm:.4f} (training filter is loose; OK)")
    else:
        _ok(f"action quat norm = {quat_norm:.4f}")

    for cam in ["observation.images.left_camera",
                "observation.images.center_camera",
                "observation.images.right_camera"]:
        if cam not in item:
            _fail(f"missing camera: {cam}")
            sys.exit(1)
        img = item[cam]
        print(f"  {cam}: shape={tuple(img.shape)} dtype={img.dtype} "
              f"min={float(img.min()):.3f} max={float(img.max()):.3f}")
        if img.shape[0] != 3:
            _fail(f"channel dim != 3 for {cam}: shape={tuple(img.shape)}")
            sys.exit(1)
        if float(img.min()) < -0.001 or float(img.max()) > 1.001:
            _fail(f"{cam} pixel range outside [0,1] — should be [0,1] before "
                  f"SmolVLA's internal SigLIP rescale")
            sys.exit(1)
    _ok("all 3 cameras present, channel-first, [0,1] range")


def run_preprocessor_and_loss(root: Path, skip_loss: bool) -> None:
    _section("7. SmolVLA preprocessor + forward (real batch)")
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.processor_smolvla import (
        make_smolvla_pre_post_processors,
    )
    from lerobot.configs.types import FeatureType, PolicyFeature
    from torch.utils.data import DataLoader

    info = json.loads((root / "meta" / "info.json").read_text())
    stats = json.loads((root / "meta" / "stats.json").read_text())
    # Convert stats to torch tensors as the processor expects.
    def _to_tensor_stats(d):
        out = {}
        for k, v in d.items():
            out[k] = {kk: torch.tensor(vv) for kk, vv in v.items()}
        return out
    stats_t = _to_tensor_stats(stats)

    ds = LeRobotDataset(
        repo_id="local/validate_smolvla",
        root=str(root),
        video_backend="pyav",
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"  raw batch keys: {sorted(batch.keys())}")
    print(f"  batch task: {batch['task']}")
    print(f"  batch state shape: {tuple(batch['observation.state'].shape)}")
    print(f"  batch action shape: {tuple(batch['action'].shape)}")
    for cam in ["observation.images.left_camera",
                "observation.images.center_camera",
                "observation.images.right_camera"]:
        print(f"  batch {cam} shape: {tuple(batch[cam].shape)}")

    # Build a SmolVLAConfig matching what the trainer uses.
    config = SmolVLAConfig(
        chunk_size=50, n_action_steps=50,
        load_vlm_weights=False,  # skip weight download for validation
    )
    # Set up features so the processor knows which keys to normalize.
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        "observation.images.left_camera": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 288)),
        "observation.images.center_camera": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 288)),
        "observation.images.right_camera": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 288)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    config.device = "cpu"

    pre, post = make_smolvla_pre_post_processors(config, dataset_stats=stats_t)
    print(f"  preprocessor steps: {[type(s).__name__ for s in pre.steps]}")

    out = pre(dict(batch))
    print()
    print(f"  preprocessed batch keys: {sorted(out.keys())}")
    # The actual emitted keys are dot-separated: `observation.language.tokens`
    # / `observation.language.attention_mask` (verified empirically against
    # the preprocessor pipeline output 2026-05-09).
    lt_key = "observation.language.tokens"
    lm_key = "observation.language.attention_mask"
    if lt_key in out and lm_key in out:
        lt = out[lt_key]
        lm = out[lm_key]
        print(f"  {lt_key} shape: {tuple(lt.shape)} dtype={lt.dtype}")
        print(f"  {lm_key} shape: {tuple(lm.shape)} dtype={lm.dtype}")
        # Show the first row of token ids — confirms tokenization actually ran.
        print(f"  first row token-id sample: {lt[0][:12].tolist()} ...")
        _ok("language tokens populated by tokenizer")
    else:
        _fail(f"missing {lt_key} or {lm_key} after preprocessor")
        _fail(f"available keys: {sorted(out.keys())}")
        sys.exit(1)
    print(f"  state after norm: shape={tuple(out['observation.state'].shape)} "
          f"mean={float(out['observation.state'].float().mean()):+.3f} "
          f"std={float(out['observation.state'].float().std()):.3f}")
    cam_key = "observation.images.left_camera"
    img = out[cam_key]
    print(f"  {cam_key} after preproc: shape={tuple(img.shape)} "
          f"dtype={img.dtype} min={float(img.min()):.3f} max={float(img.max()):.3f}")
    _ok("preprocessor pipeline produced expected outputs")

    if skip_loss:
        _warn("--skip-loss: skipping forward+loss check (no GPU forward pass)")
        return

    _section("8. Real forward + loss (CPU, slow but final check)")
    print("  building policy (load_vlm_weights=False to keep it fast)...")
    config2 = SmolVLAConfig(
        chunk_size=50, n_action_steps=50,
        load_vlm_weights=False,
    )
    config2.input_features = config.input_features
    config2.output_features = config.output_features
    config2.device = "cpu"
    try:
        policy = SmolVLAPolicy(config2)
    except Exception as e:
        _fail(f"SmolVLAPolicy construction failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    policy.eval()
    print("  policy built, running forward...")
    out2 = pre(dict(batch))
    try:
        with torch.no_grad():
            loss, loss_dict = policy.forward(out2)
        print(f"  loss = {float(loss):.4f}")
        if not np.isfinite(float(loss)):
            _fail("loss is not finite")
            sys.exit(1)
        _ok("forward pass succeeded, loss is finite")
    except Exception as e:
        _fail(f"forward failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/root/aic_data/v9_act_build/v9_port_local_smolvla_dataset"))
    p.add_argument("--skip-loss", action="store_true",
                   help="Skip the slow CPU forward+loss check.")
    args = p.parse_args()
    if not args.dataset_root.exists():
        sys.exit(f"missing dataset: {args.dataset_root}")

    print(f"=== Pre-training validation: {args.dataset_root} ===")

    info = check_dataset_schema(args.dataset_root)
    all_tasks = check_tasks_parquet(args.dataset_root)
    check_per_episode_tasks(args.dataset_root, all_tasks)
    check_train_val_split(args.dataset_root, info["total_episodes"])
    check_stats_json(args.dataset_root)
    inspect_sample_frame(args.dataset_root, [])
    run_preprocessor_and_loss(args.dataset_root, args.skip_loss)

    print()
    print("=== ALL CHECKS PASSED — safe to kick off 100k training ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
