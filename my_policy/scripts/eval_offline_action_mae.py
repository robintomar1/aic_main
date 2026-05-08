#!/usr/bin/env python3
"""Offline action-MAE evaluator — load a trained ACT checkpoint, run
inference over the val split of its dataset, report MAE statistics.

Designed for the world-frame vs port-local comparison: run twice (once
per checkpoint+dataset pair), then diff the JSON outputs. The metric is
**raw 7-d action error**, which is fair across frame conventions because
both port-local and world-frame transforms are rigid (preserve distances)
— a 1mm prediction error is 1mm of physical world error in either frame.

What we measure per frame:
  * Per-dim L1 error on the 7-d action vector (3 position + 4 quaternion).
  * Euclidean position error ||Δxyz|| in meters.
  * Geodesic rotation error in degrees (sign-invariant via |dot|).

What we report (overall):
  * Mean / median / p95 / max for position error (m).
  * Mean / median / p95 / max for rotation error (deg).
  * n_frames evaluated, n_episodes evaluated.
  * Optional JSON dump for downstream comparison.

Per-frame `policy.reset()` is called between episodes (action queue is
per-episode in ACT). Within an episode, frames are consumed in order so
the queued action chunks remain coherent.

Usage:
    pixi run python my_policy/scripts/eval_offline_action_mae.py \\
        --checkpoint-dir /root/aic_data/v9_act_build/runs/v9_act_v1/checkpoints/010000/pretrained_model \\
        --dataset-root /root/aic_data/v9_act_build/v9_act_merged_clean \\
        --val-episodes-file /root/aic_data/v9_act_build/v9_act_merged_clean/val_episodes.json \\
        --json-out /tmp/v9_act_v1_step10k_val_mae.json

    # then for port-local:
    pixi run python my_policy/scripts/eval_offline_action_mae.py \\
        --checkpoint-dir /root/aic_data/v9_act_build/runs/v9_pl_v1/checkpoints/010000/pretrained_model \\
        --dataset-root /root/aic_data/v9_act_build/v9_port_local_merged \\
        --val-episodes-file /root/aic_data/v9_act_build/v9_port_local_merged/val_episodes.json \\
        --json-out /tmp/v9_pl_v1_step10k_val_mae.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _load_policy_and_processors(checkpoint_dir: Path, device: torch.device):
    """Load ACT policy + pre/post-processor pipelines from a saved
    checkpoint. Same logic as test_runact_offline.run_tier2 — kept inline
    here so this script has zero cross-script dependencies."""
    import draccus
    from safetensors.torch import load_file
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    cfg_dict = json.loads((checkpoint_dir / "config.json").read_text())
    cfg_dict.pop("type", None)  # draccus rejects this discriminator
    config = draccus.decode(ACTConfig, cfg_dict)

    policy = ACTPolicy(config)
    policy.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")))
    policy.eval().to(device)

    pre = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir), config_filename="policy_preprocessor.json"
    )
    post = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, pre, post


def _quat_geodesic_deg(q_pred: np.ndarray, q_rec: np.ndarray) -> float:
    """Sign-invariant geodesic angle between two unit quaternions, in
    degrees. Quaternions are normalized first since the network output
    isn't guaranteed to be unit-norm.
    """
    qp = q_pred / max(np.linalg.norm(q_pred), 1e-8)
    qr = q_rec / max(np.linalg.norm(q_rec), 1e-8)
    cos_half = min(abs(float(np.dot(qp, qr))), 1.0)
    return float(np.degrees(2.0 * np.arccos(cos_half)))


def _episode_bounds(meta_episodes_table) -> dict[int, tuple[int, int]]:
    """Map episode_index → (dataset_from_index, dataset_to_index) using
    the v3.0 episodes parquet. Verified against memory
    `feedback_lerobot_dataset_api.md`: the v3.0 API is
    `meta.episodes[i]['dataset_from_index'][0]` (a 1-element list).
    """
    out: dict[int, tuple[int, int]] = {}
    eps_table = meta_episodes_table.to_pylist()
    for r in eps_table:
        ep = int(r["episode_index"])
        # Some lerobot writers store ints directly; some wrap as 1-elt lists.
        f = r["dataset_from_index"]
        t = r["dataset_to_index"]
        from_idx = int(f[0]) if isinstance(f, (list, tuple)) else int(f)
        to_idx = int(t[0]) if isinstance(t, (list, tuple)) else int(t)
        out[ep] = (from_idx, to_idx)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint-dir", type=Path, required=True,
                   help="Path to .../checkpoints/<step>/pretrained_model/")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="LeRobotDataset root (must contain val_episodes.json "
                        "or have one passed via --val-episodes-file).")
    p.add_argument("--val-episodes-file", type=Path, default=None,
                   help="Override path to val_episodes.json. Default: "
                        "<dataset-root>/val_episodes.json.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Optional cap on number of val episodes (debug).")
    p.add_argument("--max-frames-per-episode", type=int, default=None,
                   help="Optional cap on frames per episode (debug).")
    p.add_argument("--json-out", type=Path, default=None,
                   help="Write structured results to this JSON file.")
    args = p.parse_args()

    if not (args.checkpoint_dir / "model.safetensors").exists():
        sys.exit(f"missing model.safetensors in {args.checkpoint_dir}")

    val_path = args.val_episodes_file or (args.dataset_root / "val_episodes.json")
    if not val_path.exists():
        sys.exit(f"missing val_episodes.json: {val_path}")
    val_episodes = sorted(set(int(e) for e in json.loads(val_path.read_text())))
    if args.max_episodes is not None:
        val_episodes = val_episodes[:args.max_episodes]
    print(f"checkpoint:    {args.checkpoint_dir}")
    print(f"dataset:       {args.dataset_root}")
    print(f"val episodes:  {len(val_episodes)} (file: {val_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:        {device}")

    policy, pre, post = _load_policy_and_processors(args.checkpoint_dir, device)
    print(f"policy loaded; chunk_size={policy.config.chunk_size}, "
          f"n_action_steps={policy.config.n_action_steps}")

    # Load dataset, restricted to val episodes — lerobot's `episodes` arg
    # filters at load time so the dataset's __getitem__ indexes only hit
    # val frames, but it renumbers indices densely starting at 0. To keep
    # the per-episode policy.reset() semantics correct, we instead load
    # the FULL dataset and slice manually using the per-episode bounds.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq

    ds = LeRobotDataset(
        repo_id="local/eval_offline_action_mae",
        root=str(args.dataset_root),
        video_backend="pyav",
    )
    eps_meta = pq.read_table(
        str(args.dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    )
    bounds = _episode_bounds(eps_meta)
    print(f"dataset:       {len(ds)} total frames, {len(bounds)} total episodes")

    pos_errs_m: list[float] = []
    rot_errs_deg: list[float] = []
    per_dim_l1: list[np.ndarray] = []
    n_frames_seen = 0
    t_start = time.time()
    for ei, ep in enumerate(val_episodes):
        if ep not in bounds:
            print(f"  WARN: val episode {ep} not in dataset bounds; skipping")
            continue
        from_idx, to_idx = bounds[ep]
        n_in_ep = to_idx - from_idx
        if args.max_frames_per_episode is not None:
            n_in_ep = min(n_in_ep, args.max_frames_per_episode)

        policy.reset()  # ACT's action queue is per-episode
        for off in range(n_in_ep):
            i = from_idx + off
            item = ds[i]
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
            pred = action[0].cpu().numpy()
            recorded = item["action"].numpy()

            per_dim_l1.append(np.abs(pred - recorded))
            pos_errs_m.append(float(np.linalg.norm(pred[:3] - recorded[:3])))
            rot_errs_deg.append(_quat_geodesic_deg(pred[3:7], recorded[3:7]))
            n_frames_seen += 1

        if (ei + 1) % 10 == 0 or ei == len(val_episodes) - 1:
            elapsed = time.time() - t_start
            print(f"  [{ei+1}/{len(val_episodes)}] {n_frames_seen} frames done "
                  f"({elapsed:.1f}s, {n_frames_seen/max(elapsed,1e-3):.1f} fps)")

    pos_arr = np.array(pos_errs_m)
    rot_arr = np.array(rot_errs_deg)
    per_dim_arr = np.stack(per_dim_l1)

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_root": str(args.dataset_root),
        "n_episodes": len(val_episodes),
        "n_frames": n_frames_seen,
        "wall_time_s": time.time() - t_start,
        "pos_err_m": {
            "mean": float(pos_arr.mean()),
            "median": float(np.percentile(pos_arr, 50)),
            "p95": float(np.percentile(pos_arr, 95)),
            "p99": float(np.percentile(pos_arr, 99)),
            "max": float(pos_arr.max()),
        },
        "rot_err_deg": {
            "mean": float(rot_arr.mean()),
            "median": float(np.percentile(rot_arr, 50)),
            "p95": float(np.percentile(rot_arr, 95)),
            "p99": float(np.percentile(rot_arr, 99)),
            "max": float(rot_arr.max()),
        },
        "per_dim_mae": per_dim_arr.mean(axis=0).tolist(),
    }

    print()
    print(f"=== Results ===")
    print(f"  episodes evaluated: {summary['n_episodes']}")
    print(f"  frames evaluated  : {summary['n_frames']}")
    print(f"  wall time         : {summary['wall_time_s']:.1f}s")
    print(f"  position MAE (m)  : "
          f"mean={summary['pos_err_m']['mean']:.4f}  "
          f"median={summary['pos_err_m']['median']:.4f}  "
          f"p95={summary['pos_err_m']['p95']:.4f}  "
          f"max={summary['pos_err_m']['max']:.4f}")
    print(f"  rotation MAE (deg): "
          f"mean={summary['rot_err_deg']['mean']:.2f}  "
          f"median={summary['rot_err_deg']['median']:.2f}  "
          f"p95={summary['rot_err_deg']['p95']:.2f}  "
          f"max={summary['rot_err_deg']['max']:.2f}")
    print(f"  per-dim L1 MAE    : {[f'{x:.4f}' for x in summary['per_dim_mae']]}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
