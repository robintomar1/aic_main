#!/usr/bin/env python3
"""Offline action-MAE evaluator for SmolVLA — clone of
`eval_offline_action_mae.py` swapping ACT loader for SmolVLA and feeding
the per-episode `tasks` string as language input.

Same metrics, same JSON output schema → drops into `compare_eval_runs.py`
unchanged for SmolVLA-vs-ACT comparison.

Usage:
    pixi run python my_policy/scripts/smolvla/eval_offline_action_mae_smolvla.py \\
        --checkpoint-dir /root/aic_data/v9_act_build/runs/v9_pl_smolvla_v1/checkpoints/050000/pretrained_model \\
        --dataset-root /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset \\
        --val-episodes-file /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset/val_episodes.json \\
        --json-out /tmp/v9_pl_smolvla_v1_step50k_val_mae.json
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
    import draccus
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    cfg_dict = json.loads((checkpoint_dir / "config.json").read_text())
    cfg_dict.pop("type", None)
    config = draccus.decode(SmolVLAConfig, cfg_dict)

    policy = SmolVLAPolicy(config)
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
    qp = q_pred / max(np.linalg.norm(q_pred), 1e-8)
    qr = q_rec / max(np.linalg.norm(q_rec), 1e-8)
    cos_half = min(abs(float(np.dot(qp, qr))), 1.0)
    return float(np.degrees(2.0 * np.arccos(cos_half)))


def _episode_bounds_and_tasks(meta_episodes_table) -> dict[int, tuple[int, int, str]]:
    """Map episode_index → (from_idx, to_idx, task_string)."""
    out: dict[int, tuple[int, int, str]] = {}
    for r in meta_episodes_table.to_pylist():
        ep = int(r["episode_index"])
        f = r["dataset_from_index"]
        t = r["dataset_to_index"]
        from_idx = int(f[0]) if isinstance(f, (list, tuple)) else int(f)
        to_idx = int(t[0]) if isinstance(t, (list, tuple)) else int(t)
        tasks = r.get("tasks", [])
        task_str = tasks[0] if tasks else ""
        out[ep] = (from_idx, to_idx, task_str)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--val-episodes-file", type=Path, default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--max-frames-per-episode", type=int, default=None)
    p.add_argument("--json-out", type=Path, default=None)
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

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq

    ds = LeRobotDataset(
        repo_id="local/eval_offline_action_mae_smolvla",
        root=str(args.dataset_root),
        video_backend="pyav",
    )
    eps_meta = pq.read_table(
        str(args.dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    )
    bounds = _episode_bounds_and_tasks(eps_meta)
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
        from_idx, to_idx, task_str = bounds[ep]
        n_in_ep = to_idx - from_idx
        if args.max_frames_per_episode is not None:
            n_in_ep = min(n_in_ep, args.max_frames_per_episode)
        if not task_str:
            print(f"  WARN: episode {ep} has empty task string; skipping")
            continue

        policy.reset()
        for off in range(n_in_ep):
            i = from_idx + off
            item = ds[i]
            obs = {
                "observation.images.left_camera":   item["observation.images.left_camera"],
                "observation.images.center_camera": item["observation.images.center_camera"],
                "observation.images.right_camera": item["observation.images.right_camera"],
                "observation.state": item["observation.state"],
                "task": task_str,
            }
            obs = pre(obs)
            with torch.inference_mode():
                action = policy.select_action(obs)
            action = post(action)
            pred = action[0].cpu().numpy()[:7]   # SmolVLA pads to max_action_dim=32
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
