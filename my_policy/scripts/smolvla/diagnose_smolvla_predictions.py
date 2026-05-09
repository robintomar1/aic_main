#!/usr/bin/env python3
"""Per-frame SmolVLA diagnostic — clone of `diagnose_episode_predictions.py`
swapped to SmolVLAPolicy + language input.

Reads from the SmolVLA dataset (32-dim state, per-episode `tasks` string),
runs `policy.reset()` at episode start (mimicking the live shim), then
steps frame-by-frame using the dataset's pre-prepared observation tensors
plus the per-episode task string.

Reports the same per-frame trajectory diff as the ACT version: tcp_z,
recorded vs predicted action_z, quat residual, etc. Phases (early/mid/
late) split the episode into thirds for descent-commit signal.

Usage:
    pixi run python my_policy/scripts/smolvla/diagnose_smolvla_predictions.py \\
        --checkpoint-dir /root/aic_data/v9_act_build/runs/v9_pl_smolvla_v1/checkpoints/050000/pretrained_model \\
        --dataset-root /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset \\
        --val-episodes-file /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset/val_episodes.json \\
        --episode 0 \\
        --csv-out /tmp/diag_smolvla_ep0.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
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
        str(checkpoint_dir), config_filename="policy_preprocessor.json")
    post = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, pre, post


def _quat_residual(q1: np.ndarray, q2: np.ndarray) -> float:
    n1 = max(np.linalg.norm(q1), 1e-9)
    n2 = max(np.linalg.norm(q2), 1e-9)
    return 1.0 - abs(float(np.dot(q1 / n1, q2 / n2)))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--val-episodes-file", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0,
                   help="Position WITHIN val episodes (0 = first val ep).")
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    val_eps = sorted(int(e) for e in json.loads(args.val_episodes_file.read_text()))
    if args.episode >= len(val_eps):
        sys.exit(f"--episode {args.episode} OOB (only {len(val_eps)} val eps)")
    target_ep = val_eps[args.episode]
    print(f"target val episode (global idx): {target_ep}")

    eps_meta = pq.read_table(
        str(args.dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ).to_pylist()
    bounds: dict[int, tuple[int, int]] = {}
    task_per_ep: dict[int, str] = {}
    for r in eps_meta:
        ep = int(r["episode_index"])
        f = r["dataset_from_index"]
        t = r["dataset_to_index"]
        bounds[ep] = (
            int(f[0]) if isinstance(f, (list, tuple)) else int(f),
            int(t[0]) if isinstance(t, (list, tuple)) else int(t),
        )
        tasks = r.get("tasks", [])
        task_per_ep[ep] = tasks[0] if tasks else ""

    if target_ep not in bounds:
        sys.exit(f"episode {target_ep} not in dataset")
    from_idx, to_idx = bounds[target_ep]
    n_frames = to_idx - from_idx
    if args.max_frames is not None:
        n_frames = min(n_frames, args.max_frames)
    task_str = task_per_ep.get(target_ep, "")
    print(f"episode bounds: [{from_idx}, {to_idx}) — {n_frames} frames")
    print(f"task: {task_str!r}")
    if not task_str:
        sys.exit("episode has empty task string — SmolVLA needs language input")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pre, post = _load_policy_and_processors(args.checkpoint_dir, device)
    print(f"loaded policy on {device}; chunk_size={policy.config.chunk_size}, "
          f"n_action_steps={policy.config.n_action_steps}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(
        repo_id="local/diagnose_smolvla_predictions",
        root=str(args.dataset_root),
        video_backend="pyav",
    )

    policy.reset()

    rows = []
    print()
    print(f"{'fi':>4} {'tcpZ':>8} {'recZ':>8} {'predZ':>8} {'ΔrecPred_z':>11} "
          f"{'qres':>9} {'recXY':>14} {'predXY':>14}")
    for off in range(n_frames):
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
        pred = action[0].cpu().numpy()[:7]   # SmolVLA pads to 32
        recorded = item["action"].numpy()
        state = item["observation.state"].numpy()

        # State layout for SmolVLA dataset (32-dim, no task vec):
        #   [0:3]   tcp_pose.position (port-local)
        #   [3:7]   tcp_pose.orientation xyzw (port-local)
        #   [7:13]  tcp_velocity, [13:19] tcp_error, [19:26] joints, [26:32] wrench
        tcp_xyz_port = state[0:3]

        rec_xyz, rec_q = recorded[:3], recorded[3:7]
        pred_xyz, pred_q = pred[:3], pred[3:7]
        qres = _quat_residual(pred_q, rec_q)

        rows.append({
            "frame": off,
            "tcp_x": float(tcp_xyz_port[0]), "tcp_y": float(tcp_xyz_port[1]),
            "tcp_z": float(tcp_xyz_port[2]),
            "rec_x": float(rec_xyz[0]), "rec_y": float(rec_xyz[1]),
            "rec_z": float(rec_xyz[2]),
            "pred_x": float(pred_xyz[0]), "pred_y": float(pred_xyz[1]),
            "pred_z": float(pred_xyz[2]),
            "delta_z": float(pred_xyz[2] - rec_xyz[2]),
            "quat_residual": qres,
            "rec_qx": float(rec_q[0]), "rec_qy": float(rec_q[1]),
            "rec_qz": float(rec_q[2]), "rec_qw": float(rec_q[3]),
            "pred_qx": float(pred_q[0]), "pred_qy": float(pred_q[1]),
            "pred_qz": float(pred_q[2]), "pred_qw": float(pred_q[3]),
        })

        if off < 5 or off > n_frames - 6 or off % 10 == 0:
            print(
                f"{off:>4d} "
                f"{tcp_xyz_port[2]:>+8.4f} "
                f"{rec_xyz[2]:>+8.4f} "
                f"{pred_xyz[2]:>+8.4f} "
                f"{pred_xyz[2] - rec_xyz[2]:>+11.4f} "
                f"{qres:>9.4f} "
                f"({rec_xyz[0]:+.3f},{rec_xyz[1]:+.3f}) "
                f"({pred_xyz[0]:+.3f},{pred_xyz[1]:+.3f})"
            )

    n = len(rows)
    early = rows[: n // 3]
    mid = rows[n // 3 : 2 * n // 3]
    late = rows[2 * n // 3 :]

    def _phase_stats(name, slc):
        if not slc:
            return
        z_mae = np.mean([abs(r["delta_z"]) for r in slc])
        xyz_mae = np.mean([
            np.sqrt((r["pred_x"] - r["rec_x"])**2
                    + (r["pred_y"] - r["rec_y"])**2
                    + (r["pred_z"] - r["rec_z"])**2)
            for r in slc
        ])
        q_mae = np.mean([r["quat_residual"] for r in slc])
        rec_z_avg = np.mean([r["rec_z"] for r in slc])
        pred_z_avg = np.mean([r["pred_z"] for r in slc])
        tcp_z_avg = np.mean([r["tcp_z"] for r in slc])
        print(f"  {name:>5} (n={len(slc):3d}): "
              f"|Δz|_mean={z_mae:.4f}m  pos_mae={xyz_mae:.4f}m  "
              f"qres_mean={q_mae:.4f}    "
              f"avg_rec_z={rec_z_avg:+.4f}  avg_pred_z={pred_z_avg:+.4f}  "
              f"avg_tcp_z={tcp_z_avg:+.4f}")

    print()
    print("=== per-phase aggregates (port-local action target) ===")
    _phase_stats("early", early)
    _phase_stats("mid",   mid)
    _phase_stats("late",  late)

    print()
    print("=== final 10 frames trajectory (where insertion happens) ===")
    for r in rows[-10:]:
        print(f"  fi={r['frame']:>4d}  tcp_z={r['tcp_z']:+.4f}  "
              f"rec_z={r['rec_z']:+.4f}  pred_z={r['pred_z']:+.4f}  "
              f"qres={r['quat_residual']:.4f}")

    if args.csv_out is not None:
        import csv
        with open(args.csv_out, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
