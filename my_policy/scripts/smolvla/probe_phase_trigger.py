#!/usr/bin/env python3
"""Phase-trigger probe — identify which observation channel(s) signal the
HOVER → COMMIT transition in the trained SmolVLA model.

Hypothesis: live policy gets stuck near a HOVER-regime state because
some auxiliary observation channel (wrench, velocity, image, or joints)
is what training-time conditioning uses to signal "advance to commit."
Live, that channel has values matching HOVER not COMMIT, so the model
keeps emitting HOVER-regime actions (small lag, no progression).

Method:
  1. Pick two real frames from the SAME episode:
       - hover_frame: tcp_z (port-local) near -0.119 (matches live stuck state)
       - commit_frame: tcp_z near -0.085 (model committed to descent here)
  2. Run model on each → reference predictions (a_hover, a_commit).
  3. Build hybrid frames: take hover frame, swap one channel from commit:
       - W: hover state but wrench from commit
       - V: hover state but tcp velocity from commit
       - J: hover state but joints from commit
       - I: hover state but IMAGES from commit (single hardest single channel)
       - C: hover state but tcp_pose itself from commit (sanity: should give ~commit prediction)
  4. Run model on each hybrid → predicted action.
  5. Compare lag (action_z - tcp_z) for each. If a hybrid flips from
     hover-lag (-5 to -10mm) to commit-lag (-3mm or even positive), that
     channel was the discriminator.

Prints a table. Decision rule:
  - If only IMAGES flip the prediction → image distribution shift is the cause.
  - If WRENCH flips → wrench was the gating signal (and live wrench=0 keeps us stuck).
  - If multiple channels are needed → multi-channel conditioning, harder fix.
  - If NONE flip and only TCP_POSE itself does → model is purely state-driven on
    tcp_pose, and our 'wrench is the trigger' theory is wrong.

Usage:
    pixi run python my_policy/scripts/smolvla/probe_phase_trigger.py \\
        --checkpoint-dir /root/aic_data/v9_act_build/runs/v9_pl_smolvla_v1/checkpoints/050000/pretrained_model \\
        --dataset-root /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset
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
        str(checkpoint_dir), config_filename="policy_preprocessor.json"
    )
    post = DataProcessorPipeline.from_pretrained(
        str(checkpoint_dir),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, pre, post


def _run_one(policy, pre, post, obs: dict) -> np.ndarray:
    """Run policy on a single observation dict, return 7-dim action."""
    policy.reset()  # fresh queue per probe so we always see chunk[0]
    pred = policy.select_action(pre(dict(obs)))
    pred = post(pred)
    return pred[0].cpu().numpy()[:7]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--target-tcp-z-hover", type=float, default=-0.119,
                   help="Port-local tcp_z to match for the HOVER probe. "
                        "Default -0.119 ≈ where live policy gets stuck.")
    p.add_argument("--target-tcp-z-commit", type=float, default=-0.085,
                   help="Port-local tcp_z for the COMMIT probe (model fully descending).")
    p.add_argument("--n-trials", type=int, default=3,
                   help="Repeat probe over N different episodes; report each.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    policy, pre, post = _load_policy_and_processors(args.checkpoint_dir, device)
    print(f"chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}")
    print()

    # Load the dataset for image access via LeRobotDataset (handles video decode).
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(
        repo_id="local/probe",
        root=str(args.dataset_root),
        video_backend="pyav",
    )

    # Use the parquet directly to find frames by tcp_z without iterating ds (slow).
    t = pq.read_table(
        str(args.dataset_root / "data" / "chunk-000" / "file-000.parquet"))
    states = np.stack(t["observation.state"].to_pylist())
    actions = np.stack(t["action"].to_pylist())
    eps = np.array(t["episode_index"].to_pylist())

    # Per episode: find frames closest to hover-z and commit-z. Require both in
    # same episode so other-channel context is consistent.
    eps_meta = pq.read_table(
        str(args.dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ).to_pylist()
    bounds = {}
    task_strs = {}
    for r in eps_meta:
        e = int(r["episode_index"])
        f = r["dataset_from_index"]
        tt = r["dataset_to_index"]
        bounds[e] = (
            int(f[0]) if isinstance(f, (list, tuple)) else int(f),
            int(tt[0]) if isinstance(tt, (list, tuple)) else int(tt),
        )
        task_strs[e] = (r.get("tasks") or [""])[0]

    candidates = []
    for e, (lo, hi) in sorted(bounds.items()):
        ep_z = states[lo:hi, 2]
        # Closest-z indices (relative to ep start).
        i_h = int(np.argmin(np.abs(ep_z - args.target_tcp_z_hover)))
        i_c = int(np.argmin(np.abs(ep_z - args.target_tcp_z_commit)))
        # Require closeness within tolerance.
        if abs(ep_z[i_h] - args.target_tcp_z_hover) < 0.005 and \
           abs(ep_z[i_c] - args.target_tcp_z_commit) < 0.005 and \
           i_c > i_h:  # commit comes after hover in time
            candidates.append((e, lo + i_h, lo + i_c))
        if len(candidates) >= args.n_trials:
            break
    if not candidates:
        sys.exit("could not find episodes with both target tcp_z values "
                 "within tolerance")
    print(f"using {len(candidates)} episode(s): {[c[0] for c in candidates]}")
    print()

    # ------------------------------------------------------------------
    # The probe per (hover_idx, commit_idx) pair.
    # ------------------------------------------------------------------

    def lag_str(label: str, action: np.ndarray, tcp_state: np.ndarray) -> str:
        a_xyz = action[:3]
        s_xyz = tcp_state[:3]
        d = a_xyz - s_xyz
        return (f"{label:>8s}: a_xyz=({action[0]:+.4f},{action[1]:+.4f},{action[2]:+.4f}) "
                f"|  delta_xyz=({d[0]*1000:+6.2f},{d[1]*1000:+6.2f},{d[2]*1000:+6.2f})mm")

    for ep_idx, hover_gi, commit_gi in candidates:
        task_str = task_strs[ep_idx]
        print(f"=== episode {ep_idx}  task={task_str!r} ===")
        print(f"   hover global idx: {hover_gi}  tcp_z={states[hover_gi,2]:+.4f}")
        print(f"  commit global idx: {commit_gi}  tcp_z={states[commit_gi,2]:+.4f}")

        # Grab full obs items from LeRobotDataset (gets decoded images).
        item_h = ds[hover_gi]
        item_c = ds[commit_gi]

        def make_obs(state_vec: torch.Tensor, images: dict, task: str) -> dict:
            return {
                "observation.images.left_camera":   images["left_camera"],
                "observation.images.center_camera": images["center_camera"],
                "observation.images.right_camera":  images["right_camera"],
                "observation.state": state_vec,
                "task": task,
            }

        imgs_h = {
            "left_camera":   item_h["observation.images.left_camera"],
            "center_camera": item_h["observation.images.center_camera"],
            "right_camera":  item_h["observation.images.right_camera"],
        }
        imgs_c = {
            "left_camera":   item_c["observation.images.left_camera"],
            "center_camera": item_c["observation.images.center_camera"],
            "right_camera":  item_c["observation.images.right_camera"],
        }
        st_h = item_h["observation.state"].clone()
        st_c = item_c["observation.state"].clone()

        # Reference predictions (raw frames).
        a_hover = _run_one(policy, pre, post,
                           make_obs(st_h, imgs_h, task_str))
        a_commit = _run_one(policy, pre, post,
                            make_obs(st_c, imgs_c, task_str))
        rec_h = actions[hover_gi]
        rec_c = actions[commit_gi]

        print()
        print("  REFERENCE predictions (raw frames):")
        print(f"  {lag_str('rec_H', rec_h, st_h.numpy())}  (recorded oracle)")
        print(f"  {lag_str('pred_H', a_hover, st_h.numpy())}")
        print(f"  {lag_str('rec_C', rec_c, st_c.numpy())}  (recorded oracle)")
        print(f"  {lag_str('pred_C', a_commit, st_c.numpy())}")

        # Now SWAPS: hover state, but with one channel from commit.
        # State layout (32-dim):
        #   [0:7]   tcp_pose (port-local)
        #   [7:13]  tcp_velocity (port-local)
        #   [13:19] tcp_error (frame-invariant)
        #   [19:26] joint_positions
        #   [26:32] wrench (port-local)
        print()
        print("  SWAPS (start from HOVER, replace one channel with COMMIT's):")

        def swap_state(slc: slice) -> torch.Tensor:
            new = st_h.clone()
            new[slc] = st_c[slc]
            return new

        # Tcp_pose swap (sanity: should yield commit-like prediction).
        st_swap_pose = swap_state(slice(0, 7))
        a = _run_one(policy, pre, post, make_obs(st_swap_pose, imgs_h, task_str))
        print(f"  {lag_str('+TCP_POS', a, st_swap_pose.numpy())}  ← should look commit-like")

        # Velocity swap.
        st_swap_vel = swap_state(slice(7, 13))
        a = _run_one(policy, pre, post, make_obs(st_swap_vel, imgs_h, task_str))
        print(f"  {lag_str('+VEL', a, st_swap_vel.numpy())}")

        # Joints swap.
        st_swap_j = swap_state(slice(19, 26))
        a = _run_one(policy, pre, post, make_obs(st_swap_j, imgs_h, task_str))
        print(f"  {lag_str('+JOINTS', a, st_swap_j.numpy())}")

        # Wrench swap.
        st_swap_w = swap_state(slice(26, 32))
        a = _run_one(policy, pre, post, make_obs(st_swap_w, imgs_h, task_str))
        print(f"  {lag_str('+WRENCH', a, st_swap_w.numpy())}")

        # Image swap (most likely candidate — vision encoder is the heavy lift).
        a = _run_one(policy, pre, post, make_obs(st_h, imgs_c, task_str))
        print(f"  {lag_str('+IMAGES', a, st_h.numpy())}")

        # Combo: state + images all from commit (this should fully match commit pred).
        a = _run_one(policy, pre, post, make_obs(st_c, imgs_c, task_str))
        print(f"  {lag_str('all C  ', a, st_c.numpy())}  ← should equal pred_C")

        # Also probe: hover state + COMMIT-magnitude wrench (no swap, just amplify).
        # Wrench at HOVER is ~1.5N; at COMMIT ~2.5N. Try amplifying HOVER wrench 2x.
        st_amp = st_h.clone()
        st_amp[26:32] = st_h[26:32] * 2.0  # amplify wrench
        a = _run_one(policy, pre, post, make_obs(st_amp, imgs_h, task_str))
        print(f"  {lag_str('2xWRENCH', a, st_amp.numpy())}")

        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
