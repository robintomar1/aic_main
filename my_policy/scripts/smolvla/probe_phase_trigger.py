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
    p.add_argument("--port-type", choices=["sfp", "sc", "any"], default="sfp",
                   help="Filter to episodes of this plug type. Default sfp "
                        "(matches the live failure case).")
    p.add_argument("--tcp-z-lo", type=float, default=-0.125,
                   help="Lower bound (inclusive) of port-local tcp_z bucket. "
                        "Default -0.125 — matches live stuck position.")
    p.add_argument("--tcp-z-hi", type=float, default=-0.115,
                   help="Upper bound (exclusive) of port-local tcp_z bucket.")
    p.add_argument("--hover-lag-max", type=float, default=-0.008,
                   help="A frame is 'hover-regime' if (action_z - tcp_z) <= this. "
                        "Default -8mm.")
    p.add_argument("--commit-lag-min", type=float, default=-0.003,
                   help="A frame is 'commit-regime' if (action_z - tcp_z) >= this. "
                        "Default -3mm.")
    p.add_argument("--n-trials", type=int, default=3,
                   help="Number of (hover, commit) frame pairs to probe.")
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
    state = np.stack(t["observation.state"].to_pylist())
    action = np.stack(t["action"].to_pylist())
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

    # Filter episodes by port type (SFP vs SC).
    def _ep_matches_port_type(task_str: str, port_type: str) -> bool:
        if port_type == "any":
            return True
        return task_str.startswith(f"insert {port_type} plug")

    matching_eps = set(e for e, ts in task_strs.items()
                       if _ep_matches_port_type(ts, args.port_type))
    print(f"port-type filter: {args.port_type!r}  matching episodes: {len(matching_eps)}")

    # IN-DISTRIBUTION frame selection: pick pairs at the same tcp_z bucket
    # (where live policy gets stuck) but with different lag regimes.
    # We narrow to a thin tcp_z slice so any two frames from this slice are
    # at "the same" position physically. Then we split into hover-lag and
    # commit-lag clusters by their action lag.
    z_lo, z_hi = args.tcp_z_lo, args.tcp_z_hi
    lag_z = action[:, 2] - state[:, 2]
    in_bucket = (state[:, 2] >= z_lo) & (state[:, 2] < z_hi)
    in_bucket &= np.array([int(e) in matching_eps for e in eps])
    bucket_idx = np.where(in_bucket)[0]
    if len(bucket_idx) == 0:
        sys.exit(f"no frames in tcp_z=[{z_lo},{z_hi}) for port_type={args.port_type}")
    bucket_lag = lag_z[bucket_idx]
    print(f"frames in bucket: {len(bucket_idx)}  "
          f"lag p10={np.percentile(bucket_lag,10)*1000:+.2f}mm  "
          f"p50={np.percentile(bucket_lag,50)*1000:+.2f}mm  "
          f"p90={np.percentile(bucket_lag,90)*1000:+.2f}mm")

    hover_pool = bucket_idx[bucket_lag <= args.hover_lag_max]
    commit_pool = bucket_idx[bucket_lag >= args.commit_lag_min]
    print(f"hover-lag pool (lag<={args.hover_lag_max*1000:.1f}mm): {len(hover_pool)}")
    print(f"commit-lag pool (lag>={args.commit_lag_min*1000:.1f}mm): {len(commit_pool)}")
    if len(hover_pool) < args.n_trials or len(commit_pool) < args.n_trials:
        sys.exit("not enough frames in pools — relax --hover-lag-max/--commit-lag-min "
                 "or widen --tcp-z-lo/-hi")

    rng = np.random.default_rng(0)
    candidates = []
    for k in range(args.n_trials):
        h_gi = int(rng.choice(hover_pool))
        c_gi = int(rng.choice(commit_pool))
        e_h = int(eps[h_gi]); e_c = int(eps[c_gi])
        candidates.append((e_h, e_c, h_gi, c_gi))
        print(f"  pair {k}: hover ep={e_h}@{h_gi} (tcp_z={state[h_gi,2]:+.4f}, action_z={action[h_gi,2]:+.4f}, lag={(action[h_gi,2]-state[h_gi,2])*1000:+.2f}mm)  "
              f"|  commit ep={e_c}@{c_gi} (tcp_z={state[c_gi,2]:+.4f}, action_z={action[c_gi,2]:+.4f}, lag={(action[c_gi,2]-state[c_gi,2])*1000:+.2f}mm)")
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

    for e_h, e_c, hover_gi, commit_gi in candidates:
        task_h = task_strs[e_h]
        task_c = task_strs[e_c]
        print(f"=== hover ep={e_h} ({task_h!r})  |  commit ep={e_c} ({task_c!r}) ===")
        print(f"   hover global idx: {hover_gi}  tcp_z={state[hover_gi,2]:+.4f}")
        print(f"  commit global idx: {commit_gi}  tcp_z={state[commit_gi,2]:+.4f}")
        # Use hover episode's task string for inference (the model sees it).
        # The intent: ask "if model sees hover state + hover task, but commit's
        # aux channels — does prediction flip toward commit-regime?"
        task_str = task_h

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
        rec_h = action[hover_gi]
        rec_c = action[commit_gi]

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

        # Wrench swap (full 6-dim).
        st_swap_w = swap_state(slice(26, 32))
        a = _run_one(policy, pre, post, make_obs(st_swap_w, imgs_h, task_str))
        print(f"  {lag_str('+WRENCH', a, st_swap_w.numpy())}")

        # Wrench Fy alone (statistical analysis showed strongest single-channel signal).
        st_swap_fy = st_h.clone()
        st_swap_fy[27] = st_c[27]
        a = _run_one(policy, pre, post, make_obs(st_swap_fy, imgs_h, task_str))
        print(f"  {lag_str('+Fy', a, st_swap_fy.numpy())}")

        # tcp_error.err_z alone (also showed sign-flip across regimes).
        st_swap_errz = st_h.clone()
        st_swap_errz[15] = st_c[15]
        a = _run_one(policy, pre, post, make_obs(st_swap_errz, imgs_h, task_str))
        print(f"  {lag_str('+err_z', a, st_swap_errz.numpy())}")

        # tcp_error full.
        st_swap_err = swap_state(slice(13, 19))
        a = _run_one(policy, pre, post, make_obs(st_swap_err, imgs_h, task_str))
        print(f"  {lag_str('+TCP_ERR', a, st_swap_err.numpy())}")

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
