#!/usr/bin/env python3
"""Probe SmolVLA's raw output saturation vs obs.tcp_z, across every
checkpoint in a run directory, for both port types, averaged over
multiple real images per (port_type, tcp_z) cell.

For each step checkpoint under <run-dir>/checkpoints/:
  For each port_type in {sc, sfp}:
    Sample N (episode, frame) pairs from a reference dataset whose tasks
    match that port_type.
    For each tcp_z in the sweep:
      For each (ep, frame) sample:
        Build obs = state(tcp_z, port_type) + real images from that frame
        Run predict_action_chunk
        Record raw_z[max], raw_z[-1], un_z[-1]
      Print mean ± std across the N samples.

Tells you (a) how saturation evolves across training steps, (b) whether
the model behaves differently for SC vs SFP, and (c) how much image
content (vs state alone) drives the chunk output.

Usage:
    pixi run python my_policy/scripts/smolvla/probe_last_z.py \\
        --run-dir /root/aic_data/v9_act_build/runs/v9_pl_smolvla_v3_corr_cond_minmax \\
        --dataset-root /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset_with_corrections \\
        --n-images 5

Backward compat: if --run-dir points at a single .../pretrained_model
directory, only that checkpoint is probed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import draccus
import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor.pipeline import DataProcessorPipeline


# ---------------------------------------------------------------------------
# Constants — must match build_conditioned_dataset.py + RunSmolVLA.
# ---------------------------------------------------------------------------

PORT_OFFSETS = {"sc": 0.01564, "sfp": 0.0458}
D_DESCEND_THRESH = 0.10
D_INSERTED_THRESH = 0.005
F_CONTACT_THRESH = 8.0

# Defaults for port-specific task strings; --task-strs can override.
DEFAULT_TASK_STR = {
    "sc": "insert sc plug into sc_port_base on sc_port_0",
    "sfp": "insert sfp plug into sfp_port_0 on nic_card_mount_0",
}

# tcp_z sweep (port-local meters).
DEFAULT_TCP_Z_SWEEP = (-0.30, -0.20, -0.10, -0.05, -0.01, 0.00)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(run_dir: Path) -> list[Path]:
    """Returns a sorted list of pretrained_model dirs under run_dir.

    If run_dir IS a pretrained_model dir already, returns [run_dir].
    Otherwise expects <run_dir>/checkpoints/<step>/pretrained_model/ layout.
    """
    if (run_dir / "model.safetensors").exists() and (run_dir / "config.json").exists():
        return [run_dir]

    ckpts_root = run_dir / "checkpoints"
    if not ckpts_root.is_dir():
        raise SystemExit(
            f"{run_dir} is neither a pretrained_model dir nor a run dir "
            f"with a checkpoints/ subdir."
        )
    step_dirs = []
    for d in ckpts_root.iterdir():
        if not d.is_dir():
            continue
        # Skip the "last" symlink to avoid double-probing the same checkpoint.
        if d.name == "last":
            continue
        pretrained = d / "pretrained_model"
        if pretrained.is_dir() and (pretrained / "model.safetensors").exists():
            step_dirs.append(pretrained)
    if not step_dirs:
        raise SystemExit(f"no checkpoints found under {ckpts_root}")
    # Sort by step number (the parent's name).
    step_dirs.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else -1)
    return step_dirs


# ---------------------------------------------------------------------------
# Per-port-type episode discovery + image sampling
# ---------------------------------------------------------------------------

def discover_episodes_by_port_type(dataset_root: Path) -> dict[str, list[int]]:
    """Reads tasks.parquet + per-episode metadata; returns
    {port_type: [episode_index, ...]} for the episodes in the dataset.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    task_idx_to_str = {int(row["task_index"]): str(name)
                       for name, row in tasks_df.iterrows()}

    eps = pq.read_table(
        str(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ).to_pylist()

    out: dict[str, list[int]] = defaultdict(list)
    for rec in eps:
        ep = int(rec["episode_index"])
        task_str = None
        if rec.get("tasks"):
            task_str = rec["tasks"][0]
        elif "task_index" in rec:
            task_str = task_idx_to_str.get(int(rec["task_index"]))
        if task_str is None:
            continue
        t = task_str.lower()
        if "sfp" in t:
            out["sfp"].append(ep)
        elif "sc" in t:
            out["sc"].append(ep)
    return dict(out)


def find_camera_video(dataset_root: Path, camera: str, episode_idx: int) -> Path:
    """Find the mp4 for a given camera and episode in the dataset's videos dir.

    Handles both flat (<videos>/<camera>/episode_N.mp4) and nested
    (<videos>/observation.images.<camera>/.../episode_N.mp4) layouts.
    """
    videos = dataset_root / "videos"
    name = f"episode_{episode_idx:06d}.mp4"
    # Try most-likely paths first.
    for cand in (
        videos / f"observation.images.{camera}" / "chunk-000" / name,
        videos / "chunk-000" / f"observation.images.{camera}" / name,
    ):
        if cand.exists():
            return cand
    # Fallback: search.
    for p in videos.rglob(name):
        if camera in str(p):
            return p
    raise FileNotFoundError(f"no video for camera={camera} ep={episode_idx} in {videos}")


def load_image_chw(video_path: Path, frame_idx: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (288, 256), interpolation=cv2.INTER_AREA)  # (W, H)
    t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).cuda()  # (1, 3, 256, 288)


def episode_length(dataset_root: Path, episode_idx: int) -> int:
    """Look up an episode's frame count from the per-episode meta parquet."""
    import pyarrow.parquet as pq
    eps = pq.read_table(
        str(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ).to_pylist()
    for rec in eps:
        if int(rec["episode_index"]) == episode_idx:
            return int(rec["length"])
    raise KeyError(f"episode {episode_idx} not in meta")


# ---------------------------------------------------------------------------
# State composer (mirrors RunSmolVLA._build_state_*)
# ---------------------------------------------------------------------------

def make_state(state_dim: int, tcp_z: float, port_type: str) -> torch.Tensor:
    s = np.zeros(state_dim, dtype=np.float32)
    s[3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quat
    s[2] = tcp_z
    if state_dim == 38:
        entrance_offset = PORT_OFFSETS[port_type]
        depth = -tcp_z - entrance_offset
        s[33] = depth
        fmag = 0.0
        if fmag >= F_CONTACT_THRESH:
            phase_idx = 2
        elif depth <= D_INSERTED_THRESH:
            phase_idx = 3
        elif depth <= D_DESCEND_THRESH:
            phase_idx = 1
        else:
            phase_idx = 0
        s[34 + phase_idx] = 1.0
    return torch.from_numpy(s).unsqueeze(0).cuda()


# ---------------------------------------------------------------------------
# Un-normalization (auto-detects MEAN_STD vs MIN_MAX from postprocessor)
# ---------------------------------------------------------------------------

def build_unnorm_fn(ckpt: Path):
    from safetensors import safe_open
    post_json = ckpt / "policy_postprocessor.json"
    action_mode = "MEAN_STD"
    if post_json.exists():
        post_cfg = json.loads(post_json.read_text())
        for step in post_cfg.get("steps", []):
            nm = step.get("config", {}).get("norm_map") if step.get("config") else None
            if nm and "ACTION" in nm:
                action_mode = nm["ACTION"]
                break
    safetensors_path = next(
        ckpt.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"),
        None,
    )
    if safetensors_path is None:
        return action_mode, (lambda raw: float(raw))
    with safe_open(str(safetensors_path), framework="numpy") as f:
        try:
            am = f.get_tensor("action.mean")
            asd = f.get_tensor("action.std")
        except Exception:
            am = asd = None
        try:
            amin = f.get_tensor("action.min")
            amax = f.get_tensor("action.max")
        except Exception:
            amin = amax = None
    if action_mode == "MIN_MAX" and amin is not None and amax is not None:
        mid = (float(amin[2]) + float(amax[2])) / 2.0
        half = (float(amax[2]) - float(amin[2])) / 2.0
        return action_mode, (lambda raw: mid + raw * half)
    if am is not None and asd is not None:
        m, sd = float(am[2]), float(asd[2])
        return action_mode, (lambda raw: m + raw * sd)
    return action_mode, (lambda raw: float(raw))


# ---------------------------------------------------------------------------
# Per-checkpoint probe
# ---------------------------------------------------------------------------

def probe_one_checkpoint(
    ckpt: Path,
    dataset_root: Path,
    eps_by_port: dict[str, list[int]],
    tcp_z_values: list[float],
    n_images: int,
    rng: random.Random,
) -> None:
    print(f"\n{'=' * 78}\nCHECKPOINT: {ckpt.parent.name}/pretrained_model")
    print(f"  path: {ckpt}")

    cfg_dict = json.loads((ckpt / "config.json").read_text())
    cfg_dict.pop("type", None)
    cfg_dict.pop("rtc_config", None)
    config = draccus.decode(SmolVLAConfig, cfg_dict)
    policy = SmolVLAPolicy(config)
    policy.load_state_dict(load_file(str(ckpt / "model.safetensors")))
    policy.eval()
    policy.to("cuda")
    pre = DataProcessorPipeline.from_pretrained(
        str(ckpt), config_filename="policy_preprocessor.json"
    )

    state_feat = config.input_features.get("observation.state")
    state_dim = int(state_feat.shape[0]) if state_feat is not None else 26
    if state_dim not in (26, 38):
        print(f"  WARN: unexpected state_dim={state_dim}, defaulting to 26 layout")
        state_dim = 26

    action_mode, unnorm_fn = build_unnorm_fn(ckpt)
    print(f"  chunk_size={config.chunk_size}  n_action_steps={config.n_action_steps}  "
          f"state_dim={state_dim}  action_norm={action_mode}")

    # Pre-sample N (episode, frame) image triplets for each port type.
    image_samples: dict[str, list[dict[str, torch.Tensor]]] = {}
    for port_type in ("sc", "sfp"):
        eps_pool = eps_by_port.get(port_type, [])
        if not eps_pool:
            print(f"  no episodes for port_type={port_type}; skipping")
            image_samples[port_type] = []
            continue
        samples: list[dict[str, torch.Tensor]] = []
        for _ in range(n_images):
            ep = rng.choice(eps_pool)
            try:
                ep_len = episode_length(dataset_root, ep)
            except Exception:
                ep_len = 200  # fallback
            frame_idx = rng.randint(0, max(0, ep_len - 1))
            imgs = {}
            for cam in ("left_camera", "center_camera", "right_camera"):
                try:
                    vp = find_camera_video(dataset_root, cam, ep)
                    imgs[f"observation.images.{cam}"] = load_image_chw(vp, frame_idx)
                except Exception as e:
                    print(f"  WARN: failed to load {cam} ep={ep} frame={frame_idx}: {e}")
                    imgs[f"observation.images.{cam}"] = torch.zeros(
                        1, 3, 256, 288, device="cuda"
                    )
            samples.append(imgs)
        image_samples[port_type] = samples

    # Sweep.
    for port_type in ("sc", "sfp"):
        if not image_samples[port_type]:
            continue
        offset = PORT_OFFSETS[port_type]
        print()
        print(f"  --- port_type={port_type}  entrance_offset={offset:+.5f}m  "
              f"n_images={len(image_samples[port_type])} ---")
        header = (f"  {'obs.tcp_z':>10} | "
                  f"{'raw_z[max] μ±σ':>18} {'raw_z[-1] μ±σ':>18} | "
                  f"{'un_z[-1] μ±σ (m)':>22}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        task_str = DEFAULT_TASK_STR[port_type]
        for tcp_z in tcp_z_values:
            state = make_state(state_dim, tcp_z, port_type)
            raw_maxs = []
            raw_lasts = []
            un_lasts = []
            for imgs in image_samples[port_type]:
                raw_obs = {**imgs, "task": task_str, "observation.state": state}
                obs = pre(raw_obs)
                with torch.no_grad():
                    actions = policy.predict_action_chunk(obs)
                az = actions[0, :, 2].cpu().numpy()
                raw_maxs.append(float(az.max()))
                raw_lasts.append(float(az[-1]))
                un_lasts.append(unnorm_fn(float(az[-1])))
            rmax_m, rmax_s = float(np.mean(raw_maxs)), float(np.std(raw_maxs))
            rlast_m, rlast_s = float(np.mean(raw_lasts)), float(np.std(raw_lasts))
            ulast_m, ulast_s = float(np.mean(un_lasts)), float(np.std(un_lasts))
            print(f"  {tcp_z:+10.3f} | "
                  f"{rmax_m:+8.3f}±{rmax_s:.3f}    "
                  f"{rlast_m:+8.3f}±{rlast_s:.3f}   | "
                  f"{ulast_m:+10.4f}±{ulast_s:.4f}")

    # Free GPU memory before the next checkpoint loads.
    del policy
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Either a run directory (containing checkpoints/) "
                        "or a single .../pretrained_model dir.")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Reference dataset to source real images from "
                        "(e.g. v9_port_local_smolvla_dataset_with_corrections).")
    p.add_argument("--n-images", type=int, default=5,
                   help="Number of (episode, frame) samples per port type "
                        "averaged at each tcp_z value. Default 5.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for image sampling. Default 0 — same images "
                        "across checkpoints so direct comparison is meaningful.")
    p.add_argument(
        "--tcp-z-values", type=float, nargs="+", default=list(DEFAULT_TCP_Z_SWEEP),
        help=f"Port-local tcp_z values to sweep. Default {list(DEFAULT_TCP_Z_SWEEP)}.",
    )
    args = p.parse_args()

    if not args.dataset_root.is_dir():
        raise SystemExit(f"--dataset-root not a directory: {args.dataset_root}")

    print(f"discovering checkpoints under: {args.run_dir}")
    ckpts = discover_checkpoints(args.run_dir)
    print(f"found {len(ckpts)} checkpoint(s):")
    for c in ckpts:
        print(f"  {c.parent.name}")

    print(f"\nlisting episodes by port type from: {args.dataset_root}")
    eps_by_port = discover_episodes_by_port_type(args.dataset_root)
    for port, eps in eps_by_port.items():
        print(f"  {port}: {len(eps)} episodes")

    rng = random.Random(args.seed)
    for ckpt in ckpts:
        probe_one_checkpoint(
            ckpt, args.dataset_root, eps_by_port,
            tcp_z_values=list(args.tcp_z_values),
            n_images=args.n_images,
            rng=random.Random(args.seed),  # same images per checkpoint
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
