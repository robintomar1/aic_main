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

import draccus
import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor.pipeline import DataProcessorPipeline

# Path setup so we can pull ACT_VALID_TARGETS for ACT task-vec composition.
_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../aic_main
if str(_REPO_ROOT / "my_policy") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "my_policy"))


# ---------------------------------------------------------------------------
# Constants — must match build_conditioned_dataset.py + RunSmolVLA.
# ---------------------------------------------------------------------------

PORT_OFFSETS = {"sc": 0.01564, "sfp": 0.0458}
D_DESCEND_THRESH = 0.10
D_INSERTED_THRESH = 0.005
F_CONTACT_THRESH = 8.0

# Defaults for port-specific task strings (SmolVLA language input).
DEFAULT_TASK_STR = {
    "sc": "insert sc plug into sc_port_base on sc_port_0",
    "sfp": "insert sfp plug into sfp_port_0 on nic_card_mount_0",
}

# Default (mount, port) pairs to use when composing the ACT task_vec
# one-hot for each port type. Must exist in my_policy.act.labels.ACT_VALID_TARGETS.
DEFAULT_ACT_TARGET = {
    "sc": ("sc_port_0", "sc_port_base"),
    "sfp": ("nic_card_mount_0", "sfp_port_0"),
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


def build_episode_index(dataset_root: Path) -> dict[int, dict]:
    """Returns {episode_idx: {'from_index': int, 'length': int}} by reading
    the per-episode meta parquet. Used to translate (ep, local_frame) into
    the LeRobotDataset's global frame index.

    lerobot v3.0 stores episodes concatenated in chunked .mp4 files;
    LeRobotDataset[i] handles the decoding correctly so we use it instead
    of trying to parse the chunked-video layout ourselves.
    """
    import pyarrow.parquet as pq
    eps_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not eps_path.exists():
        raise SystemExit(f"missing episode meta: {eps_path}")
    eps = pq.read_table(str(eps_path)).to_pylist()
    out: dict[int, dict] = {}
    for rec in eps:
        ep = int(rec["episode_index"])
        # v3.0 uses dataset_from_index / dataset_to_index — see memory
        # feedback_lerobot_dataset_api.md.
        from_index = rec.get("dataset_from_index")
        if from_index is None:
            # Older builds may store it as a single-element list.
            df = rec.get("dataset_from_index", [None])
            from_index = df[0] if isinstance(df, (list, tuple)) and df else None
        if from_index is None:
            raise RuntimeError(
                f"episode {ep} meta is missing dataset_from_index "
                f"(keys present: {sorted(rec.keys())[:10]}…)"
            )
        out[ep] = {
            "from_index": int(from_index),
            "length": int(rec["length"]),
        }
    return out


def make_dataset_loader(dataset_root: Path):
    """Wrap LeRobotDataset so frames can be fetched by global index."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(
        repo_id="local/probe",
        root=str(dataset_root),
        video_backend="pyav",
    )


def fetch_camera_frames(
    ds, ep_info: dict[int, dict], episode_idx: int, frame_local: int,
) -> dict[str, torch.Tensor]:
    """Returns {observation.images.<cam>: (1,3,H,W) cuda tensor} for one
    (episode, frame) sample by indexing the LeRobotDataset."""
    info = ep_info.get(episode_idx)
    if info is None:
        raise KeyError(f"episode {episode_idx} not in episode index")
    if frame_local < 0 or frame_local >= info["length"]:
        raise IndexError(
            f"frame {frame_local} out of [0,{info['length']}) for ep {episode_idx}")
    global_idx = info["from_index"] + frame_local
    item = ds[global_idx]
    out = {}
    for cam in ("left_camera", "center_camera", "right_camera"):
        key = f"observation.images.{cam}"
        if key not in item:
            raise KeyError(f"dataset item missing {key}; keys={sorted(item.keys())[:10]}…")
        t = item[key]  # already CHW float [0,1]
        if t.ndim == 3:
            t = t.unsqueeze(0)
        out[key] = t.cuda()
    return out


def episode_length(ep_info: dict[int, dict], episode_idx: int) -> int:
    info = ep_info.get(episode_idx)
    if info is None:
        raise KeyError(f"episode {episode_idx} not in episode index")
    return info["length"]


# ---------------------------------------------------------------------------
# Policy-type detection + loaders
# ---------------------------------------------------------------------------

def detect_policy_type(ckpt: Path) -> str:
    """Return 'act' or 'smolvla' based on config.json's `type` field.
    Falls back to 'smolvla' if not set (backward compat with our old runs)."""
    cfg = json.loads((ckpt / "config.json").read_text())
    t = (cfg.get("type") or "smolvla").lower()
    if t not in ("smolvla", "act"):
        raise ValueError(f"unsupported policy.type={t!r} in {ckpt / 'config.json'}")
    return t


def load_policy(ckpt: Path, policy_type: str):
    """Returns (policy, config, state_dim)."""
    cfg_dict = json.loads((ckpt / "config.json").read_text())
    cfg_dict.pop("type", None)
    if policy_type == "smolvla":
        cfg_dict.pop("rtc_config", None)
        config = draccus.decode(SmolVLAConfig, cfg_dict)
        policy = SmolVLAPolicy(config)
    else:  # act
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        config = draccus.decode(ACTConfig, cfg_dict)
        policy = ACTPolicy(config)
    policy.load_state_dict(load_file(str(ckpt / "model.safetensors")))
    policy.eval()
    policy.to("cuda")
    state_feat = config.input_features.get("observation.state")
    state_dim = int(state_feat.shape[0]) if state_feat is not None else 26
    return policy, config, state_dim


# ---------------------------------------------------------------------------
# State composers
# ---------------------------------------------------------------------------

def _smolvla_state(state_dim: int, tcp_z: float, port_type: str) -> torch.Tensor:
    """Compose state for SmolVLA: 26-dim baseline or 38-dim conditioned."""
    s = np.zeros(state_dim, dtype=np.float32)
    s[3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quat
    s[2] = tcp_z
    if state_dim == 38:
        entrance_offset = PORT_OFFSETS[port_type]
        depth = -tcp_z - entrance_offset
        s[33] = depth
        if depth <= D_INSERTED_THRESH:
            phase_idx = 3
        elif depth <= D_DESCEND_THRESH:
            phase_idx = 1
        else:
            phase_idx = 0
        s[34 + phase_idx] = 1.0
    return torch.from_numpy(s).unsqueeze(0).cuda()


# Cache the ACT_VALID_TARGETS list once (importing my_policy.act.labels).
_ACT_VALID_TARGETS_CACHE = None


def _act_valid_targets():
    global _ACT_VALID_TARGETS_CACHE
    if _ACT_VALID_TARGETS_CACHE is None:
        from my_policy.act.labels import ACT_VALID_TARGETS  # noqa: WPS433
        _ACT_VALID_TARGETS_CACHE = list(ACT_VALID_TARGETS)
    return _ACT_VALID_TARGETS_CACHE


def _act_state(state_dim: int, tcp_z: float, port_type: str) -> torch.Tensor:
    """Compose state for ACT, 44-dim with task one-hot.

    Layout (from my_policy/scripts/make_port_local_dataset.py):
      [ 0:7 ]  tcp_pose (xyz + xyzw quat)
      [ 7:13]  tcp_velocity
      [13:19]  tcp_error
      [19:26]  joint_positions
      [26:32]  wrench
      [32:44]  task_vec — 12-dim one-hot over (mount, port_name) targets
    """
    s = np.zeros(state_dim, dtype=np.float32)
    s[3:7] = [0.0, 0.0, 0.0, 1.0]
    s[2] = tcp_z
    if state_dim >= 44:
        # Set the matching task_vec one-hot.
        mount, port_name = DEFAULT_ACT_TARGET[port_type]
        try:
            idx = _act_valid_targets().index((mount, port_name))
            s[32 + idx] = 1.0
        except ValueError:
            print(f"  WARN: ({mount},{port_name}) not in ACT_VALID_TARGETS — "
                  f"task_vec will be all zeros for {port_type}", file=sys.stderr)
    return torch.from_numpy(s).unsqueeze(0).cuda()


def make_state(
    policy_type: str, state_dim: int, tcp_z: float, port_type: str,
) -> torch.Tensor:
    if policy_type == "smolvla":
        return _smolvla_state(state_dim, tcp_z, port_type)
    return _act_state(state_dim, tcp_z, port_type)


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
    ep_info: dict[int, dict],
    ds_loader,
    tcp_z_values: list[float],
    n_images: int,
    rng: random.Random,
) -> None:
    print(f"\n{'=' * 78}\nCHECKPOINT: {ckpt.parent.name}/pretrained_model")
    print(f"  path: {ckpt}")

    policy_type = detect_policy_type(ckpt)
    policy, config, state_dim = load_policy(ckpt, policy_type)
    pre = DataProcessorPipeline.from_pretrained(
        str(ckpt), config_filename="policy_preprocessor.json"
    )
    if policy_type == "smolvla" and state_dim not in (26, 38):
        print(f"  WARN: unexpected smolvla state_dim={state_dim}, "
              f"forcing baseline 26 layout")
        state_dim = 26

    action_mode, unnorm_fn = build_unnorm_fn(ckpt)
    chunk_size_attr = getattr(config, "chunk_size", "?")
    n_action_steps_attr = getattr(config, "n_action_steps", "?")
    print(f"  policy_type={policy_type}  chunk_size={chunk_size_attr}  "
          f"n_action_steps={n_action_steps_attr}  state_dim={state_dim}  "
          f"action_norm={action_mode}")

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
                ep_len = episode_length(ep_info, ep)
            except KeyError:
                print(f"  WARN: ep={ep} not in episode index; skipping sample")
                continue
            frame_idx = rng.randint(0, max(0, ep_len - 1))
            try:
                imgs = fetch_camera_frames(ds_loader, ep_info, ep, frame_idx)
            except Exception as e:
                print(f"  WARN: failed to fetch ep={ep} frame={frame_idx}: {e}")
                imgs = {
                    f"observation.images.{cam}":
                        torch.zeros(1, 3, 256, 288, device="cuda")
                    for cam in ("left_camera", "center_camera", "right_camera")
                }
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
            state = make_state(policy_type, state_dim, tcp_z, port_type)
            raw_maxs = []
            raw_lasts = []
            un_lasts = []
            for imgs in image_samples[port_type]:
                raw_obs = {**imgs, "observation.state": state}
                # Language input is SmolVLA-only; ACT consumes task as a
                # one-hot inside observation.state instead.
                if policy_type == "smolvla":
                    raw_obs["task"] = task_str
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

    print(f"\nbuilding episode index from: {args.dataset_root / 'meta' / 'episodes'}")
    ep_info = build_episode_index(args.dataset_root)
    print(f"  indexed {len(ep_info)} episodes")
    sample_eps = list(ep_info.items())[:3]
    for ep, info in sample_eps:
        print(f"    ep={ep}: from_index={info['from_index']} length={info['length']}")

    print(f"\nloading LeRobotDataset from: {args.dataset_root}")
    ds_loader = make_dataset_loader(args.dataset_root)
    print(f"  total frames in dataset: {len(ds_loader)}")

    for ckpt in ckpts:
        probe_one_checkpoint(
            ckpt, args.dataset_root, eps_by_port, ep_info, ds_loader,
            tcp_z_values=list(args.tcp_z_values),
            n_images=args.n_images,
            rng=random.Random(args.seed),  # same images per checkpoint
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
