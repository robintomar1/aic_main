#!/usr/bin/env python3
"""Probe the FULL action-chunk z-trajectory of a SmolVLA (or ACT) checkpoint
for one representative (episode, frame) sample per (port_type, tcp_z) bucket.

Where probe_last_z.py reports only raw_z[max] and raw_z[-1] averaged over N
samples, this variant dumps the entire 50-frame action[0, :, 2] trajectory
(in raw normalized space AND in un-normalized meters) for a SINGLE sampled
(episode, frame) pair per bucket — so you can see the SHAPE of what the model
predicts inside the chunk, not just endpoint statistics.

Trajectory shapes worth distinguishing:
  (A) Monotone descend with sharp retract at the LAST 1-3 frames only
        - consistent with loss-weighting that under-penalizes chunk boundaries
  (B) Smooth dive-then-retract (peak somewhere in the middle of the chunk)
        - consistent with the model imitating a demonstrator's dive-and-pause
  (C) Oscillating predictions throughout the chunk
        - consistent with under-fit / under-conditioned chunk decoder
  (D) Flat 'hover' trajectories that just terminate at a slightly retracted z
        - consistent with the model 'giving up' on motion

Output per bucket (one bucket = one (port_type, tcp_z_target) pair):
  - Header line with the sampled (episode, frame, recorded_z)
  - Per-frame lines `i: raw=+0.xxx un=+0.xxxxm` for i in [0, chunk_size)
  - ASCII sparkline over the 50 un-normalized z values
  - Summary: chunk_size, max-idx, min-idx, weakly-monotone-descend?,
             n_descend_steps / n_retract_steps, retract-amplitude (m),
             retract-onset (frame index of max), n_retract_frames.

Usage:
    pixi run python my_policy/scripts/smolvla/probe_chunk_shape.py \\
        --run-dir /root/aic_data/v9_act_build/runs/v9_pl_smolvla_v3_corr_cond_minmax/checkpoints/040000 \\
        --dataset-root /root/aic_data/v9_act_build/v9_port_local_smolvla_dataset_with_corrections_cond

--run-dir accepts ANY of these:
  - a step dir          (.../checkpoints/<step>/)            -> auto-descends to pretrained_model/
  - a pretrained_model  (.../checkpoints/<step>/pretrained_model/)
  - the run root        (.../runs/<run_name>/) is NOT accepted here — use
                         probe_last_z.py for multi-checkpoint sweeps.

NOTE: helpers (episode index, image fetch, state composer, build_unnorm_fn,
policy loaders, port-type discovery) are imported from probe_last_z.py to
avoid duplication. Only the inner per-bucket loop is rewritten.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.processor.pipeline import DataProcessorPipeline

# Reuse every helper from probe_last_z.py (sibling module).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from probe_last_z import (  # noqa: E402
    DEFAULT_TASK_STR,
    DEFAULT_TCP_Z_SWEEP,
    PORT_OFFSETS,
    build_episode_index,
    build_frame_index_by_port_z,
    build_unnorm_fn,
    detect_policy_type,
    discover_episodes_by_port_type,
    fetch_camera_frames,
    load_policy,
    make_dataset_loader,
)


# ---------------------------------------------------------------------------
# Single-checkpoint resolver (more permissive than probe_last_z's variant —
# accepts either a step dir or a pretrained_model dir; explicitly rejects a
# run root since this script is single-checkpoint by design).
# ---------------------------------------------------------------------------

def resolve_single_checkpoint(run_dir: Path) -> Path:
    """Return the pretrained_model directory for the requested checkpoint."""
    if (run_dir / "model.safetensors").exists() and (run_dir / "config.json").exists():
        return run_dir
    inner = run_dir / "pretrained_model"
    if (inner / "model.safetensors").exists() and (inner / "config.json").exists():
        return inner
    # Run-root style: refuse loudly, point at probe_last_z.py.
    if (run_dir / "checkpoints").is_dir():
        raise SystemExit(
            f"--run-dir looks like a run root (has a checkpoints/ subdir). "
            f"This script probes ONE checkpoint at a time. Pass a step dir "
            f"such as {run_dir}/checkpoints/<step>/ instead, or use "
            f"probe_last_z.py for multi-checkpoint sweeps."
        )
    raise SystemExit(
        f"could not find model.safetensors under {run_dir} or "
        f"{run_dir / 'pretrained_model'}; got --run-dir={run_dir}"
    )


# ---------------------------------------------------------------------------
# ASCII sparkline
# ---------------------------------------------------------------------------

_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: np.ndarray) -> str:
    """Map a 1-D float array to a unicode sparkline."""
    if values.size == 0:
        return ""
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return _SPARK[0] * values.size
    norm = (values - lo) / (hi - lo)
    idx = np.clip((norm * (len(_SPARK) - 1)).round().astype(int), 0, len(_SPARK) - 1)
    return "".join(_SPARK[i] for i in idx)


# ---------------------------------------------------------------------------
# Per-bucket trajectory dump
# ---------------------------------------------------------------------------

def dump_bucket_trajectory(
    port_type: str,
    tcp_z_target: float,
    bucket: list[tuple[int, int, float, np.ndarray]],
    rng: random.Random,
    policy,
    policy_type: str,
    state_dim: int,
    pre: DataProcessorPipeline,
    unnorm_fn,
    ds_loader,
    ep_info: dict[int, dict],
    tol_mm: float,
) -> None:
    """Sample ONE (ep, frame) from the bucket and dump the full chunk."""
    print()
    print(f"  ===== port_type={port_type}  tgt tcp_z={tcp_z_target:+.3f}m  "
          f"(bucket size={len(bucket)}, ±{tol_mm:.0f}mm) =====")
    if not bucket:
        print(f"  --- OOD (no frames within ±{tol_mm:.0f}mm of "
              f"{tcp_z_target:+.3f}m) ---")
        return

    ep, local_frame, rec_z, rec_state = rng.sample(bucket, 1)[0]
    print(f"  sample: episode={ep}  frame={local_frame}  "
          f"recorded_tcp_z={rec_z:+.4f}m")

    if rec_state.shape[0] != state_dim:
        raise SystemExit(
            f"\nFATAL: dataset state_dim {rec_state.shape[0]} does not match "
            f"model state_dim {state_dim}. Use the dataset whose state shape "
            f"matches this checkpoint (likely the '_cond' variant for a "
            f"conditioned model)."
        )

    try:
        imgs = fetch_camera_frames(ds_loader, ep_info, ep, local_frame)
    except Exception as e:
        print(f"  WARN: failed to fetch ep={ep} fr={local_frame}: {e}")
        return

    state_t = torch.from_numpy(rec_state.astype(np.float32)).unsqueeze(0).cuda()
    raw_obs = {**imgs, "observation.state": state_t}
    if policy_type == "smolvla":
        raw_obs["task"] = DEFAULT_TASK_STR[port_type]
    obs = pre(raw_obs)
    with torch.no_grad():
        actions = policy.predict_action_chunk(obs)
    # actions: (1, chunk_size, action_dim)
    az_raw = actions[0, :, 2].detach().cpu().numpy().astype(np.float64)
    chunk_size = int(az_raw.shape[0])
    # unnorm_fn is `mid + raw * half` (MIN_MAX) or `m + raw * sd` (MEAN_STD)
    # — both vectorize cleanly over a numpy array. The fallback branch
    # (`lambda raw: float(raw)`) does not, so fall back to per-element call.
    try:
        az_un = np.asarray(unnorm_fn(az_raw), dtype=np.float64)
        if az_un.shape != az_raw.shape:
            raise TypeError("unnorm_fn returned wrong shape")
    except (TypeError, ValueError):
        az_un = np.array([unnorm_fn(float(v)) for v in az_raw], dtype=np.float64)

    print(f"  chunk_size={chunk_size}  action_tensor_shape={tuple(actions.shape)}  "
          f"(action[0, :, 2] used)")
    print()

    # Per-frame dump.
    for i in range(chunk_size):
        print(f"    {i:2d}: raw={az_raw[i]:+.3f}  un={az_un[i]:+.4f}m")

    # ASCII sparklines (over un-normalized z).
    print()
    print(f"  sparkline (un_z, low→high mapped over [{az_un.min():+.4f}, "
          f"{az_un.max():+.4f}]m):")
    print(f"    {sparkline(az_un)}")

    # Summary stats — purely on un-normalized z.
    max_idx = int(np.argmax(az_un))
    min_idx = int(np.argmin(az_un))
    diffs = np.diff(az_un)                       # frame-to-frame delta in un_z
    descend_steps = int(np.sum(diffs >= 0))      # weakly deeper (recall: deeper = z increases toward 0)
    retract_steps = int(np.sum(diffs < 0))
    weakly_monotone_descend = bool(np.all(diffs >= 0))
    retract_amplitude_m = float(az_un[max_idx] - az_un[-1])
    n_retract_frames = chunk_size - 1 - max_idx  # number of frames after the peak

    print()
    print(f"  summary:")
    print(f"    max-idx (peak descent)         = {max_idx}  (un_z={az_un[max_idx]:+.4f}m)")
    print(f"    min-idx                        = {min_idx}  (un_z={az_un[min_idx]:+.4f}m)")
    print(f"    weakly monotone descend?       = {weakly_monotone_descend}  "
          f"(diff_un_z >= 0 at every step)")
    print(f"    descend steps / retract steps  = {descend_steps} / {retract_steps}  "
          f"(of {chunk_size - 1} frame-to-frame deltas)")
    print(f"    retract amplitude (max-last)   = {retract_amplitude_m:+.4f}m")
    print(f"    retract onset (frame of max)   = {max_idx}  "
          f"(n_retract_frames after peak = {n_retract_frames})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="A step dir (.../checkpoints/<step>/) or a "
                        "pretrained_model dir. NOT a run root — this script "
                        "probes one checkpoint at a time.")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Reference dataset (must match the checkpoint's "
                        "state_dim, e.g. the '_cond' variant for conditioned "
                        "models).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for picking the single (ep, frame) sample "
                        "per bucket. Default 0 — reproducible across reruns.")
    p.add_argument(
        "--tcp-z-values", type=float, nargs="+", default=list(DEFAULT_TCP_Z_SWEEP),
        help=f"Port-local tcp_z TARGETS to probe. Default {list(DEFAULT_TCP_Z_SWEEP)}.",
    )
    p.add_argument("--tol-mm", type=float, default=10.0,
                   help="Tolerance (mm) around each target tcp_z when filtering "
                        "dataset frames. Default 10mm.")
    args = p.parse_args()

    if not args.dataset_root.is_dir():
        raise SystemExit(f"--dataset-root not a directory: {args.dataset_root}")

    ckpt = resolve_single_checkpoint(args.run_dir)
    print(f"checkpoint: {ckpt}")

    policy_type = detect_policy_type(ckpt)
    policy, config, state_dim = load_policy(ckpt, policy_type)
    pre = DataProcessorPipeline.from_pretrained(
        str(ckpt), config_filename="policy_preprocessor.json"
    )
    if policy_type == "smolvla" and state_dim not in (26, 38):
        print(f"  WARN: unexpected smolvla state_dim={state_dim}, forcing 26")
        state_dim = 26
    action_mode, unnorm_fn = build_unnorm_fn(ckpt)
    chunk_size_attr = getattr(config, "chunk_size", "?")
    n_action_steps_attr = getattr(config, "n_action_steps", "?")
    print(f"  policy_type={policy_type}  chunk_size={chunk_size_attr}  "
          f"n_action_steps={n_action_steps_attr}  state_dim={state_dim}  "
          f"action_norm={action_mode}")

    print(f"\nlisting episodes by port type from: {args.dataset_root}")
    eps_by_port = discover_episodes_by_port_type(args.dataset_root)
    for port, eps in eps_by_port.items():
        print(f"  {port}: {len(eps)} episodes")

    print(f"\nbuilding episode index")
    ep_info = build_episode_index(args.dataset_root)
    print(f"  indexed {len(ep_info)} episodes")

    print(f"\nloading LeRobotDataset")
    ds_loader = make_dataset_loader(args.dataset_root)
    print(f"  total frames: {len(ds_loader)}")

    print(f"\nbuilding per-frame index by port type + recorded tcp_z")
    frame_index_by_port = build_frame_index_by_port_z(
        args.dataset_root, eps_by_port, ep_info,
    )
    for port, frames in frame_index_by_port.items():
        zs = np.array([f[2] for f in frames])
        if len(zs):
            print(f"  {port}: {len(frames)} frames, "
                  f"tcp_z range=[{zs.min():+.4f}, {zs.max():+.4f}]")

    tol_m = args.tol_mm / 1000.0

    print(f"\nseed={args.seed}  (single (ep, frame) sample per bucket)")
    for port_type in ("sc", "sfp"):
        frames = frame_index_by_port.get(port_type, [])
        if not frames:
            print(f"\n  no frames for port_type={port_type}; skipping")
            continue
        all_z = np.array([f[2] for f in frames])
        print(f"\n  ---- port_type={port_type}  "
              f"entrance_offset={PORT_OFFSETS[port_type]:+.5f}m  "
              f"dataset z range=[{all_z.min():+.4f}, {all_z.max():+.4f}]m ----")
        # Fresh RNG per port so port ordering doesn't change the sc samples.
        rng = random.Random(args.seed)
        for tcp_z_target in args.tcp_z_values:
            bucket = [f for f in frames if abs(f[2] - tcp_z_target) <= tol_m]
            dump_bucket_trajectory(
                port_type=port_type,
                tcp_z_target=tcp_z_target,
                bucket=bucket,
                rng=rng,
                policy=policy,
                policy_type=policy_type,
                state_dim=state_dim,
                pre=pre,
                unnorm_fn=unnorm_fn,
                ds_loader=ds_loader,
                ep_info=ep_info,
                tol_mm=args.tol_mm,
            )

    del policy
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
