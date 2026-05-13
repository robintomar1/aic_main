#!/usr/bin/env python3
"""Probe the model's raw output saturation as a function of obs.tcp_z.

Confirms the saturation hypothesis: if raw_z[max] caps near +1.0 even when
obs.tcp_z is set to the demo's end (~0.00), the model literally cannot
output beyond raw=+1.0 and the un-normalized target stays at z ≈ -0.082.

Usage:
    AIC_PL_SMOLVLA_CHECKPOINT=/root/aic_data/sc_build/runs/<run>/checkpoints/last/pretrained_model \
        pixi run python my_policy/scripts/smolvla/probe_last_z.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import draccus
import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor.pipeline import DataProcessorPipeline


def main() -> int:
    ckpt_env = os.environ.get("AIC_PL_SMOLVLA_CHECKPOINT", "").strip()
    if not ckpt_env:
        raise SystemExit(
            "AIC_PL_SMOLVLA_CHECKPOINT env var is required — set it to "
            "the .../checkpoints/<step>/pretrained_model/ dir."
        )
    ckpt = Path(ckpt_env)

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

    # Detect state dim from the policy config so we build the right shape.
    state_feat = config.input_features.get("observation.state")
    state_dim = int(state_feat.shape[0]) if state_feat is not None else 26
    if state_dim not in (26, 38):
        print(f"WARNING: unexpected state_dim={state_dim}, defaulting to 26 layout")
        state_dim = 26

    # Port entrance offsets — match build_conditioned_dataset.py.
    PORT_OFFSETS = {"sc": 0.01564, "sfp": 0.0458}
    # For probing, assume SC port (matches the task string below).
    PROBE_PORT_TYPE = "sc"
    ENTRANCE_OFFSET = PORT_OFFSETS[PROBE_PORT_TYPE]

    # Phase thresholds (mirror build_conditioned_dataset.py defaults).
    D_DESCEND_THRESH = 0.10
    D_INSERTED_THRESH = 0.005
    F_CONTACT_THRESH = 8.0

    def make_state(tcp_z: float) -> torch.Tensor:
        # Baseline 26-dim layout:
        #   [0:7]   tcp_pose (xyz + xyzw quat)
        #   [7:13]  tcp_velocity
        #   [13:20] joint positions
        #   [20:26] wrench (fx,fy,fz, tx,ty,tz)
        s = np.zeros(state_dim, dtype=np.float32)
        s[3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quat
        s[2] = tcp_z
        if state_dim == 38:
            # Conditioned layout extras:
            #   [26:33] prev_action (zero-pad; matches dataset's first-frame convention)
            #   [33]    tcp.depth_above_entrance = -tcp_z - entrance_offset
            #   [34:38] phase one-hot (approach, descend, contact, inserted)
            depth = -tcp_z - ENTRANCE_OFFSET
            s[33] = depth
            # classify phase
            fmag = 0.0  # zero wrench in probe
            if fmag >= F_CONTACT_THRESH:
                phase_idx = 2  # contact
            elif depth <= D_INSERTED_THRESH:
                phase_idx = 3  # inserted
            elif depth <= D_DESCEND_THRESH:
                phase_idx = 1  # descend
            else:
                phase_idx = 0  # approach
            s[34 + phase_idx] = 1.0
        return torch.from_numpy(s).unsqueeze(0).cuda()

    dummy_imgs = {
        "observation.images.left_camera":   torch.zeros(1, 3, 256, 288, device="cuda"),
        "observation.images.center_camera": torch.zeros(1, 3, 256, 288, device="cuda"),
        "observation.images.right_camera":  torch.zeros(1, 3, 256, 288, device="cuda"),
        "task": f"insert {PROBE_PORT_TYPE} plug into sc_port_base on sc_port_0",
    }

    print(f"checkpoint: {ckpt}")
    print(f"chunk_size: {config.chunk_size}  n_action_steps: {config.n_action_steps}")
    print(f"state_dim:  {state_dim}  ({'CONDITIONED' if state_dim == 38 else 'baseline'})")
    print(f"probe port: {PROBE_PORT_TYPE}  (entrance offset = {ENTRANCE_OFFSET:.5f} m)")
    print()
    header = f"{'obs.tcp_z':>11} | {'raw_z[0]':>9} {'raw_z[-1]':>10} {'raw_z[min]':>11} {'raw_z[max]':>11} | {'un_z[0]':>9} {'un_z[-1]':>10}"
    print(header)
    print("-" * len(header))

    # Read stats from postprocessor and detect normalization mode for ACTION.
    post_json = ckpt / "policy_postprocessor.json"
    post_safetensors = next(
        ckpt.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"),
        None,
    )
    action_mode = "MEAN_STD"  # fallback
    if post_json.exists():
        post_cfg = json.loads(post_json.read_text())
        for step in post_cfg.get("steps", []):
            norm_map = step.get("config", {}).get("norm_map") if step.get("config") else None
            if norm_map and "ACTION" in norm_map:
                action_mode = norm_map["ACTION"]
                break
    if post_safetensors is not None:
        from safetensors import safe_open
        with safe_open(str(post_safetensors), framework="numpy") as f:
            am = f.get_tensor("action.mean")
            asd = f.get_tensor("action.std")
            try:
                amin = f.get_tensor("action.min")
                amax = f.get_tensor("action.max")
            except Exception:
                amin = amax = None
    else:
        am = np.full((7,), -0.1893, dtype=np.float32)
        asd = np.full((7,), 0.10692, dtype=np.float32)
        amin = amax = None

    if action_mode == "MIN_MAX" and amin is not None and amax is not None:
        mid = (float(amin[2]) + float(amax[2])) / 2.0
        half = (float(amax[2]) - float(amin[2])) / 2.0
        unnorm_fn = lambda raw: mid + raw * half
        print(f"  ACTION normalization = MIN_MAX")
        print(f"  un-norm: action_z = {mid:+.5f} + raw * {half:+.5f}  "
              f"(min={float(amin[2]):+.5f}, max={float(amax[2]):+.5f})")
    else:
        m, s = float(am[2]), float(asd[2])
        unnorm_fn = lambda raw: m + raw * s
        print(f"  ACTION normalization = {action_mode}")
        print(f"  un-norm: action_z = {m:+.5f} + raw * {s:+.5f}")

    for tcp_z in (-0.30, -0.20, -0.10, -0.05, -0.01, 0.00):
        raw_obs = dict(dummy_imgs)
        raw_obs["observation.state"] = make_state(tcp_z)
        obs = pre(raw_obs)
        with torch.no_grad():
            actions = policy.predict_action_chunk(obs)  # (1, chunk_size, A_padded)
        az = actions[0, :, 2].cpu().numpy()
        un_first = unnorm_fn(float(az[0]))
        un_last = unnorm_fn(float(az[-1]))
        print(
            f"{tcp_z:+11.3f} | {az[0]:+9.3f} {az[-1]:+10.3f} "
            f"{az.min():+11.3f} {az.max():+11.3f} | "
            f"{un_first:+9.4f} {un_last:+10.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
