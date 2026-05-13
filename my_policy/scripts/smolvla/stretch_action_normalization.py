#!/usr/bin/env python3
"""Stretch (or shift) the action[z] un-normalization range in a trained
SmolVLA checkpoint to compensate for the model's under-extension.

The flow-matching denoiser typically saturates short of the demo's true
output range (raw ≈ +0.25 instead of +1.0 in MIN_MAX mode). Stretching
the un-normalization range by 1/saturation_factor maps the saturated
raw value to where +1.0 would have un-normalized — making the model's
output cover the full demo range without retraining.

Autodetects normalization mode from policy_postprocessor.json:
  * MEAN_STD → patches action.std (multiplies by --stretch-factor)
  * MIN_MAX  → patches action.min / action.max symmetrically around
    their midpoint (the half-range multiplied by --stretch-factor)

The midpoint (mean for MEAN_STD, mid for MIN_MAX) stays fixed so the
model's "neutral" output (raw=0) still maps to the same physical action.

Backs up the postprocessor safetensors as <filename>.bak.

Usage:
    pixi run python my_policy/scripts/smolvla/stretch_action_normalization.py \\
        --checkpoint /root/aic_data/v9_act_build/runs/.../checkpoints/100000/pretrained_model \\
        --stretch-factor 4.0

Revert:
    pixi run python my_policy/scripts/smolvla/stretch_action_normalization.py \\
        --checkpoint .../pretrained_model --revert
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


ACTION_Z_IDX = 2


def find_postprocessor(ckpt: Path) -> tuple[Path, str]:
    """Returns (safetensors path, action_norm_mode)."""
    matches = list(ckpt.glob(
        "policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
    if not matches:
        raise FileNotFoundError(
            f"no postprocessor safetensors in {ckpt}")
    if len(matches) > 1:
        raise RuntimeError(f"expected 1 postprocessor file, got: {matches}")
    safetensors_path = matches[0]

    # Detect mode from policy_postprocessor.json.
    json_path = ckpt / "policy_postprocessor.json"
    mode = "MEAN_STD"  # default
    if json_path.exists():
        cfg = json.loads(json_path.read_text())
        for step in cfg.get("steps", []):
            nm = step.get("config", {}).get("norm_map") if step.get("config") else None
            if nm and "ACTION" in nm:
                mode = nm["ACTION"]
                break
    return safetensors_path, mode


def load_all_tensors(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to checkpoints/<step>/pretrained_model/.")
    p.add_argument("--stretch-factor", type=float, default=4.0,
                   help="Multiply the half-range (MIN_MAX) or std (MEAN_STD) "
                        "of action[z] by this factor. 4.0 maps raw=+0.25 "
                        "to where raw=+1.0 was originally. Default 4.0.")
    p.add_argument("--revert", action="store_true",
                   help="Restore from .bak.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing .bak.")
    args = p.parse_args()

    if not args.checkpoint.is_dir():
        raise SystemExit(f"not a directory: {args.checkpoint}")

    post_path, mode = find_postprocessor(args.checkpoint)
    bak_path = post_path.with_suffix(post_path.suffix + ".bak")
    print(f"postprocessor : {post_path}")
    print(f"backup        : {bak_path}")
    print(f"ACTION mode   : {mode}")

    if args.revert:
        if not bak_path.exists():
            raise SystemExit(f"no backup at {bak_path}")
        shutil.copy2(bak_path, post_path)
        print(f"\nREVERTED. Backup still at {bak_path}.")
        return 0

    if args.stretch_factor <= 0:
        raise SystemExit("--stretch-factor must be > 0")

    tensors = load_all_tensors(post_path)

    if mode == "MEAN_STD":
        if "action.mean" not in tensors or "action.std" not in tensors:
            raise SystemExit("missing action.mean / action.std")
        m_old = float(tensors["action.mean"][ACTION_Z_IDX])
        s_old = float(tensors["action.std"][ACTION_Z_IDX])
        s_new = s_old * args.stretch_factor
        print()
        print(f"BEFORE: mean={m_old:+.5f}  std={s_old:+.5f}")
        print(f"  raw=-1.0 unnorms to {m_old - s_old:+.5f}")
        print(f"  raw=+1.0 unnorms to {m_old + s_old:+.5f}")
        print(f"AFTER:  mean={m_old:+.5f}  std={s_new:+.5f}  (×{args.stretch_factor:.2f})")
        print(f"  raw=-1.0 unnorms to {m_old - s_new:+.5f}")
        print(f"  raw=+1.0 unnorms to {m_old + s_new:+.5f}")
        print(f"  raw=+0.25 unnorms to {m_old + 0.25 * s_new:+.5f}")
        new_tensors = dict(tensors)
        new_std = tensors["action.std"].copy()
        new_std[ACTION_Z_IDX] = np.float32(s_new)
        new_tensors["action.std"] = new_std

    elif mode == "MIN_MAX":
        if "action.min" not in tensors or "action.max" not in tensors:
            raise SystemExit("missing action.min / action.max")
        min_old = float(tensors["action.min"][ACTION_Z_IDX])
        max_old = float(tensors["action.max"][ACTION_Z_IDX])
        mid = (min_old + max_old) / 2.0
        half_old = (max_old - min_old) / 2.0
        half_new = half_old * args.stretch_factor
        min_new = mid - half_new
        max_new = mid + half_new
        print()
        print(f"BEFORE: min={min_old:+.5f}  max={max_old:+.5f}")
        print(f"  mid={mid:+.5f}  half={half_old:+.5f}")
        print(f"  raw=-1.0 unnorms to {min_old:+.5f}")
        print(f"  raw=+1.0 unnorms to {max_old:+.5f}")
        print(f"AFTER:  min={min_new:+.5f}  max={max_new:+.5f}  (half × {args.stretch_factor:.2f})")
        print(f"  raw=-1.0 unnorms to {min_new:+.5f}")
        print(f"  raw=+1.0 unnorms to {max_new:+.5f}")
        print(f"  raw=+0.25 unnorms to {mid + 0.25 * half_new:+.5f}")
        new_tensors = dict(tensors)
        new_min = tensors["action.min"].copy()
        new_max = tensors["action.max"].copy()
        new_min[ACTION_Z_IDX] = np.float32(min_new)
        new_max[ACTION_Z_IDX] = np.float32(max_new)
        new_tensors["action.min"] = new_min
        new_tensors["action.max"] = new_max

    else:
        raise SystemExit(f"unsupported normalization mode: {mode}")

    print(f"  (other action dims and all state dims unchanged)")

    if bak_path.exists() and not args.force:
        raise SystemExit(
            f"backup already exists at {bak_path} — pass --force to overwrite, "
            f"or restore first with --revert.")
    print()
    print(f"writing backup  -> {bak_path}")
    shutil.copy2(post_path, bak_path)
    print(f"writing patched -> {post_path}")
    save_file(new_tensors, str(post_path))
    print()
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
