#!/usr/bin/env python3
"""Patch the action[z] normalization stats in an existing checkpoint so
the model's saturated raw output (~+1.0) un-normalizes to the demo's
actual endpoint instead of hovering 7-8 cm above it.

Mechanism:
  - Loads policy_postprocessor_step_*_unnormalizer_processor.safetensors.
  - Replaces action.mean[2] and action.std[2] with values derived from
    the demo's [min, max] action range so the demo's full span maps to
    normalized [-1, +1] (MIN_MAX-style).
  - Backs up the original file as <filename>.bak.
  - Leaves all other dims, state stats, and preprocessor untouched.

Inference-side test only — no retraining. To revert, restore the .bak.

Usage (inside pixi):
    pixi run python my_policy/scripts/smolvla/patch_action_normalization.py \
        --checkpoint /root/aic_data/sc_build/runs/sc_2x5x5_smolvla_pl_chunk200/checkpoints/last/pretrained_model

Or with custom new mean/std:
    ... --new-mean -0.15646 --new-std 0.15307

Run with --revert to restore from .bak.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


ACTION_Z_IDX = 2


def find_postprocessor(ckpt: Path) -> Path:
    matches = list(ckpt.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
    if not matches:
        raise FileNotFoundError(
            f"no policy_postprocessor_step_*_unnormalizer_processor.safetensors in {ckpt}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"expected exactly 1 postprocessor file, got: {matches}")
    return matches[0]


def load_all_tensors(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a checkpoints/<step>/pretrained_model/ directory.")
    p.add_argument(
        "--new-mean", type=float, default=None,
        help="New action.mean[z]. Default: derived from min/max as (min+max)/2.",
    )
    p.add_argument(
        "--new-std", type=float, default=None,
        help="New action.std[z]. Default: derived from min/max as (max-min)/2.",
    )
    p.add_argument(
        "--revert", action="store_true",
        help="Restore the postprocessor from .bak (undo a previous patch).",
    )
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing .bak if a previous patch already ran.")
    args = p.parse_args()

    ckpt = args.checkpoint
    if not ckpt.is_dir():
        raise SystemExit(f"not a directory: {ckpt}")

    post_path = find_postprocessor(ckpt)
    bak_path = post_path.with_suffix(post_path.suffix + ".bak")
    print(f"postprocessor file : {post_path}")
    print(f"backup file        : {bak_path}")

    if args.revert:
        if not bak_path.exists():
            raise SystemExit(f"nothing to revert — no backup at {bak_path}")
        shutil.copy2(bak_path, post_path)
        print(f"\nREVERTED. Backup left in place at {bak_path}.")
        return 0

    tensors = load_all_tensors(post_path)
    if "action.mean" not in tensors or "action.std" not in tensors:
        raise SystemExit("action.mean / action.std not found in postprocessor file")
    if "action.min" not in tensors or "action.max" not in tensors:
        raise SystemExit("action.min / action.max not found — cannot derive new stats")

    a_min = tensors["action.min"]
    a_max = tensors["action.max"]
    a_mean_orig = tensors["action.mean"].copy()
    a_std_orig = tensors["action.std"].copy()

    print()
    print("ACTION[z] BEFORE:")
    print(f"  min   = {float(a_min[ACTION_Z_IDX]):+.5f}")
    print(f"  max   = {float(a_max[ACTION_Z_IDX]):+.5f}")
    print(f"  mean  = {float(a_mean_orig[ACTION_Z_IDX]):+.5f}")
    print(f"  std   = {float(a_std_orig[ACTION_Z_IDX]):+.5f}")
    print(f"  → raw=+1.0 un-norms to "
          f"{float(a_mean_orig[ACTION_Z_IDX] + a_std_orig[ACTION_Z_IDX]):+.5f}")
    print(f"  → raw=-1.0 un-norms to "
          f"{float(a_mean_orig[ACTION_Z_IDX] - a_std_orig[ACTION_Z_IDX]):+.5f}")

    if args.new_mean is None:
        new_mean = (float(a_min[ACTION_Z_IDX]) + float(a_max[ACTION_Z_IDX])) / 2.0
    else:
        new_mean = args.new_mean
    if args.new_std is None:
        new_std = (float(a_max[ACTION_Z_IDX]) - float(a_min[ACTION_Z_IDX])) / 2.0
    else:
        new_std = args.new_std

    if new_std <= 0:
        raise SystemExit(f"new_std must be > 0, got {new_std}")

    a_mean_new = a_mean_orig.copy()
    a_std_new = a_std_orig.copy()
    a_mean_new[ACTION_Z_IDX] = np.float32(new_mean)
    a_std_new[ACTION_Z_IDX] = np.float32(new_std)

    print()
    print("ACTION[z] AFTER:")
    print(f"  mean  = {float(a_mean_new[ACTION_Z_IDX]):+.5f}")
    print(f"  std   = {float(a_std_new[ACTION_Z_IDX]):+.5f}")
    print(f"  → raw=+1.0 un-norms to "
          f"{float(a_mean_new[ACTION_Z_IDX] + a_std_new[ACTION_Z_IDX]):+.5f}")
    print(f"  → raw=-1.0 un-norms to "
          f"{float(a_mean_new[ACTION_Z_IDX] - a_std_new[ACTION_Z_IDX]):+.5f}")
    print(f"  → raw=+0.92 un-norms to "
          f"{float(a_mean_new[ACTION_Z_IDX] + 0.92 * a_std_new[ACTION_Z_IDX]):+.5f}")
    print(f"  (other action dims and all state dims unchanged)")

    if bak_path.exists() and not args.force:
        raise SystemExit(
            f"backup already exists at {bak_path} — pass --force to overwrite, "
            f"or restore first with --revert."
        )

    print()
    print(f"writing backup     -> {bak_path}")
    shutil.copy2(post_path, bak_path)

    new_tensors = dict(tensors)
    new_tensors["action.mean"] = a_mean_new
    new_tensors["action.std"] = a_std_new

    print(f"writing patched    -> {post_path}")
    save_file(new_tensors, str(post_path))

    print()
    print("DONE. To revert:")
    print(f"  pixi run python my_policy/scripts/smolvla/patch_action_normalization.py "
          f"--checkpoint {ckpt} --revert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
