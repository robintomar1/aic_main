#!/usr/bin/env python3
"""Demo — pick random frames from a raw recorder batch, project the
recorded port pose into each camera, render side-by-side PNGs showing:

    [original 288×256 with red dot at projected port pixel + green crop box]
    [cropped CROP×CROP region with cyan crosshair at center]

…stacked vertically for left/center/right cameras (3 rows per PNG).
Open the PNGs to visually confirm:

  1. The red dot lands on the port hole (not the plug, not a distractor) —
     validates the SE(3) projection chain is correct end-to-end.
  2. The port is near the center of each cropped image regardless of
     board yaw, board XY, or which mount the port is on — validates the
     port-canonicalized image is what the model would see.
  3. The crop box doesn't fall outside the image for typical frames.

This is the visual companion to the Tier 1/2 numerical tests; together
they cover both "the math is right" and "the math is right ABOUT WHAT WE
THINK IT'S RIGHT ABOUT."

Reads from a RAW batch (e.g. /root/aic_data/batch_100_a/) because that's
where `groundtruth.port_pose` lives (the port-local datasets drop it as
a leakage channel). Random sampling is deterministic via --seed.

Usage:
    pixi run python my_policy/scripts/demo_port_crop.py \\
        --raw-batch /root/aic_data/batch_100_a \\
        --out-dir /root/aic_data/v9_act_build/port_crop_demo \\
        --n-frames 12 \\
        --crop-size 224
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.localizer.projection import (  # noqa: E402
    LEROBOT_CAM_TO_SHORT,
    compute_static_tcp_to_camera_optical,
    project_port_to_pixels,
)
from my_policy.port_local.dataset_io import (  # noqa: E402
    SRC_PORT_POSE_SLICE,
    SRC_TCP_POSE_SLICE,
    is_port_pose_valid,
)


# Dataset image resolution — verified 2026-05-08 from
# /root/aic_data/batch_100_a/meta/info.json:
#   observation.images.{cam}: shape=[256, 288, 3] dtype=video
# That's (H, W, C). Native camera is 1152W × 1024H per
# basler_camera_macro.xacro, so scale = 0.25 from native to dataset.
DATASET_W, DATASET_H = 288, 256
NATIVE_W, NATIVE_H = 1152, 1024
PIXEL_SCALE = DATASET_W / NATIVE_W  # 0.25; Y scale is identical (square pixels)

CAMERA_KEYS = (
    "observation.images.left_camera",
    "observation.images.center_camera",
    "observation.images.right_camera",
)

# Drawing constants.
DOT_RADIUS = 4              # red dot for port pixel on original
CROSSHAIR_HALF = 8          # cyan crosshair on cropped image
BOX_LINE_WIDTH = 2
DOT_COLOR_OK = (255, 0, 0)        # red — port pixel in-frame
DOT_COLOR_OFF = (255, 128, 0)     # orange — port pixel outside image bounds
BOX_COLOR = (0, 220, 0)           # green — crop box
BOX_COLOR_OFF = (220, 220, 0)     # yellow — crop box was clamped at edge
CROSSHAIR_COLOR = (0, 220, 220)   # cyan — center of crop
TEXT_COLOR = (255, 255, 0)


def _tensor_to_uint8(img_tensor) -> np.ndarray:
    """LeRobot v3.0 `observation.images.X` is a float32 [3, H, W] tensor in
    [0, 1]. Convert to uint8 [H, W, 3] for PIL.
    """
    arr = img_tensor.numpy() if hasattr(img_tensor, "numpy") else np.asarray(img_tensor)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"unexpected image shape {arr.shape}; "
                         f"expected (3, H, W)")
    arr = np.transpose(arr, (1, 2, 0))           # → (H, W, 3)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _crop_with_padding(
    img_hwc: np.ndarray, center_xy: tuple[float, float], crop_size: int,
) -> tuple[np.ndarray, bool]:
    """Crop a `crop_size × crop_size` region centered on `center_xy` from
    `img_hwc`. Out-of-bounds regions are filled with black. Returns
    (crop, was_clamped) where `was_clamped` is True if any of the requested
    region fell outside the source image.
    """
    H, W = img_hwc.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half = crop_size // 2

    # Requested region in source image coords (may be outside image).
    src_x0 = int(round(cx)) - half
    src_x1 = src_x0 + crop_size
    src_y0 = int(round(cy)) - half
    src_y1 = src_y0 + crop_size

    # Clip to source.
    clip_x0 = max(0, src_x0)
    clip_x1 = min(W, src_x1)
    clip_y0 = max(0, src_y0)
    clip_y1 = min(H, src_y1)
    was_clamped = (
        src_x0 < 0 or src_x1 > W or src_y0 < 0 or src_y1 > H
    )

    out = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    if clip_x1 > clip_x0 and clip_y1 > clip_y0:
        # Where the visible region lands in the output crop.
        out_x0 = clip_x0 - src_x0
        out_x1 = out_x0 + (clip_x1 - clip_x0)
        out_y0 = clip_y0 - src_y0
        out_y1 = out_y0 + (clip_y1 - clip_y0)
        out[out_y0:out_y1, out_x0:out_x1] = (
            img_hwc[clip_y0:clip_y1, clip_x0:clip_x1]
        )
    return out, was_clamped


def _draw_overlay_on_original(
    img_hwc: np.ndarray, port_xy: tuple[float, float], crop_size: int,
    cam_name: str, depth_m: float,
) -> Image.Image:
    """Render the original 288×256 image with red dot + green crop box."""
    H, W = img_hwc.shape[:2]
    pim = Image.fromarray(img_hwc.copy()).convert("RGB")
    draw = ImageDraw.Draw(pim)
    cx, cy = port_xy
    # Port-pixel marker. Color depends on whether the projection landed
    # inside the visible image (in front of camera + within bounds).
    in_bounds = (depth_m > 0 and 0 <= cx < W and 0 <= cy < H)
    dot_color = DOT_COLOR_OK if in_bounds else DOT_COLOR_OFF
    draw.ellipse(
        [cx - DOT_RADIUS, cy - DOT_RADIUS, cx + DOT_RADIUS, cy + DOT_RADIUS],
        outline=dot_color, fill=dot_color,
    )
    # Crop box.
    half = crop_size // 2
    box_x0, box_y0 = cx - half, cy - half
    box_x1, box_y1 = cx + half, cy + half
    box_color = (
        BOX_COLOR_OFF
        if (box_x0 < 0 or box_y0 < 0 or box_x1 > W or box_y1 > H)
        else BOX_COLOR
    )
    draw.rectangle([box_x0, box_y0, box_x1, box_y1],
                   outline=box_color, width=BOX_LINE_WIDTH)
    draw.text((4, 4), f"{cam_name}  z={depth_m:.2f}m  uv=({cx:.0f},{cy:.0f})",
              fill=TEXT_COLOR)
    return pim


def _draw_crosshair_on_crop(crop_hwc: np.ndarray, was_clamped: bool) -> Image.Image:
    """Cyan crosshair at the geometric center of the cropped image, plus a
    label warning if the source crop was clamped at an image boundary."""
    pim = Image.fromarray(crop_hwc.copy()).convert("RGB")
    draw = ImageDraw.Draw(pim)
    H, W = crop_hwc.shape[:2]
    cx, cy = W // 2, H // 2
    draw.line([(cx - CROSSHAIR_HALF, cy), (cx + CROSSHAIR_HALF, cy)],
              fill=CROSSHAIR_COLOR, width=2)
    draw.line([(cx, cy - CROSSHAIR_HALF), (cx, cy + CROSSHAIR_HALF)],
              fill=CROSSHAIR_COLOR, width=2)
    if was_clamped:
        draw.text((4, 4), "CLAMPED", fill=DOT_COLOR_OFF)
    return pim


def _stack_row(orig: Image.Image, crop: Image.Image) -> Image.Image:
    """Side-by-side: original (left) | crop (right). Same height, width sum."""
    h = max(orig.height, crop.height)
    w = orig.width + crop.width + 8  # 8 px gap
    out = Image.new("RGB", (w, h), (32, 32, 32))
    out.paste(orig, (0, (h - orig.height) // 2))
    out.paste(crop, (orig.width + 8, (h - crop.height) // 2))
    return out


def _stack_cols(panels: list[Image.Image]) -> Image.Image:
    """Stack rows vertically (one per camera)."""
    w = max(p.width for p in panels)
    h_total = sum(p.height for p in panels) + 8 * (len(panels) - 1)
    out = Image.new("RGB", (w, h_total), (32, 32, 32))
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height + 8
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw-batch", type=Path, required=True,
                   help="Path to a raw recorder batch dir (e.g. "
                        "/root/aic_data/batch_100_a). Must have "
                        "`groundtruth.port_pose` in observation.state.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write the demo PNGs.")
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--crop-size", type=int, default=224,
                   help="Crop side length in DATASET pixels (default 224; "
                        "max usable is min(H,W)=256 since dataset is 256×288).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load dataset (uses lerobot's video decoder) -----------------
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq

    print(f"loading {args.raw_batch}")
    ds = LeRobotDataset(
        repo_id="local/demo_port_crop",
        root=str(args.raw_batch),
        video_backend="pyav",
    )
    print(f"  {len(ds)} frames")

    # Per-frame raw state (47-dim) is in the parquet — pull port_pose +
    # tcp_pose + episode_index directly to avoid pulling images for
    # frames we won't use.
    table = pq.read_table(
        str(args.raw_batch / "data" / "chunk-000" / "file-000.parquet"),
        columns=["observation.state", "episode_index", "frame_index"],
    )
    states = np.stack(table["observation.state"].to_pylist()).astype(np.float32)
    eps = np.array(table["episode_index"].to_pylist(), dtype=np.int64)
    frame_idx = np.array(table["frame_index"].to_pylist(), dtype=np.int64)

    # --- 2. Compute static T_cam_in_tcp from any frame-0 (HOME) ---------
    # Per V6 verification on batch_100_a (frame-0 TCP std < 1mm across 75
    # episodes), the recorder DOES reset to HOME between trials, so any
    # frame-0 TCP is fine. Pick episode 0's frame 0 by convention.
    home_mask = (eps == 0) & (frame_idx == 0)
    if home_mask.sum() == 0:
        print("FATAL: no frame_index=0 in episode 0; using global index 0",
              file=sys.stderr)
        home_idx = 0
    else:
        home_idx = int(np.argmax(home_mask))
    home_tcp = states[home_idx, SRC_TCP_POSE_SLICE]
    print(f"  HOME TCP (ep 0 frame 0): xyz={home_tcp[:3]} q={home_tcp[3:7]}")
    static_extr = compute_static_tcp_to_camera_optical(
        home_tcp[:3].astype(np.float64), home_tcp[3:7].astype(np.float64),
    )
    for sn in ("left", "center", "right"):
        if sn not in static_extr:
            sys.exit(f"compute_static_tcp_to_camera_optical missing camera '{sn}'")

    # --- 3. Sample N valid frames ---------------------------------------
    rng = np.random.default_rng(args.seed)
    valid_global = np.where(
        np.array([is_port_pose_valid(states[i, SRC_PORT_POSE_SLICE])
                  for i in range(len(states))])
    )[0]
    if len(valid_global) < args.n_frames:
        sys.exit(f"only {len(valid_global)} valid frames; need {args.n_frames}")
    pick = rng.choice(valid_global, size=args.n_frames, replace=False)
    pick = sorted(int(x) for x in pick)
    print(f"  picked {len(pick)} random frames (seed={args.seed}): {pick[:5]}...")

    # --- 4. Render each frame -------------------------------------------
    n_in_frame_total = 0
    n_clamped_total = 0
    for n_done, gi in enumerate(pick, start=1):
        ep = int(eps[gi])
        f_off = int(frame_idx[gi])
        tcp = states[gi, SRC_TCP_POSE_SLICE]
        port = states[gi, SRC_PORT_POSE_SLICE]

        pixels_native = project_port_to_pixels(
            port_baselink=port[:3].astype(np.float64),
            tcp_baselink_xyz=tcp[:3].astype(np.float64),
            tcp_baselink_quat_xyzw=tcp[3:7].astype(np.float64),
            static_tcp_to_camera_optical=static_extr,
        )

        item = ds[gi]
        rows: list[Image.Image] = []
        for lerobot_key, short in LEROBOT_CAM_TO_SHORT.items():
            if lerobot_key not in item:
                # Some keys in the localizer module use shortened names; the
                # raw recorder stores fully-qualified `observation.images.X`.
                # The map is { "left_camera": "left", ... } so build the
                # full key here.
                full = f"observation.images.{lerobot_key}"
            else:
                full = lerobot_key
            if full not in item:
                sys.exit(f"missing camera '{full}' in dataset item")
            img_uint8 = _tensor_to_uint8(item[full])
            u_native, v_native, depth = pixels_native[short]
            # Scale native pixels → dataset pixels (×0.25).
            if np.isnan(u_native) or np.isnan(v_native):
                u_ds, v_ds = -100.0, -100.0  # off-image marker
            else:
                u_ds = u_native * PIXEL_SCALE
                v_ds = v_native * PIXEL_SCALE

            crop, clamped = _crop_with_padding(
                img_uint8, (u_ds, v_ds), args.crop_size,
            )
            in_frame = (
                depth > 0
                and 0 <= u_ds < DATASET_W
                and 0 <= v_ds < DATASET_H
            )
            if in_frame:
                n_in_frame_total += 1
            if clamped:
                n_clamped_total += 1
            orig_panel = _draw_overlay_on_original(
                img_uint8, (u_ds, v_ds), args.crop_size, short, depth,
            )
            crop_panel = _draw_crosshair_on_crop(crop, clamped)
            rows.append(_stack_row(orig_panel, crop_panel))

        out_img = _stack_cols(rows)
        out_path = args.out_dir / f"frame_{n_done:03d}_ep{ep:03d}_off{f_off:04d}.png"
        out_img.save(out_path)
        print(f"  [{n_done}/{len(pick)}] {out_path.name}")

    n_total = len(pick) * 3  # 3 cameras
    print()
    print(f"Done. Wrote {len(pick)} PNGs to {args.out_dir}")
    print(f"  per-camera projection in-frame:  {n_in_frame_total}/{n_total} "
          f"({100*n_in_frame_total/n_total:.0f}%)")
    print(f"  per-camera crop clamped at edge: {n_clamped_total}/{n_total} "
          f"({100*n_clamped_total/n_total:.0f}%)")
    print()
    print("Visual check:")
    print("  1. Red dot lands on the port hole (not plug, not distractor).")
    print("  2. Port near center of cropped image regardless of board pose.")
    print("  3. Cropped images look similar across episodes despite raw-image variance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
