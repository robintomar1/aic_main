#!/usr/bin/env python3
"""Verify the F/T tare compensation fix on a rosbag2 recording.

Reads /fts_broadcaster/wrench and /aic_controller/controller_state from an
MCAP rosbag2, aligns them by header stamp (nearest-neighbor), computes
compensated = raw - tare_offset, prints per-axis stats, and writes a PNG
showing raw / tare / compensated Fz over time.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import rosbag2_py  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402


DEFAULT_BAG = "/root/ws_aic/src/aic/bags/cheatcode_ref"
RAW_TOPIC = "/fts_broadcaster/wrench"
CSTATE_TOPIC = "/aic_controller/controller_state"


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def read_bag(bag_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw_t, raw_f[N,3], tare_t, tare_f[M,3]) arrays."""
    storage = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    conv = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, conv)

    topics_and_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for needed in (RAW_TOPIC, CSTATE_TOPIC):
        if needed not in topics_and_types:
            raise SystemExit(f"Topic {needed} not found in bag. Topics: {list(topics_and_types)}")

    raw_cls = get_message(topics_and_types[RAW_TOPIC])
    cstate_cls = get_message(topics_and_types[CSTATE_TOPIC])

    f = rosbag2_py.StorageFilter(topics=[RAW_TOPIC, CSTATE_TOPIC])
    reader.set_filter(f)

    raw_t: List[float] = []
    raw_f: List[Tuple[float, float, float]] = []
    tare_t: List[float] = []
    tare_f: List[Tuple[float, float, float]] = []

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == RAW_TOPIC:
            msg = deserialize_message(data, raw_cls)
            t = _stamp_to_sec(msg.header.stamp)
            w = msg.wrench.force
            raw_t.append(t)
            raw_f.append((w.x, w.y, w.z))
        elif topic == CSTATE_TOPIC:
            msg = deserialize_message(data, cstate_cls)
            ws = msg.fts_tare_offset
            t = _stamp_to_sec(ws.header.stamp)
            if t <= 0.0:
                t = _stamp_to_sec(msg.header.stamp)
            w = ws.wrench.force
            tare_t.append(t)
            tare_f.append((w.x, w.y, w.z))

    return (
        np.asarray(raw_t, dtype=np.float64),
        np.asarray(raw_f, dtype=np.float64),
        np.asarray(tare_t, dtype=np.float64),
        np.asarray(tare_f, dtype=np.float64),
    )


def nearest_neighbor(src_t: np.ndarray, src_v: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """For each query time, return src_v at the closest src_t (1D or ND)."""
    if src_t.size == 0:
        raise ValueError("Empty source array")
    order = np.argsort(src_t)
    st = src_t[order]
    sv = src_v[order]
    idx = np.searchsorted(st, query_t)
    idx = np.clip(idx, 0, st.size - 1)
    left = np.clip(idx - 1, 0, st.size - 1)
    choose_left = (idx == st.size) | (
        (idx > 0) & (np.abs(st[left] - query_t) < np.abs(st[idx] - query_t))
    )
    picked = np.where(choose_left, left, idx)
    return sv[picked]


def stats_line(label: str, arr: np.ndarray) -> str:
    if arr.size == 0:
        return f"  {label:>18s}:      (empty)"
    return f"  {label:>18s}:  mean = {arr.mean():+8.4f} N   std = {arr.std():7.4f} N   n = {arr.size}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=DEFAULT_BAG, help="Path to rosbag2 directory")
    ap.add_argument("--out", default="/tmp/ft_tare_check.png", help="Output PNG path")
    args = ap.parse_args()

    bag = args.bag
    if not os.path.isdir(bag):
        print(f"ERROR: bag dir not found: {bag}", file=sys.stderr)
        return 2

    print(f"Reading bag: {bag}")
    raw_t, raw_f, tare_t, tare_f = read_bag(bag)
    print(f"  raw wrench msgs:           {raw_t.size}")
    print(f"  controller_state msgs:     {tare_t.size}")

    if raw_t.size == 0 or tare_t.size == 0:
        print("ERROR: at least one topic had zero messages.", file=sys.stderr)
        return 3

    # Align tare values to each raw stamp via nearest-neighbor on t.
    tare_aligned = nearest_neighbor(tare_t, tare_f, raw_t)
    compensated = raw_f - tare_aligned

    # Normalize to bag-relative time for the plot.
    t0 = raw_t.min()
    rel_t = raw_t - t0

    print()
    print("Per-axis stats over the full bag:")
    for axis, i in (("Fx", 0), ("Fy", 1), ("Fz", 2)):
        print(f" [{axis}]")
        print(stats_line(f"raw {axis}", raw_f[:, i]))
        print(stats_line(f"tare {axis}", tare_aligned[:, i]))
        print(stats_line(f"compensated {axis}", compensated[:, i]))

    # Plot Fz raw / tare / compensated.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rel_t, raw_f[:, 2], lw=0.8, label="raw Fz", color="tab:blue")
    ax.plot(rel_t, tare_aligned[:, 2], lw=0.8, label="tare offset Fz", color="tab:orange")
    ax.plot(rel_t, compensated[:, 2], lw=0.8, label="compensated Fz (raw - tare)", color="tab:green")
    ax.axhline(0.0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("time since bag start (s)")
    ax.set_ylabel("Fz (N)")
    ax.set_title(f"F/T tare compensation check\n{os.path.basename(os.path.normpath(bag))}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print()
    print(f"Plot saved to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
