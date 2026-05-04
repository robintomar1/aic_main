#!/usr/bin/env python3
"""Dump per-frame action z (and x,y) for a few episodes to see whether
the recorded ACT supervision signal actually descends into the port.

If the action z stays high throughout the demo, the model learned what
the data showed (no descent) — fix is at the recorder side.
If the action z descends but the model output stays high, it's a model
training failure — fix is in training.

Run:
    pixi run python my_policy/scripts/inspect_action_traces.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/root/aic_data/v9_act_build/v9_act_merged"))
    p.add_argument("--n-episodes", type=int, default=5)
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/inspect", root=str(args.dataset_root),
                        video_backend="pyav")
    print(f"dataset: {len(ds)} frames across {ds.num_episodes} episodes")

    for ep_i in range(min(args.n_episodes, ds.num_episodes)):
        ep_from = int(ds.episode_data_index["from"][ep_i])
        ep_to = int(ds.episode_data_index["to"][ep_i])
        ep_len = ep_to - ep_from
        # Pull action and tcp position at start, mid, end of episode.
        sample_idx = [ep_from, ep_from + ep_len // 4, ep_from + ep_len // 2,
                      ep_from + 3 * ep_len // 4, ep_to - 1]
        print(f"\nepisode {ep_i}: len={ep_len} frames ({ep_len/20:.1f}s @ 20Hz)")
        print(f"  frame  action(x,y,z)              tcp(x,y,z)")
        for i in sample_idx:
            item = ds[i]
            a = item["action"].numpy()
            s = item["observation.state"].numpy()
            ax, ay, az = a[0], a[1], a[2]
            sx, sy, sz = s[0], s[1], s[2]
            print(f"  {i-ep_from:5d}  ({ax:+.3f},{ay:+.3f},{az:+.3f})  "
                  f"({sx:+.3f},{sy:+.3f},{sz:+.3f})")

        # Also: action_z range across the whole episode
        # Pull only every 10th frame to keep it cheap
        zs = []
        for i in range(ep_from, ep_to, 10):
            zs.append(ds[i]["action"].numpy()[2])
        zs = np.array(zs)
        print(f"  action_z over episode: min={zs.min():.3f} max={zs.max():.3f} "
              f"mean={zs.mean():.3f} drop_first_to_last={zs[0]-zs[-1]:+.3f}m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
