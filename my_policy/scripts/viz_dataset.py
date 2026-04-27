#!/usr/bin/env python3
"""Wrapper around lerobot.scripts.lerobot_dataset_viz that forces the pyav
video backend.

Why this exists: torchcodec (lerobot's default video backend) is pinned to
FFmpeg 4-7 shared libs but our pixi env ships FFmpeg 8 (libavutil.so.60),
so torchcodec fails to load. Pulling in FFmpeg 7 via `pixi add` hits a
cross-platform solver conflict (osx-arm64 has no compatible build that
fits the rest of the locked env). The pyav package ships its own
statically-linked FFmpeg and works regardless of system FFmpeg version.

Usage (single line):
    pixi run python my_policy/scripts/viz_dataset.py --repo-id local/inspect --root /root/aic_data/smoke_d4b_dataset --episode-index 0
"""

import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Monkey-patch LeRobotDataset.__init__ to default video_backend="pyav"
# before lerobot_dataset_viz constructs the dataset.
_orig_init = LeRobotDataset.__init__


def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("video_backend", "pyav")
    return _orig_init(self, *args, **kwargs)


LeRobotDataset.__init__ = _patched_init


if __name__ == "__main__":
    # Delegate to the real viz script's main(); it picks args from sys.argv.
    from lerobot.scripts.lerobot_dataset_viz import main
    sys.exit(main() or 0)
