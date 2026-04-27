"""Dataset that wraps a recorder-collected LeRobot dataset and emits per-frame
training samples for the port-localizer.

Per `__getitem__`:
  {
    "images": {cam: ndarray[H,W,3] uint8 OR torch.Tensor},  # empty if cameras=()
    "tcp_pose": ndarray[7] float32,         # xyz + xyzw quat in base_link
    "task_one_hot": ndarray[7] float32,     # which target the policy is asked for
    "target": ndarray[5] float32,           # (bx, by, sin_yaw, cos_yaw, rail_t)
    "_meta": {episode_index, frame_index, trial_key, target_module_name, port_name}
  }

Returns numpy arrays so the labeler module is torch-free at import time —
PyTorch's DataLoader auto-converts numpy returns to tensors on collation, and
this lets the killer integration test run on host without pixi. Image tensors
returned in their LeRobot-native form (which IS a torch.Tensor when running
inside pixi); we don't unwrap them since image_transform may be torch-aware.

Two modes:
  - `cameras=()` — parquet-only, no LeRobot/torchcodec dependency. Used by the
    killer integration test. Fast.
  - `cameras=("center_camera", ...)` — also opens LeRobotDataset for image
    decoding. Slower (video decode) and pulls in pixi-only deps.

Pre-computes labels in __init__ since they're per-episode constants — avoids
repeatedly running the YAML/summary join on every frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq
import yaml

from .labels import (
    LocalizerLabel,
    compute_label,
    match_episodes_to_trials,
    task_one_hot,
)


def _load_state_names(dataset_root: Path) -> list[str]:
    """Read observation.state channel names from meta/info.json."""
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    return list(info["features"]["observation.state"]["names"])


def _slice_indices(state_names: list[str], names: list[str]) -> list[int]:
    """Look up the indices in observation.state for an explicit list of channels.
    Raises if any are missing."""
    out: list[int] = []
    for n in names:
        try:
            out.append(state_names.index(n))
        except ValueError:
            raise ValueError(
                f"observation.state has no channel {n!r}; available: {state_names}"
            )
    return out


_TCP_POSE_NAMES = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.x",
    "tcp_pose.orientation.y",
    "tcp_pose.orientation.z",
    "tcp_pose.orientation.w",
]
_PORT_POSE_NAMES = [
    "groundtruth.port_pose.x",
    "groundtruth.port_pose.y",
    "groundtruth.port_pose.z",
]


class LocalizerDataset:
    """Duck-typed PyTorch Dataset (has __len__ and __getitem__).

    NOT subclassed from torch.utils.data.Dataset so the module can be imported
    without torch installed (avoids pulling in heavy deps for the killer test).
    Works with torch.utils.data.DataLoader unchanged.
    """


    def __init__(
        self,
        dataset_root: Path,
        batch_yaml: Path,
        summary_json: Path,
        *,
        cameras: tuple[str, ...] = ("center_camera",),
        image_transform: Callable | None = None,
        repo_id: str = "local/localizer",
    ):
        self._dataset_root = Path(dataset_root)
        self._cameras = tuple(cameras)
        self._image_transform = image_transform

        # --- Load batch config + summary, build episode→trial mapping
        cfg = yaml.safe_load(Path(batch_yaml).read_text())
        summary = json.loads(Path(summary_json).read_text())
        ep_to_trial = match_episodes_to_trials(summary, cfg["trials"])

        # --- Pre-compute labels per saved-episode-index
        self._ep_to_trial: dict[int, str] = ep_to_trial
        self._trials: dict[str, dict] = cfg["trials"]
        self._ep_to_label: dict[int, LocalizerLabel] = {
            ep: compute_label(self._trials[trial_key])
            for ep, trial_key in ep_to_trial.items()
        }
        # Cache the (target_module_name, port_name) per episode for O(1) access.
        self._ep_to_target: dict[int, tuple[str, str]] = {}
        self._ep_to_one_hot: dict[int, np.ndarray] = {}
        for ep, trial_key in ep_to_trial.items():
            task = self._trials[trial_key]["tasks"]["task_1"]
            self._ep_to_target[ep] = (task["target_module_name"], task["port_name"])
            self._ep_to_one_hot[ep] = task_one_hot(task["target_module_name"])

        # --- Resolve observation.state slice indices
        state_names = _load_state_names(self._dataset_root)
        self._tcp_idx = _slice_indices(state_names, _TCP_POSE_NAMES)
        # Port pose indices kept for the killer test's parity check; not in
        # __getitem__ output (would make the model cheat).
        self._port_idx = _slice_indices(state_names, _PORT_POSE_NAMES)

        # --- Load the parquet table once. observation.state is a per-row list
        # (variable-typed in older lerobot, fixed-shape numpy in newer); we
        # eagerly stack into a 2D float64 array since every row is the same
        # length. Cost: ~(num_frames × len(state)) floats = a few MB for 6k
        # frames × 47 channels.
        parquet_path = self._dataset_root / "data" / "chunk-000" / "file-000.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"expected dataset parquet at {parquet_path}; not found"
            )
        cols = ["observation.state", "episode_index", "frame_index"]
        table = pq.read_table(str(parquet_path), columns=cols)
        self._episode_index = table["episode_index"].to_numpy().astype(np.int64)
        self._frame_index = table["frame_index"].to_numpy().astype(np.int64)
        self._state = np.stack([
            np.asarray(r, dtype=np.float64) for r in table["observation.state"].to_pylist()
        ])
        # Validate frame counts vs episode mapping.
        unique_eps = sorted(set(self._episode_index.tolist()))
        expected_eps = sorted(self._ep_to_label.keys())
        if unique_eps != expected_eps:
            raise ValueError(
                f"episode_index column has {unique_eps} but summary join "
                f"yields {expected_eps}; dataset and summary out of sync"
            )

        # --- Optional LeRobot handle for image decoding. Use the plain
        # constructor (NOT .resume() — that opens in "recording" mode and
        # rejects __getitem__ until finalize() is called; we want read-only).
        # Force pyav backend: torchcodec needs FFmpeg ≤ 7 but pixi ships
        # FFmpeg 8 — same trap viz_dataset.py documents. pyav ships its own
        # FFmpeg statically.
        if self._cameras:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self._lr = LeRobotDataset(
                repo_id=repo_id,
                root=str(self._dataset_root),
                video_backend="pyav",
            )
        else:
            self._lr = None

    def __len__(self) -> int:
        return int(len(self._state))

    @property
    def num_episodes(self) -> int:
        return len(self._ep_to_label)

    def label_for_episode(self, episode_index: int) -> LocalizerLabel:
        return self._ep_to_label[episode_index]

    def target_for_episode(self, episode_index: int) -> tuple[str, str]:
        """Returns (target_module_name, port_name) for the episode."""
        return self._ep_to_target[episode_index]

    def port_pose_baselink(self, frame_index: int) -> np.ndarray:
        """Convenience: read recorded groundtruth.port_pose.{x,y,z} for a frame.
        Used by the killer test; not part of the ML input.
        """
        return self._state[frame_index, self._port_idx].astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        ep = int(self._episode_index[idx])
        label = self._ep_to_label[ep]
        target_module, port_name = self._ep_to_target[ep]
        one_hot = self._ep_to_one_hot[ep]
        tcp = self._state[idx, self._tcp_idx].astype(np.float32)
        target = label.as_target_5()

        images: dict[str, torch.Tensor] = {}
        if self._lr is not None:
            frame = self._lr[idx]
            for cam in self._cameras:
                key = f"observation.images.{cam}"
                if key not in frame:
                    raise KeyError(
                        f"camera {cam!r} not in frame; available image keys: "
                        f"{[k for k in frame if 'image' in k]}"
                    )
                img = frame[key]
                if self._image_transform is not None:
                    img = self._image_transform(img)
                images[cam] = img

        return {
            "images": images,
            "tcp_pose": tcp,
            "task_one_hot": one_hot.astype(np.float32),
            "target": target,
            "_meta": {
                "episode_index": ep,
                "frame_index": int(self._frame_index[idx]),
                "trial_key": self._ep_to_trial[ep],
                "target_module_name": target_module,
                "port_name": port_name,
            },
        }
