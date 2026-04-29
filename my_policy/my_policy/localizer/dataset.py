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


class MultiBatchLocalizerDataset:
    """Concatenates several `LocalizerDataset`s into one duck-typed dataset.

    Episode indices are remapped to globally unique consecutive ints so the
    train_localizer episode-split logic works unchanged. `_ep_to_label`,
    `_ep_to_target`, `_ep_to_one_hot`, `_ep_to_trial`, and `_episode_index`
    mirror the LocalizerDataset surface area.

    `__getitem__` dispatches to the owning child by global frame index.
    """

    def __init__(
        self,
        roots: list[Path],
        batch_yamls: list[Path],
        summary_jsons: list[Path],
        *,
        cameras: tuple[str, ...] = ("center_camera",),
        image_transform: Callable | None = None,
        repo_id: str = "local/localizer",
    ):
        if not (len(roots) == len(batch_yamls) == len(summary_jsons)):
            raise ValueError("roots, batch_yamls, summary_jsons must be same length")
        if not roots:
            raise ValueError("need at least one batch")

        self._children: list[LocalizerDataset] = [
            LocalizerDataset(
                r, y, s,
                cameras=cameras,
                image_transform=image_transform,
                repo_id=f"{repo_id}/{i}",
            )
            for i, (r, y, s) in enumerate(zip(roots, batch_yamls, summary_jsons))
        ]

        # Per-child cumulative frame offset for global → (child_idx, local_idx).
        lengths = [len(c) for c in self._children]
        self._cum_offsets = np.cumsum([0] + lengths)
        total = int(self._cum_offsets[-1])

        # Per-child episode offset; child-local episode `ep` becomes
        # `ep_offsets[k] + ep`. Offsets jump by max-local-ep+1 to keep ints
        # tight and consecutive-ish without collision.
        self._ep_offsets: list[int] = []
        running = 0
        for c in self._children:
            self._ep_offsets.append(running)
            local_max = max(c._ep_to_label.keys()) if c._ep_to_label else -1
            running += int(local_max) + 1

        # Build combined dicts and the global episode_index column.
        self._ep_to_label: dict[int, LocalizerLabel] = {}
        self._ep_to_target: dict[int, tuple[str, str]] = {}
        self._ep_to_one_hot: dict[int, np.ndarray] = {}
        self._ep_to_trial: dict[int, str] = {}
        self._ep_to_child: dict[int, int] = {}
        for k, c in enumerate(self._children):
            off = self._ep_offsets[k]
            for ep, lbl in c._ep_to_label.items():
                gep = off + ep
                self._ep_to_label[gep] = lbl
                self._ep_to_target[gep] = c._ep_to_target[ep]
                self._ep_to_one_hot[gep] = c._ep_to_one_hot[ep]
                self._ep_to_trial[gep] = c._ep_to_trial[ep]
                self._ep_to_child[gep] = k

        # Remapped per-frame episode_index. Concatenate child arrays + offsets.
        self._episode_index = np.concatenate([
            c._episode_index + self._ep_offsets[k]
            for k, c in enumerate(self._children)
        ]).astype(np.int64)
        self._frame_index = np.concatenate([
            c._frame_index for c in self._children
        ]).astype(np.int64)
        self._total = total

    @classmethod
    def from_collection_dir(
        cls,
        collection_dir: Path,
        batch_names: list[str],
        *,
        cameras: tuple[str, ...] = ("center_camera",),
        image_transform: Callable | None = None,
    ) -> "MultiBatchLocalizerDataset":
        """Convenience: assumes recorder layout `<name>/`, `<name>.yaml`,
        `<name>_logs/summary.json` under `collection_dir`.
        """
        cd = Path(collection_dir)
        roots = [cd / n for n in batch_names]
        yamls = [cd / f"{n}.yaml" for n in batch_names]
        summaries = [cd / f"{n}_logs" / "summary.json" for n in batch_names]
        for p in roots + yamls + summaries:
            if not p.exists():
                raise FileNotFoundError(f"missing: {p}")
        return cls(roots, yamls, summaries,
                   cameras=cameras, image_transform=image_transform)

    def __len__(self) -> int:
        return self._total

    @property
    def num_episodes(self) -> int:
        return len(self._ep_to_label)

    def label_for_episode(self, episode_index: int) -> LocalizerLabel:
        return self._ep_to_label[episode_index]

    def target_for_episode(self, episode_index: int) -> tuple[str, str]:
        return self._ep_to_target[episode_index]

    def port_pose_baselink(self, frame_index: int) -> np.ndarray:
        k = int(np.searchsorted(self._cum_offsets, frame_index, side="right") - 1)
        local = frame_index - int(self._cum_offsets[k])
        return self._children[k].port_pose_baselink(local)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        k = int(np.searchsorted(self._cum_offsets, idx, side="right") - 1)
        local = idx - int(self._cum_offsets[k])
        sample = self._children[k][local]
        # Remap _meta.episode_index to the global value used by _ep_to_label.
        local_ep = sample["_meta"]["episode_index"]
        sample["_meta"]["episode_index"] = self._ep_offsets[k] + local_ep
        sample["_meta"]["batch_index"] = k
        return sample
