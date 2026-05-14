"""RunACTSlim — inference shim for a slim-observation ACT checkpoint.

Mirrors `RunACT`, with two differences that match
`my_policy/scripts/act/build_act_dataset_slim.py`:

  * `observation.state` is 19-dim:
        tcp_pose(7) ‖ task_vec(12)
    (drops tcp_velocity, tcp_error, joint_positions, wrench — none of those
    are needed by the policy and they were removed from training to test
    a thinner observation hypothesis.)

  * Two cameras only: `center_camera` + `right_camera` (left dropped).

Everything else — checkpoint loading, preprocessor/postprocessor pipelines,
temporal ensembling, the 20 Hz loop, action → Pose conversion, logging —
is inherited unchanged from `RunACT`.

Run-time configuration:
  - `AIC_ACT_CHECKPOINT`  Path to the slim run's
                          `checkpoints/last/pretrained_model/` directory.
                          Required; defaults to the path produced by the
                          first sc_2x5x5_1 slim training run.
  - `AIC_ACT_TIMEOUT_S`   Per-trial budget in seconds. Default 30.
  - `AIC_ACT_TEMPORAL_ENSEMBLE_COEFF`  Optional; same semantics as RunACT.

Launch via the policy framework:
  ros2 run aic_model aic_model --ros-args \\
      -p use_sim_time:=true \\
      -p policy:=my_policy.ros.RunACTSlim
"""
from __future__ import annotations

import numpy as np
import torch

from aic_model_interfaces.msg import Observation

from my_policy.ros.RunACT import (
    IMAGE_SCALING,
    RunACT,
    _ros_image_to_chw_float,
)


def _build_slim_state(obs_msg: Observation, task_vec: np.ndarray) -> torch.Tensor:
    """Compose the 19-dim `observation.state`: tcp_pose(7) + task_vec(12),
    matching the slim dataset schema produced by build_act_dataset_slim.py.

    The channel order is fixed by `KEEP_STATE_INDICES` in the build script —
    if either side changes, the other must change too.
    """
    if task_vec.shape != (12,):
        raise ValueError(f"task_vec must be shape (12,), got {task_vec.shape}")
    p = obs_msg.controller_state.tcp_pose
    state = np.array(
        [
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
            *task_vec.tolist(),
        ],
        dtype=np.float32,
    )
    assert state.shape == (19,), f"state must be 19-dim, got {state.shape}"
    return torch.from_numpy(state)


class RunACTSlim(RunACT):
    """ACT inference with the slim observation space (19-dim state + 2 cams).

    Subclasses RunACT and overrides only what the slim model differs on:
      - the default checkpoint path
      - the observation dict assembly
    """

    DEFAULT_CHECKPOINT: str = (
        "/root/aic_data/v9_act_build/runs/sc_2x5x5_1_slim_v1/"
        "checkpoints/last/pretrained_model"
    )

    def _build_obs_dict(
        self, obs_msg: Observation, task_vec: np.ndarray
    ) -> dict[str, torch.Tensor]:
        return {
            "observation.images.center_camera":
                _ros_image_to_chw_float(obs_msg.center_image, IMAGE_SCALING),
            "observation.images.right_camera":
                _ros_image_to_chw_float(obs_msg.right_image, IMAGE_SCALING),
            "observation.state": _build_slim_state(obs_msg, task_vec),
        }
