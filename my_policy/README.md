# my_policy — oracle policy + data collection for AIC

Companion package to `aic_model` housing:

- **`my_policy/ros/CheatCodeRobust.py`** — ground-truth-TF cable insertion oracle, used during data collection. Does NOT honor cancel without the `_should_abort()` checks; do not use as an evaluation submission.
- **`scripts/gen_trial_config.py`** — randomized batch-config generator for the eval container.
- **`scripts/collect_lerobot.py`** — async-save data recorder that writes a LeRobot v3 dataset.
- **`scripts/inspect_dataset.py`** — fast post-run sanity check (no viz needed).
- **`scripts/viz_dataset.py`** — wrapper around lerobot's rerun viewer that bypasses the torchcodec/FFmpeg-version trap.
- **`scripts/test_*.py`** — host-runnable unit tests; mock rclpy/lerobot, no pixi env required.

All commands below run inside the dev container (`docker compose exec dev bash` from `aic_docker/aic/`). Single-line — your terminal copy-paste mangles `\\` line continuations.

## Three-terminal data collection workflow

**T1 — Zenoh router** (must start first; without it, engine→aic_model service calls intermittently time out):
```
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd
```

**T2 — Eval container** (sim + aic_engine; `ground_truth:=true` is mandatory for the oracle to read TF):
```
docker compose -f /root/.../aic_docker/aic/docker-compose.yml run --rm eval ground_truth:=true start_aic_engine:=true aic_engine_config_file:=/root/aic_data/<batch>.yaml
```

**T3 — Recorder** (in dev container; auto-sweeps stale aic_model from prior runs):
```
pixi run python my_policy/scripts/collect_lerobot.py --batch-config /root/aic_data/<batch>.yaml --root /root/aic_data/<batch>_dataset --repo-id local/aic_oracle_<batch> --max-episode-s 40
```

## Generate a batch config

Smoke (3 trials, mixed plug types):
```
pixi run python my_policy/scripts/gen_trial_config.py --out /root/aic_data/smoke_d4b.yaml --n-trials 3 --seed 42
```

Full collection (~500 trials):
```
pixi run python my_policy/scripts/gen_trial_config.py --out /root/aic_data/batch_500.yaml --n-trials 500 --seed 1 --task-type mixed --distractor-min 0 --distractor-max 4
```

Validate an existing config without regenerating:
```
pixi run python my_policy/scripts/gen_trial_config.py --validate /root/aic_data/<batch>.yaml --out /tmp/_unused
```

## Inspect a recorded dataset (sanity check)

```
pixi run python my_policy/scripts/inspect_dataset.py /root/aic_data/<batch>_dataset
```

Reports per-episode frames/duration/instruction, cross-checks `summary.json` (saves submitted vs succeeded vs failed), and runs sanity assertions on the precomputed stats: action.std non-zero, `groundtruth.port_pose` non-zero, `wrench` variance >0, `meta.insertion_success` reaches 1.0 in every saved episode. A discrepancy in any of these is reported as a `⚠` issue.

## Visualize in rerun

```
pixi run python my_policy/scripts/viz_dataset.py --repo-id local/inspect --root /root/aic_data/<batch>_dataset --episode-index 0 --num-workers 0
```

Notes:
- `viz_dataset.py` is a wrapper around `lerobot.scripts.lerobot_dataset_viz` that forces `video_backend="pyav"`. Direct invocation of `lerobot.scripts.lerobot_dataset_viz` fails on this env because torchcodec only supports FFmpeg ≤ 7 and pixi ships FFmpeg 8.
- `--num-workers 0` is required because the dev container's default `/dev/shm` is 64 MB; multi-worker DataLoader SIGBUSes. Durable fix: add `shm_size: '8gb'` to the dev service in `aic_docker/aic/docker-compose.yml`.

Headless / SSH variants:

```
pixi run python my_policy/scripts/viz_dataset.py --repo-id local/inspect --root /root/aic_data/<batch>_dataset --episode-index 0 --num-workers 0 --save 1 --output-dir /root/aic_data/<batch>_dataset_logs
```

```
pixi run python my_policy/scripts/viz_dataset.py --repo-id local/inspect --root /root/aic_data/<batch>_dataset --episode-index 0 --num-workers 0 --mode distant
```

## Run unit tests on host (no pixi env needed)

```
python3 my_policy/scripts/test_collect_lerobot_loop.py
```

```
python3 my_policy/scripts/test_cheatcode_robust_cancel.py
```

## Dataset schema (what lands in the dataset)

- `action` (7) — pose target: `pose.position.{x,y,z}` + `pose.orientation.{x,y,z,w}`. CheatCodeRobust publishes pose targets via `Policy.set_pose_target` (MODE_POSITION); recording `MotionUpdate.velocity` instead would always be zero.
- `observation.state` (47) — collapsed scalar vector containing: TCP pose/velocity/error (19), joint positions (7), tare-compensated wrench (6), groundtruth port pose (7), groundtruth plug pose (7), `meta.insertion_success` (1). Field names preserved in `meta/info.json` under `features['observation.state'].names`.
- `observation.images.{left,center,right}_camera` — 256×288 RGB, encoded with libsvtav1 (LeRobot v3 video chunks).

The `observation.state` packing is from `aic_utils/lerobot_robot_aic`'s `AICRobotAICController.observation_features`. The action schema is overridden in `make_or_resume_dataset` (canonical adapter advertises 6-D velocity twists for teleop; we record 7-D pose targets matching the oracle's output).

## Known not-bugs

- `info.json` reports `repo_id: ?` and `total_videos: None` — those keys aren't in this LeRobot version's info schema. The dataset is fine.
- Video file count is per-camera-per-chunk (not per-episode-per-camera). 3 cameras + 1 chunk = 3 mp4 files for any number of episodes; `inspect_dataset.py` accepts this.

See also: `~/.claude/projects/-home-robin-ssd-aic-workspace/memory/` for cross-session notes (Zenoh router, recorder restart hygiene, F/T tare semantics, oracle policy design).
