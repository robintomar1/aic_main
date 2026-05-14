# AIC Dataset Recording Guide

Practical reference for recording new IL training datasets for the AIC cable-insertion challenge. Every fact below is sourced from either the current code (`my_policy/scripts/`, `my_policy/my_policy/`, `aic_docker/aic/docker-compose.yml`, `aic_controller/src/`) or post-mortem memory of previous recording sessions. File:line citations point at current state on this branch — re-verify before relying on a constant if the code has moved.

---

## 1. Pre-flight checklist (do not skip)

Before pressing record on a new batch, confirm every line:

- [ ] Recording on the **GPU box** (NVMe SSD), not the laptop. Camera bandwidth is ~210 MB/s sustained at 20 Hz × 3 cams × 1152×1024 — laptop SSD dropped cams to 2.4 Hz on 2026-04-24.
- [ ] **Zenoh router is running** in the dev container (see §3).
- [ ] Eval container will receive **`ground_truth:=true`** as a launch arg (without it, every trial scores 0 — see §6.1).
- [ ] No stale `aic_model` / `collect_lerobot` processes from a previous run (recorder cleans on startup, but verify if the previous run crashed mid-init).
- [ ] Disk has at least 25 GB free per active bag and ~250 GB target for a 500-episode dataset.
- [ ] Trial config YAML generated with the **fixed robot spawn pose** that eval uses (see §4); robot spawn is NOT randomized.
- [ ] Outputs go under `data_collection/<batch_name>_run/` so both containers see them at `/root/aic_data/<batch_name>_run/`.

---

## 2. Architecture / mounts

Two-container split — eval-side packages (`aic_engine`, `aic_bringup`, `aic_adapter`, `aic_controller`, `aic_gazebo`) only exist in the upstream `ghcr.io/intrinsic-dev/aic/aic_eval` image. Dev container runs `aic_model` (the participant policy node) + recorder.

`aic_docker/aic/docker-compose.yml` mounts (verified current):

```yaml
dev:
  volumes:
    - ./workspace:/root/ws_aic/src        # source: aic_main as submodule
    - ./data_collection:/root/aic_data    # batch configs, outputs, logs
eval:
  volumes:
    - ./results:/root/aic_results
    - ./data_collection:/root/aic_data    # SAME path inside eval
```

The unified `/root/aic_data` mount is the trick — both sides reference batch configs and outputs at the same path, no copies.

**Open issue (not blocking but worth fixing):** `shm_size` is NOT set in `docker-compose.yml`. Default Docker shm is 64 MB → DataLoader workers (used at training & by `viz_dataset.py`) SIGBUS. Workaround in tools is `--num-workers 0`; durable fix is `shm_size: '8gb'` under both services.

---

## 3. Zenoh router (REQUIRED before everything)

Without an explicit router, peer discovery via Zenoh scouting is racy on host-network Docker. Symptom: engine logs show `Service '/aic_model/get_state' is available` followed ~30 s later by `GetState service call timed out`, and the engine bails before any trial.

Start the router **first**, leave it running for the whole recording session:

```bash
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run ros2 run rmw_zenoh_cpp rmw_zenohd'
```

Verify it's alive: `docker compose exec dev bash -c 'pgrep -af rmw_zenohd'`. If a previous smoke worked without a router, that was luck — discovery happened to converge that time. Don't rely on it.

---

## 4. Trial config generation

Script: `my_policy/scripts/gen_trial_config.py`. Generates a YAML batch config the eval container reads via `aic_engine_config_file:=...`.

### What's randomized (matches qualification eval distribution)

From `my_policy/my_policy/gen_trial_config.py` (current values; re-quote before relying on them):

| Variable | Range | Source |
|---|---|---|
| Board X | 0.13 – 0.20 m | `BOARD_X_MIN / BOARD_X_MAX` (lines 45–46) |
| Board Y | −0.10 – +0.10 m | `BOARD_Y_MIN / BOARD_Y_MAX` (lines 47–48) |
| Board Z | 1.14 m (fixed) | `BOARD_Z` (line 49) |
| Board yaw | −35° – +35° | (lines ~50) |
| Mount yaw jitter | ±0.3 rad | `MOUNT_YAW_JITTER` (line 53) |
| Distractors | `--distractor-min .. --distractor-max` (default 0..4) | CLI |
| Target task | `--task-type {sfp, sc, mixed}` (default mixed) | CLI |

**What is fixed (do NOT randomize):**

- Robot base_link: `(-0.2, 0.2, 1.14)`, yaw `−π` rad — `ROBOT_BASE_LINK_IN_WORLD` (~line 86–88 of `gen_trial_config.py`).
- Plug-in-gripper attachment (cable is in gripper at start).
- Plug/cable types: SFP→SFP_PORT, SC→SC_PORT.

**Visibility predicate** (target port must be ≥50 px from image edges, `VISIBILITY_MARGIN_PX = 50`, line 80) rejects ~24% of random samples at full board bounds. Enabling it guarantees all trials have the target port in view — matches the qualification guarantee that "target port of interest will always be within view of the robot cameras" (`aic_main/docs/qualification_phase.md` line 44).

**Why match these bounds:** the qualification eval uses this same distribution. Recording with wider bounds (e.g. randomized robot spawn) makes the policy solve a harder problem than it'll be tested on, wasting capacity.

### Common invocations

```bash
# Inside dev container, from /root/ws_aic/src/aic
pixi run python my_policy/scripts/gen_trial_config.py --n 100 --seed 42 --task-type mixed --out /root/aic_data/batch_100_a.yaml
pixi run python my_policy/scripts/gen_trial_config.py --n 100 --seed 42 --task-type sc    --out /root/aic_data/batch_100_sc.yaml
```

---

## 5. The oracle (CheatCodeRobust)

File: `my_policy/my_policy/ros/CheatCodeRobust.py`. Used at oracle data-collection time only; needs `ground_truth:=true` for TF. **NOT submission-safe.**

### Phase machine

1. **APPROACH** (~5 s, `APPROACH_STEPS=100`, `APPROACH_SLEEP=0.06`): linear interpolation from current TCP pose to `port + APPROACH_Z_OFFSET=0.20 m`. Pure feedforward.
2. **ALIGN** (gated): hover at `HOVER_Z_OFFSET_BY_PLUG = {"sfp": 0.20, "sc": 0.10}`; wait for `|plug_xy − port_xy| < ALIGN_XY_THRESHOLD_M=0.0025` stable for `ALIGN_STABLE_S=1.0`. Timeout `ALIGN_TIMEOUT_S=10.0`.
3. **INSERT**: descent at `DESCENT_STEP=0.0003` per tick (~6 mm/s @ 20 Hz) with PI XY correction + force gate + retreat-during-hold + insertion_event termination.

### Critical constants (from current code, lines cited)

```python
PROPORTIONAL_GAIN        = 0.25     # CheatCodeRobust.py:58
INTEGRATOR_GAIN          = 0.1      #                  :59
DERIVATIVE_GAIN          = 0.08     #                  :68
MAX_INTEGRATOR_WINDUP    = 0.30     #                  :69

APPROACH_Z_OFFSET        = 0.2      #                  :73
HOVER_Z_OFFSET_BY_PLUG   = {"sfp": 0.20, "sc": 0.10}   # :76-79
INSERT_Z_OFFSET          = -0.015   #                  :81
DESCENT_STEP             = 0.0003   #                  :84

FORCE_STOP_N             = 18.0     #                  :109
FORCE_RESUME_N           = 12.0     #                  :110
HOLD_RETREAT_STEP        = 0.0001   #                  :119  (2 mm/s upward)
HOLD_RETREAT_MAX         = 0.005    #                  :120

INSIDE_XY_THRESHOLD_M    = 0.002    #                  :157
INSIDE_DEPTH_BY_PLUG     = {"sfp": -0.003, "sc": -0.0015}  # :158-161

PLUG_TIGHT_AXIS_BY_PLUG  = {"sc": "y"}                # :174
ALIGN_TIGHT_THRESHOLD_M  = 0.0005   #                  :175  (SC only)
ALIGN_CHAMFER_THRESHOLD_M= 0.003    #                  :176  (SC only)
```

### Known dataset artifacts produced by this oracle

These are properties of **how the oracle commands action targets**, not bugs. They show up in every recorded dataset and require either downstream tolerance or post-hoc relabeling. Don't be surprised when you see them:

1. **Macro action[z] step at INSERT-phase start (~frame 126).** When the oracle transitions from ALIGN to INSERT, the published `pose_command` z target jumps discretely from the hover height to the descent setpoint — empirically a ~100 mm step in port-local frame for every SC episode (130/130 in `v9_port_local_smolvla_dataset_wo_tcp_error`), 0/296 SFP episodes (the SFP descent emerges interpolated from the integrated `DESCENT_STEP` at PI-rate before INSERT logically separates). `observation.state` (TCP pose from joint encoders) is smooth across the boundary because the controller does compliant tracking — **the discontinuity lives only in the action label**. SmolVLA training run on this dataset failed to learn SC inserts; post-hoc smoothing of the step (smoothstep5 backward ramp) was tried and **did not** resolve the failure (see `project_aic_z_step_smoothing.md` in memory). The lesson is: don't expect downstream smoothing to rescue a phase-discontinuous oracle, and don't make a change to the oracle hoping it'll fix the model — verify the failure mode is actually caused by the discontinuity first.
2. **Approach micro-stair-steps every 4 frames.** Oracle publishes pose targets at ~5 Hz; the recorder ticks at 20 Hz → action stays flat for 3 frames then steps. Magnitudes 5–10 mm. Both SFP and SC have these. SmolVLA learns fine through them (SFP works), so they're below the model's prediction-resolution noise floor.
3. **Force-gate engagements (~12–18 N) are normal chamfer contact.** Don't lower `FORCE_STOP_N` below 18 — it eats healthy contact and causes hold-release oscillation (verified in 2026-04-24 sweep, recorded in `project_aic_oracle_policy.md`).

### What NOT to change in the oracle (recorded as anti-patterns)

- Do not feedforward `plug_tip_gripper_offset` in XY — unstable; cable lag feeds back and diverges. PI on `(port_xy − plug_xy)` is the only stable form.
- Do not freeze the full pose during a force hold — XY integrator must keep updating so misalignment corrects while we wait.
- Do not lower `FORCE_STOP_N` below 18 N (eats healthy chamfer contact).
- For SC plug specifically, ALIGN and INSIDE-latch must be **axis-aware** (`PLUG_TIGHT_AXIS_BY_PLUG = {"sc": "y"}`); the chamfer is along local X only. Symmetric `xy_err` magnitude check passes guaranteed-jam configurations. Tests in `test_cheatcode_robust_cancel.py`.

---

## 5.5 Recommendations for next-batch oracle changes

The previous three failure modes (port-local hover-lock, SmolVLA SC failure, post-hoc-smoothing dud) all trace to the same root cause: **the current oracle generates data that's structurally hard for IL to learn from**, not data the oracle itself fails on. The prior position in `project_aic_oracle_policy.md` — *"improving the oracle past 7/10 is yak-shaving relative to training a learned policy"* — assumed data quality was adequate. The 2026-05 failure record contradicts that. **The oracle's data IS the constraint.** This section captures changes worth bundling into the next data-collection run.

None of these is proven to fix the model — they are hypotheses backed by failure-mode evidence. The disciplined approach is: implement Tier 1 together, record a 50-episode pilot, train SmolVLA short (10 k steps), check whether validation MAE on hover-region frames improves vs the current dataset, **then** commit to a 500-episode run.

### Diagnosis: data property → failure mode

| Past failure | Data property responsible | Source |
|---|---|---|
| Port-local locks at hover (~5–7 cm above port) | Training data has TCP **continuously moving** during ALIGN/INSERT (PI controller never lets it pause). At inference, model commands "stay" → controller stops TCP → `(TCP at z≈−0.13, vel=0)` is OOD. Velocity-Z bias injection didn't help — lock is robust to single-channel perturbation. | `project_aic_port_local.md` post-mortem |
| SmolVLA SC inference failure | Action lead direction is **inconsistent** in SC: during ALIGN the action label is up to 60 mm *behind* obs (action says "stay at hover" while TCP drifts toward port); after INSERT-start it flips to leading by +20–30 mm. SFP doesn't have this flip. Model can't learn one stable obs→action relation for the same task. | Today's session (2026-05-10): forward-ramp smoothing inverted lead → 75 mm wrong-direction overshoot |
| Post-hoc smoothing didn't help | The 100 mm action step is a *symptom* of a state-machine phase transition with multiple downstream effects (target formula change, integrator reset, control mode change). Erasing the step in the action label without erasing it in the underlying physics doesn't change what the model has to learn. | Today: backward ramp eliminated the 100 mm step + reduced lead-overshoot 6×, SmolVLA still failed |
| Model has no recovery in repertoire | `inspect_act_demos.py` filters out any episode with `\|F\| ≥ 18 N` or >10 chamfer-band frames as "messy" (~19.5% excluded), including episodes where the oracle successfully recovered via retreat-during-hold. **Model never sees "force engaged → retreat → re-descend → succeed".** | `project_aic_act_dataset.md` clean filter |

### Tier 1 — high impact, low risk (do these together)

**R1. Eliminate the discrete ALIGN→INSERT z-target jump.** Have the oracle's z target evolve continuously across the phase boundary: the first commanded z in INSERT phase should equal the last commanded z in ALIGN, then decrement by `DESCENT_STEP` from there. No reset of the target formula at the phase boundary. Removes the 100 mm action step at the source — every recorded trajectory becomes C⁰-continuous in action z, matching SFP's natural behavior. *Why this is different from today's post-hoc smoothing:* it changes the actual command the controller executes, not just the recorded label, so obs and action stay coherent.

**R2. Stop excluding "messy but successful" episodes from training.** Split the clean filter into two flags: `force_engaged` (informational) and `failed` (uses `/scoring/insertion_event`). Train on `not failed`, not `clean and not failed`. The recovery episodes (force-gate engaged → retreat-during-hold → re-descend → insertion_event fired) are the **most valuable** training data because they show the model what to do when something goes wrong. Currently we throw them away and then wonder why the trained policy has no recovery in its repertoire. Edit `inspect_act_demos.py` to emit a `recovered_messy_episodes.json` alongside `clean_episodes.json`; the build pipeline can opt-in via `--clean-episodes-json + --recovered-episodes-json`.

**R3. Add explicit "settled hover" dwells with TCP truly stationary.** When ALIGN converges (XY stable), instead of polling and continuing to publish hover targets through the integrator (which keeps TCP weakly moving), **freeze the published action target for N=10–20 ticks** before transitioning to INSERT. Generates training frames of `(TCP at hover, vel ≈ 0, action = hover)` followed by `(TCP at hover, vel ≈ 0, action = first descent target)`. Directly addresses the port-local hover-lock OOD failure mode. Cost: adds ~0.5–1 s per episode; trial throughput drops <2%.

### Tier 2 — medium impact, medium risk

**R4. Soften Z stiffness during INSERT (compliance descent).** Listed in `project_aic_oracle_policy.md` as an unbuilt candidate. Bypass `set_pose_target` and construct `MotionUpdate` with reduced Z stiffness so the plug finds the port via mechanical compliance rather than rigid commanded descent. Two payoffs: (a) INSERT becomes robust to small XY misalignment (plug slides along chamfer instead of jamming or triggering force gate), (b) the *recorded action target* becomes meaningful in itself — under low Z stiffness, the action represents a "soft pull toward this point", which is closer to what the model can learn to predict than "rigid command target".

**R5. Publish action targets at 20 Hz with linear interpolation between waypoints.** Currently the oracle publishes pose targets at a lower rate during APPROACH; the recorder samples at 20 Hz, producing the 4-frame stair-step pattern (~5–10 mm jumps every 4 frames). Publishing at full 20 Hz with linear interpolation removes the stair-steps. SmolVLA learned through these on SFP, but they add noise that competes with the actual signal and they amplify the SC discontinuity problem. Verify with `pixi run python my_policy/scripts/act/inspect_action_traces.py <recording>` first to confirm the stair-step pattern is still present in current code.

### Tier 3 — verification, not modification

**R6. Verify spiral search engages on chamfer contact.** Current code has `SPIRAL_FORCE_LO_N = 8.0`, `SPIRAL_RADIUS_M = 0.002`, mode `x_only` for SC (lines 134–148). Add a one-time analysis pass on the first ~20 recorded episodes counting spiral-tick occurrences per trial. If it's near zero, the threshold is too high (or the radius too small) and we're not actually generating the recovery training data we think we are.

**R7. Implement retry-from-hover if INSERT exits without `insertion_event`.** Listed in `project_aic_oracle_policy.md`. One additional approach attempt before giving up. Lifts oracle success ~7/10 → potentially 8–9/10, increasing usable training episodes per recording session by 10–20%. Free upside if R2 is also adopted (more retries = more recovery training data).

### Test-then-commit recording protocol

1. Implement R1 + R2 + R3 in a single branch off the current oracle.
2. Record a 50-episode pilot batch (mixed SFP+SC, seed 42 to enable comparison): `--n 50 --task-type mixed --seed 42`.
3. Build the dataset (port-local + SmolVLA variant).
4. Train SmolVLA for 10 k steps (~1 hr on the 48 GB box at batch 8).
5. Compare validation action MAE on **hover-region frames** (port-local z ∈ [−0.18, −0.10]) vs the equivalent metric on `v9_port_local_smolvla_dataset_wo_tcp_error`. The hover region is where past models locked — improvement here is the most direct positive signal.
6. If hover-region MAE drops by ≥30%, scale to 500 episodes. If it does not, do not commit; the diagnosis was wrong and a different approach is needed (consider R4 + R5 next, or pivot to model-side experiments).

### What we would NOT change

- Do not lower `FORCE_STOP_N` below 18 N (eats normal chamfer contact, causes hold-release oscillation — see §5).
- Do not try XY feedforward of `plug_tip_gripper_offset` (unstable, cable lag → divergence).
- Do not reduce `ALIGN_STABLE_S` or `ALIGN_TIMEOUT_S` "to save time" — R3 actively wants to spend more time in stable hover.

---

## 6. Recording session lifecycle

Three-terminal workflow. Order matters.

### 6.1 Critical: `ground_truth:=true` on the eval launch

Without this flag, ground-truth TF for the port frame is not relayed to `/tf`. CheatCodeRobust waits 10 s for `task_board/<mount>/<port>_link`, returns False, and **every trial scores 0**. Symptom: `/aic_controller/pose_commands` has 0 messages in the recorded data.

### 6.2 Three terminals

**T1 (dev container) — Zenoh router** (see §3, leave running):

```bash
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run ros2 run rmw_zenoh_cpp rmw_zenohd'
```

**T2 (host) — start eval engine with batch config:**

```bash
cd ~/ssd/aic_workspace/aic_docker/aic && docker compose run --rm eval ground_truth:=true start_aic_engine:=true aic_engine_config_file:=/root/aic_data/<batch>.yaml
```

This blocks until the engine completes all N trials, then exits. Wait ~30–60 s for engine startup before launching T3.

**T3 (dev container) — recorder:**

```bash
docker compose exec dev bash
cd /root/ws_aic/src/aic
pixi run python my_policy/scripts/collect_lerobot.py \
  --batch-config /root/aic_data/<batch>.yaml \
  --root /root/aic_data/<batch>_run \
  --repo-id local/<batch> \
  --policy my_policy.ros.CheatCodeRobust \
  --fps 20 --max-episode-s 40 --warm-up-s 180
```

`collect_lerobot.py` (current primary recorder, replaces the older MCAP-based `collect_episode.py`):

- Writes LeRobotDataset v3.0 **directly** (one episode per trial, append via `LeRobotDataset.resume()` at line 354–366).
- Sweeps stale `aic_model` processes at startup (`kill_stale_aic_model()` at line 290, called at line 726) — restart-the-recorder is a supported flow.
- Subscribes to:
  - `/aic_controller/pose_commands` (action capture)
  - `/scoring/insertion_event` (episode end signal)
  - `/insert_cable/_action/status` (trial start/end)
  - `/tf` (eval-container heartbeat)
- CLI flags (collect_lerobot.py:665–684): `--batch-config`, `--root`, `--repo-id`, `--policy`, `--fps`, `--max-episode-s`, `--warm-up-s`, `--global-timeout-s`.

### 6.3 What gets recorded per batch

`/root/aic_data/<batch>_run/` (LeRobet v3.0 layout):

- `data/chunk-XXX/file-YYY.parquet` — single concatenated parquet for all episodes (codebase_version=v3.0)
- `videos/observation.images.{left,center,right}_camera/...` — AV1-encoded MP4s
- `meta/info.json` — schema, fps, episode count
- `meta/stats.json` — aggregate per-feature stats
- `meta/episodes/chunk-XXX/file-YYY.parquet` — per-episode stats + length + video offsets
- `meta/tasks.parquet` — task-string ↔ task_index lookup. **Must be string-indexed** for SmolVLA (see §8.4).

---

## 7. Force/torque sensor semantics (don't reinvent)

The controller does orientation-aware tare. From `aic_controller/src/aic_controller.cpp`:

- **Store tare at service-call time** (lines 417–420): rotate sensed wrench from FTS frame into base_link frame, store as `tare_offset_at_base_`.
- **Rotate back every tick** (lines 1158–1161): inverse-rotate `tare_offset_at_base_` through current tool rotation → `tare_offset_at_tip_`.
- **Publish** as `controller_state.fts_tare_offset` (frame_id `ati/tool_link`, lines 1275–1277).

**Compensated wrench** = `wrench_raw − fts_tare_offset.wrench`. Cancels the rotating gravity bias correctly.

**Gotcha:** `attach_cable_to_gripper:=true` (default) means a cable hangs at tare time, so `tare_offset_at_base_` bakes in gravity + cable-tension-at-tare-pose. As the arm moves, gravity cancels but cable tension changes with pose → residual ~5–10 N in the compensated wrench. **This is real signal, not miscalibration.** Don't try to subtract it.

**Implications for recording:**

- Contact thresholds for heuristic policies: ~12–15 N above moving baseline (matches `FORCE_STOP_N=18`, `FORCE_RESUME_N=12`).
- For learned policies: store `wrench_raw`, `fts_tare_offset`, and compensated as separate channels so the model can learn the cable-tension signature if useful. (Current schema records the compensated wrench only; this is a future TODO if downstream models would benefit.)

---

## 8. Schema and format requirements (LeRobotDataset v3.0)

### 8.1 Episode-bounds API

LeRobotDataset v3.0 dropped `dataset.episode_data_index["from"/"to"]` (v2.x API). Use:

- `ds.num_episodes`
- `ds.meta.episodes[i]["dataset_from_index"]` — int (parquet stores it as 1-element list, the loader unwraps)
- `ds.meta.episodes[i]["dataset_to_index"]`, `["length"]`, `["tasks"]`

The version is in `meta/info.json` → `codebase_version`. Verify against installed lerobot source at `.pixi/envs/default/lib/python3.12/site-packages/lerobot/datasets/lerobot_dataset.py` before assuming attr names — minor versions drift.

### 8.2 Three-camera schema

From `info.json`:

- `observation.images.left_camera`, `observation.images.center_camera`, `observation.images.right_camera`
- Stored at 256×288 (downscaled from native; `info.json` is authoritative — re-read before quoting)
- Codec: AV1, yuv420p, 20 fps, no audio

Native camera resolution is 1152×1024 raw. The recorder downsamples to 256×288 for storage; full-res JPEG q85 every 5th frame would be needed for a port localizer (not currently in the recorder).

### 8.3 State schema (varies by build stage)

| Stage | Dim | Channels |
|---|---|---|
| Raw recorder | 47 | TCP pose 7 + TCP vel 6 + TCP error 6 + joint pos 7 + wrench 6 + port pose 7 + insertion success scalar (verify against current `aic_robot_aic_controller.py`) |
| ACT (`build_act_dataset.py`) | 44 | TCP pose 7 + TCP vel 6 + TCP error 6 + joint pos 7 + wrench 6 + 12-dim structured task vector (drops port_pose; task one-hot replaces it) |
| Port-local (`make_port_local_dataset.py`) | 44 | Same shape as ACT but spatial channels rotated into target-port frame; `info.json:frame_transform = "port_local"` |
| SmolVLA (`make_smolvla_dataset.py`) | 26 | tcp_pose 7 + tcp_vel 6 + joint_pos 7 + wrench 6 (drops tcp_error 6 + task one-hot 12); `frame_transform = "port_local_smolvla_no_err"` |

**Why drop `tcp_error` for SmolVLA:** it's auto-regressive on the policy's own past output (`target_pose − current_pose`). The phase-trigger probe identified `state[15] = tcp_error.z` as the channel a trained SmolVLA used as a hover-vs-commit shortcut. Removing it forces the model to rely on visual / spatial / wrench channels.

**Why drop the 12-dim task one-hot for SmolVLA:** SmolVLA gets task identity via the natural-language `tasks` string per episode.

### 8.4 `tasks.parquet` MUST be string-indexed

`dataset_reader.py:281` does `meta.tasks.iloc[task_idx].name` to retrieve the task string. `.name` returns the index value, so the parquet's INDEX must BE the task string. Build it as:

```python
import pandas as pd
df = pd.DataFrame(
    {"task_index": list(range(len(unique_tasks)))},
    index=pd.Index(unique_tasks, name="task"),
)
df.to_parquet(out_meta_dir / "tasks.parquet")
```

**NOT** as `pa.Table.from_pylist([{"task_index": i, "task": s}, ...])` — that gives a default RangeIndex with two columns. ACT silently ignores the bug; SmolVLA's `TokenizerProcessorStep` raises `Task cannot be None` because `isinstance(task, str)` is False.

**Current state on this branch (verify before relying):**

- ✅ `make_port_local_dataset.py:158–182` — string-indexed (correct).
- ❌ `build_act_dataset.py:144–162` — uses `pa.Table.from_pylist`, **does NOT** produce string-indexed parquet. ACT-only datasets work; if you train SmolVLA on a `build_act_dataset.py` output without going through `make_port_local_dataset.py`, you'll hit the `Task cannot be None` error.
- ✅ `merge_act_datasets.py`, `clean_act_dataset.py`, `make_smolvla_dataset.py` — all carry the source's `tasks.parquet` verbatim.

Validate any builder you write with:

```python
from lerobot.datasets.io_utils import load_tasks
t = load_tasks(Path(dataset_root))
assert isinstance(t.iloc[0].name, str), f"tasks.parquet index must be str, got {type(t.iloc[0].name)}"
```

---

## 9. Per-batch verification (post-recording, before building)

Before declaring a batch usable, verify:

1. **Engine stdout / log:** look for `All Trials Processed!` with non-zero score. If you see `GetState service call timed out for node 'aic_model'`, the Zenoh router was missing or the recorder didn't start in time.
2. **`meta.json`** (in batch_run dir): `observed_events == expected_trials`. If less, the recorder hit a timeout or aic_model crashed mid-batch.
3. **model.log** (in `<batch>_run/logs/`): `grep "force gate engaged"` — counts per trial. Many engagements + low score = misalignment problem; zero engagements + good scores = clean run.
4. **`/aic_controller/pose_commands` non-empty in the recorded data.** If it's empty, you forgot `ground_truth:=true` (see §6.1).
5. **Quick LeRobot v3.0 inspect:**

   ```bash
   pixi run python my_policy/scripts/inspect_dataset.py /root/aic_data/<batch>_run
   ```

6. **Visual spot-check** (rerun):

   ```bash
   pixi run python my_policy/scripts/viz_dataset.py --repo-id local/inspect --root /root/aic_data/<batch>_run --episode-index 0 --num-workers 0
   ```

   `--num-workers 0` is required because of the 64 MB shm default (see §2). For headless: add `--save 1 --output-dir <dir>` and rerun the `.rrd` on the host.

---

## 10. Build pipeline (raw recording → training-ready)

All builders are pure pyarrow + numpy + yaml — no torch/lerobot at preprocess time. All idempotent.

### 10.1 Episode classification (clean vs messy)

```bash
pixi run python my_policy/scripts/act/inspect_act_demos.py --batch <batch> --out /root/aic_data/v9_act_build/<batch>_act_clean_episodes.json
```

Episode is `messy` if:

- Any frame `|F| ≥ 18 N` (force-gate engaged), OR
- More than 10 frames in `[8, 18) N` chamfer-contact band (`CHAMFER_BAND_FRAME_BUDGET = 10`).

Empirical rate across 6 batches: **80.5% clean**.

### 10.2 Build per-batch ACT dataset

```bash
pixi run python my_policy/scripts/act/build_act_dataset.py --batch <batch> --clean-episodes-json <...>_clean_episodes.json
```

Produces `<batch>_act_dataset/` with 44-dim observation.state and 7-dim action. **Note (§8.4):** `tasks.parquet` here is NOT string-indexed. Fine for ACT, broken for SmolVLA.

### 10.3 Merge batches

```bash
pixi run python my_policy/scripts/act/merge_act_datasets.py --sources <batch_a>_act_dataset <batch_b>_act_dataset ... --out v9_act_merged
```

Verifies `tasks.parquet` is identical across sources (it should be — 12 entries indexed by `ACT_VALID_TARGETS`). Fails fast if not.

### 10.4 Clean (stale leading frames + quat sign)

```bash
pixi run python my_policy/scripts/act/clean_act_dataset.py --src v9_act_merged --dst v9_act_merged_clean
```

Two fixes (both required, both implemented in `dataset_io.patch_stale_leading_actions` + builder quat handling):

1. **Stale leading frames** (~0.07% of frames in batch_100_a; 27/9 episodes affected). Recorder occasionally captures the previous trial's `pose_command` at the start of a new episode. Detection: `||action.position − state.position|| > 50 mm`. Fix: overwrite leading bad frames with first-good action.
2. **Quaternion sign canonicalization.** ~45% of raw frames have `dot(state_q, action_q) < 0` in base_link. The `rotmat_to_quat_xyzw` w≥0 convention auto-handles most (drops to ~0.18% post-transform), but two quats can both have w≥0 and still be in opposite hemispheres. Builder explicitly negates action quat post-transform when `dot(state_q_port, action_q_port) < 0`. Without this, the v1 "wrist locked" failure returns.

### 10.5 Port-local transform (optional)

```bash
pixi run python my_policy/scripts/make_port_local_dataset.py --src v9_act_merged_clean --dst v9_port_local_dataset
```

Re-expresses TCP pose, velocity, action, wrench in the **target port's frame** (collapses world-frame variance for IL). Schema marker: `info.json:frame_transform = "port_local"`. Channel names match `v9_act_merged_clean` exactly.

**⚠ Past failure: training MAE wins, live inference locks at hover.** Port-local achieved 0.433× position-MAE vs world-frame at 10k steps (per-dim quat 100–200× better) but the live policy locks at port-local TCP_z ≈ −0.10 to −0.13 (5–7 cm above port) and never commits to descent. Lock survived multiple checkpoints (10k/50k/100k), chunk sizes (100/20), temporal ensembling (0.01–1.0), and explicit velocity-Z bias injection. Branch `v9-port-local` is preserved for reference; **do not pursue port-local without a new mitigation strategy** — chunk-size, TE, and single-channel injection sweeps are exhaustively negative. Full post-mortem in `project_aic_port_local.md` memory.

### 10.6 SmolVLA variant (optional, derived from port-local)

```bash
pixi run python my_policy/scripts/smolvla/make_smolvla_dataset.py --src v9_port_local_dataset --dst v9_port_local_smolvla_dataset
```

Slices observation.state from 44 → 26 dims (drops tcp_error[13:19] + task one-hot[32:44]). Sets `frame_transform = "port_local_smolvla_no_err"`. Symlinks `videos/`.

---

## 11. Critical gotchas (consolidated)

The seven things that have eaten the most time:

| # | Gotcha | Symptom | Fix |
|---|---|---|---|
| 1 | Forgot `ground_truth:=true` | Every trial scores 0; `pose_commands` empty in bag | Add to eval launch (see §6.1) |
| 2 | Zenoh router not started | `GetState service call timed out` after `Service is available` | Start `rmw_zenohd` first (see §3) |
| 3 | Stale `aic_model` zombies from prior crash | Engine `Found 1 node(s) ... aic_model` but it doesn't respond | `pkill -9 -f "aic_model --ros-args"` before next run; recorder does this at startup but verify if previous crash was unusual |
| 4 | Recording on laptop SSD | Cameras drop from 20 Hz → 2.4 Hz | Record on GPU box (NVMe) |
| 5 | TF frame name uses bare `port_name` | TF lookup silently fails | Use `task_board/<mount>/<port>_link` (e.g. `task_board/nic_card_mount_4/sfp_port_1_link`) |
| 6 | `tasks.parquet` not string-indexed | SmolVLA: `Task cannot be None` | Use `pd.DataFrame(..., index=pd.Index(..., name="task"))` (see §8.4) |
| 7 | Wrong port frame for visual cropping | Pixel projection lands inside the port body, not the visible mouth | Use `<port>_link_entrance`: SFP −45.8 mm along port-local −z, SC −15.64 mm. `demo_port_crop.PORT_ENTRANCE_OFFSET_M`. |

Plus two semantic confusions worth flagging:

- **Episode → trial mapping:** never use naive `trial_(ep+1)`. Discarded trials break positional alignment. Always use `my_policy.localizer.labels.match_episodes_to_trials(summary, cfg["trials"])`. The 2026-04-29 incident wasted ~20 hr chasing model architecture for a 15-line label-matching bug — `match_episodes_to_trials` had been using join key `(target_module_name, cable_name)` without `port_name`, so SFP `port_0`/`port_1` got confused; 15–50% of frames had labels off by 50–200 mm.
- **F/T sensor compensated wrench is NOT zero at rest** when a cable is in the gripper (see §7). Don't subtract it; it's signal.

---

## 12. Disk / time budget

- Native cam: 1152×1024 × 3 cams × 20 Hz ≈ **210 MB/s sustained**.
- Raw bag at 20 Hz @ 200 MB/s × 120 s/episode × 500 eps ≈ **12 TB if kept**.
- Strategy: `collect_lerobot.py` writes LeRobot v3.0 directly with downscaled 256×288 + AV1 video — no raw bag accumulation. Final dataset size for 500 episodes ≈ **250 GB**.
- A 500-trial batch wall-clock: ~2 hours (sim runs at ~0.1× RTF).
- For port-localizer training (not currently in the recorder): would need full-res JPEG q85 every 5th frame as a separate channel.

---

## 13. Anti-patterns (do NOT repeat)

For "what to try next time", see §5.5 (Recommendations for next-batch oracle changes). The list below is what to stop trying.

Compressed list of "we tried this and it didn't work":

1. **Smoothing the SC z-step in the recorded dataset (post-hoc).** Diagnosed 100 mm action[z] discontinuity at INSERT-phase start; built smoothstep5 backward ramp; retraining did not resolve SC failure. The most-prominent dataset anomaly is not always the cause of the model failure.
2. **Port-local frame transform.** 2.3× MAE win on training fit, locks at hover at inference (see §10.5).
3. **XY feedforward of `plug_tip_gripper_offset` in the oracle.** Unstable; cable lag → divergence.
4. **Lowering `FORCE_STOP_N` below 18 N.** Eats normal chamfer contact, causes hold-release oscillation.
5. **Freezing the full pose during a force hold.** XY misalignment never corrects. Keep the XY integrator live; freeze only Z.
6. **Naive `trial_(ep+1)` episode mapping.** 50–200 mm label errors.
7. **Architecture iteration before a label audit.** When a model fits train but not val (or fits neither and predicts marginal mean), audit labels on production-shaped data BEFORE any architecture / aug change. 90-min training runs aren't a substitute for a 5-min RMS-by-group label check.
8. **Quoting numbers from memory.** Re-read `info.json`, the source script that produced the dataset, and the live config before trusting any dim / rate / path. Memories drift; code is authoritative.

---

## 14. Quick-reference: full session in shell commands

(Replace `<batch>` consistently. All commands run inside the dev container unless noted.)

```bash
# ---------- T1: Zenoh router (leave running) ----------
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run ros2 run rmw_zenoh_cpp rmw_zenohd'

# ---------- T2 (host): generate config ----------
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/gen_trial_config.py --n 100 --seed 42 --task-type mixed --out /root/aic_data/<batch>.yaml'

# ---------- T3 (host): start eval engine ----------
cd ~/ssd/aic_workspace/aic_docker/aic && docker compose run --rm eval ground_truth:=true start_aic_engine:=true aic_engine_config_file:=/root/aic_data/<batch>.yaml

# ---------- T4 (dev container): recorder ----------
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/collect_lerobot.py --batch-config /root/aic_data/<batch>.yaml --root /root/aic_data/<batch>_run --repo-id local/<batch> --policy my_policy.ros.CheatCodeRobust --fps 20 --max-episode-s 40 --warm-up-s 180'

# ---------- Verify the batch ----------
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/inspect_dataset.py /root/aic_data/<batch>_run'

# ---------- Visualize one episode (rerun) ----------
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/viz_dataset.py --repo-id local/inspect --root /root/aic_data/<batch>_run --episode-index 0 --num-workers 0'

# ---------- Build pipeline ----------
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/act/inspect_act_demos.py --batch <batch> --out /root/aic_data/v9_act_build/<batch>_act_clean_episodes.json'
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/act/build_act_dataset.py --batch <batch> --clean-episodes-json /root/aic_data/v9_act_build/<batch>_act_clean_episodes.json'
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/act/merge_act_datasets.py --sources <batch_a>_act_dataset <batch_b>_act_dataset --out v9_act_merged'
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/act/clean_act_dataset.py --src v9_act_merged --dst v9_act_merged_clean'

# (Optional) port-local + smolvla variants
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/make_port_local_dataset.py --src v9_act_merged_clean --dst v9_port_local_dataset'
docker compose exec dev bash -c 'cd /root/ws_aic/src/aic && pixi run python my_policy/scripts/smolvla/make_smolvla_dataset.py --src v9_port_local_dataset --dst v9_port_local_smolvla_dataset'
```

---

## 15. Open questions to resolve before next batch

Things this guide doesn't yet have authoritative answers for; check these before recording:

- **Does the recorder need to log `wrench_raw` and `fts_tare_offset` separately?** Currently records the compensated wrench only. If we want a learned policy to model cable-tension residual, this needs adding to `aic_robot_aic_controller.py`.
- **Should `build_act_dataset.py`'s `tasks.parquet` writer be fixed to use string indexing?** Currently only fixed in `make_port_local_dataset.py`. If you plan to train SmolVLA on a non-port-local dataset, fix `build_act_dataset.py` first.
- **Does the recorder need full-res JPEG every 5th frame?** Required for any future port-localizer training. Not currently in the recorder.
- **Should `shm_size: '8gb'` be added to docker-compose.yml?** Long-overdue durable fix for DataLoader SIGBUS; currently worked around with `--num-workers 0` everywhere.
- **Does the SC oracle hover→descend transition need to be smoothed in the oracle itself** (rather than post-hoc)? Post-hoc smoothing did not fix the SmolVLA failure mode — but that doesn't rule out an in-oracle change being part of the right answer. Decide based on the next training experiment's failure analysis, not on the assumption that "discontinuity = cause".
