# v9-act — End-to-End ACT Policy Plan

**Branch:** `v9-act` (forked from `development`).
**Status:** PLAN — no code yet. Review before implementing.
**Driving observation (2026-04-30):** v7 localizer at xy_mean=11mm + chamfer (3mm) + spiral (5mm) → ~0% insertion rate in bench. v8 destabilized; v9-pathway diverged. The localizer-then-CheatCodeRobust decomposition has a compositional-error ceiling that incremental localizer accuracy probably cannot crack — you would need single-mm xy AND single-degree yaw simultaneously, neither of which the architecture has demonstrated. End-to-end ACT collapses localizer error + chamfer absorption + spiral search budget into one trained policy with no decomposition seam, which is the only intervention that addresses the actual failure mode.

## Goal

Train a single network mapping `(3 wrist-cam images, TCP pose, force/wrench, structured 12-dim task vector) → action chunk (K future 7-D pose targets)`. Replace `CheatCodeRobust.insert_cable` at submission time. Target: ≥40% insertion rate on a held-out 20-trial set (vs current ground-truth oracle ~50-70%, vs v7-localizer 0%).

## What we're not doing

- Custom transformer implementation. LeRobot's `ACTPolicy` is mature, integrated with the recorder dataset format, already used by `aic_example_policies/aic_example_policies/ros/RunACT.py` (the official example). Roll our own only if LeRobot's ACT cannot accommodate task conditioning.
- Force-model VAE / diffusion / world-model variants. ACT is the proven baseline for this kind of high-precision admittance-controlled task; escalate only if vanilla ACT plateaus.
- Joint-space actions. Stay with 7-D pose targets matching what the recorder saved and what the controller's MODE_POSITION expects.

## Phase A — Data prep (~2-3 hr)

The recorder already saved 354 episodes / ~172k frames across `batch_100_a..e` with the right images and a 7-D pose action. Three gaps to close before we can `lerobot train`:

### A0 (optional, recommended): Inspect demos for clean-vs-messy insertion patterns

ACT can only learn what's in the demos. Of the 354 saved trials, some inserted via a clean approach-align-descend sequence; others recovered via the force-gate retreat cycle and/or the spiral search. The messy ones teach the model to wiggle arbitrarily during INSERT — a real failure mode.

Write a one-shot inspection script `my_policy/scripts/inspect_act_demos.py` that walks each saved episode's actions + force history and tags it as:
- **clean**: zero force-gate engagements (peak force never crossed `FORCE_STOP_N=18N` for >250ms), zero spiral activations.
- **messy**: had at least one force-gate retreat cycle or spiral engagement.

Output: `clean_episodes.json` / `messy_episodes.json` (per-batch tag lists) + a summary table (clean vs messy counts per task type).

**Decision after running:** if clean ratio is ≥50% (i.e. ≥175 clean episodes), train on clean only. If <50%, train on all 354 and accept that ACT will inherit some messy patterns. Either way, this is data we want before committing to a 4-5 hour training run. ~30 min to write + 5 min to run.

### A1: Add task vector to `observation.state` for ACT training

The 47-dim `observation.state` saved by the recorder includes TCP pose, joint positions, wrench, ground truth port pose, etc. — but no task identity. ACT must know which port to insert into.

**Why we can't reuse the localizer's 7-dim one-hot.** `TASK_ONE_HOT_ORDER` in `my_policy/localizer/labels.py` indexes only `target_module_name` (5 NIC mounts + 2 SC modules = 7). That works for the localizer because its output is the BOARD POSE — port-agnostic — and the per-port offset (~2cm between sfp_port_0 and sfp_port_1 on the same NIC card) is added downstream by `reconstruct_port_in_baselink` using URDF math, which takes `port_name` as a separate argument outside the network.

ACT can't do that. ACT outputs ACTIONS directly. The trajectory for "insert into nic_card_mount_0/sfp_port_0" ends ~2cm away from the trajectory for "nic_card_mount_0/sfp_port_1" — same module, different port → totally different action sequence. So ACT's task vector must encode `port_name` too.

**Actual ACT task space — 12 distinct (target_module, port_name) pairs:**

```
SFP: nic_card_mount_0..4 × sfp_port_0/sfp_port_1   (5 × 2 = 10)
SC:  sc_port_0..1 × sc_port_base                    (2 × 1 =  2)
                                                    ────────
                                                          12
```

(`cable_name` adds further variation in `noise_bench.yaml` — cable_0 / cable_1 — but cable identity affects only the gripper-side appearance which the model can infer from images. Not encoded explicitly.)

**Encoding choice — structured 12-dim** rather than flat 12-dim one-hot:

```
target_module_one_hot (7) || port_in_module_one_hot (3) || port_type_one_hot (2)
   nic_card_mount_0..4         sfp_port_0                     sfp
   sc_port_0..1                sfp_port_1                     sc
                               sc_port_base
```

12 dims total. Rationale:
- **Structured** — port_0 / port_1 within an SFP mount has consistent geometry across mounts (port_0 always at +1.1cm in mount-local frame, port_1 always at -1.2cm — see `SFP_PORT_OFFSET_IN_MOUNT` in labels.py). Structured encoding lets the model share parameters across "any sfp_port_0", which a flat 12-dim one-hot wouldn't.
- **Explicit `port_type`** — SFP and SC have different chamfer geometries, different insertion depths, different stuck-recovery patterns (CheatCodeRobust's spiral mode is per-plug-type — `x_only` for SC, `circular` for SFP). The model could derive port_type from the module name, but explicit conditioning removes the inference burden and is one extra dimension.

Define this as `ACT_TASK_VECTOR_LAYOUT` in a new module `my_policy/my_policy/act/labels.py` (parallel to `localizer/labels.py`). New file rather than extending the localizer's labels.py because the vector is deliberately ACT-specific and the localizer code shouldn't grow a dependency on it. Encoded deterministically from the known module/port topology — future port additions = explicit edit, never auto-derived from a YAML.

**Decision:** add the 12-dim task vector at training-time as a dataset transform (cheaper than re-recording). Build a preprocessing script `my_policy/scripts/build_act_dataset.py` that:
- Reads the YAML + summary.json for each batch.
- Joins via `match_episodes_to_trials` (already in `my_policy/localizer/labels.py`) to map episode_index → `(target_module_name, port_name, port_type)`.
- Emits the 12-dim structured task vector per frame using `ACT_TASK_VECTOR_LAYOUT`.
- Concatenates into `observation.state` (or a new `observation.task` field if LeRobot's ACT supports separate channels — TBD per LeRobot ACT config).

Emits a NEW LeRobot dataset (don't mutate the recorder output). Reads from the existing dataset, writes a new one with augmented state. ~1 hr.

**Output state vector** (proposed, 44-dim):
- TCP position (3) + TCP orientation xyzw quat (4)
- TCP linear velocity (3) + TCP angular velocity (3)
- TCP error (6) — `current_tcp − reference_tcp`, where `reference_tcp` is the most recently commanded pose. This is *lagged* tracking info, not the action we're predicting (the action is the NEXT pose; tcp_error reports against the PREVIOUSLY commanded pose), so it's not action leakage in the literal sense. Likely useful for force-gate / chamfer-contact disambiguation. Earlier draft excluded this on weak leakage grounds — included now.
- F/T wrench tare-compensated (6)
- Joint positions (7) — wrist cams are wrist-mounted; joint config tells the model where they are
- Task vector — structured 12-dim (`module(7) || port_in_module(3) || port_type(2)`)
- **Excluded:** `groundtruth.port_pose.*` (label leakage — the policy is supposed to find this from images), `meta.insertion_success` (label leakage).

Per-channel `observation.state.names` list maintained in dataset metadata so the inference shim can re-derive the layout.

**Belt-and-suspenders task metadata.** Also populate the per-episode `task` string field that LeRobotDataset v3.0 supports natively (e.g. `"insert sfp_port_0 into nic_card_mount_3"`). LeRobot's stock ACT v0.5.1 does NOT consume this field — it's used by language-conditioned policies (SmolVLA, pi0). But populating it costs nothing in the preprocessor and means if we ever want to escalate to a language-conditioned policy on the same dataset, the labeling is already in place. The 12-dim structured vector in `observation.state` remains the primary conditioning channel for ACT.

**Episode-level train/val split.** Same trap we hit with the localizer (label-leakage if frames from one episode appear in both sets — the board pose is constant per episode, so frame-level split is silent label leakage). Solution: use the localizer's `episode_split(seed=42)` logic to produce explicit `train_episodes` / `val_episodes` Python lists. Save to JSON next to the new dataset (`train_episodes.json`, `val_episodes.json`). Pass `--dataset.episodes=...` to `lerobot-train` for the training run; pass the val list at validation time. Reusing seed=42 keeps the val set apples-to-apples with v7 / v9-pathway numbers.

**Open question (still pending decision):** whether 12 dims diluted into a 44-dim state vector is a strong enough signal, or whether we should add redundant conditioning (FiLM(task) on visual features + task token in the encoder) to make the task signal impossible to ignore. State-concat alone is the cheapest option; redundant conditioning is ~1-1.5 hr more code. Current plan is start with state-concat alone, escalate if val shows the held-out-task generalization fails.

### A2: Action format check

Recorder saves 7-D pose `(x, y, z, qx, qy, qz, qw)`. `ACTPolicy` uses **L1 loss** on actions (not MSE) — `F.l1_loss(actions, actions_hat)` per `modeling_act.py`. No per-channel weighting between rotation and translation; the 4 quaternion components are treated as 4 independent regression targets.

Implications:
- The q-vs-(-q) ambiguity exists for both L1 and MSE, but L1 is gentler. For our task the gripper quat distribution is narrow (mostly z-down with small yaw variation), so this should be fine in practice.
- L1 doesn't enforce unit-norm. Predicted quat will have mild non-unit norm (probably 0.97-1.03). Inference (Phase C) must normalize before commanding via `set_pose_target` — see §C.
- Backup: if val loss won't shrink, switch to `(x, y, z, sin_yaw, cos_yaw, sin_roll, cos_roll, sin_pitch, cos_pitch)` (9-D) or 6D rotation representation. Don't pre-build — re-check after first training run.

### A3: Sanity check on dataset

Run the existing `inspect_dataset.py` on the new ACT-format dataset:
- Confirm 354 episodes preserved.
- Confirm action.std non-zero (no constant actions).
- Confirm `observation.state` is 44-dim and per-channel `names` list matches the layout above.
- Confirm each of the three task sub-vectors sums to 1 per frame (`module` slice sums to 1, `port_in_module` slice sums to 1, `port_type` slice sums to 1) and is constant within an episode.
- Confirm full task-vector coverage: every (target_module, port_name) pair that appears in the source YAMLs has at least one episode in the dataset. If port_in_module=sfp_port_1 has zero episodes for some mount, the model can't learn that case.
- Confirm `train_episodes.json` + `val_episodes.json` partition the episodes disjointly and cover all 354 between them.
- Confirm per-episode `task` string field is populated for all 354 episodes.

## Phase B — Training (~4-6 hr training time + ~2 hr setup)

### B1: LeRobot ACT config

Pass via `lerobot-train` CLI (the canonical entrypoint per the AIC `lerobot_robot_aic` README; lerobot v0.5.1 doesn't take a config-file path the way the localizer training script did). Key hyperparams:

**Action chunking — pick ONE of the two valid combos:**
- **`temporal_ensemble_coeff=None`, `n_action_steps=8`** (RECOMMENDED for v0): standard chunked execution. Predicts a 32-action chunk, executes the first 8 (~400ms at 20Hz), re-queries. Lower inference work.
- `temporal_ensemble_coeff=0.01`, `n_action_steps=1`: query policy every tick (50ms), exponentially-weighted average of overlapping chunk predictions. Smoother but ~5× more inference. Use only if first run shows action jitter.

The combination `(temporal_ensemble_coeff=0.01, n_action_steps=8)` is **rejected** by lerobot's config validator (`configuration_act.py:106-109` raises `NotImplementedError` — temporal ensembling requires `n_action_steps=1`).

**Other hyperparams:**
- `policy.type=act`
- `policy.chunk_size=32` (1.6 s at 20 Hz). LeRobot default is 100 (5 s); we want fast reaction for force-gate / spiral situations. If model plateaus, try 16 or 64.
- `policy.vision_backbone="resnet18"` (matches v7 backbone we already validated). LeRobot's ACT enforces `vision_backbone.startswith("resnet")` — DINOv2 / ViT backbones would require forking the policy class. Stay with ResNet18.
- `policy.dim_model=512`, `policy.n_heads=8`, `policy.n_encoder_layers=4`, `policy.n_decoder_layers=1` (LeRobot ACT defaults).
- `policy.kl_weight=10.0` (LeRobot default).
- `policy.optimizer_lr=1e-5`, `policy.optimizer_lr_backbone=1e-5`, `policy.optimizer_weight_decay=1e-4` (ACT defaults — fine for ResNet18 fine-tuning).
- `training.steps=100_000` (start; revisit based on val loss; LeRobot docs estimate "~few hours on a single GPU" for this).
- `training.batch_size=8` (start; ACT uses lots of GPU memory due to action-chunk loss).
- `dataset.repo_id=local/aic_act_<batch_set>`, `dataset.root=/root/aic_data/aic_act_<batch_set>` (the new dataset built in Phase A).
- `dataset.episodes=<train_episodes.json contents>` (passes the explicit episode-level train list).
- `dataset.image_transforms.enable=true` (lerobot's built-in augmentation: brightness/contrast/saturation/hue ColorJitter + sharpness + ±5° affine. Defaults to OFF; we MUST turn it on. With 354 episodes × heavily-correlated within-episode frames, the effective sample size is far below the 172k frame count, and v9-pathway diverging at epoch 12 is direct evidence of how easily this kind of dataset overfits without aug). Tune `image_transforms.max_num_transforms` (default 3) if needed.
- `dataset.use_imagenet_stats=true` (default; matches our ResNet18 backbone — leave alone).

### B2: Training command

Canonical `lerobot-train` invocation per the AIC `lerobot_robot_aic` README and the official lerobot ACT docs (single-line — terminal copy-paste mangles backslash continuations):

```
pixi run lerobot-train --dataset.repo_id=local/aic_act --dataset.root=/root/aic_data/aic_act_dataset --dataset.image_transforms.enable=true --policy.type=act --policy.chunk_size=32 --policy.n_action_steps=8 --output_dir=/root/aic_data/act_run_1 --job_name=v9_act_run_1 --policy.device=cuda --steps=100000 --batch_size=8
```

(Substitute the real `--dataset.episodes` flag for the train list once we know the JSON syntax — verify by running `lerobot-train --help` once the env is ready.)

Output: `<output_dir>/checkpoints/last/pretrained_model/` containing `model.safetensors`, `config.json`, `policy_preprocessor_step_*_normalizer_processor.safetensors` (matches what `aic_example_policies/aic_example_policies/ros/RunACT.py` already loads — same naming convention).

### B3: Watch criteria

LeRobot's ACT training logs:
- L1 action loss (decreases steadily). This is `F.l1_loss(action, action_hat)` masked by `action_is_pad`.
- KL loss (style-VAE regularizer — should converge to a small value, ~0.01-0.1 range). Total loss is `l1_loss + kl_weight * kl_loss`.
- Val loss every N steps (computed on the held-out episodes from `val_episodes.json`).

**Stop conditions:**
- L1 action loss plateaus and val loss starts climbing → early-stop, take best ckpt.
- After 50k steps if val loss hasn't dropped meaningfully → revisit hyperparams (chunk_size, kl_weight) or architecture.

**Conditioning-strength diagnostic** (run once mid-training, e.g. at 50k steps):
Hold out one specific (target_module, port_name) pair entirely from training (e.g. `nic_card_mount_3 + sfp_port_1`) and evaluate the model on val episodes from that held-out task. If the model handles other (mount, port_0) tasks fine but completely fails on the held-out (mount_3, port_1) — that's the canary that the structured 12-dim task vector is being ignored. Trigger to escalate to FiLM(task) on visual features + task-as-token.

## Phase C — Inference (~2 hr)

Adapt `RunACT.py` into our `my_policy/my_policy/ros/RunACT.py`. Differences from the example:
- Load OUR trained checkpoint (path from env var `MY_POLICY_ACT_CHECKPOINT`, fallback to a bundled path under `weights/` in the docker image), not the example's `grkw/aic_act_policy` HuggingFace repo.
- Build the same 44-dim state vector at inference time (matching A1's training format), including the 12-dim structured task vector derived from the live `Task` message via the same `ACT_TASK_VECTOR_LAYOUT` used at training time. Import the layout from `my_policy/my_policy/act/labels.py` so training and inference can never drift.
- Use `set_pose_target` (MODE_POSITION) with the predicted 7-D pose, not `set_cartesian_twist_target`. Match the recorder's training distribution.
- **Normalize the predicted quaternion before commanding.** ACT uses L1 loss on each of the 4 quat components independently — predictions will have mild non-unit norm (probably 0.97-1.03). Without `q ← q / ||q||`, the controller interprets a 0.99-norm quat as rotation + slight scaling and the admittance loop misbehaves. One line; do not skip.
- Inference loop: 20 Hz (50 ms tick), max 40 s per task (matches engine default).
- `policy.select_action(obs)` handles chunk caching internally (and temporal ensembling if enabled). With `n_action_steps=8` we don't need to manage chunks ourselves; with `temporal_ensemble_coeff=0.01, n_action_steps=1` it auto-applies the exponential weighting.

Lifecycle correctness:
- Construction: loads ACTPolicy (~200 MB); ~2 s.
- `on_configure`: instantiate policy class (already does the heavy load).
- `on_activate`: subscribe to insertion_event topic for early exit.
- `insert_cable`: the loop above. Honors `_should_abort()` between ticks.

### Early-exit when insertion fires

Subscribe to `/scoring/insertion_event` (same QoS profile as CheatCodeRobust). When insertion event fires, exit `insert_cable` cleanly with `return True` instead of running out the time limit. Saves bench-cycle time and avoids continuing to publish actions after success.

## Phase D — Bench (~1-2 hr)

### D1: Smoke test (5 trials, ground_truth=true)

Use `collect_lerobot.py` with `--policy=my_policy.ros.RunACT.RunACT` (or whatever class path) to log the run. Watch the model.log for:
- Policy loaded successfully (`config.json` parsed, weights + normalizer loaded).
- Per-tick action values are sensible: xyz within ~50cm of board centroid, raw predicted quat magnitude in [0.95, 1.05] (with normalization applied before publish), no NaNs.
- Insertion events fire on at least 1-2 of the 5 trials. (Same-task trials — D1 is checking that the inference loop runs end-to-end, not generalization.)

If the policy commands wildly off-board actions (e.g., drives the robot into the table) or quat magnitude is way off (<0.5 or >2.0): kill, debug. Likely a state-layout mismatch (channel order or dim count differs between training preprocessor and inference preprocessor) or a normalizer-stats mismatch.

### D2: Real bench (20-trial mixed batch, ground_truth=false)

Once D1 passes, run on a fresh `gen_trial_config.py` batch with `ground_truth:=false` (submission-shaped). Measure:
- Insertion rate (target ≥40%).
- Per-trial outcome breakdown (SFP vs SC).
- Time-to-insertion distribution (vs CheatCodeRobust's typical 15-30s).

Compare to:
- v7 localizer + CheatCodeRobust: ~0% (just measured).
- CheatCodeRobust + ground-truth TF: ~50-70% (oracle ceiling).

A score in [40%, 60%] would be a clear win for ACT — most of the ground-truth oracle's success without TF dependency.

## Phase E — Submission packaging (~1 hr)

If D2 hits target:
- Add `weights/v9_act/` to `aic_main/docker/aic_model/Dockerfile` `COPY`. ~200 MB; verify image size budget.
- Set `MY_POLICY_ACT_CHECKPOINT=/weights/v9_act/pretrained_model` in the docker compose `model` service env.
- Update `docker/docker-compose.yaml` `command:` to use `RunACT` instead of CheatCode.
- Verify `docker run --network none` works (no internet at submission).
- Push to ECR / submission registry.

## Risks (and mitigations)

| Risk | Likelihood | Mitigation |
|---|---|---|
| 354 demos too few; ACT overfits | medium | Enable lerobot's built-in `dataset.image_transforms.enable=true` (brightness/contrast/saturation/hue/sharpness/affine) — disabled by default but available. Direct evidence we need it: v9-pathway diverged at epoch 12 on the same data. Collect more demos if val still won't drop (~6 hr per 100 demos). |
| Task vector ignored — model collapses to "any port" or averages tasks. Specifically dangerous for sfp_port_0 vs sfp_port_1 disambiguation on the same NIC card. | medium | Hold out one (target_module, port_name) pair from training, see if it generalizes (the held-out-task diagnostic in B3). If not, escalate to FiLM(task) on visual features + task-as-token in the encoder. |
| Predicted quaternion mildly non-unit-norm (L1 loss treats components independently) | high (this WILL happen) | Normalize at inference before commanding via `set_pose_target` — see §C. One line. Without it the admittance loop misbehaves. |
| Quaternion regression yields nonsense rotations (val won't shrink) | low (gripper mostly z-down) | Switch to 6D rotation representation if val loss won't shrink. Don't pre-build. |
| Action chunk size wrong — too long = laggy reaction; too short = no benefit over BC | medium | Start at 32; sweep {16, 32, 64} if first run is bad. |
| LeRobot ACT requires features we don't have | low | Confirmed via web research: ACT v0.5.1 accepts arbitrary state/action dims via `input_features`/`output_features`, requires only that `vision_backbone.startswith("resnet")` and that all images share the same shape (we satisfy both). |
| Force-gate / spiral recovery patterns confuse the model — it learns to "wiggle" arbitrarily | medium | Filter dataset: keep only successful trials (already done by recorder). Optionally further filter for "clean inserts" (zero force-gate engagements, zero spiral activations) — see §A0. |
| Submission docker bloat | low | Only one model to bundle; weights ~200 MB plus LeRobot deps. |
| Per-tick inference latency too high for 20 Hz | low | ResNet18 + small transformer should run in ~10-20 ms on L4. Profile during D1. With `n_action_steps=8` we only do this every ~400ms which is comfortable. |
| Frame-level vs episode-level val split silently inflates val metrics | medium → mitigated | Pass explicit `train_episodes`/`val_episodes` lists via `--dataset.episodes`; reuse the localizer's seed=42 split for apples-to-apples comparison. |

## File changes (planned)

### New
- `my_policy/my_policy/act/__init__.py`, `my_policy/my_policy/act/labels.py` — new subpackage holding `ACT_TASK_VECTOR_LAYOUT` and the `(target_module, port_name) → 12-dim vector` builder. Imported by both training preprocessor and inference shim so they can never drift.
- `my_policy/scripts/inspect_act_demos.py` — Phase A0 clean-vs-messy demo classifier; writes `clean_episodes.json` / `messy_episodes.json` per batch.
- `my_policy/scripts/build_act_dataset.py` — one-shot preprocess: existing recorder dataset → ACT-augmented dataset (44-dim state with the 12-dim task vector + tcp_error included; per-episode `task` string populated; `train_episodes.json` / `val_episodes.json` written using seed=42 episode split).
- `my_policy/my_policy/ros/RunACT.py` — our inference shim (forked from `aic_example_policies/aic_example_policies/ros/RunACT.py`, load our checkpoint, use pose targets, add task conditioning, normalize predicted quat).
- `my_policy/scripts/test_run_act_inference.py` — host-runnable smoke test mocking the framework, verifies action shapes / no NaNs / quat magnitude ≈ 1 after normalization.

### Modified
- `aic_main/docker/aic_model/Dockerfile` — bundle weights + use RunACT (Phase E only).
- `aic_main/docker/docker-compose.yaml` — change CMD policy class.

### Untouched
- `my_policy/my_policy/localizer/` — v9-act doesn't replace the localizer code; it just doesn't use it. Keep around for v9-pathway / v9-dino branch comparison.
- `my_policy/my_policy/ros/CheatCodeRobust.py` — keep as the data-collection oracle.
- Recorder (`collect_lerobot.py`) — no changes. We post-process.

## Estimated total effort

| Phase | Hours |
|---|---|
| A — data prep (incl. A0 inspect, A1 preprocess + episode split, A2 action check, A3 sanity) | 2.5-3.5 |
| B — training (incl. wait) | 6-8 (~4-5 wall hours of GPU + setup) |
| C — inference shim | 2-2.5 (extra 30 min for quat normalization + state-layout consistency check) |
| D — bench | 1-2 |
| E — submission packaging | 1 |
| **Total** | **12.5-17 hours** (~1.5-2 days realistic with debugging) |

## Decision points before coding starts

1. **Use LeRobot's `ACTPolicy` directly, or fork it?**
   Recommend: use directly. LeRobot has been maintaining ACT actively; forking creates maintenance burden and risks.

2. **Task vector encoding: structured 12-dim (`module || port_in_module || port_type`) vs flat 12-dim one-hot?**
   Recommend: structured. Lets the model share parameters across "any sfp_port_0" since geometry is consistent across mounts. Locked in §A1.

3. **Conditioning strength: state-concat alone vs add FiLM(task) on visual features + task-as-token?**
   Recommend: state-concat alone for first run, with the held-out-task diagnostic in §B3 as the trigger to escalate. Cheapest path; ~1-1.5 hr to add the redundant signals later if signal is being ignored. Specific failure to watch for: model handles `nic_card_mount_X/sfp_port_0` correctly but conflates port_0 and port_1 on the same mount — that's the case where 12 dims diluted in 44-dim state isn't enough.

4. **Action representation: 7-D pose direct, or 9-D `(xyz, sin/cos × 3)`?**
   Recommend: 7-D direct first. Our gripper rotation distribution is narrow enough that L1 loss on raw quat components should work. Escalate only if val loss won't shrink. Don't pre-build the alternative.

5. **Chunk size: 16 / 32 / 64 / 100?**
   Recommend: 32. Compromise between reactivity and the ACT paper's K=100 default.

6. **Train from all 354 episodes, or split SFP/SC and train two policies?**
   Recommend: one policy trained on all 354 with the structured 12-dim task vector. Two-policy approach loses cross-task transfer learning (insertion mechanics shared) AND complicates inference dispatch.

If any of these need to flip before implementation, surface now.
