# v9-port-local-smolvla — Plan

## Why SmolVLA

ACT @ 100k steps converged to 2.3× lower val pos-MAE than world-frame ACT (0.433 ratio confirmed). But Stage A live bench revealed an **OOD-lock failure mode**: the policy holds at hover (~5cm above port) and never commits to descent, despite the val-set diagnostic showing perfect descent prediction on training-distribution observations. Temporal ensembling (0.01 → 1.0) and checkpoint sweeps (50k vs 100k) didn't break it; chunk_size=20 retrain is queued as the ACT-side variant.

SmolVLA is the alternative architectural bet. Hypothesis:
- **Language-conditioned VLA** treats the task instruction (`"Insert the sfp plug into sfp_port_0 on nic_card_mount_3"`) as first-class input. The dataset already populates `tasks` per-episode — verified in memory `project_aic_act_dataset.md`.
- **Pretrained SmolVLM2-500M backbone** brings vision priors that ACT's ResNet18-from-scratch lacks. Less likely to lock on the narrow training distribution.
- **Flow-matching action head** (default num_steps=10) generates actions via iterative denoising rather than a single chunk-projection — different inductive bias around stuck states.
- **Larger model** (~500M params vs ACT's 85M) — more capacity for the multi-task port-local setting, at the cost of compute.

Risks: training is slower, VRAM heavier, more hyperparameters to manage, weights bundle is ~900 MB (vs ACT's ~300 MB).

## Verified facts about SmolVLA in our lerobot install

Read 2026-05-09 from `lerobot/policies/smolvla/configuration_smolvla.py`:

| Field | Default | Note |
|---|---|---|
| `chunk_size` | **50** | 2.5 s lookahead at 20 Hz (vs ACT's 100/5s). |
| `n_action_steps` | **50** | Default uses full chunk before re-encoding. |
| `n_obs_steps` | 1 | Markov in observation, like ACT. |
| `max_state_dim` | **32** | ⚠️ **Our state is 44-dim** — see [Risk: state dim](#risk-state-dim). |
| `max_action_dim` | 32 | Our action is 7-dim, pads to 32 — no issue. |
| `tokenizer_max_length` | 48 | Task instruction max tokens. Our strings (~10 tokens) fit easily. |
| `num_steps` | 10 | Flow-matching denoising steps per action. |
| `vlm_model_name` | `"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"` | Pretrained vision-language backbone. |
| `load_vlm_weights` | False (default) | ⚠️ Default trains expert from scratch. **Set True** to init from SmolVLM2 pretrained weights. |
| `freeze_vision_encoder` | True | Default freezes SmolVLM2 — only action expert trains. ~250M trainable. |
| `train_expert_only` | True | Same idea. |
| `optimizer_lr` | 1e-4 | Standard. |
| `scheduler_warmup_steps` | 1000 | Linear warmup then decay. |
| `scheduler_decay_steps` | 30000 | LR cosine-decays to `scheduler_decay_lr=2.5e-6` over this many steps. |
| `resize_imgs_with_padding` | (512, 512) | ⚠️ Pads our 256×288 cameras up to 512×512 — much larger than ACT's input. |
| `attention_mode` | `cross_attn` | Action expert cross-attends to VLM features. |

⚠️ Need to verify at code time: does `load_vlm_weights=True` require internet at training time? Yes — pulls from HF Hub. At eval / submission time the weights need to be cached or bundled.

## Risk: state dim

`max_state_dim = 32` but our `observation.state` is 44-dim (TCP pose 7 + tcp_vel 6 + tcp_err 6 + joints 7 + wrench 6 + task_vec 12 = 44). Two paths:

1. **Bump `max_state_dim` to 64** (or 48) in the SmolVLAConfig override. Verify the model's state-projection layer adapts. If it's parameterized by `max_state_dim`, this works. If it's hard-baked, we'd need to inspect the state-projection module.
2. **Drop the redundant 12-dim task one-hot** from observation.state (since SmolVLA already gets the language instruction). State becomes 32-dim: TCP pose 7 + tcp_vel 6 + tcp_err 6 + joints 7 + wrench 6 = 32. **Cleanly fits the default max_state_dim.**

Path 2 is preferred — simpler, model-default friendly, and avoids redundancy (one-hot ↔ language describe the same task). Action item: write a `make_smolvla_dataset.py` (or a flag to `make_port_local_dataset.py`) that produces a 32-dim observation.state by skipping the task one-hot append. The natural-language `task` field per episode (already populated in our dataset) is what SmolVLA uses for conditioning.

## What's reusable vs new

**Reusable as-is (no changes):**
- `my_policy/my_policy/port_local/{transforms,dataset_io}.py` — frame transform math.
- `make_port_local_dataset.py` — port-local dataset builder. Either:
  - Add a `--state-mode={act,smolvla}` flag that controls whether the 12-dim task vector is appended (default ACT, opt-in SmolVLA).
  - Or write a thin wrapper script `make_smolvla_dataset.py` that calls the builder then re-projects state to 32-dim.
- `merge_act_datasets.py`, `clean_act_dataset.py`, `build_port_local_all.py` — all schema-agnostic; pass through.
- `localizer/projection.py`, `localizer/labels.py` — pure utility, no policy dependency.

**Adapt (mostly mechanical):**
- `eval_offline_action_mae.py` — currently loads `ACTConfig` + `ACTPolicy` hard-coded. Make policy-class agnostic via a `--policy-class={act,smolvla}` flag, OR clone to `eval_offline_action_mae_smolvla.py`. Decision: **clone** to keep ACT path untouched and avoid risk to the existing decision-quality numbers.
- `diagnose_episode_predictions.py` — same pattern. Clone to `diagnose_smolvla_predictions.py`.
- `compare_eval_runs.py` — already policy-agnostic (reads JSON only). No changes.

**New:**
- `train_smolvla.py` — SmolVLA training driver. Mirrors `train_act.py` structure but with SmolVLA-specific config (LR schedule, freeze strategy, pretrained weights toggle).
- `RunSmolVLA.py` — inference shim under `my_policy/my_policy/ros/`. Loads SmolVLAPolicy, runs the same port-pose acquisition + state composition (32-dim, no task-vec append) + action transform pipeline as `RunPortLocalACT.py`. Critical difference: feeds the language instruction per call.
- `test_runsmolvla_offline.py` — Tier 1/2/3 tests adapted for SmolVLA (state shape 32 vs 44, language input mock).

## Architecture diagram

```
Training:
   v9_port_local_merged_clean (44-dim state, language `task` field per ep)
       │
       ▼ (state slicing: drop indices [32..43] task_vec)
   v9_port_local_smolvla_dataset (32-dim state, same language field)
       │
       ▼ train_smolvla.py
   v9_pl_smolvla_v1/checkpoints/<step>/pretrained_model/
       │
       ▼
   eval_offline_action_mae_smolvla.py → val MAE JSON
   diagnose_smolvla_predictions.py → per-frame trajectory diff

Inference (live):
   eval container Observation msg
       │
       ▼ RunSmolVLA._build_state_32 (port-local, no task-vec slot)
       ▼ RunSmolVLA._build_obs_dict (images + state + language string)
   SmolVLAPolicy.select_action
       │
       ▼ port-local action [7]
       ▼ transforms.transform_pose_back_to_baselink
   set_pose_target → controller
```

## Step-by-step plan

### Phase 0: Preflight (post-compact)

These gates must pass before Phase 2 training. **If gate 1 fails, the entire branch is wasted at submission time.**

1. **🚨 Submission-time HF Hub gate**: `SmolVLAPolicy(config)` may call `AutoModel.from_pretrained(vlm_model_name)` in its constructor (via `smolvlm_with_expert.py`) even when we immediately `load_state_dict(safetensors)` over it. The eval container has no internet — if the constructor needs HF Hub, the submission will dead-on-load. Test post-smoke-train with internet disabled:
   ```
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pixi run python -c "
   import json, draccus
   from safetensors.torch import load_file
   from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
   from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
   ckpt = '<path>/pretrained_model'
   cfg = json.loads(open(f'{ckpt}/config.json').read()); cfg.pop('type', None)
   p = SmolVLAPolicy(draccus.decode(SmolVLAConfig, cfg))
   p.load_state_dict(load_file(f'{ckpt}/model.safetensors'))
   print('OK')
   "
   ```
   If it fails: bundle HF cache via `HF_HOME` mounted into submission docker, OR override `load_vlm_weights=False` at inference (relying on the fine-tuned safetensors alone — verify model loads cleanly without VLM weights from hub).

2. **🚨 Dataset slot-equality gate**: byte-equal check that `make_smolvla_dataset.py` produced exactly the first 32 channels of the source:
   ```
   python3 -c "
   import pyarrow.parquet as pq, numpy as np
   src = pq.read_table('.../v9_port_local_merged_clean/data/chunk-000/file-000.parquet')
   sv  = pq.read_table('.../v9_port_local_smolvla_dataset/data/chunk-000/file-000.parquet')
   s_src = np.stack(src['observation.state'].to_pylist())
   s_sv  = np.stack(sv['observation.state'].to_pylist())
   assert np.array_equal(s_src[:, :32], s_sv), 'state slice mismatch'
   print('shapes:', s_src.shape, s_sv.shape, 'byte-equal: OK')
   "
   ```

3. **`load_vlm_weights=True` smoke**: kick off a 100-step training run to confirm HF Hub download works + the SmolVLA training loop converges (loss curve sane in the first 100 steps). Throwaway run.

4. **VRAM check** — SmolVLA-500M with frozen vision encoder fits comfortably in 24 GB at batch=4. On the local 48 GB box, batch=8 should work; tune down if OOM.

5. **Tier 1 offline tests on GPU box** — host can't run them (no torch). Run before kicking off the 100k:
   ```
   pixi run python my_policy/scripts/smolvla/test_runsmolvla_offline.py --skip-tier3
   ```

### Phase 1: Dataset (~1-2 hours)

1. Write `my_policy/scripts/smolvla/make_smolvla_dataset.py` — copies `v9_port_local_merged_clean` and rewrites `observation.state` to 32-dim (drops task vec). Keeps `tasks` strings per episode.
2. Verify with a tiny test: load both ACT and SmolVLA versions of one frame, confirm first 32 channels match exactly + new state has correct shape + episode `tasks` strings are present.
3. Output: `/root/aic_data/v9_act_build/v9_port_local_smolvla_dataset/`.

### Phase 2: Training (~1 day)

1. Write `my_policy/scripts/smolvla/train_smolvla.py` modeled on `train_act.py`. Key flags:
   - `--name v9_pl_smolvla_v1`
   - `--dataset-root <smolvla_dataset>`
   - `--steps 100000` (matching ACT's run for fair comparison; can extend)
   - `--batch-size` lower than ACT's 8 if VRAM tight (probably 4)
   - `--load-vlm-weights` (default true) — init from SmolVLM2 pretrained
   - `--freeze-vision-encoder` (default true)
2. Smoke test: 100 steps with `--dry-run` semantics (use `--steps 100`).
3. Full training: kick off 100k overnight.

### Phase 3: Inference shim (~2 hours)

1. Write `my_policy/my_policy/ros/RunSmolVLA.py` adapted from `RunPortLocalACT.py`. Key changes:
   - Load `SmolVLAPolicy` instead of `ACTPolicy`.
   - State composition is 32-dim (no task vec).
   - Pass `task` string to `select_action` (SmolVLA-specific input).
   - Same port pose acquisition (TF or localizer) + frame transforms.
2. Document env vars: `AIC_PL_SMOLVLA_CHECKPOINT`, `AIC_PL_SMOLVLA_TIMEOUT_S`, etc.

### Phase 4: Tests (~1 hour)

1. Write `test_runsmolvla_offline.py`. Tier 1+2+3 same shape as ACT version.
2. Tier 2 critical: load real port-local frame from raw batch, verify shim's state matches the smolvla-dataset's stored state byte-exact.

### Phase 5: Eval + bench (~3 hours)

1. Clone to `eval_offline_action_mae_smolvla.py` — swap ACT loader for SmolVLA.
2. Run val MAE on first checkpoint (e.g. 20k or 50k). Compare to ACT @ same step count via `compare_eval_runs.py`.
3. **Latency check on the live machine** (not just laptop): Tier 3 of `test_runsmolvla_offline.py`. SmolVLA's chunk-boundary call does 10 flow-matching denoising steps over a 500M-param model. On L4 it may exceed the 2.5 s queue-drain time of `n_action_steps=50`.
   - **Decision rule**: if Tier 3 p99 > 2500 ms, lower `--n-action-steps` (e.g. 25 → re-plan every 1.25 s). The chunk-boundary tick now stalls 1 tick instead of 50, but the controller doesn't starve.
   - If even 25 is too slow, drop to 10 (re-plan every 0.5 s) — SmolVLA's whole point is quality of plans, but a starved queue is unrecoverable.
4. If MAE + latency look reasonable, run live bench with `ground_truth=true` on `sc_tester_5.yaml`.
5. Compare to ACT v9_pl_v2's bench score (currently 120 with TE=0.25).

### Phase 6: Decision

If SmolVLA matches or beats ACT's 0.433 val-MAE ratio AND breaks the live OOD-lock → continue to Stage B (localizer integration, full bench, submission).
If SmolVLA is comparable but still locks live → may need DAgger or different fix.
If SmolVLA is worse → fall back to the chunk=20 ACT result.

## Key environment variables / paths

- Dataset (input to phase 1): `/root/aic_data/v9_act_build/v9_port_local_merged_clean/`
- Dataset (output of phase 1): `/root/aic_data/v9_act_build/v9_port_local_smolvla_dataset/`
- Training run (output of phase 2): `/root/aic_data/v9_act_build/runs/v9_pl_smolvla_v1/`
- Inference env vars (phase 3):
  - `AIC_PL_SMOLVLA_CHECKPOINT` — path to `.../checkpoints/<step>/pretrained_model/`
  - `AIC_PL_SMOLVLA_TIMEOUT_S` — per-trial inference budget
  - `AIC_PL_LOCALIZER_CHECKPOINT` — same convention as ACT shim, optional localizer override
- Policy spec for `aic_model`: `policy:=my_policy.ros.RunSmolVLA`

## Critical gotchas (carry-overs from ACT path)

- **Episode → trial mapping**: must use `match_episodes_to_trials` from `my_policy.localizer.labels` everywhere. Naive `trial_(ep+1)` is wrong with discarded trials. Carry into `make_smolvla_dataset.py`.
- **TF frame composition for port pose**: `f"task_board/{target_module_name}/{port_name}_link"` — bare `task.port_name` won't resolve.
- **Port frame ≠ insertion mouth**: `groundtruth.port_pose` is interior anchor, mouth is offset −0.0458 m (SFP) / −0.01564 m (SC) along port-local -z. Used by image cropping in `demo_port_crop.py`. Not relevant to inference if we don't add image cropping.
- **Action quat sign canonicalization (Fix 2)**: applied during dataset build via `make_port_local_dataset.py`. Carry over for SmolVLA dataset since the underlying actions come from the same source.
- **clean_act_dataset.py is a no-op for actions but writes meta/stats.json**: lerobot's `make_dataset` requires this. The SmolVLA dataset must also have stats.json. Either run `clean_act_dataset.py` after building, or write the stats inline.

## What this branch will NOT touch

- The world-frame baseline ACT path (`v9_act_merged_clean` and friends).
- `RunACT.py`, `RunPortLocalACT.py` — left intact for fallback.
- Existing `eval_offline_action_mae.py` and `diagnose_episode_predictions.py` — clones, not edits.
- The 100k v9_pl_v2 ACT checkpoint — kept as the production fallback if SmolVLA fails.

## Critical files to create / clone (summary)

| New file | From | Purpose |
|---|---|---|
| `my_policy/scripts/smolvla/make_smolvla_dataset.py` | new | Strip 12-dim task vec from port-local dataset → 32-dim state. |
| `my_policy/scripts/smolvla/train_smolvla.py` | clone of `train_act.py` | Training driver for SmolVLAPolicy. |
| `my_policy/my_policy/ros/RunSmolVLA.py` | clone of `RunPortLocalACT.py` | Inference shim. |
| `my_policy/scripts/smolvla/test_runsmolvla_offline.py` | clone of `test_runportlocalact_offline.py` | Offline test (3 tiers). |
| `my_policy/scripts/smolvla/eval_offline_action_mae_smolvla.py` | clone of `eval_offline_action_mae.py` | Val-set MAE evaluator. |
| `my_policy/scripts/smolvla/diagnose_smolvla_predictions.py` | clone of `diagnose_episode_predictions.py` | Per-frame trajectory diff. |

## How to apply this plan

After conversation compaction, read this file first and the parent plan `/home/robin/.claude/plans/check-out-the-plan-quizzical-donut.md` for the broader 7-day context. Then execute Phase 0 → 6 in order. Don't skip Phase 0 preflight — the `max_state_dim=32` constraint is the most likely surprise.

## Time budget

Optimistic: Phase 1 (1 hr) + Phase 2 training kickoff (1 hr setup, runs overnight) + Phase 3-5 in parallel during training (~6 hrs work) + Phase 6 decision (1 hr).
Realistic: 2 days of work + 1 day of training compute = decision by 2026-05-12 with deadline 2026-05-15 (3 days buffer).
