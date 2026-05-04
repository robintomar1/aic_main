#!/usr/bin/env python3
"""v9-act ACT training driver.

Wraps lerobot's `lerobot_train.train` with:
  * trackio (HF's local-first wandb-compatible tracker) replacing wandb —
    via a sys.modules['wandb'] shim installed BEFORE lerobot imports.
  * Episode filter from `train_episodes.json` (340/86 train/val split
    written by `merge_act_datasets.py`). Stock ACT v0.5.1 does no held-
    out validation during supervised training (eval block fires only
    when `cfg.env` is set), so val episodes are intentionally excluded
    from the training loader and must be evaluated separately by the
    bench in Phase D.
  * ACT hyperparams baked in per `my_policy/docs/v9_act_plan.md`:
      - chunk_size=100 (5 s lookahead at 20 Hz)
      - n_action_steps=8 (replan every 0.4 s)
      - use_vae=true (CVAE encoder; latent zeroed at inference)
      - temporal_ensemble_coeff omitted (=> None / disabled)
  * Image augmentation enabled via lerobot's built-in
    ImageTransformsConfig (brightness/contrast/hue/affine; defaults
    documented in lerobot/datasets/transforms.py).

Setup (one-time, inside pixi):
    pixi run pip install trackio

Run (inside the dev container at ~/ws_aic/src/aic):
    pixi run python my_policy/scripts/train_act.py --name v9_act_v1
    # to resume from <output-root>/<name>/ when an existing checkpoint exists:
    pixi run python my_policy/scripts/train_act.py --name v9_act_v1 --resume

If you OOM at the default batch=8, drop to 4. The 1152×1024 cams are
fed at native resolution to ResNet18 — that's ~6 GB activations per
batch sample × 3 cams. 48 GB VRAM should handle batch=8 comfortably,
but the ACT transformer's attention is O(seq²) over the chunk so 100-
step chunks aren't free either.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path


# --- trackio → wandb shim --------------------------------------------------
#
# lerobot's WandBLogger does `import wandb` lazily inside __init__ and uses
# wandb.init / wandb.log / wandb.run.{id, get_url} / wandb.Artifact /
# wandb.Video / wandb.log_artifact / wandb.define_metric.
#
# trackio is API-compatible with the FIRST THREE (init/log/finish + run).
# The artifact/video/define_metric APIs are wandb-specific; for supervised
# ACT training (no env eval, --wandb.disable_artifact=true) they are never
# called, but lerobot imports them, so we expose them as no-op stubs.
#
# We install this shim BEFORE any lerobot import — once installed,
# `import wandb` anywhere in this process resolves to the shim.

def _install_wandb_shim() -> None:
    import trackio  # noqa: F401 — must be installed; raises ImportError if not.

    shim = types.ModuleType("wandb")
    # accelerate.utils.imports._is_package_available calls
    # importlib.util.find_spec("wandb") which requires a real ModuleSpec —
    # it raises `ValueError: wandb.__spec__ is None` otherwise. Provide one.
    shim.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)

    class _RunProxy:
        """Wraps trackio's run with the .id and .get_url() that lerobot
        prints during init. Falls back to placeholders if trackio's run
        doesn't expose them."""

        def __init__(self, trackio_run):
            self._r = trackio_run

        @property
        def id(self) -> str:
            for attr in ("id", "name", "run_id"):
                v = getattr(self._r, attr, None)
                if v:
                    return str(v)
            return "trackio_run"

        def get_url(self) -> str:
            for attr in ("url", "dashboard_url"):
                v = getattr(self._r, attr, None)
                if v:
                    return str(v)
            return "(local trackio dashboard — see `trackio show` output)"

    def _init(**kwargs):
        # lerobot passes: id, project, entity, name, notes, tags, dir,
        # config, save_code, job_type, resume, mode. Trackio.init takes
        # (project, name, config, space_id, resume). Filter + rename.
        accepted = {"project", "name", "config"}
        forwarded = {k: v for k, v in kwargs.items() if k in accepted}
        if kwargs.get("id"):
            forwarded["resume"] = kwargs["id"]
        run = trackio.init(**forwarded)
        shim.run = _RunProxy(run)
        return shim.run

    def _log(data, step=None):
        if step is not None:
            trackio.log(data, step=step)
        else:
            trackio.log(data)

    def _finish():
        trackio.finish()

    class _Stub:
        """Soaks up wandb-only API calls we know are never reached for
        supervised ACT training but show up in WandBLogger import-time."""
        def __init__(self, *a, **k): pass
        def add_file(self, *a, **k): pass

    def _noop(*a, **k): pass

    shim.init = _init
    shim.log = _log
    shim.finish = _finish
    shim.Artifact = _Stub
    shim.Video = _Stub
    shim.log_artifact = _noop
    shim.define_metric = _noop
    shim.run = None  # populated on init() call.

    sys.modules["wandb"] = shim


# --- main ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--name", default="v9_act_v1",
                   help="Run name → output_dir = <output-root>/<name>/")
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/root/aic_data/v9_act_build/v9_act_merged"))
    p.add_argument("--output-root", type=Path,
                   default=Path("/root/aic_data/v9_act_build/runs"))
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-freq", type=int, default=10_000)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint in <output-root>/<name>/.")
    p.add_argument("--force", action="store_true",
                   help="Delete <output-root>/<name>/ if it exists (overrides lerobot's "
                        "FileExistsError). Without this, an existing dir aborts unless --resume.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config end-to-end (CLI parse + cfg.validate()) and exit "
                        "before the train loop; catches missing args fast without touching "
                        "the GPU or dataset.")
    p.add_argument("--no-image-transforms", action="store_true",
                   help="Disable lerobot's built-in brightness/contrast/hue/affine augs.")
    p.add_argument("--no-trackio", action="store_true",
                   help="Skip trackio entirely (lerobot's wandb stays disabled).")
    p.add_argument("--trackio-project", default="aic_v9_act")
    p.add_argument("--chunk-size", type=int, default=100,
                   help="ACT lookahead horizon (frames @ 20 Hz). Default 100 = 5 s.")
    p.add_argument("--n-action-steps", type=int, default=8,
                   help="Steps to execute before re-querying the policy. Default 8 = 0.4 s.")
    args = p.parse_args()

    train_episodes_path = args.dataset_root / "train_episodes.json"
    if not train_episodes_path.exists():
        print(f"error: missing {train_episodes_path}; run merge_act_datasets.py "
              f"first to populate the train/val split.", file=sys.stderr)
        return 1
    train_episodes: list[int] = json.loads(train_episodes_path.read_text())
    eps_arg = "[" + ",".join(str(e) for e in train_episodes) + "]"
    output_dir = args.output_root / args.name

    # Pre-flight: lerobot's TrainPipelineConfig.validate() raises FileExistsError
    # if output_dir exists and --resume is False. Surface this as a clean wrapper
    # error so the user can choose --resume or --force, instead of seeing a
    # mid-stack traceback.
    if output_dir.exists() and not args.resume:
        if args.force:
            import shutil
            print(f"--force: deleting existing output_dir {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"error: output_dir {output_dir} already exists.", file=sys.stderr)
            print("  pass --resume to continue from the latest checkpoint, OR", file=sys.stderr)
            print("  pass --force to delete and start fresh, OR", file=sys.stderr)
            print("  pick a different --name.", file=sys.stderr)
            return 1

    cli = [
        "lerobot-train",
        # Dataset.
        f"--dataset.repo_id=local/{args.name}",
        f"--dataset.root={args.dataset_root}",
        f"--dataset.episodes={eps_arg}",
        "--dataset.video_backend=pyav",
        # Policy.
        "--policy.type=act",
        f"--policy.repo_id=local/{args.name}",  # required by HubMixin even for local-only runs
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        "--policy.use_vae=true",
        # Trainer.
        f"--output_dir={output_dir}",
        f"--job_name={args.name}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--seed={args.seed}",
        "--save_checkpoint=true",
    ]
    if args.resume:
        cli.append("--resume=true")
    if not args.no_image_transforms:
        cli.append("--dataset.image_transforms.enable=true")

    if args.no_trackio:
        cli.append("--wandb.enable=false")
    else:
        _install_wandb_shim()
        cli.extend([
            "--wandb.enable=true",
            f"--wandb.project={args.trackio_project}",
            "--wandb.disable_artifact=true",
            "--wandb.add_tags=false",  # trackio doesn't model tags.
        ])

    sys.argv = cli

    print(f"=== v9-act training run: {args.name} ===")
    print(f"dataset_root  : {args.dataset_root}")
    print(f"output_dir    : {output_dir}")
    print(f"train episodes: {len(train_episodes)} (from train_episodes.json)")
    print(f"steps         : {args.steps}")
    print(f"batch_size    : {args.batch_size}  (num_workers={args.num_workers})")
    print(f"chunk_size    : {args.chunk_size}  (n_action_steps={args.n_action_steps})")
    print(f"image augs    : {'OFF' if args.no_image_transforms else 'ON (lerobot built-ins)'}")
    print(f"tracker       : {'disabled' if args.no_trackio else f'trackio (project={args.trackio_project})'}")
    print(f"resume        : {args.resume}")
    print()

    if args.dry_run:
        # Run lerobot's draccus parser + cfg.validate() against our CLI args
        # but bail before make_dataset / make_policy / Accelerator. This catches
        # missing-arg, type-mismatch, and validate() errors without burning the
        # GPU, dataloader workers, or first-batch decode time.
        from lerobot.configs.parser import wrap as parser_wrap  # noqa: F401
        from lerobot.configs.train import TrainPipelineConfig

        @parser_wrap()
        def _validate_only(cfg: TrainPipelineConfig):
            cfg.validate()
            print("=== --dry-run: cfg.validate() OK ===")
            print(f"  dataset.root        = {cfg.dataset.root}")
            print(f"  dataset.episodes    = list of {len(cfg.dataset.episodes or [])}")
            print(f"  policy.type         = {cfg.policy.type}")
            print(f"  policy.chunk_size   = {cfg.policy.chunk_size}")
            print(f"  policy.n_action_steps = {cfg.policy.n_action_steps}")
            print(f"  policy.use_vae      = {cfg.policy.use_vae}")
            print(f"  output_dir          = {cfg.output_dir}")
            print(f"  job_name            = {cfg.job_name}")
            print(f"  batch_size          = {cfg.batch_size}")
            print(f"  steps               = {cfg.steps}")
            print(f"  wandb.enable        = {cfg.wandb.enable}")
        _validate_only()
        return 0

    # Deferred import — the shim must be in place first.
    from lerobot.scripts.lerobot_train import train as lerobot_train
    lerobot_train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
