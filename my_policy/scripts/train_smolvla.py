#!/usr/bin/env python3
"""v9-port-local SmolVLA training driver.

Mirrors `train_act.py` (same trackio→wandb shim, same train-episodes
filter convention) but launches lerobot's training with
`--policy.type=smolvla` instead of `act`. SmolVLA defaults — chunk_size=50,
freeze_vision_encoder=True, train_expert_only=True, vlm=SmolVLM2-500M —
are the right starting point for this task; we override only the bits
that matter for our dataset shape (state already 32-dim from
make_smolvla_dataset.py, so max_state_dim default is fine).

Inputs:
  - Dataset built by `make_smolvla_dataset.py`. State 32-dim, action 7-dim.
    `tasks` strings per episode are SmolVLA's language conditioning input.
  - Train-episodes filter: `<dataset_root>/train_episodes.json` (carried
    over from the merged port-local split — same 340/86 = 4:1 split as
    the ACT run).

Run (inside the dev container at ~/ws_aic/src/aic):
    pixi run python my_policy/scripts/train_smolvla.py --name v9_pl_smolvla_v1
    pixi run python my_policy/scripts/train_smolvla.py --name v9_pl_smolvla_v1 --resume

VRAM: SmolVLA-500M with frozen vision encoder + 7-dim action (padded to 32)
fits comfortably in 24 GB at batch_size=4. On the 48 GB local box, batch=8
should work; tune down if OOM.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path


# --- trackio → wandb shim (verbatim from train_act.py) --------------------

def _install_wandb_shim() -> None:
    import trackio  # noqa: F401

    shim = types.ModuleType("wandb")
    shim.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)

    class _RunProxy:
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
    shim.run = None

    sys.modules["wandb"] = shim


# --- main ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--name", default="v9_pl_smolvla_v1",
                   help="Run name → output_dir = <output-root>/<name>/")
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/root/aic_data/v9_act_build/v9_port_local_smolvla_dataset"))
    p.add_argument("--output-root", type=Path,
                   default=Path("/root/aic_data/v9_act_build/runs"))
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=4,
                   help="SmolVLA-500M is heavier than ACT — start at 4 and "
                        "raise to 8 if VRAM allows.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-freq", type=int, default=10_000)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-image-transforms", action="store_true")
    p.add_argument("--no-trackio", action="store_true")
    p.add_argument("--trackio-project", default="aic_v9_smolvla")
    # SmolVLA-specific knobs.
    p.add_argument("--chunk-size", type=int, default=50,
                   help="SmolVLA action lookahead horizon (frames @ 20 Hz). "
                        "Default 50 = 2.5 s.")
    p.add_argument("--n-action-steps", type=int, default=50,
                   help="Steps to execute before re-querying. Default = chunk_size.")
    p.add_argument("--load-vlm-weights", type=lambda s: s.lower() == "true",
                   default=True,
                   help="Load pretrained SmolVLM2-500M weights from HF Hub. "
                        "Set to false to train the expert from scratch.")
    p.add_argument("--freeze-vision-encoder", type=lambda s: s.lower() == "true",
                   default=True,
                   help="Freeze the SigLIP image encoder (default for SmolVLA).")
    p.add_argument("--train-expert-only", type=lambda s: s.lower() == "true",
                   default=True,
                   help="Train only the action expert (default for SmolVLA).")
    p.add_argument("--vlm-model-name", type=str,
                   default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                   help="HF Hub model ID for the VLM backbone.")
    p.add_argument("--train-episodes-file", type=Path, default=None,
                   help="Override train episode-index list. Defaults to "
                        "<dataset-root>/train_episodes.json.")
    args = p.parse_args()

    train_episodes_path = args.train_episodes_file \
        or (args.dataset_root / "train_episodes.json")
    if not train_episodes_path.exists():
        print(f"error: missing {train_episodes_path}", file=sys.stderr)
        return 1
    train_episodes: list[int] = json.loads(train_episodes_path.read_text())
    eps_arg = "[" + ",".join(str(e) for e in train_episodes) + "]"
    output_dir = args.output_root / args.name

    if output_dir.exists() and not args.resume:
        if args.force:
            import shutil
            print(f"--force: deleting existing output_dir {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"error: output_dir {output_dir} already exists.", file=sys.stderr)
            print("  pass --resume to continue, --force to delete, or pick a "
                  "different --name.", file=sys.stderr)
            return 1

    cli = [
        "lerobot-train",
        # Dataset.
        f"--dataset.repo_id=local/{args.name}",
        f"--dataset.root={args.dataset_root}",
        f"--dataset.episodes={eps_arg}",
        "--dataset.video_backend=pyav",
        # Policy.
        "--policy.type=smolvla",
        f"--policy.repo_id=local/{args.name}",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.load_vlm_weights={str(args.load_vlm_weights).lower()}",
        f"--policy.freeze_vision_encoder={str(args.freeze_vision_encoder).lower()}",
        f"--policy.train_expert_only={str(args.train_expert_only).lower()}",
        f"--policy.vlm_model_name={args.vlm_model_name}",
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
            "--wandb.add_tags=false",
        ])

    sys.argv = cli

    print(f"=== v9-pl-smolvla training run: {args.name} ===")
    print(f"dataset_root        : {args.dataset_root}")
    print(f"output_dir          : {output_dir}")
    print(f"train episodes      : {len(train_episodes)} (from {train_episodes_path.name})")
    print(f"steps               : {args.steps}")
    print(f"batch_size          : {args.batch_size}  (num_workers={args.num_workers})")
    print(f"chunk_size          : {args.chunk_size}  (n_action_steps={args.n_action_steps})")
    print(f"vlm                 : {args.vlm_model_name}")
    print(f"load_vlm_weights    : {args.load_vlm_weights}")
    print(f"freeze_vision       : {args.freeze_vision_encoder}")
    print(f"train_expert_only   : {args.train_expert_only}")
    print(f"image augs          : {'OFF' if args.no_image_transforms else 'ON'}")
    print(f"tracker             : {'disabled' if args.no_trackio else f'trackio (project={args.trackio_project})'}")
    print(f"resume              : {args.resume}")
    print()

    if args.dry_run:
        import draccus
        import lerobot.policies  # noqa: F401  — registers smolvla via @register_subclass
        from lerobot.configs.train import TrainPipelineConfig

        cfg: TrainPipelineConfig = draccus.parse(
            config_class=TrainPipelineConfig,
            config_path=None,
            args=sys.argv[1:],
        )
        cfg.validate()
        print("=== --dry-run: cfg.validate() OK ===")
        print(f"  dataset.root             = {cfg.dataset.root}")
        print(f"  dataset.episodes         = list of {len(cfg.dataset.episodes or [])}")
        print(f"  policy.type              = {cfg.policy.type}")
        print(f"  policy.chunk_size        = {cfg.policy.chunk_size}")
        print(f"  policy.n_action_steps    = {cfg.policy.n_action_steps}")
        print(f"  policy.max_state_dim     = {cfg.policy.max_state_dim}")
        print(f"  policy.max_action_dim    = {cfg.policy.max_action_dim}")
        print(f"  policy.load_vlm_weights  = {cfg.policy.load_vlm_weights}")
        print(f"  policy.freeze_vision_enc = {cfg.policy.freeze_vision_encoder}")
        print(f"  policy.train_expert_only = {cfg.policy.train_expert_only}")
        print(f"  policy.vlm_model_name    = {cfg.policy.vlm_model_name}")
        print(f"  output_dir               = {cfg.output_dir}")
        print(f"  batch_size               = {cfg.batch_size}")
        print(f"  steps                    = {cfg.steps}")
        return 0

    from lerobot.scripts.lerobot_train import train as lerobot_train
    lerobot_train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
