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
    pixi run python my_policy/scripts/smolvla/train_smolvla.py --name v9_pl_smolvla_v1
    pixi run python my_policy/scripts/smolvla/train_smolvla.py --name v9_pl_smolvla_v1 --resume

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
    p.add_argument("--action-normalization", type=str, default="MEAN_STD",
                   choices=["MEAN_STD", "MIN_MAX", "IDENTITY"],
                   help="Normalization mode for the action stream. Default "
                        "MEAN_STD (SmolVLA default). MIN_MAX maps the demo's "
                        "[min, max] action range to normalized [-1, +1], "
                        "useful when the model under-extends the tails.")
    p.add_argument("--state-normalization", type=str, default="MEAN_STD",
                   choices=["MEAN_STD", "MIN_MAX", "IDENTITY"],
                   help="Normalization mode for observation.state.")
    p.add_argument("--n-obs-steps", type=int, default=1,
                   help="Number of observation frames stacked into a single "
                        "model input window. Default 1 (current frame only). "
                        "Set to 4 or 8 to give the policy short-term history.")
    p.add_argument("--max-state-dim", type=int, default=None,
                   help="Override policy.max_state_dim. Default: auto-detect "
                        "from dataset info.json. SmolVLA's built-in default is "
                        "32, so any state dim > 32 (e.g. conditioned datasets) "
                        "needs this bumped.")
    p.add_argument("--max-action-dim", type=int, default=None,
                   help="Override policy.max_action_dim. Default 32 (SmolVLA "
                        "built-in) which fits our 7-dim action with headroom.")
    p.add_argument("--pretrained-policy-path", type=str, default=None,
                   help="HF Hub id or local path to a pretrained SmolVLA "
                        "checkpoint (e.g. 'lerobot/smolvla_base'). When set, "
                        "loads the action expert + VLM weights from this "
                        "checkpoint as a starting point (fine-tune) instead "
                        "of random-initializing the expert. Some flags "
                        "(load_vlm_weights, vlm_model_name, max_state_dim, "
                        "max_action_dim) are inherited from the pretrained "
                        "config and cannot be overridden cleanly.")
    p.add_argument("--rename-map", type=str, default=None,
                   help="JSON-like dict mapping dataset feature keys to "
                        "policy-expected keys. Required when fine-tuning a "
                        "pretrained policy whose feature names differ from "
                        "the dataset's. Default in fine-tune mode: maps our "
                        "left/center/right cameras to camera1/2/3 (the "
                        "SO-100 convention used by lerobot/smolvla_base).")
    args = p.parse_args()

    train_episodes_path = args.train_episodes_file \
        or (args.dataset_root / "train_episodes.json")
    if not train_episodes_path.exists():
        print(f"error: missing {train_episodes_path}", file=sys.stderr)
        return 1
    train_episodes: list[int] = json.loads(train_episodes_path.read_text())
    eps_arg = "[" + ",".join(str(e) for e in train_episodes) + "]"
    output_dir = args.output_root / args.name

    # Detect dataset's state/action dims so we can bump policy.max_state_dim
    # past SmolVLA's hardcoded default of 32 when needed.
    info_path = args.dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    dataset_state_dim = int(info["features"]["observation.state"]["shape"][0])
    dataset_action_dim = int(info["features"]["action"]["shape"][0])
    max_state_dim = args.max_state_dim or max(32, dataset_state_dim)
    max_action_dim = args.max_action_dim or max(32, dataset_action_dim)
    if dataset_state_dim > max_state_dim:
        print(f"error: dataset state_dim {dataset_state_dim} > "
              f"max_state_dim {max_state_dim}", file=sys.stderr)
        return 1

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

    cli: list[str] = [
        "lerobot-train",
        # Dataset.
        f"--dataset.repo_id=local/{args.name}",
        f"--dataset.root={args.dataset_root}",
        f"--dataset.episodes={eps_arg}",
        "--dataset.video_backend=pyav",
    ]
    if args.pretrained_policy_path:
        # Fine-tune mode: load the pretrained policy (expert + VLM) from the
        # given path. The pretrained config drives architecture AND
        # normalization mode — we cannot override normalization_mapping via
        # CLI in path-load mode (draccus's cli_overrides parser rejects the
        # dict-literal syntax). The dataset's stats.json still supplies the
        # actual mean/std/min/max values used by the preprocessor.
        if (
            args.action_normalization != "MEAN_STD"
            or args.state_normalization != "MEAN_STD"
        ):
            print(
                f"[warn] --action-normalization / --state-normalization "
                f"are IGNORED when fine-tuning from a pretrained policy. "
                f"Normalization mode is inherited from "
                f"{args.pretrained_policy_path}'s config; the dataset "
                f"stats.json supplies the values.", file=sys.stderr,
            )
        print(f"[mode] FINE-TUNE from pretrained: {args.pretrained_policy_path}")
        # Default rename map for lerobot/smolvla_base (SO-100 convention).
        # User can override --rename-map for a different pretrained policy.
        default_rename = (
            '{"observation.images.left_camera": "observation.images.camera1", '
            '"observation.images.center_camera": "observation.images.camera2", '
            '"observation.images.right_camera": "observation.images.camera3"}'
        )
        rename_map = args.rename_map or default_rename
        print(f"[mode] rename_map: {rename_map}")
        cli += [
            f"--policy.path={args.pretrained_policy_path}",
            f"--policy.repo_id=local/{args.name}",
            "--policy.push_to_hub=false",
            f"--policy.n_obs_steps={args.n_obs_steps}",
            f"--rename_map={rename_map}",
        ]
    else:
        # From-scratch-expert mode: random-init the SmolVLA action expert,
        # optionally with pretrained SmolVLM2 backbone weights frozen.
        cli += [
            "--policy.type=smolvla",
            f"--policy.repo_id=local/{args.name}",
            "--policy.push_to_hub=false",
            f"--policy.chunk_size={args.chunk_size}",
            f"--policy.n_action_steps={args.n_action_steps}",
            f"--policy.load_vlm_weights={str(args.load_vlm_weights).lower()}",
            f"--policy.freeze_vision_encoder={str(args.freeze_vision_encoder).lower()}",
            f"--policy.train_expert_only={str(args.train_expert_only).lower()}",
            f"--policy.vlm_model_name={args.vlm_model_name}",
            f"--policy.normalization_mapping={{VISUAL: IDENTITY, "
            f"STATE: {args.state_normalization}, "
            f"ACTION: {args.action_normalization}}}",
            f"--policy.n_obs_steps={args.n_obs_steps}",
            f"--policy.max_state_dim={max_state_dim}",
            f"--policy.max_action_dim={max_action_dim}",
        ]
    cli += [
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
        # lerobot's resume needs --config_path pointing to the saved
        # train_config.json (TRAIN_CONFIG_NAME) from the previous run.
        # Convention: it lives at <output_dir>/checkpoints/last/pretrained_model/
        # (symlinked to the latest step's dir).
        candidate = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if not candidate.exists():
            # Fall back: search for the highest-numbered checkpoint.
            ck_root = output_dir / "checkpoints"
            if ck_root.exists():
                steps = sorted(
                    [d for d in ck_root.iterdir() if d.is_dir() and d.name.isdigit()],
                    key=lambda d: int(d.name),
                )
                if steps:
                    candidate = steps[-1] / "pretrained_model" / "train_config.json"
        if not candidate.exists():
            print(f"error: --resume requested but train_config.json not found "
                  f"under {output_dir}/checkpoints/", file=sys.stderr)
            return 1
        print(f"resume config_path  : {candidate}")
        cli.extend(["--resume=true", f"--config_path={candidate}"])
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
    if args.pretrained_policy_path:
        print(f"MODE                : FINE-TUNE from pretrained policy")
        print(f"pretrained policy   : {args.pretrained_policy_path}")
        print(f"  (chunk_size, n_action_steps, max_state_dim, max_action_dim, "
              f"load_vlm_weights, freeze_vision, train_expert_only, vlm "
              f"are INHERITED from the pretrained config — CLI overrides "
              f"are ignored except where listed below.)")
    else:
        print(f"MODE                : FROM-SCRATCH expert (random-init action expert)")
    print(f"dataset_root        : {args.dataset_root}")
    print(f"output_dir          : {output_dir}")
    print(f"train episodes      : {len(train_episodes)} (from {train_episodes_path.name})")
    print(f"steps               : {args.steps}")
    print(f"batch_size          : {args.batch_size}  (num_workers={args.num_workers})")
    if not args.pretrained_policy_path:
        print(f"chunk_size          : {args.chunk_size}  (n_action_steps={args.n_action_steps})")
        print(f"max_state_dim       : {max_state_dim}  (dataset state_dim={dataset_state_dim})")
        print(f"max_action_dim      : {max_action_dim}  (dataset action_dim={dataset_action_dim})")
        print(f"vlm                 : {args.vlm_model_name}")
        print(f"load_vlm_weights    : {args.load_vlm_weights}")
        print(f"freeze_vision       : {args.freeze_vision_encoder}")
        print(f"train_expert_only   : {args.train_expert_only}")
    print(f"normalization       : ACTION={args.action_normalization}  STATE={args.state_normalization}  (override)")
    print(f"n_obs_steps         : {args.n_obs_steps}  (override)")
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
