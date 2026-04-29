#!/usr/bin/env python3
"""Tier 1.5 unit tests for the BoardPoseRegressor model.

Requires torch + torchvision (pixi env). Verifies forward pass shapes and the
loss/metric helpers without touching real data.

Run inside the dev container:
    pixi run python my_policy/scripts/test_localizer_model.py
"""

import math
import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

try:
    import torch
except ModuleNotFoundError:
    print("SKIP: torch not installed — run in pixi env")
    sys.exit(0)

from my_policy.localizer.model import (  # noqa: E402
    BoardPoseRegressor,
    BoardPoseRegressorConfig,
    FiLM,
    denormalize_pred,
    loss_fn,
    normalize_target,
    predicted_yaw_rad,
    reconstruct_metric_errors,
)


def test_forward_shape_multicam():
    """Forward pass with (B, num_cams, 3, H, W) returns (B, 5)."""
    cfg = BoardPoseRegressorConfig(backbone_pretrained=False, num_cameras=3)
    model = BoardPoseRegressor(cfg)
    model.eval()
    B = 4
    images = torch.randn(B, 3, 3, 224, 224)
    tcp = torch.randn(B, 7)
    oh = torch.zeros(B, 7)
    oh[torch.arange(B), torch.tensor([0, 2, 5, 6])] = 1.0
    with torch.no_grad():
        out = model(images, tcp, oh)
    assert out.shape == (B, 5), f"expected (4, 5), got {out.shape}"


def test_forward_shape_singlecam_backcompat():
    """Single-cam config still works for legacy paths."""
    cfg = BoardPoseRegressorConfig(backbone_pretrained=False, num_cameras=1)
    model = BoardPoseRegressor(cfg)
    model.eval()
    B = 2
    images = torch.randn(B, 1, 3, 224, 224)  # explicit num_cams=1
    tcp = torch.randn(B, 7)
    oh = torch.zeros(B, 7); oh[:, 0] = 1.0
    with torch.no_grad():
        out = model(images, tcp, oh)
    assert out.shape == (B, 5)


def test_film_identity_at_zero_conditioning():
    """FiLM with zero conditioning leaves features unchanged at init.

    The +1 in the gamma scaling and zero-init weights guarantee this. Critical
    so the pretrained backbone doesn't get destroyed by the random FiLM init.
    """
    film = FiLM(conditioning_dim=7, feature_dim=512)
    feat = torch.randn(4, 512)
    cond = torch.zeros(4, 7)
    out = film(feat, cond)
    assert torch.allclose(out, feat, atol=1e-6), (
        "FiLM with zero conditioning should be identity at init"
    )


def test_loss_fn_zero_at_match():
    """loss_fn returns 0 when pred (normalized) matches the normalized target."""
    target_phys = torch.randn(4, 5) * 0.1  # arbitrary physical values
    pred_norm = normalize_target(target_phys)  # what a perfect model emits
    out = loss_fn(pred_norm, target_phys)
    assert out.item() < 1e-9, f"expected ~0, got {out.item()}"


def test_loss_fn_normalization_equalizes_axes():
    """An equal-relative-error perturbation on every axis should produce
    roughly equal per-axis loss components after normalization."""
    # Use a non-trivial reference target close to the expected mean.
    target = torch.tensor([[0.165, 0.0, 0.0, 0.94, 0.025]])
    target_n = normalize_target(target)
    # Pred = perfect prediction perturbed by 0.5 std on every dim.
    pred = target_n + 0.5
    comps = loss_fn(pred, target, return_components=True)
    # Each component should be ~0.25 (since (0.5)² = 0.25).
    for k in ("board_x", "board_y", "rail_t"):
        assert abs(comps[k].item() - 0.25) < 1e-6, f"{k}: {comps[k].item()}"
    # yaw_sincos averages two channels, both 0.25.
    assert abs(comps["yaw_sincos"].item() - 0.25) < 1e-6


def test_predicted_yaw_rad_atan2():
    """atan2 of (sin, cos) recovers the yaw — input must be physical units."""
    yaws = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2, 1.234])
    pred_physical = torch.zeros(len(yaws), 5)
    pred_physical[:, 2] = torch.sin(yaws)
    pred_physical[:, 3] = torch.cos(yaws)
    recovered = predicted_yaw_rad(pred_physical)
    diff = (recovered - yaws + math.pi) % (2.0 * math.pi) - math.pi
    assert torch.all(torch.abs(diff) < 1e-6), f"yaw recovery off: {diff}"


def test_reconstruct_metric_errors_basic():
    """Verify the metric conversions (m → mm, rad → deg) are consistent.

    `pred` is the model's z-scored output; metrics denormalize internally so
    we construct pred = normalize(target_perturbed_in_physical_units).
    """
    target = torch.tensor([[0.165, 0.0, 0.0, 1.0, 0.025]])  # yaw = 0
    pred_physical = torch.tensor([
        [0.166, 0.0, math.sin(0.01), math.cos(0.01), 0.027]
    ])  # xy off by 1mm, yaw off by 0.01 rad, rail off by 2mm
    pred = normalize_target(pred_physical)  # what the model would emit
    errs = reconstruct_metric_errors(pred, target)
    assert abs(errs["board_xy_mm"].item() - 1.0) < 1e-3
    assert abs(errs["yaw_deg"].item() - math.degrees(0.01)) < 1e-3
    assert abs(errs["rail_t_mm"].item() - 2.0) < 1e-3


def test_normalize_denormalize_roundtrip():
    """normalize ∘ denormalize = identity."""
    x = torch.randn(8, 5)
    y = normalize_target(denormalize_pred(x))
    assert torch.allclose(x, y, atol=1e-6)
    z = denormalize_pred(normalize_target(x))
    assert torch.allclose(x, z, atol=1e-6)


def test_pretrained_backbone_loads():
    """ResNet18 pretrained weights load without crashing.

    Skipped when offline (CI / no internet). Catches the common bug of
    ImageNet weights mismatch with torchvision version.
    """
    try:
        model = BoardPoseRegressor(BoardPoseRegressorConfig(backbone_pretrained=True))
    except Exception as ex:
        # Network errors are non-deterministic in some envs; skip rather than fail.
        if "URL" in str(ex) or "connection" in str(ex).lower():
            print(f"  skipping: {ex}")
            return
        raise
    n = sum(p.numel() for p in model.parameters())
    # ResNet18 base ~11M params + heads + film ≈ slightly more.
    assert 10e6 < n < 13e6, f"unexpected param count {n}"


if __name__ == "__main__":
    tests = [
        test_forward_shape_multicam,
        test_forward_shape_singlecam_backcompat,
        test_film_identity_at_zero_conditioning,
        test_loss_fn_zero_at_match,
        test_loss_fn_normalization_equalizes_axes,
        test_predicted_yaw_rad_atan2,
        test_reconstruct_metric_errors_basic,
        test_normalize_denormalize_roundtrip,
        test_pretrained_backbone_loads,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as ex:
            failures += 1
            import traceback
            print(f"FAIL  {t.__name__}: {type(ex).__name__}: {ex}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(0 if failures == 0 else 1)
