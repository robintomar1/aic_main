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
    loss_fn,
    predicted_yaw_rad,
    reconstruct_metric_errors,
)


def test_forward_shape():
    """Forward pass returns (B, 5)."""
    model = BoardPoseRegressor(BoardPoseRegressorConfig(backbone_pretrained=False))
    model.eval()
    B = 4
    image = torch.randn(B, 3, 224, 224)
    tcp = torch.randn(B, 7)
    oh = torch.zeros(B, 7)
    oh[torch.arange(B), torch.tensor([0, 2, 5, 6])] = 1.0
    with torch.no_grad():
        out = model(image, tcp, oh)
    assert out.shape == (B, 5), f"expected (4, 5), got {out.shape}"


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
    """loss_fn returns 0 when pred == target."""
    pred = torch.randn(4, 5)
    out = loss_fn(pred, pred)
    assert out.item() < 1e-9, f"expected ~0, got {out.item()}"


def test_loss_fn_components_sum_correctly():
    """Per-axis component breakdown is internally consistent."""
    pred = torch.zeros(4, 5)
    target = torch.tensor([
        [0.1, 0.0, 0.0, 0.0, 0.0],   # board_x off by 0.1
        [0.0, 0.2, 0.0, 0.0, 0.0],   # board_y off by 0.2
        [0.0, 0.0, 0.3, 0.0, 0.0],   # sin off by 0.3
        [0.0, 0.0, 0.0, 0.0, 0.5],   # rail_t off by 0.5
    ])
    comps = loss_fn(pred, target, return_components=True)
    # board_x: 0.01 / 4 = 0.0025
    assert abs(comps["board_x"].item() - 0.0025) < 1e-9
    assert abs(comps["board_y"].item() - 0.01) < 1e-9
    assert abs(comps["yaw_sincos"].item() - (0.09 / 8)) < 1e-9
    assert abs(comps["rail_t"].item() - 0.0625) < 1e-9


def test_predicted_yaw_rad_atan2():
    """atan2 of (sin, cos) recovers the yaw."""
    yaws = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2, 1.234])
    pred = torch.zeros(len(yaws), 5)
    pred[:, 2] = torch.sin(yaws)
    pred[:, 3] = torch.cos(yaws)
    recovered = predicted_yaw_rad(pred)
    diff = (recovered - yaws + math.pi) % (2.0 * math.pi) - math.pi
    assert torch.all(torch.abs(diff) < 1e-6), f"yaw recovery off: {diff}"


def test_reconstruct_metric_errors_basic():
    """Verify the metric conversions (m → mm, rad → deg) are consistent."""
    target = torch.tensor([[0.1, -0.2, 0.0, 1.0, 0.05]])  # yaw = 0
    # pred: board_xy off by (0.001, 0), yaw off by sin(0.01)~=0.01 rad ≈ 0.573 deg,
    # rail_t off by 0.002.
    pred = torch.tensor([[0.101, -0.2, math.sin(0.01), math.cos(0.01), 0.052]])
    errs = reconstruct_metric_errors(pred, target)
    assert abs(errs["board_xy_mm"].item() - 1.0) < 1e-3
    # Yaw should be ~0.573 degrees.
    assert abs(errs["yaw_deg"].item() - math.degrees(0.01)) < 1e-3
    assert abs(errs["rail_t_mm"].item() - 2.0) < 1e-6


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
        test_forward_shape,
        test_film_identity_at_zero_conditioning,
        test_loss_fn_zero_at_match,
        test_loss_fn_components_sum_correctly,
        test_predicted_yaw_rad_atan2,
        test_reconstruct_metric_errors_basic,
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
