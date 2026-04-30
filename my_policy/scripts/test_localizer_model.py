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
    """Forward pass with (B, num_cams, 3, H, W) returns (B, 5).
    Pinned to backbone='resnet18' so the test doesn't try to fetch DINOv2
    when this branch's default backbone is dinov2_vits14."""
    cfg = BoardPoseRegressorConfig(
        backbone="resnet18", backbone_pretrained=False, num_cameras=3,
    )
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
    """Single-cam config still works for legacy paths.
    Pinned to backbone='resnet18' for the same reason as above."""
    cfg = BoardPoseRegressorConfig(
        backbone="resnet18", backbone_pretrained=False, num_cameras=1,
    )
    model = BoardPoseRegressor(cfg)
    model.eval()
    B = 2
    images = torch.randn(B, 1, 3, 224, 224)  # explicit num_cams=1
    tcp = torch.randn(B, 7)
    oh = torch.zeros(B, 7); oh[:, 0] = 1.0
    with torch.no_grad():
        out = model(images, tcp, oh)
    assert out.shape == (B, 5)


def test_forward_return_aux_shapes():
    """v8: return_aux=True returns (pred, aux_pixels) with aux in [0,1]."""
    # Explicit aux_pathway=False — the dataclass default is now True (v9), and
    # both modes are mutually exclusive.
    cfg = BoardPoseRegressorConfig(backbone="resnet18", backbone_pretrained=False, num_cameras=3,
                                   aux_pixel_head=True, aux_pathway=False)
    model = BoardPoseRegressor(cfg)
    model.eval()
    B, NC = 4, 3
    images = torch.randn(B, NC, 3, 224, 224)
    tcp = torch.randn(B, 7)
    oh = torch.zeros(B, 7); oh[:, 0] = 1.0
    with torch.no_grad():
        pred, aux = model(images, tcp, oh, return_aux=True)
    assert pred.shape == (B, 5)
    assert aux.shape == (B, NC, 2)
    # Sigmoid output is bounded.
    assert (aux >= 0).all() and (aux <= 1).all()


def test_forward_return_aux_shapes_v9_pathway():
    """v9-pathway: return_aux=True returns (pred, aux_pixels) with aux in [0,1].

    Aux is produced by a separate conv pathway from the spatial features (not
    the pooled feature cam_fuse consumes), addressing the v8 bug where the two
    losses competed for the 512-d bottleneck.
    """
    cfg = BoardPoseRegressorConfig(backbone="resnet18", backbone_pretrained=False, num_cameras=3,
                                   aux_pixel_head=False, aux_pathway=True)
    model = BoardPoseRegressor(cfg)
    model.eval()
    B, NC = 4, 3
    images = torch.randn(B, NC, 3, 224, 224)
    tcp = torch.randn(B, 7)
    oh = torch.zeros(B, 7); oh[:, 0] = 1.0
    with torch.no_grad():
        pred, aux = model(images, tcp, oh, return_aux=True)
    assert pred.shape == (B, 5)
    assert aux.shape == (B, NC, 2)
    assert (aux >= 0).all() and (aux <= 1).all()
    # Confirm the conv pathway modules exist and the legacy v8 head doesn't.
    assert model.aux_conv is not None
    assert model.aux_pathway_head is not None
    assert model.aux_head is None


def test_forward_pose_only_no_aux():
    """Both aux modes off → pose-only model (matches v7 architecture).

    Important contract: `return_aux=True` is *allowed* even when no aux head
    is configured — the call returns `(pred, None)` rather than raising. This
    keeps the call site identical between aux-enabled and aux-disabled
    training, and is relied on by the train loop which always passes
    `return_aux=True` and skips the aux-loss term when the second tuple
    element is None. Don't add a guard that errors here.
    """
    cfg = BoardPoseRegressorConfig(backbone="resnet18", backbone_pretrained=False, num_cameras=3,
                                   aux_pixel_head=False, aux_pathway=False)
    model = BoardPoseRegressor(cfg)
    assert model.aux_head is None
    assert model.aux_conv is None
    assert model.aux_pathway_head is None
    model.eval()
    B, NC = 2, 3
    images = torch.randn(B, NC, 3, 224, 224)
    tcp = torch.randn(B, 7)
    oh = torch.zeros(B, 7); oh[:, 0] = 1.0
    with torch.no_grad():
        pred, aux = model(images, tcp, oh, return_aux=True)
    assert pred.shape == (B, 5)
    assert aux is None


def test_config_aux_modes_are_mutually_exclusive():
    """Setting both aux_pixel_head and aux_pathway must raise at config time —
    they're alternative integration points, never both."""
    try:
        BoardPoseRegressorConfig(aux_pixel_head=True, aux_pathway=True)
    except ValueError as ex:
        assert "mutually exclusive" in str(ex)
        return
    raise AssertionError("expected ValueError, got none")


def test_config_unknown_backbone_raises():
    """Unknown backbone choice fails fast at config construction."""
    try:
        BoardPoseRegressorConfig(backbone="resnet50")
    except ValueError as ex:
        assert "unknown backbone" in str(ex) or "supported" in str(ex)
        return
    raise AssertionError("expected ValueError, got none")


def test_dinov2_forward_shape():
    """v9-dino: backbone='dinov2_vits14' produces (B, 5) pose pred + (B, NC, 2)
    aux pixel pred, and the spatial feature map fed into the aux pathway is
    16x16 (vs 7x7 for ResNet18). Skip cleanly if torch.hub can't reach GitHub
    (offline CI / no internet)."""
    cfg = BoardPoseRegressorConfig(
        backbone="dinov2_vits14",
        backbone_pretrained=False,  # don't download weights for the test
        backbone_freeze=True,
        num_cameras=3,
        aux_pixel_head=False,
        aux_pathway=True,
    )
    try:
        model = BoardPoseRegressor(cfg)
    except Exception as ex:
        msg = str(ex).lower()
        if any(s in msg for s in ("url", "connection", "network", "http")):
            print(f"  skipping: torch.hub offline ({ex})")
            return
        raise
    model.eval()
    assert model.feature_dim == 384, f"expected 384, got {model.feature_dim}"
    B, NC = 2, 3
    images = torch.randn(B, NC, 3, 224, 224)
    tcp = torch.randn(B, 7)
    oh = torch.zeros(B, 7); oh[:, 0] = 1.0
    with torch.no_grad():
        pred, aux = model(images, tcp, oh, return_aux=True)
    assert pred.shape == (B, 5)
    assert aux.shape == (B, NC, 2)
    assert (aux >= 0).all() and (aux <= 1).all()


def test_dinov2_freeze_excludes_backbone_from_grad():
    """v9-dino with freeze=True: backbone params have requires_grad=False so
    the optimizer can skip them."""
    cfg = BoardPoseRegressorConfig(
        backbone="dinov2_vits14",
        backbone_pretrained=False,
        backbone_freeze=True,
        num_cameras=3,
    )
    try:
        model = BoardPoseRegressor(cfg)
    except Exception as ex:
        msg = str(ex).lower()
        if any(s in msg for s in ("url", "connection", "network", "http")):
            print(f"  skipping: torch.hub offline ({ex})")
            return
        raise
    n_trainable_backbone = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith("backbone_dinov2.") and p.requires_grad
    )
    n_total_backbone = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith("backbone_dinov2.")
    )
    assert n_trainable_backbone == 0, (
        f"freeze=True but {n_trainable_backbone} backbone params still trainable"
    )
    assert n_total_backbone > 1_000_000, (
        f"DINOv2 should have millions of params; got {n_total_backbone}"
    )
    # Toggle train mode → backbone should still be in eval (override path).
    model.train()
    assert not model.backbone_dinov2.training, (
        "frozen backbone should stay in eval() even after model.train()"
    )


def test_aux_pixel_loss_masks_invalid():
    """v8: aux loss ignores frames marked invalid (valid=0)."""
    from my_policy.localizer.model import aux_pixel_loss
    B, NC = 2, 3
    pred = torch.full((B, NC, 2), 0.5)
    # All targets valid=0 → loss should be exactly 0 (no NaN).
    target = torch.zeros((B, NC, 3))
    target[..., 0:2] = 0.9
    target[..., 2] = 0.0
    loss_zero = aux_pixel_loss(pred, target)
    assert torch.allclose(loss_zero, torch.tensor(0.0))
    # One valid sample → loss = (0.5-0.9)^2 = 0.16 averaged over 2 channels = 0.16.
    target[0, 0, 2] = 1.0
    loss_one = aux_pixel_loss(pred, target)
    assert abs(loss_one.item() - 0.16) < 1e-6, f"got {loss_one.item()}"


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


def test_relative_normalize_roundtrip():
    """normalize ∘ denormalize == identity in relative-target mode (and the
    relative-mode stats are NOT the same as absolute-mode stats — sanity that
    the `relative` flag actually swaps the constants used)."""
    x = torch.randn(8, 5)
    y = normalize_target(denormalize_pred(x, relative=True), relative=True)
    assert torch.allclose(x, y, atol=1e-6)
    # Cross-check: the relative-mode normalized value is different from
    # absolute-mode for the same physical input — confirms different stats.
    phys = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.025]])
    abs_n = normalize_target(phys, relative=False)
    rel_n = normalize_target(phys, relative=True)
    assert not torch.allclose(abs_n[:, :2], rel_n[:, :2]), (
        "relative-mode xy normalization should differ from absolute-mode "
        "(otherwise the `relative` flag is a no-op)"
    )


def test_relative_loss_zero_at_match():
    """loss_fn(relative=True) returns ~0 when pred matches the relative-
    normalized target. Catches any drift between normalize and the loss
    path's internal normalization."""
    target_relative_phys = torch.tensor(
        [[-0.05, 0.02, 0.1, 0.99, 0.025]]
    )  # already in (board − tcp) space
    pred_norm = normalize_target(target_relative_phys, relative=True)
    out = loss_fn(pred_norm, target_relative_phys, relative=True)
    assert out.item() < 1e-9, f"expected ~0, got {out.item()}"


def test_relative_metric_errors_are_invariant_to_shift():
    """The metric error in mm is the same whether we compute it in absolute
    space or in relative space (the per-frame TCP shift cancels out of a
    pred − target subtraction). Critical: this is what lets us hand the user
    "val xy mm" numbers that mean the same thing across training modes."""
    # Pick a target close to the absolute-mode mean so denormalize gives a
    # numerical result with reasonable scale.
    target_abs = torch.tensor([[0.165, 0.0, 0.0, 1.0, 0.025]])
    tcp_xy = torch.tensor([[0.45, -0.1]])
    target_rel = target_abs.clone()
    target_rel[:, :2] -= tcp_xy

    # Pred has a 1mm xy offset in physical units, in BOTH modes.
    pred_abs_phys = target_abs.clone()
    pred_abs_phys[:, 0] += 0.001
    pred_rel_phys = target_rel.clone()
    pred_rel_phys[:, 0] += 0.001

    pred_abs_n = normalize_target(pred_abs_phys, relative=False)
    pred_rel_n = normalize_target(pred_rel_phys, relative=True)

    err_abs = reconstruct_metric_errors(pred_abs_n, target_abs, relative=False)
    err_rel = reconstruct_metric_errors(pred_rel_n, target_rel, relative=True)

    # 1mm xy delta should land at ~1mm in either reporting mode.
    assert abs(err_abs["board_xy_mm"].item() - 1.0) < 1e-3
    assert abs(err_rel["board_xy_mm"].item() - 1.0) < 1e-3


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
    ImageNet weights mismatch with torchvision version. Explicit `backbone=
    "resnet18"` here so this test is decoupled from whatever the branch's
    dataclass default happens to be (v9-dino-v7arch flips the default to
    dinov2_vits14, and we don't want that to download 84MB at test time).
    """
    try:
        model = BoardPoseRegressor(BoardPoseRegressorConfig(
            backbone="resnet18", backbone_pretrained=True,
        ))
    except Exception as ex:
        # Network errors are non-deterministic in some envs; skip rather than fail.
        if "URL" in str(ex) or "connection" in str(ex).lower():
            print(f"  skipping: {ex}")
            return
        raise
    n = sum(p.numel() for p in model.parameters())
    # ResNet18 base ~11M params + heads + film ≈ slightly more.
    # Bound is generous: ResNet18 trunk (~11.7M) + heads/film/cam_fuse (~1M)
    # plus optional aux_conv (~50K) lands ~12.8M. Upper bound 14M leaves
    # headroom for small future tweaks without flagging as a regression.
    assert 10e6 < n < 14e6, f"unexpected param count {n}"


if __name__ == "__main__":
    tests = [
        test_forward_shape_multicam,
        test_forward_shape_singlecam_backcompat,
        test_forward_return_aux_shapes,
        test_forward_return_aux_shapes_v9_pathway,
        test_forward_pose_only_no_aux,
        test_config_aux_modes_are_mutually_exclusive,
        test_config_unknown_backbone_raises,
        test_dinov2_forward_shape,
        test_dinov2_freeze_excludes_backbone_from_grad,
        test_aux_pixel_loss_masks_invalid,
        test_film_identity_at_zero_conditioning,
        test_loss_fn_zero_at_match,
        test_loss_fn_normalization_equalizes_axes,
        test_relative_normalize_roundtrip,
        test_relative_loss_zero_at_match,
        test_relative_metric_errors_are_invariant_to_shift,
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
