#!/usr/bin/env python3
"""Tier 2 integration tests for the port-localizer dataset wrapper.

Runs against the existing /home/robin/ssd/aic_workspace/aic_docker/aic/data_collection/smoke_d4b_dataset.

The killer test (test_killer_rms_under_1mm) verifies that the labels produced
by `compute_label` reconstruct port poses matching `groundtruth.port_pose` to
within < 1 mm RMS across every frame in the dataset. A bug anywhere in the
label pipeline — yaw sign, rail-key lookup, URDF offset, transform composition —
shows up here.

Run inside the dev container (torch + pyarrow available):
    python3 my_policy/scripts/test_localizer_dataset.py

(No pixi env needed for these checks — torch is the only non-stdlib import
beyond pyarrow/numpy/yaml. Skipping the cameras=() image branch keeps lerobot
out of the path; that's covered by Phase D's training script.)
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.localizer.dataset import LocalizerDataset  # noqa: E402
from my_policy.localizer.labels import (  # noqa: E402
    compute_label,
    match_episodes_to_trials,
    reconstruct_port_in_baselink,
)


# ============================================================================
# Test fixture: hard-coded path to the smoke dataset. If we add more reference
# datasets later, accept a CLI arg.
# ============================================================================

_SMOKE_DATASET = Path(
    "/home/robin/ssd/aic_workspace/aic_docker/aic/data_collection/smoke_d4b_dataset"
)
_SMOKE_BATCH_YAML = Path(
    "/home/robin/ssd/aic_workspace/aic_docker/aic/data_collection/smoke_d4b.yaml"
)
_SMOKE_SUMMARY = _SMOKE_DATASET.parent / (_SMOKE_DATASET.name + "_logs") / "summary.json"


def _ensure_fixture_present() -> None:
    for p in [_SMOKE_DATASET, _SMOKE_BATCH_YAML, _SMOKE_SUMMARY]:
        if not p.exists():
            raise FileNotFoundError(
                f"missing test fixture {p}. Run a smoke collection first to "
                f"populate the dataset (see my_policy/README.md)."
            )


# ============================================================================
# Tests
# ============================================================================

def test_killer_rms_under_1mm():
    """Primary acceptance criterion. For every frame:
      predicted_port_baselink = reconstruct_port_in_baselink(label, target, port)
      recorded_port_baselink  = observation.state[groundtruth.port_pose.{x,y,z}]
      assert RMS(diff) < 1 mm AND max(|diff|) < 3 mm
    """
    _ensure_fixture_present()
    ds = LocalizerDataset(
        _SMOKE_DATASET, _SMOKE_BATCH_YAML, _SMOKE_SUMMARY,
        cameras=(),  # skip image decoding for this test
    )
    diffs = []
    for i in range(len(ds)):
        sample = ds[i]
        ep = sample["_meta"]["episode_index"]
        target_module = sample["_meta"]["target_module_name"]
        port_name = sample["_meta"]["port_name"]
        label = ds.label_for_episode(ep)
        predicted = reconstruct_port_in_baselink(label, target_module, port_name)
        recorded = ds.port_pose_baselink(i)
        diffs.append(predicted - recorded)
    diffs_arr = np.stack(diffs)
    rms_per_frame = np.linalg.norm(diffs_arr, axis=1)
    rms_overall = float(np.sqrt(np.mean(np.sum(diffs_arr ** 2, axis=1))))
    max_abs = float(np.max(np.abs(diffs_arr)))
    print(
        f"  killer test: frames={len(ds)} rms_overall={rms_overall*1000:.4f}mm "
        f"max_abs={max_abs*1000:.4f}mm "
        f"per_frame_rms_max={float(rms_per_frame.max())*1000:.4f}mm"
    )
    assert rms_overall < 1e-3, (
        f"RMS {rms_overall*1000:.3f} mm exceeds 1 mm threshold; label "
        f"pipeline has a systematic bug."
    )
    assert max_abs < 3e-3, (
        f"max abs error {max_abs*1000:.3f} mm exceeds 3 mm; outlier frame."
    )


def test_dataset_shapes_and_dtypes():
    _ensure_fixture_present()
    ds = LocalizerDataset(
        _SMOKE_DATASET, _SMOKE_BATCH_YAML, _SMOKE_SUMMARY,
        cameras=(),
    )
    sample = ds[0]
    assert sample["tcp_pose"].shape == (7,), sample["tcp_pose"].shape
    assert sample["task_one_hot"].shape == (7,)
    assert sample["target"].shape == (5,)
    # sin² + cos² = 1.
    t = sample["target"]
    assert abs(t[2] ** 2 + t[3] ** 2 - 1.0) < 1e-6, (
        f"sin²+cos² != 1: {t[2]**2 + t[3]**2}"
    )
    # Meta keys present.
    for k in ("episode_index", "frame_index", "trial_key",
              "target_module_name", "port_name"):
        assert k in sample["_meta"], f"missing _meta key: {k}"
    # cameras=() ⇒ images dict empty.
    assert sample["images"] == {}, f"expected empty images, got {sample['images']}"


def test_dataset_length_matches_total_frames():
    """len(LocalizerDataset) == info.json['total_frames']."""
    _ensure_fixture_present()
    info = json.loads((_SMOKE_DATASET / "meta" / "info.json").read_text())
    expected = int(info["total_frames"])
    ds = LocalizerDataset(
        _SMOKE_DATASET, _SMOKE_BATCH_YAML, _SMOKE_SUMMARY, cameras=(),
    )
    assert len(ds) == expected, f"expected {expected} frames, got {len(ds)}"


def test_episode_count_matches_summary():
    """num_episodes (from dataset) == count(saved_inserted) in summary.json."""
    _ensure_fixture_present()
    summary = json.loads(_SMOKE_SUMMARY.read_text())
    saved = sum(1 for t in summary["trials"] if t.get("outcome") == "saved_inserted")
    ds = LocalizerDataset(
        _SMOKE_DATASET, _SMOKE_BATCH_YAML, _SMOKE_SUMMARY, cameras=(),
    )
    assert ds.num_episodes == saved, (
        f"dataset has {ds.num_episodes} episodes; summary saved {saved}"
    )


def test_label_constant_within_episode():
    """For each episode, every frame's label is identical (caching sanity)."""
    _ensure_fixture_present()
    ds = LocalizerDataset(
        _SMOKE_DATASET, _SMOKE_BATCH_YAML, _SMOKE_SUMMARY, cameras=(),
    )
    seen: dict[int, np.ndarray] = {}
    for i in range(len(ds)):
        sample = ds[i]
        ep = sample["_meta"]["episode_index"]
        target = sample["target"]
        if ep not in seen:
            seen[ep] = target.copy()
            continue
        assert np.allclose(seen[ep], target), (
            f"label drift within ep {ep}: {seen[ep]} vs {target}"
        )


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_killer_rms_under_1mm,
        test_dataset_shapes_and_dtypes,
        test_dataset_length_matches_total_frames,
        test_episode_count_matches_summary,
        test_label_constant_within_episode,
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
