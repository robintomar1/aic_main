#!/usr/bin/env python3
"""Host-runnable tests for the v9-act task vector encoding.

No torch / lerobot / rclpy — pure numpy. Verifies the 12-dim structured
task vector for shape, coverage, and round-trip correctness.

Run: `python3 my_policy/scripts/test_act_labels.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the package importable from a checkout — same pattern as
# test_localizer_labels.py. _PACKAGE_PARENT = my_policy/ (the outer dir,
# parent of my_policy/my_policy/).
_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))

from my_policy.act.labels import (  # noqa: E402
    ACT_MODULE_DIM,
    ACT_MODULE_ORDER,
    ACT_MODULE_SLICE,
    ACT_PORT_IN_MODULE_DIM,
    ACT_PORT_IN_MODULE_ORDER,
    ACT_PORT_IN_MODULE_SLICE,
    ACT_PORT_TYPE_DIM,
    ACT_PORT_TYPE_ORDER,
    ACT_PORT_TYPE_SLICE,
    ACT_TASK_VECTOR_DIM,
    ACT_VALID_TARGETS,
    decode_task_vector,
    encode_task_vector,
    task_channel_names,
    task_string_for,
)


def test_dimensions():
    """Dimensions are exactly what the plan specifies — 7 || 3 || 2 = 12."""
    assert ACT_MODULE_DIM == 7
    assert ACT_PORT_IN_MODULE_DIM == 3
    assert ACT_PORT_TYPE_DIM == 2
    assert ACT_TASK_VECTOR_DIM == 12


def test_full_target_coverage():
    """All 12 (module, port_name) pairs encode into distinct vectors and the
    list of valid targets matches the documented topology."""
    assert len(ACT_VALID_TARGETS) == 12
    encoded = []
    for module, port in ACT_VALID_TARGETS:
        port_type = "sc" if port == "sc_port_base" else "sfp"
        v = encode_task_vector(module, port, port_type)
        encoded.append(tuple(v.tolist()))
    # Every pair must produce a distinct vector — duplicates would silently
    # alias two tasks to one in the model's view.
    assert len(set(encoded)) == 12, (
        f"some task pairs encoded to identical vectors; "
        f"distinct count = {len(set(encoded))}"
    )


def test_each_subvector_sums_to_one():
    """For each valid target, the three sub-vector slices must each sum to 1.
    This is the core sanity check the dataset preprocessor will rely on."""
    for module, port in ACT_VALID_TARGETS:
        port_type = "sc" if port == "sc_port_base" else "sfp"
        v = encode_task_vector(module, port, port_type)
        assert np.isclose(v[ACT_MODULE_SLICE].sum(), 1.0)
        assert np.isclose(v[ACT_PORT_IN_MODULE_SLICE].sum(), 1.0)
        assert np.isclose(v[ACT_PORT_TYPE_SLICE].sum(), 1.0)
        assert v.dtype == np.float32
        assert v.shape == (ACT_TASK_VECTOR_DIM,)


def test_round_trip():
    """encode → decode returns the original triple for every valid target."""
    for module, port in ACT_VALID_TARGETS:
        port_type = "sc" if port == "sc_port_base" else "sfp"
        v = encode_task_vector(module, port, port_type)
        m_out, p_out, t_out = decode_task_vector(v)
        assert m_out == module, f"module: {m_out!r} != {module!r}"
        assert p_out == port, f"port: {p_out!r} != {port!r}"
        assert t_out == port_type, f"port_type: {t_out!r} != {port_type!r}"


def test_unknown_module_raises():
    try:
        encode_task_vector("nic_card_mount_99", "sfp_port_0", "sfp")
    except ValueError as ex:
        assert "unknown target_module_name" in str(ex)
        return
    raise AssertionError("expected ValueError")


def test_unknown_port_name_raises():
    try:
        encode_task_vector("nic_card_mount_0", "bogus_port", "sfp")
    except ValueError as ex:
        assert "unknown port_name" in str(ex)
        return
    raise AssertionError("expected ValueError")


def test_unknown_port_type_raises():
    try:
        encode_task_vector("nic_card_mount_0", "sfp_port_0", "qsfp")
    except ValueError as ex:
        assert "unknown port_type" in str(ex)
        return
    raise AssertionError("expected ValueError")


def test_port_type_mismatch_raises():
    """SC port_name with sfp port_type (or vice versa) must fail fast — this
    catches the most likely caller bug (passing port_type from the wrong
    field of the trial dict)."""
    try:
        encode_task_vector("sc_port_0", "sc_port_base", "sfp")
    except ValueError as ex:
        assert "doesn't match" in str(ex)
        return
    raise AssertionError("expected ValueError")


def test_invalid_combination_raises():
    """SC module with SFP port (or vice versa) — both fields are individually
    valid but the combination is not a real target on the board."""
    # NIC card mount with sc_port_base — invalid; NIC mounts only carry SFP.
    # But this would fail port_type cross-check first if port_type='sc',
    # so test with port_type='sfp' which crashes the cross-check before
    # the combination check fires. Use the symmetric case:
    # sc_port_0 with sfp_port_0, port_type='sfp' — passes type cross-check
    # since sfp_port_0 → sfp; fails on combination check.
    try:
        encode_task_vector("sc_port_0", "sfp_port_0", "sfp")
    except ValueError as ex:
        assert "not a valid target combination" in str(ex)
        return
    raise AssertionError("expected ValueError")


def test_task_channel_names_layout():
    """The names list must have exactly 12 entries, in the same order as
    the encoded vector (so a debugger inspecting observation.state.names
    sees the right label at each index)."""
    names = task_channel_names()
    assert len(names) == ACT_TASK_VECTOR_DIM
    # First 7 are module names, next 3 are port_in_module, last 2 are types.
    for i, mod in enumerate(ACT_MODULE_ORDER):
        assert names[i] == f"task.module.{mod}"
    for i, port in enumerate(ACT_PORT_IN_MODULE_ORDER):
        assert names[ACT_MODULE_DIM + i] == f"task.port_in_module.{port}"
    for i, t in enumerate(ACT_PORT_TYPE_ORDER):
        assert names[ACT_MODULE_DIM + ACT_PORT_IN_MODULE_DIM + i] == f"task.port_type.{t}"


def test_task_string_format():
    """task_string_for() matches the documented natural-language format
    LeRobotDataset v3.0 stores per-episode."""
    s = task_string_for("nic_card_mount_3", "sfp_port_0", "sfp")
    assert s == "insert sfp plug into sfp_port_0 on nic_card_mount_3"
    s2 = task_string_for("sc_port_1", "sc_port_base", "sc")
    assert s2 == "insert sc plug into sc_port_base on sc_port_1"


def test_decode_rejects_non_one_hot():
    """A vector with sub-vectors not summing to 1 must be rejected by
    decode (e.g. an averaged / corrupted vector)."""
    bad = np.full(ACT_TASK_VECTOR_DIM, 0.5, dtype=np.float32)
    try:
        decode_task_vector(bad)
    except ValueError as ex:
        assert "sub-vector doesn't sum to 1" in str(ex)
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [
        test_dimensions,
        test_full_target_coverage,
        test_each_subvector_sums_to_one,
        test_round_trip,
        test_unknown_module_raises,
        test_unknown_port_name_raises,
        test_unknown_port_type_raises,
        test_port_type_mismatch_raises,
        test_invalid_combination_raises,
        test_task_channel_names_layout,
        test_task_string_format,
        test_decode_rejects_non_one_hot,
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
