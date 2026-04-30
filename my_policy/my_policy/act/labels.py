"""Task-vector encoding for the v9-act ACT policy.

The ACT policy needs to know which port to insert into. Unlike the localizer
(which outputs port-agnostic board pose + rail translation and adds the
per-port offset downstream via URDF math), ACT outputs ACTIONS directly —
the trajectory for "insert into nic_card_mount_0/sfp_port_0" ends ~2cm away
from "nic_card_mount_0/sfp_port_1" — same module, different port → totally
different action sequence. So the task vector must encode `port_name` too,
not just `target_module_name`.

Design: structured 12-dim concatenation of three one-hot sub-vectors:

    target_module_one_hot (7) || port_in_module_one_hot (3) || port_type_one_hot (2)

Rationale:
  - Structured (vs flat 12-dim one-hot): port_0 / port_1 within an SFP mount
    has consistent geometry across mounts (port_0 always at +1.1cm in
    mount-local frame, port_1 always at -1.2cm — see
    SFP_PORT_OFFSET_IN_MOUNT in localizer/labels.py). Structured encoding
    lets the model share parameters across "any sfp_port_0".
  - Explicit port_type: SFP and SC have different chamfer geometries,
    different insertion depths, different stuck-recovery patterns
    (CheatCodeRobust's spiral mode is per-plug-type — `x_only` for SC,
    `circular` for SFP). Removes the inference burden of deriving from the
    module name.

Total task space: 12 distinct (target_module, port_name) pairs:
  SFP: nic_card_mount_0..4 × sfp_port_0/sfp_port_1   (5 × 2 = 10)
  SC:  sc_port_0..1        × sc_port_base             (2 × 1 =  2)

Encoded deterministically from the known module/port topology — future port
additions = explicit edit here, never auto-derived from a YAML.

This module is imported by both the training preprocessor
(scripts/build_act_dataset.py) AND the inference shim
(my_policy/ros/RunACT.py) so the layout can never drift.
"""
from __future__ import annotations

import numpy as np


# --- Sub-vector orderings (LOCKED — appending only; never reorder existing) ---

ACT_MODULE_ORDER: tuple[str, ...] = (
    "nic_card_mount_0",
    "nic_card_mount_1",
    "nic_card_mount_2",
    "nic_card_mount_3",
    "nic_card_mount_4",
    "sc_port_0",
    "sc_port_1",
)
ACT_PORT_IN_MODULE_ORDER: tuple[str, ...] = (
    "sfp_port_0",
    "sfp_port_1",
    "sc_port_base",
)
ACT_PORT_TYPE_ORDER: tuple[str, ...] = (
    "sfp",
    "sc",
)


ACT_MODULE_DIM = len(ACT_MODULE_ORDER)
ACT_PORT_IN_MODULE_DIM = len(ACT_PORT_IN_MODULE_ORDER)
ACT_PORT_TYPE_DIM = len(ACT_PORT_TYPE_ORDER)
ACT_TASK_VECTOR_DIM = ACT_MODULE_DIM + ACT_PORT_IN_MODULE_DIM + ACT_PORT_TYPE_DIM
assert ACT_TASK_VECTOR_DIM == 12, f"unexpected dim {ACT_TASK_VECTOR_DIM}"


# --- Slice locations (so consumers can re-derive sub-vector positions) ---

ACT_MODULE_SLICE = slice(0, ACT_MODULE_DIM)
ACT_PORT_IN_MODULE_SLICE = slice(ACT_MODULE_DIM, ACT_MODULE_DIM + ACT_PORT_IN_MODULE_DIM)
ACT_PORT_TYPE_SLICE = slice(
    ACT_MODULE_DIM + ACT_PORT_IN_MODULE_DIM,
    ACT_TASK_VECTOR_DIM,
)


# --- Per-channel name list for inclusion in observation.state.names ---
# Keeps inspectors / debug viewers human-readable. Prefix avoids collision
# with existing recorder channel names like "tcp_pose.position.x".

def task_channel_names() -> list[str]:
    """Returns the 12 per-channel name strings for the task sub-vector,
    in the order they appear in the encoded vector. Suitable for adding to
    an extended `observation.state.names` list at preprocessing time."""
    out: list[str] = []
    for name in ACT_MODULE_ORDER:
        out.append(f"task.module.{name}")
    for name in ACT_PORT_IN_MODULE_ORDER:
        out.append(f"task.port_in_module.{name}")
    for name in ACT_PORT_TYPE_ORDER:
        out.append(f"task.port_type.{name}")
    assert len(out) == ACT_TASK_VECTOR_DIM
    return out


# --- Encoder ---

# Valid (target_module, port_name) combinations, derived from the known
# board topology. Used by encode() to fail-fast on unexpected inputs.
ACT_VALID_TARGETS: tuple[tuple[str, str], ...] = (
    # SFP: each NIC mount carries two SFP ports.
    *((mount, "sfp_port_0") for mount in (
        "nic_card_mount_0", "nic_card_mount_1", "nic_card_mount_2",
        "nic_card_mount_3", "nic_card_mount_4",
    )),
    *((mount, "sfp_port_1") for mount in (
        "nic_card_mount_0", "nic_card_mount_1", "nic_card_mount_2",
        "nic_card_mount_3", "nic_card_mount_4",
    )),
    # SC: each SC module carries one port.
    ("sc_port_0", "sc_port_base"),
    ("sc_port_1", "sc_port_base"),
)
assert len(ACT_VALID_TARGETS) == 12


def encode_task_vector(
    target_module_name: str,
    port_name: str,
    port_type: str,
) -> np.ndarray:
    """Returns a 12-dim float32 task vector encoding the named target.

    Validates that:
      - target_module_name is a known module
      - port_name is a known port-in-module
      - port_type matches the inferred type from port_name
      - (target_module_name, port_name) is a valid combination
        (e.g. rejects sc_port_0/sfp_port_0 mismatches)

    Raises ValueError on any mismatch — better to crash at preprocessing
    than to silently train on garbage labels.
    """
    if target_module_name not in ACT_MODULE_ORDER:
        raise ValueError(
            f"unknown target_module_name {target_module_name!r}; "
            f"expected one of {ACT_MODULE_ORDER}"
        )
    if port_name not in ACT_PORT_IN_MODULE_ORDER:
        raise ValueError(
            f"unknown port_name {port_name!r}; "
            f"expected one of {ACT_PORT_IN_MODULE_ORDER}"
        )
    if port_type not in ACT_PORT_TYPE_ORDER:
        raise ValueError(
            f"unknown port_type {port_type!r}; "
            f"expected one of {ACT_PORT_TYPE_ORDER}"
        )
    # port_name → expected port_type cross-check.
    expected_type = "sc" if port_name == "sc_port_base" else "sfp"
    if port_type != expected_type:
        raise ValueError(
            f"port_type {port_type!r} doesn't match port_name {port_name!r} "
            f"(expected {expected_type!r})"
        )
    if (target_module_name, port_name) not in ACT_VALID_TARGETS:
        raise ValueError(
            f"({target_module_name!r}, {port_name!r}) is not a valid target "
            f"combination; valid: {ACT_VALID_TARGETS}"
        )

    out = np.zeros(ACT_TASK_VECTOR_DIM, dtype=np.float32)
    out[ACT_MODULE_ORDER.index(target_module_name)] = 1.0
    out[ACT_MODULE_DIM + ACT_PORT_IN_MODULE_ORDER.index(port_name)] = 1.0
    out[ACT_MODULE_DIM + ACT_PORT_IN_MODULE_DIM
        + ACT_PORT_TYPE_ORDER.index(port_type)] = 1.0
    return out


def task_string_for(
    target_module_name: str,
    port_name: str,
    port_type: str,
) -> str:
    """Build the natural-language `task` string LeRobotDataset v3.0 stores
    per-episode. Stock ACT v0.5.1 doesn't consume this field — it's used by
    language-conditioned policies (SmolVLA, pi0). Populated as belt-and-
    suspenders metadata so a future escalation doesn't require re-recording.

    Example: 'insert sfp plug into sfp_port_0 on nic_card_mount_3'.
    """
    return f"insert {port_type} plug into {port_name} on {target_module_name}"


def decode_task_vector(
    vec: np.ndarray,
) -> tuple[str, str, str]:
    """Inverse of encode_task_vector — used by inspectors / sanity checks.

    Returns (target_module_name, port_name, port_type). Raises if any sub-
    vector doesn't sum to exactly 1.0 (i.e. the input wasn't a valid task
    vector).
    """
    if vec.shape != (ACT_TASK_VECTOR_DIM,):
        raise ValueError(f"expected shape ({ACT_TASK_VECTOR_DIM},), got {vec.shape}")
    mod_slice = vec[ACT_MODULE_SLICE]
    port_slice = vec[ACT_PORT_IN_MODULE_SLICE]
    type_slice = vec[ACT_PORT_TYPE_SLICE]
    for name, sub in (("module", mod_slice),
                      ("port_in_module", port_slice),
                      ("port_type", type_slice)):
        if not np.isclose(sub.sum(), 1.0, atol=1e-4):
            raise ValueError(
                f"{name} sub-vector doesn't sum to 1 (sum={sub.sum():.4f}); "
                f"input is not a valid one-hot task vector"
            )
    return (
        ACT_MODULE_ORDER[int(np.argmax(mod_slice))],
        ACT_PORT_IN_MODULE_ORDER[int(np.argmax(port_slice))],
        ACT_PORT_TYPE_ORDER[int(np.argmax(type_slice))],
    )
