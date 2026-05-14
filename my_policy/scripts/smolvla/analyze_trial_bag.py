#!/usr/bin/env python3
"""Replay a live-trial rosbag through a SmolVLA/ACT checkpoint offline,
comparing the model's predicted chunks to what the live policy actually
commanded.

For each /observations message in the bag we:
  1. Reconstruct the same obs dict that RunSmolVLA would build at
     inference time (port-local 26 or 38-dim state + 3 images + task).
  2. Run policy.predict_action_chunk(obs) to get the chunk the model
     WOULD have produced.
  3. Look up the live /aic_controller/pose_commands message active at
     that timestamp (last cmd published <= obs time) and record what
     the live system ACTUALLY commanded.

Output:
  * Per-tick CSV with columns:
      t_s, tcp_z_bl, tcp_z_port, tcp_vel_z_bl, wrench_fmag,
      live_cmd_z_bl, live_cmd_z_port,
      model_chunk_z0_port, model_chunk_zmax_port, model_chunk_zlast_port
  * stdout summary with port-local tcp_z buckets and live vs model means.

Usage:
    pixi run python my_policy/scripts/smolvla/analyze_trial_bag.py \\
        --bag /root/aic_data/.../rosbag2_<...> \\
        --checkpoint .../checkpoints/last/pretrained_model \\
        --port-frame task_board/sc_port_0/sc_port_base_link \\
        --port-type sc

If --port-frame is omitted, the first task_board/<module>/<port>_link
seen in /tf is used; warn if multiple distinct frames seen.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "my_policy") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "my_policy"))


# ---------------------------------------------------------------------------
# ROS / model imports deferred so --help works without sourcing ROS.
# ---------------------------------------------------------------------------

def _ros_imports():
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import (
        SequentialReader, StorageOptions, ConverterOptions,
    )
    return deserialize_message, get_message, SequentialReader, StorageOptions, ConverterOptions


def _model_imports():
    import torch
    import draccus
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.processor.pipeline import DataProcessorPipeline
    return {
        "torch": torch, "draccus": draccus, "load_file": load_file,
        "SmolVLAPolicy": SmolVLAPolicy, "SmolVLAConfig": SmolVLAConfig,
        "DataProcessorPipeline": DataProcessorPipeline,
    }


# ---------------------------------------------------------------------------
# Constants (mirror RunSmolVLA.py)
# ---------------------------------------------------------------------------

STATE_DIM_BASELINE = 26
STATE_DIM_CONDITIONED = 38

PORT_ENTRANCE_OFFSET_M = {"sc": 0.01564, "sfp": 0.0458}
PHASE_D_DESCEND_THRESH_M = 0.10
PHASE_D_INSERTED_THRESH_M = 0.005
PHASE_F_CONTACT_THRESH_N = 8.0

DEFAULT_TASK_STR = {
    "sc": "insert sc plug into sc_port_base on sc_port_0",
    "sfp": "insert sfp plug into sfp_port_0 on nic_card_mount_0",
}

IMAGE_SCALING = 0.25  # 1152x1024 -> 288x256


# ---------------------------------------------------------------------------
# State composers (copied verbatim semantics from RunSmolVLA)
# ---------------------------------------------------------------------------

def _compensated_wrench(obs_msg) -> tuple[float, ...]:
    raw = obs_msg.wrist_wrench.wrench
    tare = obs_msg.controller_state.fts_tare_offset.wrench
    return (
        raw.force.x - tare.force.x,
        raw.force.y - tare.force.y,
        raw.force.z - tare.force.z,
        raw.torque.x - tare.torque.x,
        raw.torque.y - tare.torque.y,
        raw.torque.z - tare.torque.z,
    )


def _build_state_26_np(
    obs_msg, port_pose_baselink: np.ndarray,
    joint_pos_fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the 26-dim port-local state.

    If `obs_msg.joint_states.position` has fewer than 7 elements (some bags
    don't carry joint_states inside /observations), `joint_pos_fallback`
    is used. Caller is responsible for sourcing the fallback from the bag's
    separate `/joint_states` topic. We do NOT silently zero-fill — that
    would produce a plausible but garbage state.
    """
    from my_policy.port_local.transforms import FrameInputs, transform_frame
    cs = obs_msg.controller_state
    tcp_pose = cs.tcp_pose
    tcp_vel = cs.tcp_velocity
    js = obs_msg.joint_states
    js_pos = list(js.position[:7]) if len(js.position) >= 7 else None
    if js_pos is None:
        if joint_pos_fallback is None or joint_pos_fallback.shape != (7,):
            raise RuntimeError(
                f"/observations.joint_states.position has "
                f"{len(js.position)} elements (<7) and no fallback from "
                f"/joint_states topic is available yet. State would be "
                f"19-dim instead of 26-dim.")
        js_pos = list(joint_pos_fallback)
    tcp_pose_bl = np.array([
        tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z,
        tcp_pose.orientation.x, tcp_pose.orientation.y,
        tcp_pose.orientation.z, tcp_pose.orientation.w,
    ], dtype=np.float64)
    tcp_vel_bl = np.array([
        tcp_vel.linear.x, tcp_vel.linear.y, tcp_vel.linear.z,
        tcp_vel.angular.x, tcp_vel.angular.y, tcp_vel.angular.z,
    ], dtype=np.float64)
    wrench_sensor = np.array(_compensated_wrench(obs_msg), dtype=np.float64)
    inp = FrameInputs(
        tcp_pose_baselink=tcp_pose_bl,
        tcp_velocity_baselink=tcp_vel_bl,
        wrench_sensorframe=wrench_sensor,
        action_baselink=tcp_pose_bl,
        port_pose_baselink=port_pose_baselink,
    )
    out = transform_frame(inp)
    return np.array([
        *out.tcp_pose_portframe,
        *out.tcp_velocity_portframe,
        *js_pos,
        *out.wrench_portframe,
    ], dtype=np.float32)


def _classify_phase(depth: float, wrench_fxyz: np.ndarray) -> int:
    fmag = float(np.linalg.norm(wrench_fxyz))
    if fmag >= PHASE_F_CONTACT_THRESH_N:
        return 2
    if depth <= PHASE_D_INSERTED_THRESH_M:
        return 3
    if depth <= PHASE_D_DESCEND_THRESH_M:
        return 1
    return 0


def _build_state_38_np(
    obs_msg, port_pose_baselink: np.ndarray,
    port_type: str, prev_action_port: np.ndarray,
    joint_pos_fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    state_26 = _build_state_26_np(obs_msg, port_pose_baselink, joint_pos_fallback)
    entrance_offset = PORT_ENTRANCE_OFFSET_M[port_type]
    tcp_z_port = float(state_26[2])
    depth = -tcp_z_port - entrance_offset
    phase = _classify_phase(depth, state_26[20:23])
    phase_onehot = np.zeros(4, dtype=np.float32)
    phase_onehot[phase] = 1.0
    return np.concatenate([
        state_26,
        prev_action_port.astype(np.float32),
        np.array([depth], dtype=np.float32),
        phase_onehot,
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Image preprocessing (mirror RunSmolVLA._ros_image_to_chw_float)
# ---------------------------------------------------------------------------

def _ros_image_to_chw_float(ros_img, torch_mod):
    """Decode a sensor_msgs/Image and downsample to (3, 256, 288) on cuda.

    Avoids cv2 because libtiff in some pixi envs throws at decode time;
    torch.nn.functional.interpolate(mode='area') is mathematically
    equivalent to cv2.INTER_AREA for integer downscale factors (we go
    1152x1024 -> 288x256, exact 4x downscale), so the model sees the
    same pixel values as during live inference.
    """
    img_np = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(
        ros_img.height, ros_img.width, 3
    )
    # HWC uint8 -> CHW float [0,1] on cuda, then resize via torch.
    t = (
        torch_mod.from_numpy(img_np.copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .cuda()
    )
    if IMAGE_SCALING != 1.0:
        target_h = int(round(ros_img.height * IMAGE_SCALING))
        target_w = int(round(ros_img.width * IMAGE_SCALING))
        t = torch_mod.nn.functional.interpolate(
            t, size=(target_h, target_w), mode="area",
        )
    return t


# ---------------------------------------------------------------------------
# Port-local transform for a single pose (TCP or commanded pose)
# ---------------------------------------------------------------------------

def transform_pose_to_portlocal(
    pose_bl_xyz: np.ndarray, pose_bl_quat: np.ndarray,
    port_xyz: np.ndarray, port_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from my_policy.port_local.transforms import FrameInputs, transform_frame
    inp = FrameInputs(
        tcp_pose_baselink=np.concatenate([pose_bl_xyz, pose_bl_quat]),
        tcp_velocity_baselink=np.zeros(6),
        wrench_sensorframe=np.zeros(6),
        action_baselink=np.concatenate([pose_bl_xyz, pose_bl_quat]),
        port_pose_baselink=np.concatenate([port_xyz, port_quat]),
    )
    out = transform_frame(inp)
    return out.tcp_pose_portframe[:3], out.tcp_pose_portframe[3:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to .../checkpoints/<step>/pretrained_model/")
    p.add_argument("--port-frame", type=str, default=None,
                   help="Active port TF frame (e.g. "
                        "'task_board/sc_port_0/sc_port_base_link'). "
                        "Auto-detected if omitted.")
    p.add_argument("--port-type", type=str, default="sc",
                   choices=["sc", "sfp"])
    p.add_argument("--output", type=Path, default=None,
                   help="CSV path (default: <bag>/replay_summary.csv).")
    p.add_argument("--max-ticks", type=int, default=0,
                   help="Limit number of observation ticks to process "
                        "(0 = all). Useful for quick smoke test.")
    p.add_argument("--per-chunk-seed", type=int, default=0,
                   help="RNG seed reset before each predict_action_chunk "
                        "call (default 0 — matches RunSmolVLA's "
                        "AIC_PL_SMOLVLA_PER_CHUNK_SEED default). Set -1 "
                        "to disable seeding and use whatever global RNG "
                        "state happens to be active.")
    args = p.parse_args()

    output = args.output or (args.bag / "replay_summary.csv")

    print(f"Bag        : {args.bag}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Port type  : {args.port_type}  entrance_offset="
          f"{PORT_ENTRANCE_OFFSET_M[args.port_type]:+.5f}m")
    print(f"Output CSV : {output}")

    # --- model load --------------------------------------------------------
    mod = _model_imports()
    cfg_dict = json.loads((args.checkpoint / "config.json").read_text())
    cfg_dict.pop("type", None); cfg_dict.pop("rtc_config", None)
    config = mod["draccus"].decode(mod["SmolVLAConfig"], cfg_dict)
    policy = mod["SmolVLAPolicy"](config)
    policy.load_state_dict(mod["load_file"](str(args.checkpoint / "model.safetensors")))
    policy.eval(); policy.to("cuda")
    pre = mod["DataProcessorPipeline"].from_pretrained(
        str(args.checkpoint), config_filename="policy_preprocessor.json",
    )

    state_feat = config.input_features.get("observation.state")
    state_dim = int(state_feat.shape[0]) if state_feat is not None else 26
    if state_dim not in (STATE_DIM_BASELINE, STATE_DIM_CONDITIONED):
        print(f"WARN: unexpected model state_dim={state_dim}; assuming baseline 26")
        state_dim = STATE_DIM_BASELINE
    print(f"Model state_dim={state_dim} "
          f"({'CONDITIONED' if state_dim == STATE_DIM_CONDITIONED else 'baseline'})")

    # Postprocessor for un-normalizing actions to port-local meters.
    from safetensors import safe_open
    post_json = args.checkpoint / "policy_postprocessor.json"
    action_mode = "MEAN_STD"
    if post_json.exists():
        pc = json.loads(post_json.read_text())
        for step in pc.get("steps", []):
            nm = step.get("config", {}).get("norm_map") if step.get("config") else None
            if nm and "ACTION" in nm:
                action_mode = nm["ACTION"]
                break
    safet = next(args.checkpoint.glob(
        "policy_postprocessor_step_*_unnormalizer_processor.safetensors"), None)
    if safet is None:
        raise SystemExit("postprocessor safetensors not found")
    with safe_open(str(safet), framework="numpy") as f:
        am = f.get_tensor("action.mean"); asd = f.get_tensor("action.std")
        try:
            amin = f.get_tensor("action.min"); amax = f.get_tensor("action.max")
        except Exception:
            amin = amax = None
    if action_mode == "MIN_MAX" and amin is not None and amax is not None:
        mid = (float(amin[2]) + float(amax[2])) / 2.0
        half = (float(amax[2]) - float(amin[2])) / 2.0
        unnorm_z = lambda raw: mid + raw * half
        print(f"Action norm = MIN_MAX  (min={amin[2]:+.5f}, max={amax[2]:+.5f})")
    else:
        m, sd = float(am[2]), float(asd[2])
        unnorm_z = lambda raw: m + raw * sd
        print(f"Action norm = MEAN_STD  (mean={m:+.5f}, std={sd:+.5f})")

    # --- bag open ---------------------------------------------------------
    deserialize_message, get_message, SequentialReader, StorageOptions, ConverterOptions = _ros_imports()
    reader = SequentialReader()
    storage_id = "mcap" if any(args.bag.glob("*.mcap")) else "sqlite3"
    reader.open(
        StorageOptions(uri=str(args.bag), storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_classes = {n: get_message(t) for n, t in topic_types.items()}
    needed = (
        "/observations",
        "/aic_controller/pose_commands",
        "/tf",
        "/tf_static",
    )
    for t in needed:
        if t not in topic_types:
            raise SystemExit(f"bag missing required topic: {t}")
    has_joint_states_topic = "/joint_states" in topic_types
    print(f"Bag topics confirmed."
          + ("" if has_joint_states_topic
             else "  (no /joint_states topic; will rely on /observations.joint_states only)"))

    # --- pass 1: find port pose ------------------------------------------
    port_pose_bl = None
    port_frame_name = args.port_frame
    print(f"\nScanning /tf and /tf_static for port frame…")
    # Auto-detection must mirror what RunSmolVLA picks up at inference:
    #   `task_board/<module>/<port_name>_link`  (NOT `_link_entrance`,
    #   which is a child frame offset by the entrance distance — using it
    #   shifts port-local tcp_z by ~15-46 mm and the model sees the wrong
    #   frame).
    # Heuristic: child must START WITH task_board/, END WITH _link, and
    # NOT contain `_link_entrance` or `_link_plug_anchor`.
    def _is_target_port_link(child: str) -> bool:
        if not child.startswith("task_board/"):
            return False
        if not child.endswith("_link"):
            return False
        # Defensive: reject any suffixed link variants.
        for suffix in ("_link_entrance", "_link_plug_anchor"):
            if suffix in child:
                return False
        return True

    while reader.has_next() and port_pose_bl is None:
        topic, data, t_ns = reader.read_next()
        if topic not in ("/tf", "/tf_static"):
            continue
        msg = deserialize_message(data, msg_classes[topic])
        for tf in msg.transforms:
            child = tf.child_frame_id
            if port_frame_name is not None:
                if child != port_frame_name:
                    continue
            else:
                if not _is_target_port_link(child):
                    continue
            tr = tf.transform.translation; rot = tf.transform.rotation
            port_pose_bl = (
                np.array([tr.x, tr.y, tr.z]),
                np.array([rot.x, rot.y, rot.z, rot.w]),
            )
            port_frame_name = child
            print(f"  found port frame: {child} → "
                  f"xyz=({tr.x:+.4f},{tr.y:+.4f},{tr.z:+.4f}) "
                  f"q=({rot.x:+.3f},{rot.y:+.3f},{rot.z:+.3f},{rot.w:+.3f})")
            break
    if port_pose_bl is None:
        raise SystemExit("could not find port frame in /tf or /tf_static")

    # Re-open reader for full pass.
    del reader
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(args.bag), storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr",
                         output_serialization_format="cdr"),
    )

    # --- pass 2: stream and process --------------------------------------
    rows: list[dict] = []
    last_cmd_z_bl = float("nan")
    last_cmd_z_port = float("nan")
    last_obs_msg = None
    last_a_port_for_conditioning = np.zeros(7, dtype=np.float32)
    # Fallback joint positions sourced from /joint_states topic; used only
    # when /observations.joint_states.position is empty (some bag flavors
    # don't carry joint_states inside /observations).
    last_joint_pos: Optional[np.ndarray] = None
    joint_fallback_used = 0
    n_obs = 0

    print(f"\nReplaying observations through model…")
    torch_mod = mod["torch"]
    first_diag = True
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == "/joint_states":
            js_msg = deserialize_message(data, msg_classes[topic])
            if len(js_msg.position) >= 7:
                last_joint_pos = np.array(js_msg.position[:7], dtype=np.float32)
            continue

        if topic == "/aic_controller/pose_commands":
            cmd = deserialize_message(data, msg_classes[topic])
            try:
                cp = cmd.pose.pose
            except AttributeError:
                try:
                    cp = cmd.pose
                except AttributeError:
                    continue
            last_cmd_z_bl = float(cp.position.z)
            cmd_xyz = np.array([cp.position.x, cp.position.y, cp.position.z])
            cmd_q = np.array([cp.orientation.x, cp.orientation.y,
                              cp.orientation.z, cp.orientation.w])
            cmd_pose_port, _ = transform_pose_to_portlocal(
                cmd_xyz, cmd_q, port_pose_bl[0], port_pose_bl[1])
            last_cmd_z_port = float(cmd_pose_port[2])
            continue

        if topic != "/observations":
            continue
        n_obs += 1
        if args.max_ticks and n_obs > args.max_ticks:
            break

        obs_msg = deserialize_message(data, msg_classes[topic])

        # Build state (matching model's state_dim).
        if len(obs_msg.joint_states.position) < 7:
            if last_joint_pos is None:
                # No fallback yet (first obs may arrive before first
                # /joint_states msg). Skip this tick.
                continue
            joint_fallback_used += 1
        try:
            if state_dim == STATE_DIM_BASELINE:
                state_np = _build_state_26_np(
                    obs_msg, np.concatenate(port_pose_bl),
                    joint_pos_fallback=last_joint_pos,
                )
            else:
                state_np = _build_state_38_np(
                    obs_msg, np.concatenate(port_pose_bl),
                    args.port_type, last_a_port_for_conditioning,
                    joint_pos_fallback=last_joint_pos,
                )
        except RuntimeError as e:
            print(f"  skip obs {n_obs} ({e})")
            continue
        if first_diag:
            first_diag = False
            print(f"  [first obs diag] joint_states.position len="
                  f"{len(obs_msg.joint_states.position)}  "
                  f"state_np.shape={state_np.shape}  expected={state_dim}")
        if state_np.shape[0] != state_dim:
            raise RuntimeError(
                f"composed state has {state_np.shape[0]} dims but model "
                f"expects {state_dim}")
        state_t = torch_mod.from_numpy(state_np).unsqueeze(0).cuda()

        # Build images.
        try:
            imgs = {
                "observation.images.left_camera":
                    _ros_image_to_chw_float(obs_msg.left_image, torch_mod),
                "observation.images.center_camera":
                    _ros_image_to_chw_float(obs_msg.center_image, torch_mod),
                "observation.images.right_camera":
                    _ros_image_to_chw_float(obs_msg.right_image, torch_mod),
            }
        except Exception as e:
            print(f"  skip obs {n_obs} (image error: {e})")
            continue

        raw_obs = {
            **imgs,
            "observation.state": state_t,
            "task": DEFAULT_TASK_STR[args.port_type],
        }
        obs = pre(raw_obs)
        # Reset RNG before each chunk inference so flow-matching noise
        # is deterministic per (state, image). Mirrors RunSmolVLA's
        # _seed_before_inference; without it the replay's per-chunk
        # outputs differ from the live trial purely due to noise drift.
        if args.per_chunk_seed >= 0:
            torch_mod.manual_seed(args.per_chunk_seed)
            if torch_mod.cuda.is_available():
                torch_mod.cuda.manual_seed_all(args.per_chunk_seed)
        with torch_mod.no_grad():
            actions = policy.predict_action_chunk(obs)
        az_raw = actions[0, :, 2].cpu().numpy()
        # Track first action's port-local pose for prev_action conditioning.
        a_full = actions[0, 0, :7].cpu().numpy().astype(np.float32)
        last_a_port_for_conditioning = a_full

        cs = obs_msg.controller_state
        tcp_xyz = np.array([cs.tcp_pose.position.x, cs.tcp_pose.position.y, cs.tcp_pose.position.z])
        tcp_q = np.array([cs.tcp_pose.orientation.x, cs.tcp_pose.orientation.y,
                          cs.tcp_pose.orientation.z, cs.tcp_pose.orientation.w])
        tcp_xyz_port, _ = transform_pose_to_portlocal(
            tcp_xyz, tcp_q, port_pose_bl[0], port_pose_bl[1])

        # Wrench |F|.
        wf = _compensated_wrench(obs_msg)
        fmag = float(np.linalg.norm(np.array(wf[:3])))

        rows.append({
            "t_ns": t_ns,
            "tcp_z_bl": float(cs.tcp_pose.position.z),
            "tcp_z_port": float(tcp_xyz_port[2]),
            "tcp_vel_z_bl": float(cs.tcp_velocity.linear.z),
            "wrench_fmag": fmag,
            "live_cmd_z_bl": last_cmd_z_bl,
            "live_cmd_z_port": last_cmd_z_port,
            "model_raw_z_0": float(az_raw[0]),
            "model_raw_z_max": float(az_raw.max()),
            "model_raw_z_last": float(az_raw[-1]),
            "model_un_z_0": float(unnorm_z(float(az_raw[0]))),
            "model_un_z_max": float(unnorm_z(float(az_raw.max()))),
            "model_un_z_last": float(unnorm_z(float(az_raw[-1]))),
        })

    print(f"  processed {len(rows)} observation ticks")
    if joint_fallback_used:
        print(f"  NOTE: /observations.joint_states was empty for "
              f"{joint_fallback_used}/{len(rows)+joint_fallback_used} ticks; "
              f"fell back to /joint_states topic for those.")
    if not rows:
        print("nothing to write")
        return 1

    # Normalize time.
    t0 = rows[0]["t_ns"]
    for r in rows:
        r["t_s"] = (r["t_ns"] - t0) / 1e9
        del r["t_ns"]

    # CSV.
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_s", "tcp_z_bl", "tcp_z_port", "tcp_vel_z_bl", "wrench_fmag",
        "live_cmd_z_bl", "live_cmd_z_port",
        "model_raw_z_0", "model_raw_z_max", "model_raw_z_last",
        "model_un_z_0", "model_un_z_max", "model_un_z_last",
    ]
    with open(output, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {output}")

    # Summary.
    tcp_z = np.array([r["tcp_z_port"] for r in rows])
    live_cmd = np.array([r["live_cmd_z_port"] for r in rows])
    model_last = np.array([r["model_un_z_last"] for r in rows])
    model_max = np.array([r["model_un_z_max"] for r in rows])
    fmag = np.array([r["wrench_fmag"] for r in rows])

    print()
    print(f"=== Summary ===  duration={rows[-1]['t_s']:.1f}s  ticks={len(rows)}")
    print(f"port-local tcp_z      : [{tcp_z.min():+.4f}, {tcp_z.max():+.4f}]  "
          f"mean={tcp_z.mean():+.4f}")
    print(f"port-local live_cmd_z : [{np.nanmin(live_cmd):+.4f}, "
          f"{np.nanmax(live_cmd):+.4f}]  mean={np.nanmean(live_cmd):+.4f}")
    print(f"model un_z_last       : [{model_last.min():+.4f}, "
          f"{model_last.max():+.4f}]  mean={model_last.mean():+.4f}")
    print(f"|F|                   : max={fmag.max():.2f}N  "
          f"mean={fmag.mean():.2f}N")

    print()
    print(f"--- Per-tcp_z bucket: live vs model commanded ---")
    print(f"  {'tcp_z range':>20}  {'n':>5}  "
          f"{'live_cmd_z μ':>14}  {'model_un_z_last μ':>18}  "
          f"{'Δ (live-model) μ':>17}  {'|F| μ':>7}")
    bins = np.arange(-0.32, 0.01, 0.025)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (tcp_z >= lo) & (tcp_z < hi)
        if not mask.any():
            continue
        n = int(mask.sum())
        l = float(np.nanmean(live_cmd[mask]))
        m_ = float(np.nanmean(model_last[mask]))
        d = l - m_
        f_ = float(np.nanmean(fmag[mask]))
        print(f"  [{lo:+.3f},{hi:+.3f})  {n:>5}  "
              f"{l:+12.4f}    {m_:+15.4f}    {d:+15.4f}    {f_:7.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
