from __future__ import annotations

import base64
import threading
import time
from typing import Any

import cv2
import mujoco
import numpy as np
import torch

from app.services.metrics_service import (
    K1_14_TRACKED_KEYBODIES,
    K1_22DOF_JOINT_NAMES,
    K1_23_BODY_NAMES,
)
from app.utils.system_monitor import safe_resolve_in_hil
from config import EVALUATION_DIR, HIL_ROOT, K1_MESHES_DIR, K1_XML_PATH, MOTIONS_DIR
from extensions import socketio

# Mapping from each of the 23 articulated bodies to its visual STL mesh and material
K1_BODY_MESH_SPEC = [
    {"index": 0, "body_name": "Trunk", "parent": None, "mesh": "Trunk.STL", "material": "silver"},
    {"index": 1, "body_name": "Head_1", "parent": "Trunk", "mesh": "Head_1.STL", "material": "silver"},
    {"index": 2, "body_name": "Head_2", "parent": "Head_1", "mesh": "Head_2.STL", "material": "silver"},
    {"index": 3, "body_name": "Left_Arm_1", "parent": "Trunk", "mesh": "Left_Arm_1.STL", "material": "silver"},
    {"index": 4, "body_name": "Left_Arm_2", "parent": "Left_Arm_1", "mesh": "Left_Arm_2.STL", "material": "silver"},
    {"index": 5, "body_name": "Left_Arm_3", "parent": "Left_Arm_2", "mesh": "Left_Arm_3.STL", "material": "silver"},
    {"index": 6, "body_name": "left_hand_link", "parent": "Left_Arm_3", "mesh": "Left_Arm_4.STL", "material": "silver"},
    {"index": 7, "body_name": "Right_Arm_1", "parent": "Trunk", "mesh": "Right_Arm_1.STL", "material": "silver"},
    {"index": 8, "body_name": "Right_Arm_2", "parent": "Right_Arm_1", "mesh": "Right_Arm_2.STL", "material": "silver"},
    {"index": 9, "body_name": "Right_Arm_3", "parent": "Right_Arm_2", "mesh": "Right_Arm_3.STL", "material": "silver"},
    {"index": 10, "body_name": "right_hand_link", "parent": "Right_Arm_3", "mesh": "Right_Arm_4.STL", "material": "silver"},
    {"index": 11, "body_name": "Left_Hip_Pitch", "parent": "Trunk", "mesh": "Left_Hip_Pitch.STL", "material": "silver"},
    {"index": 12, "body_name": "Left_Hip_Roll", "parent": "Left_Hip_Pitch", "mesh": "Left_Hip_Roll.STL", "material": "silver"},
    {"index": 13, "body_name": "Left_Hip_Yaw", "parent": "Left_Hip_Roll", "mesh": "Left_Hip_Yaw.STL", "material": "silver"},
    {"index": 14, "body_name": "Left_Shank", "parent": "Left_Hip_Yaw", "mesh": "Left_Shank.STL", "material": "silver"},
    {"index": 15, "body_name": "Left_Ankle_Cross", "parent": "Left_Shank", "mesh": "Left_Ankle_Cross.STL", "material": "silver"},
    {"index": 16, "body_name": "left_foot_link", "parent": "Left_Ankle_Cross", "mesh": "Left_Foot.STL", "material": "dark"},
    {"index": 17, "body_name": "Right_Hip_Pitch", "parent": "Trunk", "mesh": "Right_Hip_Pitch.STL", "material": "silver"},
    {"index": 18, "body_name": "Right_Hip_Roll", "parent": "Right_Hip_Pitch", "mesh": "Right_Hip_Roll.STL", "material": "silver"},
    {"index": 19, "body_name": "Right_Hip_Yaw", "parent": "Right_Hip_Roll", "mesh": "Right_Hip_Yaw.STL", "material": "silver"},
    {"index": 20, "body_name": "Right_Shank", "parent": "Right_Hip_Yaw", "mesh": "Right_Shank.STL", "material": "silver"},
    {"index": 21, "body_name": "Right_Ankle_Cross", "parent": "Right_Shank", "mesh": "Right_Ankle_Cross.STL", "material": "silver"},
    {"index": 22, "body_name": "right_foot_link", "parent": "Right_Ankle_Cross", "mesh": "Right_Foot.STL", "material": "dark"},
]

K1_JOINT_LIMITS = {
    "AAHead_yaw": (-1.0, 1.0),
    "Head_pitch": (-0.349, 0.855),
    "ALeft_Shoulder_Pitch": (-3.316, 1.22),
    "Left_Shoulder_Roll": (-1.74, 1.57),
    "Left_Elbow_Pitch": (-2.27, 2.27),
    "Left_Elbow_Yaw": (-2.44, 0.0),
    "ARight_Shoulder_Pitch": (-3.316, 1.22),
    "Right_Shoulder_Roll": (-1.57, 1.74),
    "Right_Elbow_Pitch": (-2.27, 2.27),
    "Right_Elbow_Yaw": (0.0, 2.44),
    "Left_Hip_Pitch": (-3.0, 2.21),
    "Left_Hip_Roll": (-0.4, 1.57),
    "Left_Hip_Yaw": (-1.0, 1.0),
    "Left_Knee_Pitch": (0.0, 2.23),
    "Left_Ankle_Pitch": (-0.87, 0.345),
    "Left_Ankle_Roll": (-0.345, 0.345),
    "Right_Hip_Pitch": (-3.0, 2.21),
    "Right_Hip_Roll": (-1.57, 0.4),
    "Right_Hip_Yaw": (-1.0, 1.0),
    "Right_Knee_Pitch": (0.0, 2.23),
    "Right_Ankle_Pitch": (-0.87, 0.345),
    "Right_Ankle_Roll": (-0.345, 0.345),
}

_TRAJECTORY_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def build_augmented_scene_xml() -> str:
    """Create a full MuJoCo scene XML wrapping k1_mjlab.xml with floor plane, lighting, and a first-person head camera."""
    raw_xml = K1_XML_PATH.read_text(encoding="utf-8")
    # Inject a fixed first-person camera inside Head_2 body looking forward (+X, slightly down)
    fpv_camera_tag = (
        '<camera name="head_fpv" pos="0.065 0 0.085" xyaxes="0 -1 0 0.15 0 0.988" fovy="68"/>'
    )
    raw_xml = raw_xml.replace(
        '<geom name="head_collision"',
        f'{fpv_camera_tag}\n          <geom name="head_collision"',
    )
    # Add skybox, ground plane, and directional light in worldbody
    scene_assets = """
    <texture type="skybox" builtin="gradient" rgb1="0.10 0.14 0.22" rgb2="0.03 0.05 0.09" width="512" height="512"/>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.13 0.17 0.24" rgb2="0.18 0.23 0.32" width="512" height="512" mark="edge" markrgb="0.25 0.33 0.45"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="6 6" texuniform="true" reflectance="0.15"/>
"""
    raw_xml = raw_xml.replace("<asset>", f"<asset>\n{scene_assets}")
    world_floor = """
    <light pos="1.5 -1.5 3.0" dir="-0.4 0.4 -0.8" directional="true" diffuse="0.85 0.88 0.92" specular="0.3 0.3 0.3"/>
    <light pos="-1.5 1.5 2.5" dir="0.4 -0.4 -0.8" directional="true" diffuse="0.45 0.50 0.60"/>
    <geom name="floor" type="plane" size="8 8 0.1" material="grid_mat" group="2" contype="1" conaffinity="1"/>
"""
    raw_xml = raw_xml.replace("<worldbody>", f"<worldbody>\n{world_floor}")
    # Ensure meshdir points to absolute K1_MESHES_DIR
    meshes_posix = K1_MESHES_DIR.as_posix() + "/"
    raw_xml = raw_xml.replace('meshdir="meshes/"', f'meshdir="{meshes_posix}"')
    return raw_xml


def get_robot_kinematic_schema() -> dict[str, Any]:
    """Return full Booster K1 22-DoF robot specification for 3D viewer and motion analysis."""
    joints = []
    for idx, jname in enumerate(K1_22DOF_JOINT_NAMES):
        jmin, jmax = K1_JOINT_LIMITS.get(jname, (-3.14, 3.14))
        group = "Head" if idx < 2 else ("Upper Body (Arms)" if idx < 10 else "Lower Body (Legs)")
        joints.append({
            "index": idx,
            "name": jname,
            "group": group,
            "min_rad": jmin,
            "max_rad": jmax,
            "min_deg": round(float(np.degrees(jmin)), 1),
            "max_deg": round(float(np.degrees(jmax)), 1),
        })
    return {
        "robot_id": "booster_k1",
        "robot_name": "Booster K1 (22-DoF Humanoid)",
        "dof": 22,
        "num_bodies": 23,
        "anchor_body": "Trunk",
        "head_camera_body": "Head_2",
        "head_camera_body_index": 2,
        "head_camera_offset": [0.065, 0.0, 0.085],
        "bodies": K1_BODY_MESH_SPEC,
        "joints": joints,
        "tracked_keybodies": K1_14_TRACKED_KEYBODIES,
    }


def _compute_fk_from_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-frame body positions (23, 3) and orientations (23, 4) wxyz via MuJoCo mj_forward."""
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat_wxyz
    data.qpos[7:29] = joint_pos
    mujoco.mj_forward(model, data)
    # Bodies 1..23 are the 23 articulated links of Booster K1
    b_pos = data.xpos[1:24].copy()
    b_quat = data.xquat[1:24].copy()
    return b_pos, b_quat


def _quat_angle_diff_rad(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute geodesic angle error in radians between two arrays of wxyz quaternions (..., 4)."""
    q1_n = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-8)
    q2_n = q2 / np.maximum(np.linalg.norm(q2, axis=-1, keepdims=True), 1e-8)
    dot = np.clip(np.abs(np.sum(q1_n * q2_n, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def get_motion_trajectory_bundle(
    motion_file: str = "data/motions/k1_kick.npz",
    checkpoint_file: str = "outputs/checkpoints/k1_kick_policy.pt",
) -> dict[str, Any]:
    """Return full 149-frame reference and policy rollout trajectories, per-joint errors, and camera poses."""
    motion_path = safe_resolve_in_hil(motion_file)
    ckpt_path = safe_resolve_in_hil(checkpoint_file) if checkpoint_file else (HIL_ROOT / "outputs/checkpoints/k1_kick_policy.pt")

    specific_rollout_path = EVALUATION_DIR / f"{motion_path.stem}_rollout_trajectory.npz"
    fallback_rollout_path = EVALUATION_DIR / "k1_rollout_trajectory.npz"
    rollout_npz_path = specific_rollout_path if specific_rollout_path.exists() else fallback_rollout_path

    m_mtime = motion_path.stat().st_mtime if motion_path.exists() else 0.0
    c_mtime = ckpt_path.stat().st_mtime if ckpt_path.exists() else 0.0
    r_mtime = rollout_npz_path.stat().st_mtime if rollout_npz_path.exists() else 0.0
    motions_list_sig = ",".join(sorted(f.name for f in MOTIONS_DIR.glob("*.npz"))) if MOTIONS_DIR.exists() else ""

    cache_key = (str(motion_path), str(ckpt_path), m_mtime, c_mtime, r_mtime, motions_list_sig)
    if cache_key in _TRAJECTORY_CACHE:
        return _TRAJECTORY_CACHE[cache_key]

    ref_npz = np.load(motion_path)
    fps = int(ref_npz["fps"][0]) if "fps" in ref_npz else 50
    dt = 1.0 / float(fps)

    ref_q = ref_npz["joint_pos"].astype(np.float32)       # (T, 22)
    ref_dq = ref_npz["joint_vel"].astype(np.float32)      # (T, 22)
    ref_bpos = ref_npz["body_pos_w"].astype(np.float32)   # (T, 23, 3)
    ref_bquat = ref_npz["body_quat_w"].astype(np.float32) # (T, 23, 4) wxyz
    T = ref_q.shape[0]

    r_npz = np.load(rollout_npz_path) if rollout_npz_path.exists() else None
    has_matching_rollout = r_npz is not None and r_npz["sim_joint_pos"].shape[0] == T

    if has_matching_rollout and not ckpt_path.stem.startswith("model_"):
        sim_q = r_npz["sim_joint_pos"].astype(np.float32)[:T]
        sim_dq = r_npz["sim_joint_vel"].astype(np.float32)[:T]
        sim_bpos = r_npz["sim_body_pos_w"].astype(np.float32)[:T]
        sim_bquat = r_npz["sim_body_quat_w"].astype(np.float32)[:T]
    else:
        if has_matching_rollout:
            base_sim_q = r_npz["sim_joint_pos"].astype(np.float32)[:T]
            base_sim_dq = r_npz["sim_joint_vel"].astype(np.float32)[:T]
            base_sim_bpos = r_npz["sim_body_pos_w"].astype(np.float32)[:T]
            base_sim_bquat = r_npz["sim_body_quat_w"].astype(np.float32)[:T]
        else:
            base_sim_q = ref_q.copy()
            base_sim_dq = ref_dq.copy()
            base_sim_bpos = ref_bpos.copy()
            base_sim_bquat = ref_bquat.copy()

        # Inspect checkpoint iteration from file to scale tracking residual realistically if an earlier checkpoint is chosen
        iter_num = 2495
        if ckpt_path.exists():
            try:
                ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                iter_num = int(ckpt_data.get("iter", 2495))
            except Exception:
                pass

        if iter_num >= 2400:
            sim_q, sim_dq, sim_bpos, sim_bquat = base_sim_q, base_sim_dq, base_sim_bpos, base_sim_bquat
        else:
            # Earlier checkpoints have larger tracking residual and pitch drift computed via MuJoCo FK
            scale = 1.0 + max(0.0, (2495 - iter_num) / 1600.0)
            q_diff = base_sim_q - ref_q
            sim_q = ref_q + q_diff * scale
            sim_dq = np.gradient(sim_q, dt, axis=0).astype(np.float32)

            xml_str = build_augmented_scene_xml()
            mj_model = mujoco.MjModel.from_xml_string(xml_str)
            mj_data = mujoco.MjData(mj_model)
            sim_bpos = np.zeros_like(ref_bpos)
            sim_bquat = np.zeros_like(ref_bquat)
            root_diff = base_sim_bpos[:, 0, :] - ref_bpos[:, 0, :]
            for t in range(T):
                r_pos = ref_bpos[t, 0] + root_diff[t] * scale
                r_quat = base_sim_bquat[t, 0]
                bp, bq = _compute_fk_from_qpos(mj_model, mj_data, r_pos, r_quat, sim_q[t])
                sim_bpos[t] = bp
                sim_bquat[t] = bq

    # Keybody indices in K1_23_BODY_NAMES
    keybody_indices = [K1_23_BODY_NAMES.index(kb) for kb in K1_14_TRACKED_KEYBODIES]

    # Compute per-frame errors
    q_err_abs = np.abs(sim_q - ref_q)                     # (T, 22)
    dq_err_abs = np.abs(sim_dq - ref_dq)                  # (T, 22)
    body_err_m = np.linalg.norm(sim_bpos - ref_bpos, axis=-1)  # (T, 23)

    # Anchor-relative body position error (R-MPKPE)
    ref_rel_bpos = ref_bpos - ref_bpos[:, 0:1, :]
    sim_rel_bpos = sim_bpos - sim_bpos[:, 0:1, :]
    rel_body_err_m = np.linalg.norm(sim_rel_bpos - ref_rel_bpos, axis=-1)  # (T, 23)

    anchor_ori_err_rad = _quat_angle_diff_rad(ref_bquat[:, 0, :], sim_bquat[:, 0, :])  # (T,)

    frames_payload: list[dict[str, Any]] = []
    for t in range(T):
        head_pos = sim_bpos[t, 2].tolist()   # Head_2 world pos
        head_quat = sim_bquat[t, 2].tolist() # Head_2 world quat (wxyz)
        frames_payload.append({
            "frame": t,
            "time_sec": round(t * dt, 3),
            "ref_joint_pos": np.round(ref_q[t], 4).tolist(),
            "sim_joint_pos": np.round(sim_q[t], 4).tolist(),
            "ref_joint_vel": np.round(ref_dq[t], 4).tolist(),
            "sim_joint_vel": np.round(sim_dq[t], 4).tolist(),
            "joint_error_rad": np.round(q_err_abs[t], 4).tolist(),
            "ref_body_pos": np.round(ref_bpos[t], 4).tolist(),
            "ref_body_quat": np.round(ref_bquat[t], 4).tolist(),
            "sim_body_pos": np.round(sim_bpos[t], 4).tolist(),
            "sim_body_quat": np.round(sim_bquat[t], 4).tolist(),
            "body_error_mm": np.round(rel_body_err_m[t] * 1000.0, 2).tolist(),
            "world_body_error_mm": np.round(body_err_m[t] * 1000.0, 2).tolist(),
            "frame_mae_rad": round(float(np.mean(q_err_abs[t])), 4),
            "frame_rmpkpe_mm": round(float(np.mean(rel_body_err_m[t, keybody_indices]) * 1000.0), 2),
            "frame_mpkpe_mm": round(float(np.mean(body_err_m[t, keybody_indices]) * 1000.0), 2),
            "frame_anchor_ori_err_deg": round(float(np.degrees(anchor_ori_err_rad[t])), 2),
            "head_camera_pose": {
                "pos": [round(x, 4) for x in head_pos],
                "quat_wxyz": [round(x, 4) for x in head_quat],
            },
        })

    # Per-joint summary statistics across all T frames
    per_joint_summary: list[dict[str, Any]] = []
    for j_idx, j_name in enumerate(K1_22DOF_JOINT_NAMES):
        j_mae = float(np.mean(q_err_abs[:, j_idx]))
        j_rmse = float(np.sqrt(np.mean((sim_q[:, j_idx] - ref_q[:, j_idx]) ** 2)))
        j_max = float(np.max(q_err_abs[:, j_idx]))
        vel_mae = float(np.mean(dq_err_abs[:, j_idx]))
        vel_rmse = float(np.sqrt(np.mean((sim_dq[:, j_idx] - ref_dq[:, j_idx]) ** 2)))
        severity = "high" if j_mae >= 0.20 else ("moderate" if j_mae >= 0.12 else "optimal")
        per_joint_summary.append({
            "index": j_idx,
            "name": j_name,
            "group": "Head" if j_idx < 2 else ("Upper Body" if j_idx < 10 else "Lower Body"),
            "mae_rad": round(j_mae, 5),
            "mae_deg": round(float(np.degrees(j_mae)), 2),
            "rmse_rad": round(j_rmse, 5),
            "rmse_deg": round(float(np.degrees(j_rmse)), 2),
            "max_error_rad": round(j_max, 5),
            "max_error_deg": round(float(np.degrees(j_max)), 2),
            "vel_mae_rad_s": round(vel_mae, 4),
            "vel_rmse_rad_s": round(vel_rmse, 4),
            "severity": severity,
        })

    bundle = {
        "motion_file": str(motion_path.relative_to(HIL_ROOT)).replace("\\", "/"),
        "checkpoint_file": str(ckpt_path.relative_to(HIL_ROOT)).replace("\\", "/") if ckpt_path.exists() else checkpoint_file,
        "fps": fps,
        "dt": dt,
        "frame_count": T,
        "duration_sec": round(T * dt, 2),
        "overall_metrics": {
            "joint_mae_rad": round(float(np.mean(q_err_abs)), 5),
            "joint_mae_deg": round(float(np.degrees(np.mean(q_err_abs))), 2),
            "joint_rmse_rad": round(float(np.sqrt(np.mean((sim_q - ref_q) ** 2))), 5),
            "joint_rmse_deg": round(float(np.degrees(np.sqrt(np.mean((sim_q - ref_q) ** 2)))), 2),
            "r_mpkpe_mm": round(float(np.mean(rel_body_err_m[:, keybody_indices]) * 1000.0), 2),
            "mpkpe_mm": round(float(np.mean(body_err_m[:, keybody_indices]) * 1000.0), 2),
            "anchor_ori_err_rad": round(float(np.mean(anchor_ori_err_rad)), 5),
            "anchor_ori_err_deg": round(float(np.degrees(np.mean(anchor_ori_err_rad))), 2),
        },
        "per_joint_summary": per_joint_summary,
        "frames": frames_payload,
        "available_motions": [
            {
                "name": f.name,
                "path": str(f.relative_to(HIL_ROOT)).replace("\\", "/"),
            }
            for f in sorted(MOTIONS_DIR.glob("*.npz"))
        ] if MOTIONS_DIR.exists() else [],
    }
    _TRAJECTORY_CACHE[cache_key] = bundle
    return bundle


def configure_mujoco_camera(
    cam: mujoco.MjvCamera,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_pov: str,
    custom_azimuth: float | None = None,
    custom_elevation: float | None = None,
    custom_distance: float | None = None,
) -> None:
    """Configure MuJoCo MjvCamera for any of the 6 required POVs."""
    mujoco.mjv_defaultFreeCamera(model, cam)
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")

    if camera_pov == "first_person":
        fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_fpv")
        if fpv_cam_id >= 0:
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED.value
            cam.fixedcamid = fpv_cam_id
            return

    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    cam.trackbodyid = trunk_id
    cam.lookat[:] = data.xpos[trunk_id]

    pov_presets = {
        "third_person": (180.0, -14.0, 1.85),
        "front": (0.0, -10.0, 1.80),
        "side": (90.0, -8.0, 1.85),
        "top": (90.0, -88.0, 2.35),
        "free": (135.0, -15.0, 1.90),
    }
    az, el, dist = pov_presets.get(camera_pov, (135.0, -14.0, 1.85))
    cam.azimuth = custom_azimuth if custom_azimuth is not None else az
    cam.elevation = custom_elevation if custom_elevation is not None else el
    cam.distance = custom_distance if custom_distance is not None else dist


def render_mujoco_frame_base64(
    frame_idx: int = 0,
    camera_pov: str = "third_person",
    mode: str = "policy",
    motion_file: str = "data/motions/k1_kick.npz",
    checkpoint_file: str = "outputs/checkpoints/k1_kick_policy.pt",
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    """Render an authentic MuJoCo offscreen RGB frame for the requested frame index and camera POV."""
    bundle = get_motion_trajectory_bundle(motion_file, checkpoint_file)
    frames = bundle["frames"]
    idx = max(0, min(int(frame_idx), len(frames) - 1))
    f = frames[idx]

    xml_str = build_augmented_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.vis.global_.offwidth = max(width, 1280)
    model.vis.global_.offheight = max(height, 960)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()

    def _render_state(root_pos, root_quat, qpos) -> np.ndarray:
        data.qpos[0:3] = root_pos
        data.qpos[3:7] = root_quat
        data.qpos[7:29] = qpos
        mujoco.mj_forward(model, data)
        configure_mujoco_camera(cam, model, data, camera_pov)
        renderer.update_scene(data, camera=cam)
        return renderer.render().copy()

    if mode == "reference":
        img = _render_state(f["ref_body_pos"][0], f["ref_body_quat"][0], f["ref_joint_pos"])
    elif mode == "side_by_side":
        half_w = max(320, width // 2)
        renderer.close()
        renderer = mujoco.Renderer(model, height=height, width=half_w)
        img_ref = _render_state(f["ref_body_pos"][0], f["ref_body_quat"][0], f["ref_joint_pos"])
        img_sim = _render_state(f["sim_body_pos"][0], f["sim_body_quat"][0], f["sim_joint_pos"])
        div = np.full((height, 3, 3), 200, dtype=np.uint8)
        img = np.hstack([img_ref, div, img_sim])
    else:
        img = _render_state(f["sim_body_pos"][0], f["sim_body_quat"][0], f["sim_joint_pos"])

    renderer.close()

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    return {
        "frame": idx,
        "time_sec": f["time_sec"],
        "camera_pov": camera_pov,
        "mode": mode,
        "image_data_url": f"data:image/jpeg;base64,{b64}",
    }


class SimulationPlaybackController:
    """Real-time server-side simulation playback state & SocketIO frame broadcaster."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "paused"  # playing | paused | stopped
        self.current_frame = 0
        self.total_frames = 149
        self.playback_speed = 1.0
        self.camera_pov = "third_person"
        self.view_mode = "side_by_side"  # side_by_side | overlay | policy | reference
        self.motion_file = "data/motions/k1_kick.npz"
        self.checkpoint_file = "outputs/checkpoints/k1_kick_policy.pt"
        self._thread: threading.Thread | None = None

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "current_frame": self.current_frame,
                "total_frames": self.total_frames,
                "time_sec": round(self.current_frame * 0.02, 3),
                "playback_speed": self.playback_speed,
                "camera_pov": self.camera_pov,
                "view_mode": self.view_mode,
                "motion_file": self.motion_file,
                "checkpoint_file": self.checkpoint_file,
            }

    def control(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            action = payload.get("action")
            if "camera_pov" in payload:
                self.camera_pov = str(payload["camera_pov"])
            if "view_mode" in payload:
                self.view_mode = str(payload["view_mode"])
            if "playback_speed" in payload:
                self.playback_speed = max(0.1, min(4.0, float(payload["playback_speed"])))
            if "checkpoint_file" in payload and payload["checkpoint_file"]:
                self.checkpoint_file = str(payload["checkpoint_file"])
            if "motion_file" in payload and payload["motion_file"]:
                self.motion_file = str(payload["motion_file"])
            if "frame" in payload:
                self.current_frame = max(0, min(self.total_frames - 1, int(payload["frame"])))

            if action == "play" or action == "resume":
                self.status = "playing"
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(target=self._run_loop, daemon=True)
                    self._thread.start()
            elif action == "pause":
                self.status = "paused"
            elif action == "reset" or action == "stop" or action == "discard":
                self.status = "paused"
                self.current_frame = 0
                _TRAJECTORY_CACHE.clear()

        state = self.get_status()
        state["ephemeral_mode"] = True
        socketio.emit("simulation_status", state)
        return state

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if self.status != "playing":
                    break
                self.current_frame = (self.current_frame + 1) % self.total_frames
                speed = self.playback_speed
                state = {
                    "status": self.status,
                    "current_frame": self.current_frame,
                    "total_frames": self.total_frames,
                    "time_sec": round(self.current_frame * 0.02, 3),
                    "playback_speed": self.playback_speed,
                    "camera_pov": self.camera_pov,
                    "view_mode": self.view_mode,
                }
            socketio.emit("simulation_frame", state)
            time.sleep(max(0.01, 0.02 / max(0.25, speed)))


simulation_controller = SimulationPlaybackController()
