from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Any

import cv2
import imageio
import mujoco
import numpy as np

from app.models.database import VideoRecord, runtime_store
from app.services.simulation_service import (
    build_augmented_scene_xml,
    configure_mujoco_camera,
    get_motion_trajectory_bundle,
)
from config import HIL_ROOT, VIDEOS_DIR
from extensions import socketio


def _draw_telemetry_hud(
    frame_rgb: np.ndarray,
    title: str,
    subtitle: str,
    frame_meta: dict[str, Any],
    camera_pov: str,
    include_error_overlay: bool,
) -> np.ndarray:
    """Draw a scientific HUD banner and tracking error telemetry overlay onto an RGB frame."""
    out = frame_rgb.copy()
    h, w, _ = out.shape

    banner_h = max(48, int(h * 0.085))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (11, 15, 25), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

    font_scale_title = max(0.48, w / 1450.0)
    font_scale_sub = max(0.38, w / 1850.0)
    cv2.putText(
        out,
        title,
        (16, int(banner_h * 0.46)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale_title,
        (245, 248, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"{subtitle} | POV: {camera_pov.upper()} | Frame {frame_meta['frame']:03d} ({frame_meta['time_sec']:.2f}s)",
        (16, int(banner_h * 0.84)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale_sub,
        (125, 211, 252),
        1,
        cv2.LINE_AA,
    )

    if include_error_overlay:
        panel_w = min(w - 24, 340)
        panel_h = 114
        x0 = w - panel_w - 12
        y0 = h - panel_h - 12
        hud_bg = out.copy()
        cv2.rectangle(hud_bg, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 23, 42), -1)
        cv2.addWeighted(hud_bg, 0.84, out, 0.16, 0, out)
        cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), (56, 189, 248), 1)

        mae_rad = frame_meta["frame_mae_rad"]
        rmpkpe_mm = frame_meta["frame_rmpkpe_mm"]
        mpkpe_mm = frame_meta["frame_mpkpe_mm"]
        ori_deg = frame_meta["frame_anchor_ori_err_deg"]

        lines = [
            f"Joint Tracking MAE:  {mae_rad:.4f} rad ({np.degrees(mae_rad):.1f} deg)",
            f"Keybody R-MPKPE:     {rmpkpe_mm:.1f} mm (World: {mpkpe_mm:.1f} mm)",
            f"Anchor Ori Error:    {ori_deg:.2f} deg",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(
                out,
                txt,
                (x0 + 12, y0 + 26 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (226, 232, 240),
                1,
                cv2.LINE_AA,
            )

        bar_x = x0 + 12
        bar_y = y0 + 94
        bar_max_w = panel_w - 24
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 8), (30, 41, 59), -1)
        fill_ratio = min(1.0, rmpkpe_mm / 100.0)
        fill_w = int(bar_max_w * fill_ratio)
        bar_color = (52, 211, 153) if rmpkpe_mm < 50 else ((251, 146, 60) if rmpkpe_mm < 75 else (244, 63, 94))
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + 8), bar_color, -1)

    return out


class VideoGenerationService:
    """Stateless background job runner for MuJoCo simulation video generation (no database required)."""

    def __init__(self) -> None:
        self._app = None
        self._lock = threading.Lock()

    def init_app(self, app) -> None:
        self._app = app

    def launch_video_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        video_type = str(payload.get("video_type", "policy")).strip()
        if video_type not in ("policy", "reference", "comparison", "error_overlay"):
            video_type = "policy"

        camera_pov = str(payload.get("camera_pov", "third_person")).strip()
        if camera_pov not in ("third_person", "front", "side", "top", "free", "first_person"):
            camera_pov = "third_person"

        resolution = str(payload.get("resolution", "960x720")).strip()
        res_map = {
            "640x480": (640, 480),
            "960x720": (960, 720),
            "1280x720": (1280, 720),
            "1920x1080": (1920, 1080),
        }
        if resolution not in res_map:
            resolution = "960x720"

        fps = int(payload.get("fps", 50))
        fps = max(15, min(60, fps))
        playback_speed = float(payload.get("playback_speed", 1.0))
        playback_speed = max(0.25, min(2.0, playback_speed))
        include_overlay = bool(payload.get("include_overlay", True)) or (video_type == "error_overlay")

        motion_file = str(payload.get("motion_file", "data/motions/k1_kick.npz")).strip()
        checkpoint_path = str(payload.get("checkpoint_path", "outputs/checkpoints/k1_kick_policy.pt")).strip()
        experiment_id = payload.get("experiment_id")
        checkpoint_id = payload.get("checkpoint_id")

        job_id = f"vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        type_labels = {
            "policy": "BeyondMimic PPO Policy Rollout",
            "reference": "GMR/MJLab Kinematic Reference",
            "comparison": "Side-by-Side Comparison (Ref vs Policy)",
            "error_overlay": "Policy Rollout with Tracking Error Telemetry",
        }
        title = str(payload.get("title") or f"{type_labels[video_type]} ({camera_pov.replace('_', ' ').title()}, {resolution})").strip()

        vrec = VideoRecord(
            id=runtime_store.next_vid_id(),
            job_id=job_id,
            experiment_id=int(experiment_id) if experiment_id else None,
            checkpoint_id=int(checkpoint_id) if checkpoint_id else None,
            title=title,
            video_type=video_type,
            camera_pov=camera_pov,
            resolution=resolution,
            fps=fps,
            playback_speed=playback_speed,
            include_overlay=include_overlay,
            motion_file=motion_file,
            checkpoint_path=checkpoint_path,
            status="processing",
            progress=0,
        )
        runtime_store.videos[vrec.id] = vrec
        runtime_store.add_activity(
            category="video",
            action=f"Queued Ephemeral Video Job: {title}",
            details=f"Type={video_type}, POV={camera_pov}, Resolution={resolution}, FPS={fps}, Speed={playback_speed}x",
            status="info",
            experiment_id=vrec.experiment_id,
        )
        rec_dict = vrec.to_dict()

        t = threading.Thread(
            target=self._execute_video_job,
            args=(vrec.id, res_map[resolution]),
            daemon=True,
        )
        t.start()
        return rec_dict

    def _execute_video_job(self, record_id: int, dims: tuple[int, int]) -> None:
        vrec = runtime_store.videos.get(record_id)
        if not vrec:
            return

        try:
            width, height = dims
            bundle = get_motion_trajectory_bundle(vrec.motion_file, vrec.checkpoint_path or "outputs/checkpoints/k1_kick_policy.pt")
            frames_data = bundle["frames"]
            total_steps = len(frames_data)

            xml_str = build_augmented_scene_xml()
            model = mujoco.MjModel.from_xml_string(xml_str)
            model.vis.global_.offwidth = max(width, 1920)
            model.vis.global_.offheight = max(height, 1080)
            data = mujoco.MjData(model)
            cam = mujoco.MjvCamera()

            VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            out_filename = f"{vrec.job_id}_{vrec.video_type}_{vrec.camera_pov}.mp4"
            out_path = VIDEOS_DIR / out_filename

            rendered_frames: list[np.ndarray] = []

            if vrec.video_type == "comparison":
                panel_w = max(320, width // 2)
                renderer = mujoco.Renderer(model, height=height, width=panel_w)
            else:
                renderer = mujoco.Renderer(model, height=height, width=width)

            for idx, fmeta in enumerate(frames_data):
                if vrec.video_type == "reference":
                    data.qpos[0:3] = fmeta["ref_body_pos"][0]
                    data.qpos[3:7] = fmeta["ref_body_quat"][0]
                    data.qpos[7:29] = fmeta["ref_joint_pos"]
                    mujoco.mj_forward(model, data)
                    configure_mujoco_camera(cam, model, data, vrec.camera_pov)
                    renderer.update_scene(data, camera=cam)
                    raw_rgb = renderer.render().copy()
                    frame_out = _draw_telemetry_hud(
                        raw_rgb,
                        "GMR / MJLab Reference Motion (Kinematic)",
                        "Booster K1 (22-DoF)",
                        fmeta,
                        vrec.camera_pov,
                        False,
                    )
                elif vrec.video_type == "comparison":
                    data.qpos[0:3] = fmeta["ref_body_pos"][0]
                    data.qpos[3:7] = fmeta["ref_body_quat"][0]
                    data.qpos[7:29] = fmeta["ref_joint_pos"]
                    mujoco.mj_forward(model, data)
                    configure_mujoco_camera(cam, model, data, vrec.camera_pov)
                    renderer.update_scene(data, camera=cam)
                    left_rgb = _draw_telemetry_hud(
                        renderer.render().copy(),
                        "Reference Motion (GMR / MJLab)",
                        "Booster K1 (22-DoF)",
                        fmeta,
                        vrec.camera_pov,
                        False,
                    )

                    data.qpos[0:3] = fmeta["sim_body_pos"][0]
                    data.qpos[3:7] = fmeta["sim_body_quat"][0]
                    data.qpos[7:29] = fmeta["sim_joint_pos"]
                    mujoco.mj_forward(model, data)
                    configure_mujoco_camera(cam, model, data, vrec.camera_pov)
                    renderer.update_scene(data, camera=cam)
                    right_rgb = _draw_telemetry_hud(
                        renderer.render().copy(),
                        "BeyondMimic PPO Policy (Dynamic MuJoCo)",
                        "Booster K1 (22-DoF)",
                        fmeta,
                        vrec.camera_pov,
                        vrec.include_overlay,
                    )
                    divider = np.full((height, 4, 3), 56, dtype=np.uint8)
                    frame_out = np.hstack([left_rgb, divider, right_rgb])
                else:
                    data.qpos[0:3] = fmeta["sim_body_pos"][0]
                    data.qpos[3:7] = fmeta["sim_body_quat"][0]
                    data.qpos[7:29] = fmeta["sim_joint_pos"]
                    mujoco.mj_forward(model, data)
                    configure_mujoco_camera(cam, model, data, vrec.camera_pov)
                    renderer.update_scene(data, camera=cam)
                    raw_rgb = renderer.render().copy()
                    frame_out = _draw_telemetry_hud(
                        raw_rgb,
                        "BeyondMimic PPO Policy Tracking",
                        "Booster K1 (22-DoF)",
                        fmeta,
                        vrec.camera_pov,
                        vrec.include_overlay,
                    )

                rendered_frames.append(frame_out)

                if idx % 15 == 0 or idx == total_steps - 1:
                    prog = int(round(85.0 * (idx + 1) / total_steps))
                    vrec.progress = prog
                    socketio.emit("video_progress", vrec.to_dict())

            renderer.close()

            effective_fps = max(10, int(round(vrec.fps * vrec.playback_speed)))
            looped_frames = rendered_frames + rendered_frames
            imageio.mimsave(str(out_path), looped_frames, fps=effective_fps, macro_block_size=1)

            vrec.status = "completed"
            vrec.progress = 100
            vrec.filename = out_filename
            vrec.file_path = str(out_path.relative_to(HIL_ROOT)).replace("\\", "/")
            vrec.file_size_bytes = out_path.stat().st_size
            vrec.frame_count = len(looped_frames)
            vrec.duration_sec = round(len(looped_frames) / float(effective_fps), 2)
            runtime_store.add_activity(
                category="video",
                action=f"Video Encoded: {vrec.title}",
                details=f"Saved {out_filename} ({vrec.frame_count} frames @ {effective_fps} FPS)",
                status="success",
                experiment_id=vrec.experiment_id,
            )
            socketio.emit("video_progress", vrec.to_dict())

        except Exception as exc:
            vrec.status = "failed"
            vrec.error_message = str(exc)
            socketio.emit("video_progress", vrec.to_dict())


video_service = VideoGenerationService()
