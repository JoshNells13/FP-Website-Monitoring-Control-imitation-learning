from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Checkpoint:
    id: int
    experiment_id: int | None
    name: str
    filename: str
    file_path: str
    onnx_path: str | None = None
    iteration: int = 0
    file_size_bytes: int = 0
    mean_reward: float | None = None
    tracking_error_mae: float | None = None
    tracking_error_rmse: float | None = None
    r_mpkpe_mm: float | None = None
    mpkpe_mm: float | None = None
    success_rate: float | None = None
    is_best: bool = False
    is_primary_output: bool = False
    notes: str = ""
    metrics_json: str = "{}"
    created_at: datetime = field(default_factory=utcnow)
    experiment_run_id: str | None = None
    experiment_name: str = "Live Session / Output Policy"

    def to_dict(self) -> dict[str, Any]:
        try:
            met = json.loads(self.metrics_json) if self.metrics_json else {}
        except Exception:
            met = {}
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "experiment_run_id": self.experiment_run_id,
            "experiment_name": self.experiment_name,
            "name": self.name,
            "filename": self.filename,
            "file_path": self.file_path,
            "onnx_path": self.onnx_path,
            "iteration": self.iteration,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 2) if self.file_size_bytes else 0.0,
            "mean_reward": self.mean_reward,
            "tracking_error_mae": self.tracking_error_mae,
            "tracking_error_rmse": self.tracking_error_rmse,
            "r_mpkpe_mm": self.r_mpkpe_mm,
            "mpkpe_mm": self.mpkpe_mm,
            "success_rate": self.success_rate,
            "is_best": self.is_best,
            "is_primary_output": self.is_primary_output,
            "notes": self.notes or "",
            "metrics": met,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class Experiment:
    id: int
    run_id: str
    name: str
    run_name: str = "beyondmimic_k1_kick"
    description: str = ""
    robot_model: str = "booster_k1"
    task_id: str = "Mjlab-Tracking-Flat-Booster-K1"
    motion_file: str = "data/motions/k1_kick.npz"
    status: str = "running"
    is_archived: bool = False
    num_envs: int = 2048
    max_iterations: int = 500
    save_interval: int = 100
    seed: int = 42
    resume_from: str | None = None
    start_step: int = 0
    current_step: int = 0
    final_step: int = 500
    mean_reward: float | None = None
    best_reward: float | None = None
    mean_episode_length: float | None = None
    value_loss: float | None = None
    surrogate_loss: float | None = None
    learning_rate: float | None = None
    tracking_error_mae: float | None = None
    tracking_error_rmse: float | None = None
    r_mpkpe_mm: float | None = None
    mpkpe_mm: float | None = None
    anchor_pos_error_m: float | None = None
    anchor_ori_error_rad: float | None = None
    joint_vel_error_rmse: float | None = None
    hand_ee_error_mm: float | None = None
    foot_ee_error_mm: float | None = None
    motion_completion_pct: float | None = None
    success_rate: float | None = None
    failure_rate: float | None = None
    duration_sec: float = 0.0
    total_fps: float | None = None
    log_dir: str | None = None
    best_checkpoint_path: str | None = None
    config_json: str = "{}"
    metrics_json: str = "{}"
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def to_dict(self, include_details: bool = False) -> dict[str, Any]:
        try:
            cfg = json.loads(self.config_json) if self.config_json else {}
        except Exception:
            cfg = {}
        try:
            met = json.loads(self.metrics_json) if self.metrics_json else {}
        except Exception:
            met = {}

        data = {
            "id": self.id,
            "run_id": self.run_id,
            "name": self.name,
            "run_name": self.run_name,
            "description": self.description or "",
            "robot_model": self.robot_model,
            "task_id": self.task_id,
            "motion_file": self.motion_file,
            "status": self.status,
            "is_archived": self.is_archived,
            "num_envs": self.num_envs,
            "max_iterations": self.max_iterations,
            "save_interval": self.save_interval,
            "seed": self.seed,
            "resume_from": self.resume_from,
            "start_step": self.start_step,
            "current_step": self.current_step,
            "final_step": self.final_step,
            "mean_reward": self.mean_reward,
            "best_reward": self.best_reward,
            "mean_episode_length": self.mean_episode_length,
            "value_loss": self.value_loss,
            "surrogate_loss": self.surrogate_loss,
            "learning_rate": self.learning_rate,
            "tracking_error_mae": self.tracking_error_mae,
            "tracking_error_rmse": self.tracking_error_rmse,
            "r_mpkpe_mm": self.r_mpkpe_mm,
            "mpkpe_mm": self.mpkpe_mm,
            "anchor_pos_error_m": self.anchor_pos_error_m,
            "anchor_ori_error_rad": self.anchor_ori_error_rad,
            "joint_vel_error_rmse": self.joint_vel_error_rmse,
            "hand_ee_error_mm": self.hand_ee_error_mm,
            "foot_ee_error_mm": self.foot_ee_error_mm,
            "motion_completion_pct": self.motion_completion_pct,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "duration_sec": self.duration_sec,
            "total_fps": self.total_fps,
            "log_dir": self.log_dir,
            "best_checkpoint_path": self.best_checkpoint_path,
            "checkpoint_count": len(self.checkpoints),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_details:
            data["config"] = cfg
            data["metrics"] = met
            data["checkpoints"] = [
                c.to_dict() for c in sorted(self.checkpoints, key=lambda x: x.iteration)
            ]
        return data


@dataclass
class VideoRecord:
    id: int
    job_id: str
    title: str
    video_type: str = "policy"
    camera_pov: str = "third_person"
    resolution: str = "960x720"
    fps: int = 50
    playback_speed: float = 1.0
    include_overlay: bool = True
    motion_file: str = "data/motions/k1_kick.npz"
    checkpoint_path: str | None = None
    experiment_id: int | None = None
    checkpoint_id: int | None = None
    status: str = "completed"
    progress: int = 100
    file_path: str | None = None
    filename: str | None = None
    file_size_bytes: int = 0
    duration_sec: float = 3.0
    frame_count: int = 149
    error_message: str | None = None
    ephemeral_bytes: bytes | None = None
    created_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        has_media = bool(self.file_path or self.ephemeral_bytes)
        return {
            "id": self.id,
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "checkpoint_id": self.checkpoint_id,
            "title": self.title,
            "video_type": self.video_type,
            "camera_pov": self.camera_pov,
            "resolution": self.resolution,
            "fps": self.fps,
            "playback_speed": self.playback_speed,
            "include_overlay": self.include_overlay,
            "motion_file": self.motion_file,
            "checkpoint_path": self.checkpoint_path,
            "status": self.status,
            "progress": self.progress,
            "file_path": self.file_path or "ephemeral://in-memory-stream",
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 2) if self.file_size_bytes else 0.0,
            "duration_sec": self.duration_sec,
            "frame_count": self.frame_count,
            "error_message": self.error_message,
            "stream_url": f"/api/videos/stream/{self.id}" if has_media else None,
            "download_url": f"/api/videos/download/{self.id}" if has_media else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class ActivityLog:
    id: int
    category: str
    action: str
    details: str = ""
    status: str = "info"
    experiment_id: int | None = None
    created_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "category": self.category,
            "action": self.action,
            "details": self.details,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EphemeralRuntimeStore:
    """Stateless in-memory store for active monitoring and simulation without any database persistence."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.experiments: dict[int, Experiment] = {}
        self.checkpoints: dict[int, Checkpoint] = {}
        self.videos: dict[int, VideoRecord] = {}
        self.activities: list[ActivityLog] = []
        self._exp_seq = 1
        self._ckpt_seq = 1
        self._vid_seq = 1
        self._act_seq = 1

    def next_exp_id(self) -> int:
        with self.lock:
            val = self._exp_seq
            self._exp_seq += 1
            return val

    def next_ckpt_id(self) -> int:
        with self.lock:
            val = self._ckpt_seq
            self._ckpt_seq += 1
            return val

    def next_vid_id(self) -> int:
        with self.lock:
            val = self._vid_seq
            self._vid_seq += 1
            return val

    def add_activity(
        self,
        category: str,
        action: str,
        details: str = "",
        status: str = "info",
        experiment_id: int | None = None,
    ) -> ActivityLog:
        with self.lock:
            entry = ActivityLog(
                id=self._act_seq,
                category=category,
                action=action,
                details=details,
                status=status,
                experiment_id=experiment_id,
            )
            self._act_seq += 1
            self.activities.append(entry)
            if len(self.activities) > 100:
                self.activities = self.activities[-100:]
            return entry

    def discard_ephemeral_session(self) -> None:
        """Discard all temporary in-memory simulation and video buffers immediately."""
        with self.lock:
            self.videos = {
                k: v for k, v in self.videos.items() if v.job_id.startswith("existing_")
            }
            self.add_activity(
                category="simulation",
                action="Ephemeral Simulation State Discarded",
                details="Cleared in-memory simulation rollout and temporary buffers (no database used).",
                status="info",
            )


runtime_store = EphemeralRuntimeStore()
