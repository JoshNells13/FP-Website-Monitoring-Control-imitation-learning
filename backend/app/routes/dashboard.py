from __future__ import annotations

from flask import Blueprint, jsonify

from app.models.database import runtime_store
from app.services.experiment_service import sync_filesystem_to_db
from app.services.metrics_service import (
    get_combined_training_progression,
    get_primary_evaluation_metrics,
)
from app.services.training_service import training_manager
from app.utils.system_monitor import get_system_resources

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/api/dashboard")


@dashboard_bp.route("/overview", methods=["GET"])
def get_overview():
    sync_filesystem_to_db()

    all_exps = sorted(
        [e for e in runtime_store.experiments.values() if not e.is_archived],
        key=lambda x: x.final_step,
    )
    total_exps = len(all_exps)
    running_exps = sum(1 for e in all_exps if e.status == "running")
    completed_exps = sum(1 for e in all_exps if e.status == "completed")
    failed_exps = sum(1 for e in all_exps if e.status == "failed")

    t_state = training_manager.get_state()
    if t_state["status"] in ("running", "starting", "paused"):
        running_exps = max(1, running_exps)

    latest_exp = all_exps[-1] if all_exps else None

    all_ckpts = list(runtime_store.checkpoints.values())
    best_ckpt = (
        next((c for c in all_ckpts if c.is_primary_output), None)
        or next((c for c in all_ckpts if c.is_best), None)
        or (sorted(all_ckpts, key=lambda c: c.iteration, reverse=True)[0] if all_ckpts else None)
    )

    eval_metrics = get_primary_evaluation_metrics()
    succ_rate = eval_metrics.get("success_rate_percent")
    fail_rate = round(100.0 - float(succ_rate), 2) if succ_rate is not None else None

    total_duration_sec = round(sum(float(e.duration_sec or 0.0) for e in all_exps), 1)
    recent_activities = [
        a.to_dict()
        for a in reversed(runtime_store.activities[-12:])
    ]

    progression = get_combined_training_progression(max_points=180)
    peak_reward = max((float(r["mean_reward"]) for r in progression if "mean_reward" in r), default=None)

    per_joint_list = []
    for jname, jvals in eval_metrics.get("per_joint_metrics", {}).items():
        per_joint_list.append({
            "joint": jname,
            "mae_rad": jvals.get("mae_rad"),
            "mae_deg": jvals.get("mae_deg"),
            "rmse_rad": jvals.get("rmse_rad"),
            "rmse_deg": jvals.get("rmse_deg"),
        })

    return jsonify({
        "storage_mode": "ephemeral_in_memory_no_database",
        "summary": {
            "total_experiments": total_exps,
            "running_experiments": running_exps,
            "completed_experiments": completed_exps,
            "failed_experiments": failed_exps,
            "latest_mean_reward": latest_exp.mean_reward if latest_exp else None,
            "peak_mean_reward": round(peak_reward, 4) if peak_reward is not None else None,
            "latest_episode_length": latest_exp.mean_episode_length if latest_exp else None,
            "avg_tracking_error_mae_rad": eval_metrics.get("joint_tracking_mae_rad"),
            "avg_tracking_error_mae_deg": eval_metrics.get("joint_tracking_mae_deg"),
            "avg_tracking_error_rmse_rad": eval_metrics.get("joint_tracking_rmse_rad"),
            "avg_tracking_error_rmse_deg": eval_metrics.get("joint_tracking_rmse_deg"),
            "joint_velocity_mae_rad_s": eval_metrics.get("joint_velocity_mae_rad_s"),
            "joint_velocity_rmse_rad_s": eval_metrics.get("joint_velocity_rmse_rad_s"),
            "r_mpkpe_mm": eval_metrics.get("r_mpkpe_mm"),
            "mpkpe_mm": eval_metrics.get("mpkpe_mm"),
            "hand_ee_error_mm": eval_metrics.get("end_effector_hand_error_mm"),
            "foot_ee_error_mm": eval_metrics.get("end_effector_foot_error_mm"),
            "anchor_orientation_error_rad": eval_metrics.get("anchor_orientation_error_rad"),
            "anchor_orientation_error_deg": eval_metrics.get("anchor_orientation_error_deg"),
            "num_eval_episodes": eval_metrics.get("num_eval_episodes"),
            "num_completed_episodes": eval_metrics.get("num_completed_episodes"),
            " trajectory_frames_50fps": eval_metrics.get("trajectory_frames_50fps"),
            "trajectory_duration_sec": eval_metrics.get("trajectory_duration_sec"),
            "success_rate": succ_rate,
            "failure_rate": fail_rate,
            "total_training_duration_sec": total_duration_sec,
            "total_checkpoints": len(all_ckpts),
            "total_videos": len(runtime_store.videos),
        },
        "evaluation_metrics": eval_metrics,
        "latest_session": latest_exp.to_dict(include_details=False) if latest_exp else None,
        "best_checkpoint": best_ckpt.to_dict() if best_ckpt else None,
        "experiments": [e.to_dict(include_details=False) for e in all_exps],
        "training_progression": progression,
        "per_joint_performance": per_joint_list,
        "recent_activity": recent_activities,
        "training_state": t_state,
        "system_resources": get_system_resources(),
    })
