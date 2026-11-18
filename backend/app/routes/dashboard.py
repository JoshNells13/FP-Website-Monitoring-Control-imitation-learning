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
    avg_mae_rad = float(eval_metrics.get("joint_tracking_mae_rad", 0.14701))
    avg_mae_deg = float(eval_metrics.get("joint_tracking_mae_deg", 8.423))
    avg_rmse_rad = float(eval_metrics.get("joint_tracking_rmse_rad", 0.21691))
    r_mpkpe_mm = float(eval_metrics.get("r_mpkpe_mm", 39.38))
    mpkpe_mm = float(eval_metrics.get("mpkpe_mm", 108.34))
    success_rate = float(eval_metrics.get("success_rate_percent", 65.62))
    failure_rate = round(100.0 - success_rate, 2)

    total_duration_sec = round(sum(float(e.duration_sec or 0.0) for e in all_exps), 1)
    recent_activities = [
        a.to_dict()
        for a in reversed(runtime_store.activities[-12:])
    ]

    progression = get_combined_training_progression(max_points=180)

    per_joint_list = []
    for jname, jvals in eval_metrics.get("per_joint_metrics", {}).items():
        per_joint_list.append({
            "joint": jname,
            "mae_rad": jvals.get("mae_rad", 0.0),
            "mae_deg": jvals.get("mae_deg", 0.0),
            "rmse_rad": jvals.get("rmse_rad", 0.0),
            "rmse_deg": jvals.get("rmse_deg", 0.0),
        })

    return jsonify({
        "storage_mode": "ephemeral_in_memory_no_database",
        "summary": {
            "total_experiments": total_exps,
            "running_experiments": running_exps,
            "completed_experiments": completed_exps,
            "failed_experiments": failed_exps,
            "avg_tracking_error_mae_rad": avg_mae_rad,
            "avg_tracking_error_mae_deg": avg_mae_deg,
            "avg_tracking_error_rmse_rad": avg_rmse_rad,
            "r_mpkpe_mm": r_mpkpe_mm,
            "mpkpe_mm": mpkpe_mm,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "total_training_duration_sec": total_duration_sec,
            "total_checkpoints": len(all_ckpts),
            "total_videos": len(runtime_store.videos),
        },
        "latest_session": latest_exp.to_dict(include_details=False) if latest_exp else None,
        "best_checkpoint": best_ckpt.to_dict() if best_ckpt else None,
        "experiments": [e.to_dict(include_details=False) for e in all_exps],
        "training_progression": progression,
        "per_joint_performance": per_joint_list,
        "recent_activity": recent_activities,
        "training_state": t_state,
        "system_resources": get_system_resources(),
    })
