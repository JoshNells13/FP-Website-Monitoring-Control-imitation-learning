from __future__ import annotations

import json
from flask import Blueprint, Response, abort, jsonify, request, send_from_directory

from app.models.database import runtime_store
from app.services.experiment_service import sync_filesystem_to_db
from app.services.metrics_service import (
    build_iteration_series,
    export_metrics_dataframe,
    extract_tensorboard_run,
    generate_matplotlib_report_png,
    get_combined_training_progression,
    get_primary_evaluation_metrics,
)
from app.services.simulation_service import get_motion_trajectory_bundle
from config import EVALUATION_DIR, HIL_ROOT

analytics_bp = Blueprint("analytics", __name__, url_prefix="/api/analytics")


@analytics_bp.route("/summary", methods=["GET"])
def get_analytics_summary():
    sync_filesystem_to_db()
    exp_id = request.args.get("experiment_id")
    eval_metrics = get_primary_evaluation_metrics()

    all_exps = sorted(runtime_store.experiments.values(), key=lambda x: (x.created_at, x.run_id))
    all_ckpts = sorted(runtime_store.checkpoints.values(), key=lambda x: (x.is_primary_output, x.created_at, x.iteration), reverse=True)

    if exp_id:
        exp = runtime_store.experiments.get(int(exp_id))
        if not exp:
            return jsonify({"error": "Experiment not found"}), 404
        run_dir = (HIL_ROOT / exp.log_dir).resolve() if exp.log_dir else None
        tb = extract_tensorboard_run(run_dir) if run_dir and run_dir.exists() else {"scalars": {}, "summary": {}}
        series = build_iteration_series(tb["scalars"], max_points=300)
        selected_exp_dict = exp.to_dict(include_details=True)
    else:
        exp = all_exps[-1] if all_exps else None
        if exp and exp.log_dir and (HIL_ROOT / exp.log_dir).exists():
            tb = extract_tensorboard_run((HIL_ROOT / exp.log_dir).resolve())
            series = build_iteration_series(tb["scalars"], max_points=300)
        else:
            series = get_combined_training_progression(max_points=300)
        selected_exp_dict = exp.to_dict(include_details=True) if exp else None

    target_motion = exp.motion_file if (exp and exp.motion_file) else "data/motions/k1_kick.npz"
    target_ckpt = exp.best_checkpoint_path if (exp and exp.best_checkpoint_path) else "outputs/checkpoints/k1_kick_policy.pt"
    traj_bundle = get_motion_trajectory_bundle(motion_file=target_motion, checkpoint_file=target_ckpt)

    return jsonify({
        "selected_experiment": selected_exp_dict,
        "evaluation_metrics": eval_metrics,
        "iteration_series": series,
        "per_joint_summary": traj_bundle["per_joint_summary"],
        "trajectory_overall": traj_bundle["overall_metrics"],
        "experiments": [e.to_dict(include_details=False) for e in all_exps],
        "checkpoints": [c.to_dict() for c in all_ckpts],
    })


@analytics_bp.route("/compare", methods=["POST"])
def compare_items():
    payload = request.get_json(silent=True) or {}
    exp_ids = payload.get("experiment_ids", [])
    ckpt_ids = payload.get("checkpoint_ids", [])

    compared_experiments = []
    for eid in exp_ids:
        exp = runtime_store.experiments.get(int(eid))
        if exp:
            d = exp.to_dict(include_details=True)
            run_dir = (HIL_ROOT / exp.log_dir).resolve() if exp.log_dir else None
            if run_dir and run_dir.exists():
                tb = extract_tensorboard_run(run_dir)
                d["series"] = build_iteration_series(tb["scalars"], max_points=120)
            else:
                d["series"] = []
            compared_experiments.append(d)

    compared_checkpoints = []
    for cid in ckpt_ids:
        ckpt = runtime_store.checkpoints.get(int(cid))
        if ckpt:
            compared_checkpoints.append(ckpt.to_dict())

    return jsonify({
        "experiments": compared_experiments,
        "checkpoints": compared_checkpoints,
    })


@analytics_bp.route("/export", methods=["GET"])
def export_results():
    fmt = request.args.get("format", "csv").lower()
    exp_id = request.args.get("experiment_id")

    if exp_id:
        exp = runtime_store.experiments.get(int(exp_id))
        run_dir = (HIL_ROOT / exp.log_dir).resolve() if (exp and exp.log_dir) else None
        tb = extract_tensorboard_run(run_dir) if (run_dir and run_dir.exists()) else {"scalars": {}}
        series = build_iteration_series(tb["scalars"], max_points=1000)
        filename_prefix = f"booster_k1_{exp.run_id}" if exp else "booster_k1_experiment"
    else:
        series = get_combined_training_progression(max_points=2500)
        filename_prefix = "booster_k1_full_curriculum"

    if fmt == "json":
        payload = {
            "project": "Booster K1 (22-DoF) Whole-Body Motion Imitation Learning",
            "evaluation_metrics": get_primary_evaluation_metrics(),
            "training_series": series,
        }
        return Response(
            json.dumps(payload, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename_prefix}_results.json"},
        )

    df = export_metrics_dataframe(series)
    csv_str = df.to_csv(index=False)
    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename_prefix}_metrics.csv"},
    )


@analytics_bp.route("/report.png", methods=["GET"])
def download_matplotlib_report():
    exp_id = request.args.get("experiment_id")
    run_dir = None
    if exp_id:
        exp = runtime_store.experiments.get(int(exp_id))
        if exp and exp.log_dir:
            run_dir = (HIL_ROOT / exp.log_dir).resolve()

    png_bytes = generate_matplotlib_report_png(run_dir=run_dir)
    return Response(
        png_bytes,
        mimetype="image/png",
        headers={"Content-Disposition": "inline; filename=booster_k1_ppo_report.png"},
    )


@analytics_bp.route("/evaluation-plots/<path:filename>", methods=["GET"])
def serve_evaluation_plot(filename: str):
    target = (EVALUATION_DIR / filename).resolve()
    if not target.exists() or EVALUATION_DIR.resolve() not in target.parents:
        abort(404)
    return send_from_directory(str(EVALUATION_DIR), filename)
