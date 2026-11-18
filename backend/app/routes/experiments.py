from __future__ import annotations

import json
from datetime import datetime
from flask import Blueprint, jsonify, request

from app.models.database import Experiment, runtime_store
from app.services.experiment_service import sync_filesystem_to_db
from app.services.metrics_service import build_iteration_series, extract_tensorboard_run
from config import HIL_ROOT

experiments_bp = Blueprint("experiments", __name__, url_prefix="/api/experiments")


@experiments_bp.route("", methods=["GET"])
def list_experiments():
    sync_filesystem_to_db()
    include_archived = request.args.get("include_archived", "false").lower() == "true"
    exps = list(runtime_store.experiments.values())
    if not include_archived:
        exps = [e for e in exps if not e.is_archived]
    exps.sort(key=lambda e: (e.final_step, e.created_at), reverse=True)
    return jsonify({
        "storage_mode": "ephemeral_in_memory_no_database",
        "experiments": [e.to_dict(include_details=True) for e in exps],
        "total": len(exps),
    })


@experiments_bp.route("", methods=["POST"])
def create_experiment():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    if not name:
        return jsonify({"error": "Experiment name is required."}), 400

    run_name = str(payload.get("run_name", "beyondmimic_k1_kick")).strip()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}_{run_name}"

    exp = Experiment(
        id=runtime_store.next_exp_id(),
        run_id=run_id,
        name=name,
        run_name=run_name,
        description=str(payload.get("description", "")).strip(),
        robot_model=str(payload.get("robot_model", "booster_k1")),
        task_id=str(payload.get("task_id", "Mjlab-Tracking-Flat-Booster-K1")),
        motion_file=str(payload.get("motion_file", "data/motions/k1_kick.npz")),
        status=str(payload.get("status", "queued")),
        num_envs=int(payload.get("num_envs", 2048)),
        max_iterations=int(payload.get("max_iterations", 500)),
        save_interval=int(payload.get("save_interval", 100)),
        seed=int(payload.get("seed", 42)),
        resume_from=payload.get("resume_from"),
        config_json=json.dumps(payload.get("config", {})),
    )
    runtime_store.experiments[exp.id] = exp
    runtime_store.add_activity(
        category="experiment",
        action=f"Created In-Memory Experiment Session: {name}",
        details=f"Run ID: {run_id}, Envs: {exp.num_envs}, Max Iter: {exp.max_iterations}",
        status="info",
        experiment_id=exp.id,
    )
    return jsonify(exp.to_dict(include_details=True)), 201


@experiments_bp.route("/<int:exp_id>", methods=["GET"])
def get_experiment_detail(exp_id: int):
    exp = runtime_store.experiments.get(exp_id)
    if not exp:
        return jsonify({"error": "Experiment not found"}), 404

    data = exp.to_dict(include_details=True)
    run_dir = (HIL_ROOT / exp.log_dir).resolve() if exp.log_dir else None
    if run_dir and run_dir.exists():
        tb = extract_tensorboard_run(run_dir)
        data["iteration_series"] = build_iteration_series(tb["scalars"], max_points=250)
    else:
        data["iteration_series"] = []

    vids = [v for v in runtime_store.videos.values() if v.experiment_id == exp.id]
    if not vids:
        vids = sorted(runtime_store.videos.values(), key=lambda v: v.created_at, reverse=True)[:4]
    data["videos"] = [v.to_dict() for v in vids]
    return jsonify(data)


@experiments_bp.route("/<int:exp_id>", methods=["PATCH"])
def update_experiment(exp_id: int):
    exp = runtime_store.experiments.get(exp_id)
    if not exp:
        return jsonify({"error": "Experiment not found"}), 404

    payload = request.get_json(silent=True) or {}
    if "name" in payload:
        exp.name = str(payload["name"]).strip()
    if "description" in payload:
        exp.description = str(payload["description"]).strip()
    if "is_archived" in payload:
        exp.is_archived = bool(payload["is_archived"])
    if "status" in payload:
        exp.status = str(payload["status"]).strip()

    runtime_store.add_activity(
        category="experiment",
        action=f"Updated In-Memory Session: {exp.name}",
        details=f"Archived={exp.is_archived}, Status={exp.status}",
        status="info",
        experiment_id=exp.id,
    )
    return jsonify(exp.to_dict(include_details=True))


@experiments_bp.route("/<int:exp_id>/logs", methods=["GET"])
def get_experiment_logs(exp_id: int):
    exp = runtime_store.experiments.get(exp_id)
    if not exp:
        return jsonify({"error": "Experiment not found"}), 404

    lines: list[str] = [
        "================================================================================",
        f"MJLab BeyondMimic PPO Live Experiment Log: {exp.name}",
        f"Run ID:         {exp.run_id}",
        f"Robot Model:    {exp.robot_model} (22-DoF, Trunk anchor, 14 tracked keybodies)",
        f"Task ID:        {exp.task_id}",
        f"Motion File:    {exp.motion_file}",
        f"Log Directory:  {exp.log_dir or 'N/A'}",
        f"Parallel Envs:  {exp.num_envs} | Save Interval: {exp.save_interval} | Seed: {exp.seed}",
        "================================================================================",
    ]

    run_dir = (HIL_ROOT / exp.log_dir).resolve() if exp.log_dir else None
    if run_dir and run_dir.exists():
        tb = extract_tensorboard_run(run_dir)
        series = build_iteration_series(tb["scalars"], max_points=60)
        for row in series:
            it = row.get("iteration", 0)
            rew = row.get("mean_reward", 0.0)
            eplen = row.get("mean_episode_length", 0.0)
            vloss = row.get("value_loss", 0.0)
            b_err = row.get("error_body_pos", 0.0)
            j_err = row.get("error_joint_pos", 0.0)
            fps = row.get("total_fps", 0.0)
            lines.append(
                f"[Iter {it:04d}] mean_reward={rew:+8.4f} | ep_len={eplen:6.2f} | "
                f"err_body_pos={b_err:.4f}m | err_joint_pos={j_err:.4f}rad | "
                f"value_loss={vloss:.5f} | fps={fps:,.0f}"
            )
    else:
        lines.append("No TensorBoard event file found for this session yet.")

    return jsonify({
        "experiment_id": exp.id,
        "run_id": exp.run_id,
        "log_lines": lines,
    })
