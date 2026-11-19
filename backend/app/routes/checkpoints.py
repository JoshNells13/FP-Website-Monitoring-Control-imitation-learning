from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_file

from app.models.database import runtime_store
from app.services.experiment_service import sync_filesystem_to_db
from app.utils.system_monitor import get_system_resources, safe_resolve_in_hil
from config import HIL_ROOT, K1_XML_PATH, PYTHON_EXE

checkpoints_bp = Blueprint("checkpoints", __name__, url_prefix="/api")


@checkpoints_bp.route("/checkpoints", methods=["GET"])
def list_checkpoints():
    sync_filesystem_to_db()
    exp_id = request.args.get("experiment_id")
    ckpts = list(runtime_store.checkpoints.values())
    if exp_id:
        ckpts = [c for c in ckpts if c.experiment_id == int(exp_id)]
    ckpts.sort(key=lambda c: (c.is_primary_output, c.created_at, c.iteration), reverse=True)
    return jsonify({
        "checkpoints": [c.to_dict() for c in ckpts],
        "total": len(ckpts),
    })


@checkpoints_bp.route("/checkpoints/<int:ckpt_id>", methods=["GET"])
def get_checkpoint(ckpt_id: int):
    c = runtime_store.checkpoints.get(ckpt_id)
    if not c:
        return jsonify({"error": "Checkpoint not found"}), 404
    return jsonify(c.to_dict())


@checkpoints_bp.route("/checkpoints/<int:ckpt_id>", methods=["PATCH"])
def update_checkpoint(ckpt_id: int):
    c = runtime_store.checkpoints.get(ckpt_id)
    if not c:
        return jsonify({"error": "Checkpoint not found"}), 404

    payload = request.get_json(silent=True) or {}
    if "name" in payload:
        c.name = str(payload["name"]).strip()
    if "notes" in payload:
        c.notes = str(payload["notes"]).strip()
    if "is_best" in payload:
        c.is_best = bool(payload["is_best"])

    return jsonify(c.to_dict())


@checkpoints_bp.route("/checkpoints/download/<int:ckpt_id>", methods=["GET"])
def download_checkpoint(ckpt_id: int):
    c = runtime_store.checkpoints.get(ckpt_id)
    if not c:
        abort(404)
    fmt = request.args.get("format", "pt").lower()
    rel = c.onnx_path if (fmt == "onnx" and c.onnx_path) else c.file_path
    path = safe_resolve_in_hil(rel)
    if not path.exists():
        abort(404)
    return send_file(
        str(path),
        as_attachment=True,
        download_name=path.name,
    )


@checkpoints_bp.route("/settings", methods=["GET"])
def get_platform_settings():
    return jsonify({
        "platform": "Humanoid Imitation Learning Control & Visualization Platform",
        "version": "1.0.0",
        "database": {
            "engine": "Disabled — Stateless Ephemeral In-Memory Monitoring",
            "uri_masked": "memory://ephemeral-live-session (No Database Used)",
            "connected": False,
            "mode": "ephemeral",
        },
        "environment": {
            "hil_root": str(HIL_ROOT),
            "python_exe": str(PYTHON_EXE),
            "robot_xml": str(K1_XML_PATH),
            "robot_model": "Booster K1 (22-DoF)",
            "simulator": "MuJoCo 3.16.0 + MJLab 1.6.0 (MuJoCo-Warp CUDA)",
            "rl_framework": "BeyondMimic PPO (rsl-rl-lib 5.5.2)",
            "retargeting": "GMR (General Motion Retargeting — Mink + DAQP)",
        },
        "resources": get_system_resources(),
    })
