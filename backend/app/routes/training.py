from __future__ import annotations

import subprocess
import threading
from flask import Blueprint, jsonify, request

from app.models.database import runtime_store
from app.services.training_service import training_manager
from app.utils.system_monitor import safe_resolve_in_hil
from config import HIL_ROOT, PYTHON_EXE, SCRIPTS_DIR

training_bp = Blueprint("training", __name__, url_prefix="/api/training")


@training_bp.route("/options", methods=["GET"])
def get_training_options():
    return jsonify(training_manager.get_available_options())


@training_bp.route("/status", methods=["GET"])
def get_training_status():
    return jsonify(training_manager.get_state())


@training_bp.route("/logs", methods=["GET"])
def get_training_logs():
    limit = int(request.args.get("limit", 400))
    return jsonify({
        "logs": training_manager.get_logs(limit=limit),
        "state": training_manager.get_state(),
    })


@training_bp.route("/start", methods=["POST"])
def start_training():
    payload = request.get_json(silent=True) or {}
    try:
        state = training_manager.start_training(payload)
        return jsonify({"ok": True, "state": state})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@training_bp.route("/pause", methods=["POST"])
def pause_training():
    try:
        state = training_manager.pause_training()
        return jsonify({"ok": True, "state": state})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@training_bp.route("/resume", methods=["POST"])
def resume_training():
    try:
        state = training_manager.resume_training()
        return jsonify({"ok": True, "state": state})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@training_bp.route("/stop", methods=["POST"])
def stop_training():
    try:
        state = training_manager.stop_training()
        return jsonify({"ok": True, "state": state})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@training_bp.route("/evaluate", methods=["POST"])
def run_evaluation():
    """Run scripts/evaluate_and_play.py safely in a managed background thread."""
    payload = request.get_json(silent=True) or {}
    ckpt_rel = str(payload.get("checkpoint_file", "outputs/checkpoints/k1_kick_policy.pt")).strip()
    motion_rel = str(payload.get("motion_file", "data/motions/k1_kick.npz")).strip()
    num_eval_envs = max(16, min(512, int(payload.get("num_eval_envs", 256))))

    try:
        ckpt_path = safe_resolve_in_hil(ckpt_rel)
        motion_path = safe_resolve_in_hil(motion_rel)
        if not ckpt_path.exists():
            return jsonify({"ok": False, "error": f"Checkpoint not found: {ckpt_rel}"}), 404
        if not motion_path.exists():
            return jsonify({"ok": False, "error": f"Motion file not found: {motion_rel}"}), 404

        script_path = SCRIPTS_DIR / "evaluate_and_play.py"
        cmd = [
            str(PYTHON_EXE),
            "-u",
            str(script_path),
            "--robot",
            "booster_k1",
            "--checkpoint-file",
            str(ckpt_path.relative_to(HIL_ROOT)),
            "--motion-file",
            str(motion_path.relative_to(HIL_ROOT)),
            "--num-eval-envs",
            str(num_eval_envs),
            "--output-video",
            "outputs/videos/k1_trained_policy_tracking.mp4",
            "--output-metrics",
            "outputs/evaluation/k1_metrics.json",
            "--output-plot",
            "outputs/evaluation/k1_joint_tracking_comparison.png",
        ]

        training_manager._append_log(f"[EVALUATION] Starting policy evaluation: {' '.join(cmd)}", "info")

        def _bg_eval():
            proc = subprocess.Popen(
                cmd,
                cwd=str(HIL_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    training_manager._append_log(line, "info")
            proc.wait()
            training_manager._append_log(f"[EVALUATION] Evaluation completed with code {proc.returncode}.", "info")

        threading.Thread(target=_bg_eval, daemon=True).start()
        runtime_store.add_activity(
            category="evaluation",
            action="Started Live Policy Evaluation",
            details=f"Evaluating {ckpt_rel} on {motion_rel} ({num_eval_envs} parallel envs)",
            status="info",
        )
        return jsonify({"ok": True, "message": "Evaluation job started in background. Follow progress in Terminal Logs."})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
