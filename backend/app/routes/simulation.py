from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_from_directory

from app.models.database import runtime_store
from app.services.simulation_service import (
    get_motion_trajectory_bundle,
    get_robot_kinematic_schema,
    render_mujoco_frame_base64,
    simulation_controller,
)
from config import K1_MESHES_DIR

simulation_bp = Blueprint("simulation", __name__, url_prefix="/api/simulation")


@simulation_bp.route("/schema", methods=["GET"])
def get_schema():
    return jsonify(get_robot_kinematic_schema())


@simulation_bp.route("/meshes/<path:filename>", methods=["GET"])
def serve_robot_mesh(filename: str):
    target = (K1_MESHES_DIR / filename).resolve()
    if not target.exists() or K1_MESHES_DIR.resolve() not in target.parents:
        abort(404)
    response = send_from_directory(str(K1_MESHES_DIR), filename)
    response.headers["Cache-Control"] = "public, max-age=86400"
    return response


@simulation_bp.route("/trajectory", methods=["GET"])
def get_trajectory():
    motion_file = request.args.get("motion_file", "data/motions/k1_kick.npz")
    checkpoint_file = request.args.get("checkpoint_file", "outputs/checkpoints/k1_kick_policy.pt")
    try:
        bundle = get_motion_trajectory_bundle(motion_file=motion_file, checkpoint_file=checkpoint_file)
        return jsonify(bundle)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@simulation_bp.route("/render-frame", methods=["GET"])
def render_frame():
    frame_idx = int(request.args.get("frame", 0))
    camera_pov = request.args.get("camera_pov", "third_person")
    mode = request.args.get("mode", "policy")
    motion_file = request.args.get("motion_file", "data/motions/k1_kick.npz")
    checkpoint_file = request.args.get("checkpoint_file", "outputs/checkpoints/k1_kick_policy.pt")
    width = max(240, min(1280, int(request.args.get("width", 640))))
    height = max(180, min(960, int(request.args.get("height", 480))))
    try:
        res = render_mujoco_frame_base64(
            frame_idx=frame_idx,
            camera_pov=camera_pov,
            mode=mode,
            motion_file=motion_file,
            checkpoint_file=checkpoint_file,
            width=width,
            height=height,
        )
        return jsonify(res)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@simulation_bp.route("/status", methods=["GET"])
def get_sim_status():
    return jsonify(simulation_controller.get_status())


@simulation_bp.route("/control", methods=["POST"])
def control_simulation():
    payload = request.get_json(silent=True) or {}
    state = simulation_controller.control(payload)
    return jsonify(state)


@simulation_bp.route("/discard", methods=["POST"])
def discard_ephemeral_simulation():
    """Immediately discard the current simulation rollout buffer from memory (no database persistence)."""
    state = simulation_controller.control({"action": "discard"})
    runtime_store.discard_ephemeral_session()
    return jsonify({
        "ok": True,
        "message": "Ephemeral simulation buffers discarded from memory.",
        "state": state,
    })
