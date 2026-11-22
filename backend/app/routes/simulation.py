from __future__ import annotations

import pathlib
from datetime import datetime, timezone

from flask import Blueprint, abort, jsonify, request, send_from_directory

from app.models.database import runtime_store
from app.services.experiment_service import sync_filesystem_to_db
from app.services.simulation_service import (
    get_motion_trajectory_bundle,
    get_robot_kinematic_schema,
    render_mujoco_frame_base64,
    simulation_controller,
)
from config import CHECKPOINTS_DIR, EVALUATION_DIR, HIL_ROOT, K1_MESHES_DIR, MOTIONS_DIR

simulation_bp = Blueprint("simulation", __name__, url_prefix="/api/simulation")


def _list_detected_motions() -> list[dict]:
    if not MOTIONS_DIR.exists():
        return []
    motions = []
    ckpt_files = sorted(CHECKPOINTS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True) if CHECKPOINTS_DIR.exists() else []
    for f in sorted(MOTIONS_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
        rel = str(f.relative_to(HIL_ROOT)).replace("\\", "/")
        stem = f.stem.lower()
        tag = stem.replace("k1_", "", 1) if stem.startswith("k1_") else stem
        has_rollout = (EVALUATION_DIR / f"{f.stem}_rollout_trajectory.npz").exists()

        matched_policy = None
        exact_pt = CHECKPOINTS_DIR / f"{f.stem}_policy.pt"
        if exact_pt.exists():
            matched_policy = str(exact_pt.relative_to(HIL_ROOT)).replace("\\", "/")
        else:
            for pt in ckpt_files:
                if tag and tag in pt.stem.lower():
                    matched_policy = str(pt.relative_to(HIL_ROOT)).replace("\\", "/")
                    break

        st = f.stat()
        motions.append({
            "name": f.name,
            "path": rel,
            "size_kb": round(st.st_size / 1024.0, 1),
            "has_rollout": has_rollout,
            "matched_policy": matched_policy,
            "updated_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        })
    return motions


def _resolve_valid_motion_and_checkpoint(motion_file: str | None, checkpoint_file: str | None) -> tuple[str, str]:
    motions = _list_detected_motions()
    valid_motion = motion_file or ""
    if not valid_motion or not (HIL_ROOT / valid_motion).exists():
        valid_motion = motions[0]["path"] if motions else "data/motions/k1_kick.npz"

    valid_ckpt = checkpoint_file or ""
    if not valid_ckpt or not (HIL_ROOT / valid_ckpt).exists():
        # Try matched policy for this motion first, then newest checkpoint in outputs/checkpoints/
        matched = next((m["matched_policy"] for m in motions if m["path"] == valid_motion and m["matched_policy"]), None)
        if matched and (HIL_ROOT / matched).exists():
            valid_ckpt = matched
        elif CHECKPOINTS_DIR.exists():
            pts = sorted(CHECKPOINTS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            valid_ckpt = str(pts[0].relative_to(HIL_ROOT)).replace("\\", "/") if pts else ""
        else:
            valid_ckpt = ""
    return valid_motion, valid_ckpt


@simulation_bp.route("/assets", methods=["GET"])
def get_simulation_assets():
    """Return all live-detected .npz motions and .pt policies in human-imitation-learning."""
    sync_filesystem_to_db()
    motions = _list_detected_motions()
    ckpts = sorted(
        runtime_store.checkpoints.values(),
        key=lambda c: (c.is_primary_output, c.created_at, c.iteration),
        reverse=True,
    )
    policies = []
    for c in ckpts:
        d = c.to_dict()
        fp_lower = (c.file_path or "").lower()
        matched_motion = None
        for m in motions:
            tag = pathlib.Path(m["name"]).stem.lower().replace("k1_", "", 1)
            if tag and tag in fp_lower:
                matched_motion = m["path"]
                break
        d["matched_motion"] = matched_motion
        policies.append(d)

    return jsonify({
        "motions": motions,
        "policies": policies,
        "default_motion": motions[0]["path"] if motions else "",
        "default_policy": policies[0]["file_path"] if policies else "",
    })


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
    req_motion = request.args.get("motion_file")
    req_ckpt = request.args.get("checkpoint_file")
    motion_file, checkpoint_file = _resolve_valid_motion_and_checkpoint(req_motion, req_ckpt)
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
    motion_file, checkpoint_file = _resolve_valid_motion_and_checkpoint(
        request.args.get("motion_file"),
        request.args.get("checkpoint_file"),
    )
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
