from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_file

from app.models.database import runtime_store
from app.services.experiment_service import sync_filesystem_to_db
from app.services.video_service import video_service
from app.utils.system_monitor import safe_resolve_in_hil

videos_bp = Blueprint("videos", __name__, url_prefix="/api/videos")


@videos_bp.route("", methods=["GET"])
def list_videos():
    sync_filesystem_to_db()
    videos = sorted(runtime_store.videos.values(), key=lambda v: v.created_at, reverse=True)
    return jsonify({
        "videos": [v.to_dict() for v in videos],
        "total": len(videos),
    })


@videos_bp.route("/generate", methods=["POST"])
def generate_video():
    payload = request.get_json(silent=True) or {}
    try:
        record = video_service.launch_video_job(payload)
        return jsonify({"ok": True, "video": record}), 202
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@videos_bp.route("/<int:video_id>", methods=["GET"])
def get_video_status(video_id: int):
    v = runtime_store.videos.get(video_id)
    if not v:
        return jsonify({"error": "Video record not found"}), 404
    return jsonify(v.to_dict())


@videos_bp.route("/stream/<int:video_id>", methods=["GET"])
def stream_video(video_id: int):
    v = runtime_store.videos.get(video_id)
    if not v or not v.file_path:
        abort(404)
    path = safe_resolve_in_hil(v.file_path)
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="video/mp4", conditional=True)


@videos_bp.route("/download/<int:video_id>", methods=["GET"])
def download_video(video_id: int):
    v = runtime_store.videos.get(video_id)
    if not v or not v.file_path:
        abort(404)
    path = safe_resolve_in_hil(v.file_path)
    if not path.exists():
        abort(404)
    return send_file(
        str(path),
        mimetype="video/mp4",
        as_attachment=True,
        download_name=v.filename or path.name,
    )
