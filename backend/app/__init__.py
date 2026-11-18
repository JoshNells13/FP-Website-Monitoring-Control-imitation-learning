from __future__ import annotations

from flask import Flask, jsonify

from app.routes import ALL_BLUEPRINTS
from app.services.experiment_service import sync_filesystem_to_db
from app.services.training_service import training_manager
from app.services.video_service import video_service
from config import Config
from extensions import cors, socketio
import app.sockets  # noqa: F401


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})
    socketio.init_app(app, cors_allowed_origins="*")

    training_manager.init_app(app)
    video_service.init_app(app)

    for bp in ALL_BLUEPRINTS:
        app.register_blueprint(bp)

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "booster-k1-imitation-control-backend",
            "storage_mode": "ephemeral_in_memory_no_database",
        })

    sync_filesystem_to_db()
    return app
