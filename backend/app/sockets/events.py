from __future__ import annotations

from flask_socketio import emit

from app.services.experiment_service import check_and_sync_filesystem
from app.services.simulation_service import simulation_controller
from app.services.training_service import training_manager
from app.utils.system_monitor import get_system_resources
from extensions import socketio


@socketio.on("connect")
def handle_connect() -> None:
    _, sig = check_and_sync_filesystem()
    emit("training_status", training_manager.get_state())
    emit("simulation_status", simulation_controller.get_status())
    emit("system_resources", get_system_resources())
    emit("hil_filesystem_status", {"signature": sig})


@socketio.on("request_status")
def handle_request_status() -> None:
    changed, sig = check_and_sync_filesystem()
    emit("training_status", training_manager.get_state())
    emit("simulation_status", simulation_controller.get_status())
    emit("system_resources", get_system_resources())
    if changed:
        emit("hil_filesystem_changed", {"signature": sig}, broadcast=True)


@socketio.on("simulation_control")
def handle_simulation_control(payload: dict) -> None:
    state = simulation_controller.control(payload or {})
    emit("simulation_status", state, broadcast=True)
