from __future__ import annotations

import collections
import os
import pathlib
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any

import psutil

from app.models.database import Experiment, runtime_store
from app.services.experiment_service import sync_filesystem_to_db
from app.services.metrics_service import extract_tensorboard_run
from app.utils.system_monitor import get_system_resources, safe_resolve_in_hil
from config import CHECKPOINTS_DIR, DATA_DIR, HIL_ROOT, LOGS_DIR, MOTIONS_DIR, PYTHON_EXE, SCRIPTS_DIR
from extensions import socketio

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class TrainingProcessManager:
    """Stateless in-memory lifecycle manager for live MJLab BeyondMimic PPO training and evaluation processes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._status: str = "idle"
        self._active_config: dict[str, Any] = {}
        self._active_experiment_id: int | None = None
        self._active_log_dir: pathlib.Path | None = None
        self._start_time: float | None = None
        self._current_iteration: int = 0
        self._max_iterations: int = 500
        self._latest_metrics: dict[str, Any] = {}
        self._logs: collections.deque[dict[str, Any]] = collections.deque(maxlen=1500)
        self._app = None

    def init_app(self, app) -> None:
        self._app = app

    def _append_log(self, line: str, level: str = "info") -> None:
        clean_line = ANSI_ESCAPE_RE.sub("", line).rstrip()
        if not clean_line:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "line": clean_line,
            "level": level,
        }
        self._logs.append(entry)
        try:
            socketio.emit("training_log", entry)
        except Exception:
            pass

    def get_logs(self, limit: int = 400) -> list[dict[str, Any]]:
        items = list(self._logs)
        return items[-limit:]

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            pid = self._process.pid if self._process and self._process.poll() is None else None
            elapsed = round(time.time() - self._start_time, 1) if self._start_time and self._status in ("running", "paused", "starting") else 0.0
            progress_pct = (
                round(100.0 * min(1.0, self._current_iteration / max(1, self._max_iterations)), 1)
                if self._max_iterations > 0
                else 0.0
            )
            return {
                "status": self._status,
                "pid": pid,
                "experiment_id": self._active_experiment_id,
                "config": self._active_config,
                "log_dir": str(self._active_log_dir.relative_to(HIL_ROOT)).replace("\\", "/") if self._active_log_dir else None,
                "current_iteration": self._current_iteration,
                "max_iterations": self._max_iterations,
                "progress_percent": progress_pct,
                "elapsed_sec": elapsed,
                "latest_metrics": self._latest_metrics,
                "resources": get_system_resources(),
                "storage_mode": "ephemeral_in_memory",
            }

    def validate_training_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        robot = str(payload.get("robot", "booster_k1")).strip()
        if robot not in ("booster_k1",):
            raise ValueError(f"Unsupported robot model '{robot}'. Allowed: ['booster_k1'].")

        motion_file = str(payload.get("motion_file", "data/motions/k1_kick.npz")).strip()
        motion_path = safe_resolve_in_hil(motion_file)
        if not motion_path.exists() or motion_path.suffix.lower() != ".npz":
            raise ValueError(f"Valid .npz reference motion file not found: {motion_file}")

        num_envs = int(payload.get("num_envs", 2048))
        if not (1 <= num_envs <= 8192):
            raise ValueError("num_envs must be between 1 and 8192.")

        max_iterations = int(payload.get("max_iterations", 500))
        if not (1 <= max_iterations <= 50000):
            raise ValueError("max_iterations must be between 1 and 50000.")

        save_interval = int(payload.get("save_interval", 100))
        if not (1 <= save_interval <= 10000):
            raise ValueError("save_interval must be between 1 and 10000.")

        seed = int(payload.get("seed", 42))
        run_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(payload.get("run_name", "beyondmimic_k1_kick")).strip())
        if not run_name:
            run_name = "beyondmimic_k1_kick"

        output_ckpt = str(payload.get("output_checkpoint", "outputs/checkpoints/k1_kick_policy.pt")).strip()
        out_ckpt_path = safe_resolve_in_hil(output_ckpt)
        if out_ckpt_path.suffix.lower() != ".pt":
            raise ValueError("output_checkpoint must end with .pt")

        resume_from = payload.get("resume_from")
        resume_rel = None
        if resume_from:
            resume_path = safe_resolve_in_hil(str(resume_from).strip())
            if not resume_path.exists() or resume_path.suffix.lower() != ".pt":
                raise ValueError(f"Resume checkpoint file not found: {resume_from}")
            resume_rel = str(resume_path.relative_to(HIL_ROOT)).replace("\\", "/")

        return {
            "robot": robot,
            "motion_file": str(motion_path.relative_to(HIL_ROOT)).replace("\\", "/"),
            "num_envs": num_envs,
            "max_iterations": max_iterations,
            "save_interval": save_interval,
            "seed": seed,
            "run_name": run_name,
            "output_checkpoint": str(out_ckpt_path.relative_to(HIL_ROOT)).replace("\\", "/"),
            "resume_from": resume_rel,
            "name": str(payload.get("name", f"Booster K1 Live PPO — {run_name}")).strip(),
            "description": str(payload.get("description", "")).strip(),
        }

    def start_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("A training process is already active. Stop or wait for it first.")

            validated = self.validate_training_params(payload)
            script_path = SCRIPTS_DIR / "train_ppo.py"
            if not script_path.exists():
                raise FileNotFoundError(f"Training script not found at {script_path}")

            cmd = [
                str(PYTHON_EXE),
                "-u",
                str(script_path),
                "--robot",
                validated["robot"],
                "--motion-file",
                validated["motion_file"],
                "--num-envs",
                str(validated["num_envs"]),
                "--max-iterations",
                str(validated["max_iterations"]),
                "--save-interval",
                str(validated["save_interval"]),
                "--seed",
                str(validated["seed"]),
                "--run-name",
                validated["run_name"],
                "--output-checkpoint",
                validated["output_checkpoint"],
            ]
            if validated["resume_from"]:
                cmd.extend(["--resume-from", validated["resume_from"]])

            self._logs.clear()
            self._status = "starting"
            self._active_config = validated
            self._start_time = time.time()
            self._current_iteration = 0
            self._max_iterations = validated["max_iterations"]
            self._latest_metrics = {}
            self._active_log_dir = None

            ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            provisional_run_id = f"{ts_str}_{validated['run_name']}"
            exp = Experiment(
                id=runtime_store.next_exp_id(),
                run_id=provisional_run_id,
                name=validated["name"],
                run_name=validated["run_name"],
                description=validated["description"] or "Live in-memory monitored MJLab PPO training session.",
                robot_model=validated["robot"],
                task_id="Mjlab-Tracking-Flat-Booster-K1",
                motion_file=validated["motion_file"],
                status="running",
                num_envs=validated["num_envs"],
                max_iterations=validated["max_iterations"],
                save_interval=validated["save_interval"],
                seed=validated["seed"],
                resume_from=validated["resume_from"],
                start_step=0,
                current_step=0,
                final_step=validated["max_iterations"],
            )
            runtime_store.experiments[exp.id] = exp
            runtime_store.add_activity(
                category="training",
                action=f"Started Live Training: {validated['name']}",
                details=f"Command: train_ppo.py --robot {validated['robot']} --num-envs {validated['num_envs']} --max-iterations {validated['max_iterations']}",
                status="info",
                experiment_id=exp.id,
            )
            self._active_experiment_id = exp.id

            env_vars = os.environ.copy()
            env_vars["PYTHONUNBUFFERED"] = "1"
            env_vars["MPLCONFIGDIR"] = str(HIL_ROOT / ".cache" / "matplotlib")
            env_vars["WARP_CACHE_PATH"] = str(HIL_ROOT / ".cache" / "warp")

            self._append_log(f"[LIVE MONITOR] Launching MJLab PPO Training: {' '.join(cmd)}", "info")

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
            self._process = subprocess.Popen(
                cmd,
                cwd=str(HIL_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env_vars,
                creationflags=creationflags,
            )
            self._status = "running"

            threading.Thread(target=self._monitor_stdout_loop, daemon=True).start()
            threading.Thread(target=self._monitor_tensorboard_loop, daemon=True).start()

        socketio.emit("training_status", self.get_state())
        return self.get_state()

    def _monitor_stdout_loop(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return

        iter_pattern = re.compile(r"Learning iteration\s+(\d+)/(\d+)", re.IGNORECASE)
        logdir_pattern = re.compile(r"Logging experiment in directory:\s*(.+)", re.IGNORECASE)
        rew_pattern = re.compile(r"Mean reward:\s*([-\d\.]+)", re.IGNORECASE)
        eplen_pattern = re.compile(r"Mean episode length:\s*([-\d\.]+)", re.IGNORECASE)

        for raw_line in iter(proc.stdout.readline, ""):
            if not raw_line:
                break
            clean = ANSI_ESCAPE_RE.sub("", raw_line).strip()
            if not clean:
                continue

            level = "error" if ("error" in clean.lower() or "traceback" in clean.lower()) else "info"
            self._append_log(clean, level=level)

            m_dir = logdir_pattern.search(clean)
            if m_dir:
                detected_dir = pathlib.Path(m_dir.group(1).strip())
                if detected_dir.exists():
                    with self._lock:
                        self._active_log_dir = detected_dir

            m_iter = iter_pattern.search(clean)
            if m_iter:
                cur_it = int(m_iter.group(1))
                tot_it = int(m_iter.group(2))
                with self._lock:
                    self._current_iteration = cur_it
                    self._max_iterations = tot_it
                socketio.emit("training_progress", self.get_state())

            m_rew = rew_pattern.search(clean)
            if m_rew:
                try:
                    with self._lock:
                        self._latest_metrics["mean_reward"] = float(m_rew.group(1))
                except Exception:
                    pass

            m_ep = eplen_pattern.search(clean)
            if m_ep:
                try:
                    with self._lock:
                        self._latest_metrics["mean_episode_length"] = float(m_ep.group(1))
                except Exception:
                    pass

        rc = proc.wait()
        with self._lock:
            if self._status == "stopping":
                self._status = "stopped"
                final_status = "stopped"
            elif rc == 0:
                self._status = "completed"
                self._current_iteration = self._max_iterations
                final_status = "completed"
            else:
                self._status = "failed"
                final_status = "failed"

        self._append_log(f"[LIVE MONITOR] Training process exited with return code {rc} (status: {final_status})", "info" if rc == 0 else "error")

        if self._active_experiment_id and self._active_experiment_id in runtime_store.experiments:
            exp = runtime_store.experiments[self._active_experiment_id]
            exp.status = final_status
            if self._active_log_dir and self._active_log_dir.exists():
                exp.run_id = self._active_log_dir.name
                exp.log_dir = str(self._active_log_dir.relative_to(HIL_ROOT)).replace("\\", "/")
        sync_filesystem_to_db()
        socketio.emit("training_status", self.get_state())

    def _monitor_tensorboard_loop(self) -> None:
        while True:
            with self._lock:
                proc = self._process
                status = self._status
                log_dir = self._active_log_dir

            if proc is None or proc.poll() is not None or status not in ("running", "paused", "starting"):
                break

            try:
                if log_dir and log_dir.exists():
                    tb = extract_tensorboard_run(log_dir)
                    summary = tb.get("summary", {})
                    if summary:
                        with self._lock:
                            rew_s = summary.get("Train/mean_reward")
                            if rew_s:
                                self._current_iteration = int(rew_s["final_step"])
                                self._latest_metrics["mean_reward"] = rew_s["final_value"]
                            ep_s = summary.get("Train/mean_episode_length")
                            if ep_s:
                                self._latest_metrics["mean_episode_length"] = ep_s["final_value"]
                            b_s = summary.get("Metrics/motion/error_body_pos")
                            if b_s:
                                self._latest_metrics["error_body_pos"] = b_s["final_value"]
                            j_s = summary.get("Metrics/motion/error_joint_pos")
                            if j_s:
                                self._latest_metrics["error_joint_pos"] = j_s["final_value"]
                socketio.emit("training_progress", self.get_state())
            except Exception:
                pass
            time.sleep(2.0)

    def pause_training(self) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None or self._status != "running":
                raise RuntimeError("No running training process to pause.")
            parent = psutil.Process(self._process.pid)
            for child in parent.children(recursive=True):
                child.suspend()
            parent.suspend()
            self._status = "paused"
            self._append_log("[LIVE MONITOR] Training process suspended (paused).", "info")
        socketio.emit("training_status", self.get_state())
        return self.get_state()

    def resume_training(self) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None or self._status != "paused":
                raise RuntimeError("No paused training process to resume.")
            parent = psutil.Process(self._process.pid)
            parent.resume()
            for child in parent.children(recursive=True):
                child.resume()
            self._status = "running"
            self._append_log("[LIVE MONITOR] Training process resumed.", "info")
        socketio.emit("training_status", self.get_state())
        return self.get_state()

    def stop_training(self) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._status = "idle"
                return self.get_state()

            self._status = "stopping"
            self._append_log("[LIVE MONITOR] Stopping training process safely...", "info")
            try:
                parent = psutil.Process(self._process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.resume()
                        child.terminate()
                    except Exception:
                        pass
                parent.resume()
                parent.terminate()
                _, alive = psutil.wait_procs([parent, *children], timeout=3.0)
                for p in alive:
                    try:
                        p.kill()
                    except Exception:
                        pass
            except Exception as exc:
                self._append_log(f"[LIVE MONITOR] Warning during process termination: {exc}", "error")

        socketio.emit("training_status", self.get_state())
        return self.get_state()

    def get_available_options(self) -> dict[str, Any]:
        motions = []
        if MOTIONS_DIR.exists():
            for f in sorted(MOTIONS_DIR.glob("*.npz")):
                motions.append({
                    "name": f.name,
                    "path": str(f.relative_to(HIL_ROOT)).replace("\\", "/"),
                    "size_kb": round(f.stat().st_size / 1024.0, 1),
                })

        retargeted = []
        ret_dir = DATA_DIR / "retargeted"
        if ret_dir.exists():
            for f in sorted(ret_dir.glob("*.csv")):
                retargeted.append({
                    "name": f.name,
                    "path": str(f.relative_to(HIL_ROOT)).replace("\\", "/"),
                })

        bvh_files = []
        bvh_dir = DATA_DIR / "bvh"
        if bvh_dir.exists():
            for f in sorted(bvh_dir.rglob("*.bvh")):
                bvh_files.append({
                    "name": f.name,
                    "path": str(f.relative_to(HIL_ROOT)).replace("\\", "/"),
                })

        checkpoints = []
        if CHECKPOINTS_DIR.exists():
            for f in sorted(CHECKPOINTS_DIR.glob("*.pt")):
                checkpoints.append({
                    "label": f"Production Output: {f.name}",
                    "path": str(f.relative_to(HIL_ROOT)).replace("\\", "/"),
                })
        if LOGS_DIR.exists():
            for rdir in sorted(LOGS_DIR.iterdir(), reverse=True):
                if rdir.is_dir():
                    for f in sorted(rdir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0, reverse=True):
                        checkpoints.append({
                            "label": f"{rdir.name} / {f.name}",
                            "path": str(f.relative_to(HIL_ROOT)).replace("\\", "/"),
                        })

        return {
            "robots": [
                {
                    "id": "booster_k1",
                    "name": "Booster K1 (22-DoF Humanoid)",
                    "dof": 22,
                    "anchor_body": "Trunk",
                    "tracked_keybodies": 14,
                    "actor_obs_dim": 125,
                    "critic_obs_dim": 251,
                }
            ],
            "environments": [
                {
                    "id": "Mjlab-Tracking-Flat-Booster-K1",
                    "name": "Mjlab-Tracking-Flat-Booster-K1 (MuJoCo-Warp GPU Flat Terrain)",
                    "config_fn": "booster_k1_flat_tracking_env_cfg",
                    "runner_cfg_fn": "booster_k1_tracking_ppo_runner_cfg",
                    "control_hz": 50,
                    "sim_dt": 0.005,
                    "decimation": 4,
                }
            ],
            "motions": motions,
            "retargeted_csvs": retargeted,
            "bvh_files": bvh_files,
            "checkpoints": checkpoints,
            "defaults": {
                "robot": "booster_k1",
                "environment": "Mjlab-Tracking-Flat-Booster-K1",
                "motion_file": "data/motions/k1_kick.npz",
                "num_envs": 2048,
                "max_iterations": 500,
                "save_interval": 100,
                "seed": 42,
                "run_name": "beyondmimic_k1_kick_finetune",
                "output_checkpoint": "outputs/checkpoints/k1_kick_policy.pt",
            },
        }


training_manager = TrainingProcessManager()
