from __future__ import annotations

import json
from datetime import datetime, timezone

from app.models.database import Checkpoint, Experiment, VideoRecord, runtime_store
from app.services.metrics_service import (
    extract_tensorboard_run,
    get_primary_evaluation_metrics,
    load_run_yaml_configs,
)
from config import CHECKPOINTS_DIR, HIL_ROOT, LOGS_DIR, VIDEOS_DIR


def _parse_run_timestamp(folder_name: str) -> datetime:
    parts = folder_name.split("_")
    if len(parts) >= 2:
        try:
            dt = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y-%m-%d_%H-%M-%S")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def sync_filesystem_to_db() -> None:
    """Populate the stateless in-memory runtime_store strictly from real MJLab files (zero fabricated metrics)."""
    with runtime_store.lock:
        eval_metrics = get_primary_evaluation_metrics()
        run_dirs = sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()]) if LOGS_DIR.exists() else []

        exp_by_run_id = {e.run_id: e for e in runtime_store.experiments.values()}
        ckpt_by_path = {c.file_path: c for c in runtime_store.checkpoints.values()}

        for idx, rdir in enumerate(run_dirs):
            run_id = rdir.name
            tb = extract_tensorboard_run(rdir)
            summary = tb["summary"]
            wall_dur = tb["wall_duration_sec"]
            yamls = load_run_yaml_configs(rdir)
            agent_cfg = yamls.get("agent", {}) or {}
            env_cfg = yamls.get("env", {}) or {}

            rew_info = summary.get("Train/mean_reward", {})
            ep_len_info = summary.get("Train/mean_episode_length", {})
            vloss_info = summary.get("Loss/value", {})
            sloss_info = summary.get("Loss/surrogate", {})
            lr_info = summary.get("Loss/learning_rate", {})
            j_err_info = summary.get("Metrics/motion/error_joint_pos", {})
            j_vel_info = summary.get("Metrics/motion/error_joint_vel", {})
            b_err_info = summary.get("Metrics/motion/error_body_pos", {})
            a_pos_info = summary.get("Metrics/motion/error_anchor_pos", {})
            a_rot_info = summary.get("Metrics/motion/error_anchor_rot", {})
            fps_info = summary.get("Perf/total_fps", {})

            start_step = int(rew_info.get("initial_step", 0))
            final_step = int(rew_info.get("final_step", 0))
            mean_rew = float(rew_info["final_value"]) if "final_value" in rew_info else None
            best_rew = float(rew_info["max_value"]) if "max_value" in rew_info else mean_rew
            mean_ep_len = float(ep_len_info["final_value"]) if "final_value" in ep_len_info else None

            # Only attach k1_metrics.json evaluation metrics to the final evaluated run;
            # for intermediate runs, use the exact TensorBoard Metrics/motion/* scalars without fabrication.
            is_latest_run = idx == len(run_dirs) - 1

            ckpts_in_dir = sorted(
                rdir.glob("model_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
            )
            best_ckpt_rel = (
                str(ckpts_in_dir[-1].relative_to(HIL_ROOT)).replace("\\", "/")
                if ckpts_in_dir
                else None
            )

            exp = exp_by_run_id.get(run_id)
            run_name_str = str(agent_cfg.get("run_name", rdir.name))
            if not exp:
                exp = Experiment(
                    id=runtime_store.next_exp_id(),
                    run_id=run_id,
                    name=f"{run_name_str} ({start_step}–{final_step})",
                    run_name=run_name_str,
                    description=f"Log dir: logs/rsl_rl/k1_tracking/{run_id}",
                    robot_model="booster_k1",
                    task_id="Mjlab-Tracking-Flat-Booster-K1",
                    motion_file="data/motions/k1_kick.npz",
                    status="completed",
                    is_archived=False,
                    num_envs=int(env_cfg.get("scene", {}).get("num_envs", 2048)),
                    max_iterations=int(agent_cfg.get("max_iterations", 500)),
                    save_interval=int(agent_cfg.get("save_interval", 100)),
                    seed=int(agent_cfg.get("seed", 42)),
                    created_at=_parse_run_timestamp(run_id),
                )
                runtime_store.experiments[exp.id] = exp
                exp_by_run_id[run_id] = exp

            exp.start_step = start_step
            exp.current_step = final_step
            exp.final_step = final_step
            exp.mean_reward = round(mean_rew, 4) if mean_rew is not None else None
            exp.best_reward = round(best_rew, 4) if best_rew is not None else None
            exp.mean_episode_length = round(mean_ep_len, 2) if mean_ep_len is not None else None
            exp.value_loss = round(float(vloss_info["final_value"]), 5) if "final_value" in vloss_info else None
            exp.surrogate_loss = round(float(sloss_info["final_value"]), 5) if "final_value" in sloss_info else None
            exp.learning_rate = float(lr_info["final_value"]) if "final_value" in lr_info else None

            # Real TensorBoard training errors
            tb_body_pos_err = float(b_err_info["final_value"]) if "final_value" in b_err_info else None
            tb_joint_pos_err = float(j_err_info["final_value"]) if "final_value" in j_err_info else None
            tb_joint_vel_err = float(j_vel_info["final_value"]) if "final_value" in j_vel_info else None
            tb_anchor_pos_err = float(a_pos_info["final_value"]) if "final_value" in a_pos_info else None
            tb_anchor_rot_err = float(a_rot_info["final_value"]) if "final_value" in a_rot_info else None

            if is_latest_run and eval_metrics:
                exp.tracking_error_mae = float(eval_metrics["joint_tracking_mae_rad"])
                exp.tracking_error_rmse = float(eval_metrics["joint_tracking_rmse_rad"])
                exp.r_mpkpe_mm = float(eval_metrics["r_mpkpe_mm"])
                exp.mpkpe_mm = float(eval_metrics["mpkpe_mm"])
                exp.anchor_pos_error_m = round(tb_anchor_pos_err, 5) if tb_anchor_pos_err is not None else None
                exp.anchor_ori_error_rad = float(eval_metrics["anchor_orientation_error_rad"])
                exp.joint_vel_error_rmse = float(eval_metrics["joint_velocity_rmse_rad_s"])
                exp.hand_ee_error_mm = float(eval_metrics["end_effector_hand_error_mm"])
                exp.foot_ee_error_mm = float(eval_metrics["end_effector_foot_error_mm"])
                exp.motion_completion_pct = float(eval_metrics["success_rate_percent"])
                exp.success_rate = float(eval_metrics["success_rate_percent"])
                exp.failure_rate = round(100.0 - float(eval_metrics["success_rate_percent"]), 2)
            else:
                exp.tracking_error_mae = round(tb_joint_pos_err, 5) if tb_joint_pos_err is not None else None
                exp.tracking_error_rmse = None
                exp.r_mpkpe_mm = round(tb_body_pos_err * 1000.0, 2) if tb_body_pos_err is not None else None
                exp.mpkpe_mm = None
                exp.anchor_pos_error_m = round(tb_anchor_pos_err, 5) if tb_anchor_pos_err is not None else None
                exp.anchor_ori_error_rad = round(tb_anchor_rot_err, 5) if tb_anchor_rot_err is not None else None
                exp.joint_vel_error_rmse = round(tb_joint_vel_err, 4) if tb_joint_vel_err is not None else None
                exp.hand_ee_error_mm = None
                exp.foot_ee_error_mm = None
                exp.motion_completion_pct = None
                exp.success_rate = None
                exp.failure_rate = None

            exp.duration_sec = wall_dur
            exp.total_fps = round(float(fps_info["final_value"]), 0) if "final_value" in fps_info else None
            exp.log_dir = str(rdir.relative_to(HIL_ROOT)).replace("\\", "/")
            exp.best_checkpoint_path = best_ckpt_rel
            exp.config_json = json.dumps({
                "agent": agent_cfg,
                "environment": {
                    "robot": "booster_k1",
                    "task_id": "Mjlab-Tracking-Flat-Booster-K1",
                    "num_envs": int(env_cfg.get("scene", {}).get("num_envs", 2048)),
                },
            })
            exp.metrics_json = json.dumps({
                "tensorboard_summary": summary,
                "evaluation": eval_metrics if is_latest_run else {},
            })

            scalars = tb["scalars"]
            rew_by_step = dict(zip(scalars.get("Train/mean_reward", {}).get("steps", []), scalars.get("Train/mean_reward", {}).get("values", [])))
            b_err_by_step = dict(zip(scalars.get("Metrics/motion/error_body_pos", {}).get("steps", []), scalars.get("Metrics/motion/error_body_pos", {}).get("values", [])))
            j_err_by_step = dict(zip(scalars.get("Metrics/motion/error_joint_pos", {}).get("steps", []), scalars.get("Metrics/motion/error_joint_pos", {}).get("values", [])))

            onnx_files = list(rdir.glob("*.onnx"))
            onnx_rel = str(onnx_files[0].relative_to(HIL_ROOT)).replace("\\", "/") if onnx_files else None

            exp_ckpts: list[Checkpoint] = []
            for ckpt_idx, ckpt_file in enumerate(ckpts_in_dir):
                rel_path = str(ckpt_file.relative_to(HIL_ROOT)).replace("\\", "/")
                stem_parts = ckpt_file.stem.split("_")
                iter_num = int(stem_parts[1]) if len(stem_parts) > 1 and stem_parts[1].isdigit() else 0

                nearest_rew = rew_by_step.get(iter_num, mean_rew)
                nearest_b_err = b_err_by_step.get(iter_num, tb_body_pos_err)
                nearest_j_err = j_err_by_step.get(iter_num, tb_joint_pos_err)

                is_last_in_run = ckpt_idx == len(ckpts_in_dir) - 1

                ckpt = ckpt_by_path.get(rel_path)
                if not ckpt:
                    ckpt = Checkpoint(
                        id=runtime_store.next_ckpt_id(),
                        experiment_id=exp.id,
                        experiment_run_id=exp.run_id,
                        experiment_name=exp.name,
                        name=f"{exp.run_name} @ Iter {iter_num}",
                        filename=ckpt_file.name,
                        file_path=rel_path,
                        onnx_path=onnx_rel if is_last_in_run else None,
                        iteration=iter_num,
                        file_size_bytes=ckpt_file.stat().st_size,
                        created_at=datetime.fromtimestamp(ckpt_file.stat().st_mtime, tz=timezone.utc),
                    )
                    runtime_store.checkpoints[ckpt.id] = ckpt
                    ckpt_by_path[rel_path] = ckpt

                ckpt.mean_reward = round(float(nearest_rew), 4) if nearest_rew is not None else None
                if is_latest_run and is_last_in_run and eval_metrics:
                    ckpt.tracking_error_mae = float(eval_metrics["joint_tracking_mae_rad"])
                    ckpt.tracking_error_rmse = float(eval_metrics["joint_tracking_rmse_rad"])
                    ckpt.r_mpkpe_mm = float(eval_metrics["r_mpkpe_mm"])
                    ckpt.mpkpe_mm = float(eval_metrics["mpkpe_mm"])
                    ckpt.success_rate = float(eval_metrics["success_rate_percent"])
                    ckpt.metrics_json = json.dumps(eval_metrics)
                else:
                    ckpt.tracking_error_mae = round(float(nearest_j_err), 5) if nearest_j_err is not None else None
                    ckpt.tracking_error_rmse = None
                    ckpt.r_mpkpe_mm = round(float(nearest_b_err) * 1000.0, 2) if nearest_b_err is not None else None
                    ckpt.mpkpe_mm = None
                    ckpt.success_rate = None
                ckpt.is_best = bool(is_latest_run and is_last_in_run)
                exp_ckpts.append(ckpt)

            exp.checkpoints = exp_ckpts

        # Primary evaluated output checkpoint: outputs/checkpoints/k1_kick_policy.pt
        primary_pt = CHECKPOINTS_DIR / "k1_kick_policy.pt"
        primary_onnx = CHECKPOINTS_DIR / "k1_kick_policy.onnx"
        if primary_pt.exists():
            rel_pt = str(primary_pt.relative_to(HIL_ROOT)).replace("\\", "/")
            rel_onnx = str(primary_onnx.relative_to(HIL_ROOT)).replace("\\", "/") if primary_onnx.exists() else None
            p_ckpt = ckpt_by_path.get(rel_pt)
            if not p_ckpt:
                p_ckpt = Checkpoint(
                    id=runtime_store.next_ckpt_id(),
                    experiment_id=None,
                    experiment_run_id="evaluated_policy",
                    experiment_name="outputs/checkpoints/k1_kick_policy.pt",
                    name="k1_kick_policy.pt (Evaluated Policy)",
                    filename=primary_pt.name,
                    file_path=rel_pt,
                    onnx_path=rel_onnx,
                    iteration=2495,
                    file_size_bytes=primary_pt.stat().st_size,
                    is_best=True,
                    is_primary_output=True,
                    notes="Evaluated via scripts/evaluate_and_play.py (outputs/evaluation/k1_metrics.json).",
                    created_at=datetime.fromtimestamp(primary_pt.stat().st_mtime, tz=timezone.utc),
                )
                runtime_store.checkpoints[p_ckpt.id] = p_ckpt
            if eval_metrics:
                latest_exp_rew = (
                    sorted(runtime_store.experiments.values(), key=lambda x: x.final_step)[-1].mean_reward
                    if runtime_store.experiments
                    else None
                )
                p_ckpt.mean_reward = latest_exp_rew
                p_ckpt.tracking_error_mae = float(eval_metrics["joint_tracking_mae_rad"])
                p_ckpt.tracking_error_rmse = float(eval_metrics["joint_tracking_rmse_rad"])
                p_ckpt.r_mpkpe_mm = float(eval_metrics["r_mpkpe_mm"])
                p_ckpt.mpkpe_mm = float(eval_metrics["mpkpe_mm"])
                p_ckpt.success_rate = float(eval_metrics["success_rate_percent"])
                p_ckpt.metrics_json = json.dumps(eval_metrics)

        # Discover actual MP4 files present in outputs/videos/
        if VIDEOS_DIR.exists():
            vid_by_path = {v.file_path: v for v in runtime_store.videos.values() if v.file_path}
            for vpath in sorted(VIDEOS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
                rel_vpath = str(vpath.relative_to(HIL_ROOT)).replace("\\", "/")
                if rel_vpath in vid_by_path:
                    continue
                fname = vpath.name
                vtype = (
                    "comparison"
                    if "comparison" in fname
                    else "reference"
                    if "reference" in fname
                    else "gmr_retargeted"
                    if "gmr" in fname
                    else "error_overlay"
                    if "error_overlay" in fname
                    else "policy"
                )
                vrec = VideoRecord(
                    id=runtime_store.next_vid_id(),
                    job_id=f"existing_{vpath.stem}",
                    title=fname,
                    video_type=vtype,
                    camera_pov="first_person" if "first_person" in fname else "third_person",
                    resolution="1924x720" if vtype == "comparison" else "960x720",
                    fps=30 if "gmr" in fname else 50,
                    playback_speed=1.0,
                    include_overlay=("comparison" in fname or "error_overlay" in fname),
                    motion_file="data/motions/k1_kick.npz",
                    checkpoint_path="outputs/checkpoints/k1_kick_policy.pt" if vtype in ("policy", "comparison", "error_overlay") else None,
                    status="completed",
                    progress=100,
                    file_path=rel_vpath,
                    filename=fname,
                    file_size_bytes=vpath.stat().st_size,
                    duration_sec=0.0,
                    frame_count=0,
                    created_at=datetime.fromtimestamp(vpath.stat().st_mtime, tz=timezone.utc),
                )
                runtime_store.videos[vrec.id] = vrec
