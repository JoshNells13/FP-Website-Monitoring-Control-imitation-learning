from __future__ import annotations

import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import torch

from app.models.database import Checkpoint, Experiment, VideoRecord, runtime_store
from app.services.metrics_service import (
    extract_tensorboard_run,
    get_primary_evaluation_metrics,
    load_run_yaml_configs,
)
from config import CHECKPOINTS_DIR, DATA_DIR, EVALUATION_DIR, HIL_ROOT, LOGS_DIR, MOTIONS_DIR, VIDEOS_DIR

_CKPT_ITER_CACHE: dict[tuple[str, float], int] = {}
_LAST_FS_SIGNATURE: str = ""


def _parse_run_timestamp(folder_name: str, fallback_mtime: float | None = None) -> datetime:
    parts = folder_name.split("_")
    if len(parts) >= 2:
        try:
            dt = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y-%m-%d_%H-%M-%S")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    if fallback_mtime is not None:
        return datetime.fromtimestamp(fallback_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _get_checkpoint_iteration(pt_file: pathlib.Path, default_iter: int = 500) -> int:
    """Read actual 'iter' field from a PyTorch checkpoint file, cached by (path, mtime)."""
    try:
        mtime = pt_file.stat().st_mtime
        key = (str(pt_file), mtime)
        if key in _CKPT_ITER_CACHE:
            return _CKPT_ITER_CACHE[key]
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        it = int(data.get("iter", default_iter)) if isinstance(data, dict) else default_iter
        _CKPT_ITER_CACHE[key] = it
        return it
    except Exception:
        return default_iter


def _infer_motion_file_for_run(run_name: str, env_cfg: dict[str, Any]) -> str:
    """Resolve the exact relative motion .npz path used by a training run."""
    try:
        raw_motion = env_cfg.get("commands", {}).get("motion", {}).get("motion_file")
        if raw_motion:
            p = pathlib.Path(str(raw_motion))
            if p.exists():
                try:
                    return str(p.resolve().relative_to(HIL_ROOT.resolve())).replace("\\", "/")
                except Exception:
                    pass
            candidate = MOTIONS_DIR / p.name
            if candidate.exists():
                return f"data/motions/{candidate.name}"
    except Exception:
        pass

    lower_name = run_name.lower()
    if MOTIONS_DIR.exists():
        for npz in sorted(MOTIONS_DIR.glob("*.npz")):
            tag = npz.stem.replace("k1_", "").lower()
            if tag and tag in lower_name:
                return f"data/motions/{npz.name}"
        all_npz = sorted(MOTIONS_DIR.glob("*.npz"), key=lambda f: f.stat().st_mtime, reverse=True)
        if all_npz:
            return f"data/motions/{all_npz[0].name}"
    return "data/motions/k1_kick.npz"


def compute_hil_filesystem_signature() -> str:
    """Compute a fast MD5 signature of all relevant files and modification times in human-imitation-learning."""
    entries: list[str] = []
    scan_targets = [
        (MOTIONS_DIR, "*.npz"),
        (DATA_DIR / "retargeted", "*.csv"),
        (DATA_DIR / "retargeted", "*.pkl"),
        (CHECKPOINTS_DIR, "*.pt"),
        (CHECKPOINTS_DIR, "*.onnx"),
        (EVALUATION_DIR, "*.json"),
        (EVALUATION_DIR, "*.npz"),
        (EVALUATION_DIR, "*.png"),
        (VIDEOS_DIR, "*.mp4"),
        (HIL_ROOT / "docs", "*.mp4"),
    ]
    for folder, pattern in scan_targets:
        if folder.exists():
            for f in sorted(folder.glob(pattern)):
                try:
                    st = f.stat()
                    entries.append(f"{f.name}:{st.st_size}:{int(st.st_mtime * 1000)}")
                except Exception:
                    pass

    bvh_dir = DATA_DIR / "bvh"
    if bvh_dir.exists():
        for f in sorted(bvh_dir.rglob("*.bvh")):
            try:
                st = f.stat()
                entries.append(f"bvh:{f.name}:{st.st_size}:{int(st.st_mtime)}")
            except Exception:
                pass

    if LOGS_DIR.exists():
        for rdir in sorted(LOGS_DIR.iterdir()):
            if rdir.is_dir():
                try:
                    tfs = list(rdir.glob("events.out.tfevents.*"))
                    pts = list(rdir.glob("model_*.pt"))
                    max_mt = max(
                        [rdir.stat().st_mtime]
                        + [f.stat().st_mtime for f in tfs]
                        + [f.stat().st_mtime for f in pts]
                    )
                    entries.append(f"run:{rdir.name}:{len(pts)}:{int(max_mt * 1000)}")
                except Exception:
                    pass

    raw = "|".join(entries)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def check_and_sync_filesystem() -> tuple[bool, str]:
    """Check if human-imitation-learning files changed since last check; if so, sync in-memory store."""
    global _LAST_FS_SIGNATURE
    sig = compute_hil_filesystem_signature()
    changed = sig != _LAST_FS_SIGNATURE
    if changed:
        _LAST_FS_SIGNATURE = sig
        sync_filesystem_to_db()
    return changed, sig


def sync_filesystem_to_db() -> None:
    """Populate the stateless in-memory runtime_store strictly from real MJLab files (zero fabricated metrics)."""
    with runtime_store.lock:
        eval_metrics = get_primary_evaluation_metrics()
        run_dirs = sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()]) if LOGS_DIR.exists() else []

        run_dir_names = {d.name for d in run_dirs}
        # Prune experiments whose log directory was removed from disk (keep live in-memory running/queued sessions)
        for eid, existing_exp in list(runtime_store.experiments.items()):
            if existing_exp.status not in ("running", "queued", "starting", "paused"):
                if existing_exp.run_id not in run_dir_names:
                    runtime_store.experiments.pop(eid, None)

        # Prune checkpoints whose file_path no longer exists on disk
        for cid, existing_ckpt in list(runtime_store.checkpoints.items()):
            if existing_ckpt.file_path and not (HIL_ROOT / existing_ckpt.file_path).exists():
                runtime_store.checkpoints.pop(cid, None)

        # Prune completed videos whose file_path no longer exists on disk
        for vid_id, existing_vid in list(runtime_store.videos.items()):
            if existing_vid.status == "completed" and existing_vid.file_path and not (HIL_ROOT / existing_vid.file_path).exists():
                runtime_store.videos.pop(vid_id, None)

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

            is_latest_run = idx == len(run_dirs) - 1

            ckpts_in_dir = sorted(
                rdir.glob("model_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]) if len(p.stem.split("_")) > 1 and p.stem.split("_")[1].isdigit() else 0,
            )
            best_ckpt_rel = (
                str(ckpts_in_dir[-1].relative_to(HIL_ROOT)).replace("\\", "/")
                if ckpts_in_dir
                else None
            )

            exp = exp_by_run_id.get(run_id)
            run_name_str = str(agent_cfg.get("run_name", rdir.name))
            inferred_motion = _infer_motion_file_for_run(run_name_str, env_cfg)
            if not exp:
                exp = Experiment(
                    id=runtime_store.next_exp_id(),
                    run_id=run_id,
                    name=f"{run_name_str} ({start_step}–{final_step})",
                    run_name=run_name_str,
                    description=f"Log dir: logs/rsl_rl/k1_tracking/{run_id}",
                    robot_model="booster_k1",
                    task_id="Mjlab-Tracking-Flat-Booster-K1",
                    motion_file=inferred_motion,
                    status="completed",
                    is_archived=False,
                    num_envs=int(env_cfg.get("scene", {}).get("num_envs", 2048)),
                    max_iterations=int(agent_cfg.get("max_iterations", 500)),
                    save_interval=int(agent_cfg.get("save_interval", 100)),
                    seed=int(agent_cfg.get("seed", 42)),
                    created_at=_parse_run_timestamp(run_id, rdir.stat().st_mtime),
                )
                runtime_store.experiments[exp.id] = exp
                exp_by_run_id[run_id] = exp
            else:
                exp.motion_file = inferred_motion
                exp.name = f"{run_name_str} ({start_step}–{final_step})"

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

        # Discover and refresh all primary output checkpoints in outputs/checkpoints/*.pt
        if CHECKPOINTS_DIR.exists():
            eval_ckpt_rel = str(eval_metrics.get("checkpoint_file", "")).replace("\\", "/")
            pt_files = sorted(CHECKPOINTS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            for idx_pt, pt_file in enumerate(pt_files):
                rel_pt = str(pt_file.relative_to(HIL_ROOT)).replace("\\", "/")
                onnx_file = pt_file.with_suffix(".onnx")
                rel_onnx = str(onnx_file.relative_to(HIL_ROOT)).replace("\\", "/") if onnx_file.exists() else None
                p_ckpt = ckpt_by_path.get(rel_pt)
                is_eval_match = (rel_pt == eval_ckpt_rel) or (not eval_ckpt_rel and idx_pt == 0)
                actual_iter = _get_checkpoint_iteration(pt_file, default_iter=2495 if "kick" in pt_file.name else 500)
                stat_pt = pt_file.stat()

                if not p_ckpt:
                    p_ckpt = Checkpoint(
                        id=runtime_store.next_ckpt_id(),
                        experiment_id=None,
                        experiment_run_id="evaluated_policy",
                        experiment_name=rel_pt,
                        name=f"{pt_file.name} (Output Policy)",
                        filename=pt_file.name,
                        file_path=rel_pt,
                        onnx_path=rel_onnx,
                        iteration=actual_iter,
                        file_size_bytes=stat_pt.st_size,
                        is_best=is_eval_match,
                        is_primary_output=True,
                        notes="Output policy checkpoint in outputs/checkpoints/.",
                        created_at=datetime.fromtimestamp(stat_pt.st_mtime, tz=timezone.utc),
                    )
                    runtime_store.checkpoints[p_ckpt.id] = p_ckpt
                    ckpt_by_path[rel_pt] = p_ckpt
                else:
                    p_ckpt.onnx_path = rel_onnx
                    p_ckpt.iteration = actual_iter
                    p_ckpt.file_size_bytes = stat_pt.st_size
                    p_ckpt.is_best = is_eval_match
                    p_ckpt.created_at = datetime.fromtimestamp(stat_pt.st_mtime, tz=timezone.utc)

                if is_eval_match and eval_metrics:
                    latest_exp_rew = (
                        sorted(runtime_store.experiments.values(), key=lambda x: x.created_at)[-1].mean_reward
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

        # Discover actual MP4 files present in outputs/videos/ and docs/
        video_files: list[pathlib.Path] = []
        if VIDEOS_DIR.exists():
            video_files.extend(VIDEOS_DIR.glob("*.mp4"))
        docs_dir = HIL_ROOT / "docs"
        if docs_dir.exists():
            video_files.extend(docs_dir.glob("*.mp4"))

        if video_files:
            vid_by_path = {v.file_path: v for v in runtime_store.videos.values() if v.file_path}
            for vpath in sorted(video_files, key=lambda p: p.stat().st_mtime, reverse=True):
                rel_vpath = str(vpath.relative_to(HIL_ROOT)).replace("\\", "/")
                st_vid = vpath.stat()
                if rel_vpath in vid_by_path:
                    existing_v = vid_by_path[rel_vpath]
                    existing_v.file_size_bytes = st_vid.st_size
                    existing_v.created_at = datetime.fromtimestamp(st_vid.st_mtime, tz=timezone.utc)
                    continue
                fname = vpath.name
                fl = fname.lower()
                vtype = (
                    "comparison"
                    if "comparison" in fl
                    else "reference"
                    if "reference" in fl
                    else "gmr_retargeted"
                    if "gmr" in fl
                    else "error_overlay"
                    if "error_overlay" in fl
                    else "policy"
                )
                inferred_v_motion = _infer_motion_file_for_run(fname, {})
                motion_tag = pathlib.Path(inferred_v_motion).stem
                candidate_ckpt = CHECKPOINTS_DIR / f"{motion_tag}_policy.pt"
                inferred_v_ckpt = (
                    f"outputs/checkpoints/{candidate_ckpt.name}"
                    if candidate_ckpt.exists()
                    else "outputs/checkpoints/k1_kick_policy.pt"
                )
                vrec = VideoRecord(
                    id=runtime_store.next_vid_id(),
                    job_id=f"existing_{vpath.stem}",
                    title=fname,
                    video_type=vtype,
                    camera_pov="first_person" if "first_person" in fl else "third_person",
                    resolution="1924x720" if vtype == "comparison" else "960x720",
                    fps=30 if "gmr" in fl else 50,
                    playback_speed=1.0,
                    include_overlay=("comparison" in fl or "error_overlay" in fl),
                    motion_file=inferred_v_motion,
                    checkpoint_path=inferred_v_ckpt if vtype in ("policy", "comparison", "error_overlay") else None,
                    status="completed",
                    progress=100,
                    file_path=rel_vpath,
                    filename=fname,
                    file_size_bytes=st_vid.st_size,
                    duration_sec=0.0,
                    frame_count=0,
                    created_at=datetime.fromtimestamp(st_vid.st_mtime, tz=timezone.utc),
                )
                runtime_store.videos[vrec.id] = vrec
