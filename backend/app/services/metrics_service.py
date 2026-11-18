from __future__ import annotations

import io
import json
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import EVALUATION_DIR, HIL_ROOT, LOGS_DIR

K1_22DOF_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

K1_14_TRACKED_KEYBODIES = [
    "Trunk",
    "Head_2",
    "Left_Hip_Roll",
    "Left_Shank",
    "left_foot_link",
    "Right_Hip_Roll",
    "Right_Shank",
    "right_foot_link",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
]

K1_23_BODY_NAMES = [
    "Trunk",
    "Head_1",
    "Head_2",
    "Left_Arm_1",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_1",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Shank",
    "Left_Ankle_Cross",
    "left_foot_link",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Shank",
    "Right_Ankle_Cross",
    "right_foot_link",
]

# Cache for parsed TensorBoard event directories keyed by (dir_path, mtime)
_TB_CACHE: dict[tuple[str, float], dict[str, Any]] = {}


def extract_tensorboard_run(run_dir: pathlib.Path) -> dict[str, Any]:
    """Read all TensorBoard scalar tags and metadata from an MJLab run directory."""
    if not run_dir.exists() or not run_dir.is_dir():
        return {"scalars": {}, "summary": {}, "wall_duration_sec": 0.0}

    tf_files = list(run_dir.glob("events.out.tfevents.*"))
    if not tf_files:
        return {"scalars": {}, "summary": {}, "wall_duration_sec": 0.0}

    latest_mtime = max(f.stat().st_mtime for f in tf_files)
    cache_key = (str(run_dir), latest_mtime)
    if cache_key in _TB_CACHE:
        return _TB_CACHE[cache_key]

    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])

    scalars: dict[str, dict[str, list[float]]] = {}
    summary: dict[str, dict[str, float]] = {}
    min_wall = None
    max_wall = None

    for tag in tags:
        events = acc.Scalars(tag)
        if not events:
            continue
        steps = [int(e.step) for e in events]
        values = [round(float(e.value), 6) for e in events]
        walls = [float(e.wall_time) for e in events]
        if min_wall is None or walls[0] < min_wall:
            min_wall = walls[0]
        if max_wall is None or walls[-1] > max_wall:
            max_wall = walls[-1]

        scalars[tag] = {
            "steps": steps,
            "values": values,
        }
        summary[tag] = {
            "initial_step": steps[0],
            "initial_value": values[0],
            "final_step": steps[-1],
            "final_value": values[-1],
            "min_value": min(values),
            "max_value": max(values),
            "mean_value": round(float(np.mean(values)), 6),
        }

    wall_duration = round(max_wall - min_wall, 2) if (min_wall and max_wall and max_wall >= min_wall) else 0.0
    result = {
        "scalars": scalars,
        "summary": summary,
        "wall_duration_sec": wall_duration,
    }
    _TB_CACHE[cache_key] = result
    return result


def load_run_yaml_configs(run_dir: pathlib.Path) -> dict[str, Any]:
    """Load params/agent.yaml and params/env.yaml from an MJLab run folder."""
    params_dir = run_dir / "params"
    out: dict[str, Any] = {}
    for name in ("agent", "env"):
        p = params_dir / f"{name}.yaml"
        if p.exists():
            try:
                out[name] = yaml.safe_load(p.read_text(encoding="utf-8"))
            except Exception:
                out[name] = {}
    return out


def get_primary_evaluation_metrics() -> dict[str, Any]:
    """Load outputs/evaluation/k1_metrics.json produced by evaluate_and_play.py."""
    metrics_file = EVALUATION_DIR / "k1_metrics.json"
    if not metrics_file.exists():
        return {}
    try:
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_iteration_series(scalars: dict[str, dict[str, list[float]]], max_points: int = 250) -> list[dict[str, Any]]:
    """Convert raw TensorBoard scalars dict into a unified array of objects keyed by iteration step for Recharts."""
    primary_tag = "Train/mean_reward"
    if primary_tag not in scalars:
        if not scalars:
            return []
        primary_tag = next(iter(scalars.keys()))

    steps = scalars[primary_tag]["steps"]
    n = len(steps)
    if n == 0:
        return []

    indices = np.linspace(0, n - 1, min(n, max_points), dtype=int) if n > max_points else np.arange(n)

    tag_map = {
        "Train/mean_reward": "mean_reward",
        "Train/mean_episode_length": "mean_episode_length",
        "Loss/value": "value_loss",
        "Loss/surrogate": "surrogate_loss",
        "Loss/entropy": "entropy_loss",
        "Loss/learning_rate": "learning_rate",
        "Metrics/motion/error_joint_pos": "error_joint_pos",
        "Metrics/motion/error_joint_vel": "error_joint_vel",
        "Metrics/motion/error_body_pos": "error_body_pos",
        "Metrics/motion/error_body_rot": "error_body_rot",
        "Metrics/motion/error_anchor_pos": "error_anchor_pos",
        "Metrics/motion/error_anchor_rot": "error_anchor_rot",
        "Metrics/motion/error_body_lin_vel": "error_body_lin_vel",
        "Metrics/motion/error_body_ang_vel": "error_body_ang_vel",
        "Metrics/motion/sampling_entropy": "sampling_entropy",
        "Metrics/motion/sampling_top1_prob": "sampling_top1_prob",
        "Episode_Reward/motion_body_pos": "reward_body_pos",
        "Episode_Reward/motion_body_ori": "reward_body_ori",
        "Episode_Reward/motion_global_root_pos": "reward_root_pos",
        "Episode_Reward/motion_global_root_ori": "reward_root_ori",
        "Episode_Reward/motion_body_lin_vel": "reward_body_lin_vel",
        "Episode_Reward/motion_body_ang_vel": "reward_body_ang_vel",
        "Episode_Reward/action_rate_l2": "penalty_action_rate",
        "Episode_Reward/joint_limit": "penalty_joint_limit",
        "Episode_Reward/self_collisions": "penalty_self_collision",
        "Episode_Termination/time_out": "term_time_out",
        "Episode_Termination/anchor_pos": "term_anchor_pos",
        "Episode_Termination/anchor_ori": "term_anchor_ori",
        "Episode_Termination/ee_body_pos": "term_ee_body_pos",
        "Perf/total_fps": "total_fps",
        "Perf/collection_time": "collection_time",
        "Perf/learning_time": "learning_time",
        "Policy/mean_std": "policy_std",
    }

    lookup: dict[str, dict[int, float]] = {}
    for tb_tag, key in tag_map.items():
        if tb_tag in scalars:
            s_steps = scalars[tb_tag]["steps"]
            s_vals = scalars[tb_tag]["values"]
            lookup[key] = dict(zip(s_steps, s_vals))

    rows: list[dict[str, Any]] = []
    for idx in indices:
        step = int(steps[idx])
        row: dict[str, Any] = {"iteration": step}
        for key, step_dict in lookup.items():
            if step in step_dict:
                row[key] = round(float(step_dict[step]), 5)
        # Estimate survival / completion proxy from episode_Termination or mean_episode_length
        if "term_time_out" in row:
            total_terms = (
                row.get("term_time_out", 0.0)
                + row.get("term_anchor_pos", 0.0)
                + row.get("term_anchor_ori", 0.0)
                + row.get("term_ee_body_pos", 0.0)
            )
            if total_terms > 0:
                row["timeout_survival_pct"] = round(100.0 * row["term_time_out"] / total_terms, 2)
        rows.append(row)
    return rows


def get_combined_training_progression(max_points: int = 300) -> list[dict[str, Any]]:
    """Aggregate all chronological runs in logs/rsl_rl/k1_tracking into a continuous training curve."""
    if not LOGS_DIR.exists():
        return []
    run_dirs = sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()])
    merged_by_step: dict[int, dict[str, Any]] = {}
    for rdir in run_dirs:
        tb = extract_tensorboard_run(rdir)
        rows = build_iteration_series(tb["scalars"], max_points=600)
        for r in rows:
            r["run_id"] = rdir.name
            merged_by_step[int(r["iteration"])] = r

    all_rows = [merged_by_step[k] for k in sorted(merged_by_step.keys())]
    if len(all_rows) > max_points:
        idxs = np.linspace(0, len(all_rows) - 1, max_points, dtype=int)
        all_rows = [all_rows[i] for i in idxs]
    return all_rows


def export_metrics_dataframe(series: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert iteration series into a Pandas DataFrame for CSV/JSON export."""
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series)


def generate_matplotlib_report_png(run_dir: pathlib.Path | None = None) -> bytes:
    """Generate a high-resolution scientific 4-panel Matplotlib PNG report in memory."""
    if run_dir and run_dir.exists():
        tb = extract_tensorboard_run(run_dir)
        rows = build_iteration_series(tb["scalars"], max_points=500)
        title_suffix = run_dir.name
    else:
        rows = get_combined_training_progression(max_points=500)
        title_suffix = "Full Curriculum (Iterations 0 - 2495)"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor="#0b0f19")
    for ax in axes.flat:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values():
            spine.set_color("#1f2937")
        ax.grid(True, linestyle=":", color="#374151", alpha=0.6)

    steps = [r["iteration"] for r in rows]

    # Panel 1: Mean Episode Return
    ax = axes[0, 0]
    if rows and "mean_reward" in rows[0]:
        ax.plot(steps, [r.get("mean_reward", 0) for r in rows], color="#38bdf8", linewidth=2.0, label="Mean Reward")
    ax.set_title("Mean Episode Return", color="#f3f4f6", fontsize=11, fontweight="bold")
    ax.set_xlabel("PPO Iteration", color="#9ca3af")
    ax.set_ylabel("Reward", color="#9ca3af")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb")

    # Panel 2: Episode Horizon Survival
    ax = axes[0, 1]
    if rows and "mean_episode_length" in rows[0]:
        ax.plot(steps, [r.get("mean_episode_length", 0) for r in rows], color="#34d399", linewidth=2.0, label="Episode Length (steps)")
    ax.set_title("Episode Horizon Survival (steps)", color="#f3f4f6", fontsize=11, fontweight="bold")
    ax.set_xlabel("PPO Iteration", color="#9ca3af")
    ax.set_ylabel("Steps @ 50Hz", color="#9ca3af")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb")

    # Panel 3: Tracking Error Metrics
    ax = axes[1, 0]
    for key, label, col in [
        ("error_joint_pos", "Joint Pos Error (rad)", "#fb923c"),
        ("error_body_pos", "Relative Body Pos Error (m)", "#f43f5e"),
        ("error_anchor_pos", "Anchor Pos Error (m)", "#a78bfa"),
    ]:
        if rows and key in rows[0]:
            ax.plot(steps, [r.get(key, 0) for r in rows], color=col, linewidth=1.8, label=label)
    ax.set_title("Tracking Error Convergence", color="#f3f4f6", fontsize=11, fontweight="bold")
    ax.set_xlabel("PPO Iteration", color="#9ca3af")
    ax.set_ylabel("Error", color="#9ca3af")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8)

    # Panel 4: BeyondMimic Exponential Rewards
    ax = axes[1, 1]
    for key, label, col in [
        ("reward_body_pos", "Body Pos Reward", "#38bdf8"),
        ("reward_body_ori", "Body Ori Reward", "#34d399"),
        ("reward_root_pos", "Anchor Pos Reward", "#facc15"),
        ("reward_root_ori", "Anchor Ori Reward", "#c084fc"),
    ]:
        if rows and key in rows[0]:
            ax.plot(steps, [r.get(key, 0) for r in rows], color=col, linewidth=1.6, label=label)
    ax.set_title("BeyondMimic Exponential Tracking Rewards", color="#f3f4f6", fontsize=11, fontweight="bold")
    ax.set_xlabel("PPO Iteration", color="#9ca3af")
    ax.set_ylabel("Reward Component", color="#9ca3af")
    ax.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8)

    fig.suptitle(
        f"Booster K1 (22-DoF) BeyondMimic PPO Analytics — {title_suffix}",
        color="#f9fafb",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()
