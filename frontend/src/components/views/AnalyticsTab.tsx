"use client";

import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { API_BASE, apiFetch } from "@/lib/api";
import {
  FileJson,
  FileSpreadsheet,
  Image as ImageIcon,
  Filter,
} from "lucide-react";

export default function AnalyticsTab() {
  const [selectedExpId, setSelectedExpId] = useState<string>("");
  const [analyticsData, setAnalyticsData] = useState<any>(null);
  const [selectedCkptIds, setSelectedCkptIds] = useState<number[]>([]);

  const loadAnalytics = async (expId: string) => {
    try {
      const q = expId ? `?experiment_id=${expId}` : "";
      const res = await apiFetch(`/api/analytics/summary${q}`);
      setAnalyticsData(res);
      if (selectedCkptIds.length === 0 && res.checkpoints?.length) {
        const step = Math.max(1, Math.floor(res.checkpoints.length / 6));
        const sampled = res.checkpoints
          .filter((_: any, i: number) => i % step === 0 || i === res.checkpoints.length - 1)
          .map((c: any) => c.id);
        setSelectedCkptIds(sampled);
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadAnalytics(selectedExpId);
  }, [selectedExpId]);

  if (!analyticsData) {
    return (
      <div className="bento-card p-8 text-center text-xs font-mono text-zinc-500">
        Memuat data analitik TensorBoard & k1_metrics.json dari `/api/analytics/summary`...
      </div>
    );
  }

  const evalMetrics = analyticsData.evaluation_metrics || {};
  const series = analyticsData.iteration_series || [];
  const experiments = analyticsData.experiments || [];
  const checkpoints = analyticsData.checkpoints || [];
  const selectedExp = analyticsData.selected_experiment;

  const comparedCheckpoints = checkpoints.filter((c: any) => selectedCkptIds.includes(c.id));

  const toggleCkpt = (id: number) => {
    setSelectedCkptIds((prev) =>
      prev.includes(id) ? (prev.length > 1 ? prev.filter((x) => x !== id) : prev) : [...prev, id]
    );
  };

  const exportUrl = (fmt: "csv" | "json") => {
    const q = selectedExpId ? `&experiment_id=${selectedExpId}` : "";
    return `${API_BASE}/api/analytics/export?format=${fmt}${q}`;
  };

  const reportPngUrl = () => {
    const q = selectedExpId ? `?experiment_id=${selectedExpId}` : "";
    return `${API_BASE}/api/analytics/report.png${q}`;
  };

  const succVal = selectedExp?.success_rate ?? evalMetrics.success_rate_percent;
  const failVal =
    selectedExp?.failure_rate ??
    (succVal !== undefined && succVal !== null ? Number((100 - succVal).toFixed(2)) : null);
  const rmpkpeVal = selectedExp?.r_mpkpe_mm ?? evalMetrics.r_mpkpe_mm;
  const anchorPosVal = selectedExp?.anchor_pos_error_m;
  const anchorOriVal = selectedExp?.anchor_ori_error_rad ?? evalMetrics.anchor_orientation_error_rad;
  const jointMaeVal = selectedExp?.tracking_error_mae ?? evalMetrics.joint_tracking_mae_rad;
  const jointRmseVal = selectedExp?.tracking_error_rmse ?? evalMetrics.joint_tracking_rmse_rad;
  const jointVelRmseVal = selectedExp?.joint_vel_error_rmse ?? evalMetrics.joint_velocity_rmse_rad_s;
  const epLenVal = selectedExp?.mean_episode_length;
  const valLossVal = selectedExp?.value_loss;
  const surrLossVal = selectedExp?.surrogate_loss;
  const meanRewVal = selectedExp?.mean_reward;
  const bestRewVal = selectedExp?.best_reward;

  return (
    <div className="space-y-5">
      <div className="bento-card p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5 text-zinc-500 font-semibold">
            <Filter className="w-3.5 h-3.5 text-lime-600" /> Session Filter:
          </span>
          <select
            value={selectedExpId}
            onChange={(e) => setSelectedExpId(e.target.value)}
            className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-1.5 text-zinc-900 dark:text-white font-medium"
          >
            <option value="">All Training Sessions (Combined TensorBoard Progression)</option>
            {experiments.map((e: any) => (
              <option key={e.id} value={e.id}>
                {e.run_id} (Iter {e.start_step}–{e.final_step})
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs font-semibold">
          <a
            href={exportUrl("csv")}
            download
            className="px-3.5 py-2 rounded-xl bg-lime-100 hover:bg-lime-200 text-lime-900 flex items-center gap-1.5 transition"
          >
            <FileSpreadsheet className="w-3.5 h-3.5" /> Export CSV
          </a>
          <a
            href={exportUrl("json")}
            download
            className="px-3.5 py-2 rounded-xl bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-800 text-zinc-900 dark:text-white flex items-center gap-1.5 transition"
          >
            <FileJson className="w-3.5 h-3.5" /> Export JSON
          </a>
          <a
            href={reportPngUrl()}
            target="_blank"
            rel="noreferrer"
            className="px-3.5 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-white flex items-center gap-1.5 transition"
          >
            <ImageIcon className="w-3.5 h-3.5" /> Matplotlib Report PNG
          </a>
        </div>
      </div>

      {/* 12 Authentic Metric Tiles (Zero hardcoded fallback values) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
        {[
          {
            label: "1. Success Rate",
            val: succVal !== null && succVal !== undefined ? `${succVal}%` : "—",
            sub:
              evalMetrics.num_completed_episodes !== undefined
                ? `${evalMetrics.num_completed_episodes}/${evalMetrics.num_eval_episodes} eps`
                : "Not evaluated",
            accent: "text-lime-600",
          },
          {
            label: "2. Failure Rate",
            val: failVal !== null && failVal !== undefined ? `${failVal}%` : "—",
            sub: "Early termination",
            accent: "text-orange-500",
          },
          {
            label: "3. Keybody Error",
            val: rmpkpeVal !== null && rmpkpeVal !== undefined ? `${rmpkpeVal} mm` : "—",
            sub: "error_body_pos / R-MPKPE",
            accent: "text-zinc-900 dark:text-white",
          },
          {
            label: "4. Root Position Error",
            val: anchorPosVal !== null && anchorPosVal !== undefined ? `${anchorPosVal} m` : "—",
            sub:
              evalMetrics.mpkpe_mm !== undefined
                ? `World MPKPE: ${evalMetrics.mpkpe_mm} mm`
                : "TensorBoard anchor_pos",
            accent: "text-zinc-900 dark:text-white",
          },
          {
            label: "5. Root Orientation",
            val: anchorOriVal !== null && anchorOriVal !== undefined ? `${anchorOriVal} rad` : "—",
            sub:
              evalMetrics.anchor_orientation_error_deg !== undefined
                ? `${evalMetrics.anchor_orientation_error_deg}° Trunk tilt`
                : "TensorBoard anchor_rot",
            accent: "text-zinc-900 dark:text-white",
          },
          {
            label: "6. Joint Position Error",
            val: jointMaeVal !== null && jointMaeVal !== undefined ? `${jointMaeVal} rad` : "—",
            sub: jointRmseVal !== null && jointRmseVal !== undefined ? `RMSE: ${jointRmseVal} rad` : "error_joint_pos",
            accent: "text-lime-600",
          },
          {
            label: "7. Joint Velocity Error",
            val: jointVelRmseVal !== null && jointVelRmseVal !== undefined ? `${jointVelRmseVal} rad/s` : "—",
            sub:
              evalMetrics.joint_velocity_mae_rad_s !== undefined
                ? `MAE: ${evalMetrics.joint_velocity_mae_rad_s} rad/s`
                : "error_joint_vel",
            accent: "text-zinc-900 dark:text-white",
          },
          {
            label: "8. Motion Completion",
            val: succVal !== null && succVal !== undefined ? `${succVal}%` : "—",
            sub:
              evalMetrics.trajectory_frames_50fps !== undefined
                ? `${evalMetrics.trajectory_frames_50fps} frames @ 50Hz`
                : "—",
            accent: "text-lime-600",
          },
          {
            label: "9. Episode Duration",
            val: epLenVal !== null && epLenVal !== undefined ? `${(epLenVal * 0.02).toFixed(2)} s` : "—",
            sub: epLenVal !== null && epLenVal !== undefined ? `${epLenVal} steps @ 50Hz` : "—",
            accent: "text-zinc-900 dark:text-white",
          },
          {
            label: "10. Value Loss",
            val: valLossVal !== null && valLossVal !== undefined ? `${valLossVal}` : "—",
            sub: surrLossVal !== null && surrLossVal !== undefined ? `Surrogate: ${surrLossVal}` : "Loss/value",
            accent: "text-orange-500",
          },
          {
            label: "11. Mean Return",
            val: meanRewVal !== null && meanRewVal !== undefined ? `+${meanRewVal}` : "—",
            sub: bestRewVal !== null && bestRewVal !== undefined ? `Max: +${bestRewVal}` : "Train/mean_reward",
            accent: "text-lime-600",
          },
          {
            label: "12. End-Effector Error",
            val:
              evalMetrics.end_effector_hand_error_mm !== undefined
                ? `${evalMetrics.end_effector_hand_error_mm} / ${evalMetrics.end_effector_foot_error_mm} mm`
                : "—",
            sub: "Hand / Foot (k1_metrics.json)",
            accent: "text-zinc-900 dark:text-white",
          },
        ].map((m, i) => (
          <div key={i} className="bento-card p-3.5">
            <span className="text-[11px] text-zinc-400 font-medium">{m.label}</span>
            <div className={`text-lg font-extrabold font-mono mt-1 ${m.accent}`}>{m.val}</div>
            <span className="text-[10px] text-zinc-400 font-mono">{m.sub}</span>
          </div>
        ))}
      </div>

      {/* Interactive Recharts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="bento-card p-5">
          <h3 className="text-sm font-bold text-zinc-900 dark:text-white mb-1">
            BeyondMimic 6 Exponential Tracking Reward Components
          </h3>
          <p className="text-xs text-zinc-400 mb-4 font-mono">
            Episode_Reward/* tags extracted directly from TensorBoard event files
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={series}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                <XAxis dataKey="iteration" stroke="#71717a" fontSize={11} />
                <YAxis stroke="#71717a" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e4e4e7", borderRadius: "12px", fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Line type="monotone" dataKey="reward_body_pos" name="motion_body_pos" stroke="#84cc16" dot={false} strokeWidth={2.5} />
                <Line type="monotone" dataKey="reward_body_ori" name="motion_body_ori" stroke="#4d7c0f" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="reward_body_lin_vel" name="motion_body_lin_vel" stroke="#fb923c" dot={false} strokeWidth={1.8} />
                <Line type="monotone" dataKey="reward_body_ang_vel" name="motion_body_ang_vel" stroke="#18181b" dot={false} strokeWidth={1.8} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bento-card p-5">
          <h3 className="text-sm font-bold text-zinc-900 dark:text-white mb-1">
            Adaptive Failure-Bin Curriculum Entropy & Early Terminations
          </h3>
          <p className="text-xs text-zinc-400 mb-4 font-mono">
            Metrics/motion/sampling_* & Episode_Termination/* from TensorBoard
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={series}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                <XAxis dataKey="iteration" stroke="#71717a" fontSize={11} />
                <YAxis yAxisId="left" stroke="#84cc16" fontSize={11} />
                <YAxis yAxisId="right" orientation="right" stroke="#fb923c" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e4e4e7", borderRadius: "12px", fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Line yAxisId="left" type="monotone" dataKey="sampling_entropy" name="sampling_entropy" stroke="#18181b" dot={false} strokeWidth={2} />
                <Line yAxisId="left" type="monotone" dataKey="sampling_top1_prob" name="sampling_top1_prob" stroke="#84cc16" dot={false} strokeWidth={2.2} />
                <Line yAxisId="right" type="monotone" dataKey="term_ee_body_pos" name="ee_body_pos termination" stroke="#fb923c" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Multi-Checkpoint Comparison Matrix */}
      <div className="bento-card p-6">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
              Compare Checkpoints Across Iterations
            </h3>
            <p className="text-xs text-zinc-400 font-mono">
              Logged TensorBoard error_body_pos (mm) & Train/mean_reward at checkpoint save step
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
            {checkpoints.slice(0, 14).map((c: any) => {
              const active = selectedCkptIds.includes(c.id);
              return (
                <button
                  key={c.id}
                  onClick={() => toggleCkpt(c.id)}
                  className={`px-2.5 py-1 rounded-xl text-[11px] font-mono border transition ${
                    active
                      ? "bg-zinc-900 text-white border-zinc-900 font-bold"
                      : "bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-600"
                  }`}
                >
                  {c.is_primary_output ? "★ k1_kick_policy.pt" : `Iter ${c.iteration}`}
                </button>
              );
            })}
          </div>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparedCheckpoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
              <XAxis dataKey="filename" stroke="#71717a" fontSize={11} />
              <YAxis yAxisId="left" stroke="#fb923c" fontSize={11} />
              <YAxis yAxisId="right" orientation="right" stroke="#84cc16" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e4e4e7", borderRadius: "12px", fontSize: "12px" }}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Bar yAxisId="left" dataKey="r_mpkpe_mm" name="Body Pos Error (mm)" fill="#fb923c" radius={[6, 6, 0, 0]} />
              <Bar yAxisId="right" dataKey="mean_reward" name="Train/mean_reward" fill="#84cc16" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Matplotlib Generated Evaluation Plots Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="bento-card p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold font-mono text-zinc-700 dark:text-zinc-200">
              outputs/evaluation/k1_joint_tracking_comparison.png
            </span>
            <a
              href={`${API_BASE}/api/analytics/evaluation-plots/k1_joint_tracking_comparison.png`}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-lime-600 font-semibold hover:underline"
            >
              Open Full Resolution ↗
            </a>
          </div>
          <img
            src={`${API_BASE}/api/analytics/evaluation-plots/k1_joint_tracking_comparison.png`}
            alt="Joint Tracking Comparison Plot"
            className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800"
          />
        </div>

        <div className="bento-card p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold font-mono text-zinc-700 dark:text-zinc-200">
              Live Matplotlib Report (/api/analytics/report.png)
            </span>
            <a
              href={reportPngUrl()}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-lime-600 font-semibold hover:underline"
            >
              Download PNG Report ↗
            </a>
          </div>
          <img
            src={reportPngUrl()}
            alt="Matplotlib PPO Convergence Report"
            className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800"
          />
        </div>
      </div>
    </div>
  );
}
