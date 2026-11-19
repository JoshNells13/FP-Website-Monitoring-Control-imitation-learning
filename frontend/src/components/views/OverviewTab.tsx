"use client";

import React, { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";
import { CameraPOV } from "@/lib/api";
import {
  Box,
  Cpu,
  Activity,
  Video,
  RotateCcw,
  Sliders,
  BarChart2,
  LineChart as LineChartIcon,
} from "lucide-react";

interface OverviewTabProps {
  data: any;
  onNavigate: (tab: string) => void;
  onJumpToPov?: (pov: CameraPOV) => void;
  onDiscardSim?: () => void;
}

const POV_QUICK_AVATARS: { id: CameraPOV; short: string; label: string; bg: string }[] = [
  { id: "third_person", short: "3P", label: "3rd POV", bg: "from-lime-500 to-emerald-600" },
  { id: "front", short: "FR", label: "Front", bg: "from-zinc-800 to-zinc-950" },
  { id: "side", short: "SD", label: "Side", bg: "from-amber-400 to-orange-500" },
  { id: "top", short: "TP", label: "Top", bg: "from-lime-600 to-green-700" },
  { id: "free", short: "OR", label: "Orbit", bg: "from-slate-700 to-slate-900" },
  { id: "first_person", short: "FP", label: "Head FPV", bg: "from-lime-400 to-lime-600" },
];

export default function OverviewTab({
  data,
  onNavigate,
  onJumpToPov,
  onDiscardSim,
}: OverviewTabProps) {
  const [chartMode, setChartMode] = useState<"bar" | "line">("bar");
  const [stageWindow, setStageWindow] = useState<"all" | "latest">("all");
  const [highlightedBin, setHighlightedBin] = useState<number>(0);

  const summary = data?.summary;
  const evalMetrics = data?.evaluation_metrics;
  const bestCkpt = data?.best_checkpoint;
  const latestSession = data?.latest_session;
  const progression: any[] = data?.training_progression || [];
  const jointPerf: any[] = data?.per_joint_performance || [];
  const experiments: any[] = data?.experiments || [];

  // Build chart bins strictly from real TensorBoard progression data
  const curriculumBins = useMemo(() => {
    if (!progression || progression.length === 0) return [];
    const source =
      stageWindow === "latest"
        ? progression.slice(-Math.max(8, Math.floor(progression.length / 4)))
        : progression;
    const count = Math.min(8, source.length);
    const bins = [];
    for (let i = 0; i < count; i++) {
      const idx = Math.min(
        source.length - 1,
        Math.floor((i * (source.length - 1)) / Math.max(1, count - 1))
      );
      const row = source[idx];
      bins.push({
        label: `Iter ${row.iteration}`,
        iteration: row.iteration,
        mean_reward: row.mean_reward !== undefined ? Number(row.mean_reward.toFixed(3)) : 0,
        mean_episode_length:
          row.mean_episode_length !== undefined ? Number(row.mean_episode_length.toFixed(1)) : 0,
        error_body_pos:
          row.error_body_pos !== undefined ? Number(row.error_body_pos.toFixed(4)) : 0,
        value_loss: row.value_loss !== undefined ? Number(row.value_loss.toFixed(5)) : 0,
      });
    }
    return bins;
  }, [progression, stageWindow]);

  // Compute kinematic group breakdown strictly from real k1_metrics.json per_joint_performance
  const kinematicBreakdown = useMemo(() => {
    if (!jointPerf || jointPerf.length < 22) return [];
    const groups = [
      { name: "Right Leg (6-DoF)", idxs: [16, 17, 18, 19, 20, 21], color: "bg-[#fb923c]" },
      { name: "Left Leg (6-DoF)", idxs: [10, 11, 12, 13, 14, 15], color: "bg-[#a3e635]" },
      { name: "Left Arm (4-DoF)", idxs: [2, 3, 4, 5], color: "bg-[#84cc16]" },
      { name: "Right Arm (4-DoF)", idxs: [6, 7, 8, 9], color: "bg-[#4d7c0f]" },
      { name: "Head (2-DoF)", idxs: [0, 1], color: "bg-zinc-400" },
    ];
    const totals = groups.map((g) => {
      const vals = g.idxs
        .map((i) => jointPerf[i]?.mae_rad)
        .filter((v): v is number => typeof v === "number");
      const sum = vals.reduce((a, b) => a + b, 0);
      const mean = vals.length > 0 ? sum / vals.length : 0;
      return { ...g, sum, mean };
    });
    const grandTotal = totals.reduce((a, b) => a + b.sum, 0);
    if (grandTotal <= 0) return [];
    return totals.map((t) => ({
      label: t.name,
      pct: Math.round((t.sum / grandTotal) * 100),
      meanRad: t.mean,
      color: t.color,
    }));
  }, [jointPerf]);

  if (!data || !summary) {
    return (
      <div className="bento-card p-8 text-center text-xs font-mono text-zinc-500">
        Menghubungkan ke Backend MJLab (`http://127.0.0.1:5000/api/dashboard/overview`)... Pastikan server backend sedang berjalan.
      </div>
    );
  }

  const latestReward = summary.latest_mean_reward;
  const peakReward = summary.peak_mean_reward;
  const latestEpLen = summary.latest_episode_length;
  const rmpkpeMm = summary.r_mpkpe_mm;
  const mpkpeMm = summary.mpkpe_mm;
  const jointMaeRad = summary.avg_tracking_error_mae_rad;
  const jointMaeDeg = summary.avg_tracking_error_mae_deg;
  const jointRmseRad = summary.avg_tracking_error_rmse_rad;
  const jointVelRmse = summary.joint_velocity_rmse_rad_s;
  const handEeMm = summary.hand_ee_error_mm;
  const footEeMm = summary.foot_ee_error_mm;
  const anchorOriRad = summary.anchor_orientation_error_rad;
  const anchorOriDeg = summary.anchor_orientation_error_deg;
  const successRate = summary.success_rate;
  const failureRate = summary.failure_rate;
  const numCompletedEps = summary.num_completed_episodes;
  const numEvalEps = summary.num_eval_episodes;

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
      {/* LEFT / CENTER MAIN BENTO AREA (8 Columns on XL) */}
      <div className="xl:col-span-8 flex flex-col gap-5">
        {/* TOP ROW: Real TensorBoard Convergence Chart + Real Summary Metrics */}
        <div className="bento-card p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8 flex flex-col justify-between">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="text-3xl font-extrabold tracking-tight text-zinc-900 dark:text-white font-mono">
                  {latestReward !== null && latestReward !== undefined
                    ? `${latestReward >= 0 ? "+" : ""}${latestReward.toFixed(2)}`
                    : "—"}{" "}
                  <span className="text-sm font-sans font-semibold text-lime-600 dark:text-lime-400">
                    Train/mean_reward
                  </span>
                </div>
                <p className="text-xs text-zinc-400 mt-0.5 font-mono">
                  {latestSession
                    ? `Latest Run: ${latestSession.run_id} (Iter ${latestSession.start_step}–${latestSession.final_step})`
                    : "TensorBoard Training Progression"}
                </p>
              </div>

              <div className="flex items-center gap-2">
                <select
                  value={stageWindow}
                  onChange={(e) => setStageWindow(e.target.value as any)}
                  className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200/80 dark:border-zinc-700 rounded-xl px-2.5 py-1.5 text-xs font-medium text-zinc-700 dark:text-zinc-200"
                >
                  <option value="all">All Logged Runs ({experiments.length})</option>
                  <option value="latest">Latest Run Segment</option>
                </select>

                <div className="flex items-center bg-zinc-100 dark:bg-zinc-800 rounded-xl p-0.5 border border-zinc-200/80 dark:border-zinc-700">
                  <button
                    onClick={() => setChartMode("bar")}
                    className={`p-1.5 rounded-lg transition ${
                      chartMode === "bar"
                        ? "bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white shadow-sm"
                        : "text-zinc-400"
                    }`}
                    title="Bar View"
                  >
                    <BarChart2 className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setChartMode("line")}
                    className={`p-1.5 rounded-lg transition ${
                      chartMode === "line"
                        ? "bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white shadow-sm"
                        : "text-zinc-400"
                    }`}
                    title="Line View"
                  >
                    <LineChartIcon className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-4 text-[11px] text-zinc-500 mt-2 mb-1 font-mono">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#84cc16]" /> Train/mean_reward
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#fb923c]" /> Metrics/motion/error_body_pos (m)
              </span>
            </div>

            <div className="h-56 mt-1">
              {curriculumBins.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={curriculumBins}
                    margin={{ top: 10, right: 5, left: -20, bottom: 0 }}
                    onMouseMove={(state: any) => {
                      if (state?.activeTooltipIndex !== undefined) {
                        setHighlightedBin(state.activeTooltipIndex);
                      }
                    }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e4e4e7" />
                    <XAxis
                      dataKey="label"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#71717a", fontSize: 11 }}
                    />
                    <YAxis
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#a1a1aa", fontSize: 11 }}
                    />
                    <Tooltip
                      cursor={{ fill: "rgba(132, 204, 22, 0.06)" }}
                      contentStyle={{
                        backgroundColor: "#ffffff",
                        borderColor: "#e4e4e7",
                        borderRadius: "14px",
                        boxShadow: "0 10px 25px rgba(0,0,0,0.08)",
                        fontSize: "12px",
                        color: "#18181b",
                      }}
                    />
                    {chartMode === "bar" ? (
                      <>
                        <Bar
                          dataKey="mean_reward"
                          name="Train/mean_reward"
                          radius={[8, 8, 8, 8]}
                          barSize={30}
                        >
                          {curriculumBins.map((_, index) => (
                            <Cell
                              key={`rew-${index}`}
                              fill={index === highlightedBin ? "#84cc16" : "#d4d4d8"}
                            />
                          ))}
                        </Bar>
                        <Bar
                          dataKey="error_body_pos"
                          name="error_body_pos (m)"
                          radius={[8, 8, 8, 8]}
                          barSize={30}
                        >
                          {curriculumBins.map((_, index) => (
                            <Cell
                              key={`err-${index}`}
                              fill={index === highlightedBin ? "#fb923c" : "#e4e4e7"}
                            />
                          ))}
                        </Bar>
                      </>
                    ) : (
                      <>
                        <Line
                          type="monotone"
                          dataKey="mean_reward"
                          name="Train/mean_reward"
                          stroke="#84cc16"
                          strokeWidth={3}
                          dot={{ r: 4, fill: "#84cc16" }}
                        />
                        <Line
                          type="monotone"
                          dataKey="error_body_pos"
                          name="error_body_pos (m)"
                          stroke="#fb923c"
                          strokeWidth={2.5}
                          dot={{ r: 3, fill: "#fb923c" }}
                        />
                      </>
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-xs font-mono text-zinc-400">
                  Belum ada file log TensorBoard di logs/rsl_rl/k1_tracking.
                </div>
              )}
            </div>
          </div>

          {/* 3 Stacked Authentic KPI Summary Blocks on Right of Hero Card */}
          <div className="lg:col-span-4 lg:border-l border-zinc-100 dark:border-zinc-800 lg:pl-6 flex flex-col justify-between gap-4">
            <div>
              <div className="text-xs font-medium text-zinc-400">
                Peak Logged Reward (TensorBoard)
              </div>
              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-1">
                {peakReward !== null && peakReward !== undefined ? `+${peakReward.toFixed(2)}` : "—"}
              </div>
              <div className="text-[11px] font-mono text-zinc-500 mt-1">
                Total Sessions: {summary.total_experiments} ({summary.completed_experiments} completed)
              </div>
            </div>

            <div className="border-t border-zinc-100 dark:border-zinc-800 pt-3.5">
              <div className="text-xs font-medium text-zinc-400">
                Evaluated R-MPKPE / World MPKPE
              </div>
              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-1">
                {rmpkpeMm !== null && rmpkpeMm !== undefined ? `${rmpkpeMm} mm` : "—"}
              </div>
              <div className="text-[11px] font-mono text-zinc-500 mt-1">
                World MPKPE: {mpkpeMm !== null && mpkpeMm !== undefined ? `${mpkpeMm} mm` : "—"}
              </div>
            </div>

            <div className="border-t border-zinc-100 dark:border-zinc-800 pt-3.5">
              <div className="text-xs font-medium text-zinc-400">
                Latest Mean Episode Length
              </div>
              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-1">
                {latestEpLen !== null && latestEpLen !== undefined ? `${latestEpLen} steps` : "—"}
              </div>
              <div className="text-[11px] font-mono text-zinc-500 mt-1">
                Total Wall Time: {summary.total_training_duration_sec}s
              </div>
            </div>
          </div>
        </div>

        {/* MIDDLE ROW: 2 Real Evaluation Metric Cards from outputs/evaluation/k1_metrics.json */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Card 1: Evaluated Joint Position & Velocity Tracking Error */}
          <div className="bento-card p-5 flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                  Joint Position & Velocity Tracking Error
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5 font-mono">
                  Source: outputs/evaluation/k1_metrics.json
                </p>
              </div>
              <button
                onClick={() => onNavigate("motion")}
                className="p-1.5 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-400 hover:text-zinc-900 transition"
                title="Open 22-DoF Motion Analysis"
              >
                <Sliders className="w-4 h-4" />
              </button>
            </div>

            {jointMaeRad !== null && jointMaeRad !== undefined ? (
              <div className="mt-4 space-y-3">
                <div className="w-full h-5 rounded-xl bg-zinc-100 dark:bg-zinc-800 overflow-hidden p-0.5">
                  <div
                    className="h-full rounded-lg hatch-lime transition-all duration-500"
                    style={{ width: `${Math.min(100, Math.round((jointMaeRad / 0.25) * 100))}%` }}
                  />
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                  <div>
                    <span className="text-zinc-400 block text-[10px]">JOINT MAE</span>
                    <span className="font-bold text-zinc-900 dark:text-white">
                      {jointMaeRad} rad ({jointMaeDeg}°)
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-400 block text-[10px]">JOINT RMSE</span>
                    <span className="font-bold text-zinc-900 dark:text-white">
                      {jointRmseRad} rad
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-400 block text-[10px]">VELOCITY RMSE</span>
                    <span className="font-bold text-zinc-900 dark:text-white">
                      {jointVelRmse} rad/s
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="mt-4 text-xs font-mono text-zinc-400">
                Belum ada data k1_metrics.json.
              </div>
            )}
          </div>

          {/* Card 2: Evaluated End-Effector & Torso Anchor Orientation Error (Replaces static tip card!) */}
          <div className="bento-card p-5 flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                  End-Effector & Anchor Orientation Error
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5 font-mono">
                  Task: {evalMetrics?.task_id || "Mjlab-Tracking-Flat-Booster-K1"}
                </p>
              </div>
              <span className="px-2.5 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-[11px] font-mono text-zinc-600 dark:text-zinc-300">
                {evalMetrics?.trajectory_frames_50fps || 149}f @ 50Hz
              </span>
            </div>

            <div className="mt-4 grid grid-cols-3 gap-3 text-xs font-mono">
              <div className="p-3 rounded-2xl bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200/60 dark:border-zinc-700">
                <span className="text-zinc-400 block text-[10px]">HAND EE ERROR</span>
                <span className="text-base font-extrabold text-lime-600 dark:text-lime-400">
                  {handEeMm !== null && handEeMm !== undefined ? `${handEeMm} mm` : "—"}
                </span>
              </div>
              <div className="p-3 rounded-2xl bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200/60 dark:border-zinc-700">
                <span className="text-zinc-400 block text-[10px]">FOOT EE ERROR</span>
                <span className="text-base font-extrabold text-orange-500">
                  {footEeMm !== null && footEeMm !== undefined ? `${footEeMm} mm` : "—"}
                </span>
              </div>
              <div className="p-3 rounded-2xl bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200/60 dark:border-zinc-700">
                <span className="text-zinc-400 block text-[10px]">ANCHOR ORI ERR</span>
                <span className="text-base font-extrabold text-zinc-900 dark:text-white">
                  {anchorOriDeg !== null && anchorOriDeg !== undefined ? `${anchorOriDeg}°` : "—"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* BOTTOM ROW: 3 Real Data Bento Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Bottom Card 1: 22-DoF Kinematic Group Error Split computed from k1_metrics.json */}
          <div className="bento-card p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                    22-DoF Group MAE
                  </h3>
                  <p className="text-[11px] text-zinc-400 font-mono">From per_joint_metrics</p>
                </div>
                <span className="px-2.5 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-[11px] font-mono text-zinc-600 dark:text-zinc-300">
                  {jointPerf.length} DoFs
                </span>
              </div>

              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-3">
                {jointMaeDeg !== null && jointMaeDeg !== undefined ? `${jointMaeDeg}°` : "—"}
              </div>

              {kinematicBreakdown.length > 0 && (
                <div className="mt-3 flex items-center gap-1 h-4 w-full">
                  {kinematicBreakdown.map((item, idx) => (
                    <div
                      key={idx}
                      style={{ width: `${item.pct}%` }}
                      className={`h-full rounded-md ${item.color}`}
                      title={`${item.label}: ${item.meanRad.toFixed(4)} rad (${item.pct}%)`}
                    />
                  ))}
                </div>
              )}
            </div>

            <div className="mt-4 space-y-2 text-xs">
              {kinematicBreakdown.map((item, idx) => (
                <div key={idx} className="flex items-center justify-between font-mono">
                  <span className="flex items-center gap-2 text-zinc-600 dark:text-zinc-300 truncate font-sans">
                    <span className={`w-2.5 h-2.5 rounded-sm shrink-0 ${item.color}`} />
                    <span className="truncate">{item.label}</span>
                  </span>
                  <span className="font-semibold text-zinc-900 dark:text-white">
                    {item.meanRad.toFixed(3)}r ({item.pct}%)
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom Card 2: Multi-Episode Statistical Rollout Success / Failure Rate */}
          <div className="bento-card p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                    Motion Completion
                  </h3>
                  <p className="text-[11px] text-zinc-400 font-mono">
                    {numEvalEps ? `${numEvalEps} parallel episodes` : "Evaluation Rollout"}
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-[11px] font-mono text-zinc-600 dark:text-zinc-300">
                  {evalMetrics?.trajectory_duration_sec ?? 2.98}s
                </span>
              </div>

              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-3">
                {numCompletedEps !== undefined && numEvalEps !== undefined
                  ? `${numCompletedEps} / ${numEvalEps} eps`
                  : "—"}
              </div>
              <div className="text-[11px] font-mono text-zinc-500 mt-1">
                Success: <span className="text-lime-600 font-bold">{successRate ?? "—"}%</span> | Failure:{" "}
                <span className="text-orange-500 font-bold">{failureRate ?? "—"}%</span>
              </div>
            </div>

            {successRate !== null && successRate !== undefined && (
              <div className="my-2 flex flex-col items-center">
                <svg viewBox="0 0 200 115" className="w-44 h-28 overflow-visible">
                  <path
                    d="M 25 100 A 75 75 0 0 1 175 100"
                    fill="none"
                    stroke="#e4e4e7"
                    strokeWidth="18"
                    strokeLinecap="round"
                  />
                  <path
                    d="M 25 100 A 75 75 0 0 1 175 100"
                    fill="none"
                    stroke="#84cc16"
                    strokeWidth="18"
                    strokeLinecap="round"
                    strokeDasharray={`${(Math.min(100, Math.max(0, successRate)) / 100) * 235.6} 235.6`}
                  />
                  <text
                    x="100"
                    y="85"
                    textAnchor="middle"
                    className="fill-zinc-900 dark:fill-white font-mono font-extrabold text-2xl"
                  >
                    {successRate.toFixed(1)}%
                  </text>
                  <text
                    x="100"
                    y="103"
                    textAnchor="middle"
                    className="fill-zinc-400 text-[10px] font-mono"
                  >
                    success_rate_percent
                  </text>
                </svg>
              </div>
            )}

            <p className="text-[11px] text-center text-zinc-400 font-mono">
              Perturbed Batch MAE: {evalMetrics?.perturbed_batch_joint_mae_rad ?? "—"} rad
            </p>
          </div>

          {/* Bottom Card 3: Quantitative Evaluation Metrics Summary (100% from k1_metrics.json) */}
          <div className="bento-card p-5 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                Evaluation Metrics
              </h3>
              <button
                onClick={() => onNavigate("analytics")}
                className="px-2.5 py-1 rounded-full border border-zinc-200 dark:border-zinc-700 text-[11px] font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-zinc-50"
              >
                Full Report
              </button>
            </div>

            <div className="my-3 space-y-3 text-xs font-mono">
              {[
                {
                  title: "Anchor-Relative R-MPKPE",
                  val: rmpkpeMm !== undefined ? `${rmpkpeMm} mm` : "—",
                  ratio: rmpkpeMm !== undefined ? Math.min(100, Math.round((rmpkpeMm / 100) * 100)) : 0,
                  color: "bg-[#84cc16]",
                },
                {
                  title: "World Keybody MPKPE",
                  val: mpkpeMm !== undefined ? `${mpkpeMm} mm` : "—",
                  ratio: mpkpeMm !== undefined ? Math.min(100, Math.round((mpkpeMm / 150) * 100)) : 0,
                  color: "bg-[#84cc16]",
                },
                {
                  title: "Anchor Orientation Error",
                  val:
                    anchorOriRad !== undefined
                      ? `${anchorOriRad} rad (${anchorOriDeg}°)`
                      : "—",
                  ratio:
                    anchorOriRad !== undefined
                      ? Math.min(100, Math.round((anchorOriRad / 0.35) * 100))
                      : 0,
                  color: "bg-[#fb923c]",
                },
                {
                  title: "Perturbed Batch RMSE",
                  val:
                    evalMetrics?.perturbed_batch_joint_rmse_rad !== undefined
                      ? `${evalMetrics.perturbed_batch_joint_rmse_rad} rad`
                      : "—",
                  ratio:
                    evalMetrics?.perturbed_batch_joint_rmse_rad !== undefined
                      ? Math.min(
                          100,
                          Math.round((evalMetrics.perturbed_batch_joint_rmse_rad / 0.4) * 100)
                        )
                      : 0,
                  color: "bg-[#fb923c]",
                },
              ].map((g, i) => (
                <div key={i}>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-sans font-semibold text-zinc-700 dark:text-zinc-200">
                      {g.title}
                    </span>
                    <span className="text-zinc-900 dark:text-white font-bold">{g.val}</span>
                  </div>
                  <div className="mt-1 w-full h-2 rounded-full bg-zinc-100 dark:bg-zinc-800 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${g.color}`}
                      style={{ width: `${g.ratio}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT SIDEBAR COLUMN (4 Columns on XL — Active Checkpoint Card, Quick Controls, Camera POVs, Real Runs List) */}
      <div className="xl:col-span-4 flex flex-col gap-5">
        <div className="bento-card p-5 space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                Active Policy Checkpoint
              </h3>
              <p className="text-[11px] text-zinc-400 font-mono">
                {bestCkpt?.file_path || "outputs/checkpoints/k1_kick_policy.pt"}
              </p>
            </div>
            <button
              onClick={() => onNavigate("checkpoints")}
              className="px-3 py-1.5 rounded-full border border-zinc-200 dark:border-zinc-700 text-xs font-semibold text-zinc-800 dark:text-zinc-200 hover:bg-zinc-50 transition"
            >
              All ({summary.total_checkpoints})
            </button>
          </div>

          {/* Primary Active Checkpoint Card (100% Real Checkpoint Metadata) */}
          <div
            onClick={() => onNavigate("simulation")}
            className="cursor-pointer rounded-2xl p-4 bg-gradient-to-br from-[#65a30d] via-[#84cc16] to-[#a3e635] text-white shadow-md flex flex-col justify-between h-40 relative overflow-hidden"
          >
            <div className="flex items-center justify-between z-10">
              <span className="text-xs font-mono font-semibold text-white/90">
                {evalMetrics?.robot_model || "booster_k1_22dof"}
              </span>
              <span className="font-mono font-bold text-xs bg-black/20 px-2 py-0.5 rounded-full">
                Iter {bestCkpt?.iteration ?? "—"} • {bestCkpt?.file_size_mb ?? "—"} MB
              </span>
            </div>

            <div className="z-10 my-auto">
              <div className="text-[10px] text-white/80 font-mono uppercase">
                CHECKPOINT FILE
              </div>
              <div className="text-base font-mono font-extrabold tracking-wide truncate mt-0.5">
                {bestCkpt?.filename || "—"}
              </div>
              {bestCkpt?.onnx_path && (
                <div className="text-[11px] font-mono text-white/90 truncate">
                  ONNX: {bestCkpt.onnx_path}
                </div>
              )}
            </div>

            <div className="flex items-end justify-between z-10 text-xs font-mono">
              <div>
                <span className="text-white/80">Motion: </span>
                <span className="font-bold">{evalMetrics?.motion_file || "data/motions/k1_kick.npz"}</span>
              </div>
              <div className="font-bold">MAE: {bestCkpt?.tracking_error_mae ?? "—"}r</div>
            </div>
          </div>

          {/* 5 Functional Navigation & Ephemeral Buffer Controls */}
          <div className="grid grid-cols-5 gap-2 pt-1">
            {[
              { label: "3D Sim", icon: Box, action: () => onNavigate("simulation") },
              { label: "Train", icon: Cpu, action: () => onNavigate("training") },
              { label: "Motion", icon: Activity, action: () => onNavigate("motion") },
              { label: "Video", icon: Video, action: () => onNavigate("videos") },
              { label: "Discard", icon: RotateCcw, action: () => onDiscardSim?.() },
            ].map((btn, i) => {
              const Icon = btn.icon;
              return (
                <button
                  key={i}
                  onClick={btn.action}
                  className="flex flex-col items-center gap-1.5 p-2.5 rounded-2xl bg-zinc-50 dark:bg-zinc-800/70 hover:bg-lime-50 dark:hover:bg-zinc-800 border border-zinc-200/60 dark:border-zinc-700/70 transition group"
                >
                  <div className="w-8 h-8 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-700 flex items-center justify-center text-zinc-700 dark:text-zinc-200 group-hover:text-lime-600 shadow-sm">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
                    {btn.label}
                  </span>
                </button>
              );
            })}
          </div>

          {/* 6 Functional Camera POV Launchers */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-xs font-bold text-zinc-800 dark:text-zinc-200">
                Launch 3D Viewer by Camera POV
              </span>
              <span className="text-[11px] font-mono text-zinc-400">6 Modes</span>
            </div>
            <div className="grid grid-cols-6 gap-2">
              {POV_QUICK_AVATARS.map((pov) => (
                <button
                  key={pov.id}
                  onClick={() => onJumpToPov?.(pov.id)}
                  className="flex flex-col items-center gap-1 group"
                  title={`Open 3D Simulation in ${pov.label} mode`}
                >
                  <div
                    className={`w-10 h-10 rounded-full bg-gradient-to-br ${pov.bg} text-white font-mono font-bold text-xs flex items-center justify-center shadow-sm group-hover:scale-105 transition`}
                  >
                    {pov.short}
                  </div>
                  <span className="text-[10px] text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-white truncate w-full text-center">
                    {pov.label}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom-Right Card: Real TensorBoard Runs from logs/rsl_rl/k1_tracking */}
        <div className="bento-card p-5 flex-1 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                Logged Training Runs
              </h3>
              <span className="px-2.5 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-[11px] font-mono text-zinc-600 dark:text-zinc-300">
                {experiments.length} Runs
              </span>
            </div>

            <div className="flex items-center justify-between text-[11px] text-zinc-400 pb-2 border-b border-zinc-100 dark:border-zinc-800 font-mono">
              <span>Run Directory (logs/rsl_rl/k1_tracking)</span>
              <span>Final Return</span>
            </div>

            <div className="divide-y divide-zinc-100 dark:divide-zinc-800/70 mt-1">
              {experiments.map((exp: any, idx: number) => (
                <div
                  key={exp.id}
                  onClick={() => onNavigate("experiments")}
                  className="py-3 flex items-center justify-between gap-3 cursor-pointer hover:bg-zinc-50/80 dark:hover:bg-zinc-800/40 px-1.5 rounded-xl transition"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="w-8 h-8 rounded-xl bg-zinc-900 dark:bg-zinc-800 text-white font-mono font-bold text-xs flex items-center justify-center shrink-0">
                      #{idx + 1}
                    </div>
                    <div className="min-w-0">
                      <div className="text-xs font-bold font-mono text-zinc-900 dark:text-white truncate">
                        {exp.run_id}
                      </div>
                      <div className="text-[11px] text-zinc-400 font-mono truncate">
                        Iter {exp.start_step}→{exp.final_step} • {exp.checkpoint_count} ckpts • {exp.duration_sec}s
                      </div>
                    </div>
                  </div>

                  <div className="text-right shrink-0 font-mono">
                    <div className="text-xs font-extrabold text-zinc-900 dark:text-white">
                      {exp.mean_reward !== null && exp.mean_reward !== undefined
                        ? `${exp.mean_reward >= 0 ? "+" : ""}${exp.mean_reward.toFixed(2)}`
                        : "—"}
                    </div>
                    <div className="text-[10px] text-lime-600 dark:text-lime-400">
                      ep_len: {exp.mean_episode_length ?? "—"}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
