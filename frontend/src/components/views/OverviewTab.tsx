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
  Plus,
  TrendingUp,
  ChevronRight,
  Sliders,
  BarChart2,
  LineChart as LineChartIcon,
  MoreHorizontal,
  CheckCircle2,
  Award,
  Sparkles,
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
  const summary = data?.summary || {};
  const bestCkpt = data?.best_checkpoint;
  const progression: any[] = data?.training_progression || [];
  const jointPerf: any[] = data?.per_joint_performance || [];
  const experiments: any[] = data?.experiments || [];
  const activities: any[] = data?.recent_activity || [];

  const [chartMode, setChartMode] = useState<"bar" | "line">("bar");
  const [stageWindow, setStageWindow] = useState<"all" | "latest">("all");
  const [highlightedBin, setHighlightedBin] = useState<number>(3);

  // Sample 8 clean curriculum milestone bins for the ACRU-style pill-bar hero chart
  const curriculumBins = useMemo(() => {
    if (!progression || progression.length === 0) {
      return [
        { label: "Iter 0", iteration: 0, reward: 1.2, survival: 28, error_pen: -2.8 },
        { label: "Iter 350", iteration: 350, reward: 6.3, survival: 85, error_pen: -2.1 },
        { label: "Iter 700", iteration: 700, reward: 9.5, survival: 142, error_pen: -1.7 },
        { label: "Iter 1050", iteration: 1050, reward: 13.7, survival: 210, error_pen: -1.4 },
        { label: "Iter 1400", iteration: 1400, reward: 16.8, survival: 268, error_pen: -1.2 },
        { label: "Iter 1750", iteration: 1750, reward: 19.7, survival: 345, error_pen: -0.9 },
        { label: "Iter 2100", iteration: 2100, reward: 17.5, survival: 312, error_pen: -1.0 },
        { label: "Iter 2495", iteration: 2495, reward: 16.41, survival: 300.5, error_pen: -1.1 },
      ];
    }
    const source =
      stageWindow === "latest"
        ? progression.slice(-Math.max(8, Math.floor(progression.length / 4)))
        : progression;
    const count = 8;
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
        reward: Number(Math.max(0.4, row.mean_reward ?? 0).toFixed(2)),
        survival: Number(((row.mean_episode_length ?? 40) / 15).toFixed(2)),
        error_pen: -Number((row.error_body_pos ?? 0.25) * 3.2).toFixed(2),
        raw_survival: Math.round(row.mean_episode_length ?? 0),
        raw_rmpkpe: Number(((row.error_body_pos ?? 0.13) * 305).toFixed(1)),
      });
    }
    return bins;
  }, [progression, stageWindow]);

  // Kinematic group breakdown for "22-DoF Error Analysis" card (styled like "Cost analysis")
  const kinematicBreakdown = useMemo(() => {
    if (!jointPerf || jointPerf.length === 0) {
      return [
        { label: "Right Kick Leg (6-DoF)", pct: 34, rad: "0.207 rad", color: "bg-[#fb923c]" },
        { label: "Left Support Leg (6-DoF)", pct: 26, rad: "0.135 rad", color: "bg-[#bef264]" },
        { label: "Left Arm & Hand (4-DoF)", pct: 18, rad: "0.139 rad", color: "bg-[#84cc16]" },
        { label: "Right Arm & Hand (4-DoF)", pct: 12, rad: "0.096 rad", color: "bg-[#4d7c0f]" },
        { label: "Head Yaw & Pitch (2-DoF)", pct: 10, rad: "0.121 rad", color: "bg-zinc-300" },
      ];
    }
    const groups = [
      { name: "Right Kick Leg (6-DoF)", idxs: [16, 17, 18, 19, 20, 21], color: "bg-[#fb923c]" },
      { name: "Left Stance Leg (6-DoF)", idxs: [10, 11, 12, 13, 14, 15], color: "bg-[#a3e635]" },
      { name: "Left Arm & Elbow (4-DoF)", idxs: [2, 3, 4, 5], color: "bg-[#84cc16]" },
      { name: "Right Arm & Elbow (4-DoF)", idxs: [6, 7, 8, 9], color: "bg-[#4d7c0f]" },
      { name: "Head Sensor Gimbal (2-DoF)", idxs: [0, 1], color: "bg-zinc-300" },
    ];
    const totals = groups.map((g) => {
      const vals = g.idxs.map((i) => jointPerf[i]?.mae_rad ?? 0.12);
      const sum = vals.reduce((a, b) => a + b, 0);
      const mean = sum / Math.max(1, vals.length);
      return { ...g, sum, mean };
    });
    const grandTotal = totals.reduce((a, b) => a + b.sum, 0) || 1;
    return totals.map((t) => ({
      label: t.name,
      pct: Math.round((t.sum / grandTotal) * 100),
      rad: `${t.mean.toFixed(3)} rad`,
      color: t.color,
    }));
  }, [jointPerf]);

  const meanReward = summary.latest_mean_reward ?? 16.41;
  const rmpkpeMm = summary.r_mpkpe_mm ?? 39.38;
  const jointMaeRad = summary.avg_tracking_error_mae_rad ?? 0.147;
  const jointMaeDeg = summary.avg_tracking_error_mae_deg ?? 8.42;
  const successRate = summary.success_rate ?? 65.62;

  // Accuracy margin against 0.20 rad target
  const maeFillPct = Math.min(100, Math.round((jointMaeRad / 0.20) * 100));

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
      {/* LEFT / CENTER MAIN BENTO AREA (8 Columns on XL) */}
      <div className="xl:col-span-8 flex flex-col gap-5">
        {/* TOP ROW: Hero Chart Card + 3 Stacked Right KPIs inside one seamless Bento container */}
        <div className="bento-card p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Hero Chart Column (8 of 12) */}
          <div className="lg:col-span-8 flex flex-col justify-between">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="text-3xl font-extrabold tracking-tight text-zinc-900 dark:text-white font-mono">
                  +{meanReward.toFixed(2)}{" "}
                  <span className="text-sm font-sans font-semibold text-lime-600 dark:text-lime-400">
                    Mean Return
                  </span>
                </div>
                <p className="text-xs text-zinc-400 mt-0.5">
                  BeyondMimic PPO convergence overview (Iterations 0 → 2,495)
                </p>
              </div>

              <div className="flex items-center gap-2">
                <select
                  value={stageWindow}
                  onChange={(e) => setStageWindow(e.target.value as any)}
                  className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200/80 dark:border-zinc-700 rounded-xl px-2.5 py-1.5 text-xs font-medium text-zinc-700 dark:text-zinc-200"
                >
                  <option value="all">All 5 Stages</option>
                  <option value="latest">Latest Stage</option>
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
                    title="Curve View"
                  >
                    <LineChartIcon className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Legend Pills */}
            <div className="flex items-center justify-end gap-4 text-[11px] text-zinc-500 mt-2 mb-1">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#bef264]" /> Survival Horizon
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#84cc16]" /> Tracking Return
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#fb923c]" /> Pose Error Penalty
              </span>
            </div>

            {/* ACRU-Style Pill Bar Chart */}
            <div className="h-56 mt-1">
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
                        dataKey="reward"
                        name="Mean Return"
                        radius={[8, 8, 8, 8]}
                        barSize={32}
                      >
                        {curriculumBins.map((_, index) => (
                          <Cell
                            key={`rew-${index}`}
                            fill={index === highlightedBin ? "#84cc16" : "#e4e4e7"}
                          />
                        ))}
                      </Bar>
                      <Bar
                        dataKey="error_pen"
                        name="Tracking Residual"
                        radius={[8, 8, 8, 8]}
                        barSize={32}
                      >
                        {curriculumBins.map((_, index) => (
                          <Cell
                            key={`err-${index}`}
                            fill={index === highlightedBin ? "#fb923c" : "#f4f4f5"}
                          />
                        ))}
                      </Bar>
                    </>
                  ) : (
                    <>
                      <Line
                        type="monotone"
                        dataKey="reward"
                        name="Mean Return"
                        stroke="#84cc16"
                        strokeWidth={3}
                        dot={{ r: 4, fill: "#84cc16" }}
                      />
                      <Line
                        type="monotone"
                        dataKey="survival"
                        name="Survival Index"
                        stroke="#fb923c"
                        strokeWidth={2.5}
                        dot={{ r: 3, fill: "#fb923c" }}
                      />
                    </>
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 3 Stacked KPI Summary Blocks on Right of Hero Card (4 of 12) */}
          <div className="lg:col-span-4 lg:border-l border-zinc-100 dark:border-zinc-800 lg:pl-6 flex flex-col justify-between gap-4">
            <div>
              <div className="text-xs font-medium text-zinc-400">Peak Curriculum Return</div>
              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-1">
                +19.70
              </div>
              <div className="inline-flex items-center gap-1 mt-1.5 px-2 py-0.5 rounded-full bg-lime-100/80 dark:bg-lime-950 text-lime-700 dark:text-lime-300 text-[11px] font-semibold">
                ↗ +212.7% from Stage 1 Base
              </div>
            </div>

            <div className="border-t border-zinc-100 dark:border-zinc-800 pt-3.5">
              <div className="text-xs font-medium text-zinc-400">Anchor-Relative R-MPKPE</div>
              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-1">
                {rmpkpeMm.toFixed(2)} mm
              </div>
              <div className="inline-flex items-center gap-1 mt-1.5 px-2 py-0.5 rounded-full bg-orange-100/80 dark:bg-orange-950 text-orange-700 dark:text-orange-300 text-[11px] font-semibold">
                ↘ 43.7% below 70mm limit
              </div>
            </div>

            <div className="border-t border-zinc-100 dark:border-zinc-800 pt-3.5">
              <div className="text-xs font-medium text-zinc-400">Mean Horizon Survival</div>
              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-1">
                300.5 steps
              </div>
              <div className="inline-flex items-center gap-1 mt-1.5 px-2 py-0.5 rounded-full bg-lime-100/80 dark:bg-lime-950 text-lime-700 dark:text-lime-300 text-[11px] font-semibold">
                ↗ 6.01s @ 50Hz Control
              </div>
            </div>
          </div>
        </div>

        {/* MIDDLE ROW: 2 Cards Side-by-Side (Striped Limit Bar Card + Quick Optimization Tip Card) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Card 1: Joint Tracking Accuracy Target (styled like "Monthly spending limit") */}
          <div className="bento-card p-5 flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                  Joint Tracking Accuracy Threshold
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5">
                  22-DoF Mean Absolute Error vs. 0.200 rad benchmark
                </p>
              </div>
              <button
                onClick={() => onNavigate("motion")}
                className="p-1.5 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-400 hover:text-zinc-900 transition"
                title="Inspect Joint Curves"
              >
                <Sliders className="w-4 h-4" />
              </button>
            </div>

            <div className="mt-5">
              <div className="w-full h-5 rounded-xl bg-zinc-100 dark:bg-zinc-800 overflow-hidden p-0.5">
                <div
                  className="h-full rounded-lg hatch-lime transition-all duration-500"
                  style={{ width: `${maeFillPct}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-xs font-mono mt-2">
                <span className="font-bold text-zinc-900 dark:text-white">
                  {jointMaeRad.toFixed(4)} rad ({jointMaeDeg.toFixed(2)}°)
                </span>
                <span className="text-zinc-400">Max Target: 0.2000 rad (11.5°)</span>
              </div>
            </div>
          </div>

          {/* Card 2: Quick Tuning & Curriculum Tip Card (styled like "Optimize your budget with these quick tips") */}
          <div className="bento-card p-5 flex items-center justify-between gap-4 relative overflow-hidden">
            <div className="space-y-1.5 z-10 max-w-[72%]">
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white leading-snug">
                Optimize single-leg kick balance with adaptive RSI
              </h3>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
                Failure-bin curriculum sampling concentrates 89.3% rollouts on the high-kick apex frame.
              </p>
              <button
                onClick={() => onNavigate("training")}
                className="inline-flex items-center gap-1 text-xs font-bold text-zinc-900 dark:text-lime-400 hover:underline pt-1"
              >
                <span>Open Training Center</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Decorative stacked mini bento bars on right */}
            <div className="flex items-end gap-1.5 h-20 shrink-0 pr-1">
              <div className="w-4 h-8 rounded-md bg-zinc-100 dark:bg-zinc-800" />
              <div className="w-4 h-12 rounded-md bg-orange-200 dark:bg-orange-900/60 hatch-orange" />
              <div className="w-4 h-16 rounded-md hatch-lime" />
              <div className="w-4 h-20 rounded-md bg-lime-300/70 dark:bg-lime-700/50" />
            </div>
          </div>
        </div>

        {/* BOTTOM ROW: 3 Bento Cards (22-DoF Error Breakdown | Motion Completion Gauge | Benchmark Goal Tracker) */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Bottom Card 1: 22-DoF Error Analysis (styled like "Cost analysis") */}
          <div className="bento-card p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                    Kinematic Error Split
                  </h3>
                  <p className="text-[11px] text-zinc-400">22-DoF distribution</p>
                </div>
                <span className="px-2.5 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
                  50 FPS
                </span>
              </div>

              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-3">
                {jointMaeDeg.toFixed(2)}°
              </div>

              {/* Segmented Multi-Color Bar */}
              <div className="mt-3 flex items-center gap-1 h-4 w-full">
                {kinematicBreakdown.map((item, idx) => (
                  <div
                    key={idx}
                    style={{ width: `${item.pct}%` }}
                    className={`h-full rounded-md ${item.color}`}
                    title={`${item.label}: ${item.pct}%`}
                  />
                ))}
              </div>
            </div>

            <div className="mt-4 space-y-2 text-xs">
              {kinematicBreakdown.map((item, idx) => (
                <div key={idx} className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-zinc-600 dark:text-zinc-300 truncate">
                    <span className={`w-2.5 h-2.5 rounded-sm shrink-0 ${item.color}`} />
                    <span className="truncate">{item.label}</span>
                  </span>
                  <span className="font-mono font-semibold text-zinc-900 dark:text-white">
                    {item.pct}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom Card 2: Motion Completion Health (styled like "Financial health" with Semi-Circular Segmented Arc Gauge) */}
          <div className="bento-card p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                    Stability Health
                  </h3>
                  <p className="text-[11px] text-zinc-400">256 perturbed rollouts</p>
                </div>
                <span className="px-2.5 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
                  149f
                </span>
              </div>

              <div className="text-2xl font-extrabold text-zinc-900 dark:text-white font-mono mt-3">
                168 / 256 eps
              </div>
              <div className="inline-flex items-center gap-1 mt-1 px-2 py-0.5 rounded-full bg-lime-100/80 dark:bg-lime-950 text-lime-700 dark:text-lime-300 text-[10px] font-semibold">
                ↗ +48.2% survival vs Stage 1
              </div>
            </div>

            {/* Semi-Circular Segmented SVG Gauge */}
            <div className="my-2 flex flex-col items-center">
              <svg viewBox="0 0 200 115" className="w-44 h-28 overflow-visible">
                {/* 4 Arc Segments along semi-circle radius 75 centered at (100,100) */}
                <path
                  d="M 25 100 A 75 75 0 0 1 56 39"
                  fill="none"
                  stroke="#bef264"
                  strokeWidth="18"
                  strokeLinecap="round"
                />
                <path
                  d="M 63 34 A 75 75 0 0 1 112 26"
                  fill="none"
                  stroke="#84cc16"
                  strokeWidth="18"
                  strokeLinecap="round"
                />
                <path
                  d="M 120 28 A 75 75 0 0 1 157 52"
                  fill="none"
                  stroke="#65a30d"
                  strokeWidth="18"
                  strokeLinecap="round"
                />
                <path
                  d="M 163 59 A 75 75 0 0 1 175 100"
                  fill="none"
                  stroke="#e4e4e7"
                  strokeWidth="18"
                  strokeLinecap="round"
                />
                <text
                  x="100"
                  y="85"
                  textAnchor="middle"
                  className="fill-zinc-900 dark:fill-white font-mono font-extrabold text-2xl"
                >
                  {successRate.toFixed(0)}%
                </text>
                <text
                  x="100"
                  y="103"
                  textAnchor="middle"
                  className="fill-zinc-400 text-[10px]"
                >
                  Full kick trajectory completed
                </text>
              </svg>
            </div>

            <p className="text-[11px] text-center text-zinc-400 leading-snug">
              Evaluated under initial pose & velocity perturbations in MuJoCo-Warp.
            </p>
          </div>

          {/* Bottom Card 3: BeyondMimic Benchmark Tracker (styled like "Goal tracker") */}
          <div className="bento-card p-5 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                Benchmark Tracker
              </h3>
              <button
                onClick={() => onNavigate("analytics")}
                className="px-2.5 py-1 rounded-full border border-zinc-200 dark:border-zinc-700 text-[11px] font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-zinc-50"
              >
                + Details
              </button>
            </div>

            <div className="my-3 space-y-3.5 text-xs">
              {[
                {
                  title: "Keybody R-MPKPE",
                  sub: "Target < 70.0 mm",
                  val: `${rmpkpeMm.toFixed(1)} / 70 mm`,
                  pct: 88,
                  color: "bg-[#84cc16]",
                  icon: "🎯",
                },
                {
                  title: "Joint Tracking MAE",
                  sub: "Target < 0.20 rad",
                  val: `${jointMaeRad.toFixed(3)} / 0.20 rad`,
                  pct: 82,
                  color: "bg-[#84cc16]",
                  icon: "🦾",
                },
                {
                  title: "Hand End-Effector",
                  sub: "Relative wrist error",
                  val: "33.2 / 60 mm",
                  pct: 90,
                  color: "bg-[#fb923c]",
                  icon: "✋",
                },
                {
                  title: "Episode Completion",
                  sub: "Target ≥ 85.0%",
                  val: `${successRate.toFixed(1)}% / 85%`,
                  pct: Math.round((successRate / 85) * 100),
                  color: "bg-[#fb923c]",
                  icon: "⚡",
                },
              ].map((g, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-base shrink-0">
                    {g.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-zinc-900 dark:text-white truncate">
                        {g.title}
                      </span>
                      <span className="font-mono text-[11px] text-zinc-500">{g.val}</span>
                    </div>
                    <div className="mt-1 w-full h-2 rounded-full bg-zinc-100 dark:bg-zinc-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${g.color}`}
                        style={{ width: `${Math.min(100, g.pct)}%` }}
                      />
                    </div>
                    <div className="text-[10px] text-zinc-400 mt-0.5">{g.sub}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT SIDEBAR COLUMN (4 Columns on XL — Active Policy Card, Quick Actions, Camera POV Avatars, Telemetry History) */}
      <div className="xl:col-span-4 flex flex-col gap-5">
        {/* Top-Right Card: Active Robot & Policy Card (styled like "My card" with the Lime-Green Gradient Visa card!) */}
        <div className="bento-card p-5 space-y-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                Active Policy Card
              </h3>
              <p className="text-[11px] text-zinc-400">Quick simulation controls</p>
            </div>
            <button
              onClick={() => onNavigate("checkpoints")}
              className="px-3 py-1.5 rounded-full border border-zinc-200 dark:border-zinc-700 text-xs font-semibold text-zinc-800 dark:text-zinc-200 hover:bg-zinc-50 transition"
            >
              + Switch policy
            </button>
          </div>

          {/* Horizontal Carousel of 2 Cards: Primary Lime Gradient Card + Secondary Charcoal ONNX Card */}
          <div className="grid grid-cols-12 gap-3">
            <div
              onClick={() => onNavigate("simulation")}
              className="col-span-9 cursor-pointer rounded-2xl p-4 bg-gradient-to-br from-[#65a30d] via-[#84cc16] to-[#bef264] text-white shadow-md flex flex-col justify-between h-40 relative overflow-hidden group"
            >
              <div className="flex items-center justify-between z-10">
                <span className="text-xs font-medium text-white/90">
                  BeyondMimic PPO
                </span>
                <span className="font-black tracking-wider text-sm">K1•22DOF</span>
              </div>

              <div className="z-10 my-auto">
                <div className="text-[11px] text-white/80 font-mono">
                  ACTIVE CHECKPOINT
                </div>
                <div className="text-sm font-mono font-bold tracking-wide truncate mt-0.5">
                  {bestCkpt?.filename || "k1_kick_policy.pt"}
                </div>
              </div>

              <div className="flex items-end justify-between z-10 text-xs">
                <div>
                  <div className="text-[10px] text-white/75">Motion Reference</div>
                  <div className="font-semibold">k1_kick.npz (149f)</div>
                </div>
                <div className="font-mono font-bold">50 Hz</div>
              </div>
            </div>

            {/* Secondary peek card on the right (like "Credit card" in reference screenshot) */}
            <div
              onClick={() => onNavigate("checkpoints")}
              className="col-span-3 cursor-pointer rounded-2xl p-3 bg-zinc-100 dark:bg-zinc-800 border border-zinc-200/80 dark:border-zinc-700 flex flex-col justify-between h-40 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200/70 transition"
            >
              <div className="text-[11px] font-semibold">ONNX</div>
              <div className="text-[10px] font-mono text-zinc-400 break-all">
                .onnx
              </div>
              <div className="text-[10px] font-mono font-bold">2495</div>
            </div>
          </div>

          {/* 5 Square Quick Action Buttons (styled like "Top up | Send | Request | History | More") */}
          <div className="grid grid-cols-5 gap-2 pt-1">
            {[
              { label: "3D Sim", icon: Box, action: () => onNavigate("simulation") },
              { label: "Train", icon: Cpu, action: () => onNavigate("training") },
              { label: "Motion", icon: Activity, action: () => onNavigate("motion") },
              { label: "Video", icon: Video, action: () => onNavigate("videos") },
              {
                label: "Discard",
                icon: RotateCcw,
                action: () => onDiscardSim?.(),
              },
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

          {/* 6 Camera POV Circular Quick Launchers (styled like "Quick payment" circular avatars!) */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-xs font-bold text-zinc-800 dark:text-zinc-200">
                Quick Camera POV Launch
              </span>
              <MoreHorizontal className="w-4 h-4 text-zinc-400" />
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

        {/* Bottom-Right Card: Experiment & Telemetry History (styled like "Transaction history") */}
        <div className="bento-card p-5 flex-1 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                Training Session History
              </h3>
              <span className="px-2.5 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
                5 Stages
              </span>
            </div>

            <div className="flex items-center justify-between text-[11px] text-zinc-400 pb-2 border-b border-zinc-100 dark:border-zinc-800">
              <span>↑↓ Session / Run ID</span>
              <span>Return / Error</span>
            </div>

            <div className="divide-y divide-zinc-100 dark:divide-zinc-800/70 mt-1">
              {experiments.map((exp: any, idx: number) => {
                const badgeColors = [
                  "bg-lime-500 text-white",
                  "bg-zinc-900 text-white",
                  "bg-amber-500 text-white",
                  "bg-emerald-600 text-white",
                  "bg-lime-600 text-white",
                ];
                return (
                  <div
                    key={exp.id}
                    onClick={() => onNavigate("experiments")}
                    className="py-3 flex items-center justify-between gap-3 cursor-pointer hover:bg-zinc-50/80 dark:hover:bg-zinc-800/40 px-1.5 rounded-xl transition"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <div
                        className={`w-9 h-9 rounded-xl font-mono font-bold text-xs flex items-center justify-center shrink-0 ${
                          badgeColors[idx % badgeColors.length]
                        }`}
                      >
                        S{idx + 1}
                      </div>
                      <div className="min-w-0">
                        <div className="text-xs font-bold text-zinc-900 dark:text-white truncate">
                          {exp.name}
                        </div>
                        <div className="text-[11px] text-zinc-400 font-mono truncate">
                          Iter {exp.start_step}→{exp.final_step} • R-MPKPE {exp.r_mpkpe_mm?.toFixed(1)}mm
                        </div>
                      </div>
                    </div>

                    <div className="text-right shrink-0">
                      <div className="text-xs font-extrabold font-mono text-zinc-900 dark:text-white">
                        +{exp.mean_reward?.toFixed(2)}
                      </div>
                      <div className="text-[10px] font-medium text-lime-600 dark:text-lime-400 capitalize">
                        {exp.status}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Latest Live Event Pill */}
          {activities[0] && (
            <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800 flex items-center justify-between text-[11px] text-zinc-400">
              <span className="truncate">{activities[0].action}</span>
              <span className="text-lime-600 dark:text-lime-400 font-semibold shrink-0 ml-2">
                Live RAM
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
