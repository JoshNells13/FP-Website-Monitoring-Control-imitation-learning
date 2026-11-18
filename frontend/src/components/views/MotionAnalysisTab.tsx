"use client";

import React, { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import { TrajectoryBundle } from "@/lib/api";
import { AlertTriangle } from "lucide-react";

interface MotionAnalysisTabProps {
  trajectory: TrajectoryBundle | null;
}

export default function MotionAnalysisTab({ trajectory }: MotionAnalysisTabProps) {
  const [selectedJointIndex, setSelectedJointIndex] = useState<number>(16);
  const [compareJointIndex, setCompareJointIndex] = useState<number>(19);
  const [currentFrame, setCurrentFrame] = useState<number>(68);
  const [filterSeverity, setFilterSeverity] = useState<"all" | "high" | "moderate" | "optimal">("all");

  const frames = trajectory?.frames || [];
  const jointSummaries = trajectory?.per_joint_summary || [];
  const selectedJointMeta = jointSummaries[selectedJointIndex] || jointSummaries[0];
  const compareJointMeta = jointSummaries[compareJointIndex] || jointSummaries[1];

  const jointTimeSeries = useMemo(() => {
    return frames.map((f) => {
      const refAngle = f.ref_joint_pos[selectedJointIndex] ?? 0;
      const simAngle = f.sim_joint_pos[selectedJointIndex] ?? 0;
      const refVel = f.ref_joint_vel[selectedJointIndex] ?? 0;
      const simVel = f.sim_joint_vel[selectedJointIndex] ?? 0;
      const errRad = f.joint_error_rad[selectedJointIndex] ?? 0;

      const compRef = f.ref_joint_pos[compareJointIndex] ?? 0;
      const compSim = f.sim_joint_pos[compareJointIndex] ?? 0;

      const refRightFootZ = f.ref_body_pos?.[22]?.[2] ?? 0;
      const simRightFootZ = f.sim_body_pos?.[22]?.[2] ?? 0;

      return {
        frame: f.frame,
        time_sec: f.time_sec,
        ref_angle_rad: refAngle,
        sim_angle_rad: simAngle,
        error_rad: errRad,
        ref_vel: refVel,
        sim_vel: simVel,
        comp_ref_rad: compRef,
        comp_sim_rad: compSim,
        ref_foot_z_m: refRightFootZ,
        sim_foot_z_m: simRightFootZ,
      };
    });
  }, [frames, selectedJointIndex, compareJointIndex]);

  const filteredJoints = useMemo(() => {
    if (filterSeverity === "all") return jointSummaries;
    return jointSummaries.filter((j) => j.severity === filterSeverity);
  }, [jointSummaries, filterSeverity]);

  const highErrorJoints = useMemo(
    () => jointSummaries.filter((j) => j.severity === "high"),
    [jointSummaries]
  );

  return (
    <div className="space-y-5">
      {/* Top KPI Bento Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bento-card p-4">
          <span className="text-xs text-zinc-400 font-medium">Reference Motion</span>
          <div className="mt-1 font-mono text-sm font-bold text-zinc-900 dark:text-white truncate">
            {trajectory?.motion_file || "data/motions/k1_kick.npz"}
          </div>
          <div className="mt-1 text-[11px] text-zinc-400">Martial-Arts Front Kick (GMR → MJLab)</div>
        </div>

        <div className="bento-card p-4">
          <span className="text-xs text-zinc-400 font-medium">Duration & Frames</span>
          <div className="mt-1 font-mono text-lg font-extrabold text-zinc-900 dark:text-white">
            {trajectory?.duration_sec ?? 2.98}s ({trajectory?.frame_count ?? 149}f)
          </div>
          <div className="mt-1 text-[11px] font-semibold text-lime-600">
            {trajectory?.fps ?? 50} FPS Control (dt = 0.02s)
          </div>
        </div>

        <div className="bento-card p-4">
          <span className="text-xs text-zinc-400 font-medium">Mean Joint MAE</span>
          <div className="mt-1 font-mono text-lg font-extrabold text-lime-600 dark:text-lime-400">
            {trajectory?.overall_metrics?.joint_mae_rad ?? 0.147} rad (
            {trajectory?.overall_metrics?.joint_mae_deg ?? 8.42}°)
          </div>
          <div className="mt-1 text-[11px] font-mono text-zinc-400">
            RMSE: {trajectory?.overall_metrics?.joint_rmse_rad ?? 0.217} rad
          </div>
        </div>

        <div className="bento-card p-4">
          <span className="text-xs text-zinc-400 font-medium">Keybody R-MPKPE</span>
          <div className="mt-1 font-mono text-lg font-extrabold text-zinc-900 dark:text-white">
            {trajectory?.overall_metrics?.r_mpkpe_mm ?? 39.38} mm
          </div>
          <div className="mt-1 text-[11px] font-mono text-zinc-400">
            World MPKPE: {trajectory?.overall_metrics?.mpkpe_mm ?? 108.34} mm
          </div>
        </div>

        <div className="bento-card p-4 border-orange-300/70">
          <span className="text-xs text-orange-600 font-semibold flex items-center gap-1">
            <AlertTriangle className="w-3.5 h-3.5" /> High-Error Joints (&ge;0.20 rad)
          </span>
          <div className="mt-1 font-mono text-lg font-extrabold text-orange-500">
            {highErrorJoints.length} / 22 Joints
          </div>
          <div className="mt-1 text-[11px] font-mono text-zinc-400 truncate">
            {highErrorJoints.map((j) => j.name).join(", ")}
          </div>
        </div>
      </div>

      {/* Interactive Joint Selector & Frame Scrubber */}
      <div className="bento-card p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
          <div>
            <label className="text-zinc-500 mr-2 font-sans font-semibold">Primary Joint:</label>
            <select
              value={selectedJointIndex}
              onChange={(e) => setSelectedJointIndex(Number(e.target.value))}
              className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-1.5 text-zinc-900 dark:text-white font-semibold"
            >
              {jointSummaries.map((j) => (
                <option key={j.index} value={j.index}>
                  [{j.index.toString().padStart(2, "0")}] {j.name} (MAE: {j.mae_rad.toFixed(3)} rad /{" "}
                  {j.mae_deg}°)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-zinc-500 mr-2 font-sans font-semibold">Comparison Joint:</label>
            <select
              value={compareJointIndex}
              onChange={(e) => setCompareJointIndex(Number(e.target.value))}
              className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-1.5 text-zinc-800 dark:text-zinc-200"
            >
              {jointSummaries.map((j) => (
                <option key={j.index} value={j.index}>
                  [{j.index.toString().padStart(2, "0")}] {j.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono flex-1 max-w-md">
          <span className="text-zinc-500 whitespace-nowrap">
            Frame {currentFrame} ({(currentFrame * 0.02).toFixed(2)}s):
          </span>
          <input
            type="range"
            min={0}
            max={Math.max(1, frames.length - 1)}
            value={currentFrame}
            onChange={(e) => setCurrentFrame(Number(e.target.value))}
            className="flex-1 accent-lime-500"
          />
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="bento-card p-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                Joint Angle Trajectory: {selectedJointMeta?.name} (rad)
              </h3>
              <p className="text-xs text-zinc-400 font-mono">
                MAE: {selectedJointMeta?.mae_rad} rad ({selectedJointMeta?.mae_deg}°) | RMSE:{" "}
                {selectedJointMeta?.rmse_rad} rad
              </p>
            </div>
            <span
              className={`px-2.5 py-0.5 rounded-full text-xs font-semibold capitalize ${
                selectedJointMeta?.severity === "high"
                  ? "bg-orange-100 text-orange-700"
                  : selectedJointMeta?.severity === "moderate"
                  ? "bg-amber-100 text-amber-700"
                  : "bg-lime-100 text-lime-700"
              }`}
            >
              {selectedJointMeta?.severity} error
            </span>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={jointTimeSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                <XAxis dataKey="time_sec" stroke="#71717a" fontSize={11} unit="s" />
                <YAxis stroke="#71717a" fontSize={11} unit=" rad" />
                <Tooltip
                  contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e4e4e7", borderRadius: "12px", fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <ReferenceLine x={Number((currentFrame * 0.02).toFixed(3))} stroke="#18181b" strokeDasharray="3 3" />
                <Line
                  type="monotone"
                  dataKey="ref_angle_rad"
                  name="GMR Reference (rad)"
                  stroke="#71717a"
                  strokeDasharray="5 5"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="sim_angle_rad"
                  name="BeyondMimic Policy (rad)"
                  stroke="#84cc16"
                  dot={false}
                  strokeWidth={2.5}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bento-card p-5">
          <div className="mb-3">
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
              Instantaneous Joint Tracking Error: |q_sim(t) - q_ref(t)|
            </h3>
            <p className="text-xs text-zinc-400 font-mono">
              Highlights peak deviation phases during dynamic single-leg kick apex
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={jointTimeSeries}>
                <defs>
                  <linearGradient id="errOrangeGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#fb923c" stopOpacity={0.38} />
                    <stop offset="95%" stopColor="#fb923c" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                <XAxis dataKey="time_sec" stroke="#71717a" fontSize={11} unit="s" />
                <YAxis stroke="#fb923c" fontSize={11} unit=" rad" />
                <Tooltip
                  contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e4e4e7", borderRadius: "12px", fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <ReferenceLine y={0.2} stroke="#84cc16" strokeDasharray="4 4" />
                <Area
                  type="monotone"
                  dataKey="error_rad"
                  name={`${selectedJointMeta?.name} Error (rad)`}
                  stroke="#fb923c"
                  fill="url(#errOrangeGrad)"
                  strokeWidth={2.2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bento-card p-5">
          <div className="mb-3">
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
              Joint Angular Velocity: {selectedJointMeta?.name} (rad/s)
            </h3>
            <p className="text-xs text-zinc-400 font-mono">
              Velocity RMSE: {selectedJointMeta?.vel_rmse_rad_s} rad/s | MAE:{" "}
              {selectedJointMeta?.vel_mae_rad_s} rad/s
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={jointTimeSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                <XAxis dataKey="time_sec" stroke="#71717a" fontSize={11} unit="s" />
                <YAxis stroke="#71717a" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e4e4e7", borderRadius: "12px", fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="ref_vel"
                  name="Reference Velocity (rad/s)"
                  stroke="#a1a1aa"
                  strokeDasharray="4 4"
                  dot={false}
                  strokeWidth={1.6}
                />
                <Line
                  type="monotone"
                  dataKey="sim_vel"
                  name="Simulated Velocity (rad/s)"
                  stroke="#18181b"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bento-card p-5">
          <div className="mb-3">
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
              3D End-Effector Kick Elevation (right_foot_link Z) & {compareJointMeta?.name}
            </h3>
            <p className="text-xs text-zinc-400 font-mono">
              World-frame foot height Z (m) + secondary joint angle comparison
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={jointTimeSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                <XAxis dataKey="time_sec" stroke="#71717a" fontSize={11} unit="s" />
                <YAxis yAxisId="left" stroke="#84cc16" fontSize={11} unit="m" />
                <YAxis yAxisId="right" orientation="right" stroke="#fb923c" fontSize={11} unit="rad" />
                <Tooltip
                  contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e4e4e7", borderRadius: "12px", fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="ref_foot_z_m"
                  name="Ref Right Foot Height Z (m)"
                  stroke="#71717a"
                  strokeDasharray="4 4"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="sim_foot_z_m"
                  name="Policy Right Foot Height Z (m)"
                  stroke="#84cc16"
                  dot={false}
                  strokeWidth={2.5}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="comp_sim_rad"
                  name={`${compareJointMeta?.name} (rad)`}
                  stroke="#fb923c"
                  dot={false}
                  strokeWidth={1.8}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Complete 22-DoF Per-Joint Error Table */}
      <div className="bento-card p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
              Complete 22-DoF Per-Joint Tracking Performance
            </h3>
            <p className="text-xs text-zinc-400">
              Click any joint row to load its trajectory curves into the interactive analyzers above
            </p>
          </div>
          <div className="flex items-center gap-1.5 text-xs font-medium">
            {(["all", "high", "moderate", "optimal"] as const).map((sev) => (
              <button
                key={sev}
                onClick={() => setFilterSeverity(sev)}
                className={`px-3 py-1 rounded-xl border capitalize transition ${
                  filterSeverity === sev
                    ? "bg-zinc-900 text-white border-zinc-900 font-semibold"
                    : "bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-600"
                }`}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-zinc-100 dark:border-zinc-800 text-zinc-400">
                <th className="py-2.5 px-3">Idx</th>
                <th className="py-2.5 px-3">Joint Name</th>
                <th className="py-2.5 px-3">Group</th>
                <th className="py-2.5 px-3">MAE (rad)</th>
                <th className="py-2.5 px-3">MAE (deg)</th>
                <th className="py-2.5 px-3">RMSE (rad)</th>
                <th className="py-2.5 px-3">Max Error</th>
                <th className="py-2.5 px-3">Vel RMSE</th>
                <th className="py-2.5 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
              {filteredJoints.map((j) => {
                const isSelected = j.index === selectedJointIndex;
                return (
                  <tr
                    key={j.index}
                    onClick={() => setSelectedJointIndex(j.index)}
                    className={`cursor-pointer transition ${
                      isSelected
                        ? "bg-lime-50 dark:bg-lime-950/30"
                        : "hover:bg-zinc-50 dark:hover:bg-zinc-800/40"
                    }`}
                  >
                    <td className="py-2.5 px-3 text-zinc-400">#{j.index}</td>
                    <td className="py-2.5 px-3 font-bold text-zinc-900 dark:text-white">{j.name}</td>
                    <td className="py-2.5 px-3 text-zinc-500 font-sans">{j.group}</td>
                    <td className="py-2.5 px-3 text-zinc-900 dark:text-white font-semibold">{j.mae_rad.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-zinc-600 dark:text-zinc-300">{j.mae_deg.toFixed(2)}°</td>
                    <td className="py-2.5 px-3 text-zinc-600 dark:text-zinc-300">{j.rmse_rad.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-orange-600">{j.max_error_rad.toFixed(3)} rad</td>
                    <td className="py-2.5 px-3 text-zinc-600 dark:text-zinc-300">{j.vel_rmse_rad_s.toFixed(3)} rad/s</td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[11px] font-sans font-semibold capitalize ${
                          j.severity === "high"
                            ? "bg-orange-100 text-orange-700"
                            : j.severity === "moderate"
                            ? "bg-amber-100 text-amber-700"
                            : "bg-lime-100 text-lime-700"
                        }`}
                      >
                        {j.severity === "high" ? "Significant Error" : j.severity}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
