"use client";

import React, { useEffect, useState } from "react";
import RobotCanvas3D from "@/components/RobotCanvas3D";
import { apiFetch, CameraPOV, TrajectoryBundle } from "@/lib/api";
import {
  Camera,
  Play,
  Pause,
  RotateCcw,
  Activity,
  Gauge,
  Monitor,
  RefreshCw,
} from "lucide-react";

interface DetectedMotion {
  name: string;
  path: string;
  size_kb?: number;
  has_rollout?: boolean;
  matched_policy?: string | null;
}

interface SimulationViewerTabProps {
  trajectory: TrajectoryBundle | null;
  availableMotions?: DetectedMotion[];
  checkpoints: any[];
  selectedMotion?: string;
  onSelectMotion?: (motionPath: string) => void;
  selectedCheckpoint: string;
  onSelectCheckpoint: (path: string) => void;
  onRefreshAssets?: () => void;
  initialCameraPov?: CameraPOV;
}

const CAMERA_POVS: { id: CameraPOV; label: string; desc: string }[] = [
  { id: "third_person", label: "1. Third-Person", desc: "Follow humanoid from behind" },
  { id: "front", label: "2. Front View", desc: "View humanoid from the front" },
  { id: "side", label: "3. Side View", desc: "Lateral profile perspective" },
  { id: "top", label: "4. Top View", desc: "Overhead bird's-eye view" },
  { id: "free", label: "5. Free Orbit", desc: "Interactive rotate, zoom & pan" },
  { id: "first_person", label: "6. First-Person (Head POV)", desc: "Robot Head_2 stereo eye perspective" },
];

export default function SimulationViewerTab({
  trajectory,
  availableMotions = [],
  checkpoints,
  selectedMotion = "",
  onSelectMotion,
  selectedCheckpoint,
  onSelectCheckpoint,
  onRefreshAssets,
  initialCameraPov = "third_person",
}: SimulationViewerTabProps) {
  const [isPlaying, setIsPlaying] = useState(true);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [cameraPov, setCameraPov] = useState<CameraPOV>(initialCameraPov);
  const [viewMode, setViewMode] = useState<"side_by_side" | "overlay" | "policy_only" | "reference_only">("side_by_side");
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [selectedBodyIdx, setSelectedBodyIdx] = useState<number | null>(null);
  const [nativeMujocoImg, setNativeMujocoImg] = useState<string | null>(null);
  const [loadingNative, setLoadingNative] = useState(false);
  const [showNativePip, setShowNativePip] = useState(false);

  useEffect(() => {
    if (initialCameraPov) {
      setCameraPov(initialCameraPov);
    }
  }, [initialCameraPov]);

  const frames = trajectory?.frames || [];
  const totalFrames = trajectory?.frame_count || frames.length || 1;
  const currentFrameData = frames[Math.min(frameIndex, Math.max(0, frames.length - 1))] || null;
  const jointSummaries = trajectory?.per_joint_summary || [];

  const detectedMotions: DetectedMotion[] = availableMotions.length
    ? availableMotions
    : ((trajectory as any)?.available_motions || []);

  useEffect(() => {
    setFrameIndex(0);
  }, [trajectory?.motion_file, frames.length]);

  useEffect(() => {
    if (!isPlaying || frames.length === 0) return;
    const intervalMs = Math.max(12, Math.round(20 / playbackSpeed));
    const timer = setInterval(() => {
      setFrameIndex((prev) => (prev + 1) % frames.length);
    }, intervalMs);
    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, frames.length]);

  const fetchNativeFrame = async (fIdx: number, pov: CameraPOV) => {
    setLoadingNative(true);
    try {
      const modeParam =
        viewMode === "reference_only"
          ? "reference"
          : viewMode === "side_by_side"
          ? "side_by_side"
          : "policy";
      const res = await apiFetch(
        `/api/simulation/render-frame?frame=${fIdx}&camera_pov=${pov}&mode=${modeParam}&motion_file=${encodeURIComponent(
          selectedMotion
        )}&checkpoint_file=${encodeURIComponent(
          selectedCheckpoint
        )}&width=560&height=360`
      );
      if (res.image_data_url) {
        setNativeMujocoImg(res.image_data_url);
      }
    } catch {
      // ignore
    } finally {
      setLoadingNative(false);
    }
  };

  useEffect(() => {
    if (showNativePip) {
      fetchNativeFrame(frameIndex, cameraPov);
    }
  }, [showNativePip, cameraPov, viewMode, selectedCheckpoint]);

  return (
    <div className="space-y-5">
      {/* Top Control Bento Bar */}
      <div className="bento-card p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-bold text-zinc-700 dark:text-zinc-300 mr-1.5 flex items-center gap-1">
            <Camera className="w-3.5 h-3.5 text-lime-600" /> CAMERA POV:
          </span>
          {CAMERA_POVS.map((pov) => (
            <button
              key={pov.id}
              onClick={() => setCameraPov(pov.id)}
              title={pov.desc}
              className={`px-3 py-1.5 rounded-xl text-xs font-medium transition border ${
                cameraPov === pov.id
                  ? "bg-zinc-900 dark:bg-lime-400 border-zinc-900 dark:border-lime-400 text-white dark:text-zinc-950 font-semibold shadow-sm"
                  : "bg-zinc-100 dark:bg-zinc-800 border-zinc-200/80 dark:border-zinc-700 text-zinc-600 dark:text-zinc-300 hover:text-zinc-900"
              }`}
            >
              {pov.label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <div className="flex items-center bg-zinc-100 dark:bg-zinc-800 border border-zinc-200/80 dark:border-zinc-700 rounded-xl p-0.5">
            {[
              { id: "side_by_side", label: "Side-by-Side" },
              { id: "overlay", label: "Ghost Overlay" },
              { id: "policy_only", label: "Policy Only" },
              { id: "reference_only", label: "Reference Only" },
            ].map((m) => (
              <button
                key={m.id}
                onClick={() => setViewMode(m.id as any)}
                className={`px-2.5 py-1 rounded-lg font-medium transition ${
                  viewMode === m.id
                    ? "bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white shadow-sm font-semibold"
                    : "text-zinc-500 hover:text-zinc-900"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`px-3 py-1.5 rounded-xl font-medium border transition ${
              showHeatmap
                ? "bg-lime-100 dark:bg-lime-950 border-lime-400 text-lime-800 dark:text-lime-300 font-semibold"
                : "bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-600"
            }`}
          >
            Error Heatmap
          </button>

          <button
            onClick={() => setShowSkeleton(!showSkeleton)}
            className={`px-3 py-1.5 rounded-xl font-medium border transition ${
              showSkeleton
                ? "bg-orange-100 dark:bg-orange-950 border-orange-400 text-orange-800 dark:text-orange-300 font-semibold"
                : "bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-600"
            }`}
          >
            Kinematic Links
          </button>

          <button
            onClick={() => setShowNativePip(!showNativePip)}
            className={`px-3 py-1.5 rounded-xl font-medium border flex items-center gap-1 transition ${
              showNativePip
                ? "bg-zinc-900 text-white border-zinc-900"
                : "bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-600"
            }`}
          >
            <Monitor className="w-3.5 h-3.5" /> MuJoCo Native Stream
          </button>
        </div>
      </div>

      {/* Main Viewport + Live Joint Telemetry Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-8 space-y-4">
          <div className="relative">
            <RobotCanvas3D
              frameData={currentFrameData}
              cameraPov={cameraPov}
              viewMode={viewMode}
              showHeatmap={showHeatmap}
              showSkeleton={showSkeleton}
              selectedBodyIndex={selectedBodyIdx}
              onSelectBody={(idx) => setSelectedBodyIdx(idx)}
              heightClass="h-[540px]"
            />

            {showNativePip && (
              <div className="absolute top-14 right-3 w-72 bg-white/95 dark:bg-zinc-900/95 border border-zinc-200 dark:border-zinc-700 rounded-2xl overflow-hidden shadow-2xl">
                <div className="px-3 py-2 bg-zinc-100 dark:bg-zinc-800 border-b border-zinc-200 dark:border-zinc-700 flex items-center justify-between text-[11px] font-mono">
                  <span className="text-zinc-900 dark:text-white font-bold">
                    MuJoCo Native ({cameraPov})
                  </span>
                  <button
                    onClick={() => fetchNativeFrame(frameIndex, cameraPov)}
                    className="text-lime-600 font-semibold hover:underline"
                  >
                    {loadingNative ? "Rendering..." : "Refresh"}
                  </button>
                </div>
                {nativeMujocoImg ? (
                  <img
                    src={nativeMujocoImg}
                    alt="MuJoCo Offscreen Render"
                    className="w-full h-44 object-cover"
                  />
                ) : (
                  <div className="h-44 flex items-center justify-center text-xs font-mono text-zinc-500">
                    Rendering MuJoCo frame...
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Playback Controls & Timeline Scrubber Bar */}
          <div className="bento-card p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="px-4 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 dark:bg-lime-400 dark:text-zinc-950 text-white text-xs font-semibold flex items-center gap-1.5 transition"
                >
                  {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {isPlaying ? "Pause" : "Play / Resume"}
                </button>
                <button
                  onClick={async () => {
                    setIsPlaying(false);
                    setFrameIndex(0);
                    try {
                      await apiFetch("/api/simulation/discard", { method: "POST" });
                    } catch {
                      // ignore
                    }
                  }}
                  className="px-3 py-2 rounded-xl bg-orange-100 hover:bg-orange-200 dark:bg-orange-950 text-orange-800 dark:text-orange-300 text-xs font-semibold flex items-center gap-1 transition"
                  title="Hentikan dan langsung buang buffer simulasi dari memori (Tanpa Database)"
                >
                  <RotateCcw className="w-3.5 h-3.5" /> Reset & Discard Buffer
                </button>

                <div className="flex items-center gap-1 ml-2 bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2 py-1 text-xs font-mono">
                  <Gauge className="w-3.5 h-3.5 text-lime-600 mr-1" />
                  {[0.25, 0.5, 1.0, 1.5, 2.0].map((spd) => (
                    <button
                      key={spd}
                      onClick={() => setPlaybackSpeed(spd)}
                      className={`px-2 py-0.5 rounded-lg ${
                        playbackSpeed === spd
                          ? "bg-zinc-900 text-white dark:bg-lime-400 dark:text-zinc-950 font-bold"
                          : "text-zinc-500 hover:text-zinc-900"
                      }`}
                    >
                      {spd}x
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
                <span className="text-zinc-500">Motion ({detectedMotions.length}):</span>
                <select
                  value={selectedMotion}
                  onChange={(e) => onSelectMotion?.(e.target.value)}
                  className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2.5 py-1.5 text-zinc-800 dark:text-zinc-200"
                >
                  {detectedMotions.length === 0 ? (
                    <option value="">Scanning data/motions/*.npz...</option>
                  ) : (
                    detectedMotions.map((m) => (
                      <option key={m.path} value={m.path}>
                        {m.name} {m.has_rollout ? "• Rollout Ready" : "• 50Hz Ref"}
                      </option>
                    ))
                  )}
                </select>
                <span className="text-zinc-500 ml-1">Policy ({checkpoints.length}):</span>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => onSelectCheckpoint(e.target.value)}
                  className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2.5 py-1.5 text-zinc-800 dark:text-zinc-200 max-w-xs"
                >
                  {checkpoints.length === 0 ? (
                    <option value="">No .pt policy detected</option>
                  ) : (
                    checkpoints.map((c: any) => (
                      <option key={c.file_path} value={c.file_path}>
                        {c.is_primary_output
                          ? `[Output] ${c.filename} (Iter ${c.iteration})`
                          : `${c.experiment_run_id || c.name} / ${c.filename}`}
                      </option>
                    ))
                  )}
                </select>
                {onRefreshAssets && (
                  <button
                    onClick={onRefreshAssets}
                    title="Scan ulang Motion (.npz) & Policy (.pt) dari folder human-imitation-learning"
                    className="p-1.5 rounded-xl bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-800 dark:hover:bg-zinc-700 border border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-300 transition"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-zinc-500 w-24">
                Frame {frameIndex + 1}/{totalFrames}
              </span>
              <input
                type="range"
                min={0}
                max={totalFrames - 1}
                value={frameIndex}
                onChange={(e) => {
                  setIsPlaying(false);
                  setFrameIndex(Number(e.target.value));
                }}
                className="flex-1 accent-lime-500 cursor-pointer"
              />
              <span className="text-xs font-mono font-bold text-lime-600 dark:text-lime-400 w-24 text-right">
                {(currentFrameData?.time_sec ?? 0).toFixed(2)}s / {(trajectory?.duration_sec ?? 2.98).toFixed(2)}s
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Live 22-DoF Joint Telemetry & Pose Inspector */}
        <div className="lg:col-span-4 bento-card p-5 flex flex-col h-[650px]">
          <div className="border-b border-zinc-100 dark:border-zinc-800 pb-3 mb-3 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-2">
                <Activity className="w-4 h-4 text-lime-600" />
                Live 22-DoF Joint State
              </h3>
              <p className="text-[11px] text-zinc-400 font-mono">
                Ref vs. Policy Angle (rad) @ t={currentFrameData?.time_sec.toFixed(2)}s
              </p>
            </div>
            <span className="px-2.5 py-0.5 rounded-full bg-lime-100 dark:bg-lime-950 text-lime-700 dark:text-lime-300 text-[11px] font-semibold">
              50 Hz
            </span>
          </div>

          <div className="flex-1 overflow-y-auto pr-1">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead className="sticky top-0 bg-white dark:bg-zinc-900 border-b border-zinc-100 dark:border-zinc-800 text-zinc-400 text-[11px]">
                <tr>
                  <th className="py-1.5 px-2">Joint (22-DoF)</th>
                  <th className="py-1.5 px-1.5 text-right">Ref</th>
                  <th className="py-1.5 px-1.5 text-right">Sim</th>
                  <th className="py-1.5 px-1.5 text-right">|Err|</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
                {jointSummaries.map((j) => {
                  const refVal = currentFrameData?.ref_joint_pos?.[j.index] ?? 0;
                  const simVal = currentFrameData?.sim_joint_pos?.[j.index] ?? 0;
                  const errVal = currentFrameData?.joint_error_rad?.[j.index] ?? 0;
                  const isHigh = errVal > 0.25;
                  const isMod = errVal > 0.14;
                  return (
                    <tr key={j.name} className="hover:bg-zinc-50 dark:hover:bg-zinc-800/40">
                      <td className="py-1.5 px-2 text-zinc-800 dark:text-zinc-200 font-medium truncate max-w-[135px]" title={j.name}>
                        {j.name}
                      </td>
                      <td className="py-1.5 px-1.5 text-right text-zinc-500">
                        {refVal >= 0 ? `+${refVal.toFixed(2)}` : refVal.toFixed(2)}
                      </td>
                      <td className="py-1.5 px-1.5 text-right text-zinc-900 dark:text-white font-semibold">
                        {simVal >= 0 ? `+${simVal.toFixed(2)}` : simVal.toFixed(2)}
                      </td>
                      <td className="py-1.5 px-1.5 text-right">
                        <span
                          className={`px-1.5 py-0.5 rounded-md text-[11px] ${
                            isHigh
                              ? "bg-orange-100 dark:bg-orange-950 text-orange-700 dark:text-orange-300 font-bold"
                              : isMod
                              ? "bg-amber-100/70 dark:bg-amber-950 text-amber-700 dark:text-amber-300"
                              : "text-lime-600 dark:text-lime-400 font-semibold"
                          }`}
                        >
                          {errVal.toFixed(3)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {currentFrameData && (
            <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800 grid grid-cols-2 gap-2 text-[11px] font-mono">
              <div className="p-2.5 rounded-xl bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200/60 dark:border-zinc-700">
                <div className="text-zinc-400 text-[10px]">TRUNK ANCHOR POS</div>
                <div className="text-zinc-900 dark:text-white font-bold mt-0.5">
                  [{currentFrameData.sim_body_pos[0].map((v) => v.toFixed(2)).join(", ")}]m
                </div>
              </div>
              <div className="p-2.5 rounded-xl bg-lime-50/70 dark:bg-lime-950/30 border border-lime-200 dark:border-lime-800/50">
                <div className="text-lime-700 dark:text-lime-400 text-[10px]">HEAD_2 FPV EYE</div>
                <div className="text-zinc-900 dark:text-white font-bold mt-0.5">
                  [{currentFrameData.head_camera_pose.pos.map((v) => v.toFixed(2)).join(", ")}]m
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
