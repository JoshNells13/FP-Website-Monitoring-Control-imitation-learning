"use client";

import React, { useState } from "react";
import { API_BASE, apiFetch, CameraPOV } from "@/lib/api";
import {
  Download,
  Film,
  Sparkles,
  Loader2,
} from "lucide-react";

interface VideoGenerationTabProps {
  videos: any[];
  checkpoints: any[];
  onRefresh: () => void;
}

export default function VideoGenerationTab({
  videos,
  checkpoints,
  onRefresh,
}: VideoGenerationTabProps) {
  const [selectedVideo, setSelectedVideo] = useState<any>(() => videos[0] || null);
  const [form, setForm] = useState({
    title: "",
    video_type: "comparison",
    camera_pov: "third_person" as CameraPOV,
    resolution: "960x720",
    fps: 50,
    playback_speed: 1.0,
    include_overlay: true,
    motion_file: "data/motions/k1_kick.npz",
    checkpoint_path: "outputs/checkpoints/k1_kick_policy.pt",
  });
  const [submitting, setSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  const activeVideo =
    videos.find((v) => v.id === selectedVideo?.id) || videos[0] || null;

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setFeedback(null);
    try {
      const res = await apiFetch("/api/videos/generate", {
        method: "POST",
        body: JSON.stringify(form),
      });
      if (res.video) {
        setSelectedVideo(res.video);
      }
      setFeedback("Background MuJoCo + FFmpeg video encoding job launched!");
      onRefresh();
    } catch (err: any) {
      setFeedback(`Error: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <form
          onSubmit={handleGenerate}
          className="lg:col-span-5 bento-card p-6 space-y-4"
        >
          <div className="border-b border-zinc-100 dark:border-zinc-800 pb-3">
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-2">
              <Film className="w-4 h-4 text-lime-600" />
              MuJoCo Offscreen Video Generation Studio
            </h3>
            <p className="text-xs text-zinc-400">
              Non-blocking background job rendering via MuJoCo + FFmpeg MP4 encoder
            </p>
          </div>

          <div className="space-y-3 text-xs">
            <div>
              <label className="block text-zinc-500 font-medium mb-1">Custom Video Title (Optional)</label>
              <input
                type="text"
                placeholder="e.g. Booster K1 Kick — Side-by-Side First-Person POV"
                value={form.title}
                onChange={(e) => setForm({ ...form, title: e.target.value })}
                className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
              />
            </div>

            <div>
              <label className="block text-zinc-500 font-medium mb-1">Recording / Export Mode</label>
              <select
                value={form.video_type}
                onChange={(e) => setForm({ ...form, video_type: e.target.value })}
                className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
              >
                <option value="comparison">
                  Side-by-Side Comparison (GMR Reference vs. BeyondMimic Policy)
                </option>
                <option value="policy">
                  Export Generated Policy Motion as MP4
                </option>
                <option value="reference">
                  Export Kinematic Reference Motion as MP4
                </option>
                <option value="error_overlay">
                  Policy Recording with Tracking Error Telemetry HUD Overlay
                </option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Camera Perspective (POV)</label>
                <select
                  value={form.camera_pov}
                  onChange={(e) => setForm({ ...form, camera_pov: e.target.value as CameraPOV })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
                >
                  <option value="third_person">1. Third-Person Follow</option>
                  <option value="front">2. Front Camera</option>
                  <option value="side">3. Side Profile Camera</option>
                  <option value="top">4. Top Overhead Camera</option>
                  <option value="free">5. Diagonal Isometric Free</option>
                  <option value="first_person">6. First-Person (Robot Head_2 POV)</option>
                </select>
              </div>

              <div>
                <label className="block text-zinc-500 font-medium mb-1">Video Resolution</label>
                <select
                  value={form.resolution}
                  onChange={(e) => setForm({ ...form, resolution: e.target.value })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
                >
                  <option value="640x480">640x480 (SD Fast)</option>
                  <option value="960x720">960x720 (MJLab Standard HD)</option>
                  <option value="1280x720">1280x720 (720p Widescreen)</option>
                  <option value="1920x1080">1920x1080 (1080p Full HD)</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Frame Rate (FPS)</label>
                <select
                  value={form.fps}
                  onChange={(e) => setForm({ ...form, fps: Number(e.target.value) })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
                >
                  <option value={50}>50 FPS (Native Control)</option>
                  <option value={30}>30 FPS (Standard)</option>
                  <option value={60}>60 FPS (High FPS)</option>
                  <option value={25}>25 FPS (PAL)</option>
                </select>
              </div>

              <div>
                <label className="block text-zinc-500 font-medium mb-1">Playback Speed</label>
                <select
                  value={form.playback_speed}
                  onChange={(e) => setForm({ ...form, playback_speed: Number(e.target.value) })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
                >
                  <option value={1.0}>1.0x (Real-Time)</option>
                  <option value={0.5}>0.5x (Slow-Motion)</option>
                  <option value={0.25}>0.25x (Ultra Slow)</option>
                  <option value={2.0}>2.0x (Fast Forward)</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-zinc-500 font-medium mb-1">Policy Checkpoint</label>
              <select
                value={form.checkpoint_path}
                onChange={(e) => setForm({ ...form, checkpoint_path: e.target.value })}
                className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
              >
                {checkpoints.map((c: any) => (
                  <option key={c.file_path} value={c.file_path}>
                    {c.filename} (Iter {c.iteration})
                  </option>
                ))}
              </select>
            </div>

            <label className="flex items-center gap-2 pt-1 cursor-pointer">
              <input
                type="checkbox"
                checked={form.include_overlay}
                onChange={(e) => setForm({ ...form, include_overlay: e.target.checked })}
                className="accent-lime-500"
              />
              <span className="text-zinc-600 dark:text-zinc-300 font-medium">
                Include real-time Tracking Error (MAE, R-MPKPE, Anchor Tilt) HUD overlay
              </span>
            </label>
          </div>

          {feedback && (
            <div className="p-2.5 rounded-xl bg-lime-100 border border-lime-300 text-xs font-medium text-lime-900">
              {feedback}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full py-2.5 px-4 rounded-xl bg-zinc-900 hover:bg-zinc-800 dark:bg-lime-400 dark:text-zinc-950 disabled:opacity-50 text-white text-xs font-bold flex items-center justify-center gap-2 transition shadow-sm"
          >
            {submitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Launching Background Job...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" /> Generate MP4 Experiment Video
              </>
            )}
          </button>
        </form>

        {/* Right: Video Preview Player & Download Controls */}
        <div className="lg:col-span-7 bento-card p-6 flex flex-col justify-between">
          {activeVideo ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <div>
                  <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                    {activeVideo.title}
                  </h3>
                  <p className="text-xs font-mono text-zinc-400">
                    Type: {activeVideo.video_type.toUpperCase()} | POV:{" "}
                    {activeVideo.camera_pov.toUpperCase()} | {activeVideo.resolution} @{" "}
                    {activeVideo.fps} FPS | {activeVideo.file_size_mb} MB
                  </p>
                </div>
                {activeVideo.status === "completed" && activeVideo.download_url && (
                  <a
                    href={`${API_BASE}${activeVideo.download_url}`}
                    download
                    className="px-4 py-2 rounded-xl bg-[#84cc16] hover:bg-[#65a30d] text-white text-xs font-bold flex items-center gap-1.5 transition shadow-sm"
                  >
                    <Download className="w-3.5 h-3.5" /> Download MP4
                  </a>
                )}
              </div>

              {activeVideo.status === "processing" ? (
                <div className="h-[380px] bg-zinc-100 dark:bg-zinc-950 rounded-2xl border border-zinc-200 dark:border-zinc-800 flex flex-col items-center justify-center p-6 space-y-3">
                  <Loader2 className="w-8 h-8 text-lime-600 animate-spin" />
                  <div className="text-sm font-bold text-zinc-900 dark:text-white">
                    Rendering MuJoCo Simulation Frames & Encoding MP4...
                  </div>
                  <div className="w-64 bg-zinc-200 dark:bg-zinc-800 h-3 rounded-full overflow-hidden">
                    <div
                      className="hatch-lime h-full transition-all"
                      style={{ width: `${activeVideo.progress}%` }}
                    />
                  </div>
                  <div className="text-xs font-mono font-bold text-lime-600">
                    {activeVideo.progress}% Completed
                  </div>
                </div>
              ) : activeVideo.stream_url ? (
                <div className="bg-zinc-950 rounded-2xl overflow-hidden border border-zinc-200 dark:border-zinc-800">
                  <video
                    key={activeVideo.id}
                    src={`${API_BASE}${activeVideo.stream_url}`}
                    controls
                    autoPlay
                    loop
                    muted
                    className="w-full max-h-[395px] object-contain"
                  />
                </div>
              ) : (
                <div className="h-[380px] bg-zinc-100 dark:bg-zinc-950 rounded-2xl border border-zinc-200 flex items-center justify-center text-xs font-mono text-orange-600">
                  Video generation failed: {activeVideo.error_message || "Unknown error"}
                </div>
              )}

              <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800 flex items-center justify-between text-xs font-mono text-zinc-400">
                <span>File: {activeVideo.file_path}</span>
                <span>
                  {activeVideo.frame_count} frames ({activeVideo.duration_sec}s)
                </span>
              </div>
            </>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-zinc-500">
              Select or generate a video to preview
            </div>
          )}
        </div>
      </div>

      {/* Video Generation History Table */}
      <div className="bento-card p-6">
        <h3 className="text-sm font-bold text-zinc-900 dark:text-white mb-3">
          Video Generation History ({videos.length})
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-zinc-100 dark:border-zinc-800 text-zinc-400">
                <th className="py-2.5 px-3">Title</th>
                <th className="py-2.5 px-2">Type</th>
                <th className="py-2.5 px-2">Camera POV</th>
                <th className="py-2.5 px-2">Resolution / FPS</th>
                <th className="py-2.5 px-2">Frames / Duration</th>
                <th className="py-2.5 px-2">Size</th>
                <th className="py-2.5 px-2">Status</th>
                <th className="py-2.5 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
              {videos.map((v: any) => {
                const isSelected = activeVideo?.id === v.id;
                return (
                  <tr
                    key={v.id}
                    onClick={() => setSelectedVideo(v)}
                    className={`cursor-pointer transition ${
                      isSelected ? "bg-lime-50 dark:bg-lime-950/30" : "hover:bg-zinc-50 dark:hover:bg-zinc-800/40"
                    }`}
                  >
                    <td className="py-2.5 px-3 font-sans font-bold text-zinc-900 dark:text-white">
                      {v.title}
                      <div className="text-[11px] font-mono font-normal text-zinc-400">{v.filename}</div>
                    </td>
                    <td className="py-2.5 px-2 uppercase text-lime-600 font-bold">{v.video_type}</td>
                    <td className="py-2.5 px-2 uppercase text-zinc-600 dark:text-zinc-300">{v.camera_pov}</td>
                    <td className="py-2.5 px-2 text-zinc-600 dark:text-zinc-300">
                      {v.resolution} @ {v.fps}fps ({v.playback_speed}x)
                    </td>
                    <td className="py-2.5 px-2 text-zinc-600 dark:text-zinc-300">
                      {v.frame_count}f ({v.duration_sec}s)
                    </td>
                    <td className="py-2.5 px-2 text-zinc-400">{v.file_size_mb} MB</td>
                    <td className="py-2.5 px-2">
                      {v.status === "completed" ? (
                        <span className="px-2 py-0.5 rounded-full bg-lime-100 text-lime-700 font-sans font-semibold">
                          Ready
                        </span>
                      ) : v.status === "processing" ? (
                        <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 font-sans font-semibold">
                          Encoding {v.progress}%
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full bg-orange-100 text-orange-700 font-sans font-semibold">
                          Failed
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right space-x-2 font-sans font-semibold">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedVideo(v);
                        }}
                        className="px-3 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 hover:bg-zinc-200"
                      >
                        Preview
                      </button>
                      {v.download_url && (
                        <a
                          href={`${API_BASE}${v.download_url}`}
                          download
                          onClick={(e) => e.stopPropagation()}
                          className="px-3 py-1 rounded-xl bg-zinc-900 text-white hover:bg-zinc-800 inline-block"
                        >
                          Download
                        </a>
                      )}
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
