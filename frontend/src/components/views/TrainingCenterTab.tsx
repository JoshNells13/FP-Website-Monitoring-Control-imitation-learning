"use client";

import React, { useEffect, useRef, useState } from "react";
import { apiFetch } from "@/lib/api";
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  Terminal,
  Sliders,
  Sparkles,
} from "lucide-react";

interface TrainingCenterTabProps {
  trainingState: any;
  logs: any[];
  resources: any;
  onRefresh: () => void;
}

export default function TrainingCenterTab({
  trainingState,
  logs,
  resources,
  onRefresh,
}: TrainingCenterTabProps) {
  const [options, setOptions] = useState<any>(null);
  const [form, setForm] = useState({
    name: "BeyondMimic K1 Kick — Live Session",
    description: "",
    robot: "booster_k1",
    environment: "Mjlab-Tracking-Flat-Booster-K1",
    motion_file: "data/motions/k1_kick.npz",
    num_envs: 2048,
    max_iterations: 500,
    save_interval: 100,
    seed: 42,
    run_name: "beyondmimic_k1_kick_finetune",
    output_checkpoint: "outputs/checkpoints/k1_kick_policy.pt",
    resume_from: "outputs/checkpoints/k1_kick_policy.pt",
  });
  const [feedback, setFeedback] = useState<{ type: "ok" | "err"; msg: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    apiFetch("/api/training/options")
      .then((res) => setOptions(res))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const status = trainingState?.status || "idle";
  const isRunning = status === "running" || status === "starting";
  const isPaused = status === "paused";

  const handleAction = async (endpoint: string, body?: any) => {
    setBusy(true);
    setFeedback(null);
    try {
      const res = await apiFetch(endpoint, {
        method: "POST",
        body: body ? JSON.stringify(body) : undefined,
      });
      setFeedback({
        type: "ok",
        msg: res.message || `Command ${endpoint.split("/").pop()?.toUpperCase()} executed successfully.`,
      });
      onRefresh();
    } catch (err: any) {
      setFeedback({ type: "err", msg: err.message || "Operation failed" });
    } finally {
      setBusy(false);
    }
  };

  const cpuPct = resources?.cpu?.percent ?? 0;
  const ramPct = resources?.ram?.percent ?? 0;
  const gpuPct = resources?.gpu?.utilization_percent ?? 0;
  const vramPct = resources?.gpu?.vram_percent ?? 0;

  return (
    <div className="space-y-5">
      {/* Live Hardware Resource Telemetry Bar (100% Real psutil + nvidia-smi values) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bento-card p-4">
          <div className="flex justify-between text-xs text-zinc-500 font-medium">
            <span>CPU Utilization</span>
            <span className="text-zinc-900 dark:text-white font-mono font-bold">{cpuPct}%</span>
          </div>
          <div className="mt-2.5 w-full bg-zinc-100 dark:bg-zinc-800 h-2.5 rounded-full overflow-hidden">
            <div className="bg-[#84cc16] h-full transition-all" style={{ width: `${cpuPct}%` }} />
          </div>
          <div className="mt-1.5 text-[11px] font-mono text-zinc-400">
            {resources?.cpu
              ? `${resources.cpu.cores_physical} Physical / ${resources.cpu.cores_logical} Logical Cores`
              : "—"}
          </div>
        </div>

        <div className="bento-card p-4">
          <div className="flex justify-between text-xs text-zinc-500 font-medium">
            <span>System RAM</span>
            <span className="text-zinc-900 dark:text-white font-mono font-bold">
              {resources?.ram
                ? `${resources.ram.used_gb} / ${resources.ram.total_gb} GB (${ramPct}%)`
                : "—"}
            </span>
          </div>
          <div className="mt-2.5 w-full bg-zinc-100 dark:bg-zinc-800 h-2.5 rounded-full overflow-hidden">
            <div className="bg-[#84cc16] h-full transition-all" style={{ width: `${ramPct}%` }} />
          </div>
          <div className="mt-1.5 text-[11px] font-mono text-zinc-400">Ephemeral In-Memory Store</div>
        </div>

        <div className="bento-card p-4">
          <div className="flex justify-between text-xs text-zinc-500 font-medium">
            <span>GPU Compute (CUDA)</span>
            <span className="text-zinc-900 dark:text-white font-mono font-bold">{gpuPct}%</span>
          </div>
          <div className="mt-2.5 w-full bg-zinc-100 dark:bg-zinc-800 h-2.5 rounded-full overflow-hidden">
            <div className="bg-[#fb923c] h-full transition-all" style={{ width: `${gpuPct}%` }} />
          </div>
          <div className="mt-1.5 text-[11px] font-mono text-zinc-400 truncate">
            {resources?.gpu?.name || "—"}
          </div>
        </div>

        <div className="bento-card p-4">
          <div className="flex justify-between text-xs text-zinc-500 font-medium">
            <span>GPU VRAM</span>
            <span className="text-zinc-900 dark:text-white font-mono font-bold">
              {resources?.gpu
                ? `${resources.gpu.vram_used_gb} / ${resources.gpu.vram_total_gb} GB (${vramPct}%)`
                : "—"}
            </span>
          </div>
          <div className="mt-2.5 w-full bg-zinc-100 dark:bg-zinc-800 h-2.5 rounded-full overflow-hidden">
            <div className="bg-zinc-900 dark:bg-lime-400 h-full transition-all" style={{ width: `${vramPct}%` }} />
          </div>
          <div className="mt-1.5 text-[11px] font-mono text-zinc-400">
            MuJoCo-Warp Parallel Pool
          </div>
        </div>
      </div>

      {feedback && (
        <div
          className={`px-4 py-3 rounded-2xl border text-xs font-medium flex items-center justify-between ${
            feedback.type === "ok"
              ? "bg-lime-100/80 border-lime-300 text-lime-900"
              : "bg-orange-100/80 border-orange-300 text-orange-900"
          }`}
        >
          <span>{feedback.msg}</span>
          <button onClick={() => setFeedback(null)} className="text-zinc-500 hover:text-zinc-900">
            ✕
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-5 bento-card p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-zinc-100 dark:border-zinc-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-2">
                <Sliders className="w-4 h-4 text-lime-600" />
                MJLab BeyondMimic PPO Control
              </h3>
              <p className="text-xs text-zinc-400">
                Directly executes scripts/train_ppo.py in virtual environment
              </p>
            </div>
            <span
              className={`px-2.5 py-1 rounded-full text-xs font-mono font-bold uppercase ${
                isRunning
                  ? "bg-lime-500 text-white animate-pulse"
                  : isPaused
                  ? "bg-amber-400 text-zinc-900"
                  : "bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300"
              }`}
            >
              {status}
            </span>
          </div>

          <div className="space-y-3 text-xs">
            <div>
              <label className="block text-zinc-500 font-medium mb-1">Session Label</label>
              <input
                type="text"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Robot Model</label>
                <select
                  value={form.robot}
                  onChange={(e) => setForm({ ...form, robot: e.target.value })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
                >
                  {(options?.robots || [{ id: "booster_k1", name: "booster_k1 (22-DoF)" }]).map(
                    (r: any) => (
                      <option key={r.id} value={r.id}>
                        {r.name}
                      </option>
                    )
                  )}
                </select>
              </div>
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Environment</label>
                <select
                  value={form.environment}
                  onChange={(e) => setForm({ ...form, environment: e.target.value })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
                >
                  {(options?.environments || [{ id: "Mjlab-Tracking-Flat-Booster-K1" }]).map(
                    (env: any) => (
                      <option key={env.id} value={env.id}>
                        {env.id}
                      </option>
                    )
                  )}
                </select>
              </div>
            </div>

            <div>
              <label className="block text-zinc-500 font-medium mb-1">Reference Motion (.npz @ 50 FPS)</label>
              <select
                value={form.motion_file}
                onChange={(e) => setForm({ ...form, motion_file: e.target.value })}
                className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
              >
                {(options?.motions || []).map((m: any) => (
                  <option key={m.path} value={m.path}>
                    {m.path} ({m.size_kb} KB)
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-zinc-500 font-medium mb-1">
                Resume From Checkpoint (--resume-from)
              </label>
              <select
                value={form.resume_from}
                onChange={(e) => setForm({ ...form, resume_from: e.target.value })}
                className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
              >
                <option value="">-- Train From Scratch (Random Init) --</option>
                {(options?.checkpoints || []).map((c: any) => (
                  <option key={c.path} value={c.path}>
                    {c.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Parallel Envs</label>
                <input
                  type="number"
                  value={form.num_envs}
                  min={1}
                  max={4096}
                  onChange={(e) => setForm({ ...form, num_envs: Number(e.target.value) })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2.5 py-2 text-zinc-900 dark:text-white font-mono"
                />
              </div>
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Max Iters</label>
                <input
                  type="number"
                  value={form.max_iterations}
                  min={10}
                  max={30000}
                  onChange={(e) => setForm({ ...form, max_iterations: Number(e.target.value) })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2.5 py-2 text-zinc-900 dark:text-white font-mono"
                />
              </div>
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Save Every</label>
                <input
                  type="number"
                  value={form.save_interval}
                  min={10}
                  max={5000}
                  onChange={(e) => setForm({ ...form, save_interval: Number(e.target.value) })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2.5 py-2 text-zinc-900 dark:text-white font-mono"
                />
              </div>
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Seed</label>
                <input
                  type="number"
                  value={form.seed}
                  onChange={(e) => setForm({ ...form, seed: Number(e.target.value) })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2.5 py-2 text-zinc-900 dark:text-white font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Run Tag</label>
                <input
                  type="text"
                  value={form.run_name}
                  onChange={(e) => setForm({ ...form, run_name: e.target.value })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
                />
              </div>
              <div>
                <label className="block text-zinc-500 font-medium mb-1">Output Policy</label>
                <input
                  type="text"
                  value={form.output_checkpoint}
                  onChange={(e) => setForm({ ...form, output_checkpoint: e.target.value })}
                  className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
                />
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-zinc-100 dark:border-zinc-800 grid grid-cols-2 sm:grid-cols-4 gap-2">
            <button
              disabled={busy || isRunning || isPaused}
              onClick={() => handleAction("/api/training/start", form)}
              className="flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-xl bg-[#84cc16] hover:bg-[#65a30d] disabled:opacity-40 text-white text-xs font-bold transition shadow-sm"
            >
              <Play className="w-3.5 h-3.5" /> Start PPO
            </button>

            <button
              disabled={busy || !isRunning}
              onClick={() => handleAction("/api/training/pause")}
              className="flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-xl bg-amber-400 hover:bg-amber-500 disabled:opacity-40 text-zinc-950 text-xs font-bold transition"
            >
              <Pause className="w-3.5 h-3.5" /> Pause
            </button>

            <button
              disabled={busy || !isPaused}
              onClick={() => handleAction("/api/training/resume")}
              className="flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-xl bg-zinc-900 hover:bg-zinc-800 disabled:opacity-40 text-white text-xs font-bold transition"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Resume
            </button>

            <button
              disabled={busy || (!isRunning && !isPaused)}
              onClick={() => handleAction("/api/training/stop")}
              className="flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-xl bg-orange-500 hover:bg-orange-600 disabled:opacity-40 text-white text-xs font-bold transition"
            >
              <Square className="w-3.5 h-3.5" /> Stop
            </button>
          </div>

          <div className="p-3.5 rounded-2xl bg-zinc-50 dark:bg-zinc-800/70 border border-zinc-200/70 dark:border-zinc-700 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-zinc-900 dark:text-white flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-lime-600" />
                Quantitative Evaluation (evaluate_and_play.py)
              </span>
              <button
                disabled={busy || isRunning}
                onClick={() =>
                  handleAction("/api/training/evaluate", {
                    checkpoint_file: form.resume_from || form.output_checkpoint,
                    motion_file: form.motion_file,
                    num_eval_envs: 256,
                  })
                }
                className="px-3 py-1.5 rounded-xl bg-zinc-900 text-white hover:bg-zinc-800 text-xs font-semibold transition"
              >
                Evaluate Policy
              </button>
            </div>
            <p className="text-[11px] text-zinc-500">
              Executes 256-episode statistical rollout + deterministic video capture on the selected checkpoint.
            </p>
          </div>
        </div>

        {/* Right Column: Live Process Telemetry (Zero dummy fallbacks) */}
        <div className="lg:col-span-7 flex flex-col gap-5">
          <div className="bento-card p-6">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div>
                <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
                  Real-Time Training Execution Telemetry
                </h3>
                <p className="text-xs text-zinc-400 font-mono">
                  PID: {trainingState?.pid ?? "None"} | Active Log Dir:{" "}
                  {trainingState?.log_dir ?? "None (Process Idle)"}
                </p>
              </div>
              <div className="text-right font-mono">
                <div className="text-sm font-extrabold text-zinc-900 dark:text-white">
                  Iteration {trainingState?.current_iteration ?? 0} /{" "}
                  {trainingState?.max_iterations ?? form.max_iterations}
                </div>
                <div className="text-[11px] text-lime-600 font-semibold">
                  Elapsed: {trainingState?.elapsed_sec ?? 0}s (
                  {trainingState?.progress_percent ?? 0}%)
                </div>
              </div>
            </div>

            <div className="w-full bg-zinc-100 dark:bg-zinc-800 h-4 rounded-xl overflow-hidden p-0.5">
              <div
                className="hatch-lime h-full rounded-lg transition-all duration-300"
                style={{ width: `${trainingState?.progress_percent ?? 0}%` }}
              />
            </div>

            <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="p-3 rounded-2xl bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200/60 dark:border-zinc-700">
                <span className="text-zinc-400 block text-[10px]">LIVE MEAN REWARD</span>
                <span className="text-base font-extrabold text-zinc-900 dark:text-white">
                  {trainingState?.latest_metrics?.mean_reward !== undefined
                    ? Number(trainingState.latest_metrics.mean_reward).toFixed(3)
                    : "—"}
                </span>
              </div>
              <div className="p-3 rounded-2xl bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200/60 dark:border-zinc-700">
                <span className="text-zinc-400 block text-[10px]">EPISODE LENGTH</span>
                <span className="text-base font-extrabold text-lime-600 dark:text-lime-400">
                  {trainingState?.latest_metrics?.mean_episode_length !== undefined
                    ? Number(trainingState.latest_metrics.mean_episode_length).toFixed(1)
                    : "—"}
                </span>
              </div>
              <div className="p-3 rounded-2xl bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200/60 dark:border-zinc-700">
                <span className="text-zinc-400 block text-[10px]">BODY POS ERROR</span>
                <span className="text-base font-extrabold text-orange-500">
                  {trainingState?.latest_metrics?.error_body_pos !== undefined
                    ? `${Number(trainingState.latest_metrics.error_body_pos).toFixed(3)} m`
                    : "—"}
                </span>
              </div>
              <div className="p-3 rounded-2xl bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200/60 dark:border-zinc-700">
                <span className="text-zinc-400 block text-[10px]">JOINT POS ERROR</span>
                <span className="text-base font-extrabold text-zinc-900 dark:text-white">
                  {trainingState?.latest_metrics?.error_joint_pos !== undefined
                    ? `${Number(trainingState.latest_metrics.error_joint_pos).toFixed(3)} rad`
                    : "—"}
                </span>
              </div>
            </div>
          </div>

          <div className="flex-1 bg-zinc-950 border border-zinc-800 rounded-[22px] overflow-hidden flex flex-col min-h-[370px] shadow-sm">
            <div className="bg-zinc-900 border-b border-zinc-800 px-4 py-3 flex items-center justify-between text-xs font-mono">
              <div className="flex items-center gap-2 text-zinc-200">
                <Terminal className="w-4 h-4 text-lime-400" />
                <span>MJLab Process stdout / stderr Stream</span>
              </div>
              <span className="text-[11px] text-zinc-400">{logs.length} log entries</span>
            </div>
            <div
              ref={logContainerRef}
              className="flex-1 p-4 font-mono text-xs overflow-y-auto space-y-1 max-h-[390px]"
            >
              {logs.length === 0 ? (
                <div className="text-zinc-500">
                  Tidak ada proses training yang sedang berjalan. Klik &quot;Start PPO&quot; atau &quot;Evaluate Policy&quot; untuk menjalankan subprocess MJLab.
                </div>
              ) : (
                logs.map((entry, i) => (
                  <div
                    key={i}
                    className={`leading-relaxed break-all ${
                      entry.level === "error" ? "text-orange-400" : "text-zinc-200"
                    }`}
                  >
                    <span className="text-zinc-500 mr-2">
                      [{entry.timestamp?.slice(11, 19) || ""}]
                    </span>
                    {entry.line}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
