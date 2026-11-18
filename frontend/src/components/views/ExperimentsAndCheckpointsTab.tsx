"use client";

import React, { useEffect, useState } from "react";
import { API_BASE, apiFetch } from "@/lib/api";
import {
  Plus,
  Archive,
  Terminal,
  Award,
  Download,
  Settings,
  Database,
} from "lucide-react";

interface ExperimentsViewProps {
  experiments: any[];
  onRefresh: () => void;
  onSelectCheckpointForSim: (ckptPath: string) => void;
}

export function ExperimentsTab({
  experiments,
  onRefresh,
  onSelectCheckpointForSim,
}: ExperimentsViewProps) {
  const [selectedExp, setSelectedExp] = useState<any>(() => experiments[0] || null);
  const [expLogs, setExpLogs] = useState<string[]>([]);
  const [showArchived, setShowArchived] = useState(false);
  const [allExps, setAllExps] = useState<any[]>(experiments);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newExp, setNewExp] = useState({
    name: "",
    run_name: "beyondmimic_k1_kick_custom",
    description: "",
    num_envs: 2048,
    max_iterations: 500,
    save_interval: 100,
  });

  useEffect(() => {
    apiFetch(`/api/experiments?include_archived=${showArchived}`)
      .then((res) => {
        setAllExps(res.experiments || []);
        if (!selectedExp && res.experiments?.length) {
          setSelectedExp(res.experiments[0]);
        }
      })
      .catch(() => {});
  }, [showArchived, experiments]);

  useEffect(() => {
    if (!selectedExp?.id) return;
    apiFetch(`/api/experiments/${selectedExp.id}/logs`)
      .then((res) => setExpLogs(res.log_lines || []))
      .catch(() => setExpLogs([]));
  }, [selectedExp?.id]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const created = await apiFetch("/api/experiments", {
        method: "POST",
        body: JSON.stringify(newExp),
      });
      setShowCreateModal(false);
      setNewExp({
        name: "",
        run_name: "beyondmimic_k1_kick_custom",
        description: "",
        num_envs: 2048,
        max_iterations: 500,
        save_interval: 100,
      });
      setSelectedExp(created);
      onRefresh();
    } catch {
      // ignore
    }
  };

  const handleArchiveToggle = async (exp: any) => {
    await apiFetch(`/api/experiments/${exp.id}`, {
      method: "PATCH",
      body: JSON.stringify({ is_archived: !exp.is_archived }),
    });
    onRefresh();
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 bento-card p-5">
        <div>
          <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
            Experiment Management & Training Sessions ({allExps.length})
          </h3>
          <p className="text-xs text-zinc-400">
            Inspect session metadata, YAML configurations, TensorBoard logs, and associated checkpoints
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs font-medium">
          <label className="flex items-center gap-1.5 text-zinc-500 cursor-pointer">
            <input
              type="checkbox"
              checked={showArchived}
              onChange={(e) => setShowArchived(e.target.checked)}
              className="accent-lime-500"
            />
            Show Archived
          </label>
          <button
            onClick={() => setShowCreateModal(!showCreateModal)}
            className="px-4 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 dark:bg-lime-400 dark:text-zinc-950 text-white font-semibold flex items-center gap-1.5 shadow-sm"
          >
            <Plus className="w-3.5 h-3.5" /> New Session
          </button>
        </div>
      </div>

      {showCreateModal && (
        <form
          onSubmit={handleCreate}
          className="bento-card p-5 space-y-3 text-xs border-lime-400"
        >
          <h4 className="font-bold text-sm text-zinc-900 dark:text-white">
            Create In-Memory Experiment Session
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label className="block text-zinc-500 mb-1">Experiment Name *</label>
              <input
                required
                type="text"
                value={newExp.name}
                onChange={(e) => setNewExp({ ...newExp, name: e.target.value })}
                placeholder="e.g. Booster K1 Kick Stage 6"
                className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="block text-zinc-500 mb-1">Run Tag</label>
              <input
                type="text"
                value={newExp.run_name}
                onChange={(e) => setNewExp({ ...newExp, run_name: e.target.value })}
                className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white font-mono"
              />
            </div>
            <div>
              <label className="block text-zinc-500 mb-1">Parallel Envs / Max Iterations</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="number"
                  value={newExp.num_envs}
                  onChange={(e) => setNewExp({ ...newExp, num_envs: Number(e.target.value) })}
                  className="bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2.5 py-2 text-zinc-900 dark:text-white font-mono"
                />
                <input
                  type="number"
                  value={newExp.max_iterations}
                  onChange={(e) => setNewExp({ ...newExp, max_iterations: Number(e.target.value) })}
                  className="bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-2.5 py-2 text-zinc-900 dark:text-white font-mono"
                />
              </div>
            </div>
          </div>
          <div>
            <label className="block text-zinc-500 mb-1">Description</label>
            <textarea
              rows={2}
              value={newExp.description}
              onChange={(e) => setNewExp({ ...newExp, description: e.target.value })}
              placeholder="Document hypothesis, reward weights, or termination thresholds..."
              className="w-full bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl px-3 py-2 text-zinc-900 dark:text-white"
            />
          </div>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setShowCreateModal(false)}
              className="px-3.5 py-1.5 rounded-xl bg-zinc-100 text-zinc-700"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-1.5 rounded-xl bg-zinc-900 text-white font-semibold"
            >
              Save Session
            </button>
          </div>
        </form>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-5 space-y-3">
          {allExps.map((exp) => {
            const active = selectedExp?.id === exp.id;
            return (
              <div
                key={exp.id}
                onClick={() => setSelectedExp(exp)}
                className={`p-4 bento-card cursor-pointer transition ${
                  active
                    ? "ring-2 ring-lime-500"
                    : "hover:border-zinc-300"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="text-sm font-bold text-zinc-900 dark:text-white">{exp.name}</div>
                    <div className="text-[11px] font-mono text-zinc-400 mt-0.5">{exp.run_id}</div>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase ${
                      exp.status === "completed"
                        ? "bg-lime-100 text-lime-700"
                        : "bg-zinc-100 text-zinc-700"
                    }`}
                  >
                    {exp.is_archived ? "archived" : exp.status}
                  </span>
                </div>

                <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 line-clamp-2">
                  {exp.description}
                </p>

                <div className="mt-3 pt-2.5 border-t border-zinc-100 dark:border-zinc-800 grid grid-cols-4 gap-2 text-[11px] font-mono">
                  <div>
                    <span className="text-zinc-400 block">ITERS</span>
                    <span className="text-zinc-900 dark:text-white font-bold">
                      {exp.start_step}–{exp.final_step}
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-400 block">REWARD</span>
                    <span className="text-lime-600 font-bold">+{exp.mean_reward?.toFixed(2) ?? 0}</span>
                  </div>
                  <div>
                    <span className="text-zinc-400 block">R-MPKPE</span>
                    <span className="text-zinc-900 dark:text-white font-bold">{exp.r_mpkpe_mm?.toFixed(1) ?? "-"} mm</span>
                  </div>
                  <div>
                    <span className="text-zinc-400 block">CKPTS</span>
                    <span className="text-orange-500 font-bold">{exp.checkpoint_count ?? 6} files</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="lg:col-span-7 space-y-4">
          {selectedExp && (
            <>
              <div className="bento-card p-6 space-y-4">
                <div className="flex flex-wrap items-start justify-between gap-2 border-b border-zinc-100 dark:border-zinc-800 pb-3">
                  <div>
                    <h3 className="text-base font-extrabold text-zinc-900 dark:text-white">
                      {selectedExp.name}
                    </h3>
                    <p className="text-xs font-mono text-lime-600">{selectedExp.log_dir}</p>
                  </div>
                  <button
                    onClick={() => handleArchiveToggle(selectedExp)}
                    className="px-3 py-1.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 text-zinc-700 dark:text-zinc-200 text-xs font-semibold flex items-center gap-1.5"
                  >
                    <Archive className="w-3.5 h-3.5" />
                    {selectedExp.is_archived ? "Unarchive" : "Archive"}
                  </button>
                </div>

                <p className="text-xs text-zinc-600 dark:text-zinc-300 leading-relaxed">
                  {selectedExp.description}
                </p>

                <div>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-2">
                    Associated Model Checkpoints ({selectedExp.checkpoints?.length || 0})
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-h-48 overflow-y-auto pr-1">
                    {(selectedExp.checkpoints || []).map((c: any) => (
                      <div
                        key={c.id}
                        className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-800/80 border border-zinc-200/70 dark:border-zinc-700 flex items-center justify-between text-xs font-mono"
                      >
                        <div>
                          <div className="text-zinc-900 dark:text-white font-bold">{c.filename}</div>
                          <div className="text-[10px] text-zinc-400">
                            Iter {c.iteration} | Reward: +{c.mean_reward?.toFixed(2)} | {c.file_size_mb} MB
                          </div>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() => onSelectCheckpointForSim(c.file_path)}
                            className="px-2.5 py-1 rounded-lg bg-zinc-900 text-white hover:bg-zinc-800 text-[11px] font-sans font-semibold"
                          >
                            3D Sim
                          </button>
                          <a
                            href={`${API_BASE}/api/checkpoints/download/${c.id}`}
                            download
                            className="p-1.5 rounded-lg bg-zinc-200/80 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-200"
                            title="Download .pt"
                          >
                            <Download className="w-3.5 h-3.5" />
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-zinc-950 border border-zinc-800 rounded-[22px] overflow-hidden shadow-sm">
                <div className="bg-zinc-900 border-b border-zinc-800 px-4 py-3 flex items-center justify-between text-xs font-mono">
                  <span className="flex items-center gap-2 text-zinc-200">
                    <Terminal className="w-4 h-4 text-lime-400" />
                    Experiment TensorBoard Iteration Log ({selectedExp.run_id})
                  </span>
                  <span className="text-zinc-500">{expLogs.length} lines</span>
                </div>
                <div className="p-4 font-mono text-xs text-zinc-300 space-y-1 max-h-72 overflow-y-auto">
                  {expLogs.map((line, idx) => (
                    <div key={idx} className="leading-relaxed">
                      {line}
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

interface CheckpointsTabProps {
  checkpoints: any[];
  onSelectForSim: (path: string) => void;
}

export function CheckpointsTab({ checkpoints, onSelectForSim }: CheckpointsTabProps) {
  return (
    <div className="space-y-5">
      <div className="bento-card p-6">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-2">
              <Award className="w-4 h-4 text-lime-600" />
              Model Checkpoint Repository ({checkpoints.length} Saved Policies)
            </h3>
            <p className="text-xs text-zinc-400">
              All PyTorch (.pt) and exported ONNX (.onnx) policies across logs/rsl_rl/k1_tracking and outputs/checkpoints
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-zinc-100 dark:border-zinc-800 text-zinc-400">
                <th className="py-2.5 px-3">Checkpoint Name</th>
                <th className="py-2.5 px-2">Iteration</th>
                <th className="py-2.5 px-2">Experiment Session</th>
                <th className="py-2.5 px-2">Mean Reward</th>
                <th className="py-2.5 px-2">Joint MAE</th>
                <th className="py-2.5 px-2">R-MPKPE</th>
                <th className="py-2.5 px-2">Size</th>
                <th className="py-2.5 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
              {checkpoints.map((c: any) => (
                <tr key={c.id} className="hover:bg-zinc-50 dark:hover:bg-zinc-800/40">
                  <td className="py-2.5 px-3">
                    <div className="font-bold text-zinc-900 dark:text-white flex items-center gap-1.5">
                      {c.filename}
                      {c.is_primary_output && (
                        <span className="px-2 py-0.5 rounded-full bg-lime-100 text-lime-800 text-[10px] font-sans font-bold">
                          PRIMARY POLICY
                        </span>
                      )}
                    </div>
                    <div className="text-[10px] text-zinc-400">{c.file_path}</div>
                  </td>
                  <td className="py-2.5 px-2 text-zinc-900 dark:text-white font-bold">Iter {c.iteration}</td>
                  <td className="py-2.5 px-2 text-zinc-600 dark:text-zinc-300 font-sans">{c.experiment_name}</td>
                  <td className="py-2.5 px-2 text-lime-600 font-bold">+{c.mean_reward?.toFixed(2)}</td>
                  <td className="py-2.5 px-2 text-zinc-800 dark:text-zinc-200">{c.tracking_error_mae?.toFixed(4)} rad</td>
                  <td className="py-2.5 px-2 text-orange-500 font-semibold">{c.r_mpkpe_mm?.toFixed(1)} mm</td>
                  <td className="py-2.5 px-2 text-zinc-400">{c.file_size_mb} MB</td>
                  <td className="py-2.5 px-3 text-right space-x-1.5 font-sans font-semibold">
                    <button
                      onClick={() => onSelectForSim(c.file_path)}
                      className="px-3 py-1 rounded-xl bg-zinc-900 text-white hover:bg-zinc-800"
                    >
                      3D Viewer
                    </button>
                    <a
                      href={`${API_BASE}/api/checkpoints/download/${c.id}?format=pt`}
                      download
                      className="px-2.5 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 hover:bg-zinc-200 inline-block"
                    >
                      .PT
                    </a>
                    {c.onnx_path && (
                      <a
                        href={`${API_BASE}/api/checkpoints/download/${c.id}?format=onnx`}
                        download
                        className="px-2.5 py-1 rounded-xl bg-lime-100 text-lime-800 hover:bg-lime-200 inline-block"
                      >
                        .ONNX
                      </a>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export function SettingsTab({ theme, onToggleTheme }: { theme: string; onToggleTheme: () => void }) {
  const [settings, setSettings] = useState<any>(null);

  useEffect(() => {
    apiFetch("/api/settings")
      .then((res) => setSettings(res))
      .catch(() => {});
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div className="bento-card p-6 space-y-4">
        <h3 className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-2">
          <Database className="w-4 h-4 text-lime-600" />
          MJLab & Backend Integration Paths
        </h3>
        <div className="space-y-2.5 text-xs font-mono">
          <div className="p-3.5 rounded-2xl bg-zinc-50 dark:bg-zinc-800 border border-zinc-200/70 dark:border-zinc-700">
            <span className="text-zinc-400 block text-[10px]">MJLAB REPOSITORY ROOT</span>
            <span className="text-zinc-900 dark:text-white font-semibold break-all">
              {settings?.environment?.hil_root || "c:\\Users\\ASUS\\Documents\\fp-ichiro\\human-imitation-learning"}
            </span>
          </div>
          <div className="p-3.5 rounded-2xl bg-zinc-50 dark:bg-zinc-800 border border-zinc-200/70 dark:border-zinc-700">
            <span className="text-zinc-400 block text-[10px]">PYTHON VIRTUAL ENVIRONMENT</span>
            <span className="text-lime-600 font-semibold break-all">
              {settings?.environment?.python_exe || "human-imitation-learning\\.venv\\Scripts\\python.exe"}
            </span>
          </div>
          <div className="p-3.5 rounded-2xl bg-zinc-50 dark:bg-zinc-800 border border-zinc-200/70 dark:border-zinc-700">
            <span className="text-zinc-400 block text-[10px]">BOOSTER K1 MUJOCO XML SPECIFICATION</span>
            <span className="text-zinc-900 dark:text-white break-all">
              {settings?.environment?.robot_xml || "third_party/GMR/assets/booster_k1/k1_mjlab.xml"}
            </span>
          </div>
          <div className="p-3.5 rounded-2xl bg-lime-50/70 dark:bg-lime-950/30 border border-lime-200 dark:border-lime-800">
            <span className="text-lime-700 dark:text-lime-400 block text-[10px]">STORAGE & MONITORING MODE</span>
            <span className="text-zinc-900 dark:text-white font-bold">
              {settings?.database?.engine || "Stateless Ephemeral In-Memory Monitoring (No Database Used)"}
            </span>
          </div>
        </div>
      </div>

      <div className="bento-card p-6 space-y-4">
        <h3 className="text-sm font-bold text-zinc-900 dark:text-white flex items-center gap-2">
          <Settings className="w-4 h-4 text-lime-600" />
          Interface Theme & Pipeline Architecture
        </h3>

        <div className="p-4 rounded-2xl bg-zinc-50 dark:bg-zinc-800 border border-zinc-200/70 dark:border-zinc-700 flex items-center justify-between">
          <div>
            <div className="text-xs font-bold text-zinc-900 dark:text-white">Color Theme Mode</div>
            <div className="text-[11px] text-zinc-400">
              Clean ACRU Bento Light theme primary; Dark mode optional
            </div>
          </div>
          <button
            onClick={onToggleTheme}
            className="px-4 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 dark:bg-lime-400 dark:text-zinc-950 text-white text-xs font-semibold"
          >
            Theme: {theme.toUpperCase()} (Switch)
          </button>
        </div>

        <div className="space-y-2 text-xs font-mono text-zinc-700 dark:text-zinc-300">
          <div className="p-3.5 rounded-2xl bg-zinc-50 dark:bg-zinc-800 border border-zinc-200/70 dark:border-zinc-700 flex justify-between">
            <span className="text-zinc-400">Simulator Core:</span>
            <span className="font-semibold">MuJoCo 3.16.0 + MJLab 1.6.0 (MuJoCo-Warp GPU)</span>
          </div>
          <div className="p-3.5 rounded-2xl bg-zinc-50 dark:bg-zinc-800 border border-zinc-200/70 dark:border-zinc-700 flex justify-between">
            <span className="text-zinc-400">RL Policy Architecture:</span>
            <span className="font-semibold">BeyondMimic PPO [512, 256, 128] ELU</span>
          </div>
          <div className="p-3.5 rounded-2xl bg-zinc-50 dark:bg-zinc-800 border border-zinc-200/70 dark:border-zinc-700 flex justify-between">
            <span className="text-zinc-400">Motion Retargeting:</span>
            <span className="font-semibold">GMR Two-Stage Differential IK (Mink + DAQP)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
