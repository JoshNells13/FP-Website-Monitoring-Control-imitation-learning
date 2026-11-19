"use client";

import React, { useCallback, useEffect, useState } from "react";
import { apiFetch, CameraPOV, getSocket, TrajectoryBundle } from "@/lib/api";
import OverviewTab from "@/components/views/OverviewTab";
import TrainingCenterTab from "@/components/views/TrainingCenterTab";
import SimulationViewerTab from "@/components/views/SimulationViewerTab";
import MotionAnalysisTab from "@/components/views/MotionAnalysisTab";
import AnalyticsTab from "@/components/views/AnalyticsTab";
import VideoGenerationTab from "@/components/views/VideoGenerationTab";
import {
  ExperimentsTab,
  CheckpointsTab,
  SettingsTab,
} from "@/components/views/ExperimentsAndCheckpointsTab";
import {
  LayoutGrid,
  Cpu,
  Box,
  Activity,
  FolderKanban,
  Award,
  Settings,
  Plus,
  Zap,
  ChevronsLeft,
  ChevronsRight,
  ChevronDown,
  RefreshCw,
  Radio,
} from "lucide-react";

export default function PlatformPage() {
  const [activeTab, setActiveTab] = useState<string>("overview");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);
  const [telemetryGroupOpen, setTelemetryGroupOpen] = useState<boolean>(true);
  const [socketConnected, setSocketConnected] = useState<boolean>(false);

  const [dashboardData, setDashboardData] = useState<any>(null);
  const [trainingState, setTrainingState] = useState<any>(null);
  const [trainingLogs, setTrainingLogs] = useState<any[]>([]);
  const [resources, setResources] = useState<any>(null);
  const [trajectory, setTrajectory] = useState<TrajectoryBundle | null>(null);
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  const [videos, setVideos] = useState<any[]>([]);
  const [selectedMotion, setSelectedMotion] = useState<string>(
    "data/motions/k1_dance.npz"
  );
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>(
    "outputs/checkpoints/k1_dance_policy.pt"
  );
  const [initialCameraPov, setInitialCameraPov] = useState<CameraPOV>("third_person");

  const toggleTheme = () => {
    const next = theme === "light" ? "dark" : "light";
    setTheme(next);
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", next === "dark");
    }
  };

  const fetchCoreData = useCallback(async () => {
    try {
      const [ov, ckptsRes, vidsRes, logsRes] = await Promise.all([
        apiFetch("/api/dashboard/overview"),
        apiFetch("/api/checkpoints"),
        apiFetch("/api/videos"),
        apiFetch("/api/training/logs?limit=250"),
      ]);
      setDashboardData(ov);
      setTrainingState(ov.training_state);
      setResources(ov.system_resources);
      setCheckpoints(ckptsRes.checkpoints || []);
      setVideos(vidsRes.videos || []);
      setTrainingLogs(logsRes.logs || []);
    } catch {
      // ignore
    }
  }, []);

  const fetchTrajectory = useCallback(async (motionPath: string, ckptPath: string) => {
    try {
      const bundle = await apiFetch<TrajectoryBundle>(
        `/api/simulation/trajectory?motion_file=${encodeURIComponent(
          motionPath
        )}&checkpoint_file=${encodeURIComponent(ckptPath)}`
      );
      setTrajectory(bundle);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    fetchCoreData();
    fetchTrajectory(selectedMotion, selectedCheckpoint);
  }, [fetchCoreData, fetchTrajectory, selectedMotion, selectedCheckpoint]);

  useEffect(() => {
    const socket = getSocket();

    const onConnect = () => setSocketConnected(true);
    const onDisconnect = () => setSocketConnected(false);
    const onTrainingStatus = (state: any) => {
      setTrainingState(state);
      if (state?.resources) setResources(state.resources);
    };
    const onTrainingProgress = (state: any) => {
      setTrainingState(state);
      if (state?.resources) setResources(state.resources);
    };
    const onTrainingLog = (entry: any) => {
      setTrainingLogs((prev) => [...prev.slice(-600), entry]);
    };
    const onVideoProgress = (vrec: any) => {
      setVideos((prev) => {
        const exists = prev.some((v) => v.id === vrec.id);
        if (exists) {
          return prev.map((v) => (v.id === vrec.id ? vrec : v));
        }
        return [vrec, ...prev];
      });
    };

    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("training_status", onTrainingStatus);
    socket.on("training_progress", onTrainingProgress);
    socket.on("training_log", onTrainingLog);
    socket.on("video_progress", onVideoProgress);

    if (socket.connected) setSocketConnected(true);

    const pollInterval = setInterval(() => {
      socket.emit("request_status");
    }, 4000);

    return () => {
      clearInterval(pollInterval);
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("training_status", onTrainingStatus);
      socket.off("training_progress", onTrainingProgress);
      socket.off("training_log", onTrainingLog);
      socket.off("video_progress", onVideoProgress);
    };
  }, []);

  const handleSelectCheckpointForSim = (ckptPath: string) => {
    setSelectedCheckpoint(ckptPath);
    if (ckptPath.toLowerCase().includes("dance")) {
      setSelectedMotion("data/motions/k1_dance.npz");
    } else if (ckptPath.toLowerCase().includes("kick")) {
      setSelectedMotion("data/motions/k1_kick.npz");
    }
    setActiveTab("simulation");
  };

  const handleJumpToCameraPov = (pov: CameraPOV) => {
    setInitialCameraPov(pov);
    setActiveTab("simulation");
  };

  const trainStatus = trainingState?.status || "idle";

  return (
    <div className="min-h-screen flex bg-[#F3F4F6] dark:bg-[#0b0f17] text-zinc-900 dark:text-zinc-100 p-3 sm:p-4 gap-5">
      {/* Left Minimalist Bento Sidebar */}
      <aside
        className={`${sidebarCollapsed ? "w-20" : "w-60"
          } shrink-0 flex flex-col justify-between transition-all duration-200 select-none`}
      >
        <div>
          <div className="px-3 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-zinc-900 dark:bg-lime-400 text-white dark:text-zinc-950 flex items-center justify-center font-black text-sm tracking-tighter">
                JJ
              </div>
              {!sidebarCollapsed && (
                <span className="text-xl font-black tracking-tight text-zinc-900 dark:text-white">
                  JossJiss
                </span>
              )}
            </div>
            {!sidebarCollapsed && (
              <span
                className={`h-2.5 w-2.5 rounded-full ${socketConnected ? "bg-lime-500" : "bg-amber-400"
                  }`}
                title={socketConnected ? "Socket.IO Live Connected" : "Polling"}
              />
            )}
          </div>

          <nav className="mt-4 space-y-1">
            {[
              { id: "overview", label: "Dashboard", icon: LayoutGrid },
              {
                id: "training",
                label: "Training Center",
                icon: Cpu,
                badge: trainStatus === "running" ? "LIVE" : undefined,
              },
              { id: "simulation", label: "3D Simulation", icon: Box },
            ].map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`relative w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-medium transition ${isActive
                      ? "bg-zinc-200/80 dark:bg-zinc-800 text-zinc-900 dark:text-white font-semibold"
                      : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200/40 dark:hover:bg-zinc-800/40"
                    }`}
                >
                  {isActive && (
                    <span className="absolute left-0 top-2.5 bottom-2.5 w-1 bg-zinc-900 dark:bg-lime-400 rounded-r-full" />
                  )}
                  <span className="flex items-center gap-3">
                    <Icon className="w-4 h-4 shrink-0" />
                    {!sidebarCollapsed && <span>{item.label}</span>}
                  </span>
                  {!sidebarCollapsed && item.badge && (
                    <span className="px-1.5 py-0.5 rounded-full bg-lime-500 text-white text-[10px] font-bold">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}

            <div>
              <button
                onClick={() => setTelemetryGroupOpen(!telemetryGroupOpen)}
                className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200/40 dark:hover:bg-zinc-800/40 transition"
              >
                <span className="flex items-center gap-3">
                  <Activity className="w-4 h-4 shrink-0" />
                  {!sidebarCollapsed && <span>Telemetry & Media</span>}
                </span>
                {!sidebarCollapsed && (
                  <ChevronDown
                    className={`w-3.5 h-3.5 text-zinc-400 transition-transform ${telemetryGroupOpen ? "rotate-180" : ""
                      }`}
                  />
                )}
              </button>

              {telemetryGroupOpen && !sidebarCollapsed && (
                <div className="ml-5 pl-3 border-l border-zinc-300 dark:border-zinc-700 my-1 space-y-1">
                  {[
                    {
                      id: "motion",
                      label: "Motion Analysis",
                      count: trajectory?.per_joint_summary?.length
                        ? String(trajectory.per_joint_summary.length)
                        : undefined,
                    },
                    { id: "analytics", label: "Analytics & Plots" },
                    {
                      id: "videos",
                      label: "Video Studio",
                      count: videos.length > 0 ? String(videos.length) : undefined,
                    },
                  ].map((sub) => {
                    const isSubActive = activeTab === sub.id;
                    return (
                      <button
                        key={sub.id}
                        onClick={() => setActiveTab(sub.id)}
                        className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs transition ${isSubActive
                            ? "bg-zinc-200/80 dark:bg-zinc-800 text-zinc-900 dark:text-white font-semibold"
                            : "text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white"
                          }`}
                      >
                        <span>{sub.label}</span>
                        {sub.count && (
                          <span className="px-1.5 py-0.5 rounded-full bg-zinc-900 dark:bg-zinc-700 text-white text-[10px] font-semibold">
                            {sub.count}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {[
              { id: "experiments", label: "Experiments", icon: FolderKanban },
              {
                id: "checkpoints",
                label: "Checkpoints",
                icon: Award,
                badge: checkpoints.length > 0 ? String(checkpoints.length) : undefined,
              },
              { id: "settings", label: "System Settings", icon: Settings },
            ].map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`relative w-full flex items-center justify-between px-3.5 py-2.5 rounded-2xl text-xs font-medium transition ${isActive
                      ? "bg-zinc-200/80 dark:bg-zinc-800 text-zinc-900 dark:text-white font-semibold"
                      : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200/40 dark:hover:bg-zinc-800/40"
                    }`}
                >
                  {isActive && (
                    <span className="absolute left-0 top-2.5 bottom-2.5 w-1 bg-zinc-900 dark:bg-lime-400 rounded-r-full" />
                  )}
                  <span className="flex items-center gap-3">
                    <Icon className="w-4 h-4 shrink-0" />
                    {!sidebarCollapsed && <span>{item.label}</span>}
                  </span>
                  {!sidebarCollapsed && item.badge && (
                    <span className="px-1.5 py-0.5 rounded-full bg-zinc-200 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 text-[10px] font-semibold">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Stateless Ephemeral Control Card */}
        <div className="space-y-3">
          {!sidebarCollapsed && (
            <div className="bento-card p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 rounded-full bg-zinc-900 dark:bg-lime-400 text-white dark:text-zinc-900 flex items-center justify-center">
                  <Zap className="w-4 h-4" />
                </div>
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-lime-100 dark:bg-lime-950 text-lime-700 dark:text-lime-300 font-semibold">
                  Ephemeral RAM
                </span>
              </div>
              <div>
                <div className="text-xs font-bold text-zinc-900 dark:text-white">
                  Stateless Live Monitor
                </div>
                <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5 leading-relaxed">
                  No database used. Simulation buffers discard immediately on reset.
                </p>
              </div>
              <button
                onClick={async () => {
                  await apiFetch("/api/simulation/discard", { method: "POST" });
                  fetchCoreData();
                }}
                className="w-full py-2 px-3 rounded-xl bg-zinc-900 hover:bg-zinc-800 dark:bg-lime-400 dark:hover:bg-lime-300 text-white dark:text-zinc-950 text-xs font-semibold transition shadow-sm"
              >
                Discard Sim Buffer
              </button>
            </div>
          )}

          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-full flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-zinc-500 hover:text-zinc-900 dark:hover:text-white transition"
          >
            {sidebarCollapsed ? (
              <ChevronsRight className="w-4 h-4" />
            ) : (
              <>
                <ChevronsLeft className="w-4 h-4" />
                <span>Collapse sidebar</span>
              </>
            )}
          </button>
        </div>
      </aside>

      {/* Main Content Workspace */}
      <div className="flex-1 flex flex-col min-w-0 gap-5">
        {/* Top Header Bar: 100% Real Live Telemetry & Actions (Zero dummy elements) */}
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2.5 bg-white dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-800 rounded-2xl px-4 py-2.5 text-xs shadow-[0_2px_10px_rgba(15,23,42,0.02)]">
            <span className="flex items-center gap-1.5 font-mono font-bold text-zinc-900 dark:text-white">
              <Radio
                className={`w-3.5 h-3.5 ${socketConnected ? "text-lime-500" : "text-amber-500"
                  }`}
              />
              {socketConnected ? "SOCKET.IO LIVE" : "REST POLLING"}
            </span>
            <span className="text-zinc-300 dark:text-zinc-700">|</span>
            <span className="font-mono text-zinc-600 dark:text-zinc-300">
              Process: <strong className="uppercase text-zinc-900 dark:text-white">{trainStatus}</strong>
            </span>
            {resources?.cpu && (
              <>
                <span className="text-zinc-300 dark:text-zinc-700">|</span>
                <span className="font-mono text-zinc-500">
                  CPU: {resources.cpu.percent}% • RAM: {resources.ram?.used_gb}/{resources.ram?.total_gb} GB
                </span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={fetchCoreData}
              className="w-10 h-10 rounded-full bg-white dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-800 flex items-center justify-center text-zinc-600 dark:text-zinc-300 hover:text-zinc-900 shadow-sm transition"
              title="Sync Live Filesystem & Telemetry"
            >
              <RefreshCw className="w-4 h-4" />
            </button>

            <button
              onClick={() => setActiveTab("settings")}
              className="w-10 h-10 rounded-full bg-white dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-800 flex items-center justify-center text-zinc-600 dark:text-zinc-300 hover:text-zinc-900 shadow-sm transition"
              title="Platform Settings"
            >
              <Settings className="w-4 h-4" />
            </button>

            {resources?.gpu && (
              <div className="hidden md:flex items-center gap-2.5 bg-white dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-800 rounded-full pl-2 pr-4 py-1.5 shadow-sm">
                <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-lime-500 to-emerald-500 text-white font-bold text-xs flex items-center justify-center font-mono">
                  K1
                </div>
                <div className="leading-tight">
                  <div className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                    {resources.gpu.name}
                  </div>
                  <div className="text-[10px] text-zinc-400 font-mono">
                    VRAM: {resources.gpu.vram_used_gb} / {resources.gpu.vram_total_gb} GB ({resources.gpu.utilization_percent}% util)
                  </div>
                </div>
              </div>
            )}

            <button
              onClick={() => setActiveTab("training")}
              className="flex items-center gap-2 bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800 border border-zinc-200/90 dark:border-zinc-800 rounded-full px-4 py-2.5 text-xs font-semibold text-zinc-900 dark:text-white shadow-sm transition"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Training Control</span>
            </button>
          </div>
        </header>

        <main className="flex-1">
          {activeTab === "overview" && (
            <OverviewTab
              data={dashboardData}
              onNavigate={(tab) => setActiveTab(tab)}
              onJumpToPov={handleJumpToCameraPov}
              onDiscardSim={async () => {
                await apiFetch("/api/simulation/discard", { method: "POST" });
                fetchCoreData();
              }}
            />
          )}
          {activeTab === "training" && (
            <TrainingCenterTab
              trainingState={trainingState}
              logs={trainingLogs}
              resources={resources}
              onRefresh={fetchCoreData}
            />
          )}
          {activeTab === "simulation" && (
            <SimulationViewerTab
              trajectory={trajectory}
              checkpoints={checkpoints}
              selectedMotion={selectedMotion}
              onSelectMotion={(m) => setSelectedMotion(m)}
              selectedCheckpoint={selectedCheckpoint}
              onSelectCheckpoint={(path) => {
                setSelectedCheckpoint(path);
                if (path.toLowerCase().includes("dance")) {
                  setSelectedMotion("data/motions/k1_dance.npz");
                } else if (path.toLowerCase().includes("kick")) {
                  setSelectedMotion("data/motions/k1_kick.npz");
                }
              }}
              initialCameraPov={initialCameraPov}
            />
          )}
          {activeTab === "motion" && <MotionAnalysisTab trajectory={trajectory} />}
          {activeTab === "analytics" && <AnalyticsTab />}
          {activeTab === "videos" && (
            <VideoGenerationTab
              videos={videos}
              checkpoints={checkpoints}
              onRefresh={fetchCoreData}
            />
          )}
          {activeTab === "experiments" && (
            <ExperimentsTab
              experiments={dashboardData?.experiments || []}
              onRefresh={fetchCoreData}
              onSelectCheckpointForSim={handleSelectCheckpointForSim}
            />
          )}
          {activeTab === "checkpoints" && (
            <CheckpointsTab
              checkpoints={checkpoints}
              onSelectForSim={handleSelectCheckpointForSim}
            />
          )}
          {activeTab === "settings" && (
            <SettingsTab theme={theme} onToggleTheme={toggleTheme} />
          )}
        </main>
      </div>
    </div>
  );
}
