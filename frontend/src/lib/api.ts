import { io, Socket } from "socket.io-client";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:5000";

let socketInstance: Socket | null = null;

export function getSocket(): Socket {
  if (!socketInstance) {
    socketInstance = io(API_BASE, {
      transports: ["websocket", "polling"],
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
    });
  }
  return socketInstance;
}

export async function apiFetch<T = any>(path: string, options?: RequestInit): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });
  if (!res.ok) {
    let errText = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      errText = data.error || data.message || errText;
    } catch {
      // ignore
    }
    throw new Error(errText);
  }
  return res.json();
}

export type CameraPOV =
  | "third_person"
  | "front"
  | "side"
  | "top"
  | "free"
  | "first_person";

export interface TrajectoryFrame {
  frame: number;
  time_sec: number;
  ref_joint_pos: number[];
  sim_joint_pos: number[];
  ref_joint_vel: number[];
  sim_joint_vel: number[];
  joint_error_rad: number[];
  ref_body_pos: number[][];
  ref_body_quat: number[][];
  sim_body_pos: number[][];
  sim_body_quat: number[][];
  body_error_mm: number[];
  world_body_error_mm: number[];
  frame_mae_rad: number;
  frame_rmpkpe_mm: number;
  frame_mpkpe_mm: number;
  frame_anchor_ori_err_deg: number;
  head_camera_pose: {
    pos: [number, number, number];
    quat_wxyz: [number, number, number, number];
  };
}

export interface JointSummary {
  index: number;
  name: string;
  group: string;
  mae_rad: number;
  mae_deg: number;
  rmse_rad: number;
  rmse_deg: number;
  max_error_rad: number;
  max_error_deg: number;
  vel_mae_rad_s: number;
  vel_rmse_rad_s: number;
  severity: "optimal" | "moderate" | "high";
}

export interface TrajectoryBundle {
  motion_file: string;
  checkpoint_file: string;
  fps: number;
  dt: number;
  frame_count: number;
  duration_sec: number;
  overall_metrics: {
    joint_mae_rad: number;
    joint_mae_deg: number;
    joint_rmse_rad: number;
    joint_rmse_deg: number;
    r_mpkpe_mm: number;
    mpkpe_mm: number;
    anchor_ori_err_rad: number;
    anchor_ori_err_deg: number;
  };
  per_joint_summary: JointSummary[];
  frames: TrajectoryFrame[];
}
