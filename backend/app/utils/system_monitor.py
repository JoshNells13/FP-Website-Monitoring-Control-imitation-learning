from __future__ import annotations

import pathlib
import subprocess
import time
from typing import Any

import psutil

from config import HIL_ROOT


def get_system_resources() -> dict[str, Any]:
    """Return real-time CPU, RAM, GPU, and VRAM resource telemetry."""
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_count_logical = psutil.cpu_count(logical=True) or 1
    cpu_count_physical = psutil.cpu_count(logical=False) or 1
    vm = psutil.virtual_memory()

    ram_total_gb = round(vm.total / (1024**3), 2)
    ram_used_gb = round(vm.used / (1024**3), 2)
    ram_percent = round(vm.percent, 1)

    gpu_info: dict[str, Any] = {
        "available": False,
        "name": "CPU Only",
        "utilization_percent": 0.0,
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "vram_percent": 0.0,
        "temperature_c": None,
        "power_draw_w": None,
    }

    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                name = parts[0]
                util = float(parts[1]) if parts[1].replace(".", "", 1).isdigit() else 0.0
                mem_used = float(parts[2]) if parts[2].replace(".", "", 1).isdigit() else 0.0
                mem_total = float(parts[3]) if parts[3].replace(".", "", 1).isdigit() else 1.0
                temp = float(parts[4]) if len(parts) > 4 and parts[4].replace(".", "", 1).isdigit() else None
                power = float(parts[5]) if len(parts) > 5 and parts[5].replace(".", "", 1).isdigit() else None
                gpu_info = {
                    "available": True,
                    "name": name,
                    "utilization_percent": round(util, 1),
                    "vram_used_mb": round(mem_used, 1),
                    "vram_total_mb": round(mem_total, 1),
                    "vram_used_gb": round(mem_used / 1024.0, 2),
                    "vram_total_gb": round(mem_total / 1024.0, 2),
                    "vram_percent": round(100.0 * mem_used / max(mem_total, 1.0), 1),
                    "temperature_c": temp,
                    "power_draw_w": power,
                }
    except Exception:
        pass

    return {
        "timestamp": time.time(),
        "cpu": {
            "percent": round(cpu_percent, 1),
            "cores_physical": cpu_count_physical,
            "cores_logical": cpu_count_logical,
        },
        "ram": {
            "used_gb": ram_used_gb,
            "total_gb": ram_total_gb,
            "percent": ram_percent,
        },
        "gpu": gpu_info,
    }


def safe_resolve_in_hil(rel_or_abs_path: str) -> pathlib.Path:
    """Resolve a path and ensure it remains strictly within HIL_ROOT."""
    p = pathlib.Path(rel_or_abs_path)
    if not p.is_absolute():
        resolved = (HIL_ROOT / p).resolve()
    else:
        resolved = p.resolve()

    hil_resolved = HIL_ROOT.resolve()
    if resolved != hil_resolved and hil_resolved not in resolved.parents:
        raise ValueError(f"Path '{rel_or_abs_path}' is outside the allowed repository directory.")
    return resolved
