from __future__ import annotations

import os
import pathlib

BACKEND_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
HIL_ROOT = (WORKSPACE_ROOT / "human-imitation-learning").resolve()
os.environ.setdefault("MPLCONFIGDIR", str((HIL_ROOT / ".cache" / "matplotlib").resolve()))
os.environ.setdefault("WARP_CACHE_PATH", str((HIL_ROOT / ".cache" / "warp").resolve()))

PYTHON_EXE = HIL_ROOT / ".venv" / "Scripts" / "python.exe"
SCRIPTS_DIR = HIL_ROOT / "scripts"
DATA_DIR = HIL_ROOT / "data"
MOTIONS_DIR = DATA_DIR / "motions"
RETARGETED_DIR = DATA_DIR / "retargeted"
BVH_DIR = DATA_DIR / "bvh"
LOGS_DIR = HIL_ROOT / "logs" / "rsl_rl" / "k1_tracking"
OUTPUTS_DIR = HIL_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
K1_ASSETS_DIR = HIL_ROOT / "third_party" / "GMR" / "assets" / "booster_k1"
K1_MESHES_DIR = K1_ASSETS_DIR / "meshes"
K1_XML_PATH = K1_ASSETS_DIR / "k1_mjlab.xml"


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "booster-k1-beyondmimic-mjlab-ephemeral")
    STATELESS_MODE = True
    HIL_ROOT = str(HIL_ROOT)
    PYTHON_EXE = str(PYTHON_EXE)
