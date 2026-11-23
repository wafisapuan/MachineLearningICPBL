"""Shared constants. Import from here; never redefine these elsewhere."""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_HOLDOUT = ROOT / "data" / "holdout"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FEATURES_CSV = RESULTS_DIR / "features.csv"
HOLDOUT_IDS = ROOT / "docs" / "holdout_ids.txt"

# ---------- Reproducibility ----------
RANDOM_STATE = 42

# ---------- Data ----------
FPS = 30
LABELS = {"success": 1, "fail": 0}

# Left side does the task (right arm barely moves, right hand never closes)
ARM_JOINTS = ["l_sh_p", "l_sh_r", "l_sh_y", "l_el_p", "l_wr_y", "l_wr_p", "l_wr_r"]
FINGER_JOINTS = ["l_hand_index", "l_hand_middle", "l_hand_ring", "l_hand_pinky"]
THUMB_JOINTS = ["l_hand_thumb_y", "l_hand_thumb_p"]
HAND_JOINTS = FINGER_JOINTS + THUMB_JOINTS
LEFT_JOINTS = ARM_JOINTS + HAND_JOINTS          # 13 channels

def state(cols):
    return ["state." + c for c in cols]

def action(cols):
    return ["action." + c for c in cols]

# ---------- Phase detection ----------
GRIP_CLOSED = 0.8          # mean finger angle (rad) above which the hand counts as closed

# ---------- Sequences (LSTM, optional) ----------
SEQ_LEN = 300
