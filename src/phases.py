"""Phase detection (reach -> grasp -> carry -> release -> return). Owner: Akif"""
import numpy as np
import pandas as pd

from src.config import FINGER_JOINTS, GRIP_CLOSED, state


def grip_signal(df: pd.DataFrame) -> np.ndarray:
    """Mean closure of the 4 left fingers (0 = open, ~1.57 = closed)."""
    return df[state(FINGER_JOINTS)].mean(axis=1).to_numpy()


def detect_phases(df: pd.DataFrame) -> dict:
    """Return frame indices of key moments.

    grasp_idx:   first frame the hand is closed
    release_idx: last frame the hand is closed
    n_grasps:    number of separate closings (1 = normal, >1 = regrasp)

    TODO (Akif): handle edge cases more carefully, check on plots.
    """
    closed = grip_signal(df) > GRIP_CLOSED
    idx = np.flatnonzero(closed)
    if len(idx) == 0:  # hand never closed
        return {"grasp_idx": None, "release_idx": None, "n_grasps": 0, "end_idx": len(df) - 1}

    n_grasps = int(np.sum(np.diff(closed.astype(int)) == 1) + closed[0])
    return {
        "grasp_idx": int(idx[0]),
        "release_idx": int(idx[-1]),
        "n_grasps": n_grasps,
        "end_idx": len(df) - 1,
    }
