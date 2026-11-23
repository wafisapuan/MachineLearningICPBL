"""Feature extraction: one episode -> one row of numbers.

Owner: Akif (core features + merging). Everyone adds feature functions in their section.

How to add a feature:
  1. Write `def my_feature(df, ph) -> float` in your section
     (`df` = episode DataFrame, `ph` = output of detect_phases(df))
  2. Add it to FEATURE_FUNCTIONS at the bottom
  3. Run `python scripts/build_features.py`
"""
import numpy as np
import pandas as pd

from src.config import ARM_JOINTS, FPS, action, state
from src.phases import detect_phases


# ---------- helpers ----------
def arm_speed(df: pd.DataFrame) -> np.ndarray:
    """Summed absolute joint velocity of the left arm, per frame (rad/s)."""
    arm = df[state(ARM_JOINTS)].to_numpy()
    return np.abs(np.diff(arm, axis=0)).sum(axis=1) * FPS


def t(df, idx):
    return float(df["timestamp"].iloc[idx]) if idx is not None else np.nan


# =====================================================================
# Akif: core features
# =====================================================================
def duration(df, ph):
    return t(df, -1)


def reach_time(df, ph):
    return t(df, ph["grasp_idx"])


def carry_time(df, ph):
    if ph["grasp_idx"] is None:
        return np.nan
    return t(df, ph["release_idx"]) - t(df, ph["grasp_idx"])


def mean_speed(df, ph):
    return float(arm_speed(df).mean())


def max_speed(df, ph):
    return float(arm_speed(df).max())


def track_err(df, ph):
    s = df[state(ARM_JOINTS)].to_numpy()
    a = df[action(ARM_JOINTS)].to_numpy()
    return float(np.abs(s - a).mean())

# TODO (Akif): return_time, reach_ratio, carry_speed, path_length, max_track_err,
#              joint angles at grasp / release


# =====================================================================
# Imran: success/fail separating features (due Nov 30)
# =====================================================================
def n_grasps(df, ph):
    return ph["n_grasps"]

# TODO (Imran): wr_p_at_release, wr_r_at_release


# =====================================================================
# Elisya: motion quality features (due Nov 30)
# =====================================================================
# TODO (Elisya): jerk, n_pauses, speed_variability


# =====================================================================
# Wafi: release quality features (due Nov 30)
# =====================================================================
# TODO (Wafi): release_speed, still_before_release, open_speed


# =====================================================================
# Registry: add your function here
# =====================================================================
FEATURE_FUNCTIONS = [
    duration, reach_time, carry_time, mean_speed, max_speed, track_err,  # Akif
    n_grasps,                                                            # Imran
    # Elisya
    # Wafi
]


def extract_features(df: pd.DataFrame) -> dict:
    ph = detect_phases(df)
    return {f.__name__: f(df, ph) for f in FEATURE_FUNCTIONS}


def build_table(episodes: dict, labels: dict | None = None) -> pd.DataFrame:
    """episodes: {id: DataFrame} -> feature table (one row per episode)."""
    rows = []
    for eid, df in episodes.items():
        row = {"episode": eid, **extract_features(df)}
        if labels is not None:
            row["label"] = labels.get(eid)
        rows.append(row)
    return pd.DataFrame(rows)
