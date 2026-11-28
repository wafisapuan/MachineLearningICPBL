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

from src.config import ARM_JOINTS, FINGER_JOINTS, FPS, action, state
from src.phases import detect_phases


# ---------- helpers ----------
def arm_speed(df: pd.DataFrame) -> np.ndarray:
    """Summed absolute joint velocity of the left arm, per frame (rad/s)."""
    if len(df) < 2:
        return np.array([])

    arm = df[state(ARM_JOINTS)].to_numpy()
    return np.abs(np.diff(arm, axis=0)).sum(axis=1) * FPS


def t(df: pd.DataFrame, idx) -> float:
    """Elapsed time in seconds from the first frame."""
    if idx is None or df.empty:
        return np.nan

    return float(
        df["timestamp"].iloc[idx] - df["timestamp"].iloc[0]
    )

# =====================================================================
# Akif: core features
# =====================================================================
def duration(df, ph):
    return float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])


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

def return_track_err(df, ph):
    """Mean absolute command-following error during the return phase."""
    if ph["open_idx"] is None:
        return np.nan

    start = ph["open_idx"]

    s = df[state(ARM_JOINTS)].iloc[start:].to_numpy()
    a = df[action(ARM_JOINTS)].iloc[start:].to_numpy()

    return float(np.abs(s - a).mean())

def return_time(df, ph):
    if ph["open_idx"] is None:
        return np.nan

    return t(df, -1) - t(df, ph["open_idx"])


def reach_ratio(df, ph):
    total = duration(df, ph)

    if total <= 0 or ph["grasp_idx"] is None:
        return np.nan

    return reach_time(df, ph) / total


def carry_speed(df, ph):
    start = ph["grasp_idx"]
    end = ph["release_idx"]

    if start is None or end is None:
        return np.nan

    speeds = arm_speed(df)
    phase_speeds = speeds[start:end]

    if len(phase_speeds) == 0:
        return np.nan

    return float(phase_speeds.mean())


def path_length(df, ph):
    if len(df) < 2:
        return 0.0

    arm = df[state(ARM_JOINTS)].to_numpy()
    movement = np.abs(np.diff(arm, axis=0))

    return float(movement.sum())


def max_track_err(df, ph):
    s = df[state(ARM_JOINTS)].to_numpy()
    a = df[action(ARM_JOINTS)].to_numpy()

    return float(np.abs(s - a).max())

def shoulder_roll_movement(df, ph):
    """Total absolute shoulder roll movement during grasp-to-release."""
    if ph["grasp_idx"] is None or ph["release_idx"] is None:
        return np.nan

    col = state(["l_sh_r"])[0]

    start = ph["grasp_idx"]
    end = ph["release_idx"]

    movement = df[col].iloc[start:end+1].diff().abs().sum()

    return float(movement)

# =====================================================================
# Imran: success/fail separating features (due Nov 30)
# =====================================================================
def n_grasps(df, ph):
    return ph["n_grasps"]


def grip_hold_mean(df, ph):
    """Mean finger closure (rad) while holding the bottle (grasp -> release).
    The fingers stop where they touch the bottle, so this tells us how/where it was gripped:
    fingers closing further = a thinner part or a looser grip. Higher in failed episodes."""
    if ph["grasp_idx"] is None:
        return np.nan
    fingers = df[state(FINGER_JOINTS)].iloc[ph["grasp_idx"]:ph["release_idx"] + 1]
    return float(fingers.to_numpy().mean())


def sh_r_at_grasp(df, ph):
    """Shoulder roll (rad) at the moment the hand closes: where the arm was when it grabbed.
    Higher in successful episodes."""
    if ph["grasp_idx"] is None:
        return np.nan
    return float(df[state(["l_sh_r"])[0]].iloc[ph["grasp_idx"]])


def carry_err_max(df, ph):
    """Largest command-following error of the arm while carrying (sum over 7 joints, rad).
    A spike means the arm lagged or was pushed off its commanded path while loaded.
    Higher in successful episodes."""
    if ph["grasp_idx"] is None:
        return np.nan
    a, b = ph["grasp_idx"], ph["release_idx"] + 1
    err = np.abs(df[state(ARM_JOINTS)].iloc[a:b].to_numpy() - df[action(ARM_JOINTS)].iloc[a:b].to_numpy())
    return float(err.sum(axis=1).max())

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
    duration,reach_time,carry_time,return_time,reach_ratio,mean_speed,max_speed,carry_speed,path_length,track_err,max_track_err,  # Akif
    n_grasps, grip_hold_mean, sh_r_at_grasp, carry_err_max,             # Imran
                                                                    # Elisya
                                                                    # Wafi
    #additional features
    shoulder_roll_movement,return_track_err,
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
