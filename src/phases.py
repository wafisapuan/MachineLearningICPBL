
"""Phase detection (reach -> grasp -> carry -> release -> return).
Owner: Akif
"""

import numpy as np
import pandas as pd

from src.config import FINGER_JOINTS, GRIP_CLOSED, state


def grip_signal(df: pd.DataFrame) -> np.ndarray:
    """Mean closure of the 4 left fingers."""

    columns = state(FINGER_JOINTS)

    # Check required columns
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing finger columns: {missing}")

    # Convert to numeric and reject missing/invalid values
    fingers = df[columns].apply(pd.to_numeric, errors="coerce")

    if fingers.isna().any().any():
        raise ValueError("Invalid or missing finger values detected")

    return fingers.mean(axis=1).to_numpy()


def detect_phases(df: pd.DataFrame) -> dict:
    """Detect grasp and release events."""

    grip = grip_signal(df)
    closed = grip > GRIP_CLOSED

    # Detect all transitions
    changes = np.diff(closed.astype(int))

    grasp_indices = np.flatnonzero(changes == 1) + 1

    if closed[0]:
        grasp_indices = np.insert(grasp_indices, 0, 0)

    release_indices = np.flatnonzero(changes == -1)

    if len(grip) == 0 or not closed.any():
        return {
            "grasp_idx": None,
            "release_idx": None,
            "open_idx": None,
            "n_grasps": 0,
            "end_idx": len(df) - 1,
            "grasp_events": [],
        }

    # Pair each grasp with its corresponding release
    events = []

    for i, grasp_idx in enumerate(grasp_indices):

        next_releases = release_indices[
            release_indices >= grasp_idx
        ]

        if len(next_releases) > 0:
            release_idx = int(next_releases[0])
            open_idx = release_idx + 1
        else:
            release_idx = len(df) - 1
            open_idx = None

        events.append({
            "grasp_idx": int(grasp_idx),
            "release_idx": release_idx,
            "open_idx": open_idx,
        })

    first_event = events[0]
    last_event = events[-1]

    return {
        "grasp_idx": first_event["grasp_idx"],
        "release_idx": last_event["release_idx"],
        "open_idx": last_event["open_idx"],
        "n_grasps": len(events),
        "end_idx": len(df) - 1,
        "grasp_events": events,
    }
