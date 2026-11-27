"""Quick checks on synthetic data (no real dataset needed). Run: pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LEFT_JOINTS, SEQ_LEN, action, state  # noqa: E402
from src.features import FEATURE_FUNCTIONS, build_table  # noqa: E402
from src.phases import detect_phases  # noqa: E402
from src.sequences import to_fixed_length  # noqa: E402
from src.sequences import build_sequences


def fake_episode(n=300, grasp=(100, 200)) -> pd.DataFrame:
    """A smooth fake episode where the hand closes between frames grasp[0] and grasp[1]."""
    rng = np.random.default_rng(0)
    t = np.arange(n) / 30
    df = pd.DataFrame({"timestamp": t, "frame_index": np.arange(n)})
    for c in state(LEFT_JOINTS) + action(LEFT_JOINTS):
        df[c] = np.sin(t + rng.uniform(0, 3)) * 0.3
    for c in ["l_hand_index", "l_hand_middle", "l_hand_ring", "l_hand_pinky"]:
        g = np.zeros(n)
        g[grasp[0]:grasp[1]] = 1.57
        df["state." + c] = g
        df["action." + c] = g
    return df


def test_phases_detected():
    ph = detect_phases(fake_episode())
    assert ph["grasp_idx"] == 100
    assert ph["release_idx"] == 199
    assert ph["n_grasps"] == 1


def test_hand_never_closes():
    ph = detect_phases(fake_episode(grasp=(0, 0)))
    assert ph["grasp_idx"] is None and ph["n_grasps"] == 0


def test_feature_table_shape():
    eps = {"a": fake_episode(300), "b": fake_episode(420)}
    table = build_table(eps, {"a": 1, "b": 0})
    assert len(table) == 2
    assert table.shape[1] == len(FEATURE_FUNCTIONS) + 2   # + episode, label


def test_fixed_length():
    assert to_fixed_length(fake_episode(377)).shape == (SEQ_LEN, len(LEFT_JOINTS))

def test_build_sequences():
    eps = {
        "a": fake_episode(300),
        "b": fake_episode(420)
    }

    X, ids = build_sequences(eps)

    assert X.shape == (2, SEQ_LEN, len(LEFT_JOINTS))
    assert ids == ["a", "b"]
    assert np.isfinite(X).all()