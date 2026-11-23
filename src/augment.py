"""Data augmentation for raw episodes. Owner: Imran (due Dec 2)

Rules:
  - Augment ONLY the training part of each CV fold.
  - Never augment validation, hold-out, or demo data.
  - Keep the original episode id with every copy (for GroupKFold).
  - Keep changes small: a fail must not start to look like a success.
"""
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)


def jitter(df: pd.DataFrame, sigma_frac: float = 0.01) -> pd.DataFrame:
    """Add Gaussian noise (sigma = sigma_frac × each joint's range)."""
    raise NotImplementedError("TODO (Imran)")


def speed_change(df: pd.DataFrame, low: float = 0.9, high: float = 1.1) -> pd.DataFrame:
    """Resample the episode to 90–110% of its length (update timestamp too)."""
    raise NotImplementedError("TODO (Imran)")


def time_shift(df: pd.DataFrame, max_frames: int = 15) -> pd.DataFrame:
    """Trim a few frames at the start/end."""
    raise NotImplementedError("TODO (Imran)")


def magnitude_scale(df: pd.DataFrame, low: float = 0.95, high: float = 1.05) -> pd.DataFrame:
    """Scale joint motion around its mean."""
    raise NotImplementedError("TODO (Imran)")


def augment(df: pd.DataFrame, k: int) -> list[pd.DataFrame]:
    """Return k randomly augmented copies of one episode."""
    raise NotImplementedError("TODO (Imran)")


def augment_training_set(episodes: dict, labels: dict, train_ids: list, k: int):
    """Augment only train_ids. Returns (episodes, labels, groups) where groups
    maps every (new) id to its ORIGINAL episode id, for GroupKFold."""
    raise NotImplementedError("TODO (Imran)")
