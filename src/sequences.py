"""Fixed-length sequences for the (optional) LSTM. Owner: Akif"""
import numpy as np
import pandas as pd

from src.config import LEFT_JOINTS, SEQ_LEN, state


def to_fixed_length(df: pd.DataFrame, n: int = SEQ_LEN) -> np.ndarray:
    """Resample the 13 left-side state channels to n steps -> array (n, 13)."""
    x = df[state(LEFT_JOINTS)].to_numpy()
    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, n)
    return np.stack([np.interp(new, old, x[:, j]) for j in range(x.shape[1])], axis=1)


def build_sequences(episodes: dict, n: int = SEQ_LEN) -> tuple[np.ndarray, list[str]]:
    """{id: DataFrame} -> (array (N, n, 13), list of ids)."""
    ids = list(episodes)
    return np.stack([to_fixed_length(episodes[i], n) for i in ids]), ids
