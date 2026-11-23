"""Loading episodes. Owner: Akif"""
from pathlib import Path

import pandas as pd

from src.config import LABELS


def episode_id(path: Path) -> str:
    """'episode_000012.csv' -> '000012'"""
    return Path(path).stem.replace("episode_", "")


def load_folder(folder) -> tuple[dict[str, pd.DataFrame], dict[str, int | None]]:
    """Load every CSV in a folder.

    Works for both layouts:
      - labeled:   folder/success/*.csv, folder/fail/*.csv  -> labels 1/0
      - unlabeled: folder/*.csv (e.g. demo day)              -> labels None

    Returns (episodes, labels), both keyed by episode id.
    """
    folder = Path(folder)
    episodes, labels = {}, {}
    labeled = any((folder / name).is_dir() for name in LABELS)

    if labeled:
        for name, value in LABELS.items():
            for f in sorted((folder / name).glob("*.csv")):
                episodes[episode_id(f)] = pd.read_csv(f)
                labels[episode_id(f)] = value
    else:
        for f in sorted(folder.glob("*.csv")):
            episodes[episode_id(f)] = pd.read_csv(f)
            labels[episode_id(f)] = None

    if not episodes:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return episodes, labels


def check_episodes(episodes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quick sanity report: length, duration, missing values, column count."""
    rows = []
    for eid, df in episodes.items():
        rows.append({
            "episode": eid,
            "n_frames": len(df),
            "duration_s": df["timestamp"].iloc[-1],
            "n_missing": int(df.isna().sum().sum()),
            "n_columns": df.shape[1],
            "time_increasing": bool(df["timestamp"].is_monotonic_increasing),
        })
    return pd.DataFrame(rows)
