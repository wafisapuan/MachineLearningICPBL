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
    """Validate episode structure and timestamp consistency."""
    rows = []

    # Use the first episode as the reference schema
    reference_columns = next(iter(episodes.values())).columns.tolist()

    for eid, df in episodes.items():

        # Check column structure
        correct_columns = df.columns.tolist() == reference_columns

        # Check missing values
        missing_values = int(df.isna().sum().sum())

        # Check timestamps
        timestamps = df["timestamp"]
        time_increasing = bool(timestamps.is_monotonic_increasing)

        # Check 30 FPS
        time_diff = timestamps.diff().dropna()
        correct_fps = bool(
            len(time_diff) > 0
            and (time_diff.sub(1 / 30).abs() < 1e-6).all()
        )

        rows.append({
            "episode": eid,
            "n_frames": len(df),
            "duration_s": timestamps.iloc[-1] - timestamps.iloc[0],
            "n_missing": missing_values,
            "n_columns": df.shape[1],
            "same_schema": correct_columns,
            "time_increasing": time_increasing,
            "correct_30fps": correct_fps,
        })

    return pd.DataFrame(rows)

