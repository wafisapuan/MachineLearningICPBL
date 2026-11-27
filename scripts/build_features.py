
"""Build the shared feature table from development data only.

    python scripts/build_features.py            # -> results/features.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_RAW, FEATURES_CSV  # noqa: E402
from src.features import build_table  # noqa: E402
from src.load import load_folder  # noqa: E402


def normalize_episode_id(value):
    """Convert episode_000001 or 000001 into the same ID format."""
    episode = Path(str(value)).stem

    if episode.startswith("episode_"):
        episode = episode.removeprefix("episode_")

    return episode


def read_holdout_ids(path):
    """Read holdout episode IDs from the text file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Holdout ID file not found: {path}"
        )

    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_RAW))
    parser.add_argument("--out", default=str(FEATURES_CSV))
    args = parser.parse_args()

    holdout_file = ROOT / "docs" / "holdout_ids.txt"
    holdout_ids = read_holdout_ids(holdout_file)

    # Load all episodes available in the selected data folder.
    episodes, labels = load_folder(args.data)

    # Identify which holdout episodes are present in the loaded data.
    found_holdout = {
        normalize_episode_id(eid)
        for eid in episodes
    } & holdout_ids

    # Keep development episodes only.
    dev_episodes = {
        eid: df
        for eid, df in episodes.items()
        if normalize_episode_id(eid) not in holdout_ids
    }

    dev_labels = {
        eid: labels.get(eid)
        for eid in dev_episodes
    }

    # Extract features from development episodes only.
    table = build_table(dev_episodes, dev_labels)

    # Validate the result before saving.
    if len(table) != 40:
        raise ValueError(
            f"Expected 40 development episodes, got {len(table)}. "
            "Check data/raw and docs/holdout_ids.txt."
        )

    output_ids = {
        normalize_episode_id(eid)
        for eid in table["episode"]
    }

    leaked_ids = output_ids & holdout_ids

    if leaked_ids:
        raise ValueError(
            f"Holdout leakage detected: {sorted(leaked_ids)}"
        )

    success_count = (table["label"] == 1).sum()
    fail_count = (table["label"] == 0).sum()

    if success_count != 23 or fail_count != 17:
        raise ValueError(
            f"Unexpected label counts: "
            f"{success_count} success, {fail_count} fail. "
            "Expected 23 success and 17 fail."
        )

    # Save only after all checks pass.
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    print(f"Loaded episodes: {len(episodes)}")
    print(f"Holdout IDs found and excluded: {len(found_holdout)}")
    print(f"Development episodes: {len(table)}")
    print(f"Success: {success_count}")
    print(f"Fail: {fail_count}")
    print(f"Holdout leakage: {len(leaked_ids)}")
    print(
        f"\n{len(table)} episodes × "
        f"{table.shape[1] - 2} features -> {output_path}"
    )

    print("\nFeature medians by label:")
    print(table.groupby("label").median(numeric_only=True).round(3).T)
