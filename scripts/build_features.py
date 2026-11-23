"""Build the shared feature table from the training data.

    python scripts/build_features.py            # -> results/features.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_RAW, FEATURES_CSV  # noqa: E402
from src.features import build_table  # noqa: E402
from src.load import load_folder  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_RAW))
    p.add_argument("--out", default=str(FEATURES_CSV))
    args = p.parse_args()

    episodes, labels = load_folder(args.data)
    table = build_table(episodes, labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} episodes × {table.shape[1] - 2} features -> {args.out}")
    print(table.groupby("label").median(numeric_only=True).round(3).T)
