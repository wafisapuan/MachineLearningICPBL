"""Train all demo models from results/features.csv and save them to models/.

    python scripts/train_models.py

Uses the CURRENT versions of classify.py, cluster.py and metric.py, so it works
from day 1 with the baselines and keeps working as each owner improves their part.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import classify, cluster, metric  # noqa: E402
from src.config import FEATURES_CSV, MODELS_DIR  # noqa: E402

if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    F = pd.read_csv(FEATURES_CSV).dropna()
    feature_cols = [c for c in F.columns if c not in ("episode", "label")]

    classify.train(F, feature_cols)
    cluster.fit(F, feature_cols)
    metric.fit_scaling(F)
    print(f"Trained on {len(F)} episodes, {len(feature_cols)} features. Models saved to {MODELS_DIR}/")
