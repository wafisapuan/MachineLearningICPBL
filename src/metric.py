"""0–100 quality score. Owner: Wafi (due Dec 8)

score = 100 × ( w1·success_prob + w2·efficiency + w3·smoothness + w4·release_quality )

Each term is scaled to 0–1 with min/max values saved from the TRAINING data,
so new episodes are scored consistently. Every term and weight must be justified
in results/metric.md (we submit the formula).
"""
import json

import numpy as np
import pandas as pd

from src.config import MODELS_DIR

SCALING_PATH = MODELS_DIR / "metric_scaling.json"

# TODO (Wafi): finalize weights and justify each one
WEIGHTS = {
    "success_prob": 0.5,
    "efficiency": 0.2,
    "smoothness": 0.15,
    "release_quality": 0.15,
}

# Which feature drives each term, and whether higher is better
# TODO (Wafi): choose based on Imran's top features and Elisya's clusters
TERMS = {
    "efficiency": ("duration", "lower"),
    "smoothness": ("max_speed", "lower"),          # placeholder: replace with jerk when ready
    "release_quality": ("carry_time", "lower"),    # placeholder: replace with release_speed
}


def fit_scaling(train_features: pd.DataFrame) -> dict:
    """Save min/max of each term's feature from the training data."""
    scaling = {feat: [float(train_features[feat].min()), float(train_features[feat].max())]
               for feat, _ in TERMS.values()}
    SCALING_PATH.write_text(json.dumps(scaling, indent=2))
    return scaling


def _to_unit(x, lo, hi, direction):
    u = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
    return 1 - u if direction == "lower" else u


def score(features: pd.DataFrame, success_prob: pd.Series) -> pd.Series:
    """Return a 0–100 score per episode."""
    scaling = json.loads(SCALING_PATH.read_text())
    total = WEIGHTS["success_prob"] * success_prob.to_numpy()
    for term, (feat, direction) in TERMS.items():
        lo, hi = scaling[feat]
        total = total + WEIGHTS[term] * _to_unit(features[feat].to_numpy(), lo, hi, direction)
    return pd.Series((100 * total).round(1), index=features.index, name="score")
