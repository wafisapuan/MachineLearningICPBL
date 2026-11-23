"""Success/fail classification. Owner: Imran (final model due Dec 8)

Experiments go in notebooks/imran_*.ipynb; the FINAL model code lives here.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_STATE

MODEL_PATH = MODELS_DIR / "classifier.joblib"


def make_model() -> Pipeline:
    """Baseline. TODO (Imran): replace with the best model after comparison."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def train(features: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    model = make_model().fit(features[feature_cols], features["label"])
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    return model


def predict(features: pd.DataFrame) -> pd.DataFrame:
    """Returns episode, predicted_label, success_prob."""
    bundle = joblib.load(MODEL_PATH)
    X = features[bundle["feature_cols"]]
    prob = bundle["model"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "episode": features["episode"],
        "predicted_label": ["success" if p >= 0.5 else "fail" for p in prob],
        "success_prob": prob.round(3),
    })
