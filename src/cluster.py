"""Clustering into human-readable attempt types. Owner: Elisya (final model due Dec 8)

Experiments go in notebooks/elisya_*.ipynb; the FINAL model code lives here.
Do NOT use the label to build clusters, only to describe them afterwards.
"""
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_STATE

MODEL_PATH = MODELS_DIR / "cluster.joblib"

# TODO (Elisya): fill in after analysing the clusters (number -> evidence-based name)
CLUSTER_NAMES = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
}


def make_model(k: int = 3) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("km", KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)),
    ])


def fit(features: pd.DataFrame, feature_cols: list[str], k: int = 3) -> Pipeline:
    model = make_model(k).fit(features[feature_cols])
    joblib.dump({"model": model, "feature_cols": feature_cols, "names": CLUSTER_NAMES}, MODEL_PATH)
    return model


def assign(features: pd.DataFrame) -> pd.DataFrame:
    """Returns episode, cluster_id, cluster_name."""
    bundle = joblib.load(MODEL_PATH)
    ids = bundle["model"].predict(features[bundle["feature_cols"]])
    return pd.DataFrame({
        "episode": features["episode"],
        "cluster_id": ids,
        "cluster_name": [bundle["names"].get(i, f"Cluster {i}") for i in ids],
    })
