"""Demo day pipeline: new episodes -> label + cluster + score.

    python demo.py --data path/to/new_episodes/

Works with a flat folder of CSVs (no labels) or success/ + fail/ subfolders.
Never retrains: it only loads the models saved in models/.
Owners: Akif + Elisya
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import classify, cluster, metric  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.features import build_table  # noqa: E402
from src.load import load_folder  # noqa: E402


def run(data_dir: str) -> pd.DataFrame:
    episodes, labels = load_folder(data_dir)
    features = build_table(episodes)

    pred = classify.predict(features)
    clus = cluster.assign(features)
    out = pred.merge(clus, on="episode")
    out["score"] = metric.score(features, out["success_prob"]).to_numpy()

    if any(v is not None for v in labels.values()):  # labeled folder (e.g. hold-out): show truth too
        out["true_label"] = out["episode"].map(lambda e: {1: "success", 0: "fail"}.get(labels[e]))
    return out[[c for c in ["episode", "true_label", "predicted_label", "success_prob",
                            "cluster_name", "score"] if c in out.columns]]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Classify, cluster and score robot episodes.")
    p.add_argument("--data", required=True, help="Folder with episode CSVs")
    p.add_argument("--out", default=str(RESULTS_DIR / "demo_results.csv"))
    args = p.parse_args()

    start = time.time()
    results = run(args.data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    print("\n" + results.to_string(index=False))
    if "true_label" in results:
        acc = (results["true_label"] == results["predicted_label"]).mean()
        print(f"\nAccuracy on this folder: {acc:.0%}")
    print(f"\n{len(results)} episodes processed in {time.time() - start:.1f} s -> {args.out}")
