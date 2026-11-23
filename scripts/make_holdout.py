"""Move 8 episodes (4 success + 4 fail) from data/raw to data/holdout. Run ONCE.

The split is deterministic (fixed seed), so every member gets the same hold-out set.
The chosen ids are saved to docs/holdout_ids.txt (committed) for reference.

    python scripts/make_holdout.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_HOLDOUT, DATA_RAW, HOLDOUT_IDS, RANDOM_STATE  # noqa: E402

PER_CLASS = 4


def main():
    if any(DATA_HOLDOUT.glob("*/*.csv")):
        print(f"Hold-out already exists in {DATA_HOLDOUT}; nothing to do.")
        return
    rng = np.random.default_rng(RANDOM_STATE)
    chosen = []
    for label in ["success", "fail"]:
        files = sorted((DATA_RAW / label).glob("*.csv"))
        if len(files) < PER_CLASS:
            sys.exit(f"Not enough files in {DATA_RAW / label}. Did you put the dataset in data/raw/?")
        pick = rng.choice(len(files), PER_CLASS, replace=False)
        (DATA_HOLDOUT / label).mkdir(parents=True, exist_ok=True)
        for i in sorted(pick):
            shutil.move(str(files[i]), DATA_HOLDOUT / label / files[i].name)
            chosen.append(f"{label}/{files[i].name}")
    HOLDOUT_IDS.write_text("\n".join(chosen) + "\n")
    print("Hold-out episodes:\n  " + "\n  ".join(chosen))
    print(f"Saved list to {HOLDOUT_IDS}")


if __name__ == "__main__":
    main()
