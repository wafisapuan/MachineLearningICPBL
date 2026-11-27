"""Move the 8 test-set episodes (4 success + 4 fail) from data/raw to data/holdout.

Everyone runs this once on their own laptop, after putting all 48 episodes in data/raw.

The test set is fixed by docs/holdout_ids.txt (committed), so every member gets exactly
the same 8 episodes. Only if that file does not exist does the script pick them
(stratified, fixed seed) and create it.

    python scripts/make_holdout.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_HOLDOUT, DATA_RAW, HOLDOUT_IDS, RANDOM_STATE  # noqa: E402

PER_CLASS = 4


def read_list():
    """Committed list -> ['success/episode_000014.csv', ...]."""
    return [line.strip().replace("\\", "/") for line in HOLDOUT_IDS.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def pick_new():
    """Only used if docs/holdout_ids.txt does not exist yet."""
    rng = np.random.default_rng(RANDOM_STATE)
    chosen = []
    for label in ["success", "fail"]:
        files = sorted((DATA_RAW / label).glob("*.csv"))
        if len(files) < PER_CLASS:
            sys.exit(f"Not enough files in {DATA_RAW / label}. Did you put the dataset in data/raw/?")
        chosen += [f"{label}/{files[i].name}" for i in sorted(rng.choice(len(files), PER_CLASS, replace=False))]
    HOLDOUT_IDS.write_text("\n".join(chosen) + "\n")
    print(f"Created {HOLDOUT_IDS}")
    return chosen


def main():
    chosen = read_list() if HOLDOUT_IDS.exists() else pick_new()
    moved, already = 0, 0
    for rel in chosen:
        src, dst = DATA_RAW / rel, DATA_HOLDOUT / rel
        if dst.exists():
            already += 1
            continue
        if not src.exists():
            sys.exit(f"Missing {src}. Put all 48 episodes in data/raw/ first.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst)
        moved += 1
    print(f"Test set ({len(chosen)} episodes): {moved} moved to {DATA_HOLDOUT}, {already} already there.")
    print("  " + "\n  ".join(chosen))


if __name__ == "__main__":
    main()
