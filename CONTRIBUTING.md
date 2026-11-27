# How we work together

## Branches
- `main` is always working code. **Nobody pushes to `main` directly.**
- Make a branch per task: `<name>/<topic>`, e.g. `akif/features-v1`, `elisya/kmeans`, `wafi/metric-draft`

```bash
git checkout main
git pull
git checkout -b imran/baseline-models
# ... work ...
git add src/classify.py
git commit -m "Add logistic regression baseline with 5-fold CV"
git push -u origin imran/baseline-models
```
Then open a **pull request** on GitHub.

## Pull requests
- Wafi reviews and merges.
- Changes to `src/load.py`, `src/features.py`, `src/sequences.py`, or `demo.py` also need **Akif's** OK (code owner).
- Keep PRs small (one topic). Fill in the PR template.
- Before opening: run `pytest` and make sure it passes.

## What to commit (and what not)

| Commit ✅ | Never commit ❌ |
|---|---|
| Code (`src/`, `scripts/`, `tests/`, `demo.py`) | **Raw episode CSVs** (`data/raw/`, `data/holdout/`) |
| `docs/holdout_ids.txt` (the official test-set list) | Videos |
| `results/features.csv` (**Akif only**, see below) and other small result tables | Large notebook outputs (clear them first) |
| Reports (`reports/`) and figures used in the report | `.venv/`, `__pycache__/` |
| Saved models in `models/` (needed by `demo.py`) | |

"Data" in our rules means the **raw recordings from the course**. Tables and reports *we create* from them are fine to commit.

## Rules
1. **Never commit raw data** (`data/` is in `.gitignore`).
2. **Never train or tune on `data/holdout/`.**
3. Augmentation only inside training folds.
4. Use `RANDOM_STATE` from `src/config.py` everywhere.
5. Shared constants (joint names, thresholds) live in `src/config.py`; don't redefine them.
6. Notebooks are for experiments: name them `<name>_<topic>.ipynb` and clear large outputs before committing. Code used by the demo must live in `src/`.
7. Every function used by `demo.py` must work on a **new folder** of CSVs (no labels).
8. Always `git pull` on `main` before starting a new branch.
9. Never edit `docs/holdout_ids.txt`. The test set is fixed for the whole project.

## Adding a feature (everyone)
1. Write a function in `src/features.py` under your name's section:
   ```python
   def release_speed(df, ph):
       """Mean arm speed within ±0.5 s of release."""
       # df = one episode, ph = detect_phases(df) (grasp_idx, release_idx, ...)
       ...
   ```
2. Register it in `FEATURE_FUNCTIONS` at the bottom of the file.
3. Run `python scripts/build_features.py` and check the new column (locally only).
4. Commit **only `src/features.py`**. Do not commit `results/features.csv` in a feature PR: parallel PRs would always conflict on it.
   After merging, **Akif rebuilds and commits `results/features.csv`** on `main`.
5. Open a PR; Akif reviews.

## Commit messages
Short, present tense, what + why: `Add n_grasps feature to detect regrasping`
