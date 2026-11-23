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

## Rules
1. **Never commit data** (`data/` is in `.gitignore`).
2. **Never train or tune on `data/holdout/`.**
3. Augmentation only inside training folds.
4. Use `RANDOM_STATE` from `src/config.py` everywhere.
5. Shared constants (joint names, thresholds) live in `src/config.py`; don't redefine them.
6. Notebooks are for experiments: name them `<name>_<topic>.ipynb`. Code used by the demo must live in `src/`.
7. Every function used by `demo.py` must work on a **new folder** of CSVs (no labels).

## Adding a feature (everyone)
1. Write a function in `src/features.py` under your name's section:
   ```python
   def release_speed(df: pd.DataFrame) -> float:
       """Mean arm speed within ±0.5 s of release."""
       ...
   ```
2. Register it in `FEATURE_FUNCTIONS` at the bottom of the file.
3. Run `python scripts/build_features.py` and check the new column.
4. Open a PR; Akif reviews.

## Commit messages
Short, present tense, what + why: `Add n_grasps feature to detect regrasping`
