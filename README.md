# Robot Pick-and-Place: Classify, Cluster & Score

**MMT3007 Machine Learning (Fall 2026), Hanyang University ERICA · IC-PBL Project**
Instructor: Prof. Yongjae Yoo · Industry partner: Aei Robot (Mr. Junwoo Yong)

We analyze joint-angle logs from the humanoid robot **Alice M1** performing a pick-and-place task (put a bottle into a box). Using only the CSV logs (no video), we:

1. **Classify** each episode as success (bottle upright) or fail (bottle tipped over)
2. **Cluster** episodes into human-readable types of attempts
3. **Score** each episode's quality on a 0–100 scale

---

## Team

| Member | Role | Owns |
|---|---|---|
| **Akif** | Data & Pipeline Lead | `src/load.py`, `src/phases.py`, `src/features.py`, `src/sequences.py`, `demo.py` |
| **Imran** | Classification & Augmentation Lead | `src/classify.py`, `src/augment.py` |
| **Elisya** | Clustering & Visualization Lead | `src/cluster.py`, figures, `demo.py` (with Akif) |
| **Wafi** | Metric Lead & Team Leader | `src/metric.py`, repo, reports |

Full role details: [`docs/team_roles_detailed.md`](docs/team_roles_detailed.md)

---

### Data split used everywhere
- **Test set (hold-out):** 8 episodes in `docs/holdout_ids.txt` (4 success, 4 fail). Never used for training, feature design or tuning. Used once for the practice demo (Dec 12).
- **Training + validation:** the other 40 episodes, evaluated with **repeated stratified 5-fold cross-validation**.

---

## Quick start

```bash
# 1. Clone and enter
git clone https://github.com/wafisapuan/MachineLearningICPBL
cd MachineLearningICPBL

# 2. Create environment and install
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Put ALL 48 episode CSVs in data/raw/  (see data/README.md)
#    data/raw/success/*.csv  (27)  and  data/raw/fail/*.csv  (21)

# 4. Set aside the test set: EVERYONE runs this once on their own laptop.
#    It moves the 8 episodes listed in docs/holdout_ids.txt into data/holdout/.
#    Run it again whenever you copy the dataset into data/raw again.
python scripts/make_holdout.py

# 5. Explore
python scripts/explore.py --data data/raw

# 6. Build features -> train models -> test the demo on the hold-out set
#    (results/features.csv is already committed; rebuild it only if you changed src/features.py)
python scripts/build_features.py
python scripts/train_models.py
python demo.py --data data/holdout

# 7. Run the quick checks
pytest
```

The whole pipeline already runs with **baseline** versions of every part (logistic regression, k-means with 3 clusters, placeholder score). Each owner replaces their baseline with the real thing; the pipeline keeps working.

### Run the demo (Dec 14)

```bash
python demo.py --data path/to/new_episodes/
```

Output: `results/demo_results.csv`

| episode | predicted_label | success_prob | cluster_name | score |
|---|---|---|---|---|
| test_01 | success | 0.91 | Clean & fast success | 87 |

---

## Repository structure

```
MachineLearningICPBL/
├── data/                  # NOT committed (see data/README.md)
│   ├── raw/               #   success/ and fail/ training episodes
│   └── holdout/           #   8 episodes set aside for the practice demo, never train on these
├── src/
│   ├── config.py          # Shared constants (joint names, thresholds, paths, seed)
│   ├── load.py            # Akif:  load CSV folders
│   ├── phases.py          # Akif:  detect grasp / release
│   ├── features.py        # Akif:  episode → one row of features (+ everyone's feature functions)
│   ├── sequences.py       # Akif:  fixed-length sequences for the LSTM
│   ├── augment.py         # Imran: data augmentation
│   ├── classify.py        # Imran: success/fail model
│   ├── cluster.py         # Elisya: clustering + cluster names
│   └── metric.py          # Wafi:  0–100 quality score
├── scripts/
│   ├── make_holdout.py    # One-time hold-out split
│   ├── build_features.py  # Builds results/features.csv
│   └── explore.py         # Exploration plots
├── notebooks/             # Personal experiments: <name>_<topic>.ipynb
├── models/                # Saved models used by demo.py
├── results/               # Tables and figures/
├── reports/               # Mentoring reports, final report
├── docs/                  # Project overview, role guide
├── tests/                 # Quick checks (pytest)
├── demo.py                # Full pipeline for demo day
└── requirements.txt
```

---

## Data

- 48 episodes (27 success / 21 fail), 30 fps, 9–17 s each
- Each CSV: 5 info columns + 26 `state.*` (measured) + 26 `action.*` (commanded) joint columns
- Only the **left arm** performs the task; finger closure marks grasp and release
- The dataset belongs to the course/industry partner: **do not commit it or share it publicly**

---

## Team workflow

See [`CONTRIBUTING.md`](CONTRIBUTING.md). In short:

1. Create a branch: `git checkout -b <name>/<topic>` (e.g. `imran/baseline-models`)
2. Commit small, clear changes
3. Open a pull request → Wafi reviews and merges into `main`
4. Never push directly to `main`, and never commit data

---

## Key dates

| Date | Milestone |
|---|---|
| Nov 24 | Guest lecture (Mr. Yong) |
| Nov 28 | `features.csv` v1 |
| Nov 30 | Everyone's feature functions → Akif |
| Dec 8 | All models saved |
| Dec 12 | Practice demo on hold-out set |
| **Dec 13, 23:59** | **Code freeze + upload** |
| **Dec 14, 14:30** | **Live demo** |
| Dec 16 | Repo + final report |
| Dec 17 | Individual reflective journal |
