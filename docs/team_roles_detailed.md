# Team Roles: Detailed Guide

**Project:** Classify, Cluster, and Evaluate Humanoid Robot Action Dataset (MMT3007, Prof. Yongjae Yoo)
**Code freeze:** Dec 13, 23:59 · **Demo:** Dec 14, 14:30 · **Repo + report:** Dec 16 · **Reflective journal:** Dec 17

| Member | Role | One-line summary |
|---|---|---|
| **Akif** | Data & Pipeline Lead | Turns raw CSVs into clean features and fixed-length sequences; builds the demo pipeline |
| **Imran** | Classification & Augmentation Lead | Predicts success/fail; **owns data augmentation**; tries an LSTM if time allows |
| **Elisya** | Clustering & Visualization Lead | Finds types of attempts, names them, makes the figures; co-builds the demo |
| **Wafi** | Metric Lead & Team Leader | Designs the 0–100 score; owns the repo, schedule, reports, and final report |

---

## 0. Read this first (everyone)

### 0.1 The data in one minute
- **48 episodes**: `success/` (27, bottle upright) and `fail/` (21, bottle tipped over)
- 30 frames/second, **9–17 seconds** long (≈270–520 rows), so **every episode has a different length**
- Columns: 5 info columns (`timestamp`, `frame_index`, …) + **26 `state.*`** (what the robot actually did) + **26 `action.*`** (what it was told to do)
- **Only the left arm (`l_`) does the task.** The right arm barely moves and its hand never closes.
- Left finger closure (`l_hand_index/middle/ring/pinky`) goes **0 → ~1.57** when grasping, so it marks **grasp** and **release**
- Phases: **reach → grasp → carry → release → return**
- First look: fails tend to be **slower** (median 11.4 s vs 10.4 s). Simple features give **~65% accuracy** vs 56% for always guessing "success", so we need **better features**.

### 0.2 Helpful lecture chapters (optional reference)

The lectures are **extra background, not a rulebook**. Use any method that fits the data and that you can justify, including ones outside the course. Chapters 8 and 11 were mentioned by the lecturer as especially relevant.

| Chapter | Can help with | Most useful for |
|---|---|---|
| **08 Data Augmentation** ⭐ | Noise injection, shifting, speed change for sequence data; augment **training set only**, validate on un-augmented data | Imran |
| **11 Deep Learning** ⭐ | Sequence models: TDNN, RNN, **LSTM** (textbook 12.7.4), the lecturer's suggested direction | Imran |
| 02 Supervised Learning | Train/validation/test, overfitting, model complexity | Imran |
| 03 Parametric Estimation | Gaussian assumptions, bias/variance | Imran, Wafi |
| 04 Linear Discriminant & Kernel Machines | Logistic regression, SVM | Imran |
| 05 Dimensionality Reduction (PCA) | Reduce features, 2D plots, PCA as augmentation | Elisya, Akif |
| 06 Clustering | k-means, EM / Gaussian mixtures | Elisya |
| 07 Decision Trees | Decision tree / random forest, feature importance | Imran, Wafi |
| 10 Multilayer Perceptron | MLP baseline before LSTM | Imran |

### 0.3 How the pieces connect
```
                ┌──► Augment (Ch 8) ──► LSTM on raw sequences (Ch 11) ──┐
48 CSVs ──► Explore                                                      ├──► Classify ─┐
                └──► Feature table ──┬──► Classic classifiers ──────────┘               │
                                     ├──► Clustering ───────────────────────────────────┼──► Score ──► demo.py
                                     └───────────────────────────────────────────────────┘
```

### 0.4 Team rules
1. **Hold-out set:** on day 1 Akif moves **8 episodes (4 success + 4 fail)** to `data/holdout/`. **Nobody trains or tunes on them.** They are our practice demo set.
2. **One feature table:** `features.csv` with columns `episode, label, <features…>`. Everyone may add features, but through Akif's `src/features.py`.
3. **Augmentation only inside training folds** (Ch 8 workflow). Never augment validation, hold-out, or demo data.
4. **`random_state=42`** everywhere.
5. **Git:** work on your own branch → pull request → Wafi merges into `main`. No direct pushes to `main`.
6. **Everything must run on a new folder of CSVs**, because that's what happens on demo day.
7. **Weekly check-in** (15 min): done / next / blocked.

### 0.5 Language & tools
- **Python 3.10+**
- `pandas`, `numpy`, `scipy`: data and signal processing
- `scikit-learn`: classic models, PCA, clustering, CV
- **`PyTorch`** (or `tensorflow.keras`): LSTM / MLP
- `matplotlib`, `seaborn`: plots · `joblib`: save models
- Jupyter / Colab for experiments, VS Code for final `.py` files, GitHub for the repo

---

## 1. Akif: Data & Pipeline Lead

**Mission:** Turn the raw, different-length CSVs into clean data and features everyone can use, and build the pipeline that runs on demo day.
**Not your job:** data augmentation. Imran owns it, to keep your load manageable. You only provide the fixed-length sequences and `extract_features` he needs.
**Key point:** Your job is not just to pick columns. It's to **create features**: summarize each episode's signals into one row of numbers so that episodes of any length become comparable.
**Others depend on you:** Everyone, early. Ship something simple fast, then improve.
**Chapters:** 05 (PCA)

### Deliverables
| File | Content | Due |
|---|---|---|
| `src/load.py` | `load_folder(path) → {episode_id: DataFrame}`; works with or without `success/`/`fail/` subfolders | **Nov 26** |
| `data/holdout/` | 4 success + 4 fail episodes set aside | **Nov 26** |
| `src/phases.py` | `detect_phases(df) → {grasp_idx, release_idx, …}` | **Nov 27** |
| `src/features.py` + `features.csv` v1 | Core features, one row per episode | **Nov 28** |
| `src/sequences.py` | `to_fixed_length(df, n=300)`: resampled raw sequences for the LSTM | **Nov 30** |
| `features.csv` final | Includes everyone's merged features | **Dec 6** |
| `demo.py` | Full pipeline on a new folder (with Elisya) | **Dec 11** |

### Step by step

**Step 1: Load & check (by Nov 26)**
- [ ] Load all 48 CSVs; confirm all have the same 57 columns, no missing values, timestamps increasing
- [ ] Record each episode's length and duration
- [ ] Create the hold-out split (stratified: 4 success + 4 fail, `random_state=42`) and tell the team which IDs
- [ ] Decide which columns to keep: left arm (7 joints) + left hand (6) for both `state` and `action`; right side only if a check shows it matters

**Step 2: Phase detection (by Nov 27)**
- [ ] `grip = mean of 4 left fingers (state)`
- [ ] Grasp = first frame `grip > 0.8`; release = last frame `grip > 0.8`
- [ ] Handle edge cases: hand closes more than once (count it), never closes (flag it)
- [ ] Plot phases on 5 episodes to check they look right

**Step 3: Core features v1 (by Nov 28)**

| Group | Features |
|---|---|
| Timing | `duration`, `reach_time`, `carry_time`, `return_time`, `reach_ratio` |
| Motion | `mean_speed`, `max_speed`, `carry_speed`, `path_length` |
| Control | `track_err` (mean \|state − action\|), `max_track_err` |
| Pose | wrist pitch/roll, elbow, shoulder pitch **at grasp** and **at release** |

- [ ] Write each feature as its own small function and combine them in `extract_features(df) → dict`
- [ ] `build_table(folder) → DataFrame`, then save `features.csv`
- [ ] Boxplot each feature (success vs fail) like `explore.py` Plot 3 and share them

**Step 4: Fixed-length sequences for the LSTM (by Nov 30)**
- [ ] `to_fixed_length(df, n=300)`: interpolate each of the 13 left-side `state` channels to 300 steps
- [ ] Output array shape: `(episodes, 300, 13)`; give it to Imran

**Step 5: Merge everyone's features (by Dec 6)**
- [ ] Review the 2–3 feature functions from Imran, Elisya, and Wafi; merge them; regenerate `features.csv`

**Step 6: Demo pipeline (by Dec 11, with Elisya)**
- [ ] `python demo.py --data <folder>` → `results.csv` with `episode, predicted_label, success_prob, cluster_name, score`
- [ ] Loads saved models from `models/`; **never retrains**
- [ ] Runs in under 1 minute; test on `data/holdout/` and on a fresh clone on another laptop

### Done when
`demo.py` runs on any folder on a clean laptop, and everyone has been using your feature table since Nov 28.

---

## 2. Imran: Classification & Augmentation Lead

**Mission:** Build the best success/fail predictor and find which features matter. Main route: **(A) classic models on the feature table**. Extra route if time allows: **(B) an LSTM on raw sequences + augmentation**, which is worth trying since the lecturer pointed to it, but Track A comes first.
**Augmentation:** you **own** it: build the methods, test whether they help, and write that report section.
**You depend on:** Akif (`features.csv`, fixed-length sequences, `extract_features`). Until then, start with `starter_features.csv`.
**Others depend on you:** Wafi (top features + success probability), Akif (saved model).
**Chapters:** 02, 04, 07, 10, **08**, **11**

### Deliverables
| File | Content | Due |
|---|---|---|
| Feature functions → Akif | 2–3 features that separate success/fail (e.g., `n_grasps`, `wr_p_at_release`) | **Nov 30** |
| `src/augment.py` | Augmentation methods + training-fold helper | **Dec 2** |
| `notebooks/classic_models.ipynb` | Track A experiments | **Dec 3** |
| `notebooks/lstm.ipynb` | Track B experiments | **Dec 6** |
| Top features → Wafi | Ranked list + short physical explanation | **Dec 4** |
| `src/classify.py` | `predict(features or sequences) → label, probability` | **Dec 8** |
| `models/classifier.*` | Final chosen model | **Dec 8** |
| `results/classification.md` | Comparison table, confusion matrix, augmentation results | **Dec 8** |

### Augmentation (Nov 30 – Dec 4; you own this)

. Common techniques for time-series data (Ch 8 covers several; others are welcome):

| Function | Ch 8 technique | What it does | Setting |
|---|---|---|---|
| `jitter(x)` | Noise injection | Adds Gaussian noise | σ = 1–2% of each joint's range |
| `speed_change(x)` | Changing sampling speed | Resample to 90–110% length | random per copy |
| `time_shift(x)` | Shifting | Trim/pad a few frames at start/end | ±5–15 frames |
| `magnitude_scale(x)` | Modifying | Scale motion around the mean | ×0.95–1.05 |
| `time_warp(x)` | Modifying | Stretch one phase only | ±10% |
| `pca_augment(X)` *(optional)* | "Do PCA and take a few" | Reconstruct from top PCs + small noise | Ch 5 |

- [ ] Each function takes one episode and returns a new one; `augment(x, k)` returns `k` random copies
- [ ] Plot 3 augmented copies next to the original to check they look realistic
- [ ] Keep changes small: a fail must not start looking like a success
- [ ] Provide a helper that augments **only the training fold** and keeps the original episode ID (for `GroupKFold`)
- [ ] Augment the **raw episodes** (before feature extraction). For Track A, run Akif's `extract_features` on the augmented copies; for Track B, convert them with Akif's `to_fixed_length`
- [ ] Test 0 / 1 / 3 / 5 / 10 copies per episode for Track A, and with vs. without for Track B
- [ ] Write the **augmentation section of the report**: methods, settings, example plots, and whether it helped


### Track A: Classic models (Nov 27 – Dec 3)
- [ ] `X` = features, `y` = 1 success / 0 fail; always put `StandardScaler` inside a `Pipeline`
- [ ] **Stratified 5-fold CV** repeated 5× (and leave-one-out as a check); report mean ± std
- [ ] Models: Logistic Regression (Ch 4), SVM linear + RBF (Ch 4), Decision Tree + Random Forest (Ch 7), k-NN, MLP (Ch 10)
- [ ] Metrics: accuracy, precision, recall, F1, confusion matrix; beat the dummy baseline (≈58%)
- [ ] **Feature importance** (LR coefficients, RF importance, permutation importance) → top 5–8 → **send to Wafi by Dec 4**
- [ ] Re-test with only the top features (fewer features = less overfitting with 40 episodes)
- [ ] **Augmentation test:** compare 0 / 1 / 3 / 5 / 10 copies per episode. *Early test with starter features showed no gain*, so re-test with the full features.

### Track B: LSTM on raw sequences (Nov 30 – Dec 6, if Track A is on schedule)
- [ ] Input: Akif's fixed-length sequences `(N, 300, 13)`, standardized per channel (fit on training fold only)
- [ ] Model (keep it **small**, since there's very little data):
  ```
  LSTM(13 → hidden 16–32, 1 layer) → dropout 0.3 → Linear → sigmoid
  ```
  Optional: a 1D-CNN (TDNN idea from Ch 11) for comparison
- [ ] Train with Adam (Ch 11), early stopping on the validation fold
- [ ] **Compare: without augmentation vs. with augmentation (5–10 copies)**. This is exactly the Ch 8 + Ch 11 combination, and a key result for the report.
- [ ] Use the same CV folds as Track A so the comparison is fair

### Final choice (Dec 7–8)
- [ ] Table: all models (A and B) with mean ± std accuracy and F1
- [ ] Choose the demo model: **best score, but prefer the simpler one if scores are close**
- [ ] Train on all 40 training episodes, save it, then test **once** on `data/holdout/` and record the result honestly
- [ ] Give Wafi a function that returns **success probability** (0–1) for each episode

### Watch out
- **Leakage:** never fit scalers/feature selection on all data before CV
- **Augmentation leakage:** copies of one episode must stay in the same fold (`GroupKFold` with original episode ID)
- LSTM will likely **overfit** with 40 episodes. Report that honestly if it happens; it's still a valid finding.

### Done when
The saved model predicts the hold-out set inside `demo.py`, and you can explain in one sentence why an episode was predicted to fail.

---

## 3. Elisya: Clustering & Visualization Lead

**Mission:** Find **types of attempts beyond success/fail**, give each a human-readable name backed by evidence, make the project's main figures, and co-build the demo.
**You depend on:** Akif (`features.csv`).
**Others depend on you:** Wafi (cluster descriptions), Akif (cluster model for the demo), everyone (figures).
**Chapters:** 05 (PCA), 06 (k-means, EM)

### Deliverables
| File | Content | Due |
|---|---|---|
| Feature functions → Akif | 2–3 motion-quality features (`jerk`, `n_pauses`, `path_length`) | **Nov 30** |
| `notebooks/clustering.ipynb` | Experiments | ongoing |
| Cluster names + evidence → Wafi | Table + plots | **Dec 5** |
| `src/cluster.py` | `assign(features) → cluster_name` | **Dec 8** |
| `models/cluster.joblib` | Scaler + model + number→name mapping | **Dec 8** |
| `results/clusters.md` + figures | Report section | **Dec 8** |
| `demo.py` (with Akif) | Integration + demo output display | **Dec 11** |

### Step by step

**Step 1: First look (by Nov 29)**
- [ ] Standardize features
- [ ] **PCA (Ch 5)** → 2D scatter, colored by success/fail; also check explained variance
- [ ] Do **not** use the label to build clusters; only compare with it afterwards

**Step 2: Cluster (by Dec 3)**
- [ ] **k-means (Ch 6)**: try k = 2–6; note the slides say results depend on initialization, so use `n_init=20` or PCA-based init
- [ ] **EM / Gaussian Mixture (Ch 6)**: initialized by k-means as in the lecture
- [ ] *(Optional)* hierarchical clustering + dendrogram
- [ ] Choose k with: elbow plot, silhouette score, BIC (for GMM)
- [ ] Try clustering on (a) all features and (b) top PCA components; compare

**Step 3: Name the clusters (by Dec 5)** → send to Wafi
- [ ] For each cluster: average of each feature vs overall average, and success rate inside the cluster
- [ ] Plot each cluster's joint trajectories (like `explore.py` Plot 2, colored by cluster)
- [ ] Name each one from the evidence, e.g. *"Clean & fast success"*, *"Slow but successful"*, *"Hesitant / regrasp"*, *"Rushed release"*
- [ ] Write **why**: 2–3 defining features with numbers per cluster
- [ ] Goal: find **more structure than success/fail**, e.g. two different kinds of fail

**Step 4: Finalize (by Dec 8)**
- [ ] Save scaler + model + `{cluster_number: name}` mapping; numbers change between runs, so fix `random_state`
- [ ] New episodes: assign to the nearest cluster center (k-means) or highest probability (GMM)

**Step 5: Figures + demo (by Dec 11)**
- [ ] Report figures: PCA scatter by cluster, cluster profile chart, trajectories per cluster, confusion matrix (from Imran), score distribution (from Wafi)
- [ ] With Akif: make `demo.py` output readable for the live audience (clear printed table, optional plot of the new episodes on the PCA map)

### Done when
Every cluster has a name, a description, and numbers behind it, and new episodes get a cluster name inside `demo.py`.

---

## 4. Wafi: Metric Lead & Team Leader

**Mission:** Design a defensible **0–100 quality score**, own the repository, keep the team on schedule, and deliver all reports.
**You depend on:** Imran (top features + success probability), Elisya (cluster descriptions).
**Others depend on you:** Everyone (merges, deadlines, reports), Akif (`score()` function).
**Chapters:** 03, 07 (feature importance), plus everyone's results

### Deliverables
| Item | Due |
|---|---|
| GitHub repo set up (structure, members, branch rules) | **Nov 24** |
| Mentoring report #1 (guest lecture) | after **Nov 24** |
| Feature functions → Akif (release quality: `release_speed`, `still_before_release`) | **Nov 30** |
| Score formula draft | **Dec 5** |
| `src/metric.py` + `results/metric.md` | **Dec 8** |
| Mentoring report #2 (office hours) | after **Dec 1 / Dec 8** |
| README, `requirements.txt`, repo cleanup | **Dec 12** |
| Final report | **Dec 16** |

### Part A: Repository owner
- [ ] Create the repo, invite Akif/Imran/Elisya, create folders:
  ```
  data/ (gitignored)  notebooks/  src/  models/  results/  demo.py  README.md  requirements.txt
  ```
- [ ] Protect `main`: pull requests only; you review and merge
- [ ] Akif is the **code owner** for `src/load.py`, `src/features.py`, `demo.py`, so get his approval on changes there
- [ ] Before Dec 13: clean repo, working README ("install → run demo" in 3 commands), pinned `requirements.txt`

### Part B: The metric
1. **Define "good" (by Nov 30)**
   - [ ] In plain words: *bottle ends upright, task done efficiently, smooth motion, no hesitation, gentle release*
   - [ ] Ask Mr. Yong on Nov 24 how Aei Robot judges a good attempt
2. **Build the formula (by Dec 5)**
   ```
   score = 100 × ( w1·success_prob + w2·efficiency + w3·smoothness + w4·release_quality )
   ```
   | Term | From | 0–1 meaning |
   |---|---|---|
   | `success_prob` | Imran's model | Chance the bottle ends upright |
   | `efficiency` | `duration`, `path_length` | Shorter/more direct = higher |
   | `smoothness` | Elisya's `jerk`, `n_pauses` | Smoother = higher |
   | `release_quality` | your `release_speed`, `still_before_release` | Calm, controlled release = higher |
   - [ ] Pick terms based on Imran's top features and Elisya's cluster differences
   - [ ] Weights: success matters most (e.g., w1 = 0.5); **justify every weight** in writing
   - [ ] Scale each term using **training min/max** (saved), so new episodes are scored consistently; clip to 0–1
3. **Validate (by Dec 7)**
   - [ ] Successes score higher than fails on average (plot both distributions)
   - [ ] Within successes, "clean" clusters score higher than "messy" ones
   - [ ] Scores on `data/holdout/` look sensible
   - [ ] Try 2–3 weight settings and show the ranking is stable
4. **Finalize (by Dec 8)**: `score(features) → 0–100` ready for `demo.py`; formula written clearly in `results/metric.md` (we must submit it)

### Part C: Team leader
- [ ] **Nov 23:** kick-off meeting, confirm roles, share this guide
- [ ] **Nov 24:** guest lecture: questions ready, notes, **team photo with Mr. Yong** → report #1
- [ ] **Nov 30 / Dec 1 / Dec 7 / Dec 8:** office hours: collect one question from each member beforehand → report #2
- [ ] **Weekly check-in**; remind people 2 days before each due date
- [ ] **Dec 12:** run the full **practice demo on `data/holdout/`** with everyone
- [ ] **Dec 13:** confirm final upload works (fresh clone + run)
- [ ] **Dec 14:** demo plan: who presents what (Akif: pipeline; Imran: classifier; Elisya: clusters; Wafi: score + intro/summary)
- [ ] **Dec 14–16:** collect sections and assemble the final report
- [ ] Remind everyone: **individual reflective journal due Dec 17**

### Questions to ask Mr. Yong (Nov 24)
1. The CSVs have both `state` and `action` columns; the description mentions only commanded positions. Which should we rely on?
2. Which joints or moments matter most for the bottle tipping over?
3. How does Aei Robot judge a "good" pick-and-place?
4. Will the demo-day test data have the same format and setup? Will a 2nd dataset be released?
5. What kinds of failures do they usually see?

---

## 5. Timeline

| Date | Akif | Imran | Elisya | Wafi |
|---|---|---|---|---|
| **Nov 23** | 👥 Kick-off meeting (everyone) | | | Run meeting |
| **Nov 24** | 🎤 Guest lecture (everyone) | | | Repo setup, notes, photo |
| **Nov 25–26** | Loader + **hold-out split** | Explore starter features | Explore starter features | Report #1 |
| **Nov 27–28** | Phases + **features v1** | Track A baseline | PCA first look | Define "good" |
| **Nov 29–30** | Fixed-length sequences | Feature functions → Akif | Feature functions → Akif | Feature functions → Akif |
| **Nov 30, Dec 1** | 🧑‍🏫 Office hours (everyone) | | | Collect questions |
| **Dec 2–3** | Merge features (early) | **augment.py** + Track A compare | Clustering (k-means, EM) | Formula draft |
| **Dec 4–5** | Merge features | **Top features → Wafi**, Track B LSTM | **Cluster names → Wafi** | Formula with real inputs |
| **Dec 6** | Final `features.csv` | LSTM ± augmentation | Cluster profiles | Validate score |
| **Dec 7–8** | 🧑‍🏫 Office hours (everyone) | Save final model | Save cluster model | `metric.py` + report #2 |
| **Dec 9–11** | **demo.py** (+ Elisya) | Help test | demo.py + figures | README, report outline |
| **Dec 12** | 🧪 **Practice demo on hold-out (everyone)** | | | Lead it |
| **Dec 13** | ⚠️ **Code upload, no changes after** | | | Confirm upload |
| **Dec 14** | 🎤 **Demo day** | | | |
| **Dec 15–16** | Report section | Report section | Report section | **Assemble final report** |
| **Dec 17** | ✍️ Reflective journal (everyone, individual) | | | |

---

## 6. Handoffs

| From → To | What | By |
|---|---|---|
| Akif → everyone | Hold-out IDs | Nov 26 |
| Akif → everyone | `features.csv` v1 | Nov 28 |
| Akif → Imran | Fixed-length sequences | Nov 30 |
| Imran, Elisya, Wafi → Akif | 2–3 feature functions each | Nov 30 |
| Imran → Wafi | Top features + `success_prob` function | Dec 4 |
| Elisya → Wafi | Cluster names + evidence | Dec 5 |
| Imran, Elisya, Wafi → Akif | Saved models + `predict / assign / score` | Dec 8 |
| Everyone → Wafi | Report sections | Dec 15 |

---

## 7. Final output we are aiming for

**On demo day**, `demo.py` turns a folder of unseen episodes into:

| episode | predicted_label | success_prob | cluster_name | score |
|---|---|---|---|---|
| test_01 | success | 0.91 | Clean & fast success | 87 |
| test_02 | fail | 0.22 | Rushed release | 31 |

**In the report**, we explain *why*: how accurate the classifier is and which features matter (Imran), what each cluster means and its evidence (Elisya), the score formula and why each term and weight is there (Wafi), how the data was processed (Akif), and whether augmentation helped (Imran).

> Methods in this guide are suggestions. If a member finds a better approach (inside or outside the lectures), use it, as long as it's validated properly and explained in the report.
