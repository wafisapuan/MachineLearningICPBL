# MMT3007 Machine Learning — IC-PBL Project Overview

**Project:** Classify, Cluster, and Evaluate Humanoid Robot Action Dataset
**Instructor:** Prof. Yongjae Yoo · **TA:** chungminho@hanyang.ac.kr
**Industry partner:** Aei Robot (Mr. Junwoo Yong, Researcher)

---

## 1. What the project is about

A humanoid robot (**Alice M1**) tries to **pick up a bottle/tumbler and place it in a yellow box**. Some attempts work and some don't. Our job is to understand each attempt **using only the log data (CSV files)**, without the videos.

### The data
- **48 episodes**, one CSV per episode, recorded at **30 fps**
- Each CSV contains: `timestamp`, `frame_index`, and **26 commanded joint positions**
  - Robot: dual-arm, each side has a 7-DoF arm + 6-DoF hand
  - Joint name prefixes: `l_`/`r_` = left/right, `sh` = shoulder, `el` = elbow, `wr` = wrist; `y/p/r` = yaw/pitch/roll
- Labels come from the folder each file is in:
  - `success/` (**27**): the bottle ends up **upright** in the box
  - `fail/` (**21**): the bottle **tips over**

---

## 2. What we need to build (4 tasks)

### Task 1: Preprocessing / Feature Engineering
- 48 episodes is a very small dataset, so we have to extract good features from it.
- Turn each time series into useful features, e.g.:
  - episode duration and the length of each phase (reach → grasp → lift → move → release)
  - joint velocities and accelerations, smoothness/jerk
  - grip timing and hand closing/opening behavior
  - arm height/pose at the moment of release
- Optional: **augment** the data (more samples) or **reduce** it (e.g., PCA). Any technique from class (or beyond) is allowed.

### Task 2: Classification
- Build a classifier that predicts **success / fail** for **new, unseen episodes**.
- Validate it properly (e.g., cross-validation, since the data is small).

### Task 3: Clustering
- Group the episodes into **more categories than just success/fail**.
  - Example clusters: *clean success*, *slow/hesitant success*, *dropped from too high*, *knocked over*
- Each cluster needs a **human-readable label backed by evidence** from the data.
- We must **justify why** the clustering makes sense.

### Task 4: Quality Score Metric
- Design a score (e.g., **0–1** or **0–100**) that rates how well each episode performed the task.
- Decide **which features matter and why**. Use the findings from Tasks 2 and 3 to back this up.
- We will **submit the formula** and **compute per-episode scores on the demo-day test data**.

> 💡 **Practical tip:** the demo uses **new data**, so the pipeline should take a folder of CSVs and output, for each episode: **predicted label + cluster + score**.

---

## 3. Schedule (updated per lecturer's notice)

**Implementation period: Nov 16 → Dec 14** (four weeks)

| Date | Event | Notes |
|---|---|---|
| **Nov 16 (Mon), Nov 17 (Tue)** | ❌ No classes | Implementation time |
| **Nov 23 (Mon)** | In-person lecture | |
| **Nov 24 (Tue)** | Guest lecture (Mr. Yong, Aei Robot) | Submit **PBL mentoring report #1** with evidence afterwards; team photo with Mr. Yong |
| **Nov 30 (Mon)** | Team-wise office hours | Q&A and mentoring with Prof. & TAs (details TBA) |
| **Dec 1 (Tue)** | Team-wise office hours | Q&A and mentoring (details TBA) |
| **Dec 7 (Mon)** | Team-wise office hours | Q&A and mentoring (details TBA) |
| **Dec 8 (Tue)** | Team-wise office hours | Q&A and mentoring (details TBA) |
| **Dec 13 (Sun), 23:59** | ⚠️ **Final code upload** (PBL platform) | **No changes allowed after this.** Make sure the code runs! |
| **Dec 14 (Mon), 14:30** | 🎤 **Demo day** | Live demo with our uploaded code in front of Prof., TA, and Mr. Yong |
| **Dec 16 (Wed), 23:59** | Repo + final report | GitHub repo with clean code + README, and the final report |
| **Dec 17 (Thu), 23:59** | Individual reflective journal (개인성찰일지) | **Each member submits individually** |

### Submission checklist
- [ ] PBL mentoring report #1 (after the Nov 24 guest lecture), PBL platform
- [ ] PBL mentoring report #2 (after mentoring; likely the office-hour sessions, so check the upcoming notice), PBL platform
- [ ] Source code: **Dec 13**, PBL platform (Week 14 report)
- [ ] Live demo on our laptop: **Dec 14**, classroom
- [ ] Team repository (code + README + final report): **Dec 16**, PBL platform
- [ ] Individual reflective journal: **Dec 17**, PBL platform (IRJ)

---

## 4. Team Roles (4 members)

Each person owns one task plus one shared job.

| Role | Main task | Shared job |
|---|---|---|
| **Person 1:** Data & Pipeline Lead | Task 1: Preprocessing / Features | Final demo script + GitHub repo |
| **Person 2:** Classification Lead | Task 2: Classification | Feature importance for the metric |
| **Person 3:** Clustering Lead | Task 3: Clustering | Visualizations for report & demo |
| **Person 4:** Metric Lead & Coordinator | Task 4: Quality Score | Admin, deadlines, final report |

### Person 1: Data & Pipeline Lead
- [ ] Load all 48 CSVs, check for missing values, odd lengths, and noise
- [ ] Smooth/resample trajectories if needed
- [ ] Build the **shared feature table** (one row per episode, one column per feature): duration, phase lengths, velocities, jerk, grip timing, release pose, etc.
- [ ] Try augmentation (e.g., noise, time-warping) or reduction (e.g., PCA) if it helps
- [ ] Write the **final demo script**: input = folder of CSVs → output = label + cluster + score per episode
- [ ] Keep the repo clean and write the README

### Person 2: Classification Lead
- [ ] Train several classifiers (Logistic Regression, SVM, Random Forest, k-NN, etc.)
- [ ] Use **cross-validation** (e.g., stratified k-fold or leave-one-out) because the data is small
- [ ] Report accuracy, precision, recall, F1, and confusion matrix
- [ ] Find the **most important features** for success vs. fail and pass them to Person 4
- [ ] Save the final trained model so it can be loaded in the demo script

### Person 3: Clustering Lead
- [ ] Try clustering methods (k-means, hierarchical, GMM, DBSCAN)
- [ ] Pick the number of clusters (elbow method, silhouette score, dendrogram)
- [ ] Give each cluster a **human-readable name** backed by evidence (e.g., "clean success", "dropped from too high")
- [ ] Explain **why** the clustering makes sense
- [ ] Make the main plots: PCA/t-SNE scatter, joint trajectories per cluster, feature comparison charts

### Person 4: Metric Lead & Coordinator
- [ ] Design the **quality score formula** (0–100) using insights from Persons 2 and 3
- [ ] Justify each feature and weight in the formula
- [ ] Check that the score ranks episodes sensibly (successes high, fails low, clean > messy)
- [ ] Track deadlines and submit the mentoring reports
- [ ] Put together the **final report** (each member writes their own section)

### Suggested timeline

| Week | Focus |
|---|---|
| **Nov 16–22** | Everyone explores the data together. Agree on the feature table format early. Person 1 delivers a first version by ~Nov 22. Persons 2–4 start with simple features meanwhile. |
| **Nov 23–29** | First versions of the classifier, clusters, and metric. Bring questions to the Nov 24 guest lecture. |
| **Nov 30–Dec 6** | Improve the models using office-hour feedback. Make sure the metric matches the classifier and cluster findings. |
| **Dec 7–12** | Person 1 combines everything into one script. Everyone tests it on a held-out practice set. **Freeze the code.** |
| **Dec 13–16** | Upload the code (Dec 13), demo (Dec 14), finish the repo and report (Dec 16). |

> 💡 **Tip:** Set aside ~8 episodes at the start, never train on them, and use them for a practice run of the demo.

---

## 5. Language & Tools

### Language
- **Python 3.10+**: used for everything (data processing, models, visualization, demo script)

### Libraries
| Purpose | Library |
|---|---|
| Data loading & manipulation | `pandas`, `numpy` |
| Signal processing (smoothing, derivatives, resampling) | `scipy` |
| Classification, clustering, PCA, scaling, cross-validation | `scikit-learn` |
| Visualization | `matplotlib`, `seaborn` |
| Saving/loading trained models | `joblib` |
| *(Optional)* Time-series features / clustering | `tsfresh`, `tslearn` |
| *(Optional)* Gradient boosting | `xgboost` or `lightgbm` |

### Environment & collaboration
| Purpose | Tool |
|---|---|
| Exploration & experiments | **Jupyter Notebook** / **Google Colab** |
| Final code & demo script | **VS Code** (plain `.py` files) |
| Version control & team repo | **Git + GitHub** (one branch per person, merge into `main`) |
| Dependency list | `requirements.txt` so the demo runs on any laptop |
| Team communication | KakaoTalk / Discord / Notion (whatever the team prefers) |

### Suggested repo structure
```
project/
├── data/                 # success/ and fail/ CSV folders (not pushed if too large)
├── notebooks/            # exploration notebooks per person
├── src/
│   ├── features.py       # Person 1
│   ├── classify.py       # Person 2
│   ├── cluster.py        # Person 3
│   ├── metric.py         # Person 4
│   └── demo.py           # Person 1: runs everything on a new folder
├── models/               # saved trained models
├── results/              # plots and tables
├── requirements.txt
└── README.md
```

---

## 6. Contacts
- Instructor: yongjaeyoo@hanyang.ac.kr
- TA: chungminho@hanyang.ac.kr
