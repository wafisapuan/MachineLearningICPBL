# Data folder (not committed)

Download the dataset from the PBL platform and place it like this:

```
data/
├── raw/
│   ├── success/   episode_000002.csv, ...   (27 files)
│   └── fail/      episode_000001.csv, ...   (21 files)
└── holdout/       created by scripts/make_holdout.py
    ├── success/   4 files
    └── fail/      4 files
```

- `data/` is in `.gitignore`. **Never commit the CSVs.**
- After running `python scripts/make_holdout.py`, `raw/` holds the **40 training episodes** and `holdout/` holds **8 episodes** for the practice demo.
- Every team member should run the same split (it's deterministic: fixed seed), or copy the hold-out IDs listed in `docs/holdout_ids.txt`.
