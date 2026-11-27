# Data folder (raw CSVs are never committed)

Download the dataset from the PBL platform and put **all 48 episodes** here:

```
data/
├── raw/
│   ├── success/   episode_000002.csv, ...   (27 files)
│   └── fail/      episode_000001.csv, ...   (21 files)
└── holdout/       created by scripts/make_holdout.py
    ├── success/   4 files
    └── fail/      4 files
```

Then run once on your own laptop:

```bash
python scripts/make_holdout.py
```

It moves the 8 test-set episodes listed in `docs/holdout_ids.txt` from `raw/` to `holdout/`.
Afterwards `raw/` holds the **40 training episodes**.

Notes:
- The test-set list `docs/holdout_ids.txt` is committed and is the same for everyone: success 000014, 000029, 000036, 000038 and fail 000003, 000005, 000015, 000018. Do not change it.
- `scripts/build_features.py` also excludes these 8 episodes automatically, so it works whether or not you ran `make_holdout.py`.
- `data/` is in `.gitignore`. **Never commit the CSVs.**
- In the original download the folders are nested (`26_2_machinelearning_1st_data/26_2_machinelearning_1st_data/`). Move `success/` and `fail/` directly into `data/raw/`.
