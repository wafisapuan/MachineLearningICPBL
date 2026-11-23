# Notebooks

Personal experiments. Name them `<name>_<topic>.ipynb`, e.g.:

- `akif_features_check.ipynb`
- `imran_classic_models.ipynb`, `imran_augmentation.ipynb`, `imran_lstm.ipynb`
- `elisya_pca_clustering.ipynb`
- `wafi_metric.ipynb`

Rules:
- Import shared code from `src/` instead of copying it:
  ```python
  import sys; sys.path.insert(0, "..")
  from src.load import load_folder
  from src.features import build_table
  ```
- Clear large outputs before committing.
- Anything `demo.py` needs must be moved into `src/`.
