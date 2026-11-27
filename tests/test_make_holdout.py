"""make_holdout.py: moves the listed test episodes and never leaves copies in data/raw."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_holdout", ROOT / "scripts" / "make_holdout.py")
make_holdout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_holdout)


def setup(tmp_path, monkeypatch):
    raw, hold, ids = tmp_path / "raw", tmp_path / "holdout", tmp_path / "holdout_ids.txt"
    for label, n in [("success", 6), ("fail", 6)]:
        (raw / label).mkdir(parents=True)
        for i in range(n):
            (raw / label / f"episode_{label[0]}{i:05d}.csv").write_text("x")
    ids.write_text("# test set\nsuccess/episode_s00000.csv\n  # indented comment\nfail/episode_f00001.csv\n")
    monkeypatch.setattr(make_holdout, "DATA_RAW", raw)
    monkeypatch.setattr(make_holdout, "DATA_HOLDOUT", hold)
    monkeypatch.setattr(make_holdout, "HOLDOUT_IDS", ids)
    return raw, hold


def test_comments_are_ignored(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    assert make_holdout.read_list() == ["success/episode_s00000.csv", "fail/episode_f00001.csv"]


def test_moves_listed_episodes(tmp_path, monkeypatch):
    raw, hold = setup(tmp_path, monkeypatch)
    make_holdout.main()
    assert (hold / "success/episode_s00000.csv").exists()
    assert not (raw / "success/episode_s00000.csv").exists()
    assert len(list(raw.glob("*/*.csv"))) == 10


def test_duplicate_in_raw_is_removed(tmp_path, monkeypatch):
    raw, hold = setup(tmp_path, monkeypatch)
    make_holdout.main()
    (raw / "success/episode_s00000.csv").write_text("x")      # dataset copied in again
    make_holdout.main()
    assert not (raw / "success/episode_s00000.csv").exists()  # no leak into training
    assert (hold / "success/episode_s00000.csv").exists()     # test copy kept
