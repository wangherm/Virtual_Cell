"""Actual tiny offline Qwen training through all arms; not biological evidence."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")
from test_qwen import setup
from test_specialization import features_for, challenge_fixture
from vcell.utils import write_json
from vcell.qwen import load_qwen_checkpoint, predict
from vcell.train import tensors


def test_all_routes_reload_resume_expansion_and_panel_guards(setup, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_round6")
    cfg, data = setup
    paths = features_for(data, tmp_path / "features")
    features_json = tmp_path / "features.json"
    write_json(features_json, {n: str(p) for n, p in paths.items()})
    cfg_path = tmp_path / "student.json"
    write_json(cfg_path, cfg)
    work = tmp_path / "work"
    work.mkdir()
    challenge_fixture(data, tmp_path / "challenge")
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*1024**3))
    args = ["--work-dir", str(work), "--data", cfg["data_dir"], "--features-json", str(features_json),
            "--student-config", str(cfg_path), "--model-dir", cfg["model"]["model_id"],
            "--seeds", "17", "--epochs", "1", "--head-epochs", "1", "--device", "cpu", "--modules", "2",
            "--min-specialty-targets", "1", "--challenge-data", str(tmp_path / "challenge")]
    root = runner.main(args)
    complete = json.loads((root / "COMPLETE.json").read_text())
    assert complete["arms_completed"] == 6 and complete["expansion_completed"] and not complete["test_evaluated"]
    checkpoint = root / "students/B_background_seed17/supervised/best.pt"
    model, saved = load_qwen_checkpoint(checkpoint)
    norm = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in saved["normalization"].items()}
    actual = predict(model, tensors(data, norm), data["splits"]["val"], "cpu", 4) * norm["scale"]
    with np.load(checkpoint.parent / "validation_predictions.npz") as f:
        np.testing.assert_allclose(actual, f["delta"], atol=1e-6)
    expansion = json.loads((root / "expansion/audit.json").read_text())
    assert expansion["epoch_train_rows"] == len(data["splits"]["train"])
    with tarfile.open(root / "round6_review.tar.gz") as t:
        names = t.getnames()
        assert "teacher_diagnostics.json" in names and "paired_comparisons.csv" in names
        assert "expansion/comparison.csv" in names
        assert not any(n.endswith((".npz", ".pt")) for n in names)
    runner.main(args + ["--resume"])
    with pytest.raises(ValueError, match="Resume"):
        runner.main(args + ["--resume", "--modules", "3"])
