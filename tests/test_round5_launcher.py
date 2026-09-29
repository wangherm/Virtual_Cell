"""Full orchestration on tiny offline fixtures; downloads/UCE/GPU are stubbed."""
import importlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")
from test_qwen import setup
from vcell.utils import file_sha256, write_json


def test_round5_launches_every_ablation_and_compares_identical_panel(setup, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("run_round5")
    cfg, data = setup
    work = tmp_path / "work"
    work.mkdir()
    repo = tmp_path / "test_repo"
    (repo / "configs").mkdir(parents=True)
    for name in ("uce_backbone", "uce_aux"):
        write_json(repo / f"configs/{name}.lock.json", {"files": {}, "TEST_ONLY": True})
    assets = repo / "assets/function"
    assets.mkdir(parents=True)
    gmt = assets / "test.gmt"
    gmt.write_text("test_term\tdesc\t" + "\t".join(data["perturbations"]) + "\n")
    write_json(assets / "sources.json", {"files": [{"library": "test", "sha256": file_sha256(gmt)}]})
    parent = work / "runs/qwen_421_four_teacher_01"
    parent.mkdir(parents=True)
    write_json(parent / "run_manifest.json", {"data_fingerprint": data["audit"]["fingerprint"],
               "fingerprint": "TEST ONLY", "config": {"student": cfg}})
    candidates = []
    for name in ("student_a", "student_b"):
        checkpoint = parent / name / "all/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"TEST ONLY historical checkpoint stand-in")
        candidates.append({"model": name, "checkpoint_sha256": file_sha256(checkpoint)})
    write_json(parent / "COMPLETE.json", {"run_fingerprint": "TEST ONLY", "test_evaluated": False, "candidates": candidates})
    def feature(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, features=np.random.default_rng(0).normal(size=(len(data["meta"]), 8)).astype(np.float32),
                 row_ids=data["meta"].row_id.to_numpy(dtype="U"), data_fingerprint=data["audit"]["fingerprint"])
        write_json(path.with_suffix(".json"), {"feature_file_sha256": file_sha256(path),
                   "artifact_kind": "frozen_control_features_NOT_predictions", "source": "TEST ONLY synthetic features"})
    for family in runner.FAMILIES[:-1]:
        feature(work / "teachers/four_teacher_01" / family / "features.npz")
    monkeypatch.setattr(runner, "REPO", repo)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(runner, "prepare", lambda *a, **kw: Path(cfg["model"]["model_id"]))
    monkeypatch.setattr(runner, "export_uce", lambda data, path, *a: feature(path))
    actual_head = runner.crossfit_teacher
    def head(*a, **kw):
        kw["device"] = "cpu"
        return actual_head(*a, **kw)
    monkeypatch.setattr(runner, "crossfit_teacher", head)
    actual_qwen = runner.run_qwen
    def qwen(config, **kw):
        config["device"] = "cpu"
        config["num_threads"] = 1
        config["model"]["dtype"] = "float32"
        return actual_qwen(config, **kw)
    monkeypatch.setattr(runner, "run_qwen", qwen)
    actual_prediction = runner.validation_prediction
    def prediction(path, data, arm):
        if Path(path).parent == parent:
            return np.zeros_like(data["delta"][data["splits"]["val"]]), {"TEST_ONLY": True}
        return actual_prediction(path, data, arm)
    monkeypatch.setattr(runner, "validation_prediction", prediction)
    runner.main(["--work-dir", str(work), "--data", cfg["data_dir"], "--epochs", "1", "--head-epochs", "1"])
    root = work / "runs/round5_01"
    completed = json.loads((root / "COMPLETE.json").read_text())
    audit = json.loads((root / "comparison_audit.json").read_text())
    assert completed["arms_completed"] == 7 and not completed["test_evaluated"]
    assert {r["config"]["loss_weighting"] for r in audit["runs"]} == {"uniform", "context_perturbation"}
    assert {len(r["config"]["teacher_families"]) for r in audit["runs"]} == {4, 5}
    assert all(r["config"]["data_dir"] == cfg["data_dir"] for r in audit["runs"])
    with tarfile.open(root / "round5_review.tar.gz") as archive:
        names = archive.getnames()
        assert "comparison.csv" in names and sum(n.endswith("history.csv") for n in names) == 7
        assert not any(n.endswith(".pt") for n in names)
