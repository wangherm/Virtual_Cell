import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vcell.synthetic import make_demo
from vcell.data import load_prepared
from vcell.external_models import select_symbols
from vcell.teacher_pipeline import control_groups, fit_feature_teacher, write_cache
from vcell.teachers import load_cache
from vcell.utils import write_json


@pytest.fixture
def data(tmp_path):
    return load_prepared(make_demo(tmp_path / "demo", genes=16, targets=4, cells=8))


def test_controls_are_real_unique_matched_cells(data):
    groups = list(control_groups(data, 5, 0, symbol_key=None))
    rows = []
    for indices, counts, genes, symbols, record in groups:
        rows.extend(indices.tolist())
        assert counts.shape == (5, 16)
        assert len(set(record["control_cell_ids"])) == 5
        # Synthetic source orders 16 control cells before each batch's perturbations.
        ids = [int(name.rsplit("_", 1)[1]) for name in record["control_cell_ids"]]
        assert all(i % 48 < 16 for i in ids)
        assert (data["meta"].iloc[indices].batch == record["batch"]).all()
    assert sorted(rows) == list(range(len(data["meta"])))


def test_symbol_mapping_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        select_symbols(["A", "A"], ["A", "B"])


def test_feature_head_produces_aligned_cache(data, tmp_path):
    torch.set_num_threads(1)
    features = tmp_path / "features.npz"
    np.savez_compressed(features, features=np.random.default_rng(0).normal(size=(len(data["meta"]), 8)).astype(np.float32),
                        row_ids=data["meta"].row_id.to_numpy(dtype="U"), data_fingerprint=data["audit"]["fingerprint"])
    write_json(features.with_suffix(".json"), {
        "teacher_name": "TEST_ONLY", "teacher_family": "scgpt", "model_revision": "TEST_ONLY",
        "training_contexts": [], "excluded_contexts": ["HepG2", "Jurkat"], "source": "TEST fixture",
        "license": "test", "audit_notes": "Unit test artificial features, not pretrained biology",
        "declared_no_holdout_perturbations": True, "artifact_kind": "frozen_control_features_NOT_predictions"})
    cfg = {"data_dir": str(tmp_path / "demo/prepared"), "features": str(features),
           "output_dir": str(tmp_path / "head"), "device": "cpu", "seed": 0, "hidden": 8,
           "epochs": 2, "batch_size": 8, "patience": 2, "learning_rate": .001}
    path = fit_feature_teacher(cfg)
    pred, provenance = load_cache(path, data)
    assert pred.shape == data["delta"].shape
    assert set(provenance["training_contexts"]) == {"K562", "RPE1"}
    assert provenance["prediction_space"] == "prepared_log1p_delta"
    assert provenance["selection_contexts"] == ["HepG2"]


def test_native_cache_rejects_holdout_exposure(data, tmp_path):
    provenance = {"teacher_name": "state", "model_revision": "test", "training_contexts": ["Jurkat"],
        "excluded_contexts": ["HepG2", "Jurkat"], "source": "test", "license": "test",
        "audit_notes": "test", "declared_no_holdout_perturbations": True}
    with pytest.raises(ValueError, match="held-out"):
        write_cache(tmp_path / "teacher.npz", np.zeros_like(data["delta"]), data, provenance)
    assert not (tmp_path / "teacher.npz").exists()
