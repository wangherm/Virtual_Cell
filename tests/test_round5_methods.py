import copy
import json

import numpy as np
import pytest
import torch

from vcell.data import load_prepared
from vcell.synthetic import make_demo
from vcell.functional import functional_vectors, load_function
from vcell.round5_methods import fit_head, fit_ridge, crossfit_teacher, reliability_report
from vcell.qwen import training_weights
from vcell.uce_encoder import cell_tokens, PlainUnpickler
from vcell.utils import file_sha256, write_json


@pytest.fixture
def data(tmp_path):
    torch.set_num_threads(1)
    return load_prepared(make_demo(tmp_path / "data", genes=16, targets=4, cells=8))


def test_held_labels_cannot_affect_head_or_ridge(data):
    train, val = data["splits"]["train"], data["splits"]["val"]
    features = np.random.default_rng(0).normal(size=(len(data["meta"]), 8)).astype(np.float32)
    first, _ = fit_head(data, features, train, val, epochs=1)
    ridge, _ = fit_ridge(data, train, val, 10., rank=3)
    changed = copy.deepcopy(data)
    changed["delta"][val] += 10000
    changed["delta"][changed["splits"]["test"]] -= 20000
    second, _ = fit_head(changed, features, train, val, epochs=1)
    ridge2, _ = fit_ridge(changed, train, val, 10., rank=3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(ridge, ridge2)


def test_functional_graph_preserves_order_missing_and_related_nodes(tmp_path):
    path = tmp_path / "annotations.gmt"
    path.write_text("process1\tdesc\tA\tB\nprocess2\tdesc\tC\tD\n")
    x, audit = functional_vectors(["A", "B", "C", "D", "MISSING"], [path], dimensions=16)
    np.testing.assert_array_equal(x[0], x[1])
    assert not np.array_equal(x[0], x[2])
    assert not x[4].any() and audit["missing"] == ["MISSING"]
    assert audit["directed_edges"] == 4
    out = tmp_path / "features.npz"
    np.savez(out, vectors=x, perturbations=["A", "B", "C", "D", "MISSING"])
    with pytest.raises(ValueError, match="order"):
        load_function({"path": out, "sha256": file_sha256(out)}, ["B", "A", "C", "D", "MISSING"])


def test_weighting_matches_macro_metric_and_binary_mask(data, tmp_path):
    train = data["splits"]["train"]
    # Remove most occurrences of one perturbation from one context.
    selected = train[:-3]
    weights, gate = training_weights({"teacher_mix": "equal", "loss_weighting": "context_perturbation"}, data, selected)
    meta = data["meta"].iloc[selected].copy()
    meta["weight"] = weights[selected].numpy() / len(selected)
    grouped = meta.groupby("context").weight.sum()
    np.testing.assert_allclose(grouped, 1 / len(grouped))
    assert torch.all(gate == 1)


def test_real_crossfit_cache_reports_only_training_rows(data, tmp_path):
    x = np.random.default_rng(9).normal(size=(len(data["meta"]), 8)).astype(np.float32)
    path = tmp_path / "features.npz"
    np.savez(path, features=x, row_ids=data["meta"].row_id.to_numpy(dtype="U"), data_fingerprint=data["audit"]["fingerprint"])
    write_json(path.with_suffix(".json"), {"feature_file_sha256": file_sha256(path),
               "artifact_kind": "frozen_control_features_NOT_predictions", "source": "TEST ONLY artificial features"})
    cache = crossfit_teacher(data, path, "uce", tmp_path / "head", epochs=1)
    report = reliability_report(data, {"uce": cache}, tmp_path / "reliability.json")
    assert set(report["eligible_rows"]) == set(data["splits"]["train"])
    assert report["weights"] == [1.]
    assert 0 <= report["kd_strength"] <= 1
    provenance = json.loads(cache.with_suffix(".json").read_text())
    for fold in provenance["crossfit"]:
        assert not set(fold["fit_rows"]) & set(fold["held_rows"])


def test_uce_sampling_zero_counts_and_chromosome_boundaries():
    tokens, mask = cell_tokens(np.array([[0., 9., 0.]]), np.array([4, 5, 6]),
                              np.array([0, 1, 2]), np.array([3, 2, 1]), np.random.RandomState(0), sample_size=4)
    assert tokens.tolist() == [[3, 143575, 5, 5, 5, 5, 2]]
    assert mask.sum() == 7
    with pytest.raises(ValueError, match="no expressed"):
        cell_tokens(np.zeros((1, 3)), np.array([4, 5, 6]), np.arange(3), np.arange(3), np.random.RandomState(0))
