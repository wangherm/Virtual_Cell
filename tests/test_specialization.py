import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vcell.data import load_prepared, save_prepared
from vcell.synthetic import make_demo
from vcell.specialization import (make_modules, reference_responses, calibration_folds,
    calibrate, route, nested_teachers, save_distillation, load_distillation, coverage_audit)
from vcell.expansion import prepare_expansion
from vcell.utils import file_sha256, write_json


@pytest.fixture
def data(tmp_path):
    torch.set_num_threads(1)
    return load_prepared(make_demo(tmp_path / "data", genes=12, targets=4, cells=8))


def features_for(data, folder):
    paths = {}
    for i, name in enumerate(["scgpt", "state", "scfoundation", "geneformer", "uce"]):
        path = folder / name / "features.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, features=np.random.default_rng(i).normal(size=(len(data["meta"]), 6)).astype(np.float32),
                 row_ids=data["meta"].row_id.to_numpy(dtype="U"), data_fingerprint=data["audit"]["fingerprint"])
        write_json(path.with_suffix(".json"), {"feature_file_sha256": file_sha256(path),
            "artifact_kind": "frozen_control_features_NOT_predictions", "source": "TEST ONLY artificial encoder features"})
        paths[name] = path
    return paths


def test_modules_overlap_unknown_and_no_outcome_leakage(data):
    a, _ = make_modules(data, 3)
    changed = copy.deepcopy(data)
    changed["delta"][:] = 999
    held = np.r_[data["splits"]["val"], data["splits"]["test"]]
    changed["baseline"][held] *= -100
    b, _ = make_modules(changed, 3)
    np.testing.assert_array_equal(a, b)
    genes = data["genes"].tolist()
    m, audit = make_modules(data, definition={"first": genes[:5], "second": genes[3:8]})
    np.testing.assert_allclose(m.sum(0), 1)
    assert m[0, 3] == .5 and audit["names"][-1] == "__unannotated__"
    assert m[-1, -1] == 1


def test_reference_uses_fit_only_and_batch_fallback_has_disjoint_groups(data):
    train = data["splits"]["train"]
    folds, kind = calibration_folds(data, train)
    assert kind == "context"
    fit, held = folds[0]
    expected, _ = reference_responses(data, fit)
    other = copy.deepcopy(data)
    other["delta"][held] += 100
    actual, _ = reference_responses(other, fit)
    np.testing.assert_array_equal(expected, actual)
    inner, kind = calibration_folds(data, fit)
    assert kind == "batch_fallback"
    for source, query in inner:
        assert not set(source) & set(query)
        assert not set(data["meta"].iloc[source].batch) & set(data["meta"].iloc[query].batch)


def test_complementary_teachers_route_locally_and_bad_teachers_abstain(data):
    train = data["splits"]["train"]
    modules = np.zeros((2, len(data["genes"])), dtype=np.float32)
    modules[0, :6], modules[1, 6:] = 1, 1
    prediction = np.stack([data["delta"].copy(), data["delta"].copy()])
    prediction[0, :, 6:] += 10
    prediction[1, :, :6] += 10
    # Larger reference errors expose clear constructed specializations.
    reference = data["delta"] + 100
    cal = calibrate(prediction, data, train, modules, reference, min_targets=1, shrinkage=0)
    assert cal["weights"] == [[1., 0.], [0., 1.]]
    values, gate = route(prediction, modules, cal)
    np.testing.assert_allclose(values, data["delta"], atol=1e-6)
    assert (gate > 0).all()
    bad = calibrate(prediction+100, data, train, modules, reference, min_targets=1)
    assert bad["strength"] == [0., 0.]
    few = calibrate(prediction, data, train, modules, reference, min_targets=100)
    assert few["strength"] == [0., 0.]


def test_nested_route_for_outer_context_cannot_read_its_outcomes(data, tmp_path):
    paths = features_for(data, tmp_path / "features")
    # Two teachers suffice for this library invariant; launcher requires all five.
    paths = dict(list(paths.items())[:2])
    modules, _ = make_modules(data, 2)
    first = nested_teachers(data, paths, modules, tmp_path / "one", epochs=1, min_targets=1)
    fit, held = calibration_folds(data, data["splits"]["train"])[0][0]
    changed = copy.deepcopy(data)
    changed["delta"][held] += 1000
    changed["delta"][changed["splits"]["val"]] -= 2000
    changed["delta"][changed["splits"]["test"]] += 3000
    second = nested_teachers(changed, paths, modules, tmp_path / "two", epochs=1, min_targets=1)
    for name in first:
        a, ag, _ = load_distillation(first[name], data)
        b, bg, _ = load_distillation(second[name], changed)
        np.testing.assert_array_equal(a[held], b[held])
        np.testing.assert_array_equal(ag[held], bg[held])
        assert not ag[data["splits"]["test"]].any()
    audit = json.loads((tmp_path / "one/audit.json").read_text())
    for fold in audit["folds"]:
        assert set(fold["base_calibration"]["rows"]) <= set(fold["fit_rows"])
        assert not set(fold["fit_rows"]) & set(fold["held_rows"])


def test_distillation_rejects_held_rows_and_tampering(data, tmp_path):
    target = np.zeros_like(data["delta"])
    gate = np.zeros_like(target)
    gate[data["splits"]["test"][0]] = 1
    spec = save_distillation(tmp_path / "bad.npz", data, target, gate, {})
    with pytest.raises(ValueError, match="Held-out"):
        load_distillation(spec, data)
    Path(spec["path"]).with_suffix(".json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_distillation(spec, data)


def challenge_fixture(data, output):
    meta = data["meta"].copy()
    train = data["splits"]["train"]
    # New synthetic context; single-cell data are never downloaded in tests.
    meta.loc[train, "context"] = "H1_hESC"
    meta.loc[train, "dataset"] = "test_h1"
    meta.loc[train, "row_id"] = ["H1_"+r for r in meta.loc[train, "row_id"]]
    save_prepared(output, data["baseline"][:, :-2], data["delta"][:, :-2], data["pert_idx"],
        data["genes"][:-2], data["perturbations"], meta, {"synthetic": True,
        "reference": {"fingerprint": data["audit"]["fingerprint"]},
        "normalization": data["audit"].get("normalization"), "target_sum": data["audit"].get("target_sum")})
    return load_prepared(output)


def test_expansion_preserves_every_original_held_row_and_common_gene(data, tmp_path):
    challenge = challenge_fixture(data, tmp_path / "challenge")
    paths = prepare_expansion(data, challenge, tmp_path / "expanded")
    a, b = (load_prepared(paths[k]) for k in ("original_only", "plus_h1"))
    for split in ("val", "test"):
        np.testing.assert_array_equal(a["delta"][a["splits"][split]], b["delta"][b["splits"][split]])
        assert len(a["splits"][split]) == len(data["splits"][split])
    assert len(b["splits"]["train"]) == 2 * len(a["splits"]["train"])
    assert a["genes"].tolist() == data["genes"][:-2].tolist()
    audit = coverage_audit(b, tmp_path / "coverage")
    assert "H1_hESC" in audit["training_contexts"]
