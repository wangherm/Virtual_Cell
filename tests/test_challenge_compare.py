"""Tiny random Qwen integration test; not evidence of biological performance."""
import copy
import json

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")
from test_qwen import setup
from test_training421 import experiment
from test_challenge_data import source_fixture
from vcell.challenge_data import aggregate_training, prepare_student_c
from vcell.challenge_compare import compare_students
from vcell.data import load_prepared
from vcell.qwen import run_qwen
from vcell.utils import file_sha256, write_json


def test_three_students_common_validation_and_row_guard(setup, tmp_path):
    cfg, reference = experiment(setup, tmp_path)
    parent = tmp_path / "parent"
    candidates = []
    for name, seed in [("student_a", 17), ("student_b", 29)]:
        student = copy.deepcopy(cfg["student"])
        student.update(output_dir=str(parent / name), seed=seed)
        run_qwen(student)
        candidates.append({"model": name, "checkpoint_sha256": file_sha256(parent / name / "all/best.pt")})
    write_json(parent / "COMPLETE.json", {"candidates": candidates, "test_evaluated": False})
    path, source, *_ = source_fixture(tmp_path, reference)
    ref = cfg["student"]["data_dir"]
    aggregate = aggregate_training(source, ref, tmp_path / "aggregate", local_h5ad=path)
    prepared = prepare_student_c(ref, aggregate, tmp_path / "prepared_c", min_cells=3, min_controls=3)
    cdata = load_prepared(prepared)
    student = copy.deepcopy(cfg["student"])
    student.update(data_dir=str(prepared), output_dir=str(tmp_path / "student_c"), arms=["supervised"],
                   teachers={}, kd_weight=0., allow_unverified_teachers=False)
    student.pop("teacher_families")
    run_qwen(student)
    output = compare_students(ref, parent, prepared, tmp_path / "student_c", tmp_path / "comparison")
    audit = json.loads((output / "comparison_audit.json").read_text())
    assert not audit["test_evaluated"]
    assert audit["retained_genes"] == len(reference["genes"]) - 2
    assert audit["retained_val_rows"] == len(cdata["splits"]["val"])
    assert audit["retained_val_rows"] < audit["original_val_rows"]
    assert set(audit["identities"]) == {"student_a", "student_b", "student_c"}
    for candidate in candidates:
        assert file_sha256(parent / candidate["model"] / "all/best.pt") == candidate["checkpoint_sha256"]
    predictions = tmp_path / "student_c/supervised/validation_predictions.npz"
    with np.load(predictions) as f:
        arrays = dict(f)
    arrays["row_ids"] = arrays["row_ids"][::-1]
    np.savez(predictions, **arrays)
    with pytest.raises(ValueError, match="row order"):
        compare_students(ref, parent, prepared, tmp_path / "student_c", tmp_path / "bad")
