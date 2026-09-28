"""End-to-end software tests; tiny random models/caches are TEST ONLY."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")
from test_qwen import setup
from vcell.qwen import teacher_targets
from vcell.teachers import load_cache
from vcell.teacher_pipeline import write_cache
from vcell.training421 import run_421, preflight, select_final
from vcell.utils import file_sha256


def experiment(setup, tmp_path):
    base, data = setup
    base.update(arms=["all"], max_train_rows=None, max_val_rows=None, epochs=1,
                teacher_families=["state", "scgpt", "scfoundation", "geneformer"], allow_unverified_teachers=True)
    for i, family in enumerate(base["teacher_families"]):
        path = tmp_path / (family + ".npz")
        provenance = {"teacher_name": "TEST ONLY", "teacher_family": family, "model_revision": "TEST ONLY",
                      "training_contexts": [], "excluded_contexts": [], "source": "Artificial test arrays",
                      "license": "test", "audit_notes": "Test unverified provenance gate", "declared_no_holdout_perturbations": False,
                      "benchmark_status": "exploratory_pretraining_overlap_unverified", "prediction_space": "prepared_log1p_delta"}
        write_cache(path, np.full_like(data["delta"], (i+1)*.01), data, provenance, allow_unverified=True)
        base["teachers"][family] = str(path)
    return {"student": base, "student_seeds": [17, 29], "parallel_students": 2, "output_dir": str(tmp_path / "experiment")}, data


def test_four_targets_are_equal_and_strict_mode_still_blocks(setup, tmp_path):
    cfg, data = experiment(setup, tmp_path)
    base = cfg["student"]
    targets, _, _, weights = teacher_targets(base, data, data["splits"]["val"])
    np.testing.assert_allclose(targets["all"], .025)
    assert weights == [.25]*4
    with pytest.raises(ValueError):
        load_cache(base["teachers"]["state"], data)
    base["allow_unverified_teachers"] = False
    with pytest.raises(ValueError):
        preflight(cfg)


def test_full_data_two_processes_select_reload_resume_and_alignment_guard(setup, tmp_path):
    cfg, data = experiment(setup, tmp_path)
    root = Path(cfg["output_dir"])
    final = run_421(cfg)
    selection = json.loads((final / "selection.json").read_text())
    assert not selection["test_evaluated"] and selection["allow_unverified_teachers"]
    assert len(selection["candidates"]) == 2
    assert selection["selected"]["mse_delta"] == min(c["mse_delta"] for c in selection["candidates"])
    assert file_sha256(final / "student.pt") == selection["selected"]["checkpoint_sha256"]
    configurations = {n: json.loads((root / (n + ".json")).read_text()) for n in ("student_a", "student_b")}
    for name in configurations:
        manifest = json.loads((root / name / "run_manifest.json").read_text())
        assert manifest["train_rows"] == data["splits"]["train"].tolist()
        assert manifest["val_rows"] == data["splits"]["val"].tolist()
    assert run_421(cfg, resume=True) == final
    # Equal shapes do not make mismatched validation rows acceptable.
    path = root / "student_b/all/validation_predictions.npz"
    with np.load(path) as f:
        values = dict(f)
    values["row_ids"] = values["row_ids"][::-1]
    np.savez_compressed(path, **values)
    targets, *_ = teacher_targets(cfg["student"], data, data["splits"]["val"])
    with pytest.raises(ValueError, match="IDs differ"):
        select_final(root, data, targets, configurations, "test")


def test_421_rejects_subsampling_duplicate_seeds_and_missing_teacher(setup, tmp_path):
    cfg, _ = experiment(setup, tmp_path)
    for mutate in (lambda c: c["student"].update(max_train_rows=1),
                   lambda c: c.update(student_seeds=[17, 17]),
                   lambda c: c["student"]["teachers"].pop("state")):
        bad = copy.deepcopy(cfg)
        mutate(bad)
        with pytest.raises(ValueError):
            preflight(bad)
    assert not Path(cfg["output_dir"]).exists()
