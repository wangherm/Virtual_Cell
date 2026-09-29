import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")
from test_qwen import setup
from test_training421 import experiment
from vcell.teacher_pipeline import write_cache
from vcell.qwen import run_qwen, load_qwen_checkpoint, predict, teacher_targets
from vcell.train import tensors
from vcell.utils import file_sha256, write_json


def test_five_teacher_reliable_function_weighted_train_reload_and_tamper(setup, tmp_path):
    experiment_cfg, data = experiment(setup, tmp_path)
    cfg = experiment_cfg["student"]
    provenance = json.loads(Path(cfg["teachers"]["state"]).with_suffix(".json").read_text())
    provenance["teacher_family"] = "uce"
    cache = tmp_path / "uce.npz"
    write_cache(cache, np.full_like(data["delta"], .02), data, provenance, allow_unverified=True)
    cfg["teachers"]["uce"] = str(cache)
    cfg["teacher_families"].append("uce")
    functional = tmp_path / "functional.npz"
    np.savez(functional, vectors=np.eye(len(data["perturbations"]), dtype=np.float32), perturbations=data["perturbations"])
    cfg["model"]["functional"] = {"path": str(functional), "sha256": file_sha256(functional)}
    eligible = np.zeros(len(data["meta"]), dtype=bool)
    eligible[data["splits"]["train"]] = True
    mask = tmp_path / "mask.npz"
    np.savez(mask, eligible=eligible, row_ids=data["meta"].row_id.to_numpy(dtype="U"))
    report = tmp_path / "reliability.json"
    write_json(report, {"data_fingerprint": data["audit"]["fingerprint"], "teacher_names": cfg["teacher_families"],
                       "weights": [.1, .2, .3, .1, .3], "kd_strength": .5,
                       "teacher_cache_hashes": {k: [file_sha256(p), file_sha256(Path(p).with_suffix('.json'))] for k, p in cfg["teachers"].items()},
                       "eligible_rows": data["splits"]["train"].tolist(), "selected_using": "training-context cross-fit predictions only"})
    cfg.update(teacher_mix="reliability", loss_weighting="context_perturbation",
               kd_mask={"path": str(mask), "sha256": file_sha256(mask)},
               reliability={"path": str(report), "sha256": file_sha256(report)})
    root = run_qwen(cfg)
    model, saved = load_qwen_checkpoint(root / "all/best.pt")
    assert saved["functional_vectors"] is not None
    norm = {k: v.numpy() if torch.is_tensor(v) else v for k, v in saved["normalization"].items()}
    pred = predict(model, tensors(data, norm), data["splits"]["val"], "cpu", 4) * norm["scale"]
    with np.load(root / "all/validation_predictions.npz") as f:
        np.testing.assert_allclose(pred, f["delta"], atol=1e-6)
    run_qwen(cfg, resume=True)
    # Reject an otherwise correctly hashed report that includes held-out labels.
    bad = json.loads(report.read_text())
    bad["eligible_rows"].append(int(data["splits"]["val"][0]))
    write_json(report, bad)
    cfg["reliability"]["sha256"] = file_sha256(report)
    with pytest.raises(ValueError, match="training rows"):
        teacher_targets(cfg, data, data["splits"]["val"])
