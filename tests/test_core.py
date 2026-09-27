import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
import anndata as ad
from scipy import sparse
from vcell.synthetic import make_demo
from vcell.data import load_prepared, validate_split_config, prepare, count_chunks
from vcell.models import make_model
from vcell.losses import cross_student_contrastive, mutual_loss
from vcell.train import run_suite, normalization, settings_for
from vcell.selection import fit_mix
from vcell.evaluate import row_metrics, evaluate_run
from vcell.teachers import import_cache, load_cache, check_provenance
from vcell.inference import predict_controls
from vcell.utils import read_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return make_demo(tmp_path_factory.mktemp("fixture") / "demo", genes=24, targets=8, cells=8)


def test_context_disjoint(demo):
    data = load_prepared(demo)
    assert data["meta"].groupby("context").split.nunique().max() == 1
    assert data["audit"]["synthetic"]
    assert data["delta"].shape[1] == 24
    assert np.isfinite(data["delta"]).all()


def test_reject_split_overlap():
    cfg = {"train_contexts": ["a"], "val_contexts": ["a"], "test_contexts": ["c"], "datasets": []}
    with pytest.raises(ValueError, match="disjoint"):
        validate_split_config(cfg)


@pytest.mark.parametrize("bad", [np.array([[1.2, 2.]]), np.array([[-1., 2.]]), np.array([[np.nan, 2.]])])
def test_reject_noncounts(bad):
    a = ad.AnnData(bad)
    with pytest.raises(ValueError):
        list(count_chunks(a, {"count_layer": "X"}, [0, 1], 10, 10000))


def test_normalizes_before_gene_subset():
    a = ad.AnnData(sparse.csr_matrix(np.array([[1, 9], [0, 0]], dtype=int)))
    _, _, x, keep = next(count_chunks(a, {"count_layer": "X"}, [0], 10, 10000))
    assert np.allclose(x[0, 0], np.log1p(1000))
    assert keep.tolist() == [True, False]


@pytest.mark.parametrize("kind", ["residual", "module", "mlp", "bilinear"])
def test_models_forward_backward(kind):
    cfg = read_config(ROOT / "configs/smoke.yaml")
    m, _ = make_model({"architecture": kind, "hidden": 32, "dropout": .05}, 24, 8, cfg["projection_dim"])
    y, z = m(torch.randn(4, 24), torch.tensor([0, 1, 2, 3]))
    assert y.shape == (4, 24)
    assert z.shape == (4, cfg["projection_dim"])
    y.square().mean().backward()
    assert m.output.weight.grad is not None


def test_contrastive_masks_repeated_targets():
    a = torch.randn(4, 8, requires_grad=True)
    b = torch.randn(4, 8, requires_grad=True)
    loss = cross_student_contrastive(a, b, torch.tensor([0, 0, 1, 2]))
    assert torch.isfinite(loss)
    loss.backward()
    assert a.grad is not None and b.grad is not None
    all_same = cross_student_contrastive(a, b, torch.zeros(4, dtype=torch.long))
    assert all_same.item() == 0


def test_mutual_gradient_is_stop_gradient():
    a, b = torch.tensor([1.], requires_grad=True), torch.tensor([3.], requires_grad=True)
    mutual_loss(a, b).backward()
    assert a.grad.item() == -2
    assert b.grad.item() == 2


def test_metrics_zero_prediction_and_perfect():
    y = np.array([[1., -1., .1], [.5, .2, -.3]])
    assert np.allclose(row_metrics(y, y, 2).mse_delta, 0)
    assert np.allclose(row_metrics(y, y, 2).pearson_delta, 1)
    zero = row_metrics(np.zeros_like(y), y, 2)
    assert zero.pearson_delta.isna().all()
    assert zero.topk_overlap.isna().all()


def test_mixture_selection_uses_given_validation_only():
    y = np.ones((4, 3))
    w = fit_mix([y, y * 0, y * 2, y * 3], y, np.ones(4) / 4)
    assert np.isclose(w.sum(), 1)
    assert np.allclose(np.einsum('i,irg->rg', w, np.stack([y, y*0, y*2, y*3])), y)


def test_external_cache_roundtrip_and_leakage(demo, tmp_path):
    d = load_prepared(demo)
    # Software fixture only, not real teacher labels.
    np.save(tmp_path / "fixture.npy", np.zeros_like(d["delta"]))
    d["meta"][["row_id"]].to_csv(tmp_path / "rows.csv", index=False)
    pd.DataFrame({"gene": d["genes"]}).to_csv(tmp_path / "genes.csv", index=False)
    prov = {"teacher_name": "synthetic_test_fixture", "model_revision": "fixture",
            "training_contexts": ["K562", "RPE1"], "excluded_contexts": ["HepG2", "Jurkat"],
            "source": "test", "license": "test", "audit_notes": "test only",
            "declared_no_holdout_perturbations": True}
    (tmp_path / "prov.json").write_text(json.dumps(prov))
    import_cache(tmp_path / "fixture.npy", tmp_path / "rows.csv", tmp_path / "genes.csv",
                 tmp_path / "prov.json", d, tmp_path / "cache.npz")
    pred, _ = load_cache(tmp_path / "cache.npz", d)
    assert pred.shape == d["delta"].shape
    prov["training_contexts"].append("Jurkat")
    with pytest.raises(ValueError, match="held-out"):
        check_provenance(prov, d)


def test_training_feature_selection_does_not_use_test_outcomes(demo, tmp_path):
    manifest = read_config(demo.parent / "manifest.yaml")
    for entry in manifest["datasets"]:
        entry["path"] = str(demo.parent / entry["path"])
    manifest["max_genes"] = 12
    manifest["output_dir"] = str(tmp_path / "original_prepared")
    original_config = tmp_path / "original_manifest.yaml"
    original_config.write_text(yaml.safe_dump(manifest))
    d1 = load_prepared(prepare(original_config))
    for entry in manifest["datasets"]:
        if entry["context"] == "Jurkat":
            a = ad.read_h5ad(entry["path"])
            x = a.X.toarray()
            x[a.obs.target != "control", 0] *= 7
            a.X = sparse.csr_matrix(x)
            a.write_h5ad(tmp_path / "changed_test.h5ad")
            entry["path"] = str(tmp_path / "changed_test.h5ad")
    manifest["output_dir"] = str(tmp_path / "prepared")
    config = tmp_path / "manifest.yaml"
    config.write_text(yaml.safe_dump(manifest))
    d2 = load_prepared(prepare(config))
    assert np.array_equal(d1["genes"], d2["genes"])
    ix = d1["splits"]["train"]
    assert np.array_equal(d1["delta"][ix], d2["delta"][ix])
    assert np.array_equal(normalization(d1)["mean"], normalization(d2)["mean"])
    assert not np.allclose(d1["delta"][d1["splits"]["test"]], d2["delta"][d2["splits"]["test"]])


def test_end_to_end_resume_and_control_only(demo, tmp_path):
    cfg = read_config(ROOT / "configs/smoke.yaml")
    cfg.update(data_dir=str(demo), output_dir=str(tmp_path / "run"), teacher_epochs=2,
               student_epochs=2, device="cpu", bootstrap_repeats=30)
    run = run_suite(cfg)
    assert (run / "evaluation_validation/report.html").exists()
    assert not (run / "evaluation_with_test").exists()
    assert (run / "seed_0/mutual_contrastive/best.pt").exists()
    with np.load(run / "seed_0/predictions.npz") as f:
        before = f["mutual_contrastive/mean"].copy()
        assert len([n for n in f.files if n.startswith('teachers/t_')]) == 4
        assert len([n for n in f.files if n.startswith('kd/s_')]) == 4
    run_suite(cfg, resume=True)
    with np.load(run / "seed_0/predictions.npz") as f:
        assert np.array_equal(before, f["mutual_contrastive/mean"])
    with pytest.raises(FileExistsError):
        run_suite(cfg)
    changed = dict(cfg, kd_weight=.9)
    with pytest.raises(ValueError, match="changed"):
        run_suite(changed, resume=True)
    evaluate_run(run, include_test=True)
    assert (run / "evaluation_with_test/test_access.json").exists()
    a = ad.read_h5ad(demo.parent / "Jurkat.h5ad")
    a = a[a.obs.target == "control"].copy()
    a.write_h5ad(tmp_path / "controls.h5ad")
    q = {"min_control_cells": 10, "datasets": [{"id": "new", "path": "controls.h5ad",
         "context": "unseen_context", "perturbation_key": "target", "batch_key": "batch",
         "count_layer": "X", "control_values": ["control"]}]}
    (tmp_path / "query.yaml").write_text(yaml.safe_dump(q))
    pd.DataFrame({"perturbation": ["PERT000", "PERT001"]}).to_csv(tmp_path / "targets.csv", index=False)
    predict_controls(run / "seed_0/mutual_contrastive/best.pt", tmp_path / "query.yaml",
                     tmp_path / "targets.csv", tmp_path / "inference", device="cpu")
    with np.load(tmp_path / "inference/mean_predictions.npz") as f:
        assert f["delta"].shape == (4, 24)
        assert np.isfinite(f["delta"]).all()


def test_short_control_label_is_not_truncated():
    from vcell.data import labels
    a = ad.AnnData(np.ones((2, 2)), obs=pd.DataFrame({"p": ["c", "g"], "b": ["1", "1"]}, index=["a", "b"]))
    p, _ = labels(a, {"id": "x", "perturbation_key": "p", "batch_key": "b", "control_values": ["c"]})
    assert p[0] == "__control__"


def test_row_filter_excludes_other_contexts():
    from vcell.data import count_chunks
    a = ad.AnnData(np.array([[1., 2.], [2., 3.], [3., 4.]]),
                   obs=pd.DataFrame({"context": ["a", "b", "a"]}, index=["a", "b", "c"]))
    entry = {"id": "a", "row_filter": {"context": "a"}, "count_layer": "X"}
    _, _, _, keep = next(count_chunks(a, entry, [0, 1], 10, 10000))
    assert keep.tolist() == [True, False, True]


def test_prepared_tamper_rejected(demo, tmp_path):
    import shutil
    dest = tmp_path / "tampered"
    shutil.copytree(demo, dest)
    m = pd.read_csv(dest / "metadata.csv")
    m.loc[0, "row_id"] = "edited"
    m.to_csv(dest / "metadata.csv", index=False)
    with pytest.raises(ValueError, match="fingerprint"):
        load_prepared(dest)


def test_interrupted_epoch_resume_matches_uninterrupted(demo, tmp_path, monkeypatch):
    import vcell.train as train_module
    cfg = read_config(ROOT / "configs/smoke.yaml")
    cfg.update(data_dir=str(demo), teacher_epochs=3, student_epochs=3,
               device="cpu", bootstrap_repeats=20,
               experiments=["mutual_contrastive"])
    data = load_prepared(demo)
    specs, settings = cfg["teachers"][:1], settings_for(cfg, "supervised")
    full = train_module.fit_models(specs, data, cfg, tmp_path / "full", 10, 3, settings)
    original_save = train_module.atomic_torch_save
    def interrupt_after_first_epoch(obj, path):
        original_save(obj, path)
        if Path(path).name == "last.pt" and obj["epoch"] == 0:
            raise RuntimeError("Simulated interruption")
    monkeypatch.setattr(train_module, "atomic_torch_save", interrupt_after_first_epoch)
    with pytest.raises(RuntimeError, match="Simulated"):
        train_module.fit_models(specs, data, cfg, tmp_path / "partial", 10, 3, settings)
    monkeypatch.setattr(train_module, "atomic_torch_save", original_save)
    resumed = train_module.fit_models(specs, data, cfg, tmp_path / "partial", 10, 3, settings, resume=True)
    assert np.array_equal(full[specs[0]['name']], resumed[specs[0]['name']])


def test_test_targets_do_not_change_fitted_predictions(demo, tmp_path):
    import copy
    from vcell.train import fit_models
    cfg = read_config(ROOT / "configs/smoke.yaml")
    cfg.update(teacher_epochs=2, device="cpu")
    a = load_prepared(demo)
    b = copy.deepcopy(a)
    b["delta"][b["splits"]["test"]] += 1000
    specs, settings = cfg["students"], settings_for(cfg, "mutual_contrastive")
    teachers = [np.zeros_like(a["delta"])] * 4
    pa = fit_models(specs, a, cfg, tmp_path / "a", 10, 2, settings, teachers)
    pb = fit_models(specs, b, cfg, tmp_path / "b", 10, 2, settings, teachers)
    for name in pa:
        assert np.array_equal(pa[name], pb[name])


def test_notebook_schema_and_code_syntax():
    import ast
    import nbformat
    nb = nbformat.read(ROOT / "notebooks/VCell_TeacherStudent_Colab.ipynb", as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None
            assert not cell.outputs


def test_external_teachers_are_used_without_training_references(demo, tmp_path):
    from vcell.utils import write_json
    d = load_prepared(demo)
    paths = []
    for i in range(4):
        path = tmp_path / f"external_{i}.npz"
        np.savez_compressed(path, delta=np.zeros_like(d["delta"]) + i * .01,
                            genes=d["genes"], row_ids=d["meta"].row_id.to_numpy(dtype="U"),
                            data_fingerprint=d["audit"]["fingerprint"])
        write_json(path.with_suffix(".json"), {"teacher_name": f"fixture_{i}", "model_revision": "test",
                   "training_contexts": ["K562", "RPE1"], "excluded_contexts": ["HepG2", "Jurkat"],
                   "source": "synthetic_test", "license": "test", "audit_notes": "test only",
                   "declared_no_holdout_perturbations": True})
        paths.append(str(path))
    cfg = read_config(ROOT / "configs/smoke.yaml")
    cfg.update(data_dir=str(demo), output_dir=str(tmp_path / "external_run"),
               student_epochs=1, experiments=["kd"], device="cpu")
    cfg["teachers"] = [{"name": f"external_{i}", "cache": path} for i, path in enumerate(paths)]
    run = run_suite(cfg)
    assert not (run / "seed_0/teachers").exists()
    assert (run / "seed_0/kd/best.pt").exists()
    with np.load(run / "seed_0/predictions.npz") as f:
        assert np.allclose(f["teachers/external_3"], .03)


def test_all_four_students_receive_pairwise_gradients():
    from vcell.losses import student_losses

    import torch
    outputs = [(torch.randn(5, 8, requires_grad=True), torch.randn(5, 4, requires_grad=True)) for _ in range(4)]
    peer, contrast = student_losses(outputs, torch.arange(5), .2, True, True)
    (peer + contrast).backward()
    for pred, projection in outputs:
        assert pred.grad.abs().sum() > 0
        assert projection.grad.abs().sum() > 0
