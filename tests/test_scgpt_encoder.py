import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from vcell.data import load_prepared
from vcell.external_models import ScGPTEncoder
from vcell.scgpt_encoder import FrozenScGPT, bin_values
from vcell.synthetic import make_demo
from vcell.teacher_pipeline import export_native_teacher, fit_feature_teacher
from vcell.teachers import load_cache
from vcell.utils import file_sha256, write_json


def tiny_checkpoint(tmp_path, genes):
    vocab = {g: i for i, g in enumerate([*genes, "<pad>"])}
    args = dict(embsize=8, nheads=2, d_hid=12, nlayers=2, pad_token="<pad>",
                input_emb_style="continuous", input_style="binned", no_cls=True, n_bins=51, max_seq_len=12)
    torch.manual_seed(7)
    model = FrozenScGPT(vocab, args).eval()
    # Use the real upstream FlashAttention key layout, not our load-time names.
    state = {k.replace(".self_attn.in_proj_weight", ".self_attn.Wqkv.weight")
              .replace(".self_attn.in_proj_bias", ".self_attn.Wqkv.bias"): v for k, v in model.state_dict().items()}
    state["decoder.TEST_UNUSED"] = torch.ones(1)
    torch.save(state, tmp_path / "best_model.pt")
    write_json(tmp_path / "args.json", args)
    write_json(tmp_path / "vocab.json", vocab)
    cfg = {"implementation": "vcell_frozen_scgpt_v1", "seed": 0}
    for key, file in (("checkpoint", "best_model.pt"), ("args", "args.json"), ("vocab", "vocab.json")):
        cfg[key] = str(tmp_path / file)
        cfg[key + "_sha256"] = file_sha256(tmp_path / file)
    return model, cfg


def test_fused_checkpoint_matches_unfused_and_is_frozen(tmp_path):
    torch.set_num_threads(1)
    genes = [f"G{i}" for i in range(16)]
    reference, cfg = tiny_checkpoint(tmp_path, genes)
    actual = ScGPTEncoder(cfg, "cpu")
    tokens = torch.arange(8).view(1, -1)
    values = torch.tensor([[1., 2., 4., 8., 12., 25., 40., 50.]])
    with torch.no_grad():
        expected = reference._encode(tokens, values, torch.zeros_like(tokens, dtype=torch.bool))
        got = actual.model._encode(tokens, values, torch.zeros_like(tokens, dtype=torch.bool))
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert not any(p.requires_grad for p in actual.model.parameters())
    counts = np.array([[0, 0, 1, 2, 3, 3, 5, 6, 9, 10, 11, 15, 2, 2, 2, 2]], dtype=np.float32)
    one = actual.encode(counts, genes)
    two = ScGPTEncoder(cfg, "cpu").encode(counts, genes)
    np.testing.assert_array_equal(one, two)
    assert one.shape == (1, 8) and np.isfinite(one).all()


def test_missing_encoder_tensor_and_tampered_metadata_fail(tmp_path):
    _, cfg = tiny_checkpoint(tmp_path, ["A", "B"])
    state = torch.load(cfg["checkpoint"], weights_only=True)
    del state["encoder.enc_norm.bias"]
    torch.save(state, cfg["checkpoint"])
    cfg["checkpoint_sha256"] = file_sha256(cfg["checkpoint"])
    with pytest.raises(RuntimeError, match="Missing key"):
        ScGPTEncoder(cfg, "cpu")
    Path(cfg["args"]).write_text("{}")
    with pytest.raises(ValueError, match="args checksum"):
        ScGPTEncoder(cfg, "cpu")


def test_binning_zero_ties_range_and_repeatability():
    x = np.array([0., 1., 1., 1., 1., 2., 4., 16.], dtype=np.float32)
    one = bin_values(x, 51, np.random.default_rng(8))
    np.testing.assert_array_equal(one, bin_values(x, 51, np.random.default_rng(8)))
    assert one[0] == 0 and one[-1] == 50
    assert ((one[1:] >= 1) & (one[1:] <= 50)).all()
    assert len(set(one[1:5])) > 1
    assert not bin_values(np.zeros(8), 51, np.random.default_rng(0)).any()


def test_native_encoder_to_task_head_cache(tmp_path):
    torch.set_num_threads(1)
    data_dir = make_demo(tmp_path / "demo", genes=16, targets=4, cells=8)
    data = load_prepared(data_dir)
    _, model = tiny_checkpoint(tmp_path, data["genes"].tolist())
    provenance = dict(teacher_name="TEST ONLY", teacher_family="scgpt", model_revision="TEST",
        training_contexts=[], excluded_contexts=["HepG2", "Jurkat"], source="random unit test fixture",
        license="test", audit_notes="Not biological evidence", declared_no_holdout_perturbations=True,
        prediction_space="prepared_log1p_delta")
    write_json(tmp_path / "provenance.json", provenance)
    cfg = dict(family="scgpt", data_dir=str(data_dir), output=str(tmp_path / "features.npz"),
        provenance=str(tmp_path / "provenance.json"), device="cpu", seed=0,
        max_control_cells=2, gene_symbol_key=None, model=model)
    features = export_native_teacher(cfg)
    fit = dict(data_dir=str(data_dir), features=str(features), output_dir=str(tmp_path / "head"),
        device="cpu", seed=0, hidden=8, epochs=2, patience=2, batch_size=8, learning_rate=.001)
    cache = fit_feature_teacher(fit)
    pred, info = load_cache(cache, data)
    assert pred.shape == data["delta"].shape and np.isfinite(pred).all()
    assert info["feature_file_sha256"] == file_sha256(features)
    assert set(info["training_contexts"]) == {"K562", "RPE1"}


def test_published_audit_rejects_other_datasets(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("scgpt_launcher", scripts / "run_scgpt_teacher.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = load_prepared(make_demo(tmp_path / "demo", genes=16, targets=4, cells=8))
        with pytest.raises(ValueError, match="GSE264667"):
            module.benchmark_provenance(data, {"revision": "test"})
        for source in data["audit"]["sources"]:
            context = source["id"].removeprefix("synthetic_")
            if context in ("HepG2", "Jurkat"):
                source["path"] = f"GSE264667_{context.lower()}_raw_singlecell_01.h5ad"
        audited = module.benchmark_provenance(data, {"revision": "test"})
        assert "not row-level proof" in audited["evidence_level"]
        data["meta"].loc[data["meta"].context == "Jurkat", "context"] = "Other"
        with pytest.raises(ValueError, match="only"):
            module.benchmark_provenance(data, {"revision": "test"})
    finally:
        sys.path.remove(str(scripts))
