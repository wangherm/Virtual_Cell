"""Offline software checks with a TINY RANDOM Qwen fixture, never a research teacher."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from transformers import Qwen3Config, Qwen3Model, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from vcell.qwen import QwenResponse, run_qwen, load_qwen_checkpoint, predict, teacher_targets
from vcell.data import load_prepared
from vcell.synthetic import make_demo
from vcell.train import tensors
from vcell.utils import read_config, write_json
from vcell.teacher_pipeline import write_cache

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    base = tmp_path / "tiny_random_fixture"
    base.mkdir()
    Qwen3Model(Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128)).save_pretrained(base)
    tok = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "Predict": 2, "expression": 3,
                                      "changes": 4, "after": 5, "genetic": 6, "perturbation": 7, "of": 8}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]").save_pretrained(base)
    prepared = make_demo(tmp_path / "data", genes=16, targets=4, cells=8)
    cfg = read_config(ROOT / "configs/qwen_smoke.yaml")
    cfg.update(data_dir=str(prepared), output_dir=str(tmp_path / "run"), device="cpu", num_threads=1,
               max_train_rows=8, max_val_rows=4, batch_size=3, gradient_accumulation=2)
    cfg["model"].update(model_id=str(base), revision=None, dtype="float32", local_files_only=True,
                        control_tokens=2, lora_rank=2, lora_alpha=4)
    return cfg, load_prepared(prepared)


def test_qwen_gradients_reach_inputs_and_lora_but_not_frozen_base(setup):
    cfg, data = setup
    model = QwenResponse(16, data["perturbations"], cfg["model"])
    out = model(torch.randn(3, 16), torch.tensor([0, 1, 2]))
    assert out.shape == (3, 16)
    out.square().mean().backward()
    assert model.input_projector[0].weight.grad.abs().sum() > 0
    assert model.prediction_token.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all("base_layer" not in name for name in model.adapter_state())


def test_train_resume_reload_and_source_guard(setup):
    cfg, data = setup
    run_qwen(cfg)
    run_qwen(cfg, resume=True)
    folder = Path(cfg["output_dir"])
    model, saved = load_qwen_checkpoint(folder / "supervised/best.pt")
    norm = {k: v.numpy() if torch.is_tensor(v) else v for k, v in saved["normalization"].items()}
    manifest = json.loads((folder / "run_manifest.json").read_text())
    ix = np.asarray(manifest["val_rows"])
    actual = predict(model, tensors(data, norm), ix, "cpu", cfg["batch_size"]) * norm["scale"]
    with np.load(folder / "supervised/validation_predictions.npz") as f:
        np.testing.assert_allclose(actual, f["delta"], atol=1e-6)
    assert not manifest["test_evaluated"]
    assert set(manifest["train_rows"]) <= set(data["splits"]["train"])
    changed = copy.deepcopy(cfg)
    changed["learning_rate"] *= 2
    with pytest.raises(ValueError, match="changed"):
        run_qwen(changed, resume=True)
    with pytest.raises(FileExistsError):
        run_qwen(cfg)


def test_missing_teachers_fail_before_backbone_download(setup):
    cfg, data = setup
    cfg["arms"] = ["all"]
    with pytest.raises(ValueError, match="Missing real teacher caches"):
        run_qwen(cfg)
    assert not Path(cfg["output_dir"]).exists()


def test_three_teacher_distillation_and_family_guard(setup, tmp_path):
    cfg, data = setup
    for i, family in enumerate(("state", "scgpt", "scfoundation")):
        path = tmp_path / (family + ".npz")
        # Explicit artificial caches exercise the loss, not biological performance.
        provenance = {"teacher_name": "TEST_FIXTURE", "teacher_family": family, "model_revision": "TEST_ONLY",
            "training_contexts": ["K562", "RPE1"], "excluded_contexts": ["HepG2", "Jurkat"],
            "source": "unit test random arrays", "license": "test fixture", "audit_notes": "TEST ONLY",
            "declared_no_holdout_perturbations": True, "prediction_space": "prepared_log1p_delta"}
        write_cache(path, np.random.default_rng(i).normal(0, .1, data["delta"].shape), data, provenance)
        cfg["teachers"][family] = str(path)
    cfg["arms"], cfg["epochs"] = ["all"], 1
    run_qwen(cfg)
    assert (Path(cfg["output_dir"]) / "all/best.pt").exists()
    sidecar = Path(cfg["teachers"]["state"]).with_suffix(".json")
    provenance = json.loads(sidecar.read_text())
    provenance["teacher_family"] = "scgpt"
    write_json(sidecar, provenance)
    with pytest.raises(ValueError, match="teacher_family"):
        teacher_targets(cfg, data, data["splits"]["val"])


def test_interrupted_resume_matches_uninterrupted(setup, monkeypatch):
    from vcell import qwen
    cfg, data = setup
    baseline = copy.deepcopy(cfg)
    baseline["output_dir"] += "_reference"
    run_qwen(baseline)
    save = qwen.atomic_torch_save

    def interrupt(value, path):
        save(value, path)
        if Path(path).name == "last.pt" and value["epoch"] == 1:
            raise InterruptedError("test interruption")

    monkeypatch.setattr(qwen, "atomic_torch_save", interrupt)
    with pytest.raises(InterruptedError):
        run_qwen(cfg)
    monkeypatch.setattr(qwen, "atomic_torch_save", save)
    run_qwen(cfg, resume=True)
    with np.load(Path(cfg["output_dir"]) / "supervised/validation_predictions.npz") as a, np.load(Path(baseline["output_dir"]) / "supervised/validation_predictions.npz") as b:
        np.testing.assert_allclose(a["delta"], b["delta"], atol=1e-6)
