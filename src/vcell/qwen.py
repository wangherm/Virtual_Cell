"""Pretrained Qwen + continuous control tokens + LoRA + signed regression head.

This separate trainer preserves the v0.3 reference-model workflow. Biological
teacher weights are never substituted with random models or observed outcomes.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_prepared
from .selection import row_weights, mse, fit_mix, combine
from .teachers import load_cache
from .train import normalization, tensors, simple_baselines, source_fingerprint
from .utils import seed_all, device_from, atomic_torch_save, write_json, file_sha256

DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
DEFAULT_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
TEACHERS = ("state", "scgpt", "scfoundation")
SUPPORTED_TEACHERS = (*TEACHERS, "geneformer", "uce")
ARMS = ("supervised", *SUPPORTED_TEACHERS, "all")


def teacher_names(cfg):
    names = tuple(cfg.get("teacher_families", TEACHERS))
    if len(names) not in (3, 4, 5) or len(set(names)) != len(names) or set(names) - set(SUPPORTED_TEACHERS):
        raise ValueError("teacher_families must name three, four or five distinct supported pretrained families")
    return names


def backbone_identity(spec):
    """Immutable HF revision, or content hashes for an offline local snapshot."""
    folder = Path(spec["model_id"])
    if folder.is_dir():
        files = sorted(p for p in folder.rglob("*") if p.is_file()
                       and p.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"})
        if not files:
            raise ValueError("Local backbone directory is empty")
        return {str(p.relative_to(folder)): file_sha256(p) for p in files}
    revision = spec.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin a 40-character Hugging Face commit revision (not main)")
    return {"model_id": spec["model_id"], "revision": revision}


class QwenResponse(nn.Module):
    def __init__(self, genes, perturbations, spec, prompt_ids=None, prompt_mask=None, functional_vectors=None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[spec["dtype"]]
        kw = {"revision": spec.get("revision"), "trust_remote_code": False,
              "local_files_only": bool(spec.get("local_files_only", False))}
        base = AutoModel.from_pretrained(spec["model_id"], torch_dtype=dtype,
                                        attn_implementation="sdpa", **kw)
        if base.config.model_type != "qwen3":
            raise ValueError("This adapter requires a Qwen3 backbone")
        base.config.use_cache = False
        self.backbone = get_peft_model(base, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=int(spec["lora_rank"]),
            lora_alpha=int(spec["lora_alpha"]), lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], bias="none"))
        if spec.get("gradient_checkpointing", True):
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        if prompt_ids is None:
            tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], **kw)
            tokenizer.padding_side = "right"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            encoded = tokenizer([f"Predict expression changes after genetic perturbation of {p}."
                                 for p in perturbations], padding=True, return_tensors="pt")
            prompt_ids, prompt_mask = encoded["input_ids"], encoded["attention_mask"]
        self.register_buffer("prompt_ids", prompt_ids.long())
        self.register_buffer("prompt_mask", prompt_mask.long())
        h = base.config.hidden_size
        self.n_tokens = int(spec["control_tokens"])
        self.input_projector = nn.Sequential(nn.Linear(genes, h), nn.GELU(),
                                             nn.Linear(h, self.n_tokens * h))
        self.prediction_token = nn.Parameter(torch.randn(1, 1, h) * .02)
        self.output = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, genes))
        if spec.get("functional"):
            from .functional import load_function
            values = load_function(spec["functional"], perturbations) if functional_vectors is None else functional_vectors
            values = torch.as_tensor(values, dtype=torch.float32)
            if values.ndim != 2 or len(values) != len(perturbations) or not torch.isfinite(values).all():
                raise ValueError("Invalid checkpoint functional vectors")
            self.register_buffer("functional_vectors", values)
            self.functional_projector = nn.Sequential(nn.Linear(values.shape[1], h), nn.GELU(), nn.LayerNorm(h))

    def forward(self, x, p):
        controls = self.input_projector(x.float()).reshape(len(x), self.n_tokens, -1)
        if hasattr(self, "functional_vectors"):
            functional = self.functional_projector(self.functional_vectors[p]).unsqueeze(1)
            controls = torch.cat([controls, functional], dim=1)
        words = self.backbone.get_input_embeddings()(self.prompt_ids[p])
        final = self.prediction_token.expand(len(x), -1, -1)
        inputs = torch.cat([controls.to(words.dtype), words, final.to(words.dtype)], dim=1)
        mask = torch.cat([torch.ones((len(x), controls.shape[1]), device=x.device, dtype=torch.long),
                          self.prompt_mask[p], torch.ones((len(x), 1), device=x.device, dtype=torch.long)], 1)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        hidden = self.backbone(inputs_embeds=inputs, attention_mask=mask,
                               position_ids=positions, use_cache=False).last_hidden_state[:, -1]
        return self.output(hidden.float())

    def adapter_state(self):
        # No copy of the frozen ~0.6B backbone in every experiment checkpoint.
        return {k: p.detach().cpu().clone() for k, p in self.named_parameters() if p.requires_grad}

    def load_adapter_state(self, state):
        expected = {k for k, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError("Adapter parameter names differ; use the saved backbone/LoRA configuration")
        params = dict(self.named_parameters())
        with torch.no_grad():
            for k, value in state.items():
                params[k].copy_(value.to(params[k]))


@torch.no_grad()
def predict(model, ts, indices, device, batch_size):
    model.eval()
    return np.concatenate([model(ts["x"][ix].to(device), ts["p"][ix].to(device)).float().cpu().numpy()
                           for ix in (indices[i:i + batch_size] for i in range(0, len(indices), batch_size))])


def selected_rows(indices, limit, seed):
    if limit is None:
        return indices.copy()
    if int(limit) < 1:
        raise ValueError("Row limits must be positive or null")
    return np.sort(np.random.default_rng(seed).choice(indices, min(int(limit), len(indices)), replace=False))


def validate_config(cfg):
    teacher_names(cfg)
    if not cfg["arms"] or len(set(cfg["arms"])) != len(cfg["arms"]) or set(cfg["arms"]) - set(ARMS):
        raise ValueError(f"arms must be a nonempty unique list from {ARMS}")
    for key in ("epochs", "batch_size", "gradient_accumulation", "patience", "num_threads", "log_every"):
        if int(cfg[key]) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("learning_rate", "weight_decay", "kd_weight"):
        if not math.isfinite(float(cfg[key])) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if cfg["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    if cfg["teacher_mix"] not in {"equal", "validation", "reliability"}:
        raise ValueError("teacher_mix must be equal, validation or reliability")
    if cfg.get("loss_weighting", "uniform") not in {"uniform", "context_perturbation"}:
        raise ValueError("Unknown loss_weighting")
    if cfg["model"]["dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("Use float32 or bfloat16; float16 has no scaler in this trainer")
    for key in ("control_tokens", "lora_rank", "lora_alpha"):
        if int(cfg["model"][key]) < 1:
            raise ValueError(f"Invalid model.{key}")


def teacher_targets(cfg, data, val):
    names = teacher_names(cfg)
    needed = set(names) if "all" in cfg["arms"] else set(cfg["arms"]) - {"supervised"}
    unknown = set(cfg.get("teachers", {})) - set(SUPPORTED_TEACHERS)
    if unknown:
        raise ValueError(f"Unknown teacher names: {unknown}")
    missing = needed - set(cfg.get("teachers", {}))
    if missing:
        raise ValueError(f"Missing real teacher caches: {sorted(missing)}. Use arms: [supervised] for the first Qwen check.")
    targets, provenance, hashes = {}, {}, {}
    for name in sorted(needed):
        path = Path(cfg["teachers"][name])
        targets[name], provenance[name] = load_cache(path, data,
            allow_unverified=cfg.get("allow_unverified_teachers") is True)
        if provenance[name].get("teacher_family") != name:
            raise ValueError(f"{name}: sidecar teacher_family must match the configured family")
        if provenance[name].get("prediction_space") != "prepared_log1p_delta":
            raise ValueError(f"{name}: audit and declare prediction_space=prepared_log1p_delta")
        hashes[name] = [file_sha256(path), file_sha256(path.with_suffix(".json"))]
    weights = None
    if "all" in cfg["arms"]:
        values = [targets[n] for n in names]
        if cfg["teacher_mix"] == "reliability":
            report = load_reliability(cfg, data)
            if report["teacher_names"] != list(names) or report["teacher_cache_hashes"] != hashes:
                raise ValueError("Reliability report does not match these teacher caches/order")
            weights = np.array(report["weights"])
            if weights.shape != (len(names),) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
                raise ValueError("Invalid reliability mixture weights")
        else:
            weights = (fit_mix([a[val] for a in values], data["delta"][val], row_weights(data["meta"].iloc[val]))
                   if cfg["teacher_mix"] == "validation" else np.ones(len(names)) / len(names))
        targets["all"] = combine(values, weights)
    return targets, provenance, hashes, None if weights is None else weights.tolist()


def load_reliability(cfg, data):
    spec = cfg["reliability"]
    if file_sha256(spec["path"]) != spec["sha256"]:
        raise ValueError("Reliability report changed")
    report = json.loads(Path(spec["path"]).read_text())
    if report["data_fingerprint"] != data["audit"]["fingerprint"] or not set(report["eligible_rows"]).issubset(set(data["splits"]["train"])):
        raise ValueError("Reliability must be estimated from training rows only")
    if report["selected_using"] != "training-context cross-fit predictions only" or not 0 <= report["kd_strength"] <= 1:
        raise ValueError("Invalid reliability selection provenance")
    return report


def training_weights(cfg, data, train_idx):
    weights = np.ones(len(data["meta"]), dtype=np.float32)
    if cfg.get("loss_weighting") == "context_perturbation":
        weights[train_idx] = row_weights(data["meta"].iloc[train_idx]) * len(train_idx)
    gate = np.ones(len(weights), dtype=np.float32)
    if cfg.get("kd_mask"):
        spec = cfg["kd_mask"]
        if file_sha256(spec["path"]) != spec["sha256"]:
            raise ValueError("KD row mask changed")
        with np.load(spec["path"]) as f:
            if not np.array_equal(f["row_ids"], data["meta"].row_id.to_numpy(dtype="U")) or f["eligible"].shape != gate.shape:
                raise ValueError("KD row mask/data mismatch")
            gate = f["eligible"].astype(np.float32)
        if not np.isin(gate, [0, 1]).all():
            raise ValueError("KD row mask must be binary")
    if cfg["teacher_mix"] == "reliability" and "all" in cfg["arms"]:
        gate *= load_reliability(cfg, data)["kd_strength"]
    return torch.tensor(weights), torch.tensor(gate)


def train_arm(cfg, data, targets, arm, train_idx, val_idx, dest, stamp, resume):
    device = device_from(cfg["device"])
    seed_all(cfg["seed"])
    model = QwenResponse(len(data["genes"]), data["perturbations"], cfg["model"]).to(device)
    norm = normalization(data)
    ts = tensors(data, norm)
    loss_weights, kd_gate = training_weights(cfg, data, train_idx)
    kd = None if arm == "supervised" else torch.tensor(targets[arm] / norm["scale"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    folder = dest / arm
    folder.mkdir(exist_ok=True)
    first, stale, history, best = 0, 0, [], float("inf")
    last = folder / "last.pt"
    if resume and last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["run_fingerprint"] != stamp:
            raise ValueError("Checkpoint/run mismatch")
        model.load_adapter_state(saved["trainable_state"])
        optimizer.load_state_dict(saved["optimizer"])
        first, stale, history, best = saved["epoch"], saved["stale"], saved["history"], saved["best"]
    counts = {"total_parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in parameters)}
    write_json(folder / "model_info.json", counts)
    print(f"{arm}: {counts}; train_rows={len(train_idx)} val_rows={len(val_idx)}", flush=True)
    common = {"format": "vcell-qwen-adapter-v1", "model_spec": cfg["model"],
              "backbone_identity": backbone_identity(cfg["model"]),
              "genes": data["genes"].tolist(), "perturbations": data["perturbations"].tolist(),
              "prompt_ids": model.prompt_ids.cpu(), "prompt_mask": model.prompt_mask.cpu(),
              "functional_vectors": model.functional_vectors.cpu() if hasattr(model, "functional_vectors") else None,
              "normalization": {"mean": torch.tensor(norm["mean"]), "std": torch.tensor(norm["std"]), "scale": norm["scale"]},
              "data_fingerprint": data["audit"]["fingerprint"], "run_fingerprint": stamp}
    val_weights = row_weights(data["meta"].iloc[val_idx])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    effective = cfg["batch_size"] * cfg["gradient_accumulation"]
    for epoch in range(first, cfg["epochs"]):
        if stale >= cfg["patience"]:
            break
        started = time.time()
        seed_all(cfg["seed"] + 10000 + epoch)
        order = np.random.default_rng(cfg["seed"] + epoch).permutation(train_idx)
        model.train()
        totals = np.zeros(3)
        for step, start in enumerate(range(0, len(order), effective), 1):
            window = order[start:start + effective]
            optimizer.zero_grad(set_to_none=True)
            for j in range(0, len(window), cfg["batch_size"]):
                ix = window[j:j + cfg["batch_size"]]
                prediction = model(ts["x"][ix].to(device), ts["p"][ix].to(device))
                weight = loss_weights[ix].to(device)
                sup = ((prediction - ts["y"][ix].to(device)).square().mean(1) * weight).mean()
                distill = (((prediction - kd[ix].to(device)).square().mean(1) * weight * kd_gate[ix].to(device)).mean()
                           if kd is not None else sup.detach() * 0)
                loss = sup + cfg["kd_weight"] * distill
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite Qwen loss")
                (loss * len(ix) / len(window)).backward()
                totals += np.array([sup.item(), distill.item(), loss.item()]) * len(ix)
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % cfg["log_every"] == 0:
                print(f"{arm} epoch={epoch+1}/{cfg['epochs']} step={step}/{math.ceil(len(order)/effective)} loss={loss.item():.5f}", flush=True)
        pred = predict(model, ts, val_idx, device, cfg["batch_size"]) * norm["scale"]
        score = mse(pred, data["delta"][val_idx], val_weights)
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation score")
        if score < best:
            best, stale = score, 0
            atomic_torch_save({**common, "trainable_state": model.adapter_state(), "epoch": epoch + 1}, folder / "best.pt")
        else:
            stale += 1
        record = dict(zip(("supervised_loss", "kd_loss", "loss"), (totals / len(order)).tolist()))
        record.update(epoch=epoch + 1, val_mse_delta=score, seconds=time.time() - started)
        history.append(record)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        atomic_torch_save({"trainable_state": model.adapter_state(), "optimizer": optimizer.state_dict(),
                           "epoch": epoch + 1, "stale": stale, "history": history, "best": best,
                           "run_fingerprint": stamp}, last)
        print(f"{arm} epoch={epoch+1} val_mse_delta={score:.7f} seconds={record['seconds']:.1f}", flush=True)
    best_saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_adapter_state(best_saved["trainable_state"])
    pred = predict(model, ts, val_idx, device, cfg["batch_size"]) * norm["scale"]
    np.savez_compressed(folder / "validation_predictions.npz", delta=pred, genes=data["genes"],
                        row_ids=data["meta"].iloc[val_idx].row_id.to_numpy(dtype="U"),
                        data_fingerprint=data["audit"]["fingerprint"])
    write_json(folder / "validation.json", {"mse_delta": mse(pred, data["delta"][val_idx], val_weights),
               "best_epoch": best_saved["epoch"], "val_rows": len(val_idx),
               "cuda_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
    del model, optimizer, parameters
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred


def run_qwen(cfg, resume=False):
    validate_config(cfg)
    if cfg.get("allow_unverified_teachers") is True:
        print("EXPLORATORY TEACHERS: pretraining overlap is unverified; validation is for internal selection only", flush=True)
    device = device_from(cfg["device"])
    if cfg["model"]["dtype"] == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 requires a supported CUDA GPU; set model.dtype=float32 for CPU tests")
    torch.set_num_threads(cfg["num_threads"])
    data = load_prepared(cfg["data_dir"])
    train = selected_rows(data["splits"]["train"], cfg.get("max_train_rows"), cfg["seed"])
    val = selected_rows(data["splits"]["val"], cfg.get("max_val_rows"), cfg["seed"] + 1)
    # Validate every required cache BEFORE downloading the student backbone.
    targets, provenance, hashes, weights = teacher_targets(cfg, data, val)
    identity = backbone_identity(cfg["model"])
    stamp_data = {"config": cfg, "data_fingerprint": data["audit"]["fingerprint"],
                  "source_fingerprint": source_fingerprint(), "backbone_identity": identity,
                  "teacher_cache_hashes": hashes}
    stamp = hashlib.sha256(json.dumps(stamp_data, sort_keys=True).encode()).hexdigest()
    dest = Path(cfg["output_dir"])
    if dest.exists() and any(dest.iterdir()):
        if not resume or not (dest / "run_manifest.json").exists():
            raise FileExistsError(f"Use a fresh output directory or --resume: {dest}")
        old = json.loads((dest / "run_manifest.json").read_text())
        if old["run_fingerprint"] != stamp:
            raise ValueError("Config, source, backbone, data or caches changed; use a new output directory")
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "run_manifest.json", {**stamp_data, "run_fingerprint": stamp,
               "train_rows": train.tolist(), "val_rows": val.tolist(), "test_evaluated": False,
               "teacher_sources": provenance, "teacher_weights_order": list(teacher_names(cfg)), "teacher_weights": weights,
               "normalization_fitted_on": "all prepared training rows",
               "run_kind": "smoke" if cfg.get("max_train_rows") or cfg.get("max_val_rows") else "pilot"})
    predictions = {n: a[val] for n, a in simple_baselines(data).items()}
    predictions.update({"teacher/" + n: a[val] for n, a in targets.items()})
    for arm in cfg["arms"]:
        predictions["qwen/" + arm] = train_arm(cfg, data, targets, arm, train, val, dest, stamp, resume)
        records = [{"model": name, "split": "val", "mse_delta": mse(pred, data["delta"][val], row_weights(data["meta"].iloc[val])),
                    "n_rows": len(val)} for name, pred in predictions.items()]
        pd.DataFrame(records).sort_values("mse_delta").to_csv(dest / "summary.csv", index=False)
    write_json(dest / "COMPLETE.json", {"run_fingerprint": stamp, "arms": cfg["arms"], "test_evaluated": False})
    print(f"QWEN RUN COMPLETE: {dest / 'summary.csv'}", flush=True)
    return dest


def load_qwen_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "vcell-qwen-adapter-v1":
        raise ValueError("Not a Qwen adapter checkpoint")
    if backbone_identity(saved["model_spec"]) != saved["backbone_identity"]:
        raise ValueError("Frozen backbone files changed")
    model = QwenResponse(len(saved["genes"]), saved["perturbations"], saved["model_spec"],
                         saved["prompt_ids"], saved["prompt_mask"], saved.get("functional_vectors")).to(device).eval()
    model.load_adapter_state(saved["trainable_state"])
    return model, saved


def predict_checkpoint(checkpoint, data_dir, output, device="cpu", batch_size=16):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = Path(output)
    if output.suffix != ".npz" or output.exists():
        raise ValueError("Choose a new .npz output path")
    device = device_from(device)
    data = load_prepared(data_dir)
    model, saved = load_qwen_checkpoint(checkpoint, device)
    if saved["genes"] != data["genes"].tolist() or saved["perturbations"] != data["perturbations"].tolist():
        raise ValueError("Gene order or perturbation vocabulary mismatch")
    norm = saved["normalization"]
    norm = {k: v.numpy() if torch.is_tensor(v) else v for k, v in norm.items()}
    pred = predict(model, tensors(data, norm), np.arange(len(data["meta"])), device, batch_size) * norm["scale"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, delta=pred, genes=data["genes"], row_ids=data["meta"].row_id.to_numpy(dtype="U"),
                        data_fingerprint=data["audit"]["fingerprint"])
