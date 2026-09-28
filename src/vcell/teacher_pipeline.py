"""Real-control inference, frozen-feature task adaptation, and audited caches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_prepared, open_source, labels, row_selection
from .external_models import ScGPTEncoder, ScFoundationEncoder, StatePredictor
from .selection import mse, row_weights
from .teachers import check_provenance, load_cache
from .utils import device_from, seed_all, write_json, file_sha256, atomic_torch_save, hash_array


def control_groups(data, limit, seed, symbol_key="gene_name"):
    """Read only genuine control rows, preserving each file's full count library."""
    if limit < 1:
        raise ValueError("max_control_cells must be positive")
    manifest = data["audit"]["manifest"]
    paths = {s["id"]: s["path"] for s in data["audit"]["sources"]}
    rng = np.random.default_rng(seed)
    for original in manifest["datasets"]:
        entry = {**original, "path": paths[original["id"]]}
        a, gene_ids = open_source(entry, ".")
        try:
            p, batches = labels(a, entry)
            controls = (p == "__control__") & row_selection(a, entry)
            if symbol_key is None:
                symbols = gene_ids.tolist()
            elif symbol_key in a.var and not a.var[symbol_key].isna().any():
                symbols = a.var[symbol_key].astype(str).tolist()
            else:
                raise ValueError(f"Missing var[{symbol_key}] in {entry['id']}; configure gene_symbol_key")
            meta = data["meta"]
            for batch in sorted(meta.loc[meta.dataset == entry["id"], "batch"].unique()):
                pool = np.flatnonzero(controls & (batches == batch))
                if not len(pool):
                    raise ValueError(f"No matched controls for {entry['id']}/{batch}")
                chosen = np.sort(rng.choice(pool, min(limit, len(pool)), replace=False))
                layer = entry.get("count_layer", "X")
                matrix = a.X if layer == "X" else a.layers[layer]
                raw = matrix[chosen, :]
                raw = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
                if not np.isfinite(raw).all() or (raw < 0).any() or not np.allclose(raw, np.rint(raw), atol=1e-5, rtol=0):
                    raise ValueError("Teacher controls must be finite, nonnegative integer counts")
                nonzero = raw.sum(1) > 0
                raw, chosen = raw[nonzero].astype(np.float32), chosen[nonzero]
                if not len(raw):
                    raise ValueError("Sampled controls have zero libraries")
                rows = np.flatnonzero(((meta.dataset == entry["id"]) & (meta.batch == batch)).to_numpy())
                yield rows, raw, gene_ids.tolist(), symbols, {
                    "dataset": entry["id"], "batch": batch, "control_cell_ids": a.obs_names[chosen].tolist(),
                    "sample_counts_sha256": hash_array(raw), "n_available_controls": len(pool)}
        finally:
            a.file.close()


def write_cache(output, prediction, data, provenance, *, allow_unverified=False):
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Teacher cache output must end in .npz")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    if prediction.shape != data["delta"].shape or not np.isfinite(prediction).all():
        raise ValueError("Incomplete or nonfinite teacher predictions")
    check_provenance(provenance, data, allow_unverified=allow_unverified)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, delta=prediction.astype(np.float32), genes=data["genes"],
                        row_ids=data["meta"].row_id.to_numpy(dtype="U"), data_fingerprint=data["audit"]["fingerprint"])
    write_json(output.with_suffix(".json"), provenance)
    load_cache(output, data, allow_unverified=allow_unverified)


def export_native_teacher(cfg):
    data = load_prepared(cfg["data_dir"])
    family = cfg["family"]
    if family not in {"state", "scgpt", "scfoundation", "geneformer"}:
        raise ValueError("Unknown teacher family")
    if family == "state" and float(data["audit"]["target_sum"]) != 10000:
        raise ValueError("State adapter requires prepared target_sum=10000")
    output = Path(cfg["output"])
    if output.suffix != ".npz" or output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Choose a new .npz output path")
    provenance = json.loads(Path(cfg["provenance"]).read_text())
    if provenance.get("teacher_family") != family:
        raise ValueError("Provenance teacher_family differs from adapter")
    allow_unverified = cfg.get("allow_unverified_teachers") is True
    check_provenance(provenance, data, allow_unverified=allow_unverified)
    if provenance.get("prediction_space") != "prepared_log1p_delta":
        raise ValueError("Audit and declare prepared_log1p_delta prediction space")
    seed_all(int(cfg["seed"]))
    device = device_from(cfg["device"])
    from .foundation_encoders import FoundationEncoder, GeneformerEncoder
    from .state_encoder import StateEncoder
    native_state = family == "state" and cfg["model"].get("implementation") != "vcell_state_se100m_v1"
    cls = {"state": StatePredictor if native_state else StateEncoder, "scgpt": ScGPTEncoder,
           "scfoundation": FoundationEncoder if cfg["model"].get("implementation") == "vcell_scfoundation_v1" else ScFoundationEncoder,
           "geneformer": GeneformerEncoder}[family]
    backend = cls(cfg["model"], device)
    prediction = np.full_like(data["delta"], np.nan) if native_state else None
    features, samples = None, []
    for number, (rows, counts, genes, symbols, sample) in enumerate(control_groups(
            data, int(cfg["max_control_cells"]), int(cfg["seed"]), cfg.get("gene_symbol_key", "gene_name")), 1):
        samples.append(sample)
        if native_state:
            for row in rows:
                perturbed = backend.predict(counts, genes, data["meta"].iloc[row].perturbation, data["genes"])
                prediction[row] = perturbed - data["baseline"][row]
        else:
            embedding = backend.encode(counts, genes if family == "geneformer" else symbols).mean(0)
            if hasattr(backend, "last_input_audit"):
                mapping = backend.last_input_audit
                for group in mapping["duplicate_groups"]:
                    group["source_gene_ids"] = [genes[i] for i in group["source_columns"]]
                sample["gene_symbol_mapping"] = mapping
                print(f"{family} gene_mapping input={mapping['input_columns']} unique={mapping['unique_symbols']} "
                      f"merged_columns={mapping['merged_columns']} overlap={mapping['vocabulary_overlap_symbols']}", flush=True)
            if features is None:
                features = np.full((len(data["meta"]), len(embedding)), np.nan, dtype=np.float32)
            features[rows] = embedding
        print(f"{family} control_group={number} dataset={sample['dataset']} batch={sample['batch']} cells={len(counts)}", flush=True)
    source = {"adapter_config": cfg, "control_samples": samples,
              "checkpoint_sha256": file_sha256(cfg["model"]["checkpoint"]),
              "input_kind": "genuine single control cells sampled without replacement"}
    if native_state:
        write_cache(output, prediction, data, {**provenance, **source, "adaptation": "native State Transition inference"},
                    allow_unverified=allow_unverified)
    else:
        if features is None or not np.isfinite(features).all():
            raise ValueError("Missing/nonfinite control features")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, features=features, row_ids=data["meta"].row_id.to_numpy(dtype="U"),
                            data_fingerprint=data["audit"]["fingerprint"])
        write_json(output.with_suffix(".json"), {**provenance, **source,
                   "feature_file_sha256": file_sha256(output),
                   "artifact_kind": "frozen_control_features_NOT_predictions"})
    return output


class ConditionalHead(nn.Module):
    def __init__(self, features, perts, genes, hidden=128):
        super().__init__()
        self.context = nn.Linear(features, hidden)
        self.pert = nn.Embedding(perts, hidden)
        self.output = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, genes))

    def forward(self, x, p):
        c, t = F.gelu(self.context(x)), self.pert(p)
        return self.output(torch.cat([c, t, c * t], dim=1))


def fit_feature_teacher(cfg):
    data = load_prepared(cfg["data_dir"])
    features_path = Path(cfg["features"])
    provenance = json.loads(features_path.with_suffix(".json").read_text())
    if provenance.get("feature_file_sha256") and provenance["feature_file_sha256"] != file_sha256(features_path):
        raise ValueError("Frozen feature file checksum mismatch")
    if provenance.get("artifact_kind") != "frozen_control_features_NOT_predictions":
        raise ValueError("Expected native frozen control features")
    if provenance.get("teacher_family") not in {"scgpt", "scfoundation", "state", "geneformer"}:
        raise ValueError("Unknown frozen-feature teacher family")
    provenance["training_contexts"] = sorted(set(provenance["training_contexts"]) | set(data["meta"].loc[data["meta"].split == "train", "context"]))
    allow_unverified = cfg.get("allow_unverified_teachers") is True
    check_provenance(provenance, data, allow_unverified=allow_unverified)
    with np.load(features_path, allow_pickle=False) as f:
        if str(f["data_fingerprint"].item()) != data["audit"]["fingerprint"] or not np.array_equal(f["row_ids"], data["meta"].row_id.to_numpy(dtype="U")):
            raise ValueError("Features/data alignment mismatch")
        features = f["features"].astype(np.float32)
    if features.ndim != 2 or len(features) != len(data["meta"]) or not np.isfinite(features).all():
        raise ValueError("Invalid features")
    if min(cfg["epochs"], cfg["batch_size"], cfg["patience"], cfg["hidden"]) < 1:
        raise ValueError("Head training sizes must be positive")
    out = Path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a new teacher output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = device_from(cfg["device"])
    seed_all(cfg["seed"])
    train, val = data["splits"]["train"], data["splits"]["val"]
    mean, std = features[train].mean(0), np.maximum(features[train].std(0), .01)
    scale = max(float(np.sqrt(np.mean(data["delta"][train] ** 2))), .05)
    x = torch.tensor((features - mean) / std)
    p, y = torch.tensor(data["pert_idx"]), torch.tensor(data["delta"] / scale)
    model = ConditionalHead(features.shape[1], len(data["perturbations"]), len(data["genes"]), cfg["hidden"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=.01)
    best, stale, history = float("inf"), 0, []

    @torch.no_grad()
    def infer(indices):
        model.eval()
        return np.concatenate([model(x[ix].to(device), p[ix].to(device)).cpu().numpy()
                               for ix in (indices[j:j+cfg["batch_size"]] for j in range(0, len(indices), cfg["batch_size"]))]) * scale

    for epoch in range(cfg["epochs"]):
        model.train()
        order = np.random.default_rng(cfg["seed"] + epoch).permutation(train)
        for j in range(0, len(order), cfg["batch_size"]):
            ix = order[j:j+cfg["batch_size"]]
            optim.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x[ix].to(device), p[ix].to(device)), y[ix].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite teacher head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optim.step()
        score = mse(infer(val), data["delta"][val], row_weights(data["meta"].iloc[val]))
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite teacher validation score")
        if score < best:
            best, stale = score, 0
            atomic_torch_save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "feature_mean": torch.tensor(mean), "feature_std": torch.tensor(std), "delta_scale": scale,
                "config": cfg, "epoch": epoch+1, "features_sha256": file_sha256(features_path),
                "data_fingerprint": data["audit"]["fingerprint"], "genes": data["genes"].tolist(),
                "perturbations": data["perturbations"].tolist()}, out / "head.pt")
        else:
            stale += 1
        history.append({"epoch": epoch+1, "val_mse_delta": score})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"{provenance['teacher_family']} head epoch={epoch+1} val_mse_delta={score:.7f}", flush=True)
        if stale >= cfg["patience"]:
            break
    saved = torch.load(out / "head.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state"])
    provenance.update(adaptation="frozen pretrained control encoder + VCell conditional response head",
                      prediction_space="prepared_log1p_delta", feature_file_sha256=file_sha256(features_path),
                      head_sha256=file_sha256(out / "head.pt"), head_config=cfg,
                      selection_contexts=sorted(data["meta"].loc[data["meta"].split == "val", "context"].unique().tolist()))
    write_cache(out / "predictions.npz", infer(np.arange(len(features))), data, provenance,
                allow_unverified=allow_unverified)
    write_json(out / "validation.json", {"best_epoch": saved["epoch"], "mse_delta": best, "test_evaluated": False})
    return out / "predictions.npz"
