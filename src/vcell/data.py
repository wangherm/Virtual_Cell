"""Chunked count preprocessing and strictly context-held-out mean-response data."""
from __future__ import annotations
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from .utils import read_config, write_json, hash_array


def inspect_h5ad(path):
    a = ad.read_h5ad(path, backed="r")
    try:
        return {"shape": list(a.shape), "obs_columns": list(a.obs.columns),
                "var_columns": list(a.var.columns), "layers": list(a.layers.keys()),
                "obs_examples": {k: a.obs[k].astype(str).unique()[:10].tolist() for k in a.obs.columns},
                "var_names_example": a.var_names[:10].tolist()}
    finally:
        a.file.close()


def open_source(entry, root):
    path = (Path(root) / entry["path"]).resolve()
    a = ad.read_h5ad(path, backed="r")
    gene_key = entry.get("gene_key")
    genes = np.asarray(a.var[gene_key].astype(str) if gene_key else a.var_names.astype(str))
    if len(set(genes)) != len(genes):
        a.file.close()
        raise ValueError(f"{path}: duplicate gene IDs; resolve duplicates explicitly before preparing.")
    return a, genes


def labels(a, entry):
    key = entry["perturbation_key"]
    if key not in a.obs:
        raise ValueError(f"Missing obs[{key!r}]; run vcell inspect and edit the manifest.")
    if a.obs[key].isna().any():
        raise ValueError(f"Missing perturbation labels in {entry['id']}; filter explicitly.")
    p = a.obs[key].astype(str).to_numpy()
    mapping = entry.get("perturbation_map", {})
    p = np.asarray([str(mapping.get(x, x)) for x in p], dtype=object)
    batch_key = entry.get("batch_key")
    if batch_key is None:
        warnings.warn(f"{entry['id']}: no batch column specified; using one explicitly declared batch.")
        batches = np.repeat("single_batch", len(p))
    else:
        if batch_key not in a.obs or a.obs[batch_key].isna().any():
            raise ValueError(f"Missing/invalid batch_key {batch_key!r} in {entry['id']}")
        batches = a.obs[batch_key].astype(str).to_numpy()
    controls = np.isin(p, [str(x) for x in entry["control_values"]])
    if not controls.any():
        raise ValueError(f"No controls found in {entry['id']}; check control_values.")
    p = p.copy()
    p[controls] = "__control__"
    return p, batches


def row_selection(a, entry):
    keep = np.ones(a.n_obs, dtype=bool)
    for key, value in entry.get("row_filter", {}).items():
        if key not in a.obs:
            raise ValueError(f"Missing row_filter column {key}")
        values = value if isinstance(value, list) else [value]
        keep &= a.obs[key].astype(str).isin([str(x) for x in values]).to_numpy()
    if not keep.any():
        raise ValueError(f"No rows match row_filter in {entry['id']}")
    return keep


def count_chunks(a, entry, selected_indices, chunk_size, target_sum):
    layer = entry.get("count_layer", "X")
    matrix = a.X if layer == "X" else a.layers[layer]
    row_mask = row_selection(a, entry)
    for start in range(0, a.n_obs, chunk_size):
        stop = min(a.n_obs, start + chunk_size)
        if not row_mask[start:stop].any():
            continue
        raw = matrix[start:stop, :]
        values = raw.data if sparse.issparse(raw) else np.asarray(raw)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Input must contain finite, nonnegative raw counts, not z-scores.")
        if values.size and not np.allclose(values, np.rint(values), atol=1e-5, rtol=0):
            raise ValueError("Non-integer input. Select a RAW COUNTS layer; refusing double normalization.")
        total = np.asarray(raw.sum(axis=1)).ravel().astype(np.float64)
        keep = (total > 0) & row_mask[start:stop]
        sub = raw[:, selected_indices]
        sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
        x = np.log1p(sub.astype(np.float64) * (target_sum / np.maximum(total, 1))[:, None])
        yield start, stop, x.astype(np.float32), keep


def aggregate_source(entry, root, genes, cfg, allowed_targets=None):
    a, source_genes = open_source(entry, root)
    try:
        lookup = {g: i for i, g in enumerate(source_genes)}
        missing = set(genes) - set(lookup)
        if missing:
            raise ValueError(f"Missing {len(missing)} genes in {entry['id']}; no silent zero filling.")
        idx = np.array([lookup[g] for g in genes])
        p, batch = labels(a, entry)
        out = {}
        for start, stop, x, keep in count_chunks(a, entry, idx, int(cfg["chunk_size"]), float(cfg["target_sum"])):
            for local in np.flatnonzero(keep):
                pert = p[start + local]
                if allowed_targets is not None and pert != "__control__" and pert not in allowed_targets:
                    continue
                key = (batch[start + local], pert)
                if key not in out:
                    out[key] = [np.zeros(len(genes), dtype=np.float64), 0]
                out[key][0] += x[local]
                out[key][1] += 1
        return out
    finally:
        a.file.close()


def validate_split_config(cfg):
    sets = [set(cfg[k]) for k in ("train_contexts", "val_contexts", "test_contexts")]
    if any(not s for s in sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("train/val/test contexts must be nonempty and mutually disjoint.")
    ids = [d["id"] for d in cfg["datasets"]]
    sources = [(str(Path(d["path"])), json.dumps(d.get("row_filter", {}), sort_keys=True)) for d in cfg["datasets"]]
    if len(set(ids)) != len(ids) or len(set(sources)) != len(sources):
        raise ValueError("Dataset IDs and (source path, row filter) pairs must be unique.")
    declared = set.union(*sets)
    if set(d["context"] for d in cfg["datasets"]) != declared:
        raise ValueError("Manifest contexts and split declarations must agree exactly.")


def prepare(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    cfg = read_config(manifest_path)
    validate_split_config(cfg)
    dest = (root / cfg["output_dir"]).resolve()
    if (dest / "dataset.npz").exists():
        raise FileExistsError(f"{dest} already has prepared data. Choose a new output_dir.")
    dest.mkdir(parents=True, exist_ok=True)
    common = None
    target_counts = {}
    source_info = []
    seen_rows = {}
    for entry in cfg["datasets"]:
        a, genes = open_source(entry, root)
        try:
            common = set(genes) if common is None else common.intersection(genes)
            p, _ = labels(a, entry)
            mask = row_selection(a, entry)
            actual_path = str((root / entry["path"]).resolve())
            if actual_path in seen_rows and (seen_rows[actual_path] & mask).any():
                raise ValueError("Source row filters overlap: a cell cannot appear in multiple dataset entries.")
            seen_rows[actual_path] = seen_rows.get(actual_path, np.zeros(a.n_obs, dtype=bool)) | mask
            if entry["context"] in cfg["train_contexts"]:
                for name, n in pd.Series(p[mask]).value_counts().items():
                    if name != "__control__":
                        target_counts[name] = target_counts.get(name, 0) + int(n)
            source_info.append({"id": entry["id"], "path": str((root / entry["path"]).resolve()),
                                "shape": list(a.shape), "context": entry["context"]})
        finally:
            a.file.close()
    all_genes = np.asarray(sorted(common))
    if len(all_genes) < 2:
        raise ValueError("Fewer than two common genes; check gene ID namespaces.")
    # Select high-variance features from TRAIN CELLS ONLY, not formal Seurat HVGs.
    sums = np.zeros(len(all_genes), dtype=np.float64)
    squares = sums.copy()
    n = 0
    for entry in cfg["datasets"]:
        if entry["context"] not in cfg["train_contexts"]:
            continue
        a, genes = open_source(entry, root)
        try:
            loc = {g: i for i, g in enumerate(genes)}
            idx = np.array([loc[g] for g in all_genes])
            for _, _, x, keep in count_chunks(a, entry, idx, int(cfg["chunk_size"]), float(cfg["target_sum"])):
                x = x[keep].astype(np.float64)
                sums += x.sum(0)
                squares += np.square(x).sum(0)
                n += len(x)
        finally:
            a.file.close()
    if n == 0:
        raise ValueError("No positive-library training cells.")
    variance = np.maximum(squares / n - np.square(sums / n), 0)
    ng = min(int(cfg["max_genes"]), len(all_genes))
    genes = all_genes[np.sort(np.argsort(-variance, kind="stable")[:ng])]
    chosen = sorted(target_counts, key=lambda p: (-target_counts[p], p))
    if cfg.get("max_perturbations"):
        chosen = chosen[:int(cfg["max_perturbations"])]
    targets = set(chosen)
    if any("+" in p or ";" in p for p in targets):
        raise ValueError("Compound/guide labels found. Supply an explicit perturbation_map to single target genes.")
    rows, baseline, delta, skipped = [], [], [], []
    for entry in cfg["datasets"]:
        agg = aggregate_source(entry, root, genes, cfg, targets)
        split = next(s for s in ("train", "val", "test") if entry["context"] in cfg[s + "_contexts"])
        for (batch, pert), (summed, count) in sorted(agg.items()):
            if pert == "__control__":
                continue
            ctrl = agg.get((batch, "__control__"))
            if count < cfg["min_cells"] or ctrl is None or ctrl[1] < cfg["min_control_cells"]:
                skipped.append({"dataset": entry["id"], "batch": batch, "perturbation": pert,
                                "reason": "insufficient perturbation cells or matched controls"})
                continue
            base = ctrl[0] / ctrl[1]
            baseline.append(base)
            delta.append(summed / count - base)
            rid = json.dumps([entry["id"], batch, pert], separators=(",", ":"))
            rows.append({"row_id": rid, "dataset": entry["id"], "context": entry["context"],
                         "batch": batch, "perturbation": pert, "split": split,
                         "n_cells": count, "n_controls": ctrl[1]})
    if not rows:
        raise ValueError("No valid groups after filtering.")
    meta = pd.DataFrame(rows)
    vocab = sorted(meta.loc[meta.split == "train", "perturbation"].unique())
    keep = meta.perturbation.isin(vocab).to_numpy()
    dropped_unknown = int((~keep).sum())
    meta = meta.loc[keep].reset_index(drop=True)
    base = np.asarray(baseline, dtype=np.float32)[keep]
    effect = np.asarray(delta, dtype=np.float32)[keep]
    pert_idx = np.asarray([vocab.index(p) for p in meta.perturbation], dtype=np.int64)
    save_prepared(dest, base, effect, pert_idx, genes, vocab, meta,
                  {"synthetic": False, "manifest": cfg, "sources": source_info,
                   "normalization": "mean(log1p(10000*counts/full_library))" if cfg["target_sum"] == 10000 else "mean(log1p(target_sum*counts/full_library))",
                   "target_sum": cfg["target_sum"], "feature_selection": "train-cell variance rank",
                   "n_training_cells_for_feature_selection": n,
                   "dropped_unseen_target_groups": dropped_unknown, "skipped_groups": skipped})
    return dest


def save_prepared(dest, baseline, delta, pert_idx, genes, vocab, meta, info):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    baseline, delta = baseline.astype(np.float32), delta.astype(np.float32)
    fingerprint = hash_array(baseline, delta, pert_idx, np.asarray(genes, dtype="U"),
                             np.asarray(vocab, dtype="U"), meta.row_id.to_numpy(dtype="U"),
                             meta.context.to_numpy(dtype="U"), meta.perturbation.to_numpy(dtype="U"),
                             meta.split.to_numpy(dtype="U"))
    np.savez_compressed(dest / "dataset.npz", baseline=baseline, delta=delta, pert_idx=pert_idx,
                        genes=np.asarray(genes, dtype="U"), perturbations=np.asarray(vocab, dtype="U"))
    meta.to_csv(dest / "metadata.csv", index=False)
    info.update({"fingerprint": fingerprint, "n_rows": len(meta), "n_genes": len(genes),
                 "n_perturbations": len(vocab), "split_counts": meta.split.value_counts().to_dict()})
    write_json(dest / "data_audit.json", info)
    load_prepared(dest)  # fail fast on empty partitions, duplicate keys, nonfinite values


def load_prepared(path):
    path = Path(path)
    with np.load(path / "dataset.npz", allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    d["meta"] = pd.read_csv(path / "metadata.csv", dtype={"row_id": str, "context": str,
                                                        "perturbation": str, "batch": str})
    d["audit"] = json.loads((path / "data_audit.json").read_text())
    m = d["meta"]
    if len(m) != len(d["delta"]) or m.row_id.duplicated().any():
        raise ValueError("Invalid/duplicate metadata row IDs.")
    if d["baseline"].shape != d["delta"].shape or d["baseline"].shape[1] != len(d["genes"]):
        raise ValueError("Expression shapes disagree.")
    for key in ("baseline", "delta"):
        if not np.isfinite(d[key]).all():
            raise ValueError(f"Nonfinite {key}")
    if set(m.split) != {"train", "val", "test"} or m.groupby("context").split.nunique().max() != 1:
        raise ValueError("Need nonempty, context-disjoint train/val/test splits.")
    actual = hash_array(d["baseline"], d["delta"], d["pert_idx"], d["genes"],
                        d["perturbations"], m.row_id.to_numpy(dtype="U"),
                        m.context.to_numpy(dtype="U"), m.perturbation.to_numpy(dtype="U"), m.split.to_numpy(dtype="U"))
    if actual != d["audit"]["fingerprint"]:
        raise ValueError("Prepared data fingerprint mismatch; do not edit prepared files in place.")
    if (d["pert_idx"] < 0).any() or (d["pert_idx"] >= len(d["perturbations"])).any():
        raise ValueError("Perturbation index outside vocabulary")
    if not np.array_equal(d["perturbations"][d["pert_idx"]], m.perturbation.to_numpy()):
        raise ValueError("Perturbation vocabulary and metadata disagree")
    train = (m.split == "train").to_numpy()
    if not set(d["pert_idx"]).issubset(set(d["pert_idx"][train])):
        raise ValueError("This release only supports targets seen in training.")
    d["splits"] = {s: np.flatnonzero((m.split == s).to_numpy()) for s in ("train", "val", "test")}
    return d
