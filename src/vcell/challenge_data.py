"""Official 2025 training counts aligned to an existing VCell evaluation panel."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from anndata.io import read_elem

from .data import load_prepared, save_prepared
from .remote_h5 import RangeFile
from .utils import file_sha256, hash_array, write_json


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_suffix(".part")
    with temp.open("wb") as f:
        np.savez(f, **arrays)
    temp.replace(path)


def aggregate_training(source, reference, output, *, local_h5ad=None, resume=False,
                       chunk_rows=512, checkpoint_rows=8192):
    """Stream all raw libraries; retain only common output genes in group sums."""
    reference = load_prepared(reference)
    if reference["audit"].get("target_sum") != 10000:
        raise ValueError("VCC comparison requires reference target_sum=10000")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    local_hash = file_sha256(local_h5ad) if local_h5ad else None
    plan = {"source": source, "reference_fingerprint": reference["audit"]["fingerprint"],
            "local_h5ad_sha256": local_hash, "code_sha256": file_sha256(__file__)}
    stamp = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if (out / "plan.json").exists():
        if not resume or json.loads((out / "plan.json").read_text()) != plan:
            raise ValueError("Existing aggregation: resume identical inputs or choose a new name")
    elif any(out.iterdir()):
        raise ValueError("Aggregation directory has files without a plan")
    write_json(out / "plan.json", plan)
    artifact = out / "groups.npz"
    if artifact.exists() and (out / "COMPLETE.json").exists():
        complete = json.loads((out / "COMPLETE.json").read_text())
        if complete["fingerprint"] != stamp or complete["sha256"] != file_sha256(artifact):
            raise ValueError("Completed challenge aggregation changed")
        print("VCC AGGREGATES VERIFIED: reusing completed stage", flush=True)
        return artifact
    if min(chunk_rows, checkpoint_rows) < 1:
        raise ValueError("Chunk/checkpoint sizes must be positive")
    with ExitStack() as stack:
        remote = None if local_h5ad else stack.enter_context(RangeFile(source))
        h = stack.enter_context(h5py.File(local_h5ad or remote, "r"))
        obs, var = read_elem(h["obs"]), read_elem(h["var"])
        if (len(obs), len(var)) != (source["n_obs"], source["n_vars"]):
            raise ValueError("Source shape differs from official pinned training split")
        gene_ids = var[source["gene_key"]].astype(str).to_numpy()
        if len(set(gene_ids)) != len(gene_ids):
            raise ValueError("Duplicate challenge Ensembl IDs require explicit resolution")
        gene_lookup = {g: i for i, g in enumerate(gene_ids)}
        genes = np.array([g for g in reference["genes"] if g in gene_lookup], dtype="U")
        if len(genes) < 2:
            raise ValueError("Fewer than two common output genes")
        selected = np.array([gene_lookup[g] for g in genes])
        target_key, batch_key = source["perturbation_key"], source["batch_key"]
        if obs[[target_key, batch_key]].isna().any().any():
            raise ValueError("Missing challenge target/batch annotations")
        perts, batches = obs[target_key].astype(str).to_numpy(), obs[batch_key].astype(str).to_numpy()
        if source["control"] not in perts:
            raise ValueError("No official non-targeting controls")
        pairs = sorted(set(zip(batches, perts)))
        lookup = {key: i for i, key in enumerate(pairs)}
        group_index = np.array([lookup[key] for key in zip(batches, perts)])
        sums = np.zeros((len(pairs), len(genes)), dtype=np.float64)
        counts = np.zeros(len(pairs), dtype=np.int64)
        current, skipped_zero = 0, 0
        progress = out / "partial.npz"
        if progress.exists():
            with np.load(progress, allow_pickle=False) as saved:
                if str(saved["fingerprint"].item()) != stamp:
                    raise ValueError("Partial aggregation belongs to another source")
                sums, counts = saved["sums"], saved["counts"]
                current, skipped_zero = int(saved["next_row"]), int(saved["skipped_zero"])
            if sums.shape != (len(pairs), len(genes)) or counts.shape != (len(pairs),) or not 0 <= current <= len(obs):
                raise ValueError("Invalid partial aggregation dimensions")
        matrix = h[source["matrix"]]
        encoding = matrix.attrs.get("encoding-type")
        if encoding != "csr_matrix" or tuple(matrix.attrs["shape"]) != (len(obs), len(var)):
            raise ValueError("Expected the official raw CSR count matrix")
        indptr = matrix["indptr"][:].astype(np.int64)
        print(f"VCC AGGREGATE START: cells={len(obs)} common_genes={len(genes)}/{len(reference['genes'])} "
              f"groups={len(pairs)} resume_row={current}", flush=True)
        last_saved = current
        while current < len(obs):
            stop = min(current + chunk_rows, len(obs))
            lo, hi = int(indptr[current]), int(indptr[stop])
            values = matrix["data"][lo:hi]
            if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.rint(values), atol=1e-5, rtol=0):
                raise ValueError("Challenge X must be finite nonnegative integer raw counts")
            block = sparse.csr_matrix((values, matrix["indices"][lo:hi], indptr[current:stop+1]-lo),
                                      shape=(stop-current, len(var)))
            totals = np.asarray(block.sum(1, dtype=np.float64)).ravel()
            keep = totals > 0
            normalized = np.log1p(block[:, selected].toarray().astype(np.float64) * (10000 / np.maximum(totals, 1))[:, None])
            ix = group_index[current:stop][keep]
            np.add.at(sums, ix, normalized[keep])
            np.add.at(counts, ix, 1)
            skipped_zero += int((~keep).sum())
            current = stop
            if current - last_saved >= checkpoint_rows or current == len(obs):
                atomic_npz(progress, sums=sums, counts=counts, next_row=current, skipped_zero=skipped_zero, fingerprint=stamp)
                status = {"cells_done": current, "cells_total": len(obs), "percent": 100*current/len(obs),
                          "bytes_received_this_session": remote.bytes_received if remote else 0}
                write_json(out / "progress.json", status)
                print(f"VCC AGGREGATE: {current}/{len(obs)} cells ({status['percent']:.1f}%)", flush=True)
                last_saved = current
        atomic_npz(artifact, sums=sums, counts=counts, genes=genes,
                   batches=np.array([p[0] for p in pairs], dtype="U"),
                   perturbations=np.array([p[1] for p in pairs], dtype="U"), fingerprint=stamp)
        write_json(out / "COMPLETE.json", {"fingerprint": stamp, "sha256": file_sha256(artifact), "plan": plan,
            "n_cells": len(obs), "zero_library_cells_skipped": skipped_zero,
            "common_genes": genes.tolist(), "missing_reference_genes": [g for g in reference["genes"] if g not in gene_lookup],
            "source_integrity": "local file SHA256 recorded; official byte identity not asserted" if local_h5ad else
                                "pinned GCS generation and validated range responses; whole-object CRC32C not recomputed"})
    return artifact


def prepare_student_c(reference_dir, aggregate, output, *, training_source="challenge-only", min_cells=10,
                      min_controls=30, control="non-targeting"):
    """New independent data preparation; never modify previous student data."""
    if training_source not in {"challenge-only", "combined"}:
        raise ValueError("training_source must be challenge-only or combined")
    if min(min_cells, min_controls) < 1:
        raise ValueError("Cell thresholds must be positive")
    original = load_prepared(reference_dir)
    if original["audit"].get("target_sum") != 10000:
        raise ValueError("VCC comparison requires reference target_sum=10000")
    aggregate, output = Path(aggregate), Path(output)
    complete = json.loads((aggregate.parent / "COMPLETE.json").read_text())
    if complete["sha256"] != file_sha256(aggregate) or complete["plan"]["reference_fingerprint"] != original["audit"]["fingerprint"]:
        raise ValueError("Challenge aggregate/reference integrity mismatch")
    if (output / "dataset.npz").exists():
        raise FileExistsError("Use a fresh student-C prepared directory")
    with np.load(aggregate, allow_pickle=False) as f:
        genes, sums, counts, batches, perts = [f[k] for k in ("genes", "sums", "counts", "batches", "perturbations")]
    if len(set(genes)) != len(genes) or not set(genes).issubset(set(original["genes"])):
        raise ValueError("Challenge genes do not align with reference")
    ref_cols = np.array([list(original["genes"]).index(g) for g in genes])
    controls = {batch: (sums[i] / counts[i], int(counts[i])) for i, (batch, pert) in enumerate(zip(batches, perts))
                if pert == control and counts[i] >= min_controls}
    rows, baseline, delta, skipped = [], [], [], []
    for i, (batch, pert) in enumerate(zip(batches, perts)):
        if pert == control:
            continue
        if counts[i] < min_cells or batch not in controls:
            skipped.append({"batch": str(batch), "perturbation": str(pert), "n_cells": int(counts[i])})
            continue
        base, n_controls = controls[batch]
        baseline.append(base)
        delta.append(sums[i] / counts[i] - base)
        rows.append({"row_id": json.dumps(["vcc2025_train", str(batch), str(pert)], separators=(",", ":")),
                     "dataset": "vcc2025_train", "context": "H1_hESC", "batch": str(batch),
                     "perturbation": str(pert), "split": "train", "n_cells": int(counts[i]), "n_controls": n_controls})
    if not rows:
        raise ValueError("No valid challenge training groups")
    vocab = set(r["perturbation"] for r in rows)
    if training_source == "combined":
        vocab.update(original["meta"].loc[original["meta"].split == "train", "perturbation"])
    old = original["meta"]
    include = old.perturbation.isin(vocab) & ((old.split != "train") if training_source == "challenge-only" else True)
    ref_rows = np.flatnonzero(include.to_numpy())
    if "H1_hESC" in set(old.context):
        raise ValueError("Reference contexts already contain H1_hESC; review overlap explicitly")
    baseline = np.concatenate([np.asarray(baseline, dtype=np.float32), original["baseline"][np.ix_(ref_rows, ref_cols)]])
    delta = np.concatenate([np.asarray(delta, dtype=np.float32), original["delta"][np.ix_(ref_rows, ref_cols)]])
    meta = pd.concat([pd.DataFrame(rows), old.iloc[ref_rows]], ignore_index=True)
    vocab = sorted(vocab)
    indices = np.array([vocab.index(p) for p in meta.perturbation], dtype=np.int64)
    if set(meta.split) != {"train", "val", "test"}:
        raise ValueError("No common perturbations in a held-out split; cannot compare these experiments")
    save_prepared(output, baseline, delta, indices, genes, vocab, meta,
                  {"synthetic": False, "normalization": original["audit"]["normalization"], "target_sum": 10000,
                   "feature_selection": "intersection with the frozen reference gene panel; no new variance selection",
                   "challenge_training": {"training_source": training_source, "aggregate_sha256": file_sha256(aggregate),
                       "source": complete["plan"]["source"], "n_challenge_groups": len(rows), "min_cells": min_cells,
                       "min_controls": min_controls, "skipped_groups": skipped},
                   "reference": {"data_dir": str(Path(reference_dir).resolve()), "fingerprint": original["audit"]["fingerprint"],
                       "gene_indices": ref_cols.tolist(), "included_row_ids": old.iloc[ref_rows].row_id.tolist(),
                       "original_val_rows": len(original["splits"]["val"]),
                       "common_val_rows": int((meta.split == "val").sum()), "original_n_genes": len(original["genes"])},
                   "benchmark_note": "Exploratory training-data-source comparison, not an isolated distillation ablation. "
                       "All scores use the same common genes/rows. Platforms and measured library panels differ. Test remains sealed."})
    print(f"STUDENT C DATA READY: train={sum(meta.split=='train')} val={sum(meta.split=='val')} "
          f"genes={len(genes)} training_source={training_source}", flush=True)
    return output
