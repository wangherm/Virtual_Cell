"""Validated external teacher caches; no silent fallbacks to fake predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .utils import write_json, file_sha256


REQUIRED_PROVENANCE = {"teacher_name", "model_revision", "training_contexts", "excluded_contexts",
                       "source", "license", "audit_notes", "declared_no_holdout_perturbations"}


def check_provenance(provenance, data, *, allow_unverified=False):
    missing = REQUIRED_PROVENANCE - set(provenance)
    if missing:
        raise ValueError(f"Missing teacher provenance: {sorted(missing)}")
    if allow_unverified and provenance.get("benchmark_status") == "exploratory_pretraining_overlap_unverified":
        if provenance["declared_no_holdout_perturbations"] is not False or not provenance["audit_notes"]:
            raise ValueError("Exploratory provenance must explicitly report unverified exclusion")
        return
    held = set(data["meta"].loc[data["meta"].split != "train", "context"])
    if set(provenance["training_contexts"]) & held:
        raise ValueError("Teacher reports training on held-out context perturbations.")
    if not held.issubset(set(provenance["excluded_contexts"])):
        raise ValueError("Every held-out context must be explicitly excluded in teacher provenance.")
    if provenance["declared_no_holdout_perturbations"] is not True:
        raise ValueError("Teacher training provenance not verified; refusing strict benchmark use.")
    # Declaration is auditable metadata, not automated proof of pretraining contents.


def load_cache(path, data, *, allow_unverified=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        required = {"delta", "row_ids", "genes", "data_fingerprint"}
        if not required.issubset(f.files):
            raise ValueError(f"Teacher cache needs {required}")
        if str(f["data_fingerprint"].item()) != data["audit"]["fingerprint"]:
            raise ValueError("Teacher cache belongs to another data preparation/split.")
        if not np.array_equal(f["genes"], data["genes"]):
            raise ValueError("Teacher cache gene order mismatch.")
        if not np.array_equal(f["row_ids"], data["meta"].row_id.to_numpy(dtype="U")):
            raise ValueError("Teacher cache row order mismatch.")
        pred = f["delta"].astype(np.float32)
    if pred.shape != data["delta"].shape or not np.isfinite(pred).all():
        raise ValueError("Invalid teacher prediction shape or values.")
    provenance = json.loads(path.with_suffix(".json").read_text())
    check_provenance(provenance, data, allow_unverified=allow_unverified)
    return pred, provenance


def import_cache(matrix_path, row_ids_path, genes_path, provenance_path, data, output):
    """Import mean DELTA predictions already aligned to this package's expression scale."""
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    pred = np.load(matrix_path, allow_pickle=False)
    rows = pd.read_csv(row_ids_path, dtype=str)["row_id"].to_numpy()
    genes = pd.read_csv(genes_path, dtype=str)["gene"].to_numpy()
    if len(set(rows)) != len(rows) or len(set(genes)) != len(genes):
        raise ValueError("Duplicate row IDs or genes in imported teacher output.")
    if pred.shape != (len(rows), len(genes)):
        raise ValueError("Prediction matrix shape differs from supplied IDs.")
    rmap, gmap = {x: i for i, x in enumerate(rows)}, {x: i for i, x in enumerate(genes)}
    if set(data["meta"].row_id) - set(rmap) or set(data["genes"]) - set(gmap):
        raise ValueError("Teacher output must cover every requested row and gene; no filling missing predictions.")
    pred = pred[np.ix_([rmap[x] for x in data["meta"].row_id], [gmap[x] for x in data["genes"]])]
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite teacher predictions")
    provenance = json.loads(Path(provenance_path).read_text())
    check_provenance(provenance, data)
    provenance["import_matrix_sha256"] = file_sha256(matrix_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, delta=pred.astype(np.float32), genes=data["genes"],
                        row_ids=data["meta"].row_id.to_numpy(dtype="U"),
                        data_fingerprint=data["audit"]["fingerprint"])
    write_json(output.with_suffix(".json"), provenance)
    load_cache(output, data)


def export_queries(data, output):
    """Only controls/target IDs, never true perturbed expression."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "teacher_queries.npz", baseline=data["baseline"],
                        pert_idx=data["pert_idx"], genes=data["genes"],
                        perturbations=data["perturbations"],
                        row_ids=data["meta"].row_id.to_numpy(dtype="U"))
    data["meta"][["row_id", "dataset", "context", "batch", "perturbation", "split"]].to_csv(output / "row_ids.csv", index=False)
    pd.DataFrame({"gene": data["genes"]}).to_csv(output / "genes.csv", index=False)
