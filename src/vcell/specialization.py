"""Background audits and nested, training-only teacher specialization experiments.

Validation is for development. Test outcomes are never used here. With only two
training contexts, inner calibration must fall back to grouped batch holdouts;
this limitation is recorded rather than called independent context evidence.
"""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .round5_methods import fit_head, load_features
from .selection import row_weights, mse
from .utils import file_sha256, write_json

BACKGROUND_FIELDS = ("species", "tissue", "cell_type", "cell_line", "donor", "study",
                     "platform", "intervention", "timepoint", "dose", "culture")


def coverage_audit(data, output, backgrounds=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = data["meta"].copy()
    backgrounds = backgrounds or {}
    unknown = set(backgrounds) - set(meta.dataset)
    if unknown:
        raise ValueError(f"Background metadata names unknown datasets: {sorted(unknown)}")
    for field in BACKGROUND_FIELDS:
        meta[field] = [backgrounds.get(d, {}).get(field, "unknown") for d in meta.dataset]
    meta.to_csv(output / "background_rows.csv", index=False)
    keys = ["split", "dataset", "context", "perturbation"]
    coverage = meta.groupby(keys, observed=True).agg(groups=("row_id", "size"),
                    perturbation_cells=("n_cells", "sum")).reset_index()
    coverage.to_csv(output / "coverage.csv", index=False)
    summary = meta.groupby(["split", "dataset", "context"], observed=True).agg(
        groups=("row_id", "size"), targets=("perturbation", "nunique"),
        perturbation_cells=("n_cells", "sum"), batches=("batch", "nunique")).reset_index()
    summary.to_csv(output / "sample_counts.csv", index=False)
    train = meta.iloc[data["splits"]["train"]]
    contexts = sorted(train.context.unique())
    overlap = []
    for a, b in combinations(contexts, 2):
        shared = set(train[train.context == a].perturbation) & set(train[train.context == b].perturbation)
        overlap.append({"context_a": a, "context_b": b, "shared_targets": len(shared),
                        "targets": sorted(shared)})
    controls = meta.drop_duplicates(["dataset", "context", "batch"])
    report = {"data_fingerprint": data["audit"]["fingerprint"], "training_contexts": contexts,
              "context_overlap": overlap, "known_metadata": {
                  f: sorted(set(meta[f]) - {"unknown"}) for f in BACKGROUND_FIELDS},
              "control_counts_note": "Controls are reused across target rows; never sum n_controls across perturbations.",
              "matched_control_groups": len(controls), "test_evaluated": False,
              "warning": "Cell type, cell line and study effects are not identifiable without crossed coverage. "
                         "Unknown metadata are not inferred. No post-treatment covariate enters the model."}
    write_json(output / "coverage_audit.json", report)
    return report


def make_modules(data, count=8, definition=None):
    """Return M x G memberships with per-gene sum one, including uncovered genes.

    Optional JSON: {module_name: [prepared_gene_id, ...]}. Default clusters only
    training control profiles. These are expression modules, not named pathways.
    """
    genes = list(data["genes"])
    if definition is not None:
        if not isinstance(definition, dict) or not definition:
            raise ValueError("Module definition must be a nonempty gene-list mapping")
        names, rows = [], []
        for name, members in definition.items():
            if not isinstance(members, list) or not members:
                raise ValueError("Each module must contain a gene-ID list")
            row = np.isin(genes, members).astype(float)
            if row.any():
                names.append(name)
                rows.append(row)
        if not rows:
            raise ValueError("No module genes match the prepared gene IDs")
        membership = np.stack(rows)
        missing = membership.sum(0) == 0
        if missing.any():
            names.append("__unannotated__")
            membership = np.vstack([membership, missing])
        kind = "fixed external gene sets; overlap normalized per gene"
    else:
        if count < 1 or count > len(genes):
            raise ValueError("Invalid module count")
        train = data["splits"]["train"]
        unique = ~data["meta"].iloc[train].duplicated(["dataset", "context", "batch"])
        x = data["baseline"][train[unique]].astype(np.float64).T
        x -= x.mean(1, keepdims=True)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        # Deterministic farthest-point seeds; no perturbation outcomes or held controls.
        centers = [x[0].copy()]
        distance = np.sum((x - centers[0]) ** 2, axis=1)
        for _ in range(1, count):
            centers.append(x[int(np.argmax(distance))].copy())
            distance = np.minimum(distance, np.sum((x - centers[-1]) ** 2, axis=1))
        centers = np.stack(centers)
        labels = np.zeros(len(x), dtype=int)
        for _ in range(30):
            new = np.argmin(((x[:, None] - centers) ** 2).sum(2), axis=1)
            if np.array_equal(labels, new) and _:
                break
            labels = new
            for k in range(count):
                if (labels == k).any():
                    centers[k] = x[labels == k].mean(0)
        ids = sorted(set(labels))
        names = [f"control_expression_{k+1}" for k in ids]
        membership = np.stack([labels == k for k in ids]).astype(float)
        kind = "training-control profile clusters; not causal or curated pathways"
    membership /= membership.sum(0, keepdims=True)
    return membership.astype(np.float32), {"kind": kind, "names": names,
        "genes": genes, "membership": membership.tolist(), "test_evaluated": False}


def reference_responses(data, fit):
    """Equal-context mean response; unknown targets use a labelled fallback."""
    fit = np.asarray(fit)
    meta = data["meta"].iloc[fit]
    fallback = np.average(data["delta"][fit], axis=0, weights=row_weights(meta))
    values = np.tile(fallback, (len(data["perturbations"]), 1)).astype(np.float32)
    seen = np.zeros(len(values), dtype=bool)
    for p in np.unique(data["pert_idx"][fit]):
        ix = fit[data["pert_idx"][fit] == p]
        cs = data["meta"].iloc[ix].context.to_numpy()
        values[p] = np.mean([data["delta"][ix[cs == c]].mean(0) for c in sorted(set(cs))], axis=0)
        seen[p] = True
    return values, seen


def calibration_folds(data, fit):
    """Leave context out when possible, otherwise two disjoint batch groups."""
    meta = data["meta"].iloc[fit]
    contexts = meta.context.to_numpy()
    if len(set(contexts)) >= 2:
        return [(fit[contexts != c], fit[contexts == c]) for c in sorted(set(contexts))], "context"
    groups = meta[["dataset", "context", "batch"]].astype(str).agg("|".join, axis=1).to_numpy()
    keys = sorted(set(groups))
    if len(keys) < 2:
        return [], "insufficient_batches"
    assignment = {k: j % 2 for j, k in enumerate(keys)}
    side = np.array([assignment[g] for g in groups])
    return [(fit[side != k], fit[side == k]) for k in (0, 1)], "batch_fallback"


def module_errors(prediction, truth, membership):
    return np.einsum("...rg,mg->...rm", (prediction - truth) ** 2, membership) / membership.sum(1)


def calibrate(predictions, data, rows, membership, reference, min_targets=10, shrinkage=20.):
    """Low-capacity module selector. No context identity lookup or sample oracle."""
    teachers, _, _ = predictions.shape
    modules = len(membership)
    weights = np.full((modules, teachers), 1 / teachers)
    strength = np.zeros(modules)
    if not len(rows):
        return {"weights": weights.tolist(), "strength": strength.tolist(), "targets": 0, "rows": []}
    truth = data["delta"][rows]
    rw = row_weights(data["meta"].iloc[rows])
    errors = np.einsum("trm,r->tm", module_errors(predictions[:, rows], truth, membership), rw)
    zero = rw @ module_errors(np.zeros_like(truth), truth, membership)
    transfer = rw @ module_errors(reference[rows], truth, membership)
    baseline = np.minimum(zero, transfer)
    targets = int(data["meta"].iloc[rows].perturbation.nunique())
    shrink = targets / (targets + shrinkage)
    gains = np.maximum(1 - errors / np.maximum(baseline, 1e-12), 0)
    if targets >= min_targets:
        for m in range(modules):
            if gains[:, m].sum() > 0:
                weights[m] = (1-shrink) / teachers + shrink * gains[:, m] / gains[:, m].sum()
            mix = np.einsum("t,trg->rg", weights[m], predictions[:, rows])
            error = float(rw @ module_errors(mix, truth, membership[m:m+1])[:, 0])
            strength[m] = shrink * np.clip(1 - error / max(baseline[m], 1e-12), 0, 1)
    return {"weights": weights.tolist(), "strength": strength.tolist(), "targets": targets,
            "rows": rows.tolist(), "teacher_mse": errors.tolist(), "baseline_mse": baseline.tolist(),
            "gains": gains.tolist(), "min_targets": min_targets, "shrinkage": shrinkage}


def route(predictions, membership, calibration, equal=False):
    weights = np.asarray(calibration["weights"])
    if equal:
        weights = np.full_like(weights, 1 / predictions.shape[0])
    strength = np.asarray(calibration["strength"])
    per_gene_gate = strength @ membership
    coefficients = np.einsum("mt,m,mg->tg", weights, strength, membership)
    value = np.einsum("tg,trg->rg", coefficients, predictions)
    value /= np.maximum(per_gene_gate, 1e-12)
    return value.astype(np.float32), per_gene_gate.astype(np.float32)


def specialist_loss_weights(calibration, membership, teachers):
    gains = np.asarray(calibration.get("gains", np.zeros((teachers, len(membership)))))
    if calibration["targets"] < calibration.get("min_targets", 10):
        gains = np.zeros_like(gains)
    weights = 1 + 2 * (gains @ membership)
    return (weights / weights.mean(1, keepdims=True)).astype(np.float32)


def cached_head(data, features, fit, query, folder, *, epochs, device, gene_weights=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "prediction.npz"
    spec = {"fit": fit.tolist(), "query": query.tolist(), "epochs": epochs,
            "gene_weights": None if gene_weights is None else gene_weights.tolist()}
    if path.exists() and (folder / "complete.json").exists():
        saved = json.loads((folder / "complete.json").read_text())
        if saved["spec"] != spec or file_sha256(path) != saved["sha256"]:
            raise ValueError("Nested head cache changed; choose a fresh run")
        with np.load(path) as f:
            return f["delta"]
    print(f"ROUND6 HEAD: {folder} fit={len(fit)} query={len(query)} epochs={epochs}", flush=True)
    prediction, _ = fit_head(data, features, fit, query, epochs=epochs, device=device, gene_weights=gene_weights)
    np.savez_compressed(path, delta=prediction)
    write_json(folder / "complete.json", {"spec": spec, "sha256": file_sha256(path)})
    return prediction


def inner_predictions(data, features, fit, folder, epochs, device, gene_weights=None):
    folds, kind = calibration_folds(data, fit)
    predictions = np.zeros((len(features), *data["delta"].shape), dtype=np.float32)
    reference = np.zeros_like(data["delta"])
    eligible = np.zeros(len(reference), dtype=bool)
    for number, (source, held) in enumerate(folds):
        ref, seen = reference_responses(data, source)
        reference[held] = ref[data["pert_idx"][held]]
        eligible[held] = seen[data["pert_idx"][held]]
        for teacher, (name, x) in enumerate(features.items()):
            predictions[teacher, held] = cached_head(data, x, source, held, Path(folder) / f"fold{number}" / name,
                epochs=epochs, device=device, gene_weights=None if gene_weights is None else gene_weights[teacher])
    return predictions, reference, np.flatnonzero(eligible), kind


def save_distillation(path, data, targets, gate, audit):
    # Only training rows carry KD targets. Validation/test are neither routed nor scored here.
    np.savez_compressed(path, delta=targets, gene_weights=gate, genes=data["genes"],
        row_ids=data["meta"].row_id.to_numpy(dtype="U"), data_fingerprint=data["audit"]["fingerprint"])
    write_json(Path(path).with_suffix(".json"), {**audit, "sha256": file_sha256(path),
        "test_evaluated": False, "kind": "nested_training_only_distillation"})
    return {"path": str(path), "sha256": file_sha256(path),
            "audit_sha256": file_sha256(Path(path).with_suffix(".json"))}


def load_distillation(spec, data):
    path = Path(spec["path"])
    if file_sha256(path) != spec["sha256"] or file_sha256(path.with_suffix(".json")) != spec["audit_sha256"]:
        raise ValueError("Module distillation checksum changed")
    audit = json.loads(path.with_suffix(".json").read_text())
    if audit.get("kind") != "nested_training_only_distillation" or audit.get("test_evaluated") is not False:
        raise ValueError("Invalid module distillation provenance")
    with np.load(path, allow_pickle=False) as f:
        if str(f["data_fingerprint"].item()) != data["audit"]["fingerprint"] or not np.array_equal(f["genes"], data["genes"]) or not np.array_equal(f["row_ids"], data["meta"].row_id.to_numpy(dtype="U")):
            raise ValueError("Module distillation data/row/gene mismatch")
        targets, gate = f["delta"].astype(np.float32), f["gene_weights"].astype(np.float32)
    if targets.shape != data["delta"].shape or gate.shape != targets.shape or not np.isfinite(targets).all() or not np.isfinite(gate).all() or (gate < 0).any() or (gate > 1.00001).any():
        raise ValueError("Invalid module targets/weights")
    held = np.setdiff1d(np.arange(len(targets)), data["splits"]["train"])
    if gate[held].any() or targets[held].any():
        raise ValueError("Held-out rows must not contain distillation targets")
    return targets, gate, audit


def nested_teachers(data, feature_paths, membership, output, *, epochs=20, device="cpu", min_targets=10):
    """Outer context predictions and routes never fit the outer context labels.

    Base inner OOF determines specialty loss weights; specialists are refitted
    with those weights. Inner specialist calibration is tuning, not independent
    evidence. Only outer predictions are used for the transfer diagnostic.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features = {name: load_features(path, data)[0] for name, path in feature_paths.items()}
    train = data["splits"]["train"]
    folds, kind = calibration_folds(data, train)
    if kind != "context":
        raise ValueError("At least two training backgrounds required for outer context holdouts")
    shape = data["delta"].shape
    base, specialist = (np.zeros((len(features), *shape), dtype=np.float32) for _ in range(2))
    targets = {k: np.zeros(shape, dtype=np.float32) for k in ("equal", "module", "specialist", "gated_equal")}
    gates = {k: np.zeros(shape, dtype=np.float32) for k in targets}
    reference = np.zeros(shape, dtype=np.float32)
    eligible = np.zeros(shape[0], dtype=bool)
    records = []
    for number, (fit, held) in enumerate(folds):
        folder = output / f"outer{number}"
        print(f"ROUND6 TEACHERS outer={number+1}/{len(folds)} held_context={data['meta'].iloc[held].context.unique().tolist()}", flush=True)
        inner, ref, valid, inner_kind = inner_predictions(data, features, fit, folder / "inner_base", epochs, device)
        cal = calibrate(inner, data, valid, membership, ref, min_targets)
        gene_weights = specialist_loss_weights(cal, membership, len(features))
        inner_s, ref_s, valid_s, _ = inner_predictions(data, features, fit, folder / "inner_specialist", epochs, device, gene_weights)
        cal_s = calibrate(inner_s, data, valid_s, membership, ref_s, min_targets)
        means, seen = reference_responses(data, fit)
        reference[held] = means[data["pert_idx"][held]]
        supported = seen[data["pert_idx"][held]]
        eligible[held] = supported
        for j, (name, x) in enumerate(features.items()):
            base[j, held] = cached_head(data, x, fit, held, folder / "base" / name, epochs=epochs, device=device)
            specialist[j, held] = cached_head(data, x, fit, held, folder / "specialist" / name,
                                           epochs=epochs, device=device, gene_weights=gene_weights[j])
        targets["equal"][held] = base[:, held].mean(0)
        gates["equal"][held] = supported[:, None]
        for key, values, calibration in (("module", base, cal), ("gated_equal", base, cal), ("specialist", specialist, cal_s)):
            pred, gate = route(values[:, held], membership, calibration, equal=key == "gated_equal")
            targets[key][held] = pred
            gates[key][held] = gate * supported[:, None]
        record = {"fit_rows": fit.tolist(), "held_rows": held.tolist(), "inner_kind": inner_kind,
                  "held_contexts": data["meta"].iloc[held].context.unique().tolist(),
                  "base_calibration": cal, "specialist_calibration": cal_s,
                  "specialist_gene_weight_range": [float(gene_weights.min()), float(gene_weights.max())]}
        records.append(record)
        write_json(folder / "audit.json", record)
    audit = {"teacher_names": list(features), "folds": records, "eligible_rows": np.flatnonzero(eligible).tolist(),
             "feature_hashes": {n: file_sha256(p) for n, p in feature_paths.items()},
             "pretraining_overlap": "unverified", "test_evaluated": False,
             "specialization": "same head architecture/budget; train-only specialty-weighted output loss (1 to 3 before mean normalization)",
             "module_definition": "fixed using training controls or externally supplied annotations",
             "limitation": "Two training contexts imply batch-fallback inner calibration. Outer context transfer is the diagnostic."}
    packages = {key: save_distillation(output / f"{key}.npz", data, targets[key], gates[key], audit) for key in targets}
    expert_diagnostics(data, base, specialist, reference, eligible, membership, list(features), targets, gates, output)
    write_json(output / "audit.json", audit)
    return packages


def expert_diagnostics(data, base, specialist, reference, eligible, membership, names, targets, gates, output):
    rows = np.flatnonzero(eligible)
    records = []
    if not len(rows):
        write_json(Path(output) / "diagnostics.json", {"eligible_rows": 0, "test_evaluated": False})
        return
    truth = data["delta"][rows]
    arrays = {**{n: a[rows] for n, a in zip(names, base)},
              **{n+"/specialist": a[rows] for n, a in zip(names, specialist)},
              "no_change": np.zeros_like(truth), "mean_transfer": reference[rows]}
    # A closed gate means use the training-only reference for this diagnostic;
    # students instead retain their genuine supervised loss on every gene.
    for key in targets:
        arrays[key+"/reference_fallback"] = gates[key][rows] * targets[key][rows] + (1-gates[key][rows]) * reference[rows]
    for context in sorted(data["meta"].iloc[rows].context.unique()):
        ix = np.flatnonzero(data["meta"].iloc[rows].context.to_numpy() == context)
        rw = row_weights(data["meta"].iloc[rows[ix]])
        for name, values in arrays.items():
            errors = rw @ module_errors(values[ix], truth[ix], membership)
            for m, error in enumerate(errors):
                records.append({"context": context, "model": name, "module": m,
                    "mse": float(error), "targets": int(data["meta"].iloc[rows[ix]].perturbation.nunique())})
    pd.DataFrame(records).to_csv(Path(output) / "teacher_specialties.csv", index=False)
    # Label-aware best single teacher per row/module; not a bound on mixtures.
    module_loss = module_errors(base[:, rows], truth, membership)
    rw = row_weights(data["meta"].iloc[rows])
    oracle = float(rw @ (module_loss.min(0) @ (membership.sum(1) / membership.sum())))
    pairs = []
    for a, b in combinations(range(len(names)), 2):
        ea, eb = (base[a, rows]-truth).ravel(), (base[b, rows]-truth).ravel()
        corr = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else None
        pairs.append({"a": names[a], "b": names[b], "error_correlation": corr})
    write_json(Path(output) / "diagnostics.json", {"eligible_rows": len(rows),
        "eligible_targets": int(data["meta"].iloc[rows].perturbation.nunique()),
        "oof_mse": {n: mse(p, truth, rw) for n, p in arrays.items()},
        "oracle_mse_DIAGNOSTIC_NOT_A_MODEL": oracle, "oracle_uses_true_labels": True,
        "pairs": pairs, "gate_mean": {n: float(g[rows].mean()) for n, g in gates.items()},
        "test_evaluated": False, "note": "Outer context transfer with frozen training-control modules. "
        "Repeated development diagnostics are not final test scores; correlated batches are not independent perturbations."})


def compare_validation(data, predictions, membership, output):
    """One panel, macro metrics and paired target bootstrap (descriptive only)."""
    output = Path(output)
    val = data["splits"]["val"]
    meta, truth = data["meta"].iloc[val], data["delta"][val]
    rows, groups, modules = [], [], []
    for name, prediction in predictions.items():
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError("Comparison panel mismatch")
        rows.append({"model": name, "mse_delta": mse(prediction, truth, row_weights(meta)),
                     "n_rows": len(val), "n_genes": len(data["genes"]), "n_targets": meta.perturbation.nunique()})
        for (context, pert), positions in meta.reset_index(drop=True).groupby(["context", "perturbation"]).indices.items():
            groups.append({"model": name, "context": context, "perturbation": pert,
                           "mse": float(np.mean((prediction[positions]-truth[positions])**2))})
        for m, score in enumerate(row_weights(meta) @ module_errors(prediction, truth, membership)):
            modules.append({"model": name, "module": m, "mse": float(score)})
    scores = pd.DataFrame(rows).sort_values("mse_delta")
    scores.to_csv(output / "comparison.csv", index=False)
    seeded = scores[scores.model.str.contains(r"_seed\d+$")].copy()
    if len(seeded):
        seeded["experiment"] = seeded.model.str.replace(r"_seed\d+$", "", regex=True)
        seeded.groupby("experiment").agg(mean_mse=("mse_delta", "mean"),
            std_across_seeds=("mse_delta", "std"), seeds=("model", "size")).sort_values("mean_mse").to_csv(output / "seed_summary.csv")
    frame = pd.DataFrame(groups)
    frame.to_csv(output / "by_perturbation.csv", index=False)
    pd.DataFrame(modules).to_csv(output / "by_module.csv", index=False)
    # Match the equal-context objective. Resample perturbations as clusters across
    # contexts, preserving within-target dependence rather than resampling rows.
    matrix = frame.pivot(index=["context", "perturbation"], columns="model", values="mse")
    perts = sorted(set(matrix.index.get_level_values("perturbation")))
    rng = np.random.default_rng(17)
    samples = rng.integers(0, len(perts), size=(1000, len(perts)))
    pairs = []
    for a, b in combinations(matrix.columns, 2):
        diff = matrix[a] - matrix[b]
        context_scores = []
        for context in matrix.index.get_level_values("context").unique():
            values = diff.xs(context).reindex(perts).to_numpy()
            sampled = values[samples]
            count = np.isfinite(sampled).sum(1)
            context_scores.append(np.divide(np.nansum(sampled, 1), count,
                                  out=np.full(len(samples), np.nan), where=count > 0))
        boot = np.mean(context_scores, axis=0)
        finite = boot[np.isfinite(boot)]
        lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        pairs.append({"a": a, "b": b, "mse_a_minus_b": float(diff.groupby(level="context").mean().mean()),
                      "ci025": lo, "ci975": hi, "target_context_wins_a": int((diff < 0).sum()),
                      "target_context_pairs": len(diff), "bootstrap_unit": "perturbation cluster",
                      "interpretation": "descriptive; reused validation, no multiple-comparison correction"})
    pd.DataFrame(pairs).to_csv(output / "paired_comparisons.csv", index=False)
