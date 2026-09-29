"""Train-only cross fitting, reliability estimation and low-rank Ridge baselines."""
import gc
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import svd, solve

from .selection import row_weights, mse, fit_mix, combine
from .teacher_pipeline import ConditionalHead, write_cache
from .utils import seed_all, file_sha256, write_json, atomic_torch_save


def context_folds(data):
    train = data["splits"]["train"]
    contexts = data["meta"].iloc[train].context.to_numpy()
    if len(set(contexts)) < 2:
        raise ValueError("Reliability needs at least two training contexts")
    return [(train[contexts != c], train[contexts == c]) for c in sorted(set(contexts))]


def load_features(path, data):
    path = Path(path)
    audit = json.loads(path.with_suffix(".json").read_text())
    if audit.get("feature_file_sha256") != file_sha256(path):
        raise ValueError("Frozen teacher feature checksum mismatch")
    if audit.get("artifact_kind") != "frozen_control_features_NOT_predictions":
        raise ValueError("Expected frozen pretrained features, not response predictions")
    with np.load(path) as f:
        if str(f["data_fingerprint"].item()) != data["audit"]["fingerprint"] or not np.array_equal(
                f["row_ids"], data["meta"].row_id.to_numpy(dtype="U")):
            raise ValueError("Teacher feature/data alignment mismatch")
        x = f["features"].astype(np.float32)
    if x.ndim != 2 or len(x) != len(data["meta"]) or not np.isfinite(x).all():
        raise ValueError("Invalid frozen features")
    return x, audit


def fit_head(data, features, fit, query, *, epochs=20, seed=17, device="cpu"):
    """Fixed epochs. No validation/test labels influence fitting or stopping."""
    seed_all(seed)
    mean, std = features[fit].mean(0), np.maximum(features[fit].std(0), .01)
    scale = max(float(np.sqrt(np.mean(data["delta"][fit] ** 2))), .05)
    x = torch.tensor((features - mean) / std, device=device)
    p = torch.tensor(data["pert_idx"], device=device)
    y = torch.tensor(data["delta"][fit] / scale, device=device)
    w = torch.tensor(row_weights(data["meta"].iloc[fit]) * len(fit), dtype=torch.float32, device=device)
    model = ConditionalHead(features.shape[1], len(data["perturbations"]), len(data["genes"]), 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(fit))
        for start in range(0, len(order), 64):
            selected = order[start:start+64]
            ix = fit[selected]
            prediction = model(x[ix], p[ix])
            loss = ((prediction - y[selected]).square().mean(1) * w[selected]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite cross-fitted teacher loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = np.concatenate([model(x[ix], p[ix]).cpu().numpy() for ix in
                                     (query[j:j+64] for j in range(0, len(query), 64))]) * scale
    state = {"weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "mean": torch.tensor(mean), "std": torch.tensor(std), "scale": scale,
             "fit_rows": fit.tolist(), "epochs": epochs}
    del model, optimizer, x, p, y
    gc.collect()
    return prediction, state


def crossfit_teacher(data, feature_path, family, output, *, epochs=20, device="cpu"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features, feature_audit = load_features(feature_path, data)
    prediction = np.zeros_like(data["delta"])
    eligible = np.zeros(len(prediction), dtype=bool)
    records = []
    for fold, (fit, held) in enumerate(context_folds(data)):
        values, state = fit_head(data, features, fit, held, epochs=epochs, device=device)
        valid = np.isin(data["pert_idx"][held], data["pert_idx"][fit])
        prediction[held] = values
        eligible[held] = valid
        records.append({"fit_rows": data["meta"].iloc[fit].row_id.tolist(),
                        "held_rows": data["meta"].iloc[held].row_id.tolist(),
                        "held_contexts": sorted(data["meta"].iloc[held].context.unique().tolist()),
                        "eligible_rows": int(valid.sum())})
        atomic_torch_save(state, output / f"fold_{fold}.pt")
        print(f"CROSSFIT {family} fold={fold+1} held={len(held)} eligible={valid.sum()}", flush=True)
    query = np.flatnonzero(data["meta"].split.to_numpy() != "train")
    values, final = fit_head(data, features, data["splits"]["train"], query, epochs=epochs, device=device)
    prediction[query] = values
    atomic_torch_save(final, output / "head.pt")
    np.savez_compressed(output / "oof_eligible.npz", eligible=eligible,
                        data_fingerprint=data["audit"]["fingerprint"])
    provenance = {"teacher_name": family + " frozen encoder + cross-fitted VCell head", "teacher_family": family,
        "model_revision": feature_audit.get("model_revision", feature_audit.get("checkpoint_sha256", "see frozen feature provenance")),
        "source": feature_audit.get("source", "see frozen feature provenance"), "license": "See THIRD_PARTY_NOTICES.md",
        "training_contexts": sorted(data["meta"].iloc[data["splits"]["train"]].context.unique().tolist()),
        "excluded_contexts": [], "declared_no_holdout_perturbations": False,
        "benchmark_status": "exploratory_pretraining_overlap_unverified", "prediction_space": "prepared_log1p_delta",
        "audit_notes": "Pretraining overlap unverified. Task heads fit training labels only, with fixed epochs. "
                       "Training KD uses leave-one-training-context-out predictions; unsupported fold targets are masked.",
        "feature_sha256": file_sha256(feature_path), "feature_provenance": feature_audit,
        "crossfit": records, "epochs": epochs, "test_evaluated": False,
        "eligible_sha256": file_sha256(output / "oof_eligible.npz")}
    cache = output / "predictions.npz"
    write_cache(cache, prediction, data, provenance, allow_unverified=True)
    return cache


def reliability_report(data, caches, output):
    from .teachers import load_cache
    names = list(caches)
    predictions, hashes = [], {}
    eligible = np.zeros(len(data["meta"]), dtype=bool)
    eligible[data["splits"]["train"]] = True
    for name, path in caches.items():
        values, provenance = load_cache(path, data, allow_unverified=True)
        mask_path = Path(path).parent / "oof_eligible.npz"
        if not provenance.get("crossfit") or file_sha256(mask_path) != provenance["eligible_sha256"]:
            raise ValueError("Reliability requires audited out-of-context predictions")
        with np.load(mask_path) as f:
            if str(f["data_fingerprint"].item()) != data["audit"]["fingerprint"]:
                raise ValueError("OOF mask belongs to another dataset")
            eligible &= f["eligible"]
        predictions.append(values)
        hashes[name] = [file_sha256(path), file_sha256(Path(path).with_suffix(".json"))]
    ix = np.flatnonzero(eligible)
    if len(ix) < 2:
        raise ValueError("Too few out-of-context rows with supported perturbations")
    weights = row_weights(data["meta"].iloc[ix])
    truth = data["delta"][ix]
    mixture = fit_mix([a[ix] for a in predictions], truth, weights)
    no_change = mse(np.zeros_like(truth), truth, weights)
    error = mse(combine([a[ix] for a in predictions], mixture), truth, weights)
    strength = float(np.clip(1 - error / max(no_change, 1e-12), 0, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_path = output.with_suffix(".npz")
    np.savez_compressed(mask_path, eligible=eligible, row_ids=data["meta"].row_id.to_numpy(dtype="U"))
    report = {"data_fingerprint": data["audit"]["fingerprint"], "teacher_names": names, "weights": mixture.tolist(),
              "teacher_cache_hashes": hashes, "eligible_rows": ix.tolist(), "mask_path": str(mask_path),
              "mask_sha256": file_sha256(mask_path), "kd_strength": strength, "oof_mixture_mse": error,
              "oof_zero_mse": no_change, "individual_oof_mse": {n: mse(a[ix], truth, weights) for n, a in zip(names, predictions)},
              "selected_using": "training-context cross-fit predictions only", "test_evaluated": False,
              "note": "Weights optimize train-only cross-fit errors; these errors are tuning diagnostics, not independent evaluation."}
    write_json(output, report)
    print(f"RELIABILITY: weights={dict(zip(names, mixture.round(4)))} kd_strength={strength:.4f} eligible={len(ix)}", flush=True)
    return report


def fit_ridge(data, fit, query, alpha, rank=32):
    """Weighted low-rank output regression; all transforms fitted within each fold."""
    weights = row_weights(data["meta"].iloc[fit])
    root = np.sqrt(weights * len(fit))[:, None]
    controls = data["baseline"].astype(np.float64)
    mu = np.average(controls[fit], axis=0, weights=weights)
    _, _, basis = svd((controls[fit] - mu) * root, full_matrices=False, check_finite=False)
    basis = basis[:min(rank, len(fit)-1)]
    context = (controls - mu) @ basis.T
    sd = np.maximum(context[fit].std(0), .1)
    context /= sd
    onehot = np.eye(len(data["perturbations"]))[data["pert_idx"]]
    rng = np.random.default_rng(17)
    target_code = onehot @ rng.normal(size=(onehot.shape[1], context.shape[1])) / np.sqrt(context.shape[1])
    design = np.column_stack([context, onehot, context * target_code])
    center = np.average(design[fit], axis=0, weights=weights)
    design -= center
    y = data["delta"][fit].astype(np.float64)
    ym = np.average(y, axis=0, weights=weights)
    _, _, outputs = svd((y - ym) * root, full_matrices=False, check_finite=False)
    outputs = outputs[:min(rank, len(fit)-1)]
    x = design[fit] * root
    target = ((y - ym) @ outputs.T) * root
    coef = solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ target, assume_a="pos", check_finite=False)
    prediction = (design[query] @ coef @ outputs + ym).astype(np.float32)
    return prediction, dict(control_mean=mu, control_basis=basis, control_std=sd, design_center=center,
                            output_basis=outputs, output_mean=ym, coef=coef, alpha=alpha, rank=rank)


def ridge_baseline(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scores = {}
    for alpha in (1., 10., 100.):
        errors = []
        for fit, held in context_folds(data):
            pred, _ = fit_ridge(data, fit, held, alpha)
            errors.append(mse(pred, data["delta"][held], row_weights(data["meta"].iloc[held])))
        scores[alpha] = float(np.mean(errors))
    alpha = min(scores, key=scores.get)
    val = data["splits"]["val"]
    pred, state = fit_ridge(data, data["splits"]["train"], val, alpha)
    np.savez_compressed(output / "ridge.npz", **state, genes=data["genes"], perturbations=data["perturbations"])
    np.savez_compressed(output / "validation_predictions.npz", delta=pred, row_ids=data["meta"].iloc[val].row_id.to_numpy(dtype="U"),
                        genes=data["genes"], data_fingerprint=data["audit"]["fingerprint"])
    write_json(output / "audit.json", {"alpha": alpha, "training_cv_mse": scores,
               "validation_mse": mse(pred, data["delta"][val], row_weights(data["meta"].iloc[val])),
               "test_evaluated": False, "normalization_and_svd": "fit training fold only"})
    return pred
