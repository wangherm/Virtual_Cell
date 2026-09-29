"""Compare supervised student C with frozen A/B results on identical rows/genes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_prepared
from .selection import mse, row_weights
from .train import simple_baselines
from .utils import file_sha256, write_json


def validation_prediction(folder, data, arm):
    folder = Path(folder)
    manifest = json.loads((folder / "run_manifest.json").read_text())
    complete = json.loads((folder / "COMPLETE.json").read_text())
    if manifest["data_fingerprint"] != data["audit"]["fingerprint"] or complete["run_fingerprint"] != manifest["run_fingerprint"]:
        raise ValueError("Student run/data identity mismatch")
    val = data["splits"]["val"]
    if complete["test_evaluated"] or manifest["val_rows"] != val.tolist():
        raise ValueError("Expected a completed full-validation run with test sealed")
    path = folder / arm / "validation_predictions.npz"
    with np.load(path, allow_pickle=False) as f:
        if str(f["data_fingerprint"].item()) != data["audit"]["fingerprint"] or not np.array_equal(f["genes"], data["genes"]):
            raise ValueError("Prediction gene/data identity mismatch")
        if not np.array_equal(f["row_ids"], data["meta"].iloc[val].row_id.to_numpy(dtype="U")):
            raise ValueError("Prediction row order mismatch")
        pred = f["delta"].astype(np.float32)
    if pred.shape != data["delta"][val].shape or not np.isfinite(pred).all():
        raise ValueError("Invalid validation predictions")
    report = json.loads((folder / arm / "validation.json").read_text())
    actual = mse(pred, data["delta"][val], row_weights(data["meta"].iloc[val]))
    if not np.isclose(actual, report["mse_delta"], rtol=1e-6, atol=1e-9):
        raise ValueError("Validation predictions do not reproduce recorded MSE")
    return pred, {"prediction_sha256": file_sha256(path), "run_fingerprint": manifest["run_fingerprint"],
                  "recorded_full_val_mse": actual, "best_epoch": report["best_epoch"]}


def compare_students(reference_dir, parent_run, c_data_dir, c_run, output):
    reference, challenge = load_prepared(reference_dir), load_prepared(c_data_dir)
    if challenge["audit"]["reference"]["fingerprint"] != reference["audit"]["fingerprint"]:
        raise ValueError("C was prepared against another reference experiment")
    parent = Path(parent_run)
    completed = json.loads((parent / "COMPLETE.json").read_text())
    if completed.get("test_evaluated"):
        raise ValueError("Expected parent comparison with test sealed")
    val, cval = reference["splits"]["val"], challenge["splits"]["val"]
    row_map = {r: i for i, r in enumerate(reference["meta"].iloc[val].row_id)}
    try:
        rows = np.array([row_map[r] for r in challenge["meta"].iloc[cval].row_id])
        columns = np.array([list(reference["genes"]).index(g) for g in challenge["genes"]])
    except (KeyError, ValueError) as exc:
        raise ValueError("C validation is not a row/gene subset of A/B validation") from exc
    truth = reference["delta"][np.ix_(val[rows], columns)]
    if not np.array_equal(truth, challenge["delta"][cval]) or not np.array_equal(
            reference["baseline"][np.ix_(val[rows], columns)], challenge["baseline"][cval]):
        raise ValueError("Validation labels or control expression changed between experiments")
    meta = challenge["meta"].iloc[cval]
    predictions, identities = {}, {}
    for name in ("student_a", "student_b"):
        checkpoint = parent / name / "all/best.pt"
        candidate = next(x for x in completed["candidates"] if x["model"] == name)
        if file_sha256(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint changed since the 421 selection")
        full, identities[name] = validation_prediction(parent / name, reference, "all")
        predictions[name] = full[np.ix_(rows, columns)]
    predictions["student_c"], identities["student_c"] = validation_prediction(c_run, challenge, "supervised")
    predictions["reference_mean_transfer"] = simple_baselines(reference)["mean_transfer"][np.ix_(val[rows], columns)]
    predictions["student_c_mean_transfer"] = simple_baselines(challenge)["mean_transfer"][cval]
    predictions["no_change"] = np.zeros_like(truth)
    records = [{"model": name, "split": "val_common", "mse_delta": mse(pred, truth, row_weights(meta)),
                "n_rows": len(cval), "n_genes": len(columns), "n_perturbations": int(meta.perturbation.nunique())}
               for name, pred in predictions.items()]
    frame = pd.DataFrame(records).sort_values("mse_delta")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "comparison_common.csv", index=False)
    subgroup = []
    for pert in sorted(meta.perturbation.unique()):
        ix = np.flatnonzero((meta.perturbation == pert).to_numpy())
        for name, prediction in predictions.items():
            subgroup.append({"model": name, "perturbation": pert, "n_rows": len(ix),
                             "mse_delta": mse(prediction[ix], truth[ix], row_weights(meta.iloc[ix]))})
    pd.DataFrame(subgroup).to_csv(out / "by_perturbation.csv", index=False)
    write_json(out / "comparison_audit.json", {"reference_fingerprint": reference["audit"]["fingerprint"],
               "student_c_fingerprint": challenge["audit"]["fingerprint"], "identities": identities,
               "common_row_ids": meta.row_id.tolist(), "common_genes": challenge["genes"].tolist(),
               "original_val_rows": len(val), "original_genes": len(reference["genes"]),
               "retained_val_rows": len(cval), "retained_genes": len(columns), "test_evaluated": False,
               "interpretation": "Training-data-source comparison, not a controlled estimate of distillation benefit. "
                                 "All metrics recomputed on exactly the same rows, genes, labels and row weights. "
                                 "A/B retain their original checkpoints selected on the full reference validation set. "
                                 "C uses the common validation subset for early stopping. No new overall winner is promoted.",
               "training_source": challenge["audit"]["challenge_training"]["training_source"]})
    print(frame.to_string(index=False), flush=True)
    print(f"THREE STUDENT COMPARISON COMPLETE: {out / 'comparison_common.csv'}", flush=True)
    return out
