"""Matched-panel, matched-step H1 data-expansion ablation, without downloading data."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_prepared, save_prepared
from .utils import write_json
from .specialization import make_modules, compare_validation, coverage_audit
from .train import simple_baselines
from .qwen import run_qwen
from .challenge_compare import validation_prediction


def prepare_expansion(original, challenge, output):
    """Keep ALL original validation rows; add only audited H1 training rows.

    Project BOTH controls onto the identical common gene panel. No absent gene
    is zero-filled. H1 validation/test rows are never imported or scored.
    """
    if challenge["audit"].get("reference", {}).get("fingerprint") != original["audit"]["fingerprint"]:
        raise ValueError("Challenge data must be prepared against this original reference")
    if challenge["audit"].get("target_sum") != original["audit"].get("target_sum") or challenge["audit"].get("normalization") != original["audit"].get("normalization"):
        raise ValueError("Expansion normalization differs")
    training = challenge["splits"]["train"]
    add = training[challenge["meta"].iloc[training].context.to_numpy() == "H1_hESC"]
    if not len(add) or "H1_hESC" in set(original["meta"].context):
        raise ValueError("Need new H1_hESC training rows")
    genes = [g for g in original["genes"] if g in set(challenge["genes"])]
    if not genes:
        raise ValueError("No common measured genes")
    cols = np.array([list(original["genes"]).index(g) for g in genes])
    ccols = np.array([list(challenge["genes"]).index(g) for g in genes])
    vocab = sorted(set(original["perturbations"]) | set(challenge["meta"].iloc[add].perturbation))
    output = Path(output)
    paths = {}
    for name, append in (("original_only", False), ("plus_h1", True)):
        meta = original["meta"].copy()
        baseline, delta = original["baseline"][:, cols], original["delta"][:, cols]
        if append:
            meta = pd.concat([meta, challenge["meta"].iloc[add]], ignore_index=True)
            baseline = np.vstack([baseline, challenge["baseline"][np.ix_(add, ccols)]])
            delta = np.vstack([delta, challenge["delta"][np.ix_(add, ccols)]])
        indices = np.array([vocab.index(p) for p in meta.perturbation], dtype=np.int64)
        path = output / name
        info = {"synthetic": original["audit"].get("synthetic", False),
            "normalization": original["audit"].get("normalization"), "target_sum": original["audit"].get("target_sum"),
            "reference_fingerprint": original["audit"]["fingerprint"],
            "challenge_fingerprint": challenge["audit"]["fingerprint"], "added_h1_rows": len(add) if append else 0,
            "panel": "intersection; ALL original validation rows retained", "test_evaluated": False}
        if path.exists() and (path / "dataset.npz").exists():
            saved = load_prepared(path)
            if not np.array_equal(saved["baseline"], baseline) or not np.array_equal(saved["delta"], delta) or saved["meta"].row_id.tolist() != meta.row_id.tolist() or saved["genes"].tolist() != genes or saved["perturbations"].tolist() != vocab:
                raise ValueError("Existing expansion data changed")
        else:
            save_prepared(path, baseline, delta, indices, genes, vocab, meta, info)
        paths[name] = path
    return paths


def run_expansion(original, challenge, root, base_config, seeds, resume=False):
    root = Path(root)
    paths = prepare_expansion(original, challenge, root / "prepared")
    datasets = {k: load_prepared(p) for k, p in paths.items()}
    reference = datasets["original_only"]
    val = reference["splits"]["val"]
    for data in datasets.values():
        their_val = data["splits"]["val"]
        if not np.array_equal(reference["delta"][val], data["delta"][their_val]) or not np.array_equal(reference["baseline"][val], data["baseline"][their_val]) or reference["meta"].iloc[val].row_id.tolist() != data["meta"].iloc[their_val].row_id.tolist():
            raise ValueError("Expansion held-out panel changed")
    membership, _ = make_modules(reference, min(8, len(reference["genes"])))
    predictions, records = {}, []
    budget = len(reference["splits"]["train"])
    for name, data in datasets.items():
        coverage_audit(data, root / "coverage" / name)
        for key, values in simple_baselines(data).items():
            predictions[name + "/" + key] = values[data["splits"]["val"]]
        for seed in seeds:
            cfg = copy.deepcopy(base_config)
            cfg.update(data_dir=str(paths[name]), output_dir=str(root / "students" / f"{name}_seed{seed}"),
                       seed=seed, arms=["supervised"], teachers={}, teacher_mix="equal", epoch_train_rows=budget,
                       allow_unverified_teachers=False)
            cfg["model"]["background_interaction"] = True
            run_qwen(cfg, resume=resume)
            values, identity = validation_prediction(cfg["output_dir"], data, "supervised")
            predictions[f"{name}_seed{seed}"] = values
            records.append({"name": name, "seed": seed, "identity": identity})
            compare_validation(reference, predictions, membership, root)
    write_json(root / "audit.json", {"runs": records, "epoch_train_rows": budget, "test_evaluated": False,
        "n_genes": len(reference["genes"]), "n_val_rows": len(val),
        "train_rows": {k: len(v["splits"]["train"]) for k, v in datasets.items()},
        "note": "Same shared vocabulary, genes, held-out labels, architecture, seeds and maximum optimizer steps. "
        "Early stopping can differ. Added data are sampled without replacement each epoch; all rows remain in the training pool. "
        "This tests H1 expansion at fixed compute, not cross-species generalization. Compare within this panel only."})
    return root
