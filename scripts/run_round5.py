"""Auditable fifth-teacher, functional-prior and reliability ablations on one dataset."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pandas as pd
import torch

from prepare_qwen import prepare, matches
from vcell.data import load_prepared
from vcell.functional import prepare_function
from vcell.uce_encoder import export_uce
from vcell.round5_methods import load_features, crossfit_teacher, reliability_report, ridge_baseline
from vcell.qwen import run_qwen
from vcell.challenge_compare import validation_prediction
from vcell.selection import mse, row_weights
from vcell.train import source_fingerprint, simple_baselines
from vcell.teachers import load_cache
from vcell.utils import file_sha256, write_json

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ["scgpt", "state", "scfoundation", "geneformer", "uce"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=os.environ.get("VCELL_WORK", "/root/autodl-tmp/vcell-work"))
    parser.add_argument("--name", default="round5_01")
    parser.add_argument("--data")
    parser.add_argument("--parent-run")
    parser.add_argument("--feature-root")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args(argv)
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("Use an alphanumeric run name with underscores/hyphens")
    if min(args.epochs, args.head_epochs) < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Positive epochs and distinct nonnegative seeds required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("This launcher needs CUDA with BF16 support")
    torch.set_num_threads(4)
    work = Path(args.work_dir).resolve()
    data_dir = Path(args.data).resolve() if args.data else work / "prepared/real_min10_01"
    parent = Path(args.parent_run).resolve() if args.parent_run else work / "runs/qwen_421_four_teacher_01"
    feature_root = Path(args.feature_root).resolve() if args.feature_root else work / "teachers/four_teacher_01"
    data = load_prepared(data_dir)
    manifest = json.loads((parent / "run_manifest.json").read_text())
    parent_complete = json.loads((parent / "COMPLETE.json").read_text())
    if manifest["data_fingerprint"] != data["audit"]["fingerprint"] or parent_complete["run_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Parent run differs from the round-5 dataset")
    if parent_complete["test_evaluated"]:
        raise ValueError("Keep the final test sealed")
    paths = {f: feature_root / f / "features.npz" for f in FAMILIES[:-1]}
    for path in paths.values():
        load_features(path, data)
    root = work / "runs" / args.name
    locks = {name: json.loads((REPO / f"configs/{name}.lock.json").read_text()) for name in ("uce_backbone", "uce_aux")}
    folders = {"uce_backbone": work / "models/UCE-33", "uce_aux": work / "models/UCE-aux"}
    plan = {"data_fingerprint": data["audit"]["fingerprint"], "source_fingerprint": source_fingerprint(),
            "launcher_sha256": file_sha256(__file__), "parent_complete_sha256": file_sha256(parent / "COMPLETE.json"),
            "feature_files": {f: [str(p), file_sha256(p), file_sha256(p.with_suffix('.json'))] for f, p in paths.items()},
            "locks": locks, "seeds": args.seeds, "epochs": args.epochs, "head_epochs": args.head_epochs,
            "function_sources": json.loads((REPO / "assets/function/sources.json").read_text()),
            "student_base_config": manifest["config"]["student"], "test_evaluated": False}
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root / "plan.json").exists() or json.loads((root / "plan.json").read_text()) != plan:
            raise ValueError("Existing round: --resume identical inputs, or choose a new --name")
    missing = sum(spec["size"] for name, lock in locks.items() for file, spec in lock["files"].items()
                  if not matches(folders[name] / file, spec))
    free = shutil.disk_usage(work).free
    reserve = (1 + 2 * len(args.seeds)) * 1024**3
    print(f"ROUND5 PREFLIGHT: free={free/1024**3:.1f} GiB; new model files={missing/1024**3:.1f} GiB; reserve={reserve/1024**3:.0f} GiB", flush=True)
    if free < missing + reserve:
        raise ValueError("Insufficient data-disk space for pinned UCE files plus outputs; no existing files were removed")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    def progress(stage):
        write_json(root / "progress.json", {"stage": stage, "test_evaluated": False})
        print(f"ROUND5 STAGE: {stage}", flush=True)
    progress("download and verify UCE-33")
    for name in locks:
        prepare(folders[name], args.endpoint, lock=locks[name], label=name.upper())
    paths["uce"] = root / "uce/features.npz"
    progress("extract frozen UCE control features; partial groups resumable")
    if paths["uce"].exists() and paths["uce"].with_suffix(".json").exists():
        load_features(paths["uce"], data)
    else:
        export_uce(data, paths["uce"], folders["uce_backbone"], folders["uce_aux"])
    functional = root / "functional/features.npz"
    prepare_function(data, REPO / "assets/function", functional)
    caches = {}
    for family, path in paths.items():
        progress("cross-fit teacher " + family)
        cache = root / "teachers" / family / "predictions.npz"
        if cache.exists() and cache.with_suffix(".json").exists():
            _, audit = load_cache(cache, data, allow_unverified=True)
            if audit["feature_sha256"] != file_sha256(path) or audit["epochs"] != args.head_epochs:
                raise ValueError("Saved cross-fitted head differs from this experiment")
        else:
            crossfit_teacher(data, path, family, cache.parent, epochs=args.head_epochs, device="cuda")
        caches[family] = str(cache)
        gc.collect()
        torch.cuda.empty_cache()
    reliability_path = root / "reliability.json"
    report = reliability_report(data, caches, reliability_path)
    if args.teacher_only:
        print(f"ROUND5 TEACHERS COMPLETE: {root}", flush=True)
        return
    progress("Ridge train-context cross validation")
    predictions = {"ridge_lowrank": ridge_baseline(data, root / "ridge")}
    val = data["splits"]["val"]
    for name, array in simple_baselines(data).items():
        predictions[name] = array[val]
    for name in ("student_a", "student_b"):
        candidate = next(c for c in parent_complete["candidates"] if c["model"] == name)
        if file_sha256(parent / name / "all/best.pt") != candidate["checkpoint_sha256"]:
            raise ValueError("Historical student checkpoint changed")
        predictions["historical/" + name], _ = validation_prediction(parent / name, data, "all")
    model_dir = prepare(work / "models/Qwen3-0.6B-Base", args.endpoint)
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Each adjacent KD comparison changes one factor; supervised function arm
    # checks whether annotations help independently of distillation.
    experiments = [
        ("supervised_uniform", "supervised", False, False, "equal", 5),
        ("supervised_weighted", "supervised", True, False, "equal", 5),
        ("supervised_function", "supervised", True, True, "equal", 5),
        ("kd4_equal", "all", True, False, "equal", 4),
        ("kd5_equal", "all", True, False, "equal", 5),
        ("kd5_reliable", "all", True, False, "reliability", 5),
        ("kd5_reliable_function", "all", True, True, "reliability", 5),
    ]
    records = []
    def save_comparison():
        rows = [{"model": name, "split": "val", "mse_delta": mse(value, data["delta"][val], row_weights(data["meta"].iloc[val])),
                 "n_rows": len(val), "n_genes": len(data["genes"]), "n_perturbations": int(data["meta"].iloc[val].perturbation.nunique())}
                for name, value in predictions.items()]
        pd.DataFrame(rows).sort_values("mse_delta").to_csv(root / "comparison.csv", index=False)
        by_target = []
        meta = data["meta"].iloc[val]
        for pert in sorted(meta.perturbation.unique()):
            selected = np.flatnonzero((meta.perturbation == pert).to_numpy())
            for name, value in predictions.items():
                by_target.append({"model": name, "perturbation": pert, "n_rows": len(selected),
                    "mse_delta": mse(value[selected], data["delta"][val[selected]], row_weights(meta.iloc[selected]))})
        pd.DataFrame(by_target).to_csv(root / "by_perturbation.csv", index=False)
    save_comparison()
    for seed in args.seeds:
        for name, arm, weighted, function, mix, nteachers in experiments:
            label = f"{name}_seed{seed}"
            progress("train " + label)
            cfg = copy.deepcopy(manifest["config"]["student"])
            families = FAMILIES[:nteachers]
            cfg.update(data_dir=str(data_dir), output_dir=str(root / "students" / label), seed=seed,
                       arms=[arm], teachers={f: caches[f] for f in families} if arm == "all" else {},
                       teacher_families=families, teacher_mix=mix, allow_unverified_teachers=True,
                       max_train_rows=None, max_val_rows=None, epochs=args.epochs, patience=5, num_threads=4,
                       loss_weighting="context_perturbation" if weighted else "uniform", device="cuda")
            cfg["model"].update(model_id=str(model_dir), revision=None, local_files_only=True)
            cfg["model"].pop("functional", None)
            if function:
                cfg["model"]["functional"] = {"path": str(functional), "sha256": file_sha256(functional)}
            if arm == "all":
                cfg["kd_mask"] = {"path": report["mask_path"], "sha256": report["mask_sha256"]}
            if mix == "reliability":
                cfg["reliability"] = {"path": str(reliability_path), "sha256": file_sha256(reliability_path)}
            run_qwen(cfg, resume=args.resume)
            value, identity = validation_prediction(cfg["output_dir"], data, arm)
            predictions[label] = value
            records.append({"name": label, "config": cfg, "identity": identity})
            save_comparison()
    write_json(root / "comparison_audit.json", {"data_fingerprint": data["audit"]["fingerprint"],
               "runs": records, "reliability": report, "test_evaluated": False, "seeds": args.seeds,
               "note": "Same full validation panel for every arm. Historical A/B used a different head-fitting protocol. "
                       "Pretraining overlap remains unverified. Validation has been reused for exploratory model selection; "
                       "do not present it as untouched test performance. No previous winner overwritten."})
    write_json(root / "COMPLETE.json", {"arms_completed": len(records), "comparison": str(root / "comparison.csv"),
               "test_evaluated": False, "plan_sha256": file_sha256(root / "plan.json")})
    review_files = [root / p for p in ("COMPLETE.json", "plan.json", "comparison.csv", "by_perturbation.csv",
                    "comparison_audit.json", "reliability.json", "functional/features.json", "ridge/audit.json")]
    review_files += sorted((root / "students").glob("*/*/history.csv"))
    review_files += sorted((root / "students").glob("*/*/validation.json"))
    with tarfile.open(root / "round5_review.tar.gz", "w:gz") as archive:
        for path in review_files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    progress("complete")
    print(f"ROUND5 COMPLETE: {root / 'comparison.csv'}", flush=True)
    print(f"REVIEW PACKAGE: {root / 'round5_review.tar.gz'}", flush=True)


if __name__ == "__main__":
    main()
