"""Controlled background and teacher-specialization ablations using cached encoders."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tarfile

import numpy as np
import torch

from vcell.data import load_prepared
from vcell.qwen import run_qwen, backbone_identity
from vcell.challenge_compare import validation_prediction
from vcell.specialization import coverage_audit, make_modules, nested_teachers, compare_validation
from vcell.round5_methods import load_features, ridge_baseline
from vcell.train import source_fingerprint, simple_baselines
from vcell.utils import read_config, file_sha256, write_json

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ["scgpt", "state", "scfoundation", "geneformer", "uce"]
EXPERIMENTS = {"A_supervised": (False, None), "B_background": (True, None),
               "C_equal": (True, "equal"), "D_modules": (True, "module"),
               "E_specialists": (True, "specialist"), "F_gated_equal": (True, "gated_equal")}


def package_review(root):
    # No weights or predictions: keep transfer from AutoDL small.
    with tarfile.open(root / "round6_review.tar.gz", "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".json"} and "heads" not in path.parts:
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=os.environ.get("VCELL_WORK", "/root/autodl-tmp/vcell-work"))
    parser.add_argument("--name", default="round6_01")
    parser.add_argument("--data")
    parser.add_argument("--feature-root")
    parser.add_argument("--round5-run")
    parser.add_argument("--reference-run", help="Optional completed 421 run for historical A/B comparison")
    parser.add_argument("--features-json", help="Family -> frozen feature file; required for a changed dataset")
    parser.add_argument("--backgrounds-json", help="Dataset -> explicit biological/technical attributes")
    parser.add_argument("--modules-json", help="Module -> prepared gene IDs; otherwise use training controls")
    parser.add_argument("--modules", type=int, default=8)
    parser.add_argument("--challenge-data", help="Optional existing prepared student-C dataset for a matched-step H1 expansion test")
    parser.add_argument("--min-specialty-targets", type=int, default=10)
    parser.add_argument("--model-dir")
    parser.add_argument("--student-config", default=str(REPO / "configs/qwen_three_teachers.yaml"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29])
    parser.add_argument("--arms", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("Use a simple alphanumeric run name")
    if min(args.epochs, args.head_epochs, args.modules, args.min_specialty_targets) < 1 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        parser.error("Positive counts, distinct arms and distinct nonnegative seeds required")
    torch.set_num_threads(4)
    work = Path(args.work_dir).resolve()
    data_dir = Path(args.data).resolve() if args.data else work / "prepared/real_min10_01"
    data = load_prepared(data_dir)
    challenge = load_prepared(args.challenge_data) if args.challenge_data else None
    backgrounds = json.loads(Path(args.backgrounds_json).read_text()) if args.backgrounds_json else {}
    if args.audit_only:
        dest = work / "reports" / args.name
        coverage_audit(data, dest, backgrounds)
        print(f"ROUND6 COVERAGE: {dest}", flush=True)
        return dest
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("CUDA/BF16 unavailable; use --audit-only for CPU coverage")
    feature_root = Path(args.feature_root) if args.feature_root else work / "teachers/four_teacher_01"
    round5 = Path(args.round5_run) if args.round5_run else work / "runs/round5_01"
    paths = {f: feature_root / f / "features.npz" for f in FAMILIES[:-1]}
    paths["uce"] = round5 / "uce/features.npz"
    if args.features_json:
        paths = {k: Path(v).resolve() for k, v in json.loads(Path(args.features_json).read_text()).items()}
    if set(paths) != set(FAMILIES):
        raise ValueError("Supply all five real frozen teacher feature caches")
    for path in paths.values():
        load_features(path, data)
    definition = json.loads(Path(args.modules_json).read_text()) if args.modules_json else None
    membership, module_audit = make_modules(data, args.modules, definition)
    reference_run = Path(args.reference_run) if args.reference_run else work / "runs/qwen_421_four_teacher_01"
    historical, historical_ids = {}, {}
    if (reference_run / "COMPLETE.json").exists():
        complete = json.loads((reference_run / "COMPLETE.json").read_text())
        manifest = json.loads((reference_run / "run_manifest.json").read_text())
        if manifest["data_fingerprint"] == data["audit"]["fingerprint"]:
            for name in ("student_a", "student_b"):
                candidate = next(c for c in complete["candidates"] if c["model"] == name)
                if file_sha256(reference_run / name / "all/best.pt") != candidate["checkpoint_sha256"]:
                    raise ValueError("Historical checkpoint changed")
                historical["historical/"+name], historical_ids[name] = validation_prediction(reference_run / name, data, "all")
        elif args.reference_run:
            raise ValueError("Explicit historical reference has a different dataset")
    elif args.reference_run:
        raise ValueError("Explicit reference run is incomplete")
    model_dir = Path(args.model_dir) if args.model_dir else work / "models/Qwen3-0.6B-Base"
    cfg = read_config(args.student_config)
    cfg.update(epochs=args.epochs, patience=5, batch_size=4, gradient_accumulation=4,
        max_train_rows=None, max_val_rows=None, teacher_families=FAMILIES, teachers={},
        num_threads=4, device=args.device, allow_unverified_teachers=True, loss_weighting="uniform")
    cfg["model"].update(model_id=str(model_dir.resolve()), revision=None, local_files_only=True,
                         dtype="bfloat16" if args.device == "cuda" else "float32")
    for key in ("functional", "background_interaction"):
        cfg["model"].pop(key, None)
    for key in ("reliability", "kd_mask", "distillation"):
        cfg.pop(key, None)
    plan = {"data_fingerprint": data["audit"]["fingerprint"], "data_dir": str(data_dir),
        "source_fingerprint": source_fingerprint(), "launcher_sha256": file_sha256(__file__),
        "metadata_sha256": file_sha256(data_dir / "metadata.csv"),
        "data_audit_sha256": file_sha256(data_dir / "data_audit.json"),
        "features": {n: [str(p.resolve()), file_sha256(p), file_sha256(p.with_suffix('.json'))] for n, p in paths.items()},
        "student_config": cfg, "backbone_identity": backbone_identity(cfg["model"]),
        "head_epochs": args.head_epochs, "seeds": args.seeds, "arms": args.arms,
        "min_specialty_targets": args.min_specialty_targets, "modules": module_audit,
        "backgrounds": backgrounds, "challenge_fingerprint": challenge["audit"]["fingerprint"] if challenge else None,
        "historical_references": historical_ids,
        "test_evaluated": False}
    root = work / "runs" / args.name
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root / "plan.json").exists() or json.loads((root / "plan.json").read_text()) != plan:
            raise ValueError("Resume identical code/data/config only, or use a fresh --name")
    reserve = (1 + .4 * len(args.seeds) * (len(args.arms) + (2 if challenge else 0))) * 1024**3
    # Account for outputs already present during a same-plan resume.
    existing = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    free = shutil.disk_usage(work).free
    print(f"ROUND6 PREFLIGHT: free={free/1024**3:.1f} GiB, additional reserve={max(0,reserve-existing)/1024**3:.1f} GiB; no model downloads", flush=True)
    if free < max(0, reserve-existing):
        raise ValueError("Insufficient data-disk space; no old artifacts were deleted")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    write_json(root / "modules.json", module_audit)
    coverage_audit(data, root / "coverage", backgrounds)
    def progress(stage):
        write_json(root / "progress.json", {"stage": stage, "test_evaluated": False})
        print("ROUND6 STAGE: " + stage, flush=True)
    progress("nested teacher calibration and specialization")
    packages = nested_teachers(data, paths, membership, root / "heads", epochs=args.head_epochs,
                              device=args.device, min_targets=args.min_specialty_targets)
    # Compact diagnostics belong in the review package; large fitting caches do not.
    for name in ("diagnostics.json", "teacher_specialties.csv", "audit.json"):
        if (root / "heads" / name).exists():
            destination = name if name.startswith("teacher_") else "teacher_" + name
            shutil.copyfile(root / "heads" / name, root / destination)
    if args.diagnostics_only:
        progress("diagnostics complete; students not started")
        package_review(root)
        print(f"ROUND6 DIAGNOSTICS COMPLETE: {root}", flush=True)
        return root
    progress("Ridge baseline")
    val = data["splits"]["val"]
    predictions = {n: a[val] for n, a in simple_baselines(data).items()}
    predictions.update(historical)
    predictions["ridge_lowrank"] = ridge_baseline(data, root / "ridge")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    records = []
    compare_validation(data, predictions, membership, root)
    for seed in args.seeds:
        for label in args.arms:
            background, teacher = EXPERIMENTS[label]
            name = f"{label}_seed{seed}"
            progress("train " + name)
            current = copy.deepcopy(cfg)
            current.update(seed=seed, data_dir=str(data_dir), output_dir=str(root / "students" / name),
                           arms=["all" if teacher else "supervised"], teacher_mix="module" if teacher else "equal")
            current["model"]["background_interaction"] = background
            if teacher:
                current["distillation"] = packages[teacher]
            run_qwen(current, resume=args.resume)
            predictions[name], identity = validation_prediction(current["output_dir"], data, current["arms"][0])
            records.append({"name": name, "seed": seed, "config": current, "identity": identity})
            compare_validation(data, predictions, membership, root)
    write_json(root / "comparison_audit.json", {"data_fingerprint": data["audit"]["fingerprint"],
        "runs": records, "test_evaluated": False, "note": "Identical full panel, optimizer settings and paired seeds. "
        "A->B changes the reference-plus-interaction architecture; C->D changes routing and strength; "
        "F->D isolates teacher selection at the same gate strength. D->E changes specialist teacher fitting. "
        "Specialist heads retain the same architecture and training epochs. Validation is reused development data; "
        "bootstrap intervals are descriptive, not corrected significance tests. Historical A/B used a different "
        "teacher fitting protocol and are contextual references, not matched ablations. No checkpoint auto-promoted."})
    if challenge is not None:
        from vcell.expansion import run_expansion
        progress("matched-panel H1 data expansion")
        run_expansion(data, challenge, root / "expansion", cfg, args.seeds, args.resume)
    write_json(root / "COMPLETE.json", {"arms_completed": len(records), "test_evaluated": False,
               "expansion_completed": challenge is not None,
               "plan_sha256": file_sha256(root / "plan.json")})
    progress("complete")
    package_review(root)
    print(f"ROUND6 COMPLETE: {root / 'comparison.csv'}", flush=True)
    print(f"REVIEW PACKAGE: {root / 'round6_review.tar.gz'}", flush=True)
    return root


if __name__ == "__main__":
    main()
