"""Train one supervised Qwen on VCC2025 and compare with existing A/B students."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import torch

from prepare_qwen import prepare
from vcell.challenge_data import aggregate_training, prepare_student_c
from vcell.challenge_compare import compare_students
from vcell.data import load_prepared
from vcell.qwen import run_qwen
from vcell.train import source_fingerprint
from vcell.utils import file_sha256, write_json

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=os.environ.get("VCELL_WORK", "/root/autodl-tmp/vcell-work"))
    parser.add_argument("--name", default="vcc2025_c_01")
    parser.add_argument("--parent-run")
    parser.add_argument("--data")
    parser.add_argument("--h5ad", help="Optional existing official training h5ad; otherwise stream the pinned public GCS object")
    parser.add_argument("--training-source", choices=["challenge-only", "combined"], default="challenge-only")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args(argv)
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("Use only letters, digits, underscores and hyphens in --name")
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    work = Path(args.work_dir).resolve()
    parent = Path(args.parent_run).resolve() if args.parent_run else work / "runs/qwen_421_four_teacher_01"
    data_dir = Path(args.data).resolve() if args.data else work / "prepared/real_min10_01"
    root = work / "runs" / args.name
    prepared = work / "prepared" / args.name
    aggregates = work / "prepared" / (args.name + "_aggregation")
    data = load_prepared(data_dir)
    parent_manifest = json.loads((parent / "run_manifest.json").read_text())
    parent_complete = json.loads((parent / "COMPLETE.json").read_text())
    if parent_manifest["data_fingerprint"] != data["audit"]["fingerprint"] or parent_complete["run_fingerprint"] != parent_manifest["fingerprint"]:
        raise ValueError("Parent 421 result/data mismatch")
    if parent_complete["test_evaluated"]:
        raise ValueError("Use the parent run with its test set still sealed")
    for name in ("student_a", "student_b"):
        if not (parent / name / "COMPLETE.json").is_file():
            raise ValueError(f"Missing completed {name} run")
    if not args.prepare_only and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("Training launcher requires CUDA with bfloat16 support")
    free = shutil.disk_usage(work).free
    print(f"STUDENT C PREFLIGHT: data disk free={free/1024**3:.1f} GiB; full source will not be stored", flush=True)
    if free < 3 * 1024**3:
        raise ValueError("At least 3 GiB free space required for aggregation and the new student")
    source = json.loads((REPO / "configs/vcc2025_train.lock.json").read_text())
    local = str(Path(args.h5ad).resolve()) if args.h5ad else None
    plan = {"source": source, "parent_run": str(parent), "parent_fingerprint": parent_manifest["fingerprint"],
            "parent_complete_sha256": file_sha256(parent / "COMPLETE.json"), "reference_data": str(data_dir),
            "reference_fingerprint": data["audit"]["fingerprint"], "training_source": args.training_source,
            "seed": args.seed, "local_h5ad": local, "source_fingerprint": source_fingerprint(),
            "launcher_sha256": file_sha256(__file__)}
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root / "plan.json").exists() or json.loads((root / "plan.json").read_text()) != plan:
            raise ValueError("Existing student-C experiment: resume identical inputs or use a new --name")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    print(f"STUDENT C: training_source={args.training_source}; real labels only; no teacher targets; seed={args.seed}", flush=True)
    aggregate = aggregate_training(source, data_dir, aggregates, local_h5ad=local, resume=args.resume)
    if (prepared / "dataset.npz").exists():
        cached = load_prepared(prepared)
        audit = cached["audit"]
        if audit["challenge_training"]["aggregate_sha256"] != file_sha256(aggregate) or audit["challenge_training"]["training_source"] != args.training_source:
            raise ValueError("Existing prepared C data differs from this experiment")
    else:
        prepare_student_c(data_dir, aggregate, prepared, training_source=args.training_source)
    if args.prepare_only:
        print(f"STUDENT C PREPARATION COMPLETE: {prepared}", flush=True)
        return
    cfg = copy.deepcopy(parent_manifest["config"]["student"])
    cfg.update(data_dir=str(prepared), output_dir=str(root / "student_c"), arms=["supervised"], teachers={},
               kd_weight=0., seed=args.seed, allow_unverified_teachers=False, max_train_rows=None, max_val_rows=None,
               device="cuda", num_threads=4)
    cfg.pop("teacher_families", None)
    model_dir = prepare(work / "models/Qwen3-0.6B-Base", args.endpoint)
    cfg["model"].update(model_id=str(model_dir), revision=None, local_files_only=True)
    write_json(root / "student_c_config.json", cfg)
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run_qwen(cfg, resume=args.resume)
    compared = compare_students(data_dir, parent, prepared, root / "student_c", root / "comparison")
    write_json(root / "COMPLETE.json", {"student": "student_c", "training_source": args.training_source,
               "comparison_csv": str(compared / "comparison_common.csv"), "test_evaluated": False,
               "parent_models_unchanged": True, "official_vcc_submission": False})
    print(f"STUDENT C COMPLETE: {root}", flush=True)


if __name__ == "__main__":
    main()
