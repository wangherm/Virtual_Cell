"""AutoDL: four frozen pretrained encoders, four heads, two Qwens, one selection."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch
import yaml

from prepare_qwen import prepare, matches
from vcell.data import load_prepared
from vcell.teacher_pipeline import export_native_teacher, fit_feature_teacher
from vcell.teachers import load_cache
from vcell.train import source_fingerprint
from vcell.training421 import run_421
from vcell.utils import file_sha256, write_json

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("scgpt", "state", "scfoundation", "geneformer")
FOLDERS = dict(scgpt="scGPT-human", state="State-SE-100M", scfoundation="scFoundation", geneformer="Geneformer-V1-10M")
IMPLEMENTATIONS = dict(scgpt="vcell_frozen_scgpt_v1", state="vcell_state_se100m_v1",
                       scfoundation="vcell_scfoundation_v1", geneformer="vcell_geneformer_v1")
GENEFORMER_DICT = "geneformer/gene_dictionaries_30m/"
FILES = {
    "scgpt": dict(checkpoint="best_model.pt", args="args.json", vocab="vocab.json"),
    "state": dict(checkpoint="model.safetensors", config="config.yaml", proteins="protein_embeddings.pt"),
    "scfoundation": dict(checkpoint="models.ckpt"),
    "geneformer": dict(checkpoint="Geneformer-V1-10M/model.safetensors", config="Geneformer-V1-10M/config.json",
                       vocab=GENEFORMER_DICT+"token_dictionary_gc30M.pkl", medians=GENEFORMER_DICT+"gene_median_dictionary_gc30M.pkl",
                       mapping=GENEFORMER_DICT+"ensembl_mapping_dict_gc30M.pkl"),
}


def provenance(family, lock, data):
    return {
        "teacher_name": ("State SE-100M" if family == "state" else family) + " frozen control encoder + VCell response head",
        "teacher_family": family, "model_revision": lock["revision"],
        "training_contexts": [], "excluded_contexts": [],
        "source": "https://huggingface.co/" + lock["model_id"] + "/tree/" + lock["revision"],
        "license": "See THIRD_PARTY_NOTICES.md and the pinned upstream model terms",
        "prediction_space": "prepared_log1p_delta", "declared_no_holdout_perturbations": False,
        "benchmark_status": "exploratory_pretraining_overlap_unverified",
        "audit_notes": "Pretraining row-level exclusion has not been verified. Internal scores are exploratory, "
                       "not a clean generalization benchmark. VCell head fitting uses train labels only; validation "
                       "selects its epoch; test labels are never used for fitting or selection. State uses SE, not ST. "
                       "scFoundation checkpoint is the GenBio redistribution; upstream-original checksum identity is not established.",
        "head_training_contexts": sorted(data["meta"].loc[data["meta"].split == "train", "context"].unique().tolist()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=os.environ.get("VCELL_WORK", "/root/autodl-tmp/vcell-work"))
    parser.add_argument("--data")
    parser.add_argument("--name", default="four_teacher_01")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--parallel-students", type=int, choices=[1, 2], default=2)
    args = parser.parse_args(argv)
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("--name may contain letters, digits, underscores and hyphens only")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("AutoDL 421 requires CUDA with bfloat16 support")
    torch.set_num_threads(4)
    work = Path(args.work_dir).resolve()
    data_dir = Path(args.data).resolve() if args.data else work / "prepared/real_min10_01"
    root = work / "teachers" / args.name
    run = work / "runs" / ("qwen_421_" + args.name)
    data = load_prepared(data_dir)
    locks = {family: json.loads((REPO / "configs" / (family + "_backbone.lock.json")).read_text()) for family in FAMILIES}
    qwen_lock = json.loads((REPO / "configs/qwen_backbone.lock.json").read_text())
    snapshots = [(locks[f], work / "models" / FOLDERS[f]) for f in FAMILIES]
    snapshots.append((qwen_lock, work / "models/Qwen3-0.6B-Base"))
    needed = sum(spec["size"] for lock, folder in snapshots for name, spec in lock["files"].items()
                 if not matches(folder / name, spec))
    free = shutil.disk_usage(work).free
    reserve = 5 * 1024**3
    print(f"421 PREFLIGHT: free={free/1024**3:.1f} GiB missing_models={needed/1024**3:.1f} GiB reserve=5 GiB", flush=True)
    if free < needed + reserve:
        raise ValueError("Insufficient data-disk space for missing models and 5 GiB training reserve")
    plan = {"data_dir": str(data_dir), "data_fingerprint": data["audit"]["fingerprint"], "locks": locks,
            "source_fingerprint": source_fingerprint(), "launcher_sha256": file_sha256(__file__),
            "epochs": args.epochs, "parallel_students": args.parallel_students,
            "allow_unverified_teachers": True, "max_control_cells": 8,
            "student_seeds": [17, 29], "qwen_lock": qwen_lock}
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root / "plan.json").is_file() or json.loads((root / "plan.json").read_text()) != plan:
            raise ValueError("Existing 421 teacher run: use --resume with identical settings or choose a new --name")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    print("EXPLORATORY RUN: pretrained overlap is unverified; all heads use only train labels. State = SE-100M.", flush=True)
    caches = {}
    for number, family in enumerate(FAMILIES, 1):
        print(f"421 TEACHER {number}/4 START: {family}", flush=True)
        lock = locks[family]
        folder = prepare(work / "models" / FOLDERS[family], args.endpoint, lock=lock, label=family.upper())
        model = {"implementation": IMPLEMENTATIONS[family], "seed": 0, "duplicate_symbol_policy": "sum_counts"}
        for key, name in FILES[family].items():
            model[key], model[key + "_sha256"] = str(folder / name), lock["files"][name]["sha256"]
        if family == "scgpt":
            model["max_input_genes"] = 1200
        if family == "scfoundation":
            vocab = REPO / "assets/teacher_vocab/scfoundation_gene_index.tsv"
            if not matches(vocab, lock["vocabulary"]):
                raise ValueError("Bundled scFoundation vocabulary checksum mismatch")
            model.update(vocab=str(vocab), vocab_sha256=lock["vocabulary"]["sha256"])
        stage = root / family
        stage.mkdir(exist_ok=True)
        write_json(stage / "provenance.json", provenance(family, lock, data))
        feature_cfg = {"family": family, "data_dir": str(data_dir), "output": str(stage / "features.npz"),
                       "provenance": str(stage / "provenance.json"), "device": "cuda", "seed": 0,
                       "max_control_cells": 8, "gene_symbol_key": "gene_name", "model": model,
                       "allow_unverified_teachers": True}
        write_json(stage / "export_config.json", feature_cfg)
        features = Path(feature_cfg["output"])
        if features.exists() and features.with_suffix(".json").is_file():
            saved = json.loads(features.with_suffix(".json").read_text())
            if saved.get("adapter_config") != feature_cfg or saved.get("feature_file_sha256") != file_sha256(features):
                raise ValueError("Saved features changed; use a fresh --name")
            print(f"{family} FEATURES VERIFIED: reuse", flush=True)
        else:
            if features.exists() or features.with_suffix(".json").exists():
                raise ValueError("Incomplete feature write; preserve it and use a new --name")
            export_native_teacher(feature_cfg)
        gc.collect()
        torch.cuda.empty_cache()
        head_cfg = {"data_dir": str(data_dir), "features": str(features), "output_dir": str(stage / "head"),
                    "device": "cuda", "seed": 0, "hidden": 128, "epochs": 30, "patience": 5,
                    "batch_size": 64, "learning_rate": .001, "allow_unverified_teachers": True}
        write_json(stage / "head_config.json", head_cfg)
        cache = stage / "head/predictions.npz"
        if cache.exists() and cache.with_suffix(".json").is_file():
            _, saved = load_cache(cache, data, allow_unverified=True)
            if saved.get("head_config") != head_cfg or saved.get("head_sha256") != file_sha256(stage / "head/head.pt"):
                raise ValueError("Saved teacher head changed; use a new --name")
        else:
            head = stage / "head"
            if head.exists() and any(head.iterdir()):
                # Preserve an interrupted short head fit and restart from epoch 1.
                archived = stage / ("head_interrupted_" + str(time.time_ns()))
                head.rename(archived)
                print(f"Preserved interrupted head at {archived}; restarting this head", flush=True)
            fit_feature_teacher(head_cfg)
        caches[family] = str(cache)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"421 TEACHER {number}/4 COMPLETE: {cache}", flush=True)
    prepare(work / "models/Qwen3-0.6B-Base", args.endpoint)
    cfg = yaml.safe_load((REPO / "configs/qwen_smoke.yaml").read_text())
    cfg.update(data_dir=str(data_dir), output_dir=str(run), arms=["all"], teachers=caches,
               teacher_families=list(FAMILIES), teacher_mix="equal", allow_unverified_teachers=True,
               max_train_rows=None, max_val_rows=None, epochs=args.epochs, patience=5, log_every=10, num_threads=4)
    cfg["model"].update(model_id=str(work / "models/Qwen3-0.6B-Base"), revision=None, local_files_only=True)
    experiment = {"output_dir": str(run), "student": cfg, "student_seeds": [17, 29], "parallel_students": args.parallel_students}
    write_json(root / "421_config.json", experiment)
    print(f"421 CONFIG: {root / '421_config.json'}", flush=True)
    if not args.teacher_only:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        run_421(experiment, resume=args.resume)
    print(f"421 PIPELINE COMPLETE: {root if args.teacher_only else run}", flush=True)


if __name__ == "__main__":
    main()
