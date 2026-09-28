"""Prepare the real scGPT teacher and an offline Qwen comparison on AutoDL."""
import argparse
import json
import os
from pathlib import Path

import torch
import yaml
import pandas as pd

from prepare_qwen import prepare
from vcell.data import load_prepared
from vcell.teacher_pipeline import export_native_teacher, fit_feature_teacher
from vcell.teachers import load_cache
from vcell.train import source_fingerprint, simple_baselines
from vcell.selection import mse, row_weights
from vcell.utils import file_sha256, write_json

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "configs/scgpt_backbone.lock.json"


def benchmark_provenance(data, lock):
    """Published-source audit, restricted to this specific four-dataset pilot.

    This is NOT a row-level audit of the original pretraining corpus. See the
    explicit evidence level and limitations saved with every exported cache.
    """
    meta = data["meta"]
    expected = {"train": {"K562", "RPE1"}, "val": {"HepG2"}, "test": {"Jurkat"}}
    for split, contexts in expected.items():
        if set(meta.loc[meta.split == split, "context"]) != contexts:
            raise ValueError("Automatic published-source audit supports only the K562/RPE1 -> HepG2/Jurkat pilot")
    paths = {s["id"]: Path(s["path"]).name.lower() for s in data["audit"]["sources"]}
    for context, name in (("HepG2", "hepg2"), ("Jurkat", "jurkat")):
        datasets = meta.loc[meta.context == context, "dataset"].unique()
        if any(paths.get(d) != f"gse264667_{name}_raw_singlecell_01.h5ad" for d in datasets):
            raise ValueError("Held-out data is not the documented GSE264667 input; supply a separate reviewed provenance")
    return {
        "teacher_name": "scGPT-human frozen encoder + VCell response head",
        "teacher_family": "scgpt", "model_revision": lock["revision"],
        "training_contexts": [], "excluded_contexts": ["HepG2", "Jurkat"],
        "source": "https://huggingface.co/wanglab/scGPT-human/tree/" + lock["revision"],
        "license": "Upstream scGPT code MIT; author-released weights, see source repository",
        "prediction_space": "prepared_log1p_delta",
        "declared_no_holdout_perturbations": True,
        "evidence_level": "published-source and dataset-release chronology; not row-level proof",
        "audit_notes": "Whole-human pretraining is reported as CELLxGENE Census 2023-05-15. "
            "The locked args.json identifies the May 2023 human pretraining run, with load_model=null. "
            "GSE264667 (HepG2/Jurkat) was submitted 2024-04-23 and public 2024-05-05. "
            "These records support exclusion of this held-out perturbation dataset from the reported "
            "pretraining corpus. We cannot independently reconstruct every pretraining row or rule "
            "out unpublished exposure. Context exclusions refer to perturbation training, not a "
            "claim that all related normal tissue was absent. Head fitting uses only train labels; "
            "validation selects its epoch. Test labels are never used for fitting or selection.",
        "evidence_urls": ["https://www.nature.com/articles/s41592-024-02201-0",
                          "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667"],
        "pretraining_corpus": "CELLxGENE Census 2023-05-15 whole human",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", default=os.environ.get("VCELL_WORK", "/root/autodl-tmp/vcell-work"))
    p.add_argument("--data")
    p.add_argument("--name", default="scgpt_01")
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--resume", action="store_true", help="Reuse completed stages; resume Qwen adapters")
    p.add_argument("--teacher-only", action="store_true", help="Write the Qwen config without starting the student")
    p.add_argument("--full-run", action="store_true", help="All train/validation groups, 10 student epochs")
    args = p.parse_args(argv)
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        p.error("--name must contain only letters, numbers, underscores or hyphens")
    work = Path(args.work_dir).resolve()
    data_dir = Path(args.data).resolve() if args.data else work / "prepared/real_min10_01"
    root = work / "teachers" / args.name
    torch.set_num_threads(8)
    data = load_prepared(data_dir)
    lock = json.loads(LOCK.read_text())
    provenance = benchmark_provenance(data, lock)
    plan = {"data_dir": str(data_dir), "data_fingerprint": data["audit"]["fingerprint"],
            "lock": lock, "device": args.device, "full_run": args.full_run,
            "source_fingerprint": source_fingerprint(), "launcher_sha256": file_sha256(__file__),
            "provenance": provenance}
    manifest = root / "plan.json"
    if root.exists() and any(root.iterdir()):
        if not args.resume or not manifest.is_file() or json.loads(manifest.read_text()) != plan:
            raise ValueError("Existing teacher run: use --resume with identical settings, or a new --name")
    root.mkdir(parents=True, exist_ok=True)
    write_json(manifest, plan)
    model_dir = prepare(work / "models/scGPT-human", args.endpoint, lock=lock, label="SCGPT")
    model = {"implementation": "vcell_frozen_scgpt_v1", "seed": 0, "max_input_genes": 1200}
    for key, name in (("checkpoint", "best_model.pt"), ("args", "args.json"), ("vocab", "vocab.json")):
        model[key] = str(model_dir / name)
        model[key + "_sha256"] = lock["files"][name]["sha256"]
    provenance_file = root / "provenance.json"
    write_json(provenance_file, provenance)
    feature_cfg = {"family": "scgpt", "data_dir": str(data_dir), "output": str(root / "features.npz"),
                   "provenance": str(provenance_file), "device": args.device, "seed": 0,
                   "max_control_cells": 8, "gene_symbol_key": "gene_name", "model": model}
    write_json(root / "export_config.json", feature_cfg)
    features = Path(feature_cfg["output"])
    if features.exists():
        sidecar = json.loads(features.with_suffix(".json").read_text())
        if sidecar.get("adapter_config") != feature_cfg or sidecar.get("feature_file_sha256") != file_sha256(features):
            raise ValueError("Saved scGPT features changed; use a new run name")
        print("SCGPT FEATURES VERIFIED: reusing completed extraction", flush=True)
    else:
        export_native_teacher(feature_cfg)
    head_cfg = {"data_dir": str(data_dir), "features": str(features), "output_dir": str(root / "head"),
                "device": args.device, "seed": 0, "hidden": 128, "epochs": 20, "patience": 5,
                "batch_size": 64, "learning_rate": .001}
    write_json(root / "head_config.json", head_cfg)
    cache = root / "head/predictions.npz"
    if cache.exists():
        _, saved = load_cache(cache, data)
        if saved.get("head_config") != head_cfg or saved.get("head_sha256") != file_sha256(root / "head/head.pt"):
            raise ValueError("Saved teacher head changed; use a new run name")
    else:
        fit_feature_teacher(head_cfg)
    predictions, _ = load_cache(cache, data)
    val = data["splits"]["val"]
    compared = {**simple_baselines(data), "teacher/scgpt": predictions}
    records = [{"model": name, "split": "val", "mse_delta": mse(value[val], data["delta"][val],
                row_weights(data["meta"].iloc[val])), "n_rows": len(val)} for name, value in compared.items()]
    pd.DataFrame(records).sort_values("mse_delta").to_csv(root / "teacher_summary.csv", index=False)
    print(f"SCGPT TEACHER COMPLETE: {cache}", flush=True)
    cfg = yaml.safe_load((REPO / "configs/qwen_smoke.yaml").read_text())
    cfg.update(data_dir=str(data_dir), output_dir=str(work / "runs" / ("qwen_" + args.name)),
               device=args.device, arms=["supervised", "scgpt"], teachers={"scgpt": str(cache)})
    cfg["model"].update(model_id=str(work / "models/Qwen3-0.6B-Base"), revision=None, local_files_only=True)
    if args.full_run:
        cfg.update(epochs=10, patience=3, max_train_rows=None, max_val_rows=None, log_every=10)
    if args.device == "cpu":
        cfg["model"]["dtype"] = "float32"
    write_json(root / "qwen_config.json", cfg)
    print(f"QWEN COMPARISON CONFIG: {root / 'qwen_config.json'}", flush=True)
    if not args.teacher_only:
        # Existing Qwen files must pass the same locked verification as the first smoke run.
        prepare(work / "models/Qwen3-0.6B-Base", args.endpoint)
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from vcell.qwen import run_qwen
        dest = Path(cfg["output_dir"])
        run_qwen(cfg, resume=args.resume and dest.exists() and any(dest.iterdir()))
    print("SCGPT PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
