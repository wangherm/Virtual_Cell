"""Four pretrained teachers, two independent full-data Qwen students, one selection.

The students share architecture and targets, but use distinct random seeds.
Selection uses validation only. Neither student learns from the other student.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from .data import load_prepared
from .qwen import backbone_identity, teacher_names, teacher_targets, validate_config
from .selection import mse, row_weights
from .train import simple_baselines, source_fingerprint
from .utils import file_sha256, write_json


def preflight(cfg):
    base = cfg["student"]
    validate_config(base)
    if base["arms"] != ["all"] or base.get("max_train_rows") is not None or base.get("max_val_rows") is not None:
        raise ValueError("421 requires arms: [all] and all prepared train/validation rows")
    if base["teacher_mix"] != "equal":
        raise ValueError("First 421 experiment requires equal teacher weights, fixed before selection")
    if len(teacher_names(base)) != 4 or set(base["teachers"]) != set(teacher_names(base)):
        raise ValueError("421 needs exactly the four named real teacher caches")
    seeds = cfg["student_seeds"]
    if len(seeds) != 2 or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != 2:
        raise ValueError("Specify exactly two distinct nonnegative student seeds")
    if cfg.get("parallel_students", 2) not in (1, 2):
        raise ValueError("parallel_students must be 1 or 2")
    data = load_prepared(base["data_dir"])
    targets, provenance, hashes, weights = teacher_targets(base, data, data["splits"]["val"])
    identity = backbone_identity(base["model"])
    if base["model"]["dtype"] == "bfloat16" and (
        base["device"] != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("bfloat16 students require supported CUDA; use float32 for CPU checks")
    facts = {"config": cfg, "data_fingerprint": data["audit"]["fingerprint"],
             "source_fingerprint": source_fingerprint(), "teacher_cache_hashes": hashes,
             "teacher_sources": provenance, "teacher_weights": weights,
             "teacher_order": list(teacher_names(base)), "backbone_identity": identity,
             "n_train": len(data["splits"]["train"]), "n_val": len(data["splits"]["val"]),
             "test_evaluated": False}
    facts["fingerprint"] = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    return data, targets, facts


def select_final(root, data, targets, student_configs, stamp):
    root = Path(root)
    val = data["splits"]["val"]
    rows = data["meta"].iloc[val].row_id.to_numpy(dtype="U")
    weights = row_weights(data["meta"].iloc[val])
    records, candidates = [], []
    for name, values in {**simple_baselines(data), **{"teacher/" + k: v for k, v in targets.items()}}.items():
        records.append({"model": name, "split": "val", "mse_delta": mse(values[val], data["delta"][val], weights),
                        "n_rows": len(val)})
    for name, config in student_configs.items():
        folder = Path(config["output_dir"])
        if not (folder / "COMPLETE.json").is_file():
            raise ValueError(f"{name} has not completed; refusing to select from an incomplete pair")
        complete = json.loads((folder / "COMPLETE.json").read_text())
        manifest = json.loads((folder / "run_manifest.json").read_text())
        if manifest["config"] != config or manifest["data_fingerprint"] != data["audit"]["fingerprint"]:
            raise ValueError(f"{name}: student config/data mismatch")
        if complete["run_fingerprint"] != manifest["run_fingerprint"] or complete["test_evaluated"]:
            raise ValueError(f"{name}: completion identity or test-isolation mismatch")
        if manifest["train_rows"] != data["splits"]["train"].tolist() or manifest["val_rows"] != val.tolist():
            raise ValueError("Both students must use the same complete train/validation splits")
        checkpoint = folder / "all/best.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["run_fingerprint"] != manifest["run_fingerprint"]:
            raise ValueError(f"{name}: selected checkpoint does not belong to this run")
        with np.load(folder / "all/validation_predictions.npz", allow_pickle=False) as f:
            if not np.array_equal(f["row_ids"], rows) or not np.array_equal(f["genes"], data["genes"]):
                raise ValueError("Student validation prediction IDs differ")
            if str(f["data_fingerprint"].item()) != data["audit"]["fingerprint"]:
                raise ValueError("Student prediction data fingerprint mismatch")
            pred = f["delta"].astype(np.float32)
        if pred.shape != data["delta"][val].shape or not np.isfinite(pred).all():
            raise ValueError("Invalid student validation predictions")
        score = mse(pred, data["delta"][val], weights)
        recorded = json.loads((folder / "all/validation.json").read_text())["mse_delta"]
        if not np.isclose(score, recorded, rtol=1e-6, atol=1e-9):
            raise ValueError("Recomputed validation score differs from student report")
        record = {"model": name, "split": "val", "mse_delta": score, "n_rows": len(val), "seed": config["seed"]}
        records.append(record)
        candidates.append({**record, "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)})
    winner = min(candidates, key=lambda x: (x["mse_delta"], x["model"]))
    baseline = next(r["mse_delta"] for r in records if r["model"] == "mean_transfer")
    selection = {"run_fingerprint": stamp, "criterion": "minimum validation mse_delta; exact ties choose student_a",
                 "selected": winner, "candidates": candidates, "mean_transfer_val_mse": baseline,
                 "beats_mean_transfer": winner["mse_delta"] < baseline, "test_evaluated": False,
                 "allow_unverified_teachers": any(c.get("allow_unverified_teachers") is True for c in student_configs.values()),
                 "note": "Best of these two students, not evidence of statistical significance or baseline superiority. "
                         "When unverified teachers are allowed, pretraining overlap is unknown and these are exploratory scores."}
    final = root / "final"
    final.mkdir(exist_ok=True)
    # This is one chosen model, never a weight average or a two-student ensemble.
    destination = final / "student.pt"
    if destination.exists() and file_sha256(destination) != winner["checkpoint_sha256"]:
        raise ValueError("Existing final model differs; use a new run directory")
    if not destination.exists():
        tmp = final / "student.pt.part"
        shutil.copyfile(winner["checkpoint"], tmp)
        if file_sha256(tmp) != winner["checkpoint_sha256"]:
            raise ValueError("Final checkpoint copy checksum mismatch")
        tmp.replace(destination)
    write_json(final / "selection.json", selection)
    saved = torch.load(winner["checkpoint"], map_location="cpu", weights_only=True)
    write_json(final / "backbone.json", {"model_spec": student_configs[winner["model"]]["model"],
                                        "backbone_identity": saved["backbone_identity"]})
    (final / "README.md").write_text(
        f"# Selected Qwen student\n\nSelected: {winner['model']} (seed {winner['seed']}).\n"
        f"Validation MSE: {winner['mse_delta']:.9f}. Test results remain sealed.\n\n"
        "`student.pt` contains the LoRA adapter and numerical input/output layers. Keep the "
        "verified Qwen backbone at the path recorded in `backbone.json`. It is required for loading.\n"
        "This exploratory run does not establish absence of pretraining overlap. "
        "The exported student is subject to upstream model terms, including State's non-commercial "
        "restrictions on distilled derivatives. See the included licenses and THIRD_PARTY_NOTICES.md.\n\n"
        "State citation: Adduri, A. et al. (2025). Predicting cellular responses to perturbation "
        "across diverse contexts with State. https://doi.org/10.1101/2025.06.26.661135\n\n"
        "Use `vcell qwen-predict --checkpoint /absolute/path/to/final/student.pt "
        "--data /absolute/path/to/prepared --output /absolute/new/predictions.npz --device cuda`.\n",
        encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    if (repo / "licenses").is_dir():
        shutil.copytree(repo / "licenses", final / "licenses", dirs_exist_ok=True)
        shutil.copyfile(repo / "THIRD_PARTY_NOTICES.md", final / "THIRD_PARTY_NOTICES.md")
    pd.DataFrame(records).sort_values("mse_delta").to_csv(root / "comparison.csv", index=False)
    write_json(root / "COMPLETE.json", selection)
    print(f"421 COMPLETE: selected={winner['model']} val_mse_delta={winner['mse_delta']:.7f} "
          f"beats_mean_transfer={selection['beats_mean_transfer']} final={destination}", flush=True)
    return final


def run_421(cfg, resume=False):
    data, targets, facts = preflight(cfg)  # All four teachers verified before creating student outputs.
    root = Path(cfg["output_dir"]).resolve()
    manifest = root / "run_manifest.json"
    if root.exists() and any(root.iterdir()):
        if not resume or not manifest.is_file() or json.loads(manifest.read_text())["fingerprint"] != facts["fingerprint"]:
            raise ValueError("Existing 421 run: resume identical inputs or choose a fresh output directory")
    root.mkdir(parents=True, exist_ok=True)
    write_json(manifest, facts)
    (root / "logs").mkdir(exist_ok=True)
    configs = {}
    for name, seed in zip(("student_a", "student_b"), cfg["student_seeds"]):
        spec = copy.deepcopy(cfg["student"])
        spec.update(seed=seed, output_dir=str(root / name))
        configs[name] = spec
        write_json(root / (name + ".json"), spec)
    print(f"421 START: train_groups={facts['n_train']} val_groups={facts['n_val']} "
          f"teachers={facts['teacher_order']} seeds={cfg['student_seeds']}", flush=True)
    active, finished = {}, {}
    pending = list(configs)
    try:
        last_status = 0
        while pending or active:
            while pending and len(active) < cfg.get("parallel_students", 2):
                name = pending.pop(0)
                log = (root / "logs" / (name + ".log")).open("a", encoding="utf-8")
                cmd = [sys.executable, "-u", "-m", "vcell", "qwen-run", "--config", str(root / (name + ".json"))]
                folder = Path(configs[name]["output_dir"])
                if resume and folder.exists() and any(folder.iterdir()):
                    cmd.append("--resume")
                env = {**os.environ, "PYTHONUNBUFFERED": "1",
                       "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
                try:
                    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                except BaseException:
                    log.close()
                    raise
                active[name] = process, log
                print(f"421 STUDENT START: {name} pid={process.pid} log={log.name}", flush=True)
            for name, (process, log) in list(active.items()):
                if process.poll() is not None:
                    finished[name] = process.returncode
                    log.close()
                    del active[name]
                    print(f"421 STUDENT EXIT: {name} code={process.returncode}", flush=True)
            if time.monotonic() - last_status >= 30:
                status = {name: ("running" if name in active else "pending" if name in pending else
                                 "complete" if finished[name] == 0 else "failed") for name in configs}
                write_json(root / "progress.json", status)
                print(f"421 STATUS: {status}", flush=True)
                last_status = time.monotonic()
            if active:
                time.sleep(1)
    finally:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
    if any(finished.get(name) != 0 for name in configs):
        write_json(root / "progress.json", {name: "complete" if finished.get(name) == 0 else "failed" for name in configs})
        raise RuntimeError(f"A student failed; inspect {root / 'logs'}. No final model was selected.")
    result = select_final(root, data, targets, configs, facts["fingerprint"])
    write_json(root / "progress.json", {name: "complete" for name in configs})
    return result
