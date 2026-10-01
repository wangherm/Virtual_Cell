"""One launch: controlled Qwen expansion, response basis, calibration and native reviewers."""
import argparse
import copy
import io
from collections import deque
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import torch

from vcell.data import load_prepared
from vcell.expansion import prepare_expansion
from vcell.qwen import backbone_identity
from vcell.round7 import internal_partition, prepare_native_inputs, internal_calibration, native_review, native_ensemble_audit
from vcell.specialization import make_modules, compare_validation, coverage_audit
from vcell.train import simple_baselines, source_fingerprint
from vcell.utils import read_config, file_sha256, write_json

REPO = Path(__file__).resolve().parents[1]


def run_jobs(jobs, root, gpu_slots, resume, label="ROUND7"):
    """Bounded independent processes; each owns its RNG, optimizer and log."""
    active, finished = {}, {}
    pending = list(jobs)
    (root/"logs").mkdir(exist_ok=True)
    try:
        last_status = 0
        while pending or active:
            for name in list(pending):
                job = jobs[name]
                resource = job["resource"]
                limit = gpu_slots if resource == "gpu" else 2
                if sum(jobs[n]["resource"] == resource for n in active) >= limit:
                    continue
                if resume and Path(job["complete"]).exists():
                    finished[name] = 0
                    pending.remove(name)
                    continue
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                    "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                    "MPLBACKEND": "Agg", "PYTHONPATH": str(REPO/"src")}
                if resource == "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = ""
                log = (root/"logs"/(name+".log")).open("a", encoding="utf-8")
                try:
                    proc = subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env,
                                            start_new_session=os.name != "nt")
                except BaseException:
                    log.close()
                    raise
                active[name] = proc, log
                pending.remove(name)
                print(f"{label} START {name} pid={proc.pid} log={log.name}", flush=True)
            for name, (proc, log) in list(active.items()):
                if proc.poll() is not None:
                    code = proc.returncode
                    if code == 0 and not Path(jobs[name]["complete"]).exists():
                        code = 99
                    finished[name] = code
                    log.close()
                    del active[name]
                    print(f"{label} EXIT {name} code={code}", flush=True)
            if time.monotonic()-last_status >= 30 or not active:
                status = {n: {"state": "running" if n in active else "pending" if n in pending else
                    "complete" if finished[n] == 0 else "failed", "exit_code": finished.get(n),
                    "log": str(root/"logs"/(n+".log"))} for n in jobs}
                write_json(root/"progress.json", {"jobs": status, "test_evaluated": False})
                print(f"{label} STATUS completed={sum(v==0 for v in finished.values())}/{len(jobs)} active={list(active)}", flush=True)
                last_status = time.monotonic()
            if active:
                time.sleep(1)
    finally:
        for proc, log in active.values():
            if proc.poll() is None:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait()
            log.close()
    return finished


def package(root):
    with tarfile.open(root/"round7_review.tar.gz", "w:gz") as tar:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in {".json", ".csv"}:
                tar.add(p, arcname=str(p.relative_to(root)))
            elif p.is_file() and p.suffix == ".log":
                with p.open(encoding="utf-8", errors="replace") as stream:
                    tail = "".join(deque(stream, maxlen=120)).encode("utf-8")
                info = tarfile.TarInfo(p.relative_to(root).as_posix())
                info.size = len(tail)
                tar.addfile(info, io.BytesIO(tail))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", default=os.environ.get("VCELL_WORK", "/root/autodl-tmp/vcell-work"))
    p.add_argument("--name", default="round7_01")
    p.add_argument("--data")
    p.add_argument("--challenge-data")
    p.add_argument("--model-dir")
    p.add_argument("--student-config", default=str(REPO/"configs/qwen_three_teachers.yaml"))
    p.add_argument("--seeds", nargs="+", type=int, default=[17,29,43])
    p.add_argument("--max-steps", type=int, default=4860)
    p.add_argument("--eval-every-steps", type=int, default=243)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--parallel-students", type=int, choices=[1,2], default=2)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--traditional", nargs="*", choices=["celloracle", "sctenifold"], default=["celloracle", "sctenifold"])
    p.add_argument("--native-targets", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args(argv)
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        p.error("Use a simple run name")
    if min(args.max_steps, args.eval_every_steps, args.batch_size, args.gradient_accumulation, args.native_targets) < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error("Invalid training counts or seeds")
    torch.set_num_threads(2)
    work = Path(args.work_dir).resolve()
    original_dir = Path(args.data or work/"prepared/real_min10_01").resolve()
    challenge_dir = Path(args.challenge_data or work/"prepared/vcc2025_c_01").resolve()
    original, challenge = load_prepared(original_dir), load_prepared(challenge_dir)
    root = work/"runs"/args.name
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("CUDA/BF16 unavailable")
    cfg = read_config(args.student_config)
    cfg.update(device=args.device, num_threads=2, arms=["supervised"], teachers={}, teacher_mix="equal",
        max_train_rows=None, max_val_rows=None, batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
        max_steps=args.max_steps, eval_every_steps=args.eval_every_steps, loss_weighting="uniform",
        allow_unverified_teachers=False, log_every=50)
    cfg["model"].update(model_id=str(Path(args.model_dir or work/"models/Qwen3-0.6B-Base").resolve()),
        revision=None, local_files_only=True, dtype="bfloat16" if args.device=="cuda" else "float32",
        gradient_checkpointing=args.gradient_checkpointing)
    for key in ("functional", "background_interaction", "response_rank"):
        cfg["model"].pop(key, None)
    for key in ("reliability", "kd_mask", "distillation", "epoch_train_rows", "internal_split"):
        cfg.pop(key, None)
    plan = {"source_fingerprint": source_fingerprint(), "scripts": {n:file_sha256(REPO/"scripts"/n) for n in
        ("run_round7.py", "run_traditional_teacher.py", "run_traditional_autodl.sh")},
        "original_fingerprint": original["audit"]["fingerprint"], "challenge_fingerprint": challenge["audit"]["fingerprint"],
        "base_config": cfg, "seeds": args.seeds, "traditional": args.traditional, "native_targets": args.native_targets,
        "backbone_identity": backbone_identity(cfg["model"]), "primary": "fixed final optimizer update",
        "low_rank_data": "plus_h1 predeclared; no validation data choice", "test_evaluated": False}
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root/"plan.json").exists() or json.loads((root/"plan.json").read_text()) != plan:
            raise ValueError("Resume identical Round 7 code/data/config or choose a new name")
    if shutil.disk_usage(work).free < 15*1024**3:
        raise ValueError("Round 7 reserves 15 GiB for students and isolated native environments")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"plan.json", plan)
    paths = prepare_expansion(original, challenge, root/"prepared")
    data = {k:load_prepared(v) for k,v in paths.items()}
    for name, d in data.items():
        coverage_audit(d, root/"coverage"/name)
    write_json(root/"implementation_audit.json", {"data_alignment": "load_prepared fingerprints and exact common panel checked",
        "delta": "perturbed mean log1p expression minus matched-control mean; native raw spot checks in native_inputs/audit.json",
        "background": "unchanged Round 6 reference coefficient=1, interaction output zero-initialized; not equivalent to plain at initialization",
        "initialization": "shared parameter hashes saved per job; actual equality checked after training",
        "normalization": "refit on each arm's allowed training rows; H1 changes statistics; not fitted on HepG2/Jurkat",
        "objective": "uniform rows, uniform genes in normalized delta space; differs from target-macro reported MSE",
        "steps": args.max_steps, "examples": args.max_steps*args.batch_size*args.gradient_accumulation,
        "schedule": "constant per-step LR, no early stopping; earlier curves are supplementary",
        "modules": "not used in Round 7; prior unequal cluster sizes are not an error by themselves",
        "legacy_teachers": "not refitted or reused as expression teachers; prior native preprocessing not newly certified",
        "test_evaluated": False})
    jobs, specs = {}, {}
    for seed in args.seeds:
        for dataset in paths:
            for background in (False, True):
                name = f"{'background' if background else 'plain'}_{dataset}_seed{seed}"
                current = copy.deepcopy(cfg)
                current.update(seed=seed, data_dir=str(paths[dataset]), output_dir=str(root/"students"/name))
                current["model"]["background_interaction"] = background
                specs[name] = current
        name = f"response32_plus_h1_seed{seed}"
        current = copy.deepcopy(specs[f"plain_plus_h1_seed{seed}"])
        current["output_dir"] = str(root/"students"/name)
        current["model"]["response_rank"] = 32
        specs[name] = current
    internal = copy.deepcopy(specs[f"plain_plus_h1_seed{args.seeds[0]}"])
    internal.update(internal_split=internal_partition(data["plus_h1"]), output_dir=str(root/"students/internal_calibration"))
    specs["internal_calibration"] = internal
    (root/"configs").mkdir(exist_ok=True)
    for name, current in specs.items():
        file = root/"configs"/(name+".json")
        write_json(file, current)
        cmd = [sys.executable, "-u", "-m", "vcell", "qwen-run", "--config", str(file)]
        if args.resume and Path(current["output_dir"]).exists():
            cmd.append("--resume")
        jobs[name] = {"cmd": cmd, "resource": "gpu", "complete": str(Path(current["output_dir"])/"COMPLETE.json")}
    write_json(root/"jobs.json", jobs)
    if args.plan_only:
        print(f"ROUND7 PLAN READY: {len(specs)} students, native={args.traditional}; not started", flush=True)
        return root
    if args.traditional:
        prepare_native_inputs(original, data["original_only"], root/"native_inputs")
        for family in args.traditional:
            out = root/"traditional"/family
            jobs[family] = {"resource": "cpu", "complete": str(out/"COMPLETE.json"),
                "cmd": ["bash", str(REPO/"scripts/run_traditional_autodl.sh"), family,
                    str(root/"native_inputs"), str(out), str(work), str(args.native_targets)]}
    write_json(root/"jobs.json", jobs)
    finished = run_jobs(jobs, root, args.parallel_students, args.resume)
    reference = data["original_only"]
    val = reference["splits"]["val"]
    membership, _ = make_modules(reference, min(8, len(reference["genes"])))
    predictions = {}
    for dataset, d in data.items():
        predictions.update({dataset+"/"+n:v[d["splits"]["val"]] for n,v in simple_baselines(d).items()})
    for name, current in specs.items():
        if name == "internal_calibration" or finished.get(name) != 0:
            continue
        with np.load(Path(current["output_dir"])/"supervised/fixed_predictions.npz") as f:
            if not np.array_equal(f["row_ids"], reference["meta"].iloc[val].row_id.to_numpy(dtype="U")) or not np.array_equal(f["genes"], reference["genes"]):
                raise ValueError("Round 7 result panel mismatch")
            predictions[name] = f["delta"]
    compare_validation(reference, predictions, membership, root)
    norms = [{"model": n, "prediction_rms": float(np.sqrt(np.mean(v*v))),
              "truth_rms": float(np.sqrt(np.mean(reference["delta"][val]**2)))} for n,v in predictions.items()]
    pd.DataFrame(norms).to_csv(root/"effect_norms.csv", index=False)
    if finished.get("internal_calibration") == 0:
        internal_calibration(data["plus_h1"], internal, root/"calibration")
    if args.traditional:
        native_review(reference, root/"native_inputs", root/"traditional", predictions, root)
        if finished.get("internal_calibration") == 0 and finished.get("celloracle") == 0:
            native_ensemble_audit(data["plus_h1"], internal, root/"native_inputs", root/"traditional", root/"calibration")
    initializations = {n:json.loads((Path(c["output_dir"])/"supervised/model_info.json").read_text())["shared_initial_sha256"]
        for n,c in specs.items() if (Path(c["output_dir"])/"supervised/model_info.json").exists()}
    write_json(root/"initialization_audit.json", {"hashes": initializations,
        "paired_shared_weights_match": all(len({h for n,h in initializations.items() if n.endswith(f"seed{s}")})<=1 for s in args.seeds),
        "excludes": "output head and optional background-specific parameters"})
    failed = [n for n in jobs if finished.get(n) != 0]
    status = {"failed_jobs": failed, "students_completed": sum(finished.get(n)==0 for n in specs),
        "native_requested": args.traditional, "primary": "fixed final update", "test_evaluated": False}
    write_json(root/("INCOMPLETE.json" if failed else "COMPLETE.json"), status)
    if not failed and (root/"INCOMPLETE.json").exists():
        (root/"INCOMPLETE.json").unlink()
    package(root)
    print(f"ROUND7 {'INCOMPLETE' if failed else 'COMPLETE'}: {root}; review={root/'round7_review.tar.gz'}", flush=True)
    if failed:
        raise RuntimeError(f"Jobs failed: {failed}; successful work retained. Fix dependency/OOM and use --resume.")
    return root


if __name__ == "__main__":
    main()
