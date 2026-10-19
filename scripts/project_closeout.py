"""Inventory local research runs and export explicitly selected research models.

Only the Python standard library is required. Source experiments are read-only;
this tool never runs a model, loads a pickle, calls an API, or selects a winner.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time


SKIP = {"workers", "private", "inputs", "input_cache", "contract", "contract2",
        "preflight", "provider_responses", "provider_tickets", ".git", "__pycache__"}
SUMMARY_KEYS = ("search_complete", "scientific_pass", "framework_pass",
                "task_coverage_complete", "agent_self_improvement_proven",
                "test_evaluated", "outer_evaluated", "HT29_evaluated",
                "independent_confirmation", "total_requests_including_gate",
                "total_worker_seconds")
CHECKPOINT_SUFFIXES = {".pt", ".ckpt", ".safetensors", ".pth"}
METRIC_FIELDS = ("model", "method", "arm", "task", "split", "phase", "scope",
                 "mse", "mse_delta", "mse_delta_mean", "strong_mse",
                 "specificity_gap", "amplitude_ratio", "n_rows", "n_genes",
                 "n_perturbations", "seed")
NUMERIC_FIELDS = set(METRIC_FIELDS[7:])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    # Provider responses and arbitrary configuration files are never read.
    if path.is_symlink():
        raise ValueError("Symbolic metadata files are not accepted: " + str(path))
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Metadata file exceeds 64 MiB: " + str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      parse_constant=lambda value: None)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(base, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected a confined relative path")
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symbolic paths are not exported")
    result = current.resolve()
    if not result.is_relative_to(base.resolve()) or not result.is_file():
        raise ValueError("File is missing or outside the permitted directory")
    return result


def files_under(root):
    # Do not follow directory links or enumerate large worker matrices/caches.
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if not path.is_symlink():
                yield path


def inspect_run(run, forced_unfinished=False):
    loaded, errors = {}, []
    for name in ("stage.json", "summary.json", "COMPLETE.json", "heartbeat.json",
                 "INCOMPLETE.json", "PAUSED.json", "campaign_plan.json"):
        path = run / name
        if path.exists():
            try:
                value = read_json(path)
                if not isinstance(value, dict):
                    raise ValueError("Expected an object")
                loaded[name] = value
            except (ValueError, OSError):
                errors.append(name)
    heartbeat = loaded.get("heartbeat.json", {})
    updated = heartbeat.get("updated_at")
    age = max(0, time.time() - updated) if isinstance(updated, (int, float)) else None
    fresh = heartbeat.get("state") == "running" and age is not None and age < 60
    marker = "COMPLETE.json" in loaded
    if forced_unfinished:
        state = "unfinished_user_declared"
    elif errors:
        state = "metadata_error"
    elif fresh:
        state = "running_heartbeat"
    elif marker and "INCOMPLETE.json" in loaded and (run / "INCOMPLETE.json").stat().st_mtime_ns > (run / "COMPLETE.json").stat().st_mtime_ns:
        state = "incomplete_after_completion_marker"
    elif marker and loaded.get("summary.json", {}).get("search_complete") is False:
        state = "conflicting_completion_metadata"
    elif marker:
        state = "completed_declared"
    elif "PAUSED.json" in loaded:
        state = "paused"
    elif "INCOMPLETE.json" in loaded:
        state = "incomplete"
    else:
        state = "unknown_no_completion_marker"
    summary = loaded.get("summary.json", {})
    stage = loaded.get("stage.json", {}).get("stage")
    task_counts = []
    for path in sorted((run / "tasks").glob("*.json")):
        try:
            queue = read_json(path)
            for row in queue.get("tasks", []):
                task_counts.append({"arm": path.stem, **{k: row.get(k) for k in
                                    ("task", "target", "distinct_new", "status")}})
        except (ValueError, OSError, AttributeError, TypeError):
            errors.append(path.relative_to(run).as_posix())
    if errors and state == "completed_declared":
        state = "metadata_error"
    return {"run": run.name, "state": state, "completion_marker": marker,
            "completion_integrity": "not_fully_verified",
            "fresh_running_heartbeat": fresh,
            "heartbeat_age_seconds": round(age, 1) if age is not None else None,
            "stage": stage if isinstance(stage, str) else None,
            "metadata_errors": errors,
            "historical_error_marker": "INCOMPLETE.json" in loaded,
            "tasks": task_counts,
            "summary": {key: summary[key] for key in SUMMARY_KEYS if key in summary},
            "model_export_allowed": state == "completed_declared",
            "active_process_check": "heartbeat_only; absence is not proof of a stopped process"}


def metric_inventory(run):
    result = []
    for name in ("comparison.csv", "comparison_macro.csv", "summary.csv"):
        path = run / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for index, row in enumerate(csv.DictReader(stream)):
                    if index >= 10000:
                        break
                    selected = {"run": run.name, "source_table": name,
                                "verification": "reported_only; not_recomputed"}
                    for key in METRIC_FIELDS:
                        value = row.get(key)
                        if value is None or value == "":
                            continue
                        if key in NUMERIC_FIELDS:
                            try:
                                number = float(value)
                                if math.isfinite(number):
                                    selected[key] = number
                            except (ValueError, TypeError):
                                pass
                        elif re.fullmatch(r"[A-Za-z0-9_./+ :@=-]{1,160}", value):
                            selected[key] = value
                    if any(key in selected for key in NUMERIC_FIELDS):
                        result.append(selected)
        except (OSError, UnicodeError, csv.Error):
            continue
    return result


def model_inventory(run, state):
    result = []
    for path in files_under(run):
        relative = path.relative_to(run)
        if path.suffix in CHECKPOINT_SUFFIXES:
            result.append({"run": run.name, "path": relative.as_posix(),
                           "kind": "checkpoint", "bytes": path.stat().st_size,
                           "sha256": None, "run_state": state,
                           "qualification": "not_selected; checkpoint contents not deserialized"})
        elif path.name == "assessment.json":
            try:
                record = read_json(path)
                module = path.parent / "module.py"
                if (record.get("status") == "valid" and module.is_file()
                        and not module.is_symlink()):
                    result.append({"run": run.name,
                                   "path": module.relative_to(run).as_posix(),
                                   "kind": "evolved_algorithm", "bytes": module.stat().st_size,
                                   "recorded_module_sha256": record.get("module_sha256"),
                                   "phase": record.get("identity", {}).get("phase"),
                                   "sha256": None, "run_state": state,
                                   "qualification": "research_candidate; no automatic promotion"})
            except (ValueError, OSError, AttributeError):
                continue
    return result


def code_revision(repo):
    try:
        def git(*args):
            return subprocess.check_output(["git", "-c", "safe.directory=" + str(repo),
                                            "-C", str(repo), *args], text=True,
                                           stderr=subprocess.DEVNULL, timeout=10).strip()
        return {"commit": git("rev-parse", "HEAD"),
                "working_tree_dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "working_tree_dirty": None}


def export_model(runs, selection, output, run_states, max_bytes):
    # Selection is explicit: RUN/path/to/best.pt or RUN/.../module.py.
    source = safe_file(runs, selection)
    relative = source.relative_to(runs)
    if len(relative.parts) < 2:
        raise ValueError("A model path must include its run directory")
    run_name = relative.parts[0]
    if run_states[run_name]["state"] != "completed_declared":
        raise ValueError("Model export requires a completed, inactive run: " + run_name)
    if source.suffix not in CHECKPOINT_SUFFIXES | {".py"}:
        raise ValueError("Only explicitly selected Python methods or checkpoints are exported")
    if set(relative.parts) & SKIP:
        raise ValueError("Private/worker/cache files are not exportable")
    before = source.stat()
    if before.st_size > max_bytes:
        raise ValueError("Model exceeds --max-model-mib; increase the explicit export cap")
    digest = sha256(source)
    run = runs / run_name
    marker = read_json(run / "COMPLETE.json")
    expected = marker.get("files", {}).get(source.relative_to(run).as_posix())
    if isinstance(expected, dict):
        expected = expected.get("sha256")
    if expected is not None and expected != digest:
        raise ValueError("Selected model conflicts with its run completion checksum")
    assessment_hash = None
    if source.name == "module.py" and (source.parent / "assessment.json").is_file():
        assessment_hash = read_json(source.parent / "assessment.json").get("module_sha256")
        if assessment_hash is not None and assessment_hash != digest:
            raise ValueError("Selected algorithm conflicts with its assessment checksum")
    destination = output / "selected_models" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(destination) != digest:
        raise ValueError("Model changed during export; retry after all writers stop")
    return {"run": run_name, "original_path": relative.as_posix(),
            "package_path": destination.relative_to(output).as_posix(),
            "bytes": after.st_size, "sha256": digest,
            "completion_checksum_matched": expected is not None,
            "original_run_completion_sha256": sha256(run / "COMPLETE.json"),
            "assessment_checksum_matched": assessment_hash is not None,
            "run_integrity": "selected_file_only; full run not verified",
            "reported_run_scientific_flags": run_states[run_name]["summary"],
            "independent_validation": "not_established_by_export",
            "deployment_status": "research_artifact; not a standalone deployment certification"}


def frozen_models(runs, name, states):
    """Export the existing source shortlist, without selecting on confirmation."""
    if name not in states or states[name]["state"] != "completed_declared":
        raise ValueError("Frozen export requires a completed run: " + name)
    run = runs / name
    chosen = {}
    for path in sorted(run.glob("arms/*/*/search_frozen.json")):
        safe_file(runs, path.relative_to(runs))
        value = read_json(path)
        if value.get("source_only") is not True:
            raise ValueError("Shortlist is not explicitly source-only: " + str(path))
        for record in value.get("finalists", []):
            if record.get("status") != "valid":
                raise ValueError("Invalid source finalist")
            module = Path(record["module"])
            if not module.is_absolute():
                raise ValueError("Expected the original absolute recorded module path")
            try:
                relative = module.relative_to(run)
            except ValueError:
                raise ValueError("Frozen module is outside its original run; preserve the original mount paths") from None
            source = safe_file(run, relative)
            digest = record.get("module_sha256")
            if not isinstance(digest, str) or digest != sha256(source):
                raise ValueError("Frozen source method checksum mismatch")
            chosen[source.relative_to(runs).as_posix()] = digest
    if not chosen:
        raise ValueError("No compatible frozen source finalists; use an explicit --model path")
    return chosen


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(output, runs, models, exports, revision, metrics):
    write_json(output / "runs.json", runs)
    write_json(output / "model_inventory.json", models)
    write_json(output / "selected_models.json", exports)
    write_csv(output / "reported_metrics.csv", metrics,
              ["run", "source_table", "verification", *METRIC_FIELDS])
    write_csv(output / "runs.csv", runs, ["run", "state", "stage", "completion_marker", "completion_integrity"])
    write_csv(output / "models.csv", models, ["run", "path", "kind", "bytes", "run_state", "phase", "qualification"])
    lines = ["# Virtual Cell project closeout snapshot", "",
             "This is a read-only inventory, not a new experiment or an independent result verification.",
             "The project is paused. Round27 remains unfinished when explicitly declared by the operator.",
             "Completion markers describe execution; scientific success is reported separately.", "",
             f"Source revision: `{revision['commit']}`; dirty checkout: `{revision['working_tree_dirty']}`.",
             f"Runs inventoried: {len(runs)}. Model artifacts found: {len(models)}. Explicit exports: {len(exports)}.", "",
             "| Run | Execution state | Scientific pass (reported) | Test evaluated (reported) |",
             "| --- | --- | --- | --- |"]
    for run in runs:
        name = run["run"].replace("|", "\\|").replace("\n", " ")
        summary = run["summary"]
        lines.append(f"| {name} | {run['state']} | {summary.get('scientific_pass', 'unknown')} | {summary.get('test_evaluated', 'unknown')} |")
    lines += ["", "## Interpretation", "",
              "No automatic best-model selection is made across tasks, gene panels, seeds or rounds.",
              "Repeat uses measured Rep1 support to predict Rep2; it is not zero-shot perturbation prediction.",
              "Historically inspected confirmation data are development evidence, not a new independent test.",
              "Missing completion metadata mean unknown, not failed or completed. A stale error marker may survive a resume.",
              "Round27 candidates must not be presented as a completed final experiment.", "",
              "## Resume and backup", "",
              "Keep the entire original runs, prepared inputs, raw data, model caches and parent campaigns.",
              "Also retain the original repository (including .git), environments, local configuration and dependency versions privately.",
              "Absolute parent/input paths occur in provenance records. Preserve paths or mount the backup at the same paths.",
              "This light package is NOT a resume backup. No experiment directory was modified or deleted.",
              "Check active processes before a separate filesystem backup; this inventory does not stop them.", "",
              "## Deliverables", "",
              "See runs.json, models.csv, reported_metrics.csv, selected_models.json and MODEL_CARD.md.",
              "Metrics are copied from whitelisted aggregate columns, not recomputed or compared across protocols.",
              "Raw matrices, provider replies, API credentials, logs and third-party base weights are excluded.",
              "Selected user-requested files may contain generated code or checkpoint metadata; review before public upload."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    card = """# Research model artifact card

## Status and intended use

These are explicitly selected research artifacts, not automatically selected best models.
Round27 is unfinished if so declared in runs.json. No independent validation is
established by packaging. Do not use this software for clinical decisions.

## Task and output

The project predicts group-mean gene-expression deltas. Repeat uses measured
Rep1 support; Transfer concerns source-cell-line holdouts. Output gene order,
normalization, context/target encodings and source folds must match the original
experiment. The older 1,834-gene and newer 1,809-gene panels are not interchangeable.

## Files and dependencies

selected_models.json lists the original path, SHA-256 and export verification
scope for each file. An empty list means no server model has been exported.
Python evolved methods implement a fit/predict algorithm; they generally refit
using original public source templates and require the original evaluator/runtime.
They are not standalone pretrained weights. Checkpoints may contain heads or
adapters and require the original architecture, configuration, preprocessing,
gene panel and separately licensed base model. They are not deserialized here.
Use the original documented inference entry point in the matching source revision.
Retain folds.json, run_plan.json, baseline_selection.json, templates and recorded
parent runs privately for replay; this light package does not redistribute them.

## Evaluation and limitations

Read the original run comparison tables and protocol for numerical results.
Packaging neither recomputes scores nor promotes a candidate. A successful
agent loop does not establish predictive improvement. Cross-cell generalization
and independent external validation have not been established by this export.

## Licensing

Repository code is MIT; third-party data, priors and foundation models retain
their own terms. Check THIRD_PARTY_NOTICES.md and the selected model's complete
lineage before redistributing weights. No blanket MIT claim is made for weights.
"""
    (output / "MODEL_CARD.md").write_text(card, encoding="utf-8")


def build(work, name, unfinished=(), selections=(), max_model_mib=1024, frozen_runs=()):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
        raise ValueError("Use a simple new output name")
    work = Path(work).resolve()
    runs_root = work / "runs"
    if not runs_root.is_dir() or runs_root.is_symlink():
        raise ValueError("Expected a real work/runs directory")
    closeout = work / "closeout"
    if closeout.is_symlink():
        raise ValueError("Closeout output directory cannot be a symbolic link")
    output = closeout / name
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "progress.json"
    write_json(progress, {"stage": "inventory", "complete": False})
    try:
        run_rows, models, metrics = [], [], []
        for run in sorted(runs_root.iterdir()):
            if not run.is_dir() or run.is_symlink() or run.name.startswith("."):
                continue
            print("INVENTORY:", run.name, flush=True)
            write_json(progress, {"stage": "inventory", "run": run.name, "complete": False})
            row = inspect_run(run, run.name in unfinished)
            run_rows.append(row)
            models.extend(model_inventory(run, row["state"]))
            metrics.extend(metric_inventory(run))
        states = {r["run"]: r for r in run_rows}
        frozen = {}
        for run_name in frozen_runs:
            frozen.update(frozen_models(runs_root, run_name, states))
        exports = []
        for selected in dict.fromkeys([*selections, *frozen]):
            print("EXPORT:", selected, flush=True)
            write_json(progress, {"stage": "explicit model export", "complete": False})
            item = export_model(runs_root, selected, output, states,
                                max_model_mib * 1024 * 1024)
            if selected in frozen:
                if item["sha256"] != frozen[selected]:
                    raise ValueError("Frozen method changed during export")
                item["selection_origin"] = "existing_source_only_frozen_shortlist; not a new winner selection"
            else:
                item["selection_origin"] = "explicit_operator_path"
            exports.append(item)
        revision = code_revision(Path(__file__).resolve().parents[1])
        report(output, run_rows, models, exports, revision, metrics)
        disk = shutil.disk_usage(work)
        write_json(output / "storage.json", {
            "disk_bytes": {"total": disk.total, "used": disk.used, "free": disk.free},
            "preserve_directories": {name: (work / name).exists() for name in
                                     ("runs", "raw", "prepared", "models", "envs", "logs",
                                      "agent_memory", "agent_control", "predictions")},
            "individual_directory_sizes": "not_computed; no full data-disk scan"})
        write_json(output / "snapshot.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "source_code": revision,
            "declared_unfinished_runs": list(unfinished),
            "requested_source_frozen_runs": list(frozen_runs),
            "missing_declared_runs": sorted(set(unfinished) - set(states)),
            "source_work": str(work), "source_mutated": False,
            "api_requests": 0, "full_run_integrity_verified": False,
            "is_resume_backup": False, "automatic_model_selection": False})
        write_json(progress, {"stage": "packaging", "complete": False})
        payloads = [p for p in sorted(output.rglob("*")) if p.is_file() and p != progress]
        manifest = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in payloads}
        write_json(output / "ARCHIVE_MANIFEST.json", {"files": manifest})
        archive = output / "project_closeout.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for path in payloads + [output / "ARCHIVE_MANIFEST.json"]:
                stream.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
        digest = sha256(archive)
        (output / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        write_json(progress, {"stage": "complete", "complete": True, "archive": str(archive),
                              "sha256": digest, "bytes": archive.stat().st_size})
        print("CLOSEOUT COMPLETE:", archive, flush=True)
        return output
    except Exception as error:
        # Error messages are not copied to the public package: they may contain local secrets.
        write_json(progress, {"stage": "failed", "complete": False,
                              "error_type": type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/root/autodl-tmp/vcell-work")
    parser.add_argument("--name", default="project_closeout_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--unfinished-run", action="append", default=[],
                        help="Explicitly exclude an unfinished run from completion/export claims")
    parser.add_argument("--model", action="append", default=[],
                        help="Explicit RUN/relative/model.py or checkpoint path; repeat for several files")
    parser.add_argument("--frozen-run", action="append", default=[],
                        help="Export existing source-only frozen algorithm finalists from a completed run")
    parser.add_argument("--max-model-mib", type=int, default=1024)
    args = parser.parse_args()
    if args.max_model_mib <= 0:
        parser.error("--max-model-mib must be positive")
    build(args.work_dir, args.name, args.unfinished_run, args.model, args.max_model_mib, args.frozen_run)


if __name__ == "__main__":
    main()
