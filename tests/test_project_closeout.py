"""Closeout preserves experiments and does not promote incomplete models."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import pytest


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("project_closeout", REPO / "scripts/project_closeout.py")
closeout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closeout)


def put(path, value):
    closeout.write_json(path, value)


def completed(work, name="round26_critic_02"):
    run = work / "runs" / name
    put(run / "summary.json", {"search_complete": True, "scientific_pass": False,
                              "test_evaluated": False, "secret": "DO_NOT_COPY"})
    put(run / "COMPLETE.json", {"files": {}})
    return run


def test_declared_unfinished_overrides_complete_and_blocks_export(tmp_path):
    run = completed(tmp_path, "round27_agent_01")
    (run / "best.pt").write_bytes(b"opaque checkpoint")
    row = closeout.inspect_run(run, forced_unfinished=True)
    assert row["state"] == "unfinished_user_declared"
    assert not row["model_export_allowed"]
    with pytest.raises(ValueError, match="completed"):
        closeout.build(tmp_path, "declared", [run.name], [run.name + "/best.pt"])
    assert json.loads((tmp_path / "closeout/declared/progress.json").read_text())["stage"] == "failed"


def test_inventory_is_light_and_source_is_unchanged(tmp_path):
    run = completed(tmp_path)
    (run / "best.pt").write_bytes(b"opaque pickle; must not execute")
    put(run / "workers/private/assessment.json", {"secret": "DO_NOT_COPY"})
    (run / ".env").write_text("DEEPSEEK_API_KEY=DO_NOT_COPY")
    (run / "secret.log").write_text("DO_NOT_COPY")
    (run / "comparison.csv").write_text("model,mse,secret\nfixed,0.04,DO_NOT_COPY\n")
    put(run / "tasks/meta.json", {"tasks": [{"task": "repeat", "target": 5,
                                           "distinct_new": 2, "status": "pending"}]})
    put(tmp_path / "runs/round27_agent_01/stage.json", {"stage": "search"})
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file()}
    out = closeout.build(tmp_path, "inventory", ["round27_agent_01"])
    assert all((Path(p).read_bytes(), Path(p).stat().st_mtime_ns) == value for p, value in before.items())
    assert json.loads((out / "selected_models.json").read_text()) == []
    models = json.loads((out / "model_inventory.json").read_text())
    assert len(models) == 1 and models[0]["sha256"] is None
    assert "0.04" in (out / "reported_metrics.csv").read_text()
    assert json.loads((out / "runs.json").read_text())[0]["tasks"][0]["distinct_new"] == 2
    with tarfile.open(out / "project_closeout.tar.gz") as archive:
        names = archive.getnames()
        assert not any(n.endswith((".pt", ".log", ".env")) for n in names)
        for member in archive:
            assert b"DO_NOT_COPY" not in archive.extractfile(member).read()
        manifest = json.load(archive.extractfile("ARCHIVE_MANIFEST.json"))["files"]
        for path, entry in manifest.items():
            data = archive.extractfile(path).read()
            assert entry == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_selected_file_checksum_and_integrity_scope(tmp_path):
    run = completed(tmp_path)
    model = run / "head/best.pt"
    model.parent.mkdir()
    model.write_bytes(b"not a real serialized checkpoint")
    digest = closeout.sha256(model)
    put(run / "COMPLETE.json", {"files": {"head/best.pt": digest}})
    out = closeout.build(tmp_path, "models", selections=[run.name + "/head/best.pt"])
    exported = json.loads((out / "selected_models.json").read_text())[0]
    assert exported["completion_checksum_matched"]
    assert exported["sha256"] == digest
    assert exported["independent_validation"] == "not_established_by_export"
    assert (out / exported["package_path"]).read_bytes() == model.read_bytes()
    assert json.loads((out / "snapshot.json").read_text())["api_requests"] == 0


def test_checksum_mismatch_refuses_public_package(tmp_path):
    run = completed(tmp_path)
    (run / "best.pt").write_bytes(b"changed")
    put(run / "COMPLETE.json", {"files": {"best.pt": "0" * 64}})
    with pytest.raises(ValueError, match="checksum"):
        closeout.build(tmp_path, "corrupt", selections=[run.name + "/best.pt"])
    assert not (tmp_path / "closeout/corrupt/project_closeout.tar.gz").exists()


def test_algorithm_inventory_and_assessment_hash(tmp_path):
    run = completed(tmp_path)
    module = run / "arms/meta/repeat/assessments/hash/module.py"
    module.parent.mkdir(parents=True)
    module.write_text("# Research method; do not import during inventory.\nraise RuntimeError('not executed')\n")
    put(module.parent / "assessment.json", {"status": "valid", "identity": {"phase": "search"},
                                           "module_sha256": closeout.sha256(module)})
    selection = module.relative_to(tmp_path / "runs").as_posix()
    out = closeout.build(tmp_path, "algorithm", selections=[selection])
    row = json.loads((out / "model_inventory.json").read_text())[0]
    assert row["kind"] == "evolved_algorithm" and row["phase"] == "search"
    exported = json.loads((out / "selected_models.json").read_text())[0]
    assert exported["assessment_checksum_matched"]


def test_path_private_and_size_guards(tmp_path):
    run = completed(tmp_path)
    (run / "best.pt").write_bytes(b"oversize")
    states = {run.name: closeout.inspect_run(run)}
    with pytest.raises(ValueError, match="relative"):
        closeout.safe_file(tmp_path, "../outside")
    with pytest.raises(ValueError, match="cap"):
        closeout.export_model(tmp_path / "runs", run.name + "/best.pt", tmp_path / "out", states, 1)
    secret = run / "workers/private/best.pt"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(b"private")
    with pytest.raises(ValueError, match="Private"):
        closeout.export_model(tmp_path / "runs", secret.relative_to(tmp_path / "runs"), tmp_path / "out", states, 100)


def test_live_and_ambiguous_states_are_not_completed(tmp_path):
    run = completed(tmp_path)
    put(run / "heartbeat.json", {"state": "running", "updated_at": time.time()})
    assert closeout.inspect_run(run)["state"] == "running_heartbeat"
    (run / "heartbeat.json").unlink()
    put(run / "summary.json", {"search_complete": False})
    assert closeout.inspect_run(run)["state"] == "conflicting_completion_metadata"
    put(run / "INCOMPLETE.json", {"error_type": "RuntimeError"})
    later = (run / "COMPLETE.json").stat().st_mtime + 5
    os.utime(run / "INCOMPLETE.json", (later, later))
    assert closeout.inspect_run(run)["state"] == "incomplete_after_completion_marker"
    unknown = tmp_path / "old_run"
    unknown.mkdir()
    assert closeout.inspect_run(unknown)["state"] == "unknown_no_completion_marker"


def test_symlink_and_existing_output_refused(tmp_path):
    run = completed(tmp_path)
    out = closeout.build(tmp_path, "once")
    with pytest.raises(FileExistsError):
        closeout.build(tmp_path, "once")
    assert (out / "progress.json").exists()
    source = run / "real.pt"
    source.write_bytes(b"weight")
    try:
        (run / "linked.pt").symlink_to(source)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this host")
    with pytest.raises(ValueError, match="Symbolic"):
        closeout.safe_file(tmp_path / "runs", run.name + "/linked.pt")


def test_cli_runs_with_standard_library_only_outside_checkout(tmp_path):
    completed(tmp_path)
    command = [sys.executable, "-S", str(REPO / "scripts/project_closeout.py"),
               "--work-dir", str(tmp_path), "--name", "cli", "--unfinished-run", "round27_agent_01"]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "CLOSEOUT COMPLETE" in result.stdout
    snapshot = json.loads((tmp_path / "closeout/cli/snapshot.json").read_text())
    assert snapshot["missing_declared_runs"] == ["round27_agent_01"]


def test_frozen_export_uses_existing_source_shortlist_not_confirmation(tmp_path):
    run = completed(tmp_path)
    pool = run / "arms/repair/repeat"
    module = pool / "assessments/hash/module.py"
    module.parent.mkdir(parents=True)
    module.write_text("# Frozen research algorithm\n")
    record = {"status": "valid", "module": str(module), "module_sha256": closeout.sha256(module)}
    put(pool / "search_frozen.json", {"source_only": True, "finalists": [record]})
    put(run / "comparison.json", {"winner": "a different confirmation model"})
    out = closeout.build(tmp_path, "frozen", frozen_runs=[run.name])
    rows = json.loads((out / "selected_models.json").read_text())
    assert len(rows) == 1 and rows[0]["sha256"] == record["module_sha256"]
    assert rows[0]["selection_origin"].startswith("existing_source_only")
    with pytest.raises(ValueError, match="completed"):
        closeout.frozen_models(tmp_path / "runs", run.name,
                              {run.name: {"state": "unfinished_user_declared"}})
    module.write_text("# changed\n")
    with pytest.raises(ValueError, match="checksum"):
        closeout.frozen_models(tmp_path / "runs", run.name,
                              {run.name: {"state": "completed_declared"}})


def test_current_documentation_links_resolve():
    import re
    for path in [REPO / "README.md", REPO / "docs/PROJECT_STATUS.md", REPO / "docs/PROJECT_CLOSEOUT.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https:", "http:", "#")):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (path.name, target)
