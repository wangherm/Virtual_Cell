"""Regression checks for the terminal workflow's overwrite protection."""
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/smoke_test.py"


def invoke(cwd, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd,
                          capture_output=True, text=True)


def test_smoke_help_works_outside_checkout(tmp_path):
    result = invoke(tmp_path, "--help")
    assert result.returncode == 0
    assert "--resume" in result.stdout


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_smoke_refuses_existing_content(tmp_path, existing_kind):
    work = tmp_path / "existing"
    if existing_kind == "directory":
        work.mkdir()
        sentinel = work / "important.txt"
    else:
        sentinel = work
    sentinel.write_text("preserve this", encoding="utf-8")
    result = invoke(tmp_path, "--work-dir", str(work))
    assert result.returncode == 2
    assert "not empty" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve this"


def test_smoke_resume_rejects_incomplete_preprocessing(tmp_path):
    work = tmp_path / "partial"
    (work / "demo/prepared").mkdir(parents=True)
    result = invoke(tmp_path, "--work-dir", str(work), "--resume")
    assert result.returncode == 2
    assert "complete prepared data" in result.stderr
    assert not (work / "run").exists()
