"""Use disposable synthetic fixtures only; never clean real local credentials."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/server_privacy_cleanup.py'
spec = importlib.util.spec_from_file_location('cleanup', SCRIPT)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


def put(root, name, text='synthetic fixture'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_preview_apply_preserves_research_and_access(tmp_path):
    home, work = tmp_path / 'home', tmp_path / 'work'
    removable = [put(home, '.git-credentials'), put(home, '.bash_history'),
                 put(home, '.config/gh/hosts.yml'), put(work, 'models/huggingface/token')]
    keep = [put(home, '.ssh/id_ed25519'), put(home, '.ssh/authorized_keys'),
            put(home, '.bashrc'), put(home, '.env'), put(work, 'runs/best.pt'),
            put(work, 'prepared/data.npz'), put(work, 'models/Qwen/model.safetensors'),
            put(home, 'repo/tokenizer.py'), put(work, 'knowledge/memory.sqlite')]
    before = {p: p.read_bytes() for p in removable + keep}
    preview = cleanup.run(home, work)
    assert len(preview) == len(removable)
    assert all(r['status'] == 'would_remove' for r in preview)
    assert all(p.read_bytes() == data for p, data in before.items())
    applied = cleanup.run(home, work, apply=True)
    assert all(r['status'] == 'removed' for r in applied)
    assert all(not p.exists() for p in removable)
    assert all(p.read_bytes() == before[p] for p in keep)
    assert cleanup.run(home, work, apply=True) == []


def test_sessions_require_separate_opt_in(tmp_path):
    home, work = tmp_path / 'home', tmp_path / 'work'
    sessions = [put(home, '.local/share/jupyter/runtime/jpserver-123.json'),
                put(home, '.local/share/jupyter/runtime/jupyter_cookie_secret'),
                put(home, '.config/Code/agent-host/local-endpoint/entries/abc.json')]
    keep = put(home, '.local/share/jupyter/runtime/kernel-123.json')
    cleanup.run(home, work, apply=True)
    assert all(p.exists() for p in sessions)
    rows = cleanup.run(home, work, apply=True, sessions=True)
    assert len(rows) == 3 and all(r['status'] == 'removed' for r in rows)
    assert keep.exists()


def test_log_redaction_preserves_metrics_and_never_prints_values(tmp_path):
    work = tmp_path / 'work'
    log = put(work, 'logs/download_raw.log', 'loss=0.12\nhttps://files.example/data?X-Amz-Credential=TEST&X-Amz-Signature=SECRET\nhttps://person:SECRET@files.example/a\n')
    before = log.read_bytes()
    preview = cleanup.clean_download_log(log, work)
    assert preview['url_occurrences'] == 2 and log.read_bytes() == before
    row = cleanup.clean_download_log(log, work, apply=True)
    assert row['status'] == 'redacted'
    assert 'SECRET' not in log.read_text() and 'loss=0.12' in log.read_text()
    assert 'SECRET' not in str(row)
    assert not list(log.parent.glob('.privacy-redact-*'))
    if sys.platform == 'linux':
        assert log.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux server symlink and hardlink boundary')
def test_rejects_links_and_symlink_ancestors(tmp_path):
    home, work = tmp_path / 'home', tmp_path / 'work'
    home.mkdir()
    other = put(tmp_path, 'outside/private')
    (home / '.bash_history').symlink_to(other)
    (home / '.config').symlink_to(other.parent, target_is_directory=True)
    os.link(other, home / '.git-credentials')
    rows = cleanup.run(home, work, apply=True)
    assert all(row['status'] == 'skipped_unsafe_path' for row in rows)
    assert other.read_text() == 'synthetic fixture'
    assert not cleanup.safe_file(other, home)
