"""Synthetic-only checks: never inspect this workstation's live credentials."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/server_privacy_audit.py'
spec = importlib.util.spec_from_file_location('privacy_audit', SCRIPT)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def fake_key(prefix='sk-'):
    return prefix + ''.join(chr(65 + i % 26) for i in range(40))


def test_detects_credentials_without_values_or_source_mutation(tmp_path):
    home = tmp_path / 'home'
    gh = home / '.config/gh/hosts.yml'
    gh.parent.mkdir(parents=True)
    key = fake_key('gho_')
    gh.write_text('github.com:\n  oauth_token: ' + key + '\n')
    history = home / '.bash_history'
    password = ''.join(['opaque', 'password', '347832'])
    history.write_text('client --password ' + password + '\n')
    notebook = home / 'work.ipynb'
    notebook.write_text(json.dumps({'cell': 'DEEPSEEK_API_KEY=' + fake_key()}))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (gh, history, notebook)}
    audit = audit_module.Audit(tmp_path / 'report')
    audit.walk([home])
    result = audit.finish([home], False, False)
    categories = {r['category'] for r in result['findings']}
    assert {'github_token_pattern', 'github_cli_credential_store', 'api_key_pattern', 'possible_command_line_credential'} <= categories
    for path in (audit.output / 'findings.json', audit.output / 'SUMMARY.txt', audit.output / 'progress.json'):
        text = path.read_text()
        assert key not in text and fake_key() not in text and password not in text
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in before.items())
    if sys.platform == 'linux':
        assert audit.output.stat().st_mode & 0o777 == 0o700
        assert (audit.output / 'findings.json').stat().st_mode & 0o777 == 0o600


def test_environment_processes_and_redacted_secret_filenames(tmp_path, monkeypatch):
    audit = audit_module.Audit(tmp_path / 'report')
    value = fake_key()
    audit.environment({'DEEPSEEK_API_KEY': value, 'SSH_AUTH_SOCK': '/agent/socket'}, 'fixture_environment')
    process = tmp_path / 'fake_proc/123'
    process.mkdir(parents=True)
    (process / 'environ').write_bytes(('HF_TOKEN=' + fake_key('hf_') + '\0').encode())
    (process / 'cmdline').write_bytes(('python\0--token\0' + value + '\0').encode())
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(os, 'getuid', lambda: process.stat().st_uid, raising=False)
    audit.processes(tmp_path / 'fake_proc')
    audit.add('fixture', 'HIGH', str(tmp_path / fake_key('hf_')))
    result = audit.finish([], True, True)
    text = json.dumps(result)
    assert value not in text and fake_key('hf_') not in text
    assert audit.process_scanned == 1
    assert any(r['location'].startswith('process:123:HF_TOKEN') for r in result['findings'])


def test_private_keys_and_metadata_are_distinguished(tmp_path):
    root = tmp_path / 'home'
    ssh = root / '.ssh'
    ssh.mkdir(parents=True)
    (ssh / 'id_ed25519').write_text('-----BEGIN ' + 'OPENSSH PRIVATE KEY-----\nopaque\n')
    (ssh / 'authorized_keys').write_text('ssh-ed25519 publicmaterial owner@' + 'host.example\n')
    audit = audit_module.Audit(tmp_path / 'report')
    audit.walk([root])
    categories = {f['category'] for f in audit.findings}
    assert 'private_key_material' in categories and 'ssh_private_key_file' in categories
    assert 'ssh_public_access_list_not_private_key' in categories
    assert 'email_like_personal_metadata' in categories
    assert all(f['severity'] == 'INFO' for f in audit.findings if f['category'] == 'ssh_public_access_list_not_private_key')


def test_partial_files_and_exclusions_are_explicit(tmp_path):
    root = tmp_path / 'home'
    root.mkdir()
    value = fake_key()
    (root / 'large.log').write_text('a' * 700000 + '\n' + value + '\n' + 'b' * 1500000)
    binary = root / 'weights.pt'
    binary.write_bytes(value.encode())
    git_objects = root / '.git/objects'
    git_objects.mkdir(parents=True)
    (git_objects / 'object').write_text(value)
    env = root / '.venv'
    env.mkdir()
    (env / 'fixture.py').write_text(value)
    audit = audit_module.Audit(tmp_path / 'report', max_file_mib=1)
    audit.walk([root])
    result = audit.finish([root], False, False)
    assert audit.counts['partially_scanned_files'] == 1
    assert audit.counts['binary_archive_or_database_files_skipped'] == 1
    assert audit.counts['directories_skipped'] >= 2
    assert not any(r['category'] == 'api_key_pattern' for r in audit.findings)
    assert result['no_findings_is_not_a_clean_disk_certificate']


def test_symlinks_and_output_overwrite_are_refused(tmp_path):
    audit = audit_module.Audit(tmp_path / 'report')
    with pytest.raises(FileExistsError):
        audit_module.Audit(tmp_path / 'report')
    outside = tmp_path / 'outside'
    outside.write_text(fake_key())
    root = tmp_path / 'home'
    root.mkdir()
    try:
        (root / 'link').symlink_to(outside)
    except OSError:
        pytest.skip('Host does not permit symlinks')
    audit.walk([root])
    assert not audit.findings


def test_cli_standard_library_only_on_controlled_fixture(tmp_path):
    root = tmp_path / 'home'
    root.mkdir()
    key = fake_key()
    (root / '.env').write_text('DEEPSEEK_API_KEY=' + key)
    result = subprocess.run([sys.executable, '-S', str(SCRIPT), '--root', str(root),
                             '--output', str(tmp_path / 'report'), '--no-env', '--no-processes'],
                            capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert key not in result.stdout + result.stderr
    assert 'AUDIT COMPLETE: HIGH=' in result.stdout


def test_placeholders_and_usage_counters_are_not_literal_credentials(tmp_path):
    audit = audit_module.Audit(tmp_path / 'report')
    audit.text('max_tokens: 32768\ninput_tokens: 45000\napi_key: YOUR_API_KEY\napi_key: ${KEY_FROM_ENV}\n', 'fixture')
    assert audit.findings == []
