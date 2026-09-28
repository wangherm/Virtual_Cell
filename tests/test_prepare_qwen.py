"""Downloader must pin revision, avoid implicit credentials, and reject bad bytes."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

source = Path(__file__).resolve().parents[1] / "scripts/prepare_qwen.py"
spec = importlib.util.spec_from_file_location("prepare_qwen", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def lock(content):
    return {"model_id": "test/public-model", "revision": "a" * 40,
            "files": {"config.json": {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}


def test_existing_verified_model_never_connects(tmp_path):
    content = b'{"model_type":"qwen3"}'
    (tmp_path / "config.json").write_bytes(content)
    def unavailable(**kwargs):
        raise AssertionError("Offline reuse must not contact the network")
    module.prepare(tmp_path, "https://example.invalid", lock(content), unavailable)


def test_download_is_pinned_and_public(tmp_path):
    content = b"valid"
    def download(**kwargs):
        assert kwargs["revision"] == "a" * 40
        assert kwargs["endpoint"] == "https://mirror.example"
        assert kwargs["token"] is False
        (Path(kwargs["local_dir"]) / kwargs["filename"]).write_bytes(content)
    module.prepare(tmp_path, "https://mirror.example", lock(content), download)


def test_download_bad_hash_fails(tmp_path):
    def download(**kwargs):
        (tmp_path / "config.json").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        module.prepare(tmp_path, "https://mirror.example", lock(b"valid"), download)


@pytest.mark.parametrize("error", [ChunkedEncodingError, ConnectionError, Timeout])
def test_interrupted_transfer_keeps_partial_and_retries(tmp_path, error):
    content = b"verified complete file"
    partial = tmp_path / ".cache" / "checkpoint.incomplete"
    calls, waits = [], []
    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["force_download"] is False
        if len(calls) == 1:
            partial.parent.mkdir()
            partial.write_bytes(content[:8])
            raise error("simulated interrupted transfer")
        assert partial.read_bytes() == content[:8]
        (tmp_path / "config.json").write_bytes(content)
    module.prepare(tmp_path, "https://mirror.example", lock(content), download, sleep=waits.append)
    assert len(calls) == 2 and waits == [2]
    assert module.matches(tmp_path / "config.json", lock(content)["files"]["config.json"])


def test_retry_exhaustion_is_bounded_and_does_not_report_success(tmp_path, capsys):
    calls, waits = [], []
    def download(**kwargs):
        calls.append(kwargs)
        raise ChunkedEncodingError("still broken")
    with pytest.raises(ChunkedEncodingError):
        module.prepare(tmp_path, "https://mirror.example", lock(b"valid"), download, sleep=waits.append)
    assert len(calls) == 4 and waits == [2, 4, 8]
    output = capsys.readouterr().out
    assert "attempts exhausted" in output and "SNAPSHOT VERIFIED" not in output


def test_hash_failure_after_retry_still_rejects_bytes(tmp_path):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ChunkedEncodingError("interrupted")
        (tmp_path / "config.json").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        module.prepare(tmp_path, "https://mirror.example", lock(b"valid"), download, sleep=lambda _: None)
    assert len(calls) == 2
