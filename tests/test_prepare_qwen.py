"""Downloader must pin revision, avoid implicit credentials, and reject bad bytes."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

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
