"""Download the pinned public Qwen snapshot, verify hashes, then train offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path


LOCK = Path(__file__).resolve().parents[1] / "configs/qwen_backbone.lock.json"


def matches(path, expected):
    if not path.is_file() or path.stat().st_size != expected["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected["sha256"]


def prepare(folder, endpoint, lock=None, download=None):
    lock = lock or json.loads(LOCK.read_text(encoding="utf-8"))
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    missing = [name for name, spec in lock["files"].items() if not matches(folder / name, spec)]
    if missing:
        if download is None:
            from huggingface_hub import hf_hub_download
            download = hf_hub_download
        print(f"Downloading {lock['model_id']} @ {lock['revision']} from {endpoint}", flush=True)
        for name in missing:
            print(f"Preparing {name}", flush=True)
            download(repo_id=lock["model_id"], filename=name, revision=lock["revision"],
                     local_dir=str(folder), endpoint=endpoint, token=False,
                     force_download=(folder / name).exists())
            if not matches(folder / name, lock["files"][name]):
                raise ValueError(f"Checksum mismatch: {name}. Refusing to load this file.")
    print(f"QWEN SNAPSHOT VERIFIED: {folder}", flush=True)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    args = parser.parse_args()
    prepare(args.model_dir, args.endpoint)
