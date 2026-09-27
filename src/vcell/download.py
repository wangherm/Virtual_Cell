"""Opt-in public downloads. List first; select exact file IDs, never download everything."""
from pathlib import Path
import hashlib
import requests


def figshare_catalogue(article_id=20029387):
    url = f"https://api.figshare.com/v2/articles/{int(article_id)}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()["files"]


def download_figshare(file_id, dest, article_id=20029387):
    choices = [f for f in figshare_catalogue(article_id) if int(f["id"]) == int(file_id)]
    if len(choices) != 1:
        raise ValueError("File ID not present in requested article; list catalogue first.")
    f = choices[0]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    name = Path(f["name"]).name
    path = dest / name
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(path.suffix + ".part")
    if temp.exists():
        raise FileExistsError(f"Incomplete prior download: {temp}. Inspect/move it before retrying.")
    digest = hashlib.md5()  # Figshare publishes MD5 for transfer integrity, not security.
    downloaded = 0
    with requests.get(f["download_url"], stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        with open(temp, "wb") as out:
            for block in r.iter_content(1024 * 1024):
                if block:
                    out.write(block)
                    digest.update(block)
                    downloaded += len(block)
    if downloaded != int(f["size"]):
        raise IOError("File size mismatch; incomplete .part retained for inspection")
    expected = f.get("computed_md5") or f.get("supplied_md5")
    if expected and digest.hexdigest() != expected:
        raise IOError("MD5 mismatch; .part retained for inspection")
    temp.rename(path)
    print(f"Downloaded {path} ({downloaded} bytes)")
    return path
