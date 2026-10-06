"""Download the fine-tuned XLM-R model (1.1 GB, too large for git) into model/xlmr/.

    python scripts/download_model.py

The model is attached to the public GitHub release below; no token is needed.
GITHUB_TOKEN is still used if set (e.g. to avoid API rate limits).

Without the model, the API falls back to the TF-IDF baseline in model/baseline.joblib.
Standard library only, so it runs before any requirements are installed.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "osalhk/tensorforge"
TAG = "model-xlmr-v1"
ASSET = "xlmr-v1.zip"
SHA256 = "f17f399e2e83676b5f158f5fa33f4d568f41de27739dbb2cf6afdb48890764c5"

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "model" / "xlmr"


def request(url: str, token: str | None, accept: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "tensorforge-download"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def asset_request(token: str | None) -> urllib.request.Request:
    if not token:
        return request(f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}", None, "application/octet-stream")
    # Private repositories: find the asset id through the API, then download it.
    with urllib.request.urlopen(request(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}",
                                        token, "application/vnd.github+json")) as r:
        release = json.load(r)
    asset = next((a for a in release["assets"] if a["name"] == ASSET), None)
    if asset is None:
        sys.exit(f"Release {TAG} has no asset named {ASSET}.")
    return request(asset["url"], token, "application/octet-stream")


def main():
    if (DEST / "model.onnx").exists():
        print(f"{DEST} already exists; delete it to download again.")
        return

    token = os.environ.get("GITHUB_TOKEN") or None
    tmp = Path(tempfile.mkdtemp())
    archive = tmp / ASSET
    try:
        print(f"Downloading {ASSET} from release {TAG} ...")
        try:
            with urllib.request.urlopen(asset_request(token)) as r, open(archive, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, shown = 0, -5
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    pct = int(done * 100 / total) if total else 0
                    if pct >= shown + 5:
                        print(f"  {pct:3d}%  {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
                        shown = pct
        except urllib.error.HTTPError as e:
            hint = " Check the release exists, or set GITHUB_TOKEN." if e.code == 404 and not token else ""
            sys.exit(f"Download failed: HTTP {e.code}.{hint}")

        digest = hashlib.sha256()
        with open(archive, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != SHA256:
            sys.exit("Checksum mismatch: the download is incomplete or the release file changed.")

        DEST.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(DEST)
        print(f"Model ready in {DEST}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
