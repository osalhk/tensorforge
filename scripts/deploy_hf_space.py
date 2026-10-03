"""Deploy the API (with the XLM-R model) to a public Hugging Face Docker Space.

    .venv/bin/hf auth login                 # once; needs a token with "write" access
    .venv/bin/python scripts/deploy_hf_space.py [--space tensorforge]

- Creates the Space if needed and uploads only what the image needs: Dockerfile,
  requirements.txt, app/, frontend/ and the model files (the 1.1 GB ONNX goes up as LFS).
- Stores API_KEY (from .env or the environment) as a Space secret; it is never uploaded as a file.
- The Space serves the same container as `docker run`, on https://<user>-<space>.hf.space.
"""

import argparse
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

SPACE_README = """---
title: TensorForge Ticket Triage
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Multilingual RideEat support ticket classifier API
---

TensorForge 2.0 Phase 2: classifies RideEat support tickets (English, Sinhala, Tamil, Singlish,
Tanglish) into category, secondary category and urgency, following the official OpenAPI contract.

- Demo: `/demo/`
- Health: `/health`
- Prediction endpoints require the team API key (`X-API-Key` or `Authorization: Bearer`).
"""

UPLOAD = ["Dockerfile", "requirements.txt", ".dockerignore", "app/*.py", "frontend/*",
          "model/baseline.joblib", "model/xlmr/config.json", "model/xlmr/model.onnx",
          "model/xlmr/tokenizer.json", "model/xlmr/tokenizer_config.json"]


def read_api_key() -> str:
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="tensorforge", help="Space name (under your account)")
    args = parser.parse_args()

    if not (ROOT / "model" / "xlmr" / "model.onnx").exists():
        sys.exit("model/xlmr is missing: run scripts/download_model.py first.")
    api_key = read_api_key()
    if not api_key or api_key == "replace-me":
        sys.exit("No API_KEY found in .env or the environment.")

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.space}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_secret(repo_id, "API_KEY", api_key)
    print(f"Space {repo_id}: API_KEY secret set")

    print("Uploading files (the 1.1 GB model can take a while on the first upload)...")
    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space", commit_message="Space card")
    api.upload_folder(folder_path=ROOT, repo_id=repo_id, repo_type="space",
                      allow_patterns=UPLOAD, delete_patterns=["app/*", "frontend/*"],
                      commit_message="Deploy TensorForge API")

    host = re.sub(r"[^a-z0-9-]", "-", repo_id.lower().replace("/", "-"))
    print(f"\nDone. The Space now builds the image (a few minutes).")
    print(f"  Page:    https://huggingface.co/spaces/{repo_id}")
    print(f"  API URL: https://{host}.hf.space   (submit this to the organisers)")


if __name__ == "__main__":
    main()
