"""Deploy the API (with the XLM-R model) to a public Hugging Face Docker Space.

    .venv/bin/hf auth login                 # once; needs a token with "write" access
    .venv/bin/python scripts/deploy_hf_space.py [--space tensorforge] [--model-repo tensorforge-model]

- Uploads the model files to a PRIVATE model repo, so the public Space never exposes them.
- Creates the Space if needed and uploads only code: Dockerfile.space (as the Space's Dockerfile),
  requirements.txt, app/ and frontend/. The Space downloads the model from the private repo at build time.
- Stores API_KEY and HF_TOKEN as Space secrets (from .env or the environment); they are never uploaded
  as files. HF_TOKEN comes from HF_READ_TOKEN: a read-only token, not the write token used to deploy.
- The Space serves the same API as `docker run`, on https://<user>-<space>.hf.space.
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

SPACE_FILES = ["requirements.txt", ".dockerignore", "app/*.py", "frontend/*"]
MODEL_FILES = ["baseline.joblib", "xlmr/config.json", "xlmr/model.onnx",
               "xlmr/tokenizer.json", "xlmr/tokenizer_config.json"]


def read_env(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="tensorforge", help="Space name (under your account)")
    parser.add_argument("--model-repo", default="tensorforge-model", help="Private model repo name")
    args = parser.parse_args()

    if not (ROOT / "model" / "xlmr" / "model.onnx").exists():
        sys.exit("model/xlmr is missing: run scripts/download_model.py first.")
    api_key = read_env("API_KEY")
    if not api_key or api_key == "replace-me":
        sys.exit("No API_KEY found in .env or the environment.")
    read_token = read_env("HF_READ_TOKEN")
    if not read_token:
        sys.exit("No HF_READ_TOKEN found in .env or the environment "
                 "(create a read-only token at https://huggingface.co/settings/tokens).")

    api = HfApi()
    user = api.whoami()["name"]
    model_id = f"{user}/{args.model_repo}"
    space_id = f"{user}/{args.space}"

    api.create_repo(model_id, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(model_id).private:
        sys.exit(f"{model_id} exists and is public; refusing to upload the model there.")
    print(f"Uploading model to private repo {model_id} (the 1.1 GB file can take a while)...")
    api.upload_folder(folder_path=ROOT / "model", repo_id=model_id, repo_type="model",
                      allow_patterns=MODEL_FILES, commit_message="Upload model")

    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_secret(space_id, "API_KEY", api_key)
    api.add_space_secret(space_id, "HF_TOKEN", read_token)
    api.add_space_variable(space_id, "MODEL_REPO", model_id)
    print(f"Space {space_id}: API_KEY and HF_TOKEN secrets, MODEL_REPO variable set")

    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md",
                    repo_id=space_id, repo_type="space", commit_message="Space card")
    api.upload_file(path_or_fileobj=ROOT / "Dockerfile.space", path_in_repo="Dockerfile",
                    repo_id=space_id, repo_type="space", commit_message="Space Dockerfile")
    api.upload_folder(folder_path=ROOT, repo_id=space_id, repo_type="space",
                      allow_patterns=SPACE_FILES, delete_patterns=["app/*", "frontend/*", "model/*"],
                      commit_message="Deploy TensorForge API")

    host = re.sub(r"[^a-z0-9-]", "-", space_id.lower().replace("/", "-"))
    print(f"\nDone. The Space now builds the image (a few minutes).")
    print(f"  Page:    https://huggingface.co/spaces/{space_id}")
    print(f"  API URL: https://{host}.hf.space   (submit this to the organisers)")


if __name__ == "__main__":
    main()
