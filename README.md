# TensorForge 2.0 — Phase 2: Support Ticket Classifier

Classifies RideEat customer support tickets (English, Sinhala, Tamil, Singlish, Tanglish, mixed) into:

1. `category` — the main issue (11 classes)
2. `secondary_category` — a second issue, or `null`
3. `is_urgent` — needs immediate human attention

The model is trained by us (no LLM API calls for classification) and served through a FastAPI service that follows the official OpenAPI contract.

**Deadline:** 10 October 2026, 6:00 PM

**Hosted API:** https://13-232-241-157.sslip.io (demo: https://13-232-241-157.sslip.io/demo/)
— AWS EC2, Caddy terminates TLS (Let's Encrypt) and forwards to the API container.

## Results (validation set, 800 tickets)

| Model | Category accuracy | Category macro-F1 | Secondary macro-F1 | Urgent F1 | All three correct |
|---|---|---|---|---|---|
| TF-IDF + logistic regression (`notebooks/01_baseline.py`) | 0.604 | 0.631 | 0.668 | 0.841 | 0.564 |
| **Fine-tuned XLM-RoBERTa, 3 heads** (`notebooks/02_transformer_colab.ipynb`) | **0.853** | **0.851** | **0.877** | **0.969** | **0.834** |

Served as fp32 ONNX (int8 lost 3–4 points). In Docker with `--cpus 2 --memory 4g --network none`:
`/predict` ~130 ms, 100-ticket batch 14 s, 5,000-ticket job ~13–15 min, `/predict` p95 614 ms while a job runs, 1.7 GB RAM.

## Folder layout

```
data/        dataset (train / validation) + DATA_NOTES.md
api/         official OpenAPI spec + JSON schemas (do not edit)
docs/        competition guidelines
notebooks/   experiments & training
model/       baseline.joblib (in git) + xlmr/ (1.1 GB, downloaded — see below)
scripts/     download_model.py
app/         FastAPI service
frontend/    demo website
Dockerfile
```

## Run locally

```bash
# one-time setup
uv venv .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
cp .env.example .env          # put the real API key in .env — never commit it

# get the XLM-R model (1.1 GB, attached to a GitHub release; checksum verified)
python3 scripts/download_model.py

# optional: retrain the baseline (writes model/baseline.joblib)
.venv/bin/python notebooks/01_baseline.py

# start the API, then open http://localhost:8000/docs
set -a && . ./.env && set +a && .venv/bin/uvicorn app.main:app --port 8000

# run the contract tests (checks every response against the official schemas in api/)
.venv/bin/python -m pytest -q
```

Without `model/xlmr/` the API falls back to the baseline (`model_version` starts with `baseline-`).
`MODEL_PATH` picks a model explicitly.

With Docker (download the model first; it is copied into the image so nothing is fetched at runtime):

```bash
python3 scripts/download_model.py
docker build -t tensorforge .
docker run -p 8000:8000 -e API_KEY=<key> tensorforge
```

## How the API is built (`app/`)

| File | Job |
|---|---|
| `main.py` | Endpoints, API key check, error format. Checks run in the spec order: auth → content type → size → JSON → validation |
| `validation.py` | Input rules (channel values, empty text, lengths, batch `index` details, duplicate ids) |
| `model.py` | Loads XLM-R (ONNX) or the baseline, applies the consistency rules (secondary ≠ primary, spam → not urgent, team table). Uses as many threads as the container's CPU limit, and lets live requests go before async-job work |
| `features.py` | Builds the model input from `channel`, `subject`, `text` — shared with training |
| `jobs.py` | Async batch jobs stored in SQLite; a background worker processes them in chunks. Running jobs become `failed`/`interrupted` after a restart |

`model_version` is the model name plus the first 12 characters of the model file's SHA-256 hash, so it always identifies the exact artifact.

`needs_human_review` is `true` when the confidence in the primary category is below **0.5**.

## Endpoints

| Method | Path | Auth |
|---|---|---|
| GET | `/health` | public |
| POST | `/predict` | API key |
| POST | `/predict/batch` | API key (1–100 tickets) |
| POST | `/batch/jobs` | API key (up to 5,000 tickets) |
| GET | `/batch/jobs/{job_id}` | API key |
| GET | `/batch/jobs/{job_id}/results` | API key |
| DELETE | `/batch/jobs/{job_id}` | API key |

Send the key as `X-API-Key: <key>` or `Authorization: Bearer <key>`.
