# TensorForge 2.0 — Phase 2: Support Ticket Classifier

Classifies RideEat customer support tickets (English, Sinhala, Tamil, Singlish, Tanglish, mixed) into:

1. `category` — the main issue (11 classes)
2. `secondary_category` — a second issue, or `null`
3. `is_urgent` — needs immediate human attention

The model is trained by us (no LLM API calls for classification) and served through a FastAPI service that follows the official OpenAPI contract.

**Deadline:** 10 October 2026, 6:00 PM

## Folder layout

```
data/        dataset (train / validation) + DATA_NOTES.md
api/         official OpenAPI spec + JSON schemas (do not edit)
docs/        competition guidelines
notebooks/   experiments & training
model/       saved trained model (not committed if large)
app/         FastAPI service
frontend/    demo website
Dockerfile
```

## Run locally

```bash
# one-time setup
uv venv .venv && uv pip install --python .venv/bin/python -r requirements-dev.txt
cp .env.example .env          # put the real API key in .env — never commit it

# train the baseline (writes model/baseline.joblib)
.venv/bin/python notebooks/01_baseline.py

# start the API, then open http://localhost:8000/docs
set -a && . ./.env && set +a && .venv/bin/uvicorn app.main:app --port 8000

# run the contract tests (checks every response against the official schemas in api/)
.venv/bin/python -m pytest -q
```

With Docker:

```bash
docker build -t tensorforge .
docker run -p 8000:8000 -e API_KEY=<key> tensorforge
```

## How the API is built (`app/`)

| File | Job |
|---|---|
| `main.py` | Endpoints, API key check, error format. Checks run in the spec order: auth → content type → size → JSON → validation |
| `validation.py` | Input rules (channel values, empty text, lengths, batch `index` details, duplicate ids) |
| `model.py` | Loads the model, applies the consistency rules (secondary ≠ primary, spam → not urgent, team table) |
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
