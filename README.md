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

## Run locally (once the app is built)

```bash
cp .env.example .env          # put the real API key in .env — never commit it
docker build -t tensorforge .
docker run -p 8000:8000 -e API_KEY=<key> tensorforge
```

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
