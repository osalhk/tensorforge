"""TensorForge 2.0 Phase 2 ticket classification API.

Follows api/tensorforge-phase2-openapi-v2.yaml. Every prediction/job endpoint checks, in order:
auth (401) -> content type (415) -> size (413) -> JSON parse (400) -> validation (422).

Environment:
    API_KEY          required; without it every protected endpoint returns 401
    MODEL_PATH       model folder (XLM-R) or .joblib file (baseline);
                     default model/xlmr if present, else model/baseline.joblib
    ORT_THREADS      onnxruntime threads for the XLM-R model (default: the container's CPU limit)
    JOBS_DB          default jobs.db (SQLite file for async jobs)
    JOB_RETENTION_H  default 24 (spec minimum is 6)
"""

import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader, HTTPBearer
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.jobs import FINISHED, JobStore, JobWorker
from app.model import load_predictor
from app.validation import validate_batch, validate_ticket

log = logging.getLogger("tensorforge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
MB = 1024 * 1024
PREDICT_LIMIT, BATCH_LIMIT, JOBS_LIMIT = 1 * MB, 5 * MB, 25 * MB
BATCH_MAX, JOBS_MAX = 100, 5000
RETRY_AFTER = "2"


class AsciiJSONResponse(JSONResponse):
    """ASCII-escaped JSON, so unusual Unicode in echoed ids can never crash encoding."""

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def error(status: int, code: str, message: str, details=None, headers=None):
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return AsciiJSONResponse(body, status_code=status, headers=headers)


def validation_error(details):
    return error(422, "validation_error", "Request failed validation.", details)


# ---------------------------------------------------------------- model + jobs state

class State:
    api_key: str = ""
    predictor = None
    store: JobStore | None = None
    worker: JobWorker | None = None


state = State()


def load_model():
    default = ROOT / "model" / "xlmr"
    if not (default / "model.onnx").exists():
        default = ROOT / "model" / "baseline.joblib"
    path = Path(os.environ.get("MODEL_PATH", default))
    try:
        t = time.time()
        state.predictor = load_predictor(path)
        log.info("model %s loaded in %.1fs (threads: %s)", state.predictor.version, time.time() - t,
                 getattr(state.predictor, "threads", "-"))
    except Exception:
        log.exception("failed to load model from %s", path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.api_key = os.environ.get("API_KEY", "")
    if not state.api_key:
        log.warning("API_KEY is not set: all protected endpoints will return 401")
    # Load in the background so /health can answer 503 while loading.
    threading.Thread(target=load_model, daemon=True, name="model-loader").start()
    state.store = JobStore(
        os.environ.get("JOBS_DB", str(ROOT / "jobs.db")),
        retention_hours=float(os.environ.get("JOB_RETENTION_H", "24")),
    )
    state.worker = JobWorker(state.store, lambda: state.predictor)
    state.worker.start()
    yield
    state.worker.stop_event.set()


app = FastAPI(
    title="TensorForge Ticket Classifier",
    version="2.1.0",
    lifespan=lifespan,
    default_response_class=AsciiJSONResponse,
)

# These only make the "Authorize" button appear in /docs; the real check is check_auth().
api_key_doc = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_doc = HTTPBearer(auto_error=False)
PROTECTED = [Security(api_key_doc), Security(bearer_doc)]


# ---------------------------------------------------------------- middleware + error handlers

class RequestIdMiddleware:
    """Echo X-Request-ID on every response when the client sent one."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = next((v for k, v in scope["headers"] if k == b"x-request-id"), None)
        if rid is None:
            return await self.app(scope, receive, send)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message["headers"] = [h for h in message.get("headers", []) if h[0] != b"x-request-id"]
                message["headers"].append((b"x-request-id", rid[:128]))
            await send(message)

        await self.app(scope, receive, send_with_id)


app.add_middleware(RequestIdMiddleware)

HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    code = HTTP_CODES.get(exc.status_code, "http_error")
    message = {404: "Route not found.", 405: "Method not allowed."}.get(exc.status_code, str(exc.detail))
    return error(exc.status_code, code, message, headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request, exc):
    return validation_error([{"field": ".".join(map(str, e["loc"][1:])) or "request", "issue": e["msg"]}
                             for e in exc.errors()])


@app.exception_handler(Exception)
async def unhandled_handler(request, exc):
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return error(500, "internal_error", "Internal server error.")


# ---------------------------------------------------------------- request pipeline

def check_auth(request: Request):
    """Returns an error response, or None when the key is valid."""
    candidates = []
    if key := request.headers.get("x-api-key"):
        candidates.append(key.strip())
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        candidates.append(token.strip())

    headers = {"WWW-Authenticate": "Bearer"}
    if not candidates:
        return error(401, "unauthorized", "Missing API key. Send X-API-Key or Authorization Bearer.",
                     headers=headers)
    expected = state.api_key.encode()
    # Refuse everything when no key is configured; never run open.
    if not expected or not any(hmac.compare_digest(c.encode(), expected) for c in candidates):
        return error(401, "unauthorized", "Invalid API key.", headers=headers)
    return None


class BodyTooLarge(Exception):
    pass


DRAIN_CAP = 64 * MB


async def discard_body(request: Request, size: int = 0):
    """Read and drop the rest of an unread body before an early error response.

    Otherwise the server closes the connection while the client is still uploading and the client
    sees a connection reset instead of our 401/413/415. Gives up past DRAIN_CAP.
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > DRAIN_CAP:
        return
    while size <= DRAIN_CAP:
        message = await request.receive()
        if message["type"] != "http.request":
            return
        size += len(message.get("body", b""))
        if not message.get("more_body", False):
            return


async def read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        await discard_body(request)
        raise BodyTooLarge
    chunks, size, more = [], 0, True
    while more:
        message = await request.receive()
        if message["type"] != "http.request":  # client disconnected
            break
        chunk, more = message.get("body", b""), message.get("more_body", False)
        size += len(chunk)
        if size > limit:
            if more:
                await discard_body(request, size)
            raise BodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def parse_json(body: bytes):
    return json.loads(body.decode("utf-8"))


async def read_json(request: Request, limit: int):
    """Auth -> 415 -> 413 -> 400. Returns (data, error_response)."""
    if (err := check_auth(request)) is not None:
        await discard_body(request)
        return None, err
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media_type != "application/json":
        await discard_body(request)
        return None, error(415, "unsupported_media_type", "Content-Type must be application/json.")
    try:
        body = await read_body(request, limit)
    except BodyTooLarge:
        return None, error(413, "payload_too_large", "Request body exceeds the maximum allowed size.")
    try:
        return await run_in_threadpool(parse_json, body), None
    except (ValueError, RecursionError):  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        return None, error(400, "malformed_json", "Request body is not valid JSON.")


def model_unavailable():
    return error(503, "model_not_ready", "Model is still loading. Retry shortly.",
                 headers={"Retry-After": "5"})


# ---------------------------------------------------------------- demo website

FRONTEND = ROOT / "frontend"
if FRONTEND.is_dir():
    # Public static page; it calls the protected API with a key the visitor types in.
    app.mount("/demo", StaticFiles(directory=FRONTEND, html=True), name="demo")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/demo/")


# ---------------------------------------------------------------- endpoints

TICKET_EXAMPLE = {"ticket_id": "TF-TE-000001", "channel": "chat", "subject": "",
                  "text": "bro mage order eka hour ekakata wada late, driver call ganne na. refund ekak denna"}


def body_doc(example):
    return {"requestBody": {"required": True,
                            "content": {"application/json": {"schema": {"type": "object"},
                                                             "example": example}}}}


@app.get("/health", tags=["Core"])
async def health():
    p = state.predictor
    if p is None:
        return AsciiJSONResponse({"status": "loading", "model_version": None, "model_loaded": False},
                                 status_code=503)
    return {"status": "ok", "model_version": p.version, "model_loaded": True}


@app.post("/predict", tags=["Core"], openapi_extra=body_doc(TICKET_EXAMPLE), dependencies=PROTECTED)
async def predict(request: Request):
    data, err = await read_json(request, PREDICT_LIMIT)
    if err:
        return err
    ticket, issues = validate_ticket(data, require_id=False)
    if issues:
        return validation_error(issues)
    if state.predictor is None:
        return model_unavailable()
    return (await run_in_threadpool(state.predictor.predict, [ticket]))[0]


@app.post("/predict/batch", tags=["Evaluation"], openapi_extra=body_doc({"tickets": [TICKET_EXAMPLE]}),
          dependencies=PROTECTED)
async def predict_batch(request: Request):
    started = time.perf_counter()
    data, err = await read_json(request, BATCH_LIMIT)
    if err:
        return err
    tickets, details = validate_batch(data, BATCH_MAX)
    if details:
        return validation_error(details)
    if state.predictor is None:
        return model_unavailable()
    predictions = await run_in_threadpool(state.predictor.predict, tickets)
    return {
        "predictions": predictions,
        "meta": {
            "count": len(predictions),
            "model_version": state.predictor.version,
            "processing_time_ms": int((time.perf_counter() - started) * 1000),
        },
    }


@app.post("/batch/jobs", tags=["Evaluation"], status_code=202,
          openapi_extra=body_doc({"tickets": [TICKET_EXAMPLE]}), dependencies=PROTECTED)
async def submit_job(request: Request):
    data, err = await read_json(request, JOBS_LIMIT)
    if err:
        return err
    tickets, details = await run_in_threadpool(validate_batch, data, JOBS_MAX)
    idem = request.headers.get("idempotency-key")
    if idem is not None and len(idem) > 128:
        details = (details or []) + [{"field": "Idempotency-Key", "issue": "must be at most 128 characters"}]
    if details:
        return validation_error(details)
    if state.predictor is None:
        return model_unavailable()

    row, problem = await run_in_threadpool(state.store.submit, tickets, state.predictor.version, idem)
    if problem == "too_many_jobs":
        return error(429, "too_many_jobs", "Job queue is full. Retry later.",
                     headers={"Retry-After": "30"})
    return AsciiJSONResponse(
        state.store.status_view(row), status_code=202,
        headers={"Location": f"/batch/jobs/{row['job_id']}", "Retry-After": RETRY_AFTER},
    )


def job_not_found():
    return error(404, "job_not_found", "No job with this id.")


def job_expired():
    return error(410, "job_expired", "Job results have expired.")


@app.get("/batch/jobs/{job_id}", tags=["Evaluation"], dependencies=PROTECTED)
async def job_status(job_id: str, request: Request):
    if (err := check_auth(request)) is not None:
        return err
    row = await run_in_threadpool(state.store.get, job_id)
    if row is None:
        return job_not_found()
    if state.store.is_expired(row):
        return job_expired()
    headers = {} if row["status"] in FINISHED else {"Retry-After": RETRY_AFTER}
    return AsciiJSONResponse(state.store.status_view(row), headers=headers)


@app.delete("/batch/jobs/{job_id}", tags=["Evaluation"], status_code=204, dependencies=PROTECTED)
async def delete_job(job_id: str, request: Request):
    if (err := check_auth(request)) is not None:
        return err
    if not await run_in_threadpool(state.store.delete, job_id):
        return job_not_found()
    return Response(status_code=204)


def parse_int_param(request: Request, name: str, minimum: int, maximum: int | None):
    raw = request.query_params.get(name)
    if raw is None:
        return None, None
    try:
        value = int(raw)
    except ValueError:
        return None, {"field": name, "issue": "must be an integer"}
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        return None, {"field": name, "issue": f"must be {bound}"}
    return value, None


@app.get("/batch/jobs/{job_id}/results", tags=["Evaluation"], dependencies=PROTECTED)
async def job_results(job_id: str, request: Request):
    if (err := check_auth(request)) is not None:
        return err
    offset, e1 = parse_int_param(request, "offset", 0, None)
    limit, e2 = parse_int_param(request, "limit", 1, JOBS_MAX)
    if e1 or e2:
        return validation_error([e for e in (e1, e2) if e])

    row = await run_in_threadpool(state.store.get, job_id)
    if row is None:
        return job_not_found()
    if state.store.is_expired(row):
        return job_expired()
    if row["status"] != "succeeded":
        return error(409, "job_not_ready",
                     f"Job status is '{row['status']}'. Results are available once it is 'succeeded'.")

    predictions = await run_in_threadpool(state.store.results, row)
    total = len(predictions)
    offset = offset or 0
    limit = limit if limit is not None else max(total, 1)
    page = predictions[offset:offset + limit]
    next_offset = offset + limit if offset + limit < total else None
    return {
        "job_id": row["job_id"],
        "status": "succeeded",
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": next_offset,
        "model_version": row["model_version"],
        "predictions": page,
    }
