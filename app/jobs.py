"""Async batch jobs, stored in SQLite so they survive a restart.

- One background worker thread processes queued jobs in small chunks, so /health and
  /predict stay responsive while a job runs.
- On startup, jobs that were `running` are marked `failed` with code `interrupted`
  (spec: a job must never vanish silently). Queued jobs are simply picked up again.
- Finished jobs keep their results for RETENTION_HOURS, then status/results return 410.
- DELETE removes the job row; the worker notices and stops.
"""

import json
import logging
from contextlib import contextmanager
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone

log = logging.getLogger("tensorforge.jobs")

CHUNK_SIZE = 50
POLL_SECONDS = 0.5
PURGE_EVERY_SECONDS = 60
TOMBSTONE_DAYS = 7  # how long an expired job keeps answering 410 before it becomes 404

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    total           INTEGER NOT NULL,
    processed       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    expires_at      TEXT,
    model_version   TEXT NOT NULL,
    error_code      TEXT,
    error_message   TEXT,
    idempotency_key TEXT UNIQUE,
    tickets         TEXT,
    results         TEXT
)
"""

FINISHED = ("succeeded", "failed", "cancelled")


def now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def dumps(obj) -> str:
    # ensure_ascii keeps odd Unicode (e.g. lone surrogates) storable in SQLite.
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


class JobStore:
    def __init__(self, path: str, retention_hours: float = 24, max_active: int = 4):
        self.path = path
        self.retention = timedelta(hours=retention_hours)
        self.max_active = max_active
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(SCHEMA)
            t = now()
            c.execute(
                "UPDATE jobs SET status='failed', error_code='interrupted', "
                "error_message='Service restarted while the job was running.', "
                "finished_at=?, expires_at=?, tickets=NULL WHERE status='running'",
                (iso(t), iso(t + self.retention)),
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ---------- API-facing ----------

    def status_view(self, row) -> dict:
        error = None
        if row["status"] == "failed":
            error = {"code": row["error_code"], "message": row["error_message"]}
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "total": row["total"],
            "processed": row["processed"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "expires_at": row["expires_at"],
            "model_version": row["model_version"],
            "error": error,
        }

    def is_expired(self, row) -> bool:
        return row["expires_at"] is not None and parse(row["expires_at"]) <= now()

    def get(self, job_id: str):
        with self._conn() as c:
            return c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()

    def submit(self, tickets: list[dict], model_version: str, idempotency_key: str | None):
        """Returns (row, error) where error is None, or 'too_many_jobs'."""
        with self._lock, self._conn() as c:
            if idempotency_key is not None:
                row = c.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if row is not None and not self.is_expired(row):
                    return row, None
                if row is not None:  # expired: release the key for a fresh job
                    c.execute("UPDATE jobs SET idempotency_key=NULL WHERE job_id=?", (row["job_id"],))

            active = c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if active >= self.max_active:
                return None, "too_many_jobs"

            job_id = str(uuid.uuid4())
            c.execute(
                "INSERT INTO jobs (job_id, status, total, created_at, model_version, idempotency_key, tickets) "
                "VALUES (?, 'queued', ?, ?, ?, ?, ?)",
                (job_id, len(tickets), iso(now()), model_version, idempotency_key, dumps(tickets)),
            )
            return c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone(), None

    def delete(self, job_id: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM jobs WHERE job_id=?", (job_id,)).rowcount > 0

    def results(self, row) -> list[dict]:
        return json.loads(row["results"])

    # ---------- worker-facing ----------

    def claim_next(self):
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT job_id FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            claimed = c.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE job_id=? AND status='queued'",
                (iso(now()), row["job_id"]),
            ).rowcount
            if not claimed:
                return None
            return c.execute("SELECT job_id, tickets FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()

    def progress(self, job_id: str, processed: int) -> bool:
        """Returns False if the job was deleted (cancelled) meanwhile."""
        with self._conn() as c:
            return c.execute(
                "UPDATE jobs SET processed=? WHERE job_id=? AND status='running'", (processed, job_id)
            ).rowcount > 0

    def finish(self, job_id: str, results: list[dict]):
        t = now()
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='succeeded', processed=total, finished_at=?, expires_at=?, "
                "results=?, tickets=NULL WHERE job_id=? AND status='running'",
                (iso(t), iso(t + self.retention), dumps(results), job_id),
            )

    def fail(self, job_id: str, code: str, message: str):
        t = now()
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status='failed', error_code=?, error_message=?, finished_at=?, "
                "expires_at=?, tickets=NULL WHERE job_id=? AND status='running'",
                (code, message, iso(t), iso(t + self.retention), job_id),
            )

    def purge(self):
        t = now()
        with self._conn() as c:
            c.execute("UPDATE jobs SET results=NULL, tickets=NULL WHERE expires_at <= ?", (iso(t),))
            c.execute("DELETE FROM jobs WHERE expires_at <= ?", (iso(t - timedelta(days=TOMBSTONE_DAYS)),))


class JobWorker(threading.Thread):
    def __init__(self, store: JobStore, get_predictor):
        super().__init__(daemon=True, name="job-worker")
        self.store = store
        self.get_predictor = get_predictor
        self.stop_event = threading.Event()

    def run(self):
        last_purge = 0.0
        while not self.stop_event.is_set():
            try:
                t = now().timestamp()
                if t - last_purge > PURGE_EVERY_SECONDS:
                    self.store.purge()
                    last_purge = t
                predictor = self.get_predictor()
                job = self.store.claim_next() if predictor is not None else None
                if job is None:
                    self.stop_event.wait(POLL_SECONDS)
                    continue
                self._process(job["job_id"], json.loads(job["tickets"]), predictor)
            except Exception:
                log.exception("job worker loop error")
                self.stop_event.wait(POLL_SECONDS)

    def _process(self, job_id: str, tickets: list[dict], predictor):
        try:
            results = []
            for i in range(0, len(tickets), CHUNK_SIZE):
                results.extend(predictor.predict(tickets[i:i + CHUNK_SIZE], background=True))
                if not self.store.progress(job_id, len(results)):
                    return  # deleted while running
            self.store.finish(job_id, results)
        except Exception:
            log.exception("job %s failed", job_id)
            self.store.fail(job_id, "internal_error", "The job failed while classifying tickets.")
