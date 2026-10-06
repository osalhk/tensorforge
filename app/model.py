"""Loads the trained model and turns its outputs into API predictions.

Two model types share the same decision rules:
- a folder with model.onnx + tokenizer.json + config.json  -> fine-tuned XLM-R (notebooks/02_*)
- a .joblib file                                           -> TF-IDF baseline (notebooks/01_*)
"""

import hashlib
import io
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from app.features import ticket_text, transformer_text

TEAMS = {
    "payment_refund": "Payments & Refunds",
    "ride_trip_issue": "Ride Operations",
    "lost_item": "Lost & Found",
    "order_missing_wrong": "Food Operations",
    "delivery_delay": "Delivery Operations",
    "food_quality": "Restaurant Quality",
    "account_promo": "Account Services",
    "safety_conduct": "Trust & Safety",
    "app_technical": "Tech Support",
    "general_inquiry": "Front-line Support",
    "spam_irrelevant": "Auto-close / Spam Filter",
}
NONE = "none"  # label used for "no secondary category" during training

# Tickets whose primary-category confidence falls below this are flagged for a human.
REVIEW_THRESHOLD = 0.5

# Temperature per exact model (notebooks/06_calibrate.py): softens over-confident category probabilities.
CALIBRATION = Path(__file__).with_name("calibration.json")


def apply_temperature(proba, t: float):
    """softmax(logits / t) from probabilities: same argmax, only the confidence changes."""
    q = np.power(np.clip(np.asarray(proba, np.float64), 1e-12, 1), 1 / t)
    return q / q.sum(1, keepdims=True)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def available_cpus() -> int:
    """CPUs this process may really use. Inside Docker with --cpus, os.cpu_count() reports every
    host core, but the cgroup quota only allows a few; running more threads than that makes
    onnxruntime threads fight over the quota and slows everything down several times."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        try:  # cgroup v1
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except (OSError, ValueError):
            pass
    return n


def decide(tickets, cat_proba, cat_classes, sec_proba, sec_classes, urg_proba, urg_threshold, version):
    """Probabilities -> API predictions, applying the consistency rules."""
    out = []
    for t, cp, sp, up in zip(tickets, cat_proba, sec_proba, urg_proba):
        i = int(np.argmax(cp))
        category = str(cat_classes[i])
        confidence = round(float(cp[i]), 4)

        # Secondary must differ from the primary: take the best label other than it.
        sp = np.array(sp, dtype=np.float64)
        if category in sec_classes:
            sp[sec_classes.index(category)] = -1
        secondary = sec_classes[int(np.argmax(sp))]
        secondary = None if secondary == NONE else secondary
        is_urgent = bool(up >= urg_threshold)

        if category == "spam_irrelevant":
            secondary, is_urgent = None, False

        pred = {}
        if "ticket_id" in t:
            pred["ticket_id"] = t["ticket_id"]
        pred.update({
            "category": category,
            "secondary_category": secondary,
            "team": TEAMS[category],
            "is_urgent": is_urgent,
            "confidence": confidence,
            "model_version": version,
            "needs_human_review": confidence < REVIEW_THRESHOLD,
        })
        out.append(pred)
    return out


class BaselinePredictor:
    def __init__(self, path: Path):
        import joblib

        raw = path.read_bytes()
        bundle = joblib.load(io.BytesIO(raw))
        self.version = f"{bundle.get('name', path.stem)}-{hashlib.sha256(raw).hexdigest()[:12]}"
        self.cat_model = bundle["category"]
        self.sec_model = bundle["secondary"]
        self.urg_model = bundle["urgent"]
        self.urg_threshold = bundle["urgent_threshold"]

    def predict(self, tickets: list[dict], background: bool = False) -> list[dict]:
        X = [ticket_text(t["channel"], t.get("subject"), t["text"]) for t in tickets]
        urg = self.urg_model.predict_proba(X)[:, list(self.urg_model.classes_).index(True)]
        return decide(tickets, self.cat_model.predict_proba(X), self.cat_model.classes_,
                      self.sec_model.predict_proba(X), list(self.sec_model.classes_),
                      urg, self.urg_threshold, self.version)


class TransformerPredictor:
    BATCH = 8             # live requests: small, length-sorted batches waste little time on padding
    BACKGROUND_BATCH = 2  # async jobs: tiny steps, so a live request never waits long


    def __init__(self, folder: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        cfg = json.loads((folder / "config.json").read_text())
        if cfg.get("text_format") != "transformer_text_v1":
            raise ValueError(f"unsupported text_format {cfg.get('text_format')!r}")
        self.categories = cfg["categories"]
        self.secondary = cfg["secondary"]
        self.urg_threshold = cfg["urgent_threshold"]
        self.version = f"{cfg.get('name', folder.name)}-{file_hash(folder / 'model.onnx')[:12]}"
        calibration = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
        self.temperature = calibration.get(self.version, {}).get("category_temperature")
        if self.temperature:
            self.version += f"-T{self.temperature:.2f}"

        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(cfg["max_len"])
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")

        opts = ort.SessionOptions()
        self.threads = int(os.environ.get("ORT_THREADS") or available_cpus())
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])

        # Live requests (/predict, /predict/batch) have priority over async jobs: the CPU is
        # shared, and /predict must stay under 1 s while a job runs. Runs are serialised; a job
        # step waits while any live request is pending, so live work waits for at most one step.
        self._run_lock = threading.Lock()
        self._live_lock = threading.Lock()
        self._live = 0
        self._no_live = threading.Event()
        self._no_live.set()

    @contextmanager
    def _live_request(self):
        with self._live_lock:
            self._live += 1
            self._no_live.clear()
        try:
            yield
        finally:
            with self._live_lock:
                self._live -= 1
                if self._live == 0:
                    self._no_live.set()

    def probabilities(self, tickets: list[dict], background: bool = False):
        X = [transformer_text(t["channel"], t.get("subject"), t["text"]) for t in tickets]
        n = len(X)
        cat = np.zeros((n, len(self.categories)), np.float32)
        sec = np.zeros((n, len(self.secondary)), np.float32)
        urg = np.zeros(n, np.float32)
        order = np.argsort([len(x) for x in X], kind="stable")
        step = self.BACKGROUND_BATCH if background else self.BATCH
        for i in range(0, n, step):
            idx = order[i:i + step]
            encs = self.tokenizer.encode_batch([X[j] for j in idx])
            feed = {"input_ids": np.array([e.ids for e in encs], np.int64),
                    "attention_mask": np.array([e.attention_mask for e in encs], np.int64)}
            if background:
                self._no_live.wait()
            with self._run_lock:
                c, s, u = self.session.run(None, feed)
            cat[idx], sec[idx], urg[idx] = c, s, u.reshape(-1)
        return cat, sec, urg

    def predict(self, tickets: list[dict], background: bool = False) -> list[dict]:
        if background:
            cat, sec, urg = self.probabilities(tickets, background=True)
        else:
            with self._live_request():
                cat, sec, urg = self.probabilities(tickets)
        if self.temperature:
            cat = apply_temperature(cat, self.temperature)
        return decide(tickets, cat, self.categories, sec, self.secondary, urg, self.urg_threshold, self.version)


def load_predictor(path: Path):
    if path.is_dir():
        return TransformerPredictor(path)
    return BaselinePredictor(path)

