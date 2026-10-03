"""Loads the trained model and turns its outputs into API predictions.

Two model types share the same decision rules:
- a folder with model.onnx + tokenizer.json + config.json  -> fine-tuned XLM-R (notebooks/02_*)
- a .joblib file                                           -> TF-IDF baseline (notebooks/01_*)
"""

import hashlib
import io
import json
import os
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


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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

    def predict(self, tickets: list[dict]) -> list[dict]:
        X = [ticket_text(t["channel"], t.get("subject"), t["text"]) for t in tickets]
        urg = self.urg_model.predict_proba(X)[:, list(self.urg_model.classes_).index(True)]
        return decide(tickets, self.cat_model.predict_proba(X), self.cat_model.classes_,
                      self.sec_model.predict_proba(X), list(self.sec_model.classes_),
                      urg, self.urg_threshold, self.version)


class TransformerPredictor:
    BATCH = 8  # small, length-sorted batches waste little time on padding

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

        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(cfg["max_len"])
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")

        opts = ort.SessionOptions()
        # 0 = let onnxruntime use every available core; set ORT_THREADS to pin it.
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))
        self.session = ort.InferenceSession(str(folder / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])

    def probabilities(self, tickets: list[dict]):
        X = [transformer_text(t["channel"], t.get("subject"), t["text"]) for t in tickets]
        n = len(X)
        cat = np.zeros((n, len(self.categories)), np.float32)
        sec = np.zeros((n, len(self.secondary)), np.float32)
        urg = np.zeros(n, np.float32)
        order = np.argsort([len(x) for x in X], kind="stable")
        for i in range(0, n, self.BATCH):
            idx = order[i:i + self.BATCH]
            encs = self.tokenizer.encode_batch([X[j] for j in idx])
            feed = {"input_ids": np.array([e.ids for e in encs], np.int64),
                    "attention_mask": np.array([e.attention_mask for e in encs], np.int64)}
            c, s, u = self.session.run(None, feed)
            cat[idx], sec[idx], urg[idx] = c, s, u.reshape(-1)
        return cat, sec, urg

    def predict(self, tickets: list[dict]) -> list[dict]:
        cat, sec, urg = self.probabilities(tickets)
        return decide(tickets, cat, self.categories, sec, self.secondary, urg, self.urg_threshold, self.version)


def load_predictor(path: Path):
    if path.is_dir():
        return TransformerPredictor(path)
    return BaselinePredictor(path)

