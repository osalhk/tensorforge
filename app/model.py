"""Loads the trained model and turns its outputs into API predictions."""

import hashlib
import io
from pathlib import Path

import joblib
import numpy as np

from app.features import ticket_text

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


class Predictor:
    def __init__(self, path: Path):
        raw = path.read_bytes()
        bundle = joblib.load(io.BytesIO(raw))
        # Version = model name + file hash, so it always points at an exact artifact.
        self.version = f"{bundle.get('name', path.stem)}-{hashlib.sha256(raw).hexdigest()[:12]}"
        self.cat_model = bundle["category"]
        self.sec_model = bundle["secondary"]
        self.urg_model = bundle["urgent"]
        self.urg_threshold = bundle["urgent_threshold"]

    def predict(self, tickets: list[dict]) -> list[dict]:
        """tickets: validated dicts with channel, subject, text and optional ticket_id."""
        X = [ticket_text(t["channel"], t.get("subject"), t["text"]) for t in tickets]
        cat_proba = self.cat_model.predict_proba(X)
        sec_proba = self.sec_model.predict_proba(X)
        urg_proba = self.urg_model.predict_proba(X)[:, list(self.urg_model.classes_).index(True)]
        cat_classes = self.cat_model.classes_
        sec_classes = list(self.sec_model.classes_)

        out = []
        for t, cp, sp, up in zip(tickets, cat_proba, sec_proba, urg_proba):
            i = int(np.argmax(cp))
            category = str(cat_classes[i])
            confidence = round(float(cp[i]), 4)

            # Secondary must differ from the primary: take the best label other than it.
            if category in sec_classes:
                sp = sp.copy()
                sp[sec_classes.index(category)] = -1
            secondary = sec_classes[int(np.argmax(sp))]
            secondary = None if secondary == NONE else secondary
            is_urgent = bool(up >= self.urg_threshold)

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
                "model_version": self.version,
                "needs_human_review": confidence < REVIEW_THRESHOLD,
            })
            out.append(pred)
        return out
