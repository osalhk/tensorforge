"""Baseline: TF-IDF (char + word n-grams) + logistic regression.

Trains three models on data/train.jsonl and scores them on data/validation.jsonl:
  1. category            (11 classes)
  2. secondary_category  (5 classes + "none")
  3. is_urgent           (binary, threshold tuned with cross-validation on train only)

Run from the project root:
    .venv/bin/python notebooks/01_baseline.py
Saves the fitted models to model/baseline.joblib.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.features import ticket_text  # noqa: E402  (same input builder as the API)

NONE = "none"
SEED = 42


def load(name):
    with open(ROOT / "data" / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_text(t):
    # Only channel, subject and text: the API never sends `language`.
    return ticket_text(t["channel"], t.get("subject"), t["text"])


def make_model():
    features = FeatureUnion([
        # Character n-grams cope with spelling variation and Sinhala/Tamil script.
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, max_features=300_000)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
    ])
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=3000)
    return make_pipeline(features, clf)


def postprocess(cat, sec, urgent):
    """Apply the API consistency rules."""
    if sec == cat:
        sec = NONE
    if cat == "spam_irrelevant":
        sec, urgent = NONE, False
    return cat, sec, urgent


def main():
    train, val = load("train.jsonl"), load("validation.jsonl")
    Xtr, Xva = [to_text(t) for t in train], [to_text(t) for t in val]
    y_cat_tr = np.array([t["category"] for t in train])
    y_sec_tr = np.array([t["secondary_category"] or NONE for t in train])
    y_urg_tr = np.array([t["is_urgent"] for t in train])

    print("Training category model...")
    cat_model = make_model().fit(Xtr, y_cat_tr)
    print("Training secondary model...")
    sec_model = make_model().fit(Xtr, y_sec_tr)

    # Urgency: pick the probability threshold that maximises F1 using
    # out-of-fold predictions on train, so validation stays untouched.
    print("Tuning urgency threshold (5-fold CV on train)...")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(make_model(), Xtr, y_urg_tr, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    best_t = max(thresholds, key=lambda t: f1_score(y_urg_tr, oof >= t))
    print(f"  best threshold = {best_t:.2f}")
    urg_model = make_model().fit(Xtr, y_urg_tr)

    # ---- Validation ----
    cats = cat_model.predict(Xva)
    secs = sec_model.predict(Xva)
    urgs = urg_model.predict_proba(Xva)[:, 1] >= best_t
    preds = [postprocess(c, s, bool(u)) for c, s, u in zip(cats, secs, urgs)]
    p_cat, p_sec, p_urg = map(np.array, zip(*preds))

    y_cat = np.array([t["category"] for t in val])
    y_sec = np.array([t["secondary_category"] or NONE for t in val])
    y_urg = np.array([t["is_urgent"] for t in val])

    print("\n================ VALIDATION (800 tickets) ================")
    print(f"category   accuracy {accuracy_score(y_cat, p_cat):.3f}   macro-F1 {f1_score(y_cat, p_cat, average='macro'):.3f}")
    print(f"secondary  accuracy {accuracy_score(y_sec, p_sec):.3f}   macro-F1 {f1_score(y_sec, p_sec, average='macro'):.3f}")
    print(f"urgent     accuracy {accuracy_score(y_urg, p_urg):.3f}   F1 {f1_score(y_urg, p_urg):.3f}")
    all3 = np.mean((p_cat == y_cat) & (p_sec == y_sec) & (p_urg == y_urg))
    print(f"all three correct: {all3:.3f}")

    print("\n--- category per class ---")
    print(classification_report(y_cat, p_cat, digits=3))
    print("--- urgent ---")
    print(classification_report(y_urg, p_urg, digits=3))

    print("--- category accuracy by language ---")
    langs = np.array([t["language"] for t in val])
    for lang in ["en", "singlish", "si", "ta", "tanglish", "mixed"]:
        m = langs == lang
        print(f"  {lang:9s} n={m.sum():4d}  acc {accuracy_score(y_cat[m], p_cat[m]):.3f}")

    out = ROOT / "model" / "baseline.joblib"
    joblib.dump({"name": "baseline", "category": cat_model, "secondary": sec_model, "urgent": urg_model,
                 "urgent_threshold": float(best_t)}, out)
    print(f"\nSaved model to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
