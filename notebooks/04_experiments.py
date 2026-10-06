"""Experiment log: how we got from a simple TF-IDF model to the fine-tuned XLM-R model.

    .venv/bin/python notebooks/04_experiments.py

1. Re-runs the classical category experiments (5-fold CV on train, then fit on train and score validation).
2. Scores the saved baseline (3 heads, consistency rules) on validation.
3. Adds the XLM-R runs recorded by notebooks/02_transformer_colab.ipynb (model/xlmr/history.json,
   model/xlmr/config.json) and the int8 quantisation result noted in that notebook.

Writes reports/experiments.csv and reports/figures/exp_*.png. Seeds are fixed (42).
"""

import csv
import json
import sys
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from report_style import BLUE, ORANGE, REPORTS, ROOT, TEXT_2, hbar, plt, save

sys.path.insert(0, str(ROOT))
from app.features import ticket_text  # noqa: E402
from app.model import BaselinePredictor  # noqa: E402

SEED = 42
FIELDS = ["experiment", "model", "features", "params", "cv5_category_acc", "val_category_acc",
          "val_category_macro_f1", "val_secondary_macro_f1", "val_urgent_f1", "val_all_three", "notes"]


def load(name):
    with open(ROOT / "data" / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def word():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b")


def char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                           max_features=300_000)


def both():
    return FeatureUnion([("char", char()), ("word", word())])


def lr(C=10, balanced=True):
    return LogisticRegression(C=C, class_weight="balanced" if balanced else None, max_iter=3000, random_state=SEED)


# name, features label, params label, pipeline factory, input builder, note
FULL = lambda t: ticket_text(t["channel"], t.get("subject"), t["text"])  # noqa: E731
TEXT_ONLY = lambda t: t["text"]  # noqa: E731
CLASSICAL = [
    ("E01", "word 1-2", "LR C=10, no class weights", lambda: make_pipeline(word(), lr(10, False)), FULL,
     "words only"),
    ("E02", "char 2-5", "LR C=10, no class weights", lambda: make_pipeline(char(), lr(10, False)), FULL,
     "character n-grams handle spelling variation and Sinhala/Tamil script"),
    ("E03", "char 2-5 + word 1-2", "LR C=10, no class weights", lambda: make_pipeline(both(), lr(10, False)), FULL,
     "both feature sets"),
    ("E04", "char 2-5 + word 1-2", "LR C=10, balanced", lambda: make_pipeline(both(), lr(10)), FULL,
     "class weights for rare classes (safety, spam); chosen baseline"),
    ("E05", "char 2-5 + word 1-2", "LR C=1, balanced", lambda: make_pipeline(both(), lr(1)), FULL,
     "stronger regularisation"),
    ("E06", "char 2-5 + word 1-2", "LR C=30, balanced", lambda: make_pipeline(both(), lr(30)), FULL,
     "weaker regularisation"),
    ("E07", "char 2-5 + word 1-2", "LinearSVC C=0.5, balanced",
     lambda: make_pipeline(both(), LinearSVC(C=0.5, class_weight="balanced", random_state=SEED)), FULL,
     "linear SVM instead of LR (no probabilities, so no calibrated confidence)"),
    ("E08", "char 2-5 + word 1-2 (text only)", "LR C=10, balanced", lambda: make_pipeline(both(), lr(10)), TEXT_ONLY,
     "drops channel and subject: shows they help"),
]


def r3(x):
    return None if x is None else round(float(x), 4)


def main():
    train, val = load("train.jsonl"), load("validation.jsonl")
    y_tr = np.array([t["category"] for t in train])
    y_va = np.array([t["category"] for t in val])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []

    print("Classical category experiments (5-fold CV on train + validation):")
    for name, feats, params, factory, build, note in CLASSICAL:
        t0 = time.time()
        X_tr, X_va = [build(t) for t in train], [build(t) for t in val]
        cv_acc = cross_val_score(factory(), X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
        pred = factory().fit(X_tr, y_tr).predict(X_va)
        rows.append({"experiment": name, "model": "tfidf", "features": feats, "params": params,
                     "cv5_category_acc": f"{cv_acc.mean():.4f} ± {cv_acc.std():.4f}",
                     "val_category_acc": r3(accuracy_score(y_va, pred)),
                     "val_category_macro_f1": r3(f1_score(y_va, pred, average="macro")), "notes": note})
        print(f"  {name} {feats:32s} {params:28s} cv {cv_acc.mean():.3f}  val {rows[-1]['val_category_acc']:.3f}"
              f"  ({time.time() - t0:.0f}s)")

    print("Saved 3-head baseline (model/baseline.joblib) on validation:")
    bundle = joblib.load(ROOT / "model" / "baseline.joblib")
    preds = BaselinePredictor(ROOT / "model" / "baseline.joblib").predict(val)
    rows.append({"experiment": "B1", "model": "tfidf 3 heads (served fallback)", "features": "char 2-5 + word 1-2",
                 "params": f"LR C=10 balanced x3, urgent threshold {bundle['urgent_threshold']:.2f} (5-fold CV)",
                 **full_scores(preds, val), "notes": "category + secondary + urgent with API consistency rules"})
    print(f"  B1 val category acc {rows[-1]['val_category_acc']:.3f}")

    history = json.loads((ROOT / "model" / "xlmr" / "history.json").read_text())
    for seed, epochs in history.items():
        for e in epochs:
            rows.append({"experiment": f"X-s{seed}-e{e['epoch']}", "model": "xlm-roberta-base, 3 heads",
                         "features": "channel | subject + text, max 256 tokens",
                         "params": f"seed {seed}, epoch {e['epoch']}, LR 2e-5, class weights",
                         "val_category_acc": r3(e["category_acc"]), "val_category_macro_f1": r3(e["category_f1"]),
                         "val_secondary_macro_f1": r3(e["secondary_f1"]), "val_urgent_f1": r3(e["urgent_f1"]),
                         "val_all_three": r3(e["all_three"]), "notes": "per-epoch validation, from Colab history.json"})
    rows.append({"experiment": "X-int8", "model": "xlm-roberta-base int8 ONNX", "features": "same",
                 "params": "dynamic int8 quantisation (several settings)", "val_category_acc": "0.734-0.746",
                 "notes": "earlier run; fp32 of the same run 0.775. Rejected: -3 to -4 points (notebook section 9)"})
    cfg = json.loads((ROOT / "model" / "xlmr" / "config.json").read_text())
    s = cfg["onnx_validation_scores"]
    rows.append({"experiment": "FINAL", "model": "xlm-roberta-base fp32 ONNX (served)", "features": "same",
                 "params": f"seed {cfg['seed']}, best epoch {cfg['best_epoch']}, urgent threshold {cfg['urgent_threshold']:.2f}",
                 "val_category_acc": r3(s["category_acc"]), "val_category_macro_f1": r3(s["category_f1"]),
                 "val_secondary_macro_f1": r3(s["secondary_f1"]), "val_urgent_f1": r3(s["urgent_f1"]),
                 "val_all_three": r3(s["all_three"]),
                 "notes": f"model_version xlmr-{cfg['model_sha256'][:12]}; epoch and threshold picked on validation"})

    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "experiments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("  wrote reports/experiments.csv")
    plot_progress(rows)
    plot_training(history)


def full_scores(preds, val):
    y_cat = np.array([t["category"] for t in val])
    y_sec = np.array([t["secondary_category"] or "none" for t in val])
    y_urg = np.array([t["is_urgent"] for t in val])
    p_cat = np.array([p["category"] for p in preds])
    p_sec = np.array([p["secondary_category"] or "none" for p in preds])
    p_urg = np.array([p["is_urgent"] for p in preds])
    return {"val_category_acc": r3(accuracy_score(y_cat, p_cat)),
            "val_category_macro_f1": r3(f1_score(y_cat, p_cat, average="macro")),
            "val_secondary_macro_f1": r3(f1_score(y_sec, p_sec, average="macro")),
            "val_urgent_f1": r3(f1_score(y_urg, p_urg)),
            "val_all_three": r3(np.mean((p_cat == y_cat) & (p_sec == y_sec) & (p_urg == y_urg)))}


def plot_progress(rows):
    by = {r["experiment"]: r for r in rows}
    steps = [("E01", "Words only (TF-IDF + LR)"), ("E02", "Character n-grams"), ("E03", "Words + characters"),
             ("E04", "+ class weights (baseline)"), ("E07", "Linear SVM instead"),
             ("FINAL", "Fine-tuned XLM-R (final)")]
    labels = [s[1] for s in steps]
    values = [float(by[s[0]]["val_category_acc"]) * 100 for s in steps]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    hbar(ax, labels, values, color=[ORANGE] * (len(steps) - 1) + [BLUE], fmt="{:.1f}%")
    ax.set_xlim(0, 100)
    ax.set_title("Category accuracy on validation, by experiment")
    ax.set_xlabel("% correct (800 validation tickets)")
    save(fig, "exp_progress.png")


def plot_training(history):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (seed, epochs), color in zip(history.items(), [BLUE, ORANGE]):
        x = [e["epoch"] for e in epochs]
        y = [e["category_acc"] * 100 for e in epochs]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=f"seed {seed}")
    cfg = json.loads((ROOT / "model" / "xlmr" / "config.json").read_text())
    best = history[str(cfg["seed"])][cfg["best_epoch"] - 1]
    ax.annotate(f"served: seed {cfg['seed']}, epoch {cfg['best_epoch']}", (best["epoch"], best["category_acc"] * 100),
                xytext=(0, -38), textcoords="offset points", ha="center", fontsize=8.5, color=TEXT_2,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, linewidth=0.8))
    ax.set_xlabel("epoch")
    ax.set_ylabel("category accuracy (%)")
    ax.set_title("XLM-R fine-tuning: validation accuracy per epoch")
    ax.legend(loc="lower right")
    ax.set_xlim(0.7, max(len(v) for v in history.values()) + 0.3)
    save(fig, "exp_xlmr_training.png")


if __name__ == "__main__":
    main()
