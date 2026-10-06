"""Evaluate the served models on the 800 validation tickets, through the same code the API uses.

    .venv/bin/python notebooks/05_evaluate.py

Loads model/xlmr (served) and model/baseline.joblib (fallback) with app.model.load_predictor, so the
consistency rules, team table, confidence and needs_human_review are exactly what /predict returns.
Writes reports/metrics.json, reports/classification_report.csv and reports/figures/eval_*.png.
"""

import csv
import json
import sys
import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support)

from report_style import BLUE, GRID, ORANGE, REPORTS, ROOT, TEXT, TEXT_2, plt, save

sys.path.insert(0, str(ROOT))
from app.model import REVIEW_THRESHOLD, TEAMS, load_predictor  # noqa: E402

CATEGORIES = list(TEAMS)
LANG_ORDER = ["en", "singlish", "si", "ta", "tanglish", "mixed"]
MODELS = {"xlmr": ROOT / "model" / "xlmr", "baseline": ROOT / "model" / "baseline.joblib"}


def load(name):
    with open(ROOT / "data" / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(preds, val):
    y_cat = np.array([t["category"] for t in val])
    y_sec = np.array([t["secondary_category"] or "none" for t in val])
    y_urg = np.array([t["is_urgent"] for t in val])
    p_cat = np.array([p["category"] for p in preds])
    p_sec = np.array([p["secondary_category"] or "none" for p in preds])
    p_urg = np.array([p["is_urgent"] for p in preds])
    conf = np.array([p["confidence"] for p in preds])
    langs = np.array([t["language"] for t in val])
    correct = p_cat == y_cat
    review = conf < REVIEW_THRESHOLD
    up, ur, uf, _ = precision_recall_fscore_support(y_urg, p_urg, average="binary")

    # Expected calibration error: gap between confidence and accuracy, over 10 confidence bins.
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    rules_ok = all(p["team"] == TEAMS[p["category"]] and p["secondary_category"] != p["category"]
                   and (p["category"] != "spam_irrelevant" or (not p["is_urgent"] and p["secondary_category"] is None))
                   for p in preds)
    return {
        "model_version": preds[0]["model_version"],
        "category_accuracy": accuracy_score(y_cat, p_cat),
        "category_macro_f1": f1_score(y_cat, p_cat, average="macro"),
        "team_accuracy": float(np.mean([p["team"] == TEAMS[c] for p, c in zip(preds, y_cat)])),
        "secondary_accuracy": accuracy_score(y_sec, p_sec),
        "secondary_macro_f1": f1_score(y_sec, p_sec, average="macro"),
        "urgent_precision": up, "urgent_recall": ur, "urgent_f1": uf,
        "all_three_correct": float(np.mean(correct & (p_sec == y_sec) & (p_urg == y_urg))),
        "mean_confidence": float(conf.mean()),
        "expected_calibration_error": float(ece),
        "needs_human_review": {
            "threshold": REVIEW_THRESHOLD, "share_flagged": float(review.mean()),
            "accuracy_auto_routed": float(correct[~review].mean()),
            "accuracy_flagged": float(correct[review].mean()) if review.any() else None,
        },
        "consistency_rules_hold": rules_ok,
        "category_accuracy_by_language": {lang: float(correct[langs == lang].mean()) for lang in LANG_ORDER},
        "_arrays": (y_cat, p_cat, conf, correct),
    }


def main():
    val = load("validation.jsonl")
    tickets = [{k: t[k] for k in ("ticket_id", "channel", "subject", "text")} for t in val]  # what the API sees
    results = {}
    for name, path in MODELS.items():
        t0 = time.time()
        preds = load_predictor(path).predict(tickets)
        results[name] = evaluate(preds, val)
        print(f"{name}: category acc {results[name]['category_accuracy']:.3f} on {len(val)} tickets "
              f"({time.time() - t0:.0f}s)")

    REPORTS.mkdir(exist_ok=True)
    clean = {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in results.items()}
    clean = json.loads(json.dumps(clean, default=float), parse_float=lambda x: round(float(x), 4))
    (REPORTS / "metrics.json").write_text(json.dumps({"validation_tickets": len(val), **clean}, indent=2) + "\n")
    print("  wrote reports/metrics.json")

    with open(REPORTS / "classification_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "category", "team", "precision", "recall", "f1", "support"])
        for name, r in results.items():
            y, p = r["_arrays"][:2]
            prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=CATEGORIES, zero_division=0)
            for row in zip(CATEGORIES, prec, rec, f1, sup):
                w.writerow([name, row[0], TEAMS[row[0]], *(round(float(x), 4) for x in row[1:4]), int(row[4])])
    print("  wrote reports/classification_report.csv")

    plot_confusion(results["xlmr"])
    plot_grouped("eval_per_class_f1.png", "F1 per category on validation", CATEGORIES,
                 {n: precision_recall_fscore_support(*r["_arrays"][:2], labels=CATEGORIES, zero_division=0)[2]
                  for n, r in results.items()}, "F1")
    plot_grouped("eval_accuracy_by_language.png", "Category accuracy by language on validation", LANG_ORDER,
                 {n: [r["category_accuracy_by_language"][lang] for lang in LANG_ORDER] for n, r in results.items()},
                 "accuracy")
    plot_calibration(results["xlmr"])


def plot_confusion(r):
    y, p = r["_arrays"][:2]
    cm = confusion_matrix(y, p, labels=CATEGORIES)
    share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45, ha="right")
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if share[i, j] > 0.55 else TEXT)
    ax.set_xlabel("predicted category")
    ax.set_ylabel("true category")
    ax.set_title(f"Confusion matrix, fine-tuned XLM-R ({r['category_accuracy']:.1%} correct)")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "eval_confusion_matrix.png")


def plot_grouped(name, title, labels, series, xlabel):
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    h = 0.38
    for k, ((model, values), color) in enumerate(zip([("xlmr", series["xlmr"]), ("baseline", series["baseline"])],
                                                     [BLUE, ORANGE])):
        label = "fine-tuned XLM-R" if model == "xlmr" else "TF-IDF baseline"
        ax.barh(y + (h / 2 if k == 0 else -h / 2) + 0.01 * (1 if k == 0 else -1), values, height=h - 0.02,
                color=color, label=label)
        if model == "xlmr":
            for yi, v in zip(y, values):
                ax.text(v, yi + h / 2, f" {v:.2f}", va="center", fontsize=8, color=TEXT_2)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.3)
    save(fig, name)


def plot_calibration(r):
    _, _, conf, correct = r["_arrays"]
    edges = np.linspace(0, 1, 11)
    centers, acc, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo)
        if m.sum() >= 5:
            centers.append(conf[m].mean())
            acc.append(correct[m].mean())
            counts.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--", label="perfect calibration")
    ax.plot(centers, acc, color=BLUE, linewidth=2, marker="o", markersize=6, label="fine-tuned XLM-R")
    for x, a, n in zip(centers, acc, counts):
        ax.annotate(f"n={n}", (x, a), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7.5,
                    color=TEXT_2)
    ax.axvline(REVIEW_THRESHOLD, color=ORANGE, linewidth=1.5)
    ax.text(REVIEW_THRESHOLD + 0.01, 0.04, "needs_human_review\nbelow 0.5", color=TEXT_2, fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("confidence returned by the API")
    ax.set_ylabel("share actually correct")
    ax.set_title(f"Does confidence mean what it says? (ECE {r['expected_calibration_error']:.3f})")
    ax.legend(loc="upper left")
    save(fig, "eval_calibration.png")


if __name__ == "__main__":
    main()
