"""Calibrate the XLM-R category confidence with temperature scaling.

    .venv/bin/python notebooks/06_calibrate.py

The fine-tuned model is over-confident (mean confidence 0.96, accuracy 0.85). Temperature scaling divides the
logits by one number T > 1 before the softmax. It never changes which category wins, only how sure the model
says it is. Since log p = logits - const, softmax(logits / T) == p ** (1 / T), renormalised.

T is chosen to minimise negative log-likelihood. The honest effect is measured with 5-fold cross-validation
inside the 800 validation tickets (fit T on 4/5, score the held-out 1/5); the final T is then fitted on all 800.
Writes app/calibration.json (keyed by the model hash, so it only applies to that exact model)
and reports/calibration.json.
"""

import json
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

from report_style import REPORTS, ROOT

sys.path.insert(0, str(ROOT))
from app.model import REVIEW_THRESHOLD, file_hash  # noqa: E402

XLMR = ROOT / "model" / "xlmr"
GRID = np.round(np.arange(0.5, 6.001, 0.01), 2)
SEED = 42


def scale(p, t):
    q = np.power(np.clip(p, 1e-12, 1), 1 / t)
    return q / q.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def ece(p, y):
    conf, correct = p.max(1), p.argmax(1) == y
    bins = np.minimum((conf * 10).astype(int), 9)
    return float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(10) if (bins == b).any()))


def fit(p, y):
    return float(GRID[np.argmin([nll(scale(p, t), y) for t in GRID])])


def summary(p, y):
    conf, correct = p.max(1), p.argmax(1) == y
    review = conf < REVIEW_THRESHOLD
    return {"nll": nll(p, y), "ece": ece(p, y), "mean_confidence": float(conf.mean()),
            "accuracy": float(correct.mean()), "share_flagged_for_review": float(review.mean()),
            "accuracy_of_flagged": float(correct[review].mean()) if review.any() else None,
            "accuracy_of_auto_routed": float(correct[~review].mean())}


def main():
    cfg = json.loads((XLMR / "config.json").read_text())
    saved = np.load(XLMR / "val_probs.npz")
    labels = {json.loads(line)["ticket_id"]: json.loads(line)["category"]
              for line in open(ROOT / "data" / "validation.jsonl", encoding="utf-8")}
    y = np.array([cfg["categories"].index(labels[t]) for t in saved["ticket_id"]])
    p = saved["category"].astype(np.float64)

    # Honest estimate: every ticket is scored with a T fitted without it.
    held_out = np.zeros_like(p)
    temps = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(p, y):
        t = fit(p[tr], y[tr])
        temps.append(t)
        held_out[te] = scale(p[te], t)

    t_final = fit(p, y)
    before, after = summary(p, y), summary(held_out, y)
    assert (held_out.argmax(1) == p.argmax(1)).all(), "temperature must not change predictions"
    print(f"fold temperatures {temps}; final T = {t_final}")
    for name, s in [("before", before), ("after (5-fold held-out)", after)]:
        print(f"  {name:24s} ECE {s['ece']:.3f}  NLL {s['nll']:.3f}  mean conf {s['mean_confidence']:.3f}  "
              f"acc {s['accuracy']:.3f}  flagged {s['share_flagged_for_review']:.1%} "
              f"(acc {s['accuracy_of_flagged'] or 0:.2f})")

    model_version = f"{cfg['name']}-{file_hash(XLMR / 'model.onnx')[:12]}"
    (ROOT / "app" / "calibration.json").write_text(json.dumps({
        model_version: {"category_temperature": t_final,
                        "fitted_on": "800 validation tickets, minimum NLL (notebooks/06_calibrate.py)"}}, indent=2) + "\n")
    REPORTS.mkdir(exist_ok=True)
    rnd = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    (REPORTS / "calibration.json").write_text(json.dumps({
        "model_version": model_version, "temperature": t_final, "fold_temperatures": temps,
        "before": rnd(before), "after_5fold_held_out": rnd(after)}, indent=2) + "\n")
    print("  wrote app/calibration.json and reports/calibration.json")


if __name__ == "__main__":
    main()
