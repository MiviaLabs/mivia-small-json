#!/usr/bin/env python3
"""Arm 3: train + evaluate the is_security gate (skops hashed-word LR).

Trains on data/real/grade-security/gate-train.jsonl, picks tau for F1 on
the same file's tail (dev), and ONE-SHOT evaluates on gate-eval.jsonl
(frozen in the splits). Writes models/is-security-gate.skops + meta.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "data" / "real" / "grade-security"
GATE = ROOT / "models" / "is-security-gate.skops"
GATE_SHA = ROOT / "models" / "is-security-gate.sha256"
META = ROOT / "models" / "is-security-gate-meta.json"
SEED = 42

BAR_F1 = 0.90  # pre-registered (security-severity)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _metrics(y, p, tau):
    pred = (p >= tau).astype(int)
    y = np.asarray(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tau": round(float(tau), 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "accuracy": round(float((pred == y).mean()), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> int:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import skops.io as skio

    train = _rows(SEC / "gate-train.jsonl")
    # in-file dev tail (the eval file stays one-shot)
    cut = int(len(train) * 0.95)
    dev = train[cut:]
    train = train[:cut]

    tr_x = [r["text"] for r in train]
    tr_y = np.asarray([r["label"] for r in train])
    dv_x = [r["text"] for r in dev]
    dv_y = np.asarray([r["label"] for r in dev])

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  max_features=2 ** 15, lowercase=True,
                                  sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=1000, C=8.0, random_state=SEED)),
    ])
    clf.fit(tr_x, tr_y)
    # inference-noop hygiene: sklearn keeps pruned-ngram bookkeeping that
    # dominates the skops dump size (review 2026-10-09: 117 MB -> ~3 MB)
    clf.named_steps["tfidf"].stop_words_ = set()
    dv_p = clf.predict_proba(dv_x)[:, 1]

    best_tau, best_f1 = 0.5, 0.0
    for tau in np.quantile(dv_p, np.linspace(0.05, 0.95, 181)):
        m = _metrics(dv_y, dv_p, tau)
        if m["f1"] > best_f1:
            best_f1, best_tau = m["f1"], float(tau)
    dev_metrics = _metrics(dv_y, dv_p, best_tau)
    print("dev:", json.dumps(dev_metrics))

    skio.dump(clf, GATE)
    GATE_SHA.write_text(
        hashlib.sha256(GATE.read_bytes()).hexdigest() + "\n")

    # ONE-SHOT: frozen gate-eval
    ev = _rows(SEC / "gate-eval.jsonl")
    ev_x = [r["text"] for r in ev]
    ev_y = np.asarray([r["label"] for r in ev])
    ev_p = clf.predict_proba(ev_x)[:, 1]
    ev_metrics = _metrics(ev_y, ev_p, best_tau)
    passed = ev_metrics["f1"] >= BAR_F1
    result = {
        "verdict": "PASS" if passed else "FAIL",
        "bar_f1": BAR_F1,
        "tau": dev_metrics["tau"],
        "dev": dev_metrics,
        "eval": ev_metrics,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }
    META.write_text(json.dumps({
        **result,
        "seed": SEED,
        "artifact_sha256": hashlib.sha256(GATE.read_bytes()).hexdigest(),
    }, indent=1) + "\n")
    out = ROOT / "reports" / "is-security-gate-oneshot.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print("EVAL:", json.dumps(ev_metrics))
    print(f"VERDICT {result['verdict']} (bar F1 >= {BAR_F1}); written {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
