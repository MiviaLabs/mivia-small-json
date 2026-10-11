#!/usr/bin/env python3
"""Register gate for the grade severity router (TRACK: register-router).

A cheap CPU binary classifier — TF-IDF (word 1-2 + char 3-5) + logistic
regression — deciding whether a text is BUSINESS-register phrasing
(route to the specialist severity head) or canonical register (route to
the shipped head). Positives: the combined specialist pool
(`data/incoming/grade-teacher-model/specialist_pool.jsonl`, session-LLM
+ Qwen teacher rows, provenance in that directory). Negatives: a seeded,
tier-stratified sample of canonical grade-train rows.

Threshold (tau) is chosen on a seeded DEV SPLIT ONLY, precision-first:
smallest tau achieving dev precision >= 0.98 — a canonical row misrouted
to the specialist regresses the panel, so precision dominates. The frozen
panel and business slice are never consulted here.

Persistence: skops only (pickle is banned), sha256 sidecar. Candidate
artifact under gitignored data/incoming/models/ until the router bar
clears; promotion to models/ is a separate, manifest-refreshed commit.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

POOL = ROOT / "data" / "incoming" / "grade-teacher-model" / "specialist_pool.jsonl"
GRADE_TRAIN = ROOT / "data" / "real" / "grade" / "train.jsonl"
CANDIDATE = ROOT / "data" / "incoming" / "models" / "register-gate.skops"
CANDIDATE_TAU = ROOT / "data" / "incoming" / "models" / "register-gate.meta.json"

TAU_PRECISION_FLOOR = 0.98
SEED = 11


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def build_frames(seed: int = SEED) -> tuple[list[str], list[int], list[str], list[int]]:
    """(train texts, train labels, dev texts, dev labels). 1 = business."""
    import random

    rng = random.Random(seed)
    pos = []
    seen: set[str] = set()
    for line in POOL.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n = _norm(row["text"])
        if n in seen:
            continue
        seen.add(n)
        pos.append(row["text"])
    by_tier: dict[int, list[str]] = {}
    for line in GRADE_TRAIN.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n = _norm(row["text"])
        if n in seen:
            continue
        by_tier.setdefault(row["gold"], []).append(row["text"])
    n_neg = len(pos)
    per_tier = max(1, n_neg // len(by_tier))
    neg: list[str] = []
    neg_seen: set[str] = set()
    for tier in sorted(by_tier):
        cand = by_tier[tier][:]
        rng.shuffle(cand)
        taken = 0
        for t in cand:
            if taken >= per_tier or len(neg) >= n_neg:
                break
            n = _norm(t)
            if n in neg_seen:
                continue
            neg_seen.add(n)
            neg.append(t)
            taken += 1
    rng.shuffle(neg)
    neg = neg[:n_neg]

    texts = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    order = list(range(len(texts)))
    rng.shuffle(order)
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    cut = (len(texts) * 4) // 5
    return texts[:cut], labels[:cut], texts[cut:], labels[cut:]


def fit(texts: list[str], labels: list[int]) -> tuple:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    pipe = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), analyzer="word",
                                     min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb",
                                     min_df=2, sublinear_tf=True)),
        ])),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
    ])
    pipe.fit(texts, labels)
    return pipe


def choose_tau(pipe, dev_texts: list[str], dev_labels: list[int]) -> float:
    """The pre-registered rule (ticket 03, spec): the SMALLEST tau that
    achieves dev precision >= floor — i.e. max dev recall subject to the
    precision floor; ties keep the lowest tau.

    Sensitivity note (review finding, 2026-10-07): the alternative
    precision-margin tie-break would select tau 0.6537 (dev precision
    1.000, recall 1.000) — but at that threshold the router FAILS its
    pre-registered bar (slice 46.7% < 50, panel 68.7% < 69.0; see
    reports/grade-register-router.md). The spec-literal rule was the one
    registered and measured before promotion, so it stands; the
    alternative is recorded as a sensitivity analysis, not adopted
    post-hoc.
    """
    from sklearn.metrics import precision_score, recall_score

    prob = pipe.predict_proba(dev_texts)[:, 1]
    candidates = sorted({round(float(p), 4) for p in prob}, reverse=True)
    best: tuple[float, float] | None = None  # (recall, tau)
    for tau in candidates:
        pred = (prob >= tau).astype(int)
        if not pred.any():
            continue
        prec = precision_score(dev_labels, pred, zero_division=0)
        if prec < TAU_PRECISION_FLOOR:
            continue
        rec = recall_score(dev_labels, pred, zero_division=0)
        if best is None or rec > best[0] or (rec == best[0] and tau < best[1]):
            best = (rec, tau)
    if best is None:
        return 0.9999
    return best[1]


def save(pipe, tau: float, meta: dict) -> None:
    import skops.io as sio

    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(pipe, str(CANDIDATE))
    digest = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    (CANDIDATE.parent / (CANDIDATE.name + ".sha256")).write_text(
        digest + "  " + CANDIDATE.name + "\n")
    meta["tau"] = tau
    meta["sha256"] = digest[:16]
    CANDIDATE_TAU.write_text(json.dumps(meta, indent=2) + "\n")


def main() -> int:
    train_t, train_y, dev_t, dev_y = build_frames()
    print(f"train {len(train_t)} (pos {sum(train_y)}) / dev {len(dev_t)} "
          f"(pos {sum(dev_y)})")
    pipe = fit(train_t, train_y)
    from sklearn.metrics import precision_score, recall_score

    prob = pipe.predict_proba(dev_t)[:, 1]
    tau = choose_tau(pipe, dev_t, dev_y)
    pred = (prob >= tau).astype(int)
    print(f"tau {tau:.4f} · dev precision "
          f"{precision_score(dev_y, pred, zero_division=0):.3f} · recall "
          f"{recall_score(dev_y, pred, zero_division=0):.3f}")
    meta = {"tau_dev_chosen": tau, "precision_floor": TAU_PRECISION_FLOOR,
            "seed": SEED, "train_n": len(train_t), "dev_n": len(dev_t),
            "train_positives": sum(train_y), "dev_positives": sum(dev_y),
            "negatives": len(train_t) - sum(train_y),
            "sklearn_version": __import__("sklearn").__version__,
            "positives": POOL.name}
    save(pipe, tau, meta)
    print(f"candidate -> {CANDIDATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
