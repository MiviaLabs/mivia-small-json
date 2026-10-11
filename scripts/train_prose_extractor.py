#!/usr/bin/env python3
"""Train the learned prose title extractor (Arm A, prose-extraction).

Training data: the 20k incident-ticket CSV (subject/body gold, en+de) plus
data/real/pull_request train rows — all real, all in-repo. The reality-eval
samples (seed-42 first 1,000 en + 1,000 de tickets) and the pull_request
dev split are REFUSED by text hash: they are eval, never train.

Labels: tokens of the gold title, matched greedily left-to-right (the
training text is ``title\\n\\nbody``, so the first occurrences are the
title). Dev split: pool-policy hash sha256(normalized subject) % 10 == 0.

Model: house pattern — FeatureHasher(2**18) + LogisticRegression, skops
+ sha256 sidecar + trainmeta. CPU training: sklearn has no CUDA path; the
GPU clause covers the transformer graduation if the one-shot bar fails.

Writes models/prose-title-extractor.skops (candidate until promoted by the
one-shot bar; --out controls the destination).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.extract_learned import (  # noqa: E402
    PROSE_TOKEN_RE,
    prose_title_features,
)

CSV_PATH = ROOT / "data/incoming/external/incident-severity/tickets-multi-lang-20k.csv"
PR_TRAIN = ROOT / "data/real/pull_request/train.jsonl"
PR_DEV = ROOT / "data/real/pull_request/dev.jsonl"
DEFAULT_OUT = ROOT / "models" / "prose-title-extractor.skops"
SEED = 42
EVAL_N_PER_LANG = 1000


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _nhash(text: str) -> str:
    return hashlib.sha256(_norm(text).encode()).hexdigest()


def eval_hashes() -> set[str]:
    """Exact replication of bench_reality_extract.tickets() sampling."""
    with CSV_PATH.open(newline="", encoding="utf-8", errors="replace") as fh:
        all_rows = list(csv.DictReader(fh))
    hashes: set[str] = set()
    for lang in ("en", "de"):
        pool = [r for r in all_rows if r.get("language") == lang]
        rng = random.Random(SEED)
        rng.shuffle(pool)
        for r in pool[:EVAL_N_PER_LANG]:
            hashes.add(_nhash(f"{r['subject']}\n\n{r.get('body') or ''}".strip()))
    return hashes


def pr_dev_hashes() -> set[str]:
    return {_nhash(json.loads(line)["raw"])
            for line in PR_DEV.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def load_rows() -> tuple[list[dict], list[dict], dict]:
    """(train, dev, stats) rows: {text, title_tokens, body_tokens}."""
    banned = eval_hashes() | pr_dev_hashes()
    raw: list[tuple[str, str, str]] = []  # (text, subject, body)
    refused = 0
    with CSV_PATH.open(newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            subject = (r.get("subject") or "").strip()
            body = (r.get("body") or "").strip()
            if not subject or not body:
                continue
            text = f"{subject}\n\n{body}"
            if _nhash(text) in banned:
                refused += 1
                continue
            raw.append((text, subject, body))
    pr_used = 0
    for line in PR_TRAIN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["raw"]
        if _nhash(text) in banned:
            refused += 1
            continue
        gold = row.get("gold") or {}
        subject = (gold.get("title") or "").strip()
        body = (gold.get("body") or "").strip()
        if not subject or not body:
            continue
        raw.append((text, subject, body))
        pr_used += 1

    train, dev = [], []
    for text, subject, body in raw:
        rec = {"text": text, "subject": subject,
               "title_tokens": PROSE_TOKEN_RE.findall(subject),
               "body_tokens": set(PROSE_TOKEN_RE.findall(body.lower())),
               "lang_pr": subject != ""}
        (dev if int(hashlib.sha256(_norm(subject).encode()).hexdigest(), 16)
         % 10 == 0 else train).append(rec)
    stats = {"csv_rows": len(raw) - pr_used, "pr_rows": pr_used,
             "refused_eval_or_dev": refused,
             "train": len(train), "dev": len(dev)}
    return train, dev, stats


def label_tokens(rec: dict) -> tuple[list[str], list[int]]:
    """(per-token features, labels) for one row via greedy title match."""
    feats, _spans = prose_title_features(rec["text"])
    remaining = Counter(t.lower() for t in rec["title_tokens"])
    tokens = PROSE_TOKEN_RE.findall(rec["text"])
    labels = []
    for tok in tokens:
        low = tok.lower()
        if remaining[low] > 0:
            remaining[low] -= 1
            labels.append(1)
        else:
            labels.append(0)
    return feats, labels


def main() -> int:
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from skops.io import dump as skdump

    train, dev, stats = load_rows()
    print("data:", stats)
    X_tr, y_tr = [], []
    for rec in train:
        feats, labels = label_tokens(rec)
        X_tr.extend(feats)
        y_tr.extend(labels)
    pos = sum(y_tr)
    print(f"token samples: {len(y_tr)} ({pos} title, "
          f"{100 * pos / max(len(y_tr), 1):.1f}%)")

    pipe = Pipeline([
        ("hasher", FeatureHasher(n_features=2 ** 18, input_type="string")),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=SEED)),
    ])
    pipe.fit(X_tr, y_tr)

    # Dev-side metrics BEFORE any eval touch: title exact + body recall.
    from mivia_small_json.extract_learned import extract_prose_title

    def norm_title(s: str) -> str:
        return re.sub(r"[^a-z0-9äöüß ]+", " ", s.lower())

    exact = body_rec = n = 0
    model = pipe
    for rec in dev:
        got = extract_prose_title(rec["text"], model=model)
        if got is None:
            n += 1
            continue
        title, body = got
        n += 1
        exact += norm_title(title) == norm_title(rec["subject"])
        pred_body = set(PROSE_TOKEN_RE.findall(body.lower()))
        if rec["body_tokens"]:
            body_rec += (len(pred_body & rec["body_tokens"])
                         / len(rec["body_tokens"]))
    print(f"DEV (n={n}): title exact {100 * exact / n:.1f}% · "
          f"body token recall {body_rec / n:.3f}")

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    skdump(pipe, DEFAULT_OUT)
    digest = hashlib.sha256(DEFAULT_OUT.read_bytes()).hexdigest()
    DEFAULT_OUT.with_suffix(".skops.sha256").write_text(digest + "\n")
    DEFAULT_OUT.with_suffix(".skops.trainmeta.json").write_text(json.dumps({
        "model": "prose-title-extractor",
        "algorithm": "FeatureHasher(2**18)+LogisticRegression(C=2.0)",
        "train_token_samples": len(y_tr), "rows": stats,
        "dev_title_exact_pct": round(100 * exact / n, 1),
        "dev_body_token_recall": round(body_rec / n, 3),
        "seed": SEED, "sha256": digest,
    }, indent=2) + "\n")
    print(f"saved {DEFAULT_OUT} sha {digest[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
