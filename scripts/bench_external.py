#!/usr/bin/env python3
"""External comparison benchmark: short-text classification.

Trains a small hashed linear classifier on public intent corpora and
reports macro F1, so this project's approach can be read against
published baselines. These sets are comparison data only — never router
training data (see code-review-expansion).

Usage::

    python scripts/bench_external.py

Writes reports/benchmark-external.md.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "incoming" / "external"
REPORT = ROOT / "reports" / "benchmark-external.md"
SEED = 42


def _read_csv(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8-sig") as fh:  # utf-8-sig strips a BOM
        rows = list(csv.DictReader(fh))
    return [r["text"] for r in rows], [r["category"] for r in rows]


def _read_jsonl(path: Path) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    return [r["text"] for r in rows], [r["intent"] for r in rows]


def load_banking77() -> tuple:
    return (_read_csv(EXTERNAL / "banking77" / "train.csv"),
            _read_csv(EXTERNAL / "banking77" / "test.csv"))


def load_clinc150() -> tuple:
    return (_read_jsonl(EXTERNAL / "clinc150" / "train.jsonl"),
            _read_jsonl(EXTERNAL / "clinc150" / "test.jsonl"))


def build():
    """The published-baseline shape: hashed word bigrams -> TF-IDF -> LogReg."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("features", HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 18,
                                       alternate_sign=False, norm=None)),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=SEED)),
    ])


def evaluate(train: tuple, test: tuple) -> dict:
    """Train on ``train``, score on ``test``. Returns the metric block."""
    from sklearn.metrics import accuracy_score, f1_score

    (tr_x, tr_y), (te_x, te_y) = train, test
    pipe = build().fit(tr_x, tr_y)
    pred = pipe.predict(te_x)
    return {
        "n_train": len(tr_x),
        "n_test": len(te_x),
        "n_classes": len(set(tr_y)),
        "accuracy": round(float(accuracy_score(te_y, pred)), 4),
        "f1_macro": round(float(f1_score(te_y, pred, average="macro")), 4),
    }


DATASETS = (("banking77", load_banking77), ("clinc150", load_clinc150))


def main() -> int:
    results: dict[str, dict] = {}
    for name, loader in DATASETS:
        t0 = time.perf_counter()
        try:
            results[name] = evaluate(*loader())
        except FileNotFoundError as exc:
            results[name] = {"error": f"missing data ({exc}); "
                                       "run scripts/fetch_external_benchmarks.py"}
        results[name]["seconds"] = round(time.perf_counter() - t0, 2)

    lines = ["# External comparison benchmark", "",
             "Small hashed linear classifier (word 1-2 grams -> TF-IDF -> "
             "LogisticRegression).", "Comparison data only; not router training data.", ""]
    for name, block in results.items():
        lines.append(f"## {name}")
        lines.append("")
        if "error" in block:
            lines += [block["error"], ""]
            continue
        lines += [f"- train rows: {block['n_train']}",
                  f"- test rows: {block['n_test']}",
                  f"- classes: {block['n_classes']}",
                  f"- accuracy: {block['accuracy']}",
                  f"- macro F1: {block['f1_macro']}",
                  f"- train+score seconds: {block['seconds']}", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    for name, block in results.items():
        print(f"{name}: {block}")
    print(f"report written to {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
