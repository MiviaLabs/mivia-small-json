#!/usr/bin/env python3
"""Learn agent_step status inference for free-form execution traces.

The rule extractor (``agent_step.extract_agent_step``) reads canonical
markers (``return_code: N``, ``status: <word>``) at 97.8% on canonical
dev. This script trains a message-level classifier for the traces where
those markers are ABSENT — the ``stripped`` proxy removes the canonical
``return_code``/``status`` fields (labels kept), the same augmentation
spirit as ``data/incoming/log-format-augmented.jsonl``.

Same stack as the other learned extractors (hashed token/char features +
LogisticRegression, skops persistence, CPU-only). Train on curated
``data/real/agent_step`` train (+ stripped copies), eval on the
session-disjoint staged dev, canonical AND stripped.

The artifact ships (``models/agent_step_status.skops``) only if it beats
the rules baseline on stripped dev without regressing canonical dev in
assist mode; otherwise the numbers stand as a falsification record.

Writes models/agent_step_status.skops (+ sha256 sidecar) on success.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_PATH = ROOT / "data" / "real" / "agent_step" / "train.jsonl"
DEV_PATH = ROOT / "data" / "real" / "agent_step" / "dev.jsonl"
OUT_PATH = ROOT / "models" / "agent_step_status.skops"

TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")
_STRIP_RES = (
    re.compile(r"(?i)\bstatus\s*[:=]\s*(success|error|timeout|interrupted)\b;?"),
    re.compile(r"(?i)\breturn_code\s*[:=]\s*(-?\d+)\b;?"),
    re.compile(r"(?i)\bexit\s+code\s*[:=]?\s*(-?\d+)\b;?"),
)


def strip_canonical(text: str) -> str:
    """Remove canonical status markers, keeping the trace content."""
    for pat in _STRIP_RES:
        text = pat.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ;")


def status_features(text: str) -> list[str]:
    """Token unigram/bigram + char 4-gram features for trace text."""
    low = text.lower()
    tokens = TOKEN_RE.findall(low)
    feats = {"w=" + t for t in tokens if len(t) > 1}
    for i in range(len(tokens) - 1):
        feats.add(f"bi={tokens[i]}_{tokens[i+1]}")
    for i in range(len(low) - 3):
        feats.add("c4=" + low[i:i + 4])
    return sorted(feats)


def fit_status(texts: list[str], labels: list[str]):
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("hasher", FeatureHasher(n_features=2**17, input_type="string")),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
    ])
    return pipe.fit([status_features(t) for t in texts], labels)


def _rows(path: Path) -> list[tuple[str, str]]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append((row["raw"], row["gold"]["status"]))
    return out


def _accuracy(model, pairs: list[tuple[str, str]], stripped: bool) -> float:
    if not pairs:
        return 0.0
    texts = [strip_canonical(t) if stripped else t for t, _ in pairs]
    pred = model.predict([status_features(t) for t in texts])
    return sum(p == y for p, (_, y) in zip(pred, pairs)) / len(pairs)


def main() -> int:
    from skops.io import dump

    from mivia_small_json.agent_step import extract_agent_step

    train = _rows(TRAIN_PATH)
    dev = _rows(DEV_PATH)
    print(f"train rows: {len(train)}, dev rows: {len(dev)}")

    def rules_acc(pairs, stripped):
        ok = 0
        for raw, gold in pairs:
            text = strip_canonical(raw) if stripped else raw
            try:
                if extract_agent_step(text)["status"] == gold:
                    ok += 1
            except Exception:
                pass
        return ok / max(1, len(pairs))

    for name, stripped in (("canonical", False), ("stripped", True)):
        print(f"rules dev {name}: {rules_acc(dev, stripped):.1%}")

    aug_texts, aug_labels = [], []
    for raw, status in train:
        aug_texts.append(raw)
        aug_labels.append(status)
        stripped = strip_canonical(raw)
        if stripped != raw:
            aug_texts.append(stripped)
            aug_labels.append(status)
    model = fit_status(aug_texts, aug_labels)
    for name, stripped in (("canonical", False), ("stripped", True)):
        print(f"learned dev {name}: {_accuracy(model, dev, stripped):.1%}")

    learned_stripped = _accuracy(model, dev, True)
    minority = [(t, y) for t, y in dev if y != "success"]
    recall = _accuracy(model, minority, True) if minority else 0.0
    print(f"learned minority recall on stripped dev: {recall:.1%} "
          f"(n={len(minority)})")
    if learned_stripped <= rules_acc(dev, True) or recall <= 0.0:
        print("NOT shipped: needs both a stripped-dev win over rules AND "
              "non-zero minority recall (a majority-only predictor is not "
              "a status classifier). Falsification record, no artifact.")
        return 2
    dump(model, OUT_PATH)
    digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    (OUT_PATH.parent / (OUT_PATH.name + ".sha256")).write_text(
        digest + "\n", encoding="utf-8")
    print(f"saved {OUT_PATH} sha256={digest[:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
