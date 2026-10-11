#!/usr/bin/env python3
"""Universal-structuring 1b: the intent/label pair-training corpus.

Builds ``data/real/intent-pairs/train.jsonl`` — (request, description)
pairs in the guard trainer's format ({label, request, source, split,
tool}) — from three public sets, TRAIN splits only:

- Banking77 train.csv (10,016 rows): 1 positive + 3 sampled negatives
  per row against the SAME description recipe the zero-shot select
  baseline uses (``"<intent words> (<intent words>)"``) — the pair the
  model trains on must be byte-identical to the pair inference scores.
- CLINC150 train.jsonl (7,600 rows): same construction, 151 intents.
- When2Call MCQ: sibling-tool near-miss negatives (label 0) via the
  same construction as scripts/build_tool_match_corpus.py's
  ``--nearmiss when2call`` — an aligned-but-irrelevant tool per listed
  sibling, the swap-in-a-sibling quadrant.

DISJOINTNESS (hard rule): the Banking77 TEST split, the CLINC150
validation split (the registered 1a baseline surface) and CLINC150 TEST
split never contribute a row; tests/test_intent_pairs_disjoint.py
sha-pins this corpus and recomputes the eval texts to hold it.

Deterministic (seed 42), offline, no network.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

B77 = ROOT / "data/incoming/external/banking77"
CLINC = ROOT / "data/incoming/external/clinc150"
WHEN2CALL = ROOT / "data/incoming/external/when2call/when2call_test_mcq.jsonl"
OUT = ROOT / "data/real/intent-pairs/train.jsonl"

NEG_PER_ROW = 3
SEED = 42


def desc_of(intent: str) -> str:
    """The description string inference scores — same recipe as the
    bench_banking77_select / bench_clinc150_select baselines."""
    return f"{intent.replace('_', ' ')} ({intent.replace('_', ' ')})"


def split_of(i: int, rng: random.Random) -> str:
    """90/10 train/dev, seeded per row index (stable across rebuilds)."""
    return "dev" if rng.random() < 0.10 else "train"


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def _strong_norm(text: str) -> str:
    """NFKC + casefold + punctuation-strip (the test_bfcl_pairs standard):
    catches punctuation-variant duplicates ("I lost my phone!" vs
    "i lost my phone")."""
    t = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(_PUNCT.sub(" ", t).split())


def _guard_eval_norms() -> set[str]:
    """The guard's canonical eval instruments (guard-eval-v2 = the queued
    select()-promotion no-regression surface; irrelevance_eval = the FP
    instrument). Requests here never train — a corpus minted after these
    evals exist may not bleed into them (test_bfcl_pairs standard)."""
    out: set[str] = set()
    for name in ("guard-eval-v2.jsonl", "irrelevance_eval.jsonl"):
        path = B77.parent / "bfcl" / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                t = r.get("request") or r.get("text") or ""
                if t:
                    out.add(_strong_norm(t))
    return out


def _eval_texts() -> set[str]:
    """Normalized texts of EVERY one-shot eval surface (Banking77 test,
    CLINC150 validation + test). Train rows matching any of them are
    dropped: the public corpora contain cross-split duplicate utterances
    ('hey what's up' sits in CLINC train AND validation), and a text the
    one-shot eval measures must never be trained on."""
    import csv as _csv

    texts: set[str] = set()
    with (B77 / "test.csv").open(encoding="utf-8", errors="replace",
                                 newline="") as fh:
        for r in _csv.DictReader(fh):
            if r.get("text"):
                texts.add(_norm(r["text"]))
    for name in ("validation.jsonl", "test.jsonl"):
        with (CLINC / name).open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line).get("text")
                    if t:
                        texts.add(_norm(t))
    return texts


def main() -> int:
    rng = random.Random(SEED)
    eval_texts = _eval_texts()
    eval_strong = {_strong_norm(t) for t in eval_texts}
    guard_norms = _guard_eval_norms()
    rows: list[dict] = []

    # --- Banking77 (train split ONLY) -------------------------------------
    b77_intents = json.loads((B77 / "categories.json").read_text())
    b77_texts: list[tuple[str, str]] = []
    with (B77 / "train.csv").open(encoding="utf-8", errors="replace",
                                  newline="") as fh:
        for r in csv.DictReader(fh):
            text = (r.get("text") or "").strip()
            label = (r.get("category") or "").strip()
            if text and label:
                b77_texts.append((text, label))
    skipped = 0
    for text, label in b77_texts:
        if _norm(text) in eval_texts or _strong_norm(text) in eval_strong \
                or _strong_norm(text) in guard_norms:
            skipped += 1
            continue
        split = ("dev" if int(hashlib.sha256(
            _norm(text).encode()).hexdigest(), 16) % 10 == 0 else "train")
        rows.append({"label": 1, "request": text, "source":
                     "banking77_train_pos", "split": split,
                     "tool": desc_of(label)})
        for neg in rng.sample([i for i in b77_intents if i != label],
                              NEG_PER_ROW):
            rows.append({"label": 0, "request": text, "source":
                         "banking77_train_neg", "split": split,
                         "tool": desc_of(neg)})

    # --- CLINC150 (train split ONLY — validation/test never touch this) ---
    clinc_intents = json.loads((CLINC / "labels.json").read_text())
    clinc_texts: list[tuple[str, str]] = []
    with (CLINC / "train.jsonl").open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            label = (r.get("intent") or "").strip()
            if text and label:
                clinc_texts.append((text, label))
    for text, label in clinc_texts:
        if _norm(text) in eval_texts or _strong_norm(text) in eval_strong \
                or _strong_norm(text) in guard_norms:
            skipped += 1
            continue
        split = ("dev" if int(hashlib.sha256(
            _norm(text).encode()).hexdigest(), 16) % 10 == 0 else "train")
        rows.append({"label": 1, "request": text, "source":
                     "clinc150_train_pos", "split": split,
                     "tool": desc_of(label)})
        for neg in rng.sample([i for i in clinc_intents if i != label],
                              NEG_PER_ROW):
            rows.append({"label": 0, "request": text, "source":
                         "clinc150_train_neg", "split": split,
                         "tool": desc_of(neg)})

    # --- When2Call (sibling near-miss negatives, guard-lineage format) ----
    # The MCQ file is a gitignored HF cache fetch: without it the builder
    # used to shrink silently by 7,443 rows and fail only at the sha pin.
    if not WHEN2CALL.is_file():
        print(f"when2call MCQ file missing: {WHEN2CALL}; fetch it (see "
              "data/real/intent-pairs/PROVENANCE.md) and rebuild", file=sys.stderr)
        return 2
    import build_tool_match_corpus as btc

    w2c_before = len(rows)
    for request, span, label, source in btc._when2call_pairs(btc.render_tool):
        if _strong_norm(request) in guard_norms:
            skipped += 1
            continue
        rows.append({"label": label, "request": request, "source": source,
                     "split": "train", "tool": span})
    if len(rows) - w2c_before < 1000:
        print(f"when2call contributed only {len(rows) - w2c_before} rows "
              "(expected ~7,400): refusing to write a shrunken corpus",
              file=sys.stderr)
        return 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    from collections import Counter

    by_src = Counter(r["source"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    pos = sum(r["label"] for r in rows)
    prov = f"""# intent-pairs — universal-structuring phase 1b pair-training corpus

Built by scripts/build_intent_pairs.py (seed {SEED}, deterministic,
offline). Format: the guard trainer's pairs schema
({{label, request, source, split, tool}}); `tool` is the description
string inference scores — byte-identical to the zero-shot select
baseline recipe `"<intent words> (<intent words>)"`.

Sources (TRAIN splits ONLY):
- Banking77 train.csv ({len(b77_texts)} rows, {len(b77_intents)} intents;
  positive + {NEG_PER_ROW} sampled negatives per row) — UK research corpus.
- CLINC150 train.jsonl ({len(clinc_texts)} rows, {len(clinc_intents)} labels
  incl. oos; positive + {NEG_PER_ROW} sampled negatives per row) —
  Oracle research corpus.
- When2Call test_mcq — sibling-tool near-miss NEGATIVES only (label 0),
  same construction as the guard corpus --nearmiss when2call.

DISJOINTNESS: Banking77 test.csv (3,080), CLINC150 validation.jsonl
(3,100 — the registered 1a baseline surface) and CLINC150 test.jsonl
(5,500) contributed ZERO rows. Held by tests/test_intent_pairs_disjoint.py
(sha-pin + recomputed eval texts).

sha256: {digest}
rows: {len(rows)} (skipped {skipped} train rows whose text also appears on
an eval surface — cross-split duplicates in the upstream corpora)
(label-1 {pos} / label-0 {len(rows) - pos})
by source: {dict(sorted(by_src.items()))}
by split: {dict(sorted(by_split.items()))}
"""
    (OUT.parent / "PROVENANCE.md").write_text(prov, encoding="utf-8")
    print(f"rows={len(rows)} pos={pos} sha256={digest[:12]}...")
    print(f"written {OUT} + PROVENANCE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
