#!/usr/bin/env python3
"""Merge agent-authored multilingual intent queries into pair form.

Reads data/incoming/multilang-intents/{banking77,clinc150}-{lang}.jsonl
(guideline: universal-structuring/
annotation-guideline-multilang-intents.md), drops any text that matches
an eval surface or the English corpus (contamination discipline, same as
scripts/build_intent_pairs.py), samples 3 negatives per row from the
SAME label world (English descriptions — the product scores
non-English text against English intent descriptions), and writes
data/real/intent-pairs/multilang.jsonl in the guard trainer's schema.

90/10 train/dev split, seed 42, deterministic. Refuses to write when a
language file is missing, unparseable, or under-quota — a silently
shrunken corpus must fail here, not at the sha pin.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

STAGING = ROOT / "data/incoming/multilang-intents"
OUT = ROOT / "data/real/intent-pairs/multilang.jsonl"
EN_CORPUS = ROOT / "data/real/intent-pairs/train.jsonl"
B77 = ROOT / "data/incoming/external/banking77"
CLINC = ROOT / "data/incoming/external/clinc150"

NEG_PER_ROW = 3
SEED = 42
LANGS = ("en", "de", "fr", "es", "pl", "pt", "it", "nl", "ja")


def desc_of(intent: str) -> str:
    return f"{intent.replace('_', ' ')} ({intent.replace('_', ' ')})"


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def _strong_norm(text: str) -> str:
    """NFKC + casefold + punctuation-strip (the test_bfcl_pairs standard):
    catches punctuation-variant duplicates like "what time is it" vs
    "What time is it?". """
    t = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(_PUNCT.sub(" ", t).split())


def _guard_eval_norms() -> set[str]:
    """The guard's canonical eval instruments never train (see
    scripts/build_intent_pairs.py — same exclusion, same reason)."""
    out: set[str] = set()
    for name in ("guard-eval-v2.jsonl", "irrelevance_eval.jsonl"):
        path = STAGING.parent / "external/bfcl" / name
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
    """Every one-shot eval surface + the English corpus texts."""
    import csv

    texts: set[str] = set()
    with (B77 / "test.csv").open(encoding="utf-8", errors="replace",
                                 newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("text"):
                texts.add(_norm(r["text"]))
    with (B77 / "train.csv").open(encoding="utf-8", errors="replace",
                                  newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("text"):
                texts.add(_norm(r["text"]))
    for name in ("validation.jsonl", "test.jsonl", "train.jsonl"):
        with (CLINC / name).open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line).get("text")
                    if t:
                        texts.add(_norm(t))
    for line in EN_CORPUS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            texts.add(_norm(json.loads(line)["request"]))
    return texts


def main() -> int:
    b77_intents = set(json.loads((B77 / "categories.json").read_text()))
    clinc_intents = set(json.loads((CLINC / "labels.json").read_text()))
    worlds = {"banking77": b77_intents, "clinc150": clinc_intents}
    eval_texts = _eval_texts()
    rng = random.Random(SEED)
    guard_norms = _guard_eval_norms()
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    per_lang: dict[str, int] = {}
    problems: list[str] = []

    for lang in LANGS:
        for world, intents in worlds.items():
            paths = sorted(STAGING.glob(f"{world}-{lang}*.jsonl"))
            if not paths:
                problems.append(f"missing files: {world}-{lang}")
                continue
            texts_in_file: set[str] = set()
            n_before = len(rows)
            for path in paths:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError as exc:
                        problems.append(f"{path.name}: unparseable line: {exc}")
                        continue
                    text = (r.get("text") or "").strip()
                    intent = (r.get("intent") or "").strip()
                    if not text or intent not in intents:
                        skipped["bad-intent-or-empty"] = \
                            skipped.get("bad-intent-or-empty", 0) + 1
                        continue
                    if _norm(text) in eval_texts \
                            or _strong_norm(text) in guard_norms:
                        skipped["eval-overlap"] = \
                            skipped.get("eval-overlap", 0) + 1
                        continue
                    if _norm(text) in texts_in_file:
                        skipped["duplicate"] = skipped.get("duplicate", 0) + 1
                        continue
                    texts_in_file.add(_norm(text))
                    # Hash-stable split: a text's split never changes
                    # when new waves merge (the old rng-order split
                    # rotated dev rows on every corpus growth).
                    split = "dev" if int(hashlib.sha256(
                        _norm(text).encode()).hexdigest(), 16) % 10 == 0 \
                        else "train"
                    rows.append({"label": 1, "request": text,
                                 "source": f"{world}_multilang_{lang}_pos",
                                 "split": split, "tool": desc_of(intent)})
                    for neg in rng.sample(sorted(intents - {intent}),
                                          NEG_PER_ROW):
                        rows.append({"label": 0, "request": text,
                                     "source": f"{world}_multilang_{lang}_neg",
                                     "split": split, "tool": desc_of(neg)})
            per_lang[f"{world}-{lang}"] = per_lang.get(f"{world}-{lang}", 0) \
                + len(rows) - n_before

    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print("refusing to write: staging incomplete", file=sys.stderr)
        return 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    pos = sum(r["label"] for r in rows)
    print(f"rows={len(rows)} pos={pos} langs={len(per_lang)} "
          f"skipped={skipped} sha256={digest}")
    for k, v in sorted(per_lang.items()):
        print(f"  {k}: {v}")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
