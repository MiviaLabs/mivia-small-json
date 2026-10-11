#!/usr/bin/env python3
"""Build the MISSING registered aux component: fresh HF rows for the
phase-2 OOD training mix (owner directive 2026-10-10).

The phase-2 spec (ood-routing-band, amendment) registers an aux
corpus of "fresh HF rows (agnews/wikitext/emotion/sst2 — different indices
than the v1 EVAL samples) + a freshly agent-authored free-chat/non-EN/config
batch". The 2026-10-09 freeze delivered only the agent batch (5,994 rows);
the s42 one-shot then measured the untrained categories at 6-30% rejection
(reports/ood-class-oneshot-mm12-ood-s42-int8.onnx.json). This script builds
the missing half and freezes the corrected corpus v2:

    data/real/ood-aux/ood-aux-hf.jsonl        the new component (~1,200 rows)
    data/real/ood-aux/ood-aux-train-v2.jsonl  agent batch + HF component

Disjointness is CONSTRUCTIVE (offset window 2000+; the v1 eval sample drew
from offsets 0-399, scripts/build_ood_set.py fetch()) and VERIFIED by
normalized-text hash against every eval surface and every in-domain corpus
file — the build aborts on any overlap. tests/test_ood_aux_disjoint.py pins
the frozen corpus against regressions.

Fetches serially with 2 s pauses (the 2026-10-08 rate-limit lesson: no
bursts, short pauses). Sources/licenses mirror the v1 eval set
(scripts/build_ood_set.py): AG News (public news corpus), wikitext-2-raw
(CC BY-SA 3.0), SetFit emotion and sst2 (public research sets).
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUX_DIR = ROOT / "data" / "real" / "ood-aux"
HF_OUT = AUX_DIR / "ood-aux-hf.jsonl"
V1_CORPUS = AUX_DIR / "ood-aux-train.jsonl"
V2_CORPUS = AUX_DIR / "ood-aux-train-v2.jsonl"
SERVER = "https://datasets-server.huggingface.co/rows"
SEED = 20261010

# Same datasets/configs/splits as the v1 eval set, but the offset window
# starts at 2000 — far past v1's 0-399 sample — and hash-verification is
# the definitive gate anyway.
SOURCES = [
    ("fancyzhx/ag_news", "default", "train", 300),
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", 300),
    ("SetFit/emotion", "default", "train", 300),
    ("SetFit/sst2", "default", "train", 300),
]
START_OFFSET = 2000
PAGES = 12           # per source: up to 1200 candidate rows
PAGE_LEN = 100
PAUSE_S = 2.0
# wikitext-2raw is ~40% usable (headings/short lines filtered); the spec
# pins the component's presence and disjointness, not per-source counts,
# so each source must clear this floor rather than exactly `want`.
MIN_PER_SOURCE = 200

# Eval surfaces and in-domain corpora: the corpus must NEVER contain
# these (both at fetch time and per the committed gate test).
BAN_LIST = [
    "data/incoming/ood/ood-set.jsonl",
    "data/incoming/ood/ood-set-v2.jsonl",
    "data/incoming/ood/ood-set-v2-curated.jsonl",
    "data/heldout/test.jsonl",
    "data/heldout/dev-reference.jsonl",
]
# Fetch-time only: the existing agent batch is carried into corpus v2 by
# design, so it is not in the gate ban set — but freshly fetched rows
# must not duplicate it either.
FETCH_ONLY_BAN = ["data/real/ood-aux/ood-aux-train.jsonl"]


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def banned_texts(include_fetch_only: bool = False) -> set[str]:
    seen: set[str] = set()
    paths = [ROOT / p for p in BAN_LIST]
    if include_fetch_only:
        paths += [ROOT / p for p in FETCH_ONLY_BAN]
    paths += sorted((ROOT / "data/real").glob("*/dev.jsonl"))
    paths += sorted((ROOT / "data/real").glob("*/train.jsonl"))
    for d in ("data/incoming/realworld", "data/wild"):
        paths += sorted((ROOT / d).rglob("*.jsonl"))
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for key in ("raw", "text", "body", "problem_statement"):
                val = row.get(key) if isinstance(row, dict) else None
                if isinstance(val, str) and val.strip():
                    seen.add(_norm(val))
    return seen


def fetch(dataset: str, config: str, split: str, want: int) -> list[str]:
    rows: list[str] = []
    for page in range(PAGES):
        offset = START_OFFSET + page * PAGE_LEN
        query = urllib.parse.urlencode({
            "dataset": dataset, "config": config, "split": split,
            "offset": offset, "length": PAGE_LEN})
        req = urllib.request.Request(
            f"{SERVER}?{query}",
            headers={"User-Agent": "mivia-ood-aux-hf/1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        got = payload.get("rows", [])
        rows.extend(r["row"].get("text", "") for r in got)
        print(f"  {dataset} offset {offset}: +{len(got)} rows "
              f"(pool {len(rows)})", flush=True)
        if not got or len(rows) >= want * 2:
            break
        time.sleep(PAUSE_S)
    return rows


def _clean(text: str) -> str:
    text = (text or "").replace("\\b", " ").replace("\\n", " ").strip()
    return " ".join(text.split())


def _usable(text: str, min_len: int = 60) -> bool:
    if len(text) < min_len:
        return False
    if text.startswith("=") or text.startswith("@"):
        return False
    printable = sum(c.isprintable() for c in text)
    return printable / max(len(text), 1) >= 0.9


def main() -> int:
    import random

    rng = random.Random(SEED)
    banned = banned_texts(include_fetch_only=True)
    print(f"ban list: {len(banned)} normalized texts "
          f"(every eval surface + in-domain corpus)", flush=True)

    picked_all: list[dict] = []
    for dataset, config, split, want in SOURCES:
        pool = [t for t in ( _clean(r) for r in
                             fetch(dataset, config, split, want))
                 if _usable(t)]
        # de-dup the pool, drop anything banned, seeded sample to `want`
        uniq: list[str] = []
        seen: set[str] = set()
        banned_hits = 0
        for text in pool:
            key = _norm(text)
            if key in seen:
                continue
            if key in banned:
                banned_hits += 1
                continue
            seen.add(key)
            uniq.append(text)
        rng.shuffle(uniq)
        took = uniq[:want]
        print(f"  {dataset}: pool {len(pool)} usable, {banned_hits} banned "
              f"hits, took {len(took)}", flush=True)
        if len(took) < MIN_PER_SOURCE:
            raise SystemExit(
                f"ABORT: {dataset} yielded {len(took)} disjoint rows "
                f"(floor {MIN_PER_SOURCE})")
        picked_all.extend({"source": dataset, "category": "real",
                           "text": t} for t in took)

    AUX_DIR.mkdir(parents=True, exist_ok=True)
    with HF_OUT.open("w", encoding="utf-8") as fh:
        for row in picked_all:
            fh.write(json.dumps({"text": row["text"],
                                 "category": "real"},
                                ensure_ascii=False) + "\n")

    v1 = [json.loads(line) for line in V1_CORPUS.read_text(encoding="utf-8")
          .splitlines() if line.strip()]
    # The v1 agent batch carries 17 internal duplicate texts (found by the
    # gate test at freeze): corpus v2 keeps first occurrences only.
    seen_v1: set[str] = set()
    v1_unique: list[dict] = []
    for row in v1:
        key = _norm(row["text"])
        if key in seen_v1:
            continue
        seen_v1.add(key)
        v1_unique.append(row)
    print(f"v1 batch: {len(v1)} rows -> {len(v1_unique)} after dedup",
          flush=True)

    # Final gate over the WHOLE v2 corpus (v1 batch included): the v1
    # freeze verified eval-set disjointness but not in-domain-corpus
    # disjointness, and one-word free-chat rows can collide with canonical
    # training texts ('seriously?' did). Colliding rows are dropped and
    # reported — a text that exists verbatim in a canonical class must not
    # also be an OOD example.
    gate_ban = banned_texts(include_fetch_only=False)
    dropped: list[str] = []
    final: list[dict] = []
    for row in v1_unique + [{"text": r["text"], "category": r["category"]}
                            for r in picked_all]:
        if _norm(row["text"]) in gate_ban:
            dropped.append(row["text"][:60])
            continue
        final.append(row)
    if dropped:
        print(f"dropped {len(dropped)} rows colliding with eval/in-domain "
              f"texts: {dropped}", flush=True)
    keys = [_norm(r["text"]) for r in final]
    assert len(keys) == len(set(keys)), "within-corpus duplicate texts"
    with V2_CORPUS.open("w", encoding="utf-8") as fh:
        for row in final:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for path in (HF_OUT, V2_CORPUS):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(
            digest + "\n", encoding="utf-8")
        print(f"wrote {path.name}: "
              f"{sum(1 for _ in path.open())} rows, sha256 {digest[:12]}…",
              flush=True)
    total = len(final)
    print(f"corpus v2: {total} rows "
          f"(aux share of the mix: {total}/{total + 67_951} "
          f"= {100 * total / (total + 67_951):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
