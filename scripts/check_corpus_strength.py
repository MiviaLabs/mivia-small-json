#!/usr/bin/env python3
"""Weakness probe (AGENTS.md Data preparation protocol, 6b).

Routes a corpus through the SHIPPED router and splits it into:

- easy mass    — rows the shipped router already routes to the row's
                 expected class (they add weight, not signal);
- signal       — rows the shipped router gets WRONG (the actual reason
                 to retrain), with the confusion targets per source.

Two uses:
- PRE-training (the protocol step): on a candidate increment the
  easy-mass share is the "is this just more weight?" check — probe it
  before spending the GPU.
- POST-training (absorption check): a corpus the shipped model trained
  on should read ~0% signal; rows still wrong after training are the
  honest residue for the next round's residual-flip analysis.

Read-only measurement, never a gate: the recommendation line is input
for the round's written data review. Batch-size-free (one route per
row) keeps the 2 GB CPU sandbox safe.

    python scripts/check_corpus_strength.py <corpus.jsonl> \
        [--text-key text] [--expected-key category] [--source-key source] \
        [--default-expected issue] [--limit N]

Rows missing the expected key take --default-expected (the aux chat
corpus is all `text`; pass that explicitly).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--expected-key", default="category")
    ap.add_argument("--source-key", default="source")
    ap.add_argument("--default-expected", default=None,
                    help="expected class for rows missing the key "
                         "(e.g. `text` for the chat aux)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = []
    for line in args.corpus.read_text(encoding="utf-8",
                                      errors="replace").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = r.get(args.text_key) or ""
        expected = r.get(args.expected_key) or args.default_expected
        if not text or not expected:
            continue
        rows.append({"text": text, "expected": expected,
                     "source": r.get(args.source_key) or "unspecified"})
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("no rows with both text and an expected class", file=sys.stderr)
        return 2

    from mivia_small_json import pipeline

    router = pipeline.load_active_router()
    assert router is not None, "no shipped router"

    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "easy": 0, "confusions": Counter()})
    for r in rows:
        schema, _rb, _conf = pipeline._route_with_confidence(router,
                                                             r["text"])
        s = stats[r["source"]]
        s["n"] += 1
        if schema == r["expected"]:
            s["easy"] += 1
        else:
            s["confusions"][schema] += 1

    total_n = sum(s["n"] for s in stats.values())
    total_easy = sum(s["easy"] for s in stats.values())
    out = {
        "corpus": str(args.corpus),
        "shipped_router_sha256": _sha256(ROOT / "models/router-encoder-int8.onnx"),
        "rows": total_n,
        "easy_mass_pct": round(100 * total_easy / total_n, 2),
        "per_source": {
            src: {
                "n": s["n"],
                "easy_mass_pct": round(100 * s["easy"] / s["n"], 2),
                "top_confusions": dict(s["confusions"].most_common(3)),
            }
            for src, s in sorted(stats.items())
        },
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))
    share = out["easy_mass_pct"]
    if share >= 80:
        print(f"RECOMMENDATION: {share}% easy mass — this increment is "
              "mostly weight; rethink the data before spending the GPU")
    elif share >= 50:
        print(f"RECOMMENDATION: {share}% easy mass — half of this "
              "increment teaches the model nothing new; check the "
              "confusion targets cover the registered weakness")
    else:
        print(f"RECOMMENDATION: {share}% easy mass — the increment "
              "targets live weaknesses; proceed to the audit gate")
    return 0


def _sha256(path: Path) -> str | None:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() \
        if path.is_file() else None


if __name__ == "__main__":
    raise SystemExit(main())
