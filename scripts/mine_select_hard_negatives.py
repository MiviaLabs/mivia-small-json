#!/usr/bin/env python3
"""Mine select()-shaped hard negatives from TRAIN data (follow-up 2).

The Arm C session rank (48c43bce) fixed real-session service but degraded
select() over short confusable descriptions (git attractor). The fix is
train-only hard-negative mining: score every merged-corpus TRAIN request
against a pool of confusable tool descriptions with the SHIPPED rank;
where a WRONG description scores high, emit (request, wrong_tool, 0).

Real data, deterministic (seed 42), train rows only — the BFCL dev split
that `bench_zero_shot_primitives.select_multi_candidate` samples stays
untouched, as do all test fixtures.

Output: data/incoming/external/harvest-sessions/train-merged-v2.jsonl
= the Arm C merged corpus + mined negatives (source tag select_hardneg).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MERGED = ROOT / "data" / "incoming" / "external" / "harvest-sessions" / "train-merged.jsonl"
OUT = ROOT / "data" / "incoming" / "external" / "harvest-sessions" / "train-merged-v2.jsonl"
SEED = 42
K = 10          # confusables sampled per request
SCORE_FLOOR = 0.10
MAX_NEG_PER_REQUEST = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-negatives", type=int, default=15000)
    args = ap.parse_args()

    from mivia_small_json import tool_match

    rank = tool_match.load_single_matcher(
        tool_match.ROOT / "models" / "tool-match-int8.onnx")
    if rank is None:
        raise SystemExit("shipped rank not loadable")

    rows = [json.loads(line) for line in MERGED.read_text().splitlines() if line.strip()]
    tools = sorted({r["tool"] for r in rows})
    print(f"corpus {len(rows)} rows, description pool {len(tools)}")

    rng = random.Random(SEED)
    negatives = []
    for i, r in enumerate(rows):
        if r["label"] != 1:
            continue  # mine contrast only around real invocations
        pool = rng.sample(tools, min(K + 1, len(tools)))
        confusables = [t for t in pool if t != r["tool"]][:K]
        scored = []
        for t in confusables:
            try:
                s = rank._score_raw(r["request"], t)
            except Exception:
                continue
            scored.append((s, t))
        scored.sort(reverse=True)
        hard = [(s, t) for s, t in scored if s >= SCORE_FLOOR]
        for s, t in hard[:MAX_NEG_PER_REQUEST]:
            negatives.append({"label": 0, "request": r["request"],
                              "source": "select_hardneg", "split": "train",
                              "tool": t})
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{sum(1 for x in rows if x['label'] == 1)} "
                  f"positives scanned, {len(negatives)} negatives mined")
        if len(negatives) >= args.max_negatives:
            break

    out_rows = rows + negatives
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in out_rows))
    print(f"mined {len(negatives)} hard negatives -> {OUT} "
          f"({len(out_rows)} total rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
