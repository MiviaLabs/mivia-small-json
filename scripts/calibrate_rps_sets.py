#!/usr/bin/env python3
"""Calibrate the RPS ordinal conformal decode for the grade primitive.

Ticket 13 of result-quality-improvement ``rps_tier_set``
uses the ranked probability score as the conformal nonconformity function
(Haas et al., UAI 2026, PMLR v337:1882-1912, arXiv:2606.24959), which yields
contiguous median-centred sets by construction. This script sweeps the RPS
quantile on the same slices the APS decode was calibrated on, so the two
decodes can be read side by side.

Slices (shipped grade artifact):
- LogHub dev severity rows (in-domain),
- synthetic-register dev (data/real/grade/dev.jsonl, template-level held out),
- frozen novel-register set (hand-written business/SRE phrasings).

Read-only. Prints the table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_zero_shot_primitives import (  # noqa: E402
    NOVEL_REGISTER_GRADE_CASES,
    SEVERITY_ANCHORS_EXCLUSIVE,
    load_loghub_grade_slices,
)

QUANTILES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def main() -> int:
    from mivia_small_json.tool_match import (
        GRADE_SET_QUANTILE,
        aps_tier_set,
        load_grade_matcher,
        rps_tier_set,
    )

    matcher = load_grade_matcher()
    if matcher is None:
        print("grade artifact not available", file=sys.stderr)
        return 1

    def scores_of(text: str) -> list[float]:
        return [matcher._score_raw(text, a) for a in SEVERITY_ANCHORS_EXCLUSIVE]

    slices = load_loghub_grade_slices()
    loghub = (slices["implicit"] + slices["explicit"])[:400]
    dev = [json.loads(line) for line in
           (ROOT / "data" / "real" / "grade" / "dev.jsonl").read_text().splitlines()
           if line.strip()]
    syn = [(r["text"], r["gold"]) for r in dev if r["family"] == "severity"]
    novel = list(NOVEL_REGISTER_GRADE_CASES)
    data = {"loghub": loghub, "syn": syn, "novel": novel}

    def measure(fn, q):
        out = {}
        for name, rows in data.items():
            cov = size = 0.0
            for text, gold in rows:
                s = fn(scores_of(text), q)
                cov += gold in s
                size += len(s)
            n = len(rows) or 1
            out[name] = (100 * cov / n, size / n)
        return out

    print("APS decode at the pinned quantile", GRADE_SET_QUANTILE)
    for name, (cov, size) in measure(aps_tier_set, GRADE_SET_QUANTILE).items():
        print(f"  {name:7s} coverage={cov:5.1f}% size={size:.2f}")

    print("\nRPS decode sweep (coverage / mean size)")
    print(f"{'q':>5} | " + " | ".join(f"{n:>15}" for n in data))
    for q in QUANTILES:
        cells = measure(rps_tier_set, q)
        print(f"{q:>5.2f} | " + " | ".join(
            f"{cells[n][0]:6.1f}%/{cells[n][1]:5.2f}" for n in data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
