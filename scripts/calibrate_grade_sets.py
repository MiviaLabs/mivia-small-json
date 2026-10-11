#!/usr/bin/env python3
"""Measure the grade tier-set operating point (quantile sweep).

`grade(with_set=True)` decodes to a contiguous tier set via
`tool_match.aps_tier_set`. The per-tier sigmoid scores saturate near 0/1,
which breaks the exchangeability a provably-covered conformal quantile
would need - so the pinned `GRADE_SET_QUANTILE` is a MEASURED operating
point, not a coverage guarantee. This script produces the sweep table
behind the pin: coverage (gold tier inside the set) and mean set size per
quantile on raw scores, on the calibration slice and register-shifted
eval slices.

Slices (shipped v4 artifact trained on LogHub train only):
- synthetic-register dev (data/real/grade/dev.jsonl): calibration slice,
  template-level held out,
- frozen novel-register set (hand-written business/SRE phrasings),
- LogHub dev severity rows (in-domain).

Run after any grade artifact change; re-pin GRADE_SET_QUANTILE from the
table and update the live test bands.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_zero_shot_primitives import (  # noqa: E402
    NOVEL_REGISTER_GRADE_CASES,
    SEVERITY_ANCHORS_EXCLUSIVE,
    load_loghub_grade_slices,
)

QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]


def main() -> int:
    from mivia_small_json.tool_match import load_grade_matcher

    matcher = load_grade_matcher()
    if matcher is None:
        print("grade artifact not available", file=sys.stderr)
        return 1

    def scores_of(text: str) -> list[float]:
        return [matcher._score_raw(text, a) for a in SEVERITY_ANCHORS_EXCLUSIVE]

    def set_of(scores: list[float], q: float) -> list[int]:
        from mivia_small_json.tool_match import aps_tier_set

        # aps_tier_set takes RAW per-tier scores and softmaxes once; do not
        # pre-softmax here (a double softmax flattens the distribution and
        # overstates coverage — caught by the live test band).
        return aps_tier_set(scores, q)

    slices = load_loghub_grade_slices()
    loghub = (slices["implicit"] + slices["explicit"])[:400]
    dev = [json.loads(line) for line in
           (ROOT / "data" / "real" / "grade" / "dev.jsonl").read_text().splitlines()
           if line.strip()]
    syn = [(r["text"], r["gold"]) for r in dev if r["family"] == "severity"]
    novel = list(NOVEL_REGISTER_GRADE_CASES)

    probs = {name: [(scores_of(t), gold) for t, gold in data]
             for name, data in [("syn(dev)", syn), ("novel", novel),
                                ("loghub", loghub)]}

    header = f"{'q':>6} | " + " | ".join(
        f"{name}: cover size" for name in probs)
    print(header)
    for q in QUANTILES:
        cells = []
        for name in probs:
            cov = np.mean([gold in set_of(p, q) for p, gold in probs[name]])
            size = np.mean([len(set_of(p, q)) for p, _ in probs[name]])
            cells.append(f"{cov * 100:5.1f}% {size:5.2f}")
        print(f"{q:>6.2f} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
