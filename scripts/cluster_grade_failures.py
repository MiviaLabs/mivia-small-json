#!/usr/bin/env python3
"""Report the grade artifact's failure clusters on the frozen panel.

Ticket 14 of `result-quality-improvement`. A uniform error
rate hides structure: the panel's misses are concentrated in a few
(gold -> predicted) cells, and those cells are what targeted synthesis should
be written against.

Prints, per stratum, the most common confusion cells and the adjacent-vs-far
split of the misses. Read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from bench_zero_shot_primitives import SEVERITY_ANCHORS_EXCLUSIVE  # noqa: E402

PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
TIER = {"info": 0, "warning": 1, "error": 2, "critical": 3}
NAMES = ["INFO", "WARN", "ERROR", "CRIT"]


def main() -> int:
    from mivia_small_json.tool_match import load_grade_matcher

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    matcher = load_grade_matcher()
    if matcher is None:
        print("grade artifact not available", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in
            args.panel.read_text(encoding="utf-8").splitlines() if line.strip()]
    confusion: dict[str, Counter] = defaultdict(Counter)
    adjacent = far = 0
    for row in rows:
        anchor, _score, _tier = matcher.grade(
            row["text"], SEVERITY_ANCHORS_EXCLUSIVE, with_set=True)
        pred = TIER[anchor.split(" ")[0]]
        confusion[row["stratum"]][(row["gold"], pred)] += 1
        if pred != row["gold"]:
            if abs(pred - row["gold"]) == 1:
                adjacent += 1
            else:
                far += 1

    for stratum in sorted(confusion):
        total = sum(confusion[stratum].values())
        print(f"=== {stratum} (n={total}) ===")
        for (gold, pred), n in confusion[stratum].most_common(args.top):
            off = "" if gold == pred else f"  off by {abs(gold - pred)}"
            print(f"  gold={NAMES[gold]:5s} pred={NAMES[pred]:5s} "
                  f"n={n:3d} ({100 * n / total:4.1f}%){off}")
    print(f"\nmisses: adjacent={adjacent} far={far} "
          f"total={adjacent + far} of {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
