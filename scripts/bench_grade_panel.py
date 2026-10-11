#!/usr/bin/env python3
"""Report a shipped grade head on the frozen panel, with intervals.

Ticket 04 of result-quality-improvement Prints, per
stratum: the count, the exact accuracy, the within-one-tier accuracy, a 95%
Wilson interval, and a cluster bootstrap interval that resamples whole
sources, because rows from one log system are not independent.

Two shipped heads, selected by ``--head``:

- ``grade`` (default): the cross-encoder anchor scorer
  (``tool_match.load_grade_matcher``), decoded by argmax over the anchors.
- ``ordinal``: the promoted fixed-scale cumulative head
  (``tool_match.load_severity_ordinal``), decoded by
  ``severity_head.cumulative_tier``. This is the path ``pipeline.triage_severity``
  takes for the canonical severity scale.

Read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
ANCHOR_TIER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


def main() -> int:
    from mivia_small_json.severity_head import CANONICAL_SEVERITY_ANCHORS
    from mivia_small_json.stats import cluster_bootstrap_ci, wilson_interval

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=PANEL)
    ap.add_argument("--head", choices=["grade", "ordinal"], default="grade",
                    help="grade = cross-encoder anchor scorer (default); "
                         "ordinal = promoted fixed-scale cumulative head")
    args = ap.parse_args()

    if args.head == "ordinal":
        from mivia_small_json.tool_match import load_severity_ordinal
        head = load_severity_ordinal()
        missing = "ordinal severity head not available"
    else:
        from mivia_small_json.tool_match import load_grade_matcher
        head = load_grade_matcher()
        missing = "grade artifact not available"
    if head is None:
        print(missing, file=sys.stderr)
        return 1

    anchors = list(CANONICAL_SEVERITY_ANCHORS)
    rows = [json.loads(line) for line in
            args.panel.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        anchor, _score = head.grade(row["text"], anchors)
        pred = ANCHOR_TIER[anchor.split(" ")[0]]
        row["dist"] = abs(pred - row["gold"])
        by_stratum[row["stratum"]].append(row)

    def report(label: str, group: list[dict]) -> None:
        n = len(group)
        hits = sum(1 for r in group if r["dist"] == 0)
        within = sum(1 for r in group if r["dist"] <= 1)
        lo, hi = wilson_interval(hits, n)
        clusters: dict[str, list[int]] = defaultdict(list)
        for r in group:
            clusters[r.get("source", "?")].append(1 if r["dist"] == 0 else 0)
        clo, chi = cluster_bootstrap_ci(clusters, seed=42)
        print(f"{label:18s} n={n:3d} exact={100 * hits / n:5.1f}% "
              f"within1={100 * within / n:5.1f}% "
              f"wilson=[{100 * lo:.1f}, {100 * hi:.1f}] "
              f"cluster=[{100 * clo:.1f}, {100 * chi:.1f}]")

    print(f"head: {args.head}")
    for stratum in sorted(by_stratum):
        report(stratum, by_stratum[stratum])
    report("ALL", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
