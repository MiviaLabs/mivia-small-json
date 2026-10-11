"""JevBench head-to-head: mivia-small-json vs the published decision-model board.

Scoring surface: JevBench v1 public splits (MIT,
github.com/fstandhartinger/jevbench — 48 easy / 72 original / 111 hard rows;
types choice/noul/score). mivia is mapped PRIMITIVE-to-PRIMITIVE, not
task-to-task: every row is answered by the shipped zero-shot machinery
(``SplitToolMatcher.select`` — rank artifact over candidate strings with
normalized confidence + ambiguity status), the same primitive the select API
exposes. No per-family tuning; one mapping for all rows:

- ``choice``: select over ``labels`` (label string as its own description)
- ``noul``:   select over ["yes", "no"]
- ``score``:  select over the level strings; expected values compared
              within-1 (ordinal tolerance, same rule as the grade head)

Honesty rules (pre-registered before the run):
- families are scored and REPORTED per family; the generalist families
  (policy/legal/tradeoff/...) are expected losses and are not hidden;
- confidence and ambiguity status are reported, not used to game accuracy;
- published comparisons (Jev 1.13, OpenJev clones) are quoted from their
  boards next to ours in reports/jevbench-head-to-head.md.

Usage:
    python scripts/bench_jevbench.py                # run all three splits
    python scripts/bench_jevbench.py --split original
Writes reports/jevbench-head-to-head.md (appends a dated run section).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "incoming" / "external" / "jevbench"
REPORT = ROOT / "reports" / "jevbench-head-to-head.md"
SPLITS = ("easy", "original", "hard")


def load_rows(split: str) -> list[dict]:
    rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()
            if line.strip()]
    for r in rows:
        r["_expected"] = str(r["expected"])
    return rows


def answer_row(matcher, row: dict) -> tuple[str, float, str]:
    """Answer one JevBench row with the shipped zero-shot select primitive.

    Returns (picked_label, confidence, ambiguity_status).
    """
    q = row["question"]
    qtype = q["type"]
    # JevBench states may be strings OR structured objects (Jev's real
    # interface: "a string, object, or array"). Serialize like any client.
    state = row["state"] if isinstance(row["state"], str) \
        else json.dumps(row["state"], ensure_ascii=False)
    if qtype == "noul":
        candidates = ["yes", "no"]
    elif qtype == "choice":
        candidates = [str(lab) for lab in row["labels"]]
    elif qtype == "score":
        # ordered levels; select over them and compare within-1 by index
        candidates = [str(k) for k in (q.get("criteria") or row["labels"]
                                       or {}).keys()] if isinstance(
            q.get("criteria"), dict) else [str(lab) for lab in row["labels"]]
        if not candidates:
            candidates = [str(lab) for lab in row["labels"]]
    else:
        candidates = [str(lab) for lab in row["labels"]]

    # candidate strings carry their criteria descriptions when given: the
    # rank artifact reads name+description spans, so description-aware
    # candidates are the same interface the select API documents.
    criteria = q.get("criteria")
    if isinstance(criteria, dict) and qtype in ("choice",):
        cands = []
        for lab in candidates:
            desc = criteria.get(lab)
            cands.append(f"{lab}: {desc}" if desc else lab)
    elif qtype == "noul":
        crit = criteria or {}
        cands = []
        for lab in candidates:
            desc = crit.get(lab if isinstance(crit, dict) else None, "")
            default = ("the condition holds" if lab == "yes"
                       else "the condition does not hold")
            cands.append(f"{lab}: {desc or default}")
    else:
        cands = candidates

    picked, conf, status = matcher.select(state, cands, with_status=True)
    # map the picked candidate string back to its label
    picked_label = str(picked).split(":")[0].strip() if picked else ""
    return picked_label, float(conf), status


def score_row(row: dict, picked: str) -> bool:
    q = row["question"]
    if q["type"] == "score" and row["_expected"].isdigit():
        # ordinal tolerance: |picked - expected| <= 1 level
        labels = [str(lab) for lab in row["labels"]]
        if picked in labels and row["_expected"] in labels:
            return abs(labels.index(picked)
                          - labels.index(row["_expected"])) <= 1
        return False
    return picked == row["_expected"]


def run_split(matcher, split: str) -> dict:
    rows = load_rows(split)
    by_type = defaultdict(lambda: [0, 0])
    by_family = defaultdict(lambda: [0, 0])
    by_status = defaultdict(lambda: [0, 0])
    statuses = Counter()
    for row in rows:
        picked, conf, status = answer_row(matcher, row)
        good = score_row(row, picked)
        by_type[row["question"]["type"]][0] += good
        by_type[row["question"]["type"]][1] += 1
        by_family[row["family"]][0] += good
        by_family[row["family"]][1] += 1
        by_status[status][0] += good
        by_status[status][1] += 1
        statuses[status] += 1
    total_ok = sum(v[0] for v in by_type.values())
    total_n = sum(v[1] for v in by_type.values())
    return {"split": split, "rows": total_n, "correct": total_ok,
            "accuracy": round(100 * total_ok / total_n, 2) if total_n else 0.0,
            "by_type": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_type.items())},
            "by_family": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_family.items())},
            # select ALWAYS returns the argmax pick; status is calibrated
            # uncertainty metadata — per-status accuracy is the calibration
            # measurement (ok should beat ambiguous/abstain if calibrated)
            "by_status": {k: f"{v[0]}/{v[1]}={100 * v[0] / v[1]:.0f}%"
                          for k, v in sorted(by_status.items())},
            "statuses": dict(statuses)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default=None)
    args = parser.parse_args()

    from mivia_small_json.pipeline import load_active_router  # noqa: F401
    from mivia_small_json.tool_match import load_matcher

    matcher = load_matcher()
    if matcher is None:
        raise SystemExit("shipped guard artifacts not loadable")

    splits = (args.split,) if args.split else SPLITS
    results = [run_split(matcher, s) for s in splits]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["", f"## mivia-small-json run — {stamp}",
             "",
             "Mapping: primitive-to-primitive (shipped zero-shot "
             "`select`, no per-family tuning). Ordinal tolerance within-1 "
             "on `score` rows (same rule as the grade head).",
             ""]
    for r in results:
        lines.append(f"### split `{r['split']}` — {r['correct']}/{r['rows']} "
                     f"= **{r['accuracy']}%**")
        lines.append("")
        lines.append("- by type: " + ", ".join(f"{k} {v}" for k, v in r["by_type"].items()))
        lines.append("- by family: " + ", ".join(f"{k} {v}" for k, v in r["by_family"].items()))
        lines.append("- by status (calibration check, pick always scored): "
                     + ", ".join(f"{k} {v}" for k, v in r["by_status"].items()))
        lines.append("")

    print("\n".join(lines))
    existing = REPORT.read_text() if REPORT.exists() else (
        "# JevBench head-to-head\n\nPublished references (author/board-measured):\n"
        "- Jev 1.13.0: 63.29 (JevBench v1.4.2.2 board) / 86.51 Borda (S1MB)\n"
        "- OpenJev (ZefanCai 27B clone): 72.35 Borda (S1MB)\n"
        "- OpenJev-Flash-9B (Loop AI): 79.4% — closed 10k bench, author-measured\n"
        "- Hosted Jev: 85.4% — OpenJev's closed 10k bench, author-measured\n")
    REPORT.write_text(existing + "\n".join(lines) + "\n")
    print(f"appended to {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
