"""Measure the binary select primitive on the off-JevBench validation set.

The referee set is data/eval/binary-select-v1.jsonl (500 rows, 250
expected-yes / 250 expected-no, mechanical labels, non-JevBench sources —
see its PROVENANCE). Every row is answered exactly the way the shipped
binary select call answers a JevBench noul row:
``matcher.select(text, ["yes", "no"], with_status=True)`` — the fix under
test lives inside select's mapping, so the runner stays mapping-agnostic.

--phrasing measures a SPECIFIC candidate rendering instead (diagnostic
instrument only; it bypasses whatever the select mapping ships):

  bare        ["yes", "no"]                        (production API shape)
  t0-default  yes: the condition holds             (bench status quo, the
              no: the condition does not hold       diagnosed biased pair)
  t2-truefalse    yes: true / no: false
  t3-statement    yes: the statement is true / no: the statement is false
  t4-holdsfails   yes: the condition holds / no: the condition fails
  t5-sonotso      yes: the text states this is so /
                  no: the text states this is not so
  t6-correct      yes: correct / no: incorrect
  t7-affneg       yes: affirmative / no: negative

PRE-REGISTERED DECISION RULE (stated before any off-JevBench number was
read; recorded verbatim in reports/training-report.md):
  Ship template T iff
    (a) expected-yes(T) >= expected-yes(baseline) + 5pp, where baseline is
        the WORSE of {bare, t0-default} on this set, and
    (b) expected-no(T) >= expected-no(baseline) - 1pp (no-regression),
    (c) tie-break: higher overall accuracy, then the shorter template.
  If no template satisfies (a)+(b), the phrasing fix is REFUTED: the
  bias is not phrasing-level, and the report says so. JevBench is only
  scored AFTER a template satisfies the rule on this set.
Every measured template's full table is reported (rule 15: failed lines
included, no cherry-picking).

Usage:
    python scripts/eval_binary_select.py                 # all phrasings
    python scripts/eval_binary_select.py --phrasing bare # one column
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data" / "eval" / "binary-select-v1.jsonl"

# Pre-registered template set (see module docstring). Keys are stable.
PHRASINGS = {
    "bare": ("yes", "no"),
    "t0-default": ("yes: the condition holds",
                   "no: the condition does not hold"),
    "t2-truefalse": ("yes: true", "no: false"),
    "t3-statement": ("yes: the statement is true",
                     "no: the statement is false"),
    "t4-holdsfails": ("yes: the condition holds",
                      "no: the condition fails"),
    "t5-sonotso": ("yes: the text states this is so",
                   "no: the text states this is not so"),
    "t6-correct": ("yes: correct", "no: incorrect"),
    "t7-affneg": ("yes: affirmative", "no: negative"),
}


def load_rows() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text().splitlines()
            if line.strip()]


def evaluate(matcher, rows: list[dict], phrasing: str | None) -> dict:
    """Run the binary select call over every row; phrasing=None uses the
    shipped mapping untouched (candidates passed bare)."""
    by_label = defaultdict(lambda: [0, 0])
    by_family = defaultdict(lambda: [0, 0])
    statuses = defaultdict(lambda: [0, 0])
    misses: list[dict] = []
    yes_cand, no_cand = PHRASINGS[phrasing] if phrasing else ("yes", "no")
    for r in rows:
        cands = [yes_cand, no_cand] if phrasing else ["yes", "no"]
        picked, conf, status = matcher.select(r["text"], cands,
                                              with_status=True)
        # map the picked candidate string back to its label (same rule as
        # scripts/bench_jevbench.py: label = text before the first colon)
        picked_label = str(picked).split(":")[0].strip() if picked else ""
        expected = r["expected"]
        good = picked_label == expected
        by_label[expected][0] += good
        by_label[expected][1] += 1
        by_family[r["family"]][0] += good
        by_family[r["family"]][1] += 1
        statuses[status][0] += good
        statuses[status][1] += 1
        if not good:
            misses.append({"id": r["id"], "expected": expected,
                           "picked": picked_label, "conf": round(conf, 3),
                           "status": status, "family": r["family"]})
    yes_ok, yes_n = by_label["yes"]
    no_ok, no_n = by_label["no"]
    return {
        "phrasing": phrasing or "shipped-mapping",
        "expected_yes": f"{yes_ok}/{yes_n}",
        "expected_no": f"{no_ok}/{no_n}",
        "yes_acc": yes_ok / yes_n if yes_n else 0.0,
        "no_acc": no_ok / no_n if no_n else 0.0,
        "overall": (yes_ok + no_ok) / (yes_n + no_n),
        "by_family": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_family.items())},
        "by_status": {k: f"{v[0]}/{v[1]}" for k, v in sorted(statuses.items())},
        "misses": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrasing", choices=sorted(PHRASINGS), default=None,
                    help="measure one rendering; default = shipped mapping")
    ap.add_argument("--head", choices=("served", "legacy", "select-rank"),
                    default="served",
                    help="which rank head the referee answers with: "
                         "served = the live select surface (wired "
                         "select-rank head, OWNER WIRING DECREE "
                         "2026-10-11), legacy = the guard rank bytes "
                         "explicit-path (reproduces the committed 26.8/"
                         "76.8 baseline), select-rank = the wired head "
                         "explicit-path")
    ap.add_argument("--misses", action="store_true",
                    help="print every miss for the measured column")
    args = ap.parse_args()

    from mivia_small_json.tool_match import (load_matcher,
                                             load_single_matcher,
                                             SELECT_RANK_MODEL,
                                             SELECT_RANK_TOKENIZER,
                                             DEFAULT_MODEL,
                                             DEFAULT_TOKENIZER)

    if args.head == "select-rank":
        matcher = load_single_matcher(SELECT_RANK_MODEL,
                                      tokenizer_path=SELECT_RANK_TOKENIZER)
    elif args.head == "legacy":
        matcher = load_single_matcher(DEFAULT_MODEL,
                                      tokenizer_path=DEFAULT_TOKENIZER)
    else:
        matcher = load_matcher()
    if matcher is None:
        raise SystemExit("shipped guard artifacts not loadable")
    rows = load_rows()

    if args.phrasing is not None:
        results = [evaluate(matcher, rows, args.phrasing)]
    else:
        results = [evaluate(matcher, rows, p) for p in PHRASINGS]

    print(f"set: {DATA.relative_to(ROOT)} ({len(rows)} rows, "
          f"250 yes / 250 no, non-JevBench)")
    header = (f"{'phrasing':16s} {'expected-yes':13s} {'expected-no':13s} "
              f"{'overall':8s}")
    print(header)
    for r in results:
        print(f"{r['phrasing']:16s} {r['expected_yes']:13s} "
              f"({100 * r['yes_acc']:5.1f}%) {r['expected_no']:13s} "
              f"({100 * r['no_acc']:5.1f}%) {100 * r['overall']:5.1f}%")
        print("  by family: " + ", ".join(
            f"{k} {v}" for k, v in r["by_family"].items()))
        print("  by status: " + ", ".join(
            f"{k} {v}" for k, v in r["by_status"].items()))
    if args.misses:
        for r in results:
            print(f"\nmisses ({r['phrasing']}): {len(r['misses'])}")
            for m in r["misses"][:40]:
                print(f"  {m['id']} exp={m['expected']} got={m['picked']} "
                      f"conf={m['conf']} {m['status']} [{m['family']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
