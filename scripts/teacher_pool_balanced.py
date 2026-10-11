#!/usr/bin/env python3
"""Tier-balanced teacher pool for business-phrasing severity distill (Track 3).

Failure analysis (reports/grade-business-distill.md): the 79-row tier-skewed
pool (tiers 0/1: 22 rows) shifted CORAL cutpoints; panel 70.0% -> ~55%.
This script builds a tier-balanced pool with tier-0/1 emphasis: >= 40 agreed
rows each for tiers 0 and 1, reusing the prior 79-row pool's agreed tier-2/3
rows (same Qwen teacher family, same dual-prompt protocol, temp 0) as
ballast and generating fresh tier-0/1-heavy candidates.

Protocol (identical to scripts/teacher_label_business.py):
- candidates reserved (hashes + timestamp) BEFORE any teacher call;
- hash-dedupe against grade train/dev, register panel, business slice,
  guard pairs.jsonl, guard-eval-v2, and the prior teacher candidates;
- labels from ollama qwen3-coder:30b-256k under TWO prompts; only rows where
  both prompts agree are kept. Never fake labels: without the teacher the
  script stops after reservation (DESIGN + reservation + baseline only).
- candidates are emitted tier-round-robin so a truncated run stays balanced.

Outputs (gitignored training material, never promoted automatically):
- data/incoming/grade-teacher-balanced/candidates.jsonl
- data/incoming/grade-teacher-balanced/reserve.json
- data/incoming/grade-teacher-balanced/teacher_pool_balanced.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_business_grade_slice import SLICE, corpus_hashes, nhash  # noqa: E402
from teacher_label_business import MODEL, ask, parse1, parse2  # noqa: E402

OUTDIR = ROOT / "data" / "incoming" / "grade-teacher-balanced"
PRIOR_POOL = ROOT / "data" / "incoming" / "grade-teacher" / "teacher_pool.jsonl"
PRIOR_CANDS = ROOT / "data" / "incoming" / "grade-teacher" / "candidates.jsonl"
GUARD_EVAL = ROOT / "data" / "incoming" / "external" / "bfcl" / "guard-eval-v2.jsonl"

PREFIXES_5 = ["", "Overnight, ", "During the peak window, ",
              "Per the status page, ", "As of this morning, "]
PREFIXES_4 = PREFIXES_5[:4]

# Fresh bases: disjoint by authorship from the Track-C BASE (40 rows) and the
# frozen 60-row business slice; the hash ban below enforces disjointness.
TIER0_BASES = [
    "The quarterly dividend notice was mailed to shareholders on schedule",
    "The office relocation wrapped up over the weekend with no equipment lost",
    "The staff canteen menu rotation started Monday to positive feedback",
    "The parking permit renewals were processed before the deadline",
    "The fire drill completed in under four minutes with full attendance",
    "The vendor contracts were countersigned and archived by legal",
    "The holiday gift cards were distributed to all warehouse staff",
    "The lobby renovation passed final inspection yesterday morning",
    "The timesheet system upgrade finished overnight with no data loss",
    "The company shuttle timetable was published for the new quarter",
    "The recycling audit recorded full compliance across all floors",
    "The new badge readers were installed at the side entrance",
    "The employee survey closed with record participation this year",
    "The garden courtyard reopened after the paving work finished",
    "The travel expense backlog cleared after the policy reminder",
    "The meeting room booking pilot ended and rooms show correct availability",
    "The apprenticeship interviews concluded and offers went out Friday",
    "The water coolers were replaced on every floor this week",
    "The charity run raised the target amount for the food bank",
    "The daylight-saving clock change applied cleanly to all wall displays",
    "The onboarding buddy roster was finalized for next month's starters",
    "The quarterly stationery order arrived complete and undamaged",
    "The bike shed extension opened with twenty additional spaces",
    "The all-hands recording was posted to the intranet by noon",
    "The first-aid kit checks passed on all sites this quarter",
    "The car-pool matching board paired thirty commuters successfully",
    "The document shredding pickup ran on its normal Tuesday slot",
    "The reception rota covers the front desk through the holiday break",
    "The paternity-leave policy update was published to the handbook",
    "The team offsite venue confirmed the booking for September",
]

TIER1_BASES = [
    "The invoice approval queue doubled after the approver rota changed and finance is watching it",  # noqa: E501
    "The office wifi drops in the east wing during video calls and IT is investigating",
    "The supplier deliveries slipped two days and safety stock is covering the gap",
    "The helpdesk wait time crept past ten minutes at midday and extra agents were paged",
    "The backup generator test flagged a weak battery that replacement parts will fix next week",
    "The expense report backlog grew after the travel freeze lifted and reviewers are catching up",
    "The warehouse scanner batteries drain faster in the cold store and spares are on order",
    "The customer feedback scores dipped after the packaging change and the team is monitoring returns",  # noqa: E501
    "The print servers jammed twice this week and a technician visit is booked",
    "The overtime hours rose in the packing team and supervisors are rebalancing shifts",
    "The software license renewals lag one department behind and reminders went out",
    "The delivery vans failed two retests at the garage and repairs are scheduled",
    "The meeting room air conditioning struggles on hot afternoons and maintenance is scheduled",
    "The contractor onboarding queue stretched past a week and HR added a second reviewer",
    "The stock counts drifted on three product lines and a recount is planned",
    "The email archive search slowed after the retention change and admins are tuning indexes",
    "The staff turnover ticked up in the call centre and exit interviews are underway",
    "The water pressure dropped on the upper floors and plumbers are tracing the fault",
    "The vendor invoices arrived without purchase order numbers and matching is delayed",
    "The training completion rate slipped below target and managers were asked to nudge teams",
    "The forklift servicing is overdue by a fortnight and the booking is confirmed",
    "The alarm panel throws false alerts on windy nights and engineers are adjusting sensors",
    "The courier pickup missed its slot twice and account managers escalated the route",
    "The canteen card readers reject some new badges and replacements are being issued",
    "The mileage claims piled up after the audit pause and payroll is working through them",
    "The conference bridge echoes for dial-in callers and telecoms supplied a workaround",
    "The pest-control visit found droppings in the loading bay and bait stations go in Friday",
    "The project milestone slipped a week while waiting on vendor samples under review",
    "The visitor sign-in tablet freezes occasionally and a replacement is on order",
    "The heating runs unevenly across the open-plan floor and facilities logged a ticket",
]

P1 = ("You triage operations messages. Classify this message into exactly one "
      "severity tier: INFO (routine, no action), WARNING (degraded or needs "
      "watching), ERROR (real impact, action required), CRITICAL (severe, "
      "urgent, safety/data/existential). Reply with only the tier name.\n\n"
      "Message: {text}\nTier:")
P2 = ("Rate the operational severity of this message: 3 means critical "
      "emergency, 2 means error with real impact, 1 means warning worth "
      "watching, 0 means routine info. Reply with only the digit.\n\n"
      "Message: {text}\nDigit:")


def build_candidates(seed: int) -> list[dict]:
    """Fresh tier-0/1-heavy candidates, tier-round-robin (seeded)."""
    per_tier: dict[int, list[dict]] = {}
    for intended, bases, prefixes in ((0, TIER0_BASES, PREFIXES_5),
                                      (1, TIER1_BASES, PREFIXES_5)):
        rows = []
        for base in bases:
            for pre in prefixes:
                text = base if not pre else pre + base[0].lower() + base[1:]
                rows.append({"text": text, "intended": intended})
        rng = random.Random(seed + intended)
        rng.shuffle(rows)
        per_tier[intended] = rows
    # Round-robin interleave so any prefix of the output stays tier-balanced.
    out: list[dict] = []
    for i in range(max(len(v) for v in per_tier.values())):
        for tier in (0, 1):
            if i < len(per_tier[tier]):
                out.append(per_tier[tier][i])
    return out


def frozen_hashes() -> set[str]:
    frozen = corpus_hashes()
    for line in SLICE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen.add(nhash(json.loads(line)["text"]))
    if GUARD_EVAL.is_file():
        for line in GUARD_EVAL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    for key in ("request", "text", "prompt"):
                        if isinstance(row.get(key), str):
                            frozen.add(nhash(row[key]))
                except (TypeError, ValueError):
                    pass
    for p in (PRIOR_CANDS, PRIOR_POOL):
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        frozen.add(nhash(json.loads(line)["text"]))
                    except (KeyError, TypeError, ValueError):
                        pass
    return frozen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="reserve candidates only, no teacher calls")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    cands = build_candidates(args.seed)
    if args.limit is not None:
        cands = cands[:args.limit]
    frozen = frozen_hashes()
    kept, dropped = [], 0
    for c in cands:
        if nhash(c["text"]) in frozen:
            dropped += 1
            continue
        kept.append(c)
    print(f"candidates: {len(cands)} generated, {dropped} hash-collisions "
          f"dropped, {len(kept)} kept")
    (OUTDIR / "candidates.jsonl").write_text(
        "\n".join(json.dumps(c) for c in kept) + "\n", encoding="utf-8")
    (OUTDIR / "reserve.json").write_text(json.dumps({
        "seed": args.seed,
        "teacher": MODEL,
        "reserved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_candidates": len(kept),
        "hashes": sorted(nhash(c["text"]) for c in kept),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"reserved {len(kept)} candidate hashes BEFORE generation")
    if args.dry_run:
        return 0

    agree = dis = unparse = 0
    pool: list[dict] = []
    pool_path = OUTDIR / "teacher_pool_balanced.jsonl"

    def _flush() -> None:
        pool_path.write_text(
            "\n".join(json.dumps(r) for r in pool) + ("\n" if pool else ""),
            encoding="utf-8")

    for i, c in enumerate(kept):
        a, b = parse1(ask(P1.format(text=c["text"]))), parse2(ask(P2.format(text=c["text"])))  # noqa: E501
        if a is None or b is None:
            unparse += 1
            tag = "unparsed"
        elif a == b:
            agree += 1
            tag = "agree"
            pool.append({"text": c["text"], "gold": a, "intended": c["intended"],  # noqa: E501
                         "teacher": MODEL, "prompts": 2})
        else:
            dis += 1
            tag = "disagree"
        print(f"[{i + 1}/{len(kept)}] P1={a} P2={b} -> {tag}", flush=True)
        if (i + 1) % 10 == 0:
            _flush()
    _flush()
    print(f"agreement: {agree}/{len(kept)} agree, {dis} disagree, {unparse} unparsed")  # noqa: E501
    print("new-pool tier histogram:",
          dict(sorted(Counter(r["gold"] for r in pool).items())), f"n={len(pool)}")  # noqa: E501
    if PRIOR_POOL.is_file():
        lines = PRIOR_POOL.read_text(encoding="utf-8").splitlines()
        prior = [json.loads(ln) for ln in lines if ln.strip()]
        merged = pool + [r for r in prior if r.get("gold") in (2, 3)]
        print("merged (new + prior tier-2/3 ballast) histogram:",
              dict(sorted(Counter(r["gold"] for r in merged).items())),
              f"n={len(merged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
