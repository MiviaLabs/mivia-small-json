#!/usr/bin/env python3
"""Teacher-label a business-phrasing train pool offline (Track C, step 2).

Candidate texts are original templated business/SRE phrasings (seeded,
deterministic), deduplicated against the frozen business slice, the register
panel, the grade train/dev corpus, and the guard's pairs file BEFORE any
labelling. Labels come from the offline open-weights teacher
(ollama `qwen3-coder:30b-256k`, Qwen license family — never Llama) under TWO
differently-worded prompts; only rows where both prompts agree are kept
(denoise by multi-prompt agreement). The teacher never sees the eval slice.

Outputs (gitignored training material, never promoted automatically):
- data/incoming/grade-teacher/candidates.jsonl
- data/incoming/grade-teacher/teacher_pool.jsonl (agreed rows only)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_business_grade_slice import SLICE, corpus_hashes, nhash  # noqa: E402

OUTDIR = ROOT / "data" / "incoming" / "grade-teacher"
MODEL = "qwen3-coder:30b-256k"
OLLAMA = "http://localhost:11434/api/generate"

TIER = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}

BASE: dict[int, list[str]] = {
    0: [
        "The weekly sales summary was delivered to regional managers on time",
        "The container fleet renewal passed inspection and paperwork is filed",
        "The training portal enrolled the new cohort without a hitch",
        "The inventory count matched the warehouse system line for line",
        "The sponsorship invoice was paid and the receipt is on record",
        "The compliance training completion rate hit one hundred percent",
        "The branch refurbishment finished two days early and under budget",
        "The customer newsletter went to forty thousand subscribers cleanly",
        "The pension contributions were submitted before the cutoff",
        "The facilities team serviced the backup generators on schedule",
    ],
    1: [
        "The returns queue is growing after the size-guide update and agents are monitoring it",
        "Power consumption in the east hall is trending toward the contract cap",
        "The delivery partner reports scattered delays across two postcodes",
        "The support backlog aged past the weekly target for the first time this quarter",
        "The staging cluster drifts one schema behind production until the nightly sync",
        "Chargeback rates nudged upward in the travel category this month",
        "The translation vendor missed a batch and launch copy may slip a day",
        "Sign-up confirmations are intermittently slow on one mobile carrier",
        "The catering supplier shorted two items for the conference order",
        "Disk alerts fired twice on the archive node but cleared on their own",
    ],
    2: [
        "The booking engine double-charged weekend reservations and reversals are queued",
        "The inventory sync deleted live listings and sellers cannot trade",
        "The payroll run posted wrong deductions and payday corrections are required",
        "The firmware push bricked set-top boxes in three cities",
        "The claims portal rejects valid policy numbers after the upgrade",
        "The dispatch router assigns drivers to closed depots and parcels are stranded",
        "The meter-reading upload failed and billing estimates go out tomorrow",
        "The ticketing platform oversold the Saturday show and compensation is due",
        "The loan origination feed dropped applications submitted after noon",
        "The pharmacy refill line is down and patients are queuing in store",
    ],
    3: [
        "The core banking switch is offline and branches cannot serve customers",
        "A coolant leak forced the evacuation of the fabrication plant",
        "The national lottery draw failed live on air with the jackpot unverified",
        "Intruders exfiltrated the customer database and posted samples publicly",
        "The air-traffic backup system failed during the primary's outage",
        "The pension fund's trading desk executed erroneous block trades at the open",
        "The water utility's treatment controls are unresponsive across two plants",
        "The election results portal shows tampered tallies and officials paused publication",
        "The cargo ship's manifest system lost hazardous-goods declarations mid-voyage",
        "The stock exchange halted trading after the matching engine corrupted the order book",
    ],
}

PREFIXES = ["", "Overnight, ", "During the peak window, ", "Per the status page, "]


def variants(seed: int) -> list[dict]:
    rng = random.Random(seed)
    cands: list[dict] = []
    for intended, bases in BASE.items():
        for base in bases:
            for pre in PREFIXES:
                text = base if not pre else pre + base[0].lower() + base[1:]
                cands.append({"text": text, "intended": intended})
    rng.shuffle(cands)
    return cands


P1 = ("You triage operations messages. Classify this message into exactly one "
      "severity tier: INFO (routine, no action), WARNING (degraded or needs "
      "watching), ERROR (real impact, action required), CRITICAL (severe, "
      "urgent, safety/data/existential). Reply with only the tier name.\n\n"
      "Message: {text}\nTier:")
P2 = ("Rate the operational severity of this message: 3 means critical "
      "emergency, 2 means error with real impact, 1 means warning worth "
      "watching, 0 means routine info. Reply with only the digit.\n\n"
      "Message: {text}\nDigit:")


def ask(prompt: str) -> str:
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0, "num_predict": 8}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            payload = json.loads(r.read())
    except urllib.error.URLError as e:
        raise SystemExit(
            f"teacher unreachable at {OLLAMA} ({e}); start ollama with "
            f"model {MODEL!r} (`ollama pull {MODEL}`) and retry") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(f"teacher returned a non-JSON payload: {e}") from e
    try:
        return payload["response"]
    except (KeyError, TypeError) as e:
        raise SystemExit(
            f"teacher returned no 'response' field: "
            f"{str(payload)[:200]!r}") from e


def parse1(resp: str) -> int | None:
    m = re.search(r"\b(INFO|WARNING|ERROR|CRITICAL)\b", resp.upper())
    return TIER[m.group(1)] if m else None


def parse2(resp: str) -> int | None:
    m = re.search(r"[0-3]", resp)
    return int(m.group(0)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default = 7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="write candidates only, no teacher calls")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    cands = variants(args.seed)
    if args.limit is not None:
        cands = cands[:args.limit]
    frozen = corpus_hashes()
    # The eval slice hashes are part of corpus_hashes via panel? No — the
    # slice is a NEW file; add it explicitly (reserved before generation).
    for line in SLICE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen.add(nhash(json.loads(line)["text"]))
    kept, dropped = [], 0
    for c in cands:
        if nhash(c["text"]) in frozen:
            dropped += 1
            continue
        kept.append(c)
    print(f"candidates: {len(cands)} generated, {dropped} hash-collisions dropped, "
          f"{len(kept)} kept")
    (OUTDIR / "candidates.jsonl").write_text(
        "\n".join(json.dumps(c) for c in kept) + "\n", encoding="utf-8")
    if args.dry_run:
        return 0

    agree = dis = unparse = 0
    pool: list[dict] = []
    pool_path = OUTDIR / "teacher_pool.jsonl"

    def _flush_pool() -> None:
        pool_path.write_text(
            "\n".join(json.dumps(r) for r in pool) + ("\n" if pool else ""),
            encoding="utf-8")

    for i, c in enumerate(kept):
        a, b = parse1(ask(P1.format(text=c["text"]))), parse2(ask(P2.format(text=c["text"])))
        if a is None or b is None:
            unparse += 1
            tag = "unparsed"
        elif a == b:
            agree += 1
            tag = "agree"
            pool.append({"text": c["text"], "gold": a, "intended": c["intended"],
                         "teacher": MODEL, "prompts": 2})
        else:
            dis += 1
            tag = "disagree"
        print(f"[{i + 1}/{len(kept)}] P1={a} P2={b} -> {tag}", flush=True)
        if (i + 1) % 10 == 0:
            _flush_pool()  # crash-safe: agreed rows survive a later failure
    _flush_pool()
    print(f"agreement: {agree}/{len(kept)} agree, {dis} disagree, {unparse} unparsed")
    print("pool tier histogram:",
          dict(sorted(Counter(r["gold"] for r in pool).items())), f"n={len(pool)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
