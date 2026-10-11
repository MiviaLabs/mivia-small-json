#!/usr/bin/env python3
"""CLINC150 intent selection (zero-shot, the universal-structuring 1a baseline).

Same method and primitive as scripts/bench_banking77_select.py: a seed-42
sample of the CLINC150 VALIDATION split (3,100 rows; the test split is
kept untouched as the future one-shot surface for the phase-1 rank
training) scored through the shipped zero-shot ``SplitToolMatcher.select``
— the same primitive the /v1/select API and the routing ladder expose —
with each of the 150 intents described by its name and a short natural
phrasing. No per-intent calibration. This number is the PRE-REGISTERED
baseline for universal-structuring phase 1c: the
trained rank primitive must clear baseline +15pp here AND baseline +15pp
on Banking77, with no regression on the routing bars.

Usage:
    .venv/bin/python scripts/bench_clinc150_select.py [--n 600]
Writes reports/benchmark-clinc150.md (a dated run section) and
.scratch/clinc150-select.jsonl (raw per-row output, gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPLITS = {
    "validation": ROOT / "data/incoming/external/clinc150/validation.jsonl",
    "test": ROOT / "data/incoming/external/clinc150/test.jsonl",
}
LABELS = ROOT / "data/incoming/external/clinc150/labels.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--head", choices=("served", "legacy", "select-rank"),
                    default="served",
                    help="which rank head to score: served = the live "
                         "select surface (wired select-rank head, OWNER "
                         "WIRING DECREE 2026-10-11), legacy = the guard "
                         "rank bytes explicit-path (pre-decree numbers), "
                         "select-rank = the wired head explicit-path")
    ap.add_argument("--matcher", type=Path, default=None,
                    help="candidate int8 ONNX to score instead of the "
                         "shipped guard (one-shot candidate measurement)")
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--label", type=str, default=None,
                    help="candidate label recorded in the report section")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-consumed", action="store_true",
                    help="drop the consumed sample(s) for this split "
                         "(validation seed-42 n=600; test seed-42 n=600 "
                         "x candidate runs)")
    ap.add_argument("--exclude-consumed-seeds", type=str, default="42",
                    help="comma-separated seeds whose samples of this "
                         "split are consumed (the CONSUMED ledger; "
                         "extend it when a new surface is burned)")
    ap.add_argument("--split", type=str, default="validation",
                    choices=sorted(SPLITS),
                    help="CLINC150 split; 'test' is the virgin one-shot "
                         "reserve (validation n=600 seed-42 was consumed "
                         "by the 1a baseline + 1c candidate)")
    args = ap.parse_args()

    from mivia_small_json import bootstrap
    from mivia_small_json.tool_match import (_get_guard, load_single_matcher,
                                             SELECT_RANK_MODEL,
                                             SELECT_RANK_TOKENIZER,
                                             DEFAULT_MODEL,
                                             DEFAULT_TOKENIZER)

    bootstrap.register_all_schemas()
    if args.matcher:
        matcher = load_single_matcher(args.matcher,
                                      tokenizer_path=args.tokenizer)
        if matcher is None:
            print(f"candidate matcher failed to load: {args.matcher}")
            return 2
    elif args.head == "select-rank":
        matcher = load_single_matcher(SELECT_RANK_MODEL,
                                      tokenizer_path=SELECT_RANK_TOKENIZER)
    elif args.head == "legacy":
        matcher = load_single_matcher(DEFAULT_MODEL,
                                      tokenizer_path=DEFAULT_TOKENIZER)
    else:
        matcher = _get_guard()
    if matcher is None:
        print("guard unavailable")
        return 2

    labels = json.loads(LABELS.read_text())
    # Same description recipe as the Banking77 baseline: name plus a
    # natural phrasing, composed once, no per-intent tuning.
    descs = {
        i: f"{i.replace('_', ' ')} ({i.replace('_', ' ')})"
        for i in labels
    }

    rows: list[tuple[str, str]] = []
    with SPLITS[args.split].open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            label = (r.get("intent") or "").strip()
            if text and label:
                rows.append((text, label))
    if args.exclude_consumed:
        consumed: set[tuple[str, str]] = set()
        for s_ in args.exclude_consumed_seeds.split(","):
            if s_.strip():
                consumed |= {(t, ln) for t, ln in
                             random.Random(int(s_)).sample(
                                 rows, min(600, len(rows)))}
        rest = [r for r in rows if r not in consumed]
        sample = random.Random(args.seed).sample(
            rest, min(args.n, len(rest)))
    else:
        sample = random.Random(args.seed).sample(
            rows, min(args.n, len(rows)))

    tag = f"-{args.label}-{args.split}-s{args.seed}" if args.label else ""
    out_path = ROOT / (f".scratch/cand-clinc150-select{tag}.jsonl"
                       if args.matcher
                       else f".scratch/clinc150-select{tag}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top1 = 0
    t0 = time.perf_counter()
    lat: list[float] = []
    with out_path.open("w", encoding="utf-8") as out:
        for text, gold in sample:
            t = time.perf_counter()
            best, conf = matcher.select(text, descs)
            dt = time.perf_counter() - t
            lat.append(dt)
            hit = best == gold
            if hit:
                top1 += 1
            out.write(json.dumps({"gold": gold, "pred": best, "seed": args.seed,
                                  "split": args.split,
                                  "head": args.label or "shipped",
                                  "conf": round(conf, 4), "hit": hit},
                                 ensure_ascii=False) + "\n")
    n = len(sample)
    wall = time.perf_counter() - t0
    lat.sort()
    p50 = lat[len(lat) // 2] * 1000
    p95 = lat[int(len(lat) * 0.95)] * 1000
    acc = 100 * top1 / n
    result = {
        "set": f"clinc150 {args.split} split (official)",
        "seed": args.seed,
        "split": args.split,
        "exclude_consumed": args.exclude_consumed,
        "rows_total": 3100 if args.split == "validation" else 5500,
        "sampled": n,
        "intents": len(labels),
        "primitive": ("ToolMatcher.select (candidate " + (args.label or "") + ")"
                 if args.matcher else
                 "SplitToolMatcher.select (shipped guard, zero-shot)"),
        "top1_accuracy": round(acc, 2),
        "top1_correct": top1,
        "latency_ms_p50": round(p50, 1),
        "latency_ms_p95": round(p95, 1),
        "wall_s": round(wall, 1),
    }
    print(json.dumps(result, indent=1))
    report = ROOT / "reports/benchmark-clinc150.md"
    tag = f" — CANDIDATE {args.label}" if args.label else ""
    section = (f"\n\n## Run {time.strftime('%Y-%m-%d %H:%M')}{tag}\n\n"
               f"{json.dumps(result, indent=1)}\n")
    report.write_text(report.read_text() + section
                      if report.exists() else
                      "# CLINC150 intent selection — zero-shot select()\n"
                      " (universal-structuring phase 1a baseline; the"
                      " test split is untouched for the phase-1c one-shot)\n"
                      + section)
    print(f"written {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
