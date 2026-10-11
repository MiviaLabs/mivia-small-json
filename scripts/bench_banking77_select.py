#!/usr/bin/env python3
"""Banking77 intent selection (openjev-style large-set benchmark).

The product claim this measures: dump any text plus a label schema, get
the right value predicted — no training, no per-task tuning. Banking77
(public research corpus, 77 banking intents, ~13k rows total) is the
biggest such set in the repo: the eval scores a seed-42 sample of the
3,080-row official test split through the shipped zero-shot
``SplitToolMatcher.select`` primitive — the same primitive the /v1/select
API and the routing ladder expose — with each intent described by its
name and a short natural phrasing. No per-intent calibration.

Usage:
    .venv/bin/python scripts/bench_banking77_select.py [--n 600]
Writes reports/benchmark-banking77.md (a dated run section) and
.scratch/banking77-select.jsonl (raw per-row output, gitignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CSV = ROOT / "data/incoming/external/banking77/test.csv"
CATS = ROOT / "data/incoming/external/banking77/categories.json"


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
                    help="drop every CONSUMED sample's rows (recomputed "
                         "deterministically from the csv): the fresh "
                         "one-shot reserve must not overlap them")
    ap.add_argument("--exclude-consumed-seeds", type=str, default="42,43",
                    help="comma-separated seeds whose samples are "
                         "consumed (the CONSUMED ledger; extend it when "
                         "a new surface is burned)")
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
        # OWNER WIRING DECREE 2026-10-11: explicit-path head picks so the
        # pre-wiring numbers stay reproducible regardless of serving state.
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

    intents = json.loads(CATS.read_text())
    # Each intent is described by its name plus a natural phrasing —
    # composed once, no per-intent tuning.
    descs = {
        i: f"{i.replace('_', ' ')} ({i.replace('_', ' ')})"
        for i in intents
    }

    rows: list[tuple[str, str]] = []
    with CSV.open(encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            text = (r.get("text") or "").strip()
            label = (r.get("category") or r.get("label") or "").strip()
            if text and label:
                rows.append((text, label))
    if args.exclude_consumed:
        consumed = set()
        for s_ in args.exclude_consumed_seeds.split(","):
            if s_.strip():
                consumed |= {(t, line) for t, line in
                             random.Random(int(s_)).sample(
                                 rows, min(600, len(rows)))}
        rest = [r for r in rows if r not in consumed]
        sample = random.Random(args.seed).sample(
            rest, min(args.n, len(rest)))
    else:
        sample = random.Random(args.seed).sample(
            rows, min(args.n, len(rows)))

    tag = f"-{args.label}-s{args.seed}" if args.label else ""
    out_path = ROOT / (f".scratch/cand-banking77-select{tag}.jsonl"
                       if args.matcher
                       else f".scratch/banking77-select{tag}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top1 = 0
    mrr = 0.0
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
                mrr += 1.0
            out.write(json.dumps({"gold": gold, "pred": best, "seed": args.seed,
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
        "set": "banking77 test split (official)",
        "seed": args.seed,
        "exclude_consumed": args.exclude_consumed,
        "rows_total": 3080,
        "sampled": n,
        "intents": len(intents),
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
    report = ROOT / "reports/benchmark-banking77.md"
    tag = f" — CANDIDATE {args.label}" if args.label else ""
    section = (f"\n\n## Run {time.strftime('%Y-%m-%d %H:%M')}{tag}\n\n"
               f"{json.dumps(result, indent=1)}\n")
    report.write_text(report.read_text() + section
                      if report.exists() else
                      "# Banking77 intent selection — zero-shot select()\n"
                      + section)
    print(f"written {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
