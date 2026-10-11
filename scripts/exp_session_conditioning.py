#!/usr/bin/env python3
"""Session-conditioning arm for the tool-match guard.

Hypothesis: the guard's input span is ``name: description``, and BFCL builds
its near-miss irrelevance rows by stripping an argument or swapping a sibling
tool, so the guard is asked to refuse on evidence it cannot see. The
fillability arm (reports/exp-span-redesign.md, span v3) made the missing
*argument* visible and cleared the guard. This arm tries the other place the
evidence might live: the **session** -- the prior turns and prior tool calls
that produced the request.

Corpus: ``build_tool_match_corpus.build_session`` appends a session channel to
every span (``<v1 span> | session: <prior calls / prior turns>``) and mixes in
BFCL multi-turn pairs (``multi_turn_miss_func``). Single-turn rows render
``| session: (none)``, which is exactly what the two end-to-end eval sets
render to, because neither carries a session.

Measurement is ``scripts/bench_tool_abstention.py`` verbatim, with ``_load``
wrapped so the production scorer renders the session span. The candidate is
compared to the span-v3 gate (83.5% abstention at matched retention) with an
exact McNemar test on the 884 rows.

Writes only under ``data/incoming/models/`` (artifacts) and ``/tmp`` (corpus,
training log). Never touches ``models/``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json.tool_match import render_tool  # noqa: E402

import build_tool_match_corpus as btc  # noqa: E402
import exp_span_redesign as esr  # noqa: E402

MODELS = ROOT / "data" / "incoming" / "models"
SESSION_PAIRS = Path("/tmp/session-pairs.jsonl")
DEFAULT_OUT = MODELS / "tool-match-session-int8.onnx"
DEFAULT_REPORT = Path("/tmp/session-training.md")
BASELINE = MODELS / "tool-match-spanv3-int8.onnx"
SHIPPED = ROOT / "models" / "tool-match-int8.onnx"


def session_span_fn(tool: dict, request: str, catalogue=None) -> str:
    """The session-conditioned tool span. Eval rows carry no session."""
    return btc.session_span(render_tool(tool), "")


def _bench_module():
    spec = importlib.util.spec_from_file_location(
        "bench_tool_abstention", ROOT / "scripts" / "bench_tool_abstention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scores(model: Path, span_fn, max_len: int = 256) -> tuple[list[float], list[float]]:
    """(fp scores, call scores) for one artifact under one span."""
    bench = _bench_module()
    base = bench._load(model, None, max_len)
    matcher = esr.SpanMatcher(base, span_fn)
    fp = [matcher.best_score(t) for t in bench._texts(bench.FP_EVAL)]
    calls = [matcher.best_score(t) for t in bench._texts(bench.CALLS)]
    return fp, calls


def _frontier(fp: list[float], calls: list[float]) -> dict:
    """Abstention/retention over a 0.001 grid; the best point at >=95% kept."""
    grid = [i / 1000 for i in range(1001)]
    curve = []
    for thr in grid:
        abst = 100 * sum(1 for s in fp if s < thr) / len(fp)
        kept = 100 * sum(1 for s in calls if s >= thr) / len(calls)
        curve.append((thr, abst, kept))
    keep95 = [c for c in curve if c[2] >= 95.0]
    best95 = max(keep95, key=lambda c: c[1]) if keep95 else None
    at60 = min(curve, key=lambda c: abs(c[0] - 0.60))
    return {"curve": curve, "best95": best95, "at_0.6": at60}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial) p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def _decisions(scores: list[float], thr: float) -> list[bool]:
    """True == abstain (score below the threshold)."""
    return [s < thr for s in scores]


def cmd_build(args) -> int:
    rows, stats = btc.build_session(mt_split=args.mt_split)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    for name, value in stats.items():
        print(f"{name}: {value}")
    print(f"total rows: {len(rows)}")
    print(f"written to {out}")
    return 0


def cmd_train(args) -> int:
    out = Path(args.out)
    pairs = Path(args.pairs)
    if not pairs.is_file():
        rows, _stats = btc.build_session(mt_split=args.mt_split)
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        pairs.write_text(payload, encoding="utf-8")
        print(f"built {pairs} ({len(rows)} rows)")
    cmd = [sys.executable, str(ROOT / "scripts" / "train_tool_match.py"),
           "--pairs", str(pairs), "--out", str(out),
           "--report", str(args.report), "--max-len", str(args.max_len),
           "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--seed", str(args.seed),
           "--near-miss-sources", args.near_miss_sources,
           "--near-miss-weight", str(args.near_miss_weight)]
    print("running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def cmd_measure(args) -> int:
    cand = Path(args.model)
    print(f"=== session-conditioned candidate :: {cand.name} ===")
    fp_c, call_c = _scores(cand, session_span_fn, args.max_len)
    fr_c = _frontier(fp_c, call_c)

    print(f"=== baseline span-v3 gate :: {BASELINE.name} ===")
    fp_b, call_b = _scores(BASELINE, esr.SPANS["v3"], args.max_len)
    fr_b = _frontier(fp_b, call_b)

    fp_s, call_s = _scores(SHIPPED, esr.SPANS["v1"], args.max_len)
    fr_s = _frontier(fp_s, call_s)

    def line(name, fr):
        _, a6, k6 = fr["at_0.6"]
        thr, a, k = fr["best95"]
        return f"| {name} | {a:.1f}% | {k:.1f}% | {thr:.3f} | {a6:.1f}% |"

    print()
    print("| model | best abstention at kept>=95% | kept % | thr | abstention at 0.6 |")
    print("|---|---|---|---|---|")
    print(line("shipped L6 (v1 span)", fr_s))
    print(line("span v3 gate (baseline)", fr_b))
    print(line("**session-conditioned**", fr_c))

    thr_c, abst_c, kept_c = fr_c["best95"]
    thr_b, abst_b, kept_b = fr_b["best95"]
    dec_c = _decisions(fp_c, thr_c)
    dec_b = _decisions(fp_b, thr_b)
    b = sum(1 for x, y in zip(dec_c, dec_b) if x and not y)  # cand blocks, base misses
    c = sum(1 for x, y in zip(dec_c, dec_b) if y and not x)  # base blocks, cand misses
    p = mcnemar_exact(b, c)
    print()
    print(f"McNemar on the 884 rows (candidate @{thr_c:.3f} vs span v3 @{thr_b:.3f}): "
          f"candidate blocks {b} more, misses {c}; exact p = {p:.4f}")
    delta = abst_c - abst_b
    print(f"abstention delta {delta:+.1f} pp at kept {kept_c:.1f}% (base kept {kept_b:.1f}%)")

    if args.json:
        payload = {
            "candidate": str(cand),
            "shipped": {"best95": fr_s["best95"], "at_0.6": fr_s["at_0.6"]},
            "baseline": {"best95": fr_b["best95"], "at_0.6": fr_b["at_0.6"]},
            "session": {"best95": fr_c["best95"], "at_0.6": fr_c["at_0.6"]},
            "mcnemar": {"candidate_blocks_more": b, "candidate_misses": c,
                        "p": p, "delta_pp": delta},
            "n_fp": len(fp_c), "n_call": len(call_c),
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"written to {args.json}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build the session corpus")
    p.add_argument("--mt-split", default=btc.MT_DEFAULT, choices=btc.MT_SPLITS)
    p.add_argument("--out", type=Path, default=SESSION_PAIRS)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("train", help="build (if needed) + train the candidate")
    p.add_argument("--mt-split", default=btc.MT_DEFAULT, choices=btc.MT_SPLITS)
    p.add_argument("--pairs", type=Path, default=SESSION_PAIRS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--near-miss-weight", type=float, default=5.0)
    p.add_argument("--near-miss-sources", type=str,
                   default="bfcl_distractor_aligned,bfcl_irrelevance_negatives,"
                           "mt_miss_func,mt_distractor")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("measure", help="bench + frontier + McNemar")
    p.add_argument("--model", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(func=cmd_measure)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
