#!/usr/bin/env python3
"""Catalogue-as-unit arm for the tool-match guard.

Eleven arms were falsified; the one that worked (span v3) added the missing
*argument* evidence to a single (request, tool) span. The label audit
(`reports/exp-label-audit.md`) and `docs/plans/DATA_TRAINING_PLAN.md` argue the
near-miss evidence is *relational*: BFCL builds a near-miss by stripping an
argument a tool needs or by swapping in a sibling tool, and both only make
sense beside the rest of the catalogue. The falsified v4 span tried to read the
siblings while keeping the pair as the labelled unit; that made one pair's
input a function of the catalogue it sat in, and the same pair carried two
different spans under one label (the invariance break in
`reports/exp-span-redesign.md`).

This arm changes the *unit* instead of the span: score a whole catalogue
against the request and emit one accept/abstain score. The labelled object is
``(request, catalogue)``; a pair carries no label, so the v4 confound cannot
arise. Architecture is the same MiniLM-L6 cross-encoder the guard already
uses, reading ``(request, rendered_catalogue)`` and emitting one match
probability. Training reuses `scripts/train_tool_match.py` unchanged (a
catalogue is just the second text span), so the exported artifact is the same
~23 MB int8 ONNX shape and runs CPU-only at inference.

Two corpora, two renderings
---------------------------
- ``--corpus pairs`` (default): `data/incoming/external/bfcl/pairs.jsonl`
  regrouped by request. A request's stored positive tools plus a deterministic
  sample of its negatives form a label-1 catalogue; the negatives alone form a
  label-0 catalogue. This isolates the *formulation*: the per-tool span is the
  shipped v1 ``name: description``, unchanged from the pair model.
- ``--corpus bfcl``: real BFCL catalogues rebuilt from source (the same
  ``btc._sources()`` the shipped corpus uses) with the fillability-aware v3
  span per tool. A rec with a gold tool is a label-1 catalogue; the same
  catalogue with the gold tool removed is the Hammer-style label-0 near-miss.
  This tests the formulation *plus* the known-good evidence.

Both corpora apply the builder's request-level eval filter, so no request the
two end-to-end eval sets score can train the arm.

Measurement is `scripts/bench_tool_abstention.py`'s decision (abstention on the
884 near-misses, retention on the 183 calls) with a 0.001 threshold sweep, and
an exact McNemar against the shipped span-v3 gate (the 83.5% bar) and the
shipped v1 L6 (80.1%). Writes only under ``data/incoming/models/`` (candidate
artifacts) and ``/tmp`` (corpora, training logs). Never touches ``models/``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json.tool_match import (  # noqa: E402
    _SPLIT,
    parse_catalogue_raw,
    render_tool,
)

import build_tool_match_corpus as btc  # noqa: E402
import exp_span_redesign as esr  # noqa: E402

MODELS = ROOT / "data" / "incoming" / "models"
PAIRS = btc.OUT
DEFAULT_OUT = MODELS / "tool-match-catalogue-int8.onnx"
DEFAULT_CORPUS = Path("/tmp/catalogue-pairs.jsonl")
# The two artifacts the bar is measured against. Both live in models/ and ship.
BASELINE = ROOT / "models" / "tool-match-gate-int8.onnx"   # span v3 gate, 83.5%
SHIPPED = ROOT / "models" / "tool-match-int8.onnx"          # v1 L6, 80.1%
# Catalogue rendering knobs. A catalogue is one cross-encoder input beside the
# request, so a long catalogue truncates at the tail. Per-tool caps keep the
# discriminating head (name + description head, or the v3 fill evidence) in view.
PER_TOOL_CHARS = 200
MAX_TOOLS = 6
NEG_SAMPLE = 3


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _split_of(request: str) -> str:
    """Deterministic request-level 80/20 split (the unit is the catalogue)."""
    return "dev" if int(_digest(request), 16) % 5 == 0 else "train"


def _sample(spans: list[str], k: int, salt: str) -> list[str]:
    """Deterministic content-hash sample of ``spans`` (stable across runs)."""
    ordered = sorted(spans, key=lambda s: _digest(salt + "\x00" + s))
    return ordered[:k]


# ---------------------------------------------------------------------------
# Catalogue rendering.  Signature: (tools, request, mode) -> str.
# ---------------------------------------------------------------------------
def render_catalogue(tools: list[dict], request: str, mode: str = "v1",
                     per_tool_chars: int = PER_TOOL_CHARS,
                     max_tools: int = MAX_TOOLS) -> str:
    """Render a whole catalogue as one compact string.

    ``v1`` renders each tool with the shipped span (``name: description``) and
    truncates it to ``per_tool_chars`` so several tools fit. ``v3`` renders the
    fillability-aware span (``esr.span_v3``: name + required params + the
    values the request supplies + the missing set) and keeps it whole, because
    the fill evidence lives at the tail. Tools are joined by `` || ``.
    """
    parts = []
    for tool in tools[:max_tools]:
        span = esr.span_v3(tool, request) if mode == "v3" else render_tool(tool)
        span = span.strip()
        if per_tool_chars:
            span = span[:per_tool_chars]
        if span:
            parts.append(span)
    return " || ".join(parts)


# ---------------------------------------------------------------------------
# Corpus A: pairs.jsonl regrouped by request.
# ---------------------------------------------------------------------------
def build_pairs_rows(mode: str = "v1", neg_sample: int = NEG_SAMPLE,
                     max_tools: int = MAX_TOOLS,
                     per_tool_chars: int = PER_TOOL_CHARS,
                     ) -> tuple[list[dict], dict[str, int]]:
    """Regroup the committed pairs into catalogues.

    A request's stored positive tool spans plus a deterministic sample of its
    negative spans form the label-1 catalogue; the negatives alone form the
    label-0 catalogue. The negatives include the aligned distractors and the
    irrelevance rows, which is where the sibling evidence lives. The per-tool
    span is the stored (shipped v1) string; ``mode`` is accepted for symmetry
    and only the join/caps differ.
    """
    rows_in = [json.loads(line) for line in
               PAIRS.read_text(encoding="utf-8").split("\n") if line.strip()]
    by_request: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"pos": [], "neg": []})
    for rec in rows_in:
        bucket = "pos" if rec["label"] == 1 else "neg"
        by_request[rec["request"]][bucket].append(rec["tool"])

    stats = {"cat_positive": 0, "cat_negative": 0}
    out: list[dict] = []

    def add(request: str, spans: list[str], label: int, source: str) -> None:
        text = " || ".join(s[:per_tool_chars] for s in spans[:max_tools] if s)
        if not text:
            return
        out.append({"request": request, "tool": text, "label": label,
                    "source": source, "split": _split_of(request)})
        stats[source] += 1

    for request, buckets in by_request.items():
        neg = _sample(buckets["neg"], neg_sample, request) if buckets["neg"] else []
        if buckets["pos"]:
            add(request, buckets["pos"] + neg, 1, "cat_positive")
            add(request, buckets["pos"], 1, "cat_positive")
            if neg:
                add(request, neg, 0, "cat_negative")
        elif neg:
            add(request, neg, 0, "cat_negative")
    return out, stats


# ---------------------------------------------------------------------------
# Corpus B: real BFCL catalogues (source rebuild) with the v3 span.
# ---------------------------------------------------------------------------
def build_bfcl_rows(mode: str = "v3", max_tools: int = MAX_TOOLS,
                    per_tool_chars: int = 0,
                    ) -> tuple[list[dict], dict[str, int]]:
    """Real BFCL catalogues; the negative is the catalogue minus the gold tool.

    This is BFCL's own irrelevance construction: a catalogue where the request's
    gold tool is absent. The label is set-level ("does this catalogue hold a
    tool for the request"), so the negative is a genuine near-miss set, not a
    single swapped pair.
    """
    questions, answers = btc._sources()
    blocked = btc._eval_requests()
    stats = {"cat_positive": 0, "cat_negative": 0,
             "dropped_eval_request": 0, "dropped_empty": 0}
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(request: str, tools: list[dict], label: int, source: str) -> None:
        request = btc._clean(request)
        if request in blocked:
            stats["dropped_eval_request"] += 1
            return
        text = render_catalogue(tools, request, mode, per_tool_chars, max_tools)
        text = btc._clean(text)
        if not text:
            stats["dropped_empty"] += 1
            return
        key = (request, text)
        if key in seen:
            return
        seen.add(key)
        out.append({"request": request, "tool": text, "label": label,
                    "source": source, "split": _split_of(request)})
        stats[source] += 1

    for rec in questions:
        request = rec["question"][0][0]["content"]
        called = {next(iter(g)) for g in answers.get(rec["id"], []) if g}
        catalogue = [t for t in rec["function"] if isinstance(t, dict)]
        gold = [t for t in catalogue if t.get("name") in called]
        rest = [t for t in catalogue if t.get("name") not in called]
        if gold:
            add(request, catalogue, 1, "cat_positive")
        if rest:
            add(request, rest, 0, "cat_negative")

    irr = btc.BFCL / "irrelevance.jsonl"
    if irr.is_file():
        for line in irr.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            tools, request = parse_catalogue_raw(json.loads(line)["text"])
            if tools:
                add(request, tools, 0, "cat_negative")
    return out, stats


# ---------------------------------------------------------------------------
# Corpus C: merged multi-turn rows regrouped by request (the well-defined arm).
#
# The H3 constant-label rule (`multiturn_rows_merged` in
# build_tool_match_corpus.py, `reports/exp-pair-well-definedness.md`) keeps
# only (request, tool) pairs whose label is constant across the three
# multi-turn splits: 62,885 rows kept / 3,604 dropped / 1,345 conflicting
# pairs out of 66,489. A catalogue built from unmerged rows would inherit the
# same conflict (the split changes a turn's gold while leaving the request and
# the catalogue in place), so this corpus regroups the *kept* rows only and
# reports the merge counters verbatim for comparability. The per-tool span is
# the shipped v1 ``name: description`` (``mode`` accepted for symmetry), so
# this arm isolates the unit change like corpus A, on real multi-turn
# catalogues. The builder's request-level eval filter applies, so no request
# either end-to-end eval set scores can train the arm.
# ---------------------------------------------------------------------------
def build_mt_rows(mode: str = "v1", neg_sample: int = NEG_SAMPLE,
                 max_tools: int = MAX_TOOLS,
                 per_tool_chars: int = PER_TOOL_CHARS,
                 ) -> tuple[list[dict], dict[str, int]]:
    kept, merge_stats = btc.multiturn_rows_merged()
    stats: dict[str, int] = dict(merge_stats)
    stats.update({"dropped_eval_request": 0, "cat_positive": 0,
                  "cat_negative": 0})
    blocked = btc._eval_requests()
    # Guard-eval-v2's multi-turn rows come from these same splits, so every
    # one of its 384 multi-turn requests is in the merged rows; the pair-level
    # filter above does not see them. Request-level drop keeps the v2
    # measurement unseen (the same philosophy as _eval_requests).
    gev2 = btc.BFCL / "guard-eval-v2.jsonl"
    if gev2.is_file():
        for line in gev2.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                blocked.add(btc._clean(json.loads(line).get("request") or ""))
    by_request: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"pos": [], "neg": []})
    for rec in kept:
        bucket = "pos" if rec["label"] == 1 else "neg"
        by_request[rec["request"]][bucket].append(rec["tool"])

    out: list[dict] = []

    def add(request: str, spans: list[str], label: int, source: str) -> None:
        if request in blocked:
            stats["dropped_eval_request"] += 1
            return
        text = " || ".join(s[:per_tool_chars] for s in spans[:max_tools] if s)
        if not text:
            return
        out.append({"request": request, "tool": text, "label": label,
                    "source": source, "split": _split_of(request)})
        stats[source] += 1

    for request, buckets in by_request.items():
        neg = _sample(buckets["neg"], neg_sample, request) if buckets["neg"] else []
        if buckets["pos"]:
            add(request, buckets["pos"] + neg, 1, "cat_positive")
            if neg:
                add(request, neg, 0, "cat_negative")
        elif neg:
            add(request, neg, 0, "cat_negative")
    return out, stats


def build_rows(corpus: str, mode: str, **kw) -> tuple[list[dict], dict[str, int]]:
    if corpus == "bfcl":
        return build_bfcl_rows(mode=mode, **kw)
    if corpus == "mt":
        return build_mt_rows(mode=mode, **kw)
    return build_pairs_rows(mode=mode, **kw)


def _write_rows(rows: list[dict], out: Path) -> str:
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Measurement.  bench_tool_abstention.py's decision, with the catalogue render
# swapped in for the per-pair span.
# ---------------------------------------------------------------------------
class CatalogueMatcher(esr.SpanMatcher):
    """ToolMatcher whose ``best_score`` renders the whole catalogue."""

    def __init__(self, base, mode: str, per_tool_chars: int = PER_TOOL_CHARS,
                 max_tools: int = MAX_TOOLS):
        super().__init__(base, None)
        self._mode = mode
        self._per_tool_chars = per_tool_chars
        self._max_tools = max_tools

    def best_match(self, text: str):
        tools, request = parse_catalogue_raw(text)
        if tools is None or not request:
            return None, 1.0
        if not tools:
            return None, 0.0 if _SPLIT in text else 1.0
        catalogue = render_catalogue(tools, request, self._mode,
                                     self._per_tool_chars, self._max_tools)
        score = self._score(request, catalogue)
        return tools, max(score, 0.0)


def _bench_module():
    spec = importlib.util.spec_from_file_location(
        "bench_tool_abstention", ROOT / "scripts" / "bench_tool_abstention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scores(model: Path, mode: str, max_len: int,
            per_tool_chars: int = PER_TOOL_CHARS,
            max_tools: int = MAX_TOOLS) -> tuple[list[float], list[float]]:
    """(near-miss scores, call scores) for one artifact under one render."""
    bench = _bench_module()
    base = bench._load(model, None, max_len)
    matcher = CatalogueMatcher(base, mode, per_tool_chars, max_tools)
    fp = [matcher.best_score(t) for t in bench._texts(bench.FP_EVAL)]
    calls = [matcher.best_score(t) for t in bench._texts(bench.CALLS)]
    return fp, calls


def _scores_span(model: Path, span: str, max_len: int) -> tuple[list[float], list[float]]:
    """(near-miss, call) scores for a per-pair span artifact (baselines)."""
    bench = _bench_module()
    base = bench._load(model, None, max_len)
    matcher = esr.SpanMatcher(base, esr.SPANS[span])
    fp = [matcher.best_score(t) for t in bench._texts(bench.FP_EVAL)]
    calls = [matcher.best_score(t) for t in bench._texts(bench.CALLS)]
    return fp, calls


def _frontier(fp: list[float], calls: list[float]) -> dict:
    """Threshold curves on the 0.001 and the 0.01 grid.

    The pre-registered bar quotes 83.5%, which is the shipped span-v3 gate on
    the 0.01 grid (`reports/exp-span-redesign.md`); the finer 0.001 grid gives
    that same gate 84.2%. Both are reported so the bar is judged on its own
    grid and the finer number is not hidden.
    """
    def best_over(grid):
        curve = []
        for thr in grid:
            abst = 100 * sum(1 for s in fp if s < thr) / len(fp)
            kept = 100 * sum(1 for s in calls if s >= thr) / len(calls)
            curve.append((thr, abst, kept))
        keep95 = [c for c in curve if c[2] >= 95.0]
        best95 = max(keep95, key=lambda c: c[1]) if keep95 else None
        return curve, best95

    curve, best95 = best_over([i / 1000 for i in range(1001)])
    _cc, best95_coarse = best_over([round(x / 100, 2) for x in range(101)])
    at60 = min(curve, key=lambda c: abs(c[0] - 0.60))
    return {"curve": curve, "best95": best95,
            "best95_coarse": best95_coarse, "at_0.6": at60}


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
    rows, stats = build_rows(args.corpus, args.render,
                             max_tools=args.max_tools,
                             per_tool_chars=args.per_tool_chars,
                             **({"neg_sample": args.neg_sample}
                                if args.corpus == "pairs" else {}))
    digest = _write_rows(rows, Path(args.out))
    for name, value in stats.items():
        print(f"{name}: {value}")
    print(f"total rows: {len(rows)}  sha256: {digest}")
    print(f"written to {args.out}")
    return 0


def cmd_train(args) -> int:
    pairs = Path(args.pairs)
    if not pairs.is_file():
        rows, _stats = build_rows(args.corpus, args.render,
                                  max_tools=args.max_tools,
                                  per_tool_chars=args.per_tool_chars,
                                  **({"neg_sample": args.neg_sample}
                                     if args.corpus == "pairs" else {}))
        _write_rows(rows, pairs)
        print(f"built {pairs} ({len(rows)} rows)")
    cmd = [sys.executable, str(ROOT / "scripts" / "train_tool_match.py"),
           "--pairs", str(pairs), "--out", str(args.out),
           "--report", str(args.report), "--max-len", str(args.max_len),
           "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--seed", str(args.seed),
           "--near-miss-sources", "cat_negative",
           "--near-miss-weight", str(args.near_miss_weight)]
    print("running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def cmd_measure(args) -> int:
    cand = Path(args.model)
    print(f"=== catalogue-unit candidate :: {cand.name} "
          f"(render {args.render}) ===")
    fp_c, call_c = _scores(cand, args.render, args.max_len,
                           args.per_tool_chars, args.max_tools)
    fr_c = _frontier(fp_c, call_c)

    print(f"=== baseline span-v3 gate :: {BASELINE.name} ===")
    fp_b, call_b = _scores_span(BASELINE, "v3", args.max_len)
    fr_b = _frontier(fp_b, call_b)

    print(f"=== shipped v1 L6 :: {SHIPPED.name} ===")
    fp_s, call_s = _scores_span(SHIPPED, "v1", args.max_len)
    fr_s = _frontier(fp_s, call_s)

    def line(name, fr):
        _, a6, k6 = fr["at_0.6"]
        thr, a, k = fr["best95"]
        _ct, ca, _ck = fr["best95_coarse"]
        return (f"| {name} | {a:.1f}% | {k:.1f}% | {thr:.3f} | "
                f"{ca:.1f}% | {a6:.1f}% |")

    print()
    print("| model | abstention at kept>=95% | kept % | thr | "
          "abstention (0.01 grid) | abstention at 0.6 |")
    print("|---|---|---|---|---|---|")
    print(line("shipped v1 L6 (80.1%)", fr_s))
    print(line("span v3 gate (83.5% bar)", fr_b))
    print(line("**catalogue-unit**", fr_c))

    results = {}
    for tag, (fr_other, fp_o) in (("vs span-v3 gate", (fr_b, fp_b)),
                                  ("vs shipped v1", (fr_s, fp_s))):
        # The pre-registered bar (83.5%) is quoted on the 0.01 grid, so the
        # headline comparison uses that grid; the finer 0.001 point is in the
        # table above.
        thr_c, abst_c, kept_c = fr_c["best95_coarse"]
        thr_o, abst_o, kept_o = fr_other["best95_coarse"]
        dec_c = _decisions(fp_c, thr_c)
        dec_o = _decisions(fp_o, thr_o)
        b = sum(1 for x, y in zip(dec_c, dec_o) if x and not y)
        c = sum(1 for x, y in zip(dec_c, dec_o) if y and not x)
        p = mcnemar_exact(b, c)
        delta = abst_c - abst_o
        results[tag] = {"b": b, "c": c, "p": p, "delta_pp": delta,
                        "abst": abst_c, "kept": kept_c,
                        "other_abst": abst_o, "other_kept": kept_o,
                        "thr": thr_c, "other_thr": thr_o}
        print()
        print(f"McNemar on the 884 rows (catalogue @{thr_c:.3f} vs {tag} "
              f"@{thr_o:.3f}): candidate blocks {b} more, misses {c}; "
              f"exact p = {p:.4g}")
        print(f"  abstention {abst_c:.1f}% (kept {kept_c:.1f}%) vs "
              f"{abst_o:.1f}% (kept {kept_o:.1f}%); delta {delta:+.1f} pp")

    # Invariance: the catalogue decision must not depend on tool order.
    flips = _order_flips(cand, args.render, args.max_len,
                         args.per_tool_chars, args.max_tools, results)
    print()
    print(f"order-invariance: shuffling tools within each catalogue flips "
          f"{flips['fp_flips']}/{flips['n_fp']} near-miss decisions and "
          f"{flips['call_flips']}/{flips['n_call']} call decisions")

    if args.json:
        payload = {
            "candidate": str(cand), "corpus": args.corpus,
            "render": args.render,
            "shipped": {"best95": fr_s["best95"],
                        "best95_coarse": fr_s["best95_coarse"],
                        "at_0.6": fr_s["at_0.6"]},
            "baseline": {"best95": fr_b["best95"],
                         "best95_coarse": fr_b["best95_coarse"],
                         "at_0.6": fr_b["at_0.6"]},
            "catalogue": {"best95": fr_c["best95"],
                          "best95_coarse": fr_c["best95_coarse"],
                          "at_0.6": fr_c["at_0.6"]},
            "mcnemar": results, "order_invariance": flips,
            "n_fp": len(fp_c), "n_call": len(call_c),
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"written to {args.json}")
    return 0


def _order_flips(model: Path, mode: str, max_len: int, per_tool_chars: int,
                 max_tools: int, results: dict) -> dict:
    """Decision flips when the tools inside each catalogue are shuffled.

    The unit is the set; the render is a sequence. A well-formed set scorer
    should read roughly the same catalogue whatever the tool order, so a high
    flip rate is an invariance cost to report, not a hidden one.
    """
    bench = _bench_module()
    base = bench._load(model, None, max_len)
    matcher = CatalogueMatcher(base, mode, per_tool_chars, max_tools)
    thr = results["vs span-v3 gate"]["thr"]

    def flipped(text: str) -> bool:
        tools, request = parse_catalogue_raw(text)
        if not tools:
            return False
        straight = matcher._score(
            request, render_catalogue(tools, request, mode, per_tool_chars, max_tools))
        shuffled = list(reversed(tools))
        rev = matcher._score(
            request,
            render_catalogue(shuffled, request, mode, per_tool_chars, max_tools))
        return (straight < thr) != (rev < thr)

    fp = [flipped(t) for t in bench._texts(bench.FP_EVAL)]
    calls = [flipped(t) for t in bench._texts(bench.CALLS)]
    return {"n_fp": len(fp), "n_call": len(calls),
            "fp_flips": sum(fp), "call_flips": sum(calls)}


# ---------------------------------------------------------------------------
# Set head: a listwise reranker over the gate's per-tool scores.
#
# The second instantiation of "the catalogue as the labelled unit". It keeps
# the pair scorer's per-pair invariance (the per-tool scores are untouched) and
# learns a catalogue-level decision from the *multiset* of scores. The shipped
# decision is a bare max over pairs, which ignores how many tools there are and
# how close the top two are; both are catalogue-level signals. The head is a
# tiny logistic regression, so the runtime stays sklearn/skops + one 23 MB int8
# ONNX and no torch.
# ---------------------------------------------------------------------------
SETHEAD_FEATURES = ("n", "max", "mean", "min", "std", "margin",
                    "frac_ge_0.9", "frac_ge_0.5")


def sethead_features(scores: list[float]) -> list[float]:
    """Catalogue-level features from one catalogue's per-tool match scores."""
    s = sorted(scores, reverse=True)
    n = len(s)
    mean = sum(s) / n
    std = (sum((x - mean) ** 2 for x in s) / n) ** 0.5
    margin = s[0] - s[1] if n > 1 else 0.0
    return [n, s[0], mean, s[-1], std, margin,
            sum(1 for x in s if x >= 0.9) / n,
            sum(1 for x in s if x >= 0.5) / n]


def _gate_scores(gate, request: str, tools: list[dict]) -> list[float]:
    """Per-tool span-v3 scores for one catalogue, via the shipped gate."""
    return [gate._score(request, esr.span_v3(t, request)) for t in tools]


def _eval_feature_rows(gate, mode: str = "v3") -> tuple[list[list[float]], list[list[float]]]:
    """(near-miss features, call features) from the two end-to-end eval sets."""
    bench = _bench_module()
    out = []
    for path in (bench.FP_EVAL, bench.CALLS):
        feats = []
        for text in bench._texts(path):
            tools, request = parse_catalogue_raw(text)
            if not tools or not request:
                continue
            feats.append(sethead_features(_gate_scores(gate, request, tools)))
        out.append(feats)
    return out[0], out[1]


def build_sethead_rows(gate) -> tuple[list[list[float]], list[int], list[str]]:
    """Catalogue-level features/labels from real BFCL catalogues.

    Positive = the rec's own catalogue (it holds the gold tool). Negative = the
    same catalogue with the gold tool removed (BFCL's own irrelevance
    construction), plus the 240 committed irrelevance rows. The request-level
    eval filter keeps every scored request out.
    """
    questions, answers = btc._sources()
    blocked = btc._eval_requests()
    feats: list[list[float]] = []
    labels: list[int] = []
    splits: list[str] = []

    def add(request: str, tools: list[dict], label: int) -> None:
        request = btc._clean(request)
        if request in blocked or not tools:
            return
        feats.append(sethead_features(_gate_scores(gate, request, tools)))
        labels.append(label)
        splits.append(_split_of(request))

    for rec in questions:
        request = rec["question"][0][0]["content"]
        called = {next(iter(g)) for g in answers.get(rec["id"], []) if g}
        catalogue = [t for t in rec["function"] if isinstance(t, dict)]
        rest = [t for t in catalogue if t.get("name") not in called]
        if any(t.get("name") in called for t in catalogue):
            add(request, catalogue, 1)
        if rest:
            add(request, rest, 0)

    irr = btc.BFCL / "irrelevance.jsonl"
    if irr.is_file():
        for line in irr.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            tools, request = parse_catalogue_raw(json.loads(line)["text"])
            if tools:
                add(request, tools, 0)
    return feats, labels, splits


def cmd_sethead(args) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bench = _bench_module()
    gate = bench._load(BASELINE, None, 256)
    print("building catalogue-level features from the span-v3 gate ...", flush=True)
    feats, labels, splits = build_sethead_rows(gate)
    tr = [i for i, s in enumerate(splits) if s == "train"]
    dv = [i for i, s in enumerate(splits) if s == "dev"]
    x = np.asarray(feats)
    y = np.asarray(labels)
    print(f"set-head corpus: {len(labels)} catalogues "
          f"({len(tr)} train / {len(dv)} dev), "
          f"positives {int(y.sum())} / negatives {int((y == 0).sum())}")

    pipe = Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000))])
    pipe.fit(x[tr], y[tr])
    dev_pred = pipe.predict_proba(x[dv])[:, 1]
    dev_acc = float(((dev_pred >= 0.5).astype(int) == y[dv]).mean())
    print(f"set-head dev accuracy @0.5: {dev_acc:.3f}")

    import skops.io as sio
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(pipe, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / (out.name + ".sha256")).write_text(digest + "\n", encoding="utf-8")
    print(f"exported {out.name} ({out.stat().st_size} bytes) sha256 {digest[:16]}")

    # Measure the two-stage system on the eval sets.
    fp_feats, call_feats = _eval_feature_rows(gate)
    fp_scores = list(pipe.predict_proba(np.asarray(fp_feats))[:, 1])
    call_scores = list(pipe.predict_proba(np.asarray(call_feats))[:, 1])
    fr_c = _frontier(fp_scores, call_scores)

    fp_b, call_b = _scores_span(BASELINE, "v3", 256)
    fr_b = _frontier(fp_b, call_b)
    fp_s, call_s = _scores_span(SHIPPED, "v1", 256)
    fr_s = _frontier(fp_s, call_s)

    def line(name, fr):
        _, a6, k6 = fr["at_0.6"]
        thr, a, k = fr["best95"]
        _ct, ca, _ck = fr["best95_coarse"]
        return (f"| {name} | {a:.1f}% | {k:.1f}% | {thr:.3f} | "
                f"{ca:.1f}% | {a6:.1f}% |")

    print()
    print("| model | abstention at kept>=95% | kept % | thr | "
          "abstention (0.01 grid) | abstention at 0.6 |")
    print("|---|---|---|---|---|---|")
    print(line("shipped v1 L6 (80.1%)", fr_s))
    print(line("span v3 gate (83.5% bar)", fr_b))
    print(line("**set head over v3 gate**", fr_c))

    results = {}
    for tag, (fr_other, fp_o) in (("vs span-v3 gate", (fr_b, fp_b)),
                                  ("vs shipped v1", (fr_s, fp_s))):
        thr_c, abst_c, kept_c = fr_c["best95_coarse"]
        thr_o, abst_o, kept_o = fr_other["best95_coarse"]
        dec_c = _decisions(fp_scores, thr_c)
        dec_o = _decisions(fp_o, thr_o)
        b = sum(1 for p, q in zip(dec_c, dec_o) if p and not q)
        c = sum(1 for p, q in zip(dec_c, dec_o) if q and not p)
        p = mcnemar_exact(b, c)
        results[tag] = {"b": b, "c": c, "p": p, "delta_pp": abst_c - abst_o,
                        "abst": abst_c, "kept": kept_c, "other_abst": abst_o,
                        "other_kept": kept_o, "thr": thr_c, "other_thr": thr_o}
        print()
        print(f"McNemar on the 884 rows (set head @{thr_c:.3f} vs {tag} "
              f"@{thr_o:.3f}): set head blocks {b} more, misses {c}; "
              f"exact p = {p:.4g}")
        print(f"  abstention {abst_c:.1f}% (kept {kept_c:.1f}%) vs "
              f"{abst_o:.1f}% (kept {kept_o:.1f}%)")

    if args.json:
        payload = {"model": str(out), "dev_accuracy": dev_acc,
                   "corpus": {"n": len(labels), "positives": int(y.sum())},
                   "features": list(SETHEAD_FEATURES),
                   "baseline": {"best95": fr_b["best95"],
                                "best95_coarse": fr_b["best95_coarse"]},
                   "sethead": {"best95": fr_c["best95"],
                               "best95_coarse": fr_c["best95_coarse"]},
                   "mcnemar": results}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"written to {args.json}")
    return 0


def cmd_v2(args) -> int:
    """Canonical guard-eval-v2 table with a swapped scorer.

    `bench_tool_abstention.py --eval v2` hardwires the shipped guard, so this
    reuses its canonical `measure_v2` (same catalogue grouping, same
    >=95%-retention operating-point rule) with the candidate artifact loaded
    through the production `ToolMatcher`. Same decision procedure, swapped
    weights — directly comparable to the shipped table and to Step 4.
    """
    import json as _json

    bench = _bench_module()
    rows = [_json.loads(line) for line in
            bench.GUARD_EVAL_V2.read_text(encoding="utf-8").split("\n")
            if line.strip()]
    for tag, model in (("candidate", Path(args.model)),
                       ("shipped-gate", BASELINE), ("shipped-v1", SHIPPED)):
        print(f"--- {tag} :: {model.name} ---")
        matcher = bench._load(model, None, args.max_len)
        bench.measure_v2(rows, matcher=matcher)
        print()
    return 0


def cmd_invariance(args) -> int:
    """The unit-level label is well-defined; the pair-level label is not."""
    rows, _stats = build_rows(args.corpus, args.render,
                              max_tools=args.max_tools,
                              per_tool_chars=args.per_tool_chars,
                              **({"neg_sample": args.neg_sample}
                                 if args.corpus == "pairs" else {}))
    cat_labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        cat_labels[(row["request"], row["tool"])].add(row["label"])
    cat_conflicts = {k: v for k, v in cat_labels.items() if len(v) > 1}
    print(f"(request, catalogue) units: {len(cat_labels)}")
    print(f"  units carrying two labels: {len(cat_conflicts)}")

    pair_labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        for tool in row["tool"].split(" || "):
            pair_labels[(row["request"], tool)].add(row["label"])
    pair_conflicts = [k for k, v in pair_labels.items() if len(v) > 1]
    print(f"(request, tool) pairs seen inside a catalogue: {len(pair_labels)}")
    print(f"  pairs landing in both a label-1 and a label-0 catalogue: "
          f"{len(pair_conflicts)}")
    print("  (a pair carries no label under the catalogue unit; the label "
          "belongs to the catalogue)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--corpus", choices=("pairs", "bfcl", "mt"), default="pairs")
        p.add_argument("--render", choices=("v1", "v3"), default="v1")
        p.add_argument("--max-tools", type=int, default=MAX_TOOLS)
        p.add_argument("--per-tool-chars", type=int, default=PER_TOOL_CHARS)
        p.add_argument("--neg-sample", type=int, default=NEG_SAMPLE)

    p = sub.add_parser("build", help="build the catalogue corpus")
    common(p)
    p.add_argument("--out", type=Path, default=DEFAULT_CORPUS)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("train", help="build (if needed) + train the candidate")
    common(p)
    p.add_argument("--pairs", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--report", type=Path, default=Path("/tmp/catalogue-train.md"))
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--near-miss-weight", type=float, default=3.0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("measure", help="bench + frontier + McNemar + invariance")
    common(p)
    p.add_argument("--model", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("v2", help="canonical guard-eval-v2 tables, swapped scorer")
    p.add_argument("--model", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-len", type=int, default=256)
    p.set_defaults(func=cmd_v2)

    p = sub.add_parser("invariance", help="unit-label vs pair-label conflicts")
    common(p)
    p.set_defaults(func=cmd_invariance)

    p = sub.add_parser("sethead",
                       help="listwise set head over the gate's per-tool scores")
    p.add_argument("--out", type=Path,
                   default=MODELS / "tool-match-catalogue-sethead.skops")
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(func=cmd_sethead)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
