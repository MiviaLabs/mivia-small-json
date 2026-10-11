#!/usr/bin/env python3
"""Diagnose the guard infeasible slice: stratify, score, ablate.

Eval set: ``data/incoming/external/bfcl/guard-eval-v2.jsonl`` rows with
``category == "infeasible"`` (403 rows / 150 catalogues, disjoint from the
guard's training corpus ``pairs.jsonl``). Subtype comes from the row's
``source`` prefix (``acebench_special_{irrelevant,incomplete,error_param}``);
the ACEBench raw dump is not vendored offline, so no finer source parse is
attempted and the prefix is the subtype.

For each row: score through the shipped ``SplitToolMatcher`` gate
(``score_pair`` on the fillability span) and rank raw probability on the
shipped span. Per-catalogue decision is ``max`` over listed tools at the
shipped operating threshold 0.992. Same-request ablations per subtype:

- ``arg_stripped``: numbers removed (eval builder's own strip shape) —
  capability/parameter evidence removed, topic kept.
- ``topic_masked``: alpha tokens of length >= 4 replaced with ``THING`` —
  topic removed, numbers/args kept.

Deterministic: content-hash order, no RNG. CPU-only, stdlib + existing deps.
Prints one JSON summary line.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.tool_match import (  # noqa: E402
    load_matcher,
    render_tool,
    render_tool_fillability,
    validate_fillability,
)

EVAL = ROOT / "data" / "incoming" / "external" / "bfcl" / "guard-eval-v2.jsonl"
OP_THR = 0.992
MOVE_FLOOR = 0.05
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z]{4,}")

SUBTYPES = ("irrelevant", "incomplete", "error_param")


def subtype_of(row: dict) -> str:
    """Subtype from the ACEBench source prefix; 'unknown' when absent."""
    source = str(row.get("source", ""))
    for sub in SUBTYPES:
        if f"special_{sub}" in source:
            return sub
    return "unknown"


def catalogue_id(row: dict) -> str:
    source = str(row.get("source", ""))
    return source.split("#", 1)[1] if "#" in source else source


def strip_args(request: str) -> str:
    return _WS.sub(" ", _NUM.sub(" ", request)).strip()


def mask_topic(request: str) -> str:
    return _WS.sub(" ", _WORD.sub("THING", request)).strip()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").split("\n")
            if line.strip()]
    inf = [r for r in rows if r.get("category") == "infeasible"]
    inf.sort(key=lambda r: _digest(r["request"] + "\x00" + json.dumps(r["tool"], sort_keys=True)))
    matcher = load_matcher()
    if matcher is None or not hasattr(matcher, "gate"):
        print(json.dumps({"error": "shipped SplitToolMatcher unavailable"}))
        return 2
    per_sub: dict[str, dict] = {}
    for sub in SUBTYPES:
        srows = [r for r in inf if subtype_of(r) == sub]
        cats: dict[str, list[float]] = {}
        gate_scores, rank_scores = [], []
        d_arg, d_topic = [], []
        fillable = 0
        for r in srows:
            req, tool = r["request"], r["tool"]
            span = render_tool_fillability(tool, req)
            g = matcher.score_pair(req, span)
            rk = matcher.rank._score_raw(req, render_tool(tool))
            gate_scores.append(g)
            rank_scores.append(rk)
            cats.setdefault(catalogue_id(r), []).append(g)
            sa, st = strip_args(req), mask_topic(req)
            ga = matcher.score_pair(sa, render_tool_fillability(tool, sa)) if sa != req else g
            gt = matcher.score_pair(st, render_tool_fillability(tool, st)) if st != req else g
            d_arg.append(g - ga)
            d_topic.append(g - gt)
            if validate_fillability(tool, req)[0]:
                fillable += 1
        hi_arg = [d for d, g in zip(d_arg, gate_scores) if g >= OP_THR]
        hi_topic = [d for d, g in zip(d_topic, gate_scores) if g >= OP_THR]
        cat_scores = [max(v) for v in cats.values()]
        abst = sum(1 for s in cat_scores if s < OP_THR)
        med = lambda xs: round(statistics.median(xs), 4) if xs else None  # noqa: E731
        per_sub[sub] = {
            "n_rows": len(srows),
            "n_catalogues": len(cats),
            "catalogue_abstain_at_0.992": round(abst / len(cats), 4) if cats else None,
            "pair_false_call_rate_at_0.992": round(
                sum(1 for s in gate_scores if s >= OP_THR) / len(gate_scores), 4)
            if gate_scores else None,
            "median_gate": med(gate_scores),
            "median_rank_raw": med(rank_scores),
            "median_delta_arg_stripped": med(d_arg),
            "median_delta_topic_masked": med(d_topic),
            "frac_arg_moved_ge_0.05": round(sum(1 for d in d_arg if abs(d) >= MOVE_FLOOR)
                                            / len(d_arg), 4) if d_arg else None,
            "frac_topic_moved_ge_0.05": round(sum(1 for d in d_topic if abs(d) >= MOVE_FLOOR)
                                              / len(d_topic), 4) if d_topic else None,
            "fillable_frac": round(fillable / len(srows), 4) if srows else None,
            "n_high_gate_rows": len(hi_arg),
            "hi_median_delta_arg_stripped": med(hi_arg),
            "hi_median_delta_topic_masked": med(hi_topic),
            "hi_frac_topic_moved_ge_0.05": round(
                sum(1 for d in hi_topic if abs(d) >= MOVE_FLOOR) / len(hi_topic), 4)
            if hi_topic else None,
            "hi_frac_arg_moved_ge_0.05": round(
                sum(1 for d in hi_arg if abs(d) >= MOVE_FLOOR) / len(hi_arg), 4)
            if hi_arg else None,
        }
    summary = {
        "eval": "guard-eval-v2 infeasible",
        "disjoint_from_training": True,
        "op_thr": OP_THR,
        "n_rows": len(inf),
        "n_catalogues": len({catalogue_id(r) for r in inf}),
        "n_unknown_subtype": sum(1 for r in inf if subtype_of(r) == "unknown"),
        "subtypes": per_sub,
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
