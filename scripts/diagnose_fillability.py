#!/usr/bin/env python3
"""Fillability contrast diagnosis: does the gate score move when the only
change is the missing required-argument set?

Builds minimal contrast pairs from guard-eval-v2 ``single_turn_relevance``
rows (eval set, disjoint from the guard's training corpus ``pairs.jsonl``):
each pair is (full request, numeric-stripped request) against the SAME tool.
Only pairs whose strip actually breaks fillability
(``validate_fillability`` -> False) are kept, so the should-abstain label on
the stripped side is justified. Stripping is the eval builder's own
construction (``scripts/build_guard_eval.py::_strip_arguments``).

Each pair is scored through the shipped gate artifact (``score_pair`` on the
gate span, gated AND raw to rule out a SelectiveNet collapse) and through the
rank artifact (raw probability on the shipped span).

Deterministic: pair order is content-hash ordered, no RNG. CPU-only, stdlib +
existing deps. Prints one JSON summary line plus per-row deltas.
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
SEED = 0
MOVE_FLOOR = 0.05  # a score move smaller than this is noise on this head
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_WS = re.compile(r"\s+")


def _strip(request: str) -> str:
    return _WS.sub(" ", _NUM.sub(" ", request)).strip()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_pairs() -> list[dict]:
    """Minimal contrast pairs: same tool, required arg present vs stripped."""
    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").split("\n")
            if line.strip()]
    pairs = []
    for r in rows:
        if r.get("category") != "single_turn_relevance":
            continue
        full = r["request"]
        stripped = _strip(full)
        if not stripped or stripped == full:
            continue
        if validate_fillability(r["tool"], stripped)[0]:
            continue  # strip did not break fillability: not a contrast
        pairs.append({"request_full": full, "request_stripped": stripped,
                      "tool": r["tool"], "source": r.get("source", "")})
    pairs.sort(key=lambda p: _digest(p["request_full"] + "\x00"
                                     + json.dumps(p["tool"], sort_keys=True)))
    return pairs


def main() -> int:
    matcher = load_matcher()
    if matcher is None or not hasattr(matcher, "gate"):
        print(json.dumps({"error": "shipped SplitToolMatcher unavailable"}))
        return 2
    pairs = build_pairs()
    deltas_gate, deltas_raw, deltas_rank = [], [], []
    rows_out = []
    for p in pairs:
        tool = p["tool"]
        span_full = render_tool_fillability(tool, p["request_full"])
        span_stripped = render_tool_fillability(tool, p["request_stripped"])
        gf = matcher.score_pair(p["request_full"], span_full)
        gs = matcher.score_pair(p["request_stripped"], span_stripped)
        grf = matcher.gate._score_raw(p["request_full"], span_full)
        grs = matcher.gate._score_raw(p["request_stripped"], span_stripped)
        ship = render_tool(tool)
        rf = matcher.rank._score_raw(p["request_full"], ship)
        rs = matcher.rank._score_raw(p["request_stripped"], ship)
        deltas_gate.append(gf - gs)
        deltas_raw.append(grf - grs)
        deltas_rank.append(rf - rs)
        rows_out.append({
            "source": p["source"], "tool": tool.get("name", ""),
            "gate_full": round(gf, 4), "gate_stripped": round(gs, 4),
            "delta_gate": round(gf - gs, 5),
            "delta_gate_raw": round(grf - grs, 5),
            "delta_rank": round(rf - rs, 5),
            "stripped_span_tail": span_stripped[-100:],
        })
    moved = sum(1 for d in deltas_gate if abs(d) >= MOVE_FLOOR)
    summary = {
        "seed": SEED, "eval": "guard-eval-v2 single_turn_relevance",
        "disjoint_from_training": True, "n_pairs": len(pairs),
        "median_delta_gate": round(statistics.median(deltas_gate), 5)
        if deltas_gate else None,
        "max_abs_delta_gate": round(max(map(abs, deltas_gate)), 5)
        if deltas_gate else None,
        "median_delta_gate_raw": round(statistics.median(deltas_raw), 5)
        if deltas_raw else None,
        "median_delta_rank": round(statistics.median(deltas_rank), 5)
        if deltas_rank else None,
        "n_moved_ge_0.05": moved,
    }
    print(json.dumps(summary))
    for row in rows_out[:10]:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
