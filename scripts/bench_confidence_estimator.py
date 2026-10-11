#!/usr/bin/env python3
"""Compare confidence estimators for the tool-match guard.

Ticket 09 of result-quality-improvement The guard's
confidence curve is flat across thresholds 0.4-0.7, so the question is whether
a different estimator read off the SAME forward pass separates near-miss
irrelevance from real calls. Three are compared at the fixed threshold:

- ``gated``  — the shipped score: ``1 / (1 + exp(z0 - z1))``, collapsed to 0
  when a SelectiveNet head rejects.
- ``raw``    — the same probability without the reject collapse.
- ``pnorm``  — ``max_i z_i / ||z||_p`` (Cattelan & Silva, UAI 2024,
  arXiv:2305.15508), the post-hoc fix for flat-confidence classifiers.

One forward pass per pair; every estimator is derived from its logits.

Read-only. Prints the table and optionally writes it to ``--json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json import tool_match  # noqa: E402
from mivia_small_json.tool_match import (  # noqa: E402
    THRESHOLD,
    ToolMatcher,
    parse_catalogue_raw,
    render_tool,
)

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
FP_EVAL = BFCL / "irrelevance_eval.jsonl"
CALLS = BFCL / "calls.jsonl"
DEFAULT_MODEL = ROOT / "models" / "tool-match-int8.onnx"
DEFAULT_TOKENIZER = ROOT / "models" / "router-tokenizer.json"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _load(path: Path, tokenizer: Path, max_len: int) -> ToolMatcher:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    session = ort.InferenceSession(str(path),
                                   providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(tokenizer))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=None)
    return ToolMatcher(session, tok, [i.name for i in session.get_inputs()])


def _best(text: str, matcher: ToolMatcher, p: float) -> tuple[float, float, float] | None:
    """(gated, raw, pnorm) best over the listed tools, one forward pass each."""
    tools, request = parse_catalogue_raw(text)
    if not tools or not request:
        return None
    gated = raw = pnorm = -1.0
    for tool in tools:
        rendered = render_tool(tool)
        logits = matcher._logits(request, rendered)
        # Same convention as ToolMatcher._predict: sigmoid of the negated diff.
        margin = max(-60.0, min(60.0, logits[0] - logits[1]))
        prob = 1.0 / (1.0 + math.exp(margin))
        gated = max(gated, matcher._score(request, rendered))
        raw = max(raw, prob)
        pnorm = max(pnorm, tool_match.pnorm_confidence(logits, p))
    return gated, raw, pnorm


def _rates(rows: list[dict], matcher: ToolMatcher, p: float) -> dict:
    n = g = r = pn = 0
    for row in rows:
        best = _best(row["text"], matcher, p)
        if best is None:
            continue
        n += 1
        g += best[0] >= THRESHOLD
        r += best[1] >= THRESHOLD
        pn += best[2] >= THRESHOLD
    return {"n": n, "gated": g, "raw": r, "pnorm": pn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    matcher = _load(args.model, args.tokenizer, args.max_len)
    fp = _rates(_rows(FP_EVAL), matcher, args.p)
    calls = _rates(_rows(CALLS), matcher, args.p)

    def abstention_est(x, key):
        return 100.0 * (1 - x[key] / x["n"]) if x["n"] else 0.0

    payload = {
        "threshold": THRESHOLD,
        "p": args.p,
        "irrelevance": {k: v for k, v in fp.items()},
        "calls": {k: v for k, v in calls.items()},
        "abstention_pct": {
            "gated": round(abstention_est(fp, "gated"), 1),
            "raw": round(abstention_est(fp, "raw"), 1),
            "pnorm": round(abstention_est(fp, "pnorm"), 1),
        },
        "retention_pct": {
            "gated": round(100 * calls["gated"] / calls["n"], 1),
            "raw": round(100 * calls["raw"] / calls["n"], 1),
            "pnorm": round(100 * calls["pnorm"] / calls["n"], 1),
        },
    }
    for est in ("gated", "raw", "pnorm"):
        print(f"{est:6s} abstention={payload['abstention_pct'][est]:5.1f}% "
              f"retention={payload['retention_pct'][est]:5.1f}%")
    if args.json:
        args.json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
