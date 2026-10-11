#!/usr/bin/env python3
"""Calibrate a conformal false-fire budget for the tool-match guard.

Ticket 10 of result-quality-improvement The guard's
confidence curve is flat, so instead of hunting a threshold we fit one with a
guarantee: conformal risk control (Angelopoulos et al., ICLR 2024,
arXiv:2208.02814) picks the smallest accept threshold whose finite-sample
corrected false-fire risk is at most ``alpha``.

Pool the 880 BFCL near-miss rows (should abstain) with the 183 real calls
(should accept), split each group deterministically by index parity into a
calibration half and a test half, fit lambda on calibration, and report the
achieved rates on test.

The guarantee holds on the calibration distribution. The near-misses are
structurally harder than a random deployment mix, so the test rate is a
measurement, not a second guarantee.

Writes ``data/guard-risk-calibration.json`` bound to the guard sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json import tool_match  # noqa: E402
from mivia_small_json.tool_match import (  # noqa: E402
    ToolMatcher,
    parse_catalogue_raw,
    render_tool,
)

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
FP_EVAL = BFCL / "irrelevance_eval.jsonl"
CALLS = BFCL / "calls.jsonl"
DEFAULT_MODEL = ROOT / "models" / "tool-match-int8.onnx"
DEFAULT_TOKENIZER = ROOT / "models" / "router-tokenizer.json"
OUT = ROOT / "data" / "guard-risk-calibration.json"
ALPHAS = (0.01, 0.05, 0.10)


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


def _score(text: str, matcher: ToolMatcher) -> float | None:
    tools, request = parse_catalogue_raw(text)
    if not tools or not request:
        return None
    return max(matcher._score(request, render_tool(t)) for t in tools)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    matcher = _load(args.model, args.tokenizer, args.max_len)

    # label 0 = near-miss (accepting it is a false fire), 1 = real call.
    labelled: list[tuple[float, int]] = []
    for rows, label in ((_rows(FP_EVAL), 0), (_rows(CALLS), 1)):
        for row in rows:
            score = _score(row["text"], matcher)
            if score is not None:
                labelled.append((score, label))

    # Deterministic split: index parity within each label group.
    calib, test = [], []
    seen = {0: 0, 1: 0}
    for score, label in labelled:
        bucket = calib if seen[label] % 2 == 0 else test
        seen[label] += 1
        bucket.append((score, label))

    results = {}
    for alpha in ALPHAS:
        lam = tool_match.fit_risk_threshold(
            [s for s, _ in calib], [y for _, y in calib], alpha)
        accepted = [(s, y) for s, y in test if s >= lam]
        false_fire = sum(1 for _, y in accepted if y == 0)
        coverage = sum(1 for _, y in accepted if y == 1)
        n_calls = sum(1 for _, y in test if y == 1)
        results[str(alpha)] = {
            "threshold": round(lam, 4),
            "accepted": len(accepted),
            # What CRC bounds: P(accept and near-miss) over the whole set.
            "false_fire_marginal_pct": round(100 * false_fire / len(test), 2),
            # What a caller experiences among the calls it does make.
            "false_fire_conditional_pct": round(100 * false_fire / len(accepted), 2)
            if accepted else 0.0,
            "call_coverage_pct": round(100 * coverage / n_calls, 2) if n_calls else 0.0,
        }

    payload = {
        "artifact": str(args.model.relative_to(ROOT)),
        "artifact_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "n_calibration": len(calib),
        "n_test": len(test),
        "alphas": results,
        "caveat": (
            "Guarantee holds on the calibration distribution. The BFCL "
            "near-misses are structurally harder than a random deployment "
            "mix; the test rate is a measurement, not a second guarantee."
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for alpha, r in results.items():
        print(f"alpha={alpha:>4}  lambda={r['threshold']:.3f}  "
              f"marginal={r['false_fire_marginal_pct']:5.2f}%  "
              f"conditional={r['false_fire_conditional_pct']:5.2f}%  "
              f"coverage={r['call_coverage_pct']:5.2f}%  "
              f"(accepted {r['accepted']}/{len(test)})")
    print(f"written to {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
