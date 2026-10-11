#!/usr/bin/env python3
"""Canonical guard-eval-v2 measurement for a conformance-gate candidate.

Mirrors ``bench_tool_abstention.py --eval v2`` (same op-point logic, same
catalogue grouping by ``source`` with per-catalogue ``max``) but:
- scores the CANDIDATE artifact (the canonical path hardcodes the shipped
  guard and ignores ``--model``), and
- renders each row's tool through ``render_conformance_span`` — the span
  the candidate was trained on — instead of the raw JSON blob.

The BEFORE side reproduces the canonical shipped measurement exactly
(same ``_v2_catalogue_text`` blob, shipped matcher) so before/after differ
only in artifact + span. Also reports the error_param sub-slice
(ACEBench Special, the track's target) and an exact McNemar test on
per-catalogue abstention flips. CPU-only, read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_tool_abstention import (  # noqa: E402
    V2_CATEGORIES,
    V2_RELEVANCE,
    _catalogue_scores,
    _load,
    _operating_point,
    _rates,
    _rows,
    _v2_catalogue_text,
)
from mivia_small_json.tool_match import load_matcher  # noqa: E402
from render_conformance_span import render_conformance_span  # noqa: E402

from scipy.stats import binomtest  # noqa: E402

EVAL = ROOT / "data" / "incoming" / "external" / "bfcl" / "guard-eval-v2.jsonl"


def catalogue_key(row: dict) -> str:
    return row["source"]


def score_rows(rows: list[dict], matcher, span_fn, pair_mode: bool
               ) -> dict[str, list[float]]:
    """Category -> per-catalogue max scores, canonical grouping.

    The canonical shipped path scores the parsed catalogue blob
    (``best_score``); a candidate trained on (request, span) pairs scores
    the pair directly (``score_pair``). ``best_score`` returns 1.0 for any
    non-catalogue text, so the conformance span must go through
    ``score_pair``.
    """
    by_cat: dict[str, list[tuple[str, float]]] = {c: [] for c in V2_CATEGORIES}
    for row in rows:
        if pair_mode:
            score = matcher.score_pair(row["request"], span_fn(row))
        else:
            score = matcher.best_score(span_fn(row))
        by_cat[row["category"]].append((catalogue_key(row), score))
    return {c: _catalogue_scores(v) for c, v in by_cat.items()}


def op_threshold(neg_cat: dict[str, dict[str, float]],
                 pos_cat: dict[str, float]) -> float:
    neg = [s for scores in neg_cat.values() for s in scores.values()]
    thr_op, _, _ = _operating_point(neg, list(pos_cat.values()))
    return thr_op


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    rows = _rows(EVAL)

    # BEFORE: canonical shipped measurement, reproduced here so the
    # error_param sub-slice and per-catalogue flips come from one run.
    shipped = load_matcher() or _load(
        ROOT / "models" / "tool-match-int8.onnx", None, 256)
    before = score_rows(rows, shipped, lambda r: _v2_catalogue_text(
        r["tool"], r["request"]), pair_mode=False)

    # AFTER: candidate on the conformance span it was trained on.
    cand = _load(Path(args.model), Path(args.tokenizer), 256)
    after = score_rows(rows, cand, lambda r: render_conformance_span(
        r["tool"], r["request"]), pair_mode=True)

    pos_b = before[V2_RELEVANCE]
    pos_a = after[V2_RELEVANCE]
    neg_b = {c: s for c, s in before.items() if c != V2_RELEVANCE}
    neg_a = {c: s for c, s in after.items() if c != V2_RELEVANCE}
    thr_b = op_threshold(neg_b, pos_b)
    thr_a = op_threshold(neg_a, pos_a)
    print(f"BEFORE op thr {thr_b:.4f}  retention "
          f"{_rates(list(pos_b.values()), thr_b)[1]:.1f}%")
    print(f"AFTER  op thr {thr_a:.4f}  retention "
          f"{_rates(list(pos_a.values()), thr_a)[1]:.1f}%")
    print(f"{'category':<24} {'cat':>5} {'before':>8} {'after':>8} "
          f"{'delta':>8}")
    out = {"model": args.model, "thr_before": round(thr_b, 4),
           "thr_after": round(thr_a, 4), "per_category": []}
    for category in V2_CATEGORIES:
        if category == V2_RELEVANCE:
            continue
        b = _rates(list(before[category].values()), thr_b)[0]
        a = _rates(list(after[category].values()), thr_a)[0]
        print(f"{category:<24} {len(before[category]):>5} "
              f"{b:>7.1f}% {a:>7.1f}% {a - b:>+7.1f}pp")
        out["per_category"].append({"category": category,
                                    "before": round(b, 1),
                                    "after": round(a, 1),
                                    "delta_pp": round(a - b, 1)})

    # error_param sub-slice (the track's target) + exact McNemar.
    ep = [r for r in rows
          if r["source"].startswith("acebench_special_error_param")]
    ep_keys = sorted({catalogue_key(r) for r in ep})
    flip_b = [before["infeasible"][k] < thr_b for k in ep_keys]
    flip_a = [after["infeasible"][k] < thr_a for k in ep_keys]
    b_yes = sum(flip_b)
    a_yes = sum(flip_a)
    n01 = sum(1 for x, y in zip(flip_b, flip_a) if not x and y)
    n10 = sum(1 for x, y in zip(flip_b, flip_a) if x and not y)
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    lift = 100 * (a_yes - b_yes) / len(ep_keys)
    print(f"error_param catalogues {len(ep_keys)}: before {b_yes} "
          f"({100 * b_yes / len(ep_keys):.1f}%) -> after {a_yes} "
          f"({100 * a_yes / len(ep_keys):.1f}%)  lift {lift:+.1f}pp  "
          f"McNemar b={n10} c={n01} p={p:.3g}")
    out["error_param"] = {
        "catalogues": len(ep_keys), "before_abstain": b_yes,
        "after_abstain": a_yes, "lift_pp": round(lift, 1),
        "mcnemar_b": n10, "mcnemar_c": n01, "mcnemar_p": None if not
        (n01 + n10) else round(p, 6),
    }
    # retention control: single_turn_relevance must not regress
    rel_b = _rates(list(pos_b.values()), thr_b)[1]
    rel_a = _rates(list(pos_a.values()), thr_a)[1]
    out["relevance_retention"] = {"before": round(rel_b, 1),
                                  "after": round(rel_a, 1)}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"written to {args.json}")


if __name__ == "__main__":
    main()
