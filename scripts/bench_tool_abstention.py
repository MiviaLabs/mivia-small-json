#!/usr/bin/env python3
"""Measure a tool-match artifact end to end on the unseen eval sets.

Reads any int8 ONNX artifact, scores the same catalogue-level decision the
pipeline makes (`best_score` over the listed tools against the fixed
threshold), and reports:

- **abstention** on the 884 BFCL `live_irrelevance` rows (near-miss
  catalogues; unseen by the corpus builder)
- **retention** on the 183 BFCL `calls.jsonl` rows (unseen)
- the threshold curve, to show whether the confidence carries signal
- per-pair latency and peak RSS, against the CPU-only budget

The artifact is loaded through the production `ToolMatcher`, so the
measurement cannot drift from the shipped scorer. One pair per model call:
the int8 graph is not batch-invariant (0.527 at batch 1 vs 0.489 at batch 2).

Read-only. Prints the table and optionally writes it to `--json`.

`--eval v2` is the canonical 9-category guard measurement on
`guard-eval-v2.jsonl` (per-category abstention/retention at the operating
point, shipped gate + rank); `build_guard_eval.py measure` redirects to it.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json import tool_match  # noqa: E402
from mivia_small_json.tool_match import (  # noqa: E402
    THRESHOLD,
    ToolMatcher,
)

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
FP_EVAL = BFCL / "irrelevance_eval.jsonl"
CALLS = BFCL / "calls.jsonl"
GUARD_EVAL_V2 = BFCL / "guard-eval-v2.jsonl"

# The canonical guard-eval-v2 measurement (``--eval v2``) covers these 9
# categories. ``single_turn_relevance`` is the retention control (label 1);
# every other category should abstain (label 0).
V2_RELEVANCE = "single_turn_relevance"
V2_CATEGORIES = (
    "single_turn_irrelevance",
    "single_turn_relevance",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "ask_followup",
    "infeasible",
    "foreign_tool",
    "sibling_swap",
    "no_tool_needed",
)
DEFAULT_MODEL = ROOT / "models" / "tool-match-int8.onnx"
DEFAULT_TOKENIZER = ROOT / "models" / "router-tokenizer.json"
# The shipped gate artifact — the thing ``--eval v2`` scores by default. A
# ``--model`` equal to this path is refused: the shipped operating point is
# already measured, and a "candidate" run that re-scores the gate is exactly
# how a false PASS gets made (spec §7.1).
GATE_MODEL = ROOT / "models" / "tool-match-gate-int8.onnx"


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _load(path: Path, tokenizer: Path | None, max_len: int) -> ToolMatcher:
    """The production scorer on any artifact, sha sidecar or not."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    if tokenizer is None or tokenizer == DEFAULT_TOKENIZER:
        cand_tok = path.parent / (path.name.replace("-int8.onnx", "-tokenizer.json"))
        if cand_tok.is_file():
            tokenizer = cand_tok
        else:
            tokenizer = DEFAULT_TOKENIZER

    session = ort.InferenceSession(str(path),
                                   providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(tokenizer))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=None)
    return ToolMatcher(session, tok, [i.name for i in session.get_inputs()])


def _texts(path: Path) -> list[str]:
    return [rec.get("text") or rec.get("raw") or "" for rec in _rows(path)]


# ---------------------------------------------------------------------------
# Canonical guard-eval-v2 measurement (``--eval v2``).
#
# This is the ONE canonical 9-category path:
# ``scripts/build_guard_eval.py measure`` redirects here. Scores use the
# shipped guard (``load_matcher()``, gate + rank); the decision is per
# catalogue (``max`` over the listed tools, grouped by the row's ``source``
# key); the operating point maximizes catalogue abstention subject to >= 95%
# retention on the ``single_turn_relevance`` control.
# ---------------------------------------------------------------------------
def _catalogue_scores(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Group (catalogue key, score) pairs into catalogue scores (max)."""
    out: dict[str, float] = {}
    for key, score in pairs:
        out[key] = max(out.get(key, -1.0), score)
    return out


def _operating_point(neg: list[float], pos: list[float],
                     floor: float = 95.0) -> tuple[float, float, float]:
    """(threshold, abstention %, retention %) maximizing abstention at >= floor."""
    best = None
    for thr in sorted(set(neg + pos)):
        kept = 100 * sum(s >= thr for s in pos) / len(pos) if pos else 100.0
        abst = 100 * sum(s < thr for s in neg) / len(neg) if neg else 0.0
        if kept >= floor and (best is None or abst > best[1]):
            best = (thr, abst, kept)
    return best or (THRESHOLD, 0.0, 100.0)


def _rates(scores: list[float], threshold: float) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    abst = 100 * sum(s < threshold for s in scores) / len(scores)
    return abst, 100 - abst


def _v2_catalogue_text(tool: dict, request: str) -> str:
    """The single-tool catalogue text the shipped guard scores."""
    return ("Tools available:\n" + json.dumps([tool], ensure_ascii=False)
            + "\n\nUser request: " + request)


def measure_v2(rows: list[dict], matcher=None, limit: int = 0,
               model_path: Path | None = None,
               decisions_path: Path | None = None) -> dict:
    """Score guard-eval-v2 rows per category at the operating point.

    ``model_path`` scores a candidate gate artifact through the production
    loader on the fillability span (``load_single_matcher`` with
    ``render_tool_fillability``). The payload stamps the scored artifact's
    path and sha256 so a number can never outlive its artifact; the default
    run stamps the shipped gate.
    """
    from mivia_small_json.tool_match import (load_matcher,
                                             load_single_matcher,
                                             render_tool_fillability)

    artifact_path: Path | None
    if model_path is not None:
        artifact_path = Path(model_path)
        if not artifact_path.is_file():
            raise SystemExit(f"candidate artifact not found: {artifact_path}")
        if GATE_MODEL.is_file() and _sha256(artifact_path) == _sha256(GATE_MODEL):
            raise SystemExit(
                f"--model {artifact_path} IS the shipped gate "
                f"(sha256 {_sha256(GATE_MODEL)[:16]}); the shipped operating "
                "point is already measured — pass the candidate artifact")
        matcher = load_single_matcher(artifact_path,
                                      span_fn=render_tool_fillability)
        if matcher is None:
            raise SystemExit(f"candidate artifact not loadable: {artifact_path}")
        name = f"{artifact_path.name} (candidate gate)"
    else:
        external = matcher is not None
        if matcher is None:
            matcher = load_matcher()
        if matcher is None:
            matcher = _load(DEFAULT_MODEL, None, 256)
            name = f"{DEFAULT_MODEL.name} (single; gate missing)"
            artifact_path = DEFAULT_MODEL
        elif external:
            # A caller-supplied matcher: identity is not stampable — even a
            # SplitToolMatcher here may wrap a CANDIDATE gate, and stamping
            # it as the shipped gate would be the number-outlives-artifact
            # lie this instrument exists to stop.
            name = "external matcher"
            artifact_path = None
        elif isinstance(matcher, tool_match.SplitToolMatcher):
            # load_matcher returns the split gate+rank pair only when the
            # gate loaded clean; stamp what was actually scored.
            name = "shipped split (gate + rank)"
            artifact_path = GATE_MODEL
        else:
            # load_matcher fell back to the rank artifact alone — the gate
            # is missing or failed its sha.
            name = f"{DEFAULT_MODEL.name} (rank-only; gate missing)"
            artifact_path = DEFAULT_MODEL
    print(f"guard: {name}  eval guard-eval-v2 ({len(rows)} rows)")
    if artifact_path is not None and artifact_path.is_file():
        print(f"artifact {artifact_path} sha256 {_sha256(artifact_path)[:16]}")

    if limit:
        capped: list[dict] = []
        seen: dict[str, int] = {}
        for row in rows:
            if seen.get(row["category"], 0) < limit:
                capped.append(row)
                seen[row["category"]] = seen.get(row["category"], 0) + 1
        rows = capped

    by_cat: dict[str, list[tuple[str, float]]] = {c: [] for c in V2_CATEGORIES}
    for row in rows:
        score = matcher.best_score(_v2_catalogue_text(row["tool"],
                                                     row["request"]))
        by_cat[row["category"]].append((row["source"], score))

    neg_cat = {c: _catalogue_scores(v) for c, v in by_cat.items()
               if c != V2_RELEVANCE}
    pos_cat = _catalogue_scores(by_cat[V2_RELEVANCE])
    neg = [s for scores in neg_cat.values() for s in scores.values()]
    pos = list(pos_cat.values())
    thr_op, abst_op, kept_op = _operating_point(neg, pos)
    print(f"operating point (retention >= 95%): thr {thr_op:.3f} -> "
          f"abstention {abst_op:.1f}% / retention {kept_op:.1f}% "
          f"({len(pos)} call catalogues)")

    print(f"{'category':<24} {'pairs':>5} {'catal':>5} "
          f"{'abstain@op':>10} {'retain@op':>9}")
    table = []
    for category in V2_CATEGORIES:
        pairs = by_cat[category]
        if not pairs:
            continue
        cat_scores = list(_catalogue_scores(pairs).values())
        label = 1 if category == V2_RELEVANCE else 0
        cat_abst, cat_kept = _rates(cat_scores, thr_op)
        table.append({"category": category, "label": label,
                      "pairs": len(pairs), "catalogues": len(cat_scores),
                      "catalogue_abstention_at_op": round(cat_abst, 1),
                      "catalogue_retention_at_op": round(cat_kept, 1)})
        print(f"{category:<24} {len(pairs):>5} {len(cat_scores):>5} "
              f"{cat_abst:>9.1f}% {cat_kept:>8.1f}%")
    all_cat = [s for scores in neg_cat.values() for s in scores.values()]
    all_abst, _ = _rates(all_cat, thr_op)
    print(f"{'ALL (label 0)':<24} {sum(len(v) for c, v in by_cat.items() if c != V2_RELEVANCE):>5} "
          f"{len(all_cat):>5} {all_abst:>9.1f}% {100 - all_abst:>8.1f}%")
    if decisions_path is not None:
        # Per-catalogue decisions for paired statistics (McNemar / BCa):
        # catalogue key -> score at the SAME operating point as the payload.
        # Full float precision: the consumer re-derives abstain decisions
        # against this threshold, and 6dp rounding cells (~5e-7) sit inside
        # real score-threshold gaps.
        dump = {"artifact": str(artifact_path) if artifact_path is not None
                else "external matcher",
                "artifact_sha256": (_sha256(artifact_path)
                                    if artifact_path is not None
                                    and artifact_path.is_file() else None),
                "threshold": thr_op,
                "categories": {}}
        for category, scores in by_cat.items():
            dump["categories"][category] = {
                key: score
                for key, score in _catalogue_scores(scores).items()}
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        decisions_path.write_text(json.dumps(dump, indent=1),
                                  encoding="utf-8")
        print(f"per-catalogue decisions -> {decisions_path}")
    return {"eval": "guard-eval-v2", "guard": name,
            "artifact": str(artifact_path) if artifact_path is not None
                        else "external matcher",
            "artifact_sha256": (_sha256(artifact_path)
                                if artifact_path is not None
                                and artifact_path.is_file() else None),
            "threshold": THRESHOLD, "rows": len(rows),
            "operating_point": {"thr": round(thr_op, 4),
                                "abstention": round(abst_op, 1),
                                "retention": round(kept_op, 1)},
            "retention_catalogues": len(pos),
            "catalogue_abstention_at_op": round(all_abst, 1),
            "per_category": table}


def cmd_measure_v2(json_path=None, limit: int = 0,
                   model_path: Path | None = None,
                   decisions_path: Path | None = None) -> int:
    if not GUARD_EVAL_V2.is_file():
        print(f"missing {GUARD_EVAL_V2}; run build_guard_eval.py build first",
              file=sys.stderr)
        return 1
    rows = _rows(GUARD_EVAL_V2)
    payload = measure_v2(rows, limit=limit, model_path=model_path,
                         decisions_path=decisions_path)
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"written to {json_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None,
                        help="artifact to score (default: the shipped "
                             "artifact for the chosen --eval; for v2 every "
                             "explicit path is a candidate gate, and the "
                             "shipped gate path is refused). NOTE: a v2 "
                             "candidate is scored WITHOUT the decision-layer "
                             "fillability veto (plain artifact measurement); "
                             "the shipped reference includes it — the "
                             "comparison is conservative (candidates "
                             "understate deployed abstention)")
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--tier2", type=Path, default=None,
                        help="optional Tier 2 model path for cascaded evaluation")
    parser.add_argument("--tier2-tokenizer", type=Path, default=None)
    parser.add_argument("--tier2-max-len", type=int, default=512)
    parser.add_argument("--cascaded", action="store_true",
                        help="run cascaded Tier 1 -> Tier 2 evaluation")
    parser.add_argument("--label", default="")
    parser.add_argument("--limit", type=int, default=0,
                        help="score at most N rows per set (v1) or per "
                             "category (v2) (smoke tests)")
    parser.add_argument("--eval", choices=("v1", "v2"), default="v1",
                        help="v1: the 884+183 BFCL sets; v2: the 9-category "
                             "guard-eval-v2.jsonl (canonical per-category path)")
    parser.add_argument("--span", choices=("v1", "v2"), default="v1",
                        help="v1 is the shipped name+description span; v2 "
                             "adds the parameter schema")
    parser.add_argument("--fp32", type=Path, default=None,
                        help="optional fp32 parent for an agreement check")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--decisions", type=Path, default=None,
                        help="v2 only: dump per-catalogue scores at the "
                             "operating point for paired statistics")
    args = parser.parse_args()

    if args.eval == "v2":
        # args.model stays None unless the caller typed it: for v2 every
        # explicit path is a candidate (and the shipped gate is refused).
        return cmd_measure_v2(args.json, args.limit, args.model,
                              args.decisions)

    v1_model = args.model or DEFAULT_MODEL

    if args.span == "v2":
        # The parameter-aware span. The shipped artifact is trained on v1, so
        # measuring it under v2 would report a number for a model that never
        # saw parameters.
        tool_match.render_tool = partial(tool_match.render_tool,
                                         parameters=True)

    if args.cascaded or args.tier2 is not None:
        tier1 = _load(v1_model, args.tokenizer, args.max_len)
        tier2_path = (
            args.tier2
            or (ROOT / "data" / "incoming" / "models" / "tool-match-modernbert-int8.onnx")
        )
        tier2 = _load(tier2_path, args.tier2_tokenizer, args.tier2_max_len)
        matcher = tool_match.CascadedToolMatcher(tier1, tier2)
        model_name = f"{v1_model.name} + {tier2_path.name} (cascaded)"
    else:
        matcher = _load(v1_model, args.tokenizer, args.max_len)
        model_name = v1_model.name

    fp_eval = _texts(FP_EVAL)
    calls = _texts(CALLS)
    if args.limit:
        fp_eval, calls = fp_eval[:args.limit], calls[:args.limit]
    print(f"artifact {model_name} · max_len {args.max_len} · "
          f"threshold {THRESHOLD}")

    # The production decision, scored once and reused for every threshold.
    fp_scores = [matcher.best_score(text) for text in fp_eval]
    call_scores = [matcher.best_score(text) for text in calls]

    def abstention(thr: float) -> float:
        return 100 * sum(1 for s in fp_scores if s < thr) / len(fp_scores)

    def retention(thr: float) -> float:
        return 100 * sum(1 for s in call_scores if s >= thr) / len(call_scores)

    curve = [(thr, abstention(thr), retention(thr))
             for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    print(f"abstention {abstention(THRESHOLD):.1f}% ({len(fp_scores)} rows) · "
          f"retention {retention(THRESHOLD):.1f}% ({len(call_scores)} rows)")
    print(f"{'thr':>5} {'abstain %':>10} {'kept %':>8}")
    for thr, abstain, kept in curve:
        print(f"{thr:>5.1f} {abstain:>10.1f} {kept:>8.1f}")

    sample = [t for t in fp_eval[:60] if "User request:" in t]
    latencies = []
    for text in sample:
        start = time.perf_counter()
        matcher.best_score(text)
        latencies.append((time.perf_counter() - start) * 1000)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"latency p50 {p50:.1f} ms · p95 {p95:.1f} ms · peak RSS {peak:.0f} MB")

    agreement = None
    if args.fp32 and args.fp32.is_file():
        fp32 = _load(args.fp32, args.tokenizer, args.max_len)
        diffs = []
        flips = 0
        for text in sample:
            a = matcher.best_score(text)
            b = fp32.best_score(text)
            diffs.append(abs(a - b))
            flips += (a >= THRESHOLD) != (b >= THRESHOLD)
        agreement = {"max_abs_diff": max(diffs), "decision_flips": flips,
                     "n": len(sample)}
        print(f"fp32 vs int8: max |diff| {agreement['max_abs_diff']:.3f} · "
              f"decision flips {flips}/{len(sample)}")

    payload = {
        "label": args.label, "model": str(v1_model),
        "max_len": args.max_len, "threshold": THRESHOLD,
        "abstention_pct": round(abstention(THRESHOLD), 2),
        "retention_pct": round(retention(THRESHOLD), 2),
        "abstention_rows": len(fp_scores), "retention_rows": len(call_scores),
        "curve": [{"thr": t, "abstain": round(a, 2), "kept": round(k, 2)}
                  for t, a, k in curve],
        "latency_p50_ms": round(p50, 2), "latency_p95_ms": round(p95, 2),
        "peak_rss_mb": round(peak, 1), "fp32_agreement": agreement,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
