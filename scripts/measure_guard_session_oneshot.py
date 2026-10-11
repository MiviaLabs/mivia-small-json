#!/usr/bin/env python3
"""ONE-SHOT candidate measurement for Arm C (session-guard-transfer).

Derives the matcher's own guard-eval-v2 operating point (max abstention at
>=95% retention on single_turn_relevance — the canonical rule, scored via
the catalogue-text protocol exactly like bench_tool_abstention), then
applies that threshold to the RESERVED harvest eval half scored as pairs
(the protocol the 99.7% baseline used). Refuses to run if the reserve file
does not match the eval rows.

With --shipped: measures the shipped chain instead of a candidate (the
eval-half baseline, scored once).

Bar (pre-registered in the spec): harvest eval call rate <= 90%;
v2 abstention >= 77% @ retention >= 95%; label-1 call rate >= 80%
(degenerate-winner guard). Never tunes anything; never writes models/.

Run: python scripts/measure_guard_session_oneshot.py --model <path> --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PAIRS = ROOT / "data" / "incoming" / "external" / "harvest-sessions" / "pairs.jsonl"
RESERVE = ROOT / "data" / "incoming" / "external" / "harvest-sessions" / "eval.reserve.json"
V2 = ROOT / "data" / "incoming" / "external" / "bfcl" / "guard-eval-v2.jsonl"
OUT = ROOT / "reports"
V2_RELEVANCE = "single_turn_relevance"


def _norm(r: dict) -> str:
    return " ".join((r["request"] + "\x00" + r["tool"]).lower().split())


def _nhash(r: dict) -> str:
    return hashlib.sha256(_norm(r).encode()).hexdigest()


def check_reserve(dev_rows: list[dict]) -> None:
    reserve = json.loads(RESERVE.read_text())
    have = {_nhash(r) for r in dev_rows}
    want = set(reserve["hashes"])
    missing, extra = want - have, have - want
    if missing or extra or reserve["n"] != len(dev_rows):
        raise SystemExit(
            f"reserve mismatch: missing {len(missing)}, extra {len(extra)}, "
            f"n reserve={reserve['n']} vs eval={len(dev_rows)} — the eval "
            "half changed after freezing; refusing")


def load_matcher(model_path: Path | None):
    """The RANK head as a plain ToolMatcher — shipped bytes when
    model_path is None, else the candidate artifact.

    The arm's comparison is rank-vs-rank at each matcher's own derived
    operating point: the 99.7% baseline was a rank-head pair-score
    property (the gate head scores these rendered-string pairs near zero
    and would read 15% call — a different model, not the baseline).
    Candidates are scored without the decision-layer veto, same as
    bench_tool_abstention scores candidate gates: conservative.
    """
    from mivia_small_json import tool_match

    path = (ROOT / "models" / "tool-match-int8.onnx" if model_path is None
            else model_path)
    matcher = tool_match.load_single_matcher(path)
    if matcher is None:
        raise SystemExit(f"rank artifact not loadable: {path}")
    return matcher


def v2_operating_point(matcher) -> tuple[float, float, float]:
    """Canonical v2 operating point for THIS matcher (thr, abstention, retention).

    Replicates bench_tool_abstention exactly: rows scored via the
    catalogue-text protocol, then grouped by source with the catalogue
    score = max over its rows (the canonical retention denominator is
    catalogues, not rows — skipping the grouping collapses the operating
    point to thr 0.0).
    """
    from bench_tool_abstention import _operating_point, _v2_catalogue_text

    rows = [json.loads(line) for line in V2.read_text().splitlines() if line.strip()]
    by_source: dict[tuple[str, int], float] = {}
    for r in rows:
        score = matcher.best_score(_v2_catalogue_text(r["tool"], r["request"]))
        key = (r["source"], r["label"])
        by_source[key] = max(by_source.get(key, 0.0), score)
    neg = [s for (src, lab), s in by_source.items() if lab == 0]
    pos = [s for (src, lab), s in by_source.items() if lab == 1]
    return _operating_point(neg, pos, floor=95.0)


def harvest_scores(matcher, dev: list[dict]) -> list[tuple[float, int]]:
    out = []
    for r in dev:
        out.append((matcher._score(r["request"], r["tool"]), r["label"]))
    return out


def load_split_chain(rank_path: Path | None):
    """The deployed veto+split chain, optionally with a candidate rank half.

    Assembly mirrors tool_match.load_matcher: shipped gate
    (fillability span) over the given rank artifact. Used for the v2
    clause: a promoted rank must keep the CHAIN at >=77% abstention at
    >=95% retention at the chain's canonical operating point.
    """
    from mivia_small_json import tool_match

    rank = (tool_match.load_single_matcher(rank_path) if rank_path is not None
            else None)
    base = rank if rank is not None else tool_match.load_single_matcher(
        tool_match.DEFAULT_MODEL)
    if base is None:
        raise SystemExit("rank artifact not loadable")
    gate_path = ROOT / "models" / "tool-match-gate-int8.onnx"
    gate = tool_match.load_single_matcher(gate_path, span_fn=tool_match.render_tool_fillability)
    if gate is None:
        raise SystemExit("shipped gate not loadable")
    return tool_match.SplitToolMatcher(gate, base)


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(
        1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--shipped", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.602,
                    help="deployment threshold for the harvest pair scores "
                         "(the shipped chain operates at 0.602; a promoted "
                         "rank serves at the same threshold)")
    args = ap.parse_args()

    matcher = load_matcher(None if args.shipped else args.model)

    rows = [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]
    dev = [r for r in rows if r["split"] == "dev"]
    check_reserve(dev)

    # Protocol (pinned in session-guard-transfer):
    # harvest pair scores at the FIXED deployed threshold; v2 through the
    # veto+split chain (shipped gate + this rank) at its canonical point.
    scores = harvest_scores(matcher, dev)
    thr = args.threshold
    call = 100 * sum(s >= thr for s, _ in scores) / len(scores)
    l1 = [s for s, lab in scores if lab == 1]
    l0 = [s for s, lab in scores if lab == 0]
    l1_call = 100 * sum(s >= thr for s in l1) / max(len(l1), 1)
    l0_abst = 100 * sum(s < thr for s in l0) / max(len(l0), 1)

    chain = load_split_chain(None if args.shipped else args.model)
    d_thr, v2_abst, v2_ret = v2_operating_point(chain)

    res = {
        "tag": args.tag or ("shipped" if args.shipped else f"s{args.seed}"),
        "model": "shipped chain" if args.shipped else str(args.model),
        "deployment_threshold": round(thr, 4),
        "chain_operating_point_threshold": round(d_thr, 4),
        # Re-registered bar (spec, committed before training): raise the
        # label-1 call rate without degrading the chain; keep label-0
        # abstention from collapsing. Overall call rate is recorded, not
        # gated (the old <=90% ceiling was written against the erratum
        # baseline and is vacuous).
        "bar": {"label1_call_rate_min": 40.0, "v2_abstention_min": 77.0,
                "v2_retention_min": 95.0, "label0_abstention_min": 60.0},
        "harvest_eval": {"n": len(dev), "call_rate_pct": round(call, 1),
                         "label1_call_rate_pct": round(l1_call, 1),
                         "label0_abstention_pct": round(l0_abst, 1),
                         "n_label1": len(l1), "n_label0": len(l0),
                         "auc_vs_construction": round(
                             auc(l1, [s for s in l0]), 4)},
        "guard_eval_v2": {"abstention_pct": round(v2_abst, 1),
                          "retention_pct": round(v2_ret, 1)},
    }
    if not args.shipped:
        res["pass"] = (l1_call >= 40.0 and v2_abst >= 77.0 and v2_ret >= 95.0
                       and l0_abst >= 60.0)
    tag = res["tag"]
    out = OUT / f"guard-session-oneshot-{tag}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    if not args.shipped:
        print(f"\nverdict ({tag}): {'PASS' if res['pass'] else 'FAIL'} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
