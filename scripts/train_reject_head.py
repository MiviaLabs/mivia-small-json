#!/usr/bin/env python3
"""TRACK D: trained reject option for the residual ask_followup catalogues.

Context: pipeline routes guard-abstained questions to tool_clarification
(51.8%, zero regressions); the residual ~118 ask_followup catalogues are
high-confidence topical matches (114 >= 0.99) that no rule separates from
single_turn_relevance calls. This script:

1. reproduces the residual set deterministically from guard-eval-v2 +
   the shipped guard (``load_matcher`` only; never retrains/replaces it);
2. trains a small logistic reject head (CPU sklearn, seeds 7/8) on scalar
   features the rank scorer underuses (fillability-vs-topicality gap,
   question-shape x high-score conjunction) reusing shipped scores;
3. measures canonical guard-eval-v2 per-category before/after at the same
   operating point (matched >= 95% retention).

This is NOT a re-run of any falsified arm: no cross-encoder fine-tune
(defer head), no SelectiveNet/SNGP/QAT/late-interaction/margin/span change,
no session channel, no near-miss rebalancing, no catalogue-as-unit. The
weights are a 7-feature linear model on frozen shipped scores.

Training rows come from pairs.jsonl (positives = keep, negatives =
reject), verified hash-disjoint from every guard-eval-v2 request. Only the
operating threshold tau is selected on v2 (same status as the canonical
operating point itself); weights never see v2.

CPU-only by design: a convex 7-feature logistic fit on ~8k rows takes
milliseconds; a GPU (RTX 5080 present, torch.cuda True) buys nothing.
Candidates only under gitignored data/incoming/models/, never promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.tool_match import load_matcher  # noqa: E402

V2 = ROOT / "data" / "incoming" / "external" / "bfcl" / "guard-eval-v2.jsonl"
PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
OUT_DIR = ROOT / "data" / "incoming" / "models"
SCORE_CACHE = OUT_DIR / "reject-head-v2-scores.json"

SEEDS = (7, 8)
ASK = "ask_followup"
REL = "single_turn_relevance"


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "bta", ROOT / "scripts" / "bench_tool_abstention.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BTA = _load_bench()

_Q_OPEN = re.compile(
    r"^\s*(who|what|when|where|why|how|which|whose|whom|can|could|would|"
    r"should|is|are|do|does|did|tell|explain|describe|list|give|show|"
    r"help|need|want|looking|searching|any|does anyone|what is|what are)\b",
    re.IGNORECASE,
)


def is_question(text: str) -> int:
    return int("?" in text or bool(_Q_OPEN.search(text or "")))


def _tok(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def fill_proxy(tool: dict, request: str) -> tuple[float, int]:
    """Fraction of required params whose name tokens appear in the request."""
    try:
        params = (tool or {}).get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        req = params.get("required", []) or []
    except AttributeError:
        return 1.0, 0
    if not req:
        return 1.0, 0
    words = _tok(request)
    hit = 0
    for name in req:
        desc = ""
        try:
            desc = (props.get(name, {}) or {}).get("description", "") or ""
        except AttributeError:
            desc = ""
        if (_tok(str(name)) | _tok(desc)) & words:
            hit += 1
    return hit / len(req), len(req)


def features(score: float, tool: dict, request: str) -> list[float]:
    fill, nreq = fill_proxy(tool, request)
    q = is_question(request)
    return [
        score,
        float(q),
        score * q,
        score - fill,
        (1.0 - fill) * score,
        __import__("math").log1p(len(request)) / 10.0,
        nreq / 5.0,
    ]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _digest(text: str) -> str:
    return hashlib.sha256(_norm(text).encode()).hexdigest()


def _guard_fingerprint() -> str:
    """Short digest of the shipped guard artifacts AND decision rule.

    Cache keys carry it, so scores cached under a different guard (or a
    hand-placed stale file) can never hit: a miss recomputes instead of
    silently reusing a foreign number. The decision-rule tag matters as much
    as the artifact bytes: the fillability veto (0efc741) changed
    ``SplitToolMatcher.best_score`` semantics while changing zero artifact
    bytes, and the on-disk cache silently mixed pre-veto and post-veto
    scores until the tag was added (found by adversarial review 2026-10-07:
    52/121 sampled keys stale). Bump the tag whenever the shipped decision
    rule changes.
    """
    h = hashlib.sha256()
    h.update(b"decision:v3-split-fillability-veto|")
    for name in ("tool-match-gate-int8.onnx", "tool-match-int8.onnx"):
        path = ROOT / "models" / name
        if path.is_file():
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        else:
            h.update(f"missing:{name}".encode())
    return h.hexdigest()[:16]


def _cache_key(row: dict, guard_fp: str) -> str:
    tool = row.get("tool")
    try:
        canon = json.dumps(tool, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canon = str(tool)
    tool_digest = hashlib.sha256(canon.encode()).hexdigest()[:16]
    return (
        f"v2||{guard_fp}||{row['category']}||{row['source']}"
        f"||{_digest(row['request'])}||{tool_digest}"
    )


def score_rows(rows: list[dict], matcher, cache: dict | None = None,
               guard_fp: str | None = None) -> dict:
    """Pair score per row index (single-tool catalogue text, canonical).

    The cache key carries the guard fingerprint plus the request AND the
    tool: two rows of one catalogue share category/source/request but
    score different tools, so a request-only key reuses the first tool's
    score for the whole catalogue. Fresh scores populate ``cache``.
    """
    if guard_fp is None:
        guard_fp = _guard_fingerprint()
    out: dict[str, float] = {}
    for i, row in enumerate(rows):
        key = _cache_key(row, guard_fp)
        if cache is not None and key in cache:
            out[str(i)] = cache[key]
            continue
        score = matcher.best_score(BTA._v2_catalogue_text(row["tool"], row["request"]))
        out[str(i)] = score
        if cache is not None:
            cache[key] = score
    return out


def catalogue_max(rows: list[dict], scores: dict) -> dict[str, dict]:
    cats: dict[str, dict] = {}
    for i, row in enumerate(rows):
        src = row["source"]
        s = scores[str(i)]
        c = cats.get(src)
        if c is None or s > c["score"]:
            cats[src] = {
                "score": s,
                "category": row["category"],
                "tool": row["tool"],
                "request": row["request"],
            }
    return cats


def catalogue_text(tools: list[dict], request: str) -> str:
    return (
        "Tools available:\n"
        + json.dumps(tools, ensure_ascii=False)
        + "\n\nUser request: "
        + request
    )


def reproduce_residual(rows: list[dict]) -> dict:
    """Deterministic residual via the read-only production path.

    Multi-tool catalogue text per source through
    ``pipeline._classify_known_schema`` with a forced tool_call route
    (mirrors tests/test_clarification_routing.py; no pipeline/tool_match
    file is modified). Residual = catalogues still emitting tool_call.
    """
    from collections import defaultdict
    from mivia_small_json import bootstrap, pipeline as P
    from mivia_small_json import registry

    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    af = {s: v for s, v in by_src.items() if v[0]["category"] == ASK}
    out = {
        "n_catalogues": len(af),
        "tool_clarification": 0,
        "no_tool_call": 0,
        "n_residual": 0,
        "sources": [],
    }
    with registry.temporary_schemas():
        bootstrap.register_all_schemas()
        for src in sorted(af):
            v = af[src]
            text = catalogue_text([x["tool"] for x in v], v[0]["request"])
            obj, _ = P._classify_known_schema(text, "tool_call", routed_by="track-d-repro")
            t = obj.get("type", "?")
            if t == "tool_call":
                out["n_residual"] += 1
                out["sources"].append(src)
            elif t in out:
                out[t] += 1
    return out


def train_head(train_X, train_y, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(train_X)
    clf = LogisticRegression(solver="saga", max_iter=5000, random_state=seed).fit(
        sc.transform(train_X), train_y
    )
    return sc, clf


def evaluate(rows: list[dict], matcher, head, tau: float, cache: dict | None = None,
             guard_fp: str | None = None) -> dict:
    sc, clf = head
    scores = score_rows(rows, matcher, cache, guard_fp)
    cats = catalogue_max(rows, scores)
    neg = [c["score"] for c in cats.values() if c["category"] != REL]
    pos = [c["score"] for c in cats.values() if c["category"] == REL]
    thr, _, _ = BTA._operating_point(neg, pos)
    per_cat: dict[str, dict] = {}
    mcnemar = {"b": 0, "c": 0}
    for src, c in cats.items():
        call_before = c["score"] >= thr
        feat = features(c["score"], c["tool"], c["request"])
        import numpy as np

        p_keep = float(clf.predict_proba(sc.transform(np.array([feat])))[0][1])
        call_after = bool(call_before and p_keep >= tau)
        d = per_cat.setdefault(c["category"], {"n": 0, "before": 0, "after": 0})
        d["n"] += 1
        d["before"] += call_before
        d["after"] += call_after
        if c["category"] == ASK:
            mcnemar["b"] += int(call_before and not call_after)
            mcnemar["c"] += int(call_after and not call_before)
    b, cc = mcnemar["b"], mcnemar["c"]
    p_val = 1.0
    if b + cc:
        from scipy.stats import binomtest

        p_val = float(binomtest(min(b, cc), b + cc, 0.5).pvalue)
    table = {}
    for cat, d in per_cat.items():
        table[cat] = {
            "catalogues": d["n"],
            "call_before": round(100 * d["before"] / d["n"], 1),
            "call_after": round(100 * d["after"] / d["n"], 1),
            "delta": round(100 * (d["after"] - d["before"]) / d["n"], 1),
        }
    return {
        "threshold": round(thr, 4),
        "tau": tau,
        "per_category": table,
        "mcnemar": {"b": b, "c": cc, "p": p_val},
    }


def pick_tau(rows, matcher, head, cache=None, guard_fp: str | None = None) -> float:
    """Largest ask_followup rejection with relevance retention >= 95%.

    Operating-point selection on v2 (same status as the canonical thr);
    the head weights never see v2.
    """
    import numpy as np

    sc, clf = head
    scores = score_rows(rows, matcher, cache, guard_fp)
    cats = catalogue_max(rows, scores)
    neg = [c["score"] for c in cats.values() if c["category"] != REL]
    pos = [c["score"] for c in cats.values() if c["category"] == REL]
    thr, _, _ = BTA._operating_point(neg, pos)
    rel = [c for c in cats.values() if c["category"] == REL and c["score"] >= thr]
    ask = [c for c in cats.values() if c["category"] == ASK and c["score"] >= thr]
    all_rel = [c for c in cats.values() if c["category"] == REL]
    rp = np.array(
        [
            float(
                clf.predict_proba(
                    sc.transform(np.array([features(c["score"], c["tool"], c["request"])]))
                )[0][1]
            )
            for c in rel
        ]
    )
    ap = np.array(
        [
            float(
                clf.predict_proba(
                    sc.transform(np.array([features(c["score"], c["tool"], c["request"])]))
                )[0][1]
            )
            for c in ask
        ]
    )
    # Matched-retention constraint: overall relevance catalogue retention
    # (kept / all relevance catalogues) must stay >= 95%, not just the
    # would-call slice.
    best, best_rej = 1.0, -1.0
    for tau in sorted(set(rp.tolist() + ap.tolist() + [1.0])):
        kept = (len(rel) - int((rp < tau).sum())) / len(all_rel) if all_rel else 1.0
        rej = float((ap < tau).mean()) if len(ap) else 0.0
        if kept >= 0.95 and rej > best_rej:
            best, best_rej = tau, rej
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    matcher = load_matcher()
    if matcher is None:
        print("shipped guard failed to load", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in V2.read_text().splitlines()]
    if args.limit:
        rows = [
            r
            for cat in BTA.V2_CATEGORIES
            for r in [x for x in rows if x["category"] == cat][: args.limit]
        ]
    guard_fp = _guard_fingerprint()
    cache = {}
    if SCORE_CACHE.is_file():
        cache = json.loads(SCORE_CACHE.read_text())
        # Pre-key-format entries can never hit a v2 key; drop them now so
        # the file self-heals instead of accumulating dead weight.
        cache = {k: v for k, v in cache.items()
                 if isinstance(k, str) and k.startswith(f"v2||{guard_fp}||")}

    resid = reproduce_residual(rows)
    print(
        f"residual ask_followup catalogues (pipeline path): "
        f"{resid['n_residual']}/{resid['n_catalogues']} "
        f"(clarification {resid['tool_clarification']}, "
        f"no_tool {resid['no_tool_call']})"
    )

    pairs = [json.loads(line) for line in PAIRS.read_text().splitlines()]
    eval_digests = {_digest(r["request"]) for r in rows}
    dropped = [p for p in pairs if _digest(p["request"]) in eval_digests]
    pairs = [p for p in pairs if _digest(p["request"]) not in eval_digests]
    print(f"pairs rows kept: {len(pairs)}; eval-overlapping dropped: {len(dropped)}")

    pscores = score_rows(
        [
            {"category": "train", "source": p["source"], "request": p["request"], "tool": p["tool"]}
            for p in pairs
        ],
        matcher,
        None,
    )
    import numpy as np

    X = np.array([features(pscores[str(i)], p["tool"], p["request"]) for i, p in enumerate(pairs)])
    y = np.array([1 if p["label"] == 1 else 0 for p in pairs])

    results = {}
    for seed in SEEDS:
        head = train_head(X, y, seed)
        tau = pick_tau(rows, matcher, head, cache, guard_fp)
        ev = evaluate(rows, matcher, head, tau, guard_fp=guard_fp, cache=cache)
        a = ev["per_category"][ASK]
        r = ev["per_category"][REL]
        drop = a["call_before"] - a["call_after"]
        reg = max(v["delta"] for k, v in ev["per_category"].items() if k != ASK)
        ok = drop >= 10.0 and r["delta"] == 0.0 and reg <= 1.0 and ev["mcnemar"]["p"] < 0.05
        print(
            f"seed {seed}: tau={tau:.4f} ask {a['call_before']}->"
            f"{a['call_after']} (drop {drop:.1f}pp) rel delta "
            f"{r['delta']:+.1f}pp worst-other {reg:+.1f}pp "
            f"McNemar p={ev['mcnemar']['p']:.4g} -> "
            f"{'PASS' if ok else 'FAIL'}"
        )
        results[seed] = ev
        if args.write:
            sc, clf = head
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"reject-head-s{seed}.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "mean": sc.mean_.tolist(),
                        "scale": sc.scale_.tolist(),
                        "coef": clf.coef_[0].tolist(),
                        "intercept": float(clf.intercept_[0]),
                        "tau": tau,
                        "eval": "guard-eval-v2",
                        "threshold": ev["threshold"],
                    },
                    indent=1,
                )
            )
    if args.write:
        (OUT_DIR / "reject-head-metrics.json").write_text(
            json.dumps({str(k): v for k, v in results.items()}, indent=1)
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        SCORE_CACHE.write_text(json.dumps(cache, indent=1))
        print("candidates written under", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
