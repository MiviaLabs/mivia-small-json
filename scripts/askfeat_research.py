#!/usr/bin/env python3
"""TRACK 4: ask_followup residual new-feature research (CPU-only).

Context: 118/473 residual catalogues are high-confidence topical matches
no rule separates from real calls; the 7-feature linear head managed only
-3.6pp (reports/exp-reject-head.md, FAIL). This script reuses the
train_reject_head.py reproduce path read-only and hunts ONE new feature
family the rank scorer underuses.

Pre-registered bar (written before the numbers):
  PASS iff ask_followup false-call drops >= 10pp retention-flat, no >1pp
  regressions, McNemar p < 0.05.
  DIAGNOSIS-ONLY pass iff a feature family separates residual-118 from
  single_turn_relevance keeps with max(AUC, 1-AUC) >= 0.75 (or McNemar
  p < 0.05 at matched retention for a head trained on it).
  FAIL otherwise.

Families tested (all from frozen shipped scores + request/tool text;
no training, so disjointness from training is structural):
  F1 catalogue-size normalization: n_tools, score vs size
  F2 per-tool score margin: top1 - top2 gap within multi-tool catalogues
  F3 request/tool length-ratio anomalies
  F4 required-param token coverage GAPS on high scores (uncovered count,
      gap conditioned on score >= op threshold)
  F5 tool-description entailment proxies (Jaccard, desc/request coverage)

Read-only: imports train_reject_head + bench_tool_abstention + tool_match,
never modifies pipeline.py / tool_match.py / models /. CPU-only by design
(frozen-score arithmetic + one logistic fit); no GPU command issued, so no
flock lock is needed. Candidates only under gitignored
data/incoming/models/, never promoted.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.tool_match import load_matcher  # noqa: E402

V2 = ROOT / "data" / "incoming" / "external" / "bfcl" / "guard-eval-v2.jsonl"
PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
OUT_DIR = ROOT / "data" / "incoming" / "models"
CACHE = Path("/tmp/askfeat-v2-scores.json")

ASK = "ask_followup"
REL = "single_turn_relevance"
SEEDS = (21, 22)
AUC_BAR = 0.75


def _load_reject():
    spec = importlib.util.spec_from_file_location(
        "train_reject_head", ROOT / "scripts" / "train_reject_head.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REJ = _load_reject()


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "bta", ROOT / "scripts" / "bench_tool_abstention.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BTA = _load_bench()


def _tok(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _tool_text(tool) -> str:
    if isinstance(tool, str):
        return tool
    try:
        params = (tool or {}).get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        names = " ".join(sorted(props.keys()))
        descs = " ".join(
            str((v or {}).get("description", "")) for v in props.values()
        )
        return f"{tool.get('name', '')} {tool.get('description', '')} {names} {descs}"
    except AttributeError:
        return str(tool)


def catalogue_table(rows: list[dict], scores: dict) -> dict[str, dict]:
    """One row per catalogue (source): frozen scores + text features."""
    by_src: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_src[r["source"]].append(i)
    cats: dict[str, dict] = {}
    for src, idxs in by_src.items():
        s = sorted((scores[str(i)], i) for i in idxs)
        top_score, top_i = s[-1][0], s[-1][1]
        second = s[-2][0] if len(s) > 1 else top_score
        top = rows[top_i]
        req, tool = top["request"], top["tool"]
        fill, nreq = REJ.fill_proxy(tool if isinstance(tool, dict) else {}, req)
        uncovered = 0
        if isinstance(tool, dict):
            params = (tool.get("parameters", {}) or {})
            props = params.get("properties", {}) or {}
            uncovered = sum(
                1
                for nm in (params.get("required", []) or [])
                if not (_tok(str(nm)) | _tok(
                    str((props.get(nm, {}) or {}).get("description", ""))) & _tok(req))
            )
        rt, dt = _tok(req), _tok(_tool_text(tool))
        inter = len(rt & dt)
        cats[src] = {
            "category": top["category"],
            "n_tools": len(idxs),
            "score": top_score,
            "margin": top_score - second,
            "req_len": len(req),
            "tool_len": len(_tool_text(tool)),
            "len_ratio": len(req) / max(1, len(_tool_text(tool))),
            "fill": fill,
            "nreq": nreq,
            "uncovered": uncovered,
            "gap": top_score - fill,
            "jaccard": inter / max(1, len(rt | dt)),
            "desc_cov": inter / max(1, len(dt)),
            "req_cov": inter / max(1, len(rt)),
            "request": req,
            "tool": tool,
        }
    return cats


FAMILIES: dict[str, list[str]] = {
    "F1_size": ["n_tools"],
    "F2_margin": ["margin"],
    "F3_length": ["len_ratio"],
    "F4_fillgap": ["gap", "uncovered"],
    "F5_entail": ["jaccard", "desc_cov", "req_cov"],
}


def auc_for(values: list[float], labels: list[int]) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def family_sep(cats: dict[str, dict], resid: set[str], keeps: set[str]) -> dict:
    """AUC per feature: residual-118 (label 1) vs relevance keeps (label 0)."""
    ids = sorted(set(resid) | set(keeps))
    labels = [1 if i in resid else 0 for i in ids]
    out: dict[str, dict] = {}
    for fam, feats in FAMILIES.items():
        per_feat = {}
        for f in feats:
            vals = [cats[i][f] for i in ids]
            a = auc_for(vals, labels)
            per_feat[f] = {"auc": round(a, 4), "sep": round(max(a, 1 - a), 4)}
        best = max(per_feat.values(), key=lambda d: d["sep"])
        out[fam] = {"features": per_feat, "best_sep": best["sep"]}
    return out


def margin_subset_sep(cats, resid, keeps) -> dict:
    """F2 on multi-tool catalogues only (margin undefined for singletons)."""
    multi = {i for i in set(resid) | set(keeps) if cats[i]["n_tools"] > 1}
    r = {i for i in resid if i in multi}
    k = {i for i in keeps if i in multi}
    ids = sorted(multi)
    labels = [1 if i in r else 0 for i in ids]
    a = auc_for([cats[i]["margin"] for i in ids], labels)
    return {"n_resid": len(r), "n_keep": len(k),
            "auc": round(a, 4), "sep": round(max(a, 1 - a), 4)}


def train_single_feature_head(cats, resid, keeps, feat: str, seed: int):
    """CPU logistic on ONE feature; trained on pairs.jsonl catalogue groups.

    Train catalogues are grouped from pairs.jsonl by source with the same
    feature pipeline; v2-overlapping requests dropped by hash, so weights
    never see the eval set. Only tau is picked on v2 (same status as the
    canonical operating point).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pairs = [json.loads(line) for line in PAIRS.read_text().splitlines()]
    eval_dig = {REJ._digest(r["request"]) for r in
                [json.loads(ln) for ln in V2.read_text().splitlines()]}
    n_drop = sum(1 for p in pairs if REJ._digest(p["request"]) in eval_dig)
    print(f"head train: {len(pairs) - n_drop} pairs kept, "
          f"{n_drop} v2-overlapping dropped (zero eval overlap)")
    matcher = load_matcher()
    train_rows = [
        {"category": "train", "source": p["source"], "request": p["request"],
         "tool": p["tool"], "label": p["label"]}
        for p in pairs if REJ._digest(p["request"]) not in eval_dig
    ]
    pscores = REJ.score_rows(train_rows, matcher, None)
    tgroups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(train_rows):
        tgroups[r["source"]].append(i)
    X, y = [], []
    for src, idxs in tgroups.items():
        best = max(idxs, key=lambda i: pscores[str(i)])
        r = train_rows[best]
        prow = {"category": "train", "source": src, "request": r["request"],
                "tool": r["tool"]}
        tmp = catalogue_table([prow], {"0": pscores[str(best)]})
        X.append([tmp[src][feat]])
        y.append(1 if r["label"] == 1 else 0)
    X = np.array(X)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(random_state=seed).fit(sc.transform(X), np.array(y))
    return sc, clf


def main() -> int:
    matcher = load_matcher()
    if matcher is None:
        print("shipped guard failed to load", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in V2.read_text().splitlines()]
    cache = json.loads(CACHE.read_text()) if CACHE.is_file() else None
    scores = REJ.score_rows(rows, matcher, cache)
    if not CACHE.is_file():
        CACHE.write_text(json.dumps(scores))

    resid = REJ.reproduce_residual(rows)
    print(f"residual: {resid['n_residual']}/{resid['n_catalogues']} "
          f"(clarification {resid['tool_clarification']}, "
          f"no_tool {resid['no_tool_call']})")
    cats = catalogue_table(rows, scores)
    neg = [c["score"] for c in cats.values() if c["category"] != REL]
    pos = [c["score"] for c in cats.values() if c["category"] == REL]
    thr, _, _ = BTA._operating_point(neg, pos)
    resid_set = set(resid["sources"])
    keeps = {s for s, c in cats.items()
             if c["category"] == REL and c["score"] >= thr}
    print(f"op thr {thr:.4f}; residual catalogues {len(resid_set)}; "
          f"relevance keeps {len(keeps)}")

    sep = family_sep(cats, resid_set, keeps)
    for fam, d in sep.items():
        print(fam, d)
    print("F2_margin_multitool_only", margin_subset_sep(cats, resid_set, keeps))

    best_fam = max(sep.items(), key=lambda kv: kv[1]["best_sep"])
    print(f"best family: {best_fam[0]} sep={best_fam[1]['best_sep']}")
    out = {
        "residual": resid,
        "threshold": thr,
        "n_residual": len(resid_set),
        "n_keeps": len(keeps),
        "families": sep,
        "margin_subset": margin_subset_sep(cats, resid_set, keeps),
        "auc_bar": AUC_BAR,
    }
    if best_fam[1]["best_sep"] >= AUC_BAR:
        feat = max(best_fam[1]["features"].items(),
                   key=lambda kv: kv[1]["sep"])[0]
        for seed in SEEDS:
            sc, clf = train_single_feature_head(cats, resid_set, keeps, feat, seed)
            prob = {s: float(clf.predict_proba(
                sc.transform([[cats[s][feat]]]))[0][1]) for s in cats}
            # Canonical constraint: OVERALL relevance retention
            # (kept calls / all relevance catalogues) >= 95%.
            all_rel = [s for s, c in cats.items() if c["category"] == REL]
            grid = sorted(set(prob.values()) | {1.0})
            best, brej = 1.0, -1.0
            for tau in grid:
                kept = sum(1 for s in all_rel
                           if cats[s]["score"] >= thr and prob[s] >= tau)
                rej = sum(1 for s in resid_set
                          if cats[s]["score"] >= thr and prob[s] < tau)
                if kept / len(all_rel) >= 0.95 and rej > brej:
                    best, brej = tau, rej
            per_cat: dict[str, dict] = {}
            b = c = 0
            for s, ct in cats.items():
                cb = ct["score"] >= thr
                ca = bool(cb and prob[s] >= best)
                d = per_cat.setdefault(ct["category"], {"n": 0, "b": 0, "a": 0})
                d["n"] += 1
                d["b"] += cb
                d["a"] += ca
                if ct["category"] == ASK:
                    b += int(cb and not ca)
                    c += int(ca and not cb)
            from scipy.stats import binomtest
            p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
            table = {k: {"call_before": round(100 * v["b"] / v["n"], 1),
                         "call_after": round(100 * v["a"] / v["n"], 1),
                         "delta": round(100 * (v["a"] - v["b"]) / v["n"], 1)}
                     for k, v in per_cat.items()}
            ask, rel = table[ASK], table[REL]
            drop = ask["call_before"] - ask["call_after"]
            worst = max(v["delta"] for k, v in table.items() if k != ASK)
            ok = drop >= 10.0 and rel["delta"] == 0.0 and worst <= 1.0 and p < 0.05
            print(f"seed {seed}: feat={feat} tau={best:.4f} "
                  f"ask {ask['call_before']}->{ask['call_after']} "
                  f"(drop {drop:.1f}pp) rel {rel['delta']:+.1f}pp "
                  f"worst-other {worst:+.1f}pp McNemar b={b},c={c} "
                  f"p={p:.4g} -> {'PASS' if ok else 'FAIL'}")
            out[f"seed_{seed}"] = {"tau": best, "table": table,
                                   "mcnemar": {"b": b, "c": c, "p": p},
                                   "pass": ok}
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"askfeat-head-s{seed}.json").write_text(json.dumps({
                "seed": seed, "feature": feat, "tau": best,
                "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                "coef": clf.coef_[0].tolist(),
                "intercept": float(clf.intercept_[0]),
            }, indent=1))
    (OUT_DIR / "askfeat-metrics.json").write_text(json.dumps(out, indent=1))
    print("metrics under", OUT_DIR / "askfeat-metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
