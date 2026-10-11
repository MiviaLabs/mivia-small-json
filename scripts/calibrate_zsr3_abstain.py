#!/usr/bin/env python3
"""zero-shot-select r3: fit the abstention calibration on the NEAR-DOMAIN
supply and decide the REGISTERED SHIP PATH (docs/specs/
zero-shot-select-r3/spec.md, switch registered BEFORE the GPU).

Path N (primary): the r2 registered decision path — dedicated no-match head
null score p = head(text, "{key}: {null-phrasing}"); p >= tau -> ABSTAIN;
else rank argmax over real candidates. Registered tau rule: ONE pooled tau
on train-split supply — smallest grid tau with pooled in-domain-fire <= 5%
AND pooled near-oos-fire >= 30%. NO feasible tau is the switch condition
(never a silent percentile fallback).

Path M (registered fallback): the decision-layer margin abstain — with
s(1) >= s(2) the two highest raw scores of the RANK head over the REAL
candidates, ABSTAIN iff s(1) - s(2) < tau_margin[k]. Per band
k in {4, 8, 16, 32, 77, 151}: tau_margin = smallest grid tau with
in-domain-low-margin <= 5% AND near-oos-low-margin >= 40%; no feasible tau
-> the in-domain 5th percentile (precision priority, flagged).

Fit supply (train-split ONLY; eval windows and data/heldout never read):
- in-domain rows: r1 listwise DEV-split positive rows (text, gold intent,
  dataset) — the untrained 10%;
- oos rows: r3 near-oos DEV-split texts (the untrained ~10% of the
  synthesized near-domain supply — the eval's register). The far
  telco/insurance supply's fire rate is RECORDED as disclosure, never fit.

Writes data/real/zero-shot-select-r3/abstain-calibration.json (+ sha256
sidecar) binding the chosen artifacts' sha256 and the burned alternative's
numbers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RANK = ROOT / "models/select-rank-zsr3-int8.onnx"
RANK_TOK = ROOT / "models/select-rank-zsr3-tokenizer.json"
HEAD = ROOT / "models/select-nomatch-zsr3-int8.onnx"
HEAD_TOK = ROOT / "models/select-nomatch-zsr3-tokenizer.json"
R1 = ROOT / "data/real/zero-shot-select-r1"
R3 = ROOT / "data/real/zero-shot-select-r3"
BITEXT = [ROOT / "data/incoming/external/bitext/telco.csv",
          ROOT / "data/incoming/external/bitext/insurance.csv"]
OUT = R3 / "abstain-calibration.json"

NULL_CAND = "__null__: none of these descriptions match"
BANDS = [4, 8, 16, 32, 77, 151]
ID_FIRE_MAX = 0.05
OOS_FIRE_MIN = 0.30           # Path N
M_OOS_LOW_MIN = 0.40          # Path M
N_PER_BAND = 400

_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).casefold()
    return " ".join(_PUNCT.split(t)).strip()


def q(xs: list[float], p: float) -> float:
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]


def main() -> int:
    from mivia_small_json.tool_match import load_single_matcher

    rank = load_single_matcher(RANK, tokenizer_path=RANK_TOK)
    head = load_single_matcher(HEAD, tokenizer_path=HEAD_TOK)
    if rank is None or head is None:
        print("artifact load failed", file=sys.stderr)
        return 2
    rank_sha = hashlib.sha256(RANK.read_bytes()).hexdigest()
    head_sha = hashlib.sha256(HEAD.read_bytes()).hexdigest()

    # ---- supply ----------------------------------------------------------
    id_rows: dict[str, tuple[str, str]] = {}   # norm text -> (dataset, tool)
    for line in (R1 / "listwise-pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] == "dev" and r["label"] == 1 and \
                r["source"] in ("banking77_pos", "clinc150_pos"):
            id_rows[norm(r["request"])] = (
                "banking" if r["source"] == "banking77_pos" else "clinc",
                r["tool"])
    near_dev: dict[str, str] = {}
    for line in (R3 / "near-oos-pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] == "dev":
            near_dev[norm(r["request"])] = \
                r["source"].rsplit("_", 1)[0].replace("neardo_", "")
    id_texts = list(id_rows)
    near_texts = list(near_dev)
    print(f"supply: {len(id_texts)} in-domain dev texts, "
          f"{len(near_texts)} near-oos dev texts")

    # ---- Path N: pooled tau on the dedicated head -------------------------
    id_sc = sorted(head._score_raw(t, NULL_CAND) for t in id_texts)
    near_sc = sorted(head._score_raw(t, NULL_CAND) for t in near_texts)
    grid = sorted({round(s, 6) for s in id_sc} | {round(s, 6)
                                                  for s in near_sc},
                  reverse=True)

    def fire(scores: list[float], tau: float) -> float:
        lo, hi = 0, len(scores)
        while lo < hi:
            mid = (lo + hi) // 2
            if scores[mid] >= tau:
                lo = mid + 1
            else:
                hi = mid
        return 100 * lo / len(scores)

    feasible = None
    for tau in grid:
        if fire(id_sc, tau) <= 100 * ID_FIRE_MAX and \
                fire(near_sc, tau) >= 100 * OOS_FIRE_MIN:
            feasible = tau
    path_n = {
        "feasible": feasible is not None,
        "tau": round(feasible, 6) if feasible is not None else None,
        "in_domain_fire_pct": round(fire(id_sc, feasible), 2)
        if feasible is not None else round(fire(id_sc, grid[0]), 2),
        "near_oos_fire_pct": round(fire(near_sc, feasible), 2)
        if feasible is not None else round(fire(near_sc, grid[0]), 2),
        "rule": ("smallest grid tau with pooled in-domain fire <= 5% and "
                 "pooled near-oos fire >= 30%"),
        "in_domain_null_score": {"p50": round(q(id_sc, .5), 4),
                                 "p95": round(q(id_sc, .95), 4),
                                 "p99": round(q(id_sc, .99), 4),
                                 "max": round(id_sc[-1], 4), "n": len(id_sc)},
        "near_oos_null_score": {"p50": round(q(near_sc, .5), 4),
                                "p90": round(q(near_sc, .9), 4),
                                "p99": round(q(near_sc, .99), 4),
                                "max": round(near_sc[-1], 4), "n": len(near_sc)},
    }
    # far-supply disclosure (the anti-correlated r1/r2 supply, never fit)
    ph = re.compile(r"\{\{[^}]*\}\}")
    far: list[str] = []
    consumed = {norm(json.loads(ln)["request"])
                for f in ("null-pairs.jsonl", "listwise-pairs.jsonl",
                          "synthetic-labelsets.jsonl", "bfcl-train.jsonl")
                for ln in (R1 / f).read_text().splitlines() if ln.strip()}
    for fname in BITEXT:
        with fname.open(encoding="utf-8", errors="replace", newline="") as fh:
            for r in csv.DictReader(fh):
                t = (r.get("instruction") or "").strip()
                t = re.sub(r"\s+", " ", ph.sub("the reference", t))
                if 15 <= len(t) <= 140 and norm(t) not in consumed:
                    far.append(t)
    rng_far = random.Random(20261011)
    rng_far.shuffle(far)
    far_sc = [head._score_raw(t, NULL_CAND)
              for t in far[:N_PER_BAND]]
    far_fire_at_tau = round(fire(sorted(far_sc), feasible), 2) \
        if feasible is not None else None
    print(f"Path N: feasible={path_n['feasible']} "
          f"tau={path_n['tau']} id_fire={path_n['in_domain_fire_pct']}% "
          f"near_fire={path_n['near_oos_fire_pct']}%; far-supply fire "
          f"{far_fire_at_tau}% (disclosure)")

    # ---- Path M: per-band margin taus on the rank head --------------------
    pools = {
        "banking": [json.loads(
            (ROOT / "data/incoming/external/banking77/categories.json")
            .read_text())],
        "clinc": [[ln for ln in json.loads(
            (ROOT / "data/incoming/external/clinc150/labels.json")
            .read_text()) if ln != "oos"]],
    }

    def render(i: str) -> str:
        return f"{i}: {i.replace('_', ' ')} ({i.replace('_', ' ')})"

    def margin(text: str, cands: list[str]) -> float:
        sc = sorted((rank._score_raw(text, c) for c in cands), reverse=True)
        return sc[0] - sc[1] if len(sc) > 1 else sc[0]

    rng = random.Random(42)
    tau_by_k = []
    for k in BANDS:
        id_m: list[float] = []
        near_m: list[float] = []
        for tn in id_texts:
            ds, tool = id_rows[tn]
            pool = [render(i) for i in pools[ds][0]]
            full = k >= len(pool)
            if full:
                cands = pool
            else:
                sibs = [c for c in pool if c != tool]
                cands = [tool] + rng.sample(sibs, k - 1)
            m = margin(tn, cands)
            id_m.append(m)
        for tn in near_texts:
            ds = near_dev[tn]
            pool = [render(i) for i in pools[ds][0]]
            cands = pool if k >= len(pool) else rng.sample(pool, k)
            near_m.append(margin(tn, cands))
        id_m_s, near_m_s = sorted(id_m), sorted(near_m)
        mgrid = sorted({round(v, 6) for v in id_m} |
                       {round(v, 6) for v in near_m})

        def low_share(scores: list[float], tau: float) -> float:
            lo = sum(1 for v in scores if v < tau)
            return 100 * lo / len(scores)

        m_feasible = None
        for tau in mgrid:
            if low_share(id_m_s, tau) <= 100 * ID_FIRE_MAX and \
                    low_share(near_m_s, tau) >= 100 * M_OOS_LOW_MIN:
                m_feasible = tau
        flagged = m_feasible is None
        tau_m = m_feasible if not flagged else q(id_m_s, 0.05)
        tau_by_k.append({
            "k": k, "tau_margin": round(tau_m, 6),
            "in_domain_low_pct": round(low_share(id_m_s, tau_m), 2),
            "near_oos_low_pct": round(low_share(near_m_s, tau_m), 2),
            "n_in_domain": len(id_m), "n_near_oos": len(near_m),
            "feasible": not flagged,
            "rule": ("smallest grid tau with in-domain low-margin <= 5% and "
                     "near-oos low-margin >= 40%") if not flagged else
            ("no feasible tau: in-domain 5th percentile margin "
             "(precision priority, flagged)"),
        })
        print(f"k={k}: tau_margin={tau_by_k[-1]['tau_margin']} "
              f"id_low={tau_by_k[-1]['in_domain_low_pct']}% "
              f"near_low={tau_by_k[-1]['near_oos_low_pct']}%"
              f"{' (flagged)' if flagged else ''}")

    ship_path = "N" if path_n["feasible"] else "M"
    meta = {
        "schema": "select-abstain-calibration/v2",
        "arm": "zero-shot-select-r3",
        "ship_path": ship_path,
        "head_artifact": "models/select-nomatch-zsr3-int8.onnx",
        "head_sha256": head_sha,
        "rank_artifact": "models/select-rank-zsr3-int8.onnx",
        "rank_sha256": rank_sha,
        "null_candidate": NULL_CAND,
        "registered_null_keys": ["__null__", "none", "no_match"],
        "decision_path_N": (
            "candidate set carries a registered null key -> p = head(text, "
            "rendered null candidate); p >= pooled tau -> ABSTAIN; else rank "
            "argmax over the real candidates (null key excluded). Sets "
            "without a null key never query the head."),
        "decision_path_M": (
            "rank head scores the real candidates; m = s(1) - s(2); "
            "m < tau_margin[k] -> ABSTAIN (k = real candidate count; band = "
            "nearest fitted k); else the rank argmax stands."),
        "path_N": path_n,
        "path_M": {"bands": {"shape_key": "candidate_count",
                             "tau_by_k": tau_by_k},
                   "note": ("Path M margins fitted regardless of the switch; "
                            "the burned alternative's numbers live here")},
        "supply": {
            "in_domain": "r1 listwise DEV-split positive rows",
            "oos": "r3 near-oos DEV-split texts (the eval's register)",
            "far_supply_disclosure": {
                "note": ("telco/insurance unconsumed bitext: the r1/r2 "
                         "supply the r2 RESULT proved anti-correlated with "
                         "the eval's near-domain oos; recorded, never fit"),
                "n": len(far_sc),
                "fire_at_shipped_tau_pct": far_fire_at_tau,
                "null_score_p50": round(q(sorted(far_sc), .5), 4),
            },
            "eval_windows_read": False,
        },
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = json.dumps(meta, indent=1)
    OUT.write_text(payload, encoding="utf-8")
    (OUT.parent / (OUT.name + ".sha256")).write_text(
        hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n",
        encoding="utf-8")
    print(f"SHIP PATH: {ship_path} -> wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
