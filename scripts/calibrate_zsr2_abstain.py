#!/usr/bin/env python3
"""zero-shot-select r2: fit the band-aware abstention taus for the
DEDICATED no-match head (zero-shot-select-r2).

Writes data/real/zero-shot-select-r2/nomatch-calibration.json (+ sha256
sidecar): per candidate-count band k a tau on the head's null score
s = p_match(text, "__null__: none of these descriptions match"). The band
structure and nearest-K assignment mirror models/select-calibration-meta.json.

Supply (train-split, NEVER the eval windows, never data/heldout):
- in-domain rows: r1 listwise DEV-split texts (the untrained 10%) over
  banking77 + clinc150;
- oos rows: bitext telco/insurance texts NOT consumed by the r1 corpus
  (the r1 null-pairs' 2,500 texts are excluded by normalized text; clinc
  train oos leftovers are EMPTY — all 100 rows entered the r1 corpus —
  disclosed here).

Registered deterministic rule (declared in the spec BEFORE any fit):
tau_band = smallest grid tau with in-domain-fire <= 5% AND oos-fire >=
30%; no feasible tau -> tau = in-domain 95th percentile (precision
priority, flagged in the JSON). The head's null score does not read the
other candidates, so per-band taus CAN come out equal — the calibration
measures each band on its own rows and records what it measures.
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

HEAD = ROOT / "models/select-nomatch-zsr2-int8.onnx"
TOK = ROOT / "models/select-nomatch-zsr2-tokenizer.json"
R1 = ROOT / "data/real/zero-shot-select-r1"
R2 = ROOT / "data/real/zero-shot-select-r2"
B77 = ROOT / "data/incoming/external/banking77"
BITEXT = [ROOT / "data/incoming/external/bitext/telco.csv",
          ROOT / "data/incoming/external/bitext/insurance.csv"]
OUT = R2 / "nomatch-calibration.json"

NULL_CAND = "__null__: none of these descriptions match"
BANDS = [4, 8, 16, 32, 77, 151]
N_PER_BAND = 600
ID_FIRE_MAX = 0.05
OOS_FIRE_MIN = 0.30

_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).casefold()
    return " ".join(_PUNCT.split(t)).strip()


def main() -> int:
    from mivia_small_json.tool_match import load_single_matcher

    m = load_single_matcher(HEAD, tokenizer_path=TOK)
    if m is None:
        print("nomatch head failed to load", file=sys.stderr)
        return 2
    head_sha = hashlib.sha256(HEAD.read_bytes()).hexdigest()

    # -- supply --------------------------------------------------------------
    consumed: set[str] = set()
    dev_texts: list[str] = []
    for line in (R1 / "listwise-pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] == "dev":
            dev_texts.append(r["request"])
        consumed.add(norm(r["request"]))
    for name in ("null-pairs.jsonl", "synthetic-labelsets.jsonl",
                 "bfcl-train.jsonl"):
        for line in (R1 / name).read_text().splitlines():
            if line.strip():
                consumed.add(norm(json.loads(line)["request"]))

    ph = re.compile(r"\{\{[^}]*\}\}")
    oos_pool: list[str] = []
    seen_oos: set[str] = set()
    for fname in BITEXT:
        with fname.open(encoding="utf-8", errors="replace", newline="") as fh:
            for r in csv.DictReader(fh):
                t = (r.get("instruction") or "").strip()
                t = ph.sub("the reference", t)
                t = re.sub(r"\s+", " ", t)
                n = norm(t)
                if t and 15 <= len(t) <= 140 and n not in consumed \
                        and n not in seen_oos:
                    seen_oos.add(n)
                    oos_pool.append(t)

    rng = random.Random(20261011)
    rng.shuffle(dev_texts)
    rng.shuffle(oos_pool)
    if len(dev_texts) < N_PER_BAND * len(BANDS) or \
            len(oos_pool) < N_PER_BAND * len(BANDS):
        print(f"SUPPLY FAIL: dev {len(dev_texts)} / oos {len(oos_pool)}, "
              f"need {N_PER_BAND * len(BANDS)} each", file=sys.stderr)
        return 2

    def fire_stats(rows: list[str], tau: float) -> float:
        fired = sum(1 for t in rows if m._score_raw(t, NULL_CAND) >= tau)
        return fired / len(rows)

    tau_by_k = []
    for i, k in enumerate(BANDS):
        id_rows = dev_texts[i * N_PER_BAND:(i + 1) * N_PER_BAND]
        oos_rows = oos_pool[i * N_PER_BAND:(i + 1) * N_PER_BAND]
        id_scores = sorted(m._score_raw(t, NULL_CAND) for t in id_rows)
        grid = sorted({round(s, 6) for s in id_scores} |
                      {round(m._score_raw(t, NULL_CAND), 6)
                       for t in oos_rows}, reverse=True)
        feasible = None
        for tau in grid:
            if fire_stats(id_rows, tau) <= ID_FIRE_MAX and \
                    fire_stats(oos_rows, tau) >= OOS_FIRE_MIN:
                feasible = tau
                break
        flagged = feasible is None
        if flagged:
            tau = id_scores[int(0.95 * (len(id_scores) - 1))]
        else:
            tau = feasible
        tau_by_k.append({
            "k": k, "tau": round(tau, 6),
            "in_domain_fire_pct": round(100 * fire_stats(id_rows, tau), 2),
            "oos_fire_pct": round(100 * fire_stats(oos_rows, tau), 2),
            "n_in_domain": len(id_rows), "n_oos": len(oos_rows),
            "rule": "registered: smallest tau with in-domain fire <= 5% "
                    "and oos fire >= 30%" if not flagged else
                    "no feasible tau: in-domain 95th percentile "
                    "(precision priority)",
            "flagged": flagged,
        })
        print(f"k={k}: tau={tau_by_k[-1]['tau']} "
              f"id_fire={tau_by_k[-1]['in_domain_fire_pct']}% "
              f"oos_fire={tau_by_k[-1]['oos_fire_pct']}%"
              f"{' (flagged)' if flagged else ''}")

    meta = {
        "schema": "select-nomatch-calibration/v1",
        "arm": "zero-shot-select-r2",
        "head_artifact": "models/select-nomatch-zsr2-int8.onnx",
        "head_sha256": head_sha,
        "null_candidate": NULL_CAND,
        "registered_null_keys": ["__null__", "none", "no_match"],
        "decision_path": ("candidate set carries a registered null key -> "
                          "p = head(text, rendered null candidate); "
                          "p >= tau_band(k) -> ABSTAIN; else rank argmax "
                          "over the real candidates (null key excluded). "
                          "Sets without a null key never query this head."),
        "bands": {
            "shape_key": "candidate_count",
            "small": {"k_max": 4, "tau": tau_by_k[0]["tau"],
                      "measured": tau_by_k[0]},
            "many": {
                "k_min": 5,
                "assignment": ("serve the fit point with the nearest K by "
                               "|log(k / k_fit)|; ties to the larger K"),
                "tau_by_k": tau_by_k[1:],
            },
        },
        "supply": {
            "in_domain": "r1 listwise DEV-split texts (untrained 10%)",
            "oos": ("bitext telco/insurance texts not consumed by the r1 "
                    "corpus; clinc train oos leftovers are empty (all 100 "
                    "entered the r1 corpus)"),
            "oos_pool_size": len(oos_pool),
            "eval_windows_read": False,
        },
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = json.dumps(meta, indent=1)
    OUT.write_text(payload, encoding="utf-8")
    (OUT.parent / (OUT.name + ".sha256")).write_text(
        hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n",
        encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
