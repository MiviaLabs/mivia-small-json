#!/usr/bin/env python3
"""Weakness probe for the zero-shot-select r3 corpus (protocol 6b).

Read-only, explicit paths, NO threshold is set from it. Measures:

1. near-oos texts per domain through the SHIPPED rank bytes over the full
   registered pools — the confusion texture (which intents win on
   in-topic/out-of-intent questions; they are expected to win — that is
   what oos IS; the probe discloses the texture and confirms no authored
   text is a trivial in-intent paraphrase);
2. the r2 nomatch head (demoted bytes, explicit path) null-score
   distribution on r3 near-oos DEV texts vs r1 listwise DEV in-domain
   texts — the separability the new supply must create (the r2 head never
   saw near-domain oos; if it already separates, the fix is calibration);
3. the shipped rank head's bare yes/no accuracy on r2 verdict samples
   (regression context only; the referee itself is only read at the
   mid-round gate).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

R1 = ROOT / "data/real/zero-shot-select-r1"
R3 = ROOT / "data/real/zero-shot-select-r3"
R2CAND = ROOT / "data/incoming/models/zsr2-candidates/select-nomatch-zsr2-int8.onnx"
R2CAND_TOK = ROOT / "data/incoming/models/zsr2-candidates/select-nomatch-zsr2-tokenizer.json"
SHIP = ROOT / "models/tool-match-int8.onnx"
SHIP_TOK = ROOT / "models/tool-match-tokenizer.json"
B77 = ROOT / "data/incoming/external/banking77"
CLINC = ROOT / "data/incoming/external/clinc150"
R2VERDICT = ROOT / "data/real/zero-shot-select-r2/verdict-pairs.jsonl"
OUT = ROOT / ".scratch/zs-select-r3/weakness-probe.json"
N_PER_DOMAIN = 300
N_SEP = 500


def render(intent: str) -> str:
    return f"{intent}: {intent.replace('_', ' ')} ({intent.replace('_', ' ')})"


def main() -> int:
    from mivia_small_json.tool_match import load_single_matcher

    ship = load_single_matcher(SHIP, tokenizer_path=SHIP_TOK)
    if ship is None:
        print("shipped rank head failed to load", file=sys.stderr)
        return 2
    nom2 = load_single_matcher(R2CAND, tokenizer_path=R2CAND_TOK)
    if nom2 is None:
        print("r2 nomatch candidate failed to load (explicit path)",
              file=sys.stderr)
        return 2

    rng = random.Random(11)
    intents = {
        "banking": json.loads((B77 / "categories.json").read_text()),
        "clinc": [ln for ln in json.loads(
            (CLINC / "labels.json").read_text()) if ln != "oos"],
    }
    out: dict = {
        "shipped_rank_sha256": hashlib.sha256(SHIP.read_bytes()).hexdigest(),
        "r2_nomatch_sha256": hashlib.sha256(R2CAND.read_bytes()).hexdigest(),
        "texture": {}, "separability": {}, "shipped_referee_context": {},
    }

    # -- 1. texture: shipped argmax on near-oos texts -------------------------
    near = [json.loads(ln) for ln in
            (R3 / "near-oos-pairs.jsonl").read_text().splitlines()
            if ln.strip()]
    text_domain: dict[str, str] = {}
    for r in near:
        text_domain[r["request"]] = r["source"].rsplit("_", 1)[0] \
            .replace("neardo_", "")
    for domain in ("banking", "clinc"):
        texts = [t for t, d in text_domain.items() if d == domain]
        sample = rng.sample(texts, min(N_PER_DOMAIN, len(texts)))
        pool = {f"c{i}": render(it) for i, it in enumerate(intents[domain])}
        winners: Counter[str] = Counter()
        for t in sample:
            best, conf = ship.select(t, pool)
            winners[pool[best].split(":")[0]] += 1
        out["texture"][domain] = {
            "n": len(sample),
            "distinct_winners": len(winners),
            "top_winners": dict(winners.most_common(8)),
        }

    # -- 2. separability: r2 nomatch null score on near-oos dev vs in-domain --
    dev_near = [t for t, d in text_domain.items()
                if next(r for r in near if r["request"] == t)["split"] == "dev"]
    dev_in: list[str] = []
    for line in (R1 / "listwise-pairs.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["source"] in ("banking77_pos", "clinc150_pos") \
                    and r["split"] == "dev":
                dev_in.append(r["request"])
    rng2 = random.Random(13)
    dev_near_s = rng2.sample(dev_near, min(N_SEP, len(dev_near)))
    dev_in_s = rng2.sample(dev_in, min(N_SEP, len(dev_in)))
    NULL_CAND = "__null__: none of these descriptions match"
    near_sc = sorted(nom2._score_raw(t, NULL_CAND) for t in dev_near_s)
    in_sc = sorted(nom2._score_raw(t, NULL_CAND) for t in dev_in_s)

    def q(xs, p):
        return round(xs[min(len(xs) - 1, int(p * (len(xs) - 1)))], 4)

    out["separability"] = {
        "null_candidate": NULL_CAND,
        "near_oos_dev": {"n": len(near_sc), "p50": q(near_sc, .5),
                         "p90": q(near_sc, .9), "p99": q(near_sc, .99),
                         "max": round(near_sc[-1], 4)},
        "indomain_dev": {"n": len(in_sc), "p50": q(in_sc, .5),
                         "p95": q(in_sc, .95), "p99": q(in_sc, .99),
                         "max": round(in_sc[-1], 4)},
        "note": ("r2 nomatch bytes never trained on near-domain oos; the r3 "
                 "supply must create the separation these quantiles lack"),
    }

    # -- 3. shipped bare yes/no on r2 verdict samples (context only) ----------
    verd = [json.loads(ln) for ln in R2VERDICT.read_text().splitlines()
            if ln.strip()]
    sample = rng.sample(verd, min(300, len(verd)))
    ok = 0
    for r in sample:
        picked, _ = ship.select(r["request"], ["yes", "no"])
        ok += (picked == "yes") == (r["label"] == 1)
    out["shipped_referee_context"] = {
        "n": len(sample), "shipped_bare_acc_pct": round(100 * ok / len(sample), 1),
        "note": "training-supply probe; the referee itself is only read at "
                "the mid-round gate"}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
