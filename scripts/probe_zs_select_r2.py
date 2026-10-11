#!/usr/bin/env python3
"""Weakness probe for the zero-shot-select r2 corpus (protocol 6b).

Routes sampled positives from EVERY training source (r1 listwise/null/
synthetic + r2 nli/verdict) through the SHIPPED rank artifact
(explicit-path loading, read-only) and splits them:

- ranking worlds (gold + 7 same-world sibling candidates):
  easy mass  — shipped head already ranks gold on top (weight, not signal)
  signal     — gold ranked below a sibling (with the winner's identity)
- binary worlds (r2 verdict + nli_verdict, bare yes/no):
  shipped binary accuracy per source prefix (probe-only numbers on
  training supply; the referee itself is only read at the mid-round gate)

Registered decision rule (spec, BEFORE the probe ran): the r1 synthetic
label-set file is DROPPED from the r2 trainer iff its easy-mass >= 90%
(mostly easy mass = neutral). Read-only; no threshold is set from it.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

R1 = ROOT / "data/real/zero-shot-select-r1"
R2 = ROOT / "data/real/zero-shot-select-r2"
RANK = ROOT / "models/tool-match-int8.onnx"
TOK = ROOT / "models/tool-match-tokenizer.json"
OUT = ROOT / ".scratch/zs-select-r2/weakness-probe.json"
N_PER_SOURCE = 300
B77 = ROOT / "data/incoming/external/banking77"
CLINC = ROOT / "data/incoming/external/clinc150"


def main() -> int:
    from mivia_small_json.tool_match import load_single_matcher

    m = load_single_matcher(RANK, tokenizer_path=TOK)
    if m is None:
        print("shipped rank head failed to load", file=sys.stderr)
        return 2
    rng = random.Random(7)
    out: dict = {"shipped_rank_sha256":
                 hashlib.sha256(RANK.read_bytes()).hexdigest(),
                 "ranking_sources": {}, "binary_sources": {},
                 "s3_polarity_rows_dropped_at_load": 0}

    # sibling worlds
    b77_intents = json.loads((B77 / "categories.json").read_text())
    clinc_labels = json.loads((CLINC / "labels.json").read_text())
    desc77 = lambda i: f"{i.replace('_', ' ')} ({i.replace('_', ' ')})"  # noqa: E731

    def probe_ranking(name, positives, world_of):
        """positives: list of (request, gold_tool, sib_pool_tag)."""
        sample = rng.sample(positives, min(N_PER_SOURCE, len(positives)))
        easy = 0
        winners = defaultdict(int)
        for req, gold, tag in sample:
            sibs = [d for d in world_of(tag) if d != gold]
            cands = [gold] + rng.sample(sibs, min(7, len(sibs)))
            rng.shuffle(cands)
            descs = {f"c{i}": d for i, d in enumerate(cands)}
            best, _ = m.select(req, descs)
            if descs[best] == gold:
                easy += 1
            else:
                winners[descs[best][:60]] += 1
        share = round(100 * easy / max(1, len(sample)), 1)
        out["ranking_sources"][name] = {
            "n": len(sample), "easy_mass_pct": share,
            "signal_pct": round(100 - share, 1),
            "top_winners": dict(sorted(winners.items(),
                                       key=lambda kv: -kv[1])[:5])}
        return share

    # -- r1 S1 listwise ------------------------------------------------------
    for ds, path, labels in (
            ("r1_b77_pos", R1 / "listwise-pairs.jsonl", b77_intents),
            ("r1_clinc_pos", R1 / "listwise-pairs.jsonl", clinc_labels)):
        pos = [(r["request"], r["tool"], ds)
               for line in path.read_text().splitlines() if (r := json.loads(line))
               and r["source"] == ("banking77_pos" if "b77" in ds
                                   else "clinc150_pos")]
        world = (lambda tag: [desc77(i) for i in b77_intents]) if "b77" in ds \
            else (lambda tag: [desc77(i) for i in clinc_labels])
        probe_ranking(ds, pos, world)

    # -- r1 S2 null positives --------------------------------------------------
    nulls = [(r["request"], r["tool"], None)
             for line in (R1 / "null-pairs.jsonl").read_text().splitlines()
             if (r := json.loads(line)) and r["source"] == "null_pos_oos"]
    probe_ranking("r1_null_pos", nulls,
                  lambda tag: ["__null__: none of these descriptions match",
                               "none: none of these descriptions match",
                               "no_match: the text is about none of these "
                               "topics"]
                  + [desc77(i) for i in rng.sample(
                      b77_intents, min(30, len(b77_intents)))])

    # -- r1 S3 synthetic (positives with set siblings) ------------------------
    syn = [json.loads(ln) for ln in
           (R1 / "synthetic-labelsets.jsonl").read_text().splitlines()
           if ln.strip()]
    out["s3_polarity_rows_dropped_at_load"] = sum(
        1 for r in syn if r.get("source", "").startswith("synthetic_polarity"))
    set_descs: dict[str, set[str]] = defaultdict(set)
    for r in syn:
        if r.get("set_id"):
            set_descs[r["set_id"]].add(r["tool"])
    syn_pos = [(r["request"], r["tool"], r["set_id"]) for r in syn
               if r.get("label") == 1 and r.get("set_id")]
    probe_ranking("r1_s3_synthetic", syn_pos,
                  lambda tag: sorted(set_descs.get(tag, [])))

    # -- r2 nli dict shape ----------------------------------------------------
    nli = [json.loads(ln) for ln in
           (R2 / "nli-converted.jsonl").read_text().splitlines() if ln.strip()]
    dict_rows = [r for r in nli if r["source"].startswith("nli_dict")]
    hyp_pool = [r["tool"] for r in rng.sample(
        dict_rows, min(300, len(dict_rows)))]
    probe_ranking("r2_nli_dict", [(r["request"], r["tool"], None)
                                  for r in dict_rows],
                  lambda tag: hyp_pool)

    # -- binary worlds: shipped bare yes/no accuracy per source prefix --------
    bin_groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in nli:
        if r["source"].startswith("nli_verdict"):
            key = r["source"]
            bin_groups[key].append((r["request"], r["label"]))
    for line in (R2 / "verdict-pairs.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            key = r["source"].rsplit("_", 1)[0]
            bin_groups[key].append((r["request"], r["label"]))
    for key, rows in sorted(bin_groups.items()):
        sample = rng.sample(rows, min(N_PER_SOURCE, len(rows)))
        ok = 0
        for text, lab in sample:
            picked, _ = m.select(text, ["yes", "no"])
            ok += (picked == "yes") == (lab == 1)
        out["binary_sources"][key] = {
            "n": len(sample), "shipped_bare_acc_pct": round(100 * ok / len(sample), 1)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
