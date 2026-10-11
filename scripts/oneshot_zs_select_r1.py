#!/usr/bin/env python3
"""zero-shot-select r1 ONE-SHOT bar matrix (zero-shot-select-r1/
spec.md; bars NEVER retuned): candidate vs shipped on identical surfaces,
both loaded by EXPLICIT PATH (no loader defaults — the concurrent
tool_match.py work must not touch these measurements).

Round TARGETS (promotion requires ALL):
  banking77 fresh reserve  >= 85   (seed 47, consumed seeds 42-46 excluded)
  clinc150 test reserve    >= 78   (split test, seed 47, consumed excluded)
  unseen-domain reserve    >= 65 pooled top-1 with enriched descriptions
  learned abstention       null-pick precision >= 90 AND recall >= 30
  binary referee           expected-yes >= 50 AND expected-no >= 75.8

Binding HOLD bars (guard/router side): measured by the named instruments,
not here (the gate artifact is byte-untouched); this script records the
candidate rank artifact's own guard-adjacent pair-level numbers as REPORT
lines and the shas of BOTH sides (rule 10).

Writes reports/zs-select-r1-oneshot.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CAND = ROOT / "models/select-rank-zsr1-int8.onnx"
CAND_TOK = ROOT / "models/select-rank-zsr1-tokenizer.json"
SHIP = ROOT / "models/tool-match-int8.onnx"
SHIP_TOK = ROOT / "models/tool-match-tokenizer.json"
OUT = ROOT / "reports/zs-select-r1-oneshot.json"

B77 = ROOT / "data/incoming/external/banking77"
CLINC = ROOT / "data/incoming/external/clinc150"
RESERVE = ROOT / "data/real/zero-shot-select-r1/synthetic-labelsets-reserve.jsonl"
REFEREE = ROOT / "data/eval/binary-select-v1.jsonl"

NULL_DESC = "none of these descriptions match"

B77_FRESH_BAR = 85.0
CLINC_RESERVE_BAR = 78.0
UNSEEN_BAR = 65.0
NULL_PRECISION_BAR = 90.0
NULL_RECALL_BAR = 30.0
REFEREE_YES_BAR = 50.0
REFEREE_NO_BAR = 75.8


def descs77():
    intents = json.loads((B77 / "categories.json").read_text())
    return {i: f"{i.replace('_', ' ')} ({i.replace('_', ' ')})"
            for i in intents}, intents


def descs_clinc():
    labels = json.loads((CLINC / "labels.json").read_text())
    return {i: f"{i.replace('_', ' ')} ({i.replace('_', ' ')})"
            for i in labels}


def b77_pool():
    rows = []
    with (B77 / "test.csv").open(encoding="utf-8", errors="replace",
                                 newline="") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            line = (r.get("category") or r.get("label") or "").strip()
            if t and line:
                rows.append((t, line))
    return rows


def clinc_pool(split):
    return [(r["text"], r["intent"]) for r in
            [json.loads(line) for line in
             (CLINC / f"{split}.jsonl").read_text(
                 encoding="utf-8").splitlines() if line.strip()]]


def fresh_sample(pool, seed, consumed_seeds, n=600):
    consumed = set()
    for s in consumed_seeds:
        consumed |= {t for t, _ in random.Random(s).sample(
            pool, min(600, len(pool)))}
    rest = [r for r in pool if r[0] not in consumed]
    return random.Random(seed).sample(rest, min(n, len(rest)))


def top1(matcher, descs, sample, per_row=None):
    ok = 0
    for text, gold in sample:
        best, _conf = matcher.select(text, descs)
        hit = best == gold
        ok += hit
        if per_row is not None:
            per_row.append(hit)
    return 100 * ok / max(1, len(sample))


def null_windows(matcher, descs, windows):
    """null picks per window: (name, rows, true_oos flag)."""
    out = {}
    full = dict(descs)
    full["__null__"] = NULL_DESC
    for name, rows, is_oos in windows:
        picks = 0
        for text, _gold in rows:
            best, _conf = matcher.select(text, full)
            picks += (best == "__null__")
        out[name] = {"n": len(rows), "null_picks": picks, "true_oos": is_oos}
    return out


def null_metrics(windows):
    tp = sum(w["null_picks"] for w in windows.values() if w["true_oos"])
    total_picks = sum(w["null_picks"] for w in windows.values())
    total_oos = sum(w["n"] for w in windows.values() if w["true_oos"])
    prec = 100 * tp / total_picks if total_picks else 0.0
    rec = 100 * tp / total_oos if total_oos else 0.0
    return prec, rec, total_picks, total_oos


def referee(matcher):
    rows = [json.loads(line) for line in REFEREE.read_text().splitlines()
            if line.strip()]
    yes_ok = no_ok = 0
    for r in rows:
        picked, _conf = matcher.select(r["text"], ["yes", "no"])
        if picked == "yes":
            yes_ok += (r["expected"] == "yes")
        elif picked == "no":
            no_ok += (r["expected"] == "no")
    return {"expected_yes": f"{yes_ok}/250", "expected_no": f"{no_ok}/250",
            "yes_acc": 100 * yes_ok / 250, "no_acc": 100 * no_ok / 250}


def unseen_domain(matcher):
    sets = [json.loads(line) for line in RESERVE.read_text().splitlines()
            if line.strip()]
    per_domain = {}
    tot_ok = tot = 0
    for s in sets:
        # enriched descriptions: label desc + 2 of its own reserve utterances
        descs = {}
        rows = []
        for lab in s["labels"]:
            ex = lab["utterances"][:2]
            d = lab["desc"]
            if not (d.startswith(f"{lab['name']}:")
                    or ":" in d[:30]):
                d = f"{lab['name']}: {d}"
            descs[lab["name"]] = (d + " Examples: "
                                  + " ".join(f'"{u}"' for u in ex))
        for lab in s["labels"]:
            for u in lab["utterances"]:
                rows.append((u, lab["name"]))
        ok = 0
        for text, gold in rows:
            best, _conf = matcher.select(text, descs)
            ok += (best == gold)
        per_domain[s["domain"]] = round(100 * ok / max(1, len(rows)), 2)
        tot_ok += ok
        tot += len(rows)
    return round(100 * tot_ok / max(1, tot), 2), per_domain, tot


def main() -> int:
    from mivia_small_json.tool_match import load_single_matcher

    cand = load_single_matcher(CAND, tokenizer_path=CAND_TOK)
    ship = load_single_matcher(SHIP, tokenizer_path=SHIP_TOK)
    if cand is None or ship is None:
        print("matcher load failed", file=sys.stderr)
        return 2

    out = {
        "candidate_sha256": hashlib.sha256(CAND.read_bytes()).hexdigest(),
        "shipped_sha256": hashlib.sha256(SHIP.read_bytes()).hexdigest(),
        "tokenizer_note": "same bert-base vocab family; bytes compared at "
                          "promotion",
        "bars": {}, "targets": {}, "detail": {},
    }

    # ---- 1. Banking77 fresh reserve (seed 47) --------------------------
    descs, _intents = descs77()
    pool = b77_pool()
    sample = fresh_sample(pool, 47, (42, 43, 44, 45, 46))
    t0 = time.time()
    c = top1(cand, descs, sample)
    s = top1(ship, descs, sample)
    out["detail"]["banking77_fresh_s47"] = {
        "cand": round(c, 2), "shipped": round(s, 2), "n": len(sample),
        "wall_s": round(time.time() - t0, 1)}
    out["targets"]["banking77_fresh>=85"] = c >= B77_FRESH_BAR

    # ---- 2. CLINC150 test reserve (seed 47) ----------------------------
    dcl = descs_clinc()
    sample = fresh_sample(clinc_pool("test"), 47, (42, 44, 45, 46))
    c = top1(cand, dcl, sample)
    s = top1(ship, dcl, sample)
    out["detail"]["clinc150_test_s47"] = {
        "cand": round(c, 2), "shipped": round(s, 2), "n": len(sample)}
    out["targets"]["clinc150_test_reserve>=78"] = c >= CLINC_RESERVE_BAR

    # ---- 3. unseen-domain reserve --------------------------------------
    c, per_dom, n = unseen_domain(cand)
    s, per_dom_s, _ = unseen_domain(ship)
    out["detail"]["unseen_domain"] = {
        "cand_pooled": c, "shipped_pooled": s, "n": n,
        "cand_per_domain": per_dom, "shipped_per_domain": per_dom_s}
    out["targets"]["unseen_domain>=65"] = c >= UNSEEN_BAR

    # ---- 4. learned abstention (null) ----------------------------------
    b77_s42 = [(t, g) for t, g in random.Random(42).sample(
        pool, min(600, len(pool)))[:300]]
    val = clinc_pool("validation")
    clinc_s42 = random.Random(42).sample(val, 300)
    clinc_oos = [(t, "oos") for t, g in val if g == "oos"]
    ood_valid = [(t.strip(), "oos") for t in
                 (ROOT / "data/incoming/external/banking77-oos/"
                         "ood-oos__valid__seq.in").read_text(
                     encoding="utf-8").splitlines() if t.strip()]
    id_valid = [(t.strip(), "oos") for t in
                (ROOT / "data/incoming/external/banking77-oos/"
                        "id-oos__valid__seq.in").read_text(
                    encoding="utf-8").splitlines() if t.strip()]
    windows = [("b77_ood_valid", ood_valid, True),
               ("b77_id_valid", id_valid, True),
               ("clinc_validation_oos", clinc_oos, True),
               ("b77_indomain_s42_300", b77_s42, False),
               ("clinc_indomain_s42_300", clinc_s42, False)]
    wc = null_windows(cand, descs, windows)
    prec, rec, tp, to = null_metrics(wc)
    ws = null_windows(ship, descs, windows)
    sprec, srec, _, _ = null_metrics(ws)
    out["detail"]["learned_abstention"] = {
        "cand": {"precision": round(prec, 2), "recall": round(rec, 2),
                 "null_picks_total": tp, "true_oos_total": to,
                 "windows": wc},
        "shipped": {"precision": round(sprec, 2), "recall": round(srec, 2)},
    }
    out["targets"]["null_precision>=90"] = prec >= NULL_PRECISION_BAR
    out["targets"]["null_recall>=30"] = rec >= NULL_RECALL_BAR

    # ---- 5. binary referee ----------------------------------------------
    rc = referee(cand)
    rs = referee(ship)
    out["detail"]["binary_referee"] = {"cand": rc, "shipped": rs}
    out["targets"]["referee_expected_yes>=50"] = \
        rc["yes_acc"] >= REFEREE_YES_BAR
    out["targets"]["referee_expected_no>=75.8"] = \
        rc["no_acc"] >= REFEREE_NO_BAR

    out["all_targets"] = all(out["targets"].values())
    out["failed_targets"] = [k for k, v in out["targets"].items() if not v]
    out["promote"] = out["all_targets"]
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "detail"},
                     indent=1))
    print("detail written to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
