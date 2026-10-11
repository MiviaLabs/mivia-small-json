#!/usr/bin/env python3
"""zero-shot-select r3 ONE-SHOT bar matrix (zero-shot-select-r3/
spec.md; bars NEVER retuned): candidate vs shipped on identical surfaces,
both loaded by EXPLICIT PATH (no loader defaults).

Round TARGETS (promotion requires ALL):
  banking77 fresh reserve  >= 85   (seed 47, consumed seeds 42-46 excluded)
  clinc150 test reserve    >= 78   (split test, seed 47, consumed excluded)
  unseen-domain reserve    >= 65 pooled top-1 with enriched descriptions
  learned abstention       the REGISTERED SHIP PATH from the calibration
                           JSON (N: dedicated no-match head + pooled tau;
                           M: decision-layer margin abstain, tau by band):
                           precision >= 90 AND recall >= 30 (Path N) /
                           oos-coverage >= 40 (Path M) pooled over the oos
                           windows, in-domain s42-300 windows in the
                           precision denominator; the burned alternative
                           path is reported as a disclosure line
  binary referee           expected-yes >= 50 AND expected-no >= 74.8
                           (bare select(text, ["yes","no"]))

Writes reports/zs-select-r3-oneshot.json. The instrument REFUSES to run on
a calibration/artifact sha mismatch (rule 10).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CAND = ROOT / "models/select-rank-zsr3-int8.onnx"
CAND_TOK = ROOT / "models/select-rank-zsr3-tokenizer.json"
NOMATCH = ROOT / "models/select-nomatch-zsr3-int8.onnx"
NOMATCH_TOK = ROOT / "models/select-nomatch-zsr3-tokenizer.json"
CALIB = ROOT / "data/real/zero-shot-select-r3/abstain-calibration.json"
SHIP = ROOT / "models/tool-match-int8.onnx"
SHIP_TOK = ROOT / "models/tool-match-tokenizer.json"
OUT = ROOT / "reports/zs-select-r3-oneshot.json"

B77 = ROOT / "data/incoming/external/banking77"
CLINC = ROOT / "data/incoming/external/clinc150"
RESERVE = ROOT / "data/real/zero-shot-select-r1/synthetic-labelsets-reserve.jsonl"
REFEREE = ROOT / "data/eval/binary-select-v1.jsonl"

NULL_KEY = "__null__"
NULL_DESC = "none of these descriptions match"
NULL_CAND = f"{NULL_KEY}: {NULL_DESC}"

B77_FRESH_BAR = 85.0
CLINC_RESERVE_BAR = 78.0
UNSEEN_BAR = 65.0
NULL_PRECISION_BAR = 90.0
NULL_RECALL_BAR = 30.0        # Path N
M_COVERAGE_BAR = 40.0         # Path M (registered "answered-coverage >= 40")
REFEREE_YES_BAR = 50.0
REFEREE_NO_BAR = 74.8


def descs77():
    intents = json.loads((B77 / "categories.json").read_text())
    return {i: f"{i.replace('_', ' ')} ({i.replace('_', ' ')})"
            for i in intents}


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


def top1(matcher, descs, sample):
    ok = 0
    for text, gold in sample:
        best, _conf = matcher.select(text, descs)
        ok += (best == gold)
    return 100 * ok / max(1, len(sample))


def tau_for(calib: dict, k: int) -> float:
    """nearest fitted band by |log(k / k_fit)|, ties to the larger k."""
    best = min(calib["path_M"]["bands"]["tau_by_k"],
               key=lambda p: (abs(math.log(k / p["k"])), -p["k"]))
    return best["tau_margin"]


def margin_of(matcher, text: str, pool: dict) -> float:
    sc = sorted((matcher._score_raw(text, c) for c in pool.values()),
                reverse=True)
    return sc[0] - sc[1] if len(sc) > 1 else sc[0]


def abstain_windows_pathN(matcher, nomatch, tau, descs, windows):
    out = {}
    for name, rows, is_oos in windows:
        picks = 0
        for text, _gold in rows:
            p = nomatch._score_raw(text, NULL_CAND)
            if p >= tau:
                picks += 1
        out[name] = {"n": len(rows), "abstains": picks, "true_oos": is_oos}
    return out


def abstain_windows_pathM(matcher, window_pools, windows, calib):
    """window_pools: name -> real-candidate pool dict for that window's
    domain (b77 windows over the 77 intents, clinc windows over 151)."""
    out = {}
    for name, rows, is_oos in windows:
        pool = window_pools[name]
        picks = answered = 0
        for text, _gold in rows:
            k = len(pool)
            if margin_of(matcher, text, pool) < tau_for(calib, k):
                picks += 1
            else:
                answered += 1
        out[name] = {"n": len(rows), "abstains": picks,
                     "answered": answered, "true_oos": is_oos}
    return out


def abstain_windows_r1(matcher, descs, windows):
    """The shipped r1 path: null candidate in rank contention, argmax."""
    out = {}
    full = dict(descs)
    full[NULL_KEY] = NULL_DESC
    for name, rows, is_oos in windows:
        picks = 0
        for text, _gold in rows:
            best, _conf = matcher.select(text, full)
            picks += (best == NULL_KEY)
        out[name] = {"n": len(rows), "abstains": picks, "true_oos": is_oos}
    return out


def metrics(windows):
    tp = sum(w["abstains"] for w in windows.values() if w["true_oos"])
    total_picks = sum(w["abstains"] for w in windows.values())
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
        descs = {}
        rows = []
        for lab in s["labels"]:
            ex = lab["utterances"][:2]
            d = lab["desc"]
            if not (d.startswith(f"{lab['name']}:") or ":" in d[:30]):
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
    nom = load_single_matcher(NOMATCH, tokenizer_path=NOMATCH_TOK)
    calib = json.loads(CALIB.read_text())
    if cand is None or ship is None or nom is None:
        print("matcher load failed", file=sys.stderr)
        return 2
    cand_sha = hashlib.sha256(CAND.read_bytes()).hexdigest()
    nom_sha = hashlib.sha256(NOMATCH.read_bytes()).hexdigest()
    if calib.get("rank_sha256") and calib["rank_sha256"] != cand_sha:
        print(f"REFUSING: calibration rank sha != candidate bytes "
              f"({calib['rank_sha256'][:12]} vs {cand_sha[:12]})",
              file=sys.stderr)
        return 2
    ship_path = calib.get("ship_path")
    if ship_path == "N":
        if calib.get("head_sha256") != nom_sha:
            print("REFUSING: calibration head sha mismatch", file=sys.stderr)
            return 2
        tau = calib["path_N"]["tau"]
        if tau is None:
            print("REFUSING: ship_path N but calibration infeasible",
                  file=sys.stderr)
            return 2

    out = {
        "candidate_sha256": cand_sha,
        "shipped_sha256": hashlib.sha256(SHIP.read_bytes()).hexdigest(),
        "nomatch_sha256": nom_sha,
        "ship_path": ship_path,
        "tokenizer_note": "same bert-base vocab family; bytes compared at "
                          "promotion",
        "bars": {}, "targets": {}, "detail": {},
    }

    # ---- 1. Banking77 fresh reserve (seed 47) --------------------------
    descs = descs77()
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

    # ---- 4. learned abstention (registered ship path) --------------------
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
    window_pools = {"b77_ood_valid": descs, "b77_id_valid": descs,
                    "b77_indomain_s42_300": descs,
                    "clinc_validation_oos": dcl,
                    "clinc_indomain_s42_300": dcl}

    abst_detail = {"ship_path": ship_path}
    if ship_path == "N":
        wc = abstain_windows_pathN(cand, nom, tau, descs, windows)
        prec, rec, tp, to = metrics(wc)
        out["targets"]["abstain_precision>=90"] = prec >= NULL_PRECISION_BAR
        out["targets"]["abstain_recall>=30"] = rec >= NULL_RECALL_BAR
        abst_detail["cand"] = {
            "path": "N: dedicated no-match head + pooled tau "
                    f"({tau})",
            "precision": round(prec, 2), "recall": round(rec, 2),
            "abstains_total": tp, "true_oos_total": to, "windows": wc}
        # burned alternative as disclosure: Path M on the same bytes
        wm = abstain_windows_pathM(cand, window_pools, windows, calib)
        mp, mr, _, _ = metrics(wm)
        abst_detail["cand_pathM_disclosure_nonbinding"] = {
            "precision": round(mp, 2), "recall": round(mr, 2)}
    else:
        wm = abstain_windows_pathM(cand, window_pools, windows, calib)
        prec, rec, tp, to = metrics(wm)
        answered_oos = sum(w["answered"] for w in wm.values()
                           if w["true_oos"])
        total_oos = sum(w["n"] for w in wm.values() if w["true_oos"])
        coverage = 100 * (total_oos - answered_oos) / total_oos
        out["targets"]["abstain_precision>=90"] = prec >= NULL_PRECISION_BAR
        out["targets"]["oos_coverage>=40"] = coverage >= M_COVERAGE_BAR
        abst_detail["cand"] = {
            "path": ("M: decision-layer margin abstain, tau_margin by "
                     "band (nearest fitted k)"),
            "precision": round(prec, 2), "oos_coverage": round(coverage, 2),
            "abstains_total": tp, "true_oos_total": to, "windows": wm}
        # disclosure: Path N numbers on the same bytes (at the most
        # permissive fitted tau — NOT the registered decision)
        tau_max = max(p["tau_margin"]
                      for p in calib["path_M"]["bands"]["tau_by_k"])
        wn = abstain_windows_pathN(cand, nom, tau_max, descs, windows)
        np_, nr, _, _ = metrics(wn)
        abst_detail["cand_pathN_disclosure_nonbinding"] = {
            "note": f"measured at the most permissive fitted tau "
                    f"({tau_max}); not the registered path",
            "precision": round(np_, 2), "recall": round(nr, 2)}
    # shipped baseline both ways (disclosed paths)
    ws = abstain_windows_r1(ship, descs, windows)
    sprec, srec, _, _ = metrics(ws)
    ws_m = abstain_windows_pathM(ship, window_pools, windows, calib)
    smprec, smrec, _, _ = metrics(ws_m)
    abst_detail["shipped"] = {
        "r1_path": {"precision": round(sprec, 2), "recall": round(srec, 2)},
        "margin_path_disclosure": {"precision": round(smprec, 2),
                                   "recall": round(smrec, 2)}}
    out["detail"]["learned_abstention"] = abst_detail

    # ---- 5. binary referee ----------------------------------------------
    rc = referee(cand)
    rs = referee(ship)
    out["detail"]["binary_referee"] = {"cand": rc, "shipped": rs}
    out["targets"]["referee_expected_yes>=50"] = \
        rc["yes_acc"] >= REFEREE_YES_BAR
    out["targets"]["referee_expected_no>=74.8"] = \
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
