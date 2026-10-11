#!/usr/bin/env python3
"""Reproduce the measured enrichment number through the NEW dict-form path.

Runs the exact .scratch/zs-diag/run_enriched600.py bench (Banking77 test,
seed-42 n=600, per-label first-3-train-texts examples) through
api.do_select — the served /v1/select shape with candidates as
{"name", "description", "examples"} dicts — and cross-checks per-row
parity against the diagnostic's committed jsonl. Expected: 63.8 top-1
(380/600), byte-identical predictions.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / ".scratch/zs-diag"
CSV77 = ROOT / "data/incoming/external/banking77/test.csv"
TRAIN77 = ROOT / "data/incoming/external/banking77/train.csv"
CATS = ROOT / "data/incoming/external/banking77/categories.json"


def main() -> int:
    intents = json.loads(CATS.read_text())
    ex: dict[str, list[str]] = {}
    with TRAIN77.open(encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            c = (r.get("category") or "").strip()
            if t and c:
                ex.setdefault(c, [])
                if len(ex[c]) < 3:
                    ex[c].append(t)

    # The dict-form candidates: the API shape, not hand-built strings.
    cands = [{"name": i,
              "description": f"{i.replace('_', ' ')} "
                             f"({i.replace('_', ' ')})",
              "examples": ex.get(i, [])}
             for i in intents]

    rows = []
    with CSV77.open(encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            text = (r.get("text") or "").strip()
            label = (r.get("category") or r.get("label") or "").strip()
            if text and label:
                rows.append((text, label))
    sample = random.Random(42).sample(rows, 600)
    sample_set = {t for t, _ in sample}
    leak = sum(1 for c in ex for t in ex[c] if t in sample_set)

    from mivia_small_json import bootstrap
    from mivia_small_json.tool_match import _get_guard

    bootstrap.register_all_schemas()
    matcher = _get_guard()
    if matcher is None:
        raise SystemExit("guard unavailable")

    # The 77-intent catalogue exceeds the /v1/select endpoint's pre-existing
    # MAX_CANDIDATES=64 law (unchanged — it bounds per-request cross-encoder
    # passes), so the bench rides the LIBRARY dict-form path: the exact
    # rendering the endpoint serves for catalogues of up to 64 candidates.
    top1 = 0
    preds = []
    t0 = time.perf_counter()
    with (OUT / "banking77-enriched-600-dictform.jsonl").open(
            "w", encoding="utf-8") as out:
        for idx, (text, gold) in enumerate(sample):
            choice, conf, status = matcher.select(text, cands,
                                                  with_status=True)
            hit = choice == gold
            top1 += hit
            preds.append(choice)
            out.write(json.dumps(
                {"i": idx, "gold": gold, "pred": choice,
                 "conf": round(float(conf), 4), "status": status,
                 "hit": hit}, ensure_ascii=False) + "\n")

    # per-row parity against the diagnostic's own enriched run
    diag = [json.loads(line) for line in
            (OUT / "banking77-enriched-600.jsonl").open()]
    same = sum(1 for a, b in zip(diag, preds) if a["pred"] == b)
    result = {
        "path": "matcher.select dict-form candidates (library; the "
                "/v1/select rendering for <= MAX_CANDIDATES=64 catalogues)",
        "rows": 600,
        "leak_check": leak,
        "top1_dict_form": round(100 * top1 / 600, 2),
        "correct": top1,
        "pred_parity_vs_zsdiag_enriched": f"{same}/600",
        "status_counts": {
            s: sum(1 for line in (json.loads(x) for x in
                    (OUT / "banking77-enriched-600-dictform.jsonl")
                    .read_text().splitlines()) if line["status"] == s)
            for s in ("ok", "ambiguous", "abstain")},
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(result, indent=1))
    (OUT / "reproduce-dictform-summary.json").write_text(
        json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
