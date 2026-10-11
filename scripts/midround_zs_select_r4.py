#!/usr/bin/env python3
"""zero-shot-select r4 MID-ROUND GATES (zero-shot-select-r4/
spec.md): run after the rank head's FIRST full train, before any other bar.

1. Referee gate (the round's registered iteration surface): bare
   select(text, ["yes","no"]) on data/eval/binary-select-v1.jsonl.
   PASS = expected-yes >= 50% AND expected-no >= 74.8%. On FAIL: iterate
   the NLI/verdict weights only (each iteration registered in the spec's
   RESULT before the re-train).
2. Banking77 gate (s47 fresh reserve, consumed 42-46 excluded, n=600; a
   pure argmax metric — no calibration knob exists), the REGISTERED
   three-way rule:
   - >= 85          -> continue to the full matrix (promote path);
   - [81, 85)       -> continue to the full matrix (the BAR-ADJACENT rule
                       needs the other bars to decide);
   - < 81 (below the r2 pointwise control) -> the round is DEAD: STOP,
                       record, demote, do NOT run the full matrix.

Writes .scratch/zs-select-r4/midround-gates.json (scratch, not a report).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CAND = ROOT / "data/incoming/models/zsr4-candidates/select-rank-zsr4-int8.onnx"
CAND_TOK = ROOT / ("data/incoming/models/zsr4-candidates/"
                   "select-rank-zsr4-tokenizer.json")
REFEREE = ROOT / "data/eval/binary-select-v1.jsonl"
B77 = ROOT / "data/incoming/external/banking77"
OUT = ROOT / ".scratch/zs-select-r4/midround-gates.json"

R2_CONTROL = 81.0     # the recorded r2 candidate (the isolation baseline)
BAR = 85.0


def main() -> int:
    from mivia_small_json.tool_match import load_single_matcher

    cand = load_single_matcher(CAND, tokenizer_path=CAND_TOK)
    if cand is None:
        print("candidate failed to load", file=sys.stderr)
        return 2
    out = {"candidate_sha256":
           hashlib.sha256(CAND.read_bytes()).hexdigest()}

    rows = [json.loads(line) for line in REFEREE.read_text().splitlines()
            if line.strip()]
    yes_ok = no_ok = 0
    for r in rows:
        picked, _conf = cand.select(r["text"], ["yes", "no"])
        if picked == "yes":
            yes_ok += (r["expected"] == "yes")
        elif picked == "no":
            no_ok += (r["expected"] == "no")
    out["referee"] = {"expected_yes": f"{yes_ok}/250",
                      "expected_no": f"{no_ok}/250",
                      "yes_acc": 100 * yes_ok / 250,
                      "no_acc": 100 * no_ok / 250}
    out["referee_pass"] = (out["referee"]["yes_acc"] >= 50.0 and
                           out["referee"]["no_acc"] >= 74.8)

    pool = []
    with (B77 / "test.csv").open(encoding="utf-8", errors="replace",
                                 newline="") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            line = (r.get("category") or r.get("label") or "").strip()
            if t and line:
                pool.append((t, line))
    consumed = set()
    for s in (42, 43, 44, 45, 46):
        consumed |= {t for t, _ in random.Random(s).sample(
            pool, min(600, len(pool)))}
    rest = [r for r in pool if r[0] not in consumed]
    sample = random.Random(47).sample(rest, min(600, len(rest)))
    intents = json.loads((B77 / "categories.json").read_text())
    descs = {i: f"{i.replace('_', ' ')} ({i.replace('_', ' ')})"
             for i in intents}
    ok = 0
    for text, gold in sample:
        best, _conf = cand.select(text, descs)
        ok += (best == gold)
    top1 = round(100 * ok / len(sample), 2)
    out["banking77_fresh_s47"] = {"top1": top1, "n": len(sample)}
    out["banking77_zone"] = ("PASS" if top1 >= BAR else
                             "BAR-ADJACENT" if top1 >= R2_CONTROL else
                             "DEAD")
    out["stop"] = out["banking77_zone"] == "DEAD"

    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if out["stop"]:
        print(f"STOP GATE: Banking77 {top1} < {R2_CONTROL} (the r2 "
              "pointwise control) — the round is dead; do not run the "
              "full matrix", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
