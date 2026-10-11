#!/usr/bin/env python3
"""zero-shot-select r3 MID-ROUND GATES (zero-shot-select-r3/
spec.md): run after the rank head's FIRST full train, before any other bar.

1. Referee gate (the round's registered iteration surface): bare
   select(text, ["yes","no"]) on data/eval/binary-select-v1.jsonl.
   PASS = expected-yes >= 50% AND expected-no >= 74.8%. On FAIL: iterate
   the NLI/verdict weights and group rendering only.
2. Banking77 STOP gate: the candidate's top-1 on the s47 fresh reserve
   (consumed 42-46 excluded, n=600) is a pure argmax metric — no threshold
   or calibration choice can move it (the "even at oracle calibration"
   clause). If it is < 85 -> the round is DEAD: STOP, record, demote, do
   NOT run the full matrix.

Writes .scratch/zs-select-r3/midround-gates.json (scratch, not a report).
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

CAND = ROOT / "models/select-rank-zsr3-int8.onnx"
CAND_TOK = ROOT / "models/select-rank-zsr3-tokenizer.json"
REFEREE = ROOT / "data/eval/binary-select-v1.jsonl"
B77 = ROOT / "data/incoming/external/banking77"
OUT = ROOT / ".scratch/zs-select-r3/midround-gates.json"


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
    out["banking77_fresh_s47"] = {"top1": round(100 * ok / len(sample), 2),
                                  "n": len(sample)}
    out["banking77_pass"] = out["banking77_fresh_s47"]["top1"] >= 85.0
    out["stop"] = not out["banking77_pass"]

    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if out["stop"]:
        print("STOP GATE: Banking77 < 85 — the round is dead; do not run "
              "the full matrix", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
