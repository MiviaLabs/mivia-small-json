#!/usr/bin/env python3
"""realworld-slices r1 one-shot: candidate (mm12-realworld-r1-s42) vs shipped
on the REGISTERED bars (realworld-slices-r1; never retuned).

Binding HOLD bars:
- the seven pinned realworld slice floors (tests/test_realworld_bars.py
  contract: acceptable-set scoring, ambiguous rows excluded from strict)
- chat-ood lineage holds, re-used verbatim: chat-informal >= 55,
  fa-fa >= 70, fresh pooled >= 62.4 (data/wild/realworld-v2)
- held-out routed-and-valid argmax >= 99.5; BFCL routed-to-expected >= 82.0

Round TARGETS (promotion requires BOTH): post-commit strict >= 60,
slack-snippets strict >= 55.

Writes reports/realworld-r1-oneshot.json. Deterministic surfaces only;
eval_ood_class (OOD band incl. retention) is the separate registered
instrument. Long-log credits {log_event, diagnostic} here (the
authoritative bars-test set; the older battery's {log_event}-only credit
was the disclosed instrument artifact).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CAND = ROOT / ("data/incoming/models/mm12-realworld-r1-s42/"
               "mm12-realworld-r1-s42-int8.onnx")
CTOK = ROOT / ("data/incoming/models/mm12-realworld-r1-s42/"
               "mm12-realworld-r1-s42-tokenizer.json")
CCLS = ROOT / ("data/incoming/models/mm12-realworld-r1-s42/"
               "mm12-realworld-r1-s42-classes.json")
OUT = ROOT / "reports/realworld-r1-oneshot.json"

# authoritative credit sets (tests/test_realworld_bars.py ACCEPT)
ACCEPT = {
    "issue-realworld-v2": {"issue"},
    "post-commit": {"commit", "no_tool_call", "text"},
    "chat-informal": {"no_tool_call", "text"},
    "slack-snippets": {"no_tool_call", "text", "log_event"},
    "fa-ar-ood-fa": {"no_tool_call", "text"},
    "fa-ar-ood-ar": {"no_tool_call", "text"},
    "long-log-dumps": {"log_event", "diagnostic"},
}
PINNED_FLOORS = {
    "chat-informal": 42.9, "fa-ar-ood-fa": 58.7, "fa-ar-ood-ar": 48.7,
    "issue-realworld-v2": 85.7, "long-log-dumps": 100.0,
    "post-commit": 0.0, "slack-snippets": 3.5,
}
# chat-ood lineage registered holds (chat-ood)
LINEAGE_HOLDS = {"chat-informal": 55.0, "fa-ar-ood-fa": 70.0}
FRESH_POOLED_FLOOR = 62.4
TARGETS = {"post-commit": 60.0, "slack-snippets": 55.0}


def strict_pct(path: Path, router) -> tuple[float, float, int, int]:
    """(strict % excl. ambiguous, flagged %, n, strict_n) — the
    bench_realworld_slices scoring contract."""
    from mivia_small_json import pipeline

    name = path.stem
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ok = flag_ok = strict_n = 0
    for r in rows:
        obj, _meta = pipeline.classify(r["text"], router=router)
        hit = obj.get("type") in ACCEPT[name]
        flag_ok += hit
        if not r["ambiguous"]:
            ok += hit
            strict_n += 1
    n = len(rows)
    return (100 * ok / max(1, strict_n), 100 * flag_ok / n, n, strict_n)


def main() -> int:
    from mivia_small_json import bootstrap
    from mivia_small_json.encoder import load_encoder
    from mivia_small_json.pipeline import classify as _classify

    bootstrap.register_all_schemas()
    cand = load_encoder(CAND, tokenizer_path=CTOK,
                        classes=tuple(json.loads(CCLS.read_text())))
    if cand is None:
        print("candidate failed to load", file=sys.stderr)
        return 2
    shipped = load_encoder()

    out: dict = {"bars": {}, "targets": {}, "detail": {}}

    # pinned slice floors + lineage holds + round targets (strict scoring)
    for name in sorted(ACCEPT):
        p = ROOT / f"data/wild/realworld/{name}.jsonl"
        c_strict, c_flag, n, strict_n = strict_pct(p, cand)
        s_strict, s_flag, _, _ = strict_pct(p, shipped)
        out["detail"][name] = {
            "cand_strict": round(c_strict, 1), "shipped_strict": round(s_strict, 1),
            "cand_flagged": round(c_flag, 1), "n": n, "strict_n": strict_n}
        if name in PINNED_FLOORS:
            out["bars"][f"floor:{name}>={PINNED_FLOORS[name]}"] = \
                c_strict >= PINNED_FLOORS[name]
        if name in LINEAGE_HOLDS:
            out["bars"][f"hold:{name}>={LINEAGE_HOLDS[name]}"] = \
                c_strict >= LINEAGE_HOLDS[name]
        if name in TARGETS:
            out["targets"][f"target:{name}>={TARGETS[name]}"] = \
                c_strict >= TARGETS[name]

    # fresh pooled (chat-ood registered instrument convention: all rows)
    tot_ok = tot = 0
    for name in ("chat-forum-en", "chat-conversation-non-en",
                 "smalltalk-qa-mixed"):
        p = ROOT / f"data/wild/realworld-v2/{name}.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()
                if line.strip()]
        ok = 0
        for r in rows:
            obj, _ = _classify(r["text"], router=cand)
            ok += obj.get("type") in {"text", "no_tool_call"}
        tot_ok += ok
        tot += len(rows)
        out["detail"][f"fresh:{name}"] = round(100 * ok / len(rows), 1)
    pooled = 100 * tot_ok / tot
    out["detail"]["fresh:POOLED"] = round(pooled, 1)
    out["bars"][f"fresh_pooled>={FRESH_POOLED_FLOOR}"] = \
        pooled >= FRESH_POOLED_FLOOR

    # held-out argmax
    held = [json.loads(line) for line in
            (ROOT / "data/heldout/test.jsonl").read_text().splitlines()
            if line.strip()]
    ok = sum(1 for r in held
             if _classify(r["raw"], router=cand)[0].get("type")
             == r["schema"])
    out["detail"]["heldout_cand"] = round(100 * ok / len(held), 2)
    out["bars"]["heldout>=99.5"] = out["detail"]["heldout_cand"] >= 99.5

    # BFCL routed-to-expected
    calls = [json.loads(line) for line in
             (ROOT / "data/incoming/external/bfcl/calls.jsonl").read_text()
             .splitlines() if line.strip()]
    tc = sum(1 for row in calls
             if _classify(row["text"], router=cand)[0].get("type") == "tool_call")
    out["detail"]["bfcl_tool_call"] = f"{tc}/{len(calls)}"
    out["bars"]["bfcl>=82.0"] = 100 * tc / len(calls) >= 82.0

    out["all_binding_bars"] = all(out["bars"].values())
    out["all_targets"] = all(out["targets"].values())
    out["promote"] = out["all_binding_bars"] and out["all_targets"]
    out["failed_bars"] = [k for k, v in out["bars"].items() if not v]
    out["failed_targets"] = [k for k, v in out["targets"].items() if not v]
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
