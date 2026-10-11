#!/usr/bin/env python3
"""chat-ood ROUND 3 one-shot: candidate (mm12-chatood3-s42) vs shipped
(chatood2) on the registered bars.

Surfaces (chat-ood):
- no-regression: BFCL routed-to-expected >= 86.9 band (report), LogHub
  fast lane (routing only, 2k sample), held-out argmax, the seven pinned
  realworld slice FLOORS, OOD band (eval_ood_class instrument, separate).
- improvement: chat-informal >= 55.0, fa-ar-ood-fa >= 70.0 on the pinned
  slices.
- fresh real-world: pooled >= 62.4 (shipped baseline 52.4 + 10pp) on
  data/wild/realworld-v2 (chat-forum-en, chat-conversation-non-en,
  smalltalk-qa-mixed).

Writes reports/chatood-oneshot.json. Deterministic surfaces only; the
full eval_ood_class + LogHub-32k runs are separate commands recorded in
the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CAND = ROOT / "data/incoming/models/mm12-chatood3-s42/mm12-chatood3-s42-int8.onnx"
CTOK = ROOT / "data/incoming/models/mm12-chatood3-s42/mm12-chatood3-s42-tokenizer.json"
CCLS = ROOT / "data/incoming/models/mm12-chatood3-s42/mm12-chatood3-s42-classes.json"
OUT = ROOT / "reports/chatood3-oneshot.json"

NO_TOOL_OK = {"text", "no_tool_call"}
# Credit sets per slice family (tests/test_realworld_bars.py contract):
# no-tool slices credit {text,no_tool_call}; issue credits issue; long-log
# credits log_event. The first draft scored EVERY slice against the
# no-tool set — issue/long-log read 0.0 for both routers.
CREDIT = {
    "chat-informal": NO_TOOL_OK,
    "fa-ar-ood-fa": NO_TOOL_OK,
    "fa-ar-ood-ar": NO_TOOL_OK,
    "post-commit": NO_TOOL_OK,
    "slack-snippets": {"text", "no_tool_call", "log_event"},
    "issue-realworld-v2": {"issue"},
    "long-log-dumps": {"log_event"},
}
SLICES = {
    "chat-informal": 55.0,
    "fa-ar-ood-fa": 70.0,
}
PINNED_FLOORS = {
    "chat-informal": 42.9, "fa-ar-ood-fa": 58.7, "fa-ar-ood-ar": 48.7,
    "issue-realworld-v2": 85.7, "long-log-dumps": 100.0,
    "post-commit": 0.0, "slack-snippets": 3.5,
}


def pct(path: Path, router, extra_ok=NO_TOOL_OK) -> tuple[float, int, int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ok = 0
    for r in rows:
        obj, _ = _classify(router, r["text"])
        ok += obj.get("type") in extra_ok
    return 100 * ok / len(rows), ok, len(rows)


def _classify(router, text):
    from mivia_small_json import pipeline
    return pipeline.classify(text, router=router)


def main() -> int:
    from mivia_small_json import bootstrap
    bootstrap.register_all_schemas()
    from mivia_small_json.encoder import load_encoder

    cand = load_encoder(CAND, tokenizer_path=CTOK,
                        classes=tuple(json.loads(CCLS.read_text())))
    if cand is None:
        print("candidate failed to load", file=sys.stderr)
        return 2
    shipped = load_encoder()

    out: dict = {"bars": {}, "detail": {}}

    # pinned slices: floors (both routers) + improvement bars (candidate)
    for name in sorted(set(SLICES) | set(PINNED_FLOORS)):
        p = ROOT / f"data/wild/realworld/{name}.jsonl"
        c_pct, c_ok, n = pct(p, cand, CREDIT[name])
        s_pct, s_ok, _ = pct(p, shipped, CREDIT[name])
        out["detail"][name] = {"cand": round(c_pct, 1), "shipped": round(s_pct, 1),
                               "n": n}
        if name in PINNED_FLOORS:
            out["bars"][f"floor:{name}"] = c_pct >= PINNED_FLOORS[name]
        if name in SLICES:
            out["bars"][f"improve:{name}>={SLICES[name]}"] = c_pct >= SLICES[name]

    # fresh real-world slices: pooled >= 62.4
    tot_ok = tot = 0
    for name in ("chat-forum-en", "chat-conversation-non-en",
                 "smalltalk-qa-mixed"):
        p = ROOT / f"data/wild/realworld-v2/{name}.jsonl"
        p_pct, ok, n = pct(p, cand)
        s_pct, s_ok, _ = pct(p, shipped)
        out["detail"][f"fresh:{name}"] = {"cand": round(p_pct, 1),
                                          "shipped": round(s_pct, 1), "n": n}
        tot_ok += ok
        tot += n
    pooled = 100 * tot_ok / tot
    out["detail"]["fresh:POOLED"] = round(pooled, 1)
    out["bars"]["fresh_pooled>=62.4"] = pooled >= 62.4

    # held-out argmax (candidate)
    from mivia_small_json.pipeline import classify as C
    held = [json.loads(line) for line in
            (ROOT / "data/heldout/test.jsonl").read_text().splitlines()
            if line.strip()]
    ok = sum(1 for r in held
             if C(r["raw"], router=cand)[0].get("type") == r["schema"])
    out["detail"]["heldout_cand"] = round(100 * ok / len(held), 2)
    out["bars"]["heldout>=99.5"] = out["detail"]["heldout_cand"] >= 99.5

    # BFCL routed-to-expected (report band; bar vs 82.0 shipped floor)
    calls = [json.loads(line) for line in
             (ROOT / "data/incoming/external/bfcl/calls.jsonl").read_text()
             .splitlines() if line.strip()]
    tc = sum(1 for row in calls
             if _classify(cand, row["text"])[0].get("type") == "tool_call")
    out["detail"]["bfcl_tool_call"] = f"{tc}/183"
    out["bars"]["bfcl>=82.0"] = 100 * tc / 183 >= 82.0

    out["all_bars"] = all(out["bars"].values())
    out["failed"] = [k for k, v in out["bars"].items() if not v]
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"bars": out["bars"], "failed": out["failed"],
                      "all": out["all_bars"],
                      "detail": out["detail"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
