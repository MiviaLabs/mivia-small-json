#!/usr/bin/env python3
"""Kill-shot: does a much larger model separate near-misses zero-shot?

Before spending GPU time on distillation, scaling, or new heads, answer the
question they all depend on: is "this tool structurally fits but cannot serve
the goal" separable by a model far larger than the shipped guard, with no
training on this task at all?

Protocol: take an off-the-shelf cross-encoder reranker ~25x the shipped
MiniLM-L6 (BAAI/bge-reranker-v2-m3, 568M), score every (request, tool) pair in
the 884 BFCL near-miss catalogues and the 183 real calls, and sweep the
threshold. Compare the frontier against the shipped guard (69.3% abstention at
97.3% retention) and the staged SelectiveNet arm (80.1% / 95.6%).

- If the big model's frontier is clearly better, scale is a live lever and the
  Tier 1-3 plan is worth the GPU time.
- If it collapses on the same rows, the failure is not capacity and the plan
  should be killed before it starts.

Read-only apart from the model download.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
FP_EVAL = BFCL / "irrelevance_eval.jsonl"
CALLS = BFCL / "calls.jsonl"
SHIPPED = {"abstention": 69.3, "retention": 97.3}
SELECTIVENET = {"abstention": 80.1, "retention": 95.6}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from mivia_small_json.tool_match import parse_catalogue_raw, render_tool

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} (cuda available: {torch.cuda.is_available()})")
    tok = AutoTokenizer.from_pretrained(args.model)
    net = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    net.eval()

    def best_score(text: str) -> float | None:
        tools, request = parse_catalogue_raw(text)
        if not tools or not request:
            return None
        spans = [render_tool(t) for t in tools]
        enc = tok([request] * len(spans), spans, truncation=True,
                  max_length=args.max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = net(**enc).logits.view(-1)
        return float(torch.sigmoid(logits).max().cpu())

    def scored(rows: list[dict]) -> list[float]:
        out = []
        for row in rows:
            t0 = time.perf_counter()
            s = best_score(row["text"])
            out.append(s if s is not None else -1.0)
            if len(out) % 100 == 0:
                print(f"  {len(out)} rows · {1000 * (time.perf_counter() - t0):.1f} ms/row",
                      flush=True)
        return out

    fp_rows = _rows(FP_EVAL)
    call_rows = _rows(CALLS)
    print(f"scoring {len(fp_rows)} near-misses + {len(call_rows)} calls ...")
    fp = scored(fp_rows)
    calls = scored(call_rows)

    print(f"\n{'thr':>6} {'abstention':>11} {'retention':>10}")
    frontier = []
    for i in range(1, 100):
        thr = i / 100
        ab = 100 * sum(1 for s in fp if s < thr) / len(fp)
        rt = 100 * sum(1 for s in calls if s >= thr) / len(calls)
        frontier.append((thr, ab, rt))
    for thr, ab, rt in frontier[::10]:
        print(f"{thr:>6.2f} {ab:>10.1f}% {rt:>9.1f}%")

    eligible = [f for f in frontier if f[2] >= 95.0]
    best = max(eligible, key=lambda f: f[1]) if eligible else None
    print(f"\nshipped guard      : abstention {SHIPPED['abstention']}%  "
          f"retention {SHIPPED['retention']}%")
    print(f"SelectiveNet arm   : abstention {SELECTIVENET['abstention']}%  "
          f"retention {SELECTIVENET['retention']}%")
    if best:
        print(f"kill-shot model    : abstention {best[1]:.1f}%  "
              f"retention {best[2]:.1f}%  (thr={best[0]:.2f})")
        verdict = ("SCALE HELPS — a 25x model separates near-misses better"
                   if best[1] > SELECTIVENET["abstention"]
                   else "SCALE DOES NOT HELP — a 25x model does not beat the 23 MB arm")
        print(f"\n{verdict}")
    else:
        print("kill-shot model    : no threshold reaches 95% retention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
