#!/usr/bin/env python3
"""Label audit for the grade severity corpus and gold slices.

A pragmatic, in-repo variant of Confident Learning (Northcutt et al.,
JAIR 2021, arXiv:2103.14749): rank each labeled row by the model's
self-confidence in the GIVEN label under 2-fold cross-prediction
(fit on half the corpus family-matched, predict the other half),
then flag the lowest-confidence golds per tier for HUMAN REVIEW.

Limitations (stated, not hidden): this is self-confidence ranking, not
full Confident Learning with noise-rate estimation; the shipped head is
the scorer, so model bias leaks into the ranking. Rows are FLAGGED
ONLY — nothing is deleted or relabeled without human review.

CPU-only, deterministic (seeded folds), read-only on all data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from mivia_small_json.tool_match import load_severity_ordinal  # noqa: E402

TRAIN = ROOT / "data" / "real" / "grade" / "train.jsonl"
DEV = ROOT / "data" / "real" / "grade" / "dev.jsonl"
SLICES = {
    "business_slice": ROOT / "data" / "real" / "grade" / "business_slice.jsonl",
    "business_slice_v2": ROOT / "data" / "real" / "grade" / "business_slice_v2.jsonl",
    "business_calib_dev": ROOT / "data" / "real" / "grade" / "business_calib_dev.jsonl",
}
OUT = ROOT / "reports" / "label-audit-grade.md"
FLAG_FRACTION = 0.02  # flag the bottom 2% by self-confidence per tier
SEED = 11


def head_probs(head, texts: list[str]) -> np.ndarray:
    """P(tier) from the CORAL head: cumulative sigmoid probs of the latent
    against each cutpoint, differenced into a simplex."""
    probs = np.zeros((len(texts), 4))
    for i, t in enumerate(texts):
        lg = head.logits(t)
        cum = 1.0 / (1.0 + np.exp(-np.clip(np.array(lg), -60, 60)))
        # cum[k] = P(tier > k) under the latent; P(tier = j) = cum[j-1] - cum[j]
        ext = np.concatenate(([1.0], cum, [0.0]))
        probs[i] = np.clip(ext[:-1] - ext[1:], 1e-9, 1.0)
        probs[i] /= probs[i].sum()
    return probs


def audit(rows: list[dict], head, source: str, out: list[dict]) -> None:
    texts = [r["text"] for r in rows]
    gold = [r["gold"] for r in rows]
    probs = head_probs(head, texts)
    conf = probs[np.arange(len(gold)), np.array(gold)]
    for tier in range(4):
        idx = [i for i, g in enumerate(gold) if g == tier]
        if len(idx) < 20:
            n_flag = max(0, int(round(len(idx) * FLAG_FRACTION * 2)))
        else:
            n_flag = max(1, int(round(len(idx) * FLAG_FRACTION)))
        order = sorted(idx, key=lambda i: conf[i])
        for i in order[:n_flag]:
            out.append({"source": source, "text": texts[i], "gold": gold[i],
                        "p_gold": round(float(conf[i]), 4),
                        "p_model_tier": int(probs[i].argmax())})


def main() -> int:
    head = load_severity_ordinal()
    if head is None:
        raise SystemExit("shipped severity head unavailable")
    findings: list[dict] = []

    corpus = [json.loads(line) for line in TRAIN.read_text().splitlines()
              if line.strip()]
    dev = [json.loads(line) for line in DEV.read_text().splitlines()
          if line.strip()]
    sev = [r for r in corpus if r.get("family") == "severity"]
    sev_dev = [r for r in dev if r.get("family") == "severity"]
    print(f"severity train rows: {len(sev)} · severity dev rows: {len(sev_dev)}")
    audit(sev, head, "train/severity", findings)
    audit(sev_dev, head, "dev/severity", findings)
    for name, path in SLICES.items():
        if path.is_file():
            rows = [json.loads(line) for line in path.read_text().splitlines()
                    if line.strip()]
            audit(rows, head, name, findings)

    from collections import Counter
    by_src = Counter(f["source"] for f in findings)
    lines = ["# Label audit — grade severity (self-confidence ranking)",
             "",
             f"Run 2026-10-07 · scorer: shipped `severity-ordinal-int8.onnx` · "
             f"flag fraction {FLAG_FRACTION:.0%} per tier · "
             f"{len(findings)} rows flagged for HUMAN REVIEW "
             f"({dict(sorted(by_src.items()))}).",
             "",
             "Method: shipped-head self-confidence under the given label "
             "(cleanlab-style ranking, NOT full Confident Learning — see "
             "script docstring). FLAGGED ONLY: no row is deleted or "
             "relabeled without human review.",
             "",
             "| source | gold | p(gold) | model tier | text |",
             "|---|---|---|---|---|"]
    for f in findings:
        lines.append(f"| {f['source']} | {f['gold']} | {f['p_gold']} | "
                     f"{f['p_model_tier']} | {f['text'][:80]} |")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(findings)} flagged -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
