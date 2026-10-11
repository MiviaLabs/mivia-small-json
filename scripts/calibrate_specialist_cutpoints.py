#!/usr/bin/env python3
"""Fit per-cutpoint bias deltas for the business specialist severity head.

Motivation (reports/grade-register-router.md, tier-rebalance arm): CORAL
training-prior shifts move ALL cutpoints together — the documented seesaw.
Post-hoc cutpoint adjustment is the ordinal analog of binary threshold
moving and the CORAL authors' own remedy for imbalance
(corral-pytorch docs; arXiv:1901.07884 §3.2.3; Vcerajec et al., Pattern
Recognition 137:109303, 2023). This script fits one delta per cumulative
logit on the dedicated CALIBRATION DEV split only
(`data/real/grade/business_calib_dev.jsonl`, 80 rows, reserved) and never
touches the encoder, the shipped head, the v2 slice, or the panel.

Decode contract: tier = count of positive entries of (logit + delta).
Monotonicity is enforced by requiring delta_0 >= delta_1 >= delta_2
(adding delta_k to cumulative logit k equals lowering cutpoint k; the
deltas' ordering therefore preserves the cutpoints' ordering).

Objective (fixed before the grid): macro-balanced exact accuracy — the
mean of per-tier accuracy on the calibration split.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.tool_match import load_severity_specialist  # noqa: E402

CALIB = ROOT / "data" / "real" / "grade" / "business_calib_dev.jsonl"
CALIB_RESERVE = ROOT / "data" / "real" / "grade" / "business_calib_dev.reserve.json"
OUT = ROOT / "data" / "incoming" / "models" / "specialist-cutpoint-deltas.json"

STEP = 0.1
BOUND = 1.5
K = 4  # tiers


def macro_balanced(gold: list[int], pred: list[int]) -> float:
    per_tier = []
    for tier in range(K):
        idx = [i for i, g in enumerate(gold) if g == tier]
        per_tier.append(sum(pred[i] == gold[i] for i in idx) / len(idx))
    return sum(per_tier) / K


def main() -> int:
    # Guard: the calibration split must be used exactly as reserved.
    digest = hashlib.sha256(CALIB.read_bytes()).hexdigest()
    reserve = json.loads(CALIB_RESERVE.read_text())
    if not digest.startswith(reserve["sha256"][:16]):
        raise SystemExit("calibration split does not match its reserve")

    head = load_severity_specialist()
    if head is None:
        raise SystemExit("specialist head unavailable")

    rows = [json.loads(line) for line in CALIB.read_text().splitlines()
            if line.strip()]
    gold = [r["gold"] for r in rows]
    logits = [head.logits(r["text"]) for r in rows]

    grid = [round(-BOUND + i * STEP, 1)
            for i in range(int(2 * BOUND / STEP) + 1)]
    best = None  # (objective, deltas)
    n_evals = 0
    for d0, d1, d2 in itertools.product(grid, repeat=3):
        if not (d0 >= d1 >= d2):  # monotonicity-preserving
            continue
        n_evals += 1
        pred = [sum(1 for k, z in enumerate(lg) if z + (d0, d1, d2)[k] > 0.0)
                for lg in logits]
        obj = macro_balanced(gold, pred)
        if best is None or obj > best[0]:
            best = (obj, (d0, d1, d2))
    (obj, deltas) = best

    baseline_pred = [sum(1 for z in lg if z > 0.0) for lg in logits]
    baseline_obj = macro_balanced(gold, baseline_pred)
    print(f"grid: {n_evals} monotone combos, step {STEP}, bound ±{BOUND}")
    print(f"calibration macro-balanced exact: baseline {baseline_obj:.4f} "
          f"-> fitted {obj:.4f} at deltas {deltas}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "deltas": list(deltas),
        "objective": "macro_balanced_exact",
        "calibration_macro_balanced": round(obj, 4),
        "baseline_macro_balanced": round(baseline_obj, 4),
        "calibration_split": "business_calib_dev.jsonl",
        "calibration_sha256": digest[:16],
        "grid": {"step": STEP, "bound": BOUND, "monotone": True},
        "fitted_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "head": "models/severity-ordinal-specialist-int8.onnx",
    }, indent=2) + "\n")
    print(f"deltas -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
