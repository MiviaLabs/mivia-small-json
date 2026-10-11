#!/usr/bin/env python3
"""One-shot routed measurement of the LoRA arm vs the pre-registered bar.

Mirrors the orchestrator's prior arm measurements (register-router-*-metrics
.json) exactly: the shipped register gate (models/register-gate.skops via
load_register_gate, tau frozen at the dev-chosen value) routes every v2 and
panel row to the CANDIDATE (gate p >= tau) or the SHIPPED CORAL head;
ordinal metrics come from train_severity_ordinal.ordinal_metrics; exact
McNemar (mivia_small_json.stats.mcnemar_exact) runs against the shipped
head's own predictions on the same rows; per-tier exact is reported on v2.

The bar (reports/exp-severity-lora.md, written before the arm was built) is
quoted, never reinterpreted:

  PASS iff, measured ONCE on v2 (n=300) + panel (n=217):
  1. router exact >= 60.3% (shipped 56.3 + 4pp) with exact McNemar p < 0.05;
  2. no tier regresses > 5pp vs shipped (T0 85.3 / T1 38.7 / T2 40.0 /
     T3 61.3);
  3. panel exact >= 69.0%, and if < 70.0% the McNemar p vs shipped > 0.05.

CPU-only, read-only on models/; candidates stay under data/incoming/models/.
Writes data/incoming/models/register-router-lora-metrics.json.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json.stats import mcnemar_exact  # noqa: E402
from mivia_small_json.tool_match import (  # noqa: E402
    load_register_gate,
    load_severity_ordinal,
)
from train_severity_ordinal import ordinal_metrics  # noqa: E402

V2 = ROOT / "data" / "real" / "grade" / "business_slice_v2.jsonl"
PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
CANDIDATES = {
    "s42": ROOT / "data" / "incoming" / "models"
           / "severity-ordinal-lora-s42-int8.onnx",
    "s43": ROOT / "data" / "incoming" / "models"
           / "severity-ordinal-lora-s43-int8.onnx",
}
OUT = ROOT / "data" / "incoming" / "models" / "register-router-lora-metrics.json"

# The pre-registered bar, verbatim numbers from reports/exp-severity-lora.md.
BAR = {
    "v2_exact_min": 60.3,
    "v2_mcnemar_p_max": 0.05,
    "tier_regression_pp_max": 5.0,
    "shipped_v2_tiers": [85.3, 38.7, 40.0, 61.3],
    "panel_exact_floor": 69.0,
    "panel_exact_significance_free": 70.0,
    "panel_mcnemar_p_min_when_below_70": 0.05,
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tier(head, text: str) -> int:
    return head.grade(text, with_set=True)[2][0]


def _per_tier(golds: list[int], preds: list[int], k: int = 4) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in range(k):
        idx = [i for i, g in enumerate(golds) if g == t]
        out[str(t)] = round(100.0 * sum(preds[i] == t for i in idx) / len(idx),
                            1) if idx else 0.0
    return out


def _mcnemar(golds: list[int], shipped: list[int],
             candidate: list[int]) -> dict:
    # b: shipped right, candidate wrong; c: candidate right, shipped wrong.
    b = sum(shipped[i] == golds[i] and candidate[i] != golds[i]
            for i in range(len(golds)))
    c = sum(candidate[i] == golds[i] and shipped[i] != golds[i]
            for i in range(len(golds)))
    return {"b": int(b), "c": int(c), "p": round(mcnemar_exact(b, c), 5)}


def main() -> int:
    gate, tau = load_register_gate()
    if gate is None or tau is None:
        raise SystemExit("shipped register gate unavailable")
    shipped = load_severity_ordinal()
    if shipped is None:
        raise SystemExit("shipped severity-ordinal artifact unavailable")

    v2 = _rows(V2)
    panel = _rows(PANEL)
    texts = [r["text"] for r in v2 + panel]

    classes = list(gate.classes_)
    if 1 not in classes:
        raise SystemExit("register gate has no positive class 1")
    prob = gate.predict_proba(texts)[:, classes.index(1)]
    routed = (prob >= tau).astype(bool)
    v2_routed, panel_routed = routed[:len(v2)], routed[len(v2):]

    v2_gold = [r["gold"] for r in v2]
    panel_gold = [r["gold"] for r in panel]
    # The comparator: the SHIPPED head's own predictions on every row.
    v2_ship = [_tier(shipped, r["text"]) for r in v2]
    panel_ship = [_tier(shipped, r["text"]) for r in panel]
    shipped_v2 = ordinal_metrics(v2_gold, v2_ship)
    shipped_v2_tiers = _per_tier(v2_gold, v2_ship)
    shipped_panel = ordinal_metrics(panel_gold, panel_ship)

    results: dict = {
        "tau": round(float(tau), 6),
        "bar": BAR,
        "measured_once": str(date.today()),
        "shipped_baseline": {
            "v2_exact": round(shipped_v2["exact"], 1),
            "v2_within1": round(shipped_v2["within_one"], 1),
            "v2_tiers": shipped_v2_tiers,
            "panel_exact": round(shipped_panel["exact"], 1),
        },
        "routed_counts": {
            "v2_to_candidate": int(v2_routed.sum()),
            "panel_to_candidate": int(panel_routed.sum()),
            "v2_n": len(v2),
            "panel_n": len(panel),
        },
    }

    for seed, path in CANDIDATES.items():
        if not path.is_file():
            print(f"== {seed}: artifact missing, skipped ({path.name})")
            continue
        cand = load_severity_ordinal(path)
        if cand is None:
            raise SystemExit(f"candidate artifact unusable: {path.name}")

        v2_pred = [_tier(cand if r else shipped, t)
                   for r, t in zip(v2_routed, texts[:len(v2)])]
        panel_pred = [_tier(cand if r else shipped, t)
                      for r, t in zip(panel_routed, texts[len(v2):])]

        vm = ordinal_metrics(v2_gold, v2_pred)
        v2_tiers = _per_tier(v2_gold, v2_pred)
        v2_mc = _mcnemar(v2_gold, v2_ship, v2_pred)
        pm = ordinal_metrics(panel_gold, panel_pred)
        panel_mc = _mcnemar(panel_gold, panel_ship, panel_pred)

        clauses = {
            "1_v2_exact_ge_60.3": vm["exact"] >= BAR["v2_exact_min"],
            "1_mcnemar_p_lt_0.05": v2_mc["p"] < BAR["v2_mcnemar_p_max"],
            "2_no_tier_regresses_gt_5pp": all(
                v2_tiers[str(t)] >= BAR["shipped_v2_tiers"][t]
                - BAR["tier_regression_pp_max"] for t in range(4)),
            "3_panel_ge_69.0": pm["exact"] >= BAR["panel_exact_floor"],
            "3_if_below_70_p_gt_0.05": (
                pm["exact"] >= BAR["panel_exact_significance_free"]
                or panel_mc["p"] > BAR["panel_mcnemar_p_min_when_below_70"]),
        }
        verdict = all(clauses.values())

        tier_str = " / ".join(
            f"{v2_tiers[str(t)]:.1f} ({v2_tiers[str(t)] - shipped_v2_tiers[str(t)]:+.1f})"
            for t in range(4))
        print(f"\n== {seed} ({path.name}) · tau {tau:.4f} · "
              f"routed v2 {int(v2_routed.sum())}/{len(v2)} · "
              f"panel {int(panel_routed.sum())}/{len(panel)} ==")
        print(f"v2: exact {vm['exact']:.1f}% (bar >= 60.3, shipped "
              f"{shipped_v2['exact']:.1f}) · within-one {vm['within_one']:.1f}% · "
              f"McNemar b={v2_mc['b']} c={v2_mc['c']} p={v2_mc['p']:.5f}")
        print(f"v2 tiers vs shipped {list(shipped_v2_tiers.values())}: {tier_str}")
        print(f"panel: exact {pm['exact']:.1f}% (floor 69.0, shipped "
              f"{shipped_panel['exact']:.1f}) · within-one {pm['within_one']:.1f}% · "
              f"McNemar b={panel_mc['b']} c={panel_mc['c']} p={panel_mc['p']:.5f}")
        print("clauses: " + " · ".join(
            f"{k}={'PASS' if v else 'FAIL'}" for k, v in clauses.items()))
        print(f"verdict: {'PASS' if verdict else 'FAIL'}")

        results[seed] = {
            "artifact": str(path.relative_to(ROOT)),
            "v2_exact": round(vm["exact"], 1),
            "v2_within1": round(vm["within_one"], 1),
            "v2_tiers": v2_tiers,
            "v2_tiers_delta_vs_shipped": {
                str(t): round(v2_tiers[str(t)] - shipped_v2_tiers[str(t)], 1)
                for t in range(4)},
            "v2_mcnemar": v2_mc,
            "panel_exact": round(pm["exact"], 1),
            "panel_within1": round(pm["within_one"], 1),
            "panel_mcnemar": panel_mc,
            "bar_clauses": {k: bool(v) for k, v in clauses.items()},
            "verdict": "PASS" if verdict else "FAIL",
        }

    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nmetrics -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
