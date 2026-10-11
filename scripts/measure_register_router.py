#!/usr/bin/env python3
"""Router measurement vs the pre-registered bar (register-router track).

Decision rule: register-gate p(business) >= tau -> specialist head,
else shipped CORAL head. Exactly one head serves each input.

Judges the pre-registered bar from grade-register-router:
1. frozen business slice (n=60): exact >= 50% AND within-one >= 80%;
2. frozen register panel (n=217): exact >= 69.0% AND, if exact < 70.0%,
   McNemar exact p vs shipped > 0.05.

CPU-only, read-only on the shipped artifact. Candidates stay under
data/incoming/models/; tau comes from the gate's dev-chosen meta file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json.stats import mcnemar_exact  # noqa: E402
from mivia_small_json.tool_match import load_severity_ordinal  # noqa: E402
from mivia_small_json.stats import bca_ci, mde  # noqa: E402
from train_severity_ordinal import ordinal_metrics  # noqa: E402

GATE = ROOT / "data" / "incoming" / "models" / "register-gate.skops"
GATE_META = ROOT / "data" / "incoming" / "models" / "register-gate.meta.json"
SLICE = ROOT / "data" / "real" / "grade" / "business_slice.jsonl"
PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
SPECIALISTS = {
    "s42": ROOT / "data" / "incoming" / "models"
           / "severity-ordinal-specialist-s42.onnx",
    "s43": ROOT / "data" / "incoming" / "models"
           / "severity-ordinal-specialist-s43.onnx",
}
BAR = {"slice_exact": 50.0, "slice_within_one": 80.0,
       "panel_exact_floor": 69.0, "shipped_panel_exact": 70.0, "p": 0.05}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def load_gate():
    import skops.io as sio

    untrusted = sio.get_untrusted_types(file=str(GATE))
    pipe = sio.load(str(GATE), trusted=untrusted)
    meta = json.loads(GATE_META.read_text())
    return pipe, meta["tau_dev_chosen"]


def main() -> int:
    gate, tau = load_gate()
    shipped = load_severity_ordinal()
    if shipped is None:
        raise SystemExit("shipped severity-ordinal artifact unavailable")

    slice_rows = _rows(SLICE)
    panel_rows = _rows(PANEL)

    # Route every text once through the gate.
    all_texts = [r["text"] for r in slice_rows + panel_rows]
    classes = list(gate.classes_)
    pos_col = classes.index(1) if 1 in classes else None
    if pos_col is None:
        raise SystemExit("register gate has no positive class 1")
    prob = gate.predict_proba(all_texts)[:, pos_col]
    routed = (prob >= tau).astype(bool)
    n_slice = len(slice_rows)
    slice_routed, panel_routed = routed[:n_slice], routed[n_slice:]

    results: dict = {"tau": tau, "per_seed": {}}
    for seed, path in SPECIALISTS.items():
        if not path.is_file():
            continue
        specialist = load_severity_ordinal(path)
        if specialist is None:
            raise SystemExit(f"specialist artifact unusable: {path.name}")

        def head_tier(head, text: str) -> int:
            return head.grade(text, with_set=True)[2][0]

        # Business slice, routed.
        slice_pred = [head_tier(specialist if r else shipped, t)
                      for r, t in zip(slice_routed,
                                      (r["text"] for r in slice_rows))]
        slice_gold = [r["gold"] for r in slice_rows]
        sm = ordinal_metrics(slice_gold, slice_pred)

        # Panel, routed + McNemar vs shipped.
        panel_pred = [head_tier(specialist if r else shipped, t)
                      for r, t in zip(panel_routed,
                                      (r["text"] for r in panel_rows))]
        panel_gold = [r["gold"] for r in panel_rows]
        pm = ordinal_metrics(panel_gold, panel_pred)
        shipped_panel = [shipped.grade(r["text"], with_set=True)[2][0]
                         for r in panel_rows]
        b = sum(shipped_panel[i] == panel_gold[i] and panel_pred[i] != panel_gold[i]
                for i in range(len(panel_rows)))
        c = sum(panel_pred[i] == panel_gold[i] and shipped_panel[i] != panel_gold[i]
                for i in range(len(panel_rows)))
        p = mcnemar_exact(b, c)

        misrouted_to_specialist = int(panel_routed.sum())
        misrouted_to_shipped = int((~slice_routed).sum())

        passed = (sm["exact"] >= BAR["slice_exact"]
                  and sm["within_one"] >= BAR["slice_within_one"]
                  and pm["exact"] >= BAR["panel_exact_floor"]
                  and (pm["exact"] >= BAR["shipped_panel_exact"] or p > BAR["p"]))
        # Small-n honesty: BCa CI on the paired exact-difference and the
        # minimum detectable effect at this n (review: our bars have been
        # demanding effects below the test's reliable resolution).
        slice_diff = [1 if pred == g else 0 for pred, g in zip(slice_pred, slice_gold)]
        ship_slice = [1 if sh == g else 0 for sh, g in
                      zip((shipped.grade(r["text"], with_set=True)[2][0]
                           for r in slice_rows), slice_gold)]
        paired = [s - c for s, c in zip(ship_slice, slice_diff)]
        ci = bca_ci(paired, statistic=lambda v: sum(v) / len(v), seed=7)
        mde_slice = mde(len(slice_gold), sm["exact"] / 100)
        mde_panel = mde(len(panel_gold), pm["exact"] / 100)
        print(f"slice paired-exact diff BCa 95% CI: "
              f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]pp · MDE(n=60) "
              f"±{mde_slice * 100:.0f}pp · MDE(n=217) ±{mde_panel * 100:.0f}pp")
        print(f"\n== {seed} (tau {tau:.4f}) ==")
        print(f"panel routed to specialist: {misrouted_to_specialist}/{len(panel_rows)} · "
              f"slice kept on shipped: {misrouted_to_shipped}/{len(slice_rows)}")
        print(f"slice: exact {sm['exact']:.1f}% (bar >= 50) · within-one "
              f"{sm['within_one']:.1f}% (bar >= 80)")
        print(f"panel: exact {pm['exact']:.1f}% (floor 69.0) · within-one "
              f"{pm['within_one']:.1f}% · McNemar b={b} c={c} p={p:.4f}")
        print(f"verdict: {'PASS' if passed else 'FAIL'}")
        results["per_seed"][seed] = {
            "slice_diff_bca95": [round(ci[0], 3), round(ci[1], 3)],
            "mde": {"slice": round(mde_slice, 3), "panel": round(mde_panel, 3)},
            "slice": {k: round(sm[k], 1) for k in
                      ("exact", "within_one", "aod")},
            "panel": {k: round(pm[k], 1) for k in
                      ("exact", "within_one", "aod")},
            "mcnemar": {"b": b, "c": c, "p": round(p, 4)},
            "routed": {"panel_to_specialist": misrouted_to_specialist,
                        "slice_kept_shipped": misrouted_to_shipped},
            "verdict": "PASS" if passed else "FAIL",
        }

    results["bar"] = BAR
    results["verdict"] = {s: ev["verdict"] for s, ev in results["per_seed"].items()}
    # Committed location: the falsification record reads this file, and it
    # must exist on a clean clone (review finding #5).
    out = ROOT / "reports" / "register-router-metrics.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\noverall: {results['verdict']} · metrics -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
