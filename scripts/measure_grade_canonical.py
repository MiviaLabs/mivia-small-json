#!/usr/bin/env python3
"""THE canonical grade-severity measurement (2026-10-07 onward).

Measures the shipped severity path on the two canonical surfaces and
writes reports/grade-canonical-metrics.json, gated by
tests/test_grade_canonical_state.py:

- v3 (data/real/grade/business_slice_v3.jsonl, n=600 REAL tickets,
  agent gold, orchestrator blind-checked kappa 0.717): the business-
  register reality the product serves. This is the PRIMARY signal.
- panel (data/heldout/grade-register-panel.jsonl, n=217): the canonical-
  register control the shipped head was trained for.

Why v3 is primary and v1/v2 are demoted: v2's 56.3% sat inside the
shipped head's comfort zone and masked total middle-tier blindness on
real text (v3 tiers T1 1.5% / T2 1.7%). v1 (n=60) is ±26pp resolution.
Both remain frozen history, never a primary grade signal again.

The metrics file records artifact sha256s; the gate test fails if a
shipped artifact changes without this file being regenerated — no more
stale or flattering numbers.

CPU-only, read-only on shipped artifacts. Candidates (optional,
--candidate path=tag) are measured with the shipped register gate
(tau from models/register-gate-meta.json) exactly like production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json.stats import mcnemar_exact  # noqa: E402
from mivia_small_json.tool_match import (  # noqa: E402
    load_register_gate,
    load_severity_ordinal,
    load_severity_specialist,
)
from train_severity_ordinal import ordinal_metrics  # noqa: E402

V3 = ROOT / "data" / "real" / "grade" / "business_slice_v3.jsonl"
V3_RESERVE = ROOT / "data" / "real" / "grade" / "business_slice_v3.reserve.json"
PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
OUT = ROOT / "reports" / "grade-canonical-metrics.json"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiers(gold: list[int], pred: list[int]) -> dict[str, float]:
    out = {}
    for t in range(4):
        idx = [i for i, g in enumerate(gold) if g == t]
        out[str(t)] = round(sum(pred[i] == t for i in idx) / len(idx) * 100, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", default=[],
                    metavar="PATH=TAG",
                    help="optional candidate artifact measured through the "
                         "shipped register gate (repeatable)")
    args = ap.parse_args()

    # The eval sets must be exactly as reserved.
    v3_sha = _sha(V3)
    reserve = json.loads(V3_RESERVE.read_text())
    if not v3_sha.startswith(reserve["sha256"][:16]):
        raise SystemExit("v3 slice does not match its reserve — refusing")

    shipped = load_severity_ordinal()
    if shipped is None:
        raise SystemExit("shipped severity-ordinal artifact unavailable")

    v3 = _rows(V3)
    panel = _rows(PANEL)
    g3 = [r["gold"] for r in v3]
    gpn = [r["gold"] for r in panel]

    sp3 = [shipped.grade(r["text"], with_set=True)[2][0] for r in v3]
    spn = [shipped.grade(r["text"], with_set=True)[2][0] for r in panel]
    m3 = ordinal_metrics(g3, sp3)
    mpn = ordinal_metrics(gpn, spn)
    print("== SHIPPED ==")
    print(f"v3 (n=600, PRIMARY): exact {m3['exact']:.1f}% · within-1 "
          f"{m3['within_one']:.1f}% · QWK {m3['qwk']:.3f} · "
          f"tiers {_tiers(g3, sp3)}")
    print(f"panel (n=217, control): exact {mpn['exact']:.1f}% · within-1 "
          f"{mpn['within_one']:.1f}% · QWK {mpn['qwk']:.3f}")

    results = {
        "measured_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    __import__("time").gmtime()),
        "shipped": {
            "artifact": "models/severity-ordinal-int8.onnx",
            "sha256": _sha(ROOT / "models" / "severity-ordinal-int8.onnx"),
            "v3": {"exact": round(m3["exact"], 1),
                    "within_one": round(m3["within_one"], 1),
                    "qwk": round(m3["qwk"], 3),
                    "tiers": _tiers(g3, sp3)},
            "panel": {"exact": round(mpn["exact"], 1),
                       "within_one": round(mpn["within_one"], 1),
                       "qwk": round(mpn["qwk"], 3)},
        },
        "eval_sets": {
            "v3": {"n": len(v3), "sha256": v3_sha,
                    "tier_counts": {str(t): sum(1 for g in g3 if g == t)
                                     for t in range(4)},
                    "label_quality": "agent gold; orchestrator blind "
                                      "re-check 32/40, Cohen kappa 0.717"},
            "panel": {"n": len(panel)},
        },
        "candidates": {},
    }

    # The ROUTED production path (Arm D, handoff 2026-10-07): register gate
    # -> promoted specialist, canonical head otherwise. Recorded with
    # within-1 and QWK — exact/tiers alone hid the middle tiers for weeks.
    gate, tau = load_register_gate()
    specialist = load_severity_specialist()
    rpn = rv3 = None
    if (gate, tau) != (None, None):
        rpn = gate.predict_proba([r["text"] for r in panel])[:, 1] >= tau
        rv3 = gate.predict_proba([r["text"] for r in v3])[:, 1] >= tau
    if rv3 is not None and specialist is not None:
        pr3 = [(specialist if u else shipped).grade(r["text"], with_set=True)[2][0]
               for u, r in zip(rv3, v3)]
        prpn = [(specialist if u else shipped).grade(r["text"], with_set=True)[2][0]
                for u, r in zip(rpn, panel)]
        mpr3 = ordinal_metrics(g3, pr3)
        mprpn = ordinal_metrics(gpn, prpn)
        print("\n== PRODUCTION (routed) ==")
        print(f"v3 (n=600, PRIMARY): exact {mpr3['exact']:.1f}% · within-1 "
              f"{mpr3['within_one']:.1f}% · QWK {mpr3['qwk']:.3f} · "
              f"tiers {_tiers(g3, pr3)}")
        print(f"panel (n=217, control): exact {mprpn['exact']:.1f}% · within-1 "
              f"{mprpn['within_one']:.1f}% · QWK {mprpn['qwk']:.3f}")
        results["production"] = {
            "routing": "register-gate(tau) -> specialist, else head",
            "specialist": {"artifact":
                           "models/severity-ordinal-specialist-int8.onnx",
                           "sha256": _sha(ROOT / "models" /
                                          "severity-ordinal-specialist-int8.onnx")},
            "gate": {"artifact": "models/register-gate.skops",
                     "sha256": _sha(ROOT / "models" / "register-gate.skops")},
            "v3": {"exact": round(mpr3["exact"], 1),
                   "within_one": round(mpr3["within_one"], 1),
                   "qwk": round(mpr3["qwk"], 3),
                   "tiers": _tiers(g3, pr3)},
            "panel": {"exact": round(mprpn["exact"], 1),
                      "within_one": round(mprpn["within_one"], 1),
                      "qwk": round(mprpn["qwk"], 3)},
        }
    else:
        print("\n== PRODUCTION (routed) unavailable: register gate or "
              "specialist not loadable; head-only numbers above ==")

    for spec_str in args.candidate:
        path_s, tag = spec_str.split("=", 1)
        cand = load_severity_ordinal(ROOT / path_s)
        if cand is None:
            raise SystemExit(f"candidate unusable: {path_s}")
        if gate is None:
            p3 = [cand.grade(r["text"], with_set=True)[2][0] for r in v3]
            ppn = [cand.grade(r["text"], with_set=True)[2][0] for r in panel]
        else:
            p3 = [(cand if u else shipped).grade(r["text"], with_set=True)[2][0]
                  for u, r in zip(rv3, v3)]
            ppn = [(cand if u else shipped).grade(r["text"], with_set=True)[2][0]
                   for u, r in zip(rpn, panel)]
        mc3 = ordinal_metrics(g3, p3)
        mcpn = ordinal_metrics(gpn, ppn)
        b3 = sum(sp3[i] == g3[i] and p3[i] != g3[i] for i in range(len(v3)))
        c3 = sum(p3[i] == g3[i] and sp3[i] != g3[i] for i in range(len(v3)))
        bp = sum(spn[i] == gpn[i] and ppn[i] != gpn[i] for i in range(len(panel)))
        cp = sum(ppn[i] == gpn[i] and spn[i] != gpn[i] for i in range(len(panel)))
        t3 = _tiers(g3, p3)
        worst = min(t3[str(t)] - results["shipped"]["v3"]["tiers"][str(t)]
                    for t in range(4))
        print(f"\n== {tag} (routed via shipped gate) ==")
        print(f"v3: exact {mc3['exact']:.1f}% (Δ {mc3['exact'] - m3['exact']:+.1f}pp, "
              f"McNemar p={mcnemar_exact(b3, c3):.3g}) · tiers {t3} · "
              f"worst tier Δ {worst:+.1f}pp")
        print(f"panel: exact {mcpn['exact']:.1f}% "
              f"(p={mcnemar_exact(bp, cp):.3g} vs shipped)")
        results["candidates"][tag] = {
            "path": path_s, "sha256": _sha(ROOT / path_s),
            "v3": {"exact": round(mc3["exact"], 1),
                    "delta_pp": round(mc3["exact"] - m3["exact"], 1),
                    "mcnemar_p": round(mcnemar_exact(b3, c3), 8),
                    "tiers": t3, "worst_tier_delta_pp": worst},
            "panel": {"exact": round(mcpn["exact"], 1),
                       "mcnemar_p_vs_shipped": round(mcnemar_exact(bp, cp), 5)},
        }

    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nmetrics -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
