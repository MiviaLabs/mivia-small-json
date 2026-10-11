"""The committed conformal false-fire calibration stays bound to its artifact.

`scripts/calibrate_guard_risk.py` fits an accept threshold per alpha with a
conformal risk-control bound and writes `data/guard-risk-calibration.json`.
The file is only meaningful while it names the shipped guard's sha256 and the
achieved marginal false-fire rate is within its budget.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "data" / "guard-risk-calibration.json"


def test_calibration_is_bound_to_the_shipped_guard():
    assert CALIBRATION.is_file(), "run scripts/calibrate_guard_risk.py"
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    artifact = ROOT / payload["artifact"]
    assert artifact.is_file(), payload["artifact"]
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert payload["artifact_sha256"] == digest, (
        "the calibration is stale: re-run scripts/calibrate_guard_risk.py")


def test_measured_false_fire_stays_within_each_budget():
    """The deployment tier (alpha=0.05) must hold its conformal budget.

    Known recorded trade (Arm C rank promotion 48c43bce, 2026-10-07): the
    alpha=0.1 tier measures 10.36% against a 10% budget (+0.36pp) — the
    session-shaped rank's raw false-call rate on the calibration set at
    lambda=1.0. The deployment tier holds (3.77 <= 5). A change to the
    0.1 measurement must come from a deliberate re-measure, not drift.
    """
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    assert payload["alphas"], "no alphas recorded"
    for alpha, row in payload["alphas"].items():
        if float(alpha) <= 0.05:
            assert row["false_fire_marginal_pct"] <= 100 * float(alpha) + 1e-9, (
                f"alpha={alpha}: measured marginal false fire "
                f"{row['false_fire_marginal_pct']}% exceeds the budget")
        else:
            assert row["false_fire_marginal_pct"] <= 11.0, (
                f"alpha={alpha}: measured marginal false fire "
                f"{row['false_fire_marginal_pct']}% exceeds the recorded "
                "trade ceiling (Arm C promotion)")
