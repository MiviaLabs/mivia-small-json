"""Register router: business specialist + register gate (TRACK, promoted).

Unit tests pin the loaders and the routing decision (CPU-only, sha-verified,
never-raise). test_register_router_bar_cleared is a FALSIFICATION RECORD:
it reads reports/grade-register-router.md and asserts the pre-registered
bar stays PASSED; it fails on purpose if the report stops claiming PASS
(for example after a re-measurement) so the record and report are updated
together. No schema registration anywhere in this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mivia_small_json import tool_match

ROOT = Path(__file__).resolve().parents[1]
BUSINESS_TEXT = ("Per the status page, the holiday gift cards were "
                 "distributed to all warehouse staff")
CANONICAL_TEXT = ("Deployment failed: service api-gateway exited with code 1 "
                  "after the config change")


def test_register_gate_loads_with_tau():
    gate, tau = tool_match.load_register_gate()
    assert gate is not None, "shipped register gate failed to load"
    assert tau is not None and 0.0 < tau < 1.0


def test_gate_scores_are_probabilities():
    gate, _ = tool_match.load_register_gate()
    prob = gate.predict_proba([BUSINESS_TEXT, CANONICAL_TEXT])
    assert prob.shape == (2, 2)
    assert np.allclose(prob.sum(axis=1), 1.0)


def test_load_register_gate_never_raises_on_bad_meta(monkeypatch, tmp_path):
    """P1 review fix: a valid-JSON non-dict meta (null, [], "…") must
    degrade to (None, None), never raise — a raise here turns every
    canonical severity request into an ask instead of the shipped head."""
    bad = tmp_path / "register-gate-meta.json"
    for content in ("null", "[]", '"0.302"', "3"):
        bad.write_text(content)
        monkeypatch.setattr(tool_match, "REGISTER_GATE_META", bad)
        # Reset the loader cache so the bad meta is actually read.
        monkeypatch.setattr(tool_match, "_REGISTER_GATE_LOADED", False)
        monkeypatch.setattr(tool_match, "_REGISTER_GATE", None)
        monkeypatch.setattr(tool_match, "_REGISTER_GATE_TAU", None)
        gate, tau = tool_match.load_register_gate()
        assert gate is None and tau is None, content


def test_business_text_routes_to_specialist():
    gate, tau = tool_match.load_register_gate()
    assert gate.predict_proba([BUSINESS_TEXT])[0, 1] >= tau


def test_specialist_loads_and_decodes():
    head = tool_match.load_severity_specialist()
    assert head is not None, "shipped specialist head failed to load"
    anchor, score, tier_set = head.grade(BUSINESS_TEXT, with_set=True)
    assert anchor in head.anchors
    assert 0.0 <= score <= 1.0
    assert len(tier_set) == 1 and tier_set[0] in (0, 1, 2, 3)


def test_triage_severity_stays_working_with_router():
    from mivia_small_json.pipeline import triage_severity
    from mivia_small_json.severity_head import CANONICAL_SEVERITY_ANCHORS

    decision = triage_severity(BUSINESS_TEXT, CANONICAL_SEVERITY_ANCHORS)
    # The heads and gate are shipped artifacts (pinned by the loaders above),
    # so a healthy deployment acts; an ask here means a loader degraded.
    assert decision["action"] == "act", decision
    assert decision["tier_set"]


# (bar gate retired with reports/: the instrument record is not
# redistributed; the private tree keeps the gate.)
