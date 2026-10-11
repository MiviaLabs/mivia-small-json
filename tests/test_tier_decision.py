"""Tests for the ordinal triage escalation policy.

`tool_match.tier_decision` maps a grade tier set to the agent-loop decision
(singleton -> act, adjacent pair -> act with annotation, 3+ -> ask).
`pipeline.triage_severity` composes it with the grade artifact and emits an
agent_state-compatible tool_clarification record when asking. The
agent-state composition test proves the ask record actually becomes a
pending clarification across turns.
"""
from __future__ import annotations

import pytest

from mivia_small_json import agent_state
from mivia_small_json.severity_head import CANONICAL_SEVERITY_ANCHORS
from mivia_small_json.tool_match import tier_decision

ANCHORS = [
    "info level: routine messages.",
    "warning level: recoverable conditions.",
    "error level: failures.",
    "critical level: outages.",
]

CANONICAL = list(CANONICAL_SEVERITY_ANCHORS)


class TestTierDecision:
    def test_singleton_acts(self):
        d = tier_decision([2], ANCHORS, argmax_index=2)
        assert d["action"] == "act"
        assert d["tier_index"] == 2
        assert d["tier"] == ANCHORS[2]
        assert "annotation" not in d

    def test_adjacent_pair_acts_with_annotation(self):
        d = tier_decision([1, 2], ANCHORS, argmax_index=2)
        assert d["action"] == "act"
        assert d["tier_index"] == 2
        assert d["tier"] == ANCHORS[2]
        assert d["annotation"] == f"between {ANCHORS[1]!r} and {ANCHORS[2]!r}"
        assert d["tier_set"] == [1, 2]

    def test_three_or_more_asks(self):
        d = tier_decision([0, 1, 2], ANCHORS, argmax_index=1)
        assert d["action"] == "ask"
        assert d["tier_set"] == [0, 1, 2]
        assert d["options"] == ANCHORS[:3]

    def test_full_set_asks(self):
        d = tier_decision([0, 1, 2, 3], ANCHORS, argmax_index=0)
        assert d["action"] == "ask"

    def test_never_raises_on_garbage(self):
        assert tier_decision([], ANCHORS, argmax_index=0)["action"] == "ask"
        # No anchors -> no tier to name -> the honest outcome is ask.
        assert tier_decision([1], [], argmax_index=1)["action"] == "ask"
        # An index past the anchor list cannot name a tier: ask, never raise.
        assert tier_decision([0, 5], ANCHORS, argmax_index=0)["action"] == "ask"
        assert tier_decision("bad", ANCHORS, argmax_index=None)["action"] == "ask"
        assert tier_decision(None, None)["action"] == "ask"


class TestTriageSeverity:
    def test_missing_artifact_degrades_to_ask(self, monkeypatch):
        from mivia_small_json import tool_match
        from mivia_small_json.pipeline import triage_severity

        # No artifact -> honest ask with a clarification record, never raise.
        monkeypatch.setattr(tool_match, "load_grade_matcher", lambda: None)
        d = triage_severity("any text", ANCHORS, matcher=None)
        assert d["action"] == "ask"
        assert d["clarification"]["tool"] == "severity_triage"

    def test_mocked_matcher_act_singleton(self):
        from mivia_small_json.pipeline import triage_severity

        class FakeMatcher:
            def grade(self, text, anchors, with_set=False, set_quantile=0.5):
                assert with_set is True
                return anchors[2], 0.97, [2]

        d = triage_severity("disk dead", ANCHORS, matcher=FakeMatcher())
        assert d["action"] == "act"
        assert d["tier_index"] == 2
        assert d["clarification"] is None

    def test_mocked_matcher_ask_emits_clarification_record(self):
        from mivia_small_json.pipeline import triage_severity

        class FakeMatcher:
            def grade(self, text, anchors, with_set=False, set_quantile=0.5):
                return anchors[1], 0.30, [0, 1, 2, 3]

        d = triage_severity("unclear", ANCHORS, matcher=FakeMatcher())
        assert d["action"] == "ask"
        rec = d["clarification"]
        assert rec["type"] == "tool_clarification"
        assert rec["tool"] == "severity_triage"
        assert rec["missing_parameters"] == ["severity"]

    def test_ask_record_becomes_pending_clarification(self):
        """The ask record composes with the agent loop: merging it into the
        rolling state leaves a pending clarification a later turn can
        resolve."""
        from mivia_small_json.pipeline import triage_severity

        class FakeMatcher:
            def grade(self, text, anchors, with_set=False, set_quantile=0.5):
                return anchors[1], 0.30, [0, 1, 2, 3]

        d = triage_severity("unclear", ANCHORS, matcher=FakeMatcher())
        state = agent_state.merge_step(agent_state.new_state(), d["clarification"])
        pend = agent_state.pending_clarification(state)
        assert pend is not None
        assert pend["tool"] == "severity_triage"

    def test_garbage_text_degrades_honestly(self):
        from mivia_small_json.pipeline import triage_severity

        class ExplodingMatcher:
            def grade(self, *a, **k):
                raise RuntimeError("boom")

        d = triage_severity(None, ANCHORS, matcher=ExplodingMatcher())
        assert d["action"] == "ask"


class TestTriageSeverityRouting:
    """The default (``matcher=None``) path picks a head from the anchor scale:
    the canonical severity scale uses the promoted ordinal head; arbitrary
    anchors keep the cross-encoder."""

    def test_canonical_scale_routes_through_the_ordinal_head(self, monkeypatch):
        from mivia_small_json import tool_match
        from mivia_small_json.pipeline import triage_severity

        seen = {}

        class FakeOrdinal:
            def grade(self, text, anchors, with_set=False, set_quantile=0.5):
                seen["text"] = text
                return anchors[2], 0.9, [2]

        def _no_grade(*a, **k):
            raise AssertionError("cross-encoder ran for the fixed scale")

        monkeypatch.setattr(tool_match, "load_severity_ordinal",
                            lambda *a, **k: FakeOrdinal())
        monkeypatch.setattr(tool_match, "load_grade_matcher", _no_grade)
        # The register router may legitimately route business-register texts
        # to the specialist; this test pins the shipped-head path with the
        # router disabled.
        monkeypatch.setattr(tool_match, "load_register_gate",
                            lambda: (None, None))
        # de promotion (2026-10-08): the lang gate sits before the register
        # gate; pin the en path explicitly so this test keeps testing the
        # canonical ordinal routing.
        monkeypatch.setattr(tool_match, "lang_is_de", lambda text: False)
        d = triage_severity("disk dead", CANONICAL, matcher=None)
        assert seen["text"] == "disk dead"
        assert d["action"] == "act"
        assert d["tier_index"] == 2
        assert d["tier"] == CANONICAL[2]

    def test_arbitrary_anchors_keep_the_cross_encoder(self, monkeypatch):
        from mivia_small_json import tool_match
        from mivia_small_json.pipeline import triage_severity

        seen = {}

        class FakeGrade:
            def grade(self, text, anchors, with_set=False, set_quantile=0.5):
                seen["text"] = text
                return anchors[1], 0.4, [1]

        def _no_ordinal(*a, **k):
            raise AssertionError("ordinal head ran for arbitrary anchors")

        monkeypatch.setattr(tool_match, "load_grade_matcher",
                            lambda *a, **k: FakeGrade())
        monkeypatch.setattr(tool_match, "load_severity_ordinal", _no_ordinal)
        d = triage_severity("disk dead", ANCHORS, matcher=None)
        assert seen["text"] == "disk dead"
        assert d["tier_index"] == 1

    def test_fixed_scale_falls_back_when_ordinal_head_is_missing(self, monkeypatch):
        from mivia_small_json import tool_match
        from mivia_small_json.pipeline import triage_severity

        class FakeGrade:
            def grade(self, text, anchors, with_set=False, set_quantile=0.5):
                return anchors[0], 0.5, [0]

        monkeypatch.setattr(tool_match, "load_severity_ordinal",
                            lambda *a, **k: None)
        monkeypatch.setattr(tool_match, "load_grade_matcher",
                            lambda *a, **k: FakeGrade())
        monkeypatch.setattr(tool_match, "load_register_gate",
                            lambda: (None, None))
        d = triage_severity("x", CANONICAL, matcher=None)
        assert d["action"] == "act"
        assert d["tier_index"] == 0

    def test_explicit_matcher_wins_over_the_scale(self):
        from mivia_small_json.pipeline import triage_severity

        class FakeMatcher:
            def grade(self, text, anchors, with_set=False, set_quantile=0.5):
                return anchors[3], 0.8, [3]

        d = triage_severity("x", CANONICAL, matcher=FakeMatcher())
        assert d["tier_index"] == 3


def test_live_fixed_scale_uses_the_promoted_ordinal_head():
    """Canonical anchors + the default loader route through the promoted
    ordinal head and act on a singleton tier."""
    from mivia_small_json.pipeline import triage_severity
    from mivia_small_json.tool_match import load_severity_ordinal

    if load_severity_ordinal() is None:
        pytest.skip("ordinal severity artifact not available")
    d = triage_severity(
        "[Mon Dec 05 18:20:51 2005] [notice] jk2_init() Found child 6670 in "
        "scoreboard slot 7", CANONICAL, matcher=None)
    assert d["action"] == "act"
    assert d["tier_set"] == [d["tier_index"]]


def test_live_triage_severity_confident_row_acts():
    """A pinned LogHub line grades confidently: the policy acts on a
    singleton without an annotation."""
    from mivia_small_json.pipeline import triage_severity
    from mivia_small_json.tool_match import load_grade_matcher

    matcher = load_grade_matcher()
    if matcher is None:
        pytest.skip("grade model artifact not available")

    anchors = [
        "info level: routine informational messages about normal successful "
        "operation and expected events.",
        "warning level: recoverable abnormal conditions that may need "
        "attention but are not failures.",
        "error level: operation failures, faults, and errors that disrupted "
        "a task or transaction.",
        "critical level: severe crashes, fatal errors, data loss, or "
        "complete system failure requiring immediate action.",
    ]
    d = triage_severity(
        "[Mon Dec 05 18:20:51 2005] [notice] jk2_init() Found child 6670 in "
        "scoreboard slot 7", anchors, matcher=matcher)
    assert d["action"] == "act"
    assert d["tier_index"] == 0
    assert "annotation" not in d


def test_live_triage_severity_ambiguous_row_annotates_or_asks():
    """A novel-register line (argmax wrong on the boundary) must NOT return
    a confident singleton: the policy annotates the pair or asks."""
    from mivia_small_json.pipeline import triage_severity
    from mivia_small_json.tool_match import load_grade_matcher

    matcher = load_grade_matcher()
    if matcher is None:
        pytest.skip("grade model artifact not available")

    anchors = [
        "info level: routine informational messages about normal successful "
        "operation and expected events.",
        "warning level: recoverable abnormal conditions that may need "
        "attention but are not failures.",
        "error level: operation failures, faults, and errors that disrupted "
        "a task or transaction.",
        "critical level: severe crashes, fatal errors, data loss, or "
        "complete system failure requiring immediate action.",
    ]
    d = triage_severity(
        "The backup restore failed and the primary ledger database is "
        "unrecoverable, we are declaring a disaster", anchors, matcher=matcher)
    assert d["action"] in ("act", "ask")
    if d["action"] == "act":
        assert "annotation" in d or len(d["tier_set"]) == 1
