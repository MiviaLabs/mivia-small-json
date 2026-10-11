"""Unit tests for CascadedToolMatcher (Fast Gate to Deep Escalation).

Verifies:
- Fast path short-circuiting on clear matches and clear rejections
- Escalation path on ambiguous scores within ambiguity_band
- Escalation path on long context inputs
- Graceful degradation when Tier 2 is None
- Full support for select, verify, grade primitives on cascaded matcher
"""
from __future__ import annotations

import pytest

from mivia_small_json.tool_match import (
    BaseToolMatcher,
    CascadedToolMatcher,
)


class MockMatcher(BaseToolMatcher):
    def __init__(self, score_val: float):
        self.score_val = score_val
        self.calls = 0

    def _score(self, request: str, tool: str) -> float:
        self.calls += 1
        return self.score_val

    def _score_raw(self, request: str, tool: str) -> float:
        self.calls += 1
        return self.score_val


def test_cascaded_fast_path_clear_match():
    # Score 0.85 is above 0.65 -> Tier 2 must not be called
    m1 = MockMatcher(0.85)
    m2 = MockMatcher(0.99)
    cascaded = CascadedToolMatcher(m1, m2, ambiguity_band=(0.35, 0.65))

    score = cascaded._score("user request", "tool: description")
    assert score == 0.85
    assert m1.calls == 1
    assert m2.calls == 0


def test_cascaded_fast_path_clear_rejection():
    # Score 0.10 is below 0.35 -> Tier 2 must not be called
    m1 = MockMatcher(0.10)
    m2 = MockMatcher(0.01)
    cascaded = CascadedToolMatcher(m1, m2, ambiguity_band=(0.35, 0.65))

    score = cascaded._score("user request", "tool: description")
    assert score == 0.10
    assert m1.calls == 1
    assert m2.calls == 0


def test_cascaded_escalation_in_ambiguity_band():
    # Score 0.50 is inside [0.35, 0.65] -> Tier 2 must be called and override
    m1 = MockMatcher(0.50)
    m2 = MockMatcher(0.95)
    cascaded = CascadedToolMatcher(m1, m2, ambiguity_band=(0.35, 0.65))

    score = cascaded._score("user request", "tool: description")
    assert score == 0.95
    assert m1.calls == 1
    assert m2.calls == 1


def test_cascaded_escalation_long_context():
    # Clear score 0.90, but text exceeds length_threshold (512 tokens ~= 2048 chars)
    m1 = MockMatcher(0.90)
    m2 = MockMatcher(0.99)
    cascaded = CascadedToolMatcher(m1, m2, length_threshold=50)

    long_req = "a" * 300  # 300 // 4 = 75 tokens > 50
    score = cascaded._score(long_req, "tool")
    assert score == 0.99
    assert m1.calls == 1
    assert m2.calls == 1


def test_cascaded_graceful_fallback_tier2_none():
    m1 = MockMatcher(0.50)
    cascaded = CascadedToolMatcher(m1, None, ambiguity_band=(0.35, 0.65))

    score = cascaded._score("user request", "tool: description")
    assert score == 0.50
    assert m1.calls == 1


@pytest.mark.parametrize(
    "tier1_score,expected_escalation",
    [
        (0.349, False),
        (0.350, True),
        (0.500, True),
        (0.650, True),
        (0.651, False),
    ],
)
def test_cascaded_ambiguity_band_boundaries(tier1_score, expected_escalation):
    m1 = MockMatcher(tier1_score)
    m2 = MockMatcher(0.99)
    cascaded = CascadedToolMatcher(m1, m2, ambiguity_band=(0.35, 0.65))

    should = cascaded.should_escalate(tier1_score, "req", "tool")
    assert should is expected_escalation

    score = cascaded._score("req", "tool")
    if expected_escalation:
        assert score == 0.99
        assert m2.calls == 1
    else:
        assert score == tier1_score
        assert m2.calls == 0


@pytest.mark.parametrize(
    "token_count,expected_escalation",
    [
        (49, False),
        (50, False),
        (51, True),
    ],
)
def test_cascaded_length_threshold_boundaries(token_count, expected_escalation):
    m1 = MockMatcher(0.90)  # Outside ambiguity band
    m2 = MockMatcher(0.99)
    cascaded = CascadedToolMatcher(m1, m2, length_threshold=50)

    # 1 token ~= 4 chars; length = token_count * 4
    req = "x" * (token_count * 4)
    should = cascaded.should_escalate(0.90, req, "")
    assert should is expected_escalation


def test_cascaded_property_delegation():
    class DummyWithProps(BaseToolMatcher):
        def __init__(self):
            self.tok = "mock_tok"
            self.sess = "mock_sess"
            self.inputs = {"input_ids", "attention_mask"}

        def _score(self, request: str, tool: str) -> float:
            return 0.5

    t1 = DummyWithProps()
    cascaded = CascadedToolMatcher(t1)
    assert cascaded.tok == "mock_tok"
    assert cascaded.sess == "mock_sess"
    assert cascaded.inputs == {"input_ids", "attention_mask"}


class DifferentiatedMockMatcher(BaseToolMatcher):
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def _score(self, request: str, tool: str) -> float:
        for k, v in self.scores.items():
            if k in tool:
                return v
        return 0.5

    def _score_raw(self, request: str, tool: str) -> float:
        return self._score(request, tool)


def test_cascaded_primitives_integration():
    m1 = DifferentiatedMockMatcher({"opt1": 0.10, "opt2": 0.50, "low": 0.20, "high": 0.50})
    m2 = DifferentiatedMockMatcher({"opt1": 0.10, "opt2": 0.95, "low": 0.20, "high": 0.88})
    cascaded = CascadedToolMatcher(m1, m2, ambiguity_band=(0.35, 0.65))

    # verify (0.50 escalates to 0.88)
    accepted, score = cascaded.verify("request", "high")
    assert accepted is True
    assert score == 0.88

    # select (opt2 escalates to 0.95, winning clearly)
    candidates = {"opt1": "desc opt1", "opt2": "desc opt2"}
    choice, conf = cascaded.select("request", candidates)
    assert choice == "opt2"
    assert conf > 0.60

    # grade (high escalates to 0.88, winning clearly over low)
    best_anchor, g_score = cascaded.grade("request", ["low", "high"])
    assert best_anchor == "high"
    assert g_score == 0.88
