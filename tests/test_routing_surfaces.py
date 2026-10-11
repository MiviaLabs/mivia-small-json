"""Pins for the router decision surfaces in ``pipeline._route_with_confidence``.

These are the branches the suite's routing tests do not touch:

1. The ``pred in registry.active() and conf >= 0.85`` fast path returns
   ``(pred, "model", conf)`` and never arms the zero-shot fallback.
2. ``_LOW_CONFIDENCE`` (0.40): below it a strong rules signal outranks the
   model; at/above it the model prediction stands.
3. ``_STRONG_RULE_SCORE`` (4): the raw-score override fires at 4, not below.
4. ``BaseToolMatcher.verify``'s default 0.50 threshold vs the guard's
   ``tool_match.THRESHOLD`` (0.6) abstention bar.

Everything runs on fake routers (numpy ``predict_proba``) and fake guards:
no artifacts load, so the file is fast. ``registry.temporary_schemas()``
plus ``registry._reset()`` isolate the registry (conftest registers
``pull_request``/``agent_step`` suite-wide). Each test names the assertion
that breaks if its constant or branch changes.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mivia_small_json import pipeline, registry, rules, tool_match


class _FakeRouter:
    """Fixed class set and fixed probability vector; no real inference."""

    def __init__(self, classes, probs):
        self.classes = list(classes)
        self._probs = np.array(probs)

    def predict_proba(self, texts):
        return [self._probs for _ in texts]


class _RecordingGuard:
    """Stands in for the cross-encoder guard and records fallback calls."""

    def __init__(self):
        self.select_calls = []
        self.verify_calls = []

    def select(self, text, candidates, with_status=False):
        self.select_calls.append(sorted(candidates))
        return "", 0.0

    def verify(self, text, hypothesis, threshold=0.50):
        self.verify_calls.append(hypothesis)
        return False, 0.0


def _register_described(name):
    """A zero-shot-selectable custom schema (explicit description)."""
    registry.register_schema(
        name,
        {"type": {"const": name}, "title": {"type": str}},
        {"strong": [], "weak": []},
        {"title": [r"title:\s*(.*)"]},
        {"title": ""},
        description=f"{name} semantic description",
    )


@pytest.fixture
def empty_registry():
    """An empty custom registry, restored to the suite's state after.

    ``_reset`` drops the suite's ``pull_request``/``agent_step`` so the
    fallback candidate set is exactly what a test registers, and the
    context manager puts them back on exit.
    """
    with registry.temporary_schemas():
        registry._reset()
        yield


# --- 1. the conf >= 0.85 fast path -----------------------------------------


def test_fast_path_returns_model_and_does_not_arm_fallback(
        monkeypatch, empty_registry):
    """At conf 0.85 (the inclusive boundary) a covered prediction short-
    circuits before the zero-shot block.

    Load-bearing: ``guard.select_calls == []`` fails if the fast-path
    guard is removed or its threshold raised above 0.85 — the
    non-covered ``billing_invoice`` would then be scored. ``conf >= 0.85``
    (inclusive, not ``>``) is pinned because 0.85 is the chosen conf.
    """
    guard = _RecordingGuard()
    monkeypatch.setattr(tool_match, "_get_guard", lambda: guard)
    _register_described("billing_invoice")  # not a router class: arms fallback
    router = _FakeRouter(["issue", "commit"], [0.85, 0.10])

    schema, routed_by, conf = pipeline._route_with_confidence(router, "some text")

    assert (schema, routed_by) == ("issue", "model")
    assert conf == pytest.approx(0.85)
    assert guard.select_calls == [], (
        "fast path must not arm the zero-shot fallback")
    assert guard.verify_calls == []


def test_just_below_threshold_reaches_fallback(monkeypatch, empty_registry):
    """At conf 0.849 the fast path is skipped and the non-covered candidate
    is scored.

    Load-bearing: ``guard.select_calls == [["billing_invoice"]]`` fails if
    the fast-path threshold drops to 0.849 or below (the candidate would be
    filtered out), pinning ``>= 0.85`` from below.
    """
    guard = _RecordingGuard()
    monkeypatch.setattr(tool_match, "_get_guard", lambda: guard)
    _register_described("billing_invoice")
    router = _FakeRouter(["issue", "commit"], [0.849, 0.10])

    schema, routed_by, conf = pipeline._route_with_confidence(router, "some text")

    assert guard.select_calls == [["billing_invoice"]]
    assert (schema, routed_by) == ("issue", "model")
    assert conf == pytest.approx(0.849)


# --- 2. _LOW_CONFIDENCE = 0.40 ---------------------------------------------


def test_conf_at_low_confidence_boundary_model_wins(
        monkeypatch, empty_registry):
    """At conf exactly 0.40 the model prediction stands even with a strong
    rules signal available (the comparison is strict ``<``).

    Load-bearing: ``routed_by == "model"`` fails if ``_LOW_CONFIDENCE`` is
    raised above 0.40 (the low-confidence block would run and rules would
    win), pinning the constant at or below 0.40.
    """
    monkeypatch.setattr(rules, "route_scores", lambda text: {"issue": 0, "commit": 5})
    monkeypatch.setattr(rules, "route", lambda text: "commit")
    router = _FakeRouter(["issue", "commit"], [0.40, 0.30])

    schema, routed_by, conf = pipeline._route_with_confidence(router, "some text")

    assert (schema, routed_by) == ("issue", "model")
    assert conf == pytest.approx(0.40)


def test_conf_just_below_low_confidence_rules_win(
        monkeypatch, empty_registry):
    """At conf 0.399 (< 0.40) a rules score >= _STRONG_RULE_SCORE outranks
    the model.

    Load-bearing: ``routed_by == "rules"`` fails if ``_LOW_CONFIDENCE`` is
    lowered to 0.399 or below (the block would be skipped and the model
    would win), pinning the constant above 0.399. Together with the
    boundary test this pins ``_LOW_CONFIDENCE == 0.40``.
    """
    monkeypatch.setattr(rules, "route_scores", lambda text: {"issue": 0, "commit": 5})
    monkeypatch.setattr(rules, "route", lambda text: "commit")
    router = _FakeRouter(["issue", "commit"], [0.399, 0.30])

    schema, routed_by, conf = pipeline._route_with_confidence(router, "some text")

    assert routed_by == "rules"
    assert schema == "commit"
    assert conf == pytest.approx(1.0)


# --- 3. _STRONG_RULE_SCORE = 4 ---------------------------------------------


def test_rule_score_below_strong_threshold_does_not_override(
        monkeypatch, empty_registry):
    """A best rules score of 3 does not fire the raw-score override; the
    decision falls through the ``best > 0`` path (stubbed to a non-active
    name) and the model prediction stands.

    Load-bearing: ``routed_by == "model"`` fails if ``_STRONG_RULE_SCORE``
    drops to 3 or below (``best >= threshold`` would return the raw max,
    ``issue``, as ``"rules"``), pinning the constant above 3.
    """
    monkeypatch.setattr(rules, "route_scores", lambda text: {"issue": 3, "commit": 0})
    monkeypatch.setattr(rules, "route", lambda text: "ghost")
    router = _FakeRouter(["issue", "commit"], [0.35, 0.30])

    schema, routed_by, conf = pipeline._route_with_confidence(router, "some text")

    assert (schema, routed_by) == ("issue", "model")
    assert conf == pytest.approx(0.35)


def test_rule_score_at_strong_threshold_overrides(
        monkeypatch, empty_registry):
    """A best rules score of 4 fires the raw-score override: the top-scoring
    schema wins with conf 1.0.

    Load-bearing: ``routed_by == "rules"`` and ``schema == "commit"`` fail
    if ``_STRONG_RULE_SCORE`` rises above 4 (the fall-through path, stubbed
    to a non-active name, would yield the model's ``issue``). Together with
    the below-threshold test this pins ``_STRONG_RULE_SCORE == 4``.
    """
    monkeypatch.setattr(rules, "route_scores", lambda text: {"issue": 0, "commit": 4})
    monkeypatch.setattr(rules, "route", lambda text: "ghost")
    router = _FakeRouter(["issue", "commit"], [0.35, 0.30])

    schema, routed_by, conf = pipeline._route_with_confidence(router, "some text")

    assert routed_by == "rules"
    assert schema == "commit"
    assert conf == pytest.approx(1.0)


# --- 4. verify default 0.50 vs guard THRESHOLD 0.6 --------------------------


def test_verify_default_threshold_differs_from_guard_threshold():
    """``verify`` accepts at 0.55 by default while the guard's abstention
    bar is 0.6 — a deliberate gap (guard: "is this a call at all?",
    verify: "does this text entail this description?").

    Load-bearing: the two constant asserts fail if either value moves; the
    ``accepted is True`` assert fails if the default threshold rises above
    0.55; ``score < tool_match.THRESHOLD`` fails if the two thresholds are
    ever collapsed to one value.
    """
    assert tool_match.THRESHOLD == 0.6

    default = inspect.signature(
        tool_match.BaseToolMatcher.verify).parameters["threshold"].default
    assert default == 0.50

    class _ConstScoreMatcher(tool_match.BaseToolMatcher):
        def _score(self, request, tool):
            return 0.55

    accepted, score = _ConstScoreMatcher().verify("text", "hypothesis")
    assert accepted is True
    assert score == pytest.approx(0.55)
    # 0.55 clears verify's 0.50 but sits below the guard's 0.6 bar.
    assert score < tool_match.THRESHOLD
