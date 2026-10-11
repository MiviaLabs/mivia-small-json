"""Integration tests for hybrid dynamic schema routing.

Tests:
1. High-confidence built-in events bypass dynamic zero-shot evaluation.
2. Custom user-registered schemas route correctly via zero-shot cross-encoder matching.
3. Irrelevant text safely falls back to standard routing without falsely claiming custom schema.
4. Custom schema outputs pass schema validation and repair.
"""
from __future__ import annotations

import numpy as np
import pytest

from mivia_small_json import pipeline, registry, tool_match
from mivia_small_json.encoder import load_encoder


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


def _register_minimal(name, description=None):
    registry.register_schema(
        name,
        {"type": {"const": name}, "title": {"type": str}},
        {"strong": [], "weak": []},
        {"title": [r"title:\s*(.*)"]},
        {"title": ""},
        description=(f"{name} semantic description"
                     if description is None else description),
    )


def test_fallback_not_armed_when_every_custom_schema_is_a_router_class(
        monkeypatch, empty_custom_schemas):
    """A registered schema the router can already emit must not arm the
    zero-shot fallback: re-scoring it against its own description costs
    an encoder run per candidate plus a verify, and the fine-tuned router
    already covers it."""
    guard = _RecordingGuard()
    monkeypatch.setattr(tool_match, "_get_guard", lambda: guard)
    router = _FakeRouter(["issue", "commit", "pull_request"], [0.5, 0.3, 0.2])
    _register_minimal("pull_request")

    schema, routed_by, _ = pipeline._route_with_confidence(router, "some text")

    assert schema == "issue" and routed_by == "model"
    assert guard.select_calls == [], (
        "fallback ran for a schema the router already covers")
    assert guard.verify_calls == []


def test_fallback_armed_for_custom_schema_outside_router_classes(
        monkeypatch, empty_custom_schemas):
    """A schema the router cannot emit is still a zero-shot candidate."""
    guard = _RecordingGuard()
    monkeypatch.setattr(tool_match, "_get_guard", lambda: guard)
    router = _FakeRouter(["issue", "commit"], [0.5, 0.3])
    _register_minimal("billing_invoice")

    pipeline._route_with_confidence(router, "some text")

    assert guard.select_calls == [["billing_invoice"]]


def test_fallback_not_armed_for_schema_without_a_semantic_description(
        monkeypatch, empty_custom_schemas):
    """A registered schema the router cannot emit still needs a semantic
    description to be zero-shot selectable. The auto-generated field list
    is not one: selecting on it re-scores the input against field names.
    This is the shipped ``tool_clarification`` case — it is guard-derived,
    never routed from free text, and as a lone candidate it overrode
    ``tool_call`` on 51/100 BFCL calls."""
    guard = _RecordingGuard()
    monkeypatch.setattr(tool_match, "_get_guard", lambda: guard)
    router = _FakeRouter(["issue", "commit"], [0.5, 0.3])
    _register_minimal("tool_clarification", description="")

    pipeline._route_with_confidence(router, "some text")

    assert guard.select_calls == []


def test_fallback_armed_when_router_exposes_no_class_labels(
        monkeypatch, empty_custom_schemas):
    """Without a class set the router's coverage is unknown: keep every
    custom schema a candidate (the pre-fix behaviour)."""

    class _ProbaRouterNoClasses:
        def predict_proba(self, texts):
            return [np.array([0.5, 0.3, 0.2])]

        def predict(self, texts):
            return ["issue"]

    guard = _RecordingGuard()
    monkeypatch.setattr(tool_match, "_get_guard", lambda: guard)
    _register_minimal("billing_invoice")

    pipeline._route_with_confidence(_ProbaRouterNoClasses(), "some text")

    assert guard.select_calls == [["billing_invoice"]]


def test_fallback_armed_when_a_pipeline_router_exposes_an_empty_class_list(
        monkeypatch, empty_custom_schemas):
    """A ``named_steps`` router whose ``clf`` has no ``classes_`` exposes
    an empty label set. Empty is "coverage unknown", not "covers
    nothing": dropping every candidate here would silently disable the
    fallback (and indexing the empty list would crash the route)."""

    class _PipelineRouterNoClasses:
        named_steps = {"clf": object()}

        def predict_proba(self, texts):
            return [np.array([0.5, 0.3, 0.2])]

        def predict(self, texts):
            return ["issue"]

    guard = _RecordingGuard()
    monkeypatch.setattr(tool_match, "_get_guard", lambda: guard)
    _register_minimal("billing_invoice")

    pipeline._route_with_confidence(_PipelineRouterNoClasses(), "some text")

    assert guard.select_calls == [["billing_invoice"]]


@pytest.fixture
def clean_custom_schemas():
    with registry.temporary_schemas():
        yield


@pytest.fixture
def empty_custom_schemas():
    """An empty custom registry, restored to the suite's state after.

    ``tests/conftest.py`` registers ``pull_request`` and ``agent_step``
    for the whole suite, so ``temporary_schemas`` alone leaves them in
    place. The fallback tests register a schema under a name the suite
    already holds and assert on the exact candidate set, so they need the
    empty world; ``_reset`` is the documented test-isolation hook and the
    context manager puts the suite's schemas back on exit.
    """
    with registry.temporary_schemas():
        registry._reset()
        yield


def test_high_confidence_builtin_bypasses_dynamic_evaluation(clean_custom_schemas):
    router = load_encoder()
    text = (
        "commit 1a2b3c4d5e\n"
        "Author: Alice <alice@example.com>\n"
        "Fix issue with login redirect loop\n\n"
        "2 files changed, 34 insertions(+), 12 deletions(-)\n"
        "Date: 2026-10-06"
    )
    # Register a competing custom schema
    registry.register_schema(
        name="custom_ticket",
        spec={
            "type": {"const": "custom_ticket"},
            "title": {"type": str},
        },
        routing={"strong": [r"\bcustom_ticket\b"], "weak": [r"\bticket\b"]},
        extraction={"title": [r"title:\s*(.*)"]},
        defaults={"title": "untitled"},
        description="A customer ticket or bug report with error traceback",
    )

    obj, meta = pipeline.classify(text, router=router)
    # Built-in commit routes with >= 0.85 confidence and stays commit
    assert obj["type"] == "commit"
    assert meta["routed_by"] == "model"


def test_custom_user_registered_schema_routes_zero_shot(clean_custom_schemas):
    router = load_encoder()

    registry.register_schema(
        name="billing_invoice",
        spec={
            "type": {"const": "billing_invoice"},
            "amount": {"type": int},
            "invoice_id": {"type": str},
        },
        routing={"strong": [r"\bINV-\d+\b"], "weak": [r"\binvoice\b"]},
        extraction={
            "amount": [r"\$(\d+)"],
            "invoice_id": [r"(INV-\d+-\d+)"],
        },
        defaults={"amount": 0, "invoice_id": "INV-000"},
        description="Process accounting invoice with payment dollar amount and due date",
    )

    invoice_text = (
        "Invoice #INV-2026-991: Payment of $450 for cloud consulting services due on 2026-11-01"
    )

    obj, meta = pipeline.classify(invoice_text, router=router)
    assert obj["type"] == "billing_invoice"
    assert obj["amount"] == 450
    assert obj["invoice_id"] == "INV-2026-991"
    assert meta["valid"] is True
    assert meta["routed_by"] == "model"


def test_irrelevant_text_does_not_falsely_route_to_custom_schema(clean_custom_schemas):
    router = load_encoder()

    registry.register_schema(
        name="billing_invoice",
        spec={
            "type": {"const": "billing_invoice"},
            "amount": {"type": int},
        },
        routing={"strong": [r"\binvoice_id:\b"], "weak": [r"\binvoice\b"]},
        extraction={"amount": [r"\$(\d+)"]},
        defaults={"amount": 0},
        description="Accounting invoice with dollar amount and payment line items",
    )

    # Git commit text
    commit_text = (
        "commit 9f8e7d6c5b4a\n"
        "Author: Bob <bob@example.com>\n\n"
        "refactor: clean up logging utility"
    )

    obj, meta = pipeline.classify(commit_text, router=router)
    assert obj["type"] != "billing_invoice"
    assert obj["type"] == "commit"
