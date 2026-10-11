"""The router's 0.85 fast path vs the trained ``text`` class and user intent.

``pipeline._route_with_confidence`` short-circuits when the model
prediction is a registered schema and max-softmax confidence is >= 0.85
(``routed_by == "model"``). Max-softmax is overconfident off the training
manifold, so the shipped contract has two outrankers:

- a CONFIDENT ``text`` prediction emits the text record before the fast
  path (the class itself never fast-paths);
- a user-registered schema whose STRONG marker matches the text outranks
  both — explicit user intent is what the schema was installed for.

The energy gate itself is superseded on the shipped router (meta sentinel
``-100.0``); its history and the committed-set rejection numbers live in
tests/test_energy_gate_ood.py.

These tests run the shipped ONNX encoder end to end; they skip when the
artifact (or its guard) is absent, i.e. a rules/skops-only checkout.
"""
from __future__ import annotations

import pytest

from mivia_small_json import pipeline, registry, tool_match
from mivia_small_json.encoder import load_encoder

SYSLOG = "2026-03-01T12:00:00Z ERROR [auth]: token expired"
# The invoice from tests/test_dynamic_hybrid_routing.py. Out of domain: under
# the pre-gate encoder a retrain pushed it to ``issue`` ~0.997 and it
# fast-pathed; the shipped encoder scores it 0.725 but still ``issue``.
INVOICE = ("Invoice #INV-2026-991: Payment of $450 for cloud consulting "
           "services due on 2026-11-01")


@pytest.fixture(scope="module")
def encoder():
    enc = load_encoder()
    if enc is None:
        pytest.skip("shipped ONNX encoder absent; rules/skops-only checkout")
    return enc


@pytest.fixture
def clean_registry():
    """Empty custom registry, restored to the suite state on exit."""
    with registry.temporary_schemas():
        registry._reset()
        yield


@pytest.fixture
def guard_available():
    if tool_match._get_guard() is None:
        pytest.skip("cross-encoder guard absent; zero-shot fallback disabled")


def _register_billing_invoice():
    """A described non-router schema, so it arms the zero-shot fallback."""
    registry.register_schema(
        name="billing_invoice",
        spec={"type": {"const": "billing_invoice"},
              "amount": {"type": int},
              "invoice_id": {"type": str}},
        routing={"strong": [r"\bINV-\d+\b"], "weak": [r"\binvoice\b"]},
        extraction={"amount": [r"\$(\d+)"],
                    "invoice_id": [r"(INV-\d+-\d+)"]},
        defaults={"amount": 0, "invoice_id": "INV-000"},
        description=("Process accounting invoice with payment dollar amount "
                     "and due date"),
    )


def test_syslog_fast_paths_as_log_event(encoder):
    """In-domain traffic is preserved: the syslog keeps the model route.

    The gate constant is the shipped sentinel (-100.0), so the energy
    comparison holds trivially; the load-bearing asserts are the route
    itself (``routed_by == "model"`` fails if the fast path is removed)
    and the confidence floor.
    """
    assert encoder.in_domain_scores([SYSLOG])[0] >= encoder.in_domain_energy
    schema, routed_by, conf = pipeline._route_with_confidence(encoder, SYSLOG)
    assert (schema, routed_by) == ("log_event", "model")
    assert conf >= 0.85


def test_user_intent_outranks_the_confident_router(
        encoder, clean_registry, guard_available):
    """A strong-marker registered schema wins over the router's own call.

    The shipped router scores the invoice ``issue`` (see the energy-gate
    file for the anchors); the registered billing_invoice's strong marker
    is what the user installed the schema for, so it outranks the router.
    """
    _register_billing_invoice()
    obj, meta = pipeline.classify(INVOICE, router=encoder)
    assert obj["type"] == "billing_invoice"
    assert meta["routed_by"] == "model"


def test_class_and_user_intent_decide_for_a_confident_invoice(
        encoder, clean_registry, guard_available):
    """The post-gate contract (gate superseded 2026-10-10): the trained
    text class is what non-workflow prose hits with NO user
    schema registered; with a strong-marker schema registered, explicit
    user intent outranks both the fast path and the OOD emission.
    """
    from mivia_small_json.bootstrap import register_text_schema

    register_text_schema()
    # The class's load-bearing rejection anchor on the shipped router:
    # non-workflow prose at ~0.998.
    news = ("The quarterback threw three touchdown passes in the fourth "
            "quarter as the home crowd erupted.")
    schema, routed_by, conf = pipeline._route_with_confidence(encoder, news)
    assert (schema, routed_by) == ("text", "model")
    assert conf >= 0.85

    # User intent still outranks the router: the same invoice-shaped text
    # carries the registered schema's strong marker AND the semantic cue
    # the cross-encoder verifies on.
    _register_billing_invoice()
    invoice = ("Invoice INV-2026-0042 — Acme Corporation — 40 hrs "
               "consulting @ $150/hr — Subtotal $7,200 — Tax 8.25% — "
               "Total Due $7,794.00")
    obj, meta = pipeline.classify(invoice, router=encoder)
    assert obj["type"] == "billing_invoice"
    assert meta["routed_by"] == "model"
