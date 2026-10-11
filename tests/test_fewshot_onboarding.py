"""Few-shot schema onboarding: contract tests (spec
few-shot-onboarding, TDD red-first).

The feature: ``register_schema(..., examples=[...], fewshot=True)``
trains a per-schema head (shipped encoder embeddings + small sklearn
models) so the schema's extraction fills the fields the deterministic
ladder leaves open. Honesty law: learned enum values are declared
members only; learned str/int values must appear verbatim in the text
or the field is refused.

Bar notes: F3 (onboarding time) and F4 (RSS) are MEASURED by the
scratch instrument, never timing-asserted here (Governing rule 14 /
mechanical-gates law). The exact-match floors below sit under the
measured numbers with slack; the F1 gate itself (>= 90% mean over the
5 real domains at 16 examples/class) lives in the committed instrument
report, not in a test.
"""

from __future__ import annotations

import random

import pytest

from mivia_small_json import fewshot, registry
from mivia_small_json.validate import validate

pytestmark = pytest.mark.usefixtures("fewshot_dir")


PRODUCTS = [
    "mechanical keyboard", "usb-c dock", "noise cancelling headset",
    "ergonomic mouse", "laptop stand", "monitor arm", "cable organizer",
    "webcam cover",
]
COURIERS = ["DHL", "UPS", "FedEx", "Royal Mail"]
ACTIONS = ["place", "cancel", "modify", "track"]
PRIORITIES = ["normal", "rush", "economy"]

_PRIORITY_PHRASE = {
    "rush": ["This is extremely urgent, please expedite the shipment "
             "immediately.", "Very urgent: expedite handling right away.",
             "Emergency speed needed, expedite this shipment now."],
    "normal": ["Standard delivery is fine, no rush at all, take your time.",
               "No rush whatsoever, regular shipping speed is perfectly ok.",
               "Totally relaxed timing, standard shipping is just fine."],
    "economy": ["Use the cheapest slowest economy option, cost matters more "
                "than speed.", "Budget economy post please, saving money "
                "beats saving time.", "Cheapest possible shipping, this one "
                "can take as long as it needs."],
}


def _order_spec(name: str) -> dict:
    return {
        "type": {"const": name},
        "action": {"type": str, "enum": ACTIONS},
        "product": {"type": str},
        "quantity": {"type": int},
        "priority": {"type": str, "enum": PRIORITIES},
        "courier": {"type": str, "optional": True},
    }


_ORDER_ROUTING = {"strong": [r"\border\b"], "weak": [r"\b(ship|delivery)\b"]}


def _order_text(rng: random.Random, action: str) -> tuple[str, dict]:
    """One generated order text + its gold fields (mechanical labels).

    Gold fields carry no "type": the caller stamps the registered name."""
    product = rng.choice(PRODUCTS)
    qty = rng.randint(1, 9)
    priority = rng.choice(PRIORITIES)
    phrase = rng.choice(_PRIORITY_PHRASE[priority])
    fields: dict = {"action": action, "product": product,
                    "quantity": qty, "priority": priority}
    if action == "place":
        courier = rng.choice(COURIERS)
        text = (f"Please place an order for {qty} x {product}. {phrase} "
                f"Ship via {courier}.")
        fields["courier"] = courier
    elif action == "cancel":
        text = (f"Please cancel my order for the {product}, quantity {qty}. "
                f"{phrase}")
        if rng.random() < 0.5:
            courier = rng.choice(COURIERS)
            text += f" It was shipped by {courier}."
            fields["courier"] = courier
    elif action == "modify":
        courier = rng.choice(COURIERS)
        text = (f"Change my order of the {product} to {qty} units. {phrase} "
                f"Carrier: {courier}.")
        fields["courier"] = courier
    else:  # track
        courier = rng.choice(COURIERS)
        text = (f"Where is my order of {qty} x the {product}? {phrase} "
                f"It ships via {courier}.")
        fields["courier"] = courier
    return text, fields


def _order_corpus(name: str, seed: int = 42, per_class: int = 16,
                  eval_per_class: int = 40):
    """(onboarding examples with _text, eval pool) — disjoint by construction."""
    rng = random.Random(seed)
    examples, pool = [], []
    for action in ACTIONS:
        made = 0
        while made < per_class + eval_per_class:
            text, fields = _order_text(rng, action)
            if made < per_class:
                examples.append({"_text": text, "type": name, **fields})
            else:
                pool.append((text, fields))
            made += 1
    return examples, pool


def _register_order(name: str, examples, **kwargs):
    return registry.register_schema(name, _order_spec(name), _ORDER_ROUTING,
                                    {}, examples=examples, **kwargs)


def _clean(examples):
    return [{k: v for k, v in ex.items() if k != "_text"} for ex in examples]


def _exact_match(schema: str, pool) -> float:
    """Fraction of pool records whose every non-type field matches gold."""
    hits = 0
    for text, gold in pool:
        obj, _meta = fewshot.extract_via_schema(text, schema)
        obj = dict(obj)
        obj.pop("type", None)
        want = {k: v for k, v in gold.items() if k != "type"}
        if obj == want:
            hits += 1
    return hits / max(1, len(pool))


@pytest.fixture
def fewshot_dir(tmp_path, monkeypatch):
    """Isolated runtime dir + clean head cache per test."""
    monkeypatch.setenv("MIVIA_FEWSHOT_DIR", str(tmp_path / "fewshot"))
    fewshot._HEADS.clear()
    yield tmp_path / "fewshot"
    fewshot._HEADS.clear()


def _encoder_available() -> bool:
    return fewshot.encoder() is not None


# ---------------------------------------------------------------------------
# F5: the feature is strictly additive.


def test_register_without_fewshot_wires_nothing_and_returns_none(fewshot_dir):
    examples, pool = _order_corpus("plain_order", per_class=2,
                                   eval_per_class=3)
    report = _register_order("plain_order", _clean(examples))
    assert report is None  # today's contract, byte-identical
    assert fewshot.status("plain_order") is None
    # Extraction is EXACTLY the deterministic pattern engine (which fills
    # nothing here) — no learned field ever appears.
    obj, _ = fewshot.extract_via_schema(
        "Please place an order for 3 x mechanical keyboard. Ship via DHL.",
        "plain_order")
    assert obj == {"type": "plain_order"}


def test_spec_fingerprint_unchanged_by_onboarding(fewshot_dir):
    """Onboarding/forgetting a head never moves the spec fingerprint.

    (Registering a schema moves it, as it does today — that is the
    existing registration gate's surface, not this feature's.)"""
    from mivia_small_json.registry import spec_fingerprint

    examples, _ = _order_corpus("fp_order", per_class=2, eval_per_class=2)
    _register_order("fp_order", examples)  # feature off at registration
    before = spec_fingerprint()
    report = fewshot.onboard("fp_order")
    assert report["trained"] is True
    assert spec_fingerprint() == before
    fewshot.forget("fp_order")
    assert spec_fingerprint() == before


def test_register_fewshot_true_without_examples_reports_untrained(fewshot_dir):
    report = registry.register_schema(
        "noex_order", _order_spec("noex_order"), _ORDER_ROUTING, {},
        examples=None, fewshot=True)
    assert report is not None
    assert report["trained"] is False
    assert "reason" in report
    assert fewshot.head_for("noex_order") is None


# ---------------------------------------------------------------------------
# The TDD red-first contract: the synthetic 6-field order domain.


def test_order_domain_onboarding_exact_match(fewshot_dir):
    if not _encoder_available():
        pytest.skip("shipped sentence encoder not available")
    examples, pool = _order_corpus("order_fewshot")
    baseline_examples, baseline_pool = _order_corpus("order_baseline")
    assert [t for t, _ in baseline_pool] == [t for t, _ in pool]
    # The deterministic-only baseline on the SAME pool is poor: today the
    # schema has no extraction patterns, so every learned-capable field is
    # unfilled (the red state this feature exists for).
    _register_order("order_baseline", baseline_examples)
    baseline = _exact_match("order_baseline", pool)
    assert baseline < 0.5

    report = _register_order("order_fewshot", examples, fewshot=True)
    assert report["schema"] == "order_fewshot"
    assert report["trained"] is True
    assert report["n_examples"] == len(examples)
    # THE CONTRACT: >= 0.90 exact on held-out rows at 16 examples/class.
    exact = _exact_match("order_fewshot", pool)
    assert exact >= 0.90, f"order-domain exact {exact:.3f} < 0.90"


def test_enum_head_only_emits_declared_members(fewshot_dir):
    if not _encoder_available():
        pytest.skip("shipped sentence encoder not available")
    examples, _ = _order_corpus("enum_only", per_class=4, eval_per_class=2)
    _register_order("enum_only", examples, fewshot=True)
    head = fewshot.head_for("enum_only")
    assert head is not None
    for _ in range(20):
        member = head.predict_member(
            "Please cancel my order for the laptop stand, quantity 2.",
            "action")
        assert member in ACTIONS
        member = head.predict_member("Where is my order of 3 x usb-c dock?",
                                     "action")
        assert member in ACTIONS


def test_verbatim_refusal(fewshot_dir):
    """A predicted value that is NOT a verbatim span of the text is refused."""
    if not _encoder_available():
        pytest.skip("shipped sentence encoder not available")
    examples, _ = _order_corpus("verbatim_order", per_class=4,
                                eval_per_class=2)
    _register_order("verbatim_order", examples, fewshot=True)
    head = fewshot.head_for("verbatim_order")

    text = "Please place an order for 3 x mechanical keyboard. Ship via DHL."

    class _Liar(type(head)):
        def predict_span(self, text, field):
            return "invented product name"

    liar = _Liar(head.schema, head.spec, head.meta,
                 head.enum_heads, head.taggers)
    filled = liar.fill_missing(text, {"type": "verbatim_order"})
    assert "product" not in filled  # refused: not verbatim in the text
    # The honest head, by contrast, fills a verbatim span.
    filled = head.fill_missing(text, {"type": "verbatim_order"})
    if "product" in filled:
        assert filled["product"] in text


def test_list_fields_abstain_and_single_member_trains_nothing(fewshot_dir):
    """List-typed fields are never learned; an enum field with a single
    observed member trains no head — no invented values anywhere."""
    if not _encoder_available():
        pytest.skip("shipped sentence encoder not available")
    spec = {
        "type": {"const": "listing"},
        "tags": {"type": list, "optional": True},
        "action": {"type": str, "enum": ACTIONS},
    }
    examples = [{"_text": _order_text(random.Random(i), "place")[0],
                 "type": "listing", "action": "place"}
                for i in range(6)]
    registry.register_schema("listing", spec, {"strong": [], "weak": []}, {},
                             examples=examples, fewshot=True)
    lhead = fewshot.head_for("listing")
    assert lhead is not None
    assert "tags" not in lhead.taggers
    assert "action" not in lhead.enum_heads  # one member = not a task
    filled = lhead.fill_missing("order", {"type": "listing"})
    assert "tags" not in filled and "action" not in filled


# ---------------------------------------------------------------------------
# Lifecycle: persistence, attach, unregister, determinism.


def test_unregister_cleans_artifacts(fewshot_dir):
    if not _encoder_available():
        pytest.skip("shipped sentence encoder not available")
    examples, _ = _order_corpus("doomed_order", per_class=2, eval_per_class=1)
    _register_order("doomed_order", examples, fewshot=True)
    artifacts = sorted(p.name for p in fewshot_dir.iterdir())
    assert any("doomed_order" in name for name in artifacts)
    assert fewshot.head_for("doomed_order") is not None
    registry.unregister("doomed_order")
    assert fewshot.head_for("doomed_order") is None
    assert fewshot.status("doomed_order") is None
    leftover = [p.name for p in fewshot_dir.iterdir()
                if "doomed_order" in p.name]
    assert leftover == []


def test_attach_from_disk_resume(fewshot_dir):
    if not _encoder_available():
        pytest.skip("shipped sentence encoder not available")
    examples, _ = _order_corpus("resume_order", per_class=2, eval_per_class=1)
    _register_order("resume_order", examples, fewshot=True)
    first = fewshot.head_for("resume_order")
    assert first is not None
    fewshot._HEADS.clear()  # simulate a fresh process
    assert fewshot.head_for("resume_order") is None  # no silent resurrection
    assert fewshot.attach("resume_order") is True
    second = fewshot.head_for("resume_order")
    assert second is not None
    a = first.fill_missing("Where is my order of 2 x usb-c dock?",
                           {"type": "resume_order"})
    b = second.fill_missing("Where is my order of 2 x usb-c dock?",
                            {"type": "resume_order"})
    assert a == b


def test_onboard_twice_content_identical(fewshot_dir):
    """Determinism gate (content-level): refitting twice yields bit-equal
    coefficients/classes/meta. Artifact BYTES are excluded on purpose:
    skops dumps carry __id__ memory addresses in schema.json (demonstrated
    2026-10-10), so bytes are not a function of the inputs."""
    if not _encoder_available():
        pytest.skip("shipped sentence encoder not available")
    import numpy as np
    import skops.io as sio

    examples, _ = _order_corpus("det_order", per_class=4, eval_per_class=1)
    _register_order("det_order", examples, fewshot=True)
    path = fewshot_dir / "det_order.skops"
    c1 = sio.load(path, trusted=sio.get_untrusted_types(file=path))
    fewshot.onboard("det_order")
    c2 = sio.load(path, trusted=sio.get_untrusted_types(file=path))
    assert c1["meta"] == c2["meta"]
    for field in c1["enum"]:
        a, b = c1["enum"][field]["clf"], c2["enum"][field]["clf"]
        assert np.array_equal(a.coef_, b.coef_)
        assert list(a.classes_) == list(b.classes_)
    for field in c1["tag"]:
        a, b = c1["tag"][field]["clf"], c2["tag"][field]["clf"]
        assert np.array_equal(a.coef_, b.coef_)
    # And the refit head predicts identically on a probe battery.
    probes = [
        "Please place an order for 3 x mechanical keyboard. Ship via DHL.",
        "Where is my order of 2 x usb-c dock?",
        "Change my order of the monitor arm to 5 units.",
    ]
    first = fewshot.head_for("det_order")
    before = [dict(first.fill_missing(p, {"type": "det_order"}))
              for p in probes]
    fewshot.onboard("det_order")  # refit
    refit = fewshot.head_for("det_order")
    after = [dict(refit.fill_missing(p, {"type": "det_order"}))
             for p in probes]
    assert before == after


def test_fewshot_true_without_text_pairs_degrades_honestly(fewshot_dir):
    examples, _ = _order_corpus("nopairs_order", per_class=2,
                                eval_per_class=1)
    report = _register_order("nopairs_order", _clean(examples), fewshot=True)
    assert report["trained"] is False
    assert "reason" in report
    assert fewshot.head_for("nopairs_order") is None


def test_degrades_without_encoder(fewshot_dir, monkeypatch):
    examples, _ = _order_corpus("noenc_order", per_class=2, eval_per_class=1)
    monkeypatch.setattr(fewshot, "encoder", lambda: None)
    report = _register_order("noenc_order", examples, fewshot=True)
    assert report["trained"] is False
    assert "reason" in report
    # Registration survived; extraction stays purely deterministic.
    assert "noenc_order" in registry.active()
    obj, _ = fewshot.extract_via_schema("order", "noenc_order")
    assert obj == {"type": "noenc_order"}


# ---------------------------------------------------------------------------
# Composition: deterministic first, aliases still apply, explicit extractor
# stays authoritative.


def test_deterministic_first_and_aliases_still_apply(fewshot_dir):
    spec = {
        "type": {"const": "aliased_ticket"},
        "channel": {"type": str,
                    "enum": ["email", "chat"],
                    "aliases": {"email": ["mail", "e-mail"]}},
        "note": {"type": str},
    }
    # The pattern FILLS channel via the declared alias; the learned head
    # fills what the patterns leave open. Both must compose.
    extraction = {"channel": [r"via (mail|e-mail|email|chat)\b"]}
    examples = [
        {"_text": "sent via mail about the flaky deploy job",
         "type": "aliased_ticket", "channel": "email",
         "note": "flaky deploy job"}
        for _ in range(4)
    ] + [
        {"_text": "sent via chat about the broken build job",
         "type": "aliased_ticket", "channel": "chat",
         "note": "broken build job"}
        for _ in range(4)
    ]
    registry.register_schema("aliased_ticket", spec,
                             {"strong": [r"\bticket\b"], "weak": []},
                             extraction, examples=examples, fewshot=True)
    obj, _ = fewshot.extract_via_schema(
        "sent via mail about the flaky deploy job", "aliased_ticket")
    # Alias coercion from the deterministic ladder is untouched...
    assert obj["channel"] == "email"
    # ...and the learned note (no pattern covers it) is a verbatim span.
    assert obj.get("note", "") in "sent via mail about the flaky deploy job"


def test_explicit_extractor_stays_authoritative(fewshot_dir):
    examples, _ = _order_corpus("mine_order", per_class=2, eval_per_class=1)
    seen = []

    def _mine(text):
        seen.append(text)
        return {"type": "mine_order", "product": "fixed"}

    registry.register_schema(
        "mine_order", _order_spec("mine_order"), _ORDER_ROUTING, {},
        examples=examples, extractor=_mine, fewshot=True)
    obj, _ = fewshot.extract_via_schema(
        "Please place an order for 3 x mechanical keyboard.", "mine_order")
    assert obj == {"type": "mine_order", "product": "fixed"}
    assert seen  # the user extractor ran, the learned head never composed


def test_inline_text_key_is_stripped_from_stored_examples(fewshot_dir):
    examples, _ = _order_corpus("clean_order", per_class=1, eval_per_class=1)
    _register_order("clean_order", examples, fewshot=True)
    stored = registry.examples_for("clean_order")
    assert stored
    for rec in stored:
        assert "_text" not in rec
        assert validate(rec, "clean_order") == []
    assert registry.pairs_for("clean_order")
    assert registry.fewshot_report("clean_order")["trained"] is True
