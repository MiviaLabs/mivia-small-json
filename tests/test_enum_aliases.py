"""Enum aliases: declared surface vocabulary for user-schema enums.

FEATURE (2026-10-10, owner order): a registered schema's enum field
gains an optional ``aliases`` map — declared surface words (any
language, any vocabulary) that coerce to a canonical enum member.

Coercion contract (honest-values law): extracted span -> trim +
casefold + wrap-strip (the spanfix conventions) -> exact enum member
first -> else declared alias lookup -> canonical member emitted -> else
REFUSED. Matching after normalization is EXACT — no fuzzy matching, no
engine-invented synonyms. Only the schema author's declared mapping
ever rewrites a value.

Refusal contract (the coercion-honesty fix underneath): when extraction
FOUND a value but coercion refuses it, the field is UNFILLED — it flows
through the existing missing-field path (absent for optional fields) —
never the declared default. The declared default applies only when
extraction found nothing. Before this fix, "it is urgent" + enum
[high, normal, low] silently emitted "urgency": "normal" (measured on
the pre-fix code: retries=1, the _nearest prefix ladder) — a wrong
record; the text said urgent.
"""

import pytest

from mivia_small_json import pipeline, registry, rules
from mivia_small_json.validate import repair, validate

# The owner's registration example: the order schema. The address
# pattern's non-greedy stop terminates at "as soon", so the multi-word
# alias "as soon as possible" never leaks into the address span; the
# urgency pattern captures the full surface (single- or multi-word) up
# to the next punctuation.
ORDER_SPEC = {
    "type": {"const": "order"},
    "id": {"type": str},
    "address": {"type": str},
    "email": {"type": str},
    "urgency": {"type": str, "enum": ["high", "normal", "low"],
                "aliases": {"high": ["urgent", "urgently", "asap",
                                     "as soon as possible", "eilt",
                                     "dringend", "sürgős", "urgente",
                                     "فوری", "عاجل", "срочно", "紧急",
                                     "긴급"],
                            "low": ["whenever", "no rush",
                                    "low priority"]}},
}
ORDER_ROUTING = {"strong": [r"(?i)\b(ship|order)\b"],
                 "weak": [r"(?i)\burgent|urgency\b"]}
ORDER_EXTRACTION = {
    "id": [r"(?i)\border\s+(\d+)\b"],
    "address": [r"(?i)\bto\s+(.+?)\s+as\s+soon\b"],
    "email": [r"(?i)\b([\w.+-]+@[\w-]+(?:\.[\w-]+)+)\b"],
    "urgency": [r"(?i)\bit(?:'s|\s+is)\s+(.+?)(?=[.,;!?]|$)"],
}

ACCEPTANCE_TEXT = ("Hi, please ship order 8841 to 12 Oak Rd as soon as "
                   "you can, it is urgent. You can reach me at a@x.io.")
ACCEPTANCE_GOLD = {"type": "order", "id": "8841", "address": "12 Oak Rd",
                   "email": "a@x.io", "urgency": "high"}


def _register_order(name="order", spec=None):
    registry.register_schema(name, spec or ORDER_SPEC, ORDER_ROUTING,
                             ORDER_EXTRACTION)


def _order_spec_named(name, urgency_rule):
    """ORDER_SPEC with the const re-named and the urgency rule swapped."""
    return dict(ORDER_SPEC, type={"const": name}, urgency=urgency_rule)


# --- registration validation -------------------------------------------------


def test_aliases_require_enum_on_the_same_rule():
    with registry.temporary_schemas():
        with pytest.raises(ValueError, match='"aliases" requires "enum"'):
            registry.register_schema(
                "a1", {"type": {"const": "a1"},
                       "f": {"type": str, "aliases": {"x": ["y"]}}},
                {"strong": [r"\ba1\b"], "weak": []}, {})


def test_aliases_only_on_str_typed_enum_fields():
    with registry.temporary_schemas():
        with pytest.raises(ValueError, match='"aliases" is only allowed'):
            registry.register_schema(
                "a2", {"type": {"const": "a2"},
                       "f": {"type": int, "enum": ["1", "2"],
                             "aliases": {"1": ["one"]}}},
                {"strong": [r"\ba2\b"], "weak": []}, {})


def test_alias_target_must_be_a_declared_enum_member():
    with registry.temporary_schemas():
        with pytest.raises(ValueError, match="not a declared enum member"):
            _register_order(spec=dict(ORDER_SPEC,
                              urgency={"type": str,
                                       "enum": ["high", "normal", "low"],
                                       "aliases": {"critical": ["urgent"]}}))


def test_alias_surfaces_must_be_non_empty_string_lists():
    with registry.temporary_schemas():
        with pytest.raises(ValueError, match="non-empty dict"):
            _register_order(spec=dict(ORDER_SPEC,
                              urgency={"type": str,
                                       "enum": ["high", "normal", "low"],
                                       "aliases": {}}))
        with pytest.raises(ValueError, match="non-empty"):
            _register_order(spec=dict(ORDER_SPEC,
                              urgency={"type": str,
                                       "enum": ["high", "normal", "low"],
                                       "aliases": {"high": []}}))
        with pytest.raises(ValueError, match="list of strings"):
            _register_order(spec=dict(ORDER_SPEC,
                              urgency={"type": str,
                                       "enum": ["high", "normal", "low"],
                                       "aliases": {"high": ["urgent", 7]}}))
        with pytest.raises(ValueError, match="empty surface"):
            _register_order(spec=dict(ORDER_SPEC,
                              urgency={"type": str,
                                       "enum": ["high", "normal", "low"],
                                       "aliases": {"high": ["   "]}}))


def test_ambiguous_alias_surfaces_rejected():
    with registry.temporary_schemas():
        # Same surface declared for two members.
        with pytest.raises(ValueError, match="declared for both"):
            _register_order(spec=dict(ORDER_SPEC,
                              urgency={"type": str,
                                       "enum": ["high", "normal", "low"],
                                       "aliases": {"high": ["fast"],
                                                   "low": ["fast"]}}))
        # A surface that collides with a DIFFERENT member's normalized
        # form (exact-match ambiguity).
        with pytest.raises(ValueError, match="collides with enum member"):
            _register_order(spec=dict(ORDER_SPEC,
                              urgency={"type": str,
                                       "enum": ["high", "normal", "low"],
                                       "aliases": {"low": ["HIGH"]}}))
        # Case variants of the SAME member's surface are fine.
        _register_order(spec=dict(ORDER_SPEC,
                          urgency={"type": str,
                                   "enum": ["high", "normal", "low"],
                                   "aliases": {"high": ["urgent",
                                                        "URGENT"]}}))


def test_aliases_defensively_copied_and_stored_as_tuples():
    with registry.temporary_schemas():
        aliases = {"high": ["urgent"], "low": ["no rush"]}
        surfaces = aliases["high"]
        spec = dict(ORDER_SPEC,
                    urgency={"type": str, "enum": ["high", "normal", "low"],
                             "aliases": aliases})
        _register_order(spec=spec)
        surfaces.append("asap")          # caller mutates after registration
        aliases["high"].append("noch eins")
        # Coercion is unchanged by the caller-side mutation (C6 contract).
        got = rules.extract("order 1 it is asap", "order")
        assert "urgency" not in got
        assert rules.extract("order 1 it is urgent", "order")["urgency"] == \
            "high"
        # Stored convention matches the enum tuple convention.
        stored = registry.resolve("order")["urgency"]["aliases"]
        assert stored == {"high": ("urgent",), "low": ("no rush",)}


# --- engine coercion (the actual site) ----------------------------------------


def test_exact_enum_member_wins_before_aliases_and_casefolds():
    with registry.temporary_schemas():
        _register_order()
        for surface, want in (("high", "high"), ("HIGH", "high"),
                              ("Low", "low"), ("normal", "normal")):
            got = rules.extract(f"order 1 it is {surface}", "order")
            assert got["urgency"] == want, (surface, got)


def test_multilingual_and_case_variant_aliases_coerce():
    with registry.temporary_schemas():
        _register_order()
        for surface, want in (
                # de
                ("eilt", "high"), ("dringend", "high"),
                # fa / ar
                ("فوری", "high"), ("عاجل", "high"),
                # ru / zh / ko
                ("срочно", "high"), ("紧急", "high"), ("긴급", "high"),
                # hu / ro / pt (the owner's example set)
                ("sürgős", "high"), ("urgente", "high"),
                # case variants of an en alias
                ("Urgent", "high"), ("URGENT", "high"), ("urgently", "high"),
                # multi-word alias
                ("as soon as possible", "high"), ("asap", "high"),
                # the low vocabulary
                ("whenever", "low"), ("no rush", "low"),
                ("low priority", "low"),
        ):
            got = rules.extract(f"order 1 it is {surface}", "order")
            assert got.get("urgency") == want, (surface, got)


def test_wrap_strip_applies_before_alias_lookup():
    with registry.temporary_schemas():
        _register_order()
        # Captured span carries wrapper noise: quotes and sentence
        # punctuation are stripped before the exact alias match.
        got = rules.extract('order 1 she said it is "URGENT".', "order")
        assert got["urgency"] == "high"


def test_unmatched_value_is_refused_not_rewritten():
    with registry.temporary_schemas():
        _register_order()
        # "kritisch" is neither a member nor a declared alias: the
        # field is UNFILLED (absent), never coerced to a nearest-looking
        # member, never defaulted at the extraction site.
        got = rules.extract("order 1 it is kritisch", "order")
        assert got == {"type": "order", "id": "1"}


def test_first_matching_patterns_verdict_stands():
    with registry.temporary_schemas():
        # Pattern 1 matches but its span refuses; pattern 2 would have
        # matched a canonical word. First match wins — a later pattern
        # never second-guesses an earlier one's verdict.
        registry.register_schema(
            "order3",
            _order_spec_named(
                "order3",
                {"type": str, "enum": ["high", "normal", "low"],
                 "aliases": {"high": ["urgent"]}}),
            ORDER_ROUTING,
            {"urgency": [r"(?i)\bshout(?:ing)?\s+(\w+)",
                         r"(?i)\bit(?:'s|\s+is)\s+(\w+)"]})
        got = rules.extract("order 1 shouting urg it is high", "order3")
        assert "urgency" not in got


def test_strict_validation_still_requires_canonical_members():
    with registry.temporary_schemas():
        _register_order()
        # Aliases are a COERCION feature, not a validation relaxation:
        # records carry canonical members; a raw alias surface in a
        # record is still an enum error.
        assert "enum:urgency" in validate(
            {"type": "order", "id": "1", "address": "a", "email": "e",
             "urgency": "urgent"}, "order")


# --- end to end (pipeline.classify, router=None) -------------------------------


def test_acceptance_probe_exact_specimen_end_to_end():
    with registry.temporary_schemas():
        _register_order()
        obj, meta = pipeline.classify(ACCEPTANCE_TEXT, router=None)
        assert obj == ACCEPTANCE_GOLD, obj
        assert meta["valid"] is True and meta["retries"] == 0, meta


def test_refusal_contract_end_to_end_missing_not_normal():
    """The refusal-contract probe (2026-10-10).

    BEFORE (measured on the pre-fix code, this exact text and enum):
    {"type": "order", ..., "urgency": "normal"} with retries=1 — the
    extracted "urgent" fell through _nearest to the enum middle. A wrong
    record: the text said urgent.

    AFTER: "kritisch" is found but matches no member and no declared
    alias, so the field is UNFILLED. The field is declared optional here
    — the honest refusal-to-absent choice, the same convention as the
    shipped ``commit_type`` ("absent rather than empty") — so the
    emitted record is strictly valid WITH the field absent, never with a
    fabricated value.
    """
    with registry.temporary_schemas():
        registry.register_schema(
            "order2",
            _order_spec_named(
                "order2",
                {"type": str, "enum": ["high", "normal", "low"],
                 "optional": True,
                 "aliases": {"high": ["asap", "eilt", "فوری", "срочно",
                                      "紧急", "긴급"],
                             "low": ["no rush"]}}),
            ORDER_ROUTING, ORDER_EXTRACTION)
        obj, meta = pipeline.classify(
            "Hi, please ship order 555 to 1 Elm Ave, it is kritisch. "
            "Reach me at b@y.io.", router=None)
        assert "urgency" not in obj, obj
        assert meta["valid"] is True, meta
        # The declared vocabulary still coerces end to end.
        obj2, meta2 = pipeline.classify(
            "Hi, please ship order 556 to 1 Elm Ave, it is فوری. "
            "Reach me at b@y.io.", router=None)
        assert obj2["urgency"] == "high", obj2
        assert meta2["valid"] is True


def test_refused_required_field_takes_the_declared_default_path():
    """Documented boundary (2026-10-10): a REQUIRED enum field cannot be
    emitted absent — the strictly-valid record contract fills it through
    the missing-field path, which applies the author's DECLARED default.
    Optionality is the schema author's honest refusal-to-absent choice
    (see test_refusal_contract_end_to_end_missing_not_normal)."""
    with registry.temporary_schemas():
        registry.register_schema(
            "order4",
            _order_spec_named(
                "order4",
                {"type": str, "enum": ["high", "normal", "low"],
                 "aliases": {"high": ["urgent"]}}),
            ORDER_ROUTING, ORDER_EXTRACTION)
        obj, meta = pipeline.classify(
            "order 9 to 2 Pine St, it is kritisch. mail c@z.io",
            router=None)
        # No declared default: the long-standing _default chain applies
        # (enum middle) — the found-nothing semantics, unchanged. The
        # refusal difference is that the text's own word never masquerades
        # as a match and retries stay 0 until the missing fill.
        assert obj["urgency"] == "normal", obj

        registry.unregister("order4")
        registry.register_schema(
            "order5",
            _order_spec_named(
                "order5",
                {"type": str, "enum": ["high", "normal", "low"],
                 "aliases": {"high": ["urgent"]}}),
            ORDER_ROUTING, ORDER_EXTRACTION,
            defaults={"urgency": "low"})
        obj2, _ = pipeline.classify(
            "order 9 to 2 Pine St, it is kritisch. mail c@z.io",
            router=None)
        # With a declared default, the author's declared value applies —
        # the declared default path, never an engine-invented rewrite of
        # "kritisch" itself.
        assert obj2["urgency"] == "low", obj2


def test_urgent_against_non_matching_enum_is_missing_not_normal():
    """The deliverable probe verbatim: the word "urgent" against an enum
    that admits no mapping for it. BEFORE (pre-fix, measured): the
    extracted "urgent" repaired to "urgency": "normal" (the enum middle)
    — a wrong record. AFTER: the value is REFUSED and the optional field
    stays absent; the declared default applies only when extraction
    found nothing, which is not this case."""
    with registry.temporary_schemas():
        registry.register_schema(
            "order6",
            _order_spec_named(
                "order6",
                {"type": str, "enum": ["high", "normal", "low"],
                 "optional": True,
                 "aliases": {"low": ["whenever", "no rush"]}}),
            ORDER_ROUTING, ORDER_EXTRACTION)
        obj, meta = pipeline.classify(
            "Hi, please ship order 8841 to 12 Oak Rd as soon as you can, "
            "it is urgent. You can reach me at a@x.io.", router=None)
        assert "urgency" not in obj, obj
        assert meta["valid"] is True, meta


# --- repair-side contract (hand-fed records) -----------------------------------


def test_repair_coerces_through_declared_aliases_only():
    with registry.temporary_schemas():
        _register_order()
        fixed = repair({"type": "order", "urgency": "ASAP."},
                       ["enum:urgency"], "order")
        assert fixed["urgency"] == "high"
        fixed2 = repair({"type": "order", "urgency": "kritisch"},
                        ["enum:urgency"], "order")
        assert "urgency" not in fixed2


def test_builtin_nearest_ladder_unchanged():
    """The built-in schemas keep the tuned _nearest ladder byte-for-byte
    (the golden fixture and the pre-registered bars are measured on it);
    the refusal contract is scoped to user schemas."""
    fixed = repair({"type": "issue", "priority": "urgentissimo"},
                   ["enum:priority"], "issue")
    assert fixed["priority"] == "high"  # unchanged nearest/prefix behavior
    fixed2 = repair({"type": "log_event", "level": "warning"},
                    ["enum:level"], "log_event")
    assert fixed2["level"] == "WARN"


def test_corpus_world_spec_fingerprint_carries_no_aliases():
    """Schemas without aliases freeze byte-identically, so the committed
    spec fingerprint (tests/test_gates.py) cannot move (expectation pin
    for the golden-is-byte-identical contract; the gate itself stays the
    authority)."""
    for schema in registry.active():
        for field, rule in registry.resolve(schema).items():
            assert "aliases" not in rule, (schema, field)
