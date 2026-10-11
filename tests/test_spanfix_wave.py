"""The 2026-10-10 spanfix wave: failing-tests-first fixes for the ten
re-runner-confirmed span_fill / tool_call defects (bugs 1-10).

Each test reproduces its defect BEFORE the fix. [staged] bugs (3, 9)
reproduce only with the staged un-promoted candidate present; their tests
skip loudly when it is not loadable.

Surfaces: span_fill.py, tool_call.py, and the trainer decode mirror
(scripts/train_stringparam_scorer.py). Bars B1-B8 are frozen; nothing
here moves a floor — every test is a contract reproduction.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from mivia_small_json import pipeline, span_fill
from mivia_small_json.tool_call import (
    register_no_tool_call_schema,
    register_tool_call_schema,
    register_tool_clarification_schema,
)


def _fn_tool(slots: dict) -> dict:
    """Tool whose required params are the given name -> declared-type map
    (a missing type stays undeclared).

    A STRING value is the declared type (``"number"`` -> ``{"type":
    "number"}``); a DICT value is the property spec itself — bug 8's
    ENUM_TOOL passes ``{"type": "string", "enum": [...]}`` and needs the
    enum on the property, not nested inside ``type`` (fixture repair,
    spanfix wave 2026-10-10: the cancelled agent's helper wrapped dict
    values as ``{"type": <dict>}``, a malformed spec ``family_for``
    rightly refuses, so the string-family tests could never run)."""
    props = {}
    for name, stype in slots.items():
        if isinstance(stype, dict):
            props[name] = dict(stype)
        elif stype:
            props[name] = {"type": stype}
        else:
            props[name] = {}
    return {
        "type": "function",
        "function": {
            "name": "t",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(slots),
            },
        },
    }


class _Inject:
    """Injection through the trainer's seam, always restored."""

    def __init__(self, scorer, emit=0.6, extend=0.15):
        self._scorer = scorer
        self._floors = (emit, extend)

    def __enter__(self):
        self._saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
        self._saved_floors = (span_fill._SPANFILL_EMIT_FLOOR,
                              span_fill._SPANFILL_EXTEND_FLOOR)
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = \
            self._scorer, True
        span_fill.configure_spanfill_floors(*self._floors)
        return self._scorer

    def __exit__(self, *exc):
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = self._saved
        span_fill.configure_spanfill_floors(*self._saved_floors)
        return False


# --- bug 1: currency glued to a symbol becomes a clarification --------------

class TestBug1CurrencyGlued:
    def test_currency_symbol_key_delimited_fills(self):
        tool = _fn_tool({"price": "number", "qty": "integer"})
        filled, missing = span_fill.fill_arguments(tool, "price=$12 qty=3")
        assert filled == {"price": 12, "qty": 3}, (filled, missing)
        assert missing == []

    def test_currency_thousands_both_slots_fill(self):
        tool = _fn_tool({"amount": "number", "fee": "number"})
        filled, missing = span_fill.fill_arguments(
            tool, "amount=$1,000 fee=$2")
        assert filled == {"amount": 1000, "fee": 2}, (filled, missing)
        assert missing == []

    def test_currency_code_and_trailing_symbol(self):
        tool = _fn_tool({"price": "number"})
        filled, missing = span_fill.fill_arguments(tool, "price=US$12")
        assert filled == {"price": 12}, (filled, missing)
        filled, missing = span_fill.fill_arguments(tool, "price=12,")
        assert filled == {"price": 12}, (filled, missing)

    def test_plain_and_prose_currency_still_fill(self):
        tool = _fn_tool({"price": "number"})
        assert span_fill.fill_arguments(tool, "price=12")[0] \
            == {"price": 12}
        assert span_fill.fill_arguments(tool, "the price is $12")[0] \
            == {"price": 12}

    def test_exponent_window_protection_survives(self):
        """The FILLED-BUT-INVALID protection the currency fix must not
        break: `amount=1e309` (inf) must stay missing AND must not steal
        fee's `2` — the whole point of the skip block."""
        tool = _fn_tool({"amount": "number", "fee": "number"})
        filled, missing = span_fill.fill_arguments(tool, "amount=1e309 fee=2")
        assert filled == {"fee": 2}, (filled, missing)
        assert missing == ["amount"]


# --- bug 2: 'country' is not 'count' — hint words match whole tokens --------

class TestBug2TokenBoundaryHints:
    def test_country_never_receives_a_number(self):
        """Minimum contract: 2024 never lands in the country slot."""
        tool = _fn_tool({"country": {}})
        assert span_fill.family_for(tool) is None
        filled, missing = span_fill.fill_arguments(
            tool, "Look up the country France in 2024")
        assert 2024 not in (filled or {}).values(), (filled, missing)

    def test_account_is_not_a_count(self):
        assert span_fill.family_for(_fn_tool({"account": {}})) is None

    def test_hint_slots_stay_numeric(self):
        for name in ("count", "num_items", "retry_count", "num1"):
            tool = _fn_tool({name: {}})
            assert span_fill.family_for(tool) == "numeric", name


# --- bug 3: [staged] OpenAI-wrapped tool must fill through learned_fill -----

WRAPPED_STR_TOOL = {"type": "function", "function": {
    "name": "restart_service",
    "description": "Restart a systemd service on a named host.",
    "parameters": {"type": "object", "properties": {
        "service": {"type": "string",
                    "description": "The systemd service unit."},
        "host": {"type": "string", "description": "The target host."}},
        "required": ["service", "host"]}}}


class _SlotAwareScorer:
    """Scores each slot's canonical value 1.0, everything else 0.0 —
    deterministic, slot-aware stand-in for the trained pair scorer."""

    def __init__(self, values):
        self._values = values

    def pair_probs(self, request, param, prop):
        want = self._values.get(param, "")
        return [1.0 if m.group(0) == want else 0.0
                for m in re.finditer(r"\w+", request)]


class TestBug3WrappedEnvelope:
    def test_learned_fill_unwraps_the_envelope(self):
        with _Inject(_SlotAwareScorer({"service": "nginx",
                                       "host": "staging"})):
            filled, missing = span_fill.learned_fill(
                WRAPPED_STR_TOOL, "restart nginx on staging")
        assert filled == {"service": "nginx", "host": "staging"}, \
            (filled, missing)
        assert missing == []

    def test_learned_fill_wrapped_respects_only_filter(self):
        with _Inject(_SlotAwareScorer({"host": "staging"})):
            filled, missing = span_fill.learned_fill(
                WRAPPED_STR_TOOL, "restart nginx on staging",
                only=["host"])
        assert filled == {"host": "staging"}, (filled, missing)

    @pytest.mark.skipif(span_fill.load_spanfill_extractor() is None,
                        reason="staged spanfill candidate not loadable")
    def test_wrapped_tool_fills_through_the_served_pipeline(self):
        """The real pipeline path: a wrapped catalogue tool + a natural
        request must fill (or clarify honestly) — never a silent {}-call.
        (Heavily underspecified requests legitimately clarify; this one
        carries both values verbatim.)"""
        register_tool_call_schema()
        register_no_tool_call_schema()
        register_tool_clarification_schema()
        catalogue = (f"Tools available:\n[{json.dumps(WRAPPED_STR_TOOL)}]"
                     "\n\nUser request: restart nginx on staging")
        obj, meta = pipeline.classify(catalogue)
        assert obj["type"] == "tool_call", obj
        assert json.loads(obj["arguments"]) == {"service": "nginx",
                                                "host": "staging"}, obj
        assert meta["valid"] is True


# --- bug 4: alignment ladder steals the word before "for|of <slot>" ---------

class TestBug4AlignmentAfterSide:
    def test_value_after_for_of_slot_wins(self):
        tool = _fn_tool({"path": {"type": "string"}})
        filled, missing = span_fill.fill_arguments(
            tool, "show files for path /tmp")
        assert filled == {"path": "/tmp"}, (filled, missing)
        assert missing == []

    def test_before_side_still_fallback_when_after_empty(self):
        tool = _fn_tool({"path": {"type": "string"}})
        filled, missing = span_fill.fill_arguments(
            tool, "the report for path")
        assert filled == {"path": "report"}, (filled, missing)

    def test_of_variant_prefers_after_side(self):
        tool = _fn_tool({"car": {"type": "string"}})
        filled, missing = span_fill.fill_arguments(
            tool, "show the color of car tesla")
        assert filled == {"car": "tesla"}, (filled, missing)


# --- bug 5: leading dots are content (.env is a filename) -------------------

class TestBug5LeadingDots:
    def test_learned_coerce_keeps_leading_dot(self):
        prop = {"type": "string"}
        assert span_fill._learned_coerce(".env", prop) == ".env"
        assert span_fill._learned_coerce(".gitignore", prop) == ".gitignore"

    def test_quoted_values_still_strip_their_quotes(self):
        prop = {"type": "string"}
        assert span_fill._learned_coerce("'quoted'", prop) == "quoted"
        assert span_fill._learned_coerce('"dq"', prop) == "dq"

    def test_trailing_punctuation_still_strips(self):
        prop = {"type": "string"}
        assert span_fill._learned_coerce("staging.", prop) == "staging"
        assert span_fill._learned_coerce("nginx,", prop) == "nginx"

    def test_key_delimited_dotfile_fills_whole(self):
        with _Inject(_SlotAwareScorer({"path": "env"})):
            filled, missing = span_fill.fill_arguments(
                _fn_tool({"path": {"type": "string"}}), "deploy path=.env")
        assert filled == {"path": ".env"}, (filled, missing)


# --- bug 6: comma-glued assignments swallow the next one --------------------

class TestBug6CommaGluedAssignments:
    def test_comma_glued_fills_both(self):
        with _Inject(_SlotAwareScorer({"service": "nginx",
                                       "host": "web-01"})):
            filled, missing = span_fill.fill_arguments(
                WRAPPED_STR_TOOL["function"],
                "service=nginx,host=web-01 restart now")
        assert filled == {"service": "nginx", "host": "web-01"}, \
            (filled, missing)
        assert missing == []

    def test_spaced_form_still_fills_both(self):
        with _Inject(_SlotAwareScorer({"service": "nginx",
                                       "host": "web-01"})):
            filled, missing = span_fill.fill_arguments(
                WRAPPED_STR_TOOL["function"],
                "service=nginx host=web-01 restart now")
        assert filled == {"service": "nginx", "host": "web-01"}, \
            (filled, missing)
        assert missing == []


# --- bug 7: apostrophes inside quoted values --------------------------------

class TestBug7ApostropheBoundary:
    def test_single_quoted_apostrophe_name_fills(self):
        with _Inject(_SlotAwareScorer({"host": "O'Brien"})):
            filled, missing = span_fill.fill_arguments(
                WRAPPED_STR_TOOL["function"], "host='O'Brien' restart")
        assert filled == {"host": "O'Brien"}, (filled, missing)

    def test_double_quoted_apostrophe_name_fills(self):
        with _Inject(_SlotAwareScorer({"host": "O'Brien"})):
            filled, missing = span_fill.fill_arguments(
                WRAPPED_STR_TOOL["function"], 'host="O\'Brien" restart')
        assert filled == {"host": "O'Brien"}, (filled, missing)

    def test_genuine_quoted_phrase_still_strips_quotes(self):
        with _Inject(_SlotAwareScorer({"body": "deploy finished"})):
            filled, missing = span_fill.fill_arguments(
                _fn_tool({"body": {"type": "string"}}),
                "post body='deploy finished' now")
        assert filled == {"body": "deploy finished"}, (filled, missing)


# --- bug 8: enum comparison runs after the wrap-strip -----------------------

class TestBug8EnumWrapStrip:
    ENUM_TOOL = _fn_tool({"environment": {"type": "string",
                                          "enum": ["staging"]}})

    def test_quoted_enum_member_matches(self):
        with _Inject(_SlotAwareScorer({"environment": "staging"})):
            filled, missing = span_fill.fill_arguments(
                self.ENUM_TOOL, 'deploy environment="staging" now')
        assert filled == {"environment": "staging"}, (filled, missing)

    def test_trailing_punct_enum_member_matches(self):
        tool = _fn_tool({"environment": {"type": "string",
                                         "enum": ["prod-eu-1"]}})
        with _Inject(_SlotAwareScorer({"environment": "prod-eu-1"})):
            filled, missing = span_fill.fill_arguments(
                tool, "deploy environment=prod-eu-1! now")
        assert filled == {"environment": "prod-eu-1"}, (filled, missing)

    def test_bare_enum_member_still_matches(self):
        with _Inject(_SlotAwareScorer({"environment": "staging"})):
            filled, missing = span_fill.fill_arguments(
                self.ENUM_TOOL, "deploy environment=staging now")
        assert filled == {"environment": "staging"}, (filled, missing)

    def test_enum_mismatch_still_refused(self):
        with _Inject(_SlotAwareScorer({"environment": "staging"})):
            filled, missing = span_fill.fill_arguments(
                self.ENUM_TOOL, 'deploy environment="prod" now')
        assert filled == {} and missing == ["environment"], (filled, missing)


# --- bug 9: [staged] sentence membership via sequence_ids, not type_ids -----

class _FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in
                ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, _none, feed):
        return [[_FakeSession.logits]]


class TestBug9SequenceMembership:
    def test_pair_probs_never_selects_sequence1_tokens(self):
        """Red: the fake encoding has ALL-ZERO type_ids (this XLM-R
        tokenizer's real behavior) with the maximum logit planted on a
        sequence-1 (slot-side) token whose offsets overlap the request
        words. The decode must use sequence_ids: request tokens score
        from sequence-0 logits ONLY."""
        request = "restart nginx"
        # request tokens: restart(0,7) nginx(8,13); the pair's slot-side
        # subtokens restart at offset 0 and overlap them (the real skew).
        offsets = [(0, 7), (8, 13), (0, 7), (8, 13)]
        seq_ids = [0, 0, 1, 1]
        type_ids = [0, 0, 0, 0]  # all-zero: the measured tokenizer skew
        logits = [0.0, 0.0, 10.0, 10.0]  # max sits on sequence 1

        class _Tok:
            def encode(self, text, pair=None):
                return SimpleNamespace(ids=[1, 2, 3, 4], type_ids=type_ids,
                                       attention_mask=[1] * 4,
                                       offsets=offsets,
                                       sequence_ids=list(seq_ids))

        scorer = span_fill._EncoderPairScorer(_FakeSession(), _Tok())
        _FakeSession.logits = logits
        probs = scorer.pair_probs(request, "service", {"type": "string"})
        assert len(probs) == 2
        # sequence-1 logits (sigmoid(10) ~ 0.99995) must NOT leak into
        # the request tokens; only the sequence-0 evidence (sigmoid(0)
        # = 0.5) counts.
        assert all(p <= 0.5 + 1e-9 for p in probs), probs

    def test_pair_probs_works_when_sequence_ids_is_a_method(self):
        """tokenizers exposes sequence_ids as a property list in some
        versions and a method in others; the decode must accept both."""
        request = "restart nginx"

        class _Tok:
            def encode(self, text, pair=None):
                enc = SimpleNamespace(ids=[1, 2, 3, 4], type_ids=[0] * 4,
                                      attention_mask=[1] * 4,
                                      offsets=[(0, 7), (8, 13), (0, 0),
                                               (0, 0)])
                enc.sequence_ids = lambda: [0, 0, 1, 1]  # method form
                return enc

        scorer = span_fill._EncoderPairScorer(_FakeSession(), _Tok())
        _FakeSession.logits = [10.0, 10.0, 0.0, 0.0]
        probs = scorer.pair_probs(request, "service", {"type": "string"})
        assert len(probs) == 2
        assert all(p >= 0.99 for p in probs), probs


# --- bug 10: validate must refuse arguments that are not JSON ---------------

class TestBug10ArgumentsAreJson:
    def test_validate_refuses_non_json_arguments(self):
        from mivia_small_json.bootstrap import register_all_schemas
        from mivia_small_json.validate import repair, validate

        register_all_schemas()
        obj = {"type": "tool_call", "name": "run_command",
               "arguments": "run the tests now", "result": ""}
        errors = validate(obj, "tool_call")
        assert errors, "validate accepted a non-JSON arguments string"
        fixed = repair(obj, errors, "tool_call")
        assert validate(fixed, "tool_call") == [], fixed
        assert isinstance(json.loads(fixed["arguments"]), dict), fixed

    def test_valid_json_arguments_still_validate_clean(self):
        from mivia_small_json.bootstrap import register_all_schemas
        from mivia_small_json.validate import validate

        register_all_schemas()
        obj = {"type": "tool_call", "name": "run_command",
               "arguments": '{"command": "pytest -q"}',
               "result": "157 passed"}
        assert validate(obj, "tool_call") == []

    def test_pseudo_json_arguments_repair_to_strict_json(self):
        from mivia_small_json.bootstrap import register_all_schemas
        from mivia_small_json.validate import repair, validate

        register_all_schemas()
        obj = {"type": "tool_call", "name": "run_command",
               "arguments": "{'command': 'pytest -q'}", "result": ""}
        errors = validate(obj, "tool_call")
        assert errors
        fixed = repair(obj, errors, "tool_call")
        assert validate(fixed, "tool_call") == []
        assert json.loads(fixed["arguments"]) == {"command": "pytest -q"}

    def test_nan_infinity_are_not_json(self):
        # Review 2026-10-10 (P2): python's json.loads accepts NaN/Infinity
        # by default; the strict-JSON contract must refuse them.
        from mivia_small_json.bootstrap import register_all_schemas
        from mivia_small_json.validate import validate

        register_all_schemas()
        for bad in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
            obj = {"type": "tool_call", "name": "run_command",
                   "arguments": bad, "result": ""}
            assert validate(obj, "tool_call"), f"validate accepted {bad!r}"

    def test_registered_json_default_is_strict_json_object(self):
        # Review 2026-10-10 (P3): a json-format field without a declared
        # default templates to the honest-minimum "{}", not "".
        from mivia_small_json.bootstrap import register_all_schemas
        from mivia_small_json.validate import _default

        register_all_schemas()
        assert _default("tool_call", "arguments") == "{}"
