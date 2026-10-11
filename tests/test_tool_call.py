"""Tests for the ``tool_call`` AI agent workflow signal and schema.

Verifies:
- Schema registration and field specification.
- Keyword routing and model routing to ``tool_call``.
- Extraction of name, arguments, and result fields.
- Validation, repair, and pipeline classification.
"""

from __future__ import annotations

import json

import pytest

from mivia_small_json import pipeline, registry, rules, tool_call
from mivia_small_json.model import MODEL_PATH
from mivia_small_json.validate import repair, validate


@pytest.fixture(autouse=True)
def ensure_tool_call_registered():
    tool_call.register_tool_call_schema()
    yield
    for s in ("tool_call", "no_tool_call", "tool_clarification"):
        if s in registry.active():
            registry.unregister(s)



def test_tool_call_schema_is_active():
    assert "tool_call" in registry.active()
    spec = registry.resolve("tool_call")
    assert spec["type"]["const"] == "tool_call"
    assert "name" in spec
    assert "arguments" in spec
    assert "result" in spec


def test_tool_call_routing():
    samples = [
        '<functioncall> {"name": "get_weather", "arguments": \'{"location": "Boston"}\'}',
        'FUNCTION RESPONSE: {"temperature": 72}',
        'tool_call: name=search_files arguments={"query": "test"} result={"found": 1}',
    ]
    for s in samples:
        assert rules.route(s) == "tool_call"


def test_tool_call_extraction():
    text = (
        '<functioncall> {"name": "calculate_tax", "arguments": \'{"income": 50000}\'}\n'
        'FUNCTION RESPONSE: {"tax": 7500}'
    )
    extracted = rules.extract(text, "tool_call")
    assert extracted["type"] == "tool_call"
    assert extracted["name"] == "calculate_tax"
    assert "50000" in extracted["arguments"]
    assert "7500" in extracted["result"]


def test_tool_call_validation_and_repair():
    # Valid object
    valid_obj = {
        "type": "tool_call",
        "name": "read_file",
        "arguments": '{"path": "foo.txt"}',
        "result": "contents",
    }
    assert not validate(valid_obj, "tool_call")

    # Incomplete object repaired with defaults
    incomplete = {"type": "tool_call"}
    errors = validate(incomplete, "tool_call")
    assert errors
    repaired = repair(incomplete, errors, "tool_call")
    assert not validate(repaired, "tool_call")
    assert repaired["arguments"] == "{}"


def test_pipeline_classify_tool_call():
    router = pipeline.load_router(MODEL_PATH)
    raw = (
        '<functioncall> {"name": "fetch_api", "arguments": \'{"url": "https://api.github.com"}\'}\n'
        'FUNCTION RESPONSE: {"status": 200}'
    )
    obj, meta = pipeline.classify(raw, router)
    assert meta["valid"] is True
    assert obj["type"] == "tool_call"
    assert obj["name"] == "fetch_api"
    assert "https://api.github.com" in obj["arguments"]


@pytest.mark.parametrize("exc_type", [ValueError, SyntaxError, MemoryError, TypeError])
def test_literal_eval_dict_degrades_on_parse_failures(monkeypatch, exc_type):
    """Parse failures stay inside the never-raise ladder. These four are
    what ast.literal_eval and the JSON conversion raise on bad input."""
    import ast

    def boom(_text):
        raise exc_type("bad literal")

    monkeypatch.setattr(ast, "literal_eval", boom)
    assert tool_call._literal_eval_dict("{'a': 1}") is None


def test_literal_eval_dict_propagates_internal_errors(monkeypatch):
    """A RuntimeError from the evaluator is an internal failure. Swallowing
    it turns a broken parser into a silent empty object."""
    import ast

    def boom(_text):
        raise RuntimeError("tokenizer table corrupt")

    monkeypatch.setattr(ast, "literal_eval", boom)
    with pytest.raises(RuntimeError, match="tokenizer table corrupt"):
        tool_call._literal_eval_dict("{'a': 1}")


def test_repair_arguments():
    # Single quoted
    assert tool_call.repair_arguments("{'city': 'Paris'}") == '{"city":"Paris"}'
    # Python booleans and None
    assert (
        tool_call.repair_arguments("{'enabled': True, 'count': None}")
        == '{"enabled":true,"count":null}'
    )
    # Trailing comma
    assert tool_call.repair_arguments('{"key": "val",}') == '{"key":"val"}'


def test_repair_arguments_keeps_valid_json_with_escaped_quotes():
    """A valid object whose string values contain ``\\"`` must survive.

    The old code unescaped every ``\\"`` globally, corrupting the object so
    it degraded to ``{}``; this lost 12% of harvested ``agent_step``
    arguments (a shell command embedding quoted text).
    """
    raw = '{"command": "echo \\"hi\\" && ls"}'
    assert json.loads(tool_call.repair_arguments(raw)) == {
        "command": 'echo "hi" && ls'}


def test_repair_arguments_keeps_valid_json_with_pseudo_json_inside():
    """The single-quote / Python-literal rewrites target pseudo-JSON, so a
    valid object whose *value* contains ``'key':`` or ``None`` must not be
    rewritten."""
    raw = '{"command": "python3 -c \\"d = {\'a\': None}\\""}'
    assert json.loads(tool_call.repair_arguments(raw)) == {
        "command": "python3 -c \"d = {'a': None}\""}


def test_repair_arguments_trims_captured_extra_closer():
    """The extraction regex can capture one brace too many.

    When the argument object sits inside a wrapper object the greedy pattern
    takes the wrapper's closing brace, so the value arrives as `{}}`. Repair
    must return valid JSON, not the malformed string.
    """
    assert json.loads(tool_call.repair_arguments('{"a": 1}}')) == {"a": 1}
    assert json.loads(tool_call.repair_arguments("{}}")) == {}
    assert json.loads(tool_call.repair_arguments('{"a": [1, 2]}}')) == {"a": [1, 2]}
    # A mismatched closer is dropped, and a non-object value degrades to the
    # empty object rather than to a bare scalar or invalid JSON.
    assert tool_call.repair_arguments('{"x": 1}]') == '{"x":1}'
    assert tool_call.repair_arguments("42}") == "{}"


def test_extract_tool_call_handles_double_wrapped_escaped_arguments():
    """A double-quoted escaped-JSON argument value must be captured whole.

    The naive `"[^"]+"` alternative stops at the first escaped quote, so the
    value arrived as `"{"` and the arguments parsed to the string "{". The
    capture must span the escaped quotes.
    """
    text = '{"name": "Skill", "arguments": "{\\"skill\\": \\"code-review\\"}"}'
    obj = tool_call.extract_tool_call(text)
    assert json.loads(obj["arguments"]) == {"skill": "code-review"}


def test_extract_tool_call_helper():
    text = (
        '<tool_call> {"name": "execute_bash", '
        '"arguments": \'{"command": "pytest -q"}\'}</tool_call>\n'
        'TOOL RESPONSE: 180 passed in 1.2s'
    )
    extracted = tool_call.extract_tool_call(text)
    assert extracted["type"] == "tool_call"
    assert extracted["name"] == "execute_bash"
    assert '{"command":"pytest -q"}' in extracted["arguments"]
    assert "180 passed" in extracted["result"]


def test_tool_call_marker_is_counted_once():
    """The marker must sit in one strength tier, not both."""
    scores = rules.route_scores("please tool_call the helper now")
    assert scores["tool_call"] == 2, scores


def test_pipeline_classify_normalizes_tool_call_arguments():
    """The pipeline must emit normalized JSON, not raw pseudo-JSON."""
    router = pipeline.load_router(MODEL_PATH)
    raw = (
        '<functioncall> {"name": "convert", "arguments": '
        "'{\"value\": 100, \"unit\": \"C\"}'}\n"
        'FUNCTION RESPONSE: {"ok": true}'
    )
    obj, meta = pipeline.classify(raw, router)
    assert obj["type"] == "tool_call"
    assert json.loads(obj["arguments"]) == {"value": 100, "unit": "C"}


def test_tool_clarification_registration_and_validation():
    """tool_clarification schema registers and validates cleanly."""
    tool_call.register_tool_clarification_schema()
    assert "tool_clarification" in registry.active()
    sample = {
        "type": "tool_clarification",
        "tool": "get_weather",
        "reason": "missing_arguments",
        "missing_parameters": ["location"],
        "clarification_prompt": "Please specify the location to check the weather.",
    }
    assert validate(sample, "tool_clarification") == []


def test_pipeline_classify_emits_tool_clarification_when_params_missing():
    """Pipeline emits tool_clarification when request has missing required params."""
    tool_call.register_tool_call_schema()
    tool_call.register_no_tool_call_schema()
    tool_call.register_tool_clarification_schema()
    raw = (
        'Tools available:\n[{"name": "calc_sum", "description": "Add two numbers", '
        '"parameters": {"type": "object", "properties": '
        '{"a": {"type": "number"}, "b": {"type": "number"}}, '
        '"required": ["a", "b"]}}]\n\n'
        'User request: sum'
    )
    obj, meta = pipeline.classify(raw)
    assert meta["valid"] is True
    assert obj["type"] == "tool_clarification"
    assert obj["tool"] == "calc_sum"
    assert "a" in obj["missing_parameters"] and "b" in obj["missing_parameters"]




# --- BFCL fully-specified natural-language calls (59af3e0 regression) -------

class TestFullySpecifiedCallsEmitToolCall:
    """59af3e0: 119/183 fully-specified BFCL calls were captured as
    tool_clarification because the ladder re-clarified on `required` alone
    after the fillability gate had already passed. A guard-accepted,
    fillable request emits tool_call; arguments carry only the slots
    span_fill selected (the honest minimum, never fabricated values)."""

    def _fake_guard(self, monkeypatch, tool, score):
        class _FakeGuard:
            def best_match(self, _text):
                return tool, score
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard", lambda: _FakeGuard())

    def test_string_slot_request_emits_call_with_honest_minimum(
            self, monkeypatch, ensure_tool_call_registered):
        from mivia_small_json import span_fill

        if span_fill.load_spanfill_extractor() is not None:
            pytest.skip(
                "staged spanfill candidate present in models/ (un-promoted): "
                "the string family is admitted and the arm's fill contract "
                "supersedes the pre-arm honest-{} minimum — the request "
                "fills DNA_id (spanfix wave, 2026-10-10). Revisit at "
                "promotion: the assert becomes the filled form.")
        tool_call.register_tool_clarification_schema()
        dna = {
            "type": "function",
            "function": {
                "name": "fetch_DNA_sequence",
                "description": "Fetch a DNA sequence by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"DNA_id": {"type": "string"}},
                    "required": ["DNA_id"],
                },
            },
        }
        raw = (
            'Tools available:\n[' + json.dumps(dna) + ']\n\n'
            'User request: Retrieve the sequence of DNA molecule with id '
            '`DNA123`.'
        )
        self._fake_guard(monkeypatch, dna, 0.99)
        obj, meta = pipeline._classify_known_schema(raw, "tool_call", "model")
        assert obj["type"] == "tool_call"
        assert obj["name"] == "fetch_DNA_sequence"
        assert json.loads(obj["arguments"]) == {}
        assert meta["valid"] is True

    def test_unit_attached_number_is_not_a_missing_argument(
            self, monkeypatch, ensure_tool_call_registered):
        tool_call.register_tool_clarification_schema()
        energy = {
            "type": "function",
            "function": {
                "name": "thermo.calculate_energy",
                "description": "Energy of a phase change.",
                "parameters": {
                    "type": "object",
                    "properties": {"mass": {"type": "number"}},
                    "required": ["mass"],
                },
            },
        }
        raw = (
            'Tools available:\n[' + json.dumps(energy) + ']\n\n'
            'User request: Calculate the energy absorbed during the phase '
            'change of 100g of water from ice to liquid.'
        )
        self._fake_guard(monkeypatch, energy, 0.99)
        obj, meta = pipeline._classify_known_schema(raw, "tool_call", "model")
        # The decision-layer veto keeps unit-attached values ("100g"), so a
        # supplied mass is not asked back for.
        assert obj["type"] == "tool_call"
        assert obj["name"] == "thermo.calculate_energy"
        assert meta["valid"] is True


class TestStripFencesCloserPrescreen:
    """The XML-fence finditer was quadratic when no closer existed: every
    opener rescans the whole tail, so a 256 KiB wall of "<tool_call>"
    pinned a worker ~27 s through repair_arguments (reachable from
    /v1/repair and both MCP surfaces — review 2026-10-11). The closer
    prescreen skips finditer entirely when no closer is present; these
    pin the behavior it must preserve (no timing asserts — the gates stay
    mechanical)."""

    def test_wall_of_openers_passes_through_unchanged(self):
        s = "<tool_call>" * 20_000
        assert tool_call._strip_fences(s) is s

    def test_wall_of_openers_mixed_case_passes_through(self):
        s = "<ToolCall><functioncall>" * 10_000
        assert tool_call._strip_fences(s) is s

    def test_closer_present_still_extracts_body(self):
        s = 'ignore this <tool_call> {"a": 1} </tool_call> trailing'
        assert tool_call._strip_fences(s) == '{"a": 1}'

    def test_case_insensitive_closer_still_matches(self):
        s = '<TOOL_CALL>{"a": 1}</TOOL_CALL>'
        assert tool_call._strip_fences(s) == '{"a": 1}'

    def test_functioncall_variant_with_closer(self):
        s = '<functioncall>{"b": 2}</functioncall>'
        assert tool_call._strip_fences(s) == '{"b": 2}'

    def test_markdown_fence_unchanged_behavior(self):
        s = '```json\n{"c": 3}\n```'
        assert tool_call._strip_fences(s) == '{"c": 3}'
        bare = 'no fences {"a": 1}'
        # unfenced input passes through unchanged (the function contract)
        assert tool_call._strip_fences(bare) is bare
