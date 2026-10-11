"""Span-selection arg filling for the numeric tool family (no generation)."""

from mivia_small_json import pipeline, registry, span_fill
from mivia_small_json.tool_call import (
    register_no_tool_call_schema,
    register_tool_call_schema,
    register_tool_clarification_schema,
)


def _num_tool(name="calc_sum"):
    return {
        "name": name,
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    }


def _catalogue(tool, request):
    import json

    return f"Tools available:\n[{json.dumps(tool)}]\n\nUser request: {request}"


def test_family_gate_admits_numeric_and_all_string_tools():
    """Deterministic-only family admission (2026-10-10, owner-approved):
    the numeric family is unchanged and every-required-slot-string tools
    are admitted to the string family with or without a learned
    artifact."""
    assert span_fill.family_for(_num_tool()) == "numeric"
    assert span_fill.family_for({"name": "f", "description": "d"}) is None
    text_tool = {
        "name": "greet",
        "description": "say hi",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    }
    assert span_fill.family_for(text_tool) == "string"


def test_positional_numbers_fill_in_order():
    filled, missing = span_fill.fill_arguments(_num_tool(), "calculate sum of 10 and 20")
    assert missing == []
    assert filled == {"a": 10, "b": 20}


def test_key_delimited_assignment_wins_over_position():
    filled, missing = span_fill.fill_arguments(_num_tool(), "add b=7 and a=5 please")
    assert missing == []
    assert filled == {"a": 5, "b": 7}


def test_quoted_json_span_in_request():
    filled, missing = span_fill.fill_arguments(
        _num_tool(), 'please sum {"a": 3, "b": 4} now'
    )
    assert missing == []
    assert filled == {"a": 3, "b": 4}


def test_integer_coercion_rejects_fraction():
    tool = {
        "name": "f",
        "description": "d",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    }
    filled, missing = span_fill.fill_arguments(tool, "use n=3.5 please")
    assert filled == {}
    assert missing == ["n"]


def test_unfillable_slot_is_reported_missing():
    filled, missing = span_fill.fill_arguments(_num_tool(), "please calculate the sum")
    assert filled == {}
    assert missing == ["a", "b"]


def _served_num_tool(name="calculate_sum"):
    """A numeric tool with a realistic description the guard keeps.

    The cross-encoder guard stays upstream of span filling: a thin
    synthetic description scores below threshold (measured 0.59), so the
    end-to-end tests use a description the shipped guard accepts
    (measured 0.999 with tool-match-int8.onnx c34c562c0356). If the guard
    artifact is retrained these scores — and these tests — can move with
    no code change.
    """
    tool = _num_tool(name)
    tool["description"] = (
        "Calculate the sum of two numbers a and b and return the result."
    )
    return tool


def test_pipeline_emits_filled_tool_call():
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()
    assert "tool_call" in registry.active()
    raw = _catalogue(_served_num_tool(), "calculate sum of 10 and 20")
    obj, meta = pipeline.classify(raw)
    assert obj["type"] == "tool_call", obj
    import json

    assert json.loads(obj["arguments"]) == {"a": 10, "b": 20}
    assert meta["valid"] is True


def test_pipeline_clarifies_unfillable_numeric_request():
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()
    raw = _catalogue(_served_num_tool(), "calculate the sum")
    obj, meta = pipeline.classify(raw)
    assert obj["type"] == "tool_clarification", obj
    assert set(obj["missing_parameters"]) == {"a", "b"}
    assert meta["valid"] is True


def test_module_has_no_heavy_dependencies():
    import pathlib

    src = pathlib.Path(span_fill.__file__).read_text(encoding="utf-8")
    for dep in ("torch", "transformers", "onnxruntime", "sklearn", "numpy"):
        assert f"import {dep}" not in src
