"""Deterministic-only family admission (2026-10-10, owner-approved).

string-param-span-fill, section "DETERMINISTIC-ONLY
FAMILY ADMISSION": the string-scalar family is admitted WITHOUT the
learned artifact; the deterministic key-delimited/alignment ladders fill
exact-when-they-fire, and emission requires EVERY required slot to get a
verbatim copied span. A row the ladders cannot complete with no extractor
is NOT a partial and never a new clarification — it stays on today's
decision-layer path (fillability veto -> clarification or the ``{}``
honest minimum). With the artifact present the learned leg is unchanged
(L2 byte parity, measured by the .scratch/ladderfill/ instruments).

The no-artifact world is forced through the loader's own seam — the
negative-cache globals the with-artifact tests inject through — so every
pin here runs on any checkout, staged candidate or not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mivia_small_json import pipeline, registry, span_fill
from mivia_small_json.tool_call import (
    register_no_tool_call_schema,
    register_tool_call_schema,
    register_tool_clarification_schema,
)

ROOT = Path(__file__).resolve().parents[1]

STR_TOOL = {"name": "restart_service", "description":
            "Restart a systemd service on a named host.",
            "parameters": {"properties": {
                "service": {"type": "string",
                            "description": "The systemd service unit."},
                "host": {"type": "string",
                         "description": "The target host."}},
                "required": ["service", "host"]}}
ENUM_TOOL = {"name": "deploy_release", "description":
             "Ship a release to a target environment.",
             "parameters": {"properties": {
                 "release": {"type": "string"},
                 "environment": {"type": "string",
                                 "enum": ["staging", "production",
                                          "canary"]}},
                 "required": ["release", "environment"]}}
NUM_TOOL = {"name": "calculate_sum", "description":
            "Calculate the sum of two numbers a and b and return the result.",
            "parameters": {"properties": {
                "a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"]}}
MIX_TOOL = {"name": "scale_pool", "description":
            "Resize a database connection pool.",
            "parameters": {"properties": {
                "pool": {"type": "string"}, "size": {"type": "integer"}},
                "required": ["pool", "size"]}}
BOOL_TOOL = {"name": "toggle", "description": "Flip a flag.",
             "parameters": {"properties": {
                 "flag": {"type": "boolean"}}, "required": ["flag"]}}


@pytest.fixture()
def no_artifact():
    """Force the learned extractor absent through the loader's seam (the
    same globals the with-artifact tests inject through, in the negative
    direction)."""
    saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
    span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = None, True
    yield
    span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = saved


def _catalogue(tool, request):
    return (f"Tools available:\n[{json.dumps(tool)}]"
            f"\n\nUser request: {request}")


# --- admission ---------------------------------------------------------------

def test_string_family_admitted_without_artifact(no_artifact):
    assert span_fill.load_spanfill_extractor() is None
    assert span_fill.family_for(STR_TOOL) == "string"
    assert span_fill.family_for(ENUM_TOOL) == "string"


def test_mixed_and_boolean_stay_unadmitted_without_artifact(no_artifact):
    assert span_fill.family_for(MIX_TOOL) is None
    assert span_fill.family_for(BOOL_TOOL) is None


def test_numeric_family_unchanged_without_artifact(no_artifact):
    assert span_fill.family_for(NUM_TOOL) == "numeric"
    assert span_fill.fill_arguments(
        NUM_TOOL, "calculate sum of 10 and 20") == ({"a": 10, "b": 20}, [])


# --- ladder fill: every required slot gets a verbatim copied span ------------

def test_ladder_fill_calls_with_verbatim_arguments(no_artifact):
    filled, missing = span_fill.fill_arguments(
        STR_TOOL, "service=nginx host=staging restart now")
    assert (filled, missing) == ({"service": "nginx", "host": "staging"}, [])


def test_ladder_fill_emits_declared_enum_form(no_artifact):
    filled, missing = span_fill.fill_arguments(
        ENUM_TOOL, "release=2026.40 environment=staging")
    assert (filled, missing) == ({"release": "2026.40",
                                  "environment": "staging"}, [])


def test_no_artifact_served_call_carries_verbatim_arguments(no_artifact):
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()
    obj, meta = pipeline.classify(
        _catalogue(STR_TOOL, "service=nginx host=staging restart now"))
    assert obj["type"] == "tool_call", obj
    assert json.loads(obj["arguments"]) == {"service": "nginx",
                                            "host": "staging"}
    assert meta["valid"] is True


# --- NOT a partial, NOT a new clarification ----------------------------------

def test_nonfillable_request_stays_unhandled(no_artifact):
    """The ladders cannot copy any span for either slot: the family hands
    the row back unhandled — today's decision-layer path owns it (here
    the veto's {} honest minimum), never a partial, never a new
    clarification."""
    assert span_fill.fill_arguments(
        STR_TOOL, "restart nginx on staging") == (None, [])


def test_blocked_slot_stays_unhandled_without_artifact(no_artifact):
    """A present-but-invalid own value (enum mismatch) leaves the slot
    open; with no extractor the whole row is unhandled, not a partial."""
    assert span_fill.fill_arguments(
        ENUM_TOOL, "release=2026.40 environment=prod") == (None, [])


def test_served_nonfillable_row_keeps_the_honest_minimum(no_artifact):
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()
    obj, meta = pipeline.classify(
        _catalogue(STR_TOOL, "restart nginx on staging"))
    assert obj["type"] == "tool_call", obj
    assert json.loads(obj["arguments"]) == {}
    assert meta["valid"] is True


def test_served_veto_refusal_still_clarifies(no_artifact):
    """A row the fillability veto refuses takes today's clarification path
    with today's reason — the admission adds no clarification and removes
    none."""
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()
    assert "tool_clarification" in registry.active()
    obj, meta = pipeline.classify(
        _catalogue(STR_TOOL, "restart the service"))
    assert obj["type"] == "tool_clarification", obj
    assert obj["reason"] == "underspecified_query"
    assert meta["valid"] is True


# --- with the artifact present, the learned leg must not move ----------------

def test_with_artifact_ladder_complete_rows_identical():
    """L2 at unit level: on rows the deterministic ladders complete, the
    with-artifact fill is EXACTLY the no-artifact fill (the learned leg
    never touches a ladder-complete row)."""
    if span_fill.load_spanfill_extractor() is None:
        pytest.skip("no spanfill artifact on this checkout")
    rows = [
        (STR_TOOL, "service=nginx host=staging restart now"),
        (ENUM_TOOL, "release=2026.40 environment=staging"),
        (NUM_TOOL, "calculate sum of 10 and 20"),
    ]
    for tool, request in rows:
        with_artifact = span_fill.fill_arguments(tool, request)
        saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = None, True
        try:
            without = span_fill.fill_arguments(tool, request)
        finally:
            span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = saved
        assert with_artifact == without, (tool, request)
