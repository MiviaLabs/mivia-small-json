"""Fixval regression gates (code-review-validation 2026-10-10, A1/A4/A5/N3/N4).

Each test reproduces a confirmed defect from
reports/code-review-validation-2026-10-10.md against the shipped artifacts
and pins the fixed behavior. The BFCL/retention controls at the bottom keep
the veto-honesty fixes from re-clarifying fully-specified calls (the
recorded 89.1% -> 25.1% failure mode).
"""
from __future__ import annotations

import json

import pytest

from mivia_small_json import api, pipeline, tool_match


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matcher():
    """The shipped split guard, loaded once (naming the artifact on failure)."""
    loaded = tool_match.load_matcher()
    assert loaded is not None, (
        "models/tool-match-*.onnx missing or failed its sha256 check")
    return loaded


@pytest.fixture(scope="module")
def router():
    return pipeline.load_active_router()


def _catalogue_text(tools: list[dict], request: str) -> str:
    return (f"Tools available: {json.dumps(tools, ensure_ascii=False)}"
            f"\n\nUser request: {request}")


DROP_DATABASE = {"name": "drop_database", "description": "Drops a database."}
DELETE_DATABASE = {
    "name": "delete_database",
    "description": "Delete the named database.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string",
                                "description": "the database name"}},
        "required": ["name"],
    },
}
DELETE_ROWS = {
    "name": "delete_rows",
    "description": "Delete rows from a table.",
    "parameters": {
        "type": "object",
        "properties": {
            "table": {"type": "string", "description": "the table"},
            "limit": {"type": "number", "description": "how many rows"},
        },
        "required": ["table", "limit"],
    },
}
ACCOUNT_TOOL = {
    "name": "account_manager",
    "description": "Manage user accounts.",
    "parameters": {
        "type": "object",
        "properties": {"account_ids": {"type": "array",
                                       "description": "the accounts"}},
        "required": ["account_ids"],
    },
}
MATH_ADD = {
    "name": "math.add", "description": "Add two numbers",
    "parameters": {"type": "dict", "properties": {
        "a": {"type": "number", "description": "first addend"},
        "b": {"type": "number", "description": "second addend"}},
        "required": ["a", "b"]}}
WEATHER = {
    "name": "weather.get",
    "description": "Get the weather for a place in a year.",
    "parameters": {
        "properties": {
            "location": {"type": "string", "description": "the place"},
            "year": {"type": "integer", "description": "the year"},
        },
        "required": ["location", "year"],
    },
}


# ---------------------------------------------------------------------------
# A1 — a blank catalogue request must never score a CALLable tool
# ---------------------------------------------------------------------------

def test_a1_base_best_match_blank_request_scores_zero():
    """A parsed catalogue with an empty request can never call anything.

    best_match used to return (None, 1.0) here: the score cleared THRESHOLD,
    the guard never abstained, and the pipeline's rules.extract fallback
    minted `drop_database` with {} out of the catalogue JSON.
    """
    m = tool_match.ToolMatcher(None, None, set())  # the scorer is never reached
    best, score = m.best_match(_catalogue_text([DROP_DATABASE], "   "))
    assert best is None
    assert score == 0.0


def test_a1_split_best_match_blank_request_scores_zero():
    """The SplitToolMatcher override passes the abstention through."""
    split = tool_match.SplitToolMatcher(
        tool_match.ToolMatcher(None, None, set()),
        tool_match.ToolMatcher(None, None, set()))
    best, score = split.best_match(_catalogue_text([DROP_DATABASE], ""))
    assert best is None
    assert score == 0.0


def test_a1_not_a_catalogue_still_passes_through():
    """Pin: plain text is not the guard's domain — the pass-through 1.0
    the measurement scripts rely on must survive the fix."""
    m = tool_match.ToolMatcher(None, None, set())
    assert m.best_score("just a log line") == 1.0


def test_a1_malformed_catalogue_with_a_request_still_passes_through():
    """Pin: a catalogue the guard cannot read must not drop a real call."""
    m = tool_match.ToolMatcher(None, None, set())
    text = "Tools available:\nnot json\n\nUser request: do it"
    assert m.best_score(text) == 1.0


def test_a1_malformed_catalogue_blank_request_scores_zero():
    """The broadened shape: any catalogue malformation that hides a blank
    request used to score the same 1.0 CALL (review Part 2, correction 1)."""
    m = tool_match.ToolMatcher(None, None, set())
    assert m.best_score("Tools available:\nnot json\n\nUser request: ") == 0.0


def test_a1_classify_blank_request_emits_no_tool_call(router):
    """End to end: the composite prompt with a blank request must not mint
    a tool_call named out of the catalogue JSON."""
    obj, _meta = pipeline.classify(
        _catalogue_text([DROP_DATABASE], ""), router)
    assert obj.get("type") != "tool_call", obj


def test_a1_do_guard_blank_request_never_calls(router):
    """End to end on the guard surface: no CALL for a blank request."""
    out = api.do_guard("", [DROP_DATABASE], router)
    assert out["action"] != "CALL", out


# ---------------------------------------------------------------------------
# A4 — fillability-veto holes (classify {} where the guard CLARIFYs)
# ---------------------------------------------------------------------------

def test_a4_content_check_runs_after_numerics_are_satisfied():
    """'please delete 5 rows' satisfies the numeric `limit` but supplies
    nothing for the required string `table` — the content veto must still
    run (it used to be skipped once the numeric slots were filled)."""
    ok, _reason, _missing = tool_match.fillability_veto(
        DELETE_ROWS, "please delete 5 rows")
    assert not ok


def test_a4_content_veto_has_no_eight_word_cap():
    """A 9-word request carrying nothing beyond tool vocabulary and
    fillers must still veto (the old <= 8-word cap let it through)."""
    request = "could you please help me to delete the database"
    assert len(request.split()) > 8
    ok, _reason, _missing = tool_match.fillability_veto(DELETE_DATABASE, request)
    assert not ok


def test_a4_polite_filler_words_are_veto_stopwords():
    """'just'/'simply' are politeness fillers, not slot values."""
    for request in ("just delete the database",
                    "simply delete the database",
                    "can you delete the database"):
        ok, _reason, _missing = tool_match.fillability_veto(
            DELETE_DATABASE, request)
        assert not ok, request


def test_a4_underscored_tool_word_matches_split_vocabulary():
    """The request token `delete_database` keeps its underscore while the
    tool vocabulary split into delete + database — both views must agree."""
    ok, _reason, _missing = tool_match.fillability_veto(
        DELETE_DATABASE, "can you delete_database")
    assert not ok


def test_a4_veto_stays_honest_on_real_content():
    """Controls: requests carrying a value word the tool vocabulary cannot
    explain must stay fillable — the veto may not re-clarify real calls."""
    ok, _reason, _missing = tool_match.fillability_veto(
        DELETE_DATABASE, "delete the production database now")
    assert ok
    ok, _reason, _missing = tool_match.fillability_veto(
        MATH_ADD, "add 4 and 6 for me please")
    assert ok
    ok, _reason, _missing = tool_match.fillability_veto(
        WEATHER, "weather in Paris in 2021 please")
    assert ok


def test_a4_split_guard_abstains_on_holed_requests(matcher):
    """End to end through the shipped split guard: the catalogue decisions
    for the A4 constructions must abstain (score 0.0), not CALL."""
    for tool, request in (
            (DELETE_DATABASE, "just delete the database"),
            (DELETE_DATABASE, "could you please help me to delete the database"),
            (DELETE_ROWS, "please delete 5 rows"),
    ):
        best_tool, score = matcher.best_match(_catalogue_text([tool], request))
        assert score < tool_match.THRESHOLD, (tool["name"], request, score)


# ---------------------------------------------------------------------------
# A5 — false clarifications in the same veto
# ---------------------------------------------------------------------------

def test_a5_numeric_name_hints_are_segment_bounded():
    """'count' inside 'account' must not read as a numeric hint."""
    assert tool_match._is_veto_numeric("account", {}) is False
    assert tool_match._is_veto_numeric("account_ids", {}) is False
    assert tool_match._is_veto_numeric("count", {}) is True
    assert tool_match._is_veto_numeric("item_count", {}) is True
    assert tool_match._is_veto_numeric("num_items", {}) is True


def test_a5_array_typed_params_are_not_numeric():
    assert tool_match._is_veto_numeric(
        "account_ids", {"account_ids": {"type": "array"}}) is False


def test_a5_array_catalogue_request_is_not_clarified():
    """The observed false clarification: array-typed account_ids read as
    numeric ('count' in 'account'), so 'delete accounts alice and bob'
    was asked back for account_ids."""
    ok, reason, missing = tool_match.fillability_veto(
        ACCOUNT_TOOL, "delete accounts alice and bob")
    assert ok, (reason, missing)


def test_a5_two_word_short_circuit_counts_numeric_slots_only():
    """'delete production' supplies the string slot `name`; the <= 2-word
    short-circuit exists for numeric-slot tools ('10 20') and must not
    fire when the required slots are strings."""
    ok, reason, missing = tool_match.fillability_veto(
        DELETE_DATABASE, "delete production")
    assert ok, (reason, missing)
    # The numeric case keeps its protection: one number for two slots.
    two_nums = MATH_ADD
    ok, reason, missing = tool_match.fillability_veto(two_nums, "10")
    assert not ok and reason == "underspecified_query"


# ---------------------------------------------------------------------------
# N3 — the guard's Platt calibrator must not map the intent head's scores
# ---------------------------------------------------------------------------

B77_DESCS = {
    "card_arrival": "card arrival (card arrival)",
    "pending_transfer": "pending transfer (pending transfer)",
    "account_blocked": "account blocked (account blocked)",
}


def test_n3_intent_select_serves_the_raw_softmax_en():
    _assert_intent_raw_scale(
        "I am still waiting on my card, when does it arrive?",
        tool_match.load_intent_rank())


def test_n3_intent_select_serves_the_raw_softmax_mul():
    _assert_intent_raw_scale(
        "Wann kommt meine Karte endlich an?",
        tool_match.load_intent_rank_mul())


def _assert_intent_raw_scale(text, head):
    """select_intent must report the head's own raw softmax.

    The calibrator was fitted (and sha-pinned) to the guard's rank artifact;
    applied to the intent head it served 0.74-0.93 where the raw softmax is
    0.33-0.57 (review N3). The loader's contract: a foreign sha must serve
    the raw confidence.
    """
    assert head is not None, "intent head artifact missing"
    raw_key, raw_conf = head.select(text, B77_DESCS, calibrated=False)
    out = tool_match.select_intent(text, B77_DESCS)
    assert out["choice"] == raw_key
    assert out["confidence"] == pytest.approx(round(float(raw_conf), 4))
    cal_key, cal_conf = head.select(text, B77_DESCS, calibrated=True)
    if tool_match.load_select_calibrator() is not None \
            and abs(cal_conf - raw_conf) > 1e-9:
        assert out["confidence"] != pytest.approx(cal_conf, abs=1e-6), (
            "select_intent served the guard's Platt-calibrated confidence")


# ---------------------------------------------------------------------------
# N4 — nameless catalogue entries are a caller error, never an empty CALL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tools", [
    [{"description": "book a restaurant table"}],
    [{"type": "function"}],
    [{"function": {"description": "no name inside"}}],
    [""],
    {"": "book a restaurant table"},
])
def test_n4_normalize_tools_rejects_nameless_entries(tools):
    with pytest.raises(ValueError):
        api._normalize_tools(tools)


def test_n4_guard_endpoint_rejects_nameless_catalogue(router):
    """End to end: the served guard surface maps the nameless catalogue to
    an explicit 422 — never a 200 {"action": "CALL", "tool": ""}."""
    from starlette.testclient import TestClient

    client = TestClient(base_url="http://127.0.0.1",
                        app=api.create_app(token="", router=router))
    resp = client.post("/v1/guard", json={
        "request": "book a table for two",
        "tools": [{"description": "book a restaurant table"}]})
    assert resp.status_code == 422, resp.json()
    assert "name" in resp.json().get("error", "")


def test_n4_named_catalogues_still_normalize():
    """Controls: every documented accepted shape keeps working."""
    out = api._normalize_tools({"weather": "Get the weather"})
    assert out == [{"type": "function",
                    "function": {"name": "weather",
                                 "description": "Get the weather"}}]
    out = api._normalize_tools(["ping"])
    assert out == [{"type": "function", "function": {"name": "ping"}}]
    spec = {"type": "function",
            "function": {"name": "f", "description": "d"}}
    assert api._normalize_tools([spec]) == [spec]
    flat = {"name": "f", "parameters": {"type": "object",
                                        "properties": {}}}
    assert api._normalize_tools([flat])[0]["function"]["name"] == "f"
