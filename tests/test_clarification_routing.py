"""Ask_followup routing: questions no listed tool can serve ask, not abstain.

STEP 3 (clarification routing): when the tool-match guard abstains on a
tool catalogue whose request is question-shaped (When2Call-style
info-request / cannot-answer), ``pipeline.classify`` must emit
``tool_clarification`` instead of ``no_tool_call``. Real calls and
non-question abstentions are untouched.
"""

import json

import pytest

from mivia_small_json import bootstrap
from mivia_small_json import pipeline as P
from mivia_small_json import registry
from mivia_small_json import tool_match as T
from mivia_small_json.validate import validate


@pytest.fixture(autouse=True)
def _registered_schemas():
    """Register the guard-visible schemas per test without leaking them.

    Module-level registration polluted the global registry for every later
    test file (trust basis, fingerprint, golden fixture). The snapshot is
    restored after each test (same pattern as test_curate_harvested_sessions).
    """
    with registry.temporary_schemas():
        bootstrap.register_all_schemas()
        yield

WEATHER_TOOL = {
    "name": "Weather_1_GetWeather",
    "description": ("Retrieves the weather forecast for a specified city "
                    "on a given date."),
    "parameters": {
        "type": "dict",
        "properties": {
            "city": {"type": "string",
                     "description": "The name of the city."},
            "date": {"type": "string",
                     "description": "The date."},
        },
        "required": ["city", "date"],
    },
}

HYPOT_TOOL = {
    "name": "math.hypot",
    "description": ("Calculate the Euclidean norm, sqrt(sum(squares)), the "
                    "length of the vector from the origin to point (x, y) "
                    "which is the hypotenuse of the right triangle."),
    "parameters": {
        "type": "dict",
        "properties": {
            "x": {"type": "integer",
                  "description": "The x-coordinate value."},
            "y": {"type": "integer",
                  "description": "The y-coordinate value."},
            "z": {"type": "integer",
                  "description": ("Optional. The z-coordinate value. "
                                  "Default is 0.")},
        },
        "required": ["x", "y"],
    },
}

HYPOT_REQUEST = ("Calculate the hypotenuse of a right triangle given the "
                 "lengths of the other two sides as 4 and 5.")

# Forced tool_call route: mirrors production, where the encoder routes a
# catalogue text to tool_call and the guard decides call / clarify / abstain.
CALL_ROUTER = lambda text: "tool_call"  # noqa: E731


def _catalogue(tool: dict, request: str) -> str:
    return ("Tools available:\n" + json.dumps([tool])
            + "\n\nUser request: " + request)


@pytest.fixture(scope="module")
def guard():
    g = T._get_guard()
    assert g is not None, "shipped tool-match guard must load"
    return g


def test_question_no_tool_serves_asks_for_clarification(guard):
    """The STEP 3 behavior: question + guard abstain -> tool_clarification."""
    text = _catalogue(
        WEATHER_TOOL,
        "Can you find me a movie that features Timothy Bateson?")
    _, score = guard.best_match(text)
    assert score < T.THRESHOLD, "fixture must abstain to test the hook"
    obj, meta = P.classify(text, router=CALL_ROUTER)
    assert obj["type"] == "tool_clarification"
    assert obj["reason"] == "underspecified_query"
    assert meta["valid"] is True
    assert validate(obj, "tool_clarification") == []


def test_declaration_no_tool_serves_stays_abstention(guard):
    """A command no tool serves is still a silent abstention, not a question."""
    text = _catalogue(
        WEATHER_TOOL,
        "Execute irreversible deletion of team record 42 immediately.")
    _, score = guard.best_match(text)
    assert score < T.THRESHOLD, "fixture must abstain to test the hook"
    obj, _ = P.classify(text, router=CALL_ROUTER)
    assert obj["type"] == "no_tool_call"


def test_real_call_untouched_by_clarification_routing(guard):
    """Retention guard: a fillable topical match still emits a tool call."""
    text = _catalogue(HYPOT_TOOL, HYPOT_REQUEST)
    _, score = guard.best_match(text)
    assert score >= T.THRESHOLD, "fixture must be a would-be call"
    obj, meta = P.classify(text, router=CALL_ROUTER)
    assert obj["type"] == "tool_call"
    assert obj["name"] == "math.hypot"
    assert meta["valid"] is True


@pytest.mark.parametrize(("question", "expected"), [
    ("Can you find me a movie?", True),
    ("What is the weather Wednesday?", True),
    ("Which city needs the venue?", True),
    # 2026-10-10 review N1: the polite-imperative heads (please/tell/
    # show/help) left the opener regex — an imperative command is not a
    # question form; without a '?' it stays no_tool_call. This row pinned
    # the removed `please` opener; the fix flips it (minimal edit).
    ("Please clarify the date.", False),
    ("How do I serve this?", True),
    ("Show me the list of projects?", True),
    ("Execute irreversible deletion immediately.", False),
    ("Find sculptures near Chicago.", False),
    ("Blue sky thinking synergy paradigm.", False),
    ("", False),
])
def test_ask_followup_pattern_shape(question, expected):
    assert P._is_ask_followup(question) is expected


def test_veto_abstain_asks_for_missing_arguments(guard):
    """The veto companion (fillability-veto-guard §2.1):
    when the guard abstains because the argmax tool's required arguments are
    absent from the request (the no_tool_needed construction), the pipeline
    asks for the missing parameters instead of silently abstaining."""
    stripped = "Calculate the hypotenuse of a right triangle."
    text = _catalogue(HYPOT_TOOL, stripped)
    best_tool, score = guard.best_match(text)
    ok, reason, missing = T.fillability_veto(HYPOT_TOOL, stripped)
    assert score < T.THRESHOLD, "vetoed catalogue must abstain"
    assert not ok and reason == "missing_arguments"
    obj, meta = P.classify(text, router=CALL_ROUTER)
    assert obj["type"] == "tool_clarification"
    assert obj["reason"] == "missing_arguments"
    assert obj["missing_parameters"] == missing
    assert meta["valid"] is True
    assert validate(obj, "tool_clarification") == []
