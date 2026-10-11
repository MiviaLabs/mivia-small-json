"""Survivor fixes 2026-10-10: guard/pipeline agreement + emergency amplifier.

Two confirmed defects this file pins (TDD: written red against the tree,
then fixed):

- M1 (owner-mandated): ``api._guard_downgrade_check`` documented the
  fillability veto as the ONE missing-params oracle but downgraded
  CALL -> CLARIFY unconditionally whenever the extracted arguments missed
  declared required parameters — so pipeline and /v1/guard diverged in
  opposite directions on the same request. Measured pair
  (.scratch/validate-a/v11b_gate_pass.py, 2026-10-10):
  "what is the weather in Austin" vs get_weather(query, city):
  gate 0.9993, veto fillable, span_fill None -> pipeline.classify emits
  the vetted tool_call {} while /v1/guard CLARIFYs missing=[query, city].
  The veto now gates the downgrade: veto-fillable keeps the CALL (the
  pipeline-vetted honest minimum), veto-unfillable clarifies with the
  veto's own missing list. Unconditional re-clarification is the recorded
  BFCL 89.1 -> 25.1 failure mode — never reintroduce it (see
  tests/test_fixval_guard.py and reports/code-review-validation).

- Survivor (architecture amplifier): the pipeline's emergency catch-all
  re-routed through ``rules.route`` and templated the result. For a
  tool_call-routed text (e.g. the AttributeError a malformed
  ``"required": [3]`` catalogue raises in the guard's fillability span)
  that emitted ``template("tool_call")`` — ``{"name": "", ...}`` — a
  served CALL for a nonexistent tool (reproduced end to end at HEAD:
  pipeline record ``tool_call name=""``, /v1/guard ``CALL tool=''``).
  The emergency path now never re-templates a tool schema: an internal
  error degrades to an honest no_tool_call (or tool_clarification when
  no_tool_call is not registered); non-tool schemas keep the template
  escape. The never-raise contract is unchanged.
"""

from __future__ import annotations

import json

import pytest

from mivia_small_json import api, bootstrap, pipeline
from mivia_small_json import tool_match as T
from mivia_small_json.validate import validate

# The measured divergence pair (v11b_gate_pass.py, 2026-10-10). The
# catalogue is nested-BFCL shaped, exactly as the served surface
# normalizes it.
WEATHER_NESTED = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to look up"},
                "city": {"type": "string", "description": "the city"},
            },
            "required": ["query", "city"],
        },
    },
}
DIVERGENT_REQUEST = "what is the weather in Austin"

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

# The confirmed survivor's trigger: a catalogue entry whose required list
# carries a non-string (the guard's fillability span raises AttributeError
# on it at HEAD — the parallel tool_match fix may kill that at its source,
# so the end-to-end pins below assert the world-invariant property "never
# an empty-name CALL", not a specific emission).
MALFORMED_REQUIRED = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string",
                                    "description": "the city"}},
            "required": [3],
        },
    },
}
MALFORMED_REQUEST = "what is the weather in Austin"


def _catalogue_text(tools, request):
    return ("Tools available: " + json.dumps(tools, ensure_ascii=False)
            + "\n\nUser request: " + request)


def _call_out(tool: str, arguments: str, tools_spec) -> dict:
    """A /v1/guard-shaped CALL dict as do_guard builds it pre-downgrade."""
    return {"action": "CALL", "tool": tool,
            "missing_parameters": None, "clarification_prompt": None,
            "record": {"type": "tool_call", "name": tool,
                       "arguments": arguments, "result": ""},
            "meta": {}}


@pytest.fixture(scope="module")
def router():
    return pipeline.load_active_router()


@pytest.fixture(autouse=True)
def _shipped_world():
    bootstrap.register_all_schemas()
    yield


# ---------------------------------------------------------------------------
# M1: the fillability veto is the ONE missing-params oracle
# ---------------------------------------------------------------------------

class TestM1VetoGatedDowngrade:
    def test_veto_unfillable_downgrades_with_veto_missing(self):
        """A CALL whose args miss required params AND whose request the veto
        finds unfillable downgrades to CLARIFY with the veto's missing list."""
        request = "just delete the database"
        ok, _reason, veto_missing = T.fillability_veto(DELETE_DATABASE, request)
        assert not ok, "fixture: the veto must call this request unfillable"
        assert veto_missing == ["name"]
        out = api._guard_downgrade_check(
            _call_out("delete_database", "{}", DELETE_DATABASE),
            [DELETE_DATABASE], request)
        assert out["action"] == "CLARIFY", out
        assert out["missing_parameters"] == ["name"], out
        assert "delete_database" in out["clarification_prompt"], out

    def test_veto_fillable_keeps_the_call(self):
        """The measured divergence, at the decision layer: gate-passing,
        veto-fillable, span-unfilled requests keep the vetted CALL. The
        unconditional {}-downgrade was review P1's pipeline/guard
        divergence."""
        request = DIVERGENT_REQUEST
        ok, reason, missing = T.fillability_veto(WEATHER_NESTED, request)
        assert ok, (reason, missing)
        out = api._guard_downgrade_check(
            _call_out("get_weather", "{}", WEATHER_NESTED),
            [WEATHER_NESTED], request)
        assert out["action"] == "CALL", out
        assert out["tool"] == "get_weather", out

    def test_veto_unfillable_without_missing_falls_back_to_args(
            self, monkeypatch):
        """Defensive limb: an unfillable verdict that names no parameter
        (no current veto path produces it) clarifies with the arg-derived
        list instead of an empty missing set."""
        monkeypatch.setattr(
            T, "fillability_veto",
            lambda tool, req: (False, "missing_arguments", []))
        out = api._guard_downgrade_check(
            _call_out("get_weather", "{}", WEATHER_NESTED),
            [WEATHER_NESTED], DIVERGENT_REQUEST)
        assert out["action"] == "CLARIFY", out
        assert out["missing_parameters"] == ["query", "city"], out

    def test_fully_specified_call_is_untouched(self):
        """Regression guard: args that carry every required parameter never
        reach the veto, exactly as before."""
        out = api._guard_downgrade_check(
            _call_out("delete_database", '{"name": "prod"}', DELETE_DATABASE),
            [DELETE_DATABASE], "delete the production database now")
        assert out["action"] == "CALL", out

    def test_tool_outside_the_catalogue_is_untouched(self):
        """No declared required params -> no oracle consult, no downgrade
        (the CALL names a tool the catalogue does not describe)."""
        out = api._guard_downgrade_check(
            _call_out("unlisted_tool", "{}", [DELETE_DATABASE]),
            [DELETE_DATABASE], "just do it")
        assert out["action"] == "CALL", out

    def test_guard_and_pipeline_agree_on_the_measured_pair(self, router):
        """End to end on the measured pair: pipeline CALL <=> guard CALL,
        same tool. (Before the fix the guard CLARIFYed [query, city] while
        the pipeline CALLed get_weather on the identical request.)"""
        prompt = _catalogue_text([WEATHER_NESTED], DIVERGENT_REQUEST)
        best, score = T._get_guard().best_match(prompt)
        assert best is not None and score >= T.THRESHOLD, (
            f"fixture: the gate must pass (score {score})")
        ok, reason, missing = T.fillability_veto(best, DIVERGENT_REQUEST)
        assert ok, (reason, missing)
        obj, _meta = pipeline.classify(prompt, router=router)
        assert obj.get("type") == "tool_call", obj
        out = api.do_guard(DIVERGENT_REQUEST, [WEATHER_NESTED], router)
        assert out["action"] == "CALL", out
        assert out["tool"] == obj.get("name") == "get_weather", out

    def test_guard_clarifies_when_the_pipeline_clarifies(self, router):
        """The other agreement direction stays: a veto-unfillable request
        CLARIFYs on BOTH surfaces with the same missing parameters."""
        request = "just delete the database"
        prompt = _catalogue_text([DELETE_DATABASE], request)
        best, _score = T._get_guard().best_match(prompt)
        ok, reason, veto_missing = T.fillability_veto(best, request)
        assert not ok, (reason, veto_missing)
        obj, _meta = pipeline.classify(prompt, router=router)
        assert obj.get("type") == "tool_clarification", obj
        out = api.do_guard(request, [DELETE_DATABASE], router)
        assert out["action"] == "CLARIFY", out
        assert (out["missing_parameters"]
                == obj.get("missing_parameters") == veto_missing), out


# ---------------------------------------------------------------------------
# Survivor: the emergency catch-all is never a positive action
# ---------------------------------------------------------------------------

class _RaisingGuard:
    """best_match raises exactly the recorded AttributeError shape."""

    def best_match(self, text):
        raise AttributeError("'int' object has no attribute 'lower'")


class TestEmergencyNeverCallsEmptyTool:
    def test_raising_guard_emits_no_tool_call_never_empty_call(
            self, monkeypatch):
        """Pure amplifier pin (deterministic, model-free): whatever raises
        inside the tool_call branch, the emergency emission is an honest
        no_tool_call — never template('tool_call') with its empty name."""
        monkeypatch.setattr(T, "_get_guard", lambda: _RaisingGuard())
        text = _catalogue_text([MALFORMED_REQUIRED], MALFORMED_REQUEST)
        obj, meta = pipeline._classify_known_schema(text, "tool_call", "model")
        assert obj.get("type") == "no_tool_call", obj
        assert "name" not in obj, obj
        assert validate(obj, "no_tool_call") == [], obj
        # An emergency emission is a fallback, not earned validity — the
        # honest-meta contract is unchanged.
        assert meta["routed_by"] == "template"
        assert meta["valid"] is False

    def test_raising_guard_tool_clarification_schema_never_retemplates(
            self, monkeypatch):
        """Same law when the emergency re-route lands on a tool schema:
        no_tool_call is registered, so the honest abstention wins over the
        tool_clarification template too."""
        monkeypatch.setattr(T, "_get_guard", lambda: _RaisingGuard())
        from mivia_small_json import rules

        monkeypatch.setattr(rules, "route", lambda text: "tool_clarification")
        obj, _meta = pipeline._classify_known_schema("anything", "tool_call",
                                                     "model")
        assert obj.get("type") == "no_tool_call", obj
        assert validate(obj, "no_tool_call") == [], obj

    def test_malformed_required_catalogue_never_serves_empty_call(
            self, router):
        """End to end, the confirmed survivor: the malformed
        "required": [3] catalogue (AttributeError in the guard's
        fillability span at HEAD) must never surface as
        {"action": "CALL", "tool": ""}. World-invariant by design: with the
        current tool_match the pipeline degrades to the emergency
        no_tool_call; if the parallel tool_match source fix lands, the
        pipeline's own vetted ladder answers instead — a named CALL in that
        world is correct. Either way an EMPTY-name CALL is the amplifier
        and stays dead."""
        obj, meta = pipeline.classify(
            _catalogue_text([MALFORMED_REQUIRED], MALFORMED_REQUEST),
            router=router)
        assert obj.get("type") != "tool_call" or obj.get("name"), obj
        if obj.get("type") == "no_tool_call":
            assert validate(obj, "no_tool_call") == [], obj
        out = api.do_guard(MALFORMED_REQUEST, [MALFORMED_REQUIRED], router)
        assert out["action"] != "CALL" or out["tool"], out

    def test_non_tool_schemas_keep_the_template_escape(self, monkeypatch):
        """The escape hatch stays open for non-tool schemas: a raise on an
        issue-routed text still degrades to the issue template."""
        from mivia_small_json import rules

        monkeypatch.setattr(rules, "route", lambda text: "issue")

        def _boom(text, schema):
            raise RuntimeError("extraction exploded")

        monkeypatch.setattr(rules, "extract", _boom)
        obj, meta = pipeline._classify_known_schema("some issue text",
                                                    "issue", "model")
        assert obj.get("type") == "issue", obj
        assert meta["routed_by"] == "template" and meta["valid"] is False
