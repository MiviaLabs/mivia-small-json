"""Tests for the agent_step trajectory schema, routing, and extraction."""

import pytest

from mivia_small_json import pipeline, registry, rules
from mivia_small_json.agent_step import (
    AGENT_STEP_EXAMPLES,
    extract_agent_step,
    register_agent_step_schema,
)
from mivia_small_json.schemas import AGENT_ROLES, AGENT_STATUSES
from mivia_small_json.validate import validate

SUCCESS_TRACE = (
    "step 3: tool call execute_bash; arguments: {\"command\": \"pytest -q\"}; "
    "stdout: 157 passed; stderr: ; return_code: 0; duration_ms: 120"
)
ERROR_TRACE = (
    "step 4: tool call execute_bash; arguments: {'command': 'pytest tests/test_api.py'}; "
    "stdout: ; stderr: FAILED tests/test_api.py::test_login; return_code: 1; "
    "duration_ms: 845; AssertionError: assert False"
)
TIMEOUT_TRACE = (
    "step 5: action: execute_bash; arguments: {\"command\": \"pytest -q\"}; "
    "stdout: ; stderr: execution timed out after 30000ms; "
    "return_code: 124; duration_ms: 30000"
)
# The committed harvest raw opens with the session-positional step_index and
# the derived agent_role (scripts/harvest_agent_sessions.py::_raw_for_step).
CANONICAL_RAW = (
    "step_index: 3; agent_role: tool; tool: read_file; "
    "arguments: {\"path\": \"src/app.py\"}; stdout: 12 lines; "
    "return_code: 0; status: success"
)


@pytest.fixture(autouse=True)
def _registry_snapshot():
    # Restore-on-exit is the suite-wide autouse hermetic fixture
    # (tests/conftest.py); this fixture only sets the world up.
    register_agent_step_schema()
    yield


def test_extract_agent_step_stdout_keeps_a_boundary_keyword_inside_content():
    """A captured stdout that mentions `return_code`/`stderr` must not be
    truncated at the mention: the field boundary requires the `;` separator.
    The old optional-`;` lookahead cut stdout short on 10 dev rows."""
    text = ('tool: execute_bash; arguments: {"command": "grep return_code x"}; '
            'stdout: checked return_code and stderr handling; return_code: 0')
    obj = extract_agent_step(text)
    assert obj["stdout"] == "checked return_code and stderr handling"
    assert obj["return_code"] == 0


def test_extract_agent_step_tool_field_beats_content_match():
    """A loose `tool call X` inside captured stdout must not beat the real
    `tool: X` field: earliest match wins, not pattern priority. A Go source
    line (`tool call arguments failed`) otherwise won on 6 dev rows."""
    text = ('tool: read_file; arguments: {"path": "x.go"}; '
            'stdout: errors.New("agentloop: tool call arguments failed"); '
            'return_code: 0')
    assert extract_agent_step(text)["tool"] == "read_file"


def test_agent_step_registration_and_enums():
    assert "agent_step" in registry.active()
    spec = registry.resolve("agent_step")
    assert spec["type"]["const"] == "agent_step"
    assert spec["agent_role"]["enum"] == AGENT_ROLES
    assert spec["status"]["enum"] == AGENT_STATUSES
    assert AGENT_ROLES == ("assistant", "tool", "user", "environment")
    assert AGENT_STATUSES == ("success", "error", "timeout", "interrupted")


def test_agent_step_examples_are_strictly_valid():
    assert len(AGENT_STEP_EXAMPLES) >= 3
    for gold in AGENT_STEP_EXAMPLES:
        assert validate(gold, "agent_step") == []


def test_agent_step_registration_is_idempotent():
    before = registry.active()
    register_agent_step_schema()
    assert registry.active() == before


def test_agent_step_strong_routing():
    assert rules.route("step 3: tool call execute_bash -- run the tests") == "agent_step"
    assert rules.route("tool_execution_result: stdout=ok return_code=0") == "agent_step"
    assert rules.route("command finished with exit code 1 after pytest") == "agent_step"
    assert rules.route("execution timed out after 30000ms") == "agent_step"
    assert rules.route("action: execute_bash with pytest args") == "agent_step"


def test_agent_step_weak_markers_score():
    scores = rules.route_scores("stdout: ok\nstderr: empty\nreturn_code: 0")
    assert scores["agent_step"] >= 3


def test_extract_success_step():
    obj = extract_agent_step(SUCCESS_TRACE)
    assert obj["type"] == "agent_step"
    assert obj["step_index"] == 3
    assert obj["tool"] == "execute_bash"
    assert obj["arguments"] == '{"command":"pytest -q"}'
    assert obj["stdout"] == "157 passed"
    assert obj["return_code"] == 0
    assert obj["duration_ms"] == 120
    assert obj["status"] == "success"
    assert validate(obj, "agent_step") == []


def test_extract_repairs_pseudo_json_arguments():
    obj = extract_agent_step(ERROR_TRACE)
    assert obj["arguments"] == '{"command":"pytest tests/test_api.py"}'
    assert obj["return_code"] == 1
    assert obj["status"] == "error"
    assert obj["error_type"] == "AssertionError"
    assert obj["error_message"] == "assert False"
    assert validate(obj, "agent_step") == []


def test_extract_timeout_status():
    obj = extract_agent_step(TIMEOUT_TRACE)
    assert obj["status"] == "timeout"
    assert obj["duration_ms"] == 30000
    assert validate(obj, "agent_step") == []


def test_extract_interrupted_status():
    obj = extract_agent_step("step 6: tool call execute_bash; return_code: 130; "
                             "stderr: interrupted by user")
    assert obj["status"] == "interrupted"
    assert validate(obj, "agent_step") == []


def test_extract_sanitizes_ansi_and_carriage_returns():
    raw = ("step 7: tool call execute_bash; "
           "stdout: \x1b[32mok\x1b[0m; return_code: 0")
    obj = extract_agent_step(raw)
    assert obj["stdout"] == "ok"
    assert "\x1b" not in obj["stdout"]
    assert validate(obj, "agent_step") == []


def test_extract_defaults_are_valid():
    obj = extract_agent_step("step 1: tool call something")
    assert validate(obj, "agent_step") == []
    assert obj["step_index"] == 1
    assert obj["status"] == "success"


def test_extract_canonical_step_index_and_agent_role():
    """The committed raw renders `step_index: N; agent_role: X;` first.

    `step_index` is session-positional and `agent_role` is derived, so both
    live only in the raw prefix; the extractor must recover them exactly.
    `step_index: 4` blocks the generic `step\\s*...` shape on the `_`, and
    `agent_role: tool` must not be confused with the `tool:` field.
    """
    obj = extract_agent_step(CANONICAL_RAW)
    assert obj["step_index"] == 3
    assert obj["agent_role"] == "tool"
    assert obj["tool"] == "read_file"
    assert obj["stdout"] == "12 lines"
    assert obj["return_code"] == 0
    assert obj["status"] == "success"
    assert validate(obj, "agent_step") == []


@pytest.mark.parametrize("role", ["assistant", "tool", "user", "environment"])
def test_extract_agent_role_all_enum_values(role):
    obj = extract_agent_step(
        f"step_index: 4; agent_role: {role}; tool: x; arguments: {{}}; "
        "return_code: 0; status: success")
    assert obj["step_index"] == 4
    assert obj["agent_role"] == role


def test_classify_agent_step_end_to_end():
    obj, meta = pipeline.classify(ERROR_TRACE, router=None)
    assert obj["type"] == "agent_step"
    assert meta["valid"] is True
    assert validate(obj, "agent_step") == []


def test_classify_dump_keeps_agent_step_atomic():
    text = ERROR_TRACE + "\nmultiline stderr continuation"
    records = pipeline.classify_dump(text, router=None)
    assert len(records) == 1
    assert records[0][0]["type"] == "agent_step"


def test_ten_schemas_registered():
    from mivia_small_json import bootstrap
    from mivia_small_json.diagnostic import register_diagnostic_schema
    from mivia_small_json.diff_comment import register_diff_and_review_schemas
    from mivia_small_json.tool_call import (register_no_tool_call_schema,
                                            register_tool_call_schema)

    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()
    expected = {"issue", "log_event", "commit", "pull_request", "tool_call",
                "no_tool_call", "diagnostic", "diff", "review_comment",
                "agent_step"}
    assert expected <= set(registry.active())
