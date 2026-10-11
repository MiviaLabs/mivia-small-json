"""Rolling multi-turn agent state (spec slice 04: encoder-gaps-closure).

End-to-end through the public classify path: turn-one arguments persist
into the state, turn two classifies from the current utterance plus state
keys only (never concatenated history) within the encoder window budget,
and a pending missing-arguments clarification resolves across turns.
"""

import json

import pytest

from mivia_small_json import agent_state, pipeline, registry
from mivia_small_json.agent_step import register_agent_step_schema
from mivia_small_json.tool_call import register_tool_clarification_schema
from mivia_small_json.validate import validate

TURN_ONE = (
    "step 1: tool call execute_bash; arguments: {\"command\": \"pytest -q\"}; "
    "stdout: 157 passed; stderr: ; return_code: 0; duration_ms: 120"
)
TURN_TWO = "step 2: tool_execution_result: stdout: still green; return_code: 0"
TURN_TWO_ERROR = (
    "step 2: tool_execution_result: stderr: FAILED tests/test_api.py::test_login; "
    "return_code: 1"
)


@pytest.fixture(autouse=True)
def _registry_snapshot():
    before = set(registry.active())
    register_agent_step_schema()
    register_tool_clarification_schema()
    yield
    for name in registry.active():
        if name not in before:
            registry.unregister(name)


def test_new_state_is_json_object_keyed_by_step():
    state = agent_state.new_state()
    assert json.dumps(state)  # JSON-serializable
    assert state["steps"] == {}
    obj, meta = agent_state.classify_turn(TURN_ONE, state, router=None)
    assert obj["type"] == "agent_step"
    assert meta["valid"] is True
    assert validate(obj, "agent_step") == []
    assert "step-1" in state["steps"]
    assert state["steps"]["step-1"]["tool"] == "execute_bash"


def test_two_turn_trajectory_persists_arguments():
    state = agent_state.new_state()
    agent_state.classify_turn(TURN_ONE, state, router=None)
    obj, meta = agent_state.classify_turn(TURN_TWO, state, router=None)
    assert meta["valid"] is True
    assert validate(obj, "agent_step") == []
    # Turn one arguments persist: the result-only follow-up inherits them.
    assert state["steps"]["step-2"]["tool"] == "execute_bash"
    assert state["steps"]["step-2"]["arguments"] == '{"command":"pytest -q"}'
    assert obj["tool"] == "execute_bash"
    assert obj["arguments"] == '{"command":"pytest -q"}'


def test_turn_two_feeds_state_keys_never_history():
    state = agent_state.new_state()
    agent_state.classify_turn(TURN_ONE, state, router=None)
    fed = agent_state.compose_input(TURN_TWO, state)
    assert TURN_ONE not in fed  # no concatenated history
    assert "157 passed" not in fed  # no turn-one values either
    assert TURN_TWO in fed  # the current utterance is intact
    obj, meta = agent_state.classify_turn(TURN_TWO, state, router=None)
    assert meta["window_ok"] is True
    assert meta["fed_chars"] == len(fed)


def test_long_trajectory_stays_within_window_budget():
    state = agent_state.new_state()
    long_stdout = "x" * 5000
    agent_state.classify_turn(
        f"step 1: tool call execute_bash; stdout: {long_stdout}; return_code: 0",
        state,
        router=None,
    )
    obj, meta = agent_state.classify_turn(TURN_TWO, state, router=None)
    # The 5000-char turn-one body never reaches the next turn's input.
    assert meta["window_ok"] is True
    assert meta["valid"] is True
    assert obj["tool"] == "execute_bash"


def test_state_context_drops_oldest_steps_first():
    state = agent_state.new_state()
    for i in range(80):
        agent_state.merge_step(state, {
            "type": "agent_step",
            "step_index": i,
            "tool": "execute_bash",
            "arguments": '{"command":"pytest -q"}',
            "status": "success",
            "return_code": 0,
        })
    fed = agent_state.compose_input(TURN_TWO, state)
    assert len(fed) <= agent_state.WINDOW_CHARS
    assert "step-79" in fed  # latest step keys survive truncation
    assert "step-0" not in fed  # oldest step keys dropped first


def test_pending_clarification_resolves_across_turns():
    state = agent_state.new_state()
    agent_state.merge_step(state, {
        "type": "tool_clarification",
        "tool": "calc_sum",
        "reason": "missing_arguments",
        "missing_parameters": ["b"],
        "clarification_prompt": "Please provide b to use calc_sum.",
    })
    assert agent_state.pending_clarification(state)["missing_parameters"] == ["b"]
    obj, meta = agent_state.classify_turn(
        "step 2: tool call calc_sum; arguments: {\"a\": 3, \"b\": 4}; return_code: 0",
        state,
        router=None,
    )
    assert meta["valid"] is True
    assert agent_state.pending_clarification(state) is None
    assert json.loads(state["steps"]["step-2"]["arguments"]) == {"a": 3, "b": 4}


def test_second_clarification_keeps_accumulated_fills():
    state = agent_state.new_state()
    agent_state.merge_step(state, {
        "type": "tool_clarification",
        "tool": "calc_sum",
        "reason": "missing_arguments",
        "missing_parameters": ["a", "b"],
        "clarification_prompt": "Please provide a, b to use calc_sum.",
    })
    agent_state._resolve_pending(state, {
        "type": "tool_call",
        "name": "calc_sum",
        "arguments": '{"a": 5}',
        "result": "",
    })
    assert agent_state.pending_clarification(state)["filled"] == {"a": 5}
    agent_state.merge_step(state, {
        "type": "tool_clarification",
        "tool": "calc_sum",
        "reason": "missing_arguments",
        "missing_parameters": ["b"],
        "clarification_prompt": "Please provide b to use calc_sum.",
    })
    pend = agent_state.pending_clarification(state)
    assert pend["filled"] == {"a": 5}
    assert pend["missing_parameters"] == ["b"]


def test_pending_clarification_survives_unrelated_turn():
    state = agent_state.new_state()
    agent_state.merge_step(state, {
        "type": "tool_clarification",
        "tool": "calc_sum",
        "reason": "missing_arguments",
        "missing_parameters": ["b"],
        "clarification_prompt": "Please provide b to use calc_sum.",
    })
    agent_state.classify_turn(TURN_TWO, state, router=None)
    assert agent_state.pending_clarification(state)["missing_parameters"] == ["b"]


def test_turn_status_is_its_own_and_never_carried():
    state = agent_state.new_state()
    agent_state.classify_turn(TURN_ONE, state, router=None)
    obj, _ = agent_state.classify_turn(TURN_TWO_ERROR, state, router=None)
    assert obj["status"] == "error"
    assert obj["return_code"] == 1
    assert state["steps"]["step-1"]["status"] == "success"


def test_state_survives_json_round_trip():
    state = agent_state.new_state()
    agent_state.classify_turn(TURN_ONE, state, router=None)
    agent_state.classify_turn(TURN_TWO, state, router=None)
    clone = json.loads(json.dumps(state))
    obj, meta = agent_state.classify_turn(
        "step 3: tool_execution_result: stdout: done; return_code: 0",
        clone,
        router=None,
    )
    assert meta["valid"] is True
    assert obj["tool"] == "execute_bash"


def test_never_raise_contract():
    state = agent_state.new_state()
    agent_state.classify_turn(None, state, router=None)
    agent_state.classify_turn("", state, router=None)
    agent_state.classify_turn(TURN_ONE, None, router=None)
    agent_state.merge_step(state, None)
    agent_state.merge_step(None, None)
    agent_state.merge_step("junk", "junk")
    assert agent_state.compose_input(None, None) == ""
    assert agent_state.pending_clarification(None) is None
    assert agent_state.pending_clarification({}) is None


def test_classify_turn_meta_carries_pipeline_contract():
    state = agent_state.new_state()
    _, meta = agent_state.classify_turn(TURN_ONE, state, router=None)
    for key in ("routed_by", "valid", "retries", "latency_ms"):
        assert key in meta
    assert meta["state_steps"] == 1
    assert meta["state_key"] == "step-1"
    assert pipeline.classify(TURN_ONE, router=None)[1]["valid"] is True
