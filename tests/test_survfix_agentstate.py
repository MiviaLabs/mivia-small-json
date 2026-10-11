"""Survivor fix: the fabricated cross-tool pending resolution (survfix).

Execution-confirmed red state (``.scratch/validate-b/v5_pending.py``):
``agent_state._resolve_pending`` merged ANY turn's arguments blob into the
pending clarification's slots and filed a ``status:"success"`` step for a
tool that was never invoked — turn 1 clarifies ``get_user(name)``, turn 2
is ``tool: delete_file arguments: {"name": "tmp.txt"}`` and the state
records a ``get_user`` success with ``name=tmp.txt``.

The honest contracts pinned here:

- Only a record that belongs to the pending tool may fill its slots
  (``name`` for a ``tool_call``, ``tool`` for every other record).
- A ``tool_call`` for a DIFFERENT tool supersedes the pending turn —
  filed as status ``"cancelled"`` in the state ledger (the ledger's own
  vocabulary, not the validated ``agent_step`` AGENT_STATUSES enum) — an
  honest outcome, never a fabricated success.
- A matching fill files status ``"filled"``: the slots received values,
  which is not an observed execution outcome. ``"success"`` is reserved
  for records that actually observed the tool run.
- Records without a tool identity (and non-tool_call records of another
  tool) never fill or supersede: the pending survives, exactly like any
  other unrelated turn.
"""

from __future__ import annotations

import json

import pytest

from mivia_small_json import agent_state, bootstrap, registry


@pytest.fixture(autouse=True)
def _registry_snapshot():
    """The full shipped schema world, exactly like the confirmed v5 probe
    (.scratch/validate-b/v5_pending.py): with only a subset registered the
    same utterance routes differently and the scenario stops reproducing."""
    before = set(registry.active())
    bootstrap.register_all_schemas()
    yield
    for name in registry.active():
        if name not in before:
            registry.unregister(name)


def _pending(tool, missing):
    state = agent_state.new_state()
    return agent_state.merge_step(state, {
        "type": "tool_clarification",
        "tool": tool,
        "reason": "missing_arguments",
        "missing_parameters": list(missing),
        "clarification_prompt": f"Please provide {', '.join(missing)} "
                                f"to use {tool}.",
    })


# --- the confirmed fabrication -----------------------------------------------


def test_tool_call_for_different_tool_never_fabricates_success():
    """The v5 scenario end to end: turn 1 asks for get_user(name), turn 2
    is a delete_file tool_call whose arguments happen to carry a 'name'
    key. No step may claim get_user succeeded; the pending turn is
    superseded (cancelled), and delete_file keeps its own record."""
    state = _pending("get_user", ["name"])
    obj, meta = agent_state.classify_turn(
        'tool: delete_file arguments: {"name": "tmp.txt"}; return_code: 0',
        state=state, router=None,
    )
    assert obj["type"] == "tool_call" and obj["name"] == "delete_file"
    assert meta["routed_by"] == "rules"

    # The fabrication (red today): get_user was never invoked.
    assert not any(
        v.get("tool") == "get_user" and v.get("status") == "success"
        for v in state["steps"].values()
    ), json.dumps(state["steps"], indent=2)

    # Honest supersede: pending cleared, filed as cancelled — never success.
    assert agent_state.pending_clarification(state) is None
    cancelled = [v for v in state["steps"].values()
                 if v.get("tool") == "get_user"]
    assert cancelled, "the superseded clarification left no ledger entry"
    assert all(v.get("status") == "cancelled" for v in cancelled)

    # The delete_file turn keeps its own record, untouched.
    own = [v for v in state["steps"].values()
           if v.get("tool") == "delete_file"]
    assert own, "the delete_file turn vanished from the state"
    assert json.loads(own[0]["arguments"]) == {"name": "tmp.txt"}


def test_matching_fill_files_filled_never_success():
    """Even when the turn DOES belong to the pending tool, filing the
    resolution as 'success' fabricates an execution outcome nobody
    observed: the slots were filled, that is status 'filled'."""
    state = _pending("calc_sum", ["b"])
    agent_state._resolve_pending(state, {
        "type": "tool_call",
        "name": "calc_sum",
        "arguments": '{"b": 4}',
        "result": "",
    })
    assert agent_state.pending_clarification(state) is None
    entries = [v for v in state["steps"].values()
               if v.get("tool") == "calc_sum"]
    assert entries, "the resolution was not filed"
    assert entries[0]["status"] == "filled"  # red today: "success"
    assert json.loads(entries[0]["arguments"]) == {"b": 4}
    assert not any(v.get("status") == "success"
                   for v in state["steps"].values())


def test_classify_turn_resolves_matching_pending_with_filled_status():
    """The happy path keeps working end to end (same-tool agent_step turn
    supplies the missing slot) — and files the honest 'filled' status."""
    state = _pending("calc_sum", ["b"])
    obj, meta = agent_state.classify_turn(
        'step 2: tool call calc_sum; arguments: {"b": 4}; return_code: 0',
        state=state, router=None,
    )
    assert meta["valid"] is True
    assert obj["tool"] == "calc_sum"
    assert agent_state.pending_clarification(state) is None
    entries = [v for v in state["steps"].values()
               if v.get("tool") == "calc_sum"]
    assert entries, "the matching turn no longer resolves the pending"
    assert entries[-1]["status"] == "filled"  # red today: "success"


# --- the tool-match gate ------------------------------------------------------


def test_agent_step_for_another_tool_never_fills_pending():
    """A non-tool_call record of a different tool is trajectory noise, not
    the answer: the pending survives and nothing is filed for get_user."""
    state = _pending("get_user", ["name"])
    agent_state._resolve_pending(state, {
        "type": "agent_step",
        "tool": "other_tool",
        "arguments": '{"name": "x"}',
        "status": "success",
    })
    pend = agent_state.pending_clarification(state)
    assert pend is not None  # red today: the blob filled 'name'
    assert pend["missing_parameters"] == ["name"]
    assert pend["filled"] == {}
    assert not any(v.get("tool") == "get_user"
                   for v in state["steps"].values())


def test_record_without_tool_identity_never_fills_pending():
    """No tool identity, no fill: an arguments blob alone cannot claim to
    answer a named tool's clarification."""
    state = _pending("get_user", ["name"])
    agent_state._resolve_pending(state, {
        "type": "agent_step",
        "arguments": '{"name": "x"}',
    })
    assert agent_state.pending_clarification(state) is not None
    assert not any(v.get("tool") == "get_user"
                   for v in state["steps"].values())


def test_same_tool_partial_fill_still_accumulates():
    """The pre-existing contract (test_agent_state
    .test_second_clarification_keeps_accumulated_fills) must survive the
    gate: a matching tool_call's partial fill lands in pending['filled']."""
    state = _pending("calc_sum", ["a", "b"])
    agent_state._resolve_pending(state, {
        "type": "tool_call",
        "name": "calc_sum",
        "arguments": '{"a": 5}',
        "result": "",
    })
    pend = agent_state.pending_clarification(state)
    assert pend is not None
    assert pend["filled"] == {"a": 5}
    assert pend["missing_parameters"] == ["b"]


def test_unnamed_pending_is_never_resolved():
    """A clarification without a tool name has nothing to match against;
    no blob may resolve it (and the gate must not raise on it)."""
    state = _pending("", ["name"])
    agent_state._resolve_pending(state, {
        "type": "tool_call",
        "name": "whatever",
        "arguments": '{"name": "x"}',
        "result": "",
    })
    assert agent_state.pending_clarification(state) is not None
