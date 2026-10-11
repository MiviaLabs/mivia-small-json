# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Rolling multi-turn agent state (spec slice 04: encoder-gaps-closure).

A persistent JSON object keyed by step carries trajectory progress across
turns::

    {"steps": {"step-1": {"tool": ..., "arguments": ..., "status": ...}},
     "pending": {"tool": ..., "missing_parameters": [...], "filled": {...}} | None}

Each turn merges its extraction into state; the next turn is classified
from the current utterance plus state keys only -- never concatenated
history -- so long trajectories fit the encoder window
(``encoder.MAX_LEN`` tokens, estimated at four chars per token with half
the window reserved for the utterance).

Reuse, not reinvention:

- per-turn extraction goes through the existing ``agent_step`` schema
  module via the public ``pipeline.classify`` path;
- merged multi-turn arguments are assembled with
  ``span_fill.arguments_json`` and follow-up blobs are parsed with the
  ``tool_call.repair_arguments`` repair;
- leftovers keep the existing missing-arguments clarification outcome
  (``pending``), resolved when a later turn supplies the slots.

Carry-forward is deliberately narrow: an ``agent_step`` turn with no
tool mention inherits the latest prior tool, and a turn with default
arguments inherits the latest prior non-default arguments. ``status``
is never carried -- each turn's outcome is its own, so a prior success
cannot mask a later error. The never-raise contract holds throughout:
every helper degrades to an empty context / unchanged state.
"""

from __future__ import annotations

import json

from . import pipeline
from .encoder import MAX_LEN
from .span_fill import arguments_json
from .tool_call import repair_arguments
from .validate import repair, template, validate

# Conservative chars-per-token estimate for window accounting.
WINDOW_CHARS = MAX_LEN * 4
# State keys may use at most half the window; the utterance owns the rest.
MAX_CONTEXT_CHARS = WINDOW_CHARS // 2
# One step's key line never exceeds this, so a pathological slot list
# cannot crowd out every other step.
MAX_STEP_LINE = 160

# Summary keys kept per step: values that accumulate (tool, arguments)
# plus the outcome marker (status, read-only: never carried forward).
_SUMMARY_KEYS = ("tool", "arguments", "status")

_DEFAULT_ARGUMENTS = "{}"


def new_state() -> dict:
    """A fresh rolling state: steps keyed by step, no pending clarification."""
    return {"steps": {}, "pending": None}


def _steps_of(state) -> dict:
    if isinstance(state, dict) and isinstance(state.get("steps"), dict):
        return state["steps"]
    return {}


def _writable_steps(state: dict) -> dict:
    """The state's step map, created when absent. Never raises."""
    try:
        steps = state.get("steps")
        if not isinstance(steps, dict):
            steps = {}
            state["steps"] = steps
        return steps
    except Exception:
        return {}


def _scalar(value):
    """Keep JSON-scalar summaries; anything else degrades to its string."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        try:
            return str(value)
        except Exception:
            return ""


def _step_key(obj: dict, steps: dict) -> str:
    if obj.get("type") == "agent_step":
        try:
            idx = int(obj.get("step_index", 0))
        except (TypeError, ValueError):
            idx = 0
        # Step-less agent_steps all default to index 0 and used to collide
        # on "step-0", each merge overwriting the last (review §10). Fall
        # back to the monotonic turn key when 0 is the default AND taken.
        if idx != 0 or "step-0" not in steps:
            return f"step-{idx}"
    taken = sum(1 for k in steps if k.startswith("turn-"))
    return f"turn-{taken + 1}"


def _merge_step(state, obj) -> tuple[dict, str | None]:
    """Merge core of :func:`merge_step`; returns the state and the key the
    record was actually filed under (None when nothing was filed)."""
    if not isinstance(state, dict):
        state = new_state()
    if not isinstance(obj, dict) or "type" not in obj:
        return state, None
    steps = _writable_steps(state)
    if obj.get("type") == "tool_clarification":
        missing = obj.get("missing_parameters")
        missing = ([str(m) for m in missing]
                    if isinstance(missing, list) else [])
        if missing:
            tool = str(obj.get("tool", ""))
            prev = state.get("pending")
            keep = {}
            if (isinstance(prev, dict) and prev.get("tool") == tool
                    and tool
                    and isinstance(prev.get("filled"), dict)):
                # Same tool clarifying again: keep every accumulated
                # fill. Already-filled slots are facts about supplied
                # values even when the new report no longer lists
                # them; the resolver narrows the missing list and the
                # final resolution files all fills together.
                keep = dict(prev["filled"])
            state["pending"] = {
                "tool": tool,
                "missing_parameters": missing,
                "filled": keep,
            }
        return state, None
    key = _step_key(obj, steps)
    summary = {k: _scalar(obj.get(k, "")) for k in _SUMMARY_KEYS}
    if obj.get("type") == "tool_call" and not summary.get("tool"):
        # tool_call records carry the tool in "name"; the summary and the
        # carry-forward read "tool" (review §10: the name never inherited).
        summary["tool"] = _scalar(obj.get("name", ""))
    steps[key] = summary
    return state, key


def merge_step(state, obj):
    """Merge one classified record into ``state``. Never raises.

    ``agent_step`` records file under ``step-<index>``; every other typed
    record files under a monotonic ``turn-<n>`` key. A
    ``tool_clarification`` record additionally becomes the pending
    missing-arguments entry. Untyped or non-dict records are ignored.
    Returns the (possibly fresh) state for chaining.
    """
    try:
        state, _ = _merge_step(state, obj)
        return state
    except Exception:
        return state if isinstance(state, dict) else new_state()


def pending_clarification(state):
    """The unresolved missing-arguments entry, or None. Never raises."""
    try:
        pend = state.get("pending") if isinstance(state, dict) else None
        if isinstance(pend, dict) and pend.get("missing_parameters"):
            return pend
        return None
    except Exception:
        return None


def _render_line(key: str, summary: dict) -> str:
    keys = [k for k in _SUMMARY_KEYS if k in summary]
    line = f"{key}({','.join(keys)})"
    if len(line) > MAX_STEP_LINE:
        line = line[:MAX_STEP_LINE]
    return line


def state_context(state, room: int = MAX_CONTEXT_CHARS) -> str:
    """Compact state-keys rendering for the next turn's input.

    Keys only -- no stored values, no raw history -- so the rendering is
    inert under the field extractors. Newest steps first while ``room``
    allows, then restored to chronological order, so truncation drops
    the oldest steps first. Never raises.
    """
    try:
        steps = _steps_of(state)
        if not steps or room is None or room <= 0:
            return ""
        room = int(room)
        lines = [_render_line(k, v) for k, v in steps.items()
                 if isinstance(v, dict)]
        # Newest first: the tail of insertion order fills the budget.
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            cost = len(line) + (1 if kept else 0)
            if used + cost > room:
                break
            kept.append(line)
            used += cost
        kept.reverse()
        pend = pending_clarification(state)
        tail = " pending(tool,missing)" if pend else ""
        body = " ".join(kept)
        if tail and len(body) + len(tail) <= room:
            body = (body + tail) if body else tail.strip()
        return f"prior {body}" if body else ""
    except Exception:
        return ""


def compose_input(text, state=None) -> str:
    """Current utterance plus state keys, within the window budget.

    The utterance passes through intact; only the state-keys suffix is
    truncated (oldest steps first). Never raises.
    """
    try:
        utterance = "" if text is None else (
            text if isinstance(text, str) else str(text))
    except Exception:
        utterance = ""
    try:
        room = WINDOW_CHARS - len(utterance) - 1
        ctx = state_context(state, room=room)
    except Exception:
        ctx = ""
    return f"{utterance}\n{ctx}" if ctx else utterance


def _latest_prior(steps: dict, skip: str) -> dict:
    for key in reversed(list(steps)):
        if key != skip and isinstance(steps[key], dict):
            prior = steps[key]
            if prior.get("tool") or prior.get("arguments"):
                return prior
    return {}


def _carry_forward(state, key: str, obj: dict) -> None:
    """Inherit tool/arguments from the latest prior step. Never raises.

    Applies only to ``agent_step`` turns with an empty tool mention or
    default arguments. On the (unexpected) case the inherited values fail
    strict validation, the carry is reverted and the classified record
    stands -- a turn is never template-escaped because of state.
    """
    try:
        if not isinstance(obj, dict) or obj.get("type") != "agent_step":
            return
        steps = _steps_of(state)
        prior = _latest_prior(steps, key)
        if not prior:
            return
        carried = False
        orig_tool = obj.get("tool", "")
        orig_args = obj.get("arguments", _DEFAULT_ARGUMENTS)
        if not orig_tool and prior.get("tool"):
            obj["tool"] = prior["tool"]
            carried = True
        prior_args = prior.get("arguments", "")
        if orig_args in ("", _DEFAULT_ARGUMENTS) and prior_args not in (
                "", _DEFAULT_ARGUMENTS, None):
            obj["arguments"] = prior_args
            carried = True
        if not carried:
            return
        errors = validate(obj, "agent_step")
        retries = 0
        while errors and retries < 2:
            obj.update(repair(obj, errors, "agent_step"))
            errors = validate(obj, "agent_step")
            retries += 1
        if errors:  # inherited values do not verify: revert, keep the turn
            obj["tool"] = orig_tool
            obj["arguments"] = orig_args
        steps[key] = {k: _scalar(obj.get(k, "")) for k in _SUMMARY_KEYS}
    except Exception:
        pass


def _record_tool(obj: dict) -> str:
    """The turn record's tool identity: ``name`` for a ``tool_call``,
    ``tool`` for every other record. Empty when the record names none."""
    tool = obj.get("name") if obj.get("type") == "tool_call" else obj.get("tool")
    return tool.strip() if isinstance(tool, str) else ""


def _resolve_pending(state, obj) -> None:
    """Resolve the pending clarification from a turn that belongs to it.

    Only a record whose tool identity matches the pending tool may fill
    its slots — a different tool's arguments blob is that tool's data,
    never the pending tool's answer (survfix 2026-10-10: a delete_file
    turn's ``{"name": ...}`` used to file a fabricated ``get_user``
    success). The gate:

    - a ``tool_call`` for a DIFFERENT tool supersedes the pending turn:
      it is filed as status ``"cancelled"`` in the state ledger (the
      ledger's own vocabulary, not the validated agent_step enum) with
      whatever partial fills existed — an honest outcome, never success;
    - any other non-matching record (including one with no tool identity)
      leaves the pending untouched, like any unrelated turn;
    - a matching fill files status ``"filled"``: the slots received
      values, which is not an observed execution outcome — ``"success"``
      stays reserved for records that actually observed the tool run.

    Blob keys matching missing slots are merged (assembled with the
    span-fill ``arguments_json``); a fully-filled pending clears and its
    resolution is filed as a step entry. Never raises.
    """
    try:
        pend = pending_clarification(state)
        if pend is None or not isinstance(obj, dict):
            return
        pend_tool = str(pend.get("tool") or "").strip()
        if not pend_tool:
            return  # an unnamed clarification has nothing to match against
        rec_tool = _record_tool(obj)
        if rec_tool != pend_tool:
            if obj.get("type") == "tool_call" and rec_tool:
                # The trajectory moved on to a different tool: the pending
                # clarification will not be answered by this turn. Retire
                # it honestly instead of letting it dangle or, worse,
                # absorbing this turn's arguments as its own.
                steps = _writable_steps(state)
                key = _step_key({"type": "tool_call"}, steps)
                steps[key] = {
                    "tool": pend_tool,
                    "arguments": arguments_json(pend.get("filled") or {}),
                    "status": "cancelled",
                }
                state["pending"] = None
            return
        blob = obj.get("arguments")
        if not isinstance(blob, str) or not blob.strip():
            return
        try:
            data = json.loads(repair_arguments(blob))
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict):
            return
        filled = dict(pend.get("filled") or {})
        for slot in pend["missing_parameters"]:
            if slot in data:
                try:
                    json.dumps(data[slot])
                    filled[slot] = data[slot]
                except (TypeError, ValueError):
                    continue
        missing = [m for m in pend["missing_parameters"] if m not in filled]
        if not missing:
            state["pending"] = None
            steps = _writable_steps(state)
            key = _step_key({"type": "tool_call"}, steps)
            steps[key] = {
                "tool": pend_tool,
                "arguments": arguments_json(filled),
                "status": "filled",
            }
        else:
            pend["filled"] = filled
            pend["missing_parameters"] = missing
    except Exception:
        pass


def classify_turn(
    text,
    state=None,
    router=None,
    level_extractor=None,
    component_extractor=None,
    implicit_severity_extractor=None,
    log_pool=None,
    pool_inbox=None,
) -> tuple[dict, dict]:
    """Classify one trajectory turn and roll it into ``state``.

    The classifier sees the current utterance plus state keys only
    (:func:`compose_input`); the merged extraction accumulates in
    ``state`` (mutated in place). Returns ``(obj, meta)`` on the
    ``pipeline.classify`` contract with ``state_key``, ``state_steps``,
    ``fed_chars`` and ``window_ok`` added to ``meta``. Never raises.
    """
    if not isinstance(state, dict):
        state = new_state()
    try:
        fed = compose_input(text, state)
        obj, meta = pipeline.classify(
            fed,
            router=router,
            level_extractor=level_extractor,
            component_extractor=component_extractor,
            implicit_severity_extractor=implicit_severity_extractor,
            log_pool=log_pool,
            pool_inbox=pool_inbox,
        )
        key = None
        state, key = _merge_step(state, obj)
        if key is not None:
            _carry_forward(state, key, obj)
        _resolve_pending(state, obj)
        try:
            meta = dict(meta)
            # A record that was never filed (tool_clarification's pending
            # early-return) owns no state key (review §10).
            meta["state_key"] = key or ""
            meta["state_steps"] = len(_steps_of(state))
            meta["fed_chars"] = len(fed)
            meta["window_ok"] = len(fed) <= WINDOW_CHARS
        except Exception:
            pass
        return obj, meta
    except Exception:
        try:
            obj, meta = pipeline.classify(
                "" if text is None else str(text), router=None)
            meta = dict(meta)
            meta["state_key"] = ""
            meta["state_steps"] = len(_steps_of(state))
            meta["fed_chars"] = 0
            meta["window_ok"] = True
            return obj, meta
        except Exception:
            return template("no_tool_call"), {
                "routed_by": "template", "valid": False, "retries": 0,
                "latency_ms": 0.0, "state_key": "", "state_steps": 0,
                "fed_chars": 0, "window_ok": True,
            }
