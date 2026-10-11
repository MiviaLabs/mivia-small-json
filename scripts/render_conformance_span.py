#!/usr/bin/env python3
"""Per-value conformance span renderer (TRACK 2, read-only extension).

Diagnosis (reports/exp-infeasible-diagnosis.md): 49 error_param false-call
rows carry values violating declared constraints (pattern/required), but
the gate span (tool_match.render_tool_fillability) never renders
value-conformance, so no head could learn it.

This module does NOT edit tool_match.py rendering: it imports
render_tool_fillability and appends a ``conformance:`` tail with one
pass/fail verdict per provided value against each declared constraint
(pattern, type, min/maxLength, minimum/maximum, enum, required presence).

CPU-only, stdlib-only (re/json). Deterministic: no randomness.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.tool_match import render_tool_fillability  # noqa: E402

_INT_RE = re.compile(r"^[+-]?\d+$")
_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_BOOL_OK = frozenset({"true", "false", "yes", "no", "1", "0"})


def _bare(tool: dict) -> dict:
    if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
        return tool["function"]
    return tool if isinstance(tool, dict) else {}


def check_value_conformance(value: str, spec: dict) -> list[str]:
    """Failed declared constraints for one provided value ([] = pass).

    Checks, in order: type (integer/number/boolean), pattern (regex
    search), minLength/maxLength, minimum/maximum, enum. Unknown or
    absent constraints are skipped, never failed.
    """
    fails: list[str] = []
    if not isinstance(spec, dict):
        return fails
    v = value.strip().strip("\"'")
    stype = str(spec.get("type", "") or "")
    if stype == "integer" and not _INT_RE.match(v):
        fails.append("type:integer")
    elif stype in ("number", "float") and not _NUM_RE.match(v):
        fails.append("type:number")
    elif stype == "boolean" and v.lower() not in _BOOL_OK:
        fails.append("type:boolean")
    pat = spec.get("pattern")
    if isinstance(pat, str) and pat:
        try:
            if not re.search(pat, v):
                fails.append(f"pattern:{pat}")
        except re.error:
            pass
    ml = spec.get("minLength")
    if isinstance(ml, (int, float)) and len(v) < ml:
        fails.append(f"minLength:{ml}")
    xl = spec.get("maxLength")
    if isinstance(xl, (int, float)) and len(v) > xl:
        fails.append(f"maxLength:{xl}")
    try:
        num = float(v)
        is_num = True
    except ValueError:
        is_num = False
    if is_num:
        lo, hi = spec.get("minimum"), spec.get("maximum")
        if isinstance(lo, (int, float)) and num < lo:
            fails.append(f"minimum:{lo}")
        if isinstance(hi, (int, float)) and num > hi:
            fails.append(f"maximum:{hi}")
    enum = spec.get("enum")
    if isinstance(enum, list) and enum and v not in [str(e) for e in enum]:
        fails.append("enum")
    return fails


def extract_candidate_values(tool: dict, request: str) -> dict[str, str]:
    """Greedy deterministic value extraction per required param.

    Order per slot: key-delimited (``name=value`` / ``name: value`` /
    ``"name": value``), then quoted strings assigned positionally to
    still-open slots in required order, then a token adjacent to a slot
    mention. First hit wins; slots with nothing found stay absent
    (rendered as missing, never invented).
    """
    bare = _bare(tool)
    params = bare.get("parameters")
    params = params if isinstance(params, dict) else {}
    required = params.get("required")
    required = list(required) if isinstance(required, list) else []
    found: dict[str, str] = {}
    for slot in required:
        m = re.search(
            r"(?:[\"']" + re.escape(slot) + r"[\"']|\b" + re.escape(slot)
            + r"\b)\s*[:=]\s*(?P<val>\"[^\"]*\"|'[^']*'|[^\s,;]+)",
            request, re.IGNORECASE)
        if m:
            found[slot] = m.group("val").strip().strip("\"'")
    if len(found) < len(required):
        quoted = re.findall(r"\"([^\"]+)\"|'([^']+)'", request)
        flat = [a or b for a, b in quoted]
        open_slots = [s for s in required if s not in found]
        for slot, q in zip(open_slots, flat):
            if q not in found.values():
                found[slot] = q
    for slot in required:
        if slot not in found:
            m = re.search(r"\b" + re.escape(slot)
                          + r"\b[\s\w,]{0,20}?(\S+)", request, re.IGNORECASE)
            if m and m.group(1) not in found.values():
                found[slot] = m.group(1).strip("\"',.;")
    return found


def render_conformance_span(tool: dict, request: str) -> str:
    """Gate span plus ``conformance:`` tail (base untouched, appended)."""
    base = render_tool_fillability(tool, request)
    bare = _bare(tool)
    params = bare.get("parameters")
    params = params if isinstance(params, dict) else {}
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = list(required) if isinstance(required, list) else []
    if not required:
        return base
    values = extract_candidate_values(tool, request)
    bits = []
    for slot in required:
        spec = props.get(slot) if isinstance(props.get(slot), dict) else {}
        if slot not in values:
            bits.append(f"{slot}=missing:required")
            continue
        fails = check_value_conformance(values[slot], spec)
        bits.append(f"{slot}={values[slot]}:"
                    + ("pass" if not fails else "fail(" + ",".join(fails) + ")"))
    return base + " | conformance: " + "; ".join(bits)


if __name__ == "__main__":
    demo_tool = {
        "name": "NextBusPrediction",
        "description": "Predict the arrival time of the next bus.",
        "parameters": {
            "properties": {
                "bus_service": {"type": "string", "pattern": r"^\d{1,4}$",
                                "description": "1 to 4 digits."},
                "location_name": {"type": "string", "minLength": 5,
                                  "maxLength": 100},
            },
            "required": ["location_name"],
            "type": "dict",
        },
    }
    for req in ('Can you plan a bus route from Beijing321 to Chaoyang now? '
                'location_name="Beijing321"',
                'Next bus at location_name="Beijing Central Station"?'):
        print(render_conformance_span(demo_tool, req))
        print()
