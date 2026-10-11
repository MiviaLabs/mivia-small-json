# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Schema and helpers for the ``tool_call`` AI agent workflow signal.

Enables structured parsing and routing of AI function calling and tool execution
traces (e.g. from glaive-function-calling-v2, OpenAI/Anthropic tool calls, OpenHands,
SWE-agent, CodeAct).
"""

from __future__ import annotations

import json
import re

from . import registry

TOOL_CALL_SPEC = {
    "type": {"const": "tool_call"},
    "name": {"type": str},
    # Arguments are a JSON OBJECT serialized as a string: the "json"
    # format makes validate refuse free prose ("run the tests now") and
    # repair route through repair_arguments, whose ladder always ends at
    # a valid object (spanfix wave bug 10, 2026-10-10).
    "arguments": {"type": str, "format": "json"},
    "result": {"type": str},
}

TOOL_CALL_ROUTING = {
    "strong": (
        r"(?i)<functioncall>",
        r"(?i)<tool_call>",
        r"(?i)</tool_call>",
        r"(?i)FUNCTION RESPONSE:",
        r"(?i)TOOL RESPONSE:",
        r"(?i)\btool_call\b",
        r"(?i)\bcall:\s*[A-Za-z0-9_.]+",
        r'(?i)"name":\s*"[A-Za-z0-9_.]+"',
        r'(?i)"arguments":\s*[\'"{]',
        # The BFCL / function-calling leaderboard format: a tool catalogue
        # followed by a natural-language request.
        r"(?i)\bTools available:",
        r"(?i)\bAvailable tools:",
    ),
    "weak": (
        r"(?i)\bfunction_call\b",
        r"(?i)\btool invocation\b",
        r"(?i)\bfunction invocation\b",
        r"(?i)\bexecute_bash\b",
        r"(?i)\bstr_replace_editor\b",
        r"(?i)\bUser request:",
        r"(?i)\bUser prompt:",
        r"(?i)\bparameters\":\s*\{",
    ),
}

TOOL_CALL_EXTRACTION = {
    "name": (
        # Wrapper tags first, then the explicit ``tool:``/``call:`` prefix
        # (B1, 2026-10-10 review): these name the TOOL, so they outrank the
        # bare ``"name":`` JSON key — an arguments object like
        # ``tool: get_weather arguments: {"name": "Paris"}`` used to be
        # named after the first argument VALUE ("Paris").
        r'(?i)<functioncall>\s*\{"name":\s*"([^"]+)"',
        r'(?i)<tool_call>\s*\{"name":\s*"([^"]+)"',
        r"(?i)\bcall:\s*([A-Za-z0-9_.-]+)",
        r"(?i)\btool:\s*([A-Za-z0-9_.-]+)",
        r'(?i)"name":\s*"([^"]+)"',
        r"(?i)\bname\s*[:=]\s*([A-Za-z0-9_.-]+)",
    ),
    "arguments": (
        # The object limb must tolerate newlines inside the braces
        # (pretty-printed payloads); ``[^{}]`` keeps it to a flat object
        # and stops at the first unbalanced ``}`` (nested objects go
        # through the balanced-brace scan in extract_tool_call).
        r"(?i)\"arguments\"\s*:\s*('[^']+'|\"(?:\\.|[^\"\\])*\"|\{[^{}]*\})",
        r"(?i)\barguments\s*[:=]\s*(.+?)(?:\s*;\s*result|$)",
        r"(?i)\bargs\s*[:=]\s*(.+?)(?:\s*;\s*result|$)",
        r"(?i)<arguments>\s*(.*?)\s*</arguments>",
    ),
    "result": (
        r"(?i)FUNCTION RESPONSE:\s*(.+)$",
        r"(?i)TOOL RESPONSE:\s*(.+)$",
        r'(?i)"result":\s*(.+)$',
        r"(?i)\bresult\s*[:=]\s*(.+)$",
        r"(?i)\boutput\s*[:=]\s*(.+)$",
        r"(?i)\bstdout\s*[:=]\s*(.+)$",
    ),
}

TOOL_CALL_DEFAULTS = {
    "name": "",
    "arguments": "{}",
    "result": "",
}

# The abstention class: a tool catalogue whose request needs NO tool call.
# Without it the router sends every "Tools available:" text to tool_call.
NO_TOOL_CALL_SPEC = {
    "type": {"const": "no_tool_call"},
    "reason": {"type": str},
}
NO_TOOL_CALL_ROUTING = {
    "strong": (r"(?i)\bno tool call (?:is )?(?:needed|required)\b",),
    "weak": (),
}
NO_TOOL_CALL_DEFAULTS = {"reason": "no matching tool"}
NO_TOOL_CALL_EXAMPLES = [
    {"type": "no_tool_call", "reason": "no matching tool"},
]

TOOL_CALL_EXAMPLES = [
    {
        "type": "tool_call",
        "name": "get_weather",
        "arguments": '{"location": "San Francisco, CA"}',
        "result": '{"temperature": 68, "condition": "sunny"}',
    },
    {
        "type": "tool_call",
        "name": "search_code",
        "arguments": '{"query": "def classify", "path": "src/"}',
        "result": '{"matches": 3}',
    },
    {
        "type": "tool_call",
        "name": "run_command",
        "arguments": '{"command": "pytest -q"}',
        "result": "157 passed",
    },
    {
        "type": "tool_call",
        "name": "fetch_url",
        "arguments": '{"url": "https://api.github.com/repos"}',
        "result": '{"status": 200}',
    },
]

_SINGLE_QUOTED_KEY_VAL_RE = re.compile(r"'(?P<key>[a-zA-Z0-9_\-]+)'\s*:")
_SINGLE_QUOTED_STR_VAL_RE = re.compile(r":\s*'(?P<val>[^']*)'")
_SINGLE_QUOTED_ELEM_RE = re.compile(r"([\[,]\s*)'(?P<val>[^']*)'")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

# Fences that LLMs wrap payloads in: markdown code fences and the XML tags
# small models emit around tool calls. No repair library strips these, so
# the rules layer does it first (measured 2026-10-09: 5/30 corpus collapses).
_MD_FENCE_RE = re.compile(r"```(?:json|javascript|js)?\s*\n?(?P<body>.*?)```",
                          re.DOTALL)
_XML_FENCE_RE = re.compile(
    r"<(?:tool_call|functioncall)>\s*(?P<body>.*?)\s*</(?:tool_call|functioncall)>",
    re.DOTALL | re.IGNORECASE)


def _loads_dict(text: str):
    """json.loads restricted to objects (None when not a dict). Floats that
    overflow to infinity (1e400 — Python's json is lenient, RFC 8259 is
    not) refuse the parse so the ladder never emits bare ``Infinity``
    (review P1-1), and the ``NaN``/``Infinity``/``-Infinity`` constants are
    rejected the same way (they are not JSON; letting them through made
    ``{"a": NaN}`` re-emit the bare token — review §5)."""
    def _strict_float(x: str):
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite float {x}")
        return v

    def _reject_constant(x: str):
        raise ValueError(f"non-JSON constant {x}")

    try:
        parsed = json.loads(text, parse_float=_strict_float,
                            parse_constant=_reject_constant)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _dumps_dict(parsed) -> str:
    """Strict-JSON serialization of a parsed object: ``allow_nan=False``
    keeps the contract (a non-finite value that slipped through degrades
    to ``{}`` instead of emitting bare ``NaN``/``Infinity``)."""
    try:
        return json.dumps(parsed, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return "{}"


def _pydict_to_json(value):
    """Convert a literal_eval result to a JSON-able dict (tuple->list,
    set->sorted list); None when the result is not a dict. Values outside
    the JSON domain (bytes, complex, ellipsis) raise ValueError so the
    caller degrades instead of producing unserializable output
    (review P0-1: `{'a': b'x'}` used to raise TypeError out of
    repair_arguments)."""
    if not isinstance(value, dict):
        return None

    def conv(v):
        if isinstance(v, tuple):
            return [conv(x) for x in v]
        if isinstance(v, (set, frozenset)):
            return sorted(conv(x) for x in v)
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        if isinstance(v, (str, bool, int)) or v is None:
            return v
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("non-finite float is not strict JSON")
            return v
        raise ValueError(f"non-JSON literal type {type(v).__name__}")

    return {str(k): conv(x) for k, x in value.items()}


def _literal_eval_dict(text: str):
    """ast.literal_eval fallback: exact Python-dict payloads with single
    quotes, True/None and nested tuples/sets that no regex ladder can
    reach. Literal-only evaluation — never executes anything."""
    import ast

    try:
        return _pydict_to_json(ast.literal_eval(text))
    except (ValueError, SyntaxError, MemoryError, TypeError):
        return None


def _strip_fences(s: str) -> str:
    """Extract the payload from markdown or XML fences. The first block
    that contains a brace wins; unfenced input passes through unchanged."""
    for pattern, closers in (
            (_XML_FENCE_RE, ("</tool_call>", "</functioncall>")),
            (_MD_FENCE_RE, ("```",)),
    ):
        # A fence match requires a closer. Without one, finditer rescans
        # the whole tail from every opener — quadratic backtracking that
        # pinned a worker ~27 s on a 256 KiB wall of "<tool_call>" via
        # /v1/repair and both MCP surfaces. The prescreen is
        # behavior-identical: no closer means no match, so skip finditer.
        low = s.lower()
        if not any(closer in low for closer in closers):
            continue
        for match in pattern.finditer(s):
            body = match.group("body").strip()
            if "{" in body:
                return body
    return s


def _escape_control_chars(s: str) -> str:
    """Escape literal control characters inside the payload (the OpenRouter
    dominant-defect class): a raw tab/newline inside a string value breaks
    json.loads; escaping it is always safe for the ladder."""
    return (s.replace("\t", r"\t").replace("\n", r"\n")
             .replace("\r", r"\r"))


def _complete_closers(s: str) -> str | None:
    """Truncated-JSON completion (partial-json style): append the missing
    closers for unclosed brackets, respecting string literals. Returns the
    completed text only when it then parses as an object."""
    stack = []
    in_string = False
    escaped = False
    for ch in s:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if not stack or in_string:
        return None
    completed = s + "".join("}" if c == "{" else "]" for c in reversed(stack))
    return completed if _loads_dict(completed) is not None else None


_LITERAL_OUTSIDE_RE = re.compile(r"\b(True|False|None)\b")


def _rewrite_literals_outside_strings(s: str) -> str:
    """Lower Python ``True``/``False``/``None`` to JSON literals, but only
    outside string literals (same quote scanner as ``_complete_closers``):
    a ``True`` inside an intact string value is content, not a literal."""
    _json_literal = {"True": "true", "False": "false", "None": "null"}
    result: list[str] = []
    buf: list[str] = []
    in_string = False
    escaped = False
    for ch in s:
        if in_string:
            result.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            result.append(_LITERAL_OUTSIDE_RE.sub(
                lambda m: _json_literal[m.group(1)], "".join(buf)))
            buf = []
            result.append(ch)
            in_string = True
            continue
        buf.append(ch)
    result.append(_LITERAL_OUTSIDE_RE.sub(
        lambda m: _json_literal[m.group(1)], "".join(buf)))
    return "".join(result)


def _scan_balanced_object(text: str, start: int) -> str | None:
    """The balanced ``{...}`` slice beginning at ``text[start] == '{'``,
    respecting string literals (so ``{"cmd": "echo }{"}`` closes at the
    right brace and pretty-printed objects span newlines). None when the
    braces never balance."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def repair_arguments(raw_args: str) -> str:
    """Repair messy pseudo-JSON arguments emitted by AI agents.

    Fixes:
    - Fenced payloads: ```` ```json ... ``` ````, `<tool_call>...</tool_call>`
    - Single-quoted pseudo-JSON: `{'foo': 'bar'}` -> `{"foo": "bar"}`
      (keys, values after a colon, and elements inside arrays)
    - Double-wrapped escaped JSON strings: `"{\"foo\": \"bar\"}"` -> `{"foo": "bar"}`
    - Python literals: `True` -> `true`, `False` -> `false`, `None` -> `null`
    - Trailing commas: `{"foo": "bar",}` -> `{"foo": "bar"}`
    - Raw control characters inside string values
    - Truncated objects/arrays (missing closers are completed)
    - Anything that still parses as a Python dict literal (ast.literal_eval,
      tuple/set values converted to JSON lists)
    """
    s = raw_args.strip()
    if not s:
        return "{}"

    # A raw NUL byte is invalid JSON anywhere (string value or structural
    # position) and no repair step below can produce a parseable payload
    # while it is present — the whole object used to collapse to `{}`.
    # Dropping it is lossless for every input that could parse (2026-10-10
    # validation, B3: `{"command": "echo a\x00b"}` -> `{}`).
    if "\x00" in s:
        s = s.replace("\x00", "")

    # An already-valid JSON object is returned normalized, without the text
    # rewrites below: those substitutions target pseudo-JSON, and they
    # corrupt a valid object whose *string values* contain `'key':`,
    # `None`, or `\"` (a shell command embedding a Python literal, a doc
    # quoting a field name). This was degrading 12% of harvested agent_step
    # arguments to `{}`.
    parsed = _loads_dict(s)
    if parsed is not None:
        return _dumps_dict(parsed)

    # Fenced payloads: strip the wrapper and retry the exact path.
    inner = _strip_fences(s)
    if inner != s:
        parsed = _loads_dict(inner)
        if parsed is not None:
            return _dumps_dict(parsed)
        s = inner

    # Trailing prose after the object (`{"a": 1}; return_code: 0`): the
    # trim ladder below only fires when the string ENDS with a closer,
    # so cut at the balanced object's last brace and retry (review P2).
    if not s.endswith(("}", "]")):
        brace = s.find("{")
        if brace != -1:
            balanced = _scan_balanced_object(s, brace)
            if balanced is not None:
                parsed = _loads_dict(balanced)
                if parsed is not None:
                    return _dumps_dict(parsed)
                # Exact Python-dict slice BEFORE the quote/literal rewrites
                # run (B3 residual, 2026-10-10 review-fixes validation):
                # `{"command": "echo 'hi': ok", "x": 1,}; done` carries a
                # valid dict-literal slice, but the single-quote rewrites
                # below corrupt the `'hi':` inside the string value, so
                # every LATER slice sees corrupted bytes and collapses to
                # `{}`. The pristine slice is the exact answer.
                exact = _literal_eval_dict(balanced)
                if exact is not None:
                    try:
                        return json.dumps(exact, separators=(",", ":"),
                                          allow_nan=False)
                    except (TypeError, ValueError):
                        pass

    # Python dict literal (single quotes, True/None, tuples, sets) — the
    # exact answer when it evaluates; supersedes every regex heuristic.
    exact = _literal_eval_dict(s)
    if exact is not None:
        try:
            return json.dumps(exact, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            pass

    # Strip outer single quotes or double quotes if whole object is string-wrapped
    wrapped = False
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        inner = s[1:-1].strip()
        if inner.startswith("{") and inner.endswith("}"):
            s = inner
            wrapped = True

    # Unescape escaped double quotes only for the double-wrapped form
    # (`"{\"a\": \"b\"}"`). A plain object's string values legitimately
    # contain `\"` (e.g. `{"command": "echo \"hi\""}`); unescaping those
    # corrupts the JSON, so the whole object degrades to `{}`.
    if wrapped and r'\"' in s:
        s = s.replace(r'\"', '"')

    # Fix Python booleans and None — OUTSIDE string literals only (review
    # §5/P1: rewriting inside intact quotes corrupted payloads like
    # {"cmd": "echo True things"}). The string-aware rewrite is applied to
    # a copy and kept only when it actually helps the ladder below.
    s = _rewrite_literals_outside_strings(s)

    # Convert single-quoted keys and values to double quotes
    s = _SINGLE_QUOTED_KEY_VAL_RE.sub(r'"\g<key>":', s)
    s = _SINGLE_QUOTED_STR_VAL_RE.sub(r': "\g<val>"', s)
    # ... and single-quoted elements inside arrays (`['a', 'b']`): the
    # colon-anchored pass above never touches them, which collapsed every
    # payload with a quoted list (serving-probe case). Applied to a copy
    # and kept only when the result actually parses — inside a double-
    # quoted string value (`"pick [, 'b']"`) the rewrite corrupts the
    # payload (review P1-2: a wrapped-string regression).
    s_elem = _SINGLE_QUOTED_ELEM_RE.sub(r'\g<1>"\g<val>"', s)
    if s_elem != s and _loads_dict(s_elem) is not None:
        s = s_elem

    # Strip trailing commas
    s = _TRAILING_COMMA_RE.sub(r"\1", s)

    # Validate JSON, fallback to cleaned string
    parsed = _loads_dict(s)
    if parsed is not None:
        return _dumps_dict(parsed)

    # Retry the balanced-object slice on the rewritten text (B3, 2026-10-10
    # review): the strict slice above only fired when the object parsed
    # AS-IS, but `{"command": "ls", "ok": True}; return_code: 0` needs the
    # literal/comma rewrites FIRST and the slice AFTER — rewrite-then-slice.
    # The slice is re-tried with literal_eval too (a single-quoted payload
    # followed by prose the whole-string eval cannot reach).
    brace = s.find("{")
    if brace != -1:
        balanced = _scan_balanced_object(s, brace)
        if balanced is not None:
            parsed = _loads_dict(balanced)
            if parsed is not None:
                return _dumps_dict(parsed)
            exact_slice = _literal_eval_dict(balanced)
            if exact_slice is not None:
                try:
                    return json.dumps(exact_slice, separators=(",", ":"),
                                      allow_nan=False)
                except (TypeError, ValueError):
                    pass

    # Raw control characters inside strings (the dominant real-world defect
    # class): escaping them can only help at this point.
    escaped = _loads_dict(_escape_control_chars(s))
    if escaped is not None:
        return _dumps_dict(escaped)

    # The extraction regex can capture one or more extra closers when the
    # argument object sits inside a wrapper (`"arguments": {}}}`) or is
    # followed by sibling fields (`}}}; return_code: 0`). Trim trailing
    # closers until the value parses as an object; the first parse wins
    # (review §5: a bare `break` after the first trim gave up on two or
    # more extra closers).
    trimmed = s
    while trimmed.endswith(("}", "]")) and len(trimmed) > 1:
        trimmed = trimmed[:-1].rstrip()
        parsed = _loads_dict(trimmed)
        if parsed is not None:
            return _dumps_dict(parsed)
        continue

    # Truncated payload: complete the missing closers (partial-json style).
    completed = _complete_closers(s)
    if completed is not None:
        return _dumps_dict(_loads_dict(completed))

    # Last chance on the fully normalized text (bools already lowercased,
    # quotes rewritten — a payload that only needed one of the two).
    exact = _literal_eval_dict(s)
    if exact is not None:
        try:
            return json.dumps(exact, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            pass

    # Never emit malformed JSON: a value that could not be repaired to an
    # object degrades to the empty argument object.
    return "{}"


# A captured result that begins with a JSON string literal
# (`"result": "done"}` — the regex ladder captures `"done"}` including
# the quotes and the sibling closer). The quoted payload is the honest
# value; the delimiters are the wrapper, not the result (2026-10-09
# probe, same delimiter-capture class as "[ERROR]" levels).
_RESULT_JSON_STRING_RE = re.compile(r'^\s*"((?:\\.|[^"\\])*)"')


def _clean_result(raw: str) -> str:
    """Strip wrapper delimiters from a captured result value."""
    s = raw.strip()
    if s.startswith('"'):
        m = _RESULT_JSON_STRING_RE.match(s)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return m.group(1)
        return s
    # N7 (2026-10-10 review): the `"result": (.+)$` capture keeps the
    # wrapper's sibling closers for non-string JSON values — `42}`,
    # `true}`, `null}`, `42,`. Peel trailing `}`/`,` only when the
    # remainder parses as a JSON scalar; an object/array result keeps its
    # braces (the 2026-10-09 fix covered quoted strings only).
    candidate = s.rstrip("},")
    if len(candidate) < len(s) and candidate:
        try:
            value = json.loads(candidate)
        except ValueError:
            return s
        if value is None or isinstance(value, (bool, int, float, str)):
            return candidate
    return s


def extract_tool_call(text: str) -> dict:
    """Extract structured fields from tool execution traces."""
    res = dict(TOOL_CALL_DEFAULTS)
    res["type"] = "tool_call"

    # Extract name
    for pat in TOOL_CALL_EXTRACTION["name"]:
        m = re.search(pat, text)
        if m:
            res["name"] = m.group(1).strip()
            break

    # Extract arguments. The quoted/nested-object form goes through a
    # balanced-brace scan first (review §5): the regex ladder's
    # single-line object alternative degrades nested and pretty-printed
    # objects to "{}". The scan key covers the bare ``arguments:`` /
    # ``args:`` forms too (B2, 2026-10-10 review): `call: search
    # arguments: {...}` used to keep the `{}` default and donate the
    # embedded `"name":` VALUE to the name field.
    args_done = False
    m = re.search(
        r'(?i)(?:"arguments"|\barguments\b|\bargs\b)\s*[:=]\s*', text)
    if m and m.end() < len(text) and text[m.end()] == "{":
        balanced = _scan_balanced_object(text, m.end())
        if balanced is not None:
            repaired = repair_arguments(balanced)
            # The scan tracks double-quoted strings only: a single-quoted
            # pseudo-JSON value holding a stray `}` (`{'cmd': 'echo }'}`)
            # cuts the slice early and repair degrades it to `{}`. A
            # substantive slice that repairs to the empty object is a
            # mis-slice, not an answer — fall through to the regex ladder,
            # whose `arguments ...; result` capture reaches the real end
            # (2026-10-10 review-fixes validation, B2 regression hunt).
            substantive = balanced.strip() not in ("{", "{}")
            if repaired != "{}" or not substantive:
                res["arguments"] = repaired
                args_done = True
    if not args_done:
        for pat in TOOL_CALL_EXTRACTION["arguments"]:
            m = re.search(pat, text)
            if m:
                raw_args = m.group(1).strip()
                res["arguments"] = repair_arguments(raw_args)
                break

    # Extract result
    for pat in TOOL_CALL_EXTRACTION["result"]:
        m = re.search(pat, text)
        if m:
            res["result"] = _clean_result(m.group(1))
            break

    return res


def register_tool_call_schema() -> None:
    """Register the ``tool_call`` schema in the runtime registry (idempotent)."""
    if "tool_call" in registry.active():
        return
    registry.register_schema(
        "tool_call",
        TOOL_CALL_SPEC,
        TOOL_CALL_ROUTING,
        TOOL_CALL_EXTRACTION,
        TOOL_CALL_DEFAULTS,
        TOOL_CALL_EXAMPLES,
        extractor=extract_tool_call,
    )


def register_no_tool_call_schema() -> None:
    """Register the ``no_tool_call`` abstention schema (idempotent)."""
    if "no_tool_call" in registry.active():
        return
    registry.register_schema(
        "no_tool_call",
        NO_TOOL_CALL_SPEC,
        NO_TOOL_CALL_ROUTING,
        {},
        NO_TOOL_CALL_DEFAULTS,
        NO_TOOL_CALL_EXAMPLES,
    )


TOOL_CLARIFICATION_REASONS = (
    "missing_arguments",
    "ambiguous_intent",
    "underspecified_query",
)

TOOL_CLARIFICATION_SPEC = {
    "type": {"const": "tool_clarification"},
    "tool": {"type": str},
    "reason": {"type": str, "enum": TOOL_CLARIFICATION_REASONS},
    "missing_parameters": {"type": list},
    "clarification_prompt": {"type": str},
}

TOOL_CLARIFICATION_DEFAULTS = {
    "tool": "",
    "reason": "missing_arguments",
    "missing_parameters": [],
    "clarification_prompt": "Please specify the required parameters for this tool.",
}

TOOL_CLARIFICATION_ROUTING = {
    "strong": (
        r"(?i)\bneeds?\s+clarification\b",
        r"(?i)\bmissing\s+parameters?\b",
    ),
    "weak": (
        r"(?i)\bclarify\b",
        r"(?i)\bwhich\s+parameter\b",
    ),
}

TOOL_CLARIFICATION_EXAMPLES = [
    {
        "type": "tool_clarification",
        "tool": "calculate_sum",
        "reason": "missing_arguments",
        "missing_parameters": ["number1", "number2"],
        "clarification_prompt": "Please specify number1, number2 to calculate the sum.",
    }
]


def extract_tool_clarification(text: str) -> dict:
    """Extract or construct a tool_clarification record."""
    res = dict(TOOL_CLARIFICATION_DEFAULTS)
    res["type"] = "tool_clarification"
    m_tool = re.search(r"(?i)\btool\s*[:=]\s*([a-zA-Z0-9_\-]+)", text)
    if m_tool:
        res["tool"] = m_tool.group(1)
    return res


def register_tool_clarification_schema() -> None:
    """Register the ``tool_clarification`` 3-state schema in the registry (idempotent)."""
    if "tool_clarification" in registry.active():
        return
    registry.register_schema(
        "tool_clarification",
        TOOL_CLARIFICATION_SPEC,
        TOOL_CLARIFICATION_ROUTING,
        {},
        TOOL_CLARIFICATION_DEFAULTS,
        TOOL_CLARIFICATION_EXAMPLES,
        extractor=extract_tool_clarification,
    )

