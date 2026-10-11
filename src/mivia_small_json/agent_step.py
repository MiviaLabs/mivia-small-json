# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn agent trajectory step schema and extractors.

Captures one iteration of an autonomous coding-agent loop
(SWE-agent, OpenHands, Aider style)::

    agent query -> tool invocation -> stdout/stderr/return_code -> next step

``duration_ms`` is an int (whole milliseconds): the registry contract
only admits ``str``/``int``/``list`` field types, so a float would be
rejected at registration and by the strict validator.
"""

from __future__ import annotations

import re

from . import registry
from .diagnostic import sanitize_log
from .schemas import AGENT_ROLES, AGENT_STATUSES
from .tool_call import repair_arguments

AGENT_STEP_SPEC = {
    "type": {"const": "agent_step"},
    "step_index": {"type": int},
    "agent_role": {"type": str, "enum": AGENT_ROLES},
    "tool": {"type": str},
    "arguments": {"type": str},
    "stdout": {"type": str},
    "stderr": {"type": str},
    "return_code": {"type": int},
    "duration_ms": {"type": int},
    "error_type": {"type": str},
    "error_message": {"type": str},
    "status": {"type": str, "enum": AGENT_STATUSES},
}

AGENT_STEP_ROUTING = {
    "strong": (
        r"(?i)\bstep\s*\d+\s*:\s*(?:tool|call|exec)",
        r"(?i)\btool_execution_result\b",
        r"(?i)\bcommand\s+finished\s+with\s+exit\s+code\b",
        r"(?i)\bexecution\s+timed\s+out\b",
        r"(?i)\baction\s*:\s*execute_bash\b",
    ),
    "weak": (
        r"(?i)\breturn_code\b",
        r"(?i)\bexit\s+code\s*[:=]\s*\d+",
        r"(?i)\bstdout\s*[:=]",
        r"(?i)\bstderr\s*[:=]",
    ),
}

# Terminator keywords for a captured error_message value (no-truncation
# law, owner directive 2026-10-10): a value may span lines and carry
# ordinary semicolons, so it ends only at a REAL field separator — a ";"
# followed by a field keyword, or a newline followed by a field keyword
# with its ":"/ "=" (the M2 rule the stdout/stderr captures follow).
# Newlines and semicolons inside the message are content, never cuts.
_STEP_FIELD_KW = r"(?:stdout|stderr|return_code|exit\s+code|duration_ms|status)"
_VALUE_UNTIL_FIELD_KW = (
    r"((?:[^;\n]"
    r"|\n(?!\s*" + _STEP_FIELD_KW + r"\s*[:=])"
    r"|;(?!\s*" + _STEP_FIELD_KW + r"\b)"
    r")*)"
)

AGENT_STEP_EXTRACTION = {
    "step_index": (
        r"(?i)\bstep_index\s*[:=]\s*(\d+)",
        r"(?i)\bstep[_\s]*(?:#|index\s*[:=]?)?\s*(\d+)",
    ),
    "agent_role": (
        r"(?i)\bagent_role\s*[:=]\s*(assistant|tool|user|environment)\b",
        r"(?i)\b(?:agent_)?role\s*[:=]\s*(assistant|tool|user|environment)\b",
        r"(?i)\bagent\s*[:=]\s*(assistant|tool|user|environment)\b",
    ),
    "tool": (
        r"(?i)\btool\s+call\s+([A-Za-z0-9_.-]+)",
        r"(?i)\btool\s*[:=]\s*([A-Za-z0-9_.-]+)",
        r"(?i)\baction\s*:\s*([A-Za-z0-9_.-]+)",
        r"(?i)\bcall\s*:\s*([A-Za-z0-9_.-]+)",
    ),
    "arguments": (
        r"(?i)\barguments\s*[:=]\s*(.+?)(?:\s*;\s*(?:stdout|stderr|return_code|result)\b|$)",
        r"(?i)\bargs\s*[:=]\s*(.+?)(?:\s*;\s*(?:stdout|stderr|return_code|result)\b|$)",
    ),
    "stdout": (
        r"(?is)\bstdout\s*[:=]\s*(.+?)(?=\s*\bstderr\b|\s*\breturn_code\b|\s*\bexit\s+code\b|$)",
    ),
    "stderr": (
        r"(?is)\bstderr\s*[:=]\s*(.+?)(?=\s*\bstdout\b|\s*\breturn_code\b|\s*\bexit\s+code\b|$)",
    ),
    "return_code": (
        r"(?i)\breturn_code\s*[:=]\s*(-?\d+)",
        r"(?i)\bexit\s+code\s*[:=]\s*(-?\d+)",
        r"(?i)\bexit\s+code\s+(-?\d+)",
    ),
    "duration_ms": (
        r"(?i)\bduration_ms\s*[:=]\s*(\d+)",
        r"(?i)\b(\d+)\s*ms\b",
    ),
    "error_type": (
        r"\b([A-Za-z_][A-Za-z0-9_.]*Error)\b",
    ),
    "error_message": (
        r"(?i)\berror_message\s*[:=]\s*" + _VALUE_UNTIL_FIELD_KW,
        r"[A-Za-z_][A-Za-z0-9_.]*Error:\s*" + _VALUE_UNTIL_FIELD_KW,
    ),
    "status": (
        r"(?i)\bstatus\s*[:=]\s*(success|error|timeout|interrupted)\b",
    ),
}

AGENT_STEP_DEFAULTS = {
    "step_index": 0,
    "agent_role": "assistant",
    "tool": "",
    "arguments": "{}",
    "stdout": "",
    "stderr": "",
    "return_code": 0,
    "duration_ms": 0,
    "error_type": "",
    "error_message": "",
    "status": "success",
}

AGENT_STEP_EXAMPLES = [
    {
        "type": "agent_step",
        "step_index": 3,
        "agent_role": "assistant",
        "tool": "execute_bash",
        "arguments": '{"command":"pytest -q"}',
        "stdout": "157 passed",
        "stderr": "",
        "return_code": 0,
        "duration_ms": 120,
        "error_type": "",
        "error_message": "",
        "status": "success",
    },
    {
        "type": "agent_step",
        "step_index": 4,
        "agent_role": "tool",
        "tool": "execute_bash",
        "arguments": '{"command":"pytest tests/test_api.py"}',
        "stdout": "",
        "stderr": "FAILED tests/test_api.py::test_login",
        "return_code": 1,
        "duration_ms": 845,
        "error_type": "AssertionError",
        "error_message": "assert False",
        "status": "error",
    },
    {
        "type": "agent_step",
        "step_index": 5,
        "agent_role": "assistant",
        "tool": "execute_bash",
        "arguments": '{"command":"pytest -q"}',
        "stdout": "",
        "stderr": "execution timed out after 30000ms",
        "return_code": 124,
        "duration_ms": 30000,
        "error_type": "TimeoutError",
        "error_message": "execution timed out",
        "status": "timeout",
    },
]

# "timeout" must not match option spellings inside captured arguments
# (`--connect-timeout 30`, `--timeout=5`): a lookbehind on `-` or a word
# char anchors the marker to prose/field positions (review §11).
_TIMEOUT_RE = re.compile(
    r"(?i)(?<![-\w])\btimed?\s*out\b|(?<![-\w])timeout(?![\w-])")
_INTERRUPTED_RE = re.compile(
    r"(?i)\binterrupted\b|\bcancelled\b|\bcanceled\b|\bKeyboardInterrupt\b")
_DURATION_S_RE = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*(?:s(?:ec(?:ond)?s?)?|seconds?)\b")
_RETURN_CODE_RES = (
    # Most specific first (B9, review-val 2026-10-10): the bare `exit code`
    # shape matches inside echoed commands (`echo exit code 0`) and, tried
    # first, outranked the executed command's own outcome.
    re.compile(r"(?i)\bcommand\s+finished\s+with\s+exit\s+code\s+(-?\d+)"),
    re.compile(r"(?i)\breturn_code\s*[:=]\s*(-?\d+)"),
    re.compile(r"(?i)\bexit\s+code\s*[:=]?\s*(-?\d+)"),
)
_DURATION_RES = (
    re.compile(r"(?i)\bduration_ms\s*[:=]\s*(\d+)"),
    re.compile(r"(?i)\b(\d+)\s*ms\b"),
)

# Field keywords that can terminate a captured stdout/stderr value (M2,
# 2026-10-10). A value may span lines, so a terminator is a `;` separator
# before a keyword, or a NEWLINE followed by a keyword with its `:`/`=`
# (the B10 keyword-led line) — a bare newline inside captured output no
# longer truncates the value at the first line.
_FIELD_KW = r"(?:stdout|stderr|return_code|exit\s+code|duration_ms)"
_STEP_VALUE_RE_TEMPLATE = (
    r"(?is)\b{field}\s*[:=]\s*((?:"
    r"[^;\n]"
    r"|\n(?!\s*" + _FIELD_KW + r"\s*[:=])"
    r"|;(?!\s*" + _FIELD_KW + r"\b)"
    r")*)"
)
_STDOUT_VALUE_RE = re.compile(_STEP_VALUE_RE_TEMPLATE.format(field="stdout"))
_STDERR_VALUE_RE = re.compile(_STEP_VALUE_RE_TEMPLATE.format(field="stderr"))


def _first_int(patterns, text, default=0):
    return _first_int_found(patterns, text, default)[0]


def _first_int_found(patterns, text, default=0):
    """Like ``_first_int`` but also reports whether an rc was found.

    The status policy needs the distinction: "rc absent" is consistent
    with timeout/interrupted markers, "rc 0" is not (rc 0 wins over prose
    markers in captured output).
    """
    for pat in patterns:
        m = pat.search(text)
        if m:
            try:
                return int(m.group(1)), True
            except (ValueError, IndexError):
                continue
    return default, False


def extract_agent_step(text: str) -> dict:
    """Extract one agent trajectory step from raw execution-trace text.

    Sanitizes ANSI/CI noise via ``diagnostic.sanitize_log``, repairs
    pseudo-JSON arguments via ``tool_call.repair_arguments``, and derives
    ``status`` from the return code (0 -> success, non-zero -> error)
    with timeout/interrupted markers taking precedence.
    """
    clean = sanitize_log(text)
    res = dict(AGENT_STEP_DEFAULTS)
    res["type"] = "agent_step"

    # The canonical raw opens with `step_index: N; agent_role: X;` (see
    # scripts/harvest_agent_sessions.py::_raw_for_step). The `_` in
    # `step_index` blocks the generic `step\s*...` shape, so the explicit
    # alternative is tried first; `step[_\s]*` also lets `step_index` fall
    # through to the generic branch.
    m = re.search(r"(?i)\bstep_index\s*[:=]\s*(\d+)", clean)
    if not m:
        m = re.search(r"(?i)\bstep[_\s]*(?:#|index\s*[:=]?)?\s*(\d+)", clean)
    if m:
        res["step_index"] = int(m.group(1))
    m = re.search(
        r"(?i)\b(?:agent_)?role\s*[:=]\s*(assistant|tool|user|environment)\b",
        clean)
    if m:
        res["agent_role"] = m.group(1).lower()
    # Earliest match wins, not first-pattern-in-list: a loose `tool call X`
    # can appear inside captured stdout (e.g. Go source mentioning
    # "tool call arguments"), and pattern priority would let it beat the
    # real `tool: X` field at position 0.
    tool_hits = []
    for pat in (r"(?i)\btool\s+call\s+([A-Za-z0-9_.-]+)",
                r"(?i)\btool\s*[:=]\s*([A-Za-z0-9_.-]+)",
                r"(?i)\baction\s*:\s*([A-Za-z0-9_.-]+)",
                r"(?i)\bcall\s*:\s*([A-Za-z0-9_.-]+)"):
        m = re.search(pat, clean)
        if m:
            tool_hits.append((m.start(), m.group(1)))
    if tool_hits:
        res["tool"] = min(tool_hits, key=lambda hit: hit[0])[1]
    args_span = None
    m = re.search(
        r"(?i)\barguments\s*[:=]\s*(.+?)(?:\s*;\s*"
        r"(?:stdout|stderr|return_code|result)\b|$)",
        clean, re.DOTALL)
    if not m:
        m = re.search(
            r"(?i)\bargs\s*[:=]\s*(.+?)(?:\s*;\s*"
            r"(?:stdout|stderr|return_code|result)\b|$)",
            clean, re.DOTALL)
    if m:
        res["arguments"] = repair_arguments(m.group(1).strip())
        args_span = m.span(1)

    # The canonical layout separates fields with ";". The boundary must
    # require that separator: an optional ";" makes the lookahead fire on a
    # bare `stderr`/`return_code`/`duration_ms` token *inside* captured
    # output (a shell command or a doc that mentions the field name) and
    # truncates the value. The value group tolerates an ordinary `;`
    # (shell one-liners: `echo a; echo b`) but not one followed by a
    # field keyword, and matches empty — so an EMPTY field
    # (`stdout: ; stderr: boom`) cannot swallow the next `;`-separated
    # field (review §11: `.+?` needed one char and ran on).
    # Newlines are value content unless KEYWORD-led (M2, 2026-10-10): the
    # B10 fix excluded `\n` outright and truncated genuinely multi-line
    # captured output at the first line. A value now ends at a `;` field
    # separator or at a newline followed by a field keyword with its
    # `:`/`=` (`stderr:`, `return_code: 1`) — not merely at any newline.
    stdout_span = None
    m = _STDOUT_VALUE_RE.search(clean)
    if m:
        # A trailing separator is layout, not captured output.
        res["stdout"] = m.group(1).strip().rstrip(";").strip()
        stdout_span = m.span(1)
    stderr_span = None
    m = _STDERR_VALUE_RE.search(clean)
    if m:
        res["stderr"] = m.group(1).strip().rstrip(";").strip()
        stderr_span = m.span(1)

    rc, rc_found = _first_int_found(_RETURN_CODE_RES, clean)
    res["return_code"] = rc

    duration = _first_int((_DURATION_RES[0],), clean, default=-1)
    if duration < 0:
        m = _DURATION_RES[1].search(clean)
        duration = int(m.group(1)) if m else 0
        if not m:
            s = _DURATION_S_RE.search(clean)
            if s:
                try:
                    duration = int(float(s.group(1)) * 1000)
                except ValueError:
                    duration = 0
    res["duration_ms"] = duration

    # Timeout/interrupted markers and the *Error token must not read the
    # captured arguments (B9, review-val 2026-10-10): `git log
    # --grep=cancelled`, `echo timed out`, or `python -c 'raise
    # TimeoutError'` are command content, not execution outcomes. The
    # span is blanked with spaces rather than removed so offsets stay
    # stable, and the SAME blanked text feeds the error_type scan.
    marker_text = clean
    if args_span:
        start, end = args_span
        marker_text = clean[:start] + " " * (end - start) + clean[end:]

    error_type_pos = None
    m = re.search(r"\b([A-Za-z_][A-Za-z0-9_.]*Error)\b", marker_text)
    if m:
        res["error_type"] = m.group(1)
        error_type_pos = m.start()
    # Full carry (no-truncation law, 2026-10-10): the error_message value
    # runs to the next REAL field separator — never to the first ";" or
    # newline, which are value content (same M2 rule as stdout/stderr).
    # An explicit field whose value strips empty falls through to the
    # Error:-led form so real content still wins.
    error_message = ""
    m = re.search(r"(?i)\berror_message\s*[:=]\s*" + _VALUE_UNTIL_FIELD_KW,
                  clean)
    if m:
        error_message = m.group(1).strip().rstrip(";").strip()
    if not error_message and res["error_type"]:
        m = re.search(re.escape(res["error_type"]) + r":\s*"
                      + _VALUE_UNTIL_FIELD_KW, clean)
        if m:
            error_message = m.group(1).strip().rstrip(";").strip()
    res["error_message"] = error_message

    # Status policy: return_code is the primary signal (0 -> success,
    # non-zero -> error). Timeout/interrupted markers upgrade only where
    # they are trustworthy: a hit inside captured stdout/stderr is
    # captured output, so it upgrades only when the rc is consistent
    # with it (timeout: exit code 124 or no rc at all; interrupted: a
    # non-zero rc or no rc — rc 0 wins over prose in captured output).
    # A hit in the non-captured remainder is the narrator's own verdict
    # of the run and keeps the old behavior. The arguments span is
    # already blanked above, so a marker there is never a signal. An
    # explicit `status:` field still overrides everything.
    def _in_captured(pos):
        return (stdout_span and stdout_span[0] <= pos < stdout_span[1]) or \
            (stderr_span and stderr_span[0] <= pos < stderr_span[1])

    def _marker_hits(pattern):
        """Split marker hits into (non-captured seen, captured seen)."""
        uncaptured = captured = False
        for hit in pattern.finditer(marker_text):
            if _in_captured(hit.start()):
                captured = True
            else:
                uncaptured = True
        return uncaptured, captured

    int_uncaptured, int_captured = _marker_hits(_INTERRUPTED_RE)
    to_uncaptured, to_captured = _marker_hits(_TIMEOUT_RE)
    # A TimeoutError token outside the arguments is a timeout signal too;
    # its trust mirrors where it was found (captured output needs a
    # consistent rc, the non-captured remainder does not).
    is_timeout_error = res["error_type"] == "TimeoutError"
    to_uncaptured = to_uncaptured or (
        is_timeout_error and not _in_captured(error_type_pos))
    to_captured = to_captured or (
        is_timeout_error and _in_captured(error_type_pos))
    timeout_rc_ok = (not rc_found) or rc == 124
    interrupted_rc_ok = (not rc_found) or rc != 0

    if int_uncaptured or (int_captured and interrupted_rc_ok):
        res["status"] = "interrupted"
    elif to_uncaptured or (to_captured and timeout_rc_ok):
        res["status"] = "timeout"
    elif res["return_code"] != 0:
        res["status"] = "error"
    else:
        res["status"] = "success"
    m = re.search(
        r"(?i)\bstatus\s*[:=]\s*(success|error|timeout|interrupted)\b", clean)
    if m:
        res["status"] = m.group(1).lower()
    return res


def register_agent_step_schema() -> None:
    """Register the ``agent_step`` trajectory schema (idempotent)."""
    if "agent_step" in registry.active():
        return
    registry.register_schema(
        "agent_step",
        AGENT_STEP_SPEC,
        AGENT_STEP_ROUTING,
        AGENT_STEP_EXTRACTION,
        AGENT_STEP_DEFAULTS,
        AGENT_STEP_EXAMPLES,
        extractor=extract_agent_step,
    )
