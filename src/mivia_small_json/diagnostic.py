# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Schema and helpers for compiler, linter, and test diagnostics.

Captures structured errors from rustc, tsc, ruff, gcc, clang, and pytest
following SARIF and Language Server Protocol (LSP) conventions.
"""

from __future__ import annotations

import re
from . import registry

DIAGNOSTIC_SEVERITIES = ("error", "warning", "note")

DIAGNOSTIC_SPEC = {
    "type": {"const": "diagnostic"},
    "tool": {"type": str},
    "file": {"type": str},
    "line": {"type": int},
    "column": {"type": int},
    "severity": {"type": str, "enum": DIAGNOSTIC_SEVERITIES},
    "code": {"type": str},
    "message": {"type": str},
}

DIAGNOSTIC_ROUTING = {
    "strong": (
        r"(?i)\berror\[[A-Z]\d+\]:",
        r"(?i)\bwarning\[[A-Z]\d+\]:",
        r"(?i)\berror\s+TS\d+:",
        r"(?i)\b\w+\.[a-zA-Z0-9_]+\(\d+,\s*\d+\):\s*error",
        r"(?i)\b[a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+:\d+:\d+:\s*(?:[A-Z]\d{3,4}\b|error|warning|note)",
        r"(?i)-->\s+[a-zA-Z0-9_./\-]+:\d+:\d+",
        r"(?i)\bTraceback \(most recent call last\):",
        r"(?i)\bAssertionError\b",
        r"(?i)\bFAILED\s+\(failures=\d+\)",
        r"(?i):::error\s+file=",
        r"(?i)\bdiagnostic\b",
    ),
    "weak": (
        r"(?i)\bcompiler error\b",
        r"(?i)\blinter error\b",
        r"(?i)\bcompilation failed\b",
        r"(?i)\bbuild failed\b",
        r"(?i)\bassertion failed\b",
    ),
}

DIAGNOSTIC_EXTRACTION = {
    "tool": (
        r"(?i)(\berror\[E\d+\]|-->\s*.*\.rs:\d+|rustc)",
        r"(?i)(error\s+TS\d+|TypeScript|\.ts\(\d+,\d+\))",
        r"(?i)\b(ruff|eslint|gcc|clang|pytest|mypy)\b",
    ),
    "file": (
        r"(?i)-->\s*([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+):\d+",
        r"([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+)\(\d+,\s*\d+\):\s*error",
        r"([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+):\d+:\d+:\s*"
        r"(?:[A-Z]\d{3,4}\b|error|warning|note)",
        # Plain path:line:col: ladder (gcc `src/main.c:12:5: error:`,
        # ruff `server.py:3:1: F401`); anchored at line start so log
        # timestamps cannot read as paths (review §8a/§8b). Indented
        # continuations keep the anchor via leading-blank tolerance
        # (B4, 2026-10-10 review: `  src/main.c:12:5:` fell to defaults).
        r"(?m)^[ \t]*([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+):\d+:\d+:",
        r'File "([^"]+)", line \d+',
    ),
    "line": (
        r"-->\s*[a-zA-Z0-9_./\-]+:(\d+)",
        r"\.[a-zA-Z0-9_]+\((\d+),\s*\d+\)",
        r"(?m)^[ \t]*[a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+:(\d+):\d+:",
        r":[a-zA-Z0-9_./\-]+:(\d+):\d+",
        r"line (\d+)",
    ),
    "column": (
        r"-->\s*[a-zA-Z0-9_./\-]+:\d+:(\d+)",
        r"\.[a-zA-Z0-9_]+\(\d+,\s*(\d+)\)",
        r"(?m)^[ \t]*[a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+:\d+:(\d+):",
        r":[a-zA-Z0-9_./\-]+:\d+:(\d+):",
    ),
    "severity": (r"(?i)\b(error|warning|note)\b",),
    "code": (
        r"(?i)\b(?:error|warning)\[([A-Z]\d+)\]",
        r"(?i)\b(TS\d{4,5})\b",
        # Case-sensitive on purpose (B11, 2026-10-10 review): the (?i)
        # flag let lowercase tokens like `pid1234` read as codes; only
        # uppercase-prefixed codes (F401, E0308, TS1234) match.
        r"\b([A-Z]{1,3}\d{3,4})\b",
        r"(?i)\b([A-Za-z]+Error)\b",
    ),
    # Full carry (no-truncation law, owner directive 2026-10-10): the
    # message is the COMPLETE diagnostic content after the marker — every
    # following line (source snippet, caret lines, help URL), not just
    # the marker line. Output is bounded by the request's 256 KiB input
    # cap (api.MAX_TEXT_CHARS); no length cap is imposed here.
    "message": (
        r"(?i)(?:error|warning)(?:\[[A-Z]\d+\])?:\s*((?s:.+))",
        r"(?i)(?:error TS\d+:)\s*((?s:.+))",
        r"(?i)(?:error|warning|note):\s*((?s:.+))",
        r"(?i)(?:AssertionError|TypeError|ValueError|SyntaxError):\s*((?s:.+))",
        # Fallback for the bare-code ladder (B5, 2026-10-10 review):
        # `server.py:3:1: F401 \`os\` imported but unused` carries its
        # message after the code token, with no error:/warning:/note: word.
        r"(?m)^[ \t]*[a-zA-Z0-9_./\-]+\.[a-zA-Z0-9_]+:\d+:\d+:"
        r"\s*[A-Z]\d{3,4}\s+((?s:.+))",
    ),
}

DIAGNOSTIC_DEFAULTS = {
    "tool": "unknown",
    "file": "",
    "line": 1,
    "column": 1,
    "severity": "error",
    "code": "",
    "message": "",
}

DIAGNOSTIC_EXAMPLES = [
    {
        "type": "diagnostic",
        "tool": "rustc",
        "file": "src/main.rs",
        "line": 12,
        "column": 18,
        "severity": "error",
        "code": "E0308",
        "message": "mismatched types: expected `u32`, found `&str`",
    },
    {
        "type": "diagnostic",
        "tool": "tsc",
        "file": "src/api/client.ts",
        "line": 45,
        "column": 7,
        "severity": "error",
        "code": "TS2322",
        "message": "Type 'string' is not assignable to type 'number'.",
    },
    {
        "type": "diagnostic",
        "tool": "ruff",
        "file": "backend/server.py",
        "line": 3,
        "column": 1,
        "severity": "warning",
        "code": "F401",
        "message": "`os` imported but unused",
    },
    {
        "type": "diagnostic",
        "tool": "pytest",
        "file": "tests/test_router.py",
        "line": 84,
        "column": 1,
        "severity": "error",
        "code": "AssertionError",
        "message": "assert 'issue' == 'commit'",
    },
]


# Compiled view of DIAGNOSTIC_EXTRACTION for the hand-written extractor
# below (same patterns, first match wins per field — the registry engine's
# contract, plus the tool normalization the pattern table cannot express).
_DIAG_EXTRACTION_COMPILED = tuple(
    (field, tuple(re.compile(p) for p in patterns))
    for field, patterns in DIAGNOSTIC_EXTRACTION.items())


def extract_diagnostic(text: str) -> dict:
    """Extract structured diagnostic fields with tool normalization.

    The rustc shapes capture marker substrings (`error[E0308]`,
    `--> src/main.rs:12`) that no regex can turn into the word "rustc";
    the normalization below does (review §8c). Everything else follows
    the registered pattern table: patterns tried in order per field,
    first match wins, misses fall into the validate/repair ladder.
    """
    obj = {"type": "diagnostic"}
    # Sanitize first (same contract as extract_agent_step): ANSI-wrapped
    # diagnostics must extract the real path/message, not the escape
    # residue (`31msrc/app.py`, `boom\x1b[0m`).
    clean = sanitize_log(text)
    for field, patterns in _DIAG_EXTRACTION_COMPILED:
        for pattern in patterns:
            m = pattern.search(clean)
            if m:
                obj[field] = m.group(1)
                break
    tool = str(obj.get("tool", ""))
    lowered = tool.lower()
    if lowered.startswith("error[e") or lowered.startswith("-->") \
            or ".rs:" in lowered:
        obj["tool"] = "rustc"
    elif lowered.startswith("error ts") or ".ts(" in lowered:
        obj["tool"] = "tsc"
    return obj


def register_diagnostic_schema() -> None:
    """Register the ``diagnostic`` schema in the runtime registry (idempotent)."""
    if "diagnostic" in registry.active():
        return
    registry.register_schema(
        "diagnostic",
        DIAGNOSTIC_SPEC,
        DIAGNOSTIC_ROUTING,
        DIAGNOSTIC_EXTRACTION,
        DIAGNOSTIC_DEFAULTS,
        DIAGNOSTIC_EXAMPLES,
        extractor=extract_diagnostic,
    )


# ANSI escape sequences for log sanitization — single-ownership
# alternatives (review §9): each alternative consumes a disjoint byte
# shape, so no alternative can start a match inside another's span and
# the quadratic backtracking on long `[0-9;]` runs is gone.
#   CSI:          ESC [ params [intermediates] final-byte. The parameter
#                 class admits `?` (private-mode markers: `ESC[?2004h`,
#                 `ESC[?25l` — v12, 2026-10-10) and 0x20-0x2F
#                 intermediate bytes (`ESC[0 q` cursor style); without
#                 them the sequences survive sanitization and glue into
#                 extracted paths (`2004hsrc/app.py`).
#   9b CSI:       the same on the 8-bit C1 spelling
#   charset:      ESC ( or ESC ) plus one designator ("ESC ( 0")
#   OSC:          ESC ] ... BEL or ESC \  (titles, OSC-8 hyperlinks)
ANSI_ESCAPE_RE = re.compile(
    r"\x1b\[[0-9;:?]*[ -/]*[a-zA-Z]"
    r"|\x9b[0-9;:?]*[ -/]*[a-zA-Z]"
    r"|\x1b[()][0-9A-Za-z]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
)
CI_WRAPPER_RE = re.compile(
    r"^(?:##\[(?:group|endgroup|error|warning)\]|::(?:error|warning).*?::)\s*"
)
TIMESTAMP_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+")


def sanitize_log(text: str) -> str:
    """Strip ANSI escape sequences, squash carriage returns, remove CI envelopes."""
    # Strip ANSI colors/codes
    text = ANSI_ESCAPE_RE.sub("", text)
    # Normalize \r: split on carriage returns and take final overwrite per line
    lines = []
    for line in text.split("\n"):
        if "\r" in line:
            parts = [p for p in line.split("\r") if p]
            line = parts[-1] if parts else ""
        # Strip CI wrapper directives and timestamps
        line = TIMESTAMP_PREFIX_RE.sub("", line)
        line = CI_WRAPPER_RE.sub("", line)
        lines.append(line)
    return "\n".join(lines).strip()
