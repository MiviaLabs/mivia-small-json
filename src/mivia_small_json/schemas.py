# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Declarative schema registry — the single source of truth for output formats.

One entry per record type: fields, types, enums, formats, optionality.
``validate`` derives its checks from this table, ``rules`` imports the
enums and alias maps, and ``synth`` draws values from them. Adding an
output format means adding one entry here plus extraction patterns and
training examples — nothing else.

Level enums and aliases follow the OpenTelemetry log severity model
(TRACE, DEBUG, INFO, WARN, ERROR, FATAL), so records map cleanly onto
observability tooling. ``commit_type`` follows Conventional Commits and
is optional: when the message carries no conventional prefix, the field
is absent rather than empty.
"""

from __future__ import annotations

import re

SCHEMAS = ("issue", "log_event", "commit")
PRIORITIES = ("low", "medium", "high", "critical")
LEVELS = ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL")
STATES = ("open", "closed")
COMMIT_TYPES = ("feat", "fix", "chore", "docs", "style", "refactor",
                "perf", "test", "build", "ci")
# Multi-turn agent trajectory roles and terminal step states
# (single source of truth for the ``agent_step`` schema).
AGENT_ROLES = ("assistant", "tool", "user", "environment")
AGENT_STATUSES = ("success", "error", "timeout", "interrupted")

# Common severity spellings -> canonical OTel level (repair + extraction).
LEVEL_ALIASES = {
    "VERBOSE": "TRACE", "TRACE": "TRACE",
    "WARNING": "WARN", "WARN": "WARN",
    "ERR": "ERROR", "ERROR": "ERROR",
    "CRIT": "FATAL", "CRITICAL": "FATAL", "FATAL": "FATAL",
    # Additive enum extension (LogPai benchmark, 2026-10-05): syslog-style
    # severities that appear in real platform logs.
    "NOTICE": "INFO", "SEVERE": "ERROR", "EMERG": "FATAL", "ALERT": "FATAL",
    # Android LogCat / syslog single-letter severities
    "V": "TRACE", "D": "DEBUG", "I": "INFO", "W": "WARN", "E": "ERROR", "F": "FATAL",
}

# Fractional seconds up to 9 digits: platforms emit nanosecond
# timestamps and extraction must be able to carry them (normalize_ts
# truncates to the 6 digits datetime keeps).
ISO_TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$"
)
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

# schema -> field -> rule. Rule keys: "const" | "type", optional "enum",
# "format" ("iso8601" | "sha" | "json") and "optional" (field may be absent).
SPEC: dict[str, dict] = {
    "issue": {
        "type": {"const": "issue"},
        "id": {"type": str},
        "title": {"type": str},
        "body": {"type": str, "optional": True},
        "labels": {"type": list},
        "priority": {"type": str, "enum": PRIORITIES},
        "state": {"type": str, "enum": STATES},
    },
    "log_event": {
        "type": {"const": "log_event"},
        "level": {"type": str, "enum": LEVELS},
        "component": {"type": str},
        "message": {"type": str},
        # A log line without a timestamp omits `ts` — it must not learn
        # the wall clock of whoever classified it (owner sweep 2026-10-10).
        "ts": {"type": str, "format": "iso8601", "optional": True},
    },
    "commit": {
        "type": {"const": "commit"},
        # Commit-shaped text without a hex sha omits `sha` — "0000000" was
        # a fabricated placeholder (owner sweep 2026-10-10).
        "sha": {"type": str, "format": "sha", "optional": True},
        "message": {"type": str},
        "files_changed": {"type": int},
        "author": {"type": str},
        "commit_type": {"type": str, "enum": COMMIT_TYPES, "optional": True},
    },
}

FIELDS = {s: tuple(SPEC[s]) for s in SCHEMAS}
OPTIONAL = {s: frozenset(f for f, r in SPEC[s].items() if r.get("optional"))
            for s in SCHEMAS}

# schema -> {"strong": [regex...], "weak": [regex...]} routing marker
# regexes for the 2-points-strong / 1-point-weak keyword router, in the
# same shape ``registry.register_schema`` accepts for user schemas.
# Served through ``registry.routing_for`` like any registered schema;
# ``rules`` holds no routing tables of its own.
BUILTIN_ROUTING = {
    "issue": {
        "strong": (
            r"\bISS-\d{2,6}\b", r"(?i)\bissue\b", r"(?i)\bpriority\b",
            r"(?i)\blabels?\b",
        ),
        "weak": (
            r"#\d{2,6}\b", r"(?i)\b(?:open|closed)\b",
            r"(?i)\b(?:low|medium|high|critical)\b",
        ),
    },
    "log_event": {
        "strong": (
            r"\b(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\b", r"\[[\w.-]+\]\s*:",
            r"(?i)\bcomp(?:onent)?\s*[:=]", r"\d{4}-\d{2}-\d{2}T",
            r"\[(?:err|out|stdout|stderr)\]",
        ),
        "weak": (
            r"(?i)\blev?el\s*[:=]", r"(?i)\blvl\s*[:=]", r"(?i)\bmsg\s*[:=]",
            r"(?i)\b\d+ms\b",
        ),
    },
    "commit": {
        "strong": (
            r"(?i)\bcommit\b", r"\b[0-9a-f]{40}\b", r"\d+\s+files?\s+changed",
            r"(?i)\bauthor\s*[:=]",
        ),
        "weak": (
            r"\b[0-9a-f]{7,39}\b",
            r"(?i)^(?:fix|feat|chore|docs|refactor|test|perf)\b",
            r"(?i)\bby\s+[A-Z]",
        ),
    },
}

# Repair/template fallback values for the built-in specs. Served through
# ``registry.defaults_for`` like any registered schema. Fields with a
# computed default (iso8601/sha formats: ts, sha) stay unstated here and
# fall through the rule-shape ladder in ``validate._default``.
BUILTIN_DEFAULTS = {
    "id": "", "title": "", "labels": [], "priority": "medium", "state": "open",
    "level": "INFO", "component": "", "message": "", "files_changed": 0,
    "author": "",
}
