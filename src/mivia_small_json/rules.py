# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Grep/rules baseline: keyword schema routing + regex field extraction.

Honesty constraint (AGENTS.md rule 1): this module sees only the raw
record text — never annotated records, labels, or split metadata. It is
benched standalone as the baseline, and reused as the primary extraction
layer of the full pipeline. Imperfect fields are expected; validity is
restored downstream by validate/repair, while F1 vs the reference
annotations shows what grep can and cannot do.
"""

from __future__ import annotations

import re

from . import registry
from .schemas import LEVEL_ALIASES
from .validate import (
    normalize_sha,
    normalize_ts,
    repair,
    template,
    validate,
)

# --- routing ---------------------------------------------------------------


def route_scores(text: str) -> dict:
    """2 points per strong marker, 1 per weak one. Deterministic.

    Every schema — built-in or registered — resolves its compiled
    markers through the registry (declarative table in ``schemas.py``).
    """
    out = {}
    for schema in registry.active():
        strong_p, weak_p = registry.routing_for(schema)
        out[schema] = 2 * sum(1 for p in strong_p if p.search(text)) + sum(
            1 for p in weak_p if p.search(text)
        )
    return out


def route(text: str) -> str:
    """Keyword router: highest score wins, built-ins break ties first,
    then earlier registration order."""
    scores = route_scores(text)
    schemas = registry.active()
    return max(schemas, key=lambda s: (scores[s], -schemas.index(s)))


# --- extraction ------------------------------------------------------------

_TS_ANY = (r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
           r"(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?")
_LEVEL_BODY = (r"\b(?:DEBUG|INFO|WARN|WARNING|ERR|ERROR|FATAL|"
               r"CRIT(?:ICAL)?|TRACE|VERBOSE|NOTICE|SEVERE|EMERG|ALERT)\b")
# Captured-group contract: group(1) is ALWAYS the bare level token. The
# optional "[level:]" / "level=" style prefixes are matched but NEVER
# captured — every consumer reads group(1) (the "[ERROR]" strip and the
# alias lookup both read group(0) before, which leaked brackets, then
# "LEVEL=" prefixes, into the token and silently repaired to INFO).
_LEVEL_ANY = (r"\[?(?:level\s*[:=]\s*)?(" + _LEVEL_BODY + r")\]?")


def _first(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1) if m else ""


def _longest_chunk(text, min_len=4):
    parts = re.split(r"[|\n]|\s::\s|\s{2,}|(?:^|\s)[—–]\s*", text)
    parts = [p.strip(" \t[](){}<>,") for p in parts]
    parts = [p for p in parts if len(p) >= min_len and re.search(r"[A-Za-z]", p)]
    return max(parts, key=len) if parts else ""


# Technical-unit tokens that match the PROJECT-123 shape but are not ids
# (review P1-3: SHA-256, UTF-16, ISO-8859, CVE-2024-…).
_ID_TECH_TOKENS = frozenset({
    "SHA", "MD5", "UTF", "ISO", "CVE", "RGB", "RGBA", "UUID", "JWT",
    "HTTP", "HTTPS", "DNS", "TLS", "SSL", "URL", "URI", "URN", "API",
    "SQL", "CSS", "HTML", "XML", "JSON", "YAML", "CSV", "PDF", "PNG",
    "GIF", "SVG", "JPEG", "AWS", "GCP", "CPU", "GPU", "RAM", "SSD",
    "HDD", "BIOS", "VPN", "SMTP", "IMAP", "FTP", "SSH", "TCP", "UDP",
    "IP", "MAC", "LAN", "WAN", "USB", "PCI", "RFC", "GPS",
})


def _extract_issue(text: str) -> dict:
    low = text.lower()
    # Prefixed ids: the ISS- ladder first, then a generic PROJECT-123
    # pattern (PAY-2214, GH-4711, ...). First match wins; kept uppercase.
    m = re.search(r"\bISS-(\d{2,6})\b", text)
    if m:
        fid = "ISS-" + m.group(1)
    else:
        fid = ""
        # PROJECT-123 scan first; a bare "#digits" is only a fallback when
        # no explicit ticket id exists (review 2026-10-09, F2d).
        for m in re.finditer(r"\b([A-Z][A-Z0-9]{1,9}-\d{2,6})\b", text):
            token = m.group(1)
            head = token.split("-")[0]
            if head in _ID_TECH_TOKENS:
                continue
            # A trailing -<digits> segment (CVE-2024-31337) is a technical
            # identifier, not a ticket id.
            rest = text[m.end():]
            if re.match(r"-\d{2,6}\b", rest):
                continue
            fid = token
            break
        if not fid:
            m = re.search(r"#(\d{2,6})\b", text)
            if m:
                fid = "ISS-" + m.group(1)
    # "urgent" is a plausible priority word that is NOT in the enum: an
    # unmatched priority used to fall through empty and get repaired to
    # the default "medium" — a silent fabricated value. Map it instead.
    _PRIORITY_WORDS = r"(low|medium|high|critical|urgent)"
    priority = (
        _first(rf"priority\s*[:=#]?\s*{_PRIORITY_WORDS}\b", low)
        or _first(rf"\bp(?:rio)?\s*[:=]\s*{_PRIORITY_WORDS}\b", low)
        or _first(rf"\({_PRIORITY_WORDS}\)", low)
        or _first(rf"\b{_PRIORITY_WORDS}\b", low)
    )
    priority = {"urgent": "critical"}.get(priority, priority)
    state = (
        _first(r"state\s*[:=]?\s*(open|closed)\b", low)
        or _first(r"\b(open|closed)\b", low)
    )
    labels_raw = _first(r"labels?\s*[:=]\s*([^\n|]+)", low)
    labels_raw = re.split(r"(?i)\bstate\b|\bpriority\b", labels_raw)[0]
    labels = []
    for token in re.split(r"[,;/]+", labels_raw):
        token = re.sub(r"[^a-z0-9._-]", "", token.strip())
        if token and not token.isdigit():
            labels.append(token)
    cleaned = re.sub(r"\bISS-\d+\b|#\d+\b", " ", text)
    # A ticket id leaks into the title when the id word stays in the
    # chunk text ("PAY-2214: checkout failed" -> title starting with the
    # id); strip any PROJECT-123 token the id-scan family recognizes.
    cleaned = re.sub(r"\b[A-Z][A-Z0-9]{1,9}-\d{2,6}\b", " ", cleaned)
    cleaned = re.sub(r"(?i)labels?\s*[:=][^\n|]*", " ", cleaned)
    cleaned = re.sub(r"(?i)\b(?:state|priority)\s*[:=#]?\s*\w+", " ", cleaned)
    cleaned = re.sub(r"(?i)\bp(?:rio)?\s*[:=]\s*\w+", " ", cleaned)
    title = re.sub(r"^(?:bug|issue|task)\s*#?\d*\s*[:\-]\s*", "", _longest_chunk(cleaned),
                   flags=re.I).strip(" -:—–")
    return {"type": "issue", "id": fid, "title": title, "labels": labels,
            "priority": priority, "state": state}


def _extract_log(text: str) -> dict:
    # An EXPLICIT level label wins over an incidental severity word in
    # the message ("msg=connection error level=WARN" is WARN, not
    # ERROR — the first severity word used to win, review P1 2026-10-10).
    m = re.search(r"(?i)\blevel\s*[:=]\s*(" + _LEVEL_BODY + r")", text)
    if not m:
        m = re.search(_LEVEL_ANY, text, re.I)
    level = LEVEL_ALIASES.get(m.group(1).upper(), m.group(1).upper()) if m else ""
    component = (
        _first(r"\b([A-Za-z0-9_.-]+)(?:\([^)]*\))?\[\d+\]:\s*", text)
        or _first(r"\[([\w.-]{1,40})\]\s*:", text)
        or _first(r"(?i)\bcomp(?:onent)?\s*[:=]\s*([\w.-]{1,40})", text)
        or _first(r"^(?!(?i:ERROR|WARN|WARNING|INFO|DEBUG|TRACE|FATAL|"
                  r"CRITICAL|NOTICE|ERR|SEVERITY)\b)"
                  r"(?!(?i:level|priority|severity|msg|message|comp(?:onent)?|ts)\s*[:#=])"
                  r"([\w.-]{2,24}):\s+\S",
                  text, re.M)
    )
    # Meta labels are not components — "level:" "priority:" "msg:" keys
    # matched the line-initial name[: ] pattern (review 2026-10-10).
    # The leftover ": value" colon prefix is stripped from the message
    # below regardless of whether the component survived.
    meta_component = bool(component and re.match(
        r"(?i)^(?:level|priority|severity|msg|message|"
        r"comp(?:onent)?|ts|"
        r"debug|info|warn|warning|err(?:or)?|fatal|"
        r"crit(?:ical)?|trace|verbose|notice|severe|"
        r"emerg|alert)$", component))
    if meta_component:
        component = ""
    if component and component.isdigit():
        component = ""
    colon_msg = meta_component or bool(
        re.match(r"(?i)^(?:level|priority|severity|crit(?:ical)?|warn(?:ing)?|"
                 r"err(?:or)?|info|debug|trace|notice|severe|emerg|alert)\s*[:#]",
                 text.strip()))
    ts_m = re.search(_TS_ANY, text)
    ts = normalize_ts(ts_m.group(0).replace(",", ".")) if ts_m else ""
    msg = _first(r"(?i)\b(?:message|msg)\s*[:=]\s*(.+)$", text)
    if not msg:
        # The LAST "]" before the message may be the level's bracket
        # ("[ts] [error] mod_jk..."): take the text after the final
        # bracket-cluster that still carries the level or a client tag,
        # not the first "]" (which leaked "[error] ..." into the
        # message on exactly the Apache shape the level fix targeted).
        msg = ""
        m_last = None
        for m_last in re.finditer(r"\]\s*:?\s*", text):
            pass
        if m_last and m_last.end() < len(text):
            tail = text[m_last.end():]
            if tail.strip() and not re.match(
                    r"(?i)^\[?(?:DEBUG|INFO|WARN|WARNING|ERR|ERROR|FATAL|"
                    r"CRIT(?:ICAL)?|TRACE|VERBOSE|NOTICE|SEVERE|EMERG|"
                    r"ALERT)\b", tail.strip()):
                msg = tail
    if not msg:
        cleaned = re.sub(_TS_ANY, " ", text)
        cleaned = re.sub(_LEVEL_ANY, " ", cleaned, flags=re.I)
        cleaned = re.sub(r"(?i)\b(?:lev?el|lvl|comp(?:onent)?|ts)\s*[:=]", " ", cleaned)
        # A meta-label key whose value was consumed as the level
        # ("severity: ERROR disk full" -> "severity:  disk full") is
        # stripped BEFORE chunk selection, not after — the bare key can
        # win the longest-chunk tie and its later removal erased the
        # whole message (review P1, 2026-10-10).
        cleaned = re.sub(r"(?i)(?:^|\s)(?:severity|level|priority)\s*[:#-]\s*", " ", cleaned)
        msg = _longest_chunk(cleaned, min_len=3)
    msg = msg.strip()
    if (component or colon_msg) and msg.startswith(":"):
        # A meta-label component was dropped above ("CRIT: db down"):
        # the leftover colon prefix goes too.
        msg = msg.lstrip(": ").strip()
    if component and msg.lower().startswith(component.lower()):
        msg = re.sub(rf"^{re.escape(component)}\s*:\s*", "", msg, flags=re.I)
    # No length cap (no-truncation law, owner directive 2026-10-10): the
    # message is the user's data and is carried verbatim. Output is
    # bounded by the request's 256 KiB input cap (api.MAX_TEXT_CHARS), so
    # a record can never exceed what the user sent — no cap needed here.
    return {"type": "log_event", "level": level, "component": component,
            "message": msg, "ts": ts}


def _extract_commit(text: str) -> dict:
    # The bare-fallback sha (no "commit" keyword) must carry at least one
    # hex LETTER: pure-digit runs in prose are dates, phone numbers and
    # order ids far more often than shas (2026-10-09 probe: "release
    # 4015551234" and "for 20261009" both extracted as a valid sha — a
    # wrong-but-valid id of exactly the masked-default bug class). A
    # pure-digit sha still extracts through the "commit <sha>" keyword
    # path above. The digit requirement also keeps 7-letter English
    # words ("defaced") out; an all-letter hex object name is no longer
    # extracted — a deliberate recall trade (review P2).
    m = re.search(r"\bcommit\s+([0-9a-f]{7,40})\b", text, re.I) or re.search(
        r"(?<![0-9a-zA-Z])((?=[0-9a-f]*[a-f][0-9a-f]*\d|\d[0-9a-f]*[a-f])[0-9a-f]{7,40})(?![0-9a-zA-Z])",
        text,
    )
    sha = m.group(1).lower() if m else ""
    files_m = re.search(r"(\d+)\s+files?\s+changed", text) or re.search(
        r"(?i)files?\s*[:=]\s*(\d+)", text
    ) or re.search(r"(?i)files?_?changed\s*[:=]\s*(\d+)", text)
    files = int(files_m.group(1)) if files_m else 0
    ct = _first(r"^\s*(feat|fix|chore|docs|style|refactor|perf|test|build|ci)"
                r"(?:\([a-z0-9._/-]+\))?:", text, re.I)
    author = _first(r"(?i)\bauthor\s*[:=]\s*([^\n|—–]+)", text)
    if author:
        author = re.sub(r"\s*\([^)]*\)\s*$", "", author).strip()
    author = author or _first(
        r"(?i)\bby\s+([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,3})", text
    )
    cleaned = re.sub(r"\bcommit\b", " ", text, flags=re.I)
    # The digit requirement keeps 7-letter English words ("defaced") out
    # of the sha slot; an all-letter hex object name is no longer
    # extracted — a deliberate recall trade (review P2).
    cleaned = re.sub(
        r"(?<![0-9a-zA-Z])(?=[0-9a-f]*[a-f][0-9a-f]*\d|\d[0-9a-f]*[a-f])[0-9a-f]{7,40}(?![0-9a-zA-Z])",
        " ", cleaned)
    cleaned = re.sub(r"\d+\s+files?\s+changed", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"(?i)\b\(?files?\s*[:=]?\s*\d*\)?", " ", cleaned)
    cleaned = re.sub(r"(?i)\bauthor\s*[:=][^\n|—–]*", " ", cleaned)
    cleaned = re.sub(r"(?i)\bby\s+[A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,3}", " ", cleaned)
    cleaned = re.sub(r"<[^>]*>", " ", cleaned)
    # No length cap (no-truncation law, owner directive 2026-10-10): the
    # selected chunk is carried verbatim. Output is bounded by the
    # request's 256 KiB input cap (api.MAX_TEXT_CHARS) — no cap needed.
    message = _longest_chunk(cleaned, min_len=3).lstrip(":; ")
    # No sha in the text -> empty string, NOT a zero-padded placeholder:
    # 'sha' is optional, and repair drops the empty field (a fabricated
    # '0000000' would pass the format check and never reach repair).
    out = {"type": "commit", "sha": (normalize_sha(sha) if sha else ""),
           "message": message,
           "files_changed": files, "author": (author or "").strip()}
    if ct:
        out["commit_type"] = ct.lower()
    return out


BUILTIN_EXTRACTORS = {"issue": _extract_issue, "log_event": _extract_log,
                      "commit": _extract_commit}


def extract(text: str, schema: str) -> dict:
    """Extractor selection is uniform: the registry hands out the tuned
    hand-written extractor for built-ins (cross-field logic stays code)
    and the compiled pattern engine for registered schemas."""
    return registry.extractor_for(schema)(text)


def emit(text: str, router=route):
    """Constrained-emission pipeline for one raw record.

    1. route to a schema, 2. regex-extract fields, 3. strict-validate,
    4. on errors: error-fed repair at most twice, 5. last resort: minimal
    valid template (validity is never sacrificed).

    Returns ``(obj, repair_attempts, schema, template_fallback)`` — the
    last flag is True only when the minimal template was emitted, so
    callers count fallbacks honestly instead of guessing.
    """
    schema = router(text)
    obj = extract(text, schema)
    errors = validate(obj, schema)
    attempts = 0
    while errors and attempts < 2:
        obj = repair(obj, errors, schema)
        errors = validate(obj, schema)
        attempts += 1
    if errors:
        return template(schema), attempts, schema, True
    return obj, attempts, schema, False
