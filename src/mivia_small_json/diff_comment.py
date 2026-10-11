# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Git diffs, unified patches, and code review comments schema and extractors.

Implements two schemas:
1. ``diff``:
   - fields: ``old_path``, ``new_path``, ``change_type``, ``additions``, ``deletions``, ``patch``
   - change_type: ``add``, ``modify``, ``delete``, ``rename``
   - captures unified git diffs, hunk lines, and patch changesets.
2. ``review_comment``:
   - fields: ``file``, ``line``, ``comment``, ``severity``, optional ``suggested_code``
   - severity: ``nit``, ``suggestion``, ``warning``, ``blocking``
   - captures human and AI code review inline comments with optional code suggestions.
"""

from __future__ import annotations

import re

from . import registry

DIFF_CHANGE_TYPES = ("add", "modify", "delete", "rename")
REVIEW_SEVERITIES = ("nit", "suggestion", "warning", "blocking")

DIFF_SPEC = {
    "type": {"const": "diff"},
    "old_path": {"type": str},
    "new_path": {"type": str},
    "change_type": {"type": str, "enum": DIFF_CHANGE_TYPES},
    "additions": {"type": int},
    "deletions": {"type": int},
    "patch": {"type": str},
}

REVIEW_COMMENT_SPEC = {
    "type": {"const": "review_comment"},
    "file": {"type": str},
    "line": {"type": int},
    "comment": {"type": str},
    "severity": {"type": str, "enum": REVIEW_SEVERITIES},
    "suggested_code": {"type": str, "optional": True},
}

DIFF_ROUTING = {
    "strong": (
        r"(?m)^diff --git a/.* b/.*",
        r"(?m)^@@ -\d+,\d+ \+\d+,\d+ @@",
        r"(?m)^--- (?:a/|/dev/null)",
        r"(?m)^\+\+\+ (?:b/|/dev/null)",
    ),
    "weak": (
        r"\bindex [0-9a-f]{7,40}\.\.[0-9a-f]{7,40}\b",
        r"\bnew file mode \d+\b",
        r"\bdeleted file mode \d+\b",
        r"\bsimilarity index \d+%\b",
    ),
}

REVIEW_COMMENT_ROUTING = {
    "strong": (
        r"(?i)\b(?:LGTM|nit|suggestion|blocking)\b:?",
        r"(?i)```suggestion",
        r"(?i)\bcode review comment\b",
        r"(?i)\bPR review comment\b",
    ),
    "weak": (
        r"(?i)\bline \d+:\s*(?:please|could you|consider|should)\b",
        r"(?i)\bneeds test\b",
        r"(?i)\btypo\b",
        r"(?i)\bapprove(?:d)? with comments\b",
    ),
}

DIFF_DEFAULTS = {
    "old_path": "",
    "new_path": "",
    "change_type": "modify",
    "additions": 0,
    "deletions": 0,
    "patch": "",
}

REVIEW_COMMENT_DEFAULTS = {
    "file": "",
    "line": 1,
    "comment": "",
    "severity": "suggestion",
    "suggested_code": "",
}

# Quoted git paths (`diff --git a/"src/my file.py" b/...`) alongside the
# usual \S+ spellings (review §8d).
_QUOTED_PATH = r'"(?:[^"\\]|\\.)*"'

_DIFF_HEADER_RE = re.compile(
    rf'^diff --git a/(?P<old>{_QUOTED_PATH}|\S+) b/(?P<new>{_QUOTED_PATH}|\S+)',
    re.MULTILINE)
# `/dev/null` must be tried FIRST, outside the named group (N6,
# review-val 2026-10-10): the greedy `\S+` alternative matched `/dev/null`
# before the literal branch could, so the group never stayed unset and
# add/delete diffs reported `/dev/null` as a real path instead of letting
# the header-path backfill run.
_DIFF_OLD_RE = re.compile(
    rf'^--- (?:/dev/null|(?:a/)?(?P<old>{_QUOTED_PATH}|\S+))',
    re.MULTILINE)
_DIFF_NEW_RE = re.compile(
    rf'^\+\+\+ (?:/dev/null|(?:b/)?(?P<new>{_QUOTED_PATH}|\S+))',
    re.MULTILINE)

_REVIEW_FILE_RE = re.compile(r"(?i)(?:file|path):\s*`?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`?")
_REVIEW_LINE_RE = re.compile(r"(?i)\bline\s*[:#]?\s*(\d+)")
_SUGGESTION_BLOCK_RE = re.compile(r"```suggestion\s*\n(.*?)\n```", re.DOTALL)


def _clean_path(raw: str) -> str:
    """Unquote a quoted git path and drop the a/ b/ prefix git puts inside
    the quotes; unquoted paths pass through unchanged."""
    if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
        return re.sub(r"^[ab]/", "", raw[1:-1])
    return raw


def _diff_sections(text: str):
    """Split a multi-file patch into (header_match, section_text) pairs at
    each ``diff --git`` line; empty when the text has no git header."""
    heads = list(_DIFF_HEADER_RE.finditer(text))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append((m, text[m.start():end]))
    return out


def _section_header(section_text: str) -> str:
    """The metadata lines of one file section: everything before the first
    hunk marker. Change-type probes must never read hunk content (review
    §8a: a `-rename from here` hunk line flipped the change type)."""
    lines = []
    for line in section_text.splitlines():
        if line.startswith("@@"):
            break
        lines.append(line)
    return "\n".join(lines)


def _count_patch_changes(lines):
    """Count (additions, deletions) over one file's patch lines.

    Hunk-tracked whenever a ``@@`` marker exists. A diff with ``---``/``+++``
    headers and +/- body lines but NO hunk headers (hand-written or
    truncated patches) never entered a hunk and reported 0/0 (B12,
    review-val 2026-10-10), so without any ``@@`` the +/- lines after the
    ``---``/``+++`` header block count (a bare +/- patch with no headers
    counts from the top).

    Hunk-less semantics are a pinned choice (survfix 2026-10-10, v13):
    such a patch is intrinsically ambiguous, so ONLY the first
    ``--- ``-shaped line and the first ``+++ ``-shaped line are treated
    as the header block. Every later line follows the body rules: a
    ``--- text``-shaped line inside the body is a deletion (a deleted
    line whose original content began with ``-- `` renders exactly like
    that — the old any-``--- ``-skips rule miscounted it 1/0 vs 1/1, and
    dropped the content before an email-style ``--- `` separator, 1/0 vs
    2/1), a ``+++ payload`` line follows the addition rules, ``++++
    payload`` stays an addition and ``---- payload`` a deletion (review
    §8b), and a bare ``---`` rule line is not a deletion.
    """
    adds = 0
    dels = 0
    if not any(line.startswith("@@") for line in lines):
        seen_old_header = False
        seen_new_header = False
        for line in lines:
            if line.startswith("--- ") and not seen_old_header:
                seen_old_header = True
                continue
            if line.startswith("+++ ") and not seen_new_header:
                seen_new_header = True
                continue
            if line == "---":
                continue
            if line.startswith("+"):
                adds += 1
            elif line.startswith("-"):
                dels += 1
        return adds, dels
    in_hunk = False
    for line in lines:
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            adds += 1
        elif line.startswith("-"):
            dels += 1
    return adds, dels


def _change_type_of(header: str) -> str:
    if "new file mode" in header or re.search(r"^--- /dev/null", header,
                                              re.MULTILINE):
        return "add"
    if "deleted file mode" in header or re.search(r"^\+\+\+ /dev/null",
                                                  header, re.MULTILINE):
        return "delete"
    if re.search(r"^rename from ", header, re.MULTILINE):
        return "rename"
    return "modify"


def extract_diff(text: str) -> dict:
    """Extract structured fields from unified git diff text.

    Multi-file patches are segmented per ``diff --git`` section; the
    reported file is the ADDED section when one exists, else the first
    section (review §8a: a modify+add patch used to report the first
    file's paths with the second file's change type).
    """
    res = dict(DIFF_DEFAULTS)
    res["type"] = "diff"
    res["patch"] = text

    sections = _diff_sections(text)
    if sections:
        chosen = next(
            ((m, sec) for m, sec in sections
             if _change_type_of(_section_header(sec)) == "add"),
            sections[0])
        m_hdr, section = chosen
        header = _section_header(section)
        res["old_path"] = _clean_path(m_hdr.group("old"))
        res["new_path"] = _clean_path(m_hdr.group("new"))
        m_old = _DIFF_OLD_RE.search(header)
        if m_old and m_old.group("old"):
            res["old_path"] = _clean_path(m_old.group("old"))
        m_new = _DIFF_NEW_RE.search(header)
        if m_new and m_new.group("new"):
            res["new_path"] = _clean_path(m_new.group("new"))
        res["change_type"] = _change_type_of(header)

        if not res["old_path"] and res["new_path"]:
            res["old_path"] = res["new_path"]
        elif not res["new_path"] and res["old_path"]:
            res["new_path"] = res["old_path"]

        # Count additions & deletions inside the chosen section only.
        res["additions"], res["deletions"] = _count_patch_changes(
            section.splitlines())
        return res

    # No git header: keep the whole-text ladder (bare hunk blocks, etc.).
    m_old = _DIFF_OLD_RE.search(text)
    if m_old and m_old.group("old"):
        res["old_path"] = _clean_path(m_old.group("old"))
    m_new = _DIFF_NEW_RE.search(text)
    if m_new and m_new.group("new"):
        res["new_path"] = _clean_path(m_new.group("new"))

    if not res["old_path"] and res["new_path"]:
        res["old_path"] = res["new_path"]
    elif not res["new_path"] and res["old_path"]:
        res["new_path"] = res["old_path"]

    header = _section_header(text)
    res["change_type"] = _change_type_of(header)

    res["additions"], res["deletions"] = _count_patch_changes(
        text.splitlines())

    return res


def extract_review_comment(text: str) -> dict:
    """Extract structured fields from code review comment text."""
    res = dict(REVIEW_COMMENT_DEFAULTS)
    res["type"] = "review_comment"
    res["comment"] = text.strip()

    m_file = _REVIEW_FILE_RE.search(text)
    if m_file:
        res["file"] = m_file.group(1)

    m_line = _REVIEW_LINE_RE.search(text)
    if m_line:
        try:
            res["line"] = int(m_line.group(1))
        except ValueError:
            res["line"] = 1

    # Word-bound anchored severity probes (review §8c): the old substring
    # checks read "nonblocking" as blocking and "minority" as a nit. A
    # preceding hyphen is anchored too (B8, review-val 2026-10-10):
    # `non-blocking: looks fine` is a negation, not a blocking verdict.
    lower = text.lower()
    if re.search(r"(?<![\w-])blocking\b", lower):
        res["severity"] = "blocking"
    elif re.search(r"(?<![\w-])(?:nit|minor)\b", lower):
        res["severity"] = "nit"
    elif re.search(r"\bwarnings?\b|\bwarn(?:ed|ing)?\b", lower):
        res["severity"] = "warning"
    else:
        res["severity"] = "suggestion"

    m_sug = _SUGGESTION_BLOCK_RE.search(text)
    if m_sug:
        res["suggested_code"] = m_sug.group(1)

    return res


DIFF_EXTRACTION = {
    "old_path": (
        r"^diff --git a/(\S+) b/\S+",
        r"^--- a/(\S+)",
    ),
    "new_path": (
        r"^diff --git a/\S+ b/(\S+)",
        r"^\+\+\+ b/(\S+)",
    ),
    "change_type": (
        r"\b(add|modify|delete|rename)\b",
    ),
    "additions": (
        r"\b(\d+)\s+additions?\b",
    ),
    "deletions": (
        r"\b(\d+)\s+deletions?\b",
    ),
    "patch": (
        r"([\s\S]+)",
    ),
}

REVIEW_COMMENT_EXTRACTION = {
    "file": (
        r"(?i)(?:file|path):\s*`?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`?",
        r"`([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`",
    ),
    "line": (
        r"(?i)\bline\s*[:#]?\s*(\d+)",
    ),
    "comment": (
        r"^([^\n]+)",
    ),
    "severity": (
        r"(?i)\b(blocking|nit|suggestion|warning)\b",
    ),
    "suggested_code": (
        r"```suggestion\s*\n(.*?)\n```",
    ),
}

DIFF_EXAMPLES = [
    {
        "type": "diff",
        "old_path": "src/main.py",
        "new_path": "src/main.py",
        "change_type": "modify",
        "additions": 1,
        "deletions": 1,
        "patch": (
            "diff --git a/src/main.py b/src/main.py\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        ),
    }
]

REVIEW_COMMENT_EXAMPLES = [
    {
        "type": "review_comment",
        "file": "src/main.py",
        "line": 10,
        "comment": "nit: rename variable",
        "severity": "nit",
        "suggested_code": "new_name = 1",
    }
]


def register_diff_and_review_schemas() -> None:
    """Register both diff and review_comment schemas in the registry."""
    if "diff" not in registry.active():
        registry.register_schema(
            "diff",
            DIFF_SPEC,
            DIFF_ROUTING,
            DIFF_EXTRACTION,
            DIFF_DEFAULTS,
            DIFF_EXAMPLES,
            extractor=extract_diff,
        )
    if "review_comment" not in registry.active():
        registry.register_schema(
            "review_comment",
            REVIEW_COMMENT_SPEC,
            REVIEW_COMMENT_ROUTING,
            REVIEW_COMMENT_EXTRACTION,
            REVIEW_COMMENT_DEFAULTS,
            REVIEW_COMMENT_EXAMPLES,
            extractor=extract_review_comment,
        )
