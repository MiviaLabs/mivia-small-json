# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Collect fresh real records into the improvement pool (iteration 4).

Deterministic, stdlib-only, idempotent. Sources:

- git log of a repository (subject + body)
- inbox files dropped by the owner under ``data/pool/inbox/`` (one raw
  record per line, or blank-line-separated blocks for stack traces)

Every record leaves :func:`collect_all` as ``{"id", "source", "raw"}``.
Content is kept as-is (owner decision 2026-10-05: no privacy scrubbing;
see data/real/PROVENANCE.md). Length/ASCII guards are still applied,
exact duplicates removed by normalized-text sha256, near duplicates
(5-gram Jaccard >= 0.8) collapsed, and any text whose hash appears in
the frozen sets (wild, heldout, synthetic train/dev) rejected outright —
the pool must never see benchmark data.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from pathlib import Path

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TRAILER_RE = re.compile(r"^(Signed-off-by|Co-authored-by|Reviewed-by|Acked-by):", re.I)
NEARDUP_JACCARD = 0.8
# Near-dup comparison is windowed to the most recent NEARDUP_WINDOW kept
# gram-sets — O(n·W) instead of O(n²), fine at any inbox size. Exact
# duplicates stay globally deduped by normalized-text hash.
NEARDUP_WINDOW = 2000


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]


def _grams(text: str, n: int = 5) -> set[str]:
    t = normalize(text).replace(" ", "")
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def near_dup(text: str, seen_grams) -> bool:
    """True if text is a near-duplicate of any recent kept gram-set."""
    g = _grams(text)
    if not g:
        return True
    len_g = len(g)
    min_len = len_g * NEARDUP_JACCARD
    max_len = len_g / NEARDUP_JACCARD
    for s in seen_grams:
        len_s = len(s)
        if len_s < min_len or len_s > max_len:
            continue
        inter = len(g & s)
        if inter / (len_g + len_s - inter) >= NEARDUP_JACCARD:
            return True
    return False


def scrub(text: str) -> str:
    """No-op since the owner decision of 2026-10-05: dataset-provided
    content is collected as-is (the earlier email/trailer scrubbing is
    superseded — see data/real/PROVENANCE.md and docs/plans/MEMORY.md).
    Kept as an identity seam so callers have one place to change if the
    policy flips back."""
    return text.strip()


def _keep(text: str) -> bool:
    if len(text) < 10:
        return False
    printable = sum(ch.isprintable() or ch.isspace() for ch in text)
    return printable / len(text) >= 0.9


def collect_inbox(inbox_dir: Path) -> list[dict]:
    """Owner-dropped exports. ``.log`` and ``.jsonl`` files are
    line-per-record (one platform log line / one exported JSON row each);
    ``.txt`` files use blank-line-separated blocks (multi-line records
    such as stack traces). The ``*.jsonl`` seam keeps the pipeline's
    ``live_stream.jsonl`` exports collectable."""
    records: list[dict] = []
    if not inbox_dir.is_dir():
        return records
    for path in (sorted(inbox_dir.glob("*.log"))
                 + sorted(inbox_dir.glob("*.jsonl"))
                 + sorted(inbox_dir.glob("*.txt"))):
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix in (".log", ".jsonl"):
            lines = [line for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl":
                # Pool-seam exports (e.g. pipeline's live_stream.jsonl)
                # carry one JSON row per line, {"raw": ...}; a row that
                # does not parse falls back to its raw line so no
                # exported record is silently dropped.
                parsed = []
                for line in lines:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        parsed.append(line)
                        continue
                    parsed.append(row.get("raw") or row.get("text") or line)
                lines = [ln for ln in parsed if isinstance(ln, str) and ln.strip()]
            blocks = lines
        else:
            blocks = []
            buf: list[str] = []
            for line in text.splitlines():
                if line.strip():
                    buf.append(line)
                elif buf:
                    blocks.append("\n".join(buf))
                    buf = []
            if buf:
                blocks.append("\n".join(buf))
        for block in blocks:
            records.append({"source": f"inbox:{path.name}", "raw": block})
    return records


def collect_git_log(repo: Path, limit: int = 200) -> list[dict]:
    """Subject+body of the repo's own commit history (no author fields).

    A missing git binary, an interrupted call, or any other subprocess
    failure collects nothing instead of aborting the cycle: the inbox
    seam still feeds the pool."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "log", f"-{limit}", "--format=%B%x00"],
            cwd=repo, capture_output=True, text=True, check=False, timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    records = []
    for msg in out.split("\0"):
        msg = msg.strip()
        if msg:
            records.append({"source": "git-log", "raw": msg})
    return records


# Every frozen jsonl the pool must never re-collect, as (base, glob)
# pairs walked once per collect. This replaces the earlier six-file
# hardcoded list so a NEW frozen slice (e.g. a nested data/wild/realworld
# dump or a new data/heldout panel) is banned without touching this
# module. data/real train splits are deliberately absent: they are the
# training pools, not frozen audit sets. data/synthetic train/dev are
# kept (module docstring guarantee: the pool never sees benchmark data).
BANNED_GLOBS = (
    ("data/wild", "**/*.jsonl"),
    ("data/heldout", "**/*.jsonl"),
    ("data/synthetic", "**/*.jsonl"),
    ("data/real", "*/dev.jsonl"),
    ("data/real", "*/*/dev.jsonl"),
    ("data/pool/promoted", "*.jsonl"),
)


def _banned_hashes_with_errors(root: Path) -> tuple[set[str], int]:
    """Frozen-text hashes plus the count of skipped unreadable lines.

    One malformed line — a JSON decode failure, a valid-JSON non-object
    row, or a row whose text field is not a string — in ANY banned jsonl
    is skipped and counted, never fatal to the collect cycle (review
    2026-10-10, N9); the count lands in the collect stats under
    ``banned_read_errors``. Files with undecodable bytes are read with
    ``errors="replace"`` (same seam collect_inbox uses) so a single bad
    file cannot kill the cycle either.

    The hash chain covers every text key the frozen files use: "raw",
    "text", and "request" — the spanfill reserve panel
    (data/heldout/spanfill-reserve.jsonl) is keyed by "request" and its
    held-out texts must be banned from the pool (review 2026-10-10, C5).
    """
    banned: set[str] = set()
    errors = 0
    for base, pattern in BANNED_GLOBS:
        base_dir = root / base
        if not base_dir.is_dir():
            continue
        for p in sorted(base_dir.glob(pattern)):
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8",
                                    errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    errors += 1
                    continue
                if not isinstance(row, dict):
                    errors += 1
                    continue
                text = (row.get("raw") or row.get("text")
                        or row.get("request") or "")
                if not isinstance(text, str):
                    errors += 1
                    continue
                if text:
                    banned.add(text_hash(text))
                    # Line-level fragments too: a single inbox line that is a
                    # line of a frozen record leaks benchmark data just as
                    # much as the whole record does.
                    for ln in text.splitlines():
                        if ln.strip():
                            banned.add(text_hash(ln))
    return banned, errors


def _banned_hashes(root: Path) -> set[str]:
    """The frozen-text hash set; read errors are counted by collect_all."""
    return _banned_hashes_with_errors(root)[0]


def collect_all(root: Path, git_limit: int = 200) -> tuple[list[dict], dict]:
    """Return (records, stats). Records are scrubbed, guarded, deduped."""
    raw = collect_inbox(root / "data" / "pool" / "inbox")
    raw += collect_git_log(root, limit=git_limit)

    banned, banned_read_errors = _banned_hashes_with_errors(root)
    records: list[dict] = []
    stats = {"scanned": len(raw), "scrubbed_out": 0, "frozen_collision": 0,
             "duplicate": 0, "kept": 0,
             # Banned-file lines skipped as unreadable (N9, 2026-10-10):
             # honest count of what the frozen-set read could not hash.
             "banned_read_errors": banned_read_errors}
    seen: set[str] = set()
    grams: deque = deque(maxlen=NEARDUP_WINDOW)
    for rec in raw:
        text = scrub(rec["raw"])
        if not _keep(text):
            stats["scrubbed_out"] += 1
            continue
        h = text_hash(text)
        if h in banned:
            stats["frozen_collision"] += 1
            continue
        if h in seen or near_dup(text, grams):
            stats["duplicate"] += 1
            continue
        seen.add(h)
        grams.append(_grams(text))
        records.append({"id": h, "source": rec["source"], "raw": text})
        stats["kept"] += 1
    return records, stats
