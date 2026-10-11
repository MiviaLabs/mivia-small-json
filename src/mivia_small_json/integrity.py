# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared sha256 sidecar verification for model artifacts.

Every artifact loader (encoder, tool_match, pipeline, extract_learned)
verifies its artifact against a sha256 sidecar before loading. One helper
here so the streaming read and the hex validation cannot drift apart.
Sidecar conventions in use: ``<name>.sha256`` next to the artifact
(encoder, tool_match, extract_learned) and ``<stem>.sha256`` (the skops
router, as written by model.main()).
"""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

_CHUNK = 65536  # streamed; artifacts run to ~144 MB under the 2 GB cap
_HEX = frozenset("0123456789abcdef")


def sha256_file(path: Path) -> str:
    """Streaming sha256 hexdigest (never holds a whole artifact in memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def expected_digest(sidecar: Path) -> tuple[str | None, str]:
    """(expected hex digest, problem) read from a sha256 sidecar.

    ``problem`` is "" when a valid 64-hex digest was read, ``"missing"``
    when the sidecar file does not exist, ``"malformed"`` otherwise.
    """
    if not sidecar.is_file():
        return None, "missing"
    parts = sidecar.read_text(encoding="utf-8").split()
    if not parts:
        # An empty sidecar is malformed, not an IndexError (review §11).
        return None, "malformed"
    expected = parts[0].lower()
    if len(expected) != 64 or not set(expected) <= _HEX:
        return None, "malformed"
    return expected, ""


def digests_match(actual: str, expected: str) -> bool:
    """Constant-time digest comparison."""
    return hmac.compare_digest(actual, expected)


def verified(path: Path, sidecar: Path | None = None) -> bool:
    """True when ``path`` exists and matches its sha256 sidecar.

    Default sidecar convention: ``<name>.sha256`` next to the artifact.
    """
    if not path.is_file():
        return False
    if sidecar is None:
        sidecar = path.parent / (path.name + ".sha256")
    expected, problem = expected_digest(sidecar)
    if problem:
        return False
    return digests_match(sha256_file(path), expected)
