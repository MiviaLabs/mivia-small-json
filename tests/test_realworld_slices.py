"""Drift gate for the real-world routing eval slices (data/wild/realworld).

Eval-only frozen slices: row shape, label vocabulary, minimum sizes,
PII masking, and sha256 pins against the provenance doc. Deterministic,
offline, CPU-only (mechanical gates convention).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The real-world slice corpora are not redistributed (third-party
# sources; see docs/DATA.md). Without them these floors cannot be
# measured — skip, never fail; the private tree runs them for real.
if not (ROOT / "data" / "wild").exists():
    pytest.skip("real-world slice corpora absent (public tree)",
                allow_module_level=True)

SLICES = ROOT / "data" / "wild" / "realworld"

# slice -> (minimum rows, allowed expected labels)
SPEC = {
    "issue-realworld-v2": (200, {"issue"}),
    "post-commit": (150, {"commit", "no_tool_call"}),
    "chat-informal": (150, {"no_tool_call"}),
    "slack-snippets": (150, {"no_tool_call", "log_event"}),
    "fa-ar-ood-fa": (150, {"no_tool_call"}),
    "fa-ar-ood-ar": (150, {"no_tool_call"}),
    "long-log-dumps": (50, {"log_event", "diagnostic"}),
}

# sha256 pins (first 16 hex) — must match data/wild/REALWORLD_SLICES_PROVENANCE.md
HASHES = {
    "issue-realworld-v2": "47c37b26c8eca30d",
    "post-commit": "8cd5c528e69586e4",
    "chat-informal": "1bb6865433873f60",
    "slack-snippets": "bde3d47284738b21",
    "fa-ar-ood-fa": "c1428b83f705966e",
    "fa-ar-ood-ar": "ddd011b00ddab193",
    "long-log-dumps": "26d7d4eae73be83d",
}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
PHONE = re.compile(r"(?<!<)(?<!\w)(\+?\d[\d\s().-]{9,}\d)(?!\w)")


def _rows(name: str) -> list[dict]:
    path = SLICES / f"{name}.jsonl"
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("name", sorted(SPEC))
def test_size_and_labels(name: str) -> None:
    min_rows, allowed = SPEC[name]
    rows = _rows(name)
    assert len(rows) >= min_rows, f"{name}: {len(rows)} < {min_rows}"
    for row in rows:
        assert set(row) == {"text", "expected", "source", "ambiguous", "note"}, row
        assert isinstance(row["ambiguous"], bool)
        assert row["expected"] in allowed, (name, row["expected"])
        assert isinstance(row["text"], str) and row["text"].strip()
        assert isinstance(row["source"], str) and row["source"]


def test_long_log_dumps_are_long() -> None:
    for row in _rows("long-log-dumps"):
        assert 5000 <= len(row["text"]) <= 12000, len(row["text"])


def test_post_commit_keeps_both_labels() -> None:
    rows = _rows("post-commit")
    commits = [r for r in rows if r["expected"] == "commit"]
    assert len(commits) >= 20, "commit-content rows must not silently vanish"
    assert all(r["ambiguous"] for r in rows if r["expected"] == "no_tool_call")


@pytest.mark.parametrize("name", sorted(SPEC))
def test_pii_masked(name: str) -> None:
    for row in _rows(name):
        assert not EMAIL.search(row["text"]), row["source"]
        masked_phone = "<phone>" in row["text"]
        if not masked_phone and name == "chat-informal":
            assert not PHONE.search(row["text"]), row["source"]


@pytest.mark.parametrize("name", sorted(SPEC))
def test_frozen_sha256(name: str) -> None:
    data = (SLICES / f"{name}.jsonl").read_bytes()
    digest = hashlib.sha256(data).hexdigest()[:16]
    assert digest == HASHES[name], (
        f"{name} changed (sha {digest}); this is a frozen eval slice. "
        "If a regeneration was intended, update HASHES here AND the "
        "provenance doc AND re-run scripts/bench_realworld_slices.py."
    )
