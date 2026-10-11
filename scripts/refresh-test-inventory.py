#!/usr/bin/env python3
"""Regenerate tests/test-inventory.txt from the suite's collected ids.

Run from the repository root with the project interpreter:

    .venv/bin/python scripts/refresh-test-inventory.py

The gate test (test_the_suite_matches_its_committed_inventory) compares
the sha256 of the sorted collected node ids against that file, so a
test added, deleted, renamed or re-parameterized fails the gate until
the inventory is regenerated here — deliberately.

addopts -q collapses plain ``pytest --collect-only -q`` output to
per-file counts, so collection runs with addopts cleared and the ids
are taken from a recorder plugin, not parsed out of prose.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "tests" / "test-inventory.txt"


class _IdRecorder:
    """Collect the node ids exactly as the live session would see them."""

    def __init__(self) -> None:
        self.ids: list[str] = []

    def pytest_collection_modifyitems(self, session, config, items) -> None:
        self.ids = sorted(item.nodeid for item in items)


def main() -> int:
    recorder = _IdRecorder()
    code = pytest.main(
        ["--collect-only", "-q", "-o", "addopts=", str(ROOT / "tests")],
        plugins=[recorder],
    )
    if code != 0 or not recorder.ids:
        print(f"collection failed (exit {code})", file=sys.stderr)
        return 1
    before = (hashlib.sha256(INVENTORY.read_bytes()).hexdigest()
              if INVENTORY.exists() else "no inventory yet")
    INVENTORY.write_bytes(("\n".join(recorder.ids) + "\n").encode())
    print(f"{len(recorder.ids)} collected test ids -> "
          f"{INVENTORY.relative_to(ROOT)}")
    print(f"sha256 {before[:12]}... -> "
          f"{hashlib.sha256(INVENTORY.read_bytes()).hexdigest()[:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
