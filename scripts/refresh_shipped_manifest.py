#!/usr/bin/env python3
"""Write the shipped-artifact manifest.

`models/<name>.sha256` sidecars prove a file matches itself; they do not stop a
silent swap, because whoever swaps the file can regenerate its sidecar. During
one session two concurrent agents and the parent each replaced
`models/tool-match-int8.onnx` and updated its sidecar, and every gate stayed
green while the shipped guard changed underneath the other two.

This manifest is the second key: the committed sha256 of every shipped
artifact. `tests/test_gates.py::test_shipped_artifact_manifest_is_current`
fails when a file and the manifest disagree, so changing a shipped artifact
now requires a manifest update in the same commit, which a reviewer can see.

Run this after a deliberate artifact change.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MANIFEST = MODELS / "SHIPPED.json"
# Sidecars and the manifest itself are not artifacts.
SKIP_SUFFIXES = (".sha256", ".trainmeta.json")


def main() -> int:
    entries: dict[str, dict] = {}
    for path in sorted(MODELS.iterdir()):
        if path.name in {"SHIPPED.json", ".gitkeep"}:
            continue
        if path.name.endswith(SKIP_SUFFIXES):
            continue
        if not path.is_file():
            continue
        entries[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    MANIFEST.write_text(json.dumps({"artifacts": entries}, indent=2) + "\n",
                        encoding="utf-8")
    for name, meta in entries.items():
        print(f"{name}: {meta['bytes']} B {meta['sha256'][:16]}")
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(entries)} artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
