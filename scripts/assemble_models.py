#!/usr/bin/env python3
"""Assemble split model artifacts from models/parts/.

GitHub rejects files > 100 MB, so the few oversized shipped artifacts
(currently models/router-encoder-int8.onnx and models/
intent-rank-mul-int8.onnx) are stored as byte-split parts plus a
manifest recording the assembled file's sha256. This script joins them
and verifies integrity. Idempotent: an assembled file that already
matches its manifest sha256 is left untouched, so running it on a
machine that has the real file (the training box) is a no-op.

Run after checkout, BEFORE the tests or the image build (CI and the
Dockerfile call it). Never raises for missing parts — exits 2 with the
missing list instead.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "models" / "parts"


def main() -> int:
    if not PARTS.is_dir():
        print("no models/parts/ — nothing to assemble")
        return 0
    missing: list[str] = []
    for manifest_path in sorted(PARTS.glob("*.manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        name = manifest["file"]
        target = ROOT / "models" / name
        digest = hashlib.sha256(target.read_bytes()).hexdigest() \
            if target.is_file() else None
        if digest == manifest["sha256"]:
            print(f"{name}: already assembled, sha256 ok")
            continue
        part_paths = [PARTS / p for p in manifest["parts"]]
        absent = [str(p) for p in part_paths if not p.is_file()]
        if absent:
            missing.extend(absent)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            for part in part_paths:
                out.write(part.read_bytes())
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != manifest["sha256"]:
            print(f"{name}: ASSEMBLED SHA MISMATCH ({got[:12]} != "
                  f"{manifest['sha256'][:12]})", file=sys.stderr)
            return 2
        print(f"{name}: assembled from {len(part_paths)} parts, "
              f"sha256 ok")
    if missing:
        print("missing parts:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
