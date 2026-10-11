#!/usr/bin/env python3
"""Pin every committed training corpus by sha256 and row count.

A silent edit to a training file is invisible until it changes a metric,
often much later and often on another machine. This mirrors the
test-inventory gate for data: it hashes each committed JSONL corpus into
``data/dataset-manifest.json`` and, in verify mode, fails when a file is
changed, deleted, or added relative to that manifest.

Covered (committed, small, versioned):

    data/synthetic/*.jsonl
    data/real/*/*.jsonl
    data/heldout/*.jsonl
    data/wild/*.jsonl
    data/pool/*.jsonl

``data/incoming/`` is deliberately excluded: it holds bulky raw upstream
dumps that are large and, for some sources, gitignored.

Regenerate the manifest (one documented command, from the repo root):

    python scripts/check_dataset_manifest.py --write

Verify (the CI gate; exits non-zero and names every drifted path):

    python scripts/check_dataset_manifest.py

Stdlib only, deterministic ordering, no network, no git dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = Path("data") / "dataset-manifest.json"

# Globs are relative to the root and cover only committed corpora.
CORPUS_GLOBS = (
    "data/synthetic/*.jsonl",
    "data/real/*/*.jsonl",
    "data/heldout/*.jsonl",
    "data/wild/*.jsonl",
    "data/pool/*.jsonl",
)


def corpus_files(root: Path) -> list[Path]:
    """Every committed corpus file under ``root``, in deterministic order."""
    found: set[Path] = set()
    for pattern in CORPUS_GLOBS:
        found.update(root.glob(pattern))
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    """JSONL row count: non-empty lines, so a trailing newline is not a row."""
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def build_manifest(root: Path) -> dict[str, dict[str, object]]:
    """Map relative posix path -> {"sha256", "rows"} for every corpus file."""
    manifest: dict[str, dict[str, object]] = {}
    for path in corpus_files(root):
        rel = path.relative_to(root).as_posix()
        manifest[rel] = {"sha256": _sha256(path), "rows": _count_rows(path)}
    return {rel: manifest[rel] for rel in sorted(manifest)}


def write_manifest(root: Path) -> Path:
    """Regenerate the committed manifest and return its path."""
    manifest = build_manifest(root)
    target = root / MANIFEST_REL
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def verify(root: Path) -> list[str]:
    """Return one message per drifted path; an empty list means clean."""
    target = root / MANIFEST_REL
    if not target.exists():
        return [f"missing manifest: {MANIFEST_REL.as_posix()} (run --write)"]
    committed = json.loads(target.read_text(encoding="utf-8"))
    current = build_manifest(root)
    drift: list[str] = []
    for rel in sorted(set(committed) | set(current)):
        if rel not in committed:
            drift.append(f"added: {rel}")
        elif rel not in current:
            drift.append(f"missing: {rel}")
        elif committed[rel] != current[rel]:
            drift.append(f"changed: {rel}")
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the manifest from the current corpora",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="repository root to operate on (default: this script's repo)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else ROOT

    if args.write:
        target = write_manifest(root)
        count = len(build_manifest(root))
        print(f"wrote {count} corpus hashes -> {target.relative_to(root).as_posix()}")
        return 0

    drift = verify(root)
    if drift:
        for message in drift:
            print(message, file=sys.stderr)
        print(
            f"dataset manifest drift: {len(drift)} path(s); "
            "regenerate with scripts/check_dataset_manifest.py --write",
            file=sys.stderr,
        )
        return 1
    print(f"dataset manifest up to date ({len(build_manifest(root))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
