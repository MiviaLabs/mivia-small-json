#!/usr/bin/env python3
"""Promote reviewed hard rows into a gitignored labeled pool.

Reads ``data/pool/hard.jsonl``. A row a human marked with
``{"review": "accept"}`` is merged into
``data/pool/promoted/<schema>.jsonl``, which ``label.dev_rows`` includes
in the labeled development set. This closes the otherwise one-way
weak-label loop.

The merge is additive and deduped by the row text (keep-first: an
earlier promoted row stands), so promoting a new batch never silently
discards the batches before it. Malformed lines and accepted rows that
cannot be promoted (missing schema or raw) are counted and reported in
the summary — they are data loss in disguise, not silence.

The promoted tree is gitignored: reviewed rows are local training material,
not committed data.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARD = ROOT / "data" / "pool" / "hard.jsonl"
PROMOTED = ROOT / "data" / "pool" / "promoted"


def _text_key(raw: str) -> str:
    """Dedupe key for a row text (sha256, so huge texts do not pile up
    as dict keys)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def promote(detailed: bool = False) -> dict:
    """Merge accepted hard rows into the promoted pool; return the counts.

    The default shape is the historical one, ``{schema: added}``. With
    ``detailed=True`` the result names every disposition separately —
    per schema ``added`` / ``already_promoted`` / ``total``, plus
    ``read_errors`` (malformed hard.jsonl lines),
    ``promoted_read_errors`` (malformed lines already sitting in the
    promoted files) and ``accepted_unpromotable`` (rows marked accept
    whose schema or raw is missing). Nothing accepted is dropped without
    a count; ``main()`` reports the detailed view.
    """
    PROMOTED.mkdir(parents=True, exist_ok=True)
    accepted: dict[str, list[dict]] = {}
    stats = {"read_errors": 0, "promoted_read_errors": 0,
             "accepted_unpromotable": 0, "review_pending": 0}
    for line in HARD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            stats["read_errors"] += 1
            continue
        if not isinstance(row, dict):
            stats["read_errors"] += 1
            continue
        if row.get("review") != "accept":
            stats["review_pending"] += 1
            continue
        schema = row.get("schema")
        raw = row.get("raw")
        if not schema or not raw:
            stats["accepted_unpromotable"] += 1
            continue
        accepted.setdefault(str(schema), []).append(
            {"raw": raw, "schema": str(schema)})
    schemas: dict[str, dict] = {}
    for schema, rows in accepted.items():
        path = PROMOTED / f"{schema}.jsonl"
        seen: set[str] = set()
        out: list[dict] = []
        existing = 0
        duplicates_already = 0
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    stats["promoted_read_errors"] += 1
                    continue
                if not isinstance(row, dict) or not row.get("raw"):
                    stats["promoted_read_errors"] += 1
                    continue
                key = _text_key(row["raw"])
                if key in seen:
                    duplicates_already += 1
                    continue
                seen.add(key)
                out.append(row)
                existing += 1
        added = 0
        duplicates_in_batch = 0
        for row in rows:
            key = _text_key(row["raw"])
            if key in seen:
                duplicates_in_batch += 1
                continue
            seen.add(key)
            out.append(row)
            added += 1
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
            encoding="utf-8")
        schemas[schema] = {"added": added,
                           "already_promoted": existing,
                           "duplicates_in_file": duplicates_already,
                           "total": len(out),
                           "duplicates_in_batch": duplicates_in_batch}
    if detailed:
        return {"schemas": schemas, **stats}
    # Historical shape: schema -> rows added this run. The detailed view
    # above carries the merge and skip counts an operator needs.
    return {schema: c["added"] for schema, c in schemas.items()}


def main() -> int:
    if not HARD.is_file():
        print(f"no hard pool at {HARD}", file=sys.stderr)
        return 1
    result = promote(detailed=True)
    schemas = result["schemas"]
    for schema, c in schemas.items():
        print(f"{schema}: {c['added']} new promoted, "
              f"{c['already_promoted']} kept from earlier promotions, "
              f"{c['duplicates_in_batch']} batch duplicates skipped, "
              f"{c['total']} total in {schema}.jsonl")
    if result["read_errors"]:
        print(f"read_errors: {result['read_errors']} malformed line(s) in "
              f"{HARD.name} skipped (counted, not fatal)")
    if result["promoted_read_errors"]:
        print(f"promoted_read_errors: {result['promoted_read_errors']} "
              "malformed line(s) inside the promoted files skipped")
    if result["review_pending"]:
        print(f"review_pending: {result['review_pending']} row(s) not marked "
              "review=accept (left in the hard pool)")
    if result["accepted_unpromotable"]:
        print(f"accepted_unpromotable: {result['accepted_unpromotable']} "
              "row(s) marked review=accept but missing schema or raw — "
              "NOT promoted")
    if not schemas:
        if result["accepted_unpromotable"] or result["read_errors"]:
            print("nothing promoted: accepted rows existed but none could "
                  "be promoted (see the counts above)")
            return 1
        print("no rows marked review=accept; nothing promoted")
        return 0
    try:
        where = PROMOTED.relative_to(ROOT)
    except ValueError:  # a relocated pool (tests, scratch roots): show it as-is
        where = PROMOTED
    print(f"promoted -> {where} (gitignored, merged with earlier promotions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
