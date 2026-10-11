#!/usr/bin/env python3
"""Render the docs/USAGE.md §7.2 severity-taxonomy section from the module.

The crosswalk table, the per-taxonomy source notes and the enum-alias
packs in docs/USAGE.md §7.2 are GENERATED from
``src/mivia_small_json/severity_taxonomies.py`` so they cannot drift
from the tables the tests pin. This script is the only writer of the
region between the BEGIN/END markers; hand edits there are overwritten
by the next run (and ``--check`` flags them).

Usage:
    python scripts/gen_severity_alias_packs.py           # rewrite in place
    python scripts/gen_severity_alias_packs.py --check   # exit 1 on drift
    python scripts/gen_severity_alias_packs.py --print   # stdout only

Deterministic output: no wall-clock, no environment, no network.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
USAGE_PATH = REPO_ROOT / "docs" / "USAGE.md"

sys.path.insert(0, str(REPO_ROOT / "src"))

from mivia_small_json.severity_taxonomies import (  # noqa: E402
    USAGE_BEGIN_MARK,
    USAGE_END_MARK,
    render_usage_markdown,
)


def replace_region(text: str, block: str) -> str:
    """Swap the marker region for ``block``, keeping the marker pair."""
    begin = text.find(USAGE_BEGIN_MARK)
    end = text.find(USAGE_END_MARK)
    if begin < 0 or end < 0 or end < begin:
        raise SystemExit(
            f"{USAGE_PATH}: markers not found — add the pair "
            f"{USAGE_BEGIN_MARK!r} ... {USAGE_END_MARK!r} where the "
            "section belongs, then re-run")
    tail = text.find(USAGE_BEGIN_MARK, begin + 1)
    if tail >= 0:
        raise SystemExit(f"{USAGE_PATH}: duplicated BEGIN marker")
    return (text[:begin] + USAGE_BEGIN_MARK + "\n" + block + "\n"
            + USAGE_END_MARK + text[end + len(USAGE_END_MARK):])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed section matches the "
                         "module; exit 1 on drift, write nothing")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the rendered block to stdout")
    args = ap.parse_args()

    block = render_usage_markdown()
    if args.show:
        sys.stdout.write(block + "\n")
        return 0

    text = USAGE_PATH.read_text(encoding="utf-8")
    updated = replace_region(text, block)
    if updated == text:
        print(f"OK {USAGE_PATH}: section already matches the module")
        return 0
    if args.check:
        print(f"DRIFT {USAGE_PATH}: committed section differs from the "
              "module — run scripts/gen_severity_alias_packs.py and "
              "commit the output", file=sys.stderr)
        return 1
    USAGE_PATH.write_text(updated, encoding="utf-8")
    print(f"WROTE {USAGE_PATH}: severity-taxonomies section "
          "regenerated from the module")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
