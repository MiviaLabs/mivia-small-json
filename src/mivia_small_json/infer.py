# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""CLI: raw text lines on stdin -> JSONL records on stdout.

Usage::

    python -m mivia_small_json.infer [--model PATH]

Blank lines are skipped. Every non-blank line that classifies cleanly
emits exactly one JSONL record ``{"input", "obj", "meta"}``; classify()'s
never-raise contract means a hostile line still produces a record. A line
that somehow raises is reported to stderr and skipped, and never changes
the exit code (0).
"""

from __future__ import annotations

import argparse
import json
import sys

from . import bootstrap
from .pipeline import classify, load_active_router, load_router


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mivia_small_json.infer",
        description="Classify raw text lines into strict JSON records (JSONL on stdout).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="classify with this router artifact instead of the shipped "
             "active router (the ONNX encoder, skops fallback)",
    )
    args = parser.parse_args(argv)
    # Boot the full shipped schema world before routing, exactly like
    # load_active_router does: without it a diagnostic/diff/... prediction
    # from the loaded router falls back to a wrong built-in with
    # routed_by == "rules".
    bootstrap.register_all_schemas()
    # Default to the ACTIVE router — the same bytes every served surface
    # (serve.py / api.py / mcp.py) answers from — so the CLI can never
    # disagree with the server on the same input (survfix 2026-10-10: the
    # direct skops load routed 'سلام خوبی چطوری' review_comment while every
    # served surface routed text). An explicit --model still overrides.
    router = load_router(args.model) if args.model else load_active_router()
    # Read bytes and decode leniently: a hostile byte on stdin must not
    # kill the batch mid-stream (the never-fail contract covers each
    # classified line; this covers the read itself).
    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.strip():
            continue
        try:
            obj, meta = classify(line, router)
            sys.stdout.write(
                json.dumps({"input": line, "obj": obj, "meta": meta}) + "\n"
            )
        except Exception as exc:  # belt and braces: never fail the batch
            print(
                f"[mivia-small-json] line skipped ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
        finally:
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
