"""Pre-registered bars for the six+1 curated realworld routing slices.

Owner process decision (2026-10-08, after the long-log-dumps regression
shipped silently: 100.0 strict -> 69.6 on the promoted router): every
future retrain or decision-layer change must clear these floors BEFORE
promotion. The floors are the measured post-fix values of the structural
decision layer (pipeline log-shape rule, 2026-10-08); they are one-sided
minimums — improvements never trip them, any regression does.

Eval-only by construction: data/wild/ is measurement data and never
enters training. Rows are scored exactly like
scripts/bench_realworld_slices.py (acceptable-set scoring, ambiguous
rows excluded from strict but counted separately).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mivia_small_json import bootstrap
from mivia_small_json.diagnostic import register_diagnostic_schema
from mivia_small_json.diff_comment import register_diff_and_review_schemas
from mivia_small_json.encoder import load_encoder
from mivia_small_json.pipeline import classify, load_active_router
from mivia_small_json.tool_call import (
    register_no_tool_call_schema,
    register_tool_call_schema,
)

ROOT = Path(__file__).resolve().parents[1]

# The real-world slice corpora are not redistributed (third-party
# sources; see docs/DATA.md). Without them these floors cannot be
# measured — skip, never fail; the private tree runs them for real.
if not (ROOT / "data" / "wild").exists():
    pytest.skip("real-world slice corpora absent (public tree)",
                allow_module_level=True)

SLICES = ROOT / "data" / "wild" / "realworld"

# slice -> (strict floor %, n, strict_n). n/strict_n pin the slice files
# themselves: a re-curation that changes the row set must fail here and
# force a conscious re-registration of the floors, never a silent drift.
BARS = {
    "chat-informal": (42.9, 170, 170),
    "fa-ar-ood-ar": (48.7, 160, 160),
    # 94/160 = 58.75 exact; the floor sits below the value by design
    "fa-ar-ood-fa": (58.7, 160, 160),
    "issue-realworld-v2": (85.7, 260, 260),
    "long-log-dumps": (100.0, 55, 46),
    "post-commit": (0.0, 200, 28),
    "slack-snippets": (3.5, 170, 170),
}

ACCEPT = {
    "issue-realworld-v2": {"issue"},
    "post-commit": {"commit", "no_tool_call", "text"},
    "chat-informal": {"no_tool_call", "text"},
    "slack-snippets": {"no_tool_call", "text", "log_event"},
    "fa-ar-ood-fa": {"no_tool_call", "text"},
    "fa-ar-ood-ar": {"no_tool_call", "text"},
    "long-log-dumps": {"log_event", "diagnostic"},
}

pytestmark = pytest.mark.skipif(
    load_encoder() is None,
    reason="shipped ONNX encoder absent: the floors pin the encoder chain "
           "(a skops fallback would fail them spuriously)"
)


def _strict_rate(name: str) -> float:
    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()
    router = load_active_router()
    rows = [json.loads(line) for line in
            (SLICES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ok = strict_n = 0
    for row in rows:
        obj, _meta = classify(row["text"], router)
        hit = obj.get("type") in ACCEPT[name]
        if not row["ambiguous"]:
            ok += hit
            strict_n += 1
    assert (len(rows), strict_n) == (BARS[name][1], BARS[name][2]), (
        f"{name}: slice shape drifted (n={len(rows)}, strict_n={strict_n}); "
        "re-register the bars consciously")
    return 100.0 * ok / max(1, strict_n)


@pytest.mark.parametrize("name", sorted(BARS))
def test_slice_meets_its_pre_registered_floor(name):
    floor, _, _ = BARS[name]
    rate = _strict_rate(name)
    assert rate >= floor, (
        f"{name}: strict {rate:.1f}% below the pre-registered floor "
        f"{floor}% — a realworld-slices regression must not ship; "
        "re-run scripts/bench_realworld_slices.py and re-register the bar "
        "consciously if the trade is intended")
