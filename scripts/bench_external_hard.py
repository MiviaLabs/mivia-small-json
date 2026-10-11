#!/usr/bin/env python3
"""Out-of-distribution routing eval on harder external sets.

Each fetched external set is text that should route to one known schema:

- NLBSE issue reports        -> ``issue``
- CleanCodeReview comments   -> ``review_comment``
- BFCL function calls        -> ``tool_call``
- Loghub-2.0 log lines       -> ``log_event``

This is a harder, OOD check than the synthetic held-out set: the text comes
from real projects with different vocabulary. It reports, per set, the
fraction routed to the expected schema and the fraction that is valid JSON.

A missing set is reported as missing, not counted. Writes
reports/benchmark-external-ood.md. Deterministic, offline, CPU-only.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json import bootstrap  # noqa: E402
from mivia_small_json.diagnostic import register_diagnostic_schema  # noqa: E402
from mivia_small_json.diff_comment import (  # noqa: E402
    register_diff_and_review_schemas,
)
from mivia_small_json.pipeline import classify, load_active_router  # noqa: E402
from mivia_small_json.tool_call import (  # noqa: E402
    register_no_tool_call_schema,
    register_tool_call_schema,
)

EXTERNAL = ROOT / "data" / "incoming" / "external"
REPORT = ROOT / "reports" / "benchmark-external-ood.md"
CAP = 1000  # rows scored per set
# BFCL live-irrelevance: a tool catalogue with a request that needs NO tool
# call. Routing these to tool_call is a false positive. Drawn from a different
# split than the 240 rows the guard trains its abstention signal on, so the
# whole set is unseen.
IRRELEVANCE = ROOT / "data" / "incoming" / "external" / "bfcl" / "irrelevance_eval.jsonl"


def _sets() -> list[tuple[str, Path, str]]:
    out = [
        ("nlbse", EXTERNAL / "nlbse" / "issues.jsonl", "issue"),
        ("cleancodereview", EXTERNAL / "cleancodereview" / "comments.jsonl",
         "review_comment"),
        ("bfcl", EXTERNAL / "bfcl" / "calls.jsonl", "tool_call"),
    ]
    loghub2 = EXTERNAL / "loghub2"
    if loghub2.is_dir():
        for path in sorted(loghub2.glob("*.jsonl")):
            out.append((f"loghub2:{path.stem}", path, "log_event"))
    return out


def _rows(path: Path) -> list[str]:
    if not path.is_file():
        return []
    texts = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        text = (rec.get("text") or rec.get("raw") or "").strip()
        if text:
            texts.append(text)
        if len(texts) >= CAP:
            break
    return texts


def main() -> int:
    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()
    router = load_active_router()

    results = []
    for name, path, expected in _sets():
        texts = _rows(path)
        if not texts:
            results.append((name, expected, None, None, 0))
            continue
        routed = valid = 0
        for text in texts:
            obj, meta = classify(text, router)
            valid += bool(meta["valid"])
            routed += obj.get("type") == expected
        results.append((name, expected, 100 * routed / len(texts),
                        100 * valid / len(texts), len(texts)))

    lines = [
        "# Out-of-distribution routing benchmark",
        "",
        "Real external text that should route to one known schema. Harder",
        "than the synthetic held-out set: different vocabulary, real projects.",
        "",
        f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d')} · "
        "script: `scripts/bench_external_hard.py`",
        "",
        "| Set | Expected schema | rows | routed to expected % | valid % |",
        "|---|---|---|---|---|",
    ]
    for name, expected, routed, valid, n in results:
        if routed is None:
            lines.append(f"| {name} | {expected} | 0 | missing | missing |")
        else:
            lines.append(f"| {name} | {expected} | {n} | {routed:.1f} | {valid:.1f} |")

    irr_texts = _rows(IRRELEVANCE)
    fp = sum(1 for t in irr_texts
             if classify(t, router)[0].get("type") == "tool_call")
    lines += [
        "",
        "## No-tool false positives (BFCL live-irrelevance)",
        "",
        "A tool catalogue with a request that needs no tool call. Routing it",
        "to `tool_call` is a false positive: the cross-encoder tool-match",
        "guard should send it to `no_tool_call` instead.",
        "",
        "These tools are near-misses: topically adjacent but not applicable",
        "(`find_roots` for a quadratic equation against a question about a",
        "linear one). That is a finer call than topical relevance.",
        "",
        f"- rows: {len(irr_texts)}",
        f"- routed to tool_call (false positive): {fp} "
        f"({(100 * fp / len(irr_texts)) if irr_texts else 0:.1f}%)",
        "",
    ]
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    for row in results:
        print(row)
    print(f"report written to {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
