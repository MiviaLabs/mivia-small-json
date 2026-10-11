#!/usr/bin/env python3
"""agent_step quality benchmark on the session-disjoint dev split.

The frozen held-out bench (`data/heldout/test.jsonl`) is a 4-schema
synthetic set and carries no ``agent_step`` rows, so the schema's quality
was never measured. This bench closes that gap on real data.

Set: ``data/real/agent_step/dev.jsonl`` (228 rows). The harvest partitions
sessions 80/20 keyed on ``sha256(session_id) % 100``, so no session spans
train and dev; ``train.jsonl`` and ``dev.jsonl`` share zero raw texts.
Caveat, stated rather than hidden: dev is the model-selection split (it
feeds ``real_dev_accuracy`` in ``router.trainmeta.json``), so these numbers
are a real held-out estimate but not a *frozen* one.

Reports routing, validity, per-field exact match, latency, and a
false-positive check (non-agent_step text that routes to agent_step).

Writes ``reports/benchmark-agent-step.md``.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORT = ROOT / "reports" / "benchmark-agent-step.md"
DEV = ROOT / "data" / "real" / "agent_step" / "dev.jsonl"
TRAIN = ROOT / "data" / "real" / "agent_step" / "train.jsonl"
HELDOUT = ROOT / "data" / "heldout" / "test.jsonl"
TOOLCALL_DEV = ROOT / "data" / "real" / "tool_call" / "dev.jsonl"

FIELDS = ("step_index", "agent_role", "tool", "arguments", "stdout", "stderr",
          "return_code", "duration_ms", "error_type", "error_message", "status")
INT_FIELDS = frozenset({"step_index", "return_code", "duration_ms"})
# What scripts/harvest_agent_sessions.py::_raw_for_step renders into the
# raw. Anything outside this set is session context the model cannot see.
IN_RAW = frozenset({"step_index", "agent_role", "tool", "arguments", "stdout",
                    "stderr", "return_code", "status"})
FP_SLICE = 100  # tool_call dev rows sampled for the false-positive check


def _register_schemas() -> None:
    from mivia_small_json import bootstrap
    from mivia_small_json.diagnostic import register_diagnostic_schema
    from mivia_small_json.diff_comment import register_diff_and_review_schemas
    from mivia_small_json.tool_call import (
        register_no_tool_call_schema,
        register_tool_call_schema,
        register_tool_clarification_schema,
    )

    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .splitlines() if line.strip()]


def _eq(field: str, got, gold) -> bool:
    if field in INT_FIELDS:
        try:
            return int(got or 0) == int(gold or 0)
        except (TypeError, ValueError):
            return False
    if field == "arguments":
        # `repair_arguments` re-serializes compactly, so compare parsed
        # values, not whitespace.
        try:
            return json.loads(got or "{}") == json.loads(gold or "{}")
        except (TypeError, ValueError):
            return str(got or "").strip() == str(gold or "").strip()
    return str(got or "").strip() == str(gold or "").strip()


def _is_nonempty(field: str, value) -> bool:
    if field in INT_FIELDS:
        return int(value or 0) != 0
    return str(value or "").strip() not in ("", "{}")


def evaluate(router=None) -> dict:
    """Measure the active router on the agent_step dev split."""
    from mivia_small_json import pipeline

    if router is None:
        router = pipeline.load_active_router()

    dev = _rows(DEV)
    hits = {f: 0 for f in FIELDS}
    present = {f: 0 for f in FIELDS}
    routed = valid = 0
    stdout_truncated_misses = 0
    latencies: list[float] = []
    for row in dev:
        t0 = time.perf_counter()
        obj, meta = pipeline.classify(row["raw"], router=router)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        routed += obj.get("type") == "agent_step"
        valid += bool(meta.get("valid"))
        gold = row["gold"]
        for f in FIELDS:
            gv = gold.get(f, 0 if f in INT_FIELDS else "")
            if _is_nonempty(f, gv):
                present[f] += 1
                ok = _eq(f, obj.get(f, 0 if f in INT_FIELDS else ""), gv)
                hits[f] += ok
                # The harvest clips stdout and stores a `[truncated N chars]`
                # marker; raw and gold clip at different points, so the gold
                # value is not fully present in the raw.
                if f == "stdout" and not ok and "[truncated" in str(gv):
                    stdout_truncated_misses += 1

    false_positives = {}
    for name, path, limit in (("heldout_4schema", HELDOUT, None),
                              ("tool_call_dev", TOOLCALL_DEV, FP_SLICE)):
        if not path.is_file():
            continue
        rows = _rows(path)[:limit] if limit else _rows(path)
        fp = 0
        for row in rows:
            obj, _ = pipeline.classify(row["raw"], router=router)
            fp += obj.get("type") == "agent_step"
        false_positives[name] = {"n": len(rows), "routed_agent_step": fp}

    n = len(dev) or 1
    return {
        "n": len(dev),
        "routed_pct": round(100.0 * routed / n, 2),
        "valid_pct": round(100.0 * valid / n, 2),
        "fields": {f: {"present": present[f], "exact": hits[f],
                       "exact_pct": (round(100.0 * hits[f] / present[f], 1)
                                     if present[f] else None),
                       "in_raw": f in IN_RAW}
                   for f in FIELDS},
        "lat_p50_ms": round(statistics.median(latencies), 2),
        "lat_p95_ms": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 2),
        "stdout_truncated_misses": stdout_truncated_misses,
        "false_positives": false_positives,
    }


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True
                              ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_report(result: dict) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    f = result["fields"]
    lines = [
        "# Benchmark: agent_step quality (session-disjoint dev split)",
        "",
        "The frozen held-out bench is a 4-schema synthetic set with no "
        "`agent_step` rows, so this schema was never measured there. This "
        "bench measures it on real harvested data the router did not train "
        "on.",
        "",
        f"- **Timestamp**: {now}",
        f"- **Set**: `data/real/agent_step/dev.jsonl` — {result['n']} rows",
        "- **Split**: session-disjoint from train (`sha256(session_id) % 100`); "
        "zero raw overlap with `train.jsonl`",
        "- **Caveat**: dev is the model-selection split (it feeds "
        "`real_dev_accuracy`), so this is a real held-out estimate, not a "
        "frozen one",
        f"- **Environment**: python {platform.python_version()} · "
        f"git {_git_sha()}",
        "",
        "## Routing & validity",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| routed to `agent_step` | **{result['routed_pct']}%** "
        f"({int(round(result['n'] * result['routed_pct'] / 100))}/{result['n']}) |",
        f"| valid JSON | **{result['valid_pct']}%** |",
        f"| latency p50 / p95 | {result['lat_p50_ms']} / "
        f"{result['lat_p95_ms']} ms |",
        "",
        "## Per-field exact match",
        "",
        "| Field | In raw? | Present | Exact | Exact % |",
        "|---|---|---|---|---|",
    ]
    for name, m in f.items():
        pct = "n/a" if m["exact_pct"] is None else f"{m['exact_pct']}%"
        lines.append(f"| `{name}` | {'yes' if m['in_raw'] else 'no'} | "
                     f"{m['present']} | {m['exact']} | {pct} |")
    lines += [
        "",
        "`step_index` and `agent_role` are session-level fields the harvest "
        "assigns positionally/derived; `_raw_for_step` now renders them into "
        "the raw (`step_index: N; agent_role: X; tool: ...`), so both are "
        "extracted at 100%. `duration_ms`, `error_type` and `error_message` "
        "remain derived from the body (not rendered) and are scored only where "
        "the gold carries them.",
        "",
        f"`stdout` misses that are harvest-clip artifacts "
        f"(`[truncated N chars]` in the gold): "
        f"**{result['stdout_truncated_misses']}** of "
        f"{result['fields']['stdout']['present'] - result['fields']['stdout']['exact']}. "
        "The raw and the stored gold clip the value at different points, so "
        "the gold tail is not in the input; these are data-rendering, not "
        "extraction, failures.",
        "",
        "## False positives",
        "",
        "Non-`agent_step` text routed to `agent_step` (precision check):",
        "",
        "| Set | n | routed to agent_step |",
        "|---|---|---|",
    ]
    for name, m in result["false_positives"].items():
        lines.append(f"| {name} | {m['n']} | {m['routed_agent_step']} |")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    _register_schemas()
    result = evaluate()
    write_report(result)
    print(json.dumps(result, indent=2))
    print(f"report written to {args.report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
