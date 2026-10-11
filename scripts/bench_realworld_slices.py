#!/usr/bin/env python3
"""Real-world routing eval: the 6 curated wild slices (2026-10-08).

Scores the shipped chain (classify on the active ONNX router, full schema
world) over data/wild/realworld/*.jsonl. Each slice declares the set of
acceptable outcomes; flagged slices (post-commit, slack-snippets,
long-log-dumps) accept both sides of the flag. Rows are
{"text", "expected", "source", "ambiguous", "note"}; ambiguous rows are
reported both ways — they are torn by construction, so the strict number
excludes them and the flagged number includes them. Deterministic,
offline, CPU-only. Writes reports/benchmark-realworld-slices.md.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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

SLICES = ROOT / "data" / "wild" / "realworld"
REPORT = ROOT / "reports" / "benchmark-realworld-slices.md"

# slice -> acceptable routed classes. `text` is the renamed no-tool/OOD
# outcome of the retrained router (mm12); legacy `no_tool_call` still
# exists in the 11-class map, so both credit as the no-tool outcome.
ACCEPT = {
    "issue-realworld-v2": {"issue"},
    "post-commit": {"commit", "no_tool_call", "text"},
    "chat-informal": {"no_tool_call", "text"},
    "slack-snippets": {"no_tool_call", "text", "log_event"},
    "fa-ar-ood-fa": {"no_tool_call", "text"},
    "fa-ar-ood-ar": {"no_tool_call", "text"},
    "long-log-dumps": {"log_event", "diagnostic"},
}


def main() -> int:
    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()
    router = load_active_router()

    results = []
    for name in sorted(ACCEPT):
        path = SLICES / f"{name}.jsonl"
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        strict_ok = flag_ok = valid = in_domain = 0
        confusions: Counter[str] = Counter()
        n_strict = 0
        for row in rows:
            obj, meta = classify(row["text"], router)
            valid += bool(meta["valid"])
            in_domain += bool(obj.get("in_domain", True))
            routed = obj.get("type")
            ok = routed in ACCEPT[name]
            flag_ok += ok
            if ok:
                confusions["__ok__"] += 1
            else:
                confusions[routed or "<none>"] += 1
            if not row["ambiguous"]:
                strict_ok += ok
                n_strict += 1
        results.append({
            "name": name, "n": len(rows), "strict_n": n_strict,
            "strict": 100 * strict_ok / max(1, n_strict),
            "flagged": 100 * flag_ok / len(rows),
            "valid": 100 * valid / len(rows),
            "in_domain": 100 * in_domain / len(rows),
            "confusions": {k: v for k, v in confusions.most_common()
                           if k != "__ok__"},
        })

    import subprocess

    git_sha = "unknown"
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True,
                                 timeout=30, check=False).stdout.strip() or "unknown"
    except Exception:
        pass

    import sklearn

    lines = [
        "# Real-world routing eval — curated wild slices",
        "",
        "Seven curated real-data slices pinning known OOD routing"
        " weaknesses (informal chat, social text, non-English prose,"
        " long dumps). Gold labels + provenance:"
        " `data/wild/realworld/` + `data/wild/REALWORLD_SLICES_PROVENANCE.md`."
        " Collect/curate: `scripts/collect_realworld_slices.py`.",
        "",
        f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d')} ·"
        " script: `scripts/bench_realworld_slices.py` · shipped chain"
        " (`classify` on the active router, full schema world)",
        "",
        f"Env: git {git_sha[:12]} · python"
        f" {sys.version.split()[0]} · scikit-learn {sklearn.__version__}",
        "",
        "| slice | n | strict % (excl. ambiguous) | flagged-set % |"
        " valid % | in_domain % | top confusions |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        conf = ", ".join(f"{k} {v}" for k, v in
                         list(r["confusions"].items())[:3]) or "—"
        lines.append(
            f"| {r['name']} | {r['n']} | {r['strict']:.1f} (n={r['strict_n']})"
            f" | {r['flagged']:.1f} | {r['valid']:.1f}"
            f" | {r['in_domain']:.1f} | {conf} |")
    lines.append("")
    lines.append("`strict %` scores only rows with a confident gold label;"
                 " `flagged-set %` credits any acceptable outcome for the"
                 " slice (both sides of a flagged pair). Ambiguous rows are"
                 " excluded from strict scoring, never hidden: they are the"
                 " torn rows by construction. Each slice's strict value is"
                 " a PRE-REGISTERED BAR (one-sided floor):"
                 " tests/test_realworld_bars.py fails on any regression,"
                 " and a floor may only move via a conscious re-registration"
                 " in that file.")
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    for r in results:
        print(r["name"], f"strict={r['strict']:.1f}",
              f"flagged={r['flagged']:.1f}", f"valid={r['valid']:.1f}")
    print(f"report written to {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
