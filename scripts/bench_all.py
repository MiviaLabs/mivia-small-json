#!/usr/bin/env python3
"""Unified benchmark report: internal per-schema plus external comparisons.

One command, one report a release can be judged from:

1. Internal per-schema metrics on the frozen held-out split (routing,
   validity, latency, artifact size, per-class F1).
2. LogHub template parsing (GA / PA / FGA / FTA) vs published parsers.
3. External short-text classification (banking77, clinc150) macro F1.
4. tool_call field extraction on the committed Glaive dev slice.

Writes reports/benchmark-unified.md. Deterministic, offline.
"""
from __future__ import annotations

import importlib.metadata as im
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPORT = ROOT / "reports" / "benchmark-unified.md"

# Published references, for context only.
PUBLISHED_DRAIN_PA = 0.865   # ICSE 2019, arXiv:1811.03509, Table IV
PUBLISHED_BANKING77_F1 = 0.93  # TF-IDF + linear baseline, upper end
PUBLISHED_CLINC150_F1 = 0.93


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _versions() -> list[str]:
    def v(name):
        try:
            return im.version(name)
        except Exception:
            return "?"
    return [f"python {platform.python_version()}",
            f"scikit-learn {v('scikit-learn')}", f"numpy {v('numpy')}",
            f"skops {v('skops')}", f"scipy {v('scipy')}"]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True
                              ).stdout.strip()
    except Exception:
        return "unknown"


def run_per_schema() -> dict:
    from mivia_small_json import bench

    return bench.main()


def run_loghub_templates() -> tuple[dict, dict]:
    mod = _load("bench_loghub_templates", "scripts/bench_loghub_templates.py")
    from mivia_small_json.logtemplate import evaluate

    per: dict[str, dict] = {}
    gt_e, pr_g, gt_t, pr_t = [], [], [], []
    for system in mod._systems():
        metrics, s_gt_e, s_pr_g, s_gt_t, s_pr_t = mod.score_system(system)
        per[system] = metrics
        gt_e += [f"{system}:{x}" for x in s_gt_e]
        pr_g += [f"{system}:{x}" for x in s_pr_g]
        gt_t += s_gt_t
        pr_t += s_pr_t
    return evaluate(gt_e, pr_g, gt_t, pr_t), per


def run_external() -> dict:
    mod = _load("bench_external", "scripts/bench_external.py")
    out = {}
    for name, loader in mod.DATASETS:
        out[name] = mod.evaluate(*loader())
    return out


def _is_json_object(text: str) -> bool:
    """True only for a JSON object.

    The `arguments` field must be an object. Parsing any JSON would count the
    string `"{"` (a truncated capture) as valid, which is how the earlier
    96.33% reading overstated the rate.
    """
    try:
        return isinstance(json.loads(text or ""), dict)
    except ValueError:
        return False


def run_tool_call() -> dict:
    from mivia_small_json import bootstrap, pipeline
    from mivia_small_json.model import MODEL_PATH
    from mivia_small_json.tool_call import register_tool_call_schema, repair_arguments

    bootstrap.register_example_schemas()
    register_tool_call_schema()
    router = pipeline.load_router(MODEL_PATH)
    rows = [json.loads(line) for line in
            (ROOT / "data" / "real" / "tool_call" / "dev.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    routed = name_ok = args_ok = args_ok_repaired = 0
    for row in rows:
        obj, _meta = pipeline.classify(row["raw"], router)
        gold = row["gold"]
        routed += obj.get("type") == "tool_call"
        name_ok += obj.get("name") == gold.get("name")
        raw_args = obj.get("arguments") or ""
        args_ok += _is_json_object(raw_args)
        args_ok_repaired += _is_json_object(repair_arguments(raw_args))
    n = len(rows) or 1
    return {"n": len(rows), "routed_pct": round(100 * routed / n, 2),
            "name_acc": round(name_ok / n, 4),
            "args_valid_pct": round(100 * args_ok / n, 2),
            "args_valid_after_repair_pct": round(100 * args_ok_repaired / n, 2)}


def run_agent_step() -> dict:
    """agent_step quality on the session-disjoint dev split (see
    scripts/bench_agent_step.py): the frozen held-out set is 4-schema and
    carries no agent_step rows."""
    mod = _load("bench_agent_step", "scripts/bench_agent_step.py")
    mod._register_schemas()
    return mod.evaluate()


def _active_router_size() -> tuple[int, str]:
    """Bytes and artifact name for the router the pipeline actually serves.

    ``load_active_router`` prefers the ONNX encoder and falls back to the
    skops router only when the encoder artifact or its runtime is
    unavailable. Reading ``MODEL_PATH`` unconditionally reported the
    fallback (353 KB) even while the ~23 MB encoder was serving, so the
    row names the artifact it measured.
    """
    from mivia_small_json import encoder, pipeline
    from mivia_small_json.model import MODEL_PATH

    if isinstance(pipeline.load_active_router(), encoder.EncoderRouter):
        path = encoder.DEFAULT_MODEL
    else:
        path = MODEL_PATH
    return (path.stat().st_size if path.is_file() else 0), path.name


def main() -> int:
    per_schema = run_per_schema()
    loghub_overall, loghub_per = run_loghub_templates()
    external = run_external()
    tool_call = run_tool_call()
    agent_step = run_agent_step()
    size, size_label = _active_router_size()

    r, f = per_schema["rules"], per_schema["full"]
    lines = [
        "# Unified benchmark report — mivia-small-json",
        "",
        f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} · "
        "script: `scripts/bench_all.py`",
        "",
        "## 1. Internal per-schema (frozen held-out, 4-schema world)",
        "",
        f"Held-out n = {f['n']} (never trained on).",
        "",
        "| Metric | rules baseline | full pipeline |",
        "|---|---|---|",
        f"| valid JSON % | {r['valid_pct']:.2f} | {f['valid_pct']:.2f} |",
        f"| routed-and-valid % (strict) | {r['routed_and_valid_pct']:.2f} | "
        f"{f['routed_and_valid_pct']:.2f} |",
        f"| F1 macro (routing) | {r['f1_macro']:.4f} | {f['f1_macro']:.4f} |",
        f"| latency p50 (ms) | {r['p50_ms']:.3f} | {f['p50_ms']:.3f} |",
        f"| latency p95 (ms) | {r['p95_ms']:.3f} | {f['p95_ms']:.3f} |",
        f"| model size (bytes, active router: {size_label}) | 0 | {size} |",
        "",
        "F1 per class (full pipeline):",
        "",
        "| Schema | F1 |",
        "|---|---|",
        *[f"| {s} | {v:.4f} |" for s, v in f["f1_per_class"].items()],
        "",
        "## 2. LogHub template parsing (GA / PA / FGA / FTA)",
        "",
        f"Overall over {len(loghub_per)} systems: "
        f"**GA {loghub_overall['GA']:.4f} · PA {loghub_overall['PA']:.4f} · "
        f"FGA {loghub_overall['FGA']:.4f} · FTA {loghub_overall['FTA']:.4f}**.",
        f"Published Drain average (ICSE 2019): {PUBLISHED_DRAIN_PA} "
        "message-level (closest to GA).",
        "",
        "| System | GA | PA | FGA | FTA |",
        "|---|---|---|---|---|",
        *[f"| {s} | {m['GA']:.4f} | {m['PA']:.4f} | {m['FGA']:.4f} | "
          f"{m['FTA']:.4f} |" for s, m in loghub_per.items()],
        "",
        "## 3. External short-text classification (macro F1)",
        "",
        "Comparison data only; never router training data.",
        "",
        "| Set | Classes | Accuracy | Macro F1 | Published (linear) |",
        "|---|---|---|---|---|",
    ]
    pub = {"banking77": PUBLISHED_BANKING77_F1, "clinc150": PUBLISHED_CLINC150_F1}
    for name, m in external.items():
        lines.append(f"| {name} | {m['n_classes']} | {m['accuracy']} | "
                     f"{m['f1_macro']} | {pub.get(name, 'n/a')} |")
    lines += [
        "",
        "## 4. tool_call field extraction (Glaive dev slice)",
        "",
        f"n = {tool_call['n']} · routed to tool_call "
        f"{tool_call['routed_pct']}% · name accuracy "
        f"{tool_call['name_acc']}.",
        f"- arguments valid JSON as emitted by the pipeline: "
        f"{tool_call['args_valid_pct']}%",
        f"- arguments valid JSON after the schema's own `repair_arguments`: "
        f"{tool_call['args_valid_after_repair_pct']}%",
        "Source: `data/real/tool_call/dev.jsonl` (public Glaive "
        "function-calling traces, held-out dev slice).",
        "Finding: the registered extractor calls `repair_arguments` on every "
        "captured argument object, so single-quoted pseudo-JSON and a captured "
        "extra closer are normalised before emission.",
        "",
        "## 5. agent_step quality (session-disjoint dev split)",
        "",
        "The frozen held-out set is 4-schema and carries no `agent_step` "
        "rows, so this schema is measured separately on real data the router "
        "did not train on: `data/real/agent_step/dev.jsonl` "
        f"(n = {agent_step['n']}, session-disjoint from train).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| routed to `agent_step` | **{agent_step['routed_pct']}%** |",
        f"| valid JSON | **{agent_step['valid_pct']}%** |",
        f"| latency p50 / p95 | {agent_step['lat_p50_ms']} / "
        f"{agent_step['lat_p95_ms']} ms |",
        f"| false positives (350 held-out + 100 tool_call dev) | "
        f"{sum(m['routed_agent_step'] for m in agent_step['false_positives'].values())} |",
        "",
        "Per-field exact match (in-raw fields):",
        "",
        "| Field | Exact % |",
        "|---|---|",
        *[f"| `{f}` | {m['exact_pct']}% |"
          for f, m in agent_step["fields"].items()
          if m["in_raw"] and m["exact_pct"] is not None],
        "",
        "`step_index` and `agent_role` are rendered into the raw and "
        "extracted at 100%. `stdout` misses are "
        f"dominated by harvest clip markers "
        f"({agent_step['stdout_truncated_misses']} of "
        f"{agent_step['fields']['stdout']['present'] - agent_step['fields']['stdout']['exact']}).",
        "Full write-up: `reports/benchmark-agent-step.md`.",
        "",
        "## Environment fingerprint",
        "",
        f"- {' · '.join(_versions())}",
        f"- git SHA {_git_sha()}",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"loghub overall: {loghub_overall}")
    print(f"external: {external}")
    print(f"tool_call: {tool_call}")
    print(f"report written to {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
