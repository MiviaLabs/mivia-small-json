#!/usr/bin/env python3
"""Guard latency + prune-quality report (spec guard-serve-performance).

A REPORT, never a CI gate: no timing asserts anywhere (AGENTS.md law); the
numbers below are the record, whatever they are.

Protocol (named, one box, one process):

- Tool pool: every unique tool dict harvested from the committed BFCL
  corpora (calls.jsonl, irrelevance_eval.jsonl catalogues + the per-row
  tools of guard-eval-v2.jsonl), deduped by canonical JSON, sorted by name.
- Catalogues: seeded (default seed 20261010) fixed catalogues of 10/50/200
  tools (MAX_TOOLS) from that pool — deterministic per seed.
- Requests: a fixed seeded set of real call requests from the
  ``single_turn_relevance`` (label 1) rows of guard-eval-v2.jsonl.
- Path: the public guard surface (``api.do_guard`` — composite classify +
  catalogue decision + veto), with the process guard singleton reset per
  (mode, size) so every cold sample is a fresh loader's FIRST decision.
- Modes: shipped (prune off — the default) vs prune-on
  (MIVIA_GUARD_PRUNE=1, k=12) via the documented environment passthrough.
- Samples: one cold decision per (mode, size); ``--warm-iters`` warm
  decisions cycling the request set; p50/p95 over the warm samples.
- Pruner overhead (spec bar 4): the first stage alone at each catalogue
  size, p50/p95 over the request set.
- Fingerprint: python / scikit-learn / numpy / onnxruntime / tokenizers
  versions, git SHA, host — plus the sha256 of every artifact scored
  (instruments name the bytes they scored).

``--bars`` measures the pre-registered prune-recall and decision-agreement
bars on the canonical corpora (calls.jsonl retention control, the 884-row
BFCL irrelevance set, guard-eval-v2's catalogue set). The bars decide the
DEFAULT, not this report:

1. prune recall: shipped argmax survives pruning on 100% of the 183-row
   retention control; >= 99.5% pooled across the full eval catalogue set.
2. decision agreement (after veto, at THRESHOLD): >= 99% pooled; per guard
   eval category no regression > 2.0pp; ``no_tool_needed`` not regressing.

Any miss => the prune default stays OFF and the miss is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
CALLS = BFCL / "calls.jsonl"
FP_EVAL = BFCL / "irrelevance_eval.jsonl"
GUARD_EVAL_V2 = BFCL / "guard-eval-v2.jsonl"
SIZES = (10, 50, 200)
SEED = 20261010
WARM_ITERS = 12
N_REQUESTS = 8
PRUNE_ENV = "MIVIA_GUARD_PRUNE"
PRUNE_K_ENV = "MIVIA_GUARD_PRUNE_K"

V2_RELEVANCE = "single_turn_relevance"
V2_CATEGORIES = (
    "single_turn_irrelevance",
    "single_turn_relevance",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "ask_followup",
    "infeasible",
    "foreign_tool",
    "sibling_swap",
    "no_tool_needed",
)

ARTIFACTS = (
    "models/tool-match-gate-int8.onnx",
    "models/tool-match-gate-tokenizer.json",
    "models/tool-match-int8.onnx",
    "models/tool-match-tokenizer.json",
    "models/router-encoder-int8.onnx",
    "models/router-tokenizer.json",
)


def _sha256(path: Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _catalogue_text(tools: list[dict], request: str) -> str:
    return ("Tools available:\n" + json.dumps(tools, ensure_ascii=False)
            + "\n\nUser request: " + request)


def _pctl(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _fingerprint() -> dict:
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        sha = git.stdout.strip() if git.returncode == 0 else "unknown"
    except Exception:
        sha = "unknown"
    versions = {}
    for pkg in ("scikit-learn", "numpy", "onnxruntime", "tokenizers",
                "fastapi", "mivia-small-json"):
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            versions[pkg] = "unavailable"
    return {
        "python": platform.python_version(),
        "packages": versions,
        "git_sha": sha,
        "host": platform.node(),
        "platform": platform.platform(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifacts_sha256": {name: _sha256(ROOT / name)
                             for name in ARTIFACTS},
    }


# ---------------------------------------------------------------------------
# Fixed corpora: tool pool, catalogues, request set (all seeded/deterministic)
# ---------------------------------------------------------------------------

def _tool_pool() -> list[dict]:
    """Every unique tool dict in the committed BFCL corpora, sorted by name."""
    pool: dict[str, dict] = {}
    for path in (CALLS, FP_EVAL):
        for rec in _rows(path):
            text = rec.get("text") or rec.get("raw") or ""
            head, sep, _req = text.partition("\n\nUser request:")
            if not sep:
                continue
            start = head.find("[")
            if start < 0:
                continue
            try:
                tools = json.loads(head[start:])
            except ValueError:
                continue
            for tool in tools:
                if isinstance(tool, dict) and tool.get("name"):
                    key = json.dumps(tool, sort_keys=True, ensure_ascii=False)
                    pool.setdefault(key, tool)
    for rec in _rows(GUARD_EVAL_V2):
        tool = rec.get("tool")
        if isinstance(tool, dict) and tool.get("name"):
            key = json.dumps(tool, sort_keys=True, ensure_ascii=False)
            pool.setdefault(key, tool)
    return sorted(pool.values(),
                  key=lambda t: json.dumps(t, sort_keys=True,
                                           ensure_ascii=False))


def _catalogues(pool: list[dict], seed: int) -> dict[int, list[list[dict]]]:
    """Fixed catalogues per size: disjoint seeded chunks, size replicas each."""
    out: dict[int, list[list[dict]]] = {}
    for size in SIZES:
        rng = random.Random(seed + size)
        order = list(range(len(pool)))
        rng.shuffle(order)
        reps: list[list[dict]] = []
        start = 0
        for _rep in range(3):
            chunk: list[dict] = []
            while len(chunk) < size:  # wrap: reuse from the top if exhausted
                take = min(size - len(chunk), len(order) - start)
                chunk.extend(pool[i] for i in order[start:start + take])
                start += take
                if start >= len(order):
                    start = 0
            reps.append(chunk)
        out[size] = reps
    return out


def _request_set(seed: int, n: int = N_REQUESTS) -> list[str]:
    """Real call requests from the v2 relevance control (label 1), seeded."""
    rows = [r["request"] for r in _rows(GUARD_EVAL_V2)
            if r.get("category") == V2_RELEVANCE and r.get("request")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    picked: list[str] = []
    for r in rows:
        if r not in picked:
            picked.append(r)
        if len(picked) >= n:
            break
    return picked


# ---------------------------------------------------------------------------
# Latency protocol
# ---------------------------------------------------------------------------

def _set_prune_env(enabled: bool) -> None:
    if enabled:
        os.environ[PRUNE_ENV] = "1"
        os.environ[PRUNE_K_ENV] = "12"
    else:
        os.environ.pop(PRUNE_ENV, None)
        os.environ.pop(PRUNE_K_ENV, None)


def _fresh_guard():
    """Reset the process guard singleton so the next decision constructs a
    fresh loader (the cold protocol). Script-only: tests use monkeypatch."""
    from mivia_small_json import tool_match

    tool_match._GUARD = None
    tool_match._GUARD_LOADED = False
    return tool_match._get_guard()


def measure_latency(warm_iters: int, seed: int) -> dict:
    from mivia_small_json import api, pipeline, tool_match
    from mivia_small_json.tool_match import DEFAULT_PRUNE_K

    router = pipeline.load_active_router()
    pool = _tool_pool()
    catalogues = _catalogues(pool, seed)
    requests = _request_set(seed)
    print(f"tool pool: {len(pool)} unique tools · "
          f"requests: {len(requests)} · sizes {SIZES}")

    saved_env = {k: os.environ.get(k) for k in (PRUNE_ENV, PRUNE_K_ENV)}
    results: dict[str, dict] = {}
    try:
        for mode, enabled in (("shipped", False), ("prune_on", True)):
            mode_out: dict[str, dict] = {}
            for size in SIZES:
                _set_prune_env(enabled)
                reps = catalogues[size]
                # Cold: fresh loader, first decision (construction untimed;
                # the decision itself is the cold sample).
                _fresh_guard()
                cold_ms = time.perf_counter()
                out0 = api.do_guard(requests[0], list(reps[0]), router)
                cold = (time.perf_counter() - cold_ms) * 1000.0
                # Warm: cycle the request set across the catalogue replicas.
                warm: list[float] = []
                i = 0
                while len(warm) < warm_iters:
                    request = requests[i % len(requests)]
                    tools = reps[(i // len(requests)) % len(reps)]
                    t0 = time.perf_counter()
                    out = api.do_guard(request, list(tools), router)
                    warm.append((time.perf_counter() - t0) * 1000.0)
                    i += 1
                mode_out[str(size)] = {
                    "cold_first_decision_ms": round(cold, 2),
                    "warm_p50_ms": round(_pctl(warm, 0.50), 2),
                    "warm_p95_ms": round(_pctl(warm, 0.95), 2),
                    "warm_samples_ms": [round(x, 2) for x in warm],
                    "routed_record_types": sorted({
                        out0["record"]["type"], out["record"]["type"]}),
                }
                print(f"  {mode:<9} n={size:<4} cold {cold:8.1f} ms · "
                      f"warm p50 {mode_out[str(size)]['warm_p50_ms']:8.1f} · "
                      f"p95 {mode_out[str(size)]['warm_p95_ms']:8.1f} ms "
                      f"({out0['record']['type']})")
            results[mode] = mode_out
    finally:
        _set_prune_env(False)
        for key, val in saved_env.items():
            if val is not None:
                os.environ[key] = val
        # Leave the process with the shipped default guard.
        _fresh_guard()

    # Pruner overhead alone (bar 4): the first stage at each size.
    pruner = tool_match.LexicalToolPruner()
    overhead: dict[str, dict] = {}
    for size in SIZES:
        tools = catalogues[size][0]
        samples = []
        for request in requests:
            t0 = time.perf_counter()
            pruner.select_survivors(request, tools, DEFAULT_PRUNE_K)
            samples.append((time.perf_counter() - t0) * 1000.0)
        overhead[str(size)] = {"p50_ms": round(_pctl(samples, 0.50), 3),
                               "p95_ms": round(_pctl(samples, 0.95), 3)}
        print(f"  pruner    n={size:<4} p50 {overhead[str(size)]['p50_ms']}"
              f" ms · p95 {overhead[str(size)]['p95_ms']} ms")

    # Pre-registered latency bar 3 (recorded, never a gate):
    # prune-on p95 @200 <= 1.5x shipped p95 @10.
    base = results["shipped"]["10"]["warm_p95_ms"]
    big = results["prune_on"]["200"]["warm_p95_ms"]
    ratio = round(big / base, 3) if base > 0 else None
    latency_bar = {"ship_p95_10_ms": base, "prune_p95_200_ms": big,
                   "ratio": ratio, "bound": 1.5,
                   "pass": bool(ratio is not None and ratio <= 1.5)}
    print(f"latency bar (prune p95@200 <= 1.5x shipped p95@10): "
          f"{big} / {base} = {ratio} -> "
          f"{'PASS' if latency_bar['pass'] else 'MISS'}")
    return {"protocol": {"catalogue_sizes": list(SIZES), "seed": seed,
                         "requests": requests,
                         "warm_iterations": warm_iters,
                         "prune_k": DEFAULT_PRUNE_K,
                         "path": "api.do_guard (public guard surface)"},
            "latency": results, "pruner_overhead": overhead,
            "latency_bar": latency_bar}


# ---------------------------------------------------------------------------
# Pre-registered quality bars (the recall/agreement instrument)
# ---------------------------------------------------------------------------

def measure_bars(k: int) -> dict:
    from mivia_small_json import pipeline, tool_match
    from mivia_small_json.tool_match import (PruneConfig, THRESHOLD,
                                             parse_catalogue_raw)

    guard_shipped = tool_match.load_matcher(prune=PruneConfig(enabled=False))
    guard_pruned = tool_match.load_matcher(
        prune=PruneConfig(enabled=True, k=k))
    if guard_shipped is None or guard_pruned is None:
        raise SystemExit("shipped guard not loadable")
    # The catalogue scan surface (the split's gate when shipped). The split
    # owns the prune config, never the gate.
    scan = guard_shipped.gate if hasattr(guard_shipped, "gate") \
        else guard_shipped
    assert guard_pruned.prune is not None and guard_pruned.prune.enabled \
        and guard_pruned.prune.k == k, \
        "the prune-on guard must actually carry the config"
    pruner = tool_match.LexicalToolPruner()
    pipeline.load_active_router()

    def decision(text: str) -> str:
        _tool, score = guard_shipped.best_match(text)
        keep_base = "keep" if score >= THRESHOLD else "abstain"
        _tool2, score2 = guard_pruned.best_match(text)
        keep_prune = "keep" if score2 >= THRESHOLD else "abstain"
        return keep_base if keep_base == keep_prune else "differs"

    def prune_hits(tools: list[dict] | None, request: str) -> bool:
        if not tools or len(tools) <= 1:
            # No catalogue, or min(k, 1) = 1: nothing the prune can drop.
            return True
        # The shipped argmax over the FULL catalogue (gate scan, shipped
        # spans) must appear in the survivor set.
        argmax_tool, _score = scan._argmax_over(request, tools, tools)
        survivors = pruner.select_survivors(request, tools, k)
        return any(s is argmax_tool for s in survivors)

    sets: dict[str, dict] = {}
    # -- retention control + BFCL irrelevance (multi-tool catalogue sets) --
    for name, path in (("retention_control_calls", CALLS),
                       ("bfcl_irrelevance", FP_EVAL)):
        recall_hits = recall_n = 0
        agree = false_abstains = 0
        for rec in _rows(path):
            text = rec.get("text") or rec.get("raw") or ""
            tools, request = parse_catalogue_raw(text)
            if not tools or not request:
                continue
            recall_n += 1
            recall_hits += 1 if prune_hits(tools, request) else 0
            verdict = decision(text)
            agree += 1 if verdict != "differs" else 0
            if verdict == "abstain" and rec.get("label") == 1:
                false_abstains += 1
        sets[name] = {"catalogues": recall_n,
                      "prune_recall": round(recall_hits / recall_n, 6),
                      "recall_misses": recall_n - recall_hits,
                      "decision_agreement": round(agree / recall_n, 6),
                      "label1_false_abstentions_under_prune": false_abstains}
        print(f"  {name:<28} n={recall_n:<5} recall "
              f"{sets[name]['prune_recall']:.4f} · agreement "
              f"{sets[name]['decision_agreement']:.4f}")

    # -- guard-eval-v2 catalogue set: pooled agreement + per-category --
    v2_rows = _rows(GUARD_EVAL_V2)
    by_cat: dict[str, dict] = {c: {"n": 0, "abstain_base": 0,
                                   "abstain_prune": 0, "agree": 0}
                               for c in V2_CATEGORIES}
    pooled = {"n": 0, "agree": 0, "recall_n": 0, "recall_hits": 0}
    for rec in v2_rows:
        text = _catalogue_text(rec["tool"], rec["request"])
        tools, request = parse_catalogue_raw(text)
        cat = rec["category"]
        keep_base, _s = guard_shipped.best_match(text)
        keep_base = "keep" if _s >= THRESHOLD else "abstain"
        keep_prune, _s2 = guard_pruned.best_match(text)
        keep_prune = "keep" if _s2 >= THRESHOLD else "abstain"
        by_cat[cat]["n"] += 1
        by_cat[cat]["abstain_base"] += keep_base == "abstain"
        by_cat[cat]["abstain_prune"] += keep_prune == "abstain"
        by_cat[cat]["agree"] += keep_base == keep_prune
        pooled["n"] += 1
        pooled["agree"] += keep_base == keep_prune
        pooled["recall_n"] += 1
        # Single-tool catalogues cannot miss (min(k, 1) = 1) — but verify
        # instead of assuming: any multi-tool row here is actually measured.
        pooled["recall_hits"] += 1 if prune_hits(tools, request) else 0

    categories = {}
    worst = 0.0
    for cat, v in by_cat.items():
        base = 100.0 * v["abstain_base"] / v["n"] if v["n"] else 0.0
        prune = 100.0 * v["abstain_prune"] / v["n"] if v["n"] else 0.0
        categories[cat] = {
            "catalogues": v["n"],
            "abstention_shipped_pct": round(base, 2),
            "abstention_prune_pct": round(prune, 2),
            "delta_pp": round(prune - base, 2),
            "agreement": round(v["agree"] / v["n"], 6) if v["n"] else 1.0,
        }
        worst = min(worst, prune - base)
    ntn_base = categories["no_tool_needed"]["abstention_shipped_pct"]
    ntn_prune = categories["no_tool_needed"]["abstention_prune_pct"]

    total_n = (sets["retention_control_calls"]["catalogues"]
               + sets["bfcl_irrelevance"]["catalogues"] + pooled["n"])
    total_hits = (sets["retention_control_calls"]["prune_recall"]
                  * sets["retention_control_calls"]["catalogues"]
                  + sets["bfcl_irrelevance"]["prune_recall"]
                  * sets["bfcl_irrelevance"]["catalogues"]
                  + pooled["recall_hits"])
    total_agree = (sets["retention_control_calls"]["decision_agreement"]
                   * sets["retention_control_calls"]["catalogues"]
                   + sets["bfcl_irrelevance"]["decision_agreement"]
                   * sets["bfcl_irrelevance"]["catalogues"]
                   + pooled["agree"])
    recall_bar = {
        "retention_control_required": 1.0,
        "retention_control": sets["retention_control_calls"]["prune_recall"],
        "retention_control_pass": (
            sets["retention_control_calls"]["prune_recall"] == 1.0),
        "pooled_required": 0.995,
        "pooled": round(total_hits / total_n, 6),
        "pooled_pass": (total_hits / total_n) >= 0.995,
    }
    agreement = total_agree / total_n
    agreement_bar = {
        "required": 0.99, "pooled": round(agreement, 6),
        "pass": agreement >= 0.99,
        "worst_category_delta_pp": round(worst, 2),
        "no_category_regresses_gt_2pp": worst >= -2.0,
        "no_tool_needed_delta_pp": round(ntn_prune - ntn_base, 2),
        "no_tool_needed_not_regressing": ntn_prune >= ntn_base,
    }
    verdict = "PASS" if (recall_bar["retention_control_pass"]
                         and recall_bar["pooled_pass"]
                         and agreement_bar["pass"]
                         and agreement_bar["no_category_regresses_gt_2pp"]
                         and agreement_bar["no_tool_needed_not_regressing"]) \
        else "MISS"
    print(f"  recall bar: retention-control "
          f"{recall_bar['retention_control']} (need 1.0), pooled "
          f"{recall_bar['pooled']} (need >= 0.995)")
    print(f"  agreement bar: pooled {agreement_bar['pooled']} "
          f"(need >= 0.99), worst category delta "
          f"{agreement_bar['worst_category_delta_pp']}pp, no_tool_needed "
          f"delta {agreement_bar['no_tool_needed_delta_pp']}pp")
    print(f"BARS VERDICT: {verdict} (a miss keeps the prune default OFF)")
    return {"prune_k": k, "sets": sets, "v2_categories": categories,
            "v2_pooled": {"catalogues": pooled["n"],
                          "agreement": round(pooled["agree"] / pooled["n"], 6)},
            "recall_bar": recall_bar, "agreement_bar": agreement_bar,
            "verdict": verdict}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(out_dir: Path, latency: dict | None, bars: dict | None,
                 fingerprint: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    payload = {"spec": "guard-serve-performance",
               "report": "latency + prune-quality record, never a CI gate",
               "fingerprint": fingerprint, "latency": latency, "bars": bars}
    json_path = out_dir / f"guard-serve-performance-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    lines = [f"# Guard serve performance — {stamp}",
             "",
             "Spec: `guard-serve-performance`. A record,",
             "never a CI gate. One box, one process, public guard path",
             "(`api.do_guard`), fixed seeded catalogues from the committed",
             "BFCL tool pool, shipped bytes sha256-stamped below.",
             "",
             "## Fingerprint",
             "",
             f"- python {fingerprint['python']} · git "
             f"`{fingerprint['git_sha'][:12]}` · host "
             f"`{fingerprint['host']}`",
             "- packages: "
             + ", ".join(f"{k} {v}"
                         for k, v in fingerprint["packages"].items()),
             "- artifacts:",
             ]
    for name, sha in fingerprint["artifacts_sha256"].items():
        if sha:
            lines.append(f"  - `{name}` sha256 {sha[:16]}…")
    if latency:
        lines += ["", "## Latency (ms) — shipped vs prune-on (k=12)", "",
                  "| mode | n | cold first | warm p50 | warm p95 |",
                  "|---|---|---|---|---|"]
        for mode in ("shipped", "prune_on"):
            for size in ("10", "50", "200"):
                cell = latency["latency"][mode][size]
                lines.append(
                    f"| {mode} | {size} | "
                    f"{cell['cold_first_decision_ms']} | "
                    f"{cell['warm_p50_ms']} | {cell['warm_p95_ms']} |")
        over = latency["pruner_overhead"]
        lines += ["", f"Pruner overhead alone (BM25 first stage): "
                  f"n=10 p95 {over['10']['p95_ms']} ms · n=50 p95 "
                  f"{over['50']['p95_ms']} ms · n=200 p95 "
                  f"{over['200']['p95_ms']} ms.", ""]
        bar = latency["latency_bar"]
        lines += [
            f"Latency bar (prune-on p95 @200 <= 1.5x shipped p95 @10): "
            f"{bar['prune_p95_200_ms']} / {bar['ship_p95_10_ms']} = "
            f"{bar['ratio']} -> "
            f"{'PASS' if bar['pass'] else 'MISS (recorded; report only)'}."]
    if bars:
        lines += ["", f"## Prune quality bars (k={bars['prune_k']})", ""]
        rb, ab = bars["recall_bar"], bars["agreement_bar"]
        lines += [
            f"- Recall bar: retention control {rb['retention_control']} "
            f"(required 1.0) -> "
            f"{'PASS' if rb['retention_control_pass'] else 'MISS'}; "
            f"pooled {rb['pooled']} (required >= {rb['pooled_required']}) "
            f"-> {'PASS' if rb['pooled_pass'] else 'MISS'}.",
            f"- Agreement bar: pooled {ab['pooled']} "
            f"(required >= {ab['required']}) -> "
            f"{'PASS' if ab['pass'] else 'MISS'}; worst category delta "
            f"{ab['worst_category_delta_pp']}pp (bound -2.0) -> "
            f"{'ok' if ab['no_category_regresses_gt_2pp'] else 'REGRESSION'};"
            f" no_tool_needed delta "
            f"{ab['no_tool_needed_delta_pp']}pp -> "
            f"{'ok' if ab['no_tool_needed_not_regressing'] else 'REGRESSION'}.",
            "", "| v2 category | abstain shipped % | abstain prune % | "
            "delta pp | agreement |", "|---|---|---|---|---|"]
        for cat, v in bars["v2_categories"].items():
            lines.append(f"| {cat} | {v['abstention_shipped_pct']} | "
                         f"{v['abstention_prune_pct']} | {v['delta_pp']} | "
                         f"{v['agreement']} |")
        lines += ["", f"**QUALITY BARS: {bars['verdict']}** (a miss keeps "
                  "the prune default OFF; bars are never retuned after "
                  "measurement)."]
    if latency and bars:
        # The default decision is data: the pre-registered rule flips the
        # default only when EVERY registered bar passes; the latency bar is
        # one of them, and the spec names the multilingual recall surface as
        # the follow-up any flip must wait for beyond English.
        all_pass = latency["latency_bar"]["pass"] and bars["verdict"] == "PASS"
        if all_pass:
            lines += [
                "", "**DEFAULT DECISION: all registered bars PASS.** The "
                "flip of the shipped default remains gated on the "
                "multilingual recall follow-up the spec names before any "
                "default flip is trusted beyond English."]
        else:
            missed = []
            if not latency["latency_bar"]["pass"]:
                missed.append(
                    f"latency bar MISS "
                    f"({latency['latency_bar']['ratio']} > "
                    f"{latency['latency_bar']['bound']}: the fixed "
                    "per-decision overhead dominates the 10-tool baseline)")
            if bars["verdict"] != "PASS":
                missed.append("quality bars MISS")
            lines += ["", "**DEFAULT DECISION: the prune ships OFF.** "
                      f"Recorded miss: {'; '.join(missed)}."]
    md_path = out_dir / f"guard-serve-performance-{stamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {md_path}")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warm-iters", type=int, default=WARM_ITERS)
    parser.add_argument("--bars", action="store_true",
                        help="also measure the pre-registered prune-recall "
                             "and decision-agreement bars on the canonical "
                             "corpora")
    parser.add_argument("--bars-only", action="store_true")
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports")
    args = parser.parse_args()

    fingerprint = _fingerprint()
    latency = None
    bars = None
    if not args.bars_only:
        latency = measure_latency(args.warm_iters, args.seed)
    if not args.latency_only:
        from mivia_small_json.tool_match import DEFAULT_PRUNE_K

        bars = measure_bars(DEFAULT_PRUNE_K)
    write_report(args.out, latency, bars, fingerprint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
