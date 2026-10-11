# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Benchmark rules-only vs full (model-routed) pipeline on the FROZEN
held-out split, and write a dated bench report with the six metrics
plus an environment fingerprint.

Two validity readings, reported side by side (never conflated):

- ``valid-json %`` (weak): the emitted object passes the strict
  validator for the schema it was routed to. A template escape does
  not count — that object is valid by construction, not earned.
- ``routed-and-valid %`` (strict): that AND the pipeline routed to the
  record's true schema. Routing quality is separately reported as F1
  macro/per class. The >= 99% stop-rule verdict uses the STRICT one.

The report is dated and written under ``reports/bench/``; the frozen
``reports/phase1.md`` shipped with v0 is never overwritten.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata as im
import json

from .registry import spec_fingerprint
import subprocess
import time
from pathlib import Path

from sklearn.metrics import f1_score

from mivia_small_json import bootstrap, model, rules
from mivia_small_json.pipeline import load_active_router
from mivia_small_json.validate import SCHEMAS, validate

ROOT = Path(__file__).resolve().parents[2]
CYCLES_USED = 1  # iteration cycles spent before this report (max 2, kill rule)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _run(rows: list[dict], router) -> dict:
    latencies, n_valid, n_strict, retries, fallbacks = [], 0, 0, 0, 0
    gold, pred = [], []
    for row in rows:
        t0 = time.perf_counter()
        obj, attempts, schema, escaped = rules.emit(row["raw"], router)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        retries += attempts
        fallbacks += 1 if escaped else 0
        gold.append(row["schema"])
        pred.append(schema)
        if not escaped and not validate(obj, schema):
            n_valid += 1
            if schema == row["schema"]:
                n_strict += 1
    n = len(rows)
    ls = sorted(latencies)

    def q(p: float) -> float:
        return ls[min(n - 1, int(p * (n - 1)))]
    per = f1_score(gold, pred, average=None, labels=list(SCHEMAS))
    return {
        "n": n,
        "valid_pct": 100.0 * n_valid / n,
        "routed_and_valid_pct": 100.0 * n_strict / n,
        "p50_ms": q(0.50),
        "p95_ms": q(0.95),
        "retries_per_example": retries / n,
        "fallbacks": fallbacks,
        "f1_macro": float(f1_score(gold, pred, average="macro")),
        "f1_per_class": {s: float(f) for s, f in zip(SCHEMAS, per)},
    }


def _fingerprint() -> list[str]:
    import platform

    def v(name):
        try:
            return im.version(name)
        except Exception:
            return "?"
    lines = [
        f"python {platform.python_version()}",
        f"scikit-learn {v('scikit-learn')}", f"numpy {v('numpy')}",
        f"skops {v('skops')}", f"scipy {v('scipy')}", f"joblib {v('joblib')}",
    ]
    # The ONNX runtime deps: recorded only when present, so the skops-only
    # fingerprint stays honest about what actually ran.
    for name in ("onnxruntime", "tokenizers"):
        try:
            lines.append(f"{name} {im.version(name)}")
        except Exception:
            pass
    return lines


def _backend_of(router) -> str:
    """Name the backend that actually loaded, from the object itself.

    File existence is not evidence: the ONNX artifact can sit on disk
    while its sha-verified load failed and the skops router (or the rules
    layer) answered the queries. The report must attribute its numbers to
    what actually served, never to a file that was merely present (V2,
    2026-10-10: the note claimed the int8 ONNX encoder while skops ran).
    """
    if router is None:
        return "rules"
    from .encoder import EncoderRouter

    return "encoder" if isinstance(router, EncoderRouter) else "skops"


def _as_backend(value) -> str:
    """Accept the historical True/False flags as backend names."""
    if value is True:
        return "encoder"
    if value is False:
        return "skops"
    return str(value)


def _artifact_attribution(encoder_active, size: int) -> str:
    """The report's model-artifact line, naming what was measured.

    ``encoder_active`` takes a backend name from :func:`_backend_of`
    (``"encoder"`` / ``"skops"`` / ``"rules"``); the historical booleans
    are still accepted.
    """
    backend = _as_backend(encoder_active)
    if backend == "encoder":
        return (f"- model artifact: models/router-encoder-int8.onnx + "
                f"models/router-tokenizer.json ({size} bytes total, "
                f"sha256-verified before load)")
    if backend == "skops":
        return f"- model artifact: models/router.skops ({size} bytes) + sha256 sidecar"
    return ("- model artifact: none (rules-only routing — no model artifact "
            "loaded, so the full-pipeline column measured the rules layer)")


def _method_note(encoder_active) -> str:
    """The report's full-pipeline method note, matching the router used.

    ``encoder_active`` takes a backend name from :func:`_backend_of`; the
    historical booleans are still accepted.
    """
    backend = _as_backend(encoder_active)
    if backend == "encoder":
        return ("- Full pipeline: same extraction/repair; routing by the "
                "fine-tuned int8 ONNX mMiniLM-L12 encoder router "
                "(sha-verified load), skops fallback.")
    if backend == "skops":
        return ("- Full pipeline: same extraction/repair; routing by TF-IDF (word+char "
                "n-grams) -> LogisticRegression, skops artifact loaded through the "
                "sha-verifying pipeline loader.")
    return ("- Full pipeline: same extraction/repair; routing by the rules "
            "router only — no model artifact could be loaded, so this is a "
            "rules-only measurement, not a model comparison.")


def main() -> dict:
    bootstrap.register_example_schemas()
    rows = [json.loads(line)
            for line in (ROOT / "data" / "heldout" / "test.jsonl").read_text(
                encoding="utf-8").splitlines() if line]
    pipe = load_active_router()  # shipped router: ONNX encoder, else skops
    # Attribute every number to the backend that actually loaded — derived
    # from the loaded object, never from file existence (a failed ONNX load
    # that fell back to skops must not be reported as the encoder).
    backend = _backend_of(pipe)
    full_router = (lambda t: str(pipe.predict([t])[0])) if pipe is not None else rules.route
    results = {"rules": _run(rows, rules.route), "full": _run(rows, full_router)}
    # Report the size of the artifact actually serving: the encoder when it
    # loaded, otherwise the skops router, otherwise nothing.
    from . import encoder as _enc

    if backend == "encoder":
        size = _enc.DEFAULT_MODEL.stat().st_size + _enc.DEFAULT_TOKENIZER.stat().st_size
    elif backend == "skops":
        size = model.MODEL_PATH.stat().st_size
    else:
        size = 0
    r, f = results["rules"], results["full"]
    verdict = (
        f"STOP RULE MET — full pipeline reached {f['routed_and_valid_pct']:.2f}% "
        f"routed-and-valid JSON on the frozen held-out set (>= 99%). Ship v0: "
        f"no further iteration, no fine-tuning (stop rule)."
        if f["routed_and_valid_pct"] >= 99.0 else
        f"STOP RULE NOT MET — {f['routed_and_valid_pct']:.2f}% routed-and-valid "
        f"< 99% after {CYCLES_USED} of max 2 iteration cycles. Kill-rule "
        f"bookkeeping applies; no metric hacking."
    )
    manifest = json.loads((ROOT / "data" / "heldout" / "manifest.json").read_text())
    sha = _git_sha()
    lines = [
        "# Bench report — mivia-small-json (frozen held-out split)",
        "",
        f"Date: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"· Seed: 42 · Held-out n: {r['n']} (frozen at generation, never trained on)",
        "",
        "## Stop-rule verdict (strict: routed-and-valid)",
        "",
        verdict,
        "",
        "## Six metrics — rules baseline vs full pipeline",
        "",
        "| Metric | rules baseline | full pipeline | delta (full - rules) |",
        "|---|---|---|---|",
        f"| valid JSON % (schema-valid) | {r['valid_pct']:.2f} | {f['valid_pct']:.2f} | "
        f"{f['valid_pct'] - r['valid_pct']:+.2f} pp |",
        f"| routed-and-valid % (strict) | {r['routed_and_valid_pct']:.2f} | "
        f"{f['routed_and_valid_pct']:.2f} | "
        f"{f['routed_and_valid_pct'] - r['routed_and_valid_pct']:+.2f} pp |",
        f"| latency p50 (ms) | {r['p50_ms']:.3f} | {f['p50_ms']:.3f} | "
        f"{f['p50_ms'] - r['p50_ms']:+.3f} |",
        f"| latency p95 (ms) | {r['p95_ms']:.3f} | {f['p95_ms']:.3f} | "
        f"{f['p95_ms'] - r['p95_ms']:+.3f} |",
        f"| retries / example | {r['retries_per_example']:.4f} | "
        f"{f['retries_per_example']:.4f} | "
        f"{f['retries_per_example'] - r['retries_per_example']:+.4f} |",
        f"| F1 macro (routing) | {r['f1_macro']:.4f} | {f['f1_macro']:.4f} | "
        f"{100 * (f['f1_macro'] - r['f1_macro']):+.2f} pp |",
        f"| model size (bytes) | 0 (no model) | {size} | {size:+d} |",
        "",
        "F1 per class (routing):",
        "",
        "| Schema | rules F1 | full F1 |",
        "|---|---|---|",
        *[f"| {s} | {r['f1_per_class'][s]:.4f} | {f['f1_per_class'][s]:.4f} |"
          for s in SCHEMAS],
        "",
        f"Template-fallback emissions (last resort): rules={r['fallbacks']}, "
        f"full={f['fallbacks']}.",
        "",
        "## Method notes",
        "",
        "- valid JSON % (weak) = emitted object passes the strict validator for "
        "the routed schema, without the template escape (repair allowed, "
        "<= 2 attempts).",
        "- routed-and-valid % (strict) = that AND routed to the record's true "
        "schema; this is the stop-rule gate metric.",
        "- Template-fallback emissions = rows emitted from the minimal valid "
        "template (routed_by == \"template\" on the classify path).",
        "- Latency: time.perf_counter per example, CPU-only, includes routing + "
        "extraction + validation + repair.",
        "- Rules baseline: keyword routing + regex extraction (rules.py); no "
        "learned state, no access to annotations.",
        _method_note(backend),
        "- Retries = repair attempts (0-2); template escapes reported separately.",
        "",
        "## Environment fingerprint",
        "",
        f"- {' · '.join(_fingerprint())}",
        f"- git SHA {sha}",
        f"- corpus: test.jsonl sha256 {manifest['test_sha256'][:12]}... "
        f"- spec fingerprint: {spec_fingerprint()}",
        f"({manifest['records']['test']} records, seed {manifest['seed']})",
        _artifact_attribution(backend, size),
        "",
    ]
    out_dir = ROOT / "reports" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = out_dir / f"bench-{stamp}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(verdict)
    print(f"rules: valid={r['valid_pct']:.2f}% routed-and-valid="
          f"{r['routed_and_valid_pct']:.2f}% f1={r['f1_macro']:.4f} "
          f"p95={r['p95_ms']:.2f}ms | full: valid={f['valid_pct']:.2f}% "
          f"routed-and-valid={f['routed_and_valid_pct']:.2f}% "
          f"f1={f['f1_macro']:.4f} p95={f['p95_ms']:.2f}ms model={size}B")
    print(f"report written to {report.relative_to(ROOT)}")
    return results


if __name__ == "__main__":
    main()
