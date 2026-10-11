#!/usr/bin/env python3
"""Confident-learning label audit for the tool-match guard (CPU only, read-only).

Motivation (Northcutt et al., JAIR 2021, arXiv:1911.00068): typical benchmark
label error is 3-6%. The repo reads the guard's residual abstention on BFCL
near-misses as a representational ceiling (80.1% abstention at 95.6% retention
on 884 rows). At a ~70-80% ceiling, a few percent of mislabelled rows can
masquerade as a ceiling. This script audits the labels without training
anything.

What it scores
--------------
- ``pairs.jsonl`` (8421 pair rows: request, tool, label, source, split). The
  shipped guard was fine-tuned on ``split == "train"``; ``split == "dev"`` is
  the out-of-sample set. Confident learning needs out-of-sample probabilities,
  so the headline CL estimate is on ``dev``; the full file is also reported
  for reference (its ``train`` rows are in-sample and understate error).
- ``irrelevance_eval.jsonl`` (884 near-miss catalogues, ground truth: abstain)
  and ``calls.jsonl`` (183 real calls, ground truth: keep). These are the
  shipped end-to-end eval sets, request-level disjoint from ``pairs.jsonl``
  (tests/test_bfcl_pairs.py).

Method
------
The shipped guard is ``models/tool-match-int8.onnx`` (SelectiveNet L6). For a
pair, ``ToolMatcher._predict`` returns ``(match_prob, selection)``; the shipped
decision (``_score``) collapses to 0.0 when ``selection < 0.5``. For confident
learning the *raw* ``match_prob`` is the model probability (the gate destroys
the probability), so:

    p_given   = match_prob if label == 1 else 1 - match_prob
    p_alt     = 1 - p_given
    margin    = p_given - p_alt = (2*label - 1) * (2*match_prob - 1)

A row is a candidate mislabel when the model prefers the alternative
(``p_alt > 0.5``, ``margin < 0``). Counts are reported at several thresholds
on ``p_alt`` and ``margin``. The calibrated confident joint follows the binary
cleanlab construction: a single threshold ``t`` is set so the number of
predicted positives equals the observed positive count (the observed class
prior is unbiased for the true prior under the CL assumption); ``Q[1,0]`` and
``Q[0,1]`` then estimate the label errors in each class. cleanlab is not
installed, so this is implemented here; no dependency is added.

The two eval sets are single-class (all irrelevance, or all calls), so the
prior-preserving calibration is undefined; instead every false fire / dropped
call is dumped with its raw score so it can be read by hand, and the ceiling
impact is tabulated as a function of an assumed mislabel fraction.

The guard is loaded as a *single* artifact, exactly as
``scripts/bench_tool_abstention.py::_load`` does. ``load_matcher()`` would
auto-cascade to a gitignored tier-2 (``data/incoming/models/…modernbert…``)
and would not reproduce the shipped 80.1/95.6 numbers.

Read-only. Prints a report; writes no file.

Commands
--------
    python scripts/exp_label_audit.py                 # full report
    python scripts/exp_label_audit.py --limit 200     # smoke test
    python scripts/exp_label_audit.py --top 30        # more pair examples
    python scripts/exp_label_audit.py --json          # machine-readable blob
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.tool_match import (  # noqa: E402
    _SPLIT,
    MAX_LEN,
    THRESHOLD,
    ToolMatcher,
    parse_catalogue_raw,
    render_tool,
    validate_fillability,
)

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
PAIRS = BFCL / "pairs.jsonl"
IRRELEVANCE = BFCL / "irrelevance_eval.jsonl"
CALLS = BFCL / "calls.jsonl"
DEFAULT_MODEL = ROOT / "models" / "tool-match-int8.onnx"
DEFAULT_TOKENIZER = ROOT / "models" / "router-tokenizer.json"

P_ALT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
MARGIN_THRESHOLDS = (0.0, -0.2, -0.5, -0.8)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _load_single(path: Path, tokenizer: Path, max_len: int) -> ToolMatcher:
    """The shipped measurement path: one artifact, CPU, no cascade."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = min(4, __import__("os").cpu_count() or 1)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(tokenizer))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=None)
    return ToolMatcher(session, tok, [i.name for i in session.get_inputs()])


def _decide(matcher: ToolMatcher, request: str, tool: str) -> tuple[float, float]:
    """(raw match probability, shipped gated score) from one forward pass.

    Mirrors ``ToolMatcher._score``: the SelectiveNet gate collapses the score
    to 0.0 when ``selection < 0.5``.
    """
    prob, sel = matcher._predict(request, tool)
    gated = 0.0 if (sel is not None and sel < 0.50) else prob
    return prob, gated


def _best_over_tools(matcher: ToolMatcher, text: str):
    """Best (tool, raw prob, gated score) over a catalogue's listed tools."""
    tools, request = parse_catalogue_raw(text)
    if tools is None or not request:
        return None, 1.0, 1.0, ""
    if not tools:
        return None, (0.0 if _SPLIT in text else 1.0), \
               (0.0 if _SPLIT in text else 1.0), request
    best_gated, best_raw, best_tool = -1.0, -1.0, None
    for tool in tools:
        span = render_tool(tool)
        prob, gated = _decide(matcher, request, span)
        if gated > best_gated:
            best_gated, best_tool = gated, tool
        if prob > best_raw:
            best_raw = prob
    return best_tool, max(best_raw, 0.0), max(best_gated, 0.0), request


def _cl_columns(probs: np.ndarray, labels: np.ndarray):
    """(p_given, p_alt, margin) for the confident-learning criteria."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    p_given = np.where(labels == 1, probs, 1.0 - probs)
    p_alt = 1.0 - p_given
    margin = p_given - p_alt
    return p_given, p_alt, margin


def _confident_joint(probs: np.ndarray, labels: np.ndarray):
    """Binary confident joint with prior-preserving calibration.

    Threshold ``t`` is the observed positive rate's quantile of ``probs`` so
    that ``#{p >= t} == #{label == 1}``; the observed class prior is treated as
    an unbiased estimate of the true prior (the CL assumption). Returns
    ``(t, Q)`` with ``Q[a, b]`` = count of rows labelled ``a`` predicted ``b``,
    and ``n_est_errors = Q[1,0] + Q[0,1]``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    n1 = int((labels == 1).sum())
    if n1 == 0 or n1 == n:
        t = 0.5  # degenerate prior: nothing to calibrate against
    else:
        t = float(np.sort(probs)[::-1][n1 - 1])
    pred = probs >= t
    Q = np.zeros((2, 2), dtype=np.int64)
    for a in (0, 1):
        for b in (0, 1):
            Q[a, b] = int(np.sum((labels == a) & (pred == b)))
    return t, Q


def _pct(a: int, b: int) -> float:
    return 100.0 * a / b if b else 0.0


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def audit_pairs(matcher: ToolMatcher, rows: list[dict], top: int, out: dict) -> None:
    print("\n" + "=" * 78)
    print("PART 1  pairs.jsonl  (pair-level confident-learning audit)")
    print("=" * 78)
    scored = []
    for r in rows:
        prob, gated = _decide(matcher, r["request"], r["tool"])
        scored.append({**r, "p": prob, "gated": gated})

    by_split: dict[str, list[dict]] = {}
    for r in scored:
        by_split.setdefault(r["split"], []).append(r)

    out["pairs"] = {}
    for split in sorted(by_split):
        sub = by_split[split]
        probs = np.array([r["p"] for r in sub])
        labels = np.array([r["label"] for r in sub])
        p_given, p_alt, margin = _cl_columns(probs, labels)
        t, Q = _confident_joint(probs, labels)
        n_est = int(Q[1, 0] + Q[0, 1])
        off = int(np.sum((probs >= 0.5).astype(int) != labels))
        print(f"\n-- split={split!r}  n={len(sub)}  "
              f"(pos={int((labels == 1).sum())}, neg={int((labels == 0).sum())})")
        print(f"   calibrated threshold t={t:.4f}  "
              f"confident joint [[{Q[0,0]} {Q[0,1]}],[{Q[1,0]} {Q[1,1]}]]")
        print(f"   CL-estimated label errors = {n_est} "
              f"({_pct(n_est, len(sub)):.2f}%): "
              f"{Q[1,0]} labelled-1 predicted-0, {Q[0,1]} labelled-0 predicted-1")
        print(f"   'predicted != given' at 0.5 (off-diagonal) = {off} "
              f"({_pct(off, len(sub)):.2f}%)")
        print("   p_alt threshold sweep (candidate mislabels, most confident first):")
        sweep = {}
        for th in P_ALT_THRESHOLDS:
            sel = p_alt >= th
            k = int(sel.sum())
            k_pos = int(np.sum(sel & (labels == 1)))
            k_neg = int(np.sum(sel & (labels == 0)))
            sweep[str(th)] = {"n": k, "of_positives": k_pos,
                              "of_negatives": k_neg,
                              "pct": round(_pct(k, len(sub)), 3)}
            print(f"     p_alt >= {th:<5} : {k:5d} ({_pct(k, len(sub)):5.2f}%)"
                  f"  [labelled-1: {k_pos}, labelled-0: {k_neg}]")
        print("   margin threshold sweep (margin = p_given - p_alt):")
        for th in MARGIN_THRESHOLDS:
            k = int((margin < th).sum())
            print(f"     margin < {th:<5} : {k:5d} ({_pct(k, len(sub)):5.2f}%)")
        out["pairs"][split] = {
            "n": len(sub), "pos": int((labels == 1).sum()),
            "neg": int((labels == 0).sum()),
            "calibrated_t": round(t, 4),
            "confident_joint": Q.tolist(),
            "cl_est_errors": n_est, "cl_est_errors_pct": round(_pct(n_est, len(sub)), 3),
            "offdiag_0p5": off, "offdiag_0p5_pct": round(_pct(off, len(sub)), 3),
            "p_alt_sweep": sweep,
            "margin_sweep": {str(th): int((margin < th).sum())
                             for th in MARGIN_THRESHOLDS},
        }

    # Examples from the out-of-sample split (fall back to all rows).
    ref = by_split.get("dev") or scored

    def _p_alt(r: dict) -> float:
        p = r["p"] if r["label"] == 1 else 1.0 - r["p"]
        return 1.0 - p

    order = sorted(range(len(ref)), key=lambda i: -_p_alt(ref[i]))
    print(f"\n-- top {top} most-likely-mislabelled pair rows "
          f"(split={'dev' if 'dev' in by_split else 'all'}):")
    examples = []
    for rank, i in enumerate(order[:top], 1):
        r = ref[i]
        p_alt = _p_alt(r)
        print(f"\n   [{rank}] label={r['label']}  p(match)={r['p']:.4f}  "
              f"p_alt={p_alt:.4f}  source={r['source']}  split={r['split']}")
        print(f"       request: {_clip(r['request'], 400)}")
        print(f"       tool:    {_clip(r['tool'], 400)}")
        examples.append({"rank": rank, "label": r["label"], "p_match": round(r["p"], 4),
                         "p_alt": round(p_alt, 4), "source": r["source"],
                         "split": r["split"], "request": r["request"],
                         "tool": r["tool"]})
    out["pairs_top_examples"] = examples


def audit_eval_set(matcher: ToolMatcher, path: Path, kind: str, out: dict,
                   full_tools: bool = False) -> None:
    """kind: 'irrelevance' (truth=abstain) or 'calls' (truth=keep)."""
    rows = _rows(path)
    print("\n" + "=" * 78)
    print(f"PART {'2' if kind == 'irrelevance' else '3'}  {path.name}  "
          f"(n={len(rows)}, truth={'abstain' if kind == 'irrelevance' else 'keep'})")
    print("=" * 78)
    recs = []
    for r in rows:
        tool, raw, gated, request = _best_over_tools(matcher, r["text"])
        fillable, reason, missing = (validate_fillability(tool, request)
                                     if tool else (True, "", []))
        recs.append({"text": r["text"], "request": request, "tool": tool,
                     "raw": raw, "gated": gated, "source": r.get("source"),
                     "fillable": fillable, "reason": reason, "missing": missing})

    gated = np.array([r["gated"] for r in recs])
    if kind == "irrelevance":
        errors = [r for r in recs if r["gated"] >= THRESHOLD]  # false fires
        good = [r for r in recs if r["gated"] < THRESHOLD]
        label = "false fires (guard FAILS to abstain)"
    else:
        errors = [r for r in recs if r["gated"] < THRESHOLD]   # dropped calls
        good = [r for r in recs if r["gated"] >= THRESHOLD]
        label = "dropped calls (guard FAILS to keep)"
    print(f"\n   {label}: {len(errors)}/{len(recs)}  "
          f"({'abstention' if kind == 'irrelevance' else 'retention'} "
          f"{_pct(len(good), len(recs)):.1f}% at threshold {THRESHOLD})")
    raw_scores = np.array([r["raw"] for r in recs])
    print(f"   raw match prob: min {raw_scores.min():.4f}  "
          f"p50 {np.median(raw_scores):.4f}  p90 {np.percentile(raw_scores, 90):.4f}  "
          f"max {raw_scores.max():.4f}")
    for th in (0.9, 0.99):
        k = int((raw_scores >= th).sum())
        print(f"   raw match prob >= {th}: {k} rows ({_pct(k, len(recs)):.2f}%)")
    n_unfillable = sum(1 for r in errors if not r["fillable"])
    print(f"   of the {len(errors)} errors, {n_unfillable} fail the deterministic "
          f"fillability gate (required arg absent -> unambiguously not callable)")

    errors.sort(key=lambda r: -r["raw"])
    print(f"\n   --- every {label}, sorted by raw match prob ---")
    for i, r in enumerate(errors, 1):
        flag = "fillable" if r["fillable"] else f"unfillable:{r['reason']}"
        print(f"\n   [{i}] raw={r['raw']:.4f} gated={r['gated']:.4f} "
              f"source={r['source']} {flag}")
        print(f"       request: {_clip(r['request'], 500)}")
        print(f"       best tool: {_clip(render_tool(r['tool']) if r['tool'] else '(none)', 500)}")
        if full_tools and r["tool"]:
            print(f"       full tool JSON: {json.dumps(r['tool'], ensure_ascii=False)}")

    out[path.stem] = {
        "n": len(recs), "n_errors": len(errors),
        "rate_pct": round(_pct(len(good), len(recs)), 2),
        "n_unfillable_errors": n_unfillable,
        "raw_p90": round(float(np.percentile(raw_scores, 90)), 4),
        "raw_ge_0p9": int((raw_scores >= 0.9).sum()),
        "raw_ge_0p99": int((raw_scores >= 0.99).sum()),
        "errors": [{"rank": i, "raw": round(r["raw"], 4),
                    "gated": round(r["gated"], 4), "source": r["source"],
                    "fillable": r["fillable"], "missing": r["missing"],
                    "request": r["request"],
                    "tool": render_tool(r["tool"]) if r["tool"] else None,
                    "tool_json": r["tool"]}
                   for i, r in enumerate(errors, 1)],
    }


def ceiling_table(abstained: int, n: int, out: dict) -> None:
    print("\n" + "=" * 78)
    print("PART 4  ceiling impact: corrected abstention vs assumed mislabel fraction")
    print("=" * 78)
    print(f"\n   shipped: abstained {abstained}/{n} = {_pct(abstained, n):.1f}%")
    print("   if m of the n rows are actually legitimate calls (mislabelled as")
    print("   irrelevance) and the guard correctly fires on them, the true")
    print("   abstention over the genuinely-irrelevant rows is abstained/(n-m):")
    print(f"\n   {'assumed mislabel %':>20} {'m':>5} {'true abstention %':>18}")
    table = []
    for x in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0):
        m = int(round(x / 100.0 * n))
        true_abst = _pct(abstained, n - m) if n - m > 0 else float("nan")
        table.append({"mislabel_pct": x, "m": m, "true_abstention_pct": round(true_abst, 2)})
        print(f"   {x:>19.1f}% {m:>5d} {true_abst:>17.1f}%")
    out["ceiling_table"] = table


def env_fingerprint(model: Path, out: dict) -> None:
    import onnxruntime
    import tokenizers
    try:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"
    sidecar = model.parent / (model.name + ".sha256")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    try:
        rel = str(model.resolve().relative_to(ROOT))
    except ValueError:
        rel = str(model)
    out["env"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "onnxruntime": onnxruntime.__version__,
        "tokenizers": tokenizers.__version__,
        "git_sha": sha,
        "artifact": rel,
        "artifact_sha256": digest,
        "artifact_sidecar": sidecar.read_text().strip() if sidecar.is_file() else None,
        "sidecar_matches": sidecar.is_file() and sidecar.read_text().strip() == digest,
        "max_len": MAX_LEN,
        "threshold": THRESHOLD,
    }
    print("=" * 78)
    print("ENV")
    print("=" * 78)
    for k, v in out["env"].items():
        print(f"   {k}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--pairs", type=Path, default=PAIRS)
    ap.add_argument("--irrelevance", type=Path, default=IRRELEVANCE)
    ap.add_argument("--calls", type=Path, default=CALLS)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--full-tools", action="store_true",
                    help="print the full parameter JSON of each error's best tool")
    ap.add_argument("--limit", type=int, default=0,
                    help="score at most N rows per set (smoke tests)")
    ap.add_argument("--json", action="store_true",
                    help="also emit a JSON blob on stdout (fenced)")
    args = ap.parse_args()

    out: dict = {}
    env_fingerprint(args.model, out)

    matcher = _load_single(args.model, args.tokenizer, args.max_len)
    pairs = _rows(args.pairs)
    irrelevance = _rows(args.irrelevance)
    calls = _rows(args.calls)
    if args.limit:
        pairs = pairs[: args.limit]
        irrelevance = irrelevance[: args.limit]
        calls = calls[: args.limit]

    audit_pairs(matcher, pairs, args.top, out)

    audit_eval_set(matcher, args.irrelevance, "irrelevance", out, args.full_tools)
    n_irr = out["irrelevance_eval"]["n"]
    abstained = n_irr - out["irrelevance_eval"]["n_errors"]
    ceiling_table(abstained, n_irr, out)

    audit_eval_set(matcher, args.calls, "calls", out, args.full_tools)
    n_calls = out["calls"]["n"]
    kept = n_calls - out["calls"]["n_errors"]
    print(f"\n   retention: kept {kept}/{n_calls} = {_pct(kept, n_calls):.1f}%")
    print("   if d of the calls are actually irrelevant (mislabelled as calls)")
    print("   and the guard correctly drops them, true retention = kept/(n-d):")
    print(f"\n   {'assumed mislabel %':>20} {'d':>5} {'true retention %':>18}")
    tbl = []
    for x in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0):
        d = int(round(x / 100.0 * n_calls))
        true_ret = _pct(kept, n_calls - d) if n_calls - d > 0 else float("nan")
        tbl.append({"mislabel_pct": x, "d": d, "true_retention_pct": round(true_ret, 2)})
        print(f"   {x:>19.1f}% {d:>5d} {true_ret:>17.1f}%")
    out["retention_table"] = tbl

    if args.json:
        print("\n===JSON-BEGIN===")
        print(json.dumps(out, indent=1))
        print("===JSON-END===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
