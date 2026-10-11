#!/usr/bin/env python3
"""Span-redesign arm for the tool-match guard.

The measured cause (reports/exp-label-audit.md): on the 884 BFCL
`live_irrelevance` rows the labels are clean, and BFCL builds them by (a)
stripping the arguments the tool needs or (b) pairing the request with a
sibling tool. The shipped guard reads ``name: description``
(``tool_match.render_tool``), where neither the missing argument nor the
competing sibling is visible. The model is asked to refuse on evidence it
cannot see.

This script re-renders the guard's training corpus with a different tool span
and trains/measures the L6 cross-encoder on it. It edits no source file: it
imports ``build_tool_match_corpus`` for the BFCL fetch/assembly helpers and
``train_tool_match.py`` (as a subprocess) for the trainer.

Arms
----
- ``v1``  shipped span: ``name: description``.
- ``v2``  control (the falsified parameter-aware span, ``render_tool(...,
          parameters=True)``): name + full parameter schema.
- ``v3``  fillability-aware span (idea ii): name + required parameters +
          the argument values the request supplies + the missing set. Uses
          ``span_fill.fill_arguments`` and ``tool_match.validate_fillability``.
- ``v4``  catalogue-aware span (idea iii): v3 plus a short list of the other
          tools in the same catalogue. NOT single-pair invariant; measured as
          a probe with the invariance cost stated.

The corpus stores pre-rendered tool strings, so a new span means re-rendering
from the BFCL sources. ``build_rows`` reproduces ``build()``'s assembly
verbatim (same rows, labels, request-level filter and 80/20 split) and only
swaps the tool string; ``verify-v1`` checks the reproduction against the
shipped ``pairs.jsonl``.

Writes only under ``data/incoming/models/`` (artifacts) and ``/tmp`` (corpora,
training logs). Never touches ``models/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json.tool_match import (  # noqa: E402
    _SPLIT,
    ToolMatcher,
    parse_catalogue_raw,
    render_tool,
    validate_fillability,
)
from mivia_small_json.span_fill import fill_arguments  # noqa: E402

import build_tool_match_corpus as btc  # noqa: E402

BFCL = btc.BFCL
SHIPPED_PAIRS = btc.OUT
MODELS = ROOT / "data" / "incoming" / "models"
SIBLING_DESC_CHARS = 70
SIBLING_MAX = 5


# ---------------------------------------------------------------------------
# Spans.  Signature: (tool_dict, request, catalogue) -> str.
# ---------------------------------------------------------------------------
def _bare(tool: dict) -> dict:
    if isinstance(tool.get("function"), dict):
        return tool["function"]
    return tool


def _schema(tool: dict) -> tuple[dict, list[str]]:
    params = _bare(tool).get("parameters")
    params = params if isinstance(params, dict) else {}
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = list(required) if isinstance(required, list) else []
    return props, required


def _ptype(props: dict, name: str) -> str:
    spec = props.get(name)
    return str(spec.get("type", "any")) if isinstance(spec, dict) else "any"


def span_v1(tool: dict, request: str, catalogue=None) -> str:
    return render_tool(tool)


def span_v2(tool: dict, request: str, catalogue=None) -> str:
    return render_tool(tool, parameters=True)


def span_v3(tool: dict, request: str, catalogue=None) -> str:
    """Fillability-aware span (idea ii).

    ``requires:`` lists the required parameters with their schema types.
    ``provides:`` is the value set ``span_fill.fill_arguments`` selected from
    the request (numeric family only). ``missing:`` is
    ``tool_match.validate_fillability``'s missing set, so a required argument
    the request does not supply becomes visible text.
    """
    base = render_tool(tool)
    props, required = _schema(tool)
    if not required:
        return base
    parts = [base, "requires: "
             + ", ".join(f"{p} ({_ptype(props, p)})" for p in required)]
    filled, missing_fill = fill_arguments(tool, request)
    _ok, _reason, missing = validate_fillability(tool, request)
    if filled is not None:
        values = ", ".join(f"{k}={v}" for k, v in filled.items()) or "(none)"
        parts.append(f"provides: {values}")
        parts.append(f"missing: {', '.join(missing_fill) or '(none)'}")
    elif missing:
        parts.append(f"missing: {', '.join(missing)}")
    return " | ".join(parts)


def span_v4(tool: dict, request: str, catalogue=None) -> str:
    """Catalogue-aware span (idea iii): v3 plus the listed siblings.

    The pair's score now depends on what else the catalogue holds, which
    breaks the repo's deliberate single-pair scoring. Measured as a probe.
    """
    base = span_v3(tool, request)
    if not catalogue:
        return base
    mine = _bare(tool).get("name", "")
    siblings = []
    for other in catalogue:
        name = _bare(other).get("name", "")
        if not name or name == mine:
            continue
        desc = str(_bare(other).get("description", "")).strip()
        desc = desc[:SIBLING_DESC_CHARS]
        siblings.append(f"{name}: {desc}" if desc else name)
        if len(siblings) >= SIBLING_MAX:
            break
    if not siblings:
        return base
    return base + " | also listed: " + "; ".join(siblings)


SPANS = {"v1": span_v1, "v2": span_v2, "v3": span_v3, "v4": span_v4}


# ---------------------------------------------------------------------------
# Corpus re-render.  Reproduces build_tool_match_corpus.build() verbatim,
# swapping only the rendered tool string.
# ---------------------------------------------------------------------------
def build_rows(span_name: str) -> tuple[list[dict], dict]:
    span_fn = SPANS[span_name]
    questions, answers = btc._sources()
    blocked = btc._eval_requests()
    stats = {
        "bfcl_positives": 0, "bfcl_distractor_negatives": 0,
        "bfcl_distractor_aligned": 0, "bfcl_foreign_negatives": 0,
        "bfcl_irrelevance_negatives": 0, "dropped_eval_request": 0,
        "dropped_duplicate": 0,
    }
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []

    def add(request: str, tool: dict, label: int, counter: str,
            catalogue: list[dict], split: str | None = None) -> None:
        request = btc._clean(request)
        if not request:
            return
        if request in blocked:
            stats["dropped_eval_request"] += 1
            return
        # Dedup and split key off the v1 span so the row set and 80/20 split
        # are identical to the shipped corpus; only the stored tool string
        # changes.
        v1_span = btc._clean(render_tool(tool))
        key = (request, v1_span)
        if key in seen:
            stats["dropped_duplicate"] += 1
            return
        seen.add(key)
        tool_span = btc._clean(span_fn(tool, request, catalogue))
        if not tool_span:
            return
        rows.append({"request": request, "tool": tool_span, "label": label,
                     "source": counter,
                     "split": split or btc._split_of(request, v1_span)})
        stats[counter] += 1

    matched: list[tuple[str, dict, list[dict]]] = []
    for rec in questions:
        request = rec["question"][0][0]["content"]
        called = {next(iter(g)) for g in answers.get(rec["id"], []) if g}
        gold_signatures = {btc._signature(t) for t in rec["function"]
                           if t["name"] in called}
        gold_signatures.discard(())
        catalogue = rec["function"]
        for tool in catalogue:
            if tool["name"] in called:
                matched.append((request, tool, catalogue))
            elif btc._signature(tool) and btc._signature(tool) in gold_signatures:
                add(request, tool, 0, "bfcl_distractor_aligned", catalogue)
            else:
                add(request, tool, 0, "bfcl_distractor_negatives", catalogue)

    span = len(matched) // 2
    for i, (request, tool, catalogue) in enumerate(matched):
        add(request, tool, 1, "bfcl_positives", catalogue)
        for step in range(1, 5):
            foreign = matched[(i + span * step) % len(matched)][1]
            if foreign["name"].split(":")[0] != tool["name"].split(":")[0]:
                add(request, foreign, 0, "bfcl_foreign_negatives", catalogue)
                break

    irr = BFCL / "irrelevance.jsonl"
    if irr.is_file():
        for line in irr.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            tools, request = parse_catalogue_raw(json.loads(line)["text"])
            for tool in tools or []:
                add(request, tool, 0, "bfcl_irrelevance_negatives", tools or [])
    return rows, stats


def _write_rows(rows: list[dict], out: Path) -> str:
    import hashlib
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def cmd_build(args) -> int:
    rows, stats = build_rows(args.span)
    digest = _write_rows(rows, args.out)
    for name, value in stats.items():
        print(f"{name}: {value}")
    print(f"total rows: {len(rows)}  sha256: {digest}")
    print(f"written to {args.out}")
    return 0


def cmd_verify_v1(args) -> int:
    """The re-render with the v1 span must equal the shipped pairs.jsonl."""
    rows, _stats = build_rows("v1")
    mine = {json.dumps(r, sort_keys=True) for r in rows}
    theirs = {json.dumps(json.loads(line), sort_keys=True)
              for line in SHIPPED_PAIRS.read_text(encoding="utf-8").split("\n")
              if line.strip()}
    only_mine = mine - theirs
    only_theirs = theirs - mine
    print(f"re-rendered v1 rows: {len(mine)}  shipped rows: {len(theirs)}")
    print(f"only in re-render: {len(only_mine)}  only in shipped: "
          f"{len(only_theirs)}")
    if only_mine:
        print("example only in re-render:", next(iter(only_mine))[:300])
    if only_theirs:
        print("example only in shipped:", next(iter(only_theirs))[:300])
    ok = not only_mine and not only_theirs
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Measurement.  Runs scripts/bench_tool_abstention.py verbatim, with _load
# wrapped so the production scorer renders the arm's span (the stock bench
# only knows v1/v2 and only the tool, not the request/catalogue).
# ---------------------------------------------------------------------------
class SpanMatcher(ToolMatcher):
    """ToolMatcher whose best_match renders the arm's span.

    Mirrors ``BaseToolMatcher.best_match`` exactly, including the
    catalogue-level ``max`` decision and the not-a-catalogue short circuits,
    and differs only in the span passed to ``_score``.
    """

    def __init__(self, base: ToolMatcher, span_fn):
        self.sess = base.sess
        self.tok = base.tok
        self.inputs = base.inputs
        self._span_fn = span_fn

    def best_match(self, text: str):
        tools, request = parse_catalogue_raw(text)
        if tools is None or not request:
            return None, 1.0
        if not tools:
            return None, 0.0 if _SPLIT in text else 1.0
        best_tool = None
        best_score = -1.0
        for tool in tools:
            score = self._score(request, self._span_fn(tool, request, tools))
            if score > best_score:
                best_score = score
                best_tool = tool
        return best_tool, max(best_score, 0.0)


def _bench_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bench_tool_abstention", ROOT / "scripts" / "bench_tool_abstention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_bench(model: Path, span_name: str, max_len: int = 256,
              label: str = "") -> dict:
    bench = _bench_module()
    orig_load = bench._load
    span_fn = SPANS[span_name]

    def patched_load(path, tokenizer, ml):
        return SpanMatcher(orig_load(path, tokenizer, ml), span_fn)

    bench._load = patched_load
    argv = ["bench_tool_abstention.py", "--model", str(model),
            "--max-len", str(max_len)]
    if label:
        argv += ["--label", label]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = bench.main()
    finally:
        sys.argv = old_argv
    return {"returncode": rc}


def frontier(model: Path, span_name: str, max_len: int = 256) -> dict:
    """Fine threshold sweep on the same decision the bench makes."""
    bench = _bench_module()
    matcher = SpanMatcher(bench._load(model, None, max_len), SPANS[span_name])
    fp_scores = [matcher.best_score(t) for t in bench._texts(bench.FP_EVAL)]
    call_scores = [matcher.best_score(t) for t in bench._texts(bench.CALLS)]
    grid = [round(x / 100, 2) for x in range(0, 101)]
    curve = []
    for thr in grid:
        abst = 100 * sum(1 for s in fp_scores if s < thr) / len(fp_scores)
        kept = 100 * sum(1 for s in call_scores if s >= thr) / len(call_scores)
        curve.append((thr, abst, kept))
    at60 = next(c for c in curve if c[0] == 0.60)
    # max abstention whose retention stays >= 95
    keep95 = [c for c in curve if c[2] >= 95.0]
    best95 = max(keep95, key=lambda c: c[1]) if keep95 else None
    # retention at the first threshold reaching >= 80 abstention
    reach80 = [c for c in curve if c[1] >= 80.0]
    first80 = reach80[0] if reach80 else None
    return {"at_0.6": at60, "frontier_95": best95, "first_80": first80,
            "curve": curve, "n_fp": len(fp_scores), "n_call": len(call_scores)}


def cmd_measure(args) -> int:
    model = Path(args.model)
    print(f"=== bench_tool_abstention.py :: {args.span} :: {model.name} ===")
    run_bench(model, args.span, args.max_len, args.label)
    fr = frontier(model, args.span, args.max_len)
    t, a, k = fr["at_0.6"]
    print(f"\n[frontier] 0.6 -> abstention {a:.1f}% / retention {k:.1f}%")
    if fr["frontier_95"]:
        t, a, k = fr["frontier_95"]
        print(f"[frontier] max abstention at retention>=95%: {a:.1f}% "
              f"@thr {t:.2f} (retention {k:.1f}%)")
    if fr["first_80"]:
        t, a, k = fr["first_80"]
        print(f"[frontier] first >=80% abstention @thr {t:.2f}: abstention "
              f"{a:.1f}% / retention {k:.1f}%")
    if args.json:
        payload = {"span": args.span, "model": str(model),
                   "at_0.6": fr["at_0.6"], "frontier_95": fr["frontier_95"],
                   "first_80": fr["first_80"],
                   "n_fp": fr["n_fp"], "n_call": fr["n_call"]}
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"written to {args.json}")
    return 0


def cmd_train(args) -> int:
    """Build the arm's corpus, then call train_tool_match.py on it."""
    import subprocess
    pairs = Path(args.pairs) if args.pairs else Path(
        f"/tmp/span-{args.span}-pairs.jsonl")
    if not args.pairs:
        rows, stats = build_rows(args.span)
        _write_rows(rows, pairs)
        print(f"built {pairs} ({len(rows)} rows)")
    cmd = [sys.executable, str(ROOT / "scripts" / "train_tool_match.py"),
           "--pairs", str(pairs), "--out", str(args.out),
           "--report", str(args.report), "--max-len", str(args.max_len),
           "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--seed", str(args.seed)]
    print("running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="re-render the corpus for one span")
    p.add_argument("--span", choices=list(SPANS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("verify-v1", help="check the v1 re-render == pairs.jsonl")
    p.set_defaults(func=cmd_verify_v1)

    p = sub.add_parser("train", help="build corpus + train the arm")
    p.add_argument("--span", choices=list(SPANS), required=True)
    p.add_argument("--pairs", type=Path, default=None,
                   help="use this corpus instead of building one")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, default=Path("/tmp/span-train.md"))
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("measure", help="bench + frontier for one artifact")
    p.add_argument("--span", choices=list(SPANS), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--label", default="")
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(func=cmd_measure)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
