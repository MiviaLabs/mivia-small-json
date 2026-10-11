#!/usr/bin/env python3
"""Diagnose the spike-vs-pipeline discrepancy in the tool-match guard.

The spike reported 7.1% no-tool false positives at threshold 0.5; the pipeline
reported 65%. The cause was the scorer feeding all-zero ``token_type_ids`` to
the cross-encoder, so it could not tell the request span from the tool span.
This script scores the SAME rows through five paths to size that:

- fp32 ONNX weights + the spike's HuggingFace tokenizer (the spike's scoring)
- fp32 ONNX weights + the guard's own scorer (the fixed path)
- int8 ONNX weights + the guard's own scorer (what ships)
- fp32 and int8 ONNX weights with the ids zeroed (the historical bug)

The correct paths call ``ToolMatcher`` itself rather than a copy, so the
diagnostic cannot drift from production. It also splits the result by the
spike's own train/test hash, because the spike's ``evaluate()`` scored every
pair, including the 80% the model trained on.

Read-only: loads artifacts, writes nothing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from mivia_small_json.tool_match import (  # noqa: E402
    MAX_LEN,
    THRESHOLD,
    ToolMatcher,
    parse_catalogue,
)

CALLS = ROOT / "data" / "incoming" / "external" / "bfcl" / "calls.jsonl"
IRRELEVANCE = ROOT / "data" / "incoming" / "external" / "bfcl" / "irrelevance.jsonl"
FP32 = ROOT / "data" / "incoming" / "models" / "tool-match.onnx"
INT8 = ROOT / "models" / "tool-match-int8.onnx"
RUST_TOK = ROOT / "models" / "router-tokenizer.json"
HF_TOK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# The abstention eval set: BFCL live-irrelevance, disjoint from the 240 rows
# the guard trains its abstention signal on.
FP_EVAL = (ROOT / "data" / "incoming" / "external" / "bfcl"
           / "irrelevance_eval.jsonl")


def _rows(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build() -> tuple[dict[str, list[tuple[str, bool]]], dict[str, tuple[str, bool]]]:
    """(irrelevance requests, call requests), each tool tagged is_test."""
    irr: dict[str, list[tuple[str, bool]]] = {}
    for line in _rows(IRRELEVANCE):
        tools, req = parse_catalogue(json.loads(line)["text"])
        if not req:
            continue
        for tool in tools:
            irr.setdefault(req, []).append(
                (tool, int(_digest(req + tool), 16) % 5 == 0))
    calls: dict[str, tuple[str, bool]] = {}
    for line in _rows(CALLS):
        tools, req = parse_catalogue(json.loads(line)["text"])
        if len(tools) == 1 and req:
            calls[req] = (tools[0], int(_digest(req + tools[0]), 16) % 5 == 0)
    return irr, calls


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _softmax(logits: np.ndarray) -> float:
    exp = np.exp(logits - logits.max())
    return float((exp / exp.sum())[1])


def hf_scorer(sess, tok):
    """The spike's scoring path: HuggingFace tokenizer, one pair at a time."""
    def score(req: str, tool: str) -> float:
        enc = tok([req], [tool], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="np")
        feed = {k: np.asarray(v) for k, v in enc.items()
                if k in {"input_ids", "attention_mask", "token_type_ids"}}
        return _softmax(sess.run(None, feed)[0][0])
    return score


def shipped_scorer(sess, tok):
    """The production path: the guard's own scorer, so it cannot drift."""
    return ToolMatcher(sess, tok, [i.name for i in sess.get_inputs()])._score


def zeroed_scorer(sess, tok):
    """The historical bug, kept only to show its size.

    Zeros instead of the tokenizer's segment ids. This mirrors
    ``ToolMatcher._feed``; it exists to reproduce a defect that is no
    longer in the tree, so it is deliberately its own copy.
    """
    def score(req: str, tool: str) -> float:
        enc = tok.encode_batch([(req, tool)])[0]
        ids = np.array([enc.ids], dtype=np.int64)
        feed = {"input_ids": ids,
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        if "token_type_ids" in {i.name for i in sess.get_inputs()}:
            feed["token_type_ids"] = np.zeros_like(ids)
        return _softmax(sess.run(None, feed)[0][0])
    return score


def rates(score, irr, calls, subset: str) -> tuple[float, float, int, int]:
    """(no-tool FP %, calls kept %, n_fp, n_requests) for train/test/all."""
    want = {"train": False, "test": True}.get(subset)

    def keep(is_test: bool) -> bool:
        return want is None or is_test == want

    fp = n = 0
    for req, tools in irr.items():
        selected = [name for name, is_test in tools if keep(is_test)]
        if not selected:
            continue
        n += 1
        if max(score(req, t) for t in selected) >= THRESHOLD:
            fp += 1
    kept = kn = 0
    for req, (tool, is_test) in calls.items():
        if not keep(is_test):
            continue
        kn += 1
        if score(req, tool) >= THRESHOLD:
            kept += 1
    return (100 * fp / max(1, n), 100 * kept / max(1, kn), fp, n)


def main() -> int:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    irr, calls = build()
    pairs = sum(len(v) for v in irr.values()) + len(calls)
    test_pairs = sum(1 for v in irr.values() for _n, t in v if t) + \
        sum(1 for _n, t in calls.values() if t)
    print(f"pairs: {pairs} ({pairs - test_pairs} train / {test_pairs} test)")
    print(f"irrelevance requests: {len(irr)}   single-tool call requests: {len(calls)}")
    print()

    fp32, int8 = _session(FP32), _session(INT8)
    rust = Tokenizer.from_file(str(RUST_TOK))
    rust.enable_truncation(max_length=MAX_LEN)
    rust.enable_padding(length=None)
    hf = AutoTokenizer.from_pretrained(HF_TOK)

    # Tokenizer equivalence on a real pair, the thing to rule out first.
    req, (tool, _t) = next(iter(calls.items()))
    a = hf([req], [tool], truncation=True, max_length=MAX_LEN,
           padding=True, return_tensors="np")
    b = rust.encode_batch([(req, tool)])[0]
    same = (list(a["input_ids"][0]) == b.ids
            and list(a["token_type_ids"][0]) == b.type_ids)
    print(f"tokenizer ids+type_ids identical: {same}")
    print(f"shipped tool_match.THRESHOLD: {THRESHOLD}")
    print()

    paths = [
        ("fp32 + HF tokenizer  (the spike)", hf_scorer(fp32, hf)),
        ("fp32 + Rust, segment ids (fixed)", shipped_scorer(fp32, rust)),
        ("int8 + Rust, segment ids (shipped)", shipped_scorer(int8, rust)),
        ("fp32 + Rust, ZERO ids (the bug)", zeroed_scorer(fp32, rust)),
        ("int8 + Rust, ZERO ids (the bug)", zeroed_scorer(int8, rust)),
    ]
    header = f"{'path':<36} {'subset':<7} {'no-tool FP %':>12} {'calls kept %':>13} {'n':>5}"
    print(header)
    print("-" * len(header))
    for label, score in paths:
        for subset in ("all", "train", "test"):
            fp, kept, n_fp, n = rates(score, irr, calls, subset)
            print(f"{label:<36} {subset:<7} {fp:>12.1f} {kept:>13.1f} {n:>5}"
                  f"   ({n_fp} fp)")
        print()

    # The pipeline's own evaluator, on the abstention eval set.
    matcher = ToolMatcher(int8, rust, [i.name for i in int8.get_inputs()])
    dev = [json.loads(line) for line in _rows(FP_EVAL)]
    abstained = sum(1 for r in dev if matcher.best_score(r["text"]) < THRESHOLD)
    print(f"pipeline best_score on {FP_EVAL.name}: {abstained}/{len(dev)} "
          f"abstained, {len(dev) - abstained} false positives "
          f"({100 * (len(dev) - abstained) / len(dev):.1f}%)")
    print()

    # End-to-end operating curve: every dev row, every call row.
    calls_all = [json.loads(line) for line in _rows(CALLS)]
    curve = f"{'thr':>5} {'no-tool FP %':>13} {'calls kept %':>13}"
    print("end-to-end, all rows (shipped int8)")
    print(curve)
    print("-" * len(curve))
    for thr in (0.4, 0.5, 0.6, 0.7):
        fp = sum(1 for r in dev if matcher.best_score(r["text"]) >= thr)
        kept = sum(1 for c in calls_all
                   if matcher.best_score(c["text"]) >= thr)
        print(f"{thr:>5.1f} {100 * fp / len(dev):>13.1f} "
              f"{100 * kept / len(calls_all):>13.1f}")
    print()

    # Operating point: held-out only, shipped int8 weights.
    shipped = matcher._score
    irr_test = {r: [(t, True) for t, _ in v] for r, v in irr.items()}
    calls_test = calls
    print("threshold sweep, held-out pairs only (shipped int8)")
    head2 = f"{'thr':>5} {'no-tool FP %':>13} {'calls kept %':>13} " \
            f"{'precision':>10} {'recall':>7} {'F1':>7}"
    print(head2)
    print("-" * len(head2))
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        n_fp = fp = 0
        for req, tools in irr_test.items():
            sel = [t for t, t_is in tools if t_is]
            if not sel:
                continue
            n_fp += 1
            fp += max(shipped(req, t) for t in sel) >= thr
        tp = fn = 0
        for req, (tool, is_test) in calls_test.items():
            if not is_test:
                continue
            if shipped(req, tool) >= thr:
                tp += 1
            else:
                fn += 1
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"{thr:>5.1f} {100 * fp / max(1, n_fp):>13.1f} "
              f"{100 * tp / max(1, tp + fn):>13.1f} {prec:>10.3f} {rec:>7.3f} "
              f"{f1:>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
