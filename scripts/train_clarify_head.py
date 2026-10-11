#!/usr/bin/env python3
"""Train a defer head for the tool-match guard (ticket 11).

The shipped guard is a binary match scorer. Its remaining false positives are
"structural alignment bias" near-misses: the request's attributes map onto the
tool's parameters, so the pair scores high, even though the tool cannot serve
the goal (SABEval, ACL 2026). Abstention cannot recover those — every
rebalancing arm has been falsified — but *deferral* can: a wrong call becomes
a question.

Two stages, so the binary match signal is not diluted:

1. the shipped guard decides match / no-match (unchanged);
2. on a match, this head decides ``call`` or ``defer``.

The head is trained on:

- ``call``  — a real BFCL call whose request carries the parameters;
- ``defer`` — (a) the same request with its concrete values stripped, the
  ASKTOACT recipe (EMNLP 2025), and (b) the structurally-aligned near-misses
  from `pairs.jsonl`, which the guard matches but cannot serve.

``pairs.jsonl`` already excludes the eval requests (`dropped_eval_request`), so
the eval set stays disjoint.

Pre-registered bar: blocked near-miss false positives >= 85% at call retention
>= 95% on the 884 BFCL irrelevance rows and 183 real calls. The shipped
composite rule scores 79.5% / 97.3%.

Candidate artifact only; writes under data/incoming/models/ and never promotes.
GPU when present (owner policy).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.determinism import seed_everything  # noqa: E402

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
FP_EVAL = ROOT / "data" / "incoming" / "external" / "bfcl" / "irrelevance_eval.jsonl"
CALLS = ROOT / "data" / "incoming" / "external" / "bfcl" / "calls.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "tool-defer-int8.onnx"
BAR = {"blocked": 85.0, "retention": 95.0}

DEFER, CALL = 0, 1

# Structurally-aligned negatives: the guard matches them, the tool cannot serve.
HARD_NEGATIVES = {"bfcl_distractor_aligned", "bfcl_irrelevance_negatives"}

_VALUE_PATTERNS = (
    re.compile(r"\b\d+(?:\.\d+)?\b"),
    re.compile(r"'[^']*'|\"[^\"]*\""),
    re.compile(r"https?://\S+"),
    re.compile(r"\b[\w.-]+@[\w.-]+\b"),
)


def _underspecify(request: str) -> str:
    """Strip concrete values so the request is missing its parameters.

    ASKTOACT (EMNLP 2025): a request with its key parameters removed is the
    training signal for "ask the user" — the tool is right, the arguments are
    not there.
    """
    text = request
    for pattern in _VALUE_PATTERNS:
        text = pattern.sub("", text)
    return " ".join(text.split())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_corpus(include_hard: bool = True) -> tuple[list[dict], dict[str, int]]:
    """(request, tool, label) rows: 1 = call, 0 = defer."""
    stats = {"call": 0, "defer_removed": 0, "defer_hard_negative": 0,
             "skipped_unchanged": 0, "easy_negative_dropped": 0,
             "hard_negative_dropped": 0}
    rows: list[dict] = []
    for row in _rows(PAIRS):
        request, tool = row["request"], row["tool"]
        if row["label"] == 1:
            rows.append({"request": request, "tool": tool, "label": CALL,
                         "split": row["split"]})
            stats["call"] += 1
            vague = _underspecify(request)
            if vague and vague != request:
                rows.append({"request": vague, "tool": tool, "label": DEFER,
                             "split": row["split"]})
                stats["defer_removed"] += 1
            else:
                stats["skipped_unchanged"] += 1
        elif row["source"] in HARD_NEGATIVES:
            if not include_hard:
                stats["hard_negative_dropped"] += 1
                continue
            # Matched-but-unservable: stage 1 lets these through, so stage 2
            # is exactly where they must become a question.
            rows.append({"request": request, "tool": tool, "label": DEFER,
                         "split": row["split"]})
            stats["defer_hard_negative"] += 1
        else:
            # Easy negatives never reach stage 2; keeping them would teach the
            # head a separation the guard has already made.
            stats["easy_negative_dropped"] += 1
    return rows, stats


def main() -> int:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-hard-negatives", action="store_true",
                    help="train the defer class from underspecification only, "
                         "without the structurally-aligned near-misses")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} (cuda available: {torch.cuda.is_available()})")

    rows, stats = build_corpus(include_hard=not args.no_hard_negatives)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    print(f"corpus {stats} · train {len(train)} / dev {len(dev)}")

    tok = AutoTokenizer.from_pretrained(BASE)

    class Pairs(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            r = self.data[i]
            enc = tok(r["request"], r["tool"], truncation=True,
                      max_length=args.max_len, padding="max_length")
            return (torch.tensor(enc["input_ids"]),
                    torch.tensor(enc["attention_mask"]),
                    torch.tensor(r["label"]))

    seed_everything(args.seed)
    net = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, ignore_mismatched_sizes=True).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    loader = DataLoader(Pairs(train), batch_size=args.batch, shuffle=True)
    net.train()
    for epoch in range(args.epochs):
        total = batches = 0
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = net(input_ids=ids, attention_mask=mask).logits
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        print(f"epoch {epoch + 1}: loss {total / max(1, batches):.4f}")

    net.eval()
    dev_loader = DataLoader(Pairs(dev), batch_size=64)
    correct = total = 0
    with torch.no_grad():
        for ids, mask, labels in dev_loader:
            logits = net(input_ids=ids.to(device),
                         attention_mask=mask.to(device)).logits
            correct += int((logits.argmax(-1).cpu() == labels).sum())
            total += len(labels)
    print(f"dev pair accuracy: {100 * correct / max(1, total):.1f}%")

    from onnxruntime.quantization import QuantType, quantize_dynamic

    net.to("cpu").eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = args.out.parent / args.out.name.replace("-int8", "")
    dummy = tok(["a request"], ["a tool: does a thing"], return_tensors="pt",
                truncation=True, max_length=args.max_len, padding=True)
    torch.onnx.export(
        net,
        (dummy["input_ids"], dummy["attention_mask"]),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(args.out), weight_type=QuantType.QInt8)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    (args.out.parent / (args.out.name + ".sha256")).write_text(digest + "\n",
                                                              encoding="utf-8")
    print(f"exported {args.out.name} ({args.out.stat().st_size} bytes, "
          f"sha256 {digest[:16]})")

    # Two-stage decision: shipped guard, then this head on the matched tool.
    import numpy as np
    import onnxruntime as ort

    from mivia_small_json import tool_match
    from mivia_small_json.tool_match import render_tool

    # Stage 1 must be the SHIPPED artifact. `load_matcher()` returns the
    # cascade built from the staged directory, and the bar is defined against
    # the shipped guard, so using the cascade would not be comparable.
    guard_path = ROOT / "models" / "tool-match-int8.onnx"
    if not guard_path.is_file():
        print("shipped guard unavailable", file=sys.stderr)
        return 1
    from tokenizers import Tokenizer

    guard_session = ort.InferenceSession(str(guard_path),
                                         providers=["CPUExecutionProvider"])
    guard_tok = Tokenizer.from_file(str(ROOT / "models" / "router-tokenizer.json"))
    guard_tok.enable_truncation(max_length=256)
    guard_tok.enable_padding(length=None)
    guard = tool_match.ToolMatcher(
        guard_session, guard_tok, [i.name for i in guard_session.get_inputs()])
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])

    def defer_prob(request: str, tool: str) -> float:
        enc = tok([request], [tool], truncation=True, max_length=args.max_len,
                  padding=True)
        logits = sess.run(["logits"], {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]})[0][0]
        e = np.exp(logits - logits.max())
        return float((e / e.sum())[DEFER])

    def stage2(text: str) -> tuple[bool, float]:
        """(stage-1 matched, defer probability on the matched tool)."""
        best_tool, score = guard.best_match(text)
        if best_tool is None or score < tool_match.THRESHOLD:
            return False, 0.0
        _, request = tool_match.parse_catalogue_raw(text)
        if not request:
            return False, 0.0
        return True, defer_prob(request, render_tool(best_tool))

    fp_rows = _rows(FP_EVAL)
    call_rows = _rows(CALLS)

    # Score each row once; the sweep is then arithmetic. Re-running the guard
    # per tau would be ~100k guard calls.
    fp_scored = [stage2(r["text"]) for r in fp_rows]
    call_scored = [stage2(r["text"]) for r in call_rows]

    frontier = []
    for i in range(5, 100):
        tau = i / 100
        blocked = 100 * sum(1 for matched, p in fp_scored
                            if not matched or p >= tau) / len(fp_scored)
        retained = 100 * sum(1 for matched, p in call_scored
                             if matched and p < tau) / len(call_scored)
        frontier.append((tau, blocked, retained))

    for tau, blocked, retained in frontier[::15]:
        print(f"tau={tau:.2f}  blocked={blocked:5.1f}%  retention={retained:5.1f}%")

    eligible = [f for f in frontier if f[2] >= BAR["retention"]]
    if eligible:
        tau, blocked, retained = max(eligible, key=lambda f: f[1])
        print(f"best at retention >= {BAR['retention']:.0f}%: "
              f"blocked {blocked:.1f}% (tau={tau:.2f}, retention {retained:.1f}%)")
        passed = blocked >= BAR["blocked"]
    else:
        print(f"no tau reaches retention >= {BAR['retention']:.0f}%")
        passed = False
    print(f"bar {BAR} -> {'CLEARED' if passed else 'FALSIFIED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
