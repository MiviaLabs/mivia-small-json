#!/usr/bin/env python3
"""Fine-tune the tool-match cross-encoder and export it to int8 ONNX.

The guard's relation is between two spans: a request and a listed tool. A
cross-encoder attends across both, so it can express "does this tool serve
this request?". The corpus comes from
`scripts/build_tool_match_corpus.py` and carries its own `split` field, so the
held-out rows are unseen by construction.

Rare structurally-aligned negatives (the BFCL irrelevance rows and the
aligned distractors) are up-weighted in the loss. Without that the easy
negatives dominate and the model never has to learn the quadrant the
abstention eval measures.

An optional pairwise error-regularization term (``--error-reg``; Xin et al.,
ACL 2021) penalizes wrong examples that out-confidence right ones in the
mini-batch. It is training-time only: the exported artifact keeps its
single-logit shape and the runtime is untouched.

Training runs on the GPU when one is present. This script only writes a
candidate artifact and its sha256 sidecar; promoting it to `models/` is a
deliberate, separate step.

Writes the artifact to `--out` and a report to `--report`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.determinism import seed_everything  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _Module = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    _Module = object

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "tool-match-int8.onnx"
DEFAULT_REPORT = ROOT / "reports" / "tool-match-training.md"
# The BFCL near-miss negatives (aligned distractors and irrelevance rows) are
# ~411 rows against ~8k easy negatives. Left alone, the loss barely notices
# the quadrant the abstention eval measures. The external synthetic near-misses
# (Hammer, arXiv 2410.04587) are the same quadrant; they are up-weighted too
# unless --near-miss-sources says otherwise.
NEAR_MISS_SOURCES = frozenset({
    "bfcl_distractor_aligned", "bfcl_irrelevance_negatives",
    "synthetic_nearmiss_xlam", "synthetic_nearmiss_when2call"})
NEAR_MISS_WEIGHT = 5.0


def error_reg_loss(logits, labels):
    """Pairwise error regularization (Xin et al., ACL 2021, Eqs. 9-11).

    The mean of max(0, c_i - c_j)^2 over batch pairs where example i is
    currently wrong and j is right (c is the max softmax probability, the
    confidence the guard's threshold reads). A wrong example must not be
    more confident than a right one. Two stated deviations from the paper:
    mean over pairs instead of sum (batch-size invariance), and 0/1
    current-iteration error instead of a history-averaged error bank.
    Zero when the batch has no wrong/right pair. Pure: takes logits and
    labels, needs no model.
    """
    import torch
    import torch.nn.functional as F

    conf, pred = F.softmax(logits, dim=-1).max(dim=-1)
    wrong = pred != labels
    if not bool(wrong.any()) or bool(wrong.all()):
        return logits.new_zeros(())
    gaps = torch.clamp(conf[wrong, None] - conf[~wrong], min=0.0) ** 2
    return gaps.mean()


def contrastive_margin_loss(probs_full, probs_stripped, margin: float = 0.3):
    """Pairwise presence-contrast margin (fillability twins).

    The mean of relu(margin - (p_full - p_stripped)) over twin pairs: the
    fillable twin must outscore its stripped twin by `margin`. Pointwise
    cross-entropy never constrains the twin ordering, so both sides can
    saturate near 1.0 while the `missing:` clause is ignored; this term puts
    gradient directly on that ordering. Zero when there are no pairs. Pure:
    takes probabilities, needs no model.
    """
    import torch

    if probs_full.numel() == 0:
        return probs_full.new_zeros(())
    gaps = torch.clamp(margin - (probs_full - probs_stripped), min=0.0)
    return gaps.mean()


def group_twins(rows: list[dict]) -> tuple[list[list[int]],
                                            dict[int, int], dict[int, str]]:
    """Twin groups for contrastive batching: (groups, pair_of, role_of).

    A stripped row (`twin_key`, label 0) joins the base positive whose `key`
    matches; every other row is a singleton group. Pure python, no torch, so
    tests can read the same contract the trainer runs.
    """
    key_to_idx: dict[str, int] = {}
    for i, row in enumerate(rows):
        key = row.get("key")
        if key and row.get("label") == 1 and key not in key_to_idx:
            key_to_idx[key] = i
    groups: list[list[int]] = []
    pair_of: dict[int, int] = {}
    role_of: dict[int, str] = {}
    used: set[int] = set()
    for i, row in enumerate(rows):
        twin_key = row.get("twin_key")
        if not (twin_key and row.get("label") == 0):
            continue
        full = key_to_idx.get(twin_key)
        if full is None or full in used:
            continue
        gid = len(groups)
        groups.append([full, i])
        used.add(full)
        used.add(i)
        pair_of[full] = gid
        pair_of[i] = gid
        role_of[full] = "full"
        role_of[i] = "stripped"
    for i in range(len(rows)):
        if i not in used:
            groups.append([i])
    return groups, pair_of, role_of


def selectivenet_loss(
    logits,
    selection,
    aux_logits,
    labels,
    weights=None,
    target_coverage: float = 0.75,
    alpha: float = 0.5,
    coverage_weight: float = 32.0,
    eps: float = 1e-7,
):
    """SelectiveNet loss (Geifman & El-Yaniv, arXiv:1901.09192).

    Combines:
    1. Selective risk on prediction head f(x) weighted by selection s(x).
    2. Auxiliary risk on representation head h(x) to prevent backbone collapse.
    3. Quadratic coverage penalty when empirical coverage < target_coverage.
    """
    import torch
    import torch.nn.functional as F

    s = selection.view(-1)
    ce_f = F.cross_entropy(logits, labels, reduction="none")
    ce_h = F.cross_entropy(aux_logits, labels, reduction="none")
    if weights is not None:
        w = weights.view(-1)
        sel_risk = (w * ce_f * s).sum() / ((w * s).sum() + eps)
        aux_risk = (w * ce_h).sum() / (w.sum() + eps)
    else:
        sel_risk = (ce_f * s).sum() / (s.sum() + eps)
        aux_risk = ce_h.mean()
    emp_cov = s.mean()
    cov_gap = torch.clamp(target_coverage - emp_cov, min=0.0)
    cov_pen = coverage_weight * (cov_gap ** 2)
    total_loss = alpha * sel_risk + (1.0 - alpha) * aux_risk + cov_pen
    return total_loss, sel_risk, aux_risk, emp_cov, cov_pen


class SelectiveCrossEncoder(_Module):
    """Cross-encoder with prediction head f(z), selection head g(z), and aux head h(z)."""

    def __init__(self, base_model, hidden_size: int | None = None, num_labels: int = 2):
        super().__init__()
        if hidden_size is None:
            hidden_size = getattr(getattr(base_model, "config", None), "hidden_size", 384)
        if hasattr(base_model, "bert"):
            self.encoder = base_model.bert
        elif hasattr(base_model, "model"):
            self.encoder = base_model.model
        elif hasattr(base_model, "roberta"):
            self.encoder = base_model.roberta
        else:
            self.encoder = base_model
        self.bert = self.encoder
        self.head = getattr(base_model, "head", None)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.aux_classifier = nn.Linear(hidden_size, num_labels)
        self.selection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        if hasattr(base_model, "classifier"):
            if getattr(base_model.classifier, "out_features", None) == num_labels:
                self.classifier.weight.data.copy_(base_model.classifier.weight.data)
                self.classifier.bias.data.copy_(base_model.classifier.bias.data)
                self.aux_classifier.weight.data.copy_(base_model.classifier.weight.data)
                self.aux_classifier.bias.data.copy_(base_model.classifier.bias.data)
        # Initialize selection head bias to ~1.1 so initial coverage ~0.75 (sigmoid(1.1) ~= 0.75)
        nn.init.constant_(self.selection_head[2].bias, 1.1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        return_aux: bool = False,
    ):
        kwargs = {}
        if (
            token_type_ids is not None
            and hasattr(self.encoder, "embeddings")
            and hasattr(self.encoder.embeddings, "token_type_embeddings")
        ):
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        if hasattr(outputs, "__getitem__") and len(outputs) > 1 and outputs[1] is not None:
            pooled = outputs[1]
        else:
            pooled = outputs[0][:, 0]
        if self.head is not None:
            pooled = self.head(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        selection = self.selection_head(pooled)
        if return_aux:
            aux_logits = self.aux_classifier(pooled)
            return logits, selection, aux_logits
        return logits, selection


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _rel(path: Path) -> str:
    """Path relative to the repo when it is inside it, else as given."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--base", type=str, default=BASE,
                        help=f"base cross-encoder backbone (default {BASE})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-miss-weight", type=float,
                        default=NEAR_MISS_WEIGHT)
    parser.add_argument("--near-miss-sources", type=str, default="",
                        help="comma-separated source tags up-weighted by "
                             "--near-miss-weight (default: the BFCL and "
                             "synthetic near-misses)")
    parser.add_argument("--error-reg", type=float, default=0.0,
                        help="pairwise error-regularization weight (arm B1); "
                             "0.0 preserves the shipped loss exactly")
    parser.add_argument("--margin-barrier", type=float, default=0.0,
                        help="margin barrier loss weight (arm M1): "
                             "penalizes near-misses > 0.3 and positives < 0.7")
    parser.add_argument("--contrast-weight", type=float, default=0.0,
                        help="presence-contrast twin margin weight (fillability "
                             "axis): the fillable twin must outscore its "
                             "stripped twin by --contrast-margin; 0.0 "
                             "preserves the shipped loss exactly")
    parser.add_argument("--contrast-margin", type=float, default=0.3,
                        help="required fillable-minus-stripped probability gap "
                             "(default 0.3)")
    parser.add_argument("--selectivenet", action="store_true",
                        help="train SelectiveNet reject option (arm S1)")
    parser.add_argument("--target-coverage", type=float, default=0.75,
                        help="SelectiveNet target coverage c (default 0.75)")
    parser.add_argument("--coverage-weight", type=float, default=32.0,
                        help="SelectiveNet coverage penalty weight lambda (default 32.0)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SelectiveNet trade-off weight (default 0.5)")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    seed_everything(args.seed)
    rows = load_rows(args.pairs)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    print(f"corpus: {len(rows)} rows ({len(train)} train / {len(dev)} dev) "
          f"from {args.pairs}")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device; training on CPU", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    tok = AutoTokenizer.from_pretrained(args.base)
    base_net = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, ignore_mismatched_sizes=True)
    if args.selectivenet:
        net = SelectiveCrossEncoder(base_net).to(device)
    else:
        net = base_net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    near_miss_sources = frozenset(
        s.strip() for s in args.near_miss_sources.split(",") if s.strip()
    ) or NEAR_MISS_SOURCES

    def weight_of(row: dict) -> float:
        return (args.near_miss_weight if row["source"] in near_miss_sources
                else 1.0)

    if args.contrast_weight > 0.0 and args.selectivenet:
        raise SystemExit("--contrast-weight needs the plain head, "
                         "not --selectivenet")
    twin_groups, pair_of, role_of = group_twins(train)
    n_twins = sum(1 for g in twin_groups if len(g) == 2)
    if args.contrast_weight > 0.0 and n_twins == 0:
        print("WARNING: --contrast-weight set but no twin pairs in train",
              file=sys.stderr)

    net.train()
    margin_total = 0.0
    margin_batches = 0
    for epoch in range(args.epochs):
        torch.manual_seed(args.seed + epoch)
        # Shuffle twin groups, not rows, so a fillable twin and its stripped
        # twin land in the same batch and the margin term sees both sides.
        order = [i for g in torch.randperm(len(twin_groups)).tolist()
                 for i in twin_groups[g]]
        total = 0.0
        for start in range(0, len(order), args.batch):
            batch_idx = order[start:start + args.batch]
            chunk = [train[i] for i in batch_idx]
            enc = tok([r["request"] for r in chunk],
                      [r["tool"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = torch.tensor([r["label"] for r in chunk]).to(device)
            weights = torch.tensor([weight_of(r) for r in chunk],
                                   dtype=torch.float).to(device)
            if args.selectivenet:
                logits, selection, aux_logits = net(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    token_type_ids=enc.get("token_type_ids"),
                    return_aux=True,
                )
                loss, _sel_risk, _aux_risk, _cov, _cov_pen = selectivenet_loss(
                    logits=logits,
                    selection=selection,
                    aux_logits=aux_logits,
                    labels=labels,
                    weights=weights,
                    target_coverage=args.target_coverage,
                    alpha=args.alpha,
                    coverage_weight=args.coverage_weight,
                )
            else:
                out = net(**enc)
                loss = F.cross_entropy(out.logits, labels, reduction="none")
                loss = (loss * weights).sum() / weights.sum()
                if args.error_reg:
                    loss = loss + args.error_reg * error_reg_loss(
                        out.logits, labels)
                if args.margin_barrier > 0.0:
                    probs = F.softmax(out.logits, dim=-1)[:, 1]
                    nm_mask = torch.tensor(
                        [r["source"] in near_miss_sources and r["label"] == 0
                         for r in chunk],
                        dtype=torch.bool, device=device)
                    pos_mask = (labels == 1)
                    barrier = torch.tensor(0.0, device=device)
                    if nm_mask.any():
                        nm_err = torch.clamp(probs[nm_mask] - 0.3, min=0.0).pow(2).mean()
                        barrier = barrier + nm_err
                    if pos_mask.any():
                        pos_err = torch.clamp(0.7 - probs[pos_mask], min=0.0).pow(2).mean()
                        barrier = barrier + pos_err
                    loss = loss + args.margin_barrier * barrier
                if args.contrast_weight > 0.0:
                    by_group: dict[int, dict[str, int]] = {}
                    for pos, idx in enumerate(batch_idx):
                        gid = pair_of.get(idx)
                        if gid is None:
                            continue
                        by_group.setdefault(gid, {})[role_of[idx]] = pos
                    sides = [(d["full"], d["stripped"]) for d in
                             by_group.values()
                             if "full" in d and "stripped" in d]
                    if sides:
                        cprobs = F.softmax(out.logits, dim=-1)[:, 1]
                        margin = contrastive_margin_loss(
                            cprobs[[p[0] for p in sides]],
                            cprobs[[p[1] for p in sides]],
                            args.contrast_margin)
                        loss = loss + args.contrast_weight * margin
                        margin_total += float(margin.detach())
                        margin_batches += 1
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1}: loss {total / max(1, len(train) // args.batch):.4f}")
    net.eval()

    def score(request: str, tool: str) -> float:
        enc = tok([request], [tool], truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            if args.selectivenet:
                logits, selection = net(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    token_type_ids=enc.get("token_type_ids"),
                    return_aux=False,
                )
                prob = float(torch.softmax(logits, -1)[0][1])
                sel = float(selection[0][0])
                return prob if sel >= 0.50 else 0.0
            else:
                return float(torch.softmax(net(**enc).logits, -1)[0][1])

    def pair_rates(subset: list[dict], threshold: float) -> tuple[float, float, float]:
        tp = fn = tn = fp = 0
        for row in subset:
            matched = score(row["request"], row["tool"]) >= threshold
            if row["label"] == 1:
                tp += matched
                fn += not matched
            else:
                fp += matched
                tn += not matched
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        return 100 * rec, 100 * tn / max(1, tn + fp), prec

    dev_groups, _, _ = group_twins(dev)
    dev_twins = [g for g in dev_groups if len(g) == 2]
    twin_order_line = "twin pairs in dev: 0"
    if dev_twins:
        gaps = []
        correct = 0
        for full_i, strip_i in dev_twins:
            pf = score(dev[full_i]["request"], dev[full_i]["tool"])
            ps = score(dev[strip_i]["request"], dev[strip_i]["tool"])
            gaps.append(pf - ps)
            correct += pf > ps
        twin_order_line = (
            f"twin pairs in dev: {len(dev_twins)} ordered full>stripped "
            f"{100 * correct / len(dev_twins):.1f}% "
            f"(mean gap {sum(gaps) / len(gaps):+.3f})")

    sabeval_dev = [r for r in dev if r["source"].startswith("sabeval")]
    lines = ["# Tool-match training", "",
             f"corpus: `{_rel(args.pairs)}` ({len(rows)} rows)",
             f"train {len(train)} / dev {len(dev)}",
             f"base {args.base} · epochs {args.epochs} · batch {args.batch} · "
             f"lr {args.lr} · max_len {args.max_len} · seed {args.seed}",
             f"near-miss weight {args.near_miss_weight} for "
             f"{sorted(near_miss_sources)}",
             f"selectivenet {args.selectivenet} "
             f"(cov={args.target_coverage}, lambda={args.coverage_weight}, alpha={args.alpha})",
             f"error-reg {args.error_reg}",
             f"contrast weight {args.contrast_weight} "
             f"(margin {args.contrast_margin}, train twins {n_twins})",
             ("mean twin margin loss "
              f"{margin_total / max(1, margin_batches):.4f} "
              f"over {margin_batches} paired batches"
              if args.contrast_weight > 0.0 else "contrast off"),
             twin_order_line,
             f"device {device}", "",
             "Held-out rows only, scored at the pair level.", "",
             "| threshold | positives kept % | negatives rejected % | precision |",
             "|---|---|---|---|"]
    for threshold in (0.3, 0.5, 0.6, 0.7, 0.9):
        kept, rejected, prec = pair_rates(dev, threshold)
        lines.append(f"| {threshold} | {kept:.1f} | {rejected:.1f} | {prec:.3f} |")
    if sabeval_dev:
        kept, rejected, _prec = pair_rates(sabeval_dev, 0.6)
        lines += ["", f"Held-out SABEval templates (n={len(sabeval_dev)}): "
                      f"positives kept {kept:.1f}%, siblings rejected "
                      f"{rejected:.1f}%"]
    lines.append("")

    int8_path, fp32_path = export(net, tok, args.out, args.max_len)
    digest = hashlib.sha256(int8_path.read_bytes()).hexdigest()
    (int8_path.parent / (int8_path.name + ".sha256")).write_text(
        digest + "\n", encoding="utf-8")
    tok_path = int8_path.parent / (
        int8_path.name.replace("-int8.onnx", "-tokenizer.json")
        if "-int8.onnx" in int8_path.name
        else int8_path.stem + "-tokenizer.json"
    )
    if hasattr(tok, "backend_tokenizer"):
        tok.backend_tokenizer.save(str(tok_path))
    elif hasattr(tok, "save"):
        tok.save(str(tok_path))
    lines += [f"Exported {int8_path.name} ({int8_path.stat().st_size} bytes)",
              f"sha256 {digest[:16]}", f"fp32 parent {fp32_path.name}", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


def export(net, tok, out: Path, max_len: int) -> tuple[Path, Path]:
    """Export fp32 + int8 ONNX beside ``out``. Returns (int8, fp32)."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    net.to("cpu").eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    if "-int8" in out.name:
        fp32_path = out.parent / out.name.replace("-int8", "")
    else:
        fp32_path = out.parent / f"{out.stem}-fp32{out.suffix}"
    dummy = tok(["a request"], ["a tool: does a thing"], return_tensors="pt",
                truncation=True, max_length=max_len, padding=True)
    if "token_type_ids" in dummy:
        inputs = (dummy["input_ids"], dummy["attention_mask"],
                  dummy["token_type_ids"])
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
    else:
        inputs = (dummy["input_ids"], dummy["attention_mask"])
        input_names = ["input_ids", "attention_mask"]

    if isinstance(net, SelectiveCrossEncoder):
        output_names = ["logits", "selection"]
    else:
        output_names = ["logits"]

    dynamic_axes = {k: {0: "batch", 1: "seq"} for k in input_names}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        net, inputs, fp32_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    return out, fp32_path


if __name__ == "__main__":
    sys.exit(main())
