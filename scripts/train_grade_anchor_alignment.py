#!/usr/bin/env python3
"""Fine-tune the Tier-1 cross-encoder with an ordinal anchor-alignment task.

The zero-shot `grade` primitive needs the cross-encoder to rank arbitrary
ordered anchor descriptions for a given text. The shipped guard was trained
only on binary BFCL match pairs, so its scores collapse on ordinal anchors
(measured: 0% exact on implicit-severity LogHub dev rows, 25-36% on explicit
rows, against an 80% bar - see reports/training-report.md).

This script co-trains two tasks on one backbone (cross-encoder/
ms-marco-MiniLM-L-12-v2, the Tier 1 guard backbone):

1. **Guard task**: the existing BFCL pair corpus with the SelectiveNet loss
   (selection head g at coverage 0.75) - unchanged behavior contract.
2. **Ordinal task**: listwise softmax over the K anchors of one text, with
   ordinal soft targets (a discretized Gaussian around the gold tier) - the
   MLL formulation (arXiv:2405.11775): a proper scoring rule with dense
   gradients from every anchor, ordinal structure encoded in the target
   distribution instead of a separate head.

Anchor strings come from fixed tier templates, so the same phrasing family
works at serve time. The primitive stays zero-shot w.r.t. anchor *strings*:
any new anchor text can be scored at inference; only calibration quality
depends on how close a new anchor reads to the training templates.

Training uses the GPU when present (house rule). The artifact is a candidate:
written under data/incoming/models/ (gitignored) with a sha256 sidecar and
never promoted automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.determinism import seed_everything  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - training extra only
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from train_tool_match import SelectiveCrossEncoder, selectivenet_loss  # noqa: E402

BASE = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
DEFAULT_LOG = ROOT / "data" / "real" / "log_event" / "train.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "tool-match-grade-int8.onnx"
DEFAULT_REPORT = ROOT / "reports" / "grade-anchor-training.md"

# Ordered severity tiers and the anchor phrasing templates. Serve-time anchors
# should use the same phrasing family; the model zero-shots new strings but
# was calibrated on these. Several paraphrases per tier: training samples one
# per batch, so the model learns the tier's semantics, not one string.
SEVERITY_TIERS = ["INFO", "WARNING", "ERROR", "CRITICAL"]
SEVERITY_ANCHORS = {
    "INFO": [
        "info level: routine informational messages about normal successful "
        "operation and expected events.",
        "informational: normal operation, status updates, successful "
        "completions, routine events.",
        "info severity: everything is working as expected; standard "
        "operational messages.",
    ],
    "WARNING": [
        "warning level: recoverable abnormal conditions that may need "
        "attention but are not failures.",
        "warning: something unexpected happened; the task continued but "
        "needs attention soon.",
        "warning severity: non-fatal issues, degraded behavior, retryable "
        "conditions.",
    ],
    "ERROR": [
        "error level: operation failures, faults, and errors that disrupted "
        "a task or transaction.",
        "error: an operation failed; a task could not complete as requested.",
        "error severity: failed requests, exceptions, and faults needing "
        "intervention.",
    ],
    "CRITICAL": [
        "critical level: severe crashes, fatal errors, data loss, or "
        "complete system failure requiring immediate action.",
        "critical: the system is unusable or crashing; fatal failures and "
        "data loss.",
        "critical severity: catastrophic failures, outages, unrecoverable "
        "errors.",
    ],
}

# Difficulty anchor templates (synthetic tiers for the complexity report line),
# with paraphrases per tier.
DIFFICULTY_TIERS = ["BASIC", "INTERMEDIATE", "ADVANCED"]
DIFFICULTY_ANCHORS = {
    "BASIC": [
        "basic difficulty: simple arithmetic, addition, subtraction, "
        "single-step calculations.",
        "basic: elementary math, one-step arithmetic problems.",
        "basic difficulty: primary-school arithmetic and simple sums.",
    ],
    "INTERMEDIATE": [
        "intermediate difficulty: algebra, polynomial equations, quadratics, "
        "multi-step problems.",
        "intermediate: secondary-school algebra and equation solving.",
        "intermediate difficulty: multi-step algebraic manipulation, "
        "quadratic formulas.",
    ],
    "ADVANCED": [
        "advanced difficulty: calculus, differential equations, linear "
        "algebra, proofs.",
        "advanced: university-level mathematics, derivatives, integrals, "
        "eigenvalues.",
        "advanced difficulty: multivariable calculus, differential "
        "equations, abstract algebra.",
    ],
}

# Ordinal soft-target width (in tier units). sigma=0.5 puts ~90% of the
# target mass on the gold tier and its nearest neighbours, matching the
# within-1 tolerance the benchmark reports.
SIGMA = 0.5

# Priority anchor family (issue-register): LOW < MEDIUM < HIGH. Trained on
# GitHub issues weak-labeled via the Caddy & Treude MSR 2024 prioritised-label
# mapping (grade-data-expansion).
PRIORITY_TIERS = ["LOW", "MEDIUM", "HIGH"]
PRIORITY_ANCHORS = {
    "LOW": [
        "low priority: nice to have, no timeline; work can wait indefinitely "
        "without harm.",
        "low priority: cosmetic or minor issues that do not block anyone.",
        "low priority: backlog item, little urgency and little impact.",
    ],
    "MEDIUM": [
        "medium priority: should be addressed in the normal course of work, "
        "neither urgent nor negligible.",
        "medium priority: meaningful impact on some users; schedule it soon.",
        "medium priority: moderate importance, planned for an upcoming "
        "iteration.",
    ],
    "HIGH": [
        "high priority: drop other work; this is urgent and impacts many "
        "users or critical functionality.",
        "high priority: blocker or critical issue that must be handled "
        "immediately.",
        "high priority: severe impact, needs attention right away.",
    ],
}

# Anchor families for the ordinal task: tier order and paraphrase pools.
FAMILIES = {
    "severity": (SEVERITY_TIERS, SEVERITY_ANCHORS),
    "difficulty": (DIFFICULTY_TIERS, DIFFICULTY_ANCHORS),
    "priority": (PRIORITY_TIERS, PRIORITY_ANCHORS),
}

# Guard-loss weight vs ordinal-loss weight. 1.0 keeps the guard task at its
# usual strength; the ordinal task rides along on the shared backbone.
GUARD_WEIGHT = 1.0
ORDINAL_WEIGHT = 1.0


def soft_targets(k: int, gold: int, sigma: float) -> list[float]:
    """Discretized Gaussian over tiers, normalized to sum 1."""
    weights = [math.exp(-((i - gold) ** 2) / (2 * sigma * sigma)) for i in range(k)]
    total = sum(weights)
    return [w / total for w in weights]


def load_rows(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def build_ordinal_rows(log_rows: list[dict]) -> list[dict]:
    """(text, family, gold) rows from LogHub train severity labels."""
    alias = {"FATAL": "CRITICAL", "WARN": "WARNING", "ERR": "ERROR"}
    tiers = SEVERITY_TIERS
    out = []
    for r in log_rows:
        level = (r.get("gold") or {}).get("level")
        if not level:
            continue
        tier = alias.get(level, level)
        if tier not in tiers:
            continue
        text = r["raw"].strip()
        if not text:
            continue
        out.append({"text": text, "family": "severity", "gold": tiers.index(tier)})
    return out


def build_difficulty_rows(seed: int, n_per_tier: int = 400) -> list[dict]:
    """Synthetic difficulty texts per tier (arithmetic vs algebra vs calculus).

    Written as programmatic templates (seeded) so the dev benchmark's natural
    language examples stay unseen; difficulty is expressed through operation
    vocabulary the anchor templates also name.
    """
    import random

    rng = random.Random(seed)
    basics = [
        "What is {a} plus {b}?",
        "Add {a} and {b} together",
        "Compute {a} - {b}",
        "Calculate {a} * {b}",
        "What is {a} + {b} + {c}?",
    ]
    inter = [
        "Solve the equation {a}x + {b} = {c}",
        "Expand the polynomial (x + {a})(x - {b})",
        "Find the roots of x^2 - {a}x + {b} = 0",
        "Simplify the algebraic expression {a}x^2 + {b}x - {c}",
        "Solve the system {a}x + {b}y = {c} and x - y = {d}",
    ]
    adv = [
        "Find the derivative of f(x) = {a}x^3 - {b}x + {c}",
        "Evaluate the integral of {a}x^2 over x from 0 to {b}",
        "Find the eigenvalues of a {n}x{n} matrix",
        "Solve the differential equation y'' + {a}y = 0",
        "Compute the gradient of f(x, y) = {a}xy + {b}x^2",
    ]
    rows = []
    for tier_idx, templates in enumerate([basics, inter, adv]):
        for _ in range(n_per_tier):
            t = rng.choice(templates)
            text = t.format(
                a=rng.randint(2, 99), b=rng.randint(2, 99), c=rng.randint(2, 99),
                d=rng.randint(2, 20), n=rng.choice([2, 3, 4]),
            )
            rows.append({"text": text, "family": "difficulty", "gold": tier_idx})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--corpus", type=Path, default=None,
                        help="pre-built grade corpus (data/real/grade/"
                             "train.jsonl); overrides --log")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--ordinal-batch", type=int, default=16,
                        help="texts per ordinal listwise batch")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-coverage", type=float, default=0.75)
    parser.add_argument("--coverage-weight", type=float, default=32.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--near-miss-weight", type=float, default=5.0)
    args = parser.parse_args()

    if torch is None:
        print("torch unavailable; install the training extra", file=sys.stderr)
        return 1

    import torch as _t

    _t.manual_seed(args.seed)
    device = "cuda" if _t.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({_t.cuda.get_device_name(0)})" if device == "cuda" else ""))

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    rows = load_rows(args.pairs)
    guard_train = [r for r in rows if r["split"] == "train"]
    guard_dev = [r for r in rows if r["split"] == "dev"]
    print(f"guard corpus: {len(rows)} rows ({len(guard_train)} train / {len(guard_dev)} dev)")

    # Ordinal rows: either a pre-built corpus (data/real/grade/train.jsonl,
    # built by scripts/build_grade_corpus.py; families read from rows) or the
    # legacy inline LogHub + synthetic difficulty construction.
    if args.corpus is not None:
        corpus_rows = [r for r in load_rows(args.corpus)
                       if r.get("family") in FAMILIES]
        diff_rows = build_difficulty_rows(args.seed)
        ordinal_rows = corpus_rows + diff_rows
        fam_counts = Counter(r["family"] for r in ordinal_rows)
        print(f"ordinal rows: {len(ordinal_rows)} from corpus "
              f"{_rel(args.corpus)} {dict(fam_counts)} "
              f"+ {len(diff_rows)} synthetic difficulty")
    else:
        log_rows = load_rows(args.log)
        sev_rows = build_ordinal_rows(log_rows)
        diff_rows = build_difficulty_rows(args.seed)
        ordinal_rows = sev_rows + diff_rows
        print(f"ordinal rows: {len(ordinal_rows)} ({len(sev_rows)} severity "
              f"from {_rel(args.log)}, {len(diff_rows)} synthetic difficulty)")

    # Tier-balanced sampling pools: LogHub severity is ~75% INFO, and an
    # unbalanced listwise batch teaches "always INFO" instead of ordinal
    # structure. Index (family, gold) -> row list; each ordinal batch draws
    # equally from every tier.
    tier_pools: dict[tuple[str, int], list[dict]] = {}
    for r in ordinal_rows:
        tier_pools.setdefault((r["family"], r["gold"]), []).append(r)
    fam_tiers = {
        "severity": len(SEVERITY_TIERS),
        "difficulty": len(DIFFICULTY_TIERS),
        "priority": len(PRIORITY_TIERS),
    }
    import random as _random

    rng = _random.Random(args.seed)

    def draw_ordinal_batch(n: int) -> list[dict]:
        # Rotates across families, balanced across tiers within each family.
        out: list[dict] = []
        fams = [f for f in ("severity", "difficulty", "priority")
                if any((f, t) in tier_pools for t in range(fam_tiers[f]))]
        per_fam = n // len(fams)
        for fam in fams:
            k = fam_tiers[fam]
            for tier in range(k):
                pool = tier_pools.get((fam, tier), [])
                if not pool:
                    continue
                take = max(1, per_fam // k)
                out.extend(rng.choice(pool) for _ in range(take))
        return out[:n]

    tok = AutoTokenizer.from_pretrained(BASE)
    base_net = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, ignore_mismatched_sizes=True)
    net = SelectiveCrossEncoder(base_net).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    near_miss_sources = {"bfcl_distractor_aligned", "bfcl_irrelevance_negatives"}

    def weight_of(row: dict) -> float:
        return args.near_miss_weight if row["source"] in near_miss_sources else 1.0

    k_sev = len(SEVERITY_TIERS)
    k_diff = len(DIFFICULTY_TIERS)
    targets_sev = torch.tensor(
        [soft_targets(k_sev, g, SIGMA) for g in range(k_sev)], dtype=torch.float)
    targets_diff = torch.tensor(
        [soft_targets(k_diff, g, SIGMA) for g in range(k_diff)], dtype=torch.float)
    k_pri = len(PRIORITY_TIERS)
    targets_pri = torch.tensor(
        [soft_targets(k_pri, g, SIGMA) for g in range(k_pri)], dtype=torch.float)

    seed_everything(args.seed)

    net.train()
    for epoch in range(args.epochs):
        torch.manual_seed(args.seed + epoch)
        perm = torch.randperm(len(guard_train)).tolist()
        total_guard = total_ord = 0.0
        batches = 0

        for start in range(0, len(guard_train), args.batch):
            chunk = [guard_train[i] for i in perm[start:start + args.batch]]
            enc = tok([r["request"] for r in chunk], [r["tool"] for r in chunk],
                      truncation=True, max_length=args.max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = torch.tensor([r["label"] for r in chunk], device=device)
            weights = torch.tensor([weight_of(r) for r in chunk],
                                   dtype=torch.float, device=device)

            logits, selection, aux_logits = net(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                token_type_ids=enc.get("token_type_ids"),
                return_aux=True,
            )
            loss_g, _, _, _, _ = selectivenet_loss(
                logits=logits, selection=selection, aux_logits=aux_logits,
                labels=labels, weights=weights,
                target_coverage=args.target_coverage,
                alpha=args.alpha, coverage_weight=args.coverage_weight,
            )

            # One ordinal listwise batch per guard batch (alternating steps).
            o_batch = draw_ordinal_batch(args.ordinal_batch)
            loss_o = torch.tensor(0.0, device=device)
            fam_groups: dict[str, list[dict]] = {}
            for r in o_batch:
                fam_groups.setdefault(r["family"], []).append(r)
            for family, group in fam_groups.items():
                tiers, anchor_pool = FAMILIES[family]
                k = len(tiers)
                # Listwise group: each text against every tier, one paraphrase
                # drawn per tier per batch so no single anchor string is
                # memorized.
                tier_texts = [rng.choice(anchor_pool[t]) for t in tiers]
                pairs_t, pairs_a = [], []
                for r2 in group:
                    for a in tier_texts:
                        pairs_t.append(r2["text"])
                        pairs_a.append(a)
                enc_o = tok(pairs_t, pairs_a, truncation=True,
                            max_length=args.max_len, padding=True,
                            return_tensors="pt")
                enc_o = {k2: v.to(device) for k2, v in enc_o.items()}
                logits_o, _sel, _aux = net(
                    input_ids=enc_o["input_ids"],
                    attention_mask=enc_o.get("attention_mask"),
                    token_type_ids=enc_o.get("token_type_ids"),
                    return_aux=True,
                )
                # ListNet over anchors: softmax across the K anchors of one
                # text on the match logit (l1 - l0). Per-anchor sigmoids
                # saturate under the co-trained guard CE and their gradients
                # vanish (arms v1/v2 collapsed to p==1 on every anchor); the
                # across-anchor softmax is scale-invariant and trains exactly
                # the relative ranking the argmax decode reads.
                anchor_logits = (logits_o[:, 1] - logits_o[:, 0]).view(len(group), k)
                log_p = F.log_softmax(anchor_logits, dim=-1)
                tgt = {"severity": targets_sev, "difficulty": targets_diff,
                       "priority": targets_pri}[family].to(device)
                golds = torch.tensor([r2["gold"] for r2 in group], device=device)
                # CE against the ordinal soft-target distribution.
                loss_o = loss_o + -(tgt[golds] * log_p).sum(-1).mean()
            loss = GUARD_WEIGHT * loss_g + ORDINAL_WEIGHT * loss_o
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_guard += float(loss_g.detach())
            total_ord += float(loss_o.detach())
            batches += 1

        print(f"epoch {epoch + 1}: guard loss {total_guard / max(1, batches):.4f} · "
              f"ordinal loss {total_ord / max(1, batches):.4f}")

    net.eval()

    net.to("cpu").eval()
    int8_path, fp32_path = export(net, tok, args.out, args.max_len)
    digest = hashlib.sha256(int8_path.read_bytes()).hexdigest()
    (int8_path.parent / (int8_path.name + ".sha256")).write_text(digest + "\n", encoding="utf-8")
    print(f"exported {int8_path.name} ({int8_path.stat().st_size} bytes, sha256 {digest[:16]})")
    return 0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def export(net, tok, out: Path, max_len: int) -> tuple[Path, Path]:
    """Export fp32 + int8 ONNX beside ``out`` (dual-output logits+selection)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    net.to("cpu").eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = out.parent / out.name.replace("-int8", "")
    dummy = tok(["a request"], ["a tool: does a thing"], return_tensors="pt",
                truncation=True, max_length=max_len, padding=True)
    torch.onnx.export(
        net,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        fp32_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits", "selection"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
            "selection": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    return out, fp32_path


if __name__ == "__main__":
    sys.exit(main())
