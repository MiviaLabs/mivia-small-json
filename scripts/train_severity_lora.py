#!/usr/bin/env python3
"""LoRA frozen-backbone CORAL severity specialist (pre-registered arm).

Arm of reports/exp-severity-lora.md (bar written BEFORE this script ran):
every prior arm re-fine-tuned the full backbone and paid a panel regression
because training moved every cutpoint-bearing weight. This arm FREEZES the
shipped encoder entirely (requires_grad=False on all backbone parameters)
and trains only manual LoRA adapters on the attention query/value
projections (rank 8, alpha 16, dropout 0.05, scale alpha/rank; A ~
N(0, 0.01), B = 0 so training starts as the identity) plus the ordinal
latent head and the CORAL cutpoints. No peft dependency: LoRA is a wrapper
module defined here.

Training data and protocol are exactly train_severity_ordinal.py
--extra-train <specialist_pool_v3.jsonl> --soft-adjacent 0.3: corpus
severity train rows + the 549-row teacher pool, collision-refused against
dev/panel/slices/pairs, inverse-frequency balanced sampling, CORAL loss
over soft cumulative targets (Diaz & Marathe, CVPR 2019, via
train_severity_ordinal.soft_cumulative_targets). Seeds 42/43.

Export: the LoRA deltas are FOLDED into the wrapped Linear weights
(W = W0 + (alpha/r) B A) with a numerical allclose check per module, the
wrappers are removed, and the exported graph is a plain encoder with the
SAME contract as the shipped head (input_ids, attention_mask ->
cumulative_logits [B, 3]). fp32 + dynamic int8 + sha256 sidecars exactly
like train_severity_ordinal.py. Candidate artifacts only, under
data/incoming/models/ (gitignored); models/ is never touched.

GPU for training; CPU for export/eval. The pre-registered verdict is NOT
decided here: scripts/measure_register_lora.py measures the routed
one-shot against the bar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json.determinism import seed_everything  # noqa: E402
from train_severity_ordinal import (  # noqa: E402
    BASE,
    CORPUS,
    K,
    PANEL,
    _rows,
    _severity,
    coral_decode,
    ordinal_metrics,
    soft_cumulative_targets,
)

EXTRA_POOL = (ROOT / "data" / "incoming" / "grade-teacher-model"
              / "specialist_pool_v3.jsonl")
DEFAULT_OUT = (ROOT / "data" / "incoming" / "models"
               / "severity-ordinal-lora-s42-int8.onnx")

LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.05
ATTENTION_PROJECTIONS = ("attention.self.query", "attention.self.value")

# Pre-registered bar (reports/exp-severity-lora.md), printed before any
# training and recorded in the results sidecar. The verdict is computed by
# scripts/measure_register_lora.py, not here.
BAR = {
    "v2_exact_min": 60.3,          # shipped 56.3 + 4pp, n=300
    "v2_mcnemar_p_max": 0.05,
    "tier_regression_pp_max": 5.0,  # vs shipped T0 85.3 / T1 38.7 / T2 40.0 / T3 61.3
    "panel_exact_floor": 69.0,
    "panel_shipped_exact": 70.0,    # if below this, McNemar p vs shipped > 0.05
}


class LoRALinear(nn.Module):
    """y = W0 x + (alpha/rank) * B(A(x)); B zero at init so the wrapped
    module is exactly the frozen base at the start of training."""

    def __init__(self, base: nn.Linear, rank: int = LORA_RANK,
                 alpha: float = LORA_ALPHA, dropout: float = LORA_DROPOUT):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.randn(self.rank, base.in_features) * 0.01)
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank))

    def forward(self, x):
        ax = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
        return self.base(x) + self.scale * (ax @ self.lora_B.transpose(0, 1))

    def delta_weight(self) -> torch.Tensor:
        """(alpha/rank) * B @ A, shaped like the frozen base weight."""
        return self.scale * (self.lora_B @ self.lora_A)


def attach_lora(encoder: nn.Module, rank: int = LORA_RANK,
                alpha: float = LORA_ALPHA,
                dropout: float = LORA_DROPOUT) -> int:
    """Wrap every attention query/value nn.Linear in a LoRALinear.

    BERT keeps those projections at
    ``encoder.layer.{i}.attention.self.{query,value}``; matching on the
    suffix keeps this working for any encoder with the same layout.
    Returns the number of wrapped modules (24 for MiniLM-L-12: 12 layers
    x q/v).
    """
    targets = []
    for name, mod in encoder.named_modules():
        if isinstance(mod, nn.Linear) and name.endswith(
                ATTENTION_PROJECTIONS):
            targets.append(name)
    for name in targets:
        parent = encoder.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[1]
        setattr(parent, attr, LoRALinear(getattr(parent, attr),
                                         rank=rank, alpha=alpha,
                                         dropout=dropout))
    return len(targets)


def fold_lora(module: nn.Module) -> int:
    """Fold every LoRALinear into its frozen base weight and unwrap it.

    For each wrapper: W_folded = W0 + (alpha/r) B A; the wrapper's output on
    a random tensor must match the folded base's output (allclose, eval mode
    so dropout is the identity) BEFORE the wrapper is replaced. After this
    returns, the module tree contains no LoRALinear and the graph is a plain
    encoder.
    """
    folded = 0
    for name, parent in list(module.named_modules()):
        for attr, child in list(parent.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            base = child.base
            x = torch.randn(64, base.in_features)
            child.eval()
            with torch.no_grad():
                wrapped = child(x)
                base.weight.data.add_(child.delta_weight())
                folded_out = base(x)
            if not torch.allclose(wrapped, folded_out,
                                  atol=1e-4, rtol=1e-4):
                raise AssertionError(
                    f"LoRA fold mismatch at {name}.{attr}: max abs diff "
                    f"{(wrapped - folded_out).abs().max().item():.3e}")
            setattr(parent, attr, base)
            folded += 1
    return folded


class OrdinalSeverityLoRA(nn.Module):
    """Frozen encoder + LoRA q/v + shared latent and K-1 ordered cutpoints.

    Same forward contract as train_severity_ordinal.OrdinalSeverity:
    (input_ids, attention_mask) -> latent.unsqueeze(-1) - cutpoints, i.e.
    K-1 cumulative logits per row (CORAL).
    """

    def __init__(self, encoder: nn.Module, k: int = K, pool: str = "cls",
                 rank: int = LORA_RANK, alpha: float = LORA_ALPHA,
                 dropout: float = LORA_DROPOUT):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.n_lora = attach_lora(encoder, rank=rank, alpha=alpha,
                                  dropout=dropout)
        # Freeze the ENTIRE backbone (embeddings, attention, FFN, untouched
        # projections), then re-enable the adapter parameters the wrapper
        # owns. Only LoRA A/B + latent head + cutpoints ever receive grads.
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for sub in self.encoder.modules():
            if isinstance(sub, LoRALinear):
                sub.lora_A.requires_grad_(True)
                sub.lora_B.requires_grad_(True)
        hidden = encoder.config.hidden_size
        self.latent = nn.Linear(hidden, 1)
        self.cut0 = nn.Parameter(torch.zeros(1))
        self.deltas = nn.Parameter(torch.zeros(max(0, k - 2)))

    def cutpoints(self):
        if self.deltas.numel() == 0:
            return self.cut0
        return torch.cat([
            self.cut0,
            self.cut0 + torch.cumsum(F.softplus(self.deltas), dim=0),
        ])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if self.pool == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        latent = self.latent(pooled).squeeze(-1)
        return latent.unsqueeze(-1) - self.cutpoints().unsqueeze(0)


def build_ordinal_lora(encoder: nn.Module, k: int = K, pool: str = "cls",
                       rank: int = LORA_RANK, alpha: float = LORA_ALPHA,
                       dropout: float = LORA_DROPOUT) -> OrdinalSeverityLoRA:
    """Model factory used by both training (pretrained encoder) and the
    CPU-only unit tests (tiny random-init encoder)."""
    return OrdinalSeverityLoRA(encoder, k=k, pool=pool, rank=rank,
                               alpha=alpha, dropout=dropout)


def load_extra_train(xp: Path) -> list[dict]:
    """specialist_pool_v3.jsonl as train-only rows, with the SAME
    collision-refusal as train_severity_ordinal.py --extra-train (dev /
    panel / business slice / BFCL pairs). Transcribed from that script's
    main(); the dedupe/type-guard/refusal semantics are identical.
    """
    import re as _re

    extra = [{"text": r["text"], "gold": int(r["gold"])} for r in _rows(xp)
             if isinstance(r.get("text"), str)
             and type(r.get("gold")) is int and 0 <= r["gold"] < K]
    seen = set()
    extra = [r for r in extra
             if not (_re.sub(r"\s+", " ", r["text"].strip().lower()) in seen
                     or seen.add(_re.sub(r"\s+", " ", r["text"].strip().lower())))]

    def _norm(text: str) -> str:
        return _re.sub(r"\s+", " ", text.strip().lower())

    # Review finding (2026-10-07): the guard must cover THIS arm's actual
    # eval surfaces — v2 (the bar's primary surface) and the calibration
    # split — not only the v1 slice the reference script froze against.
    frozen_paths = [p for p in (CORPUS / "dev.jsonl", PANEL,
                                CORPUS / "business_slice.jsonl",
                                CORPUS / "business_slice_v2.jsonl",
                                CORPUS / "business_calib_dev.jsonl")
                    if p.is_file()]
    frozen: set[str] = set()
    for path in frozen_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                frozen.add(_norm(json.loads(line)["text"]))
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
    pairs_path = (ROOT / "data" / "incoming" / "external" / "bfcl"
                  / "pairs.jsonl")
    if pairs_path.is_file():
        for line in pairs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                prow = json.loads(line)
            except ValueError:
                continue
            for key in ("request", "text", "prompt"):
                if isinstance(prow.get(key), str):
                    frozen.add(_norm(prow[key]))
    clash = [r["text"] for r in extra
             if _re.sub(r"\s+", " ", r["text"].strip().lower()) in frozen]
    if clash:
        raise SystemExit(f"REFUSED: {len(clash)} extra rows collide "
                         f"with dev/panel/slice/pairs, e.g. {clash[0][:80]!r}")
    return extra


def main() -> int:
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="LoRA adapters + fresh ordinal head need a higher "
                         "lr than the 2e-5 full fine-tune")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--balance", choices=["none", "inv", "sqrt"],
                    default="inv")
    ap.add_argument("--soft-adjacent", type=float, default=0.3)
    ap.add_argument("--rank", type=int, default=LORA_RANK)
    ap.add_argument("--alpha", type=float, default=LORA_ALPHA)
    ap.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="int8 candidate path (default: seed-stamped "
                         "severity-ordinal-lora-s<seed>-int8.onnx)")
    ap.add_argument("--extra-train", type=Path, default=EXTRA_POOL)
    args = ap.parse_args()

    out = args.out if args.out is not None else DEFAULT_OUT.with_name(
        DEFAULT_OUT.name.replace("s42", f"s{args.seed}"))
    out = out if out.is_absolute() else ROOT / out
    fp32_path = out.parent / out.name.replace("-int8", "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} (cuda available: {torch.cuda.is_available()})")
    print(f"arm: LoRA frozen backbone (rank {args.rank}, alpha {args.alpha}, "
          f"dropout {args.lora_dropout}) · loss: CORAL soft-adjacent "
          f"{args.soft_adjacent} · backbone: {BASE} (frozen) · pool: "
          f"{args.pool} · balance: {args.balance}")
    print(f"pre-registered bar (reports/exp-severity-lora.md): {BAR}")

    train_rows = _severity(CORPUS / "train.jsonl")
    dev_rows = _severity(CORPUS / "dev.jsonl")
    xp = args.extra_train if args.extra_train.is_absolute() else ROOT / args.extra_train
    extra = load_extra_train(xp)
    train_rows = train_rows + extra
    print(f"extra train rows: {len(extra)} from {xp.name} "
          f"(histogram {dict(sorted(Counter(r['gold'] for r in extra).items()))})")
    print(f"severity rows: {len(train_rows)} train / {len(dev_rows)} dev")
    print(f"train tier histogram: "
          f"{dict(sorted(Counter(r['gold'] for r in train_rows).items()))}")

    tok = AutoTokenizer.from_pretrained(BASE)

    seed_everything(args.seed)
    encoder = AutoModel.from_pretrained(BASE)
    net = build_ordinal_lora(encoder, k=K, pool=args.pool, rank=args.rank,
                             alpha=args.alpha, dropout=args.lora_dropout)
    net = net.to(device)

    frozen = [n for n, p in net.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    n_lora = sum(n.endswith(("lora_A", "lora_B")) for n in trainable)
    assert n_lora == 2 * net.n_lora, \
        f"expected {2 * net.n_lora} LoRA params, found {n_lora}"
    assert all(n.endswith(("lora_A", "lora_B"))
               or n.startswith(("latent", "cut0", "deltas"))
               for n in trainable), "unexpected trainable backbone param"
    print(f"wrapped {net.n_lora} q/v Linears · frozen params {len(frozen)} · "
          f"trainable {len(trainable)} ({n_lora} LoRA + "
          f"{len(trainable) - n_lora} head/cutpoints)")

    def make_sampler():
        if args.balance == "none":
            return None
        counts = Counter(r["gold"] for r in train_rows)
        expo = 1.0 if args.balance == "inv" else 0.5
        weights = [1.0 / (counts[r["gold"]] ** expo) for r in train_rows]
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(weights, num_samples=len(train_rows),
                                     replacement=True)

    def coral_loss(logits, golds):
        # Soft (--soft-adjacent): cumulative targets are P(tier > k) under
        # the Diaz & Marathe adjacent-tier distribution; helper imported
        # from train_severity_ordinal (pinned by test_soft_ordinal_targets).
        if args.soft_adjacent <= 0.0:
            idx = torch.arange(K - 1, device=logits.device).unsqueeze(0)
            targets = (golds.unsqueeze(1) > idx).float()
        else:
            soft = soft_cumulative_targets(golds.tolist(), K,
                                           args.soft_adjacent)
            targets = torch.tensor(soft, device=logits.device,
                                   dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets)

    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, (len(train_rows) + args.batch - 1) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def batch_indices():
        sampler = make_sampler()
        if sampler is None:
            return torch.randperm(len(train_rows)).tolist()
        return list(sampler)

    def evaluate(rows) -> tuple[float, list[int]]:
        net.eval()
        preds: list[int] = []
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            enc = tok([r["text"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = net(enc["input_ids"],
                             enc["attention_mask"]).cpu().tolist()
            preds.extend(coral_decode(row) for row in logits)
        correct = sum(p == r["gold"] for p, r in zip(preds, rows))
        return (100.0 * correct / len(rows) if rows else 0.0), preds

    step = 0
    for epoch in range(args.epochs):
        net.train()
        perm = batch_indices()
        total = 0.0
        batches = 0
        for start in range(0, len(perm), args.batch):
            chunk = [train_rows[i] for i in perm[start:start + args.batch]]
            enc = tok([r["text"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            golds = torch.tensor([r["gold"] for r in chunk], device=device)
            logits = net(enc["input_ids"], enc["attention_mask"])
            loss = coral_loss(logits, golds)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            total += float(loss.detach())
            batches += 1
        dev_acc, _ = evaluate(dev_rows)
        print(f"epoch {epoch + 1}: loss {total / max(1, batches):.4f} · "
              f"corpus dev exact {dev_acc:.1f}%")

    dev_acc, dev_preds = evaluate(dev_rows)
    print(f"corpus dev exact: {dev_acc:.1f}%")
    print("dev pred histogram:", dict(sorted(Counter(dev_preds).items())))
    print("cutpoints:", [round(x, 3) for x in
                         net.cutpoints().detach().cpu().tolist()])

    # Export: fold LoRA into the frozen weights (verified allclose per
    # module), unwrap, then export the plain encoder exactly like
    # train_severity_ordinal.py (fp32 + dynamic int8 + sha256 sidecars).
    from onnxruntime.quantization import QuantType, quantize_dynamic

    net.to("cpu").eval()
    n_folded = fold_lora(net)
    assert n_folded == net.n_lora, \
        f"folded {n_folded} modules, wrapped {net.n_lora}"
    assert not any(isinstance(m, LoRALinear) for m in net.modules()), \
        "LoRALinear left in the module tree after folding"
    print(f"folded {n_folded} LoRA modules into frozen weights "
          f"(per-module allclose verified)")

    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = tok(["a log line"], return_tensors="pt", truncation=True,
                max_length=args.max_len, padding=True)
    torch.onnx.export(
        net,
        (dummy["input_ids"], dummy["attention_mask"]),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["cumulative_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "cumulative_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    fp32_digest = hashlib.sha256(fp32_path.read_bytes()).hexdigest()
    (out.parent / (out.name + ".sha256")).write_text(digest + "\n",
                                                     encoding="utf-8")
    (fp32_path.parent / (fp32_path.name + ".sha256")).write_text(
        fp32_digest + "\n", encoding="utf-8")
    print(f"exported {out.name} ({out.stat().st_size} bytes, "
          f"sha256 {digest[:16]}) · fp32 {fp32_path.name} "
          f"({fp32_path.stat().st_size} bytes)")

    # Diagnostic panel pass (candidate head ALONE, unrouted — NOT the bar;
    # the bar is the routed one-shot in scripts/measure_register_lora.py).
    from mivia_small_json.stats import mcnemar_exact
    from mivia_small_json.tool_match import load_severity_ordinal

    shipped = load_severity_ordinal()
    panel = _rows(PANEL)
    cand_head = load_severity_ordinal(out)
    assert shipped is not None and cand_head is not None

    def tier_of(head, text: str) -> int:
        return head.grade(text, with_set=True)[2][0]

    cand_panel = [tier_of(cand_head, r["text"]) for r in panel]
    ship_panel = [tier_of(shipped, r["text"]) for r in panel]
    gold_panel = [r["gold"] for r in panel]
    cm = ordinal_metrics(gold_panel, cand_panel)
    sm = ordinal_metrics(gold_panel, ship_panel)
    b = sum(ship_panel[i] == gold_panel[i] and cand_panel[i] != gold_panel[i]
            for i in range(len(panel)))
    c = sum(cand_panel[i] == gold_panel[i] and ship_panel[i] != gold_panel[i]
            for i in range(len(panel)))
    p = mcnemar_exact(b, c)
    print(f"\ndiagnostic (unrouted, not the bar): panel exact candidate "
          f"{cm['exact']:.1f}% vs shipped {sm['exact']:.1f}% · McNemar "
          f"b={b} c={c} p={p:.4f}")

    results = {
        "arm": "lora-frozen-backbone",
        "loss": "coral",
        "soft_adjacent": args.soft_adjacent,
        "base": BASE,
        "lora": {"rank": args.rank, "alpha": args.alpha,
                 "dropout": args.lora_dropout,
                 "targets": list(ATTENTION_PROJECTIONS),
                 "wrapped": net.n_lora, "folded": n_folded},
        "pool": args.pool,
        "balance": args.balance,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_len": args.max_len,
        "seed": args.seed,
        "extra_train": str(xp.relative_to(ROOT)),
        "extra_train_rows": len(extra),
        "train_rows": len(train_rows),
        "artifact": str(out.relative_to(ROOT)),
        "artifact_bytes": out.stat().st_size,
        "artifact_sha256": digest,
        "fp32_artifact": str(fp32_path.relative_to(ROOT)),
        "fp32_sha256": fp32_digest,
        "corpus_dev_exact": dev_acc,
        "cutpoints": net.cutpoints().detach().cpu().tolist(),
        "bar": BAR,
        "bar_note": "verdict decided by scripts/measure_register_lora.py "
                    "(routed one-shot on v2 + panel), not by this sidecar",
        "panel_diagnostic_unrouted": {
            "candidate": cm, "shipped": sm,
            "mcnemar": {"b": b, "c": c, "p": p},
        },
    }
    sidecar = out.parent / (out.name + ".results.json")
    sidecar.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.name}.results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
