#!/usr/bin/env python3
"""SNGP tool-match guard arm: spectral-normalized encoder + RBF-GP head.

The guard's score curve is flat across thresholds 0.4-0.7, so no threshold
separates near-miss tool invocations from real calls. This arm replaces the
final softmax with the distance-aware Gaussian-process posterior of SNGP
(Liu et al., NeurIPS 2020, arXiv:2006.10108):

1. **Spectral normalization** on the cross-encoder's Linear weights bounds the
   encoder's Lipschitz constant, so the penultimate features preserve input
   distance (a requirement for the RBF kernel to mean anything).
2. **A Laplace-approximated RBF GP output layer** over those features,
   approximated by random Fourier features ``phi(h) = sqrt(2/D) cos(Wh + b)``.
   Training fits the MAP logit ``f = phi(h) @ beta``; the posterior covariance
   ``Sigma = (ridge*I + mean(phi^T phi))^-1`` is accumulated over the training
   features and frozen. At inference the logit is scaled by the predictive
   stddev, ``z = f / sqrt(1 + pi/8 * ridge * phi^T Sigma phi)`` (the probit
   approximation), so a pair far from the training feature span is shrunk
   toward the decision boundary.

Single forward pass at inference: the covariance lookup is a constant matmul,
so it adds ~no latency. The head is expressed with plain ONNX ops (MatMul,
Cos, Sqrt, Div, Log), so it exports cleanly; the export writes the same
pseudo-logit convention as the SelectiveNet arm (``[log(1-p), log(p)]``) so
the shipped ``ToolMatcher`` reads it with zero contract changes.

``--arm plain`` trains a plain 2-class softmax head on the same backbone,
data, epochs and loss, to isolate the SNGP effect from the encoder.

Writes a candidate artifact under ``data/incoming/models/`` (gitignored) and
the arm report to ``--report``. Never touches ``models/``.

Usage:
    python scripts/exp_sngp_guard.py --arm sngp --epochs 4 --batch 16
    python scripts/exp_sngp_guard.py --arm plain --epochs 4 --batch 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _Module = nn.Module
except ImportError:  # pragma: no cover - training extra only
    torch = None
    nn = None
    F = None
    _Module = object

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "tool-match-sngp-int8.onnx"
DEFAULT_REPORT = ROOT / "reports" / "exp-sngp.md"
# The BFCL near-miss negatives (aligned distractors and irrelevance rows) are
# rare against the easy negatives; up-weight them or the loss ignores the
# quadrant the abstention eval measures. Same weighting as train_tool_match.py.
NEAR_MISS_SOURCES = {"bfcl_distractor_aligned", "bfcl_irrelevance_negatives"}
NEAR_MISS_WEIGHT = 5.0
PI_OVER_8 = math.pi / 8.0


# --------------------------------------------------------------------------
# Spectral normalization (own implementation: deterministic and foldable)
# --------------------------------------------------------------------------
class SNLinear(_Module):
    """Linear layer whose weight is rescaled by its spectral norm.

    The weight is stored un-normalized as ``weight_orig``; every forward
    divides it by an estimate of its largest singular value (power iteration
    on ``u``/``v``). ``fold()`` returns a plain ``nn.Linear`` holding the last
    normalized weight, so the exported ONNX graph has no power-iteration
    state and is deterministic across calls.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, n_power_iter: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_power_iter = n_power_iter
        self.weight_orig = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_orig, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("u", F.normalize(torch.randn(out_features), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(in_features), dim=0))

    @classmethod
    def from_linear(cls, lin: "nn.Linear", n_power_iter: int = 1) -> "SNLinear":
        sn = cls(lin.in_features, lin.out_features,
                 lin.bias is not None, n_power_iter)
        sn.weight_orig.data.copy_(lin.weight.data)
        if lin.bias is not None:
            sn.bias.data.copy_(lin.bias.data)
        # Converge u/v on the pretrained weight before any forward, so an
        # eval-mode pass with no training step still has a sane sigma (random
        # u/v can under-estimate it and blow up W / sigma to NaN).
        sn._update_uv(iters=8)
        return sn

    @torch.no_grad()
    def _update_uv(self, iters: int | None = None) -> None:
        W = self.weight_orig
        for _ in range(self.n_power_iter if iters is None else iters):
            v = F.normalize(W.t() @ self.u, dim=0, eps=1e-12)
            u = F.normalize(W @ v, dim=0, eps=1e-12)
            self.u.copy_(u)
            self.v.copy_(v)

    def _sigma(self):
        return torch.dot(self.u, self.weight_orig @ self.v)

    def forward(self, x):
        if self.training:
            self._update_uv()
        sigma = self._sigma().clamp(min=1e-12)
        return F.linear(x, self.weight_orig / sigma, self.bias)

    @torch.no_grad()
    def fold(self) -> "nn.Linear":
        self._update_uv()
        sigma = self._sigma().clamp(min=1e-12)
        lin = nn.Linear(self.in_features, self.out_features,
                        bias=self.bias is not None)
        lin.weight.data.copy_((self.weight_orig / sigma).detach())
        if self.bias is not None:
            lin.bias.data.copy_(self.bias.data)
        return lin


def apply_spectral_norm(module: "nn.Module", n_power_iter: int) -> None:
    """Replace every ``nn.Linear`` descendant in place with an ``SNLinear``."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, SNLinear.from_linear(child, n_power_iter))
        else:
            apply_spectral_norm(child, n_power_iter)


def fold_spectral_norm(module: "nn.Module") -> None:
    """Replace every ``SNLinear`` descendant with its folded plain Linear."""
    for name, child in module.named_children():
        if isinstance(child, SNLinear):
            setattr(module, name, child.fold())
        else:
            fold_spectral_norm(child)


# --------------------------------------------------------------------------
# Random-feature RBF Gaussian process head
# --------------------------------------------------------------------------
class RandomFeatureGP(_Module):
    """Laplace-approximated RBF GP over the penultimate features.

    Random Fourier features approximate the RBF kernel. ``beta`` is the MAP
    output weight (trained by SGD); the posterior precision is the running
    mean of ``phi^T phi`` over the training features, and the predictive
    variance is ``ridge * phi^T inv(ridge*I + precision) phi`` (Edward2,
    ``LaplaceRandomFeatureCovariance``, Gaussian likelihood).
    """

    def __init__(self, in_dim: int, num_features: int = 1024,
                 length_scale: float | None = None, ridge: float = 1e-6,
                 normalize_input: bool = True, seed: int = 42):
        super().__init__()
        if length_scale is None:
            length_scale = math.sqrt(in_dim)
        self.in_dim = in_dim
        self.num_features = num_features
        self.length_scale = float(length_scale)
        self.ridge = float(ridge)
        self.normalize_input = normalize_input
        # Per-dimension standardization de-anisotropizes the BERT features:
        # the pooled output has a dominant common direction (effective rank
        # ~1-17 of 384), which collapses the random features to rank 1 and
        # makes the GP variance meaningless. BatchNorm (affine=False) removes
        # the per-dim mean/scale using batch stats in training and running
        # stats at inference, and exports as a plain BatchNormalization op.
        self.bn = (nn.BatchNorm1d(in_dim, affine=False, momentum=0.1)
                   if normalize_input else nn.Identity())
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(num_features, in_dim, generator=g) / self.length_scale
        b = torch.rand(num_features, generator=g) * 2 * math.pi
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.beta = nn.Parameter(torch.zeros(num_features, 1))
        self.out_bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("precision", torch.zeros(num_features, num_features))
        self.register_buffer("precision_inv", torch.eye(num_features))
        self.feature_scale = math.sqrt(2.0 / num_features)

    def features(self, h):
        h = self.bn(h)
        return self.feature_scale * torch.cos(h @ self.W.t() + self.b)

    def map_logit(self, h):
        """(MAP logit f, random features phi) for a batch of features."""
        phi = self.features(h)
        return phi @ self.beta + self.out_bias, phi

    @torch.no_grad()
    def set_exact_precision(self, phi) -> None:
        """Precision = mean(phi^T phi) over the training features (exact)."""
        self.precision.copy_((phi.t() @ phi) / phi.shape[0])

    @torch.no_grad()
    def finalize_covariance(self) -> None:
        d = self.num_features
        eye = torch.eye(d, device=self.precision.device,
                        dtype=self.precision.dtype)
        self.precision_inv.copy_(torch.linalg.inv(self.ridge * eye + self.precision))

    def scaled_logit(self, h):
        """(probit-scaled logit z, predictive variance) for a batch."""
        phi = self.features(h)
        f = phi @ self.beta + self.out_bias
        var = self.ridge * ((phi @ self.precision_inv) * phi).sum(-1, keepdim=True)
        return f / torch.sqrt(1.0 + PI_OVER_8 * var), var


class SNGPCrossEncoder(_Module):
    """MiniLM cross-encoder with spectral-normalized encoder + RBF-GP head."""

    def __init__(self, base: str = BASE, num_features: int = 1024,
                 length_scale: float | None = None, ridge: float = 1e-6,
                 n_power_iter: int = 1, normalize_input: bool = True,
                 seed: int = 42):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        base_net = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=2, ignore_mismatched_sizes=True)
        if hasattr(base_net, "bert"):
            encoder = base_net.bert
        elif hasattr(base_net, "roberta"):
            encoder = base_net.roberta
        elif hasattr(base_net, "model"):
            encoder = base_net.model
        else:
            encoder = base_net
        apply_spectral_norm(encoder, n_power_iter)
        self.encoder = encoder
        dim = getattr(encoder.config, "hidden_size", 384)
        self.dropout = nn.Dropout(p=0.1)
        self.gp = RandomFeatureGP(dim, num_features, length_scale, ridge,
                                  normalize_input, seed=seed)

    def encode(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {}
        if (token_type_ids is not None
                and hasattr(self.encoder, "embeddings")
                and hasattr(self.encoder.embeddings, "token_type_embeddings")):
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           **kwargs)
        hidden = out[0]
        # Masked mean pooling. The [CLS]/pooler output is nearly constant
        # across pairs for this backbone (effective rank ~1 after
        # standardization), so mean pooling is the only pooling that leaves
        # the GP input with usable per-dimension variance.
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.dropout(pooled)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        h = self.encode(input_ids, attention_mask, token_type_ids)
        return self.gp.map_logit(h)


class SNGPInference(_Module):
    """Export wrapper: emits ``[log(1-p), log(p)]`` from the GP posterior."""

    def __init__(self, net: SNGPCrossEncoder, scale: bool = True):
        super().__init__()
        self.net = net
        self.scale = scale

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        h = self.net.encode(input_ids, attention_mask, token_type_ids)
        if self.scale:
            z, _var = self.net.gp.scaled_logit(h)
        else:
            # Ablation: MAP logit only, no GP variance scaling (ridge->0).
            z, _phi = self.net.gp.map_logit(h)
        p = torch.sigmoid(z).clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.cat([torch.log(1.0 - p), torch.log(p)], dim=-1)


# --------------------------------------------------------------------------
# Data / metrics helpers
# --------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def auc(scores, labels) -> float:
    """Mann-Whitney AUC; ``labels == 1`` is the positive class."""
    import numpy as np

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def encode_batch(tok, requests, tools, max_len, device):
    enc = tok(requests, tools, truncation=True, max_length=max_len,
              padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("sngp", "plain"), default="sngp")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--base", type=str, default=BASE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-miss-weight", type=float, default=NEAR_MISS_WEIGHT)
    parser.add_argument("--num-features", type=int, default=1024,
                        help="random Fourier feature dimension D")
    parser.add_argument("--length-scale", type=float, default=None,
                        help="RBF length scale; default sqrt(hidden_size)")
    parser.add_argument("--ridge", type=float, default=1e-3,
                        help="GP covariance ridge penalty (prior precision)")
    parser.add_argument("--n-power-iter", type=int, default=1)
    parser.add_argument("--no-normalize-input", action="store_true",
                        help="skip LayerNorm on the GP input")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap train rows (smoke tests)")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--no-scale", action="store_true",
                        help="export the MAP logit only (no GP variance scaling); "
                             "attribution ablation for the SNGP arm")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    rows = load_rows(args.pairs)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if args.limit:
        train = train[:args.limit]
        dev = dev[:max(64, args.limit // 4)]
    print(f"corpus: {len(rows)} rows ({len(train)} train / {len(dev)} dev) "
          f"from {_rel(args.pairs)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    tok = AutoTokenizer.from_pretrained(args.base)

    def weight_of(row: dict) -> float:
        return (args.near_miss_weight if row["source"] in NEAR_MISS_SOURCES
                else 1.0)

    if args.arm == "sngp":
        net = SNGPCrossEncoder(
            args.base, num_features=args.num_features,
            length_scale=args.length_scale, ridge=args.ridge,
            n_power_iter=args.n_power_iter,
            normalize_input=not args.no_normalize_input, seed=args.seed,
        ).to(device)
        enc_params = list(net.encoder.parameters())
        head_params = list(net.gp.parameters())
        opt = torch.optim.AdamW(
            [{"params": enc_params, "lr": args.lr},
             {"params": head_params, "lr": args.head_lr}],
            weight_decay=0.01)
    else:
        net = AutoModelForSequenceClassification.from_pretrained(
            args.base, num_labels=2, ignore_mismatched_sizes=True).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    net.train()
    for epoch in range(args.epochs):
        torch.manual_seed(args.seed + epoch)
        perm = torch.randperm(len(train)).tolist()
        total = 0.0
        n_batches = 0
        t0 = time.time()
        for start in range(0, len(train), args.batch):
            chunk = [train[i] for i in perm[start:start + args.batch]]
            enc = encode_batch(tok, [r["request"] for r in chunk],
                               [r["tool"] for r in chunk], args.max_len, device)
            labels = torch.tensor([r["label"] for r in chunk],
                                  dtype=torch.float).to(device)
            weights = torch.tensor([weight_of(r) for r in chunk],
                                   dtype=torch.float).to(device)
            if args.arm == "sngp":
                logit, _phi = net(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    token_type_ids=enc.get("token_type_ids"))
                logit = logit.view(-1)
            else:
                out = net(**enc)
                logit = out.logits[:, 1] - out.logits[:, 0]
            loss = F.binary_cross_entropy_with_logits(
                logit, labels, weight=weights, reduction="mean")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        print(f"epoch {epoch + 1}: loss {total / max(1, n_batches):.4f} "
              f"({time.time() - t0:.1f}s)")
    net.eval()

    # Frozen GP covariance from the training features (exact mean).
    if args.arm == "sngp":
        with torch.no_grad():
            phis = []
            for start in range(0, len(train), 64):
                chunk = train[start:start + 64]
                enc = encode_batch(tok, [r["request"] for r in chunk],
                                   [r["tool"] for r in chunk],
                                   args.max_len, device)
                h = net.encode(input_ids=enc["input_ids"],
                               attention_mask=enc.get("attention_mask"),
                               token_type_ids=enc.get("token_type_ids"))
                phis.append(net.gp.features(h).float().cpu())
            phi_all = torch.cat(phis, dim=0).to(device)
            net.gp.set_exact_precision(phi_all)
            net.gp.finalize_covariance()
        del phi_all

    # Dev diagnostics: does the GP variance separate near-misses from calls?
    diag = {}
    if args.arm == "sngp":
        with torch.no_grad():
            map_probs, scaled_probs, vars_ = [], [], []
            for start in range(0, len(dev), 64):
                chunk = dev[start:start + 64]
                enc = encode_batch(tok, [r["request"] for r in chunk],
                                   [r["tool"] for r in chunk],
                                   args.max_len, device)
                h = net.encode(input_ids=enc["input_ids"],
                               attention_mask=enc.get("attention_mask"),
                               token_type_ids=enc.get("token_type_ids"))
                f, _phi = net.gp.map_logit(h)
                z, var = net.gp.scaled_logit(h)
                map_probs += torch.sigmoid(f.view(-1)).cpu().tolist()
                scaled_probs += torch.sigmoid(z.view(-1)).cpu().tolist()
                vars_ += var.view(-1).cpu().tolist()
        near = [i for i, r in enumerate(dev)
                if r["label"] == 0 and r["source"] in NEAR_MISS_SOURCES]
        pos = [i for i, r in enumerate(dev) if r["label"] == 1]
        y = [1] * len(pos) + [0] * len(near)
        idx = pos + near
        diag = {
            "n_pos": len(pos), "n_near": len(near),
            "auc_map_prob": auc([map_probs[i] for i in idx], y),
            "auc_scaled_prob": auc([scaled_probs[i] for i in idx], y),
            # variance should be HIGHER on near-misses: positive class = near
            "auc_variance": auc([vars_[i] for i in near] + [vars_[i] for i in pos],
                                [1] * len(near) + [0] * len(pos)),
            "var_pos_mean": float(sum(vars_[i] for i in pos) / max(1, len(pos))),
            "var_near_mean": float(sum(vars_[i] for i in near) / max(1, len(near))),
        }
        print("dev diagnostics: " + json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in diag.items()}))

    result = {"arm": args.arm, "diag": diag}
    if not args.skip_export:
        out = args.out
        if out is None:
            suffix = "sngp" if args.arm == "sngp" else "plain"
            out = ROOT / "data" / "incoming" / "models" / f"tool-match-{suffix}-int8.onnx"
        if args.arm == "sngp":
            fold_spectral_norm(net.encoder)
            export_net = SNGPInference(net, scale=not args.no_scale)
        else:
            export_net = net
        int8_path, fp32_path = export(export_net, tok, out, args.max_len)
        digest = hashlib.sha256(int8_path.read_bytes()).hexdigest()
        (int8_path.parent / (int8_path.name + ".sha256")).write_text(
            digest + "\n", encoding="utf-8")
        tok_path = int8_path.parent / (
            int8_path.name.replace("-int8.onnx", "-tokenizer.json"))
        if hasattr(tok, "backend_tokenizer"):
            tok.backend_tokenizer.save(str(tok_path))
        elif hasattr(tok, "save"):
            tok.save(str(tok_path))
        result.update({"int8": str(int8_path), "fp32": str(fp32_path),
                       "bytes": int8_path.stat().st_size, "sha256": digest})
        print(f"exported {int8_path.name} ({int8_path.stat().st_size} bytes, "
              f"sha256 {digest[:16]})")
        print(f"fp32 parent {fp32_path.name}")
        print(f"tokenizer {tok_path.name}")

    # Persist the run metadata next to the artifact for the report.
    meta_path = ROOT / "data" / "incoming" / "models" / f"tool-match-{args.arm}-run.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "arm": args.arm, "base": args.base, "epochs": args.epochs,
        "batch": args.batch, "lr": args.lr, "head_lr": args.head_lr,
        "seed": args.seed, "max_len": args.max_len,
        "num_features": args.num_features, "ridge": args.ridge,
        "length_scale": args.length_scale, "n_power_iter": args.n_power_iter,
        "normalize_input": not args.no_normalize_input,
        "near_miss_weight": args.near_miss_weight,
        "train_rows": len(train), "dev_rows": len(dev),
        "device": device, "diag": diag,
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"run metadata {_rel(meta_path)}")
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

    dynamic_axes = {k: {0: "batch", 1: "seq"} for k in input_names}
    dynamic_axes["logits"] = {0: "batch"}
    torch.onnx.export(
        net, inputs, fp32_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    return out, fp32_path


if __name__ == "__main__":
    sys.exit(main())
