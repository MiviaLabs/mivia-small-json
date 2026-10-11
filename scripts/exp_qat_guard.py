#!/usr/bin/env python3
"""QAT arm for the tool-match guard: fake-quant training, then int8 export.

The shipped guard is an int8 ONNX cross-encoder produced by *dynamic
post-training quantization* (`quantize_dynamic`: QInt8 weights, per-batch
dynamic activation ranges). Two things are wrong with that path for a small
encoder:

1. PTQ loses accuracy on small encoders (Q8BERT, arXiv:1909.12993;
   nlp-architect report <= 1.32% relative on GLUE). QAT recovers it.
2. Dynamic activation quantization is *not batch-invariant*: the activation
   scale is recomputed from whatever rows share a call, so the same pair
   scores differently at batch 1 and batch 2. Measured on the shipped
   artifact: max |diff| 0.33 and 4/400 decision flips on held-out positives.

This script fine-tunes the same backbone with **fake quantization in the
forward pass** (per-channel symmetric int8 weights, per-tensor affine int8
activations, straight-through estimator) and exports an ONNX graph whose
QuantizeLinear/DequantizeLinear scales are *constants* learned during
training. Constant scales are what make the graph batch-invariant.

Only torch / transformers / onnx / onnxruntime are used; no new dependency.
The exported QDQ graph stores float weights, so a small graph rewrite folds
each constant weight ``QuantizeLinear`` into an int8 initializer, bringing
the file back to the shipped artifact's size class.

Nothing here writes to ``models/``. The candidate lands in
``data/incoming/models/`` with a sha256 sidecar; promotion is a separate,
deliberate step.

Usage:
    python scripts/exp_qat_guard.py --selectivenet --epochs 5 --batch 16
    python scripts/exp_qat_guard.py --no-qat --out data/incoming/models/tool-match-ptq5-int8.onnx
    python scripts/exp_qat_guard.py --batch-check data/incoming/models/tool-match-qat-int8.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.ao.quantization.observer import (
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
    )
    _Module = nn.Module
except ImportError:  # pragma: no cover - torch is a training extra
    torch = None
    nn = None
    F = None
    MovingAverageMinMaxObserver = None
    MovingAveragePerChannelMinMaxObserver = None
    _Module = object

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "tool-match-qat-int8.onnx"
NEAR_MISS_SOURCES = {"bfcl_distractor_aligned", "bfcl_irrelevance_negatives"}
NEAR_MISS_WEIGHT = 5.0
QMIN, QMAX = -128, 127


def _load_train_module():
    """Import scripts/train_tool_match.py by path (reuse, do not edit)."""
    spec = importlib.util.spec_from_file_location(
        "train_tool_match", ROOT / "scripts" / "train_tool_match.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Fake-quant modules (manual observers, straight-through estimator)
# --------------------------------------------------------------------------
class _QATBase(_Module):
    """Shared fake-quant state: warmup -> fake-quant -> frozen buffers."""

    def __init__(self):
        super().__init__()
        self._frozen = False
        self.apply_fake_quant = False


class QATLinear(_QATBase):
    """nn.Linear with fake-quantized weights (per-channel) and input (per-tensor).

    Three states:

    * training, ``apply_fake_quant`` False: observers collect min/max, weights
      pass through unchanged (warmup);
    * training, ``apply_fake_quant`` True: observers keep updating, and the
      forward applies fake quant with the current observed scales;
    * ``_frozen`` True (export): the scale/zero-point are constant buffers and
      the forward reads only them, so ONNX export emits Q/DQ with constant
      parameters.
    """

    def __init__(self, lin, act_obs, w_obs):
        super().__init__()
        self.lin = lin
        self.act_obs = act_obs
        self.w_obs = w_obs
        self.register_buffer("_a_scale", torch.tensor(1.0))
        self.register_buffer("_a_zp", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("_w_scale", torch.ones(lin.out_features))
        self.register_buffer("_w_zp", torch.zeros(lin.out_features, dtype=torch.int32))

    def forward(self, x):
        w = self.lin.weight
        if self._frozen:
            a_scale, a_zp = self._a_scale, self._a_zp
            w_scale, w_zp = self._w_scale, self._w_zp
        else:
            if self.training:
                self.act_obs(x.detach())
                self.w_obs(w.detach())
            a_scale, a_zp = self.act_obs.calculate_qparams()
            w_scale, w_zp = self.w_obs.calculate_qparams()
        if self._frozen or self.apply_fake_quant:
            wq = torch.fake_quantize_per_channel_affine(
                w, w_scale.to(w.dtype).reshape(-1),
                w_zp.to(torch.int32).reshape(-1), 0, QMIN, QMAX)
            xq = torch.fake_quantize_per_tensor_affine(
                x, a_scale.to(x.dtype).reshape(()),
                a_zp.to(torch.int32).reshape(()), QMIN, QMAX)
        else:
            wq, xq = w, x
        return F.linear(xq, wq, self.lin.bias)

    @torch.no_grad()
    def freeze(self):
        a_scale, a_zp = self.act_obs.calculate_qparams()
        w_scale, w_zp = self.w_obs.calculate_qparams()
        self._a_scale.copy_(a_scale.to(self._a_scale.dtype).reshape(()))
        self._a_zp.copy_(a_zp.to(torch.int32).reshape(()))
        self._w_scale.copy_(w_scale.to(self._w_scale.dtype).reshape(-1))
        self._w_zp.copy_(w_zp.to(torch.int32).reshape(-1))
        self._frozen = True


class QATEmbedding(_QATBase):
    """nn.Embedding with per-tensor fake-quantized weights (int8 table).

    The shipped dynamic-PTQ artifact quantizes the embedding tables too
    (``word_embeddings.weight_quantized`` uint8, per-tensor scale), which is
    most of the int8 file; the QAT export has to match or the artifact is
    ~2.5x larger. Per-tensor keeps the per-step observer cost small (a
    per-channel observer over the 30522-row table dominated step time).
    """

    def __init__(self, emb, w_obs):
        super().__init__()
        self.emb = emb
        self.w_obs = w_obs
        self.register_buffer("_w_scale", torch.tensor(1.0))
        self.register_buffer("_w_zp", torch.tensor(0, dtype=torch.int32))

    def forward(self, x):
        w = self.emb.weight
        if self._frozen:
            w_scale, w_zp = self._w_scale, self._w_zp
        else:
            if self.training:
                self.w_obs(w.detach())
            w_scale, w_zp = self.w_obs.calculate_qparams()
        if self._frozen or self.apply_fake_quant:
            w = torch.fake_quantize_per_tensor_affine(
                w, w_scale.to(w.dtype).reshape(()),
                w_zp.to(torch.int32).reshape(()), QMIN, QMAX)
        return F.embedding(x, w, self.emb.padding_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq,
                           self.emb.sparse)

    @torch.no_grad()
    def freeze(self):
        w_scale, w_zp = self.w_obs.calculate_qparams()
        self._w_scale.copy_(w_scale.to(self._w_scale.dtype).reshape(()))
        self._w_zp.copy_(w_zp.to(torch.int32).reshape(()))
        self._frozen = True


def swap_quantizable(module):
    """Recursively replace every nn.Linear / nn.Embedding with a QAT wrapper."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, QATLinear(
                child,
                MovingAverageMinMaxObserver(
                    dtype=torch.qint8, qscheme=torch.per_tensor_affine),
                MovingAveragePerChannelMinMaxObserver(
                    dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
            ))
        elif isinstance(child, nn.Embedding):
            setattr(module, name, QATEmbedding(
                child,
                MovingAverageMinMaxObserver(
                    dtype=torch.qint8, qscheme=torch.per_tensor_affine),
            ))
        else:
            swap_quantizable(child)


def set_qat(module, on):
    for m in module.modules():
        if isinstance(m, _QATBase):
            m.apply_fake_quant = on


def freeze_all(module):
    for m in module.modules():
        if isinstance(m, _QATBase):
            m.freeze()


# --------------------------------------------------------------------------
# ONNX export: QDQ graph, then fold constant weight QuantizeLinear to int8
# --------------------------------------------------------------------------
def fold_weight_qdq(qdq_path: Path, out_path: Path) -> dict:
    """Fold constant ``QuantizeLinear(weight)`` nodes into int8 initializers.

    The torch exporter writes the QAT weight as a float initializer plus a
    QuantizeLinear/DequantizeLinear pair with constant scale. Folding turns the
    float initializer into an int8 one, so the file matches the shipped
    artifact's size while every remaining activation Q/DQ keeps its constant
    (training-time) scale.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(qdq_path))
    graph = model.graph
    init = {i.name: i for i in graph.initializer}
    # Constants are emitted either as initializers or as Constant-node outputs;
    # the exporter picks one per tensor, so resolve both.
    const = {}
    for node in graph.node:
        if node.op_type == "Constant":
            val = next((a.t for a in node.attribute if a.name == "value"), None)
            if val is not None:
                const[node.output[0]] = val
    consumers: dict[str, list] = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    def resolve(name):
        if name in init:
            return numpy_helper.to_array(init[name]).astype(np.float64)
        if name in const:
            return numpy_helper.to_array(const[name]).astype(np.float64)
        return None

    removed_ids = set()
    new_inits = []
    folded = 0
    for node in graph.node:
        if node.op_type != "QuantizeLinear" or not node.input:
            continue
        w_name = node.input[0]
        if w_name not in init or len(node.input) < 2:
            continue
        scale = resolve(node.input[1])
        if scale is None:
            continue
        zp = resolve(node.input[2]) if len(node.input) > 2 else None
        if zp is None:
            zp = np.float64(0.0)  # symmetric weight quant
        x = numpy_helper.to_array(init[w_name]).astype(np.float64)
        axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
        if scale.ndim > 0:
            shape = [1] * x.ndim
            shape[axis] = scale.shape[0]
            scale = scale.reshape(shape)
            zp = np.broadcast_to(zp, scale.shape) if zp.ndim == 0 else zp.reshape(shape)
        q = np.clip(np.rint(x / scale) + zp, QMIN, QMAX).astype(np.int8)
        new_name = w_name + "_q8"
        new_inits.append(numpy_helper.from_array(q, name=new_name))
        for consumer in consumers.get(node.output[0], []):
            for idx, inp in enumerate(consumer.input):
                if inp == node.output[0]:
                    consumer.input[idx] = new_name
        removed_ids.add(id(node))
        folded += 1

    kept = [n for n in graph.node if id(n) not in removed_ids]
    del graph.node[:]
    graph.node.extend(kept)
    referenced = {i for n in graph.node for i in n.input}
    keep_inits = [i for i in graph.initializer if i.name in referenced]
    del graph.initializer[:]
    graph.initializer.extend(keep_inits)
    graph.initializer.extend(new_inits)
    onnx.save(model, str(out_path))
    return {"mode": "qat_qdq", "folded_weight_qnodes": folded,
            "qdq_bytes": qdq_path.stat().st_size,
            "int8_bytes": out_path.stat().st_size}


def export_qdq(net, tok, out: Path, max_len: int, selective: bool) -> dict:
    net.to("cpu").eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = tok(["a request"], ["a tool: does a thing"], return_tensors="pt",
                truncation=True, max_length=max_len, padding=True)
    names = ["input_ids", "attention_mask", "token_type_ids"]
    inputs = tuple(dummy[n] for n in names)
    output_names = ["logits", "selection"] if selective else ["logits"]
    dynamic_axes = {n: {0: "batch", 1: "seq"} for n in names}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}
    qdq_path = out.parent / (out.name.replace("-int8.onnx", "-qdq.onnx"))
    torch.onnx.export(net, inputs, qdq_path, input_names=names,
                      output_names=output_names, dynamic_axes=dynamic_axes,
                      opset_version=17, dynamo=False)
    stats = fold_weight_qdq(qdq_path, out)
    qdq_path.unlink(missing_ok=True)
    return stats


# --------------------------------------------------------------------------
# Measurement (production ToolMatcher path)
# --------------------------------------------------------------------------
def load_matcher(path: Path, tokenizer: Path | None, max_len: int):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from mivia_small_json.tool_match import ToolMatcher

    if tokenizer is None:
        cand_name = path.name.replace("-int8.onnx", "-tokenizer.json")
        cand = path.parent / cand_name
        if cand_name != path.name and cand.is_file():
            tokenizer = cand
        else:
            tokenizer = ROOT / "models" / "router-tokenizer.json"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(tokenizer))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=None)
    return ToolMatcher(session, tok, [i.name for i in session.get_inputs()])


def batched_score(matcher, pairs):
    """Production score (SelectiveNet gate applied) for a list of pairs, one call."""
    encs = matcher.tok.encode_batch([(r, t) for r, t in pairs])
    feed = {
        "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
    }
    if "token_type_ids" in matcher.inputs:
        feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
    outs = matcher.sess.run(None, feed)
    logits = outs[0]
    diff = np.clip(logits[:, 0] - logits[:, 1], -60.0, 60.0)
    probs = 1.0 / (1.0 + np.exp(diff))
    if len(outs) > 1:
        sel = outs[1].reshape(-1)
        probs = np.where(sel >= 0.50, probs, 0.0)
    return probs


def batch_invariance(matcher, pairs, threshold=0.6):
    """Max score shift and decision flips between batch-1 and batch-2 calls."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "max_abs_diff": None, "mean_abs_diff": None,
                "decision_flips": 0, "threshold": threshold}
    b1 = [float(batched_score(matcher, [p])[0]) for p in pairs]
    diffs, flips = [], 0
    for i, p in enumerate(pairs):
        j = (i + 1) % n
        b2 = float(batched_score(matcher, [p, pairs[j]])[0])
        diffs.append(abs(b1[i] - b2))
        flips += (b1[i] >= threshold) != (b2 >= threshold)
    return {"n": n, "max_abs_diff": round(max(diffs), 4),
            "mean_abs_diff": round(sum(diffs) / n, 4), "decision_flips": flips,
            "threshold": threshold}


def dev_pair_table(matcher, dev):
    scores = [float(batched_score(matcher, [(r["request"], r["tool"])])[0])
              for r in dev]
    labels = [r["label"] for r in dev]
    out = {}
    for thr in (0.3, 0.5, 0.6, 0.7, 0.9):
        tp = fn = tn = fp = 0
        for s, y in zip(scores, labels):
            matched = s >= thr
            if y == 1:
                tp += matched
                fn += not matched
            else:
                fp += matched
                tn += not matched
        out[str(thr)] = {"positives_kept": round(100 * tp / max(1, tp + fn), 1),
                         "negatives_rejected": round(100 * tn / max(1, tn + fp), 1),
                         "precision": round(tp / max(1, tp + fp), 3)}
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--base", type=str, default=BASE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-miss-weight", type=float, default=NEAR_MISS_WEIGHT)
    ap.add_argument("--qat-warmup", type=int, default=1,
                    help="epochs of float training before fake-quant turns on")
    ap.add_argument("--no-qat", action="store_true",
                    help="train float, export dynamic-PTQ int8 (control arm)")
    ap.add_argument("--selectivenet", action="store_true")
    ap.add_argument("--target-coverage", type=float, default=0.75)
    ap.add_argument("--coverage-weight", type=float, default=32.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--batch-check", type=Path, default=None,
                    help="skip training; run the batch-invariance probe on this artifact")
    ap.add_argument("--check-n", type=int, default=400)
    ap.add_argument("--limit-train", type=int, default=0, help="smoke-test row cap")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    ttm = _load_train_module()

    if args.batch_check is not None:
        rows = ttm.load_rows(args.pairs)
        dev = [r for r in rows if r["split"] == "dev" and r["label"] == 1]
        pairs = [(r["request"], r["tool"]) for r in dev[:args.check_n]]
        matcher = load_matcher(args.batch_check, None, args.max_len)
        print(f"batch-invariance {args.batch_check.name}: "
              f"{batch_invariance(matcher, pairs)}")
        return 0

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    rows = ttm.load_rows(args.pairs)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if args.limit_train:
        train = train[:args.limit_train]
        dev = dev[:max(64, args.limit_train // 8)]
    print(f"corpus: {len(rows)} rows ({len(train)} train / {len(dev)} dev)")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device; training on CPU", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    tok = AutoTokenizer.from_pretrained(args.base)
    base_net = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, ignore_mismatched_sizes=True)
    if args.selectivenet:
        net = ttm.SelectiveCrossEncoder(base_net)
    else:
        net = base_net
    n_q = 0
    if not args.no_qat:
        swap_quantizable(net)
        n_q = sum(1 for m in net.modules() if isinstance(m, _QATBase))
    net = net.to(device)
    print(f"wrapped {n_q} Linear/Embedding layers with fake-quant modules "
          f"(qat={'off' if args.no_qat else 'on'}, warmup={args.qat_warmup})")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def weight_of(row):
        return args.near_miss_weight if row["source"] in NEAR_MISS_SOURCES else 1.0

    # Pre-tokenize once: per-step tokenization was a measurable share of step
    # time and the corpus fits in memory many times over.
    print("tokenizing train set ...")
    enc_all = tok([r["request"] for r in train], [r["tool"] for r in train],
                  truncation=True, max_length=args.max_len, padding=True,
                  return_tensors="pt")
    ids_all = enc_all["input_ids"].to(device)
    am_all = enc_all["attention_mask"].to(device)
    tt_all = enc_all.get("token_type_ids")
    tt_all = tt_all.to(device) if tt_all is not None else None
    y_all = torch.tensor([r["label"] for r in train]).to(device)
    w_all = torch.tensor([weight_of(r) for r in train],
                         dtype=torch.float).to(device)

    for epoch in range(args.epochs):
        use_qat = (not args.no_qat) and epoch >= args.qat_warmup
        set_qat(net, use_qat)
        net.train()
        torch.manual_seed(args.seed + epoch)
        perm = torch.randperm(len(train)).tolist()
        total, steps = 0.0, 0
        t0 = time.time()
        for start in range(0, len(train), args.batch):
            idx = torch.tensor(perm[start:start + args.batch], device=device)
            enc = {"input_ids": ids_all[idx], "attention_mask": am_all[idx]}
            if tt_all is not None:
                enc["token_type_ids"] = tt_all[idx]
            labels = y_all[idx]
            weights = w_all[idx]
            if args.selectivenet:
                logits, selection, aux = net(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    token_type_ids=enc.get("token_type_ids"), return_aux=True)
                loss, *_ = ttm.selectivenet_loss(
                    logits=logits, selection=selection, aux_logits=aux,
                    labels=labels, weights=weights,
                    target_coverage=args.target_coverage, alpha=args.alpha,
                    coverage_weight=args.coverage_weight)
            else:
                out = net(**enc)
                loss = F.cross_entropy(out.logits, labels, reduction="none")
                loss = (loss * weights).sum() / weights.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / max(1, steps):.4f} "
              f"qat={'on' if use_qat else 'off'} ({time.time() - t0:.1f}s)")

    net.eval()
    freeze_all(net)
    if args.no_qat:
        int8_path, _fp32 = ttm.export(net, tok, args.out, args.max_len)
        stats = {"mode": "dynamic_ptq", "int8_bytes": int8_path.stat().st_size}
    else:
        stats = export_qdq(net, tok, args.out, args.max_len, args.selectivenet)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    (args.out.parent / (args.out.name + ".sha256")).write_text(digest + "\n",
                                                              encoding="utf-8")
    tok_path = args.out.parent / args.out.name.replace("-int8.onnx", "-tokenizer.json")
    tok.backend_tokenizer.save(str(tok_path))
    print(f"exported {args.out.name} ({args.out.stat().st_size} bytes) "
          f"sha256 {digest[:16]} · {stats}")

    matcher = load_matcher(args.out, tok_path, args.max_len)
    table = dev_pair_table(matcher, dev)
    dev_pairs = [(r["request"], r["tool"]) for r in dev if r["label"] == 1]
    inv = batch_invariance(matcher, dev_pairs[:args.check_n])
    print("dev pair-level (exported int8 artifact):")
    for thr, v in table.items():
        print(f"  thr {thr}: kept {v['positives_kept']}% rejected "
              f"{v['negatives_rejected']}% prec {v['precision']}")
    print(f"batch-invariance: {inv}")

    payload = {
        "artifact": str(args.out), "sha256": digest, "bytes": args.out.stat().st_size,
        "base": args.base, "selectivenet": args.selectivenet,
        "qat": not args.no_qat, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "seed": args.seed, "qat_warmup": args.qat_warmup,
        "max_len": args.max_len, "wrapped_linears": n_q, "export": stats,
        "dev_pair_table": table, "batch_invariance": inv,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"metrics -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
