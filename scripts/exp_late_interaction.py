#!/usr/bin/env python3
"""Late-interaction (ColBERT-style) tool-match arm.

The shipped guard scores a (request, tool) pair with one cross-encoder vector.
NevIR (EACL 2024, aclanthology 2024.eacl-long.139) measured that on *negation*
near-misses the ordering is cross-encoders > late-interaction > bi-encoders.
The guard's failure mode is exactly that quadrant: a tool that structurally
fits (right domain, right shape) but cannot serve the goal, because the request
negates or contradicts what the tool does. So late interaction is a live
hypothesis, not a style choice.

What this arm does
------------------
Encode the request and the tool as per-token vectors (MiniLM-L6 backbone plus a
projection to ``--dim``), L2-normalize them, and score with MaxSim: for every
request token take the max cosine to any tool token and sum. Training uses the
same corpus, the same near-miss up-weighting and the same optimizer settings as
``scripts/train_tool_match.py``; only the scoring head changes.

Late interaction, not a cross-encoder with a MaxSim pooling head
----------------------------------------------------------------
A ColBERT encoder must not let request tokens attend to tool tokens (that is
what makes the tool side precomputable). This model enforces that with a
block-diagonal additive attention mask and per-segment position ids, so the
query half of the hidden states is numerically identical to encoding the query
alone, and the tool half to encoding the tool alone. The tool vectors are then
a function of the tool only, which is what the precompute measurement assumes.

The artifact keeps the production cross-encoder's input/output contract
(``input_ids``/``attention_mask``/``token_type_ids`` in, a 2-logit tensor out
where the match probability is ``sigmoid(logit[1] - logit[0])``), so
``scripts/bench_tool_abstention.py --model <artifact>`` measures it directly
through the production ``ToolMatcher`` with no runtime change.

This script only writes a candidate artifact and its sidecars under
``data/incoming/models/``. It never touches ``models/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _Module = nn.Module
except ImportError:  # pragma: no cover - training-only dependency
    torch = None
    nn = None
    F = None
    _Module = object

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "tool-match-lateint-int8.onnx"
DEFAULT_REPORT = ROOT / "reports" / "exp-late-interaction.md"
NEAR_MISS_SOURCES = {"bfcl_distractor_aligned", "bfcl_irrelevance_negatives"}
NEAR_MISS_WEIGHT = 5.0


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


class LateInteractionEncoder(_Module):
    """Shared MiniLM backbone + projection. One forward per span."""

    def __init__(self, bert, dim: int):
        super().__init__()
        self.bert = bert
        self.proj = nn.Linear(bert.config.hidden_size, dim)

    def forward(self, input_ids, attention_mask, position_ids=None):
        emb = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=None,
        )
        hidden = self.bert.encoder(emb, attention_mask=None)[0]
        return F.normalize(self.proj(hidden), dim=-1)


class LateInteractionModel(_Module):
    """Per-token vectors, MaxSim score, calibrated to a match logit."""

    def __init__(self, bert, dim: int):
        super().__init__()
        self.encoder = LateInteractionEncoder(bert, dim)
        # learnable temperature + bias turn MaxSim (a cosine sum in [0, n_q])
        # into a calibrated logit the production sigmoid can read.
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def maxsim(self, input_ids, attention_mask, token_type_ids):
        B, L = input_ids.shape
        valid = attention_mask.bool()
        is_q = (token_type_ids == 0).long()
        is_t = (token_type_ids == 1).long()
        # Per-segment position ids: query tokens 0..nq-1, tool tokens 0..nt-1.
        # This is what lets the tool half be a function of the tool alone.
        pos_q = torch.cumsum(is_q, dim=1) - 1
        pos_t = torch.cumsum(is_t, dim=1) - 1
        position_ids = torch.where(token_type_ids == 1, pos_t, pos_q).clamp(min=0)
        vecs = self.encoder(input_ids, attention_mask, position_ids)  # [B,L,D]
        sim = torch.bmm(vecs, vecs.transpose(1, 2))  # [B,L,L]
        tmask = (token_type_ids == 1).unsqueeze(1) & valid.unsqueeze(2)
        sim = sim.masked_fill(~tmask, -1.0)
        per_q = sim.max(dim=2).values * ((token_type_ids == 0) & valid)
        qmask = (token_type_ids == 0) & valid
        score = per_q.sum(dim=1) / qmask.sum(dim=1).clamp(min=1)
        return score

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        B, L = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        valid = attention_mask.bool()
        neg = torch.finfo(self.scale.dtype).min
        same = token_type_ids.unsqueeze(2) == token_type_ids.unsqueeze(1)
        allow = same & valid.unsqueeze(1) & valid.unsqueeze(2)
        mask4d = torch.where(
            allow,
            torch.zeros((), dtype=self.scale.dtype, device=input_ids.device),
            torch.full((), neg, dtype=self.scale.dtype, device=input_ids.device),
        ).unsqueeze(1)
        emb = self.encoder.bert.embeddings(
            input_ids=input_ids,
            position_ids=self._positions(input_ids, token_type_ids),
            token_type_ids=None,
        )
        hidden = self.encoder.bert.encoder(emb, attention_mask=mask4d)[0]
        vecs = F.normalize(self.encoder.proj(hidden), dim=-1)
        sim = torch.bmm(vecs, vecs.transpose(1, 2))
        tmask = (token_type_ids == 1).unsqueeze(1) & valid.unsqueeze(2)
        sim = sim.masked_fill(~tmask, -1.0)
        qmask = (token_type_ids == 0) & valid
        per_q = sim.max(dim=2).values * qmask
        score = per_q.sum(dim=1) / qmask.sum(dim=1).clamp(min=1)
        logit = self.scale * score + self.bias
        # Production convention (tool_match.ToolMatcher._predict): the match
        # probability is sigmoid(logits[1] - logits[0]), i.e. class 1 is the
        # match. Emit [0, logit] so the guard reads sigmoid(logit) directly.
        return torch.stack([torch.zeros_like(logit), logit], dim=-1)

    @staticmethod
    def _positions(input_ids, token_type_ids):
        is_q = (token_type_ids == 0).long()
        is_t = (token_type_ids == 1).long()
        pos_q = torch.cumsum(is_q, dim=1) - 1
        pos_t = torch.cumsum(is_t, dim=1) - 1
        return torch.where(token_type_ids == 1, pos_t, pos_q).clamp(min=0)


class ClsControl(_Module):
    """Same backbone, single-vector [CLS] head. Same-recipe control for the
    late-interaction arm, so the comparison isolates the scoring head."""

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        pooled = out[0][:, 0]
        return self.classifier(pooled)


def build_model(head: str, base: str, dim: int):
    from transformers import AutoModel
    bert = AutoModel.from_pretrained(base)
    if head == "lateint":
        return LateInteractionModel(bert, dim)
    return ClsControl(bert)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--base", type=str, default=BASE)
    parser.add_argument("--head", choices=("lateint", "cls"), default="lateint")
    parser.add_argument("--dim", type=int, default=96,
                        help="projection width for the late-interaction head")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-miss-weight", type=float, default=NEAR_MISS_WEIGHT)
    parser.add_argument("--contrastive", type=float, default=0.0,
                        help="in-batch InfoNCE weight (lateint head only)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap training rows (smoke tests)")
    parser.add_argument("--save-torch", type=Path, default=None,
                        help="also save the trained state dict (diagnostics)")
    parser.add_argument("--latency", action="store_true",
                        help="also measure the decomposed CPU latency")
    parser.add_argument("--report-append", action="store_true",
                        help="append to --report instead of overwriting")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    rows = load_rows(args.pairs)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if args.limit:
        train = train[:args.limit]
    print(f"corpus: {len(rows)} rows ({len(train)} train / {len(dev)} dev) "
          f"from {args.pairs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device; training on CPU", file=sys.stderr)
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    tok = AutoTokenizer.from_pretrained(args.base)
    net = build_model(args.head, args.base, args.dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def weight_of(row: dict) -> float:
        return args.near_miss_weight if row["source"] in NEAR_MISS_SOURCES else 1.0

    net.train()
    for epoch in range(args.epochs):
        torch.manual_seed(args.seed + epoch)
        perm = torch.randperm(len(train)).tolist()
        total = 0.0
        for start in range(0, len(train), args.batch):
            chunk = [train[i] for i in perm[start:start + args.batch]]
            enc = tok([r["request"] for r in chunk], [r["tool"] for r in chunk],
                      truncation=True, max_length=args.max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = torch.tensor([r["label"] for r in chunk]).to(device)
            weights = torch.tensor([weight_of(r) for r in chunk],
                                   dtype=torch.float).to(device)
            logits = net(input_ids=enc["input_ids"],
                         attention_mask=enc.get("attention_mask"),
                         token_type_ids=enc.get("token_type_ids"))
            diff = logits[:, 1] - logits[:, 0]
            loss = F.binary_cross_entropy_with_logits(diff, labels.float(),
                                                      reduction="none")
            loss = (loss * weights).sum() / weights.sum()
            if args.contrastive and args.head == "lateint":
                loss = loss + args.contrastive * _contrastive_term(net, enc)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        print(f"epoch {epoch + 1}: loss {total / max(1, len(train) // args.batch):.4f}",
              flush=True)
    net.eval()

    def pair_rates(subset: list[dict], threshold: float):
        tp = fn = tn = fp = 0
        with torch.no_grad():
            for i in range(0, len(subset), args.batch):
                chunk = subset[i:i + args.batch]
                enc = tok([r["request"] for r in chunk], [r["tool"] for r in chunk],
                          truncation=True, max_length=args.max_len, padding=True,
                          return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = net(input_ids=enc["input_ids"],
                             attention_mask=enc.get("attention_mask"),
                             token_type_ids=enc.get("token_type_ids"))
                probs = torch.sigmoid(logits[:, 1] - logits[:, 0]).cpu().tolist()
                for r, p in zip(chunk, probs):
                    matched = p >= threshold
                    if r["label"] == 1:
                        tp += matched
                        fn += not matched
                    else:
                        fp += matched
                        tn += not matched
        prec = tp / max(1, tp + fp)
        return 100 * tp / max(1, tp + fn), 100 * tn / max(1, tn + fp), prec

    lines = ["# Late-interaction tool-match arm", "",
             f"corpus: `{_rel(args.pairs)}` ({len(rows)} rows)",
             f"train {len(train)} / dev {len(dev)}",
             f"base {args.base} · head {args.head} · dim {args.dim} · "
             f"epochs {args.epochs} · batch {args.batch} · lr {args.lr} · "
             f"max_len {args.max_len} · seed {args.seed}",
             f"near-miss weight {args.near_miss_weight} for "
             f"{sorted(NEAR_MISS_SOURCES)} · contrastive {args.contrastive}",
             f"device {device}", "",
             "Held-out rows only, scored at the pair level.", "",
             "| threshold | positives kept % | negatives rejected % | precision |",
             "|---|---|---|---|"]
    for threshold in (0.3, 0.5, 0.6, 0.7, 0.9):
        kept, rejected, prec = pair_rates(dev, threshold)
        lines.append(f"| {threshold} | {kept:.1f} | {rejected:.1f} | {prec:.3f} |")
    lines.append("")

    if args.save_torch:
        torch.save({"head": args.head, "dim": args.dim, "base": args.base,
                    "state": net.state_dict()}, args.save_torch)
        print(f"saved torch state to {args.save_torch}")

    int8_path, fp32_path = export(net, tok, args.out, args.max_len)
    digest = hashlib.sha256(int8_path.read_bytes()).hexdigest()
    (int8_path.parent / (int8_path.name + ".sha256")).write_text(
        digest + "\n", encoding="utf-8")
    tok_path = int8_path.parent / (
        int8_path.name.replace("-int8.onnx", "-tokenizer.json"))
    if hasattr(tok, "backend_tokenizer"):
        tok.backend_tokenizer.save(str(tok_path))
    elif hasattr(tok, "save"):
        tok.save(str(tok_path))
    lines += [f"Exported {int8_path.name} ({int8_path.stat().st_size} bytes)",
              f"sha256 {digest[:16]}", f"fp32 parent {fp32_path.name}",
              f"tokenizer sidecar {tok_path.name}", ""]

    if args.latency:
        lines += latency_breakdown(net, tok, args)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.report_append else "w"
    with args.report.open(mode, encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _contrastive_term(net, enc):
    """In-batch InfoNCE: each pair's MaxSim against its own tool as the
    positive, every other tool in the batch as a negative."""
    import torch
    import torch.nn.functional as F
    ids, am, tt = enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]
    B = ids.shape[0]
    qs = [_span_vectors(net, ids[i], am[i], tt[i], 0) for i in range(B)]
    ts = [_span_vectors(net, ids[i], am[i], tt[i], 1) for i in range(B)]
    logit_mat = torch.stack([
        torch.cat([_maxsim(qs[i], ts[j]) for j in range(B)]) for i in range(B)
    ])
    target = torch.arange(B, device=ids.device)
    return F.cross_entropy(logit_mat, target)


def _span_vectors(net, ids_row, am_row, tt_row, seg):
    """Encode one segment alone, exactly as ColBERT would: only that
    segment's tokens, positions starting at 0, no cross-segment attention.
    Numerically equal to the corresponding half of the block-diagonal pass."""
    keep = (tt_row == seg) & am_row.bool()
    sub = ids_row[keep].unsqueeze(0)
    pos = torch.arange(sub.shape[1], device=sub.device).unsqueeze(0)
    emb = net.encoder.bert.embeddings(input_ids=sub, position_ids=pos,
                                      token_type_ids=None)
    h = net.encoder.bert.encoder(emb, attention_mask=None)[0]
    return F.normalize(net.encoder.proj(h), dim=-1)  # [1, n, D]


def _maxsim(q, t):
    """Sum over query tokens of the max cosine to any tool token."""
    sim = torch.einsum("bld,btd->blt", q, t)  # [1, nq, nt]
    return sim.max(dim=2).values.sum(dim=1)


def latency_breakdown(net, tok, args) -> list[str]:
    """Decomposed CPU latency in int8, with a precomputed tool side.

    The tool catalogue is fixed, so the tool vectors are computed once and
    reused; a request pays only its own encoder pass plus MaxSim. This
    exports the single-span encoder (the same weights) as its own int8 graph
    and times each stage on one thread.
    """
    import time
    import numpy as np
    import onnxruntime as ort

    net.to("cpu").eval()
    span_int8 = args.out.parent / args.out.name.replace("-int8.onnx", "-span-int8.onnx")
    span_fp32 = args.out.parent / args.out.name.replace("-int8.onnx", "-span.onnx")
    export_span_encoder(net, args.max_len, span_fp32, span_int8)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(span_int8), so,
                                providers=["CPUExecutionProvider"])
    _span_inputs = {i.name for i in sess.get_inputs()}

    def run_span(ids):
        a = np.array([ids], dtype=np.int64)
        feed = {"input_ids": a}
        if "attention_mask" in _span_inputs:
            feed["attention_mask"] = np.ones_like(a)
        return sess.run(None, feed)[0]

    rows = load_rows(args.pairs)
    dev = [r for r in rows if r["split"] == "dev"]
    sample = [r for r in dev if r["label"] == 1][:80]
    out = ["", "## CPU latency breakdown (int8, 1 thread, tool side precomputed)",
           f"single-span graph `{span_int8.name}` "
           f"({span_int8.stat().st_size} bytes)", "",
           "| stage | p50 ms | p95 ms |", "|---|---|---|"]
    reps = 30

    def timed(fn):
        xs = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t0) * 1000)
        xs.sort()
        return xs[len(xs) // 2], xs[min(len(xs) - 1, int(len(xs) * 0.95))]

    enc_q, enc_t, maxsim, total = [], [], [], []
    for r in sample:
        e = tok([r["request"]], [r["tool"]], truncation=True,
                max_length=args.max_len, padding=True, return_tensors="pt")
        ids = e["input_ids"][0].tolist()
        tt = e["token_type_ids"][0].tolist()
        q_ids = [i for i, t in zip(ids, tt) if t == 0]
        t_ids = [i for i, t in zip(ids, tt) if t == 1]
        qv = run_span(q_ids)
        tv = run_span(t_ids)
        def ms():
            sim = np.einsum("bld,btd->blt", qv, tv)
            sim.max(axis=2).sum(axis=1)
        enc_q.append(timed(lambda: run_span(q_ids)))
        enc_t.append(timed(lambda: run_span(t_ids)))
        maxsim.append(timed(ms))
        total.append(timed(lambda: (run_span(q_ids), ms())))
    for name, xs in (("query encoder (per request)", enc_q),
                     ("tool encoder (amortized, once per tool)", enc_t),
                     ("MaxSim (numpy)", maxsim),
                     ("per-request total with precompute", total)):
        p50 = sorted(v[0] for v in xs)[len(xs) // 2]
        p95 = sorted(v[1] for v in xs)[len(xs) // 2]
        out.append(f"| {name} | {p50:.2f} | {p95:.2f} |")
    return out


class SpanEncoder(_Module):
    """One span in, per-token vectors out. Exported separately so the
    precompute path can be timed without the concatenated-pair graph."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask=None):
        pos = torch.arange(input_ids.shape[1],
                           device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        emb = self.encoder.bert.embeddings(input_ids=input_ids, position_ids=pos,
                                           token_type_ids=None)
        h = self.encoder.bert.encoder(emb, attention_mask=None)[0]
        return F.normalize(self.encoder.proj(h), dim=-1)


def export_span_encoder(net, max_len: int, fp32_path: Path, int8_path: Path) -> Path:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    span = SpanEncoder(net.encoder).eval()
    dummy = torch.ones((1, 8), dtype=torch.long)
    torch.onnx.export(
        span, (dummy, dummy), fp32_path,
        input_names=["input_ids", "attention_mask"], output_names=["vectors"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "vectors": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


def export(net, tok, out: Path, max_len: int):
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
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    inputs = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])
    dynamic_axes = {k: {0: "batch", 1: "seq"} for k in input_names}
    dynamic_axes["logits"] = {0: "batch"}
    torch.onnx.export(
        net, inputs, fp32_path,
        input_names=input_names, output_names=["logits"],
        dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    return out, fp32_path


if __name__ == "__main__":
    sys.exit(main())
