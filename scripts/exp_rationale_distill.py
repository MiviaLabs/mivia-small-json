#!/usr/bin/env python3
"""Rationale-guided distillation into a ~110M cross-encoder (experiment arm).

Agrawal et al., COLING 2025 Industry (aclanthology 2025.coling-industry.12):
LLM-written rationales as an *auxiliary* training task let a 110M BERT
cross-encoder match a 7B LLM within <1% ROC-AUC at ~50x the speed, with the
rationale decoder dropped at inference. This script reproduces the two halves
of that recipe on the BFCL tool-match corpus:

1. ``generate`` -- a small local instruct model (Qwen2.5-1.5B-Instruct, no
   external API) writes a <=2-sentence rationale for every (request, tool)
   pair saying whether the tool can serve the request. Rationales are kept
   deliberately short: at small scale long chain-of-thought hurts (Small Model
   Learnability Gap, arXiv:2502.12143 / ACL Findings 2025).
2. ``train`` -- a 110M cross-encoder (``bert-base-uncased``) is trained on the
   label *plus* an auxiliary rationale-generation task. The rationale decoder
   is a small transformer decoder with cross-attention to the encoder; it is
   dropped at export, so the shipped artifact is the encoder and label head
   only.

The exported int8 ONNX is byte-compatible with the production ``ToolMatcher``
(inputs ``input_ids``/``attention_mask``/``token_type_ids``, output ``logits``
[batch, 2]) so ``scripts/bench_tool_abstention.py`` scores it unchanged.

Rationales are cached to a gitignored path (``data/incoming/models/``) because
generation is the slow step and must not be repeated. The student artifact and
its tokenizer land beside the other candidates in ``data/incoming/models/``;
nothing under ``models/`` is touched.

Usage::

    python scripts/exp_rationale_distill.py generate \
        --rationales data/incoming/models/rationales-rationale110m.jsonl
    python scripts/exp_rationale_distill.py train \
        --rationales data/incoming/models/rationales-rationale110m.jsonl \
        --out data/incoming/models/tool-match-rationale110m-int8.onnx \
        --epochs 3 --batch 8 --aux-weight 0.5
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

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _Module = nn.Module
except ImportError:  # pragma: no cover - training extra not installed
    torch = None
    nn = None
    F = None
    _Module = object

DEFAULT_PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
DEFAULT_RATIONALES = ROOT / "data" / "incoming" / "models" / "rationales-rationale110m.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "tool-match-rationale110m-int8.onnx"
TEACHER = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT = "bert-base-uncased"
# BFCL near-miss negatives are ~400 rows against ~8k easy negatives. Without an
# up-weight the loss barely notices the quadrant the abstention eval measures.
# Same recipe as scripts/train_tool_match.py, so the comparison is like-for-like.
NEAR_MISS_SOURCES = {"bfcl_distractor_aligned", "bfcl_irrelevance_negatives"}
NEAR_MISS_WEIGHT = 5.0

_SYS = ("You write very short rationales about whether an API tool can serve a "
        "user request. Always answer in one or two short, concrete sentences. "
        "No lists, no code, no preamble.")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _two_sentences(text: str, max_chars: int = 400) -> str:
    """First <=2 sentences, single-spaced, bounded. Keeps rationales short."""
    text = " ".join(text.replace("\n", " ").split())
    text = re.sub(r"^(rationale|answer)\s*[:\-]\s*", "", text, flags=re.I)
    parts = _SENT_SPLIT.split(text)
    out = " ".join(parts[:2]).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].rstrip() + "."
    return out


# --------------------------------------------------------------------------- #
# Stage 1: teacher rationale generation
# --------------------------------------------------------------------------- #
def cmd_generate(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = _rows(args.pairs)
    done: dict[str, str] = {}
    if args.rationales.is_file():
        for r in _rows(args.rationales):
            done[r["key"]] = r["rationale"]
        print(f"resuming: {len(done)} rationales already cached")

    def key_of(row: dict) -> str:
        return hashlib.sha1((row["request"] + "\x1f" + row["tool"]).encode(
            "utf-8")).hexdigest()

    todo = [r for r in rows if key_of(r) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"teacher {args.teacher}: {len(todo)} pairs to generate "
          f"({len(done)} cached, {len(rows)} total)")

    if todo:
        if not torch.cuda.is_available():
            print("WARNING: no CUDA device; teacher generation on CPU will be slow",
                  file=sys.stderr)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(args.teacher)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=torch.bfloat16).to(dev).eval()
        print(f"teacher loaded on {dev}")

        args.rationales.parent.mkdir(parents=True, exist_ok=True)
        # Append as we go so an interrupted run resumes without redoing work.
        with args.rationales.open("a", encoding="utf-8") as fh:
            for start in range(0, len(todo), args.batch):
                chunk = todo[start:start + args.batch]
                prompts = []
                for r in chunk:
                    verdict = ("The tool CAN serve this request."
                               if r["label"] == 1
                               else "The tool CANNOT serve this request.")
                    tool = " ".join(str(r["tool"]).split())[:args.tool_chars]
                    req = " ".join(str(r["request"]).split())[:args.req_chars]
                    user = (f"Tool: {tool}\n"
                            f"Request: {req}\n"
                            f"Reference answer: {verdict}\n"
                            "In one or two short sentences, give the single "
                            "concrete reason (capability match, or a missing or "
                            "mismatched argument).")
                    prompts.append(tok.apply_chat_template(
                        [{"role": "system", "content": _SYS},
                         {"role": "user", "content": user}],
                        tokenize=False, add_generation_prompt=True))
                enc = tok(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.prompt_max_len).to(dev)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                         do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                for r, ids in zip(chunk, gen):
                    text = tok.decode(ids[enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
                    rat = _two_sentences(text) or "The tool cannot serve this request."
                    fh.write(json.dumps({"key": key_of(r), "label": r["label"],
                                         "source": r["source"], "split": r["split"],
                                         "rationale": rat}) + "\n")
                fh.flush()
                n = min(start + args.batch, len(todo))
                print(f"  {n}/{len(todo)}", flush=True)

    # Report length stats over the full cache.
    cache = _rows(args.rationales)
    words = [len(r["rationale"].split()) for r in cache]
    words.sort()
    if words:
        print(f"rationales: {len(words)} · median {words[len(words)//2]} words · "
              f"p95 {words[int(len(words)*0.95)]} words · max {words[-1]}")
    print("examples:")
    for r in cache[:3]:
        print(f"  label={r['label']} · {r['rationale']}")
    return 0


# --------------------------------------------------------------------------- #
# Stage 2: student training with an auxiliary rationale decoder
# --------------------------------------------------------------------------- #
class RationaleDistill(_Module):
    """110M cross-encoder + a small rationale decoder (dropped at export).

    The encoder scores the (request, tool) pair. The decoder attends to the
    encoder's last hidden states and reconstructs the teacher's rationale
    (teacher forcing). Only the encoder and label head survive to ONNX.
    """

    def __init__(self, base, d_model: int = 256, nhead: int = 4,
                 dec_layers: int = 2, dec_ff: int = 512, dropout: float = 0.1,
                 max_rat: int = 72):
        super().__init__()
        # Delegate classification to the checkpoint's own head: BERT pools with
        # a pooler, ELECTRA takes token 0 inside its head. Keeping the original
        # seq-classification module means the trained head matches whatever the
        # backbone expects, and the ONNX export is that same module.
        self.base = base
        hidden = base.config.hidden_size
        vocab = base.config.vocab_size
        self.proj = nn.Linear(hidden, d_model)
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_rat + 2, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model, nhead, dec_ff, dropout=dropout, batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_layers)
        self.lm_head = nn.Linear(d_model, vocab)
        self.max_rat = max_rat

    def encode(self, input_ids, attention_mask, token_type_ids):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_hidden_states=True)
        return out.logits, out.hidden_states[-1]

    def logits_only(self, input_ids, attention_mask, token_type_ids):
        """Logits alone, for the decoder-free ONNX export."""
        return self.base(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).logits

    def forward(self, input_ids, attention_mask, token_type_ids,
                tgt_ids=None):
        logits, hidden = self.encode(input_ids, attention_mask, token_type_ids)
        if tgt_ids is None:
            return logits, None
        mem = self.proj(hidden)                      # [B, S, d]
        mem_pad = attention_mask == 0
        dec_in = tgt_ids[:, :-1]
        tgt_pad = dec_in == 0
        L = dec_in.size(1)
        pos = torch.arange(L, device=dec_in.device).unsqueeze(0)
        x = self.tok_emb(dec_in) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(L, L, device=dec_in.device,
                                       dtype=torch.bool), diagonal=1)
        causal = causal.float().masked_fill(causal, float("-inf"))
        dec = self.decoder(x, mem, tgt_mask=causal,
                           tgt_key_padding_mask=tgt_pad,
                           memory_key_padding_mask=mem_pad)
        return logits, self.lm_head(dec)


class LabelOnly(_Module):
    """Export wrapper: the encoder + label head, no decoder."""

    def __init__(self, model: RationaleDistill):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model.logits_only(input_ids, attention_mask, token_type_ids)


def cmd_train(args: argparse.Namespace) -> int:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    rows = _rows(args.pairs)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if args.limit:
        train, dev = train[:args.limit], dev[:max(64, args.limit // 8)]

    rats: dict[str, str] = {}
    if args.aux_weight > 0.0:
        for r in _rows(args.rationales):
            rats[r["key"]] = r["rationale"]

    def key_of(row: dict) -> str:
        return hashlib.sha1((row["request"] + "\x1f" + row["tool"]).encode(
            "utf-8")).hexdigest()

    if args.aux_weight > 0.0:
        missing = sum(1 for r in train if key_of(r) not in rats)
        print(f"aux on: {len(train) - missing}/{len(train)} train rows have a "
              f"rationale (missing -> generic fallback)")

    dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_name == "cpu":
        print("WARNING: no CUDA device; training on CPU", file=sys.stderr)
    print(f"training on {dev_name} · base {args.base} · aux_weight "
          f"{args.aux_weight} · epochs {args.epochs} · batch {args.batch}")

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, ignore_mismatched_sizes=True)
    n_params = sum(p.numel() for p in base.parameters())
    print(f"student params {n_params} (~{n_params/1e6:.0f}M)")
    net = RationaleDistill(base, max_rat=args.max_rat).to(dev_name)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def weight_of(row: dict) -> float:
        return NEAR_MISS_WEIGHT if row["source"] in NEAR_MISS_SOURCES else 1.0

    fallback = "The tool cannot serve this request."
    net.train()
    for epoch in range(args.epochs):
        torch.manual_seed(args.seed + epoch)
        perm = torch.randperm(len(train)).tolist()
        run_lbl = run_lm = 0.0
        steps = 0
        for start in range(0, len(train), args.batch):
            chunk = [train[i] for i in perm[start:start + args.batch]]
            enc = tok([r["request"] for r in chunk], [r["tool"] for r in chunk],
                      truncation=True, max_length=args.max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(dev_name) for k, v in enc.items()}
            labels = torch.tensor([r["label"] for r in chunk]).to(dev_name)
            weights = torch.tensor([weight_of(r) for r in chunk],
                                   dtype=torch.float).to(dev_name)
            tgt = None
            if args.aux_weight > 0.0:
                texts = [rats.get(key_of(r), fallback) for r in chunk]
                tgt = tok(texts, truncation=True, max_length=args.max_rat,
                          padding=True, return_tensors="pt")["input_ids"].to(dev_name)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(dev_name == "cuda")):
                logits, lm = net(enc["input_ids"], enc["attention_mask"],
                                 enc.get("token_type_ids"), tgt)
                ce = F.cross_entropy(logits.float(), labels, reduction="none")
                lbl = (ce * weights).sum() / weights.sum()
                loss = lbl
                if lm is not None:
                    lm = lm.float()
                    lm_ce = F.cross_entropy(
                        lm.reshape(-1, lm.size(-1)),
                        tgt[:, 1:].reshape(-1), ignore_index=0)
                    loss = lbl + args.aux_weight * lm_ce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run_lbl += float(lbl.detach())
            run_lm += float(lm_ce.detach()) if lm is not None else 0.0
            steps += 1
        print(f"epoch {epoch + 1}: label {run_lbl/max(1,steps):.4f} · "
              f"rationale {run_lm/max(1,steps):.4f}")
    net.eval()

    def score(req: str, tool: str) -> float:
        e = tok([req], [tool], truncation=True, max_length=args.max_len,
                padding=True, return_tensors="pt")
        e = {k: v.to(dev_name) for k, v in e.items()}
        with torch.no_grad():
            logits, _ = net.encode(e["input_ids"], e.get("attention_mask"),
                                   e.get("token_type_ids"))
        return float(torch.softmax(logits.float(), -1)[0][1])

    def pair_rates(subset: list[dict], thr: float) -> tuple[float, float]:
        tp = fn = tn = fp = 0
        for row in subset:
            matched = score(row["request"], row["tool"]) >= thr
            if row["label"] == 1:
                tp += matched
                fn += not matched
            else:
                fp += matched
                tn += not matched
        return 100 * tp / max(1, tp + fn), 100 * tn / max(1, tn + fp)

    dev_lines = ["| threshold | positives kept % | negatives rejected % |",
                 "|---|---|---|"]
    for thr in (0.3, 0.5, 0.6, 0.7, 0.9):
        kept, rejected = pair_rates(dev, thr)
        dev_lines.append(f"| {thr} | {kept:.1f} | {rejected:.1f} |")
        print(f"  dev thr {thr}: kept {kept:.1f} rejected {rejected:.1f}")

    int8_path, fp32_path = export(net, tok, args.out, args.max_len)
    digest = hashlib.sha256(int8_path.read_bytes()).hexdigest()
    (int8_path.parent / (int8_path.name + ".sha256")).write_text(
        digest + "\n", encoding="utf-8")
    tok_path = int8_path.parent / (
        int8_path.name.replace("-int8.onnx", "-tokenizer.json"))
    tok.backend_tokenizer.save(str(tok_path))
    size = int8_path.stat().st_size
    print(f"exported {int8_path.name} ({size} bytes, {size/1e6:.1f} MB) · "
          f"sha256 {digest[:16]}")
    print(f"tokenizer {tok_path.name} · fp32 {fp32_path.name}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join([
            f"# rationale-distill · {args.base} · aux_weight {args.aux_weight}",
            f"params {n_params} (~{n_params/1e6:.0f}M) · epochs {args.epochs} · "
            f"batch {args.batch} · lr {args.lr} · max_len {args.max_len}",
            f"train {len(train)} / dev {len(dev)}", "",
            "Held-out dev rows, pair level.", "", *dev_lines, "",
            f"int8 {int8_path.name} {size} B ({size/1e6:.1f} MB) sha256 {digest}",
            f"fp32 {fp32_path.name}",
        ]), encoding="utf-8")
    return 0


def export(net: RationaleDistill, tok, out: Path,
           max_len: int) -> tuple[Path, Path]:
    """Export encoder + label head to fp32 and int8 ONNX. Returns (int8, fp32)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out.parent.mkdir(parents=True, exist_ok=True)
    if "-int8" in out.name:
        fp32_path = out.parent / out.name.replace("-int8", "")
    else:
        fp32_path = out.parent / f"{out.stem}-fp32{out.suffix}"

    wrapper = LabelOnly(net).to("cpu").eval()
    dummy = tok(["a request"], ["a tool: does a thing"], return_tensors="pt",
                truncation=True, max_length=max_len, padding=True)
    input_names = ["input_ids", "attention_mask"]
    inputs = [dummy["input_ids"], dummy["attention_mask"]]
    if "token_type_ids" in dummy:
        input_names.append("token_type_ids")
        inputs.append(dummy["token_type_ids"])
    dynamic_axes = {k: {0: "batch", 1: "seq"} for k in input_names}
    dynamic_axes["logits"] = {0: "batch"}
    torch.onnx.export(wrapper, tuple(inputs), fp32_path,
                      input_names=input_names, output_names=["logits"],
                      dynamic_axes=dynamic_axes, opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    return out, fp32_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    g = sub.add_parser("generate", help="teacher rationales")
    g.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    g.add_argument("--rationales", type=Path, default=DEFAULT_RATIONALES)
    g.add_argument("--teacher", type=str, default=TEACHER)
    g.add_argument("--batch", type=int, default=16)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--max-new-tokens", type=int, default=64)
    g.add_argument("--prompt-max-len", type=int, default=512)
    g.add_argument("--tool-chars", type=int, default=300)
    g.add_argument("--req-chars", type=int, default=300)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="train the student cross-encoder")
    t.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    t.add_argument("--rationales", type=Path, default=DEFAULT_RATIONALES)
    t.add_argument("--base", type=str, default=STUDENT)
    t.add_argument("--out", type=Path, default=DEFAULT_OUT)
    t.add_argument("--report", type=Path, default=None)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--max-len", type=int, default=256)
    t.add_argument("--max-rat", type=int, default=64)
    t.add_argument("--aux-weight", type=float, default=0.5,
                   help="rationale-decoder loss weight; 0 disables the decoder "
                        "(label-only control)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--limit", type=int, default=0)
    t.set_defaults(func=cmd_train)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
