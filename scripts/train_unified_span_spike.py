#!/usr/bin/env python3
"""Unified span extractor — PHASE-1 SPIKE trainer (arm ``unified-span-spike``).

Registered arm: unified-span-extractor (registered
2026-10-10 BEFORE any candidate measurement). Proves or falsifies that ONE
GLiNER2-style unified span model (request + schema conditioning in, span
boundaries for every required slot out, ONE forward pass) beats the shipped
pairwise per-slot scorer architecture on the frozen dev split.

Dev-only measurement. Never promotes: candidates + sidecars + trainmeta go
to .scratch/unified-span/, models/ is never touched. The trainer REFUSES to
start without a passed audit stamp carrying arm == "unified-span-spike"
(fail-closed, the train_stringparam_scorer.py contract).

Registered contract highlights (the spec governs; this file implements it):
- ONE input sequence: [CLS] <schema> [SEP] <request> [SEP]; the model sees
  input_ids + attention_mask ONLY (no token_type_ids anywhere — the
  spanfix wave's segment-skew bug class is designed out; request/schema
  membership is tracked by construction via the tokenizer's sequence ids).
- Field queries: mean-pooled encoder hidden states over each field's
  rendered schema tokens (same forward pass — GLiNER2-style conditioning).
- Heads: bilinear per-token start/end scores, field-conditioned:
  logits(f, i) = q_f · W_{start|end} h_i.
- Decode-D (PRIMARY): per-field best span (start <= end, <= 8 request
  tokens) by mean sigmoid score; below the emit floor the slot abstains;
  strict coercion via the SHIPPED module's _learned_coerce (one source of
  truth: enum case-insensitive declared member, integer digit-parse,
  string wrap-strip; failure = abstain); greedy span exclusivity.
- Decode-C (registered ablation): enum slots restricted to spans whose
  text matches a declared member; otherwise identical.
- Emit floor tuned on dev only over the shipped grid under the shipped
  invented <= 0.10 budget; the chosen floor travels with the trainmeta.
- Int8 via house quantize_dynamic(QInt8); serving shape is PER-ROW
  session.run (int8 graphs requantize per tensor — co-batching shifts
  logits); a batch-equivalence spot check is REPORTED, the contract stays
  per-row.
- GPU training mandatory (rule 9): refuses without CUDA.

Usage:
    python scripts/train_unified_span_spike.py --candidate both
    python scripts/train_unified_span_spike.py --candidate B --seed 43
    python scripts/train_unified_span_spike.py --candidate A --eval-only
Run under: flock /tmp/mivia_gpu.lock (house GPU convention).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json import span_fill as _sf  # noqa: E402  (coercion parity)
from mivia_small_json.determinism import seed_everything  # noqa: E402

# --- registered constants (the spec is the source of truth) -----------------
ARM = "unified-span-spike"
SPEC = "unified-span-extractor"
AUDIT_STAMP = ROOT / ".scratch/audit-mix-stamp.json"
OUT = ROOT / ".scratch/unified-span"
FROZEN = {
    "train": ROOT / "data/real/spanfill/train.jsonl",
    "dev": ROOT / "data/real/spanfill/dev.jsonl",
}
FROZEN_SHA = {
    "train": "8bf69189eac6f0e11d8dc88d19af85940998227fb7a52554016bde909d7705b6",
    "dev": "3233c8cc5d988088b4b01e17a06d8d6881dd52bdaa58f47df74951e2f9080108",
}
FROZEN_N = {"train": 12_407, "dev": 1_402}
CANDIDATES = {
    "A": {"label": "modernbert-base", "hf_id": "answerdotai/ModernBERT-base"},
    # dated substitution note (spec): the task named "mMiniLM-L6"; no
    # multilingual 6-layer MiniLM exists upstream, so the 30 MB-class slot
    # is filled by the locally cached all-MiniLM-L6-v2 (English-only; the
    # corpus is declared English-only this spike).
    "B": {"label": "all-MiniLM-L6-v2",
          "hf_id": "sentence-transformers/all-MiniLM-L6-v2"},
}
RECIPE = {"epochs": 3, "batch": 32, "lr": 3e-5, "warmup_frac": 0.1,
          "clip": 1.0, "max_len": 256, "weight_decay": 0.01}
EMIT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)   # shipped tune_floors grid
DEV_INVENTED_MAX = 0.10                       # shipped dev budget
SPAN_MAX_TOKENS = 8                           # registered span length cap
FIELD_CAP = 8                                 # defensive; assert corpus <=
PRIMARY_DECODE = "D"                          # registered primary variant

# --- HYBRID DECODE INCREMENT (2026-10-10; spec amendment, frozen BEFORE any
# decode-E measurement) -------------------------------------------------------
HYBRID_VARIANT = "E"        # decode-E: word-attributed snap decode
HYBRID_SEEDS = (42, 43)     # A-s42 + A-s43 (seed-stability re-check)
# The shipped span_fill decode's extend floor, taken as frozen parity (NOT
# tuned; the tests pin the equality to span_fill._SPANFILL_EXTEND_FLOOR).
EXTEND_FLOOR = 0.15
_WORD_RE = re.compile(r"\w+")


def _glued_words(request: str, left, right) -> bool:
    """True when NO whitespace sits between two adjacent word spans — the
    gap is contiguous non-whitespace glue (hyphen/dot/at/underscore/...).
    Spike-local port of ``span_fill._glued`` (the hybrid decode mirrors the
    CONTRACT, nothing is edited or imported-for-mutation in src/)."""
    gap = request[left[1]:right[0]]
    return bool(gap) and not any(ch.isspace() for ch in gap)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_stamp() -> dict:
    """Fail-closed: a passed stamp for THIS arm or no training."""
    if not AUDIT_STAMP.is_file():
        sys.exit(f"refusing to train: no audit stamp at {AUDIT_STAMP} "
                 f"(run scripts/audit_training_mix.py --arm {ARM})")
    data = json.loads(AUDIT_STAMP.read_text())
    if data.get("passed") is not True or data.get("arm") != ARM:
        sys.exit(f"refusing to train: stamp at {AUDIT_STAMP} is "
                 f"passed={data.get('passed')} arm={data.get('arm')!r}; "
                 f"this trainer requires passed=true for arm {ARM!r}")
    extra = data.get("extra_sha256") or {}
    for part, path in FROZEN.items():
        got = extra.get(f"data/real/spanfill/{part}.jsonl")
        if got and got != FROZEN_SHA[part]:
            sys.exit(f"refusing to train: stamp sha for {part} drifts "
                     "from the registered frozen pin")
        if sha256_of(path) != FROZEN_SHA[part]:
            sys.exit(f"refusing to train: {part} bytes drift from the "
                     "registered frozen pin")
    return data


def field_render(name: str, prop: dict) -> str:
    """One field's rendering — the information content of the shipped
    slot_text (name | description | one of: ... | type: T)."""
    parts = [str(name)]
    desc = str((prop or {}).get("description", "")).strip()
    if desc:
        parts.append(desc)
    enum = (prop or {}).get("enum") or []
    if enum:
        parts.append("one of: " + ", ".join(str(v) for v in enum))
    parts.append(f"type: {str((prop or {}).get('type', 'string'))}")
    return " | ".join(parts)


def render_schema(row: dict):
    """The registered rendering: the tool's REQUIRED fields in schema
    order joined by ' ;; '. Returns (schema_text, names, props, ranges)
    where ranges are each field render's [start, end) char offsets inside
    schema_text — computed from the same strings, never searched."""
    params = row["tool"].get("parameters") or {}
    props_map = params.get("properties") or {}
    names, props, renders, ranges, pos = [], [], [], [], 0
    for name in list(params.get("required") or []):
        prop = props_map.get(name) \
            if isinstance(props_map.get(name), dict) else {}
        render = field_render(name, prop)
        names.append(name)
        props.append(prop)
        ranges.append((pos, pos + len(render)))
        renders.append(render)
        pos += len(render) + len(" ;; ")
    return " ;; ".join(renders), names, props, ranges


def load_raw_tokenizer(path: Path):
    """The raw `tokenizers` object — the ONLY tokenizer surface in this
    script (train and runtime alike; one offsets contract). Never
    AutoTokenizer: transformers 5.18's TokenizersBackend returned broken
    offset mappings for the ModernBERT checkpoint (measured 2026-10-10,
    the smoke test), and ModernBERT's tokenizer.json carries no pair
    post-processor — both designed out by encode_pair's manual assembly."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def _trim_offsets(text: str, offsets) -> list:
    """Trim whitespace out of a token's char span (ByteLevel BPE carries
    the leading space INSIDE the token offset — ' sessions' measures
    (7,16) where the value starts at 8; the containment gold-token
    contract and the verbatim-value decode need content spans). This is
    what BERT/XLM-R offsets do natively, so it is a no-op for candidate
    B and the parity-preserving fix for candidate A. Measured 2026-10-10:
    without it, 857/1,402 dev rows fail gold containment on A."""
    out = []
    for (s, e) in offsets:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        out.append((s, e))
    return out


def encode_pair(tok, schema_text: str, request: str, max_len: int):
    """[CLS] schema [SEP] request [SEP] assembled MANUALLY from two
    single-sequence encodes — correct for any tokenizer.json (with or
    without a pair post-processor; the ModernBERT one has none — measured).
    Sequence membership is BY CONSTRUCTION: type 0 = schema + specials,
    type 1 = request; the model input never sees token_type_ids."""
    a = tok.encode(schema_text, add_special_tokens=True)
    b = tok.encode(request, add_special_tokens=True)
    ids = list(a.ids) + list(b.ids[1:-1]) + [b.ids[-1]]
    if len(ids) >= max_len:
        return None
    offsets = _trim_offsets(schema_text, a.offsets) \
        + _trim_offsets(request, b.offsets[1:-1]) + [(0, 0)]
    type_ids = [0] * len(a.ids) + [1] * (len(b.ids) - 1)
    return {"ids": ids, "mask": [1] * len(ids), "offsets": offsets,
            "type_ids": type_ids}


def encode_row(tok, row: dict):
    """Encode one row; returns (record, None) or (None, skip_reason).
    Skips are FAIL-LOUD at the caller (the audit declared 0 absent /
    0 truncated — any skip means the rendering contract drifted)."""
    schema_text, names, props, ranges = render_schema(row)
    request = row["request"]
    enc = encode_pair(tok, schema_text, request, RECIPE["max_len"])
    if enc is None:
        return None, "truncated"
    offsets = enc["offsets"]
    type_ids = enc["type_ids"]
    field_tokens: list[list[int]] = [[] for _ in names]
    req_positions: list[int] = []
    for i, (s, e) in enumerate(offsets):
        if s == e:                       # special tokens
            continue
        if type_ids[i] == 0:             # schema segment
            for f, (cs, ce) in enumerate(ranges):
                if s < ce and e > cs:    # overlap
                    field_tokens[f].append(i)
                    break
        else:                            # request segment
            req_positions.append(i)
    if not req_positions or not all(field_tokens):
        return None, "empty-segment"
    gold = row["gold"]
    if set(gold) != set(names):
        return None, "gold-required-mismatch"
    starts, ends = [], []
    for name in names:
        value = gold[name]
        needle = value if isinstance(value, str) else json.dumps(value)
        idx = request.find(needle)
        if idx < 0:
            return None, "gold-absent-verbatim"
        cs, ce = idx, idx + len(needle)
        hits = [i for i in req_positions
                if offsets[i][0] >= cs and offsets[i][1] <= ce]
        if not hits:                     # containment parity with the
            return None, "gold-no-contained-token"  # shipped trainer
        starts.append(min(hits))
        ends.append(max(hits))
    return {"ids": enc["ids"], "mask": enc["mask"], "offsets": offsets,
            "field_tokens": field_tokens, "req_positions": req_positions,
            "starts": starts, "ends": ends}, None


def build_split(tok, rows: list[dict], split: str):
    out, skip = [], {}
    for row in rows:
        rec, why = encode_row(tok, row)
        if rec is None:
            skip[why] = skip.get(why, 0) + 1
            continue
        rec["row"] = row
        out.append(rec)
    if skip:
        sys.exit(f"{split}: {sum(skip.values())} row(s) skipped {skip} — "
                 "the audit declared 0 absent / 0 truncated; the rendering "
                 "contract drifted, refusing to measure")
    n_fields = max(len(r["starts"]) for r in out)
    if n_fields > FIELD_CAP:
        sys.exit(f"{split}: a row carries {n_fields} required fields > "
                 f"FIELD_CAP {FIELD_CAP} — re-register the cap")
    print(f"{split}: {len(out)} encoded rows; max fields {n_fields}",
          flush=True)
    return out, n_fields


# --- model -------------------------------------------------------------------
def resolve_checkpoint(cand: dict) -> Path:
    """Prefer the local HF snapshot (deterministic bytes, offline); fall
    back to the hf_id (network) — the substitution rules are registered."""
    snap = Path.home() / ".cache/huggingface/hub"
    d = snap / ("models--" + cand["hf_id"].replace("/", "--"))
    if d.is_dir():
        revs = sorted((d / "snapshots").glob("*"))
        # prefer a snapshot with safetensors, else one with pytorch_model.bin
        for want in ("model.safetensors", "pytorch_model.bin"):
            for rev in revs:
                if (rev / want).is_file():
                    return rev
    return Path(cand["hf_id"])


def build_model(key: str, device: str):
    import torch
    from transformers import AutoModel

    base = resolve_checkpoint(CANDIDATES[key])
    try:
        backbone = AutoModel.from_pretrained(str(base),
                                             attn_implementation="eager")
    except (ValueError, KeyError) as exc:
        # some archs refuse 'eager'; the default (sdpa) is fine for training
        print(f"attn_implementation=eager unavailable ({exc!r}); using default")
        backbone = AutoModel.from_pretrained(str(base))

    import torch.nn as nn

    class UnifiedSpanNet(nn.Module):
        """ONE forward pass: encode [schema; request], pool field queries
        from the schema segment, bilinear start/end scores per request
        token per field. No token_type_ids input exists anywhere."""

        def __init__(self):
            super().__init__()
            self.encoder = backbone
            hidden = backbone.config.hidden_size
            self.start_proj = nn.Linear(hidden, hidden, bias=False)
            self.end_proj = nn.Linear(hidden, hidden, bias=False)

        def forward(self, input_ids, attention_mask, field_mask):
            h = self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask
                             ).last_hidden_state            # (B,T,H)
            counts = field_mask.sum(-1).clamp(min=1.0).unsqueeze(-1)
            q = torch.bmm(field_mask, h) / counts           # (B,F,H)
            zs = self.start_proj(h)
            ze = self.end_proj(h)
            start_logits = torch.bmm(q, zs.transpose(1, 2))  # (B,F,T)
            end_logits = torch.bmm(q, ze.transpose(1, 2))
            return start_logits, end_logits

    net = UnifiedSpanNet().to(device)
    return net, base


def pad_batch(recs: list[dict], n_fields: int, device: str):
    """House convention: pad to the recipe cap so training shapes and the
    export-time shapes agree."""
    import torch

    n = RECIPE["max_len"]
    B = len(recs)

    def ids_pad(key):
        out = torch.zeros((B, n), dtype=torch.long)
        for i, r in enumerate(recs):
            seq = r[key][:n]
            out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out

    fm = torch.zeros((B, n_fields, n), dtype=torch.float32)
    starts = torch.full((B, n_fields), -100, dtype=torch.long)
    ends = torch.full((B, n_fields), -100, dtype=torch.long)
    req = torch.zeros((B, n), dtype=torch.float32)
    for i, r in enumerate(recs):
        for f, toks in enumerate(r["field_tokens"]):
            fm[i, f, toks] = 1.0
        for f in range(len(r["starts"])):
            starts[i, f] = r["starts"][f]
            ends[i, f] = r["ends"][f]
        req[i, r["req_positions"]] = 1.0
    return (ids_pad("ids").to(device), ids_pad("mask").to(device),
            fm.to(device), starts.to(device), ends.to(device),
            req.to(device))


def train_net(key: str, seed: int, train_recs, n_fields, device: str,
              t0: float):
    import torch
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import LambdaLR

    net, base = build_model(key, device)
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["lr"],
                            weight_decay=RECIPE["weight_decay"])
    steps_per_epoch = max(1, (len(train_recs) + RECIPE["batch"] - 1)
                          // RECIPE["batch"])
    total = steps_per_epoch * RECIPE["epochs"]
    warm = max(1, int(total * RECIPE["warmup_frac"]))

    def lr_lambda(step):
        if step < warm:
            return step / warm
        return max(0.0, (total - step) / max(1, total - warm))

    sched = LambdaLR(opt, lr_lambda)
    gen = torch.Generator().manual_seed(seed)
    net.train()
    losses = []
    for epoch in range(RECIPE["epochs"]):
        perm = torch.randperm(len(train_recs), generator=gen).tolist()
        ep_loss, nb = 0.0, 0
        for s in range(0, len(perm), RECIPE["batch"]):
            recs = [train_recs[i] for i in perm[s:s + RECIPE["batch"]]]
            ids, mask, fm, st, en, req = pad_batch(recs, n_fields, device)
            sl, el = net(ids, mask, fm)                     # (B,F,T)
            NEG = -10000.0
            sl = sl.masked_fill(req[:, None, :] == 0, NEG)
            el = el.masked_fill(req[:, None, :] == 0, NEG)
            loss = F.cross_entropy(sl.flatten(0, 1), st.flatten(),
                                   ignore_index=-100) \
                + F.cross_entropy(el.flatten(0, 1), en.flatten(),
                                  ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE["clip"])
            opt.step()
            sched.step()
            ep_loss += float(loss.detach())
            nb += 1
        losses.append(ep_loss / max(1, nb))
        print(f"epoch {epoch}: mean_loss {losses[-1]:.7f}", flush=True)
    gpu_s = time.time() - t0
    sw = base / "model.safetensors"
    meta = {"base_ckpt": str(base),
            "base_safetensors_sha256": sha256_of(sw) if sw.is_file() else None,
            "hidden": net.encoder.config.hidden_size,
            "recipe": RECIPE, "seed": seed, "device": device,
            "mean_loss_per_epoch": losses,
            "train_rows": len(train_recs), "n_fields": n_fields,
            "gpu_seconds": round(gpu_s, 1)}
    return {"net": net, "meta": meta}


def export_candidate(net, tok, key: str, seed: int, cand_dir: Path) -> dict:
    import torch

    cand_dir.mkdir(parents=True, exist_ok=True)
    stem = f"unified-span-{CANDIDATES[key]['label']}-s{seed}"
    fp32 = cand_dir / f"{stem}.onnx"
    int8 = cand_dir / f"{stem}-int8.onnx"
    tokj = cand_dir / f"{stem}-tokenizer.json"
    net.eval()
    net.to("cpu")   # export on CPU: device-free graph, no CUDA-export quirks
    ids, mask, fm, _, _, _ = pad_batch(
        [{"ids": [1, 2], "mask": [1, 1], "field_tokens": [[1]],
          "req_positions": [1], "starts": [1], "ends": [1]}], 1, "cpu")
    torch.onnx.export(
        net, (ids, mask, fm), str(fp32),
        input_names=["input_ids", "attention_mask", "field_mask"],
        output_names=["start_logits", "end_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "field_mask": {0: "batch", 1: "fields", 2: "seq"},
                      "start_logits": {0: "batch", 1: "fields", 2: "seq"},
                      "end_logits": {0: "batch", 1: "fields", 2: "seq"}},
        opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    tok.save(str(tokj))
    for p in (fp32, int8, tokj):
        (cand_dir / (p.name + ".sha256")).write_text(sha256_of(p) + "\n")
    return {"stem": stem, "fp32": fp32, "int8": int8, "tokenizer": tokj,
            "int8_bytes": int8.stat().st_size,
            "int8_sha256": sha256_of(int8),
            "tokenizer_sha256": sha256_of(tokj)}


# --- decode + measurement on the int8 bytes (per-row session.run) -----------
class SpanSession:
    """The registered serving shape: PER-ROW session.run, batch=1, CPU,
    onnxruntime + the saved tokenizer. Never co-batches (int8 graphs
    requantize per tensor — this repo measured co-batch logit drift)."""

    def __init__(self, onnx_path: Path, tokj_path: Path, max_len: int):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokj_path))
        self.tok.no_padding()
        self.sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])
        self.max_len = max_len

    def encode(self, schema_text: str, request: str):
        enc = encode_pair(self.tok, schema_text, request, self.max_len)
        if enc is None:
            return None
        schema_pos, req_pos = [], []
        for i, (tid, (s, e)) in enumerate(zip(enc["type_ids"],
                                              enc["offsets"])):
            if s == e:
                continue
            (schema_pos if tid == 0 else req_pos).append(i)
        return {"ids": enc["ids"], "mask": enc["mask"],
                "offsets": enc["offsets"],
                "schema_pos": schema_pos, "req_pos": req_pos}

    def row_logits(self, enc: dict, field_tokens: list[list[int]]):
        n = len(enc["ids"])
        F = len(field_tokens)
        fm = np.zeros((1, F, n), dtype=np.float32)
        for f, toks in enumerate(field_tokens):
            fm[0, f, toks] = 1
        feed = {"input_ids": np.array([enc["ids"]], dtype=np.int64),
                "attention_mask": np.array([enc["mask"]], dtype=np.int64),
                "field_mask": fm}
        sl, el = self.sess.run(["start_logits", "end_logits"], feed)
        return sl[0], el[0]          # (F,T) each — PER-ROW run


def _sigmoid(x) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


class Decode:
    """decode-D (primary) and decode-C (registered enum-constrained
    ablation) over cached per-row logits; strict coercion via the shipped
    module's _learned_coerce (one source of truth).

    decode-E (HYBRID DECODE INCREMENT, registered 2026-10-10 BEFORE any
    decode-E measurement): the pairwise arm's two decode-level fixes ported
    spike-locally (commit c4e521a's mechanism; span_fill.py itself is
    read-only for this arm):
      1. subtoken/WORD attribution by OVERLAP, never strict containment —
         the field's start/end sigmoids are max-pooled onto the request's
         ``\\w+`` words, so a head boundary landing inside a
         subword-fragmented word is that WHOLE word's evidence (the
         ``pair_probs`` port);
      2. end-boundary snapping across contiguous non-whitespace glue while
         the next word's own pooled end confidence stays >= EXTEND_FLOOR
         (0.15, shipped span_fill parity). Whitespace is a hard snapping
         limit in BOTH directions: the seed span — decode-D's registered
         construct on word-pooled scores — remains the ONLY
         whitespace-crossing authority, so the multi-word native capability
         is preserved and no first-space truncation is reintroduced. No
         start-side snapping (the measured truncation class is 100%
         end-side strict prefixes; mid-word starts are repaired by overlap
         attribution). Emission floor, exclusivity (checked on the SNAPPED
         span) and strict coercion are UNCHANGED — S5 stays structural.
    """

    def __init__(self, sess: SpanSession):
        self.sess = sess

    @staticmethod
    def _word_pool(request: str, sl_f, el_f, req_pos, offsets):
        """Max-pool the field's per-token start/end sigmoids onto the
        request's ``\\w+`` words BY OVERLAP (a token counts when its
        whitespace-trimmed char span overlaps the word). Returns
        (words, start_probs, end_probs)."""
        words = [(m.start(), m.end()) for m in _WORD_RE.finditer(request)]
        wsp = [0.0] * len(words)
        epp = [0.0] * len(words)
        for i in req_pos:
            s, e = offsets[i]
            if e <= s:
                continue
            sp = _sigmoid(sl_f[i])
            ep = _sigmoid(el_f[i])
            for w, (a, b) in enumerate(words):
                if s < b and e > a:
                    if sp > wsp[w]:
                        wsp[w] = sp
                    if ep > epp[w]:
                        epp[w] = ep
        return words, wsp, epp

    def _seed_word_span(self, request, sl_f, el_f, req_pos, offsets):
        """decode-E seed: the decode-D construct (argmax mean boundary
        score, length <= SPAN_MAX_TOKENS) over WORD pairs of the pooled
        scores. Returns (score, a, b, words, epp) or None."""
        words, wsp, epp = self._word_pool(request, sl_f, el_f, req_pos,
                                          offsets)
        if not words:
            return None
        best = None
        n = len(words)
        for a in range(n):
            sa = wsp[a]
            for b in range(a, min(n, a + SPAN_MAX_TOKENS)):
                score = (sa + epp[b]) / 2.0
                if best is None or score > best[0]:
                    best = (score, a, b)
        return best[0], best[1], best[2], words, epp

    def fill(self, request: str, names: list[str], props: list[dict],
             enc: dict, field_tokens: list[list[int]], logits, floor: float,
             variant: str):
        sl, el = logits
        offsets = enc["offsets"]
        req_pos = enc["req_pos"]
        enum_lower = [{str(v).lower() for v in (p.get("enum") or [])}
                      for p in props]
        cands = []
        debug = {"collisions": 0}
        for f, name in enumerate(names):
            if variant == "E":
                seed = self._seed_word_span(request, sl[f], el[f], req_pos,
                                            offsets)
                if seed is None:
                    continue
                cands.append({"f": f, "name": name, "prop": props[f],
                              "score": seed[0], "kind": "word",
                              "a": seed[1], "b": seed[2],
                              "words": seed[3], "epp": seed[4]})
                continue
            best = None                       # (score, i, j)
            alt = None                        # decode-C enum-matching best
            for ii, i in enumerate(req_pos):
                si = _sigmoid(sl[f, i])
                for jj, j in enumerate(req_pos):
                    if jj < ii or jj - ii + 1 > SPAN_MAX_TOKENS:
                        continue
                    score = (si + _sigmoid(el[f, j])) / 2.0
                    if best is None or score > best[0]:
                        best = (score, i, j)
                    if variant == "C" and enum_lower[f]:
                        text = request[offsets[i][0]:offsets[j][1]]
                        if _sf._wrap_strip(text).lower() in enum_lower[f] \
                                and (alt is None or score > alt[0]):
                            alt = (score, i, j)
            pick = best
            if variant == "C" and enum_lower[f]:
                pick = alt if alt is not None else None  # no member: abstain
            if pick is None:
                continue
            cands.append({"f": f, "name": name, "prop": props[f],
                          "score": pick[0], "kind": "tok",
                          "i": pick[1], "j": pick[2]})
        cands.sort(key=lambda c: -c["score"])
        filled: dict = {}
        claimed: list[tuple[int, int]] = []
        for c in cands:
            if c["kind"] == "word":
                # End-boundary snapping: extend across contiguous
                # non-whitespace glue while the next word's OWN pooled end
                # confidence holds; whitespace is a hard limit (never
                # crossed by snapping — only the seed's own boundary
                # evidence crosses whitespace).
                we = c["b"]
                words, epp = c["words"], c["epp"]
                while we + 1 < len(words) \
                        and epp[we + 1] >= EXTEND_FLOOR \
                        and _glued_words(request, words[we], words[we + 1]):
                    we += 1
                s, e = words[c["a"]][0], words[we][1]
            else:
                s, e = offsets[c["i"]][0], offsets[c["j"]][1]
            if c["score"] < floor:
                continue
            if any(not (e <= cs or s >= ce) for cs, ce in claimed):
                debug["collisions"] += 1
                continue
            raw = request[s:e]
            value = _sf._learned_coerce(raw, c["prop"])
            if value is None or (isinstance(value, str) and not value):
                continue
            claimed.append((s, e))
            filled[c["name"]] = value
        missing = [n for n in names if n not in filled]
        return filled, missing, debug


def multiword_row(row: dict) -> bool:
    return any(isinstance(v, str) and " " in v.strip()
               for v in row["gold"].values())


def _row_exact(filled: dict, missing: list, gold: dict) -> bool:
    """The shipped dev-grid exact construct, verbatim."""
    return bool(filled) and not missing and set(filled) == set(gold) \
        and all(gold.get(k) == v for k, v in filled.items())


def evaluate(sess: SpanSession, decode: Decode, recs: list[dict],
             n_total: int, floors=EMIT_GRID, variants=("D", "C")):
    """ONE session.run per row (logits cached), decode at every floor —
    the grid never re-runs the graph. exact_fill's denominator is n_total
    (ALL dev rows — the shipped grid's denominator; a row the runtime
    cannot encode counts as a miss, never vanishes)."""
    per_row, enc_fail = [], 0
    for r in recs:
        row = r["row"]
        schema_text, names, props, ranges = render_schema(row)
        enc = sess.encode(schema_text, row["request"])
        if enc is None:
            enc_fail += 1
            continue
        ftoks = [[] for _ in names]
        for i in enc["schema_pos"]:
            s, e = enc["offsets"][i]
            for f, (cs, ce) in enumerate(ranges):
                if s < ce and e > cs:
                    ftoks[f].append(i)
                    break
        if not all(ftoks) or not enc["req_pos"]:
            enc_fail += 1
            continue
        logits = sess.row_logits(enc, ftoks)
        per_row.append({"row": row, "enc": enc, "names": names,
                        "props": props, "ftoks": ftoks, "logits": logits})
    if enc_fail:
        sys.exit(f"dev runtime encode failures: {enc_fail} — refusing to "
                 "measure a shrinking denominator")
    out = {}
    for variant in variants:
        cells = []
        for floor in floors:
            exact = wrong = emitted = 0
            mw_exact = mw_n = 0
            enum_ok = enum_n = enum_viol = 0
            collisions = 0
            for pr in per_row:
                row = pr["row"]
                filled, missing, dbg = decode.fill(
                    row["request"], pr["names"], pr["props"], pr["enc"],
                    pr["ftoks"], pr["logits"], floor, variant)
                collisions += dbg["collisions"]
                gold = row["gold"]
                for k, v in (filled or {}).items():
                    emitted += 1
                    if k not in gold or gold[k] != v:
                        wrong += 1
                if _row_exact(filled, missing, gold):
                    exact += 1
                if multiword_row(row):
                    mw_n += 1
                    if _row_exact(filled, missing, gold):
                        mw_exact += 1
                for k, v in (filled or {}).items():
                    f_idx = pr["names"].index(k)
                    members = [str(m) for m in
                               (pr["props"][f_idx].get("enum") or [])]
                    if members:
                        enum_n += 1
                        if v not in members:
                            enum_viol += 1
                        elif gold.get(k) == v:
                            enum_ok += 1
            cells.append({
                "floor": floor, "n": n_total, "exact_fill": round(
                    exact / max(1, n_total), 4),
                "exact_rows": exact,
                "invented_rate": round(wrong / emitted, 4) if emitted else 0.0,
                "emitted": emitted, "wrong": wrong,
                "s3_rows": mw_n, "s3_exact_rows": mw_exact,
                "s3_exact_fill": round(mw_exact / max(1, mw_n), 4),
                "enum_slots": enum_n, "enum_violations": enum_viol,
                "enum_acc": round(enum_ok / max(1, enum_n), 4),
                "collisions": collisions})
        chosen = None
        for c in cells:                     # shipped tie-break: first
            if c["invented_rate"] <= DEV_INVENTED_MAX:   # (lowest) wins
                chosen = c
                break
        if chosen is None:
            chosen = min(cells, key=lambda c: c["invented_rate"])
        out[variant] = {"cells": cells, "chosen": chosen}
    return out, per_row


def _feed_padded(enc: dict, pad_to: int):
    n = max(len(enc["ids"]), pad_to)
    ids = np.zeros((1, n), dtype=np.int64)
    mask = np.zeros((1, n), dtype=np.int64)
    fm = np.zeros((1, 1, n), dtype=np.float32)
    ids[0, :len(enc["ids"])] = enc["ids"]
    mask[0, :len(enc["mask"])] = enc["mask"]
    if enc["schema_pos"]:
        fm[0, 0, enc["schema_pos"]] = 1
    return {"input_ids": ids, "attention_mask": mask, "field_mask": fm}


def latency_report(sess: SpanSession, recs: list[dict]) -> dict:
    """CPU per-row latency (natural lengths) + a padded-512 probe; stated
    numbers only (no timing asserts anywhere)."""
    prepared = []
    for r in recs:
        schema_text, names, _, _ = render_schema(r["row"])
        enc = sess.encode(schema_text, r["row"]["request"])
        if enc is not None:
            prepared.append(enc)
    for enc in prepared[:10]:               # warmup
        sess.sess.run(["start_logits", "end_logits"],
                      _feed_padded(enc, len(enc["ids"])))
    times = []
    for enc in prepared:
        t = time.perf_counter()
        sess.sess.run(["start_logits", "end_logits"],
                      _feed_padded(enc, len(enc["ids"])))
        times.append(time.perf_counter() - t)
    times.sort()
    p512 = []
    for enc in prepared[:50]:
        for _ in range(3):
            t = time.perf_counter()
            sess.sess.run(["start_logits", "end_logits"],
                          _feed_padded(enc, 512))
            p512.append(time.perf_counter() - t)
    p512.sort()
    rep = {"n": len(times),
           "p50_ms": round(1000 * times[len(times) // 2], 2),
           "p95_ms": round(1000 * times[int(len(times) * 0.95)], 2),
           "max_ms": round(1000 * times[-1], 2),
           "padded512_p95_ms": round(1000 * p512[int(len(p512) * 0.95)], 2)}
    print("latency (CPU, per-row):", json.dumps(rep), flush=True)
    return rep


def batch_invariance(sess: SpanSession, recs: list[dict], n: int = 32) -> dict:
    """Spot check ONLY: per-row runs vs one padded batch on the same rows.
    The registered serving contract is per-row regardless of the result."""
    sample = recs[:n]
    encs, per_row_logits = [], []
    for r in sample:
        schema_text, names, _, ranges = render_schema(r["row"])
        enc = sess.encode(schema_text, r["row"]["request"])
        ftoks = [[] for _ in names]
        for i in enc["schema_pos"]:
            s, e = enc["offsets"][i]
            for f, (cs, ce) in enumerate(ranges):
                if s < ce and e > cs:
                    ftoks[f].append(i)
                    break
        encs.append((enc, ftoks))
        per_row_logits.append(sess.row_logits(enc, ftoks))
    maxlen = max(len(e["ids"]) for e, _ in encs)
    F = max(len(ft) for _, ft in encs)
    ids = np.zeros((len(encs), maxlen), dtype=np.int64)
    mask = np.zeros((len(encs), maxlen), dtype=np.int64)
    fm = np.zeros((len(encs), F, maxlen), dtype=np.float32)
    for bi, (enc, ftoks) in enumerate(encs):
        L = len(enc["ids"])
        ids[bi, :L] = enc["ids"]
        mask[bi, :L] = 1
        for f, toks in enumerate(ftoks):
            fm[bi, f, toks] = 1
    sl_b, el_b = sess.sess.run(
        ["start_logits", "end_logits"],
        {"input_ids": ids, "attention_mask": mask, "field_mask": fm})
    max_diff = 0.0
    for bi, (sl_r, el_r) in enumerate(per_row_logits):
        L = len(encs[bi][0]["ids"])
        d = max(float(np.abs(sl_b[bi, :len(sl_r), :L] - sl_r[:, :L]).max()),
                float(np.abs(el_b[bi, :len(el_r), :L] - el_r[:, :L]).max()))
        max_diff = max(max_diff, d)
    return {"n": len(encs),
            "max_abs_logit_diff_row_vs_batch": round(max_diff, 6),
            "contract": "per-row session.run regardless (registered)"}


ENSEMBLE_RSS_CODE = r"""
import json, sys
root = sys.argv[1]
cand_int8 = sys.argv[2]
loaded, errors = [], []
def rss():
    # VmHWM, NOT getrusage: a forked child INHERITS the parent's ru_maxrss
    # high-water mark (measured 2026-10-10: a 5.1 GB "baseline" from a
    # fork of the GPU trainer); VmHWM lives in the mm_struct and resets
    # on exec, so a freshly exec'd interpreter reports its own true peak.
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0
    return -1.0
baseline = rss()
import numpy as np
try:
    from sklearn.linear_model import LogisticRegression  # noqa
    import skops.io as sio
    for name in ("level_extractor.skops", "component_extractor.skops"):
        try:
            path = root + "/models/" + name
            trusted = sio.get_untrusted_types(file=path)
            sio.load(path, trusted=trusted)
            loaded.append(name)
        except Exception as e:
            errors.append(f"{name}: {e!r}")
except Exception as e:
    errors.append(f"sklearn/skops: {e!r}")
after_extractors = rss()
try:
    import onnxruntime as ort
    plan = [("router-encoder-int8.onnx", "enc"),
            ("tool-match-int8.onnx", "enc"),
            ("tool-match-gate-int8.onnx", "enc"),
            ("spanfill-extractor-int8.onnx", "spanfill"),
            (None, "candidate")]
    sessions = []
    for name, kind in plan:
        try:
            path = cand_int8 if kind == "candidate" else root + "/models/" + name
            s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            sessions.append((s, kind))
            loaded.append(name or "candidate")
        except Exception as e:
            errors.append(f"{name or 'candidate'}: {e!r}")
    after_sessions = rss()
    ids = np.zeros((1, 32), dtype=np.int64)
    mask = np.ones((1, 32), dtype=np.int64)
    fm = np.zeros((1, 1, 32), dtype=np.float32)
    fm[0, 0, :4] = 1
    for s, kind in sessions:
        try:
            names = {i.name for i in s.get_inputs()}
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in names:      # BERT-style graphs declare it
                feed["token_type_ids"] = np.zeros_like(ids)
            if "field_mask" in names:
                feed["field_mask"] = fm
            s.run(None, feed)
        except Exception as e:
            errors.append(f"run: {e!r}")
    print(json.dumps({"baseline_mb": round(baseline, 1),
                      "after_extractors_mb": round(after_extractors, 1),
                      "after_sessions_mb": round(after_sessions, 1),
                      "peak_rss_mb": round(rss(), 1), "loaded": loaded,
                      "errors": errors}))
except Exception as e:
    print(json.dumps({"baseline_mb": round(baseline, 1),
                      "after_extractors_mb": round(after_extractors, 1),
                      "peak_rss_mb": None, "loaded": loaded,
                      "errors": errors + [f"ort: {e!r}"]}))
"""


def ensemble_rss(cand_int8: Path) -> dict:
    """Peak-RSS measurement of the full serving ensemble (router + guard +
    spanfill extractor + skops extractors + this candidate) in a fresh
    subprocess. Stated, not asserted. Reports exactly what loaded."""
    out = subprocess.run([sys.executable, "-c", ENSEMBLE_RSS_CODE,
                          str(ROOT), str(cand_int8)],
                         capture_output=True, text=True, timeout=300)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": (out.stdout[-400:] + out.stderr[-400:])}


def env_fingerprint() -> dict:
    import platform

    import onnxruntime
    import transformers
    import torch

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        git = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnxruntime": onnxruntime.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None,
            "git_sha": git}


def run_candidate(key: str, seed: int, train_rows, dev_rows, eval_only: bool,
                  stamp: dict) -> dict:
    cand_dir = OUT / f"cand-{key}-s{seed}"
    stem = f"unified-span-{CANDIDATES[key]['label']}-s{seed}"
    int8 = cand_dir / f"{stem}-int8.onnx"
    tokj = cand_dir / f"{stem}-tokenizer.json"
    meta_path = cand_dir / f"{stem}-trainmeta.json"

    meta = {}
    if eval_only and int8.is_file() and tokj.is_file():
        meta = json.loads(meta_path.read_text())
        print(f"[eval-only] reusing {int8.name}")
    else:
        import torch

        if not torch.cuda.is_available():
            sys.exit("refusing to train: CUDA unavailable (rule 9: GPU "
                     "training is mandatory; a CPU fallback needs a stated "
                     "reason and this spike registers none)")
        t0 = time.time()
        base = resolve_checkpoint(CANDIDATES[key])
        tok = load_raw_tokenizer(base / "tokenizer.json")
        train_recs, n_fields = build_split(tok, train_rows, "train")
        trained = train_net(key, seed, train_recs, n_fields, "cuda", t0)
        net, meta = trained["net"], trained["meta"]
        paths = export_candidate(net, tok, key, seed, cand_dir)
        meta.update({"candidate": CANDIDATES[key]["label"], "key": key,
                     "seed": seed, "arm": ARM, "spec": SPEC,
                     "corpus_sha256": FROZEN_SHA,
                     "stamp_extra_sha256": stamp.get("extra_sha256"),
                     "env": env_fingerprint(),
                     "stem": paths["stem"],
                     "int8": str(paths["int8"]),
                     "tokenizer": str(paths["tokenizer"]),
                     "int8_bytes": paths["int8_bytes"],
                     "int8_sha256": paths["int8_sha256"],
                     "tokenizer_sha256": paths["tokenizer_sha256"]})
        meta_path.write_text(json.dumps(meta, indent=1))
        del net
        torch.cuda.empty_cache()

    # measurement ALWAYS runs on the int8 bytes (the served numbers' class)
    dev_recs, _ = build_split_for_eval(tokj, dev_rows)
    sess = SpanSession(int8, tokj, RECIPE["max_len"])
    decode = Decode(sess)
    metrics, per_row = evaluate(sess, decode, dev_recs, n_total=len(dev_rows))
    chosen = metrics[PRIMARY_DECODE]["chosen"]
    latency = latency_report(sess, dev_recs)
    inv = batch_invariance(sess, dev_recs)
    rss = ensemble_rss(int8)
    probes = []     # selfcheck: first 3 multi-word dev rows, filled vs gold
    for pr in per_row:
        if len(probes) >= 3:
            break
        if not multiword_row(pr["row"]):
            continue
        filled, missing, _ = decode.fill(
            pr["row"]["request"], pr["names"], pr["props"], pr["enc"],
            pr["ftoks"], pr["logits"], chosen["floor"], PRIMARY_DECODE)
        probes.append({"request": pr["row"]["request"],
                       "gold": pr["row"]["gold"], "filled": filled,
                       "missing": missing})
    result = {
        "candidate": CANDIDATES[key]["label"], "key": key, "seed": seed,
        "int8_path": str(int8), "int8_sha256": sha256_of(int8),
        "int8_bytes": int8.stat().st_size,
        "tokenizer_path": str(tokj), "tokenizer_sha256": sha256_of(tokj),
        "metrics": metrics, "chosen_floor": chosen["floor"],
        "primary_variant": PRIMARY_DECODE, "batch_invariance": inv,
        "ensemble_rss": rss, "selfcheck_probes": probes,
        "latency_cpu": latency, "trainmeta_path": str(meta_path)}
    (cand_dir / f"{stem}-devmetrics.json").write_text(
        json.dumps(result, indent=1))
    return result


def build_split_for_eval(tokj: Path, dev_rows):
    """Re-encode dev with the SAVED tokenizer bytes (the runtime path) and
    the same fail-loud skip contract."""
    tok = load_raw_tokenizer(tokj)
    return build_split(tok, dev_rows, "dev(runtime-tokenizer)")


# --- bars --------------------------------------------------------------------
def score_bars(result: dict, variant: str = PRIMARY_DECODE,
               p95_ms: float | None = None) -> dict:
    """S1-S5 for one decode variant. ``p95_ms`` overrides the bare-forward
    p95 with the FULL serving decode path's p95 (the hybrid increment
    re-measures S4 on the path that actually ships: encode + session.run +
    decode — decode adds CPU)."""
    c = result["metrics"][variant]["chosen"]
    lat = result.get("latency_cpu") or {}
    rss_peak = (result.get("ensemble_rss") or {}).get("peak_rss_mb")
    int8_mb = result["int8_bytes"] / (1024 * 1024)
    measured_p95 = p95_ms if p95_ms is not None else lat.get("p95_ms")
    s4 = (int8_mb <= 200 and (measured_p95 or 1e9) <= 50
          and rss_peak is not None and rss_peak <= 2048)
    bars = {
        "S1_exact_fill": {"value": c["exact_fill"], "floor": 0.9165,
                          "pass": c["exact_fill"] >= 0.9165},
        "S2_invented_rate": {"value": c["invented_rate"], "ceiling": 0.0441,
                             "pass": c["invented_rate"] <= 0.0441},
        "S3_multiword_exact": {"value": c["s3_exact_fill"], "floor": 0.50,
                               "n_rows": c["s3_rows"],
                               "pass": c["s3_exact_fill"] >= 0.50},
        "S4_artifact": {"int8_mb": round(int8_mb, 1), "ceiling_mb": 200,
                        "cpu_p95_ms": measured_p95,
                        "p95_ceiling_ms": 50,
                        "p95_source": ("full serving decode path "
                                       "(encode + session.run + decode-"
                                       + variant + ")") if p95_ms is not None
                        else "bare forward",
                        "ensemble_peak_rss_mb": rss_peak,
                        "ensemble_ceiling_mb": 2048,
                        "ensemble_measurement": result.get("ensemble_rss"),
                        "pass": bool(s4)},
        "S5_enum_violations": {"value": c["enum_violations"], "ceiling": 0,
                               "enum_acc_reported": c["enum_acc"],
                               "pass": c["enum_violations"] == 0},
    }
    bars["PASS"] = all(v["pass"] for k, v in bars.items()
                       if isinstance(v, dict))
    return bars


# --- HYBRID DECODE INCREMENT (2026-10-10) measurement ------------------------
def _verify_candidate_bytes(cand_dir: Path, stem: str) -> tuple[Path, Path]:
    """Sidecar-verify the staged int8 + tokenizer bytes (rule 10); refuse
    loudly when anything is missing or drifts."""
    int8 = cand_dir / f"{stem}-int8.onnx"
    tokj = cand_dir / f"{stem}-tokenizer.json"
    problems = []
    for p in (int8, tokj):
        if not p.is_file():
            problems.append(f"missing bytes: {p}")
        elif not Path(str(p) + ".sha256").is_file():
            problems.append(f"missing sidecar: {p}.sha256")
    if problems:
        sys.exit("hybrid increment: staged candidate incomplete "
                 f"(artifact-gated, refusing loudly): {problems}")
    for p in (int8, tokj):
        expected = Path(str(p) + ".sha256").read_text().split()[0]
        got = sha256_of(p)
        if got != expected:
            sys.exit(f"hybrid increment: sha sidecar mismatch for {p}: "
                     f"sidecar {expected} != bytes {got}")
    return int8, tokj


def _is_prefix(gold, val) -> bool:
    return (isinstance(gold, str) and isinstance(val, str) and val
            and val != gold and gold.startswith(val))


def delta_decomposition(per_row: list[dict], decode: Decode,
                        floor_d: float, floor_e: float) -> dict:
    """Slot-level decode-D -> decode-E delta at the two variants' chosen
    floors on identical cached logits. ``D_prefix_slots`` cross-checks the
    committed phase-1 decomposition (424 strict-prefix truncations at
    decode-D's chosen floor — rule 15: read the number back, never assume)."""
    counts = {"slots_total": 0, "exact_both": 0, "D_prefix_slots": 0,
              "truncation_recovered": 0, "truncation_still_partial": 0,
              "truncation_now_other_wrong": 0, "exact_D_wrong_E": 0,
              "abstain_or_other_recovered": 0, "wrong_or_missing_both": 0}
    rows = {"rows": 0, "exact_D_rows": 0, "exact_E_rows": 0,
            "exact_both_rows": 0}
    examples = {"truncation_recovered": [], "exact_D_wrong_E": [],
                "truncation_still_partial": []}
    for pr in per_row:
        row = pr["row"]
        gold = row["gold"]
        fD, mD, _ = decode.fill(row["request"], pr["names"], pr["props"],
                                pr["enc"], pr["ftoks"], pr["logits"],
                                floor_d, "D")
        fE, mE, _ = decode.fill(row["request"], pr["names"], pr["props"],
                                pr["enc"], pr["ftoks"], pr["logits"],
                                floor_e, "E")
        fD, fE = fD or {}, fE or {}
        rows["rows"] += 1
        d_row = _row_exact(fD, mD, gold)
        e_row = _row_exact(fE, mE, gold)
        rows["exact_D_rows"] += int(d_row)
        rows["exact_E_rows"] += int(e_row)
        rows["exact_both_rows"] += int(d_row and e_row)
        for k in set(gold) | set(fD) | set(fE):
            counts["slots_total"] += 1
            g = gold.get(k)
            d_ok = k in gold and fD.get(k) == g
            e_ok = k in gold and fE.get(k) == g
            d_val = fD.get(k)
            e_val = fE.get(k)
            d_prefix = (not d_ok and _is_prefix(g, d_val))
            if d_prefix:
                counts["D_prefix_slots"] += 1
            if d_ok and e_ok:
                counts["exact_both"] += 1
            elif d_prefix and e_ok:
                counts["truncation_recovered"] += 1
                if len(examples["truncation_recovered"]) < 10:
                    examples["truncation_recovered"].append(
                        {"request": row["request"], "slot": k, "gold": g,
                         "decode_D": d_val, "decode_E": e_val})
            elif d_prefix and _is_prefix(g, e_val):
                counts["truncation_still_partial"] += 1
                if len(examples["truncation_still_partial"]) < 10:
                    examples["truncation_still_partial"].append(
                        {"request": row["request"], "slot": k, "gold": g,
                         "decode_D": d_val, "decode_E": e_val})
            elif d_prefix:
                counts["truncation_now_other_wrong"] += 1
            elif d_ok and not e_ok:
                counts["exact_D_wrong_E"] += 1        # REGRESSION — audited
                if len(examples["exact_D_wrong_E"]) < 10:
                    examples["exact_D_wrong_E"].append(
                        {"request": row["request"], "slot": k, "gold": g,
                         "decode_D": d_val, "decode_E": e_val})
            elif e_ok:
                counts["abstain_or_other_recovered"] += 1
            else:
                counts["wrong_or_missing_both"] += 1
    return {"floors": {"D": floor_d, "E": floor_e}, "counts": counts,
            "rows": rows, "examples": examples}


def fill_latency_report(sess: SpanSession, decode: Decode,
                        per_row: list[dict], variant: str,
                        floor: float) -> dict:
    """The FULL serving decode path per row (render_schema + encode +
    session.run + decode fill) — S4's re-measured p95 for the hybrid
    (decode adds CPU). Stated numbers only; no timing asserts."""
    times = []
    for pr in per_row:
        row = pr["row"]
        t0 = time.perf_counter()
        schema_text, names, props, ranges = render_schema(row)
        enc = sess.encode(schema_text, row["request"])
        ftoks = [[] for _ in names]
        for i in enc["schema_pos"]:
            s, e = enc["offsets"][i]
            for f, (cs, ce) in enumerate(ranges):
                if s < ce and e > cs:
                    ftoks[f].append(i)
                    break
        logits = sess.row_logits(enc, ftoks)
        decode.fill(row["request"], names, props, enc, ftoks, logits,
                    floor, variant)
        times.append(time.perf_counter() - t0)
    times.sort()
    rep = {"variant": variant, "floor": floor, "n": len(times),
           "path": "render_schema + encode + session.run + decode",
           "p50_ms": round(1000 * times[len(times) // 2], 2),
           "p95_ms": round(1000 * times[int(len(times) * 0.95)], 2),
           "max_ms": round(1000 * times[-1], 2)}
    print("full-fill latency (CPU, per-row):", json.dumps(rep), flush=True)
    return rep


def run_hybrid(key: str, seed: int, dev_rows: list[dict]) -> dict:
    """The registered hybrid measurement: decode-D vs decode-E on the SAME
    staged int8 bytes (no training, no GPU, models/ untouched)."""
    cand_dir = OUT / f"cand-{key}-s{seed}"
    stem = f"unified-span-{CANDIDATES[key]['label']}-s{seed}"
    int8, tokj = _verify_candidate_bytes(cand_dir, stem)
    if sha256_of(FROZEN["dev"]) != FROZEN_SHA["dev"] \
            or len(dev_rows) != FROZEN_N["dev"]:
        sys.exit("refusing: frozen dev bytes/count drifted from the "
                 "registered pin")
    print(f"=== hybrid decode increment: candidate {key} seed {seed} "
          f"({int8.name}) ===", flush=True)
    dev_recs, _ = build_split_for_eval(tokj, dev_rows)
    sess = SpanSession(int8, tokj, RECIPE["max_len"])
    decode = Decode(sess)
    metrics, per_row = evaluate(sess, decode, dev_recs,
                                n_total=len(dev_rows), variants=("D", "E"))
    latency_bare = latency_report(sess, dev_recs)
    fill_lat = fill_latency_report(sess, decode, per_row, HYBRID_VARIANT,
                                   metrics[HYBRID_VARIANT]["chosen"]["floor"])
    rss = ensemble_rss(int8)
    delta = delta_decomposition(per_row, decode,
                                metrics["D"]["chosen"]["floor"],
                                metrics[HYBRID_VARIANT]["chosen"]["floor"])
    shim = {"metrics": metrics, "latency_cpu": latency_bare,
            "ensemble_rss": rss, "int8_bytes": int8.stat().st_size}
    result = {
        "candidate": CANDIDATES[key]["label"], "key": key, "seed": seed,
        "int8_path": str(int8), "int8_sha256": sha256_of(int8),
        "int8_bytes": int8.stat().st_size,
        "tokenizer_path": str(tokj), "tokenizer_sha256": sha256_of(tokj),
        "corpus_sha256": dict(FROZEN_SHA), "dev_rows": len(dev_rows),
        "primary_variant": HYBRID_VARIANT, "ablation_variant": "D",
        "chosen_floor_D": metrics["D"]["chosen"]["floor"],
        "chosen_floor_E": metrics[HYBRID_VARIANT]["chosen"]["floor"],
        "metrics": metrics,
        "bars_D": score_bars(shim, "D"),
        "bars_E": score_bars(shim, HYBRID_VARIANT,
                             p95_ms=fill_lat["p95_ms"]),
        "latency_bare_forward_cpu": latency_bare,
        "latency_full_fill_cpu": fill_lat,
        "ensemble_rss": rss,
        "delta": delta,
    }
    (cand_dir / f"{stem}-hybridmetrics.json").write_text(
        json.dumps(result, indent=1, default=str))
    print(f"decode-D bars: {json.dumps(result['bars_D'])}", flush=True)
    print(f"decode-E bars: {json.dumps(result['bars_E'])}", flush=True)
    return result


def hybrid_verdict(results: list[dict]) -> dict:
    """The frozen verdict rule: HYBRID PASSES only if decode-E meets ALL of
    S1-S5 on BOTH seeds; ANY S1/S2 miss -> SECOND and FINAL falsification
    and the registered fallback (Qwen-0.5B + grammar PoC) fires."""
    seeds = {}
    s1s2_missed = []
    for r in results:
        be = r["bars_E"]
        seeds[str(r["seed"])] = {
            "int8_sha256": r["int8_sha256"],
            "int8_bytes": r["int8_bytes"],
            "tokenizer_sha256": r["tokenizer_sha256"],
            "chosen_floor_D": r["chosen_floor_D"],
            "chosen_floor_E": r["chosen_floor_E"],
            "bars_D": r["bars_D"], "bars_E": be,
            "delta_counts": r["delta"]["counts"],
            "delta_rows": r["delta"]["rows"],
            "delta_examples": r["delta"]["examples"],
            "latency_bare_forward_cpu": r["latency_bare_forward_cpu"],
            "latency_full_fill_cpu": r["latency_full_fill_cpu"],
            "ensemble_rss": r["ensemble_rss"],
        }
        if not (be["S1_exact_fill"]["pass"] and be["S2_invented_rate"]["pass"]):
            s1s2_missed.append(r["seed"])
    all_pass = all(r["bars_E"]["PASS"] for r in results)
    if all_pass:
        verdict, reason = "HYBRID PASSES", (
            "decode-E meets S1-S5 on both seeds "
            f"({', '.join('A-s' + str(r['seed']) for r in results)}); the "
            "unified architecture lives and phase 2 is unblocked")
    elif s1s2_missed:
        verdict, reason = "SECOND FALSIFICATION", (
            "S1/S2 miss on seed(s) "
            f"{', '.join('A-s' + str(s) for s in s1s2_missed)} — the hybrid "
            "is the architecture's SECOND and FINAL falsification; the "
            "registered fallback (Qwen-0.5B + grammar PoC) fires")
    else:
        verdict, reason = "SECOND FALSIFICATION", (
            "an S3/S4/S5 bar miss on the registered PASS rule (S1/S2 held); "
            "recorded verbatim per the amendment")
    return {"arm": ARM, "spec": SPEC, "date": "2026-10-10",
            "increment": "HYBRID DECODE INCREMENT (2026-10-10)",
            "mechanism": "decode-E: word-overlap attribution (never strict "
                         "containment) + glue-only end-boundary snapping "
                         "(extend floor 0.15, shipped span_fill parity); "
                         "whitespace is a hard snapping limit; the seed is "
                         "decode-D's construct on word-pooled scores",
            "registered_bars": {"S1_floor": 0.9165, "S2_ceiling": 0.0441,
                                "S3_floor": 0.5, "S3_rows": 117,
                                "S4": "int8 <= 200 MB; full-serving-path "
                                      "CPU p95 <= 50 ms (re-measured); "
                                      "ensemble < 2048 MB",
                                "S5": "zero enum violations"},
            "primary_variant": HYBRID_VARIANT, "ablation_variant": "D",
            "corpus_sha256": dict(FROZEN_SHA),
            "seeds": seeds,
            "verdict": verdict, "verdict_reason": reason}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="unified-span-spike: phase-1 architecture spike "
                    "(dev-only; see unified-span-extractor)")
    ap.add_argument("--candidate", choices=("A", "B", "both"), default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true",
                    help="reuse existing artifacts, re-run measurement")
    ap.add_argument("--hybrid", action="store_true",
                    help="HYBRID DECODE INCREMENT (2026-10-10): decode-D vs "
                         "decode-E on the STAGED int8 bytes (no training, no "
                         "GPU, models/ untouched); measures --candidate A at "
                         "seeds 42 and 43 and writes hybrid-verdict.json")
    args = ap.parse_args()
    seed_everything(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = require_stamp()
    train_rows = _rows(FROZEN["train"])
    dev_rows = _rows(FROZEN["dev"])
    if len(train_rows) != FROZEN_N["train"] \
            or len(dev_rows) != FROZEN_N["dev"]:
        sys.exit("refusing: frozen row counts drifted")
    if args.hybrid:
        if args.candidate not in ("A", "B"):
            sys.exit("--hybrid measures ONE candidate's staged bytes; pass "
                     "--candidate A (the registered hybrid bytes)")
        results = [run_hybrid(args.candidate, seed, dev_rows)
                   for seed in HYBRID_SEEDS]
        verdict = hybrid_verdict(results)
        out_path = OUT / "hybrid-verdict.json"
        out_path.write_text(json.dumps(verdict, indent=1, default=str))
        print(f"verdict: {verdict['verdict']} -> {out_path}")
        print(json.dumps({f"A-s{s}": v["bars_E"]
                          for s, v in verdict["seeds"].items()},
                         indent=1, default=str))
        return 0
    keys = ("A", "B") if args.candidate == "both" else (args.candidate,)
    results = {}
    for key in keys:
        print(f"=== candidate {key}: {CANDIDATES[key]['label']} "
              f"(seed {args.seed}) ===", flush=True)
        r = run_candidate(key, args.seed, train_rows, dev_rows,
                          args.eval_only, stamp)
        r["bars"] = score_bars(r)
        results[key] = r
        print(json.dumps(r["bars"], indent=1, default=str))
    verdict = {
        "arm": ARM, "spec": SPEC, "date": "2026-10-10", "seed": args.seed,
        "primary_variant": PRIMARY_DECODE,
        "candidates": {k: {"label": v["candidate"],
                           "int8_sha256": v["int8_sha256"],
                           "int8_path": v["int8_path"],
                           "bars": v["bars"],
                           "chosen_floor": v["chosen_floor"],
                           "decode_C_chosen": v["metrics"]["C"]["chosen"],
                           "latency_cpu": v.get("latency_cpu"),
                           "ensemble_rss": v.get("ensemble_rss")}
                       for k, v in results.items()},
        "decision_rule": "smallest candidate meeting S1-S5; none -> the "
                         "architecture is falsified (registered fallback: "
                         "Qwen-0.5B + grammar PoC, NOT run by this spike)",
    }
    passing = [k for k, v in results.items() if v["bars"]["PASS"]]
    if len(passing) == 2:
        verdict["verdict"] = "B"
        verdict["verdict_reason"] = "both candidates pass; B is the smaller"
    elif len(passing) == 1:
        verdict["verdict"] = passing[0]
        verdict["verdict_reason"] = f"only {passing[0]} meets S1-S5"
    else:
        verdict["verdict"] = "FALSIFIED"
        verdict["verdict_reason"] = ("no candidate meets S1-S5; phase 2 of "
                                     "the encoder-only pivot is dead per "
                                     "the registered decision rule")
    out_path = OUT / f"spike-verdict-s{args.seed}.json"
    out_path.write_text(json.dumps(verdict, indent=1, default=str))
    print(f"PASS per candidate: "
          f"{ {k: bool(results[k]['bars']['PASS']) for k in results} }")
    print(f"verdict: {verdict['verdict']} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
