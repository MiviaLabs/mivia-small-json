#!/usr/bin/env python3
"""Fine-tune the string-param span scorer (registered arm
``string-param-span-fill``, string-param-span-fill).

A pair-scoring fine-tune of the ROUTER LINEAGE checkpoint (mMiniLM-L12,
data/incoming/models/mm12-text-s42/model) into a span scorer: the pair
(sentence A = slot rendering, sentence B = request) is encoded once and a
per-token linear head scores every request token for membership in the
slot's value span. This is the arm's ONE registered deviation from the
router recipe (the scoring head); base checkpoint, recipe, and seed are
the router lineage's, unchanged.

Training pairs come from the corpus the audit-gated builder wrote:
the round-1 frozen train split + the builder's bfcl-silver + harvest-gold
files. Dev-side floor tuning uses the frozen dev split ONLY; the reserve
(data/heldout/spanfill-reserve.jsonl) is NEVER loaded here — it is
touched exactly once by the arm's one-shot bar B1 measurement.

Modes:
    (default)      GPU fine-tune + int8 ONNX export + dev floor tuning.
                   Requires CUDA: hard-fails with a clear message when
                   absent (Governing rule 9: falling back to CPU is a
                   stated reason, never an accident).
    --dry-run      2 batches through the training loop on GPU / MPS /
                   CPU-fallback (a loud warning on fallback); proves the
                   script executes without writing a candidate.
    --selfcheck    run 5 fixture pairs end-to-end through the WIRED
                   runtime decoder (mivia_small_json.span_fill) against a
                   candidate artifact; verifies loader contract, floors,
                   greedy decode, exclusivity, strict coercion.

The trainer writes ONLY into the gitignored candidates area
(data/incoming/models/stringparam-s42/): the int8 ONNX + tokenizer +
decode-meta sidecar (emit/extend floors BOUND to the artifact bytes via
the meta's artifact_sha256 — the loader refuses a mismatch) + sha256
sidecars. It NEVER promotes: nothing under models/ is touched; promotion
is the standing checklist's job after the pre-registered bars are
measured.

Wave-2 command (after the data plan is un-blocked and the audit gate is
extended per the builder's report):

    flock /tmp/mivia_gpu.lock python scripts/train_stringparam_scorer.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))  # noqa: E402

from mivia_small_json import span_fill as sf  # noqa: E402
from mivia_small_json.determinism import seed_everything  # noqa: E402

# Router lineage (string-param-span-fill: same base
# checkpoint, recipe, and seed; the head is the registered deviation).
BASE_CKPT = ROOT / "data/incoming/models/mm12-text-s42/model"
CANDIDATES = ROOT / "data/incoming/models/stringparam-s42"
STEM = "stringparam-scorer"
SEED = 42
RECIPE = {"epochs": 3, "batch": 32, "lr": 3e-5, "warmup_frac": 0.1,
          "clip": 1.0, "max_len": 256, "seed": SEED, "weight_decay": 0.01}

FROZEN_TRAIN = ROOT / "data/real/spanfill/train.jsonl"
FROZEN_DEV = ROOT / "data/real/spanfill/dev.jsonl"
SILVER = ROOT / "data/real/spanfill-stringparam/bfcl-silver.jsonl"
GOLD = ROOT / "data/real/spanfill-stringparam/harvest-gold.jsonl"
# the audit stamp the standing gate writes; the trainer refuses without it
AUDIT_STAMP = ROOT / ".scratch/audit-mix-stamp.json"

EMIT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
EXTEND_GRID = (0.15, 0.25, 0.35)
DEV_INVENTED_MAX = 0.10  # the round-1 dev budget (pre-registered)

_TOKEN_RE = re.compile(r"\w+")


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line
            in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def slot_text(param: str, prop: dict) -> str:
    """Sentence A — the same rendering the runtime scorer uses
    (span_fill._EncoderPairScorer.slot_text; one source of truth)."""
    return sf._EncoderPairScorer.slot_text(param, prop)


def gold_char_span(request: str, value) -> tuple[int, int] | None:
    """The gold value's verbatim char span in the request (first exact
    occurrence); None when absent — a value the request does not carry
    verbatim cannot teach span selection and is dropped mechanically."""
    needle = value if isinstance(value, str) else json.dumps(value)
    idx = request.find(needle)
    return None if idx < 0 else (idx, idx + len(needle))


def pair_examples(rows: list[dict]):
    """(slot_text, request, gold token mask) training examples, one per
    (row, required slot). Positives are the subtokens inside the gold
    value's char span; rows whose gold value is absent verbatim are
    skipped (mechanical honesty, never fabricated)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(BASE_CKPT))
    examples = []
    skipped = 0
    for row in rows:
        request = row["request"]
        props = (row["tool"].get("parameters") or {}).get("properties") or {}
        for slot, value in row["gold"].items():
            span = gold_char_span(request, value)
            if span is None:
                skipped += 1
                continue
            enc = tok(request, text_pair=slot_text(slot, props.get(slot) or {}),
                      truncation=True, max_length=RECIPE["max_len"],
                      return_offsets_mapping=True)
            # NOTE (wave-B3 fix, 2026-10-10): the argument is text_pair=,
            # NOT pair= — transformers has no `pair` parameter and
            # silently swallows the unknown kwarg (verified: decode shows
            # <s> A </s> with NO slot segment). With pair= the model
            # trained SLOT-BLIND (identical per-slot distributions: it
            # learned value-word-ness, never value-to-slot pairs) and the
            # runtime raw-tokenizers encode(request, pair=...) would have
            # fed a B segment the model never saw.
            mask = [0] * len(enc["input_ids"])
            for i, (s, e) in enumerate(enc["offset_mapping"]):
                if s == e:  # special tokens
                    continue
                # the request is sequence 0 (text), the slot rendering
                # sequence 1 (pair) — positives live on sequence 0
                if enc.sequence_ids()[i] == 0 and s >= span[0] \
                        and e <= span[1]:
                    mask[i] = 1
            if not any(mask):
                skipped += 1
                continue
            examples.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids",
                                          [0] * len(enc["input_ids"])),
                "span_mask": mask})
    n_pos = sum(sum(e["span_mask"]) for e in examples)
    print(f"pair examples: {len(examples)} "
          f"(positive tokens {n_pos}, skipped {skipped})")
    return examples, tok, int(n_pos)


class SpanPairScorer:
    """Encoder backbone + per-token linear head (the registered head
    deviation). Exported to ONNX with per-token logits (B, T, 1)."""

    def __init__(self, tok):
        import torch
        from transformers import AutoModel

        import torch.nn as nn

        self.torch = torch
        self.tok = tok
        backbone = AutoModel.from_pretrained(
            str(BASE_CKPT), attn_implementation="eager")
        hidden = backbone.config.hidden_size

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = backbone
                self.head = nn.Linear(hidden, 1)

            def forward(self, input_ids, attention_mask, token_type_ids):
                out = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
                return self.head(out.last_hidden_state)  # (B, T, 1)

        self.net = _Net()


def _pad_batch(examples: list[dict]):
    """Pad a batch to max_length (house recipe: pad to the recipe cap so
    the export-time static shapes and the training shapes agree)."""
    import torch

    n = RECIPE["max_len"]

    def pad(key, pad_value):
        return torch.tensor(
            [e[key][:n] + [pad_value] * max(0, n - min(len(e[key]), n))
             for e in examples], dtype=torch.long)

    return {"input_ids": pad("input_ids", 0),
            "attention_mask": pad("attention_mask", 0),
            "token_type_ids": pad("token_type_ids", 0),
            # float: the BCE-with-logits target
            "span_mask": pad("span_mask", 0).to(torch.float32)}


def train(examples, tok, device: str) -> None:
    import torch

    net = net_holder["net"]
    net.to(device)
    net.train()
    enc = _pad_batch(examples)
    enc = {k: v for k, v in enc.items()}
    span = enc.pop("span_mask")
    steps_per_epoch = (len(examples) + RECIPE["batch"] - 1) // RECIPE["batch"]
    total = max(1, steps_per_epoch * RECIPE["epochs"])
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["lr"],
                            weight_decay=RECIPE["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=RECIPE["lr"], total_steps=total,
        pct_start=RECIPE["warmup_frac"], anneal_strategy="linear")
    pos_weight = torch.tensor(
        [max(1.0, (span.numel() - span.sum()) / max(1.0, span.sum()))],
        device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    gen = torch.Generator()
    gen.manual_seed(RECIPE["seed"])
    t_all = time.perf_counter()
    epochs_meta = []
    for epoch in range(RECIPE["epochs"]):
        perm = torch.randperm(len(examples), generator=gen).tolist()
        losses, nan = [], 0
        t0 = time.perf_counter()
        for s in range(0, len(perm), RECIPE["batch"]):
            idx = perm[s:s + RECIPE["batch"]]
            batch = {k: v[idx].to(device)
                     for k, v in enc.items()}
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits = net(**batch).squeeze(-1)
            # only supervised positions carry loss: request-side tokens
            attn = batch["attention_mask"].float()
            loss = lossf(logits * attn, span[idx].to(device) * attn)
            if not torch.isfinite(loss):
                nan += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), RECIPE["clip"])
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(float(loss))
        epochs_meta.append({"epoch": epoch + 1,
                            "mean_loss": sum(losses) / max(1, len(losses)),
                            "nan_steps": nan,
                            "seconds": round(time.perf_counter() - t0, 1)})
        print(epochs_meta[-1], flush=True)
    net.eval()
    net_holder["meta_epochs"] = epochs_meta
    net_holder["train_seconds"] = round(time.perf_counter() - t_all, 1)
    net.eval()


net_holder: dict = {}


class CandidateScorer:
    """The candidate as a runtime scorer (torch path, dev tuning +
    selfcheck): mirrors _EncoderPairScorer.pair_probs on the torch net.

    Mirror contract (kept true, increment 3, 2026-10-10): this class
    mirrors the PRODUCTION scorer's per-token pooling only; the DECODE
    is not duplicated — ``tune_floors`` and ``selfcheck`` consume the
    production ``span_fill.learned_fill`` through the module injection
    seam, so dev grids and selfchecks always run the shipped decode,
    including increment 3's per-slot top-K (slot, span) assignment
    (K = 3, floors unchanged). Any future pooling change here MUST
    re-mirror span_fill._EncoderPairScorer.pair_probs in the same
    commit."""

    def __init__(self, net, tok):
        self.net = net
        self.tok = tok

    def pair_probs(self, request: str, param: str, prop: dict):
        import torch

        enc = self.tok(request, text_pair=slot_text(param, prop),
                       truncation=True, max_length=RECIPE["max_len"],
                       return_tensors="pt",
                       return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        seq_ids = enc.sequence_ids(0)
        enc.pop("offset_mapping")
        # The mMiniLM/XLM-R tokenizer returns NO token_type_ids; the
        # backbone's convention is all-zero type ids — what pair_examples
        # and the ONNX export already use. Without this fallback the
        # net(**enc) call raises on EVERY pair and the never-raise
        # contract of _slot_token_probs silently degrades every row to
        # (None, []): the injection of the candidate scorer measured
        # nothing (second half of the wave-B2 null-scorer defect,
        # fixed 2026-10-10).
        if "token_type_ids" not in enc:
            enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
        device = next(self.net.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.net(**enc).squeeze(-1)[0].tolist()
        out = []
        matches = list(_TOKEN_RE.finditer(request))
        for m in matches:
            best = 0.0
            for i, (s, e) in enumerate(offsets):
                if i >= len(logits) or s == e:
                    continue
                # OVERLAP pooling (registered increment 2026-10-10,
                # whitespace-bounded span extension) — MUST mirror
                # span_fill._EncoderPairScorer.pair_probs exactly so
                # this dev grid measures the decode production runs: a
                # subtoken straddling a glue boundary (`-07` in
                # `app-07`) is the glued token's evidence; the old
                # strict-containment test dropped it and the token read
                # 0.0 (the falsified candidate's truncation mechanism).
                if seq_ids[i] == 0 and s < m.end() and e > m.start():
                    best = max(best, 1.0 / (1.0 + math.exp(-logits[i])))
            out.append(best)
        return out


def tune_floors(scorer, dev_rows: list[dict]) -> tuple[float, float, dict]:
    """Dev-only emit/extend floor grid, the round-1 budget (invented <=
    0.10) as the binding constraint. Floors travel with the artifact.

    The candidate scorer is INJECTED into the span-fill module seam for
    the grid: ``learned_fill`` reads ``sf._SPANFILL_CLF`` (via the
    sidecar loader, which negative-caches when models/ has no artifact),
    not this function's argument — without the injection every grid cell
    measured the artifact-free null path and the "chosen" floors were
    the first grid point (trainer defect, fixed 2026-10-10). The
    previous module state is restored after, mirroring ``selfcheck``."""
    saved = (sf._SPANFILL_CLF, sf._SPANFILL_LOADED)
    sf._SPANFILL_CLF, sf._SPANFILL_LOADED = scorer, True
    try:
        best = None
        for emit in EMIT_GRID:
            for extend in EXTEND_GRID:
                sf.configure_spanfill_floors(emit, extend)
                exact = emitted = wrong = 0
                for row in dev_rows:
                    filled, missing = sf.learned_fill(
                        row["tool"], row["request"])
                    gold = row["gold"]
                    for k, v in (filled or {}).items():
                        emitted += 1
                        if k not in gold or gold[k] != v:
                            wrong += 1
                    if (filled and not missing and set(filled) == set(gold)
                            and all(gold.get(k) == v
                                    for k, v in filled.items())):
                        exact += 1
                m = {"n": len(dev_rows),
                     "exact_fill": round(exact / max(1, len(dev_rows)), 4),
                     "invented_rate": round(wrong / emitted, 4)
                     if emitted else 0.0}
                if m["invented_rate"] > DEV_INVENTED_MAX:
                    continue
                if best is None or m["exact_fill"] > best[2]["exact_fill"]:
                    best = (emit, extend, m)
        if best is None:  # nothing met the budget: lowest invented wins
            for emit in EMIT_GRID:
                for extend in EXTEND_GRID:
                    sf.configure_spanfill_floors(emit, extend)
                    wrong = emitted = 0
                    for row in dev_rows:
                        filled, _ = sf.learned_fill(row["tool"],
                                                    row["request"])
                        gold = row["gold"]
                        for k, v in (filled or {}).items():
                            emitted += 1
                            if k not in gold or gold[k] != v:
                                wrong += 1
                    rate = round(wrong / emitted, 4) if emitted else 0.0
                    if best is None or rate < best[2]["invented_rate"]:
                        best = (emit, extend,
                                {"n": len(dev_rows), "exact_fill": None,
                                 "invented_rate": rate})
    finally:
        sf._SPANFILL_CLF, sf._SPANFILL_LOADED = saved
    emit, extend, metrics = best
    sf.configure_spanfill_floors(emit, extend)
    return float(emit), float(extend), metrics


def export_and_quantize(tok, device: str) -> tuple[Path, Path, Path]:
    import torch

    net = net_holder["net"]
    net.to("cpu").eval()
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    fp32 = CANDIDATES / f"{STEM}-fp32.onnx"
    int8 = CANDIDATES / f"{STEM}-int8.onnx"
    tokj = CANDIDATES / f"{STEM}-tokenizer.json"
    tok.backend_tokenizer.save(str(tokj))
    dummy = tok(["a request"], text_pair=["slot | type: string"],
                truncation=True, max_length=RECIPE["max_len"],
                padding="max_length", return_tensors="pt")
    ttids = dummy.get("token_type_ids")
    torch.onnx.export(
        net, (dummy["input_ids"], dummy["attention_mask"],
              ttids if ttids is not None
              else torch.zeros_like(dummy["input_ids"])),
        fp32, input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "token_type_ids": {0: "batch", 1: "seq"},
                      "logits": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return fp32, int8, tokj


def agreement_gate(fp32: Path, int8: Path, tok) -> dict:
    """int8 vs fp32 agreement on a seeded sample of the training text
    (the chatood4 gate, adapted to a per-token head: pearson on logits +
    sigmoid-sign agreement).

    Measured on IN-TEXT positions only (attention_mask == 1) — the exact
    position set the runtime decoder scores. The padded positions are
    undefined garbage in BOTH graphs (never supervised, never decoded;
    wave-B2 decomposition: padding pearson 0.685-0.891 vs in-text
    >= 0.999902, .scratch/adjud/tiebreak/
    agreement-gate-decomposition.json) and comparing them manufactured a
    false gate failure (trainer defect, fixed 2026-10-10). Thresholds
    unchanged: pearson >= 0.99 AND sign-agreement >= 0.995."""
    import numpy as np
    import onnxruntime as ort

    sample = [
        ("restart nginx on staging", "service | the unit | type: string"),
        ("ship release=2026.40 to staging", "environment | type: string"),
        ("look up order ORD-48213", "order_id | type: string"),
        ("post 'deploy done' to incidents", "body | type: string"),
        ("rotate the key for ci@svc in production", "account | type: string"),
    ]
    feeds = []
    for request, slot in sample:
        enc = tok(request, text_pair=slot, truncation=True,
                  max_length=RECIPE["max_len"], padding="max_length",
                  return_tensors="np")
        feeds.append({"input_ids": enc["input_ids"].astype(np.int64),
                      "attention_mask":
                          enc["attention_mask"].astype(np.int64),
                      "token_type_ids":
                          enc.get("token_type_ids",
                                  np.zeros_like(enc["input_ids"])
                                  ).astype(np.int64)})
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.intra_op_num_threads = 2
    s_fp32 = ort.InferenceSession(str(fp32), so,
                                  providers=["CPUExecutionProvider"])
    s_int8 = ort.InferenceSession(str(int8), so,
                                  providers=["CPUExecutionProvider"])
    lf, li, n_text, lf_all, li_all = [], [], 0, [], []
    for feed in feeds:
        attn = feed["attention_mask"].astype(bool).ravel()
        fp = s_fp32.run(None, feed)[0].reshape(-1)
        i8 = s_int8.run(None, feed)[0].reshape(-1)
        lf_all.append(fp)
        li_all.append(i8)
        lf.append(fp[attn])  # in-text positions only
        li.append(i8[attn])
        n_text += int(attn.sum())
    lf, li = np.concatenate(lf), np.concatenate(li)
    pear = float(np.corrcoef(lf, li)[0, 1])
    agree = float(((lf >= 0) == (li >= 0)).mean())
    # all-positions numbers recorded for continuity with the wave-B2
    # decomposition; the GATE acts on the in-text metrics only
    pear_all = float(np.corrcoef(np.concatenate(lf_all),
                                 np.concatenate(li_all))[0, 1])
    agree_all = float(((np.concatenate(lf_all) >= 0)
                       == (np.concatenate(li_all) >= 0)).mean())
    print(f"agreement: pearson={pear:.6f} sign={agree:.4f} "
          f"(in-text n={n_text}; all-positions pearson={pear_all:.6f} "
          f"sign={agree_all:.4f} — padding is garbage, never decoded)")
    if pear < 0.99 or agree < 0.995:
        print("int8/fp32 agreement gate FAILED", file=sys.stderr)
        raise SystemExit(2)
    return {"pearson": pear, "sign_agreement": agree, "n": n_text,
            "position_set": "in_text_attention_mask_1",
            "pearson_all_positions": pear_all,
            "sign_agreement_all_positions": agree_all}


def write_candidate(int8: Path, tokj: Path, emit: float, extend: float,
                    dev_metrics: dict, agreement: dict,
                    corpus_meta: dict) -> Path:
    sha = sha256_of(int8)
    (CANDIDATES / (int8.name + ".sha256")).write_text(sha + "\n")
    (CANDIDATES / (tokj.name + ".sha256")).write_text(
        sha256_of(tokj) + "\n")
    meta = {
        "arm": "string-param-span-fill",
        "spec": "string-param-span-fill",
        "base_checkpoint": str(BASE_CKPT),
        "recipe": RECIPE,
        "head": "per-token linear span head over the router-lineage "
                "encoder (the registered deviation)",
        "seed": SEED,
        "corpus": corpus_meta,
        "emit_floor": emit,
        "extend_floor": extend,
        "dev_metrics": dev_metrics,
        "artifact_sha256": sha,  # floors bound to THESE bytes (rule 10)
        "max_len": RECIPE["max_len"],
        "int8_fp32_agreement": agreement,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": net_holder.get("meta_epochs", []),
        "train_seconds": net_holder.get("train_seconds"),
        "note": "CANDIDATE — never promoted; the loader reads models/ "
                "only, promotion is the standing checklist's job",
    }
    meta_path = CANDIDATES / f"{STEM}-trainmeta.json"
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    (CANDIDATES / (meta_path.name + ".sha256")).write_text(
        sha256_of(meta_path) + "\n")
    return meta_path


# --- selfcheck ---------------------------------------------------------------

SELFCHECK_FIXTURES = [
    {"tool": {"name": "restart_service", "description":
              "Restart a systemd service on a named host.",
              "parameters": {"properties": {
                  "service": {"type": "string",
                              "description": "The systemd service unit."},
                  "host": {"type": "string",
                           "description": "The target host."}},
                  "required": ["service", "host"]}},
     "request": "restart nginx on staging",
     "expect_gold": {"service": "nginx", "host": "staging"}},
    {"tool": {"name": "deploy_release", "description":
              "Ship a release to a target environment.",
              "parameters": {"properties": {
                  "release": {"type": "string"},
                  "environment": {"type": "string",
                                  "enum": ["staging", "production",
                                           "canary"]}},
                  "required": ["release", "environment"]}},
     "request": "ship release 2026.40 to the production environment",
     "expect_gold": {"release": "2026.40", "environment": "production"}},
    {"tool": {"name": "lookup_order", "description":
              "Fetch an order by identifier.",
              "parameters": {"properties": {
                  "order_id": {"type": "string"}},
                  "required": ["order_id"]}},
     "request": "what is the status of order ORD-48213?",
     "expect_gold": {"order_id": "ORD-48213"}},
    {"tool": {"name": "send_message", "description":
              "Post a message to a channel.",
              "parameters": {"properties": {
                  "channel": {"type": "string"},
                  "body": {"type": "string"}},
                  "required": ["channel", "body"]}},
     "request": "post 'deploy finished' to the incidents channel",
     "expect_gold": {"channel": "incidents", "body": "deploy finished"}},
    {"tool": {"name": "rotate_key", "description":
              "Rotate an API credential for a service account.",
              "parameters": {"properties": {
                  "account": {"type": "string"},
                  "environment": {"type": "string",
                                  "enum": ["staging", "production"]}},
                  "required": ["account", "environment"]}},
     "request": "rotate the api key for ci-runner@svc.mivia.example.com in staging",
     "expect_gold": {"account": "ci-runner@svc.mivia.example.com",
                     "environment": "staging"}},
]


def selfcheck(meta_path: Path) -> int:
    """5 fixture pairs end-to-end through the WIRED decoder against the
    candidate bytes (injected, never promoted). Verifies the loader
    contract (meta floors honored), greedy decode, span exclusivity and
    strict coercion on the candidate scorer."""
    meta = json.loads(meta_path.read_text())
    scorer = CandidateScorer(net_holder["net"], net_holder["tok"])
    saved_floors = (sf._SPANFILL_EMIT_FLOOR, sf._SPANFILL_EXTEND_FLOOR)
    sf.configure_spanfill_floors(meta["emit_floor"], meta["extend_floor"])
    saved = (sf._SPANFILL_CLF, sf._SPANFILL_LOADED)
    sf._SPANFILL_CLF, sf._SPANFILL_LOADED = scorer, True
    try:
        ok = 0
        for fx in SELFCHECK_FIXTURES:
            filled, missing = sf.fill_arguments(fx["tool"], fx["request"])
            gold = fx["expect_gold"]
            hit = set(filled or {}) == set(gold) and \
                all(gold[k] == v for k, v in (filled or {}).items())
            print(f"  {'PASS' if hit else 'MISS'} "
                  f"{fx['tool']['name']}: filled={filled} missing={missing}")
            ok += 1 if hit else 0
        print(f"selfcheck: {ok}/{len(SELFCHECK_FIXTURES)} exact")
        return 0 if ok == len(SELFCHECK_FIXTURES) else 1
    finally:
        sf._SPANFILL_CLF, sf._SPANFILL_LOADED = saved
        sf.configure_spanfill_floors(*saved_floors)


def _require_gpu(dry_run: bool) -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if dry_run:
        if getattr(torch.backends, "mps", None) \
                and torch.backends.mps.is_available():
            print("WARNING: --dry-run falling back to MPS", flush=True)
            return "mps"
        print("WARNING: --dry-run falling back to CPU", flush=True)
        return "cpu"
    print("REFUSING: no CUDA device. The arm trains on the GPU "
          "(Governing rule 9); a CPU fallback is a stated reason, never "
          "an accident. Use --dry-run for the 2-batch smoke.",
          file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    global net_holder
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="2 batches through the training loop, no "
                         "candidate written")
    ap.add_argument("--selfcheck", action="store_true",
                    help="5 fixtures through the wired decoder with a "
                         "freshly trained (in-memory) scorer")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    device = _require_gpu(args.dry_run)
    seed_everything(RECIPE["seed"])

    corpus_meta = {"frozen_train": {"path": str(FROZEN_TRAIN),
                                    "sha256": sha256_of(FROZEN_TRAIN)},
                   "frozen_dev": {"path": str(FROZEN_DEV),
                                  "sha256": sha256_of(FROZEN_DEV)}}
    train_rows = _rows(FROZEN_TRAIN)
    dev_rows = _rows(FROZEN_DEV)
    for name, path in (("bfcl-silver", SILVER), ("harvest-gold", GOLD)):
        rows = _rows(path)
        corpus_meta[name] = {"path": str(path),
                             "sha256": sha256_of(path) if rows else None,
                             "rows": len(rows)}
        train_rows += rows
    print(f"corpus: frozen train {len(_rows(FROZEN_TRAIN))}, dev "
          f"{len(dev_rows)}, + "
          f"{corpus_meta['bfcl-silver']['rows']} silver + "
          f"{corpus_meta['harvest-gold']['rows']} gold")

    if not args.dry_run:
        if not BASE_CKPT.is_dir():
            print(f"base checkpoint missing: {BASE_CKPT}", file=sys.stderr)
            return 2
        stamp = AUDIT_STAMP
        stamp_data = json.loads(stamp.read_text()) \
            if stamp.is_file() else {}
        # Fail CLOSED (house pattern, retrain_router_chatood4.py): the
        # stamp must be passed AND cover THIS arm — a stale `passed` stamp
        # from another arm's corpus (e.g. the router round's, which sits at
        # this exact path) must not let a GPU fine-tune start unaudited
        # (AGENTS.md rule 13; review 2026-10-10). The gate extension writes
        # the field when this arm's mix is audited.
        if stamp_data.get("passed") is not True \
                or stamp_data.get("arm") != "string-param-span-fill":
            print("REFUSING: the pre-training audit gate has not passed "
                  "for THIS arm (.scratch/audit-mix-stamp.json missing, "
                  "unpassed, or not carrying arm="
                  "'string-param-span-fill'); extend the gate for this "
                  "arm first (the builder's report names the extension)",
                  file=sys.stderr)
            return 2

    holder = SpanPairScorer(AutoTokenizer.from_pretrained(str(BASE_CKPT)))
    net_holder = {"net": holder.net, "tok": holder.tok}
    examples, tok, n_pos = pair_examples(train_rows)
    if args.dry_run:
        tiny = examples[:2 * RECIPE["batch"]]
        if not tiny:
            print("no examples — corpus empty", file=sys.stderr)
            return 2
        net_holder["net"].to(device)
        net_holder["net"].train()
        opt = torch.optim.AdamW(net_holder["net"].parameters(),
                                lr=RECIPE["lr"])
        for s in range(0, len(tiny), RECIPE["batch"]):
            enc = _pad_batch(tiny[s:s + RECIPE["batch"]])
            span = enc.pop("span_mask")
            batch = {k: v.to(device) for k, v in enc.items()}
            logits = net_holder["net"](**batch).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, span.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()
            print(f"dry-run batch: loss {float(loss):.4f}")
        print("dry-run OK (2 batches, no candidate written)")
        return 0

    train(examples, tok, device)
    # dev floor tuning runs on the GPU (the scorer moves tensors to the
    # net's device): the 21-point grid over ~1.4k dev rows is the slow
    # leg otherwise. Export/quantize are CPU-side and run after.
    emit, extend, dev_metrics = tune_floors(
        CandidateScorer(net_holder["net"], tok), dev_rows)
    print("dev-chosen floors:", json.dumps(
        {"emit_floor": emit, "extend_floor": extend, **dev_metrics}))
    net_holder["net"].to("cpu")
    fp32, int8, tokj = export_and_quantize(tok, device)
    agreement = agreement_gate(fp32, int8, tok)
    meta_path = write_candidate(int8, tokj, emit, extend, dev_metrics,
                                agreement, corpus_meta)
    print(f"candidate staged: {int8} (sha {sha256_of(int8)[:16]}...)")
    print(f"meta: {meta_path}")
    return selfcheck(meta_path)


if __name__ == "__main__":
    sys.exit(main())
