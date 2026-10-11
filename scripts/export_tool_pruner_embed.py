#!/usr/bin/env python3
"""Export the multilingual embedding tool-pruner (spec embed-tool-pruner).

ZERO-SHOT arm: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(Apache-2.0), mean-pool + L2-normalize INSIDE the graph, fp32 ONNX export,
int8 dynamic quantization (the house recipe,
scripts/export_encoder_onnx.py). Writes, after verification:

  models/tool-pruner-embed-int8.onnx        (+ .sha256; >100 MB -> also
                                             models/parts/ byte-split shards
                                             + manifest, gitignored in
                                             models/)
  models/tool-pruner-embed-tokenizer.json   (+ .sha256, the model's own
                                             XLM-R vocab copy, truncation 128)
  models/tool-pruner-embed-meta.json        (+ .sha256; provenance: model id,
                                             license, lineage, recipe, and
                                             the MEASURED verification record)

Verification before promotion (all recorded in the meta):
 1. identical texts -> bitwise identical embeddings (ORT CPU determinism);
 2. cosine sanity: an EN request is closer to its matching tool span than
    to an unrelated span (the absolute cosine is recorded — "sane" here
    means the matching-pair cosine is >= 0.5 on this model family);
 3. multilingual sanity: a German weather request ranks the weather tool
    above a DNA tool (the cross-lingual property BM25 lacks);
 4. the TOOL-SIDE BATCHING MEASUREMENT (spec Implementation Decisions):
    solo vs batched embeddings over real rendered spans from the committed
    BFCL pool + survivor-set identity on seeded catalogues. Tool-side
    batching ships only if this passes; the exact numbers are pinned in
    tests/test_embedpruner.py. The REQUEST side stays one forward, always.

Runs under the GPU flock like every GPU-box job (short CPU export; never
trains). Staging under data/incoming/models/ (gitignored); only verified
bytes are copied into models/.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LICENSE = "apache-2.0"
MAX_LEN = 128          # the model's sentence-transformers training profile
EMB_DIM = 384
PART_BYTES = 60 * 1024 * 1024   # the house 60 MiB byte-split convention
LOCK = "/tmp/mivia_gpu.lock"

STAGING = ROOT / "data" / "incoming" / "models"
MODELS = ROOT / "models"
PARTS = MODELS / "parts"
NAME = "tool-pruner-embed-int8.onnx"
TOK_NAME = "tool-pruner-embed-tokenizer.json"
META_NAME = "tool-pruner-embed-meta.json"

EN_REQUEST = "What is the weather like in Paris today?"
EN_WEATHER = "get_weather: Get the current weather forecast for a city."
EN_DNA = "sequence_dna: Sequence a DNA molecule by its identifier."
DE_REQUEST = "Wie ist das Wetter heute in Paris?"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        sha256_file(path) + "\n", encoding="utf-8")


def build_and_export() -> tuple[Path, Path, Path]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    STAGING.mkdir(parents=True, exist_ok=True)
    print(f"loading {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    net = AutoModel.from_pretrained(MODEL_ID)
    net.eval()

    class PoolEncoder(torch.nn.Module):
        """Mean-pool the token states (mask-weighted) + L2 normalize."""

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
            h = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return mean / mean.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)

    wrapped = PoolEncoder(net)

    fp32_path = STAGING / "tool-pruner-embed-fp32.onnx"
    int8_path = STAGING / NAME
    dummy_ids = torch.ones((1, 8), dtype=torch.int64)
    dummy_mask = torch.ones((1, 8), dtype=torch.int64)
    print("exporting fp32 ONNX ...", flush=True)
    torch.onnx.export(
        wrapped, (dummy_ids, dummy_mask), str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "embedding": {0: "batch"}},
        opset_version=17, dynamo=False)
    print("quantizing to int8 (dynamic, QInt8) ...", flush=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path),
                     weight_type=QuantType.QInt8)

    # Tokenizer copy: the model's own XLM-R vocab, truncation baked at 128.
    backend = tok.backend_tokenizer
    backend.enable_truncation(max_length=MAX_LEN)
    tok_path = STAGING / TOK_NAME
    tok_path.write_text(backend.to_str(pretty=True), encoding="utf-8")
    return fp32_path, int8_path, tok_path


def session_for(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def embed_batch(sess, tok, texts: list[str], pad_id: int):
    import numpy as np

    encs = tok.encode_batch([t if isinstance(t, str) else "" for t in texts])
    width = max(len(e.ids) for e in encs)
    ids = np.zeros((len(encs), width), dtype=np.int64)
    mask = np.zeros((len(encs), width), dtype=np.int64)
    for i, e in enumerate(encs):
        ids[i, :len(e.ids)] = e.ids
        mask[i, :len(e.ids)] = 1
    feed = {"input_ids": ids, "attention_mask": mask}
    if pad_id is not None:
        feed["input_ids"] = np.where(mask == 1, ids, pad_id)
    out = sess.run(["embedding"], feed)[0]
    # Consumers receive L2-normalized rows from the graph; re-normalize to
    # absorb the fp round-off a padded batch can introduce at the norm.
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-12, None)


def verify(int8_path: Path, tok_path: Path) -> dict:
    import numpy as np
    from tokenizers import Tokenizer

    import bench_guard_latency as bgl
    from mivia_small_json.tool_match import render_tool

    sess = session_for(int8_path)
    tok = Tokenizer.from_file(str(tok_path))
    tok.no_padding()
    pad_id = tok.token_to_id("<pad>")
    out_names = [o.name for o in sess.get_outputs()]
    assert out_names == ["embedding"], out_names

    def one(text: str) -> np.ndarray:
        return embed_batch(sess, tok, [text], pad_id)[0]

    # 1. determinism: identical texts -> bitwise identical embeddings
    a, b = one(EN_REQUEST), one(EN_REQUEST)
    assert np.array_equal(a, b), "embedding not deterministic"

    # 2. EN cosine sanity + 3. German cross-lingual sanity
    en_w = float(a @ one(EN_WEATHER))
    en_d = float(a @ one(EN_DNA))
    de = one(DE_REQUEST)
    de_w = float(de @ one(EN_WEATHER))
    de_d = float(de @ one(EN_DNA))
    assert en_w >= 0.5, f"EN matching-pair cosine too low: {en_w:.3f}"
    assert en_w > en_d, f"EN sanity failed: {en_w:.3f} vs {en_d:.3f}"
    assert de_w > de_d, f"DE sanity failed: {de_w:.3f} vs {de_d:.3f}"

    # 4. tool-side batching measurement on REAL rendered spans
    pool = bgl._tool_pool()
    spans = [render_tool(t) for t in pool]
    rng = random.Random(20261010)
    sample = [spans[i] for i in rng.sample(range(len(spans)), 64)]
    solo = np.stack([one(s) for s in sample])
    batched = np.concatenate([embed_batch(sess, tok, sample[i:i + 16], pad_id)
                              for i in range(0, len(sample), 16)])
    delta = float(np.abs(solo - batched).max())
    cos_solo = solo @ solo.T
    cos_batch = batched @ batched.T
    max_cos_delta = float(np.abs(cos_solo - cos_batch).max())

    # survivor-set identity: seeded catalogues (the dcagree recipe) with
    # real EN requests, survivors computed from solo vs batched span vectors
    rows = []
    for line in (ROOT / "data/incoming/external/bfcl/calls.jsonl") \
            .read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rec = json.loads(line)
            text = rec.get("text") or rec.get("raw") or ""
            head, sep, req = text.partition("\n\nUser request:")
            if sep:
                rows.append((json.loads(head[head.find("["):]),
                             req.strip()))

    def survivors(vecs: np.ndarray, qvec: np.ndarray, k: int = 12) -> list[int]:
        scores = vecs @ qvec
        n = len(scores)
        top = sorted(range(n), key=lambda i: (-float(scores[i]), i))[:min(k, n)]
        return sorted(top)

    canon = lambda t: json.dumps(t, sort_keys=True, ensure_ascii=False)  # noqa: E731
    sets_identical = True
    worst_top1_cos_delta = 0.0
    checked = 0
    for rid in range(min(30, len(rows))):
        own, request = rows[rid]
        own_keys = {canon(t) for t in own}
        own_names = {t.get("name") for t in own}
        rng = random.Random(20261010 + rid * 10007 + 50)
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        cat = list(own)
        for i in idx:
            t = pool[i]
            if canon(t) in own_keys or t.get("name") in own_names:
                continue
            cat.append(t)
            if len(cat) == 50:
                break
        spans = [render_tool(t) for t in cat]
        solo_v = np.stack([one(s) for s in spans])
        batch_v = embed_batch(sess, tok, spans, pad_id)
        q = one(request)
        s_solo = survivors(solo_v, q)
        s_batch = survivors(batch_v, q)
        checked += 1
        if s_solo != s_batch:
            sets_identical = False
        top1_delta = abs(float((solo_v @ q)[s_solo[0]]
                               - (batch_v @ q)[s_batch[0]]))
        worst_top1_cos_delta = max(worst_top1_cos_delta, top1_delta)
    return {
        "deterministic": True,
        "en_cosine_request_weather": round(en_w, 4),
        "en_cosine_request_dna": round(en_d, 4),
        "de_cosine_request_weather": round(de_w, 4),
        "de_cosine_request_dna": round(de_d, 4),
        "solo_vs_batch_max_vec_delta": delta,
        "solo_vs_batch_max_cos_delta": max_cos_delta,
        "batch_bitwise_identical": bool(delta == 0.0),
        "survivor_sets_checked": checked,
        "survivor_sets_identical_solo_vs_batch": bool(sets_identical),
        "worst_top1_cos_delta": round(worst_top1_cos_delta, 8),
    }


def main() -> int:
    t0 = time.time()
    with open(LOCK, "w") as lock_fh:
        print(f"acquiring {LOCK} ...", flush=True)
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        print("lock held; exporting", flush=True)
        fp32_path, int8_path, tok_path = build_and_export()
        report = verify(int8_path, tok_path)

    meta = {
        "arm": "embed-tool-pruner (Tier-2, zero-shot)",
        "spec": "embed-tool-pruner",
        "model_id": MODEL_ID,
        "license": LICENSE,
        "lineage": ("sentence-transformers paraphrase-multilingual-"
                    "MiniLM-L12-v2: XLM-R MiniLM-L12 backbone (118M params), "
                    "mean pooling + L2 normalization trained in; exported "
                    "zero-shot, no fine-tuning"),
        "graph": ("input_ids/attention_mask (int64, dynamic batch+seq) -> "
                  "embedding (fp32 rows, L2-normalized, inside the graph)"),
        "max_len": MAX_LEN,
        "emb_dim": EMB_DIM,
        "quantization": "onnxruntime quantize_dynamic, QInt8 weights",
        "verify": report,
        "decision": {
            "request_side": "one single-input forward, always",
            "tool_side_batching_allowed": bool(
                report["batch_bitwise_identical"]
                or report["solo_vs_batch_max_cos_delta"] < 1e-4),
        },
        "created": time.strftime("%Y-%m-%d"),
    }

    # promote verified bytes into models/
    MODELS.mkdir(exist_ok=True)
    shutil.copy2(int8_path, MODELS / NAME)
    shutil.copy2(tok_path, MODELS / TOK_NAME)
    (MODELS / META_NAME).write_text(json.dumps(meta, indent=1) + "\n",
                                    encoding="utf-8")
    write_sidecar(MODELS / NAME)
    write_sidecar(MODELS / TOK_NAME)
    write_sidecar(MODELS / META_NAME)

    # byte-split parts for the >100 MB artifact (house convention)
    target = MODELS / NAME
    if target.stat().st_size > 100 * 1024 * 1024:
        PARTS.mkdir(exist_ok=True)
        part_names = []
        with target.open("rb") as fh:
            while chunk := fh.read(PART_BYTES):
                pn = f"{NAME}.part{len(part_names):02d}"
                (PARTS / pn).write_bytes(chunk)
                part_names.append(pn)
        manifest = {"file": NAME, "parts": part_names,
                    "sha256": sha256_file(target),
                    "size": target.stat().st_size}
        (PARTS / f"{NAME}.manifest.json").write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print(f"parts: {len(part_names)} shards, manifest written")

    # keep SHIPPED.json current (the gates test fails otherwise)
    subprocess.run([sys.executable, str(ROOT / "scripts"
                                         / "refresh_shipped_manifest.py")],
                   check=True)

    print(json.dumps({k: v for k, v in meta.items() if k != "verify"},
                     indent=1))
    print("verify:", json.dumps(report, indent=1))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
