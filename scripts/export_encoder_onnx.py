#!/usr/bin/env python3
"""Export the fine-tuned router to int8 ONNX and re-measure.

Option #1: replace the hashed router with a fine-tuned MiniLM encoder.
This trains on the FULL mix, exports to ONNX, quantizes to int8, and
compares hashed vs PyTorch vs ONNX-int8 on held-out, the OOD sets, latency,
and artifact size.

Writes reports/spike-onnx.md and the artifacts under
data/incoming/models/ (gitignored). Requires torch, transformers,
onnxruntime, tokenizers (requirements-spike.txt).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.determinism import seed_everything  # noqa: E402
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = ROOT / "data" / "incoming" / "models"
REPORT = ROOT / "reports" / "spike-onnx.md"
EPOCHS = 3
BATCH = 32
# 512-token context with head-tail truncation (see encoder.HEAD_LEN/TAIL_LEN).
# The MiniLM encoder carries 512 positional embeddings natively.
MAX_LEN = 512
SEED = 42


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .split("\n") if line.strip()]


def train(seed: int = SEED):
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    from mivia_small_json import bootstrap, model

    bootstrap.register_example_schemas()
    X, y = model.training_mix()
    classes = sorted(set(y))
    index = {c: i for i, c in enumerate(classes)}
    print(f"full mix: {len(X)} rows, {len(classes)} classes")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    net = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(classes))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    net.to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-5)
    seed_everything(seed)
    for epoch in range(EPOCHS):
        total = 0.0
        for i in range(0, len(X), BATCH):
            batch = X[i:i + BATCH]
            labels = torch.tensor([index[c] for c in y[i:i + BATCH]]).to(device)
            enc = tok(batch, truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = net(**enc, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total += float(out.loss.detach())
        print(f"epoch {epoch + 1}: loss {total / max(1, len(X) // BATCH):.4f}")
    net.eval().to("cpu")
    return net, tok, classes


def export_onnx(net, tok, classes, seed: int = SEED):
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if seed == SEED else f"-s{seed}"
    pt_path = OUT_DIR / f"router-encoder{tag}.pt"
    onnx_path = OUT_DIR / f"router-encoder{tag}.onnx"
    int8_path = OUT_DIR / f"router-encoder-int8{tag}.onnx"
    tok_dir = OUT_DIR / ("minilm" if not tag else f"minilm{tag}")
    tok.save_pretrained(str(tok_dir))
    classes_path = OUT_DIR / f"router-encoder-classes{tag}.json"
    classes_path.write_text(json.dumps(list(classes), indent=2) + "\n",
                            encoding="utf-8")

    torch.save(net.state_dict(), pt_path)
    dummy = tok(["warm up"], return_tensors="pt", truncation=True,
                max_length=MAX_LEN, padding=True)
    torch.onnx.export(
        net, (dummy["input_ids"], dummy["attention_mask"],
              dummy.get("token_type_ids")),
        onnx_path, input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"], dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"}}, opset_version=17, dynamo=False)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    return pt_path, onnx_path, int8_path


class OnnxRouter:
    def __init__(self, path: Path, tokenizer, classes):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path),
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = tokenizer
        self.classes = list(classes)

    def _logits(self, texts):
        import numpy as np

        enc = self.tok(list(texts), truncation=True, max_length=MAX_LEN,
                       padding=True)
        ids = np.array(enc["input_ids"], dtype=np.int64)
        mask = np.array(enc["attention_mask"], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        return self.sess.run(None, feed)[0]

    def predict(self, texts):
        return [self.classes[i] for i in self._logits(texts).argmax(-1).tolist()]

    def predict_proba(self, texts):
        import numpy as np

        logits = self._logits(texts)
        e = np.exp(logits - logits.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)


def main() -> int:
    import argparse

    from mivia_small_json import bootstrap
    from mivia_small_json.model import MODEL_PATH
    from mivia_small_json.pipeline import classify, load_router
    from mivia_small_json.agent_step import register_agent_step_schema
    from mivia_small_json.diagnostic import register_diagnostic_schema
    from mivia_small_json.diff_comment import register_diff_and_review_schemas
    from mivia_small_json.tool_call import (register_no_tool_call_schema,
                                            register_tool_call_schema)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED,
                    help="training seed (both-seed rule: 42 and 43; "
                         "non-default seeds get -s<seed> filename tags)")
    args = ap.parse_args()

    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()
    register_agent_step_schema()

    net, tok, classes = train(seed=args.seed)
    pt_path, onnx_path, int8_path = export_onnx(net, tok, classes,
                                                seed=args.seed)
    onnx_router = OnnxRouter(int8_path, tok, classes)
    hashed = load_router(MODEL_PATH)

    def heldout(router) -> float:
        rows = _rows(ROOT / "data" / "heldout" / "test.jsonl")
        ok = sum(1 for r in rows
                 if (lambda o, m: m["valid"] and o.get("type") == r["schema"])(
                     *classify(r["raw"], router)))
        return 100 * ok / len(rows)

    ood = [
        ("nlbse->issue",
         ROOT / "data" / "incoming" / "external" / "nlbse" / "issues.jsonl",
         "issue"),
        ("cleancodereview->review_comment",
         ROOT / "data" / "incoming" / "external" / "cleancodereview" / "comments.jsonl",
         "review_comment"),
        ("bfcl->tool_call",
         ROOT / "data" / "incoming" / "external" / "bfcl" / "calls.jsonl",
         "tool_call"),
        ("no_tool_call",
         ROOT / "data" / "real" / "no_tool_call" / "dev.jsonl",
         "no_tool_call"),
    ]

    def ood_score(router, path: Path, expected: str, cap: int = 400) -> float:
        rows = _rows(path)[:cap]
        ok = sum(1 for r in rows
                 if classify(r.get("text") or r["raw"], router)[0].get("type")
                 == expected)
        return 100 * ok / len(rows) if rows else float("nan")

    texts = [r["raw"] for r in _rows(ROOT / "data" / "heldout" / "test.jsonl")][:150]

    def latency(router):
        times = []
        for t in texts:
            t0 = time.perf_counter()
            classify(t, router)
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        return times[len(times) // 2], times[min(len(times) - 1,
                                                 int(0.95 * len(times)))]

    h_lat, o_lat = latency(hashed), latency(onnx_router)
    lines = ["# Spike 3: fine-tuned int8 ONNX router vs hashed router", "",
             f"Fine-tuned {MODEL_ID} on the FULL mix ({EPOCHS} epochs)", "",
             "| Task | hashed | fine-tuned int8 ONNX |", "|---|---|---|",
             f"| held-out routed-and-valid % | {heldout(hashed):.1f} | "
             f"{heldout(onnx_router):.1f} |"]
    for name, path, expected in ood:
        lines.append(f"| OOD {name} % | {ood_score(hashed, path, expected):.1f} | "
                     f"{ood_score(onnx_router, path, expected):.1f} |")
    lines += [
        f"| latency p50 (ms) | {h_lat[0]:.3f} | {o_lat[0]:.3f} |",
        f"| latency p95 (ms) | {h_lat[1]:.3f} | {o_lat[1]:.3f} |",
        f"| artifact bytes | {MODEL_PATH.stat().st_size} | "
        f"{int8_path.stat().st_size + (OUT_DIR / 'minilm' / 'tokenizer.json').stat().st_size} |",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
