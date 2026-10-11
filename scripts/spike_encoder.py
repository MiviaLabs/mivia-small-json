#!/usr/bin/env python3
"""Spike: does a learned sentence encoder beat hashed n-grams for routing?

Two changes are measured against the committed hashed router:

1. **Encoder head (SetFit-style tier 1).** A frozen int8 ONNX MiniLM
   sentence encoder produces embeddings; the existing scikit-learn
   LogisticRegression head is retrained on those embeddings. No hand-coded
   features anywhere.
2. **Two-stage.** The same encoder, but the head only needs to be right on
   the relation-heavy classes; the hashed router stays the fast path.

Reports held-out routed-and-valid, the OOD sets, and p50/p95 latency for
encode+head vs the hashed router. Writes reports/spike-encoder.md.

Requires onnxruntime + tokenizers and the downloaded encoder under
data/incoming/models/minilm/ (gitignored). Not part of the shipped model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODEL_DIR = ROOT / "data" / "incoming" / "models" / "minilm"
REPORT = ROOT / "reports" / "spike-encoder.md"
MAX_TRAIN = 12_000          # the spike caps training rows to stay tractable
BATCH = 128
MAX_LEN = 128

SCHEMAS = ("issue", "log_event", "commit", "pull_request", "tool_call",
           "no_tool_call", "diagnostic", "diff", "review_comment")


class OnnxEncoder:
    """Frozen int8 MiniLM: mean-pooled embeddings over an ONNX session."""

    def __init__(self, model_path: Path, tokenizer_path: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.enable_truncation(max_length=MAX_LEN)
        self.tok.enable_padding(length=None)
        self.sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str]):
        import numpy as np

        out = []
        for i in range(0, len(texts), BATCH):
            encs = self.tok.encode_batch(texts[i:i + BATCH])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]          # (B, T, H)
            m = mask[..., None].astype("float32")
            out.append((hidden * m).sum(1) / m.sum(1).clip(min=1e-6))
        return np.vstack(out)


class EncoderRouter:
    """Encoder + LR head, shaped like the pipeline's router."""

    def __init__(self, encoder, head, classes):
        self.encoder = encoder
        self.head = head
        self.classes = list(classes)

    def predict(self, texts):
        return [str(c) for c in self.head.predict(self.encoder.encode(list(texts)))]

    def predict_proba(self, texts):
        return self.head.predict_proba(self.encoder.encode(list(texts)))


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .split("\n") if line.strip()]


def _train_rows() -> tuple[list[str], list[str]]:
    from mivia_small_json import bootstrap, model

    bootstrap.register_example_schemas()
    X, y = model.training_mix()
    return X[:MAX_TRAIN], y[:MAX_TRAIN]


def _score(router, rows: list[dict], schema_key: str = "schema") -> float:
    from mivia_small_json.pipeline import classify

    ok = 0
    for row in rows:
        obj, meta = classify(row["raw"], router)
        ok += bool(meta["valid"]) and obj.get("type") == row[schema_key]
    return 100 * ok / len(rows) if rows else 0.0


def _latency(router, texts: list[str], n: int = 200) -> tuple[float, float]:
    from mivia_small_json.pipeline import classify

    times = []
    for text in texts[:n]:
        t0 = time.perf_counter()
        classify(text, router)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], times[min(len(times) - 1, int(0.95 * len(times)))]


def main() -> int:
    from sklearn.linear_model import LogisticRegression

    from mivia_small_json import bootstrap
    from mivia_small_json.model import MODEL_PATH
    from mivia_small_json.pipeline import load_router
    from mivia_small_json.diagnostic import register_diagnostic_schema
    from mivia_small_json.diff_comment import register_diff_and_review_schemas
    from mivia_small_json.tool_call import (register_no_tool_call_schema,
                                            register_tool_call_schema)

    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()

    encoder = OnnxEncoder(MODEL_DIR / "model.onnx", MODEL_DIR / "tokenizer.json")
    X, y = _train_rows()
    print(f"embedding {len(X)} training rows ...")
    Xe = encoder.encode(X)
    head = LogisticRegression(max_iter=1000, C=4.0, random_state=42).fit(Xe, y)
    enc_router = EncoderRouter(encoder, head, head.classes_)
    hashed = load_router(MODEL_PATH)

    heldout = _rows(ROOT / "data" / "heldout" / "test.jsonl")
    ood = [
        ("nlbse->issue",
         ROOT / "data" / "incoming" / "external" / "nlbse" / "issues.jsonl",
         "issue", "text"),
        ("cleancodereview->review_comment",
         ROOT / "data" / "incoming" / "external" / "cleancodereview" / "comments.jsonl",
         "review_comment", "text"),
        ("bfcl->tool_call",
         ROOT / "data" / "incoming" / "external" / "bfcl" / "calls.jsonl",
         "tool_call", "text"),
    ]

    def ood_score(router, path: Path, expected: str, key: str, cap=400) -> float:
        rows = _rows(path)[:cap]
        if not rows:
            return float("nan")
        from mivia_small_json.pipeline import classify

        ok = sum(1 for r in rows
                 if classify(r[key], router)[0].get("type") == expected)
        return 100 * ok / len(rows)

    results = {}
    results["heldout"] = {"hashed": _score(hashed, heldout),
                          "encoder": _score(enc_router, heldout)}
    for name, path, expected, key in ood:
        results[name] = {"hashed": ood_score(hashed, path, expected, key),
                         "encoder": ood_score(enc_router, path, expected, key)}

    texts = [r["raw"] for r in heldout][:200]
    results["latency"] = {
        "hashed": _latency(hashed, texts),
        "encoder": _latency(enc_router, texts),
    }
    results["artifact_bytes"] = {
        "hashed": MODEL_PATH.stat().st_size,
        "encoder": (MODEL_DIR / "model.onnx").stat().st_size
        + (MODEL_DIR / "tokenizer.json").stat().st_size,
    }
    results["train_rows"] = len(X)

    lines = ["# Spike: encoder head vs hashed n-grams", "",
             f"Training rows embedded: {len(X)} (capped)", "",
             "| Task | hashed router | encoder head |", "|---|---|---|",
             f"| held-out routed-and-valid % | {results['heldout']['hashed']:.1f} | "
             f"{results['heldout']['encoder']:.1f} |"]
    for name, _p, _e, _k in ood:
        r = results[name]
        lines.append(f"| OOD {name} % | {r['hashed']:.1f} | {r['encoder']:.1f} |")
    lines += [
        f"| latency p50 (ms) | {results['latency']['hashed'][0]:.3f} | "
        f"{results['latency']['encoder'][0]:.3f} |",
        f"| latency p95 (ms) | {results['latency']['hashed'][1]:.3f} | "
        f"{results['latency']['encoder'][1]:.3f} |",
        f"| artifact bytes | {results['artifact_bytes']['hashed']} | "
        f"{results['artifact_bytes']['encoder']} |",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
