#!/usr/bin/env python3
"""Spike 2: FINE-TUNE the encoder on this task, then compare.

Spike 1 showed a *frozen* MiniLM encoder loses to the hashed router. This
fine-tunes MiniLM as a 9-way router on the same training mix, then measures
held-out routed-and-valid, the OOD sets, CPU latency, and artifact size
against the committed hashed router.

Writes reports/spike-finetune.md. Requires torch + transformers (spike-only
deps, see requirements-spike.txt). Not part of the shipped model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPORT = ROOT / "reports" / "spike-finetune.md"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 3
BATCH = 32
MAX_LEN = 128
MAX_TRAIN = 12_000
SEED = 42


class FinetunedRouter:
    """A fine-tuned MiniLM classifier shaped like the pipeline's router."""

    def __init__(self, model, tokenizer, classes):
        self.model = model
        self.tokenizer = tokenizer
        self.classes = list(classes)

    def _logits(self, texts):
        import torch

        enc = self.tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                             padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            return self.model(**enc).logits

    def predict(self, texts):
        return [self.classes[i] for i in self._logits(texts).argmax(-1).tolist()]

    def predict_proba(self, texts):
        import torch

        return torch.softmax(self._logits(texts), dim=-1).cpu().numpy()


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .split("\n") if line.strip()]


def _train():
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    from mivia_small_json import bootstrap, model

    bootstrap.register_example_schemas()
    X, y = model.training_mix()
    X, y = X[:MAX_TRAIN], y[:MAX_TRAIN]
    classes = sorted(set(y))
    index = {c: i for i, c in enumerate(classes)}

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    net = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(classes))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    net.to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-5)

    torch.manual_seed(SEED)
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
    net.eval()
    return net, tok, classes


def main() -> int:
    from mivia_small_json import bootstrap
    from mivia_small_json.model import MODEL_PATH
    from mivia_small_json.pipeline import classify, load_router
    from mivia_small_json.diagnostic import register_diagnostic_schema
    from mivia_small_json.diff_comment import register_diff_and_review_schemas
    from mivia_small_json.tool_call import (register_no_tool_call_schema,
                                            register_tool_call_schema)

    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()

    net, tok, classes = _train()
    ft = FinetunedRouter(net, tok, classes)
    hashed = load_router(MODEL_PATH)

    def heldout_score(router) -> float:
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
    ]

    def ood_score(router, path: Path, expected: str, cap: int = 400) -> float:
        rows = _rows(path)[:cap]
        ok = sum(1 for r in rows
                 if classify(r["text"], router)[0].get("type") == expected)
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

    import torch

    buf = ROOT / "data" / "incoming" / "models" / "finetuned.pt"
    torch.save(net.state_dict(), buf)
    results = {
        "heldout": (heldout_score(hashed), heldout_score(ft)),
        "ood": [(n, ood_score(hashed, p, e), ood_score(ft, p, e))
                for n, p, e in ood],
        "latency": (latency(hashed), latency(ft)),
        "size": (MODEL_PATH.stat().st_size, buf.stat().st_size),
    }
    lines = ["# Spike 2: fine-tuned encoder vs hashed router", "",
             f"Fine-tuned {MODEL_ID} ({EPOCHS} epochs, {MAX_TRAIN} rows)", "",
             "| Task | hashed router | fine-tuned encoder |", "|---|---|---|",
             f"| held-out routed-and-valid % | {results['heldout'][0]:.1f} | "
             f"{results['heldout'][1]:.1f} |"]
    for name, h, f in results["ood"]:
        lines.append(f"| OOD {name} % | {h:.1f} | {f:.1f} |")
    lines += [
        f"| latency p50 (ms) | {results['latency'][0][0]:.3f} | "
        f"{results['latency'][1][0]:.3f} |",
        f"| latency p95 (ms) | {results['latency'][0][1]:.3f} | "
        f"{results['latency'][1][1]:.3f} |",
        f"| artifact bytes | {results['size'][0]} | {results['size'][1]} |",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
