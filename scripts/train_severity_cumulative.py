#!/usr/bin/env python3
"""Train a cumulative-threshold ordinal severity head (ticket 12).

The shipped grade artifact scores a text against each ordered anchor with an
independent sigmoid. Those saturates near 0/1, and the residual error is
adjacent-tier boundary mass: exact accuracy is 100% on LogHub-style rows but
33.8% on business/SRE phrasing and 26.7% on adjacent-tier boundaries (the
n=210 panel, reports/training-report.md).

This trains a CORAL-style head instead: one latent severity score and K-1
ordered cutpoints, so adjacent boundaries share a single decision axis
(Cao et al., arXiv:1901.07884; Shi et al., arXiv:2111.08851). The head applies
to the *fixed* severity scale; arbitrary user anchors keep the anchor-alignment
path.

Candidate artifact only: writes under data/incoming/models/ and never
promotes. Pre-registered bar (see result-quality-improvement
ticket 12): novel_register exact >= 60% AND boundary >= 50% AND
template_disjoint >= 95% on the frozen panel.

GPU when present (owner policy), CPU is a stated fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.determinism import seed_everything  # noqa: E402

BASE = "sentence-transformers/all-MiniLM-L6-v2"
CORPUS = ROOT / "data" / "real" / "grade"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "severity-cumulative-int8.onnx"
PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
K = 4


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _severity(path: Path) -> list[dict]:
    return [r for r in _rows(path)
            if r.get("family") == "severity" and isinstance(r.get("gold"), int)]


def main() -> int:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--extra", type=Path, default=None,
                    help="extra severity rows to add to training (ticket 14's "
                         "targeted synthesis)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} (cuda available: {torch.cuda.is_available()})")

    train_rows = _severity(CORPUS / "train.jsonl")
    dev_rows = _severity(CORPUS / "dev.jsonl")
    if args.extra is not None:
        extra = _severity(args.extra)
        train_rows = train_rows + extra
        print(f"targeted rows added: {len(extra)}")
    print(f"severity rows: {len(train_rows)} train / {len(dev_rows)} dev")

    tok = AutoTokenizer.from_pretrained(BASE)

    class CoralSeverity(nn.Module):
        """Single latent + K-1 ordered cutpoints (CORAL)."""

        def __init__(self, base_name: str, k: int):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_name)
            hidden = self.encoder.config.hidden_size
            self.latent = nn.Linear(hidden, 1)
            self.cut0 = nn.Parameter(torch.zeros(1))
            self.deltas = nn.Parameter(torch.zeros(max(0, k - 2)))

        def cutpoints(self):
            if self.deltas.numel() == 0:
                return self.cut0
            return torch.cat([
                self.cut0,
                self.cut0 + torch.cumsum(F.softplus(self.deltas), dim=0),
            ])

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            latent = self.latent(pooled).squeeze(-1)
            return latent.unsqueeze(-1) - self.cutpoints().unsqueeze(0)

    seed_everything(args.seed)
    net = CoralSeverity(BASE, K).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def targets(golds):
        # 1{gold > k} for k = 0..K-2
        idx = torch.arange(K - 1, device=device).unsqueeze(0)
        return (torch.tensor(golds, device=device).unsqueeze(1) > idx).float()

    net.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train_rows)).tolist()
        total = 0.0
        batches = 0
        for start in range(0, len(train_rows), args.batch):
            chunk = [train_rows[i] for i in perm[start:start + args.batch]]
            enc = tok([r["text"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = net(**enc)
            loss = F.binary_cross_entropy_with_logits(
                logits, targets([r["gold"] for r in chunk]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        print(f"epoch {epoch + 1}: loss {total / max(1, batches):.4f}")

    # Held-out (corpus dev) exact accuracy before export.
    net.eval()
    from mivia_small_json.severity_head import cumulative_tier

    def exact(rows) -> float:
        correct = 0
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            enc = tok([r["text"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = net(**enc).cpu().tolist()
            correct += sum(cumulative_tier(row) == r["gold"]
                           for row, r in zip(logits, chunk))
        return 100.0 * correct / len(rows) if rows else 0.0

    print(f"corpus dev exact: {exact(dev_rows):.1f}%")

    from collections import Counter
    hist = Counter()
    for start in range(0, len(dev_rows), 64):
        chunk = dev_rows[start:start + 64]
        enc = tok([r["text"] for r in chunk], truncation=True,
                  max_length=args.max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = net(**enc).cpu().tolist()
        hist.update(cumulative_tier(row) for row in logits)
    print("dev pred histogram:", dict(sorted(hist.items())))
    print("dev gold histogram:",
          dict(sorted(Counter(r["gold"] for r in dev_rows).items())))
    print("cutpoints:", [round(x, 3) for x in
                         net.cutpoints().detach().cpu().tolist()])

    # Export fp32 + int8.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    net.to("cpu").eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = args.out.parent / args.out.name.replace("-int8", "")
    dummy = tok(["a log line"], return_tensors="pt", truncation=True,
                max_length=args.max_len, padding=True)
    torch.onnx.export(
        net,
        (dummy["input_ids"], dummy["attention_mask"]),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["cumulative_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "cumulative_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(args.out), weight_type=QuantType.QInt8)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    (args.out.parent / (args.out.name + ".sha256")).write_text(digest + "\n",
                                                              encoding="utf-8")
    print(f"exported {args.out.name} ({args.out.stat().st_size} bytes, "
          f"sha256 {digest[:16]})")

    # Measure on the frozen panel (stratum breakdown).
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    panel = _rows(PANEL)
    by: dict[str, list[dict]] = {}
    for row in panel:
        by.setdefault(row["stratum"], []).append(row)

    def panel_exact(rows) -> tuple[int, float]:
        correct = 0
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            enc = tok([r["text"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True)
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            logits = sess.run(["cumulative_logits"], {
                "input_ids": ids, "attention_mask": mask})[0]
            correct += sum(cumulative_tier(row.tolist()) == r["gold"]
                           for row, r in zip(logits, chunk))
        return len(rows), 100.0 * correct / len(rows) if rows else 0.0

    results = {}
    for stratum, rows in sorted(by.items()):
        n, acc = panel_exact(rows)
        results[stratum] = acc
        print(f"panel {stratum:18s} n={n:3d} exact {acc:5.1f}%")
    bar = {"novel_register": 60.0, "boundary": 50.0, "template_disjoint": 95.0}
    passed = all(results.get(k, 0.0) >= v for k, v in bar.items())
    print(f"bar {bar} -> {'CLEARED' if passed else 'FALSIFIED'}")
    # A falsified arm must fail the process, so a pipeline cannot read the run
    # as promotable.
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
