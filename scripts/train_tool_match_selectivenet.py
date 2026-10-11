#!/usr/bin/env python3
"""Fine-tune tool-match cross-encoder using SelectiveNet (ICML 2019).

SelectiveNet integrates a dedicated selection head g(x) in [0, 1] alongside
the classification head f(x) and an auxiliary head h(x).
Target coverage c (e.g. 0.95-0.98) forces the network to optimize selective
risk on the covered distribution while penalizing under-coverage.

The exported ONNX graph computes combined probability:
    P_select = softmax(f_logits)[:, 1] * g(x)
and outputs logit-equivalents so the shipped ToolMatcher runtime reads it
with zero contract changes.
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "tool-match-selectivenet-int8.onnx"
DEFAULT_REPORT = ROOT / "reports" / "tool-match-selectivenet-training.md"
NEAR_MISS_SOURCES = {"bfcl_distractor_aligned", "bfcl_irrelevance_negatives"}
NEAR_MISS_WEIGHT = 5.0


class SelectiveCrossEncoder(nn.Module):
    """MiniLM cross-encoder with SelectiveNet architecture (ICML 2019)."""

    def __init__(self, base_name: str = BASE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_name)
        dim = self.encoder.config.hidden_size  # 384 for MiniLM
        self.dropout = nn.Dropout(0.1)
        self.f_head = nn.Linear(dim, 2)  # Prediction head
        self.g_head = nn.Sequential(     # Selection head
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.h_head = nn.Linear(dim, 2)  # Auxiliary head

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        # Use mean pooling with attention mask
        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        cls_rep = sum_embeddings / sum_mask

        pooled = self.dropout(cls_rep)
        f_logits = self.f_head(pooled)
        g_score = self.g_head(pooled).squeeze(-1)  # in [0, 1]
        h_logits = self.h_head(pooled)

        if not self.training:
            # At inference: combined score = softmax(f_logits)[:, 1] * g_score
            p_f = F.softmax(f_logits, dim=-1)[:, 1]
            combined = torch.clamp(p_f * g_score, min=1e-6, max=1.0 - 1e-6)
            # Output pseudo-logits [log(1-p), log(p)]
            return torch.stack([torch.log(1.0 - combined), torch.log(combined)], dim=-1)

        return f_logits, g_score, h_logits


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage", type=float, default=0.96,
                        help="target coverage c in SelectiveNet")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="trade-off between selective loss and auxiliary loss")
    parser.add_argument("--lambda-cov", type=float, default=32.0,
                        help="penalty weight for coverage deficit")
    args = parser.parse_args()

    seed_everything(args.seed)
    rows = load_rows(args.pairs)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    print(f"corpus: {len(rows)} rows ({len(train)} train / {len(dev)} dev)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    tok = AutoTokenizer.from_pretrained(BASE)
    net = SelectiveCrossEncoder(BASE).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def weight_of(row: dict) -> float:
        return NEAR_MISS_WEIGHT if row["source"] in NEAR_MISS_SOURCES else 1.0

    net.train()
    for epoch in range(args.epochs):
        torch.manual_seed(args.seed + epoch)
        perm = torch.randperm(len(train)).tolist()
        total_loss = 0.0
        total_cov = 0.0

        for start in range(0, len(train), args.batch):
            chunk = [train[i] for i in perm[start:start + args.batch]]
            enc = tok([r["request"] for r in chunk],
                      [r["tool"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = torch.tensor([r["label"] for r in chunk]).to(device)
            weights = torch.tensor([weight_of(r) for r in chunk],
                                   dtype=torch.float, device=device)

            f_logits, g_score, h_logits = net(**enc)

            # Selective Empirical Risk over chosen samples
            loss_f = F.cross_entropy(f_logits, labels, reduction="none")
            denom = torch.clamp((weights * g_score).sum(), min=1e-8)
            r_f = (weights * loss_f * g_score).sum() / denom

            # Coverage constraint
            cov = g_score.mean()
            loss_cov = torch.clamp(args.coverage - cov, min=0.0) ** 2

            # Auxiliary loss to prevent backbone degradation
            loss_h = F.cross_entropy(h_logits, labels, reduction="none")
            r_h = (weights * loss_h).sum() / weights.sum()

            loss = args.alpha * (r_f + args.lambda_cov * loss_cov) + (1.0 - args.alpha) * r_h

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.detach())
            total_cov += float(cov.detach())

        batches = max(1, len(train) // args.batch)
        print(f"epoch {epoch + 1}: loss {total_loss / batches:.4f} · "
              f"avg coverage {total_cov / batches:.3f}")

    net.eval()
    print("exporting ONNX artifact...")
    int8_path, fp32_path = export(net, tok, args.out, args.max_len)
    digest = hashlib.sha256(int8_path.read_bytes()).hexdigest()
    (int8_path.parent / (int8_path.name + ".sha256")).write_text(digest + "\n", encoding="utf-8")
    print(f"exported {int8_path.name} ({int8_path.stat().st_size} bytes, sha256 {digest[:16]})")
    return 0


def export(net, tok, out: Path, max_len: int) -> tuple[Path, Path]:
    """Export fp32 + int8 ONNX beside ``out``."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    net.to("cpu").eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = out.parent / out.name.replace("-int8", "")
    dummy = tok(["a request"], ["a tool: does a thing"], return_tensors="pt",
                truncation=True, max_length=max_len, padding=True)
    torch.onnx.export(
        net, (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        fp32_path, input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"], dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    return out, fp32_path


if __name__ == "__main__":
    sys.exit(main())
