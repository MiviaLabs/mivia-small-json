#!/usr/bin/env python3
"""zero-shot-select r2: train the DEDICATED binary no-match head
(zero-shot-select-r2) and export it to int8 ONNX.

The abstention defect fix: a head trained ONLY on the no-match decision,
scored alongside the rank head at serve time (registered decision path in
the spec; the promotion-commit src item wires it). Corpus:

  r1/null-pairs.jsonl   weight 2.0 — the head's primary task: true-oos
                        texts vs null candidates (label 1), in-domain texts
                        vs null candidates (label 0), oos texts vs their
                        most-confusable real descriptions (label 0); the
                        cross-domain telco/insurance + clinc-oos supply is
                        inside this file.
  nli-converted.jsonl   DICT-shape rows only (weight 1.0) — general
                        "applies / does-not-apply" anchors so the null
                        decision generalizes to near-domain oos instead of
                        memorizing a phrasing. The verdict rows are binary-
                        surface rows, not the head's serve shape; excluded.

Same stamp, recipe and export contract as the rank head. Writes the
candidate UNTRACKED to models/select-nomatch-zsr2-int8.onnx.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_select_rank_zsr2 import (  # noqa: E402  (same arm + recipe)
    BATCH, BASE, EPOCHS, LR, MAX_LEN, NEAR_MISS_SOURCES, NEAR_MISS_WEIGHT,
    R1, R2, SEED, check_stamp,
)

OUT = ROOT / "models/select-nomatch-zsr2-int8.onnx"
REPORT = ROOT / ".scratch/zs-select-r2/train-report-nomatch.md"

# file -> (path, loss weight, keep rule)
FILES = {
    "r1/null-pairs.jsonl": (R1 / "null-pairs.jsonl", 2.0, None),
    "nli-converted.jsonl": (
        R2 / "nli-converted.jsonl", 1.0,
        lambda r: r["tool"] in ("yes", "no")),   # dict rows only
}


def main() -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from mivia_small_json.determinism import seed_everything
    from train_tool_match import export

    stamp = check_stamp()
    print(f"audit stamp OK (arm {stamp['arm']}, {len(stamp['extra_sha256'])} "
          "corpus shas bound)")

    rows: list[dict] = []
    for key, (path, _w, keep_excl) in FILES.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if keep_excl is not None and keep_excl(r):
                continue
            r["file"] = key
            rows.append(r)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    print(f"corpus: {len(rows)} pairs ({len(train)} train / {len(dev)} dev)")
    pos = sum(r["label"] for r in rows)
    print(f"label balance: {pos} pos / {len(rows) - pos} neg")

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device; training on CPU", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda"
             else ""))

    seed_everything(SEED)
    tok = AutoTokenizer.from_pretrained(BASE)
    net = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, ignore_mismatched_sizes=True).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)

    def weight_of(r: dict) -> float:
        w = FILES[r["file"]][1]
        if r["source"] in NEAR_MISS_SOURCES:
            w *= NEAR_MISS_WEIGHT
        return w

    net.train()
    for epoch in range(EPOCHS):
        torch.manual_seed(SEED + epoch)
        order = torch.randperm(len(train)).tolist()
        total = 0.0
        nb = 0
        for start in range(0, len(order), BATCH):
            chunk = [train[i] for i in order[start:start + BATCH]]
            enc = tok([r["request"] for r in chunk],
                      [r["tool"] for r in chunk], truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = torch.tensor([r["label"] for r in chunk]).to(device)
            weights = torch.tensor([weight_of(r) for r in chunk],
                                   dtype=torch.float).to(device)
            out = net(**enc)
            loss = F.cross_entropy(out.logits, labels, reduction="none")
            loss = (loss * weights).sum() / weights.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        print(f"epoch {epoch + 1}: loss {total / max(1, nb):.4f}", flush=True)
    net.eval()

    def score(request: str, tool: str) -> float:
        enc = tok([request], [tool], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            return float(torch.softmax(net(**enc).logits, -1)[0][1])

    lines = ["# zero-shot-select r2 dedicated no-match head training", "",
             f"corpus: {len(rows)} pairs ({len(train)} train / {len(dev)} "
             f"dev)", f"base {BASE} · epochs {EPOCHS} · batch {BATCH} · "
             f"lr {LR} · max_len {MAX_LEN} · seed {SEED} · device {device}",
             "weights: r1 null-pairs 2.0; nli dict rows 1.0", "",
             "Held-out (dev) pairs only:", "",
             "| threshold | positives kept % | negatives rejected % |",
             "|---|---|---|"]
    for thr in (0.3, 0.5, 0.6, 0.7, 0.9):
        kp = sum(1 for r in dev
                 if r["label"] == 1 and score(r["request"], r["tool"]) >= thr)
        kn = sum(1 for r in dev
                 if r["label"] == 0 and score(r["request"], r["tool"]) < thr)
        p = sum(1 for r in dev if r["label"] == 1)
        n = sum(1 for r in dev if r["label"] == 0)
        lines.append(f"| {thr} | {100 * kp / max(1, p):.1f} | "
                     f"{100 * kn / max(1, n):.1f} |")

    import random as _random

    agree_sample = _random.Random(0).sample(dev, min(200, len(dev)))
    fp32_scores = {i: score(r["request"], r["tool"])
                   for i, r in enumerate(agree_sample)}

    int8_path, fp32_path = export(net, tok, OUT, MAX_LEN)
    digest = hashlib.sha256(int8_path.read_bytes()).hexdigest()
    (int8_path.parent / (int8_path.name + ".sha256")).write_text(
        digest + "\n", encoding="utf-8")
    tok_path = int8_path.parent / (
        int8_path.name.replace("-int8.onnx", "-tokenizer.json"))
    if hasattr(tok, "backend_tokenizer"):
        tok.backend_tokenizer.save(str(tok_path))
    (tok_path.parent / (tok_path.name + ".sha256")).write_text(
        hashlib.sha256(tok_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8")

    from mivia_small_json.tool_match import load_single_matcher

    m8 = load_single_matcher(int8_path, tokenizer_path=tok_path)
    agree = 0
    n_check = 0
    if m8 is not None:
        for i, r in enumerate(agree_sample):
            a = fp32_scores[i]
            b = m8._score_raw(r["request"], r["tool"])
            agree += ((a >= 0.5) == (b >= 0.5)) and abs(a - b) <= 0.05
            n_check += 1
    lines += ["", f"int8/fp32 dev agreement (same side & |d|<=0.05): "
              f"{agree}/{n_check}",
              f"exported {int8_path.name} "
              f"({int8_path.stat().st_size} bytes) sha256 {digest[:16]}",
              f"fp32 parent {fp32_path.name}", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
