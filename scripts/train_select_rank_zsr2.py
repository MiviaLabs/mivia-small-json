#!/usr/bin/env python3
"""zero-shot-select r2: train the select/rank head candidate on the round's
mix and export it to int8 ONNX (zero-shot-select-r2).

Refuses to start without the arm's fail-closed audit stamp
(.scratch/audit-mix-stamp.json: arm == zero-shot-select-r2, passed true,
every bound sha matching the bytes on disk — r1 files AND the r2 files).

Registered recipe (UNCHANGED from r1): base cross-encoder/
ms-marco-MiniLM-L-6-v2, binary cross-entropy, epochs 3, batch 64, lr 3e-5,
max_len 256, seed 42, BFCL near-miss sources at the shipped 5.0 weight.
The TWO registered deviations (declared in the spec before any
measurement): (1) r1 listwise pairs at loss weight 2.0 (the r1 residual
analysis #1 increment: per-intent budget on banking/commerce); (2) the r1
synthetic file's `synthetic_polarity_*` rows are DROPPED at load (the r1
falsification names them the claim-restating teacher; 768 rows) — the r2
verdict file replaces that mass with the referee's exact construction.

Writes the candidate UNTRACKED to models/select-rank-zsr2-int8.onnx
(+ tokenizer + sha sidecars) — nothing in models/ is overwritten;
promotion is a separate deliberate step.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

R1 = ROOT / "data/real/zero-shot-select-r1"
R2 = ROOT / "data/real/zero-shot-select-r2"
STAMP = ROOT / ".scratch/audit-mix-stamp.json"
OUT = ROOT / "models/select-rank-zsr2-int8.onnx"
REPORT = ROOT / ".scratch/zs-select-r2/train-report-rank.md"

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EPOCHS, BATCH, LR, MAX_LEN, SEED = 3, 64, 3e-5, 256, 42

# file -> (path, loss weight, drop rule)
FILES = {
    "r1/listwise-pairs.jsonl": (R1 / "listwise-pairs.jsonl", 2.0, None),
    "r1/null-pairs.jsonl": (R1 / "null-pairs.jsonl", 1.0, None),
    "r1/synthetic-labelsets.jsonl": (
        R1 / "synthetic-labelsets.jsonl", 1.0,
        lambda r: r.get("source", "").startswith("synthetic_polarity_")),
    "r1/bfcl-train.jsonl": (R1 / "bfcl-train.jsonl", 1.0, None),
    "nli-converted.jsonl": (R2 / "nli-converted.jsonl", 1.0, None),
    "verdict-pairs.jsonl": (R2 / "verdict-pairs.jsonl", 1.0, None),
}

# the shipped near-miss weight set (train_tool_match.NEAR_MISS_SOURCES)
NEAR_MISS_SOURCES = frozenset({
    "bfcl_distractor_aligned", "bfcl_irrelevance_negatives",
    "synthetic_nearmiss_xlam", "synthetic_nearmiss_when2call"})
NEAR_MISS_WEIGHT = 5.0


def check_stamp() -> dict:
    if not STAMP.is_file():
        raise SystemExit("REFUSING: no audit stamp (run "
                         "scripts/audit_training_mix.py --arm "
                         "zero-shot-select-r2)")
    stamp = json.loads(STAMP.read_text())
    if stamp.get("arm") != "zero-shot-select-r2" or not stamp.get("passed"):
        raise SystemExit(f"REFUSING: stamp not valid for this arm: "
                         f"{stamp.get('arm')} passed={stamp.get('passed')}")
    for key, sha in stamp.get("extra_sha256", {}).items():
        if key.startswith("r1/"):
            p = R1 / key[len("r1/"):]
        else:
            p = R2 / key
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != sha:
            raise SystemExit(f"REFUSING: corpus file {key} does not match "
                             "the stamped sha (stale corpus or stale stamp)")
    return stamp


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
    dropped = 0
    for key, (path, _w, drop) in FILES.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if drop is not None and drop(r):
                dropped += 1
                continue
            r["file"] = key
            rows.append(r)
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    print(f"corpus: {len(rows)} pairs ({len(train)} train / {len(dev)} dev); "
          f"polarity rows dropped at load: {dropped}")
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

    # dev pair metrics + int8/fp32 agreement (internal export gate)
    def score(request: str, tool: str) -> float:
        enc = tok([request], [tool], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            return float(torch.softmax(net(**enc).logits, -1)[0][1])

    lines = ["# zero-shot-select r2 rank-head training", "",
             f"mix: {len(rows)} pairs ({len(train)} train / {len(dev)} dev);"
             f" polarity rows dropped at load: {dropped}",
             f"base {BASE} · epochs {EPOCHS} · batch {BATCH} · lr {LR} · "
             f"max_len {MAX_LEN} · seed {SEED} · device {device}",
             "weights: r1 listwise 2.0 (registered); bfcl near-miss 5.0 "
             "(shipped); everything else 1.0", "",
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
            same_side = (a >= 0.5) == (b >= 0.5)
            close = abs(a - b) <= 0.05
            agree += same_side and close
            n_check += 1
    agree_line = (f"int8/fp32 dev agreement (same side & |d|<=0.05): "
                  f"{agree}/{n_check}")
    lines += ["", agree_line,
              f"exported {int8_path.name} "
              f"({int8_path.stat().st_size} bytes) sha256 {digest[:16]}",
              f"fp32 parent {fp32_path.name}", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
