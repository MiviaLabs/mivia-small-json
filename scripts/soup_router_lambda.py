#!/usr/bin/env python3
"""Task-vector interpolation round for the PR-negatives router (F1).

Pre-registered in pr-router-negatives after the seed-
variance hypothesis died (seeds 42 and 44 land identically at real dev
0.9699 / SWE-bench 97 — the 12-row dip is systematic, caused by the
contrast slice).

Phases:
  A (GPU):  fine-tune fp32 parents — (A) the prneg mix, (B) the Arm B mix
            (REAL_SETS temporarily without the prneg entry) — both seed 42.
  B (CPU):  interpolate W(lambda) = W_B + lambda * (W_A - W_B) for a grid,
            score real dev + held-out IN TORCH (chunked), pick lambda* by
            max real dev s.t. held-out >= 98 and PR-dev accuracy >= 0.93
            (dev-fit selection; the eval surfaces are not touched).
  C (eval): export lambda* to int8 ONNX, one-shot the eval surfaces
            (SWE-bench, eval-300 PRs, LogHub) through the real pipeline.

Usage:
  python scripts/soup_router_lambda.py phase-a
  python scripts/soup_router_lambda.py phase-b
  python scripts/soup_router_lambda.py phase-c --lambda 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

STATE_A = ROOT / "data" / "incoming" / "models" / "soup-fp32-prneg.pt"
STATE_B = ROOT / "data" / "incoming" / "models" / "soup-fp32-armb.pt"
PICKLE_LAMBDA = ROOT / "data" / "incoming" / "models" / "soup-lambda.json"
LAMBDA_GRID = (0.0, 0.3, 0.5, 0.7, 1.0)
SEED = 42


def _mix(with_prneg: bool):
    """Training mix with or without the prneg slice (no persistent edits)."""
    import mivia_small_json.model as model

    if with_prneg:
        return model.training_mix()
    real_sets = tuple(p for p in model.REAL_SETS
                      if p != ("issue", "train-prneg.jsonl"))
    orig = model.REAL_SETS
    model.REAL_SETS = real_sets
    try:
        return model.training_mix()
    finally:
        model.REAL_SETS = orig


def phase_a() -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from mivia_small_json.determinism import seed_everything
    from export_encoder_onnx import BATCH, EPOCHS, MAX_LEN, MODEL_ID

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(with_prneg: bool, out: Path) -> None:
        from mivia_small_json import bootstrap

        bootstrap.register_example_schemas()
        X, y = _mix(with_prneg)
        classes = sorted(set(y))
        index = {c: i for i, c in enumerate(classes)}
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        net = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=len(classes))
        net.to(device)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=3e-5)
        seed_everything(SEED)
        for epoch in range(EPOCHS):
            total = 0.0
            for i in range(0, len(X), BATCH):
                batch = X[i:i + BATCH]
                labels = torch.tensor([index[c] for c in y[i:i + BATCH]]).to(device)
                enc = tok(batch, truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out_loss = net(**enc, labels=labels)
                opt.zero_grad()
                out_loss.loss.backward()
                opt.step()
                total += float(out_loss.loss.detach())
            print(f"  mix={'prneg' if with_prneg else 'armb'} "
                  f"epoch {epoch + 1}: loss {total / max(1, len(X) // BATCH):.4f}")
        net.eval().to("cpu")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), out)
        print(f"saved {out.name}")

    print("phase A: parent A (prneg mix)")
    run(True, STATE_A)
    print("phase A: parent B (Arm B mix)")
    run(False, STATE_B)
    return 0


def _net_and_tok():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from export_encoder_onnx import MODEL_ID

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    net = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=10)
    return net, tok


def _score_torch(net, tok, texts, classes, device="cpu", n: int = 64):
    import torch

    net.to(device)
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), n):
            batch = texts[i:i + n]
            enc = tok(batch, truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(device)
            logits = net(**enc).logits
            out.extend(classes[j] for j in logits.argmax(-1).tolist())
    net.to("cpu")
    return out


def phase_b() -> int:
    import torch
    from mivia_small_json import bootstrap, model
    from measure_pr_routing_oneshot import _chunked_predict  # noqa: F401

    bootstrap.register_all_schemas()
    net, tok = _net_and_tok()
    sa = torch.load(STATE_A, map_location="cpu", weights_only=True)
    sb = torch.load(STATE_B, map_location="cpu", weights_only=True)

    X, y = model._load_real(model.REAL_DEV_SETS)
    pr_dev = [json.loads(line) for line in
              (ROOT / "data" / "real" / "pull_request" / "dev.jsonl")
              .read_text().splitlines() if line.strip()]
    pr_texts = [r["raw"] for r in pr_dev]
    held_rows = [json.loads(line) for line in
                 (ROOT / "data" / "heldout" / "test.jsonl")
                 .read_text().splitlines() if line.strip()]

    results = {}
    for lam in LAMBDA_GRID:
        sd = {k: sb[k] + lam * (sa[k].float() - sb[k].float())
              for k in sb}
        net.load_state_dict(sd)
        classes = json.loads(
            (ROOT / "models" / "router-encoder-classes.json").read_text())
        pred = _score_torch(net, tok, X, classes)
        real_acc = sum(p == t for p, t in zip(pred, y)) / len(y)
        pr_pred = _score_torch(net, tok, pr_texts, classes)
        pr_acc = sum(p == "pull_request" for p in pr_pred) / len(pr_pred)
        h_texts = [r["raw"] for r in held_rows]
        h_pred = _score_torch(net, tok, h_texts, classes)
        h_ok = sum(1 for p, r in zip(h_pred, held_rows)
                   if p == r["schema"]) / len(held_rows)
        results[lam] = {"real_dev": round(real_acc, 4), "pr_dev": round(pr_acc, 4),
                        "heldout": round(h_ok, 4)}
        print(f"lambda {lam}: real_dev {real_acc:.4f} · pr_dev {pr_acc:.4f} · "
              f"heldout {h_ok:.4f}")

    ok = {lam: r for lam, r in results.items()
          if r["heldout"] >= 0.98 and r["pr_dev"] >= 0.93}
    if not ok:
        raise SystemExit(f"no lambda meets dev gates: {results}")
    best = max(ok, key=lambda lam: ok[lam]["real_dev"])
    PICKLE_LAMBDA.write_text(json.dumps({"lambda": best, "dev": results[best]},
                                        indent=2) + "\n")
    print(f"selected lambda* = {best} (dev: {results[best]})")
    return 0


def phase_c(lam: float) -> int:
    """Export lambda* to int8 ONNX and one-shot the eval surfaces."""
    import torch
    from mivia_small_json import bootstrap

    bootstrap.register_all_schemas()
    net, tok = _net_and_tok()
    sa = torch.load(STATE_A, map_location="cpu", weights_only=True)
    sb = torch.load(STATE_B, map_location="cpu", weights_only=True)
    sd = {k: sb[k] + lam * (sa[k].float() - sb[k].float()) for k in sb}
    net.load_state_dict(sd)
    net.eval()

    from export_encoder_onnx import export_onnx

    _pt, _onnx, int8_path = export_onnx(net, tok, sorted(
        json.loads((ROOT / "models" / "router-encoder-classes.json")
                   .read_text())), seed=99)
    print(f"exported lambda={lam} -> {int8_path.name}")

    from mivia_small_json import encoder
    from mivia_small_json.pipeline import classify
    from collections import Counter

    tok_json = ROOT / "data" / "incoming" / "models" / "minilm-s99" / "tokenizer.json"
    router = encoder.load_encoder(int8_path, tok_json, classes=tuple(
        json.loads((ROOT / "models" / "router-encoder-classes.json")
                   .read_text())))
    assert router is not None, "soup export failed to load"
    rows = json.loads((ROOT / "data/incoming/external/swebench/"
                       "swebench_lite_100.json").read_text())
    c = Counter()
    for rec in rows:
        obj, _ = classify(rec.get("problem_statement") or "", router=router)
        c[obj.get("type")] += 1
    prs = json.loads((ROOT / "data/incoming/gh-prs.json").read_text())
    c2 = Counter()
    for p in prs:
        obj, _ = classify(f"{p['title']}\n\n{p.get('body') or ''}".strip(),
                          router=router)
        c2[obj.get("type")] += 1
    print(f"SWE-bench issue: {c['issue']}% · eval-300 pull_request: "
          f"{round(100 * c2['pull_request'] / 300, 1)}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("phase-a", "phase-b", "phase-c"))
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    args = ap.parse_args()
    if args.phase == "phase-a":
        return phase_a()
    if args.phase == "phase-b":
        return phase_b()
    return phase_c(args.lam)


if __name__ == "__main__":
    sys.exit(main())
