#!/usr/bin/env python3
"""chat-ood ROUND 4: greeting-led register completion (chat-ood, ROUND 4).

Same base checkpoint (mm12-text-s42), same recipe/seed, AUX v5 UNCHANGED
(sha-pinned — the chat floors live there); the ONLY delta vs the promoted round-3
chatood3 run is the EXTRA corpus upgrade v10 -> v11: +500 greeting-led
informal tickets (de 150 / fr 150 / pl 200, half greeting-led), blind
double-adjudicated, agreement 1.0. Pre-registered bars: RECOVERY
(tickets-non-en >= 0.98, tickets-de >= 0.99, min_retention >= 0.98) +
HOLD (every round-3 bar, chatood3 promoted). One-shot: run once, record the
verdict either way. GPU only (flock /tmp/mivia_gpu.lock). Candidate
bytes never touch models/.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BASE_CKPT = ROOT / "data/incoming/models/mm12-text-s42/model"
AUX = ROOT / "data/real/ood-aux/ood-aux-train-v5-chat.jsonl"
AUX_SHA = "5b9968c053cede26a60451580c7a16a549509276e04f09bc16b899e7a66cebed"
EXTRA = ROOT / "data/real/multilang-tickets/multilang-tickets-v11.jsonl"
EXTRA_SHA = "3f39693efdd67f90d9eb17683b881835127702b66a38a6d70166103c9310e30c"
OUT = ROOT / "data/incoming/models/mm12-chatood4-s42"
RECIPE = dict(epochs=3, batch=32, lr=3e-5, warmup_frac=0.1, clip=1.0,
              max_len=512, seed=42, weight_decay=0.01)
CLASSES = ["agent_step", "commit", "diagnostic", "diff", "issue",
           "log_event", "no_tool_call", "pull_request", "review_comment",
           "text", "tool_call"]


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def relabeled_mix(seed: int) -> tuple[list, list]:
    """training_mix() with aux v5 + the round-4 tickets extra (v11)."""
    import os

    from mivia_small_json import bootstrap, model

    bootstrap.register_example_schemas()
    if sha256_of(AUX) != AUX_SHA:
        print(f"aux corpus sha mismatch: {AUX}", file=sys.stderr)
        raise SystemExit(2)
    if sha256_of(EXTRA) != EXTRA_SHA:
        print(f"extra corpus sha mismatch: {EXTRA}", file=sys.stderr)
        raise SystemExit(2)
    stamp = ROOT / ".scratch/audit-mix-stamp.json"
    if not stamp.is_file() or not json.loads(stamp.read_text()).get("passed"):
        print("REFUSING: data-audit gate has not passed", file=sys.stderr)
        raise SystemExit(2)
    stamped_extra = json.loads(stamp.read_text()).get("extra_sha256")
    if stamped_extra != EXTRA_SHA:
        print(f"audit stamp extra sha {str(stamped_extra)[:12]} does not "
              "cover this corpus", file=sys.stderr)
        raise SystemExit(2)
    os.environ["MIVIA_OOD_AUX"] = str(AUX)
    os.environ["MIVIA_EXTRA_INDOMAIN"] = str(EXTRA)
    X, y = model.training_mix(seed=seed)
    os.environ.pop("MIVIA_OOD_AUX", None)
    os.environ.pop("MIVIA_EXTRA_INDOMAIN", None)
    print(f"mix rows={len(X)}")
    return X, y


def main() -> int:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from mivia_small_json import determinism

    if not BASE_CKPT.is_dir():
        print(f"base checkpoint missing: {BASE_CKPT}", file=sys.stderr)
        return 2
    determinism.seed_everything(RECIPE["seed"])

    net = AutoModelForSequenceClassification.from_pretrained(
        BASE_CKPT, num_labels=len(CLASSES), attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(BASE_CKPT)

    X, y = relabeled_mix(RECIPE["seed"])
    index = {c: i for i, c in enumerate(CLASSES)}
    unknown = sorted({t for t in y if t not in index})
    if unknown:
        print(f"mix labels outside the class map: {unknown}", file=sys.stderr)
        return 2
    y_ids = np.array([index[t] for t in y], dtype=np.int64)

    device = "cuda"
    net.to(device)
    enc = tok(X, truncation=True, max_length=RECIPE["max_len"],
              padding="max_length", return_tensors="pt")
    labels = torch.from_numpy(y_ids)
    steps_per_epoch = (len(X) + RECIPE["batch"] - 1) // RECIPE["batch"]
    total = steps_per_epoch * RECIPE["epochs"]
    opt = torch.optim.AdamW(net.parameters(), lr=RECIPE["lr"],
                            weight_decay=RECIPE["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=RECIPE["lr"], total_steps=total,
        pct_start=RECIPE["warmup_frac"], anneal_strategy="linear")
    gen = torch.Generator()
    gen.manual_seed(RECIPE["seed"])

    net.train()
    epochs_meta = []
    t_all = time.perf_counter()
    for epoch in range(RECIPE["epochs"]):
        perm = torch.randperm(len(X), generator=gen).tolist()
        losses = []
        nan = 0
        t0 = time.perf_counter()
        for s in range(0, len(perm), RECIPE["batch"]):
            idx = perm[s:s + RECIPE["batch"]]
            batch = {k: v[idx].to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(**batch, labels=labels[idx].to(device))
            loss = out.loss
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

    # --- held-out sanity in torch (measurement only, never training) ------
    net.eval()
    heldout_rows = [json.loads(line) for line in
                    (ROOT / "data/heldout/test.jsonl").read_text().splitlines()
                    if line.strip()]
    ok = 0
    with torch.no_grad():
        for i in range(0, len(heldout_rows), 32):
            chunk = heldout_rows[i:i + 32]
            he = tok([r["raw"] for r in chunk], truncation=True,
                     max_length=RECIPE["max_len"], padding="max_length",
                     return_tensors="pt")
            logits = net(input_ids=he["input_ids"].to(device),
                         attention_mask=he["attention_mask"].to(device)
                         ).logits
            preds = logits.argmax(-1).tolist()
            ok += sum(1 for r, p in zip(chunk, preds)
                      if CLASSES[p] == r["schema"])
    heldout_acc = 100 * ok / len(heldout_rows)
    print(f"held-out (torch, fp32): {heldout_acc:.2f}%")

    # --- export ------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model").mkdir(exist_ok=True)
    net.save_pretrained(OUT / "model")
    tok.save_pretrained(OUT / "model")
    net.to("cpu").eval()  # export + quantization are CPU-side
    stem = "mm12-chatood4-s42"
    fp32_path = OUT / f"{stem}-fp32.onnx"
    int8_path = OUT / f"{stem}-int8.onnx"
    dummy = tok(["a request"], return_tensors="pt",
                truncation=True, max_length=RECIPE["max_len"],
                padding="max_length")
    torch.onnx.export(
        net, (dummy["input_ids"], dummy["attention_mask"]), fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(int8_path),
                     weight_type=QuantType.QInt8)

    # agreement gate: int8 vs fp32 on a seed-42 sample of the mix
    import onnxruntime as ort
    sample = np.random.RandomState(RECIPE["seed"]).choice(
        len(X), size=1000, replace=False)
    s_texts = [X[i] for i in sample]
    se = tok(s_texts, truncation=True, max_length=RECIPE["max_len"],
             padding="max_length", return_tensors="np")
    feed = {"input_ids": se["input_ids"].astype(np.int64),
            "attention_mask": se["attention_mask"].astype(np.int64)}
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.intra_op_num_threads = 2
    lf = ort.InferenceSession(str(fp32_path), so,
                              providers=["CPUExecutionProvider"]
                              ).run(None, feed)[0]
    li = ort.InferenceSession(str(int8_path), so,
                              providers=["CPUExecutionProvider"]
                              ).run(None, feed)[0]
    pear = float(np.corrcoef(lf.ravel(), li.ravel())[0, 1])
    agree = float((lf.argmax(-1) == li.argmax(-1)).mean())
    print(f"agreement: pearson={pear:.6f} argmax={agree:.4f}")
    if pear < 0.99 or agree < 0.995:
        print("int8/fp32 agreement gate FAILED", file=sys.stderr)
        return 2

    (OUT / f"{stem}-classes.json").write_text(json.dumps(CLASSES, indent=1))
    (OUT / f"{stem}-tokenizer.json").write_text(
        (BASE_CKPT / "tokenizer.json").read_text(encoding="utf-8"))
    for p in (int8_path, OUT / f"{stem}-classes.json",
              OUT / f"{stem}-tokenizer.json"):
        (OUT / (p.name + ".sha256")).write_text(sha256_of(p) + "\n")
    meta = {
        "phase": "chat-ood round 4 (greeting-led register completion, audit-gated)",
        "base_checkpoint": str(BASE_CKPT),
        "recipe": RECIPE,
        "n_train_rows": len(X),
        "aux_corpus": {"path": str(AUX), "sha256": AUX_SHA, "rows": 17173},
        "extra_corpus": {"path": str(EXTRA), "sha256": EXTRA_SHA,
                         "rows": 5666},
        "classes": CLASSES,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "nan_steps": sum(e["nan_steps"] for e in epochs_meta),
        "epochs": epochs_meta,
        "train_seconds": round(time.perf_counter() - t_all, 1),
        "heldout_torch_accuracy": heldout_acc,
        "int8_fp32_agreement": {"pearson": pear, "argmax": agree,
                                "n": 1000},
    }
    (OUT / f"{stem}-meta.json").write_text(json.dumps(meta, indent=1))
    (OUT / f"{stem}-meta.json.sha256").write_text(
        sha256_of(OUT / f"{stem}-meta.json") + "\n")
    print("staged:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
