#!/usr/bin/env python3
"""realworld-slices r1: post-commit register + slack boundary
(realworld-slices-r1).

Same base checkpoint (mm12-text-s42), same recipe/seed as the chatood4
lineage; the ONLY delta vs the promoted round is the corpus upgrade:
AUX v5 -> v6 (v5 bytes + P post-commit register + S slack chat, all
`text`) and EXTRA v11 -> v12 (v11 bytes + D wrapper-paste `log_event` +
C review_comment counterbalance). One combined data round, both
textures, one GPU run, one bar matrix (the spec's pre-registered bars;
never retuned). Requires THIS arm's audit stamp (arm name + passed +
the exact corpus shas the trainer will load). Candidate bytes never
touch models/. GPU only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE_CKPT = ROOT / "data/incoming/models/mm12-text-s42/model"
AUX = ROOT / "data/real/realworld-r1/ood-aux-train-v6-realworld.jsonl"
EXTRA = ROOT / "data/real/realworld-r1/extra-realworld-r1.jsonl"
STAMP = ROOT / ".scratch/audit-mix-stamp.json"
OUT = ROOT / "data/incoming/models/mm12-realworld-r1-s42"
STEM = "mm12-realworld-r1-s42"
RECIPE = dict(epochs=3, batch=32, lr=3e-5, warmup_frac=0.1, clip=1.0,
              max_len=512, seed=42, weight_decay=0.01)
CLASSES = ["agent_step", "commit", "diagnostic", "diff", "issue",
           "log_event", "no_tool_call", "pull_request", "review_comment",
           "text", "tool_call"]


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_stamp() -> dict:
    """Fail-closed: THIS arm's stamp must be passed AND must bind the
    exact aux/extra bytes this trainer loads (never another arm's stale
    pass, never drifted corpus bytes)."""
    if not STAMP.is_file():
        print("REFUSING: no audit stamp", file=sys.stderr)
        raise SystemExit(2)
    stamp = json.loads(STAMP.read_text())
    if not (stamp.get("passed") and stamp.get("arm") == "realworld-slices-r1"):
        print(f"REFUSING: stamp arm={stamp.get('arm')} "
              f"passed={stamp.get('passed')}", file=sys.stderr)
        raise SystemExit(2)
    bound = stamp.get("extra_sha256") or {}
    for path in (AUX, EXTRA):
        sha = sha256_of(path)
        # the gate binds repo-relative paths (its _rel convention); accept
        # either that or the bare filename (fix 2026-10-11, orchestrator)
        if bound.get(path.name) != sha and bound.get(
                str(path.relative_to(ROOT))) != sha:
            print(f"REFUSING: audit stamp does not bind {path.name} "
                  f"(stamp={str(bound.get(path.name))[:12]} "
                  f"actual={sha[:12]})", file=sys.stderr)
            raise SystemExit(2)
    return stamp


def relabeled_mix(seed: int) -> tuple[list, list]:
    """training_mix() with the assembled v6 aux + v12 extra."""
    import os

    from mivia_small_json import bootstrap, model

    bootstrap.register_example_schemas()
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
    stamp = check_stamp()
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
    fp32_path = OUT / f"{STEM}-fp32.onnx"
    int8_path = OUT / f"{STEM}-int8.onnx"
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

    (OUT / f"{STEM}-classes.json").write_text(json.dumps(CLASSES, indent=1))
    (OUT / f"{STEM}-tokenizer.json").write_text(
        (BASE_CKPT / "tokenizer.json").read_text(encoding="utf-8"))
    for p in (int8_path, OUT / f"{STEM}-classes.json",
              OUT / f"{STEM}-tokenizer.json"):
        (OUT / (p.name + ".sha256")).write_text(sha256_of(p) + "\n")
    # meta: every field computed from the actual run/corpus bytes — the
    # sed-remnant lesson (stale strings in recorded metas) does not apply
    aux_rows = sum(1 for line in AUX.read_text(encoding="utf-8",
                                            errors="replace").splitlines()
                   if line.strip())
    extra_rows = sum(1 for line in EXTRA.read_text(encoding="utf-8",
                                                errors="replace").splitlines()
                     if line.strip())
    meta = {
        "phase": "realworld-slices r1 (post-commit register + slack "
                 "boundary, audit-gated, one combined data round)",
        "spec": "realworld-slices-r1",
        "base_checkpoint": str(BASE_CKPT),
        "recipe": RECIPE,
        "n_train_rows": len(X),
        "aux_corpus": {"path": AUX.name, "sha256": sha256_of(AUX),
                       "rows": aux_rows},
        "extra_corpus": {"path": EXTRA.name, "sha256": sha256_of(EXTRA),
                         "rows": extra_rows},
        "audit_stamp": {"arm": stamp.get("arm"),
                        "increment_text_share": stamp.get("report", {})
                        .get("composition", {}).get("increment_text_share")},
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
    (OUT / f"{STEM}-meta.json").write_text(json.dumps(meta, indent=1))
    (OUT / f"{STEM}-meta.json.sha256").write_text(
        sha256_of(OUT / f"{STEM}-meta.json") + "\n")
    print("staged:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
