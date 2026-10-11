#!/usr/bin/env python3
"""zero-shot-select r4 (the completing run): train the DEDICATED binary
no-match head on the NEAR-DOMAIN oos supply — the r3-registered recipe,
FINALLY RUN (zero-shot-select-r4; the r3 trainer was
written but never executed after the round died at its Banking77 stop
gate).

Refuses to start without the arm's fail-closed audit stamp
(.scratch/audit-mix-stamp.json: arm == zero-shot-select-r4, passed true,
every bound sha matching the bytes on disk).

Registered recipe (declared in the r3 spec, carried verbatim):

  r1/null-pairs.jsonl        POINTWISE weight 2.0 (the r2 recipe verbatim)
  r3/near-oos-pairs.jsonl    GROUPS weight 2.0 (multi-positive softmax:
                             the near-oos null renderings are the gold set
                             against the lexical-hard confusable negative —
                             the oos shape the eval actually serves)
  r2/nli-converted.jsonl     dict-shape rows POINTWISE weight 1.0 (the r2
                             recipe verbatim; general apply/no-apply anchors)

REGISTERED ACCEPTANCE TARGET (declared BEFORE this training, in the r4
spec): null precision >= 90 AND recall >= 30 pooled over the registered
oos windows (in-domain s42-300 windows in the precision denominator),
measured through the registered ship path on the one-shot.

Same stamp, recipe constants and export contract as the rank head. Writes
the candidate UNTRACKED to data/incoming/models/zsr4-candidates/
select-nomatch-zsr4-int8.onnx.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
import sys as _sys  # noqa: E402

if sys_path not in _sys.path:
    _sys.path.insert(0, sys_path)
_sys.path.insert(0, str(ROOT / "scripts"))

from train_select_rank_zsr4 import (  # noqa: E402
    BASE, EPOCHS, LR, MAX_LEN, SEED, check_stamp,
)

R1 = ROOT / "data/real/zero-shot-select-r1"
R2 = ROOT / "data/real/zero-shot-select-r2"
R3 = ROOT / "data/real/zero-shot-select-r3"
OUT = ROOT / "data/incoming/models/zsr4-candidates/select-nomatch-zsr4-int8.onnx"
REPORT = ROOT / ".scratch/zs-select-r4/train-report-nomatch.md"

EPOCHS_N, BATCH = EPOCHS, 64
UNITS_PER_BATCH = 32
POINT_ROWS_PER_UNIT_BATCH = 6

_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).casefold()
    return " ".join(_PUNCT.split(t)).strip()


def main() -> int:
    import random as _random

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from mivia_small_json.determinism import seed_everything
    from train_tool_match import export

    stamp = check_stamp()
    print(f"audit stamp OK (arm {stamp['arm']}, {len(stamp['extra_sha256'])} "
          "corpus shas bound)")

    # pointwise: r1 null-pairs (2.0, r2 recipe) + r2 nli dict rows (1.0)
    point_rows: list[dict] = []
    for line in (R1 / "null-pairs.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["split"] == "train":
                point_rows.append({**r, "_w": 2.0, "file": "r1/null"})
    for line in (R2 / "nli-converted.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["split"] == "train" and r["tool"] not in ("yes", "no"):
                point_rows.append({**r, "_w": 1.0, "file": "r2/nli_dict"})
    # groups: r3 near-oos (2.0, multi-positive)
    by_key: dict[tuple, dict] = {}
    dev_rows: list[dict] = []
    for line in (R3 / "near-oos-pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] == "dev":
            dev_rows.append(r)
            continue
        k = ("r3/near-oos", norm(r["request"]))
        u = by_key.setdefault(k, {"request": r["request"], "cands": [],
                                  "weight": 2.0, "file": "r3/near-oos"})
        u["cands"].append((r["tool"], int(r["label"])))
    units = list(by_key.values())
    print(f"corpus: {len(units)} near-oos groups + {len(point_rows)} "
          f"pointwise rows train; {len(dev_rows)} dev pairs")
    tot = sum(len(u["cands"]) for u in units) + len(point_rows)
    pos = sum(ln for u in units for _t, ln in u["cands"]) + \
        sum(r["label"] for r in point_rows)
    print(f"label balance: {pos} pos / {tot - pos} neg "
          f"({100 * pos / max(1, tot):.2f}% pos)")

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

    def run_batch(u_chunk: list[dict], p_chunk: list[dict]) -> float:
        reqs, cands, meta = [], [], []
        idx = 0
        for ui, u in enumerate(u_chunk):
            s = idx
            for tool, _lab in u["cands"]:
                reqs.append(u["request"])
                cands.append(tool)
                idx += 1
            meta.append(("u", ui, s, idx, u_chunk))
        for pj, r in enumerate(p_chunk):
            reqs.append(r["request"])
            cands.append(r["tool"])
            meta.append(("p", pj, idx, idx + 1, p_chunk))
            idx += 1
        enc = tok(reqs, cands, truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = net(**enc)
        z1 = out.logits.float()[:, 1]   # the candidate-score logit
        losses, wsum = [], 0.0
        for kind, i, s, e, chunk in meta:
            if kind == "u":
                u = chunk[i]
                g = [j for j, (_t, lab) in enumerate(u["cands"]) if lab == 1]
                gp = torch.tensor(g, device=device)
                # listwise softmax ACROSS the candidate set (the r3
                # registered form: negatives enter through the denominator)
                losses.append(
                    (torch.logsumexp(z1[s:e], dim=0)
                     - torch.logsumexp(z1[s:e][gp], dim=0)) * u["weight"])
                wsum += u["weight"]
            else:
                r = chunk[i]
                y = torch.tensor([r["label"]], device=device)
                losses.append(F.cross_entropy(
                    out.logits[s].unsqueeze(0).float(), y) * r["_w"])
                wsum += r["_w"]
        loss = torch.stack(losses).sum() / max(1e-9, wsum)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.detach())

    net.train()
    for epoch in range(EPOCHS_N):
        rng = _random.Random(SEED + epoch)
        pu, pp = list(units), list(point_rows)
        rng.shuffle(pu)
        rng.shuffle(pp)
        pi = 0
        total, nb = 0.0, 0
        for start in range(0, len(pu), UNITS_PER_BATCH):
            u_chunk = pu[start:start + UNITS_PER_BATCH]
            p_chunk = pp[pi:pi + POINT_ROWS_PER_UNIT_BATCH]
            pi += len(p_chunk)
            if u_chunk:
                total += run_batch(u_chunk, p_chunk)
                nb += 1
        rest = pp[pi:]
        for start in range(0, len(rest), BATCH):
            total += run_batch([], rest[start:start + BATCH])
            nb += 1
        print(f"epoch {epoch + 1}: loss {total / max(1, nb):.4f} "
              f"({nb} steps)", flush=True)
    net.eval()

    def score(request: str, tool: str) -> float:
        enc = tok([request], [tool], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            return float(torch.softmax(net(**enc).logits, -1)[0][1])

    lines = ["# zero-shot-select r4 dedicated no-match head training "
             "(near-domain supply, the r3 recipe finally run)", "",
             f"corpus: {len(units)} near-oos groups + {len(point_rows)} "
             f"pointwise rows; {len(dev_rows)} dev pairs",
             f"base {BASE} · epochs {EPOCHS_N} · units/batch "
             f"{UNITS_PER_BATCH} · lr {LR} · max_len {MAX_LEN} · seed {SEED}"
             f" · device {device}",
             "weights: r1 null-pairs pointwise 2.0 (r2 recipe); r3 near-oos "
             "groups 2.0 (multi-positive); r2 nli dict pointwise 1.0", "",
             "Held-out (dev) pairs only:", "",
             "| threshold | positives kept % | negatives rejected % |",
             "|---|---|---|"]
    for thr in (0.3, 0.5, 0.6, 0.7, 0.9):
        kp = sum(1 for r in dev_rows
                 if r["label"] == 1 and score(r["request"], r["tool"]) >= thr)
        kn = sum(1 for r in dev_rows
                 if r["label"] == 0 and score(r["request"], r["tool"]) < thr)
        p = sum(1 for r in dev_rows if r["label"] == 1)
        n = sum(1 for r in dev_rows if r["label"] == 0)
        lines.append(f"| {thr} | {100 * kp / max(1, p):.1f} | "
                     f"{100 * kn / max(1, n):.1f} |")

    agree_sample = _random.Random(0).sample(dev_rows, min(200, len(dev_rows)))
    fp32_scores = {i: score(r["request"], r["tool"])
                   for i, r in enumerate(agree_sample)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
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
