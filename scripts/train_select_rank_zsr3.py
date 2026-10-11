#!/usr/bin/env python3
"""zero-shot-select r3: train the select/rank head on the LISTWISE softmax
objective (zero-shot-select-r3) and export it to int8
ONNX.

Refuses to start without the arm's fail-closed audit stamp
(.scratch/audit-mix-stamp.json: arm == zero-shot-select-r3, passed true,
every bound sha matching the bytes on disk — r1 files, r2 files AND the r3
file + adjudication evidence).

Registered objective (declared in the spec BEFORE training): groups keyed
by strong-normalized request; group loss = -log( sum of softmax probs over
gold members ) (standard CE when the gold is single, the multi-positive
form for oos null groups); pointwise BCE for the two declared exceptions
(BFCL rows at 1.0 with near-miss x5.0, NLI dict rows at 0.5). Weights are
the de-dilution lever: listwise groups 4.0, null oos groups 1.0, S3 groups
1.0 (synthetic_polarity_* rows dropped at load), NLI dict pointwise 0.5,
NLI verdict {yes,no} groups 0.5, verdict-file groups 0.5. Registered join:
r1 in-domain null negatives (label 0 vs the null candidate) ride the
same-key listwise group as an extra candidate the softmax pushes down;
unmatched ones stay in null oos groups.

UNCHANGED recipe constants: base cross-encoder/ms-marco-MiniLM-L-6-v2,
epochs 3, lr 3e-5, max_len 256, seed 42, AdamW; the ONE declared deviation
is the batch shape: 32-UNIT mixed batches (~6 pointwise rows each) plus
dedicated 64-row pointwise batches for the remainder — every row is seen
every epoch.

Writes the candidate UNTRACKED to models/select-rank-zsr3-int8.onnx
(+ tokenizer + sha sidecars) — nothing in models/ is overwritten.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
import sys  # noqa: E402

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
sys.path.insert(0, str(ROOT / "scripts"))

R1 = ROOT / "data/real/zero-shot-select-r1"
R2 = ROOT / "data/real/zero-shot-select-r2"
R3 = ROOT / "data/real/zero-shot-select-r3"
STAMP = ROOT / ".scratch/audit-mix-stamp.json"
OUT = ROOT / "models/select-rank-zsr3-int8.onnx"
REPORT = ROOT / ".scratch/zs-select-r3/train-report-rank.md"

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EPOCHS, LR, MAX_LEN, SEED = 3, 3e-5, 256, 42
UNITS_PER_BATCH = 32
POINT_ROWS_PER_UNIT_BATCH = 6    # the ~85/15 unit mix inside phase-A batches

_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).casefold()
    return " ".join(_PUNCT.split(t)).strip()


# file tag -> (path, unit weight, drop rule)
FILES = {
    "r1/listwise-pairs.jsonl": (R1 / "listwise-pairs.jsonl", 4.0, None),
    "r1/null-pairs.jsonl": (R1 / "null-pairs.jsonl", 1.0, None),
    "r1/synthetic-labelsets.jsonl": (
        R1 / "synthetic-labelsets.jsonl", 1.0,
        lambda r: r.get("source", "").startswith("synthetic_polarity_")),
    "r1/bfcl-train.jsonl": (R1 / "bfcl-train.jsonl", 1.0, None),
    "r2/nli-converted.jsonl": (R2 / "nli-converted.jsonl", 0.5, None),
    "r2/verdict-pairs.jsonl": (R2 / "verdict-pairs.jsonl", 0.5, None),
}

NEAR_MISS_SOURCES = frozenset({
    "bfcl_distractor_aligned", "bfcl_irrelevance_negatives",
    "synthetic_nearmiss_xlam", "synthetic_nearmiss_when2call"})
NEAR_MISS_WEIGHT = 5.0


def check_stamp() -> dict:
    if not STAMP.is_file():
        raise SystemExit("REFUSING: no audit stamp (run "
                         "scripts/audit_training_mix.py --arm "
                         "zero-shot-select-r3)")
    stamp = json.loads(STAMP.read_text())
    if stamp.get("arm") != "zero-shot-select-r3" or not stamp.get("passed"):
        raise SystemExit(f"REFUSING: stamp not valid for this arm: "
                         f"{stamp.get('arm')} passed={stamp.get('passed')}")
    for key, sha in stamp.get("extra_sha256", {}).items():
        if key.startswith("r1/"):
            p = R1 / key[len("r1/"):]
        elif key.startswith("r2/"):
            p = R2 / key[len("r2/"):]
        else:
            p = R3 / key
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != sha:
            raise SystemExit(f"REFUSING: corpus file {key} does not match "
                             "the stamped sha (stale corpus or stale stamp)")
    return stamp


def load_rows_full():
    """Units (groups) + pointwise rows + dev pairs (see module docstring)."""
    raw: dict[str, list[dict]] = defaultdict(list)
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
            raw[key].append(r)

    dev_rows: list[dict] = [r for rows in raw.values() for r in rows
                            if r["split"] == "dev"]

    point_rows: list[dict] = []
    for r in raw["r1/bfcl-train.jsonl"]:
        if r["split"] != "train":
            continue
        w = FILES["r1/bfcl-train.jsonl"][1]
        if r["source"] in NEAR_MISS_SOURCES:
            w *= NEAR_MISS_WEIGHT
        point_rows.append({**r, "_w": w})
    nli_w = FILES["r2/nli-converted.jsonl"][1]
    for r in raw["r2/nli-converted.jsonl"]:
        if r["split"] == "train":     # BOTH shapes: registered pointwise
            point_rows.append({**r, "_w": nli_w})

    by_key: dict[tuple, dict] = {}
    for key, weight in (("r1/listwise-pairs.jsonl", 4.0),
                        ("r1/null-pairs.jsonl", 1.0),
                        ("r1/synthetic-labelsets.jsonl", 1.0),
                        ("r2/verdict-pairs.jsonl", 0.5)):
        for r in raw[key]:
            if r["split"] != "train":
                continue
            if key == "r1/null-pairs.jsonl" and \
                    r["source"] == "null_neg_indomain":
                continue          # owned by the registered join below
            k = (key, norm(r["request"]))
            u = by_key.setdefault(k, {"request": r["request"], "cands": [],
                                      "weight": weight, "file": key})
            u["cands"].append((r["tool"], int(r["label"])))
    lw_keys = {norm(r["request"]) for r in raw["r1/listwise-pairs.jsonl"]
               if r["split"] == "train"}
    joined = unmatched = 0
    for r in raw["r1/null-pairs.jsonl"]:
        if r["split"] != "train" or r["source"] != "null_neg_indomain":
            continue
        k = norm(r["request"])
        if k in lw_keys:
            by_key[("r1/listwise-pairs.jsonl", k)]["cands"].append(
                (r["tool"], int(r["label"])))
            joined += 1
        else:
            u = by_key.setdefault(
                ("r1/null-pairs.jsonl", k),
                {"request": r["request"], "cands": [], "weight": 1.0,
                 "file": "r1/null-pairs.jsonl"})
            u["cands"].append((r["tool"], int(r["label"])))
            unmatched += 1
    units = [u for u in by_key.values() if u["cands"]]
    goldless = [u for u in units if not any(ln for _t, ln in u["cands"])]
    if goldless:
        raise SystemExit(f"GOLDLESS GROUPS: {len(goldless)} groups with no "
                         "positive member (group construction bug)")
    stats = {"dropped_polarity": dropped, "joined_null_negs": joined,
             "unmatched_null_negs": unmatched}
    return units, point_rows, dev_rows, stats


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

    units, point_rows, dev_rows, stats = load_rows_full()
    print(f"corpus: {len(units)} groups + {len(point_rows)} pointwise rows "
          f"train; {len(dev_rows)} dev pairs; {stats}")
    tot = sum(len(u["cands"]) for u in units) + len(point_rows)
    pos = sum(line for u in units for _t, line in u["cands"]) + \
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

    net.train()
    for epoch in range(EPOCHS):
        rng = _random.Random(SEED + epoch)
        pu = list(units)
        pp = list(point_rows)
        rng.shuffle(pu)
        rng.shuffle(pp)
        pi = 0
        total, nb = 0.0, 0

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
                    g = [j for j, (_t, lab) in enumerate(u["cands"])
                         if lab == 1]
                    gp = torch.tensor(g, device=device)
                    # REGISTERED listwise softmax ACROSS the candidate set:
                    # -log( sum_gold e^{z1} / sum_all e^{z1} ) — the
                    # negatives enter through the denominator (an earlier
                    # draft scored only the gold pair per-candidate, which
                    # never trained the set contrast and collapsed the
                    # surface; caught at the mid-round gate, fixed BEFORE
                    # any bar was read)
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

        # phase A: 32-unit mixed batches with ~6 pointwise rows each
        for start in range(0, len(pu), UNITS_PER_BATCH):
            u_chunk = pu[start:start + UNITS_PER_BATCH]
            p_chunk = pp[pi:pi + POINT_ROWS_PER_UNIT_BATCH]
            pi += len(p_chunk)
            if u_chunk:
                total += run_batch(u_chunk, p_chunk)
                nb += 1
        # phase B: dedicated 64-row pointwise batches for the remainder
        # (every pointwise row is seen every epoch — the BFCL share holds)
        rest = pp[pi:]
        for start in range(0, len(rest), 64):
            total += run_batch([], rest[start:start + 64])
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

    lines = ["# zero-shot-select r3 rank-head training (listwise softmax)", "",
             f"mix: {len(units)} groups + {len(point_rows)} pointwise rows "
             f"train; {len(dev_rows)} dev pairs; {stats}",
             f"base {BASE} · epochs {EPOCHS} · units/batch {UNITS_PER_BATCH} "
             f"· lr {LR} · max_len {MAX_LEN} · seed {SEED} · device {device}",
             "weights: listwise groups 4.0; null oos groups 1.0; S3 groups "
             "1.0; nli (both shapes) pointwise 0.5; verdict groups 0.5; bfcl "
             "pointwise 1.0 (near-miss x5.0); registered join of in-domain "
             "null negs into listwise groups", "",
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
