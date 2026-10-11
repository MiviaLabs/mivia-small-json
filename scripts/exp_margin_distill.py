#!/usr/bin/env python3
"""Margin-MSE distillation for the tool-match guard (arm: margin-distill).

Hofstätter et al., "Improving Efficient Neural Ranking Models with Cross-
Architecture Knowledge Distillation" (arXiv:2010.02666) showed that a small
student cross-encoder learns from a large teacher by matching *score margins*
rather than absolute scores or 0/1 labels. The teacher and student logits live
on different scales, so an absolute MSE is meaningless; the difference between
a positive and a negative for the same query is scale-comparable, and that is
what the student regresses onto. Logits-only, zero inference cost: the shipped
artifact keeps its shape and the runtime is untouched.

Teacher: BAAI/bge-reranker-v2-m3 (568M, cross-encoder, ~25x the student).
Student: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M, the shipped backbone),
exported int8 ONNX exactly the way scripts/train_tool_match.py does.

Loss per request group (a "group" is all pairs sharing a request string):

  margin  = MSE( (s_pos - s_neg) , (t_pos - t_neg) )   over pos x neg pairs
  anchor  = near-miss-weighted BCE on every member (keeps the absolute scale
            calibrated so the 0.6 threshold still means something)

Groups with only positives or only negatives fall back to BCE alone, which is
the "pairs without both members" path. Both teacher and student scores are
sigmoid probabilities, so the margin target is bounded in [-1, 1].

This script only writes a candidate artifact and its sha256 sidecar under
data/incoming/models/ (gitignored); promoting to models/ is a separate,
deliberate step. It refuses to write anywhere under models/.

  python scripts/exp_margin_distill.py                 # train + export + eval
  python scripts/exp_margin_distill.py --eval-only     # measure an artifact
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the shipped trainer's constants and export path so the artifact is
# byte-for-byte the same *kind* of thing train_tool_match.py produces.
from train_tool_match import (  # noqa: E402
    BASE,
    NEAR_MISS_SOURCES,
    NEAR_MISS_WEIGHT,
    export as _export,
)

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
DEFAULT_PAIRS = BFCL / "pairs.jsonl"
FP_EVAL = BFCL / "irrelevance_eval.jsonl"
CALLS = BFCL / "calls.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "tool-match-marginmse-int8.onnx"
DEFAULT_TEACHER_CACHE = ROOT / ".scratch" / "margin-distill" / "teacher_scores.json"
DEFAULT_SUMMARY = ROOT / ".scratch" / "margin-distill" / "summary.json"
DEFAULT_TOKENIZER = ROOT / "models" / "router-tokenizer.json"
TEACHER = "BAAI/bge-reranker-v2-m3"
SHIPPED = {"abstention": 80.1, "retention": 95.6}
PREVIOUS_L6 = {"abstention": 69.3, "retention": 97.3}
THRESHOLD = 0.6
_MODELS_DIR = ROOT / "models"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def guard_out_path(out: Path) -> None:
    """Never let this script overwrite a shipped artifact."""
    resolved = out.resolve()
    if _MODELS_DIR.resolve() in resolved.parents or resolved.parent == _MODELS_DIR.resolve():
        raise SystemExit(f"refusing to write under models/ ({out}); "
                         "candidates go to data/incoming/models/")


# --------------------------------------------------------------------------
# Teacher scoring
# --------------------------------------------------------------------------
def teacher_scores(rows: list[dict], cache: Path, batch: int, max_len: int,
                   device: str) -> list[float]:
    """Sigmoid probability for every (request, tool) pair, cached by corpus sha."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    corpus_sha = _sha256(DEFAULT_PAIRS)
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("sha256") == corpus_sha and len(payload.get("probs", [])) == len(rows):
            print(f"teacher cache hit ({len(rows)} rows) at {_rel(cache)}")
            return [float(p) for p in payload["probs"]]

    print(f"scoring {len(rows)} pairs with teacher {TEACHER} (batch {batch}) ...")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    net = AutoModelForSequenceClassification.from_pretrained(TEACHER).to(device).eval()
    probs: list[float] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            enc = tok([r["request"] for r in chunk], [r["tool"] for r in chunk],
                      truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = net(**enc).logits.view(-1)
            probs.extend(float(x) for x in torch.sigmoid(logits).cpu())
            if (start // batch) % 50 == 0:
                done = min(start + batch, len(rows))
                rate = done / max(1e-6, time.perf_counter() - t0)
                print(f"  {done}/{len(rows)}  {rate:.0f} pair/s", flush=True)
    del net
    torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"sha256": corpus_sha, "teacher": TEACHER,
                                 "probs": probs}), encoding="utf-8")
    print(f"teacher cache written to {_rel(cache)}")
    return probs


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def group_batches(groups: list[list[dict]], batch: int, seed: int):
    """Shuffle groups, pack greedily so a step sees <= `batch` member rows."""
    import torch
    order = torch.randperm(len(groups), generator=torch.Generator().manual_seed(seed)).tolist()
    current: list[list[dict]] = []
    members = 0
    for i in order:
        g = groups[i]
        if current and members + len(g) > batch:
            yield current
            current, members = [], 0
        current.append(g)
        members += len(g)
    if current:
        yield current


def train(rows: list[dict], teacher: list[float], args) -> object:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train_rows = [r for r in rows if r["split"] == "train"]
    tmap = {id(r): teacher[i] for i, r in enumerate(rows)}

    by_request: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
        by_request[r["request"]].append(r)
    groups = list(by_request.values())
    n_both = sum(1 for g in groups
                 if any(r["label"] == 1 for r in g) and any(r["label"] == 0 for r in g))
    print(f"corpus {len(rows)} rows · train {len(train_rows)} rows · "
          f"{len(groups)} request groups ({n_both} with both pos and neg)")

    tok = AutoTokenizer.from_pretrained(args.base)
    net = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, ignore_mismatched_sizes=True).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def step_loss(groups_batch: list[list[dict]]):
        member_rows = [r for g in groups_batch for r in g]
        enc = tok([r["request"] for r in member_rows],
                  [r["tool"] for r in member_rows], truncation=True,
                  max_length=args.max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = net(**enc).logits
        probs = F.softmax(logits, dim=-1)[:, 1]
        labels = torch.tensor([r["label"] for r in member_rows],
                              device=device)
        weights = torch.tensor(
            [args.near_miss_weight if r["source"] in NEAR_MISS_SOURCES else 1.0
             for r in member_rows], dtype=torch.float, device=device)
        tprob = torch.tensor([tmap[id(r)] for r in member_rows],
                             dtype=torch.float, device=device)
        # Separate weight for the margin term. Default 1.0 = a plain margin
        # MSE (the arm as specified). >1 upweights pairs whose negative is a
        # near-miss, the quadrant the abstention eval measures.
        mweights = torch.tensor(
            [args.margin_near_miss_weight if r["source"] in NEAR_MISS_SOURCES
             else 1.0 for r in member_rows], dtype=torch.float, device=device)

        margin_sum = logits.new_zeros(())
        n_pairs = 0
        bce_sum = logits.new_zeros(())
        w_sum = 0.0
        start = 0
        for g in groups_batch:
            n = len(g)
            p = probs[start:start + n]
            y = labels[start:start + n]
            w = weights[start:start + n]
            mw = mweights[start:start + n]
            t = tprob[start:start + n]
            lg = logits[start:start + n]
            pos, neg = (y == 1), (y == 0)
            if pos.any() and neg.any():
                s_margin = p[pos][:, None] - p[neg][None, :]
                t_margin = t[pos][:, None] - t[neg][None, :]
                mse = (s_margin - t_margin) ** 2
                pair_w = mw[neg][None, :].expand(int(pos.sum()), -1)
                margin_sum = margin_sum + (mse * pair_w).sum()
                n_pairs += float(pair_w.sum())
            bce = F.cross_entropy(lg, y, reduction="none")
            bce_sum = bce_sum + (bce * w).sum()
            w_sum += float(w.sum())
            start += n
        margin_loss = margin_sum / max(1e-6, n_pairs)
        bce_loss = bce_sum / max(1e-6, w_sum)
        loss = args.margin_weight * margin_loss + args.bce_weight * bce_loss
        return loss, float(margin_loss.detach()), float(bce_loss.detach()), int(n_pairs)

    net.train()
    for epoch in range(args.epochs):
        torch.manual_seed(args.seed + epoch)
        total = ml = bl = 0.0
        pairs_seen = 0
        steps = 0
        for gb in group_batches(groups, args.batch, args.seed + epoch):
            loss, m, b, n = step_loss(gb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            ml += m
            bl += b
            pairs_seen += n
            steps += 1
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / max(1, steps):.4f} "
              f"(margin {ml / max(1, steps):.4f} · bce {bl / max(1, steps):.4f} · "
              f"{pairs_seen} margin pairs)")
    net.eval()
    return net, tok, device


# --------------------------------------------------------------------------
# Export (mirrors scripts/train_tool_match.py)
# --------------------------------------------------------------------------
def export_artifact(net, tok, out: Path, max_len: int) -> tuple[Path, Path, str]:
    int8_path, fp32_path = _export(net, tok, out, max_len)
    digest = _sha256(int8_path)
    (int8_path.parent / (int8_path.name + ".sha256")).write_text(
        digest + "\n", encoding="utf-8")
    tok_path = int8_path.parent / (int8_path.name.replace("-int8.onnx",
                                                          "-tokenizer.json"))
    if hasattr(tok, "backend_tokenizer"):
        tok.backend_tokenizer.save(str(tok_path))
    elif hasattr(tok, "save"):
        tok.save(str(tok_path))
    return int8_path, fp32_path, digest


# --------------------------------------------------------------------------
# Evaluation with the production scorer
# --------------------------------------------------------------------------
def _texts(path: Path) -> list[str]:
    return [rec.get("text") or rec.get("raw") or "" for rec in load_rows(path)]


def evaluate(model: Path, tokenizer: Path | None, max_len: int) -> dict:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    from mivia_small_json.tool_match import ToolMatcher

    if tokenizer is None:
        cand = model.parent / (model.name.replace("-int8.onnx", "-tokenizer.json"))
        tokenizer = cand if cand.is_file() else DEFAULT_TOKENIZER
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(tokenizer))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=None)
    matcher = ToolMatcher(session, tok, [i.name for i in session.get_inputs()])

    fp = [matcher.best_score(t) for t in _texts(FP_EVAL)]
    calls = [matcher.best_score(t) for t in _texts(CALLS)]

    def abstention(thr):
        return 100 * sum(1 for s in fp if s < thr) / len(fp)

    def retention(thr):
        return 100 * sum(1 for s in calls if s >= thr) / len(calls)

    curve = [(t, abstention(t), retention(t)) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    # Fine frontier: best abstention at retention >= 95%.
    frontier = []
    for i in range(0, 201):
        thr = i / 200
        frontier.append((thr, abstention(thr), retention(thr)))
    eligible = [f for f in frontier if f[2] >= 95.0]
    best = max(eligible, key=lambda f: f[1]) if eligible else None
    return {
        "model": str(model),
        "tokenizer": str(tokenizer),
        "abstention": abstention(THRESHOLD),
        "retention": retention(THRESHOLD),
        "n_fp": len(fp), "n_calls": len(calls),
        "curve": curve,
        "frontier_best": best,
    }


def print_eval(res: dict) -> None:
    print(f"\nartifact {Path(res['model']).name} · threshold {THRESHOLD}")
    print(f"abstention {res['abstention']:.1f}% ({res['n_fp']} rows) · "
          f"retention {res['retention']:.1f}% ({res['n_calls']} rows)")
    print(f"{'thr':>5} {'abstain %':>10} {'kept %':>8}")
    for t, a, k in res["curve"]:
        print(f"{t:>5.1f} {a:>10.1f} {k:>8.1f}")
    if res["frontier_best"]:
        t, a, k = res["frontier_best"]
        print(f"frontier: abstention {a:.1f}% at retention {k:.1f}% (thr={t:.3f})")
    print(f"baselines: shipped {SHIPPED['abstention']}/{SHIPPED['retention']} · "
          f"previous L6 {PREVIOUS_L6['abstention']}/{PREVIOUS_L6['retention']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--base", type=str, default=BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--teacher-cache", type=Path, default=DEFAULT_TEACHER_CACHE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--teacher-max-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--teacher-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-miss-weight", type=float, default=NEAR_MISS_WEIGHT)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--margin-near-miss-weight", type=float, default=1.0,
                        help="weight on margin pairs whose negative is a "
                             "near-miss source (default 1.0 = plain margin MSE)")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--bench", action="store_true",
                        help="also run the official bench_tool_abstention.py")
    args = parser.parse_args()

    rows = load_rows(args.pairs)
    print(f"corpus {_rel(args.pairs)} ({len(rows)} rows)")

    if args.eval_only:
        res = evaluate(args.out, None, args.max_len)
        print_eval(res)
        return 0

    guard_out_path(args.out)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA device; training on CPU", file=sys.stderr)

    teacher = teacher_scores(rows, args.teacher_cache, args.teacher_batch,
                             args.teacher_max_len, device)

    # Sanity: the teacher should separate labelled pairs.
    import statistics
    tp = [teacher[i] for i, r in enumerate(rows) if r["label"] == 1]
    tn = [teacher[i] for i, r in enumerate(rows) if r["label"] == 0]
    print(f"teacher mean prob: pos {statistics.mean(tp):.3f} · "
          f"neg {statistics.mean(tn):.3f}")

    net, tok, device = train(rows, teacher, args)
    int8_path, fp32_path, digest = export_artifact(net, tok, args.out, args.max_len)
    print(f"exported {_rel(int8_path)} ({int8_path.stat().st_size} bytes) · "
          f"sha256 {digest[:16]} · fp32 {_rel(fp32_path)}")

    res = evaluate(int8_path, None, args.max_len)
    print_eval(res)

    if args.bench:
        print("\n=== official bench_tool_abstention.py ===")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_tool_abstention.py"),
                        "--model", str(int8_path)], check=False)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({
        "artifact": _rel(int8_path), "sha256": digest,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "margin_weight": args.margin_weight, "bce_weight": args.bce_weight,
        "margin_near_miss_weight": args.margin_near_miss_weight,
        "teacher": TEACHER, "base": args.base,
        "eval": res,
    }, indent=1), encoding="utf-8")
    print(f"\nsummary written to {_rel(args.summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
