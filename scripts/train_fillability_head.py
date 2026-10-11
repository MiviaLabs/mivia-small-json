#!/usr/bin/env python3
"""Objective-level fillability term for the tool-match gate (TRACK 1).

Diagnosis (reports/exp-infeasible-diagnosis.md): on the 40 incomplete
false-call rows the score is 100% topic-driven (topic-mask moves 40/40,
arg-strip moves 0/40) — the `missing:` tail gets zero gradient. Step-4
presence-contrast training is already falsified on this slice, so this track
does NOT repeat minimal-pair training. Instead it adds an objective-level
fillability term: a multitask auxiliary head predicting request-side
fillability alongside the match decision.

One-line justification: pointwise cross-entropy never constrains the
missing-tail ordering, so forcing the shared pooled representation to also
predict "does this request carry fillable values" puts gradient on the
value-presence mechanism at the objective level, while the exported artifact
keeps the shipped backbone, single-`logits[batch,2]` shape, and export path
(the aux head is training-time only, like `--error-reg`).

Honest label caveat: full `validate_fillability` needs a structured tool
dict, which `pairs.jsonl` rows do not carry (rendered "name: description"
strings only). The aux target is therefore its request-side projection —
`request_concreteness`: whether the request text supplies a concrete value
(number, number-word, URL, quoted string) using the same patterns
`tool_match` itself uses. No synthetic rows are added, so train/eval
disjointness is the shipped corpus's own (verified here: zero request
overlap with guard-eval-v2 and the v1 evals).

GPU is SERIAL: wrap every invocation in
`flock -w 7200 /tmp/mivia_gpu.lock`. Candidates only under
`data/incoming/models/`, never promoted without clearing the bar in
reports/exp-fillability-head.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_tool_match as ttm  # noqa: E402
from mivia_small_json.determinism import seed_everything  # noqa: E402
from mivia_small_json.tool_match import (  # noqa: E402
    _NUM_PATTERN,
    _NUM_WORDS,
    _URL_PATTERN,
)

OUT_STEM = "tool-match-fillability"
METRICS = ROOT / "data" / "incoming" / "models" / "fillability-head-metrics.json"
GUARD_EVAL_V2 = ROOT / "data" / "incoming" / "external" / "bfcl" / "guard-eval-v2.jsonl"
V1_EVALS = (
    ROOT / "data" / "incoming" / "external" / "bfcl" / "irrelevance_eval.jsonl",
    ROOT / "data" / "incoming" / "external" / "bfcl" / "calls.jsonl",
)
_QUOTED = re.compile(r"'[^']{1,60}'|\"[^\"]{1,60}\"")


def request_concreteness(request: str) -> int:
    """Request-side fillability projection: 1 when the request carries a
    concrete fillable value (number, number-word, URL, quoted string), else 0.

    Pure: the aux head's target. Uses the same value patterns the
    `validate_fillability` gate reads, minus the tool-dict side that
    `pairs.jsonl` rows do not carry.
    """
    words = re.findall(r"\w+", request.lower())
    if _NUM_PATTERN.search(request):
        return 1
    if any(w in _NUM_WORDS for w in words):
        return 1
    if _URL_PATTERN.search(request):
        return 1
    if _QUOTED.search(request):
        return 1
    return 0


def multitask_loss(match_logits, aux_logits, match_labels, aux_labels,
                   weights=None, aux_weight: float = 0.5):
    """Weighted match CE + aux_weight * unweighted fillability BCE.

    Pure given tensors: the match term is the shipped loss exactly;
    aux_weight 0.0 reproduces it bit-for-bit modulo batching.
    """
    import torch.nn.functional as F

    ce = F.cross_entropy(match_logits, match_labels, reduction="none")
    if weights is not None:
        match_term = (ce * weights).sum() / (weights.sum() + 1e-12)
    else:
        match_term = ce.mean()
    aux_term = F.binary_cross_entropy_with_logits(
        aux_logits.view(-1), aux_labels.float())
    return match_term + aux_weight * aux_term, match_term, aux_term


def verify_disjoint(train_requests: set[str]) -> dict:
    """Zero request overlap between training rows and every eval set."""
    out = {}
    for path in (GUARD_EVAL_V2, *V1_EVALS):
        rows = ttm.load_rows(path)
        if path.name == "calls.jsonl":
            eval_reqs = {
                m.group(1).strip()
                for r in rows
                for m in [re.search(r"User request: (.*)$",
                                    r.get("text", ""), re.S)]
                if m and m.group(1).strip()
            }
        else:
            eval_reqs = {r.get("request", "") for r in rows}
        hits = len(train_requests & eval_reqs)
        out[path.name] = {"eval_rows": len(rows), "request_hits": hits}
        assert hits == 0, f"{path.name}: {hits} train/eval request overlaps"
    return out


V2_RELEVANCE = "single_turn_relevance"
V2_INCOMPLETE_KEY = "incomplete"
SHIPPED_MODEL = ROOT / "models" / "tool-match-int8.onnx"
SHIPPED_TOKENIZER = ROOT / "models" / "router-tokenizer.json"


def _catalogue_text(tool: dict, request: str) -> str:
    return ("Tools available:\n" + json.dumps([tool], ensure_ascii=False)
            + "\n\nUser request: " + request)


def _score_onnx(int8_path: Path, tok_path: Path, texts: list[str],
                max_len: int = 256, batch: int = 64) -> list[float]:
    """Batched single-logit ONNX scorer (CPU). Candidate shape must equal
    shipped: one `logits[batch,2]` output."""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    session = ort.InferenceSession(str(int8_path),
                                   providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    assert [o.name for o in session.get_outputs()] == ["logits"], \
        f"export shape drift: {[o.name for o in session.get_outputs()]}"
    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=None)
    out: list[float] = []
    for start in range(0, len(texts), batch):
        enc = tok.encode_batch(texts[start:start + batch])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc],
                                       dtype=np.int64),
        }
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in enc], dtype=np.int64)
        logits = session.run(["logits"], feed)[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        out.extend(float(p) for p in (e / e.sum(axis=1, keepdims=True))[:, 1])
    return out


def _catalogue_max(rows: list[dict], scores: list[float],
                   key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row, score in zip(rows, scores):
        k = f"{row['category']}||{row['source']}||{key}"
        out[k] = max(out.get(k, -1.0), score)
    return out


def _op_point(neg: list[float], pos: list[float],
              floor: float = 95.0) -> tuple[float, float, float]:
    best = None
    for thr in sorted(set(neg + pos)):
        kept = 100 * sum(s >= thr for s in pos) / len(pos)
        abst = 100 * sum(s < thr for s in neg) / len(neg)
        if kept >= floor and (best is None or abst > best[1]):
            best = (thr, abst, kept)
    return best or (0.6, 0.0, 100.0)


def measure_candidate(eval_rows: list[dict], cand_path: Path,
                      cand_tok: Path | None = None) -> dict:
    """Canonical guard-eval-v2 per-category before/after at each model's own
    >=95%-retention operating point (shipped single vs candidate single)."""
    from scipy.stats import binomtest

    texts = [_catalogue_text(r["tool"], r["request"]) for r in eval_rows]
    before = _score_onnx(SHIPPED_MODEL, SHIPPED_TOKENIZER, texts)
    after = _score_onnx(cand_path,
                        cand_tok or SHIPPED_TOKENIZER, texts)
    cats = sorted({r["category"] for r in eval_rows})

    def per_cat(scores: list[float], thr: float) -> dict:
        tab = {}
        for cat in cats:
            sub = [s for r, s in zip(eval_rows, scores)
                   if r["category"] == cat]
            keys = {r["source"] for r in eval_rows if r["category"] == cat}
            cat_scores = [_catalogue_max(
                [r for r in eval_rows if r["category"] == cat],
                sub, "")[f"{cat}||{k}||"] for k in keys]
            abst = 100 * sum(s < thr for s in cat_scores) / len(cat_scores)
            tab[cat] = {"catalogues": len(cat_scores),
                        "catalogue_abstention_at_op": round(abst, 1),
                        "catalogue_retention_at_op": round(100 - abst, 1)}
        return tab

    # Catalogue-level op points (max over tools per source catalogue).
    bcat = _catalogue_max(eval_rows, before, "b")
    acat = _catalogue_max(eval_rows, after, "a")
    bpos = [v for k, v in bcat.items() if k.startswith(V2_RELEVANCE)]
    apos = [v for k, v in acat.items() if k.startswith(V2_RELEVANCE)]
    bneg = [v for k, v in bcat.items() if not k.startswith(V2_RELEVANCE)]
    aneg = [v for k, v in acat.items() if not k.startswith(V2_RELEVANCE)]
    thr_b, _, _ = _op_point(bneg, bpos)
    thr_a, _, _ = _op_point(aneg, apos)
    tab_b, tab_a = per_cat(before, thr_b), per_cat(after, thr_a)

    inc_keys = sorted({r["source"] for r in eval_rows
                       if r.get("source", "").startswith(
                           "acebench_special_incomplete")})
    dec_b = {k: bcat[f"infeasible||{k}||b"] < thr_b for k in inc_keys}
    dec_a = {k: acat[f"infeasible||{k}||a"] < thr_a for k in inc_keys}
    b_reg = sum(1 for k in inc_keys if dec_b[k] and not dec_a[k])
    c_fix = sum(1 for k in inc_keys if not dec_b[k] and dec_a[k])
    p = float(binomtest(min(b_reg, c_fix), b_reg + c_fix, 0.5).pvalue) \
        if (b_reg + c_fix) else 1.0
    lift = (tab_a.get("infeasible", {}).get("catalogue_abstention_at_op", 0)
            - tab_b.get("infeasible", {}).get("catalogue_abstention_at_op", 0))
    # Incomplete-slice lift: abstention on the incomplete source prefix.
    inc_b = 100 * sum(1 for k in inc_keys if dec_b[k]) / max(1, len(inc_keys))
    inc_a = 100 * sum(1 for k in inc_keys if dec_a[k]) / max(1, len(inc_keys))
    inc_lift = inc_a - inc_b
    rel_hold = (tab_a[V2_RELEVANCE]["catalogue_retention_at_op"]
                - tab_b[V2_RELEVANCE]["catalogue_retention_at_op"])
    irr_reg = (tab_b["single_turn_irrelevance"]["catalogue_abstention_at_op"]
               - tab_a["single_turn_irrelevance"]["catalogue_abstention_at_op"])
    verdict = ("PASS" if (inc_lift >= 15.0 and p < 0.05 and rel_hold == 0.0
                          and irr_reg <= 1.0) else "FAIL")
    return {
        "eval": "guard-eval-v2", "rows": len(eval_rows),
        "shipped_thr": round(thr_b, 4), "candidate_thr": round(thr_a, 4),
        "before": tab_b, "after": tab_a,
        "incomplete_catalogues": len(inc_keys),
        "mcnemar": {"regressions_b": b_reg, "fixes_c": c_fix,
                    "exact_p": p},
        "bar": {"incomplete_lift_pp": round(inc_lift, 1),
                "infeasible_lift_pp": round(lift, 1),
                "relevance_retention_delta_pp": rel_hold,
                "irrelevance_regression_pp": round(irr_reg, 1),
                "mcnemar_p": p,
                "verdict": verdict},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ttm.DEFAULT_PAIRS)
    parser.add_argument("--base", type=str, default=ttm.BASE)
    parser.add_argument("--out-stem", type=str, default=OUT_STEM)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--near-miss-weight", type=float,
                        default=ttm.NEAR_MISS_WEIGHT)
    parser.add_argument("--eval-only", type=Path, default=None,
                        help="score this existing candidate instead of training")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    seed_everything(args.seed)
    rows = ttm.load_rows(args.pairs)
    train = [r for r in rows if r["split"] == "train"]
    disjoint = verify_disjoint({r["request"] for r in train})
    print(f"disjoint: {json.dumps(disjoint)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device != "cuda":
        print("WARNING: no CUDA device; training on CPU", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.base)
    out_int8 = (ROOT / "data" / "incoming" / "models"
                / f"{args.out_stem}-s{args.seed}-int8.onnx")
    if args.eval_only is None:
        base_net = AutoModelForSequenceClassification.from_pretrained(
            args.base, num_labels=2, ignore_mismatched_sizes=True).to(device)
        hidden = base_net.config.hidden_size
        aux_head = nn.Linear(hidden, 1).to(device)
        opt = torch.optim.AdamW(
            list(base_net.parameters()) + list(aux_head.parameters()),
            lr=args.lr)
        near_miss_sources = ttm.NEAR_MISS_SOURCES

        def weight_of(row: dict) -> float:
            return (args.near_miss_weight
                    if row["source"] in near_miss_sources else 1.0)

        aux_labels_all = torch.tensor(
            [request_concreteness(r["request"]) for r in train],
            dtype=torch.float)
        print(f"aux positive rate on train: {float(aux_labels_all.mean()):.3f} "
              f"(n={len(train)})")
        base_net.train()
        for epoch in range(args.epochs):
            order = torch.randperm(len(train)).tolist()
            total = 0.0
            for start in range(0, len(order), args.batch):
                idx = order[start:start + args.batch]
                chunk = [train[i] for i in idx]
                enc = tok([r["request"] for r in chunk],
                          [r["tool"] for r in chunk], truncation=True,
                          max_length=args.max_len, padding=True,
                          return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = torch.tensor(
                    [r["label"] for r in chunk]).to(device)
                aux_t = aux_labels_all[idx].to(device)
                weights = torch.tensor(
                    [weight_of(r) for r in chunk],
                    dtype=torch.float).to(device)
                out = base_net(**enc, output_hidden_states=True)
                pooled = out.hidden_states[-1][:, 0]
                aux_logits = aux_head(pooled)
                loss, _, _ = multitask_loss(
                    out.logits, aux_logits, labels, aux_t,
                    weights=weights, aux_weight=args.aux_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach())
            print(f"epoch {epoch + 1}: loss "
                  f"{total / max(1, len(train) // args.batch):.4f}")
        base_net.eval()
        int8_path, _ = ttm.export(base_net, tok, out_int8, args.max_len)
        digest = hashlib.sha256(int8_path.read_bytes()).hexdigest()
        (int8_path.parent / (int8_path.name + ".sha256")).write_text(
            digest + "\n", encoding="utf-8")
        print(f"exported {int8_path.name} ({int8_path.stat().st_size} bytes) "
              f"sha256 {digest[:16]}")
    else:
        int8_path = args.eval_only

    cand_tok = int8_path.parent / (
        int8_path.name.replace("-int8.onnx", "-tokenizer.json"))
    if hasattr(tok, "backend_tokenizer"):
        tok.backend_tokenizer.save(str(cand_tok))
    payload = measure_candidate(
        ttm.load_rows(GUARD_EVAL_V2), int8_path, cand_tok)
    payload["config"] = {"base": args.base, "epochs": args.epochs,
                         "batch": args.batch, "lr": args.lr,
                         "seed": args.seed, "aux_weight": args.aux_weight,
                         "max_len": args.max_len,
                         "eval_set": "guard-eval-v2 (disjoint, see disjoint)",
                         "device": device}
    payload["disjoint"] = disjoint
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(METRICS.read_text()) if METRICS.is_file() else {}
    prior[f"seed{args.seed}-aux{args.aux_weight}"] = payload
    METRICS.write_text(json.dumps(prior, indent=1), encoding="utf-8")
    print(f"metrics -> {METRICS} :: incomplete lift "
          f"{payload['bar']['incomplete_lift_pp']}pp "
          f"p={payload['bar']['mcnemar_p']:.4g} verdict={payload['bar']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
