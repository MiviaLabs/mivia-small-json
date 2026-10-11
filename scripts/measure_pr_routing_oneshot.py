#!/usr/bin/env python3
"""ONE-SHOT candidate measurement for Arm B (pr-routing-real).

Loads an int8 ONNX router CANDIDATE from the gitignored staging dir
(data/incoming/models/), measures it exactly once against the
pre-registered bar, and writes the verdict JSON. Never tunes anything;
never writes to models/.

Bar (pre-registered in the spec, before any candidate existed):
- real-PR routing >= 50% -> pull_request on the 300 eval PRs
- data/heldout/test.jsonl routed-and-valid == 100%
- synthetic dev accuracy >= 0.99 AND real dev accuracy >= 0.99

Run: python scripts/measure_pr_routing_oneshot.py --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json import bootstrap, encoder, model  # noqa: E402
from mivia_small_json.pipeline import classify  # noqa: E402
from bench_reality_extract import gh_prs  # noqa: E402

STAGE = ROOT / "data" / "incoming" / "models"
OUT = ROOT / "reports"


def _sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.parent / (path.name + ".sha256")).write_text(digest + "\n")


def load_candidate(seed: int):
    tag = "" if seed == 42 else f"-s{seed}"
    onnx = STAGE / f"router-encoder-int8{tag}.onnx"
    tok = STAGE / f"minilm{tag}" / "tokenizer.json"
    classes = tuple(json.loads(
        (STAGE / f"router-encoder-classes{tag}.json").read_text()))
    for p in (onnx, tok):
        if not p.is_file():
            raise SystemExit(f"candidate missing: {p}")
        _sidecar(p)
    router = encoder.load_encoder(onnx, tok, classes=classes)
    if router is None:
        raise SystemExit("candidate failed sha-verified load")
    return router


def _chunked_predict(router, texts, n: int = 64) -> list:
    """EncoderRouter._logits pads the WHOLE list densely in one array, so
    a bulk predict over thousands of rows allocates tens of GB (this
    crashed the desktop twice). Score in small chunks — same logits,
    bounded memory."""
    out: list = []
    for i in range(0, len(texts), n):
        out.extend(router.predict(texts[i:i + n]))
    return out


def heldout_routed_and_valid(router) -> float:
    rows = [json.loads(line) for line in
            (ROOT / "data" / "heldout" / "test.jsonl").read_text().splitlines()
            if line.strip()]
    ok = 0
    for r in rows:
        obj, meta = classify(r["raw"], router=router)
        ok += bool(meta.get("valid") and obj.get("type") == r["schema"])
    return 100.0 * ok / len(rows)


def dev_accuracy(router, pairs) -> tuple[float, int]:
    X, y = model._load_real(pairs)
    pred = _chunked_predict(router, X)
    return (sum(p == t for p, t in zip(pred, y)) / len(y), len(y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    bootstrap.register_all_schemas()
    router = load_candidate(args.seed)

    pr = gh_prs(n=300, router=router)
    pr_pct = 100.0 * pr["routing"].get("pull_request", 0) / pr["n"]
    hov = heldout_routed_and_valid(router)
    syn_X, syn_y = model._load_jsonl(ROOT / "data" / "synthetic" / "dev.jsonl")
    syn_pred = _chunked_predict(router, syn_X)
    syn_acc = sum(p == t for p, t in zip(syn_pred, syn_y)) / len(syn_y)
    real_acc, real_n = dev_accuracy(router, model.REAL_DEV_SETS)

    verdict = {
        "seed": args.seed,
        "bar": {"pr_routing_pct": 50.0, "heldout_pct": 100.0,
                "synthetic_dev": 0.99, "real_dev": 0.99},
        "pr_routing_pct": round(pr_pct, 1),
        "pr_routing_dist": pr["routing"],
        "heldout_routed_and_valid_pct": round(hov, 1),
        "synthetic_dev_accuracy": round(syn_acc, 4),
        "real_dev_accuracy": round(real_acc, 4),
        "real_dev_n": real_n,
    }
    verdict["pass"] = (
        pr_pct >= 50.0 and hov == 100.0
        and syn_acc >= 0.99 and real_acc >= 0.99)
    out = OUT / f"pr-routing-oneshot-s{args.seed}.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"\nverdict (seed {args.seed}): {'PASS' if verdict['pass'] else 'FAIL'}"
          f" -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
