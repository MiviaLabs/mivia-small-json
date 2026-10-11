#!/usr/bin/env python3
"""Weakness probe for the zero-shot-select r1 corpus (protocol 6b).

Routes sampled positives from the round's corpus through the SHIPPED rank
artifact (explicit-path loading; read-only) and splits them:

- easy mass  — the shipped head already ranks gold on top of gold + 7
               same-world siblings (weight, not signal);
- signal     — gold ranked below a sibling (the texture this round trains),
               with the winning sibling's identity for the residual review.

Read-only measurement, no gate: the shares are input for the written data
review in zero-shot-select-r1
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CORPUS = ROOT / "data/real/zero-shot-select-r1"
RANK = ROOT / "models/tool-match-int8.onnx"
TOK = ROOT / "models/tool-match-tokenizer.json"
N_PER_SOURCE = 300


def main() -> int:
    from mivia_small_json.tool_match import load_single_matcher

    m = load_single_matcher(RANK, tokenizer_path=TOK)
    if m is None:
        print("shipped rank head failed to load", file=sys.stderr)
        return 2
    rng = random.Random(7)

    out: dict = {"shipped_rank_sha256": None, "per_source": {}}
    import hashlib

    out["shipped_rank_sha256"] = hashlib.sha256(RANK.read_bytes()).hexdigest()

    # source -> (positives, candidate pool)
    worlds: dict[str, list] = defaultdict(list)
    for name in ("listwise-pairs.jsonl", "null-pairs.jsonl",
                 "synthetic-labelsets.jsonl"):
        for line in (CORPUS / name).read_text(
                encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["label"] != 1:
                continue
            if name == "listwise-pairs.jsonl":
                world = r["source"].replace("_pos", "")
            elif name == "null-pairs.jsonl":
                world = "null_pos_oos"
            else:
                world = r["source"]
            worlds[world].append((r["request"], r["tool"]))

    for world, rows in sorted(worlds.items()):
        sample = rng.sample(rows, min(N_PER_SOURCE, len(rows)))
        all_descs = sorted({t for _, t in rows})
        easy = 0
        winners: Counter = Counter()
        for text, gold in sample:
            sibs = rng.sample([d for d in all_descs if d != gold],
                              min(7, len(all_descs) - 1))
            cands = [gold] + sibs
            scores = [m._score_raw(text, d) for d in cands]
            best = scores.index(max(scores))
            if best == 0:
                easy += 1
            else:
                winners[sibs[best - 1].split(":")[0].strip()] += 1
        n = len(sample)
        out["per_source"][world] = {
            "n": n,
            "easy_mass_pct": round(100 * easy / max(1, n), 1),
            "top_confusions": dict(winners.most_common(3)),
        }
        print(f"{world:28s} n={n:4d} easy {100 * easy / max(1, n):5.1f}% "
              f"top-sibling-winners {dict(winners.most_common(2))}")

    all_easy = [v["easy_mass_pct"] for v in out["per_source"].values()]
    mean = sum(all_easy) / max(1, len(all_easy))
    out["mean_easy_mass_pct"] = round(mean, 1)
    if mean >= 80:
        verdict = ("mostly weight — rethink before spending the GPU")
    elif mean >= 50:
        verdict = ("half the increment teaches nothing new — check "
                   "confusion targets cover the registered weakness")
    else:
        verdict = "targets live weaknesses — proceed to the audit gate"
    out["verdict"] = verdict
    dest = ROOT / ".scratch/zs-select-r1/weakness-probe.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nmean easy mass {mean:.1f}% — {verdict}")
    print(f"written {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
