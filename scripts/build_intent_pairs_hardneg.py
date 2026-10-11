#!/usr/bin/env python3
"""Arm H1+H2 corpus rebuild: quality over volume, volume ~FLAT.

Reads data/real/intent-pairs/train.jsonl (the v1 corpus the shipped EN
head trained on) plus the mining artifacts in .scratch/arm-h/ and writes
data/real/intent-pairs/train-hardneg.jsonl:

- positives: unchanged, MINUS confirmed-noise texts (noise_confirmed.jsonl
  written by the H2 adjudication step; absent file = no drops yet);
- negatives per EN row: 2 sampled random + 1 mined-hard wrong sibling
  (the top-scoring wrong description for that text) — the published
  ~1/3-hard mix; pure-hard ablations score worse than random;
- When2Call sibling negatives: passed through unchanged;
- B77-OOS (ood-oos valid, 200 banking requests the 77 intents do NOT
  cover): label-0 rows against the text's MOST CONFUSABLE banking
  description (the mined top-1) + 2 sampled others — true in-domain
  hard negatives.

Deterministic (seed 42), offline. The v1 corpus and its sha pin are
untouched.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_intent_pairs import B77, CLINC  # noqa: E402

V1 = ROOT / "data/real/intent-pairs/train.jsonl"
ARM_H = ROOT / ".scratch/arm-h"
OUT_V4 = ROOT / "data/real/intent-pairs/train-hardneg.jsonl"
B77_OOS = ROOT / "data/incoming/external/banking77-oos"
SEED = 42


def split_of(text: str) -> str:
    """The v1 hash-stable 90/10 split — same text, same split."""
    h = int(hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest(), 16)
    return "dev" if h % 10 == 0 else "train"


def _oos_top_desc(oos_texts: list[str], descs: dict[str, str]) -> dict[str, str]:
    """Most-confusable banking description per OOS text, scored with the
    shipped head (200 x 77 pairs — a tiny deterministic pass)."""
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np

    from mivia_small_json.tool_match import load_single_matcher

    m = load_single_matcher(ROOT / "models/intent-rank-int8.onnx",
                            tokenizer_path=ROOT / "models/intent-rank-tokenizer.json")
    if m is None:
        raise SystemExit("shipped intent head failed to load")
    names = list(descs)
    out: dict[str, str] = {}
    for t in oos_texts:
        scores = [m.score_pair(t, descs[i]) for i in names]
        out[t] = names[int(np.argmax(scores))]
    return out


def main() -> int:
    rng = random.Random(SEED)
    hard: dict[tuple[str, str], list[tuple[str, float]]] = {}
    hard_path = ARM_H / "hard_negatives.jsonl"
    if not hard_path.is_file():
        print("mining artifacts missing — run scripts/mine_hard_negatives.py "
              "first", file=sys.stderr)
        return 2
    for line in hard_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        hard[(r["world"], r["text"])] = [(i, p) for i, p in r["hard"]]

    noise: set[str] = set()
    noise_path = ROOT / "data/real/intent-pairs/h2-noise-drop.jsonl"
    if noise_path.is_file():
        for line in noise_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                noise.add(json.loads(line)["text"])
        print(f"H2 drops: {len(noise)} confirmed-noise texts excluded")
    relabel: dict[str, str] = {}
    relabel_path = ROOT / "data/real/intent-pairs/h2-noise-relabel.jsonl"
    if relabel_path.is_file():
        for line in relabel_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                relabel[r["text"]] = r["better"]
        print(f"H2 relabels: {len(relabel)} confirmed-noise labels corrected")

    worlds = {
        "banking77": {
            "intents": json.loads((B77 / "categories.json").read_text()),
            "pos": "banking77_train_pos",
        },
        "clinc150": {
            "intents": json.loads((CLINC / "labels.json").read_text()),
            "pos": "clinc150_train_pos",
        },
    }
    desc_of_map = {w: {i: f"{i.replace('_', ' ')} ({i.replace('_', ' ')})"
                       for i in m["intents"]}
                   for w, m in worlds.items()}

    rows: list[dict] = []
    dropped_noise = 0
    relabeled = 0
    for line in V1.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["source"] == "synthetic_nearmiss_when2call":
            rows.append(r)
            continue
        world = next((w for w, m in worlds.items()
                      if r["source"].startswith(w)), None)
        if world is None:
            rows.append(r)
            continue
        text = r["request"]
        if r["label"] == 1:
            if text in noise:
                dropped_noise += 1
                continue
            if text in relabel:
                r = dict(r, tool=desc_of_map[world][relabel[text]])
                relabeled += 1
            rows.append(r)
            # negatives: 2 random + 1 hard (kept together with the positive
            # so v4 volume tracks v1 per row)
            intents = worlds[world]["intents"]
            true_desc = r["tool"]
            negs = []
            for cand in rng.sample([i for i in intents
                                    if desc_of_map[world][i] != true_desc], 2):
                negs.append(desc_of_map[world][cand])
            hard_descs = [d for d, _ in hard.get((world, text), [])
                          if d != true_desc]
            if hard_descs:
                negs.append(hard_descs[0])
            else:  # mining gap (shouldn't happen): fall back to random
                negs.append(desc_of_map[world][rng.choice(
                    [i for i in intents if desc_of_map[world][i] != true_desc])])
            split = split_of(text)
            for nd in negs:
                rows.append({"label": 0, "request": text,
                             "source": f"{world}_hardneg_neg",
                             "split": split, "tool": nd})

    # --- B77-OOS: banking requests the 77 intents do not cover ------------
    oos_texts = [t.strip() for t in (B77_OOS / "ood-oos__valid__seq.in")
                 .read_text(encoding="utf-8").splitlines() if t.strip()]
    b77_intents = worlds["banking77"]["intents"]
    confusable = _oos_top_desc(oos_texts, desc_of_map["banking77"])
    n_oos = 0
    for t in oos_texts:
        split = split_of(t)
        top = confusable[t]
        others = [i for i in b77_intents
                  if desc_of_map["banking77"][i] != top]
        rows.append({"label": 0, "request": t, "source": "banking77_oos_neg",
                     "split": split, "tool": top})
        for cand in rng.sample(others, 2):
            rows.append({"label": 0, "request": t,
                         "source": "banking77_oos_neg", "split": split,
                         "tool": desc_of_map["banking77"][cand]})
        n_oos += 1

    OUT_V4.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    digest = hashlib.sha256(OUT_V4.read_bytes()).hexdigest()
    from collections import Counter
    by_src = Counter(r["source"] for r in rows)
    pos = sum(r["label"] for r in rows)
    print(f"rows={len(rows)} pos={pos} oos_texts={n_oos} "
          f"noise_dropped={dropped_noise} relabeled={relabeled} "
          f"sha256={digest[:12]}...")
    print("by source:", dict(sorted(by_src.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
