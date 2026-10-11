#!/usr/bin/env python3
"""Arm H1+H2: hard-negative mining + label-noise flags on the EN corpus.

Scores every English training text of data/real/intent-pairs/train.jsonl
against its OWN label world's sibling descriptions with the SHIPPED
intent-rank head (models/intent-rank-int8.onnx) and writes:

- the mined per-text top wrong-sibling descriptions (hard negatives), and
- noise flags: texts whose TRUE description is outscored by a wrong
  sibling (head confidently contradicts the label) — the H2 adjudication
  input.

The int8 graph is NOT batch-invariant (see ToolMatcher._predict), so a
text's whole sibling set is scored in ONE fixed-length batch: rankings
within a text are internally consistent, which is all mining and
flagging need.

Deterministic, offline. Writes .scratch/arm-h/ artifacts (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CORPUS = ROOT / "data/real/intent-pairs/train.jsonl"
HEAD = ROOT / "models/intent-rank-int8.onnx"
TOK = ROOT / "models/intent-rank-tokenizer.json"
B77 = ROOT / "data/incoming/external/banking77"
CLINC = ROOT / "data/incoming/external/clinc150"
OUT_DIR = ROOT / ".scratch/arm-h"

MAX_LEN = 128  # training texts are short utterances; faster than the 256 default


def desc_of(intent: str) -> str:
    return f"{intent.replace('_', ' ')} ({intent.replace('_', ' ')})"


def load_matcher():
    from tokenizers import Tokenizer

    import onnxruntime as ort
    from mivia_small_json import integrity

    if not integrity.verified(HEAD) or not integrity.verified(TOK):
        print("shipped head or tokenizer failed its sha256 check", file=sys.stderr)
        raise SystemExit(2)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0  # all cores
    sess = ort.InferenceSession(str(HEAD), so, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(TOK))
    tok.enable_truncation(max_length=MAX_LEN)
    tok.no_padding()
    inputs = [i.name for i in sess.get_inputs()]
    return sess, tok, inputs


def score_batch(sess, tok, inputs, requests: list[str], descs: list[str]) -> np.ndarray:
    """(n,) sigmoid probabilities for zipped (request, desc) pairs, one
    session.run; fixed-length padding keeps the int8 graph's input shape
    constant so every pair in the run sees the same quantization frame."""
    encs = tok.encode_batch(list(zip(requests, descs)))
    pad = max(len(e.ids) for e in encs)
    ids = np.zeros((len(encs), pad), dtype=np.int64)
    att = np.zeros((len(encs), pad), dtype=np.int64)
    seg = np.zeros((len(encs), pad), dtype=np.int64)
    for i, e in enumerate(encs):
        ids[i, : len(e.ids)] = e.ids
        att[i, : len(e.ids)] = e.attention_mask
        seg[i, : len(e.ids)] = e.type_ids
    feed = {"input_ids": ids, "attention_mask": att}
    if "token_type_ids" in inputs:
        feed["token_type_ids"] = seg
    logits = sess.run(None, feed)[0]
    # The shipped head's convention (ToolMatcher._predict): match
    # probability = 1 / (1 + exp(logits[0] - logits[1])) — i.e. sigmoid of
    # logits[1] - logits[0], NOT the usual sigmoid(logit0 - logit1).
    diff = logits[:, 1] - logits[:, 0]
    return 1.0 / (1.0 + np.exp(-np.clip(diff, -60, 60)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N texts per world (timing probe)")
    args = ap.parse_args()

    worlds = {
        "banking77": {
            "intents": json.loads((B77 / "categories.json").read_text()),
            "prefixes": ("banking77_train_pos", "banking77_train_neg"),
        },
        "clinc150": {
            "intents": json.loads((CLINC / "labels.json").read_text()),
            "prefixes": ("clinc150_train_pos", "clinc150_train_neg"),
        },
    }
    desc_of_map = {
        w: {i: desc_of(i) for i in meta["intents"]}
        for w, meta in worlds.items()
    }

    # unique positive texts per world (negatives share the same texts)
    texts_by_world: dict[str, dict[str, str]] = {w: {} for w in worlds}
    # text -> its true intent (per world; a text lives in exactly one world)
    label_by_world: dict[str, dict[str, str]] = {w: {} for w in worlds}
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["label"] != 1:
            continue
        for w, meta in worlds.items():
            if r["source"] == meta["prefixes"][0]:
                texts_by_world[w][r["request"]] = r["tool"]
                # invert desc -> intent
                inv = {v: k for k, v in desc_of_map[w].items()}
                label_by_world[w][r["request"]] = inv[r["tool"]]
                break

    sess, tok, inputs = load_matcher()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hard_out = (OUT_DIR / "hard_negatives.jsonl").open("w", encoding="utf-8")
    flag_out = (OUT_DIR / "noise_flags.jsonl").open("w", encoding="utf-8")
    n_scored = n_flagged = 0

    for w, meta in worlds.items():
        intents = meta["intents"]
        descs = [desc_of_map[w][i] for i in intents]
        items = sorted(texts_by_world[w].items())
        if args.limit:
            items = items[: args.limit]
        print(f"{w}: scoring {len(items)} texts x {len(intents)} descs")
        for k, (text, _) in enumerate(items):
            true_intent = label_by_world[w][text]
            probs = score_batch(sess, tok, inputs,
                                [text] * len(descs), descs)
            order = np.argsort(-probs)
            ranked = [(intents[j], float(probs[j])) for j in order]
            wrong = [(i, p) for i, p in ranked if i != true_intent]
            hard_out.write(json.dumps({
                "world": w, "text": text, "true": true_intent,
                "hard": wrong[:3],
            }, ensure_ascii=False) + "\n")
            if ranked[0][0] != true_intent:
                n_flagged += 1
                flag_out.write(json.dumps({
                    "world": w, "text": text, "true": true_intent,
                    "pred": ranked[0][0], "pred_p": ranked[0][1],
                    "true_p": dict(ranked)[true_intent],
                }, ensure_ascii=False) + "\n")
            n_scored += 1
            if n_scored % 500 == 0:
                print(f"  {n_scored} texts scored, {n_flagged} flagged",
                      flush=True)

    hard_out.close()
    flag_out.close()
    print(f"DONE scored={n_scored} flagged={n_flagged} "
          f"({n_flagged / max(1, n_scored):.1%}) -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
