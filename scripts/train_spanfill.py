"""Arm 2: train the spanfill value-span extractor (token x parameter pair
scorer, house FeatureHasher+LogisticRegression pattern, skops artifact).

Phases:
    --train     fit on data/real/spanfill/train.jsonl, tune the emit and
                extend thresholds on dev.jsonl, persist
                models/spanfill-extractor.skops (+ sha + trainmeta)
    --eval      ONE-SHOT on the frozen reserve (data/heldout/
                spanfill-reserve.jsonl, sha-checked)

Deterministic single train (seed 42): the both-seed rule targets GPU
fine-tuning variance; a hashed-feature LR has none (recorded deviation,
spanfill-value).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

from mivia_small_json.extract_learned import PROSE_TOKEN_RE  # noqa: E402

TRAIN = ROOT / "data" / "real" / "spanfill" / "train.jsonl"
DEV = ROOT / "data" / "real" / "spanfill" / "dev.jsonl"
RESERVE = ROOT / "data" / "heldout" / "spanfill-reserve.jsonl"
ARTIFACT = ROOT / "models" / "spanfill-extractor.skops"
META = ROOT / "models" / "spanfill-extractor-trainmeta.json"
SEED = 42


def shape(tok: str) -> str:
    return "".join("X" if c.isupper() else "x" if c.islower()
                   else "d" if c.isdigit() else c for c in tok)


def pair_features(tokens: list[str], idx: int, param: str,
                  prop: dict) -> list[str]:
    """Hashed features for one (token, parameter) pair: token context,
    parameter identity/description, and their interactions."""
    t = tokens[idx]
    low = t.lower()
    n = len(tokens)
    name_low = param.lower()
    desc = str(prop.get("description", "")).lower()
    enum = [str(v).lower() for v in prop.get("enum", [])]
    feats = {
        "tok=" + low,
        "shape=" + shape(t)[:8],
        f"len={min(len(t), 12)}",
        "posbucket=" + str(idx * 10 // max(n, 1)),
        "param=" + name_low,
        "ptype=" + str(prop.get("type", "string")),
    }
    if prop.get("enum"):
        feats.add("has_enum")
        if low in enum:
            feats.add("tok_in_enum")
    if low == name_low or name_low.endswith(low) or low.endswith(name_low):
        feats.add("tok_matches_param_name")
    for dtok in re.findall(r"\w+", desc)[:10]:
        if dtok == low:
            feats.add("tok_in_param_desc")
            break
    if any(low == e for e in enum):
        feats.add("tok_equals_enum_value")
    # value-shape interactions: digits for numeric-ish params, capitals for
    # identifier-ish params
    if any(c.isdigit() for c in t) and prop.get("type") in ("integer", "number"):
        feats.add("digit_for_numeric_param")
    if t[:1].isupper() and prop.get("type") == "string":
        feats.add("capital_for_string_param")
    # neighbors (request side)
    for off, tag in ((1, "next"), (2, "next2"), (1, "prev"), (2, "prev2")):
        j = idx + off if tag.startswith("next") else idx - off
        if 0 <= j < n:
            feats.add(f"{tag}=" + tokens[j].lower()[:16])
    # position relative to the parameter name when it appears verbatim
    name_positions = [j for j, tk in enumerate(tokens)
                      if tk.lower() == name_low]
    if name_positions:
        near = min(abs(idx - j) for j in name_positions)
        feats.add(f"near_param_name={min(near, 5)}")
    for i in range(len(low) - 2):
        feats.add("c=" + low[i:i + 3])
    return sorted(feats)


def gold_spans(tokens: list[str], value) -> list[int] | None:
    """Token indices of the gold value inside the request (exact subsequence
    of the value's own tokens — gold is verbatim by construction)."""
    vtokens = PROSE_TOKEN_RE.findall(_fmt(value))
    if not vtokens:
        return None
    vl = [v.lower() for v in vtokens]
    tl = [t.lower() for t in tokens]
    for start in range(len(tokens) - len(vl) + 1):
        if tl[start:start + len(vl)] == vl:
            return list(range(start, start + len(vl)))
    return None


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def training_pairs(rows: list[dict]):
    """(feature-lists, labels) over (token, required-param) pairs."""
    X: list[list[str]] = []
    y: list[int] = []
    for row in rows:
        tokens = PROSE_TOKEN_RE.findall(row["request"])
        if not tokens or len(tokens) > 60:
            continue
        props = row["tool"]["parameters"]["properties"]
        required = row["tool"]["parameters"]["required"]
        for param in required:
            span = gold_spans(tokens, row["gold"][param])
            prop = props[param]
            for idx in range(len(tokens)):
                X.append(pair_features(tokens, idx, param, prop))
                y.append(1 if span and idx in span else 0)
    return X, np.asarray(y, dtype=np.int64)


def _score_split(clf, rows, emit_floor, extend_floor):
    exact = emitted = wrong = filled_rows = 0
    for row in rows:
        filled, missing = decode_row(clf, row, emit_floor, extend_floor)
        gold = row["gold"]
        for k, v in (filled or {}).items():
            emitted += 1
            if k not in gold or gold[k] != v:
                wrong += 1
        if filled:
            filled_rows += 1
        if (filled and not missing
                and all(gold.get(k) == v for k, v in filled.items())
                and set(filled) == set(gold)):
            exact += 1
    n = len(rows)
    return {
        "n": n,
        "exact_fill": round(exact / n, 4),
        "invented_rate": round(wrong / emitted, 4) if emitted else 0.0,
        "rows_emitting": filled_rows,
    }


def decode_row(clf, row: dict, emit_floor: float, extend_floor: float):
    """Decode via the RUNTIME implementation (single source of truth) with
    the candidate thresholds configured. Returns (filled, missing)."""
    import mivia_small_json.span_fill as sf

    sf.configure_spanfill_floors(emit_floor, extend_floor)
    real_clf = sf.load_spanfill_extractor()
    # The trainer evaluates candidate thresholds against the in-memory clf;
    # swap it in for the duration (process-local, restored after).
    original = real_clf
    sf._SPANFILL_CLF = clf
    try:
        filled, missing = sf.learned_fill(row["tool"], row["request"])
    finally:
        sf._SPANFILL_CLF = original
    return filled, missing


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]



def train() -> int:
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import skops.io as skio

    train_rows = _rows(TRAIN)
    dev_rows = _rows(DEV)
    print(f"train {len(train_rows)} dev {len(dev_rows)}")
    X, y = training_pairs(train_rows)
    print(f"pairs: {len(X)} (positives {int(y.sum())})")
    clf = Pipeline([
        ("hash", FeatureHasher(n_features=2 ** 18, input_type="string")),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)),
    ])
    clf.fit(X, y)

    best = None
    for emit_floor in (0.3, 0.4, 0.5, 0.6, 0.7):
        for extend_floor in (0.15, 0.25, 0.35):
            m = _score_split(clf, dev_rows, emit_floor, extend_floor)
            if m["invented_rate"] > 0.10:
                continue
            key = m["exact_fill"]
            if best is None or key > best[0]:
                best = (key, emit_floor, extend_floor, m)
    if best is None:  # nothing met the dev invented budget: lowest rate wins
        for emit_floor in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for extend_floor in (0.15, 0.25, 0.35):
                m = _score_split(clf, dev_rows, emit_floor, extend_floor)
                if best is None or m["invented_rate"] < best[3]["invented_rate"]:
                    best = (m["exact_fill"], emit_floor, extend_floor, m)
    _, emit_floor, extend_floor, dev_metrics = best
    print("dev-chosen:", json.dumps(
        {"emit_floor": emit_floor, "extend_floor": extend_floor,
         **dev_metrics}))

    skio.dump(clf, ARTIFACT)
    sha = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    (ARTIFACT.parent / (ARTIFACT.stem + ".sha256")).write_text(sha + "\n")
    META.write_text(json.dumps({
        "seed": SEED,
        "n_train": len(train_rows),
        "n_pairs": int(len(y)),
        "positives": int(y.sum()),
        "emit_floor": emit_floor,
        "extend_floor": extend_floor,
        "dev_metrics": dev_metrics,
        "corpus_seed": 20261009,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": sha,
        "note": "token x param pair scorer; deterministic LR (single train, "
                "both-seed deviation recorded in the spec)",
    }, indent=1) + "\n")
    print(f"wrote {ARTIFACT.name} sha {sha[:16]}... + trainmeta")
    return 0


def evaluate() -> int:
    """ONE-SHOT on the frozen reserve — through the SHIPPED path: the
    verified loader and the meta-bound floors (a previous run measured an
    unverified backdoor configuration; recorded in the spec RESULT)."""
    meta = json.loads(META.read_text())
    rsha = hashlib.sha256(RESERVE.read_bytes()).hexdigest()
    import re as _re

    m = _re.search(r"reserve sha256: `([0-9a-f]+)`",
                   (ROOT / "data" / "real" / "spanfill" / "PROVENANCE.md")
                   .read_text())
    assert m and m.group(1) == rsha, "reserve drift — eval refused"

    from mivia_small_json import span_fill as sf

    clf = sf.load_spanfill_extractor()
    assert clf is not None, (
        "spanfill extractor refused to load (sidecar/integrity) — eval "
        "refused rather than measured a degraded path")
    emit_floor = float(meta["emit_floor"])
    extend_floor = float(meta["extend_floor"])
    sf.configure_spanfill_floors(emit_floor, extend_floor)

    rows = _rows(RESERVE)
    res = {"n": len(rows), "exact_fill": 0.0, "invented_rate": 0.0,
           "rows_emitting": 0}
    exact = emitted = wrong = filled_rows = 0
    per_family = {"reserve-only-templates": [0, 0, 0], "hash-slice": [0, 0, 0]}
    for row in rows:
        filled, missing = sf.learned_fill(row["tool"], row["request"])
        gold = row["gold"]
        for k, v in (filled or {}).items():
            emitted += 1
            if k not in gold or gold[k] != v:
                wrong += 1
        if filled:
            filled_rows += 1
        ok = (filled and not missing
              and all(gold.get(k) == v for k, v in filled.items())
              and set(filled) == set(gold))
        exact += 1 if ok else 0
        fam = "reserve-only-templates" if row.get("reserve_only") \
            else "hash-slice"
        per_family[fam][0] += 1
        per_family[fam][1] += 1 if ok else 0
        per_family[fam][2] += len(filled or {})
    res = {
        "n": len(rows),
        "exact_fill": round(exact / len(rows), 4),
        "invented_rate": round(wrong / emitted, 4) if emitted else 0.0,
        "rows_emitting": filled_rows,
    }
    # baseline: shipped span_fill on the same rows (string-param tools)
    base_exact = 0
    for row in rows:
        filled, _ = sf.fill_arguments(row["tool"], row["request"])
        if filled and all(row["gold"].get(k) == v for k, v in filled.items()) \
                and set(filled) == set(row["gold"]):
            base_exact += 1
    result = {
        "verdict": None,
        "bars": {
            "exact_fill>=0.50": res["exact_fill"] >= 0.50,
            "invented<=0.10": res["invented_rate"] <= 0.10,
        },
        "learned": res,
        "per_family": {k: {"n": v[0], "exact": v[1],
                            "emitted_values": v[2]}
                        for k, v in per_family.items()},
        "baseline_exact": round(base_exact / len(rows), 4),
        "reserve_rows": len(rows),
        "emit_floor": emit_floor,
        "extend_floor": extend_floor,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }
    result["verdict"] = "PASS" if all(result["bars"].values()) else "FAIL"
    print(json.dumps(result, indent=1))
    out = ROOT / "reports" / "spanfill-oneshot.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"written {out}")
    return 0 if result["verdict"] == "PASS" else 1





def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if args.train:
        return train()
    if args.eval:
        return evaluate()
    parser.error("choose --train / --eval")
    return 2


if __name__ == "__main__":
    sys.exit(main())
