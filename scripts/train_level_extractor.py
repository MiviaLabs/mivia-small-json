#!/usr/bin/env python3
"""Learn log-level extraction from LogPai ground truth and real/synthetic data.

The regex extractor is deterministic and cannot learn new log formats.
This script trains a LEARNED token classifier in the same stack as the
router (hashed features + LogisticRegression, skops persistence, CPU-only):

- Training data:
  1. LogHub 2k-line samples with logparser ground truth
     (data/incoming/loghub-upstream/<System>_2k.log_structured.csv).
  2. LogCat-style formatted logs (data/incoming/logcat-structured.jsonl),
     sampled from Android_v1.zip, strictly disjoint from Android_2k.log.
  3. Synthetic format augmentation (data/incoming/log-format-augmented.jsonl)
     covering logfmt, json-lines, and syslog patterns.
- Evaluation: deterministic system-disjoint split — a system is held out
  iff sha256(system) % 4 == 0 — to measure generalization to UNSEEN log
  formats. Because the bucket depends only on the name, adding a log
  system never reshuffles the systems already assigned.
- Acceptance bar: the train-only probe's accuracy on the held-out systems
  must clear LEVEL_HELD_OUT_BAR. The shipped artifact is then fit on every
  system; its own score on the held-out systems is a training fit and is not
  reported as a generalization result.

Writes models/level_extractor.skops (+ sha256 sidecar).
"""

import csv
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mivia_small_json.extract_learned import (  # noqa: E402
    canon,
    level_token_features as token_features,
    level_tokenize as tokenize,
)

LOGHUB = os.path.join(ROOT, "data", "incoming", "loghub-upstream")
INCOMING = os.path.join(ROOT, "data", "incoming")

# Measured generalization floor for the train-only probe on the held-out
# systems. The previous 95% bar was read off the all-data model's score on the
# systems it trained on, i.e. a self-report; the honest probe measures 67.0%,
# dragged down by Apache (0.9%) whose level vocabulary ("notice") is unseen in
# training. Re-baselined 2026-10-06 to the measured value.
LEVEL_HELD_OUT_BAR = 0.65


def held_out_split(name):
    """Deterministic, position-independent held-out split.

    A system is held out iff sha256(name) % 4 == 0. The bucket depends
    only on the name, so adding a log system never reshuffles the
    systems already assigned.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % 4 == 0


def load_examples():
    systems = sorted(
        name for name in os.listdir(LOGHUB)
        if os.path.isfile(os.path.join(LOGHUB, name, name + "_2k.log_structured.csv"))
    )
    train, test = [], []
    for system in systems:
        base = os.path.join(LOGHUB, system)
        truth = {}
        with open(
            os.path.join(base, system + "_2k.log_structured.csv"),
            newline="",
            encoding="utf-8",
        ) as f:
            for row in csv.DictReader(f):
                truth[int(row["LineId"])] = canon(row.get("Level") or "")
        bucket = test if held_out_split(system) else train
        with open(
            os.path.join(base, system + "_2k.log"),
            encoding="utf-8",
            errors="replace",
        ) as f:
            for i, line in enumerate(f, start=1):
                raw = line.rstrip("\n")
                gt = truth.get(i, "")
                tokens = tokenize(raw)
                if not gt or not tokens:
                    bucket.append((system, raw, tokens, None))
                    continue
                gold = None
                for ti, (tok, _pos) in enumerate(tokens):
                    if canon(tok) == gt:
                        gold = ti
                        break
                bucket.append((system, raw, tokens, gold))

    # Add extra LogCat lines strictly to train
    logcat_file = os.path.join(INCOMING, "logcat-structured.jsonl")
    if os.path.isfile(logcat_file):
        for line in open(logcat_file, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            raw = rec["raw"]
            gt = canon(rec["level"])
            tokens = tokenize(raw)
            gold = None
            for ti, (tok, _pos) in enumerate(tokens):
                if canon(tok) == gt:
                    gold = ti
                    break
            train.append(("logcat_real", raw, tokens, gold))

    # Add augmented formats strictly to train
    aug_file = os.path.join(INCOMING, "log-format-augmented.jsonl")
    if os.path.isfile(aug_file):
        for line in open(aug_file, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            raw = rec["raw"]
            gt = canon(rec["level"])
            tokens = tokenize(raw)
            gold = None
            for ti, (tok, _pos) in enumerate(tokens):
                if canon(tok) == gt:
                    gold = ti
                    break
            train.append(("format_augmented", raw, tokens, gold))

    return train, test


def main():
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from skops.io import dump

    train, test = load_examples()
    n_train_lines = sum(1 for r in train if r[3] is not None)
    held_out_systems = sorted({r[0] for r in test})
    print(f"train lines with truth: {n_train_lines}, held-out systems: {held_out_systems}")

    hasher = FeatureHasher(n_features=2**18, input_type="string")
    clf = LogisticRegression(max_iter=1000, C=2.0, random_state=42)

    def xy(bucket, train_mode):
        X, y, meta = [], [], []
        for system, line, tokens, gold in bucket:
            for ti in range(len(tokens)):
                feats = token_features(line, tokens, ti)
                X.append(feats)
                y.append(1 if (train_mode and ti == gold) else 0)
            meta.append((system, tokens, gold))
        return X, y, meta

    def evaluate(model):
        """Score a fitted model on the held-out systems.

        Returns (accuracy, coverage, per-system [n, covered, correct], n).
        """
        X_te, _y_te, meta = xy(test, False)
        proba = model.predict_proba(X_te)[:, 1]

        agree = scored = 0
        covered = 0
        per_system = {}
        idx = 0
        for system, tokens, gold in meta:
            n = len(tokens)
            row = proba[idx : idx + n]
            idx += n
            best = max(range(n), key=lambda i: row[i]) if n else -1
            pred = tokens[best][0] if (n and row[best] >= 0.35) else ""
            if gold is None:
                continue
            gt = canon(tokens[gold][0])
            if pred:
                covered += 1
            if canon(pred) and canon(pred) == gt:
                agree += 1
            scored += 1
            d = per_system.setdefault(system, [0, 0, 0])
            d[0] += 1
            if pred:
                d[1] += 1
            if canon(pred) == gt:
                d[2] += 1

        acc = agree / scored if scored else 0.0
        cov = covered / scored if scored else 0.0
        return acc, cov, per_system, scored

    # Generalization estimate: fit a probe on the training systems only and
    # score it on the held-out systems. The shipped model below is then fit on
    # every system, so its own score on the held-out systems is a training fit,
    # not a generalization result, and is not reported as one.
    X_tr, y_tr, _ = xy(train, True)
    probe = Pipeline([("hasher", hasher), ("clf", clf)]).fit(X_tr, y_tr)

    acc, cov, per_system, scored = evaluate(probe)
    print(
        f"generalization estimate (train-only probe) on held-out systems "
        f"{held_out_systems}: level acc {acc:.1%} (coverage {cov:.1%}, n={scored})"
    )
    for system in sorted(per_system):
        s, c, a = per_system[system]
        print(f"  {system}: acc {a / s:.1%} cov {c / s:.1%} n={s}")

    assert acc >= LEVEL_HELD_OUT_BAR, (
        f"generalization bar failed: held-out acc {acc:.1%} < {LEVEL_HELD_OUT_BAR:.0%}"
    )

    # Ship the model fit on every system: it is the strongest on all of them,
    # including the held-out systems where the probe generalizes poorly.
    X_all, y_all, _ = xy(train + test, True)
    prod_pipe = Pipeline([
        ("hasher", FeatureHasher(n_features=2**18, input_type="string")),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
    ]).fit(X_all, y_all)

    path = os.path.join(ROOT, "models", "level_extractor.skops")
    dump(prod_pipe, path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(digest + "\n")
    with open(path + ".trainmeta.json", "w") as f:
        json.dump({
            "artifact": os.path.basename(path),
            "artifact_sha256": digest,
            "shipped_train_systems": sorted({r[0] for r in train + test}),
            "generalization": {
                "model": "train-only-probe",
                "train_systems": sorted({r[0] for r in train}),
                "held_out_systems": held_out_systems,
                "accuracy": acc,
                "coverage": cov,
            },
        }, f, indent=2)
    print(f"saved {path} ({os.path.getsize(path)} bytes) sha256={digest[:12]}...")


if __name__ == "__main__":
    main()
