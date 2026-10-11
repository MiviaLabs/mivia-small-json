#!/usr/bin/env python3
"""Train a learned component extractor for log lines.

Uses token classification over hashed features (FeatureHasher +
LogisticRegression, skops persistence, CPU-only).

Ground truth sources:
- 15 LogHub systems (Component / Program columns in _structured.csv).
- LogCat real corpus (data/incoming/logcat-structured.jsonl).

Writes models/component_extractor.skops (+ sha256 sidecar).
"""

import csv
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

LOGHUB = os.path.join(ROOT, "data", "incoming", "loghub-upstream")
INCOMING = os.path.join(ROOT, "data", "incoming")

TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")


def held_out_split(name: str) -> bool:
    """Deterministic, position-independent held-out split.

    A system is held out iff sha256(name) % 4 == 0. The bucket depends
    only on the name, so adding a log system never reshuffles the
    systems already assigned.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % 4 == 0


def tokenize(line: str) -> list[tuple[str, int]]:
    return [(m.group(0), m.start()) for m in TOKEN_RE.finditer(line)]


def shape(tok: str) -> str:
    return re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", re.sub(r"\d", "d", tok)))


def token_features(line: str, tokens: list[tuple[str, int]], i: int) -> list[str]:
    tok = tokens[i][0]
    t = tok.lower()
    feats = {"tok=" + t, "shape=" + shape(tok), "len=" + str(min(len(t), 12))}
    start, end = tokens[i][1], tokens[i][1] + len(tok)
    if start > 0:
        feats.add("before=" + line[start - 1])
    if end < len(line):
        feats.add("after=" + line[end])
    feats.update("c=" + t[j : j + 3] for j in range(max(0, len(t) - 2)))
    if i > 0:
        feats.add("prev=" + tokens[i - 1][0].lower())
        if i > 1:
            feats.add("prev2=" + tokens[i - 2][0].lower())
    if i + 1 < len(tokens):
        feats.add("next=" + tokens[i + 1][0].lower())
        if i + 2 < len(tokens):
            feats.add("next2=" + tokens[i + 2][0].lower())
    return sorted(feats)


def load_component_data():
    systems = sorted(os.listdir(LOGHUB))
    train, test = [], []
    for sys_name in systems:
        csv_path = os.path.join(LOGHUB, sys_name, sys_name + "_2k.log_structured.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            comp_key = [k for k in fields if k.lower() in ("component", "program")]
            if not comp_key:
                continue
            k = comp_key[0]
            log_path = os.path.join(LOGHUB, sys_name, sys_name + "_2k.log")
            lines = [ln.strip() for ln in open(log_path, errors="replace") if ln.strip()]
            bucket = test if held_out_split(sys_name) else train
            for i, row in enumerate(reader):
                if i >= len(lines):
                    break
                comp_gt = row[k].strip()
                raw = lines[i]
                tokens = tokenize(raw)
                gold = None
                if comp_gt:
                    for ti, (tok, _pos) in enumerate(tokens):
                        if (
                            tok.lower() == comp_gt.lower()
                            or comp_gt.lower().startswith(tok.lower())
                        ):
                            gold = ti
                            break
                bucket.append((sys_name, raw, tokens, comp_gt, gold))

    # Add extra LogCat lines to train
    logcat_file = os.path.join(INCOMING, "logcat-structured.jsonl")
    if os.path.isfile(logcat_file):
        for line in open(logcat_file, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            raw = rec["raw"]
            comp_gt = rec.get("component", "").strip()
            tokens = tokenize(raw)
            gold = None
            if comp_gt:
                for ti, (tok, _pos) in enumerate(tokens):
                    if (
                        tok.lower() == comp_gt.lower()
                        or comp_gt.lower().startswith(tok.lower())
                    ):
                        gold = ti
                        break
            train.append(("logcat_real", raw, tokens, comp_gt, gold))

    return train, test


def main():
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from skops.io import dump

    train, test = load_component_data()
    print(f"Loaded component data: train={len(train)}, test={len(test)}")

    hasher = FeatureHasher(n_features=2**18, input_type="string")
    clf = LogisticRegression(max_iter=1000, C=2.0, random_state=42)

    def xy(bucket, is_train):
        X, y, meta = [], [], []
        for sys_name, raw, tokens, comp_gt, gold in bucket:
            for ti in range(len(tokens)):
                feats = token_features(raw, tokens, ti)
                X.append(feats)
                y.append(1 if (is_train and ti == gold) else 0)
            meta.append((sys_name, tokens, comp_gt, gold))
        return X, y, meta

    def evaluate(model):
        """Score a fitted model on the held-out systems.

        Returns (accuracy, per-system [correct, total]).
        """
        X_te, _y_te, meta = xy(test, False)
        proba = model.predict_proba(X_te)[:, 1]

        idx = 0
        correct = total = 0
        per_sys = {}
        for sys_name, tokens, comp_gt, gold in meta:
            n = len(tokens)
            row = proba[idx : idx + n]
            idx += n
            if gold is None or not comp_gt:
                continue
            best = max(range(n), key=lambda i: row[i]) if n else -1
            pred = tokens[best][0] if (n and row[best] >= 0.20) else ""
            total += 1
            d = per_sys.setdefault(sys_name, [0, 0])
            d[1] += 1
            if (
                pred.lower() == comp_gt.lower()
                or pred.lower() in comp_gt.lower()
                or comp_gt.lower() in pred.lower()
            ):
                correct += 1
                d[0] += 1

        acc = correct / total if total else 0.0
        return acc, per_sys

    # Generalization estimate: fit a probe on the training systems only and
    # score it on the held-out systems. The shipped model below is then fit on
    # every system, so its own score on the held-out systems is a training fit.
    X_tr, y_tr, _ = xy(train, True)
    probe = Pipeline([("hasher", hasher), ("clf", clf)]).fit(X_tr, y_tr)

    held_out_systems = sorted({r[0] for r in test})
    acc, per_sys = evaluate(probe)
    print(
        f"generalization estimate (train-only probe) on held-out systems "
        f"{held_out_systems}: component acc {acc:.1%}"
    )
    for sys_name, (c, t) in sorted(per_sys.items()):
        print(f"  {sys_name:12s}: {c}/{t} ({c / t:.1%})")

    assert acc >= 0.85, f"Component extractor failed acceptance bar: {acc:.1%} < 85%"

    # Ship the model fit on every system (strongest on all of them).
    X_all, y_all, _ = xy(train + test, True)
    prod_pipe = Pipeline([
        ("hasher", FeatureHasher(n_features=2**18, input_type="string")),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
    ]).fit(X_all, y_all)

    out_path = os.path.join(ROOT, "models", "component_extractor.skops")
    dump(prod_pipe, out_path)
    digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    with open(out_path + ".sha256", "w") as f:
        f.write(digest + "\n")
    with open(out_path + ".trainmeta.json", "w") as f:
        json.dump({
            "artifact": os.path.basename(out_path),
            "artifact_sha256": digest,
            "shipped_train_systems": sorted({r[0] for r in train + test}),
            "generalization": {
                "model": "train-only-probe",
                "train_systems": sorted({r[0] for r in train}),
                "held_out_systems": held_out_systems,
                "accuracy": acc,
            },
        }, f, indent=2)
    print(f"Saved component extractor to {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
