#!/usr/bin/env python3
"""Train an implicit severity classifier for syslog and daemon logs.

Syslog (Linux, OpenSSH, Mac, Thunderbird) often lacks explicit level tokens
in the raw text. This model trains on message content across LogHub systems
and syslog auth/kernel events to learn implicit severities:
- failure, timeout, denied, unable, refused -> ERROR / WARN
- break-in attempt, invalid user, link down -> WARN
- session opened/closed, accepted, connected, completed -> INFO
- debug, trace, probing -> DEBUG
- fatal, panic, abort -> FATAL

Writes models/implicit_severity.skops (+ sha256 sidecar).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.schemas import LEVEL_ALIASES, LEVELS  # noqa: E402

LOGHUB = ROOT / "data" / "incoming" / "loghub-upstream"
TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")

# Measured generalization floor for the train-only probe on the held-out
# systems. The previous 99.58% headline was the all-data model's score on the
# systems it trained on (a self-report); the honest probe measures 77.5%.
# Re-baselined 2026-10-06 to the measured value.
IMPLICIT_HELD_OUT_BAR = 0.75


def held_out_split(name: str) -> bool:
    """Deterministic, position-independent held-out split.

    A system is held out iff sha256(name) % 4 == 0. The bucket depends
    only on the name, so adding a log system never reshuffles the
    systems already assigned.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % 4 == 0


def canon(level: str) -> str:
    lv = (level or "").strip().upper()
    return LEVEL_ALIASES.get(lv, lv)


def message_features(text: str) -> list[str]:
    """Token unigram, bigram, and char trigram features for message text."""
    low = text.lower()
    tokens = TOKEN_RE.findall(low)
    feats = {"w=" + t for t in tokens if len(t) > 1}
    for i in range(len(tokens) - 1):
        feats.add(f"bi={tokens[i]}_{tokens[i+1]}")
    for i in range(len(tokens) - 2):
        feats.add(f"tri={tokens[i]}_{tokens[i+1]}_{tokens[i+2]}")
    # char ngrams for stem matching
    for i in range(len(low) - 3):
        feats.add("c4=" + low[i : i + 4])
    return sorted(feats)


def load_implicit_training_data() -> list[tuple[str, str, str]]:
    """Return (system, message, level) triples so the held-out split can be
    derived from the system name rather than row position."""
    pairs = []

    # 1. Structured LogHub messages with ground truth levels
    systems_with_levels = [
        "Android", "Apache", "BGL", "HDFS", "Hadoop", "OpenStack",
        "Spark", "Windows", "Zookeeper"
    ]
    for sys_name in systems_with_levels:
        csv_path = LOGHUB / sys_name / f"{sys_name}_2k.log_structured.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            l_key = [k for k in (reader.fieldnames or []) if k.lower() in ("level", "severity")]
            c_key = [k for k in (reader.fieldnames or []) if k.lower() in ("content", "message")]
            if not l_key or not c_key:
                continue
            for r in reader:
                lv_raw = r[l_key[0]].strip()
                c = r[c_key[0]].strip()
                c_lv = canon(lv_raw)
                if c_lv in LEVELS and c:
                    pairs.append((sys_name, c, c_lv))

    # 2. Syslog auth/kernel messages (Linux, OpenSSH, Mac)
    syslog_rules = [
        # Errors
        (r"(?i)\bauthentication failure\b", "ERROR"),
        (r"(?i)\bfailed password\b", "ERROR"),
        (r"(?i)\bfailed login\b", "ERROR"),
        (r"(?i)\bpermission denied\b", "ERROR"),
        (r"(?i)\bconnection refused\b", "ERROR"),
        (r"(?i)\bexited abnormally\b", "ERROR"),
        (r"(?i)\bunable to\b", "ERROR"),
        (r"(?i)\bnetwork is not reachable\b", "ERROR"),
        (r"(?i)\bsegmentation fault\b|\bsegfault\b", "ERROR"),
        (r"(?i)\bcore dump(?:ed)?\b", "ERROR"),
        (r"(?i)\btimed? ?out\b", "ERROR"),
        # Warnings
        (r"(?i)\bbreak-in attempt\b", "WARN"),
        (r"(?i)\binvalid user\b", "WARN"),
        (r"(?i)\billegal user\b", "WARN"),
        (r"(?i)\buser unknown\b", "WARN"),
        (r"(?i)\bdeny\(\d+\)\b", "WARN"),
        (r"(?i)\blink down\b", "WARN"),
        (r"(?i)\bunexpected\b", "WARN"),
        (r"(?i)\bretry(?:ing)?\b", "WARN"),
        # Info
        (r"(?i)\bsession opened\b", "INFO"),
        (r"(?i)\bsession closed\b", "INFO"),
        (r"(?i)\baccepted password\b", "INFO"),
        (r"(?i)\baccepted publickey\b", "INFO"),
        (r"(?i)\bconnection closed\b", "INFO"),
        (r"(?i)\breceived disconnect\b", "INFO"),
        (r"(?i)\blistening on\b", "INFO"),
        (r"(?i)\bstarted\b", "INFO"),
        (r"(?i)\brun-parts\b", "INFO"),
        (r"(?i)\bcalculate completed\b", "INFO"),
        (r"(?i)\bopen through proxy\b", "INFO"),
        # Fatal
        (r"(?i)\bfatal(?: error)?\b", "FATAL"),
        (r"(?i)\bkernel panic\b", "FATAL"),
        (r"(?i)\bout of memory: kill\b", "FATAL"),
    ]

    for sys_name in ("Linux", "OpenSSH", "Mac", "Thunderbird", "HealthApp", "Proxifier"):
        csv_path = LOGHUB / sys_name / f"{sys_name}_2k.log_structured.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            c_key = [k for k in (reader.fieldnames or []) if k.lower() in ("content", "message")]
            if not c_key:
                continue
            for r in reader:
                c = r[c_key[0]].strip()
                if not c:
                    continue
                for pat, lv in syslog_rules:
                    if re.search(pat, c):
                        pairs.append((sys_name, c, lv))
                        break

    return pairs


def main():
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from skops.io import dump

    data = load_implicit_training_data()
    print(f"Loaded {len(data)} implicit severity pairs")
    counts = {}
    for _, _, lv in data:
        counts[lv] = counts.get(lv, 0) + 1
    print("Class distribution:", counts)

    held_out_systems = sorted({s for s, _, _ in data if held_out_split(s)})
    print(f"Held-out systems (sha256(name) % 4 == 0): {held_out_systems}")

    train = [(msg, lv) for system, msg, lv in data if not held_out_split(system)]
    val = [(msg, lv) for system, msg, lv in data if held_out_split(system)]

    X = [message_features(msg) for _, msg, _ in data]
    y = [lv for _, _, lv in data]
    X_train = [message_features(msg) for msg, _ in train]
    y_train = [lv for _, lv in train]
    X_val = [message_features(msg) for msg, _ in val]
    y_val = [lv for _, lv in val]

    # Generalization estimate: fit a probe on the training systems only and
    # score it on the held-out systems. The shipped model below is then fit on
    # every system, so its own score on the held-out systems is a training fit.
    probe = Pipeline([
        ("hasher", FeatureHasher(n_features=2**17, input_type="string")),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
    ])
    probe.fit(X_train, y_train)
    val_acc = probe.score(X_val, y_val)
    print(
        f"generalization estimate (train-only probe) on held-out systems "
        f"{held_out_systems}: accuracy {val_acc:.3%}"
    )
    assert val_acc >= IMPLICIT_HELD_OUT_BAR, (
        f"generalization bar failed: held-out acc {val_acc:.3%} < {IMPLICIT_HELD_OUT_BAR:.0%}"
    )

    # Ship the model fit on every system (strongest on all of them).
    prod_pipe = Pipeline([
        ("hasher", FeatureHasher(n_features=2**17, input_type="string")),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
    ]).fit(X, y)
    prod_pipe.named_steps["clf"].coef_ = prod_pipe.named_steps["clf"].coef_.astype("float32")

    out_path = ROOT / "models" / "implicit_severity.skops"
    dump(prod_pipe, out_path)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    (ROOT / "models" / "implicit_severity.skops.sha256").write_text(digest + "\n", encoding="utf-8")
    (ROOT / "models" / "implicit_severity.skops.trainmeta.json").write_text(
        json.dumps({
            "artifact": "implicit_severity.skops",
            "artifact_sha256": digest,
            "shipped_train_systems": sorted({s for s, _, _ in data}),
            "generalization": {
                "model": "train-only-probe",
                "train_systems": sorted({s for s, _, _ in data if not held_out_split(s)}),
                "held_out_systems": held_out_systems,
                "accuracy": val_acc,
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved implicit severity model to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
