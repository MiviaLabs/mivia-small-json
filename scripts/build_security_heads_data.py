#!/usr/bin/env python3
"""Arm 3: prepare train/eval files for the three security-severity heads.

All splits are frozen here, BEFORE any training:
- security head: train/dev from the temporal-train side of the GHSA corpus
  ({text, gold, family:"severity"} — the shipped ordinal-trainer shape);
  the temporal reserve stays untouched (its own file).
- is_security gate: hashed-LR classifier slices — positives from GHSA,
  negatives from the in-domain operational corpora (loghub + issue train,
  both training-side files), 50:50 eval slice n>=400 frozen.
- de head: de priority rows from the in-repo 20k CSV mapped to bands
  {0,1,2} (band 3 never fabricated), hash-disjoint reserve n>=300.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GHSA = ROOT / "data" / "incoming" / "external" / "ghsa" / "advisories.jsonl"
SEC = ROOT / "data" / "real" / "grade-security"
DE_CSV = (ROOT / "data" / "incoming" / "external" / "incident-severity"
          / "tickets-multi-lang-20k.csv")
SEED = 42


def _norm(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text.strip().lower())


def main() -> int:
    rng = random.Random(SEED)
    rows = [json.loads(line) for line in GHSA.read_text().splitlines()
            if line.strip()]
    dates = sorted((r["published"][:10] for r in rows), reverse=True)
    cutoff = (datetime.fromisoformat(dates[0]).date()
              - __import__("datetime").timedelta(days=14)).isoformat()
    temporal_train = [r for r in rows if r["published"][:10] <= cutoff]

    # security head: train/dev (5% dev) — the ordinal-trainer shape
    seen = set()
    head_rows = []
    for r in temporal_train:
        key = _norm(r["text"])
        if key in seen:
            continue
        seen.add(key)
        head_rows.append({"text": r["text"], "gold": r["band"],
                          "family": "severity"})
    dev = [r for r in head_rows if int(hashlib.sha256(
        _norm(r["text"]).encode()).hexdigest(), 16) % 100 < 5]
    train = [r for r in head_rows if r not in dev]
    with (SEC / "train.jsonl").open("w") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (SEC / "dev.jsonl").open("w") as fh:
        for r in dev:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"security head: train {len(train)} dev {len(dev)}")

    # is_security gate: positives from the same temporal-train pool,
    # negatives from operational training corpora
    neg_pool = []
    for rel in ("data/real/log_event/train.jsonl",
                "data/real/issue/train.jsonl"):
        for line in (ROOT / rel).read_text(encoding="utf-8",
                                           errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            text = r.get("raw") or r.get("text") or ""
            if isinstance(text, str) and len(text) >= 40:
                neg_pool.append(_norm(text))
    neg_pool = sorted(set(neg_pool))
    rng.shuffle(neg_pool)
    pos_all = sorted({_norm(r["text"]) for r in temporal_train})
    rng.shuffle(pos_all)
    gate_train = ([{"text": t, "label": 1} for t in pos_all[:20000]]
                  + [{"text": t, "label": 0} for t in neg_pool[:20000]])
    gate_eval = ([{"text": t, "label": 1} for t in pos_all[20000:21000]]
                 + [{"text": t, "label": 0} for t in neg_pool[20000:21000]])
    rng.shuffle(gate_train)
    rng.shuffle(gate_eval)
    with (SEC / "gate-train.jsonl").open("w") as fh:
        for r in gate_train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (SEC / "gate-eval.jsonl").open("w") as fh:
        for r in gate_eval:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"gate: train {len(gate_train)} eval {len(gate_eval)}")

    # de severity: priority low/medium/high -> bands {0,1,2}
    de_rows = []
    with DE_CSV.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("language") != "de":
                continue
            prio = {"low": 0, "medium": 1, "high": 2}.get(
                (row.get("priority") or "").strip().lower())
            body = (row.get("body") or "").strip()
            if prio is None or len(body) < 40:
                continue
            key = _norm(body)
            if key in seen:
                continue
            seen.add(key)
            de_rows.append({"text": body, "gold": prio, "family": "severity"})
    de_reserve = [r for r in de_rows
                  if int(hashlib.sha256(
                      _norm(r["text"]).encode()).hexdigest(), 16) % 100 < 5]
    de_train = [r for r in de_rows if r not in de_reserve]
    with (SEC / "de-train.jsonl").open("w") as fh:
        for r in de_train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (SEC / "de-reserve.jsonl").open("w") as fh:
        for r in de_reserve:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    dsha = hashlib.sha256((SEC / "de-reserve.jsonl").read_bytes()).hexdigest()
    print(f"de head: train {len(de_train)} reserve {len(de_reserve)} "
          f"sha {dsha[:12]}...")

    prov = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "temporal_cutoff": cutoff,
        "security_head": {"train": len(train), "dev": len(dev)},
        "gate": {"train": len(gate_train), "eval": len(gate_eval)},
        "de_head": {"train": len(de_train), "reserve": len(de_reserve),
                    "reserve_sha256": dsha},
        "notes": "all splits frozen before training; gate negatives are "
                 "training-corpus operational text; de rows map priority "
                 "to bands {0,1,2} only",
    }
    (SEC / "splits.json").write_text(json.dumps(prov, indent=1) + "\n")
    print("frozen:", json.dumps(prov, indent=1)[:200], "...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
