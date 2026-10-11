#!/usr/bin/env python3
"""Generate severity rows targeted at the model's measured failure clusters.

Ticket 14 of `result-quality-improvement`. The failure
report on the frozen panel (`scripts/cluster_grade_failures.py`) shows two
dominant clusters, not uniform noise:

- business/SRE rows predicted one to three tiers too high: 17 INFO and 14
  WARN rows land on CRITICAL;
- WARNING on unseen systems, confused both ways (18 to INFO, 17 to ERROR).

So the families here are written against those clusters, not as more of the
same register: routine business messages that *mention* alarming words, mild
warnings, and the WARN band on log-shaped text. Each row is verified against
the frozen panel and the training corpus by the hash ban before it is written.

Writes `data/incoming/grade-targeted.jsonl`. Deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "data" / "incoming" / "grade-targeted.jsonl"
PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
CORPUS = ROOT / "data" / "real" / "grade"

# (family_id, register, tier, template). Tier: 0 INFO, 1 WARN, 2 ERROR, 3 CRIT.
FAMILIES: list[tuple[str, str, int, str]] = [
    # Cluster 1: routine business messages that mention alarming words.
    ("tg-biz-i-1", "business", 0,
     "The nightly job retried once after a transient timeout and then completed"),
    ("tg-biz-i-2", "business", 0,
     "One replica briefly lagged during the rollout and caught up without help"),
    ("tg-biz-i-3", "business", 0,
     "The rejected login from {user} was expected and the account stayed locked"),
    ("tg-biz-i-4", "business", 0,
     "The disk usage alert cleared by itself once log rotation ran"),
    ("tg-biz-i-5", "business", 0,
     "The backup verification found no corruption and every checksum matched"),
    ("tg-biz-i-6", "business", 0,
     "A few requests were rate limited and the client backed off correctly"),
    ("tg-biz-i-7", "business", 0,
     "The certificate renewal finished before expiry and no connection dropped"),
    ("tg-biz-i-8", "business", 0,
     "The migration finished with zero rows skipped and the totals reconciled"),
    # Cluster 1b: SRE phrasing for the same routine events.
    ("tg-sre-i-1", "sre", 0,
     "Error budget burn is flat and the rollout stayed within its window"),
    ("tg-sre-i-2", "sre", 0,
     "The failover drill ran on schedule and the secondary took over cleanly"),
    ("tg-sre-i-3", "sre", 0,
     "Paging fired once for {svc} and cleared before anyone acknowledged"),
    ("tg-sre-i-4", "sre", 0,
     "The incident review closed with no follow-up actions outstanding"),
    # Cluster 2: mild, recoverable warnings.
    ("tg-biz-w-1", "business", 1,
     "The retry budget for the settlement job is half spent and the run is late"),
    ("tg-biz-w-2", "business", 1,
     "One region is serving from the secondary after a brief failover"),
    ("tg-biz-w-3", "business", 1,
     "Replica lag is above target but still below the alerting threshold"),
    ("tg-biz-w-4", "business", 1,
     "The nightly export finished late and dashboards are an hour behind"),
    ("tg-biz-w-5", "business", 1,
     "A minority of webhook deliveries needed a second attempt"),
    ("tg-biz-w-6", "business", 1,
     "The cache hit rate dropped after the deploy and is inside budget"),
    ("tg-sre-w-1", "sre", 1,
     "The error budget is burning faster than planned and needs watching"),
    ("tg-sre-w-2", "sre", 1,
     "Two of the three replicas are healthy and the third is catching up"),
    ("tg-sre-w-3", "sre", 1,
     "Latency at the tail is over target for one endpoint only"),
    ("tg-sre-w-4", "sre", 1,
     "The dependency is degraded and the fallback is serving traffic"),
    # Cluster 3: the WARN band on log-shaped text.
    ("tg-log-w-1", "applog", 1,
     "{t} WARN  {svc} request took {ms}ms, over the {n}ms target but under timeout"),
    ("tg-log-w-2", "applog", 1,
     "{t} WARN  {svc} connection pool at {m}% of capacity"),
    ("tg-log-w-3", "syslog", 1,
     "{host} {svc}[{n}]: retrying upstream call after a transient reset"),
    ("tg-log-w-4", "syslog", 1,
     "{host} {svc}[{n}]: slow query took {ms}ms and was served from cache"),
]

_VALUES = {
    "host": ["web-01", "web-02", "db-primary", "cache-03", "worker-7", "edge-lon-2"],
    "user": ["alice", "svc-deploy", "backup", "www-data", "maria"],
    "svc": ["nginx", "orders-api", "billing-worker", "auth-svc", "searchd", "postgres"],
    "t": ["2026-10-06T08:14:02Z", "2026-10-06T09:41:55Z", "2026-10-05T22:03:11Z"],
    "ms": ["120", "458", "1", "2031", "88"],
}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _banned() -> set[str]:
    """Normalized hashes of everything the rows must stay away from."""
    banned: set[str] = set()
    paths = [PANEL, *(CORPUS.glob("*.jsonl"))]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("raw") or ""
            if text:
                banned.add(hashlib.sha256(_norm(text).encode()).hexdigest())
    return banned


def build(seed: int, per_family: int) -> tuple[list[dict], int]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for family_id, register, tier, template in FAMILIES:
        for _ in range(per_family):
            vals = {k: rng.choice(v) for k, v in _VALUES.items()}
            vals["n"] = rng.randint(2, 9999)
            vals["m"] = rng.randint(2, 99)
            rows.append({"text": template.format(**vals), "family": "severity",
                         "gold": tier, "source": "targeted",
                         "template": family_id, "register": register})
    banned = _banned()
    kept, dropped = [], 0
    for row in rows:
        if hashlib.sha256(_norm(row["text"]).encode()).hexdigest() in banned:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--per-family", type=int, default=60)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows, dropped = build(args.seed, args.per_family)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
                        encoding="utf-8")
    from collections import Counter

    tiers = Counter(r["gold"] for r in rows)
    print(f"targeted rows: {len(rows)} (hash-ban dropped {dropped}) "
          f"tiers={dict(sorted(tiers.items()))}")
    print(f"written to {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
