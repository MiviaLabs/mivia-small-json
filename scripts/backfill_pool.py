"""Backfill the improvement pool with trusted tier-2 log rows.

One-time, reviewed backfill for the wild log-routing gap (SMAL-8). The
per-cycle class-share quota (``label.CAPS["log_event_share"]``) bounds
autonomous flywheel cycles, where rows of the other classes fill the
rest of the batch. The platform-log export in the inbox is almost all
log lines, so the quota admits about one row per cycle and the router
never gains production-shaped training data.

This script runs the documented collection path end to end and waives
only that per-cycle quota - deliberately, and only when asked for.
Without ``--waive-quota`` it refuses to run. Every other rule is the
standard machinery, never waived: ``collect_all`` (scrub, length guard,
exact/near dedupe, frozen-set hash ban) and tier-2 admission (a rules
vote with normalized margin >= 0.50 for a class whose trust was measured
on labeled development data - today ``log_event`` only). Frozen audit
data is never a source; rows colliding with it are rejected by the hash
ban. Read ``data/pool/PROVENANCE.md`` before waiving the quota.

Usage (repository root, project venv):

    python scripts/backfill_pool.py --waive-quota
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def backfill(root: Path = ROOT, waive_quota: bool | None = None) -> dict:
    """Collect, weak-label (tier 2 only), and append to confident.jsonl.

    The per-cycle class-share quota is waived only when ``waive_quota``
    is True; the default is None, which keeps the command-line refusal
    (``--waive-quota``) authoritative while letting tests pass an
    explicit choice.
    """
    from mivia_small_json import bootstrap, collect, flywheel, label

    if waive_quota is None:
        print("refusing to run: this script waives the per-cycle "
              "class-share quota on purpose; pass --waive-quota after "
              "reading data/pool/PROVENANCE.md", file=sys.stderr)
        raise SystemExit(2)
    bootstrap.register_example_schemas()
    root = Path(root)
    records, cstats = collect.collect_all(root)
    measure = label.measure_dev_rules(root)
    trust = label._trust(measure)
    admitted: list[dict] = []
    per_class: dict[str, int] = {}
    for rec in records:
        lab = label.label_record(rec["raw"], router=None, trust=trust)
        if lab["tier"] != "confident":
            continue
        admitted.append({"id": rec["id"], "source": rec["source"],
                         "raw": rec["raw"], "schema": lab["schema"],
                         "reason": lab["reason"]})
        per_class[lab["schema"]] = per_class.get(lab["schema"], 0) + 1
    pool = root / "data" / "pool" / "confident.jsonl"
    appended = flywheel._append_confident(admitted, pool)
    total = len(flywheel._read_jsonl(pool))
    summary = {"collected": cstats, "tier2": per_class, "appended": appended,
               "pool_total": total,
               "trusted_classes": measure["trusted_classes"],
               "quota_waived": bool(waive_quota)}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    backfill(waive_quota="--waive-quota" in sys.argv[1:])
