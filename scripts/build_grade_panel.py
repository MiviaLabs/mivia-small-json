#!/usr/bin/env python3
"""Build the frozen grade register panel (ticket 03).

Ticket 03 of result-quality-improvement The novel-register
gate was n=40, where p-hat=0.55 has an SE of ~7.9pp and a Wilson CI of
~[40%, 69%], so a ship/falsify verdict sits inside the noise. This assembles a
larger, stratified, immutable panel:

- ``novel_register``  — hand-written business/SRE severity phrasings, never
  trained on (the existing 40 plus 30 new).
- ``boundary``        — hand-written adjacent-tier boundary lines.
- ``template_disjoint`` — LogHub dev rows from the held-out systems
  (sha256(name) % 4 == 0), which the grade artifact never trained on.

Rows are committed to ``data/heldout/grade-register-panel.jsonl`` with a gold
tier index and a stratum. The hash ban is checked against the grade training
corpus before writing.

Deterministic. Read-only apart from the panel file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from bench_zero_shot_primitives import NOVEL_REGISTER_GRADE_CASES  # noqa: E402

LOGHUB = ROOT / "data" / "incoming" / "loghub-upstream"
PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
CORPUS = ROOT / "data" / "real" / "grade"

# New novel-register rows (business / SRE phrasing, no level tokens).
NOVEL_NEW = [
    (0, "The nightly reconciliation finished on schedule and matched the ledger"),
    (0, "All regional caches warmed successfully during the maintenance window"),
    (0, "The quarterly report was published to the finance portal as planned"),
    (0, "Batch settlement completed and every transaction balanced"),
    (0, "The support team closed the remaining tickets before the release freeze"),
    (0, "Data synchronisation between the two clusters is running normally"),
    (0, "The new onboarding flow was enabled for the pilot cohort"),
    (0, "Backups for the primary database finished within the expected window"),
    (0, "The pricing service returned results within the agreed latency budget"),
    (0, "Certificate rotation for the public endpoints completed without incident"),
    (1, "The settlement job is running slower than usual and may miss its window"),
    (1, "Replica lag on the reporting cluster is climbing and needs watching"),
    (1, "Two payment retries failed before succeeding on the third attempt"),
    (1, "The search index is stale by about an hour and needs a refresh"),
    (1, "Memory pressure on the queue workers is rising under peak load"),
    (1, "A small number of webhook deliveries are being retried repeatedly"),
    (1, "The nightly export finished late and downstream dashboards are behind"),
    (1, "Disk usage on the logging volume is approaching the soft threshold"),
    (1, "A background compaction is competing with live traffic for IO"),
    (1, "The partner feed arrived with malformed rows that were skipped"),
    (2, "The order service is returning errors for a subset of checkout requests"),
    (2, "Payment capture failed for several carts and customers were not charged"),
    (2, "The inventory synchronisation job aborted before committing its changes"),
    (2, "Authentication is failing for one tenant after the credential rotation"),
    (2, "A schema migration did not apply and the reporting pipeline is broken"),
    (2, "The message broker rejected a batch of events and they were dropped"),
    (2, "Several invoices were generated with the wrong tax jurisdiction"),
    (2, "The reconciliation job could not reach the ledger and gave up"),
    (2, "File uploads above a certain size are failing with a storage error"),
    (2, "The recommendation service is timing out and serving stale results"),
    (3, "The primary ledger database is unrecoverable and we are declaring a disaster"),
    (3, "A region-wide outage has taken down checkout for every customer"),
    (3, "The encryption keys were destroyed and the data cannot be decrypted"),
    (3, "A bad deploy corrupted production state and rollback is not possible"),
    (3, "The payment provider is down and no transactions can be processed"),
    (3, "A ransomware event has encrypted the primary and replica file stores"),
    (3, "Total loss of the event log means the audit trail cannot be reconstructed"),
    (3, "The core switch failed and the data centre is unreachable"),
    (3, "Customer records were deleted in bulk by a runaway job"),
    (3, "The identity provider is offline and nobody can authenticate"),
]

# Adjacent-tier boundary rows: the tier is genuinely arguable.
BOUNDARY = [
    (0, "The queue is draining more slowly than usual but still within budget"),
    (1, "The queue is draining more slowly than usual and may breach its target"),
    (1, "The cache miss rate ticked up after the deploy but is not yet a problem"),
    (2, "The cache miss rate spiked after the deploy and requests are failing"),
    (1, "One replica fell behind and caught up on its own within a minute"),
    (2, "One replica fell behind and never caught up, so reads are stale"),
    (2, "The batch job failed after retrying but the data can be replayed"),
    (3, "The batch job failed and the source data was already purged"),
    (1, "A single customer reported slow checkout and the issue resolved itself"),
    (2, "Every customer is seeing slow checkout and the issue persists"),
    (2, "The region is degraded and traffic has shifted to the secondary"),
    (3, "Every region is down and there is no secondary to fail over to"),
    (0, "The deploy completed and no alerts fired in the first hour"),
    (1, "The deploy completed but the error budget burn is slightly elevated"),
    (2, "The deploy completed and the error budget burn is exhausting fast"),
    (3, "The deploy completed and it has taken the whole platform offline"),
    (1, "Latency at the ninety-ninth percentile is above target for one endpoint"),
    (2, "Latency at the median is above target for every endpoint"),
    (1, "The disk is filling faster than expected but has days of headroom"),
    (2, "The disk is filling faster than expected and has hours of headroom"),
    (0, "A routine failover test ran and the secondary took over cleanly"),
    (1, "A routine failover test ran and the secondary took over slowly"),
    (2, "A failover was triggered by the primary becoming unhealthy"),
    (3, "A failover failed and the service has no healthy instance"),
    (1, "Retries are elevated but still below the alerting threshold"),
    (2, "Retries have crossed the alerting threshold and are still climbing"),
    (1, "A non-critical dependency is degraded and the fallback is serving"),
    (2, "A critical dependency is degraded and there is no fallback"),
    (2, "The backup job failed and will retry on the next cycle"),
    (3, "The backup job failed and there is no usable backup"),
]


def _held_out(name: str) -> bool:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % 4 == 0


def _template_disjoint_rows(limit: int) -> list[dict]:
    """LogHub dev severity rows from the systems the artifact never trained on.

    Balanced across tiers: taking the first rows of the first system yields a
    degenerate stratum (all of Apache's leading rows are ERROR), which makes
    the stratum a single-tier probe rather than a stratified one.
    """
    systems = sorted(
        name for name in (p.name for p in LOGHUB.iterdir() if p.is_dir())
        if _held_out(name) and (LOGHUB / name / f"{name}_2k.log_structured.csv").is_file()
    )
    by_tier: dict[int, list[dict]] = {}
    for system in systems:
        path = LOGHUB / system / f"{system}_2k.log_structured.csv"
        for row in csv.DictReader(path.open(encoding="utf-8")):
            level = (row.get("Level") or "").strip().upper()
            tier = {"INFO": 0, "WARN": 1, "ERROR": 2, "FATAL": 3}.get(level)
            content = (row.get("Content") or "").strip()
            if tier is None or not content:
                continue
            by_tier.setdefault(tier, []).append(
                {"text": content, "gold": tier, "stratum": "template_disjoint",
                 "source": f"loghub-{system}"})
    per_tier = max(1, limit // 4)
    out: list[dict] = []
    for tier in sorted(by_tier):
        # Interleave by source so one system cannot dominate a tier: taking the
        # first rows in sorted-system order repeats the rejected first build's
        # failure mode one level down.
        by_source: dict[str, list[dict]] = {}
        for row in by_tier[tier]:
            by_source.setdefault(row["source"], []).append(row)
        picked: list[dict] = []
        depth = 0
        while len(picked) < per_tier:
            added = False
            for source in sorted(by_source):
                if depth < len(by_source[source]):
                    picked.append(by_source[source][depth])
                    added = True
                    if len(picked) >= per_tier:
                        break
            if not added:
                break
            depth += 1
        out.extend(picked)
    return out


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _training_hashes() -> set[str]:
    hashes = set()
    if not CORPUS.is_dir():
        return hashes
    for path in CORPUS.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("raw") or ""
            if text:
                hashes.add(hashlib.sha256(_normalize(text).encode()).hexdigest())
    return hashes


def main() -> int:
    rows: list[dict] = []
    for text, gold in NOVEL_REGISTER_GRADE_CASES:
        rows.append({"text": text, "gold": gold, "stratum": "novel_register",
                     "source": "frozen-40"})
    for gold, text in NOVEL_NEW:
        rows.append({"text": text, "gold": gold, "stratum": "novel_register",
                     "source": "hand-written"})
    for gold, text in BOUNDARY:
        rows.append({"text": text, "gold": gold, "stratum": "boundary",
                     "source": "hand-written"})
    rows.extend(_template_disjoint_rows(140))

    banned = _training_hashes()
    leaked = [r for r in rows
              if hashlib.sha256(_normalize(r["text"]).encode()).hexdigest() in banned]
    if leaked:
        print(f"hash ban violated by {len(leaked)} rows, e.g. {leaked[0]['text'][:60]!r}",
              file=sys.stderr)
        return 1

    # Deterministic order: stratum, then text.
    rows.sort(key=lambda r: (r["stratum"], r["gold"], r["text"]))
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    PANEL.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
                     encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
    print(f"panel n={len(rows)} {counts}")
    print(f"written to {PANEL.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
