#!/usr/bin/env python3
"""Build the frozen business-phrasing severity eval slice (Track C, step 1).

60 hand-written business/SRE phrasings (15 per tier, no explicit level
tokens), reserved BEFORE any teacher generation. Hash-disjointness against
the grade train/dev corpus, the frozen register panel, and the guard's
pairs.jsonl is checked before writing; the slice carries a reservation
record (timestamp + sha256) proving it predates any teacher-labelled pool.

Writes (grade-owned data + provenance only):
- data/real/grade/business_slice.jsonl
- data/real/grade/business_slice.reserve.json
- provenance appendix in data/real/grade/PROVENANCE.md (idempotent)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SLICE = ROOT / "data" / "real" / "grade" / "business_slice.jsonl"
RESERVE = ROOT / "data" / "real" / "grade" / "business_slice.reserve.json"
PROV = ROOT / "data" / "real" / "grade" / "PROVENANCE.md"
GRADE_CORPUS = ROOT / "data" / "real" / "grade"
PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"

# Hand-written gold rows: (tier, text). 0 info, 1 warning, 2 error, 3 critical.
ROWS: list[tuple[int, str]] = [
    (0, "The month-end close completed on Friday with every account reconciled"),
    (0, "The loyalty points payout ran overnight and balances updated correctly"),
    (0, "The board pack was distributed to directors ahead of the meeting"),
    (0, "The new refund policy page went live and the help centre links resolve"),
    (0, "The graduate intake finished induction and all logins were provisioned"),
    (0, "The fleet tracking dashboard refreshed cleanly through the holiday peak"),
    (0, "The annual audit wrapped up with no findings against the platform team"),
    (0, "The customer advisory webinar ran to a full house with no tech issues"),
    (0, "The lease renewal for the downtown office was signed and filed"),
    (0, "The holiday rota was published and every shift has cover"),
    (0, "The procurement review approved all three shortlisted vendors"),
    (0, "The press release went out on time and coverage has been positive"),
    (0, "The internship programme converted four trainees to full-time offers"),
    (0, "The energy contract renewal locked in rates below last year's"),
    (0, "The archive migration moved two million records with zero checksum errors"),
    (1, "Checkout responses slowed during the lunch rush and engineers are watching the dashboards"),  # noqa: E501
    (1, "Storage on the analytics cluster passed seventy percent and a cleanup job is scheduled"),
    (1, "The card processor reports sporadic timeouts that retries are absorbing for now"),
    (1, "The search service has burned half its error budget with a week left in the quarter"),
    (1, "The standby region lags production by one release and catches up overnight"),
    (1, "Refund volumes rose after the fare change and the trend is under review"),
    (1, "The auditor data export is running late and may miss the morning handover"),
    (1, "Cache hits dipped after the release and pages load a beat slower"),
    (1, "The night-shift pager fired once for a restart that resolved itself"),
    (1, "A stalled supplier feed delayed invoice approvals but the backlog is clearing"),
    (1, "The app's store rating slipped under four stars this week"),
    (1, "Database connections brush the pool ceiling at peak times"),
    (1, "A hardware vendor flagged a two-day slip on the server delivery"),
    (1, "The failover exercise surfaced an outdated runbook page to rewrite"),
    (1, "Renewal emails are landing in spam folders for one major provider"),
    (2, "The checkout page returned errors for twenty minutes and orders were lost"),
    (2, "The primary database is down and failover did not trigger"),
    (2, "Customer billing ran twice and duplicate charges need reversing"),
    (2, "The deployment corrupted the product catalogue and the storefront shows wrong prices"),
    (2, "Password resets are not being delivered and locked-out users cannot sign in"),
    (2, "The warehouse feed stopped syncing and shipments are piling up"),
    (2, "Last night's backup is unusable and the retention window closes tomorrow"),
    (2, "The fraud filter is blocking legitimate cards and sales are dropping"),
    (2, "The API returns stale account balances after the ledger migration"),
    (2, "Email receipts have not gone out since Tuesday and support is swamped"),
    (2, "The payroll file was rejected by the bank and salaries will be late"),
    (2, "The recommendation engine serves empty results after the model rollout"),
    (2, "Appointment reminders failed to send and the clinic phones are jammed"),
    (2, "The tax report generator produces wrong totals for cross-border orders"),
    (2, "The single sign-on provider is rejecting logins for the whole company"),
    (3, "The production database cluster lost all replicas and customer data is at risk"),
    (3, "A security breach exposed customer payment details and regulators have been notified"),
    (3, "The trading platform is offline during market hours with millions at stake"),
    (3, "Ransomware encrypted the file shares and the note demands payment by Friday"),
    (3, "The data centre fire took down every service and the recovery site is not responding"),
    (3, "A faulty migration wiped the audit trail needed for next week's compliance review"),
    (3, "The payment processor leaked card numbers and the network ordered a shutdown"),
    (3, "The hospital scheduling system is down and emergency procedures are on paper"),
    (3, "The airline booking system collapsed on the busiest travel day of the year"),
    (3, "Customer personal data was published on a public forum by a former contractor"),
    (3, "The power grid control interface is unreachable and operators are blind"),
    (3, "The bank's core ledger shows inconsistent balances with no known good copy"),
    (3, "A supply-chain attack backdoored the release shipped to all customers"),
    (3, "The emergency alert system failed silently during the severe weather event"),
    (3, "All backups including offsite copies were found corrupted after the outage"),
]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def nhash(text: str) -> str:
    return hashlib.sha256(norm(text).encode()).hexdigest()


def corpus_hashes() -> set[str]:
    out: set[str] = set()
    for name in ("train.jsonl", "dev.jsonl"):
        p = GRADE_CORPUS / name
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.add(nhash(json.loads(line)["text"]))
                except (KeyError, TypeError):
                    pass
    for line in PANEL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(nhash(json.loads(line)["text"]))
    if PAIRS.is_file():
        for line in PAIRS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    for key in ("request", "text", "prompt"):
                        if isinstance(row.get(key), str):
                            out.add(nhash(row[key]))
                except (KeyError, TypeError):
                    pass
    return out


def main() -> int:
    assert len(ROWS) == 60, f"expected 60 rows, got {len(ROWS)}"
    from collections import Counter
    assert Counter(g for g, _ in ROWS) == {0: 15, 1: 15, 2: 15, 3: 15}
    banned = re.compile(r"\b(INFO|WARNING|WARN|ERROR|CRITICAL|DEBUG|FATAL)\b")
    for gold, text in ROWS:
        assert 0 <= gold <= 3
        assert not banned.search(text), f"level token in slice: {text!r}"
    seen = {nhash(t) for _, t in ROWS}
    assert len(seen) == 60, "duplicate rows inside the slice"
    banned_hashes = corpus_hashes()
    overlap = [t for _, t in ROWS if nhash(t) in banned_hashes]
    if overlap:
        print(f"REFUSED: {len(overlap)} slice rows collide with frozen sets:", file=sys.stderr)
        for t in overlap:
            print(f"  {t}", file=sys.stderr)
        return 1
    payload = "\n".join(
        json.dumps({"text": t, "gold": g, "source": "hand-written",
                    "stratum": "business"})
        for g, t in ROWS) + "\n"
    SLICE.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    RESERVE.write_text(json.dumps({
        "slice": SLICE.name, "n": 60, "sha256": digest,
        "reserved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "frozen BEFORE any teacher generation; train pools must be "
                "hash-disjoint from this digest",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SLICE} (n=60, sha256 {digest[:16]})")
    print(f"wrote {RESERVE}")
    entry = ("| business-slice | `data/real/grade/business_slice.jsonl`, 60 hand-written "
             "business/SRE phrasings (15/tier), frozen eval only — never trained on "
             f"(sha256[:16] `{digest[:16]}`) | original content, human gold labels | 60 |\n")
    prov = PROV.read_text(encoding="utf-8")
    if "business-slice" not in prov:
        if not prov.endswith("\n"):
            prov += "\n"
        PROV.write_text(prov + entry, encoding="utf-8")
        print("provenance entry appended")
    else:
        print("provenance entry already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
