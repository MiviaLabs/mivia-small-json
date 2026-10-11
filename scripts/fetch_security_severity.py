#!/usr/bin/env python3
"""Arm 3 data: fetch GHSA advisories (CC-BY-4.0) for the security band head.

Fetches published GitHub security advisories from the public
api.github.com/advisories endpoint (no auth needed for the low request
rate; 10-30s pause between pages per the unauthenticated budget), keeps
English descriptions with a severity enum, and splits TEMPORALLY: the
newest `--reserve-days` days are the frozen reserve, everything earlier
is train/dev. Snyk is not a source; Ubuntu CC-BY-SA rows are excluded by
source filter (ghsa only here).

Usage:
    .venv/bin/python scripts/fetch_security_severity.py --pages 250
    .venv/bin/python scripts/fetch_security_severity.py --freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "incoming" / "external" / "ghsa"
OUT_DIR = ROOT / "data" / "real" / "grade-security"
TRAIN = OUT_DIR / "train.jsonl"
RESERVE = OUT_DIR / "reserve-temporal.jsonl"
PROVENANCE = OUT_DIR / "PROVENANCE.md"
RESERVE_DAYS = 14
MIN_LEN = 80


NVD = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _nvd_band(severity: str):
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(severity)


def _fetch(days: int) -> list[dict]:
    """NVD API 2.0 (public domain — better licensing than CC-BY): walk
    pubDate windows backward, 2,000 rows per request, 6s pause (the
    keyless budget is 5 requests / 30s). CVSS baseSeverity -> band."""
    from datetime import datetime, timedelta

    out: list[dict] = []
    end = datetime(2026, 10, 8)
    window = timedelta(days=10)
    misses = 0
    while len(out) < 80_000 and misses < 6:
        start_dt = end - window
        url = (f"{NVD}?pubStartDate={start_dt.isoformat()}.000"
               f"&pubEndDate={end.isoformat()}.000&resultsPerPage=2000")
        req = urllib.request.Request(url, headers={
            "User-Agent": "mivia-small-json-corpus"})
        batch = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    batch = json.loads(resp.read())
                break
            except Exception as exc:
                print(f"window {start_dt.date()}: {exc}; sleeping 45s",
                      flush=True)
                time.sleep(45)
        if not isinstance(batch, dict) or not batch.get("vulnerabilities"):
            misses += 1
            end = start_dt
            continue
        misses = 0
        for item in batch["vulnerabilities"]:
            cve = item.get("cve", {})
            desc = next((x["value"] for x in cve.get("descriptions", [])
                         if x.get("lang") == "en"), "")
            metrics = cve.get("metrics", {})
            sev = None
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if key in metrics and metrics[key]:
                    sev = metrics[key][0]["cvssData"].get("baseSeverity")
                    break
            band = _nvd_band(sev) if sev else None
            if band is None or len(desc) < MIN_LEN:
                continue
            if sum(c.isascii() and c.isalpha() for c in desc) < len(desc) * 0.6:
                continue
            out.append({
                "ghsa_id": cve.get("id", ""),
                "text": desc,
                "band": band,
                "cvss_base_severity": sev,
                "published": cve.get("published", ""),
            })
        print(f"window {start_dt.date()}..{end.date()}: "
              f"+{batch.get('totalResults')} total-results, "
              f"kept {len(out)} so far", flush=True)
        end = start_dt
        time.sleep(6)
    return out


def _dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        text = r["text"].strip()
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        r["text"] = text
        out.append(r)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1100,
                    help="10-day windows walked backward")
    parser.add_argument("--freeze", action="store_true",
                        help="split train/reserve temporally and freeze")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / "advisories.jsonl"
    if not raw_path.is_file():
        rows = _fetch(args.days)
        rows = _dedupe(rows)
        with raw_path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"fetched {len(rows)} advisories -> {raw_path}", flush=True)
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()
            if line.strip()]
    print(f"raw pool: {len(rows)}")

    if not args.freeze:
        return 0

    # Temporal split: reserve = the newest RESERVE_DAYS days present.
    dates = sorted((r["published"][:10] for r in rows if r["published"]),
                   reverse=True)
    if not dates:
        print("no dated rows")
        return 1
    cutoff = (datetime.fromisoformat(dates[0]).date()
              - timedelta(days=RESERVE_DAYS)).isoformat()
    reserve = [r for r in rows
               if r["published"][:10] > cutoff]
    train = [r for r in rows if r not in reserve]
    # Balance check: the reserve must carry every band (report, not filter).
    from collections import Counter
    counts = Counter(r["band"] for r in reserve)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN.open("w") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with RESERVE.open("w") as fh:
        for r in reserve:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    rsha = hashlib.sha256(RESERVE.read_bytes()).hexdigest()
    PROVENANCE.write_text(
        "# security severity corpus provenance\n\n"
        f"- frozen_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"- source: NVD API 2.0 (public domain); English descriptions >= "
        f"{MIN_LEN} chars with a CVSS baseSeverity; deduplicated by "
        f"description sha256\n"
        f"- TEMPORAL split (random splits banned for this arm): reserve = "
        f"published after {cutoff}; train = everything earlier\n"
        f"- train: {len(train)} rows; reserve: {len(reserve)} rows "
        f"{dict(sorted(counts.items()))}\n"
        f"- bands: 0 low / 1 medium / 2 high / 3 critical (CVSS qualitative)\n"
        f"- reserve sha256: `{rsha}` (frozen before training)\n"
        f"- excluded by plan: Snyk (proprietary), Ubuntu CC-BY-SA rows\n")
    print(f"train {len(train)} reserve {len(reserve)} "
          f"(cutoff {cutoff}) bands {dict(sorted(counts.items()))}")
    print(f"reserve sha256 {rsha[:16]}... (frozen)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
