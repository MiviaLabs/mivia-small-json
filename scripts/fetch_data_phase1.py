#!/usr/bin/env python3
"""Fetch the phase-1 datasets from DATA_TRAINING_PLAN.md (bounded, no Loghub-2.0).

Raw dumps land under ``data/incoming/external/``. Nothing here curates or
admits training rows; ``scripts/curate_data_phase1.py`` does that.

Sources and licenses (verified at fetch time):

- **NLBSE'24 issue report classification** — ``nlbse2024/issue-report-classification``
  (GitHub). 1,500 train / 1,500 test issue reports from five repos, one label
  each (bug / feature / question). License: the competition repository ships
  no explicit license file; the underlying issue text is public GitHub content
  and the dataset is redistributed by the organisers for research. Recorded as
  "GitHub public data / no license file".
- **GitBugs** — ``av9ash/gitbugs`` (GitHub), CC BY 4.0 (README, Patil 2025,
  arXiv 2504.09651). Per-project Jira-style bug CSVs: summary, id, status,
  priority, resolution, created, resolved, description.
- **CommitChronicle** — HF ``JetBrains-Research/commit-chronicle``. Card
  license is "other"; each record carries its own ``license`` field, which is
  what the curator filters on (permissive SPDX names only). Zenodo copy is
  CC BY 4.0. A bounded slice is fetched through the datasets-server rows API.
- **ACEBench** — ``ACEBench/ACEBench`` (GitHub), MIT (Microsoft). The
  ``Special`` split: infeasible / incomplete / malformed-parameter calls.

Bounded on purpose: no Loghub-2.0 (965 MB), no full commit-chronicle
(14 GB download). Deterministic and resumable (a file that already exists is
not re-fetched). Stdlib only.

    python scripts/fetch_data_phase1.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "incoming" / "external"
UA = {"User-Agent": "mivia-small-json/1.0"}
RETRIES = 3

# NLBSE'24 issue report classification (1,500 train / 1,500 test).
NLBSE24_BASE = ("https://raw.githubusercontent.com/nlbse2024/"
                "issue-report-classification/main/data")
NLBSE24_FILES = ("issues_train.csv", "issues_test.csv")

# GitBugs per-project full bug CSVs. ``zip`` names the member inside the
# archive when the remote is zipped; ``None`` means the URL is the CSV.
GITBUGS_BASE = "https://raw.githubusercontent.com/av9ash/gitbugs/main"
GITBUGS = {
    "cassandra": ("cassandra/cassandra_bugs.csv", None),
    "firefox": ("firefox/Firefox_bugs.csv.zip", "Firefox_bugs.csv"),
    "hadoop": ("hadoop/hadoop_bugs.csv", None),
    "hbase": ("hbase/hbase_bugs.csv", None),
    # mozillacore's 85k-report CSV is only on Google Drive; the file served is
    # a zip named Core_bugs.csv.zip.
    "mozillacore": ("gdrive:1zdjfEIDkdsFdCkClo4Tw80Ihdk6DAi6j", "Core_bugs.csv"),
    "ms_vscode_bugs": ("ms_vscode_bugs/vscode_bugs.csv.zip", "vscode_bugs.csv"),
    "seamonkey": ("seamonkey/SeaMonkey_bugs.csv", None),
    "spark": ("spark/spark_bugs.csv", None),
    "thunderbird": ("thunderbird/Thunderbird_bugs.csv", None),
}

# CommitChronicle bounded slice: 80 pages x 100 rows = 8k commits, offset 0.
# Bounded well below the 14 GB full download; the rows API is rate-limited, so
# this is also the practical ceiling for one run.
CC_ROWS = ("https://datasets-server.huggingface.co/rows"
           "?dataset=JetBrains-Research%2Fcommit-chronicle"
           "&config=default&split=train&offset={offset}&length=100")
CC_PAGES = 80

# ACEBench Special split (infeasible / incomplete / malformed params).
ACE_BASE = ("https://raw.githubusercontent.com/ACEBench/ACEBench/main/"
            "data_all/data_en")
ACE_FILES = ("data_special_irrelevant.json",
             "data_special_incomplete.json",
             "data_special_error_param.json")


def _get(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(RETRIES + 2):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 404:
                break
            # The datasets-server throttles aggressively (429); back off.
            time.sleep((6.0 if exc.code == 429 else 1.5) * (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url} ({last})")


def _write_if_absent(dest: Path, data: bytes) -> bool:
    if dest.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def fetch_nlbse24() -> list[str]:
    out = []
    for name in NLBSE24_FILES:
        dest = EXT / "nlbse24" / name
        if dest.is_file():
            out.append(f"nlbse24/{name}: cached")
            continue
        data = _get(f"{NLBSE24_BASE}/{name}")
        _write_if_absent(dest, data)
        rows = max(0, data.count(b"\n") - 1)
        out.append(f"nlbse24/{name}: {len(data)} B, ~{rows} lines")
    return out


def fetch_gitbugs() -> list[str]:
    out = []
    raw = EXT / "gitbugs" / "_raw"
    for project, (remote, member) in sorted(GITBUGS.items()):
        dest = EXT / "gitbugs" / f"{project}.csv"
        if dest.is_file():
            out.append(f"gitbugs/{project}.csv: cached")
            continue
        if remote.startswith("gdrive:"):
            fid = remote.split(":", 1)[1]
            url = (f"https://drive.usercontent.google.com/download?id={fid}"
                   "&export=download&confirm=t")
        else:
            url = f"{GITBUGS_BASE}/{remote}"
        blob = _get(url)
        if member is None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
        else:
            raw.mkdir(parents=True, exist_ok=True)
            archive = raw / f"{project}.zip"
            archive.write_bytes(blob)
            with zipfile.ZipFile(archive) as z:
                names = [n for n in z.namelist() if n == member]
                if not names:
                    raise RuntimeError(f"{project}: {member} not in archive")
                dest.write_bytes(z.read(names[0]))
        rows = max(0, dest.read_bytes().count(b"\n") - 1)
        out.append(f"gitbugs/{project}.csv: {dest.stat().st_size} B, ~{rows} rows")
    return out


def fetch_commit_chronicle() -> list[str]:
    dest = EXT / "commit-chronicle" / "commits-raw.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    if dest.is_file():  # resume a throttled run
        for line in dest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                kept.append(json.loads(line))
    start = len(kept) // 100
    with dest.open("a", encoding="utf-8") as fh:
        for page in range(start, CC_PAGES):
            payload = json.loads(_get(CC_ROWS.format(offset=page * 100)))
            rows = payload.get("rows", [])
            if not rows:
                break
            for item in rows:
                row = item["row"]
                # Drop the diff blobs: only the message-relevant fields are
                # needed, and the full row (with diffs) is hundreds of MB per
                # shard.
                kept.append({"hash": row.get("hash"),
                             "message": row.get("message"),
                             "repo": row.get("repo"),
                             "license": row.get("license"),
                             "language": row.get("language"),
                             "date": row.get("date")})
                fh.write(json.dumps(kept[-1], ensure_ascii=False) + "\n")
            fh.flush()
            time.sleep(0.7)  # stay under the datasets-server rate limit
    return [f"commit-chronicle/commits-raw.jsonl: {len(kept)} rows "
            f"({dest.stat().st_size} B)"]


def fetch_acebench() -> list[str]:
    out = []
    for name in ACE_FILES:
        dest = EXT / "acebench" / name
        if dest.is_file():
            out.append(f"acebench/{name}: cached")
            continue
        data = _get(f"{ACE_BASE}/{name}")
        _write_if_absent(dest, data)
        rows = sum(1 for line in data.decode("utf-8").splitlines() if line.strip())
        out.append(f"acebench/{name}: {len(data)} B, {rows} rows")
    return out


def main() -> int:
    entries: list[str] = []
    for label, fn in (("nlbse24", fetch_nlbse24),
                      ("gitbugs", fetch_gitbugs),
                      ("commit-chronicle", fetch_commit_chronicle),
                      ("acebench", fetch_acebench)):
        try:
            entries += fn()
        except Exception as exc:  # keep the other sources going
            print(f"skipping {label}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    for entry in entries:
        print(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
