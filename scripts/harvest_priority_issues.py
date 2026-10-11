#!/usr/bin/env python3
"""Harvest GitHub issues carrying unambiguous priority labels.

Source: the Caddy & Treude (MSR 2024) prioritised-label mapping (Zenodo
11210209, CC-BY-4.0) at data/incoming/external/issue-priority/. Only
directionally-unambiguous labels are searched: the mapping itself is
contradictory on numeric labels (repos number priorities differently), so
P1/P2/... and star-count labels are excluded. Each label used here was
cross-checked against the CSV mapping.

Writes one JSON line per issue to data/incoming/external/issue-priority/
issues-raw.jsonl: {url, repo, title, body, labels, priority}. The raw dump
stays in data/incoming/; the curated corpus builder turns it into
data/real/grade/ rows. Real user text: never commit the raw dump.

Requires the gh CLI (token with repo read scope). Search API: ~30
requests/min authenticated; this script sleeps between pages.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = (ROOT / "data" / "incoming" / "external" / "issue-priority"
           / "all-unique-labels-prioritised.csv")
OUT = ROOT / "data" / "incoming" / "external" / "issue-priority" / "issues-raw.jsonl"

# Directional labels only; every entry must exist in the Zenodo mapping with
# the same direction (checked at startup). Numeric (P1/P2) and star-count
# labels are excluded on purpose.
HIGH_LABELS = [
    "priority: high", "high priority", "Priority: High", "prio: high",
    "critical", "Critical", "blocker", "Blocker", "urgent", "Urgent",
    "bug: critical", "bug/major", "major", "release blocker",
    "severity: critical", "severity: high", "severe",
]
MEDIUM_LABELS = [
    "priority: medium", "medium priority", "Priority: Medium", "moderate",
    "severity: medium", "bug: medium", "bug moderate",
]
LOW_LABELS = [
    "priority: low", "low priority", "Priority: Low", "prio: low",
    "minor", "Minor", "trivial", "Trivial", "backburner", "nice to have",
    "severity: minor", "severity: low", "bug: minor", "bug minor",
]
WANTED = {label: "HIGH" for label in HIGH_LABELS}
WANTED.update({label: "MEDIUM" for label in MEDIUM_LABELS})
WANTED.update({label: "LOW" for label in LOW_LABELS})

PER_LABEL = 150
PER_PAGE = 100
SEARCH_TIMEOUT = 300


def main() -> int:
    if not MAPPING.is_file():
        print(f"missing mapping: {MAPPING} — fetch the Zenodo record first", file=sys.stderr)
        return 1
    mapping = {r["label"]: r["priority"] for r in csv.DictReader(MAPPING.open())}
    direction = {"High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
    wanted: dict[str, str] = {}
    for label, want in WANTED.items():
        got = mapping.get(label)
        if got is None:
            print(f"skipping {label!r}: not in the Zenodo mapping")
        elif direction[got] != want:
            print(f"skipping {label!r}: mapping says {got!r}, expected {want}")
        else:
            wanted[label] = want
    if not wanted:
        print("no usable labels left", file=sys.stderr)
        return 1

    seen: dict[str, dict] = {}
    if OUT.is_file():
        for line in OUT.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen[row["url"]] = row
    print(f"resuming with {len(seen)} issues already harvested")

    for label, want in wanted.items():
        query = f'label:"{label}" type:issue'
        fetched = 0
        for page in (1, 2):
            if fetched >= PER_LABEL or len(seen) >= 4000:
                break
            import urllib.parse
            import urllib.request

            q = urllib.parse.quote(query)
            url = (f"https://api.github.com/search/issues?q={q}"
                   f"&per_page={PER_PAGE}&page={page}")
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "mivia-grade-corpus",
            })
            token = subprocess.run(["gh", "auth", "token"],
                                   capture_output=True, text=True,
                                   timeout=30)
            if token.returncode == 0 and token.stdout.strip():
                req.add_header("Authorization",
                               f"Bearer {token.stdout.strip()}")
            try:
                with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
                    items = json.loads(resp.read().decode("utf-8")).get("items", [])
            except Exception as exc:
                print(f"search failed for {label!r}: {exc}", file=sys.stderr)
                time.sleep(3)
                break
            if not items:
                break
            for it in items:
                url = it.get("html_url", "")
                if not url or url in seen:
                    continue
                body = (it.get("body") or "").strip()
                title = (it.get("title") or "").strip()
                if len(title) + len(body) < 120:
                    continue
                seen[url] = {
                    "url": url,
                    "repo": it.get("repository_url", "").rsplit("/", 2)[-2:][0]
                    if it.get("repository_url") else "",
                    "title": title,
                    "body": body,
                    "labels": [lab["name"] for lab in it.get("labels", [])],
                    "priority": want,
                    "matched_label": label,
                }
                fetched += 1
            time.sleep(2.5)
        print(f"{label!r} -> {want}: total now {len(seen)}")

    OUT.write_text("\n".join(json.dumps(r) for r in seen.values()) + "\n", encoding="utf-8")
    print(f"wrote {len(seen)} issues to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
