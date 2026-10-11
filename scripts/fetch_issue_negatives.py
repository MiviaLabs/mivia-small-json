#!/usr/bin/env python3
"""Fetch real ISSUES that mention pull requests — router hard negatives.

Follow-up to Arm B (pr-router-negatives): after the
PR-training promotion, PR-looking issue statements route to
`pull_request` (SWE-bench problems 98 -> 86). The fix is contrast, not
thresholds: real issues from the SAME repos as the PR training data,
whose text talks about pull requests, labeled `issue` — so the router
learns "an issue that mentions a PR is still an issue".

Raw dump -> data/incoming/external/gh-issue-negatives/issues-raw.json;
curated rows -> data/real/issue/train-prneg.jsonl (loaded via a second
REAL_SETS entry). Hash-disjoint from the eval-300 PRs, the reality-eval
samples, the SWE-bench file, and all frozen sets; deduped against
data/real/issue/train.jsonl. Idempotent with --fetch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "incoming" / "external" / "gh-issue-negatives" / "issues-raw.json"
OUT = ROOT / "data" / "real" / "issue" / "train-prneg.jsonl"

REPOS = [
    "pytest-dev/pytest",
    "psf/requests",
    "pallets/werkzeug",
    "encode/httpx",
    "sympy/sympy",
    "scikit-learn/scikit-learn",
]
PER_REPO = 60
MIN_TEXT_LEN = 80


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def nhash(text: str) -> str:
    return hashlib.sha256(_norm(text).encode()).hexdigest()


def _existing_hashes() -> set[str]:
    hashes: set[str] = set()
    for path in (
        ROOT / "data" / "incoming" / "gh-prs.json",
        ROOT / "data" / "incoming" / "external" / "swebench" / "swebench_lite_100.json",
        ROOT / "data" / "real" / "issue" / "train.jsonl",
        ROOT / "data" / "real" / "issue" / "dev.jsonl",
        ROOT / "data" / "real" / "pull_request" / "train.jsonl",
        ROOT / "data" / "real" / "pull_request" / "dev.jsonl",
        ROOT / "data" / "heldout" / "test.jsonl",
        ROOT / "data" / "synthetic" / "train.jsonl",
        ROOT / "data" / "synthetic" / "dev.jsonl",
    ):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name == "gh-prs.json":
            for p in json.loads(text):
                hashes.add(nhash(f"{p['title']}\n\n{p.get('body') or ''}".strip()))
        elif "swebench" in path.name:
            for rec in json.loads(text):
                hashes.add(nhash(rec.get("problem_statement") or ""))
        else:
            for line in text.splitlines():
                if line.strip():
                    try:
                        hashes.add(nhash(json.loads(line)["raw"]))
                    except (json.JSONDecodeError, KeyError):
                        continue
    return hashes


def _search(repo: str, page: int) -> dict:
    query = f"repo:{repo} is:issue pull request in:body"
    url = ("search/issues?q=" + query.replace(" ", "+")
           + f"&per_page=100&page={page}")
    proc = subprocess.run(["gh", "api", url], capture_output=True,
                          text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api failed: {proc.stderr[:200]}")
    return json.loads(proc.stdout)


def fetch() -> list[dict]:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for repo in REPOS:
        got = 0
        for page in (1, 2):
            try:
                payload = _search(repo, page)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                print(f"  {repo} page {page}: {exc}", file=sys.stderr)
                break
            for item in payload.get("items", []):
                body = (item.get("body") or "").strip()
                title = (item.get("title") or "").strip()
                if not title or not body:
                    continue
                out.append({"repo": repo, "number": item.get("number"),
                            "title": title, "body": body})
                got += 1
            time.sleep(2.0)  # search API rate limit is per-minute
        print(f"{repo}: {got} issue rows mentioning pull requests")
    RAW.write_text(json.dumps(out, indent=1) + "\n")
    print(f"raw dump -> {RAW} ({len(out)} rows)")
    return out


def curate(raw_rows: list[dict]) -> None:
    banned = _existing_hashes()
    seen: set[str] = set()
    kept, refused, dupes = [], 0, 0
    for it in raw_rows:
        text = f"{it['title']}\n\n{it['body']}".strip()
        if len(text) < MIN_TEXT_LEN:
            continue
        h = nhash(text)
        if h in banned:
            refused += 1
            continue
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        kept.append({"source": f"gh-issue:{it['repo']}#{it['number']}",
                     "raw": text,
                     "gold": {"title": it["title"], "body": it["body"]}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in kept), encoding="utf-8")
    print(f"curated {len(kept)} rows (refused {refused} collisions, "
          f"{dupes} dupes) -> {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.force or not RAW.is_file():
        if not args.fetch and not args.force:
            print(f"{RAW} missing; run with --fetch", file=sys.stderr)
            return 2
        rows = fetch()
    else:
        rows = json.loads(RAW.read_text())
        print(f"using existing raw dump ({len(rows)} rows)")
    curate(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
