#!/usr/bin/env python3
"""Fetch real closed PRs (title+body) as pull_request router training data.

Arm B (pr-routing-real): the router has never seen a
real PR row, so the 300 reality-eval PRs route 0% -> pull_request. This
fetches closed PRs from public repos OTHER than the eval's flask/fastapi
(cross-repo generalization), writes the raw dump to
data/incoming/external/gh-prs-train/, and curates hash-disjoint
data/real/pull_request/{train,dev}.jsonl rows (90/10, seed 42).

The 300-row eval (data/incoming/gh-prs.json) is never written here and
never trained on; any text-hash collision with it (or with the frozen
held-out/synthetic sets) is refused, not skipped silently.

Idempotent: --fetch skips the API when the raw dump exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "incoming" / "external" / "gh-prs-train" / "prs-raw.json"
OUT_DIR = ROOT / "data" / "real" / "pull_request"
SEED = 42

# Eval PRs are flask+fastapi; train on different repos so the eval measures
# cross-repo transfer, not domain memorization.
REPOS = [
    "pytest-dev/pytest",
    "psf/requests",
    "pallets/werkzeug",
    "encode/httpx",
    "sympy/sympy",
    "scikit-learn/scikit-learn",
]
PAGES_PER_REPO = 5
PER_PAGE = 100
MIN_TEXT_LEN = 80


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def nhash(text: str) -> str:
    return hashlib.sha256(_norm(text).encode()).hexdigest()


def _frozen_hashes() -> set[str]:
    """Text hashes the training rows must never collide with."""
    hashes: set[str] = set()
    eval_prs = json.loads((ROOT / "data" / "incoming" / "gh-prs.json").read_text())
    for p in eval_prs:
        hashes.add(nhash(f"{p['title']}\n\n{p.get('body') or ''}".strip()))
    for path in (ROOT / "data" / "heldout" / "test.jsonl",
                 ROOT / "data" / "synthetic" / "train.jsonl",
                 ROOT / "data" / "synthetic" / "dev.jsonl"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    hashes.add(nhash(json.loads(line)["raw"]))
    return hashes


def _gh_api(endpoint: str, page: int) -> list[dict]:
    """One gh api call with a hard timeout (gh can hang; see MEMORY 2026-10-07)."""
    url = f"{endpoint}?state=closed&per_page={PER_PAGE}&page={page}"
    proc = subprocess.run(
        ["gh", "api", url], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api {url} failed: {proc.stderr[:200]}")
    data = json.loads(proc.stdout)
    if not isinstance(data, list):
        raise RuntimeError(f"gh api {url} returned {type(data).__name__}")
    return data


def fetch() -> list[dict]:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for repo in REPOS:
        got = 0
        for page in range(1, PAGES_PER_REPO + 1):
            try:
                rows = _gh_api(f"repos/{repo}/pulls", page)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                print(f"  {repo} page {page}: {exc} — stopping this repo",
                      file=sys.stderr)
                break
            if not rows:
                break
            for pr in rows:
                body = (pr.get("body") or "").strip()
                title = (pr.get("title") or "").strip()
                if not title or not body:
                    continue
                out.append({"repo": repo, "number": pr.get("number"),
                            "title": title, "body": body})
                got += 1
            time.sleep(0.5)  # be a good API citizen
        print(f"{repo}: {got} PR rows with title+body")
    RAW.write_text(json.dumps(out, indent=1) + "\n")
    print(f"raw dump -> {RAW} ({len(out)} rows)")
    return out


def curate(raw_rows: list[dict]) -> None:
    frozen = _frozen_hashes()
    seen: set[str] = set()
    kept, refused_frozen, dupes = [], 0, 0
    for pr in raw_rows:
        text = f"{pr['title']}\n\n{pr['body']}".strip()
        if len(text) < MIN_TEXT_LEN:
            continue
        h = nhash(text)
        if h in frozen:
            refused_frozen += 1
            continue
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        kept.append({"source": f"gh:{pr['repo']}#{pr['number']}",
                     "raw": text,
                     "gold": {"title": pr["title"], "body": pr["body"]}})
    rng = random.Random(SEED)
    rng.shuffle(kept)
    n_dev = max(1, len(kept) // 10)
    dev, train = kept[:n_dev], kept[n_dev:]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train),
        encoding="utf-8")
    (OUT_DIR / "dev.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dev),
        encoding="utf-8")
    print(f"curated: {len(train)} train / {len(dev)} dev "
          f"(refused {refused_frozen} frozen-collisions, {dupes} dupes) "
          f"-> {OUT_DIR}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="hit the GitHub API (skipped when the raw dump "
                         "exists unless --force)")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if the raw dump exists")
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
