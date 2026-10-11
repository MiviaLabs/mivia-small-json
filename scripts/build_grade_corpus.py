#!/usr/bin/env python3
"""Build the ordinal grade corpus: data/real/grade/{train,dev}.jsonl.

Combines three sources into one leakage-safe corpus of
(text, family, gold, split) rows:

1. LogHub severity labels (data/real/log_event/train.jsonl gold.level) -
   the log-domain register the v4 artifact already learned.
2. Synthetic register-diverse severity lines
   (scripts/register_severity_generator.py) - syslog/app-log/business/SRE
   phrasings of the same incident kinds, split at the template-family level.
3. GitHub priority-labeled issues (data/incoming/external/issue-priority/
   issues-raw.jsonl, weak-labeled via the Caddy & Treude MSR 2024 mapping) -
   real issue-register text, split at the repository level.

Leakage rules (learned from the LogHub template overlap, see the training
report): synthetic rows split by template family, GitHub rows split by repo,
LogHub rows are train-only (its dev slice stays an untouched benchmark).
Exact dedupe within and across sources, checked against the frozen
novel-register eval cases in scripts/bench_zero_shot_primitives.py - an eval
row that collides with any training row is an error, not a warning.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from register_severity_generator import build_rows, template_split  # noqa: E402

LOG_TRAIN = ROOT / "data" / "real" / "log_event" / "train.jsonl"
ISSUES_RAW = ROOT / "data" / "incoming" / "external" / "issue-priority" / "issues-raw.jsonl"
OUT_DIR = ROOT / "data" / "real" / "grade"
BENCH = ROOT / "scripts" / "bench_zero_shot_primitives.py"

# Family tier orders; "gold" is the index within the family.
FAMILY_TIERS = {
    "severity": ["INFO", "WARNING", "ERROR", "CRITICAL"],
    "priority": ["LOW", "MEDIUM", "HIGH"],
}
PRIORITY_ALIAS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
REPO_HOLDOUT_FRAC = 0.15
MAX_ISSUE_CHARS = 600


def loghub_rows() -> list[dict]:
    """(text, family=severity, gold) from LogHub train severity labels."""
    alias = {"FATAL": "CRITICAL", "WARN": "WARNING", "ERR": "ERROR"}
    tiers = FAMILY_TIERS["severity"]
    out = []
    for line in LOG_TRAIN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        level = (row.get("gold") or {}).get("level")
        if not level:
            continue
        tier = alias.get(level, level)
        if tier not in tiers:
            continue
        text = row["raw"].strip()
        if text:
            out.append({"text": text, "family": "severity",
                        "gold": tiers.index(tier), "source": "loghub"})
    return out


def issue_text(row: dict) -> str:
    """Title + body, truncated; no scrubbing (owner data policy)."""
    parts = [row.get("title", "").strip()]
    body = row.get("body", "").strip()
    if body:
        parts.append(body)
    text = "\n\n".join(p for p in parts if p)
    return text[:MAX_ISSUE_CHARS]


def priority_rows(seed: int) -> list[dict]:
    """(text, family=priority, gold) from the harvested GitHub issues."""
    if not ISSUES_RAW.is_file():
        print(f"warning: no harvest at {ISSUES_RAW}; priority family skipped")
        return []
    seen: dict[str, dict] = {}
    for line in ISSUES_RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = issue_text(row)
        if len(text) < 120:
            continue
        if text in seen:
            continue
        seen[text] = {
            "text": text,
            "family": "priority",
            "gold": PRIORITY_ALIAS[row["priority"]],
            "repo": row.get("repo") or "unknown",
            "source": "github-issues",
        }
    rows = list(seen.values())
    # Repo-level split: all issues of a repo land on one side.
    repos = sorted({r["repo"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(repos)
    n_hold = max(1, round(len(repos) * REPO_HOLDOUT_FRAC))
    holdout = set(repos[:n_hold])
    out = []
    for r in rows:
        r["split"] = "dev" if r["repo"] in holdout else "train"
        out.append(r)
    return out


def eval_collision_check(all_rows: list[dict]) -> None:
    """A frozen novel-register eval row inside the training corpus is an error."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from bench_zero_shot_primitives import NOVEL_REGISTER_GRADE_CASES

    eval_texts = {text for text, _tier in NOVEL_REGISTER_GRADE_CASES}
    corpus_texts = {r["text"] for r in all_rows}
    collide = eval_texts & corpus_texts
    if collide:
        raise SystemExit(
            f"FATAL: {len(collide)} frozen eval rows appear in the corpus: "
            f"{sorted(collide)[:3]}"
        )
    print(f"eval collision check: {len(eval_texts)} frozen eval rows, "
          f"0 in corpus")


def main() -> int:
    seed = 42
    train: list[dict] = []
    dev: list[dict] = []

    # 1. LogHub severity: train only (its dev slice is a frozen benchmark).
    log = loghub_rows()
    train.extend(log)
    print(f"loghub severity: {len(log)} (train)")

    # 2. Synthetic register severity: template-level split.
    synth = build_rows(seed)
    syn_tr, syn_dev = template_split(synth, holdout_frac=0.15, seed=seed)
    for r in syn_tr:
        train.append({**r, "source": "synthetic", "split": "train"})
    for r in syn_dev:
        dev.append({**r, "source": "synthetic", "split": "dev"})
    print(f"synthetic severity: {len(syn_tr)} train / {len(syn_dev)} dev "
          f"(template-level split)")

    # 3. GitHub priority issues: repo-level split.
    pri = priority_rows(seed)
    train.extend(r for r in pri if r["split"] == "train")
    dev.extend(r for r in pri if r["split"] == "dev")
    print(f"github priority: {len(pri)} total "
          f"({sum(1 for r in pri if r['split'] == 'train')} train / "
          f"{sum(1 for r in pri if r['split'] == 'dev')} dev, repo-level split)")

    # Cross-source exact dedupe on (text).
    seen: set[str] = set()
    deduped_tr: list[dict] = []
    for r in train:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        deduped_tr.append(r)
    deduped_dev = [r for r in dev if r["text"] not in seen]
    dropped = len(train) + len(dev) - len(deduped_tr) - len(deduped_dev)
    print(f"exact dedupe: dropped {dropped}")

    all_rows = deduped_tr + deduped_dev
    eval_collision_check(all_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", deduped_tr), ("dev", deduped_dev)]:
        path = OUT_DIR / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        tiers = Counter((r["family"], r["gold"]) for r in rows)
        print(f"{path}: {len(rows)} rows")
        for fam, tiers_of_fam in FAMILY_TIERS.items():
            counts = [tiers.get((fam, i), 0) for i in range(len(tiers_of_fam))]
            print(f"  {fam}: {counts}")

    # Provenance sidecar for the committed corpus.
    import hashlib

    provenance = OUT_DIR / "PROVENANCE.md"
    digest = hashlib.sha256(
        (OUT_DIR / "train.jsonl").read_bytes()).hexdigest()[:16]
    provenance.write_text(
        "# data/real/grade provenance\n\n"
        "Ordinal grade corpus for the zero-shot `grade` primitive (built by\n"
        "`scripts/build_grade_corpus.py`, deterministic at seed 42).\n\n"
        "| source | content | license / basis | rows |\n"
        "|---|---|---|---|\n"
        "| loghub | `data/real/log_event/train.jsonl` severity labels, "
        "train split only | LogHub license accepted by owner decision "
        "2026-10-05 | see counts above |\n"
        "| synthetic | `scripts/register_severity_generator.py`, seeded "
        "templates across syslog/app-log/business/SRE registers | original "
        "content | see counts above |\n"
        "| github-issues | GitHub search over priority labels, weak-labeled "
        "via the Caddy & Treude (MSR 2024) mapping (Zenodo 11210209, "
        "CC-BY-4.0; raw dump `data/incoming/external/issue-priority/"
        "issues-raw.jsonl`); unambiguous directional labels only | GitHub "
        "ToS (public data), label mapping CC-BY-4.0 | see counts above |\n\n"
        f"train.jsonl sha256[:16]: {digest}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
