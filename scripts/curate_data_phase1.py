#!/usr/bin/env python3
"""Curate the phase-1 datasets into the committed corpora.

Reads the raw dumps fetched by ``scripts/fetch_data_phase1.py`` and writes:

- ``data/real/issue/{train,dev}.jsonl``   GitBugs bug reports (CC BY 4.0),
  priority mapped onto the issue enum, appended idempotently to the existing
  split. ``data/real/PROVENANCE.md`` and ``data/dataset-manifest.json`` record
  the new counts/sha256.
- ``data/real/commit/{train,dev}.jsonl``  CommitChronicle commits, per-record
  permissive licenses only, appended idempotently.
- ``data/incoming/external/nlbse24/issues.jsonl``  NLBSE'24 held-out issue
  reports (the competition test split, deduped against the admitted train
  rows), for router OOD measurement.
- ``data/incoming/external/phase1-ood/commit.jsonl``  project/sha-disjoint
  CommitChronicle held-out commits (the five projects with fewer than 200 raw
  commits, excluded from train), for router OOD measurement.
- ``data/incoming/external/bfcl/pairs-ace.jsonl``  ACEBench ``Special``
  should-abstain pairs (new file; the committed ``pairs.jsonl`` is untouched).

Leakage rules (same as ``scripts/extend_real_data.py``):

1. Keep records with 10-1000 chars and >= 90% ASCII-printable.
2. Drop duplicates by normalized text (and by commit sha).
3. Never admit a frozen record (``data/wild``, ``data/heldout``,
   ``data/synthetic``): whole row, any single line, or its scrubbed preimage.
4. Seeded, hash-based splits and caps; re-runs are idempotent.

Stdlib only. Run from the repository root.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.tool_match import render_tool  # noqa: E402

DATA = ROOT / "data"
REAL = DATA / "real"
EXT = DATA / "incoming" / "external"
GITBUGS_DIR = EXT / "gitbugs"
NLBSE24_DIR = EXT / "nlbse24"
CC_RAW = EXT / "commit-chronicle" / "commits-raw.jsonl"
ACE_DIR = EXT / "acebench"
BFCL = EXT / "bfcl"

# CommitChronicle projects reserved as the project-disjoint OOD held-out: the
# five projects with fewer than 200 raw commits. Their rows never enter the
# train admission, so the held-out shares no project (and no sha) with the
# admitted rows — a router that memorizes a repo cannot self-report here.
CC_HELDOUT_REPOS = (
    "igniterealtime/smack",
    "jonnybgod/angular2-webpack-advance-starter",
    "irmen/pyro4",
    "pterodactyl/wings",
    "worldwonderer/worldwonderer.github.io",
)
CC_HELDOUT = EXT / "phase1-ood" / "commit.jsonl"

SEED = 20261006
# Phase-1 sources are admitted TRAIN-ONLY. The dev sets feed two fixed
# measurements — the rules-baseline trust table (`label.measure_dev_rules`) and
# the router calibration — and growing them re-baselines both: these real
# issue/commit rows drop every class below the 0.85 trust floor (commit F1
# 0.369, issue 0.351; log_event falls 0.893 -> 0.767 on misrouted rows) and the
# flywheel stops admitting weak labels. The rows still train the router; only
# the measurement basis stays pinned. Revisit when the trust is re-derived on a
# frozen reference rather than the growing dev split.
DEV_PCT = 0
GITBUGS_TRAIN_CAP = 15000
GITBUGS_DEV_CAP = 1500
CC_TRAIN_CAP = 6000
CC_DEV_CAP = 800

COMMIT_TYPES = ("feat", "fix", "chore", "docs", "style", "refactor", "perf",
                "test", "build", "ci")
# Jira and Mozilla priority scales -> the issue schema's 4 buckets. Rows whose
# priority does not map (Mozilla "--", VS Code's missing column) are skipped so
# the gold priority is never invented.
GITBUGS_PRIORITY = {
    "blocker": "critical", "critical": "critical", "urgent": "critical",
    "high": "high", "major": "high",
    "normal": "medium", "medium": "medium",
    "minor": "low", "low": "low", "trivial": "low",
    "p1": "critical", "p2": "high", "p3": "medium", "p4": "low", "p5": "low",
}

_ASCII = re.compile(r"[\x20-\x7e\t]")
_SEPARATORS = re.compile("[\u2028\u2029\x85\x1c\x1d\x1e]")
_ROLE = re.compile(r"^(?:user|assistant|system)\s*:\s*", re.IGNORECASE)
_CC_TYPE = re.compile(r"^\s*([A-Za-z]+)(?:\([^)]*\))?!?:")


def _clean(text: str) -> str:
    return _SEPARATORS.sub(" ", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _keep(text: str) -> bool:
    if not (10 <= len(text) <= 1000):
        return False
    return sum(1 for c in text if _ASCII.match(c)) / len(text) >= 0.90


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_of(key: str) -> str:
    return "dev" if int(_digest(key), 16) % 100 < DEV_PCT else "train"


def _cap(rows: list[dict], cap: int) -> list[dict]:
    """Deterministic content-hash subsample, truncated to ``cap``."""
    ordered = sorted(rows, key=lambda r: _digest(r["source"] + "\x00" + r["raw"]))
    return ordered[:cap]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    # Split on "\n" only: splitlines() also splits on U+2028/U+2029.
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .split("\n") if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in rows), encoding="utf-8")


def frozen_keys() -> tuple[set[str], set[str]]:
    """(text keys, commit shas) of every frozen raw record and line.

    The wild/heldout slices are frozen snapshots; a raw row that would have
    been scrubbed into one of them differs from it, so each key also covers
    its would-have-been-scrubbed preimage.
    """
    texts: set[str] = set()
    shas: set[str] = set()
    frozen = ["data/wild/issue.jsonl", "data/wild/commit.jsonl",
              "data/wild/platform-logs.jsonl", "data/heldout/test.jsonl",
              "data/heldout/grade-register-panel.jsonl",
              "data/synthetic/train.jsonl", "data/synthetic/dev.jsonl"]
    for rel in frozen:
        path = ROOT / rel
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            gold = row.get("gold")
            if isinstance(gold, dict) and "sha" in gold:
                shas.add(str(gold["sha"]))
            text = row.get("raw") or ""
            if not text:
                continue
            variants = {text, re.sub(r"\[email\]", "[email]", text),
                        re.sub(r"\S+@\S+\.\S+", "[email]", text)}
            for variant in variants:
                texts.add(_norm(variant))
                texts.update(_norm(line) for line in variant.splitlines()
                             if line.strip())
    return texts, shas


def build_gitbugs(frozen_texts: set[str]) -> list[dict]:
    csv.field_size_limit(10 ** 9)
    rows: list[dict] = []
    seen: set[str] = set()
    for path in sorted(GITBUGS_DIR.glob("*.csv")):
        project = path.stem
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            for rec in csv.DictReader(fh):
                priority = GITBUGS_PRIORITY.get(
                    (rec.get("Priority") or "").strip().lower())
                if not priority:
                    continue
                title = (rec.get("Summary") or "").strip()
                body = (rec.get("Description") or "").strip()
                if not title:
                    continue
                raw = _clean(title + "\n" + body[:400])[:1000]
                key = _norm(raw)
                if not _keep(raw) or key in seen or key in frozen_texts:
                    continue
                seen.add(key)
                rows.append({
                    "source": f"gitbugs:{project}",
                    "raw": raw,
                    "gold": {"type": "issue", "title": title, "labels": [],
                             "priority": priority},
                })
    return rows


def build_commit_chronicle(frozen_texts: set[str],
                           frozen_shas: set[str]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for rec in _read_jsonl(CC_RAW):
        sha = str(rec.get("hash") or "").strip()
        message = _clean(rec.get("message") or "")
        if rec.get("repo") in CC_HELDOUT_REPOS:
            continue
        if not _permissive(rec.get("license")):
            continue
        key = _norm(message)
        if not sha or sha in frozen_shas or key in seen or key in frozen_texts \
                or not _keep(message):
            continue
        seen.add(key)
        gold = {"sha": sha, "message": message}
        match = _CC_TYPE.match(message)
        if match and match.group(1).lower() in COMMIT_TYPES:
            gold["commit_type"] = match.group(1).lower()
        rows.append({"source": f"commit-chronicle:{rec.get('repo')}",
                     "raw": message, "gold": gold})
    return rows


def build_commit_chronicle_holdout(frozen_texts: set[str],
                                   frozen_shas: set[str],
                                   train_texts: set[str]) -> list[dict]:
    """Project/sha-disjoint CommitChronicle held-out rows (OOD, never trained).

    Only rows from ``CC_HELDOUT_REPOS`` — projects excluded from the train
    admission — so the held-out shares no project or sha with the admitted
    rows. Same hygiene, permissive-license and frozen-set filters as train,
    and also deduped against every admitted train row.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for rec in _read_jsonl(CC_RAW):
        if rec.get("repo") not in CC_HELDOUT_REPOS:
            continue
        sha = str(rec.get("hash") or "").strip()
        message = _clean(rec.get("message") or "")
        if not _permissive(rec.get("license")):
            continue
        key = _norm(message)
        if not sha or sha in frozen_shas or key in seen or key in frozen_texts \
                or key in train_texts or not _keep(message):
            continue
        seen.add(key)
        rows.append({"source": f"commit-chronicle:{rec.get('repo')}",
                     "sha": sha, "text": message, "schema": "commit"})
    _write_jsonl(CC_HELDOUT, rows)
    return rows


def _permissive(license_name: object) -> bool:
    """True for clearly permissive per-record licenses only."""
    text = str(license_name or "").strip().lower()
    if not text:
        return False
    if any(bad in text for bad in ("gpl", "agpl", "lgpl", "mozilla", "mpl",
                                   "eclipse", "cddl", "eupl")):
        return False
    return any(good in text for good in ("apache", "mit", "bsd", "isc",
                                         "unlicense", "0bsd", "cc0",
                                         "public domain"))


def append_split(schema: str, split: str, new_rows: list[dict]) -> int:
    path = REAL / schema / f"{split}.jsonl"
    rows = _read_jsonl(path)
    seen = {_norm(r["raw"]) for r in rows}
    added = 0
    for row in new_rows:
        key = _norm(row["raw"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        added += 1
    _write_jsonl(path, rows)
    return added


def _admitted_train_texts() -> set[str]:
    """Normalized text of every admitted train row, across all schemas.

    The held-out sets are scored against the router that trains on these
    rows, so any row that reappears here is a self-report, not a result.
    Read after the appends, so the newly admitted rows are included.
    """
    rels = ["data/synthetic/train.jsonl", "data/pool/confident.jsonl"] + [
        f"data/real/{s}/train.jsonl" for s in
        ("issue", "commit", "log_event", "tool_call", "no_tool_call",
         "diagnostic", "diff", "review_comment", "agent_step")]
    out: set[str] = set()
    for rel in rels:
        for row in _read_jsonl(DATA.parent / rel):
            text = row.get("raw") or row.get("text") or ""
            if text:
                out.add(_norm(text))
    return out


def build_nlbse24_holdout(train_texts: set[str]) -> int:
    """The NLBSE'24 test split as a held-out OOD eval set (not training).

    Deduped against every admitted train row: the split is a real GitHub
    corpus and one of its issues already sits in the existing ``issue`` train
    pool, so without this filter the held-out would score a row the router
    trained on.
    """
    path = NLBSE24_DIR / "issues_test.csv"
    csv.field_size_limit(10 ** 9)
    rows = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        for rec in csv.DictReader(fh):
            text = _clean((rec.get("title") or "") + "\n"
                          + (rec.get("body") or ""))[:2000]
            label = (rec.get("label") or "").strip()
            key = _norm(text)
            if not text or not label or key in seen or key in train_texts:
                continue
            seen.add(key)
            rows.append({"source": "nlbse2024", "text": text, "label": label})
    _write_jsonl(NLBSE24_DIR / "issues.jsonl", rows)
    return len(rows)


def build_acebench_pairs() -> dict[str, int]:
    """Should-abstain pairs from the ACEBench Special split."""
    existing: set[tuple[str, str]] = set()
    for rec in _read_jsonl(BFCL / "pairs.jsonl"):
        existing.add((rec["request"], rec["tool"]))
    kinds = (("data_special_irrelevant.json", "irrelevant"),
             ("data_special_incomplete.json", "incomplete"),
             ("data_special_error_param.json", "error_param"))
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for name, kind in kinds:
        counts[kind] = 0
        for line in (ACE_DIR / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            request = _clean(_ROLE.sub("", rec.get("question") or ""))
            if not request:
                continue
            for tool in rec.get("function") or []:
                span = _clean(render_tool(tool))
                if not span or (request, span) in existing or (request, span) in seen:
                    continue
                seen.add((request, span))
                split = "dev" if int(_digest(request + span), 16) % 5 == 0 \
                    else "train"
                rows.append({"request": request, "tool": span, "label": 0,
                             "source": f"acebench_special_{kind}",
                             "split": split})
                counts[kind] += 1
    _write_jsonl(BFCL / "pairs-ace.jsonl", rows)
    return counts


def _update_provenance(file_entries: dict[str, tuple[int, str]],
                       source_rows: list[str], bullets: list[str]) -> None:
    path = REAL / "PROVENANCE.md"
    text = path.read_text(encoding="utf-8")
    # Idempotent: drop a previous phase-1 section and any copies of the new
    # source rows before re-inserting, so a re-run rewrites rather than
    # appends.
    marker = "\n## Phase-1 additions"
    text = text.partition(marker)[0].rstrip("\n") + "\n"
    for row in source_rows:
        text = text.replace(row + "\n", "")
    for rel, (count, digest) in file_entries.items():
        text = re.sub(
            rf"^\| `{re.escape(rel)}` \| \d+ \| [0-9a-f]{{64}} \|$",
            f"| `{rel}` | {count} | {digest} |", text, flags=re.M)
    # Append the new source rows to the end of the Sources table (the table
    # runs up to the next top-level heading).
    anchor = "\n## Privacy policy"
    head, sep, tail = text.partition(anchor)
    head = head.rstrip("\n") + "\n" + "\n".join(source_rows) + "\n"
    text = head + sep + tail
    section = "\n## Phase-1 additions (2026-10-06)\n\n" + "\n".join(bullets) + "\n"
    text = text.rstrip("\n") + "\n" + section
    path.write_text(text, encoding="utf-8")


def main() -> int:
    frozen_texts, frozen_shas = frozen_keys()

    # The candidate pool is built and capped before any committed rows are
    # removed, so the selected set is stable across runs and `append_split`
    # dedupes it to a no-op on a re-run (idempotent).
    issue_rows = build_gitbugs(frozen_texts)
    issue_train = _cap([r for r in issue_rows if _split_of(r["raw"]) == "train"],
                       GITBUGS_TRAIN_CAP)
    issue_dev = _cap([r for r in issue_rows if _split_of(r["raw"]) == "dev"],
                     GITBUGS_DEV_CAP)
    added_issue_train = append_split("issue", "train", issue_train)
    added_issue_dev = append_split("issue", "dev", issue_dev)

    commit_rows = build_commit_chronicle(frozen_texts, frozen_shas)
    commit_train = _cap([r for r in commit_rows
                         if _split_of(r["gold"]["sha"]) == "train"],
                        CC_TRAIN_CAP)
    commit_dev = _cap([r for r in commit_rows
                       if _split_of(r["gold"]["sha"]) == "dev"], CC_DEV_CAP)
    added_commit_train = append_split("commit", "train", commit_train)
    added_commit_dev = append_split("commit", "dev", commit_dev)

    train_texts = _admitted_train_texts()
    nlbse = build_nlbse24_holdout(train_texts)
    cc_holdout = build_commit_chronicle_holdout(frozen_texts, frozen_shas,
                                                train_texts)
    ace = build_acebench_pairs()

    file_entries = {}
    for rel in ("data/real/issue/train.jsonl", "data/real/issue/dev.jsonl",
                "data/real/commit/train.jsonl", "data/real/commit/dev.jsonl"):
        raw = (DATA.parent / rel).read_bytes()
        n = sum(1 for line in raw.decode("utf-8").splitlines() if line.strip())
        file_entries[rel] = (n, hashlib.sha256(raw).hexdigest())

    source_rows = [
        "| issue | GitBugs bug reports, 9 projects (raw dump "
        "data/incoming/external/gitbugs/, gitignored); Jira and Mozilla "
        "priority scales mapped to low/medium/high/critical | CC BY 4.0 "
        "(github.com/av9ash/gitbugs; Patil 2025, arXiv 2504.09651) | 2026-10-06 |",
        "| commit | CommitChronicle commits, bounded slice (raw dump "
        "data/incoming/external/commit-chronicle/commits-raw.jsonl); only "
        "per-record permissive licenses admitted | per-record license field; "
        "HF card 'other', Zenodo copy CC BY 4.0 | 2026-10-06 |",
        "| issue (held-out) | NLBSE'24 issue report classification, test split "
        "(raw CSV data/incoming/external/nlbse24/, gitignored) | no license "
        "file in nlbse2024/issue-report-classification; public GitHub issue "
        "text | 2026-10-06 |",
        "| no_tool_call (guard) | ACEBench Special split, should-abstain pairs "
        "(raw data/incoming/external/acebench/) | MIT (ACEBench/ACEBench, "
        "Microsoft) | 2026-10-06 |",
    ]

    def _stat(rel: str) -> tuple[int, str]:
        raw = (DATA.parent / rel).read_bytes()
        n = sum(1 for line in raw.decode("utf-8").splitlines() if line.strip())
        return n, hashlib.sha256(raw).hexdigest()

    nlbse_n, nlbse_sha = _stat("data/incoming/external/nlbse24/issues.jsonl")
    cc_n, cc_sha = _stat("data/incoming/external/phase1-ood/commit.jsonl")
    ace_n, ace_sha = _stat("data/incoming/external/bfcl/pairs-ace.jsonl")
    bullets = [
        f"- `data/incoming/external/nlbse24/issues.jsonl` — {nlbse_n} rows, "
        f"sha256 {nlbse_sha} (NLBSE'24 held-out).",
        f"- `data/incoming/external/phase1-ood/commit.jsonl` — {cc_n} rows, "
        f"sha256 {cc_sha} (project/sha-disjoint CommitChronicle held-out: the "
        "5 projects with fewer than 200 raw commits, excluded from train).",
        f"- `data/incoming/external/bfcl/pairs-ace.jsonl` — {ace_n} rows, "
        f"sha256 {ace_sha} (ACEBench Special should-abstain pairs).",
        f"- GitBugs appended: {len(issue_train)} issue train / "
        f"{len(issue_dev)} issue dev.",
        f"- CommitChronicle appended: {len(commit_train)} commit train / "
        f"{len(commit_dev)} commit dev.",
        "- Curated by `scripts/curate_data_phase1.py` (seed 20261006, "
        "hash-based split, frozen-set disjoint); raw dumps fetched by "
        "`scripts/fetch_data_phase1.py` and gitignored when bulky.",
    ]
    _update_provenance(file_entries, source_rows, bullets)

    print(f"gitbugs issue: +{added_issue_train} train, +{added_issue_dev} dev "
          f"({len(issue_rows)} candidates)")
    print(f"commit-chronicle: +{added_commit_train} train, "
          f"+{added_commit_dev} dev ({len(commit_rows)} candidates)")
    print(f"nlbse24 held-out: {nlbse} rows")
    print(f"commit held-out: {len(cc_holdout)} rows")
    print(f"acebench pairs: {sum(ace.values())} rows {ace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
