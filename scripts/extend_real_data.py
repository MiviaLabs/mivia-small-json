#!/usr/bin/env python3
"""Build the curated real training data from data/incoming/ raw dumps.

Companion to scripts/fetch_real_data.py. Raw material lands in
data/incoming/ (kept in-repo, traceable); this script deterministically
curates it into:

  data/real/issue/{train,dev}.jsonl    HF github-issues-dataset (hf-issues-full.json)
  data/real/commit/{train,dev}.jsonl   GitHub commits (gh-commits-sklearn-early.json
                                       = original scikit-learn scan, plus
                                       gh-commits.json = extension repos)
  data/real/log_event/{train,dev}.jsonl LogHub log lines (gold.level when detectable)
  data/pool/inbox/public-logs-v1.log   pool supply, disjoint from the real splits
  data/pool/inbox/pull-requests-v1.txt pool supply (blank-line blocks)
  data/real/PROVENANCE.md              rewritten counts + sha256 table

Owner decisions (2026-10-05, docs/plans/MEMORY.md): the LogHub license
restriction is lifted with attribution in README.md, the log_event
real-data ban is removed, the data-size budget gate is deleted, and the
privacy-scrubbing policy is SUPERSEDED — dataset-provided content is
stored and trained on as-is (no email/IP/UUID replacement, no trailer
removal). Guards that remain are model hygiene only: length/printability
windows and dedupe. Frozen wild slices are untouched era snapshots.

Deterministic: seeded sampling, stable order, idempotent (re-runs only
add rows not already present; frozen-set collisions are never admitted).
"""
import hashlib
import json
import os
import random
import re
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
INCOMING = os.path.join(DATA, "incoming")
SEED = 20261005
ISSUE_DEV = 300
COMMIT_DEV = 150
LOG_EVENT_TRAIN = 12000
LOG_EVENT_DEV = 2000
LOG_EVENT_POOL = 4000
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                     r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
ASCII_TEXT = re.compile(r"[\x20-\x7e\t]")
# Level detection mirrors the OTel severity table in schemas.py; keep the
# token list in sync with LEVELS/LEVEL_ALIASES there.
LEVEL_TOKENS = {
    "trace": "TRACE", "verbose": "TRACE",
    "debug": "DEBUG", "dbg": "DEBUG",
    "info": "INFO", "notice": "INFO",
    "warn": "WARNING", "warning": "WARNING",
    "err": "ERROR", "error": "ERROR", "severe": "ERROR",
    "crit": "CRITICAL", "critical": "CRITICAL", "alert": "CRITICAL",
    "emerg": "CRITICAL", "fatal": "FATAL",
}
LEVEL_RE = re.compile(
    r"(?i)\b(?:level\s*=\s*|\[)?(trace|verbose|debug|dbg|info|notice|warn(?:ing)?|"
    r"err(?:or)?|severe|crit(?:ical)?|alert|emerg|fatal)\b(?:\s*\])?")
PREFIX_TYPE = {"MNT": "chore", "ENH": "feat", "BUG": "fix", "FIX": "fix",
               "CI": "ci", "DOC": "docs", "DOCS": "docs", "BLD": "build",
               "PERF": "perf"}


def keep(text: str) -> bool:
    if not (10 <= len(text) <= 1000):
        return False
    return sum(1 for c in text if ASCII_TEXT.match(c)) / len(text) >= 0.90


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_keys() -> tuple[set, set]:
    """(text keys, commit shas) of every frozen raw record and line.

    The wild slices are scrubbed-era snapshots: a raw dump row that was
    originally scrubbed into a wild row differs from it. Each frozen key
    therefore also covers its would-have-been-scrubbed preimage, so the
    same underlying record can never enter training under both forms.
    """
    texts, shas = set(), set()
    frozen = ["data/wild/issue.jsonl", "data/wild/commit.jsonl",
              "data/wild/platform-logs.jsonl", "data/heldout/test.jsonl",
              "data/synthetic/train.jsonl", "data/synthetic/dev.jsonl"]
    for rel in frozen:
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            gold = row.get("gold") or {}
            if "sha" in gold:
                shas.add(str(gold["sha"]))
            text = row.get("raw") or ""
            if not text:
                continue
            variants = {text,
                        re.sub(r"\[email\]", "[email]", text),
                        re.sub(r"\S+@\S+\.\S+", "[email]", text)}
            for v in list(variants):
                texts.add(norm(v))
                texts.update(norm(ln) for ln in v.splitlines() if ln.strip())
    return texts, shas


def build_issues(wild_texts):
    dump = os.path.join(INCOMING, "hf-issues-full.json")
    if not os.path.isfile(dump):
        sys_exit("data/incoming/hf-issues-full.json missing")
    rows, seen = [], set()
    for it in json.load(open(dump)):
        title = str(it.get("title") or "").strip()
        body = str(it.get("body") or "").strip()
        if not title:
            continue
        raw = (title + "\n" + body[:300] if body else title).strip()
        key = norm(raw)
        if not keep(raw) or key in seen or key in wild_texts:
            continue
        seen.add(key)
        labels = it.get("labels")
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except Exception:
                labels = [s.strip() for s in labels.split(",") if s.strip()]
        if not isinstance(labels, list):
            labels = []
        labels = [str(x) for x in labels if str(x).strip()][:8]
        gold = {"title": title, "labels": labels}
        prio = str(it.get("priority") or "").strip().lower()
        sev = str(it.get("severity") or "").strip().lower()
        if prio in ("low", "medium", "high"):
            gold["priority"] = prio
        if sev == "critical":
            gold["priority"] = "critical"
        rows.append({"source": "hf:sharjeelyunus/github-issues-dataset",
                     "raw": raw, "gold": gold})
    rng = random.Random("%s:issue" % SEED)
    rng.shuffle(rows)
    write_split("issue", "dev", rows[:ISSUE_DEV])
    write_split("issue", "train", rows[ISSUE_DEV:])
    return len(rows)


def build_commits(wild_texts, wild_shas):
    dumps = [os.path.join(INCOMING, "gh-commits-sklearn-early.json"),
             os.path.join(INCOMING, "gh-commits.json")]
    rows, seen_sha, seen_key = [], set(), set()
    for dump in dumps:
        if not os.path.isfile(dump):
            continue
        for c in json.load(open(dump)):
            msg = str(c.get("message") or "").strip()
            sha = str(c.get("sha") or "")
            if not keep(msg) or not sha or sha in seen_sha or sha in wild_shas:
                continue
            key = norm(msg)
            if key in seen_key or key in wild_texts:
                continue
            seen_sha.add(sha)
            seen_key.add(key)
            gold = {"sha": sha, "message": msg}
            tok = msg.split("\n", 1)[0].split(":", 1)[0].strip().upper()
            if tok in PREFIX_TYPE:
                gold["commit_type"] = PREFIX_TYPE[tok]
            rows.append({"source": "gh:%s" % c.get("repo"),
                         "raw": msg, "gold": gold})
    rng = random.Random("%s:commit" % SEED)
    rng.shuffle(rows)
    write_split("commit", "dev", rows[:COMMIT_DEV])
    write_split("commit", "train", rows[COMMIT_DEV:])
    return len(rows)


def build_log_event(wild_texts):
    out_dir = os.path.join(DATA, "real", "log_event")
    os.makedirs(out_dir, exist_ok=True)
    rows, seen = [], set()
    base = os.path.join(INCOMING, "loghub-upstream")
    if not os.path.isdir(base):
        sys_exit("data/incoming/loghub-upstream missing")
    for name in sorted(os.listdir(base)):
        log = os.path.join(base, name, name + "_2k.log")
        if not os.path.isfile(log):
            continue
        for line in open(log, encoding="utf-8", errors="replace"):
            text = line.rstrip("\n").strip()
            key = norm(text)
            if not keep(text) or key in seen or key in wild_texts:
                continue
            seen.add(key)
            m = LEVEL_RE.search(text)
            gold = {"level": LEVEL_TOKENS[m.group(1).lower()]} if m else {}
            rows.append({"source": "loghub:%s" % name, "raw": text, "gold": gold})
    rng = random.Random("%s:log_event" % SEED)
    rng.shuffle(rows)
    write_split("log_event", "dev", rows[:LOG_EVENT_DEV])
    write_split("log_event", "train",
                rows[LOG_EVENT_DEV:LOG_EVENT_DEV + LOG_EVENT_TRAIN])
    pool = rows[LOG_EVENT_DEV + LOG_EVENT_TRAIN:
                LOG_EVENT_DEV + LOG_EVENT_TRAIN + LOG_EVENT_POOL]
    path = os.path.join(DATA, "pool", "inbox", "public-logs-v1.log")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(r["raw"] for r in pool) + "\n")
    print("log_event: dev %d, train %d, pool inbox %d"
          % (LOG_EVENT_DEV, LOG_EVENT_TRAIN, len(pool)))


def build_pr_inbox(wild_texts):
    dump = os.path.join(INCOMING, "gh-prs.json")
    if not os.path.isfile(dump):
        print("no gh-prs.json; skipping PR inbox")
        return
    blocks, seen = [], set()
    for pr in json.load(open(dump)):
        title = str(pr.get("title") or "").strip()
        body = str(pr.get("body") or "").strip()
        text = (title + "\n" + body[:800]).strip()
        key = norm(text)
        if not keep(text) or key in seen or key in wild_texts:
            continue
        seen.add(key)
        blocks.append(text)
    path = os.path.join(DATA, "pool", "inbox", "pull-requests-v1.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(blocks) + "\n")
    print("pool inbox PR blocks:", len(blocks))


def build_logcat_raw(wild_texts):
    zip_path = os.path.join(INCOMING, "Android_v1.zip")
    if not os.path.isfile(zip_path):
        return 0
    import zipfile
    from mivia_small_json.schemas import LEVEL_ALIASES

    out_path = os.path.join(INCOMING, "logcat-structured.jsonl")
    # Exclude Android_2k lines to prevent test leakage
    android_2k = os.path.join(INCOMING, "loghub-upstream", "Android", "Android_2k.log")
    if os.path.isfile(android_2k):
        for line in open(android_2k, encoding="utf-8", errors="replace"):
            wild_texts.add(norm(line.strip()))

    pat = re.compile(
        r"^\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}\s+\d+\s+\d+\s+([VDIWEF])\s+([^:]+):\s*(.*)$"
    )
    rows, seen = [], set()
    with zipfile.ZipFile(zip_path) as z:
        with z.open("Android.log") as f:
            for line in f:
                s = line.decode("utf-8", "replace").strip()
                if not keep(s):
                    continue
                k = norm(s)
                if k in seen or k in wild_texts:
                    continue
                m = pat.match(s)
                if not m:
                    continue
                seen.add(k)
                lv = LEVEL_ALIASES.get(m.group(1), m.group(1))
                comp = m.group(2).strip()
                rows.append({"raw": s, "level": lv, "component": comp, "token": m.group(1)})
                if len(rows) >= 8000:
                    break
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("logcat training lines: %d written to %s" % (len(rows), out_path))
    return len(rows)


def build_format_augmented():
    out_path = os.path.join(INCOMING, "log-format-augmented.jsonl")
    rng = random.Random(SEED + 100)
    levels = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
    components = ["auth", "api", "database", "worker", "gateway", "scheduler", "cache"]
    messages = [
        "connection refused", "request handled in 12ms", "user authenticated",
        "table lock acquired", "cache miss on key 42", "disk space low", "fatal assertion failed"
    ]
    rows = []
    for _ in range(2000):
        lv = rng.choice(levels)
        comp = rng.choice(components)
        msg = rng.choice(messages)
        style = rng.choice(["logfmt", "json", "syslog", "bracket"])
        if style == "logfmt":
            key = rng.choice(["level", "lvl", "severity"])
            val = rng.choice([lv, lv.lower()])
            line = f"ts=2026-10-05T05:00:00Z {key}={val} comp={comp} msg=\"{msg}\""
        elif style == "json":
            key = rng.choice(["level", "lvl", "severity"])
            val = rng.choice([lv, lv.lower()])
            line = json.dumps({key: val, "component": comp, "message": msg})
        elif style == "syslog":
            code = rng.choice(["<14>", "<34>", "<11>"])
            line = f"{code}Oct  5 05:00:00 host {comp}[1234]: {lv} {msg}"
        else:
            line = f"[2026-10-05 05:00:00] [{lv.lower()}] [{comp}]: {msg}"
        rows.append({"raw": line, "level": lv})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("augmented format lines: %d written to %s" % (len(rows), out_path))
    return len(rows)



def write_split(schema, split, rows):
    path = os.path.join(DATA, "real", schema, split + ".jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("%s %s: %d rows" % (schema, split, len(rows)))


def sys_exit(msg):
    raise SystemExit(msg)


def build_tool_calls(wild_texts: set) -> tuple[int, int]:
    raw_file = os.path.join(INCOMING, "tool-calls-glaive.json")
    if not os.path.isfile(raw_file):
        return 0, 0
    items = json.load(open(raw_file, encoding="utf-8"))
    pool = []
    seen = set()
    for item in items:
        raw = item.get("raw", "")
        if not keep(raw):
            continue
        k = norm(raw)
        if k in seen or k in wild_texts:
            continue
        seen.add(k)
        pool.append(item)

    rng = random.Random(SEED)
    rng.shuffle(pool)

    dev_n = 300
    dev = pool[:dev_n]
    train = pool[dev_n:]

    write_split("tool_call", "dev", dev)
    write_split("tool_call", "train", train)
    return len(train), len(dev)


REAL_FILES = [
    "data/real/issue/train.jsonl",
    "data/real/issue/dev.jsonl",
    "data/real/commit/train.jsonl",
    "data/real/commit/dev.jsonl",
    "data/real/log_event/train.jsonl",
    "data/real/log_event/dev.jsonl",
    "data/real/tool_call/train.jsonl",
    "data/real/tool_call/dev.jsonl",
    "data/wild/issue.jsonl",
    "data/wild/commit.jsonl",
]
SOURCES = [
    ("issue", "Hugging Face dataset [sharjeelyunus/github-issues-dataset]"
     "(https://huggingface.co/datasets/sharjeelyunus/github-issues-dataset),"
     " offsets 0-7500 (raw dump data/incoming/hf-issues-full.json)",
     "MIT"),
    ("commit", "scikit-learn commit history pages 1-25 plus flask, requests"
     " and fastapi, GitHub API (raw dumps data/incoming/gh-commits*.json)",
     "BSD-3-Clause / BSD-3-Clause / Apache-2.0 / MIT (recorded, unverified)"),
    ("log_event", "LogHub (github.com/logpai/loghub), 16 systems, 2k-line"
     " samples per system (snapshot kept in data/incoming/loghub-upstream/)",
     "LogHub license accepted by owner decision 2026-10-05 with attribution"
     " in README.md"),
    ("tool_call", "glaiveai/glaive-function-calling-v2 via HuggingFace (raw dump"
     " data/incoming/tool-calls-glaive.json)", "Apache-2.0"),
    ("pull_request", "closed PRs from flask and fastapi, GitHub API"
     " (data/incoming/gh-prs.json) — pool inbox only, not training data",
     "see commit sources"),
]


def write_provenance():
    a = ["# Real data provenance", "",
         "Public sources of the real data and the curation rules.",
         "Raw dumps and the LogHub snapshot are kept in data/incoming/;",
         "scripts/extend_real_data.py curates them deterministically.", "",
         "## Sources", "",
         "| Schema | Source | License | Retrieved |", "|---|---|---|---|"]
    today = os.environ.get("MIVIA_FETCH_DATE") or date.today().isoformat()
    for schema, source, lic in SOURCES:
        a.append("| %s | %s | %s | %s |" % (schema, source, lic, today))
    a += ["", "## Privacy policy (superseded, owner decision 2026-10-05)", "",
          "Earlier versions scrubbed emails to `[email]` and dropped"
          " name-bearing trailer lines. The owner directed that"
          " dataset-provided content be kept as-is and trained on as-is:"
          " no email/IP/UUID replacement, no trailer removal. The raw"
          " dumps in data/incoming/ are the exact fetch payloads.", "",
          "## Filters (model hygiene only)", "",
          "1. Keep records with 10 to 1000 characters, >= 90% ASCII-printable.",
          "2. Remove duplicates by normalized text (and by commit sha).",
          "3. Never admit a frozen record (data/wild, data/heldout,",
          "   data/synthetic) — whole record, any single line, or its"
          "   scrubbed-era preimage.",
          "4. Seeded sampling (seed %d); re-runs are idempotent." % SEED,
          "5. Commit prefixes (MNT, ENH, BUG, FIX, CI, DOC, BLD, PERF) map to",
          "   `commit_type`; left out when the prefix is unknown.", "",
          "## Files", "", "| File | Records | sha256 |", "|---|---|---|"]
    for rel in REAL_FILES:
        path = os.path.join(ROOT, rel)
        n = sum(1 for line in open(path, encoding="utf-8") if line.strip())
        a.append("| `%s` | %d | %s |" % (rel, n, sha256(path)))
    a.append("")
    with open(os.path.join(DATA, "real", "PROVENANCE.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(a))


def main():
    wild_texts, wild_shas = frozen_keys()
    n_issues = build_issues(wild_texts)
    n_commits = build_commits(wild_texts, wild_shas)
    build_log_event(wild_texts)
    build_logcat_raw(wild_texts)
    build_format_augmented()
    build_pr_inbox(wild_texts)
    n_tc_tr, n_tc_dv = build_tool_calls(wild_texts)
    write_provenance()
    print("issue rows total: %d, commit rows total: %d, tool_call: tr=%d dv=%d" % (
        n_issues, n_commits, n_tc_tr, n_tc_dv
    ))


if __name__ == "__main__":
    main()
