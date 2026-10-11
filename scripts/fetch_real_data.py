#!/usr/bin/env python3
"""Fetch and curate public real-world data for mivia-small-json.

Stdlib only. Deterministic: fixed order, seeded sampling, stable output.

Builds:
  data/real/issue/{train,dev}.jsonl   Hugging Face github-issues-dataset (MIT)
  data/real/commit/{train,dev}.jsonl  scikit-learn commits, GitHub API (BSD-3-Clause)
  data/wild/{issue,commit}.jsonl      frozen benchmark, never trained on
  data/real/PROVENANCE.md             sources, rules, counts, sha256

log_event (LogHub) is verified but NOT used: its license allows research or
academic work only. See the "Skipped source" section of PROVENANCE.md.

Privacy: only public text is kept. No author, email, username, or avatar
fields are stored. Email addresses in kept text are replaced with [email].

The GitHub commits endpoint does not return changed-file counts, so commit
gold rows carry no files_changed value.
"""
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SEED = 20261003
SCAN_ISSUES = 4000
PAGES_COMMITS = 15
WILD_N = 150
TRAIN_DEV = {"issue": (1200, 300), "commit": (1200, 300)}
PERMISSIVE = {"mit", "bsd-2-clause", "bsd-3-clause", "apache-2.0", "cc-by-4.0", "cc0-1.0"}
UA = {"User-Agent": "mivia-small-json-fetch/1.0"}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TRAILER_RE = re.compile(r"(?i)^\s*(co-authored-by|signed-off-by|reviewed-by|reported-by)\s*:")
PREFIX_TYPE = {"MNT": "chore", "ENH": "feat", "BUG": "fix", "FIX": "fix",
               "CI": "ci", "DOC": "docs", "DOCS": "docs", "BLD": "build",
               "PERF": "perf"}
ASCII_TEXT = re.compile(r"[\x20-\x7e\t]")


def get(url, tries=6):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=dict(UA, Accept="application/json"))
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                raise
            delay = float(e.headers.get("Retry-After") or 0) or 10.0 * (i + 1)
            delay = min(delay, 30.0)
            print("  http %s on %s ... retry in %.0fs" % (e.code, url[:80], delay))
            time.sleep(delay)
        except Exception as e:  # transient network errors: retry, then fail
            last = e
            time.sleep(2 * (i + 1))
    raise last


def http_json(url):
    return json.loads(get(url))


def scrub(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = "\n".join(line for line in text.split("\n")
                     if not TRAILER_RE.match(line))
    return EMAIL_RE.sub("[email]", text).strip()


def keep(raw):
    if not (10 <= len(raw) <= 1000):
        return False
    return sum(1 for c in raw if ASCII_TEXT.match(c)) / len(raw) >= 0.90


def collect_issues():
    recs, seen = [], set()
    base = ("https://datasets-server.huggingface.co/rows"
            "?dataset=sharjeelyunus%2Fgithub-issues-dataset"
            "&config=default&split=train&length=100&offset=")
    for off in range(0, SCAN_ISSUES, 100):
        rows = (http_json(base + str(off)).get("rows")) or []
        if not rows:
            break
        for entry in rows:
            row = entry.get("row") or {}
            title = str(row.get("title") or "").strip()
            body = str(row.get("body") or "").strip()
            if not title:
                continue
            raw = scrub(title + ("\n" + body[:300] if body else ""))
            if not keep(raw) or raw in seen:
                continue
            labels = row.get("labels")
            if isinstance(labels, str):
                try:
                    labels = json.loads(labels)
                except Exception:
                    labels = [s.strip() for s in labels.split(",") if s.strip()]
            if not isinstance(labels, list):
                labels = []
            labels = [str(x) for x in labels if str(x).strip()][:8]
            gold = {"title": title, "labels": labels}
            prio = str(row.get("priority") or "").strip().lower()
            sev = str(row.get("severity") or "").strip().lower()
            if prio in ("low", "medium", "high"):
                gold["priority"] = prio
            if sev == "critical":
                gold["priority"] = "critical"
            seen.add(raw)
            recs.append((raw, gold))
        time.sleep(0.3)
    return recs


def collect_commits():
    recs, seen = [], set()
    for page in range(1, PAGES_COMMITS + 1):
        d = http_json("https://api.github.com/repos/scikit-learn/scikit-learn"
                      "/commits?per_page=100&page=%d" % page)
        if not isinstance(d, list) or not d:
            break
        for c in d:
            sha = str(c.get("sha") or "")
            msg = scrub(str((c.get("commit") or {}).get("message") or ""))
            if not sha or not keep(msg) or sha in seen:
                continue
            seen.add(sha)
            gold = {"sha": sha, "message": msg}
            tok = msg.split("\n", 1)[0].split(":", 1)[0].strip().upper()
            if tok in PREFIX_TYPE:
                gold["commit_type"] = PREFIX_TYPE[tok]
            recs.append((msg, gold))
        time.sleep(0.2)
    return recs


def build(schema, source, recs):
    rng = random.Random("%s:%s" % (SEED, schema))
    idx = list(range(len(recs)))
    wild = rng.sample(idx, min(WILD_N, len(idx)))
    rest = [i for i in idx if i not in set(wild)]
    rng.shuffle(rest)
    n_train, n_dev = TRAIN_DEV[schema]
    picks = {"wild": wild, "train": rest[:n_train], "dev": rest[n_train:n_train + n_dev]}
    counts = {}
    for split, rows in picks.items():
        path = (os.path.join(DATA, "wild", schema + ".jsonl") if split == "wild"
                else os.path.join(DATA, "real", schema, split + ".jsonl"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i in rows:
                f.write(json.dumps({"source": source, "raw": recs[i][0],
                                    "gold": recs[i][1]}, ensure_ascii=False) + "\n")
        counts[os.path.relpath(path, ROOT).replace(os.sep, "/")] = len(rows)
    return counts


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(counts, lic, today):
    a = []
    a += ["# Real data provenance", "",
          "This file lists the public sources of the real data and the cleaning rules.",
          "`scripts/fetch_real_data.py` rebuilds every file below in the same way.", "",
          "## Sources", "",
          "| Schema | Source | License | Retrieved |",
          "|---|---|---|---|",
          "| issue | Hugging Face dataset [sharjeelyunus/github-issues-dataset]"
          "(https://huggingface.co/datasets/sharjeelyunus/github-issues-dataset) | %s | %s |"
          % (lic["issue"], today),
          "| commit | scikit-learn commit history, GitHub API"
          " (api.github.com/repos/scikit-learn/scikit-learn/commits) | %s | %s |"
          % (lic["commit"], today),
          "", "## Skipped source", "",
          "- log_event (LogHub, github.com/logpai/loghub): **not used**."
          " The repository license allows research or academic work only."
          " That is restrictive, so no log data is committed here.", "",
          "## Filters", "",
          "1. Keep records with 10 to 1000 characters.",
          "2. Remove duplicate records within each schema.",
          "3. Drop records that are not dominated by ASCII text characters.",
          "4. Sample with a fixed seed (%d). Re-runs give the same files." % SEED,
          "5. Split: first sample 150 wild benchmark rows per schema, then train and dev.",
          "6. Map commit prefixes (MNT, ENH, BUG, FIX, CI, DOC, BLD, PERF) to `commit_type`.",
          "   Leave `commit_type` out when the prefix is unknown.",
          "", "## Privacy", "",
          "1. Store only public text: title, body, labels, commit message, sha.",
          "2. Store no author, committer, username, avatar, or issue id fields.",
          "3. Replace email addresses in kept text with `[email]`.",
          "4. Remove name-bearing trailer lines, for example `Signed-off-by:`.",
          "5. Use only three keys per row: `source`, `raw`, `gold`.",
          "", "## Files", "",
          "| File | Records | sha256 |", "|---|---|---|"]
    for rel in sorted(counts):
        a.append("| `%s` | %d | %s |" % (rel, counts[rel], sha256(os.path.join(ROOT, rel))))
    a += ["", "## Note on `files_changed`", "",
          "The GitHub commits endpoint does not return changed-file counts.",
          "Commit gold rows therefore carry no `files_changed` value.", ""]
    with open(os.path.join(DATA, "real", "PROVENANCE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(a))


def main():
    today = os.environ.get("MIVIA_FETCH_DATE") or date.today().isoformat()
    hf = http_json("https://huggingface.co/api/datasets/sharjeelyunus/github-issues-dataset")
    hf_lic = str((hf.get("cardData") or {}).get("license") or "")
    if not hf_lic:
        for t in hf.get("tags") or []:
            if str(t).startswith("license:"):
                hf_lic = str(t).split(":", 1)[1]
                break
    gh = http_json("https://api.github.com/repos/scikit-learn/scikit-learn")
    gh_lic = str((gh.get("license") or {}).get("spdx_id") or "")
    lh = http_json("https://api.github.com/repos/logpai/loghub")
    lh_lic = str((lh.get("license") or {}).get("spdx_id") or "")
    lic = {"issue": hf_lic, "commit": gh_lic, "loghub": lh_lic}
    print("license check:", lic)
    counts = {}
    if hf_lic.lower() in PERMISSIVE:
        print("collecting issues ...")
        counts.update(build("issue", "hf:sharjeelyunus/github-issues-dataset", collect_issues()))
    else:
        print("SKIP issues: license %r is not permissive" % hf_lic)
    if gh_lic.lower() in PERMISSIVE:
        print("collecting commits ...")
        counts.update(build("commit", "gh:scikit-learn/scikit-learn", collect_commits()))
    else:
        print("SKIP commits: license %r is not permissive" % gh_lic)
    if not counts:
        print("ALL sources failed the license gate; nothing written.")
        sys.exit(1)
    write_provenance(counts, lic, today)
    total = 0
    for rel in sorted(counts):
        path = os.path.join(ROOT, rel)
        size = os.path.getsize(path)
        total += size
        print("%s: %d rows, %d bytes, sha256=%s" % (rel, counts[rel], size, sha256(path)[:12]))
    print("new data bytes:", total)


if __name__ == "__main__":
    main()
