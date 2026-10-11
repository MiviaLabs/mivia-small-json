#!/usr/bin/env python3
"""Collect + curate the 6 real-world routing eval slices (2026-10-08 task).

Slices (task table: realworld-routing-slices):

- issue-realworld-v2  real GitHub issues, varied wording, NOT template-shaped
                      -> issue            (GitHub REST, unauthenticated urllib)
- post-commit         public social posts mentioning a commit/push/PR merge
                      (task name ``tweet-commit``; X has no ToS-compliant
                      source, so these are Mastodon posts — recorded in
                      provenance)         -> commit | no_tool_call (flagged)
- chat-informal       real 1:1 informal messages (UCI SMS ham; WhatsApp-style
                      analog, see provenance) -> no_tool_call
- slack-snippets      public Slack export messages (HF spencer/software_slacks)
                                          -> no_tool_call | log_event (flagged)
- fa-ar-ood           Persian (news + Q&A) and Arabic (tweets)
                                          -> no_tool_call
- long-log-dumps      multi-thousand-char real log blocks (loghub-upstream)
                                          -> log_event | diagnostic (flagged)

Raw uncurated dumps land in data/incoming/realworld/raw-*.jsonl; curated
labeled slices land in data/wild/realworld/<slice>.jsonl with rows
    {"text", "expected", "source", "ambiguous": bool, "note"}
Eval-only: never train on these. Deterministic given the upstream corpora;
network collection is rate-limit friendly (sleep between calls, no gh CLI).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from html import unescape as _html_unescape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "incoming" / "realworld"
OUT = ROOT / "data" / "wild" / "realworld"
UA = {"User-Agent": "mivia-realworld-slices/1.0 (eval-set collection)"}

LOGHUB = ROOT / "data" / "incoming" / "loghub-upstream"


def http_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def seen_key(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def existing_hashes() -> set[str]:
    """Hashes of every title/body text already in the repo's corpora so the
    new slices are hash-disjoint from training + previously frozen data."""
    keys: set[str] = set()

    def eat(text: str | None) -> None:
        if text and len(text) >= 40:
            keys.add(seen_key(text[:600]))
            keys.add(seen_key(text))

    for path in [
        *sorted((ROOT / "data" / "real").rglob("*.jsonl")),
        ROOT / "data" / "wild" / "issue.jsonl",
        ROOT / "data" / "wild" / "commit.jsonl",
        ROOT / "data" / "wild" / "platform-logs.jsonl",
        ROOT / "data" / "incoming" / "external" / "nlbse" / "issues.jsonl",
        ROOT / "data" / "incoming" / "external" / "nlbse24" / "issues.jsonl",
        ROOT / "data" / "incoming" / "external" / "cleancodereview" / "comments.jsonl",
        ROOT / "data" / "incoming" / "external" / "gitbugs" / "gitbugs.jsonl",
    ]:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            raw = rec.get("raw") or rec.get("text") or rec.get("body") or ""
            if isinstance(raw, dict):
                raw = raw.get("body") or ""
            gold = rec.get("gold") or {}
            eat(raw)
            if isinstance(gold, dict):
                eat(gold.get("title"))
            eat(rec.get("title") if isinstance(rec.get("title"), str) else None)
    big = ROOT / "data" / "incoming" / "hf-issues-full.json"
    if big.is_file():
        try:
            for rec in json.loads(big.read_text(encoding="utf-8")):
                if isinstance(rec, dict):
                    eat(rec.get("title"))
                    eat(rec.get("body"))
        except (ValueError, MemoryError):
            pass
    return keys


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    return re.sub(r"[ \t]+", " ", " ".join(p.parts)).strip()


def html_unescape(text: str) -> str:
    return _html_unescape(text)


def write_raw(name: str, rows: list[dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"raw-{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  raw dump: {path.relative_to(ROOT)} ({len(rows)} rows)")


def write_slice(name: str, rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
    masked = 0
    for row in rows:
        new = EMAIL.sub("<email>", row["text"])
        if new != row["text"]:
            masked += 1
            row["text"] = new
            row["note"] = (row["note"] + "; " if row["note"] else "") \
                + "email masked"
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    extra = f", {masked} rows email-masked" if masked else ""
    print(f"  slice: {path.relative_to(ROOT)} ({len(rows)} rows{extra})")
    return path


# ---------------------------------------------------------------- issue-v2
BOILER = re.compile(
    r"(?i)^\s*(#+\s*)?(steps to reproduce|expected behavior|actual behavior|"
    r"what (version|operating system|did you do)|environment|reproduction "
    r"steps|your environment|system information|additional context|"
    r"description of the (issue|problem)|to reproduce)\b"
)
CHECKBOX = re.compile(r"^\s*-\s*\[[ x]\]", re.MULTILINE)

ISSUE_REPOS = [
    "django/django", "ansible/ansible", "scrapy/scrapy", "redis/redis",
    "curl/curl", "grafana/grafana", "matplotlib/matplotlib", "moby/moby",
    "envoyproxy/envoy", "ipython/ipython", "beetbox/beets",
    "webrtc-mirror",  # placeholder never used; real list below
]
ISSUE_REPOS = ISSUE_REPOS[:-1] + [
    "pandas-dev/pandas", "gradle/gradle", "jquery/jquery",
]


def collect_issue(budget: int = 260) -> list[dict]:
    """Real GitHub issues, prose-heavy, template-light, via unauthenticated
    REST (sleep between calls; gh CLI is known to hang, urllib is used)."""
    seen = existing_hashes()
    raw: list[dict] = []
    for repo in ISSUE_REPOS:
        url = (f"https://api.github.com/repos/{repo}/issues"
               f"?state=open&per_page=100&page=1&sort=created&direction=desc")
        try:
            batch = http_json(url)
        except Exception as exc:  # noqa: BLE001 — provenance note, keep going
            print(f"  {repo}: fetch error {exc}")
            continue
        time.sleep(2.5)
        for it in batch:
            if "pull_request" in it or not it.get("body"):
                continue
            raw.append({
                "repo": repo, "number": it["number"], "title": it["title"],
                "body": it["body"], "url": it["html_url"],
            })
        print(f"  {repo}: {len(batch)} items ({len(raw)} raw total)")
        if len(raw) >= budget * 3:
            break
    write_raw("issue", raw)

    rows, taken = [], 0
    for rec in raw:
        title, body = (rec["title"] or "").strip(), (rec["body"] or "").strip()
        text = f"{title}\n\n{body}".strip()
        if not (60 <= len(title) + len(body) <= 4200) or len(body) < 250:
            continue
        if CHECKBOX.search(body) and len(CHECKBOX.findall(body)) >= 2:
            continue  # template checklist
        boiler = sum(1 for line in body.splitlines() if BOILER.match(line))
        if boiler >= 3:
            continue  # template-shaped
        if re.match(r"(?i)^(issue\s*#|#\d+)", title):
            continue
        k = seen_key(text)
        if k in seen or seen_key(body) in seen:
            continue
        seen.add(k)
        rows.append({
            "text": text[:6000], "expected": "issue",
            "source": f"github:{rec['repo']}#{rec['number']}",
            "ambiguous": False, "note": "markdown body, non-template prose",
        })
        taken += 1
        if taken >= budget:
            break
    write_slice("issue-realworld-v2", rows)
    return rows


# ------------------------------------------------------------- post-commit
COMMIT_CONTENT = re.compile(
    r"(?i)\b(pushed|committed|merged|merge of|merged in|pull request|"
    r"pr #\d+|deployed to (main|master|prod(uction)?)|landed in)\b"
)
RELEASE_CONTENT = re.compile(
    r"(?i)\b(v?\d+\.\d+(\.\d+)?\s*(is out|is here|is live|released)|"
    r"released? v?\d|new (release|version)|just (shipped|released)|"
    r"changelog|release notes|shipped v?\d)\b"
)
COMMIT_PASSING = re.compile(r"(?i)\b(commit|commits|git|github|gitlab)\b")
COMMIT_NEGATIVE = re.compile(
    r"(?i)(\bcve[- ]?\d|vulnerab|patch (your|now)|threat|malware|"
    r"stars?\s*·|fastest-growing|tutorial|how to\b|guide:|course|"
    r"new face|top contributor|health\s*[:\d]|breach|extortion|"
    r"attackers|credential)"
)

MASTO_TAGS = ["git", "gitlab", "forgejo", "github", "devops",
              "100DaysOfCode", "opensource"]


# Manual review overrides (2026-10-08, curator pass over every commit-labeled
# row + a sample of the rest). Matched as normalized substring of the text.
OVERRIDES: list[tuple[str, str, bool, str]] = [
    # demotions: git vocabulary but the post is a blog post, tool tip, quiz,
    # product changelog, pricing post or incident report — not a push
    ("new blog post “maybe a better way to track", "no_tool_call", True,
     "blog post about release history, not a push"),
    ("til about `nbstripout`", "no_tool_call", True,
     "tool tip, not a commit event"),
    ("tool called difit", "no_tool_call", True,
     "tool review, pushing only mentioned"),
    ("answer three questions — pushed or not", "no_tool_call", True,
     "tool promo quiz"),
    ("qwen38flash", "no_tool_call", True,
     "self-hosted server update log, not a git push"),
    ("was getting consistently a 500 error", "no_tool_call", True,
     "incident report about a forge, not a commit"),
    ("selected models in github copilot deprecated", "no_tool_call", True,
     "SaaS product changelog"),
    ("accessibility statements highlighted on repository overview",
     "no_tool_call", True, "SaaS product changelog"),
    ("cpanel's per-account pricing", "no_tool_call", True,
     "hosting pricing post; 'pushed' is not a git push"),
    ("git knowledge check", "no_tool_call", True,
     "git quiz, not a commit event"),
    # promotions: release/merge news whose content is the shipped change
    ("gitea v28.0.0", "commit", False, "release announcement, shipped change is the content"),
    ("localai v4.11.0", "commit", True, "multi-project release digest"),
    ("يقدم الإصدار 2.56", "commit", False, "git release news (Arabic)"),
    ("highlights from git 2.56", "commit", True, "release news link post"),
    ("git 2.56 lansat", "commit", False, "git release news (Romanian)"),
    ("latest git release brings a dedicated", "commit", False, "git release news"),
    ("gitea 28.1.0 is out", "commit", False, "release announcement"),
]


def collect_commit(budget: int = 200) -> list[dict]:
    """Mastodon public timeline posts mentioning commit/push/merge activity.

    X/Twitter has no ToS-compliant source, so the task's ``tweet-commit``
    slice is built from Mastodon posts instead; the rename and rationale are
    recorded in the provenance doc. Bot accounts and link-dominant reposts
    are dropped; posts where the commit/push is the CONTENT are labeled
    ``commit``, passing mentions get ``no_tool_call`` + ``ambiguous``.
    """
    seen = existing_hashes()
    local_keys: set[str] = set()
    raw: list[dict] = []
    for tag in MASTO_TAGS:
        url = (f"https://mastodon.social/api/v1/timelines/tag/{tag}"
               f"?limit=40")
        pages = 0
        while pages < 6 and len(raw) < 1200:
            try:
                batch = http_json(url)
            except Exception as exc:  # noqa: BLE001
                print(f"  #{tag}: fetch error {exc}")
                break
            if not batch:
                break
            for post in batch:
                acct = post["account"]["acct"]
                if post.get("bot") or "bot" in acct.split("@")[0].lower():
                    continue
                text = strip_html(post.get("content", ""))
                if not (80 <= len(text) <= 1400):
                    continue
                if not COMMIT_PASSING.search(text):
                    continue
                raw.append({"acct": acct, "text": text,
                            "url": post.get("url", ""), "tag": tag})
            url = (f"https://mastodon.social/api/v1/timelines/tag/{tag}"
                   f"?limit=40&max_id={batch[-1]['id']}")
            pages += 1
            time.sleep(0.6)
        print(f"  #{tag}: cumulative raw {len(raw)}")
    # dedup by content
    deduped = []
    for rec in raw:
        k = seen_key(rec["text"])
        if k in local_keys:
            continue
        local_keys.add(k)
        deduped.append(rec)
    write_raw("commit", deduped)

    rows = []
    for rec in deduped:
        text = rec["text"]
        k = seen_key(text)
        if k in seen:
            continue
        seen.add(k)
        ntext = norm(text)
        override = next((o for o in OVERRIDES if o[0] in ntext), None)
        if override:
            _, expected, ambiguous, note = override
        elif COMMIT_NEGATIVE.search(text) or text.count("http") > 3:
            expected, ambiguous, note = "no_tool_call", True, \
                "git vocabulary, but the post is about security news, " \
                "rankings, tutorials or links — not a commit/push"
        elif COMMIT_CONTENT.search(text) or RELEASE_CONTENT.search(text):
            expected, ambiguous, note = "commit", False, \
                "commit/push/merge/release is the content of the post"
            if len(text) > 900:
                ambiguous, note = True, "commit mention in longer post"
        else:
            expected, ambiguous = "no_tool_call", True
            note = "commit/git mentioned but not the content"
        rows.append({
            "text": text, "expected": expected,
            "source": f"mastodon:{rec['acct']}", "ambiguous": ambiguous,
            "note": note,
        })
    # keep the label mix honest: cap no_tool_call so commit rows survive
    commit_rows = [r for r in rows if r["expected"] == "commit"]
    other_rows = [r for r in rows if r["expected"] != "commit"][:budget - len(commit_rows)]
    rows = commit_rows + other_rows
    write_slice("post-commit", rows)
    return rows


# ----------------------------------------------------------- chat-informal
PHONE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)")


def mask_phones(text: str) -> tuple[str, bool]:
    masked = PHONE.sub("<phone>", text)
    return masked, masked != text


def collect_chat(budget: int = 170) -> list[dict]:
    """UCI SMS Spam Collection, ham only (label 0): real 1:1 informal
    messages — the WhatsApp-style analog, since no clean anonymized real
    WhatsApp corpus with a usable license was found on HF (provenance)."""
    seen = existing_hashes()
    raw: list[dict] = []
    offset = 0
    while offset < 5600:
        url = ("https://datasets-server.huggingface.co/rows?dataset="
               "ucirvine%2Fsms_spam&config=plain_text&split=train"
               f"&offset={offset}&length=100")
        try:
            batch = http_json(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  sms offset {offset}: fetch error {exc}")
            break
        rows_ = batch.get("rows", [])
        if not rows_:
            break
        for r in rows_:
            raw.append({"sms": r["row"]["sms"], "label": r["row"]["label"]})
        offset += 100
    write_raw("chat", raw)

    ham = [r for r in raw if r["label"] == 0]
    rows = []
    for rec in ham:
        text = rec["sms"].strip()
        if not (20 <= len(text) <= 600):
            continue
        text, masked = mask_phones(text)
        k = seen_key(text)
        if k in seen:
            continue
        seen.add(k)
        note = "real 1:1 SMS (ham); phone masked" if masked else "real 1:1 SMS (ham)"
        rows.append({
            "text": text, "expected": "no_tool_call",
            "source": "hf:ucirvine/sms_spam (ham)", "ambiguous": False,
            "note": note,
        })
        if len(rows) >= budget:
            break
    write_slice("chat-informal", rows)
    return rows


# ------------------------------------------------------------ slack-snips
STACKY = re.compile(
    r"(?i)(traceback|exception|uncaught|internal error|error:|failed:|"
    r"fatal\b|panic:|exit code|callstack|segmentation fault|aborted|"
    r"stack trace|compilation failed)"
)


def collect_slack(budget: int = 170) -> list[dict]:
    seen = existing_hashes()
    raw: list[dict] = []
    offset = 0
    while offset < 2400:
        url = ("https://datasets-server.huggingface.co/rows?dataset="
               "spencer%2Fsoftware_slacks&config=default&split=train"
               f"&offset={offset}&length=100")
        batch = None
        for attempt in range(4):
            try:
                batch = http_json(url)
                break
            except Exception as exc:  # noqa: BLE001
                wait = 20 * (attempt + 1)
                print(f"  slack offset {offset}: {exc}; retry in {wait}s")
                time.sleep(wait)
        if batch is None:
            break
        rows_ = batch.get("rows", [])
        if not rows_:
            break
        for r in rows_:
            raw.append({"workspace": r["row"]["workspace"],
                        "channel": r["row"]["channel"],
                        "text": r["row"]["text"]})
        offset += 100
        time.sleep(1.0)
    if not raw:
        raise SystemExit("slack collection failed: no rows fetched, "
                         "existing slice left untouched")
    write_raw("slack", raw)

    pool = []
    for rec in raw:
        text = html_unescape(rec["text"]).strip()
        if not (25 <= len(text) <= 1500):
            continue
        if text.count("http") > len(text) / 60:
            continue  # link dump
        k = seen_key(text)
        if k in seen:
            continue
        seen.add(k)
        err_in_fence = any(STACKY.search(block)
                           for block in re.findall(r"```(.*?)```", text, re.S))
        err_anywhere = bool(STACKY.search(text))
        multiline = text.count("\n") >= 3
        if err_in_fence or (err_anywhere and multiline):
            expected, ambiguous, note = "log_event", False, "pasted error/log text"
        elif err_anywhere:
            expected, ambiguous = "no_tool_call", True
            note = "mentions an error but reads as work chatter"
        else:
            expected, ambiguous, note = "no_tool_call", False, "work chatter"
        pool.append({
            "text": text, "expected": expected,
            "source": f"hf:spencer/software_slacks:{rec['workspace']}/{rec['channel']}",
            "ambiguous": ambiguous, "note": note,
        })
    # stratified selection: keep every pasted-error row, fill with chatter
    rows = ([r for r in pool if r["expected"] == "log_event"]
            + [r for r in pool if r["expected"] != "log_event"])
    rows = rows[:budget]
    write_slice("slack-snippets", rows)
    return rows


# --------------------------------------------------------------- fa-ar-ood
def _ds_rows(dataset: str, config: str, split: str, offsets: int) -> list[dict]:
    out: list[dict] = []
    for offset in range(0, offsets, 100):
        url = ("https://datasets-server.huggingface.co/rows?dataset="
               f"{urllib.parse.quote(dataset, safe='')}&config={config}"
               f"&split={split}&offset={offset}&length=100")
        batch = None
        for attempt in range(4):
            try:
                batch = http_json(url)
                break
            except Exception as exc:  # noqa: BLE001
                wait = 15 * (attempt + 1)
                print(f"  {dataset} offset {offset}: {exc}; retry in {wait}s")
                time.sleep(wait)
        if batch is None:
            continue
        rows_ = batch.get("rows", [])
        if not rows_:
            break
        out.extend(r["row"] for r in rows_)
        time.sleep(0.5)
    return out


def collect_ood() -> list[dict]:
    seen = existing_hashes()
    fa_rows: list[dict] = []
    news = _ds_rows("RohanAiLab/persian_news_dataset", "default", "train", 300)
    for rec in news:
        text = (rec.get("text") or "").strip()
        if 120 <= len(text) <= 1500:
            fa_rows.append({"text": text, "expected": "no_tool_call",
                            "source": "hf:RohanAiLab/persian_news_dataset",
                            "ambiguous": False, "note": "Persian news prose"})
    qa = _ds_rows("Kamtera/Persian-conversational-dataset", "default", "train", 200)
    for rec in qa:
        text = (rec.get("question") or "").strip()
        if 40 <= len(text) <= 600:
            fa_rows.append({"text": text, "expected": "no_tool_call",
                            "source": "hf:Kamtera/Persian-conversational-dataset",
                            "ambiguous": False,
                            "note": "Persian conversational Q&A"})
    ar_rows: list[dict] = []
    tweets = _ds_rows("pain/Arabic-Tweets", "default", "train", 400)
    for rec in tweets:
        text = (rec.get("text") or "").strip()
        if 30 <= len(text) <= 600:
            ar_rows.append({"text": text, "expected": "no_tool_call",
                            "source": "hf:pain/Arabic-Tweets",
                            "ambiguous": False, "note": "Arabic tweet"})
    write_raw("fa", fa_rows)
    write_raw("ar", ar_rows)

    def finalize(rows: list[dict], cap: int) -> list[dict]:
        out = []
        for row in rows:
            k = seen_key(row["text"])
            if k in seen:
                continue
            seen.add(k)
            out.append(row)
            if len(out) >= cap:
                break
        return out

    fa_final = finalize(fa_rows, 160)
    ar_final = finalize(ar_rows, 160)
    write_slice("fa-ar-ood-fa", fa_final)
    write_slice("fa-ar-ood-ar", ar_final)
    return fa_final + ar_final


# --------------------------------------------------------- long-log-dumps
TRACE = re.compile(
    r"(?im)(^\s+at \S+\(|exception|traceback|caused by:|error:|fatal|"
    r"panic:|segfault|segmentation fault|core dump|assertion failed|"
    r"\w+\.c:\d+|\w+\.java:\d+)"
)


def collect_logs(budget: int = 55) -> list[dict]:
    """Windows of 5k-12k consecutive chars from the committed loghub
    upstream dumps (real systems: supercomputer, Android, HPC, sshd, ...).
    diagnostic when the window is stack-trace/compiler dominated, else
    log_event; mixed windows flagged ambiguous."""
    systems = ["BGL", "Thunderbird", "HealthApp", "Android", "HPC",
               "OpenSSH", "Mac", "Windows", "Zookeeper", "Spark", "Hadoop",
               "Linux", "Apache", "Proxifier", "OpenStack", "HDFS"]
    files = []
    for sysname in systems:
        for path in sorted((LOGHUB / sysname).glob("*.log")):
            files.append(path)
    if not files:
        raise SystemExit("loghub-upstream dumps missing")
    rows: list[dict] = []
    per_file = max(1, budget // len(files) + 1)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) < 12000:
            continue
        taken = 0
        step = max(30000, len(text) // (per_file * 3))
        pos = 0
        while taken < per_file and len(rows) < budget:
            start = min(pos, max(0, len(text) - 12000))
            window = text[start:start + 11000]
            # snap to line boundaries
            nl = window.find("\n")
            end = window.rfind("\n", 0, len(window))
            if nl > 0 and end > nl:
                window = window[nl + 1:end]
            if not 5000 <= len(window) <= 12000:
                pos += step
                if start + 12000 >= len(text):
                    break
                continue
            nlines = window.count("\n") + 1
            traces = len(TRACE.findall(window))
            sysname = path.parent.name
            if traces >= max(3, nlines // 10):
                expected, note = "diagnostic", "stack/exception-dense window"
                ambiguous = traces < nlines // 4
            else:
                expected, note = "log_event", "burst of real system log lines"
                ambiguous = traces >= 2
            rows.append({
                "text": window, "expected": expected,
                "source": f"loghub:{sysname}/{path.name}:{start}",
                "ambiguous": ambiguous, "note": note,
            })
            taken += 1
            pos += step
            if start + 12000 >= len(text):
                break
        if len(rows) >= budget:
            break
    write_raw("logs", rows)
    write_slice("long-log-dumps", rows)
    return rows


SLICES = {
    "issue": collect_issue,
    "commit": collect_commit,
    "chat": collect_chat,
    "slack": collect_slack,
    "ood": collect_ood,
    "logs": collect_logs,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", choices=[*SLICES, "all"], default="all")
    args = ap.parse_args()
    names = list(SLICES) if args.slice == "all" else [args.slice]
    for name in names:
        print(f"== slice: {name}")
        rows = SLICES[name]()
        sizes = {}
        for row in rows:
            sizes[(row["expected"], row["ambiguous"])] = \
                sizes.get((row["expected"], row["ambiguous"]), 0) + 1
        print("  labels:", dict(sorted(sizes.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
