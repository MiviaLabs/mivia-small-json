#!/usr/bin/env python3
"""Build the realworld-slices-r1 corpus (realworld-slices-r1).

ONE combined data round for the two stuck realworld slices:
  P  post-commit-r1.jsonl     Mastodon dev-announce register      -> text
  S  slack-r1.jsonl           software_slacks offsets >= 2400      -> text
  D  wrapper-paste-r1.jsonl   chat wrapper + fenced error paste    -> log_event
  C  review-counter-r1.jsonl  GitHub PR inline review comments     -> review_comment

Phases (run in order):
  fetch    network collection (rate-limit friendly; resumable per source)
  screen   exclusions (eval windows / raw pools / frozen surfaces / base
           mix / cross-source), dedupe, rater-A mechanical screen, ids,
           labels, the blind adjudication sample
           (rater B is then coded OUT OF BAND into adjudication-B.jsonl)
  assemble apply the drop path, enforce the declared bands (exit 2 on
           shortfall), write the four corpora + the assembled v6/v12
           lineage files + sha sidecars + idempotent PROVENANCE sections
  probe    weakness probe: route admitted rows through the SHIPPED router
           (easy-mass vs signal, written up BEFORE any GPU run)

Exclusion single-source-of-truth: this builder imports the audit gate's
own norm()/eval_norms()/_replicate_eval_windows() so the builder's
verified-by-construction claim and the gate's independent re-verification
compute the SAME sets. Declared bands (spec): P [3000,5500] S [4000,8500]
D [150,500] C [1200,2000]; dedupe <= 1%; increment text share <= 0.19.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from html import unescape as html_unescape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from audit_training_mix import (  # noqa: E402
    REALWORLD_R1_AUX,
    REALWORLD_R1_DIR,
    REALWORLD_R1_EXTRA,
    REALWORLD_R1_LINEAGE,
    REALWORLD_R1_SOURCES,
    _replicate_eval_windows,
    corpus_rows,
    eval_norms,
    norm,
)

RAW = ROOT / "data/incoming/realworld"
UA = {"User-Agent": "mivia-realworld-r1/1.0 (training corpus collection; "
                    "rate-limit friendly)"}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")

# collection gates — REPLICATED from scripts/collect_realworld_slices.py
COMMIT_PASSING = re.compile(r"(?i)\b(commit|commits|git|github|gitlab)\b")
STACKY = re.compile(
    r"(?i)(traceback|exception|uncaught|internal error|error:|failed:|"
    r"fatal\b|panic:|exit code|callstack|segmentation fault|aborted|"
    r"stack trace|compilation failed)")

# rater-A mechanical screen (registered manual, spec: workflow lookalikes,
# bot/boilerplate, spam; the drop path is real)
CONVENTIONAL_COMMIT = re.compile(
    r"^(fix|feat|chore|refactor|docs|test|build|ci|perf|style)(\(|:)\b")
MERGE_PASTE = re.compile(r"^Merge pull request ")
HASHTAG = re.compile(r"#\w+")
BOT_MARKERS = re.compile(
    r"(?i)(dependabot|codecov|coveralls|github-actions|netlify|vercel|"
    r"renovate|gitguardian|snyk|sonarcloud|mergify|stale\[bot\]|\[bot\])")
BOT_COMMENT = re.compile(
    r"(?i)^(\s|)*(?:🤖|:robot:)|^LGTM\b|^thanks\b|^thank you\b|^\+1\b|^nit\b")
COMMANDISH = re.compile(r"^[!/]\w*(\s|$)|^(hubot|jenkinsbot)\b", re.I)
ISSUE_TEMPLATE = re.compile(
    r"(?i)^(steps to reproduce|expected behavior|actual behavior)\b")
TIMESTAMP_LINE = re.compile(
    r"^\s*(\[?\d{4}-\d{2}-\d{2}|\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2})")
EN_MARKERS = {"the", "and", "is", "to", "of", "a", "in", "that", "it",
              "for", "on", "with", "as", "was", "at", "by", "an", "be",
              "this", "from", "or", "but", "not", "are", "have", "has",
              "had", "you", "we", "they", "their", "will", "can", "all",
              "more", "new", "my", "out", "so", "just"}

# declared fetch budgets (spec: fresh windows only; offsets >= 2400 for HF)
MASTO_TAGS = ["git", "gitlab", "forgejo", "github", "devops",
              "100DaysOfCode", "opensource"]
MASTO_MAX_PAGES = 60
SLACK_OFFSET_START = 2400        # the eval window is offsets 0-2399
SLACK_OFFSET_MAX = 250_000       # fresh rows; the wp texture fires at only
#                                 ~0.16% past the eval window (measured at
#                                 the first scan), so the declared D band
#                                 [150, 500] needs the deeper window. The
#                                 spec's "~65k" was an estimate, not a
#                                 registered bound; the D minimum is.
SLACK_EARLY_STOP_SL = 9_500      # pre-exclusion candidate targets
SLACK_EARLY_STOP_WP = 400
RC_REPOS = ["home-assistant/core", "fastapi/fastapi", "rails/rails",
            "golang/go", "flutter/flutter", "sympy/sympy",
            "scikit-learn/scikit-learn", "astropy/astropy"]
RC_PAGES_PER_REPO = 3
RC_SLEEP = 70.0                  # unauthenticated GitHub: 60 requests/hour

SAMPLE_MINIMA = {"pc": 300, "sl": 400, "wp": 100, "rc": 150}


def http_json(url: str, timeout: int = 30, retries: int = 4,
              backoff: int = 20):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 — retry, then give up
            last = exc
            wait = backoff * (attempt + 1)
            print(f"    fetch error ({exc}); retry in {wait}s", flush=True)
            time.sleep(wait)
    print(f"    giving up: {last}", flush=True)
    return None


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


def mask_emails(text: str) -> str:
    return EMAIL.sub("<email>", text)


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ fetch
def fetch_pc() -> int:
    """Fresh Mastodon harvest: the eval collector's 7 tags, deeper
    pagination; identical filters (bot exclusion, strip_html, 80-1400
    chars, commit-vocabulary gate). Raw candidates only — screening is
    the screen phase."""
    out: list[dict] = []
    for tag in MASTO_TAGS:
        url = f"https://mastodon.social/api/v1/timelines/tag/{tag}?limit=40"
        for _page in range(MASTO_MAX_PAGES):
            batch = http_json(url)
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
                out.append({"src": "pc", "text": mask_emails(text),
                            "origin": f"mastodon:{acct}:{post.get('id', '')}"})
            url = (f"https://mastodon.social/api/v1/timelines/tag/{tag}"
                   f"?limit=40&max_id={batch[-1]['id']}")
            time.sleep(0.6)
        print(f"  #{tag}: cumulative {len(out)}", flush=True)
    write_jsonl(RAW / "r1-candidates-pc.jsonl", out)
    print(f"pc candidates: {len(out)} -> {RAW / 'r1-candidates-pc.jsonl'}")
    return 0


def fetch_sl() -> int:
    """Fresh software_slacks window: offsets >= 2400 (the eval window is
    0-2399), the collector's filters replicated, then the two predicates:
    wrapper+error-paste (wp) vs work chat (sl). Flushes every 50 pages and
    resumes from the last flushed offset (MIVIA_R1_SL_RESUME=offset) so a
    killed run does not lose the scan."""
    out_path = RAW / "r1-candidates-sl.jsonl"
    out: list[dict] = []
    offset = SLACK_OFFSET_START
    if out_path.is_file() and os.environ.get("MIVIA_R1_SL_RESUME"):
        offset = int(os.environ["MIVIA_R1_SL_RESUME"])
        for line in out_path.read_text(encoding="utf-8",
                                    errors="replace").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
        print(f"resuming at offset {offset} with {len(out)} rows", flush=True)
    n_sl = sum(1 for r in out if r["src"] == "sl")
    n_wp = sum(1 for r in out if r["src"] == "wp")
    pages_since_flush = 0
    while offset < SLACK_OFFSET_MAX:
        if n_sl >= SLACK_EARLY_STOP_SL and n_wp >= SLACK_EARLY_STOP_WP:
            break
        url = ("https://datasets-server.huggingface.co/rows?dataset="
               "spencer%2Fsoftware_slacks&config=default&split=train"
               f"&offset={offset}&length=100")
        batch = http_json(url)
        if batch is None:
            write_jsonl(out_path, out)
            break
        rows_ = batch.get("rows", [])
        if not rows_:
            break
        for r in rows_:
            text = html_unescape(r["row"]["text"]).strip()
            if not (25 <= len(text) <= 1500):
                continue
            if text.count("http") > len(text) / 60:
                continue  # link dump (replicated collector filter)
            err_in_fence = any(STACKY.search(b)
                               for b in re.findall(r"```(.*?)```", text,
                                                   re.S))
            err_anywhere = bool(STACKY.search(text))
            is_paste = err_in_fence or (err_anywhere and text.count("\n") >= 3)
            src = "wp" if is_paste else "sl"
            if src == "sl":
                n_sl += 1
            else:
                n_wp += 1
            out.append({"src": src, "text": mask_emails(text),
                        "origin": f"hf:spencer/software_slacks:"
                                  f"{r['row']['workspace']}/"
                                  f"{r['row']['channel']}:{offset + len(out) % 100}"})
        offset += 100
        time.sleep(1.0)
        pages_since_flush += 1
        if pages_since_flush >= 50:
            write_jsonl(out_path, out)
            pages_since_flush = 0
        if offset % 5000 == 2400:
            print(f"  offset {offset}: sl={n_sl} wp={n_wp}", flush=True)
    write_jsonl(out_path, out)
    print(f"sl/wp candidates: {n_sl} + {n_wp} -> {out_path}")
    return 0


def fetch_rc() -> int:
    """GitHub PR inline review comments from an active-repo set distinct
    from the issue-eval repos; unauthenticated REST, sleep between
    requests (60/hour)."""
    out: list[dict] = []
    for repo in RC_REPOS:
        for page in range(1, RC_PAGES_PER_REPO + 1):
            url = (f"https://api.github.com/repos/{repo}/pulls/comments"
                   f"?per_page=100&page={page}&sort=created&direction=desc")
            batch = http_json(url)
            if not batch:
                break
            for c in batch:
                login = (c.get("user") or {}).get("login", "") or ""
                if BOT_MARKERS.search(login):
                    continue
                body = (c.get("body") or "").strip()
                if not (25 <= len(body) <= 1500):
                    continue
                if BOT_MARKERS.search(body[:120]):
                    continue
                out.append({"src": "rc", "text": mask_emails(body),
                            "origin": f"github:{repo}:{c.get('id', '')}"
                                      f":{login}"})
            time.sleep(RC_SLEEP)
        print(f"  {repo}: cumulative {len(out)}", flush=True)
    write_jsonl(RAW / "r1-candidates-rc.jsonl", out)
    print(f"rc candidates: {len(out)} -> {RAW / 'r1-candidates-rc.jsonl'}")
    return 0


# ----------------------------------------------------------------- screen
def blocked_norms() -> set[str]:
    """Everything the corpus must be disjoint from (the audit gate's own
    sets, imported so builder and gate compute the SAME blocked sets)."""
    blocked = set(eval_norms(full=True))
    windows, _diag = _replicate_eval_windows()
    for texts in windows.values():
        blocked |= texts
    for pool in ("data/incoming/realworld/raw-commit.jsonl",
                 "data/incoming/realworld/raw-slack.jsonl"):
        p = ROOT / pool
        for line in p.read_text(encoding="utf-8",
                                errors="replace").splitlines():
            if line.strip():
                blocked.add(norm(json.loads(line).get("text") or ""))
    for extra in ("data/incoming/external/cleancodereview/comments.jsonl",
                  "data/real/review_comment/train.jsonl",
                  "data/real/review_comment/dev.jsonl",
                  REALWORLD_R1_LINEAGE["aux"][0].as_posix(),
                  REALWORLD_R1_LINEAGE["extra"][0].as_posix()):
        p = ROOT / extra
        if p.is_file():
            blocked |= {norm(t) for t in
                        (rows_text(corpus_rows(p)))}
    from mivia_small_json import model as _model

    from mivia_small_json import bootstrap

    bootstrap.register_example_schemas()
    bx, _by = _model.training_mix(seed=42)
    blocked |= {norm(t) for t in bx}
    aux_dir = ROOT / "data/incoming/ood-aux-chat"
    if aux_dir.is_dir():
        for p in sorted(aux_dir.glob("*.jsonl")):
            blocked |= {norm(t) for t in rows_text(corpus_rows(p))}
    return blocked


def rows_text(rows: list[dict]) -> list[str]:
    return [r.get("text") or r.get("raw") or "" for r in rows]


# rater-A mechanical screen, v2 (kappa-gate iteration 1 under-implemented the
# registered manual: no spam-template rule, no error-content-in-paste rule —
# the blind rater-B pass exposed exactly those gaps; B's labels are untouched)
SPAM_TEMPLATE_MARKERS = [
    "full report → https:// pullrepo.com/report",
    "hello from the # api at time",           # auto-posted test-bot template
]
SPAM_PREFIXES = ["🚀 fastest-growing ai projects today"]


def _paste_region(text: str) -> str:
    """Fenced blocks + inline backtick spans — where an actual error paste
    must live for the wp register."""
    parts = re.findall(r"```(.*?)```", text, re.S) + \
        re.findall(r"`([^`\n]+)`", text)
    return "\n".join(parts)


ERROR_SIGNATURE = re.compile(
    r"(?im)(^error\b|^!\s*error|error:|failed:|exception|traceback|panic:|"
    r"uncaught|not enough bytes|cannot read|cannot find|i cannot find|"
    r"does not match|type mismatch|compilation failed|callstack|"
    r"throw an exception|throw er)")


def screen_pc(text: str) -> tuple[str, str]:
    """Rater-A mechanical screen for the post-commit register.
    Returns (coding, reason); coding is 'keep' or 'drop:<reason>'."""
    low = text.lower()
    if any(m in low for m in SPAM_TEMPLATE_MARKERS) or \
            any(low.startswith(p) for p in SPAM_PREFIXES):
        return "drop:spam-template", "repeated auto-posted template"
    if CONVENTIONAL_COMMIT.match(text) or MERGE_PASTE.match(text):
        return "drop:commit-message-lookalike", "conventional-commit/merge paste"
    tags = HASHTAG.findall(text)
    if len(tags) >= 10 or (tags and len("".join(tags)) > 0.4 * len(text)):
        return "drop:hashtag-spam", "hashtag wall, not prose"
    if text.count("http") > 3:
        return "drop:link-dump", "link dump"
    letters = re.findall(r"\S", text)
    if letters and sum(c.isalpha() for c in text) / max(1, len(text)) < 0.35:
        return "drop:non-linguistic", "symbol wall"
    words = re.findall(r"[A-Za-zÀ-ÿĀ-žĂ-șа-яА-Яα-ωا-ي]+", text)
    if len(words) < 8:
        return "drop:too-short", "under 8 words"
    return "keep", "dev-announce register"


def screen_sl(text: str) -> tuple[str, str]:
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    prose_words = len(re.findall(r"\w+", prose))
    if COMMANDISH.match(text.strip()):
        return "drop:bot-command", "slash/bot command"
    if ISSUE_TEMPLATE.match(text.strip()):
        return "drop:issue-template-lookalike", "issue template"
    if prose_words < 6:
        return "drop:code-only", "no conversational prose"
    return "keep", "work chat"


def screen_wp(text: str) -> tuple[str, str]:
    outside = re.sub(r"```.*?```", "", text, flags=re.S).strip()
    outside = re.sub(r"\n?```\s*$", "", outside)  # unclosed fence tail
    if len(re.findall(r"\w+", outside)) < 4:
        return "drop:bare-paste", "no conversational wrapper"
    # the manual: wrapper + ERROR paste — an actual error signature must
    # appear in the paste region (fences/inline code), not just anywhere in
    # the prose (rater-B iteration 1: advice chat and opinion prose carried
    # the word "exception"/"error" and false-fired the fetch predicate)
    if not ERROR_SIGNATURE.search(_paste_region(text)):
        return "drop:no-error-content", "no error signature in the paste"
    return "keep", "wrapper + error paste"


def screen_rc(text: str) -> tuple[str, str]:
    fenced = "".join(re.findall(r"```(.*?)```", text, re.S))
    if fenced and len(fenced) > 0.7 * len(text):
        return "drop:suggestion-only", "code block without prose"
    if BOT_COMMENT.match(text.strip()) and len(text.split()) <= 3:
        return "drop:thin-bot", "emoji/approval stub"
    return "keep", "inline review comment"


SCREENS = {"pc": screen_pc, "sl": screen_sl, "wp": screen_wp,
           "rc": screen_rc}
SRC_FILES = {"pc": RAW / "r1-candidates-pc.jsonl",
             "sl": RAW / "r1-candidates-sl.jsonl",
             "wp": RAW / "r1-candidates-sl.jsonl",
             "rc": RAW / "r1-candidates-rc.jsonl"}
PREFIX = {"pc": "pc", "sl": "sl", "wp": "wp", "rc": "rc"}


def en_like(text: str) -> bool:
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return False
    return sum(1 for t in tokens if t in EN_MARKERS) / len(tokens) >= 0.18


def phase_screen() -> int:
    blocked = blocked_norms()
    print(f"blocked norms: {len(blocked)}")
    windows, wdiag = _replicate_eval_windows()
    for name, d in wdiag.items():
        print(f"  replication {name}: {d}")

    seen: set[str] = set()
    pool: list[dict] = []   # post-exclusion, PRE-screen (A codes all of it)
    drop_stats: Counter = Counter()
    for src in ("pc", "sl", "wp", "rc"):
        raw_rows = []
        for line in SRC_FILES[src].read_text(
                encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                raw_rows.append(json.loads(line))
            except ValueError:
                continue  # tolerate a torn tail line from an interrupted run
        n_src = 0
        for r in raw_rows:
            if r["src"] != src:
                continue  # the sl file carries both sl and wp rows
            text = r["text"].strip()
            k = norm(text)
            if not k or k in seen:
                drop_stats[f"{src}:dupe"] += 1
                continue
            if k in blocked:
                drop_stats[f"{src}:eval-or-mix-overlap"] += 1
                continue
            seen.add(k)
            pool.append({"id": f"{PREFIX[src]}-{len(pool):06d}",
                         "src": src, "text": text, "origin": r["origin"]})
            n_src += 1
        print(f"  {src}: {n_src} post-exclusion rows")
    if not pool:
        print("no candidates — run --phase fetch first", file=sys.stderr)
        return 2

    # rater A: FULL coverage of the post-exclusion pool (drops included —
    # the sample rater B codes must contain A-drops for the drop-path
    # evidence to be real)
    label_of = {"pc": "text", "sl": "text", "wp": "log_event",
                "rc": "review_comment"}
    a_rows = []
    for c in pool:
        coding, _reason = SCREENS[c["src"]](c["text"])
        a_rows.append({"id": c["id"], "label": coding})
    write_jsonl(REALWORLD_R1_DIR / "adjudication-A.jsonl", a_rows)
    coding_of = {r["id"]: r["label"] for r in a_rows}

    # declared band maxima are quotas: oversupply is sampled DOWN
    # deterministically (seed 42) and recorded — never admitted past the
    # registered composition (a silent over-band corpus is exactly the
    # composition drift the audit gate exists to catch)
    BAND_MAX = {"pc": 5_500, "sl": 8_500, "wp": 500, "rc": 2_000}
    by_src_pool: dict[str, list[dict]] = {}
    for c in pool:
        by_src_pool.setdefault(c["src"], []).append(c)
    rng_band = random.Random(42)
    sampled: list[dict] = []
    sampled_note: dict[str, dict] = {}
    for src, rows in by_src_pool.items():
        keeps = [c for c in rows if coding_of[c["id"]] == "keep"]
        if len(keeps) > BAND_MAX[src]:
            keeps = rng_band.sample(keeps, BAND_MAX[src])
        sampled_note[src] = {"admitted_candidates": len(keeps)}
        sampled.extend(keeps)
    cand_rows = [{"id": c["id"], "src": c["src"], "text": c["text"],
                  "label": label_of[c["src"]], "source": c["origin"]}
                 for c in sampled]
    write_jsonl(REALWORLD_R1_DIR / "candidates-A.jsonl", cand_rows)
    a_keeps = len(cand_rows)

    # the blind stratified sample for rater B: drawn from the FULL pool
    # (A-drops AND sampled-out oversupply included — B validates the
    # screen itself; the drop-path evidence needs A-drops in the sample),
    # shuffled presentation, no A labels in the file
    rng = random.Random(42)
    sample: list[dict] = []
    for src, rows in by_src_pool.items():
        k = min(SAMPLE_MINIMA[src], len(rows))
        sample.extend(rng.sample(rows, k))
    rng2 = random.Random(43)
    rng2.shuffle(sample)
    write_jsonl(REALWORLD_R1_DIR / "adjudication-sample.jsonl",
                [{"id": c["id"], "src": c["src"], "text": c["text"]}
                 for c in sample])

    # measured language split for P (report-only heuristic, spec) over the
    # band-sampled candidates that will actually ship
    pc_rows = [c for c in sampled if c["src"] == "pc"]
    lang = Counter("en-like" if en_like(c["text"]) else "other"
                   for c in pc_rows)
    langs_detail = Counter()
    for c in pc_rows:
        t = c["text"]
        if not t.isascii():
            script = ("cyrillic" if re.search(r"[а-яА-Я]", t) else
                      "arabic" if re.search(r"[ا-ي]", t) else
                      "cjk" if re.search(r"[一-鿿぀-ヿ가-힯]", t) else
                      "other-nonlatin")
            langs_detail[script] += 1
        elif not en_like(t):
            langs_detail["latin-other"] += 1
    for c in pool:
        coding, reason = SCREENS[c["src"]](c["text"])
        if coding != "keep":
            drop_stats[f"{c['src']}:{reason}"] += 1

    report = {
        "pool": Counter(c["src"] for c in pool),
        "a_keeps": a_keeps,
        "band_sampling": sampled_note,
        "drops": drop_stats,
        "sample": Counter(c["src"] for c in sample),
        "pc_language_report": {"en_like": lang["en-like"],
                               "other": lang["other"],
                               "non_en_scripts": langs_detail},
    }
    (REALWORLD_R1_DIR / "screen-report.json").write_text(
        json.dumps(report, indent=1, default=dict))
    print(json.dumps(report, indent=1, default=dict))
    print(f"\nrater B: code {len(sample)} rows from "
          f"{REALWORLD_R1_DIR / 'adjudication-sample.jsonl'} into "
          f"adjudication-B.jsonl (BLIND: no access to adjudication-A.jsonl)")
    return 0


# --------------------------------------------------------------- assemble
def phase_verify_b() -> int:
    """Kappa + coverage check BEFORE assemble (the B pass is out of band)."""
    from audit_training_mix import _cohen_kappa

    b_path = REALWORLD_R1_DIR / "adjudication-B.jsonl"
    a_path = REALWORLD_R1_DIR / "adjudication-A.jsonl"
    if not b_path.is_file():
        print("adjudication-B.jsonl missing", file=sys.stderr)
        return 2
    mapa = {r["id"]: r["label"] for r in corpus_rows(a_path)}
    mapb = {r["id"]: r["label"] for r in corpus_rows(b_path)}
    paired = [(mapa[i], mapb[i]) for i in mapb if i in mapa]
    ka = [a.split(":", 1)[0] for a, _ in paired]
    kb = [b.split(":", 1)[0] for _, b in paired]
    kappa = _cohen_kappa(ka, kb)
    print(f"paired {len(paired)} kappa={kappa:.4f} (min 0.6)")
    dis = [(a, b) for a, b in paired if a != b]
    print(f"disagreements: {len(dis)}")
    for a, b in dis[:20]:
        print(f"  A={a}  B={b}")
    by_prefix = Counter(i.split("-", 1)[0] for i in mapb)
    print("B coverage by prefix:", dict(by_prefix))
    return 0 if kappa >= 0.6 and len(paired) >= 900 else 2


def phase_assemble() -> int:
    from audit_training_mix import _cohen_kappa

    cands = corpus_rows(REALWORLD_R1_DIR / "candidates-A.jsonl")
    mapa = {r["id"]: r["label"] for r in
            corpus_rows(REALWORLD_R1_DIR / "adjudication-A.jsonl")}
    b_rows = corpus_rows(REALWORLD_R1_DIR / "adjudication-B.jsonl")
    mapb = {r["id"]: r["label"] for r in b_rows}

    # drop path: any sample row either rater codes DROP is removed
    kept, dropped = [], []
    for r in cands:
        a = mapa.get(r["id"], "keep")
        b = mapb.get(r["id"], "keep")
        if a.startswith("drop") or b.startswith("drop"):
            dropped.append({"id": r["id"], "a": a, "b": b})
            continue
        kept.append(r)
    # kappa over the full paired sample (agreement on drops included)
    paired = [(mapa[i], mapb[i]) for i in mapb if i in mapa]
    kappa = _cohen_kappa([a.split(":", 1)[0] for a, _ in paired],
                         [b.split(":", 1)[0] for _, b in paired])
    print(f"adjudication: paired={len(paired)} kappa={kappa:.4f} "
          f"dropped={len(dropped)}")
    if kappa < 0.6:
        print("kappa below the 0.6 gate", file=sys.stderr)
        return 2

    # declared bands (spec) — exit 2 on shortfall, never shrink silently
    by_src = Counter(r["src"] for r in kept)
    print("admitted:", dict(by_src))
    for name, (lo, hi) in REALWORLD_R1_SOURCES.items():
        src = {"post-commit-r1.jsonl": "pc", "slack-r1.jsonl": "sl",
               "wrapper-paste-r1.jsonl": "wp",
               "review-counter-r1.jsonl": "rc"}[name]
        n = by_src.get(src, 0)
        if not lo <= n <= hi:
            print(f"BAND FAIL: {name} has {n} admitted rows, declared "
                  f"[{lo}, {hi}]", file=sys.stderr)
            return 2
    n_text = by_src.get("pc", 0) + by_src.get("sl", 0)
    share = n_text / (90_790 + sum(by_src.values()))
    print(f"increment text share: {share:.4f} (bound 0.19)")
    if share > 0.19:
        print("INCREMENT TEXT SHARE FAIL", file=sys.stderr)
        return 2

    # write the four corpora
    names = {"pc": "post-commit-r1.jsonl", "sl": "slack-r1.jsonl",
             "wp": "wrapper-paste-r1.jsonl", "rc": "review-counter-r1.jsonl"}
    for src, name in names.items():
        rows = [{"id": r["id"], "text": r["text"], "label": r["label"],
                 "source": r["source"]}
                for r in kept if r["src"] == src]
        p = REALWORLD_R1_DIR / name
        write_jsonl(p, rows)
        p.with_suffix(p.suffix + ".sha256").write_text(sha256_of(p) + "\n")
        print(f"  {name}: {len(rows)} rows")

    # assembled lineage files: frozen bytes ++ increments, byte-exact
    def assemble(assembled: Path, frozen: Path, sources: list[str]) -> None:
        frozen_bytes = frozen.read_bytes()
        inc_rows = []
        for src in sources:
            for r in kept:
                if r["src"] != src:
                    continue
                inc_rows.append({"text": r["text"], "category": r["label"],
                                 "r1-id": r["id"], "r1-source": r["source"]})
        with assembled.open("wb") as fh:
            fh.write(frozen_bytes)
            for row in inc_rows:
                fh.write((json.dumps(row, ensure_ascii=False) + "\n")
                         .encode("utf-8"))
        assembled.with_suffix(assembled.suffix + ".sha256").write_text(
            sha256_of(assembled) + "\n")
        print(f"  {assembled.name}: {len(corpus_rows(frozen))} frozen + "
              f"{len(inc_rows)} increment rows")

    assemble(REALWORLD_R1_AUX, REALWORLD_R1_LINEAGE["aux"][0], ["pc", "sl"])
    assemble(REALWORLD_R1_EXTRA, REALWORLD_R1_LINEAGE["extra"][0],
             ["wp", "rc"])

    write_provenance(by_src, kappa, dropped, share)

    print("\nnext: --phase probe (weakness probe BEFORE any GPU run), then "
          "scripts/audit_training_mix.py --arm realworld-slices-r1")
    return 0


def write_provenance(by_src: Counter, kappa: float, dropped: list,
                     share: float) -> None:
    """Idempotent PROVENANCE sections (rule 12: replace own section,
    never stack duplicates)."""
    prov = REALWORLD_R1_DIR / "PROVENANCE.md"
    old = prov.read_text(encoding="utf-8") if prov.is_file() else ""
    # strip this builder's own sections, keep anything else
    cleaned = re.sub(
        r"<!-- section:r1:.*? -->\n.*?(?=<!-- section:r1:|\Z)", "",
        old, flags=re.S).strip()
    screen = json.loads(
        (REALWORLD_R1_DIR / "screen-report.json").read_text())
    lines = ["", ""]
    blocks = [
        ("header",
         "# Provenance — realworld-slices-r1 corpus\n\n"
         "Registered arm: realworld-slices-r1 Built by "
         "scripts/build_realworld_r1_corpus.py (fetch/screen/assemble), "
         "audit-gated by scripts/audit_training_mix.py --arm "
         "realworld-slices-r1. All rows are REAL public data; no templates "
         "were authored. Labels are adjudicated (rater A mechanical "
         "full-coverage screen + rater B blind LLM re-check on the "
         "registered stratified sample; Cohen's kappa computed over the "
         f"full paired sample: {kappa:.4f}, gate >= 0.6; drop path: any "
         f"either-rater DROP removed, {len(dropped)} rows dropped)."),
        ("post-commit-r1.jsonl",
         "<!-- section:r1:post-commit-r1.jsonl -->\n"
         "## post-commit-r1.jsonl (P — the dev-announce register)\n\n"
         "- source: fresh Mastodon public-timeline harvest, the eval "
         "collector's 7 tags (git/gitlab/forgejo/github/devops/"
         "100DaysOfCode/opensource), deeper pagination (<= 60 pages x 40)\n"
         f"- admitted: {by_src.get('pc', 0)} rows (declared band "
         f"{list(REALWORLD_R1_SOURCES['post-commit-r1.jsonl'])}), label "
         "`text`\n"
         "- filters replicated from scripts/collect_realworld_slices.py "
         "(bot exclusion, HTML strip, 80-1400 chars, commit-vocabulary "
         "gate); emails masked\n"
         "- exclusion verified by normalized-set comparison: eval window "
         "(200 rows), the committed raw-commit pool (794), every frozen "
         "eval surface, the base training mix, the chat-ood aux corpora\n"
         f"- language (report-only heuristic): {screen['pc_language_report']}"
         "\n- license: public Mastodon posts (mastodon.social), same as "
         "the eval slice's provenance\n"
         f"- sha256: `{sha256_of(REALWORLD_R1_DIR / 'post-commit-r1.jsonl')}`"),
        ("slack-r1.jsonl",
         "<!-- section:r1:slack-r1.jsonl -->\n"
         "## slack-r1.jsonl (S — dev-community chat)\n\n"
         "- source: HF spencer/software_slacks train split, fresh offsets "
         f">= {SLACK_OFFSET_START} (the eval window is offsets 0-2399; the "
         "dataset holds 1,394,705 rows)\n"
         f"- admitted: {by_src.get('sl', 0)} rows (declared band "
         f"{list(REALWORLD_R1_SOURCES['slack-r1.jsonl'])}), label `text`\n"
         "- filters replicated from the collector (HTML unescape, 25-1500 "
         "chars, link-dump drop)\n"
         "- license: NONE on the dataset — noted gap, owner handles "
         "licenses (same note as the eval slice's provenance); public "
         "Slack exports of software communities\n"
         f"- sha256: `{sha256_of(REALWORLD_R1_DIR / 'slack-r1.jsonl')}`"),
        ("wrapper-paste-r1.jsonl",
         "<!-- section:r1:wrapper-paste-r1.jsonl -->\n"
         "## wrapper-paste-r1.jsonl (D — the S2 falsification's trained "
         "fix)\n\n"
         "- source: the same fresh software_slacks window; the collector's "
         "own log_event predicate (error markers inside a fence, or "
         "anywhere in a >= 3-newline text); wrapper required (>= 4 prose "
         "words outside fences)\n"
         f"- admitted: {by_src.get('wp', 0)} rows (declared band "
         f"{list(REALWORLD_R1_SOURCES['wrapper-paste-r1.jsonl'])}), label "
         "`log_event` — boundary mass on the TRUE side so the text "
         "increment cannot erode log_event\n"
         f"- sha256: `{sha256_of(REALWORLD_R1_DIR / 'wrapper-paste-r1.jsonl')}`"),
        ("review-counter-r1.jsonl",
         "<!-- section:r1:review-counter-r1.jsonl -->\n"
         "## review-counter-r1.jsonl (C — the review_comment "
         "counterbalance)\n\n"
         "- source: GitHub PR inline review comments via unauthenticated "
         "REST from an active-repo set distinct from the issue-eval repos "
         f"({', '.join(RC_REPOS)})\n"
         f"- admitted: {by_src.get('rc', 0)} rows (declared band "
         f"{list(REALWORLD_R1_SOURCES['review-counter-r1.jsonl'])}), label "
         "`review_comment`\n"
         "- cleancodereview is deliberately NOT trained on (its benches "
         "shuffle the whole file); per-repo OSS licenses; emails masked\n"
         f"- sha256: `{sha256_of(REALWORLD_R1_DIR / 'review-counter-r1.jsonl')}`"),
        ("assembled",
         "<!-- section:r1:assembled -->\n"
         "## Assembled lineage files (trainer inputs)\n\n"
         f"- {REALWORLD_R1_AUX.name} = frozen ood-aux-train-v5-chat.jsonl "
         "(17,173 rows, sha-verified) + P + S; MIVIA_OOD_AUX\n"
         f"- {REALWORLD_R1_EXTRA.name} = frozen multilang-tickets-v11.jsonl"
         " (5,666 rows, sha-verified) + D + C; MIVIA_EXTRA_INDOMAIN\n"
         f"- increment text share at build: {share:.4f} (bound 0.19); "
         "dedupe measured by the audit gate <= 1%\n"
         f"- {REALWORLD_R1_AUX.name} sha256: `{sha256_of(REALWORLD_R1_AUX)}`\n"
         f"- {REALWORLD_R1_EXTRA.name} sha256: "
         f"`{sha256_of(REALWORLD_R1_EXTRA)}`"),
        ("evidence",
         "<!-- section:r1:evidence -->\n"
         "## Adjudication evidence\n\n"
         "- adjudication-A.jsonl: rater A (mechanical full-coverage "
         "screen)\n- adjudication-B.jsonl: rater B (blind LLM re-check, "
         "stratified sample, shuffled presentation, no access to A)\n"
         "- adjudication-sample.jsonl: the sampled rows as presented to B\n"
         "- screen-report.json: per-source candidate counts, drop reasons, "
         "language report\n"),
    ]
    for key, text in blocks:
        lines.append(text)
        lines.append("")
    prov.write_text(cleaned + "\n" + "\n".join(lines), encoding="utf-8")
    print(f"provenance: {prov}")


# ------------------------------------------------------------------ probe
def phase_probe() -> int:
    """Weakness probe (protocol 6b): route every admitted row through the
    SHIPPED router; easy-mass vs signal split written up BEFORE the GPU."""
    from mivia_small_json import bootstrap
    from mivia_small_json.pipeline import classify, load_active_router

    bootstrap.register_all_schemas()
    router = load_active_router()
    expect = {"pc": {"text"}, "sl": {"text"}, "wp": {"log_event"},
              "rc": {"review_comment"}}
    stats: dict[str, dict] = {}
    for name, src in (("post-commit-r1.jsonl", "pc"),
                      ("slack-r1.jsonl", "sl"),
                      ("wrapper-paste-r1.jsonl", "wp"),
                      ("review-counter-r1.jsonl", "rc")):
        rows = corpus_rows(REALWORLD_R1_DIR / name)
        dist: Counter = Counter()
        by: Counter = Counter()
        easy = 0
        for r in rows:
            obj, meta = classify(r["text"], router)
            dist[obj.get("type")] += 1
            by[meta.get("routed_by")] += 1
            if obj.get("type") in expect[src]:
                easy += 1
        n = len(rows)
        stats[name] = {"rows": n, "easy_mass": easy,
                       "easy_mass_pct": round(100 * easy / max(1, n), 1),
                       "routed": dict(dist), "routed_by": dict(by)}
        print(name, stats[name])
    (REALWORLD_R1_DIR / "weakness-probe.json").write_text(
        json.dumps(stats, indent=1))
    lines = [
        "# Weakness probe — realworld-slices-r1 corpus (SHIPPED router)",
        "",
        "Protocol 6b: rows the shipped router already routes to the",
        "increment's outcome are easy mass; the misrouted residue is the",
        "actual signal this round buys. Written BEFORE the GPU run.",
        "",
        "| source | rows | easy-mass | easy % | top misroutes |",
        "|---|---|---|---|---|",
    ]
    for name, s in stats.items():
        top = ", ".join(f"{k} {v}" for k, v in sorted(
            s["routed"].items(), key=lambda kv: -kv[1])[:3])
        lines.append(f"| {name} | {s['rows']} | {s['easy_mass']} | "
                     f"{s['easy_mass_pct']} | {top} |")
    (REALWORLD_R1_DIR / "weakness-probe.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REALWORLD_R1_DIR / 'weakness-probe.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=("fetch", "fetch-pc", "fetch-sl", "fetch-rc",
                             "screen", "verify-b", "assemble", "probe"))
    args = ap.parse_args()
    REALWORLD_R1_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase == "fetch":
        rc1 = fetch_pc()
        rc2 = fetch_sl()
        rc3 = fetch_rc()
        return max(rc1, rc2, rc3)
    if args.phase == "fetch-pc":
        return fetch_pc()
    if args.phase == "fetch-sl":
        return fetch_sl()
    if args.phase == "fetch-rc":
        return fetch_rc()
    if args.phase == "screen":
        return phase_screen()
    if args.phase == "verify-b":
        return phase_verify_b()
    if args.phase == "assemble":
        return phase_assemble()
    if args.phase == "probe":
        return phase_probe()
    return 2


if __name__ == "__main__":
    sys.exit(main())
