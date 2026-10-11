#!/usr/bin/env python3
"""Build the string-param span-fill training corpus (registered arm
``string-param-span-fill``, string-param-span-fill).

Three sources, each with a DECLARED minimum contribution the builder FAILS
below (volume-verified; a data gate that only prints is vacuous):

1. **Round-1 frozen corpus, reused as-is** — data/real/spanfill/{train,dev}
   .jsonl (12,407 / 1,402) and the frozen reserve data/heldout/
   spanfill-reserve.jsonl (3,191, sha-pinned by round 1's provenance).
   NOT regenerated, never copied: validated by count + sha and referenced.
2. **BFCL-derived silver, minimum 1,000 rows** — training-side BFCL v4
   splits + possible_answer ground truth ONLY. The eval files
   (irrelevance_eval.jsonl, the 183-row calls.jsonl, guard-eval-v2.jsonl)
   are excluded by REPLICATING their window logic (the house request-level
   filter of build_tool_match_corpus._eval_requests, plus the v2 eval's
   request set), verified by normalized-set comparison inside this
   builder. Alignment is mechanical high-confidence only: span_fill's own
   key-delimited and string-alignment ladders, and the aligned text must
   EQUAL the gold value. Per-row alignment evidence is recorded.
3. **Harvest gold candidates, minimum 500 admitted rows** — real tool_call
   records from data/incoming/harvested-sessions/ (gitignored; the
   zero-leakage redaction + session-disjoint partition contract lives in
   scripts/harvest_agent_sessions.py + manifest.json) and the
   data/incoming/gh-user/ owner harvest. A row is admitted ONLY when a
   natural-language request text accompanies the call (NOT the invocation
   JSON itself — a row whose only request-side text is the invocation
   cannot teach span selection) and EVERY gold argument value appears
   verbatim in that request text (mechanical filter).

Composition is measured against the declared bounds (frozen 40-70% of the
assembled train side; each new source 10-40%; enum >= 10% of positive
slots; new authored templates 0 <= cap 800) and the weakness probe (spec:
route the increment through the SERVED path; easy mass = the deterministic
ladders already fill correctly) is run BEFORE any GPU spend, with the
split counts written into the provenance.

DEFAULT behavior below a declared minimum or outside a declared bound is
FAIL (exit 2) writing NOTHING. ``--emit-despite-shortfall`` consciously
overrides: it writes the corpus, stamps every shortfall DATA-BLOCKED in
the composition report and provenance, and never touches the frozen
sources. The orchestrator owns that decision; substituting sources
silently is forbidden.

Outputs (new directory, house convention: data/real/<corpus>/ +
PROVENANCE.md; the frozen round-1 files are untouched):
    data/real/spanfill-stringparam/bfcl-silver.jsonl      (+ .sha256)
    data/real/spanfill-stringparam/harvest-gold.jsonl     (+ .sha256)
    data/real/spanfill-stringparam/composition-report.json
    data/real/spanfill-stringparam/PROVENANCE.md          (idempotent
    replace-my-own-section provenance per source: source, rules, counts,
    sha256)

Deterministic and offline after the first BFCL fetch (the same /tmp cache
the house pair-corpus builder uses).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))  # noqa: E402

from mivia_small_json import span_fill as sf  # noqa: E402
from mivia_small_json.tool_match import parse_catalogue_raw  # noqa: E402

OUT = ROOT / "data" / "real" / "spanfill-stringparam"
FROZEN_DIR = ROOT / "data" / "real" / "spanfill"
RESERVE = ROOT / "data" / "heldout" / "spanfill-reserve.jsonl"
HARVEST = ROOT / "data" / "incoming" / "harvested-sessions"
GH_USER = ROOT / "data" / "incoming" / "gh-user"
BFCL_EVAL_DIR = ROOT / "data" / "incoming" / "external" / "bfcl"

# --- declared plan (string-param-span-fill) -------------
# RE-REGISTERED 2026-10-10 (owner-directed curation; the spec carries the
# same-dated DATA-PLAN RE-REGISTRATION amendment):
#   - harvest-gold source DROPPED (measured reason: the committed harvest
#     slice carries no user turns — 0 rows admissible), gold_max 0;
#   - silver = BFCL string-family rows adjudicated by TWO BLIND labelers
#     under the clarified no-gold construct (verbatim minimal span,
#     positive context, exact enum match, all-or-nothing per row),
#     kappa-gated (>= 0.6), agreements-only; minimum 1,000 ADMITTED rows
#     post-kappa (unchanged, never lowered);
#   - composition shares amended: frozen 60-95%, bfcl-silver 5-40%;
#   - enum-slot share REPORTED, non-binding this round.
MIN_FROZEN = {"train": 12407, "dev": 1402, "reserve": 3191}
MIN_SILVER = 1000        # admitted rows post-kappa — never lowered
GOLD_MAX = 0             # harvest-gold DROPPED by the 2026-10-10
                         # re-registration (was MIN_GOLD = 500)
AUTHORED_CAP = 800
AUTHORED_DECLARED = 0  # this arm authors nothing (declared English-only,
                       # real/silver/gold supply instead)
BOUND_FROZEN_SHARE = (0.60, 0.95)
BOUND_NEW_SOURCE_SHARE = (0.05, 0.40)
ENUM_SLOT_SHARE_BINDING = False  # reported, non-binding this round
DUPE_RATE_MAX = 0.01
# Declared language register (spec data plan: "this arm is deliberately
# English-only — declared now so the audit compares against the
# declaration rather than discovering a mismatch"). Enforced MECHANICALLY
# at admission since 2026-10-10: a request carrying CJK/Hangul characters
# is refused and counted (language_violation), never admitted silently.
DECLARED_LANGUAGES = ("en",)

# CJK/Hangul detection ranges: Han ideographs (unified + extensions +
# compatibility), Kana, Hangul syllables and jamo (compat + extended).
# Deliberately EXCLUDES CJK punctuation/radical blocks and fullwidth
# Latin/digits (NFKC-foldable ASCII) so an English request is never
# flagged for typography alone.
_CJK_HANGUL_RANGES = (
    (0x1100, 0x11FF),    # Hangul Jamo
    (0x3040, 0x30FF),    # Hiragana + Katakana
    (0x3130, 0x318F),    # Hangul Compatibility Jamo
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xA960, 0xA97F),    # Hangul Jamo Extended-A
    (0xAC00, 0xD7A3),    # Hangul Syllables
    (0xD7B0, 0xD7FF),    # Hangul Jamo Extended-B
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0xFF66, 0xFF9D),    # Halfwidth Katakana
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2EBEF),  # CJK Extensions C-F
    (0x30000, 0x3134F),  # CJK Extensions G-H
)


def non_english_chars(text: str) -> str:
    """The CJK/Hangul characters a request carries; '' when none. The ONE
    mechanical predicate of the declared English-only register — the
    builder's admission gate and the audit gate's re-verification both
    use it."""
    return "".join(ch for ch in (text or "")
                   if any(lo <= ord(ch) <= hi
                          for lo, hi in _CJK_HANGUL_RANGES))

BFCL_BASE = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
             "berkeley-function-call-leaderboard/bfcl_eval/data")
# The training-side splits EXACTLY as the house pair-corpus builder pins
# them (build_tool_match_corpus.SPLITS): live_relevance (the 183-call eval
# set's source) and live_irrelevance (the abstention eval) are NOT here;
# multi_turn_* are the guard-eval-v2 windows and are not either.
BFCL_SPLITS = ("simple_python", "simple_java", "simple_javascript",
               "parallel", "multiple", "parallel_multiple",
               "live_simple", "live_multiple", "live_parallel",
               "live_parallel_multiple")
BFCL_EVAL_FILES = ("calls.jsonl", "irrelevance_eval.jsonl",
                   "guard-eval-v2.jsonl")


def norm(text: str) -> str:
    """Strong normalization (protocol 2a): NFKC + casefold + punctuation
    strip + whitespace collapse."""
    t = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(re.compile(r"[\W_]+", re.UNICODE).sub(" ", t).split())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line
            in path.read_text(encoding="utf-8").split("\n") if line.strip()]


# --- eval-window replication ------------------------------------------------

def eval_request_norms() -> set[str]:
    """Every normalized request the eval surfaces score — the SAME window
    logic the house pair builder replicates (request-level, not pair
    level: BFCL reuses a question across splits), plus the guard-eval-v2
    window. This is the exclusion contract of the spec's data plan.

    A MISSING declared surface is a build failure (FileNotFoundError),
    never a skip: the exclusion contract cannot be verified against an
    absent file, and a silent skip would build against an empty blocked
    set (2026-10-10 gate fix)."""
    blocked: set[str] = set()
    missing: list[str] = []
    for name in BFCL_EVAL_FILES:
        path = BFCL_EVAL_DIR / name
        if not path.is_file():
            missing.append(name)
            continue
        for row in _jsonl(path):
            text = row.get("text") or row.get("request") or ""
            if not text:
                continue
            if row.get("request") is not None:
                blocked.add(norm(text))  # v2 rows carry the bare request
                continue
            try:
                _tools, request = parse_catalogue_raw(text)
            except Exception:
                request = ""
            if request:
                blocked.add(norm(request))
            else:
                blocked.add(norm(text))
    if missing:
        raise FileNotFoundError(
            "eval surface(s) missing — the eval-window exclusion contract "
            f"cannot be verified: {', '.join(missing)} (looked in "
            f"{BFCL_EVAL_DIR}); the build FAILS instead of skipping a "
            "declared eval window")
    return blocked


# --- source 1: the frozen round-1 corpus (referenced, never rebuilt) -------

def check_frozen() -> dict:
    train = _jsonl(FROZEN_DIR / "train.jsonl")
    dev = _jsonl(FROZEN_DIR / "dev.jsonl")
    reserve = _jsonl(RESERVE) if RESERVE.is_file() else []
    prov = (FROZEN_DIR / "PROVENANCE.md").read_text(encoding="utf-8")
    m = re.search(r"reserve sha256: `([0-9a-f]+)`", prov)
    reserve_sha = sha256_of(RESERVE) if RESERVE.is_file() else ""
    counts = {"train": len(train), "dev": len(dev), "reserve": len(reserve)}
    problems = []
    for part, minimum in MIN_FROZEN.items():
        if counts[part] != minimum:
            problems.append(
                f"frozen corpus drift: {part} has {counts[part]} rows, "
                f"round 1 froze {minimum}")
    if not m:
        problems.append("round-1 provenance lost the reserve sha pin")
    elif m.group(1) != reserve_sha:
        problems.append("frozen reserve sha drift vs round-1 provenance pin")
    return {"counts": counts, "reserve_sha256": reserve_sha,
            "paths": {"train": str(FROZEN_DIR / "train.jsonl"),
                      "dev": str(FROZEN_DIR / "dev.jsonl"),
                      "reserve": str(RESERVE)},
            "problems": problems}


# --- source 2: BFCL-derived silver ------------------------------------------

def _fetch(url: str, cached: Path) -> list[dict]:
    if not cached.is_file():
        req = urllib.request.Request(
            url, headers={"User-Agent": "mivia-small-json/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(resp.read())
    return _jsonl(cached)


def _gold_string(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def build_silver() -> tuple[list[dict], dict]:
    """Mechanically aligned BFCL silver with per-row alignment evidence.

    A call is admitted only when every required parameter is declared
    string (string-scalar family) AND every required slot aligns via
    span_fill's own high-confidence ladders (key-delimited, then the
    string alignment ladder) with the aligned text EQUAL to the gold
    value after quote-strip. Window/positional/containment alignments are
    excluded: silver is only where alignment is unambiguous."""
    blocked = eval_request_norms()
    rows: list[dict] = []
    stats = {"questions": 0, "string_family_calls": 0, "eval_blocked": 0,
             "aligned": 0, "evidence": {"key_delimited": 0, "alignment": 0},
             "enum_refused": 0, "language_violation": 0}
    for split in BFCL_SPLITS:
        questions = _fetch(f"{BFCL_BASE}/BFCL_v4_{split}.json",
                           Path(f"/tmp/bfcl_v4_{split}.json"))
        answers = {r["id"]: r["ground_truth"] for r in _fetch(
            f"{BFCL_BASE}/possible_answer/BFCL_v4_{split}.json",
            Path(f"/tmp/bfcl_v4_{split}_answer.json"))}
        for rec in questions:
            stats["questions"] += 1
            request = rec["question"][0][0]["content"]
            funcs = {f["name"]: f for f in rec["function"]}
            for turn in answers.get(rec["id"], []):
                for fname, params in turn.items():
                    tool = funcs.get(fname)
                    if not isinstance(tool, dict):
                        continue
                    tparams = tool.get("parameters") or {}
                    props = tparams.get("properties") or {}
                    required = tparams.get("required") or []
                    if not required or not props:
                        continue
                    if not all(str((props.get(s) or {}).get("type", ""))
                               == "string" for s in required):
                        continue
                    stats["string_family_calls"] += 1
                    if norm(request) in blocked:
                        stats["eval_blocked"] += 1
                        continue
                    if non_english_chars(request):
                        # declared English-only (spec data plan): refused
                        # and COUNTED at admission, never admitted
                        stats["language_violation"] += 1
                        continue
                    evidence = []
                    gold: dict = {}
                    ok = True
                    for slot in required:
                        prop = props.get(slot) or {}
                        gv = params.get(slot)
                        if gv is None:
                            ok = False
                            break
                        gs = _gold_string(gv[0] if isinstance(gv, list)
                                          else gv)
                        enum = prop.get("enum") or []
                        if enum and gs not in [str(v) for v in enum]:
                            # strict coercion would refuse it anyway
                            stats["enum_refused"] += 1
                            ok = False
                            break
                        found = None
                        for s, e, raw in sf._key_delimited_spans(request, slot):
                            if raw.strip().strip("\"'") == gs:
                                found = ("key_delimited", s, e, raw)
                                break
                        if found is None:
                            for s, e, raw in sf._alignment_spans_str(
                                    request, slot):
                                if raw.strip().strip("\"'") == gs:
                                    found = ("alignment", s, e, raw)
                                    break
                        if found is None:
                            ok = False
                            break
                        stats["evidence"][found[0]] += 1
                        evidence.append({
                            "slot": slot, "pattern": found[0],
                            "span": [found[1], found[2]], "raw": found[3],
                            "gold": gs})
                        gold[slot] = gs
                    if not ok:
                        continue
                    stats["aligned"] += 1
                    rows.append({
                        "source": f"bfcl-silver:{split}",
                        "tool": tool, "request": request, "gold": gold,
                        "evidence": evidence, "split": "train"})
    return rows, stats


# --- source 2b: ADJUDICATED silver (re-registration 2026-10-10) --------------

def build_silver_adjudicated(path) -> tuple[list[dict], dict]:
    """Silver from the MERGED blind-adjudication agreements
    (.scratch/merge-adjud/merged-agreements.jsonl, produced by
    .scratch/merge-adjud/merge.py from pass1 x pass2-clr under the
    clarified no-gold construct).

    The two labelers' EXACT agreements only (same answered/null per slot,
    same verbatim span; both-unanswerable rows are agreements — they carry
    empty gold and teach abstention). The builder re-verifies, mechanically
    and per row, what the construct claims (measured, never assumed):

    - eval-window exclusion: a request inside any eval surface is dropped
      (the derivation already excluded them; zero is the expected count);
    - enum slots: the agreed value must match a declared member exactly
      (strict coercion would refuse it anyway);
    - verbatim presence: the agreed value must appear in the request text
      (a label the request does not carry cannot teach span selection).

    Keep-first dedupe on the strong-normalized request runs downstream in
    dedupe_check (the standing protocol; parallel instances and multi-tool
    questions collapse, counted, never hidden).
    """
    blocked = eval_request_norms()
    rows: list[dict] = []
    stats = {"merged": 0, "answerable": 0, "unanswerable": 0,
             "eval_blocked": 0, "enum_violation": 0, "verbatim_absent": 0,
             "language_violation": 0,
             "admitted": 0, "evidence": {"adjudicated": 0}}
    for rec in _jsonl(Path(path)):
        stats["merged"] += 1
        request = rec["request"]
        tool = rec["tool"]
        if norm(request) in blocked:
            stats["eval_blocked"] += 1
            continue
        if non_english_chars(request):
            # declared English-only (spec data plan): refused and COUNTED
            # at admission even when every other mechanical filter would
            # pass the row (eval window, enum, verbatim presence)
            stats["language_violation"] += 1
            continue
        props = (tool.get("parameters") or {}).get("properties") or {}
        if rec.get("agreed") == "unanswerable":
            stats["unanswerable"] += 1
            stats["admitted"] += 1
            rows.append({
                "source": "bfcl-silver-adjudicated:" + str(rec.get("split")),
                "tool": tool, "request": request, "gold": {},
                "evidence": [], "split": "train", "unanswerable": True})
            continue
        stats["answerable"] += 1
        gold: dict = {}
        evidence = []
        ok = True
        for slot, value in rec["slots"].items():
            if value is None:  # cannot happen in an agreements-only row
                ok = False
                break
            enum = (props.get(slot) or {}).get("enum") or []
            if enum and value not in [str(v) for v in enum]:
                stats["enum_violation"] += 1
                ok = False
                break
            idx = request.find(value)
            if idx < 0:
                stats["verbatim_absent"] += 1
                ok = False
                break
            evidence.append({
                "slot": slot, "pattern": "adjudicated",
                "span": [idx, idx + len(value)], "raw": value,
                "gold": value})
            gold[slot] = value
        if not ok:
            continue
        stats["evidence"]["adjudicated"] += len(evidence)
        stats["admitted"] += 1
        rows.append({
            "source": "bfcl-silver-adjudicated:" + str(rec.get("split")),
            "tool": tool, "request": request, "gold": gold,
            "evidence": evidence, "split": "train", "unanswerable": False})
    return rows, stats


# --- source 3: harvest gold candidates --------------------------------------

def _tool_from_gold(gold: dict) -> dict | None:
    """Derive a tool schema from a gold tool_call record (mechanical: the
    arguments' JSON types declare the slots; all are required)."""
    try:
        args = json.loads(gold.get("arguments") or "{}")
    except (ValueError, TypeError):
        return None
    if not isinstance(args, dict) or not args:
        return None
    props = {}
    for key, value in args.items():
        if isinstance(value, str):
            t = "string"
        elif isinstance(value, bool):
            t = "boolean"
        elif isinstance(value, int):
            t = "integer"
        elif isinstance(value, float):
            t = "number"
        else:
            return None  # structured slots are out of scope this arm
        props[key] = {"type": t}
    return {"name": str(gold.get("name", "")),
            "description": "", "parameters": {
                "type": "object", "properties": props,
                "required": sorted(args)}}


_INVOCATION_SHAPE = re.compile(r"^\s*\{?\s*\"name\"")


def _admit_harvest_row(raw: str, gold: dict) -> tuple[dict | None, str]:
    """The spec's mechanical admission filter. Returns (row, reason).

    The request side must be a natural-language turn: the committed slice
    stores the invocation JSON itself in ``raw``, and an invocation whose
    only 'request' is its own JSON teaches the model to read JSON spans,
    not request spans — such rows are refused (refusal reason recorded,
    never silently). Admitted rows additionally need EVERY gold argument
    value verbatim in the request text."""
    if not isinstance(gold, dict) or gold.get("type") != "tool_call":
        return None, "not_a_tool_call"
    tool = _tool_from_gold(gold)
    if tool is None:
        return None, "no_derivable_string_schema"
    request = (raw or "").strip()
    if not request or _INVOCATION_SHAPE.match(request):
        # the invocation JSON is not a user request: no request turn
        # accompanies this call in the committed slice
        return None, "no_request_turn"
    if non_english_chars(request):
        # declared English-only register: refused and counted like every
        # other source's admission path
        return None, "language_violation"
    try:
        args = json.loads(gold.get("arguments") or "{}")
    except (ValueError, TypeError):
        return None, "no_derivable_string_schema"
    for key, value in args.items():
        needle = value if isinstance(value, str) else json.dumps(value)
        if needle not in request:
            return None, "value_absent_from_request"
    return ({"source": "harvest-gold", "tool": tool, "request": request,
             "gold": {k: v for k, v in args.items()}, "evidence": [
                 {"slot": k, "pattern": "verbatim-presence",
                  "gold": v if isinstance(v, str) else json.dumps(v)}
                 for k, v in args.items()],
             "split": "train"}, "")


def build_gold() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats = {"scanned": 0, "refusals": {}, "admitted": 0}
    sources: list[Path] = []
    if HARVEST.is_dir():
        sources += sorted(HARVEST.glob("*/tool_call.jsonl"))
    if GH_USER.is_dir():
        sources += sorted(GH_USER.glob("*/tool_call.jsonl")) + \
            sorted(GH_USER.glob("*/*/tool_call.jsonl"))
    for path in sources:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            stats["scanned"] += 1
            rec = json.loads(line)
            row, refusal = _admit_harvest_row(rec.get("raw") or "",
                                              rec.get("gold") or {})
            if row is None:
                stats["refusals"][refusal] = \
                    stats["refusals"].get(refusal, 0) + 1
                continue
            stats["admitted"] += 1
            rows.append(row)
    return rows, stats


# --- weakness probe (spec: run BEFORE the GPU spends) ------------------------

def run_weakness_probe(rows: list[dict]) -> dict:
    """Split the increment into easy mass vs signal — the spec's probe,
    run BEFORE the GPU spends.

    Easy mass: the deterministic ladders ALONE already fill every slot
    with exactly the gold values (measured with a null learned scorer
    injected, so the probe is independent of whether a promoted artifact
    exists today). Signal: the ladders leave slots open or fill them
    wrong — only the learned scorer could fill these, which is the
    increment's actual mass. Guard-rejected: a rendered catalogue the
    shipped guard scores below THRESHOLD never reaches the span-fill
    seam at all; named separately, never counted as either side.
    """
    from mivia_small_json import tool_match
    from mivia_small_json.tool_call import (
        register_no_tool_call_schema, register_tool_call_schema,
        register_tool_clarification_schema)

    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()

    class _NullScorer:
        def pair_probs(self, request, param, prop):
            import re as _re
            return [0.0] * len(_re.findall(r"\w+", request))

    guard = tool_match._get_guard()
    saved = (sf._SPANFILL_CLF, sf._SPANFILL_LOADED)
    sf._SPANFILL_CLF, sf._SPANFILL_LOADED = _NullScorer(), True
    try:
        split = {"n": len(rows), "easy_mass": 0, "signal": 0,
                 "guard_rejected": 0, "abstain_target": 0}
        for row in rows:
            tool, request, gold = row["tool"], row["request"], row["gold"]
            if not gold:
                # adjudicated-unanswerable: the request does NOT carry the
                # values verbatim — the target behavior is the honest
                # abstention (the clarification), never a fill; counted
                # separately so "signal" stays "only the scorer could fill".
                split["abstain_target"] += 1
                continue
            filled, missing = sf.fill_arguments(tool, request)
            if (filled is not None and not missing
                    and set(filled) == set(gold)
                    and all(gold[k] == v or str(gold[k]) == str(v)
                            for k, v in filled.items())):
                split["easy_mass"] += 1
                continue
            if guard is not None:
                catalogue = ("Tools available:\n["
                             + json.dumps(tool)
                             + "]\n\nUser request: " + request)
                try:
                    _best, score = guard.best_match(catalogue)
                except Exception:
                    score = None
                if score is not None and score < tool_match.THRESHOLD:
                    split["guard_rejected"] += 1
                    continue
            split["signal"] += 1
        return split
    finally:
        sf._SPANFILL_CLF, sf._SPANFILL_LOADED = saved


# --- dedupe + composition ----------------------------------------------------

def dedupe_check(rows: list[dict], frozen_norms: set[str]) -> tuple[
        list[dict], dict]:
    """Protocol 2: within-corpus strong-normalized rate <= 1% (duplicates
    DROPPED, keep-first, counted); overlap against the existing mix
    (the frozen corpus) reported, never hidden."""
    seen: set[str] = set()
    kept: list[dict] = []
    dupes = 0
    in_frozen = 0
    for row in rows:
        key = norm(row["request"])
        if key in seen:
            dupes += 1
            continue
        if key in frozen_norms:
            in_frozen += 1
        seen.add(key)
        kept.append(row)
    rate = dupes / max(1, len(rows))
    return kept, {"in": len(rows), "kept": len(kept), "duplicates": dupes,
                  "duplicate_rate": round(rate, 4),
                  "in_frozen_mix": in_frozen}


def composition(frozen: dict, silver: int, gold: int,
                silver_rows: list[dict], gold_rows: list[dict]) -> dict:
    total = frozen["counts"]["train"] + silver + gold
    shares = {
        "frozen": round(frozen["counts"]["train"] / max(1, total), 4),
        "bfcl-silver": round(silver / max(1, total), 4),
        "harvest-gold": round(gold / max(1, total), 4),
    }
    positive_slots = 0
    enum_slots = 0
    for row in silver_rows + gold_rows:
        props = (row["tool"].get("parameters") or {}).get("properties") or {}
        for slot in row["gold"]:
            positive_slots += 1
            if (props.get(slot) or {}).get("enum"):
                enum_slots += 1
    report = {
        "train_total": total,
        "per_source": {"frozen": frozen["counts"]["train"],
                       "bfcl-silver": silver, "harvest-gold": gold,
                       "authored-en": AUTHORED_DECLARED},
        "shares": shares,
        "bounds": {"frozen": list(BOUND_FROZEN_SHARE),
                   "bfcl-silver": list(BOUND_NEW_SOURCE_SHARE),
                   "re_registered": "2026-10-10"},
        "enum_slot_share": round(enum_slots / max(1, positive_slots), 4),
        "enum_slots": enum_slots, "positive_slots": positive_slots,
        "enum_binding": ENUM_SLOT_SHARE_BINDING,
        "authored_cap": AUTHORED_CAP,
    }
    breaches = []
    if not (BOUND_FROZEN_SHARE[0] <= shares["frozen"]
            <= BOUND_FROZEN_SHARE[1]):
        breaches.append(f"frozen share {shares['frozen']} outside "
                        f"{BOUND_FROZEN_SHARE}")
    for name, bound in (("bfcl-silver", shares["bfcl-silver"]),):
        if bound and not (BOUND_NEW_SOURCE_SHARE[0] <= bound
                          <= BOUND_NEW_SOURCE_SHARE[1]):
            breaches.append(f"{name} share {bound} outside "
                            f"{BOUND_NEW_SOURCE_SHARE}")
    if ENUM_SLOT_SHARE_BINDING and positive_slots \
            and enum_slots / positive_slots < 0.10:
        breaches.append(
            f"enum slot share {report['enum_slot_share']} < 0.10")
    if AUTHORED_DECLARED > AUTHORED_CAP:
        breaches.append("authored EN rows above the template cap")
    report["breaches"] = breaches
    return report


# --- provenance (idempotent: replace-my-own-section) -------------------------

def write_provenance(sections: dict, header: str) -> None:
    path = OUT / "PROVENANCE.md"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    text = existing
    for name, body in sections.items():
        pattern = re.compile(
            r"<!-- section:" + re.escape(name) + r" -->\n.*?"
            r"<!-- /section:" + re.escape(name) + r" -->\n?",
            re.S)
        block = (f"<!-- section:{name} -->\n{body}\n"
                 f"<!-- /section:{name} -->\n")
        if pattern.search(text):
            text = pattern.sub(block, text)
        else:
            text = text.rstrip("\n") + ("\n\n" if text.strip() else "") \
                + block
    if header and not text.lstrip("\n").startswith("# "):
        text = header + "\n\n" + text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                      for r in rows)
    path.write_text(payload, encoding="utf-8")
    (path.parent / (path.name + ".sha256")).write_text(
        sha256_of(path) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--probe-only", action="store_true",
                    help="re-run the weakness probe on the written corpus")
    ap.add_argument("--adjudicated", type=Path, default=None,
                    help="path to merged-agreements.jsonl (the blind "
                         "double-adjudication merge, re-registration "
                         "2026-10-10); replaces the mechanical alignment "
                         "ladder as the silver source")
    ap.add_argument("--emit-despite-shortfall", action="store_true",
                    help="consciously emit a shortfall corpus with "
                         "DATA-BLOCKED stamps (orchestrator decision; the "
                         "default FAILs and writes nothing)")
    args = ap.parse_args(argv)
    if not (args.build or args.probe_only):
        ap.error("choose --build / --probe-only")

    frozen = check_frozen()
    if args.probe_only:
        silver = _jsonl(OUT / "bfcl-silver.jsonl") \
            if (OUT / "bfcl-silver.jsonl").is_file() else []
        gold = _jsonl(OUT / "harvest-gold.jsonl") \
            if (OUT / "harvest-gold.jsonl").is_file() else []
        probe = run_weakness_probe(silver + gold)
        print(json.dumps(probe, indent=1))
        report_path = OUT / "composition-report.json"
        report = json.loads(report_path.read_text()) \
            if report_path.is_file() else {}
        report["weakness_probe"] = probe
        OUT.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=1) + "\n")
        write_provenance({"weakness-probe": (
            f"- probe rows: {probe['n']}; easy_mass {probe['easy_mass']}; "
            f"signal {probe['signal']}; guard_rejected "
            f"{probe['guard_rejected']}; abstain_target "
            f"{probe.get('abstain_target', 0)}\n"
            f"- measured {datetime.now(timezone.utc).isoformat()}")}, "")
        return 0

    frozen_norms = set()
    for row in _jsonl(FROZEN_DIR / "train.jsonl"):
        frozen_norms.add(norm(row["request"]))
    try:
        if args.adjudicated is not None:
            silver, silver_stats = build_silver_adjudicated(args.adjudicated)
        else:
            silver, silver_stats = build_silver()
    except FileNotFoundError as exc:
        # a declared eval surface is absent: the exclusion contract cannot
        # be verified — FAIL, write nothing, never skip the window
        print(f"\nGATE FAIL: {exc}", file=sys.stderr)
        return 2
    silver, silver_dedupe = dedupe_check(silver, frozen_norms)
    gold, gold_stats = build_gold()
    gold, gold_dedupe = dedupe_check(gold, frozen_norms)
    probe = run_weakness_probe(silver + gold)
    comp = composition(frozen, len(silver), len(gold), silver, gold)

    shortfalls = []
    # rule 12: a recorded breach that does not act is vacuous — EVERY
    # declared-bound breach the composition step computed FAILS the build,
    # named (per-source shares, template cap, enum when binding)
    for breach in comp["breaches"]:
        shortfalls.append(f"composition bound breach: {breach}")
    if len(silver) < MIN_SILVER:
        shortfalls.append(
            f"bfcl-silver: {len(silver)} admitted rows < declared minimum "
            f"{MIN_SILVER} (post-kappa agreements, re-registered "
            f"2026-10-10; never lowered)")
    if len(gold) > GOLD_MAX:
        shortfalls.append(
            f"harvest-gold: {len(gold)} rows > DROPPED source max "
            f"{GOLD_MAX} (re-registered 2026-10-10; a further "
            f"re-registration is required, never silent weight)")
    for p in frozen["problems"]:
        shortfalls.append(p)

    verdict = {
        "frozen": {**frozen, "dedupe": None},
        "bfcl-silver": {**silver_stats, "dedupe": silver_dedupe},
        "harvest-gold": {**gold_stats, "dedupe": gold_dedupe},
        "composition": comp,
        "weakness_probe": probe,
        "language_enforcement": {
            "declared": list(DECLARED_LANGUAGES),
            "detector": "CJK/Hangul character classes (Han ideographs, "
                        "Kana, Hangul syllables + jamo)",
            "refused_cjk_hangul": {
                "bfcl-silver": silver_stats.get("language_violation", 0),
                "harvest-gold": gold_stats.get("refusals", {}).get(
                    "language_violation", 0)},
            "enforced_since": "2026-10-10",
        },
        "shortfalls": shortfalls,
        "data_blocked": bool(shortfalls),
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(verdict, indent=1, default=str))

    if shortfalls and not args.emit_despite_shortfall:
        print("\nGATE FAIL: declared minimums/bounds unmet — nothing "
              "written.", file=sys.stderr)
        for s in shortfalls:
            print(f"  - {s}", file=sys.stderr)
        print("The arm is DATA-BLOCKED. --emit-despite-shortfall is the "
              "conscious orchestrator override.", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUT / "bfcl-silver.jsonl", silver)
    _write_jsonl(OUT / "harvest-gold.jsonl", gold)
    (OUT / "composition-report.json").write_text(
        json.dumps(verdict, indent=1, default=str) + "\n")
    stamp = ("**DATA-BLOCKED** — declared minimums unmet:\n"
             + "".join(f"- {s}\n" for s in shortfalls)) \
        if shortfalls else "all declared minimums met"
    write_provenance({
        "frozen-corpus": (
            "- source: round-1 frozen spanfill corpus (data/real/spanfill, "
            "reserve in data/heldout), reused AS-IS, never regenerated\n"
            f"- counts: {json.dumps(frozen['counts'])}\n"
            f"- reserve sha256: `{frozen['reserve_sha256']}` (round-1 pin)\n"
            f"- license: original synthetic content (round-1 provenance)"),
        "bfcl-silver": (
            ("- source: blind double-adjudicated BFCL string-family rows "
             "(pass1 x pass2-clr EXACT agreements under the clarified "
             "no-gold construct: verbatim minimal span, positive context, "
             "exact enum match, all-or-nothing per row); merged from "
             + str(args.adjudicated) + "\n"
             "- rules: kappa-gated (>= 0.6 row-level), agreements-only, "
             "disagreements dropped; enum exact-match + verbatim-presence "
             "re-verified mechanically per row; eval windows (calls.jsonl, "
             "irrelevance_eval.jsonl, guard-eval-v2.jsonl) excluded by "
             "replicated request-level window logic\n"
             "- language: declared English-only (spec data plan); "
             "CJK/Hangul requests refused at admission and counted: "
             f"{silver_stats.get('language_violation', 0)} "
             "(enforced 2026-10-10)\n"
             f"- counts: merged {silver_stats['merged']} (answerable "
             f"{silver_stats['answerable']}, unanswerable "
             f"{silver_stats['unanswerable']}), eval-blocked "
             f"{silver_stats['eval_blocked']}, enum-violation "
             f"{silver_stats['enum_violation']}, verbatim-absent "
             f"{silver_stats['verbatim_absent']}, admitted {len(silver)} "
             f"(minimum {MIN_SILVER} post-kappa); dedupe "
             f"{json.dumps(silver_dedupe)}\n"
             f"- sha256: {sha256_of(OUT / 'bfcl-silver.jsonl')
                if (OUT / 'bfcl-silver.jsonl').is_file()
                else 'pending'}\n"
             f"- measured "
             f"{datetime.now(timezone.utc).isoformat()}")
            if args.adjudicated is not None else
            ("- source: BFCL v4 training-side splits + possible_answer "
             "(ShishirPatil/gorilla; the house pair-corpus builder's split "
             "list and /tmp cache)\n"
             "- rules: string-scalar families only; span_fill key-delimited "
             "+ string-alignment ladders; aligned text must EQUAL gold; "
             "eval windows (calls.jsonl, irrelevance_eval.jsonl, "
             "guard-eval-v2.jsonl) excluded by replicated request-level "
             "window logic, zero-overlap verified by normalized set\n"
             "- language: declared English-only (spec data plan); "
             "CJK/Hangul requests refused at admission and counted: "
             f"{silver_stats.get('language_violation', 0)} "
             "(enforced 2026-10-10)\n"
             f"- counts: scanned questions {silver_stats['questions']}, "
             f"string-family calls {silver_stats['string_family_calls']}, "
             f"eval-blocked {silver_stats['eval_blocked']}, admitted "
             f"{len(silver)} (minimum {MIN_SILVER}); dedupe "
             f"{json.dumps(silver_dedupe)}\n"
             f"- sha256: {sha256_of(OUT / 'bfcl-silver.jsonl')
                if (OUT / 'bfcl-silver.jsonl').is_file()
                else 'pending'}\n"
             f"- measured "
             f"{datetime.now(timezone.utc).isoformat()}")),
        "harvest-gold": (
            "- source: multi-harness agent-session harvest "
            "(data/incoming/harvested-sessions/, gitignored; zero-leakage "
            "redaction + session-disjoint partition per its manifest.json) "
            "and data/incoming/gh-user/ (gitignored)\n"
            "- rules: admitted ONLY with a natural-language request turn "
            "(the invocation JSON itself is not a request) and EVERY gold "
            "argument value verbatim in that request (mechanical filter); "
            "declared English-only — CJK/Hangul requests refused and "
            "counted (enforced 2026-10-10)\n"
            "- RE-REGISTERED 2026-10-10: source DROPPED (measured reason: "
            "the committed harvest slice carries no user turns — "
            "4,556/4,842 scanned rows refused no_request_turn, 0 "
            "admitted); max 0 rows\n"
            f"- counts: scanned {gold_stats['scanned']}, admitted "
            f"{len(gold)} (max {GOLD_MAX}); refusals "
            f"{json.dumps(gold_stats['refusals'])}; dedupe "
            f"{json.dumps(gold_dedupe)}\n"
            f"- sha256: {sha256_of(OUT / 'harvest-gold.jsonl')
                if (OUT / 'harvest-gold.jsonl').is_file()
                else 'pending'}\n"
            f"- measured {datetime.now(timezone.utc).isoformat()}"),
        "composition": (
            f"- measured vs declared bounds: {json.dumps(comp, default=str)}\n"
            f"- weakness probe (served path, pre-GPU): "
            f"{json.dumps(probe)}"),
        "verdict": f"- {stamp}",
    }, "# spanfill-stringparam corpus provenance\n")
    print(f"\nwritten: {OUT}")
    if shortfalls:
        print("EMITTED DESPITE SHORTFALL — arm DATA-BLOCKED; the "
              "composition report and provenance carry the stamps.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
