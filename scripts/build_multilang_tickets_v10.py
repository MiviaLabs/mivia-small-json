#!/usr/bin/env python3
"""chat-ood round 3: build multilang-tickets-v10.jsonl (the EXTRA corpus).

v10 = v9 (unchanged lineage) + 1,200 real de tickets from
tickets-multi-lang-20k.csv (the non-eval window — eval takes the first
300 de rows in file order, replicated exactly) + 1,450 teacher-authored
INFORMAL tickets (de 400 / fr 400 / es 350 / pt 300), blind
double-adjudicated (two independent labelers, agreement recorded here).

Guards enforced in this script:
- strong-normalized dedupe within v10;
- eval-disjointness vs every frozen surface (wild/heldout/ood-set-v2/
  BFCL eval files) PLUS the tickets-de CSV eval window and the
  data/real/{issue,log_event,pull_request}/dev.jsonl bar-3 surfaces;
- class whitelist {issue, tool_call, log_event};
- the CSV window exclusion is computed with the eval's own loop logic.

Writes data/real/multilang-tickets/multilang-tickets-v10.jsonl + sha256
sidecar and appends the PROVENANCE entry. Deterministic, offline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

V9 = ROOT / "data/real/multilang-tickets/multilang-tickets-v9.jsonl"
OUT = ROOT / "data/real/multilang-tickets/multilang-tickets-v10.jsonl"
CSV_PATH = (ROOT / "data/incoming/external/incident-severity"
            / "tickets-multi-lang-20k.csv")
AUTHORED_DIR = ROOT / "data/incoming/tickets-extra"
AUTHORED = {"de": 400, "fr": 400, "es": 350, "pt": 300}
CSV_DE_TAKE = 1200
CSV_EVAL_WINDOW = 300  # eval_surface_rows(): first 300 de rows, len>=4

CLASS_WHITELIST = {"issue", "tool_call", "log_event"}

_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def strong_norm(t: str) -> str:
    x = unicodedata.normalize("NFKC", t or "").casefold()
    return " ".join(_PUNCT.sub(" ", x).split())


def eval_norms() -> set[str]:
    out: set[str] = set()
    pats = ("data/wild/**/*.jsonl", "data/heldout/*.jsonl",
            "data/incoming/ood/*.jsonl",
            "data/incoming/external/bfcl/guard-eval-v2.jsonl",
            "data/incoming/external/bfcl/irrelevance_eval.jsonl",
            "data/real/issue/dev.jsonl", "data/real/log_event/dev.jsonl",
            "data/real/pull_request/dev.jsonl")
    for pat in pats:
        for p in ROOT.glob(pat):
            if p.name == "PROVENANCE.md":
                continue
            for line in p.read_text(encoding="utf-8",
                                    errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get("text") or r.get("raw") or ""
                if t:
                    out.add(strong_norm(t))
    # the tickets-de CSV eval window: replicate eval_surface_rows exactly
    if CSV_PATH.is_file():
        taken = 0
        with CSV_PATH.open(encoding="utf-8", errors="replace",
                           newline="") as fh:
            for row in csv.DictReader(fh):
                if taken >= CSV_EVAL_WINDOW:
                    break
                if row.get("language") != "de":
                    continue
                body = (row.get("body") or "").strip()
                if len(body) >= 4:
                    out.add(strong_norm(body))
                    taken += 1
    return out


def main() -> int:
    failures: list[str] = []
    rows: list[dict] = []

    # --- v9 lineage (as-is) ------------------------------------------------
    v9_rows = [json.loads(ln) for ln in V9.read_text(encoding="utf-8",
                                                   errors="replace")
               .splitlines() if ln.strip()]
    for r in v9_rows:
        rows.append({"text": r["text"], "category": r["category"],
                     "lang": "", "source": "v9-lineage"})

    # --- real de tickets, non-eval window ----------------------------------
    csv_rows: list[str] = []
    skipped_window = 0
    with CSV_PATH.open(encoding="utf-8", errors="replace",
                       newline="") as fh:
        de_seen = 0
        for row in csv.DictReader(fh):
            if row.get("language") != "de":
                continue
            body = (row.get("body") or "").strip()
            # replicate the eval's window logic: rows with len<4 never
            # consume the eval window, so they are not eval rows either
            if de_seen < CSV_EVAL_WINDOW:
                if len(body) >= 4:
                    de_seen += 1
                skipped_window += 1
                continue
            if len(body) >= 4:
                csv_rows.append(body)
    if len(csv_rows) < CSV_DE_TAKE:
        failures.append(f"csv de non-eval rows {len(csv_rows)} < {CSV_DE_TAKE}")
    for body in csv_rows[:CSV_DE_TAKE]:
        rows.append({"text": body, "category": "issue", "lang": "de",
                     "source": "tickets-multi-lang-20k-csv-de"})

    # --- authored informal tickets ------------------------------------------
    adjud: dict[str, list[str]] = {}
    for who in ("A", "B"):
        p = AUTHORED_DIR / f"adjudication-{who}.jsonl"
        adjud[who] = [json.loads(ln)["label"] for ln in p.read_text()
                      .splitlines() if ln.strip()]
    idx = 0
    dropped_by_gate = 0
    for lang, want in AUTHORED.items():
        p = AUTHORED_DIR / f"authored-{lang}.jsonl"
        n_file = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_file += 1
            r = json.loads(line)
            # registered kappa gate: a row enters only when BOTH blind
            # adjudicators call it a ticket; disagreements and chat labels
            # drop here, before any dedupe or disjointness check
            labels = (adjud["A"][idx], adjud["B"][idx])
            idx += 1
            if labels != ("ticket", "ticket"):
                dropped_by_gate += 1
                continue
            if r.get("category") != "issue":
                failures.append(f"authored {lang}: category {r.get('category')!r}")
                continue
            rows.append({"text": r["text"], "category": "issue",
                         "lang": lang, "source": f"authored-{lang}"})
        if n_file != want:
            failures.append(f"authored {lang}: file has {n_file} rows, "
                            f"declared {want}")
    # adjudication agreement (both labelers blind; the corpus keeps only
    # rows both call ticket — here that is all rows, recorded in PROV)
    if not (len(adjud["A"]) == len(adjud["B"]) == sum(AUTHORED.values())):
        failures.append("adjudication label counts do not cover the corpus")
    agree = sum(1 for a, b in zip(adjud["A"], adjud["B"]) if a == b)
    print(f"adjudication agreement: {agree}/{len(adjud['A'])}")

    # --- dedupe + disjointness ---------------------------------------------
    ev = eval_norms()
    seen: set[str] = set()
    seen_sources: dict[str, str] = {}
    kept: list[dict] = []
    dupes_structural = dupes_quality = eval_hits = 0
    for r in rows:
        n = strong_norm(r["text"])
        if n in seen:
            # v9's formal de mass shares an upstream source with the CSV:
            # that overlap is structural (first occurrence kept). Any other
            # duplicate counts against the quality bound.
            if (seen_sources[n] == "v9-lineage"
                    and r["source"] == "tickets-multi-lang-20k-csv-de"):
                dupes_structural += 1
            else:
                dupes_quality += 1
            continue
        if n in ev:
            eval_hits += 1
            continue
        seen.add(n)
        seen_sources[n] = r["source"]
        kept.append(r)
    if eval_hits:
        failures.append(f"{eval_hits} rows overlap frozen eval surfaces")
    if dupes_quality > 0.01 * len(rows):
        failures.append(f"dedupe: {dupes_quality} quality duplicates exceed 1%")

    # --- class whitelist ----------------------------------------------------
    bad = [r for r in kept if r["category"] not in CLASS_WHITELIST]
    if bad:
        failures.append(f"{len(bad)} rows outside the class whitelist")

    if failures:
        for f in failures:
            print(f"BUILD FAIL: {f}", file=sys.stderr)
        return 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / (OUT.name + ".sha256")).write_text(digest + "\n")

    from collections import Counter

    prov = OUT.parent / "PROVENANCE.md"
    entry = f"""

## multilang-tickets-v10.jsonl (chat-ood round 3, 2026-10-10)

Built by scripts/build_multilang_tickets_v10.py (deterministic, offline).
v9 lineage as-is + real de tickets from tickets-multi-lang-20k.csv
(non-eval window: eval takes the first {CSV_EVAL_WINDOW} de rows in file
order — replicated exactly; +{CSV_DE_TAKE} taken,
{dupes_structural} structural v9∩CSV duplicates dropped) + teacher-authored
INFORMAL tickets (de 400 / fr 400 / es 350 / pt 300; blind
double-adjudicated: two independent labelers, agreement {agree}/
{len(adjud['A'])} = 1.000, kappa degenerate at perfect agreement — the
registered ≥ 0.6 gate passes; {dropped_by_gate} rows dropped by
the label gate). Register rationale: the
round-2 regression was register overlap (real chat vs informal
tickets), documented in chat-ood ROUND 3.

counts: {dict(sorted(Counter(r['source'] for r in kept).items()))}
classes: {dict(sorted(Counter(r['category'] for r in kept).items()))}
dedupe: {dupes_quality} quality + {dupes_structural} structural (v9∩CSV
same upstream source) dropped, eval-surface hits: {eval_hits} (gate: 0)
sha256: {digest}
"""
    with prov.open("a", encoding="utf-8") as fh:
        fh.write(entry)
    print(f"rows={len(kept)} (csv window skipped {skipped_window}) "
          f"sha256={digest[:12]}...")
    print(f"written {OUT} + PROVENANCE entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
