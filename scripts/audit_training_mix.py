#!/usr/bin/env python3
"""Pre-training DATA AUDIT GATE (owner directive 2026-10-10).

No training run starts without this gate PASSING. It replaces the
"reviewed the data, looks ok" step that failed three times (corpus-v3
volume poison, round-1 template/distribution mismatch + BFCL tax).

ARM-AWARE (2026-10-10): the gate stamps one arm at a time; the stamp
carries ``arm`` so a trainer cannot start on another arm's stale pass
(fail-closed, the train_stringparam_scorer.py contract).

``--arm chat-ood`` (default) — the chat-ood router rounds:
 Checks (all must pass; any failure exits 2 and names the row):
  1. composition — per-source and per-language counts vs the DECLARED
     targets in the round's spec (drift is a conscious re-registration,
     never an accident);
  2. class balance — the aux text share of the total mix stays within
     bounds (round-1 measured the BFCL tax above ~16% text share with
     template-heavy EN; the bound is per-round policy);
  3. dedupe — normalized-duplicate rate within the aux corpus;
  4. eval-disjointness — ZERO normalized-text overlaps vs every frozen
     eval surface (data/wild/**, data/wild/realworld-v2/**, held-out,
     ood-set-v2, BFCL eval files incl. the 183-row calls set);
  5. provenance — every aux source file has sha256 + a PROVENANCE entry.

``--arm string-param-span-fill`` — the string-param span-fill arm
(string-param-span-fill; re-registered declaration
dated 2026-10-10, encoded in STRINGPARAM_TARGETS below). MEASURES the
frozen corpus, the bfcl-silver file, and the (dropped) harvest-gold
file from the bytes on disk — never from the builder's composition
report (rule 13: the audit MEASURES, never derives) — and re-verifies
the builder's eval-window exclusion by independent normalized-set
comparison. Writes the stamp with ``arm``, ``passed`` (true AND false:
a failed arm audit leaves a failed stamp so the trainer is blocked),
the measured numbers, and ``extra_sha256`` binding the corpus files'
shas.

``--arm realworld-slices-r1`` — the combined post-commit + slack data
round (realworld-slices-r1, registered 2026-10-10
BEFORE any candidate measurement; encoded in REALWORLD_R1_TARGETS below).
MEASURES the corpus bytes on disk — per-source counts inside the declared
bands, the increment text share <= 0.19 (declared mix math: worst case
14,000/107,290 = 0.131), dedupe rate, lineage integrity (the assembled
aux/extra files must start with the frozen v5/v11 bytes they claim to
extend), the eval-window exclusion re-verified INDEPENDENTLY by
replicating the eval selection rules from the committed raw pools, and
the blind double adjudication (kappa recomputed from the two committed
label files, >= 0.6, full id coverage). Writes the fail-closed stamp with
``arm`` and ``extra_sha256`` binding every file the trainer will load.

Writes .scratch/audit-mix-stamp.json on PASS; the trainer refuses to
train without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STAMP = ROOT / ".scratch/audit-mix-stamp.json"

ARM_CHATOOD = "chat-ood"
ARM_STRINGPARAM = "string-param-span-fill"
ARM_REALWORLD_R1 = "realworld-slices-r1"

# Round-2 declared targets (chat-ood ROUND 2).
# language -> (min_rows, max_rows) in the aux chat corpus.
TARGETS = {
    "en": (2800, 4500),   # 800 templates + real Discord/OASST (round-2)
    "fa": (3800, 5200),   # reinforced: the 65->70 gap
    "ar": (250, 500),
    "de": (600, 820), "fr": (600, 820), "es": (600, 820),
    "pl": (350, 620), "ja": (300, 520), "nl": (200, 500),
}
# Round-3 EXTRA audit (chat-ood ROUND 3, re-declared
# 2026-10-10): the tickets extra corpus (MIVIA_EXTRA_INDOMAIN) gets the
# same gate the aux chat corpus has had since round 2.
EXTRA_OUT = ROOT / "data/real/multilang-tickets/multilang-tickets-v11.jsonl"
EXTRA_AUTHORED_DIR = ROOT / "data/incoming/tickets-extra"
EXTRA_AUTHORED_TARGETS = {"de": 400, "fr": 400, "es": 350, "pt": 300,
                          "greet-de": 150, "greet-fr": 150, "pl": 200}
EXTRA_CSV_DE = 950   # 1,200 taken from the CSV; 250 are structural
                     # v9∩CSV duplicates (same upstream source) dropped
                     # by the builder's dedupe — post-dedupe expectation.
EXTRA_CLASSES = {"issue", "tool_call", "log_event"}
# authored + real-* files both count toward their language; the real-*
# prefix is folded before the target lookup.
EN_TEMPLATE_CAP = 800          # authored EN templates capped (en.jsonl)
AUX_TEXT_SHARE_MAX = 0.19      # round-3 re-declaration (spec): the shipped
                               # chatood2 state is 0.189; the binding
                               # text-side gate is BFCL >= 82, measured.
DUPE_RATE_MAX = 0.01

AUX_DIR = ROOT / "data/incoming/ood-aux-chat"
BASE_AUX = ROOT / "data/real/ood-aux/ood-aux-train-v3.jsonl"
MIX_TOTAL_EXPECTED = (89_000, 92_000)  # round-3 measured band:
# base 67,951 + aux v5 17,173 + v10 5,166 ~= 90.3k (the old
# 92-95k window came from chatood2's disputable 90,909 meta)

# --- string-param-span-fill arm (string-param-span-fill) --
# RE-REGISTERED DECLARATION (orchestrator, on the owner's curation mandate,
# dated 2026-10-10; the spec amendment carries the same date):
#   - harvest-gold source DROPPED (0 rows; measured reason: the committed
#     harvest slice carries no user turns — 4,556/4,842 scanned rows refused
#     "no_request_turn", 0 admitted);
#   - bfcl-silver via ADJUDICATED alignment, admitted minimum 1,000 rows
#     post-kappa (the mechanical-only ladder admitted 1 row);
#   - frozen round-1 corpus as the base (12,407 train / 1,402 dev, reserve
#     3,191 sha-pinned by round-1 provenance), reused as-is;
#   - template count 0 this round (the arm authors nothing);
#   - enum-slot share REPORTED, non-binding this round (measured BFCL
#     enum-string availability is ~4 rows);
#   - composition shares amended: frozen 60-95%, bfcl-silver 5-40%
#     (was 40-70% / 10-40% — the frozen base dominates an ~8-9% silver
#     share at the declared minimums: 12,407 / 13,408 = 92.5%).
STRINGPARAM_CORPUS = ROOT / "data/real/spanfill-stringparam"
STRINGPARAM_FROZEN = ROOT / "data/real/spanfill"
STRINGPARAM_RESERVE = ROOT / "data/heldout/spanfill-reserve.jsonl"
# train/dev sha pins measured 2026-10-10 from the round-1 frozen bytes
# (no round-1 sidecar exists for them; this registration CREATES the pin);
# the reserve pin is read from the round-1 provenance file, never
# hardcoded here (the gate measures the pin, then the bytes against it).
STRINGPARAM_FROZEN_SHA_PINS = {
    "train": "8bf69189eac6f0e11d8dc88d19af85940998227fb7a52554016bde909d7705b6",
    "dev": "3233c8cc5d988088b4b01e17a06d8d6881dd52bdaa58f47df74951e2f9080108",
}
STRINGPARAM_TARGETS = {
    "frozen_train": 12_407,
    "frozen_dev": 1_402,
    "frozen_reserve": 3_191,
    "silver_min": 1_000,        # admitted rows post-kappa
    "gold_max": 0,              # harvest-gold DROPPED this round
    "templates_max": 0,         # authored templates declared 0 this round
    "share_frozen": (0.60, 0.95),
    "share_silver": (0.05, 0.40),
}
# enum-slot share: REPORTED, non-binding this round (see declaration above)
# dedupe ceiling stays the house DUPE_RATE_MAX (protocol 2a).
# The eval surfaces the builder's exclusion contract replicates — the gate
# re-verifies them INDEPENDENTLY (spot re-verification, request-level).
STRINGPARAM_EVAL_FILES = {
    "calls.jsonl": ROOT / "data/incoming/external/bfcl/calls.jsonl",
    "guard-eval-v2.jsonl": ROOT / "data/incoming/external/bfcl/guard-eval-v2.jsonl",
    "irrelevance_eval.jsonl": ROOT / "data/incoming/external/bfcl/irrelevance_eval.jsonl",
}


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").casefold()
    return " ".join(re.compile(r"[\W_]+", re.UNICODE).sub(" ", t).split())


def rows_of(p: Path, key: str = "text") -> list[str]:
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t = r.get(key) or r.get("raw") or ""
        if t:
            out.append(t)
    return out


def eval_norms(full: bool = False) -> set[str]:
    out: set[str] = set()
    pats = ["data/wild/**/*.jsonl", "data/heldout/*.jsonl",
            "data/incoming/ood/*.jsonl",
            "data/incoming/external/bfcl/guard-eval-v2.jsonl",
            "data/incoming/external/bfcl/irrelevance_eval.jsonl",
            # the 183-row fully-specified calls eval set (gap fix
            # 2026-10-10: it was the one BFCL eval surface the gate did
            # not scan; catalogue renderings are composite texts, so the
            # request-level exclusion the arm contract needs lives in the
            # stringparam arm's eval_request_norms below)
            "data/incoming/external/bfcl/calls.jsonl"]
    if full:
        # the bar-3 surfaces eval_ood_class reads + the tickets-de CSV
        # window (the extra's de rows come from that same CSV)
        pats += ["data/real/issue/dev.jsonl", "data/real/log_event/dev.jsonl",
                 "data/real/pull_request/dev.jsonl"]
        sw = ROOT / "data/incoming/external/swebench/swebench_lite_100.json"
        if sw.is_file():
            for entry in json.loads(sw.read_text()):
                t = entry.get("problem_statement") or ""
                if t:
                    out.add(norm(t))
    for pat in pats:
        for p in ROOT.glob(pat):
            if p.name == "PROVENANCE.md":
                continue
            for t in rows_of(p):
                out.add(norm(t))
    if full:
        csv_path = (ROOT / "data/incoming/external/incident-severity"
                    / "tickets-multi-lang-20k.csv")
        if csv_path.is_file():
            import csv as _csv

            taken = 0
            with csv_path.open(encoding="utf-8", errors="replace",
                               newline="") as fh:
                for row in _csv.DictReader(fh):
                    if taken >= 300:
                        break
                    if row.get("language") != "de":
                        continue
                    body = (row.get("body") or "").strip()
                    if len(body) >= 4:
                        out.add(norm(body))
                        taken += 1
    return out


def audit_chatood() -> int:
    failures: list[str] = []
    report: dict = {"composition": {}, "balance": {}, "dupes": {},
                    "disjointness": {}, "provenance": {}}

    # --- 1. composition ------------------------------------------------
    lang_rows: dict[str, list[str]] = {}
    prov_missing = []
    for f in sorted(AUX_DIR.glob("*.jsonl")):
        rs = rows_of(f)
        lang = f.stem[5:] if f.stem.startswith("real-") else f.stem
        lang_rows.setdefault(lang, []).extend(rs)
        if lang not in TARGETS:
            failures.append(f"composition: unexpected language file {f.name}")
        sidecar = f.parent / "PROVENANCE.md"
        if not sidecar.is_file() or f.name not in sidecar.read_text():
            prov_missing.append(f.name)
    for lang, (lo, hi) in TARGETS.items():
        n = len(lang_rows.get(lang, []))
        report["composition"][lang] = n
        if not (lo <= n <= hi):
            failures.append(f"composition: {lang} has {n} rows, target [{lo},{hi}]")
    en_authored = len(rows_of(AUX_DIR / "en.jsonl"))
    if en_authored > EN_TEMPLATE_CAP:
        failures.append(f"composition: authored EN templates {en_authored} > cap {EN_TEMPLATE_CAP}")
    if prov_missing:
        failures.append(f"provenance: no PROVENANCE entry for {prov_missing}")

    # --- 2. balance ------------------------------------------------------
    aux_total = sum(len(v) for v in lang_rows.values())
    base_n = len(rows_of(BASE_AUX))
    extra_n = 0
    if EXTRA_OUT.is_file():
        extra_n = sum(1 for line in EXTRA_OUT.read_text(encoding="utf-8",
                                                     errors="replace")
                      .splitlines() if line.strip())
    # MEASURED, not derived: the shipped meta's n_train_rows has a known
    # 3,019-row unexplained gap for chatood2 (90,909 meta vs 87,890 from
    # the identical computation today) — the mix total is measured from
    # the real inputs instead of trusting any recorded total. The text
    # mass is the file the trainers actually load (MIVIA_OOD_AUX=v5), not
    # the aux source dir.
    aux_trained = len(rows_of(ROOT / "data/real/ood-aux/"
                              "ood-aux-train-v5-chat.jsonl"))
    base_norms: set[str] = set()
    try:
        from mivia_small_json import model as _model

        bx, _by = _model.training_mix(seed=42)
        base_mix = len(bx)
        # protocol 2c: new-corpus rows already inside the existing mix are
        # hidden weight, not new signal — reported and bounded below
        base_norms = {norm(t) for t in bx}
    except Exception as exc:  # pragma: no cover - measurement is best-effort
        base_mix = 76153 - 5436 + base_n
        report["balance"]["base_measure_error"] = repr(exc)
    # base (no aux) + the trained aux file + the tickets extra
    mix_total = base_mix + aux_trained + extra_n
    share = aux_trained / mix_total if mix_total else 1.0
    report["balance"] = {"aux_rows": aux_trained, "extra_rows": extra_n,
                         "est_mix_total": mix_total,
                         "text_share": round(share, 4)}
    if share > AUX_TEXT_SHARE_MAX:
        failures.append(f"balance: aux text share {share:.3f} > {AUX_TEXT_SHARE_MAX}")
    if not (MIX_TOTAL_EXPECTED[0] <= mix_total <= MIX_TOTAL_EXPECTED[1]):
        failures.append(f"balance: mix total {mix_total} outside "
                        f"{MIX_TOTAL_EXPECTED}")

    # --- 3. dedupe -------------------------------------------------------
    all_norms = [norm(t) for rs in lang_rows.values() for t in rs]
    dupes = len(all_norms) - len(set(all_norms))
    rate = dupes / max(1, len(all_norms))
    report["dupes"] = {"n": dupes, "rate": round(rate, 4)}
    if rate > DUPE_RATE_MAX:
        failures.append(f"dupes: {dupes} normalized duplicates ({rate:.3f})")

    # --- 4. eval-disjointness ---------------------------------------------
    ev = eval_norms()
    hits = sum(1 for n in set(all_norms) if n in ev)
    report["disjointness"] = {"eval_surfaces_normed": len(ev), "hits": hits}
    if hits:
        failures.append(f"disjointness: {hits} aux rows overlap frozen eval surfaces")

    # --- 5. round-3 EXTRA (tickets v10) ------------------------------------
    extra_report: dict = {}
    extra_norms: list = []
    if EXTRA_OUT.is_file():
        extra_rows = []
        for line in EXTRA_OUT.read_text(encoding="utf-8",
                                     errors="replace").splitlines():
            if line.strip():
                r = json.loads(line)
                extra_rows.append(r)
        extra_report["rows"] = len(extra_rows)
        cats = {}
        for r in extra_rows:
            cats[r.get("category")] = cats.get(r.get("category"), 0) + 1
        extra_report["classes"] = cats
        bad_cats = set(cats) - EXTRA_CLASSES
        if bad_cats:
            failures.append(f"extra: classes outside whitelist {sorted(bad_cats)}")
        by_src = {}
        for r in extra_rows:
            by_src[r.get("source")] = by_src.get(r.get("source"), 0) + 1
        extra_report["sources"] = by_src
        for lang, want in EXTRA_AUTHORED_TARGETS.items():
            n = by_src.get(f"authored-{lang}", 0)
            if n != want:
                failures.append(f"extra: authored-{lang} {n} != {want}")
        n_csv = by_src.get("tickets-multi-lang-20k-csv-de", 0)
        if n_csv != EXTRA_CSV_DE:
            failures.append(f"extra: csv-de {n_csv} != {EXTRA_CSV_DE}")
        extra_norms = [norm(r.get("text") or "") for r in extra_rows]
        ev_full = eval_norms(full=True)
        extra_hits = sum(1 for n in set(extra_norms) if n in ev_full)
        extra_report["disjointness_full"] = {"hits": extra_hits}
        if extra_hits:
            failures.append(f"extra: {extra_hits} rows overlap eval surfaces "
                            "(incl. CSV window + dev splits)")
        extra_dupes = len(extra_norms) - len(set(extra_norms))
        extra_report["dupes"] = extra_dupes
        if extra_dupes > DUPE_RATE_MAX * len(extra_norms):
            failures.append(f"extra: {extra_dupes} normalized duplicates")
        prov = EXTRA_OUT.parent / "PROVENANCE.md"
        if not (prov.is_file() and EXTRA_OUT.name in prov.read_text()):
            failures.append(f"extra: no PROVENANCE entry for {EXTRA_OUT.name}")
        sidecar = EXTRA_OUT.parent / (EXTRA_OUT.name + ".sha256")
        if sidecar.is_file():
            extra_report["sha256"] = sidecar.read_text().strip()[:16]
        else:
            failures.append("extra: missing sha256 sidecar")
        # blind adjudication artifacts must exist and cover their round:
        # A/B label the round-3 authored files, r4-A/r4-B the round-4 ones
        for who, langs in (("A", ("de", "fr", "es", "pt")),
                           ("B", ("de", "fr", "es", "pt")),
                           ("r4-A", ("greet-de", "greet-fr", "pl")),
                           ("r4-B", ("greet-de", "greet-fr", "pl"))):
            n_authored = sum(EXTRA_AUTHORED_TARGETS[lang] for lang in langs)
            ap = EXTRA_AUTHORED_DIR / f"adjudication-{who}.jsonl"
            n_labels = 0
            if ap.is_file():
                n_labels = sum(1 for line in ap.read_text().splitlines()
                               if line.strip())
            if n_labels != n_authored:
                failures.append(f"extra: adjudication-{who} has {n_labels} "
                                f"labels, its round has {n_authored} rows")
    else:
        failures.append(f"extra: {EXTRA_OUT.name} missing (round-3 gate)")
    # cross-source weight: aux + extra rows already inside the base mix
    if base_norms:
        in_base = sum(1 for n in set(all_norms) | set(extra_norms)
                      if n in base_norms)
        report["cross_source_overlap"] = {"rows_in_base_mix": in_base}
        smaller = min(aux_total or 1, len(base_norms))
        if in_base > 0.01 * smaller:
            failures.append(f"cross-source overlap: {in_base} aux rows are "
                            "duplicates of the existing training mix "
                            "(hidden weight)")
    report["extra"] = extra_report

    # --- verdict ----------------------------------------------------------
    if failures:
        for f in failures:
            print(f"GATE FAIL: {f}", file=sys.stderr)
        print(json.dumps(report, indent=1))
        return 2
    stamp = {"passed": True, "report": report,
             "aux_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                            for f in sorted(AUX_DIR.glob("*.jsonl"))},
             "extra_sha256": (hashlib.sha256(EXTRA_OUT.read_bytes()).hexdigest()
                              if EXTRA_OUT.is_file() else None)}
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps(stamp, indent=1))
    print(json.dumps(report, indent=1))
    print(f"GATE PASS -> {STAMP}")
    return 0


# --- string-param-span-fill arm ---------------------------------------------
# (string-param-span-fill; re-registered declaration
# dated 2026-10-10, encoded in STRINGPARAM_TARGETS above. This audit
# MEASURES the corpus bytes on disk — per-source counts, dedupe rate,
# enum-slot share, template count, the frozen splits' shas — and never
# reads the builder's composition-report.json for a verdict number
# (rule 13: the gate measures, never derives). The builder's eval-window
# exclusion is re-verified INDEPENDENTLY by normalized-set comparison
# against the eval surfaces' own request windows.)


def corpus_rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _rel(path: Path) -> str:
    """Repo-relative key for the stamp's sha binding; absolute for paths
    outside the repo (the test fixtures)."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def eval_request_norms(eval_files: dict) -> set[str]:
    """Request-level window the eval surfaces score — the SAME exclusion
    contract the builder replicates (calls.jsonl / irrelevance_eval.jsonl
    catalogues parsed to their bare request; guard-eval-v2's bare request
    field), re-implemented here so the gate independently re-verifies the
    builder's exclusion instead of trusting its verified-by-construction
    claim (spot re-verification; rule 13)."""
    from mivia_small_json.tool_match import parse_catalogue_raw

    blocked: set[str] = set()
    for name in sorted(eval_files):
        path = Path(eval_files[name])
        for row in corpus_rows(path):
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
            blocked.add(norm(request) if request else norm(text))
    return blocked


def audit_stringparam(corpus_dir: Path = STRINGPARAM_CORPUS,
                      frozen_dir: Path = STRINGPARAM_FROZEN,
                      reserve_path: Path = STRINGPARAM_RESERVE,
                      eval_files: dict | None = None,
                      frozen_sha_pins: dict | None = None,
                      stamp_path: Path = STAMP) -> int:
    """Audit + stamp the string-param-span-fill arm's corpus.

    Writes the stamp with ``passed`` true AND false (fail-closed: a failed
    arm audit leaves a failed stamp at the shared path so the trainer's
    arm+passed check blocks the GPU run), the measured numbers, and
    ``extra_sha256`` binding every corpus file the trainer will load."""
    eval_files = dict(STRINGPARAM_EVAL_FILES) if eval_files is None \
        else dict(eval_files)
    pins = dict(STRINGPARAM_FROZEN_SHA_PINS) if frozen_sha_pins is None \
        else dict(frozen_sha_pins)
    failures: list[str] = []
    report: dict = {
        "declaration": {
            "re_registered": "2026-10-10",
            "harvest_gold": "DROPPED (0 rows; measured reason: the "
                            "committed harvest slice carries no user "
                            "turns)",
            "silver": "adjudicated alignment, admitted minimum 1,000 "
                      "rows post-kappa",
            "enum_slot_share": "REPORTED, non-binding this round "
                               "(measured BFCL enum-string availability "
                               "is ~4 rows)",
            "language": "declared English-only (spec data plan: "
                        "'deliberately English-only'); CJK/Hangul "
                        "requests are refused at admission by the "
                        "builder and re-verified here against the "
                        "corpus bytes",
            "shares": {"frozen": list(STRINGPARAM_TARGETS["share_frozen"]),
                       "bfcl-silver": list(STRINGPARAM_TARGETS["share_silver"])},
        },
        "composition": {}, "frozen_integrity": {}, "dupes": {},
        "disjointness": {}, "provenance": {}}

    # --- 1. frozen corpus integrity (counts + shas vs the round-1 pins) ---
    frozen: dict[str, int] = {}
    paths = {"train": frozen_dir / "train.jsonl",
             "dev": frozen_dir / "dev.jsonl",
             "reserve": reserve_path}
    extra_sha: dict[str, str] = {}
    for part, path in paths.items():
        if not path.is_file():
            failures.append(f"frozen: {part} split missing: {path}")
            continue
        frozen[part] = len(corpus_rows(path))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        extra_sha[_rel(path)] = sha
        report["frozen_integrity"][f"{part}_sha256"] = sha
        if part in pins:
            if sha != pins[part]:
                failures.append(f"frozen: {part} sha drift vs the "
                                "2026-10-10 measured pin")
        else:  # reserve: measured against the round-1 provenance pin
            prov = (frozen_dir / "PROVENANCE.md")
            m = re.search(r"reserve sha256: `([0-9a-f]+)`",
                          prov.read_text(encoding="utf-8")) \
                if prov.is_file() else None
            if not m:
                failures.append("frozen: round-1 provenance lost the "
                                "reserve sha pin")
            else:
                report["frozen_integrity"]["reserve_pin"] = m.group(1)[:16]
                if sha != m.group(1):
                    failures.append("frozen: reserve sha drift vs the "
                                    "round-1 provenance pin")
    for part, want in (("train", STRINGPARAM_TARGETS["frozen_train"]),
                       ("dev", STRINGPARAM_TARGETS["frozen_dev"]),
                       ("reserve", STRINGPARAM_TARGETS["frozen_reserve"])):
        got = frozen.get(part)
        report["composition"][f"frozen_{part}"] = got
        if got is not None and got != want:
            failures.append(f"frozen: {part} has {got} rows, round 1 "
                            f"froze {want}")

    # --- 2. corpus dir: the declared sources and nothing else -------------
    corpus_files: dict[str, Path] = {}
    if not corpus_dir.is_dir():
        failures.append(f"corpus: directory missing: {corpus_dir}")
    else:
        expected = {"bfcl-silver.jsonl", "harvest-gold.jsonl"}
        for p in sorted(corpus_dir.glob("*.jsonl")):
            if p.name not in expected:
                failures.append(f"corpus: unexpected file {p.name} "
                                f"(declared sources: {sorted(expected)})")
            else:
                corpus_files[p.name] = p
        if "bfcl-silver.jsonl" not in corpus_files:
            failures.append("corpus: bfcl-silver.jsonl missing (the arm's "
                            "silver source)")
        for role, p in corpus_files.items():
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            extra_sha[_rel(p)] = sha
            sidecar = p.parent / (p.name + ".sha256")
            if not sidecar.is_file():
                failures.append(f"corpus: missing sha256 sidecar for "
                                f"{p.name}")
            elif sidecar.read_text().strip() != sha:
                failures.append(f"corpus: sha256 sidecar drift for "
                                f"{p.name}")
        prov = corpus_dir / "PROVENANCE.md"
        if not prov.is_file():
            failures.append("corpus: no PROVENANCE.md")
        else:
            prov_text = prov.read_text(encoding="utf-8", errors="replace")
            for name in corpus_files:
                # the house builder writes idempotent sections keyed by the
                # source stem ("<!-- section:bfcl-silver -->"); either the
                # filename or its stem counts as the entry — the bytes stay
                # bound by the sidecar sha check above
                if name in prov_text or Path(name).stem in prov_text:
                    continue
                failures.append(f"corpus: PROVENANCE has no entry for "
                                f"{name}")

    # --- 3. per-source measurement ----------------------------------------
    silver_rows = corpus_rows(corpus_files["bfcl-silver.jsonl"]) \
        if "bfcl-silver.jsonl" in corpus_files else []
    gold_rows = corpus_rows(corpus_files["harvest-gold.jsonl"]) \
        if "harvest-gold.jsonl" in corpus_files else []
    n_silver, n_gold = len(silver_rows), len(gold_rows)
    report["composition"]["bfcl_silver"] = n_silver
    report["composition"]["harvest_gold"] = n_gold
    if n_silver < STRINGPARAM_TARGETS["silver_min"]:
        failures.append(f"composition: bfcl-silver has {n_silver} admitted "
                        f"rows, declared minimum "
                        f"{STRINGPARAM_TARGETS['silver_min']} post-kappa "
                        "(re-registered 2026-10-10)")
    if n_gold > STRINGPARAM_TARGETS["gold_max"]:
        failures.append(f"composition: harvest-gold has {n_gold} rows; the "
                        "source is DROPPED in the 2026-10-10 "
                        "re-registration (a further re-registration is "
                        "required, never silent weight)")
    # templates: authored rows are declared 0 this round — measured from
    # the rows' own source tags, never from a cap arithmetic
    n_authored = sum(1 for r in silver_rows + gold_rows
                     if str(r.get("source", "")).startswith("authored"))
    report["composition"]["templates"] = n_authored
    if n_authored > STRINGPARAM_TARGETS["templates_max"]:
        failures.append(f"composition: authored/template rows {n_authored} "
                        f"> declared "
                        f"{STRINGPARAM_TARGETS['templates_max']} this round")
    # dedupe (protocol 2a), measured on the request text
    src_norms = [norm(r.get("request") or "") for r in silver_rows + gold_rows]
    dupes = len(src_norms) - len(set(src_norms))
    rate = dupes / max(1, len(src_norms))
    report["dupes"] = {"n": dupes, "rate": round(rate, 4)}
    if rate > DUPE_RATE_MAX:
        failures.append(f"dupes: {dupes} normalized duplicates "
                        f"(rate {rate:.4f})")
    # enum-slot share: REPORTED, non-binding this round
    positive_slots = enum_slots = 0
    for r in silver_rows + gold_rows:
        props = ((r.get("tool") or {}).get("parameters") or {}) \
            .get("properties") or {}
        for slot in (r.get("gold") or {}):
            positive_slots += 1
            if (props.get(slot) or {}).get("enum"):
                enum_slots += 1
    report["enum_slots"] = {
        "enum": enum_slots, "positive": positive_slots,
        "share": round(enum_slots / max(1, positive_slots), 4),
        "binding": False}

    # declared language register (spec: "deliberately English-only —
    # declared now so the audit compares against the declaration rather
    # than discovering a mismatch"). The SAME mechanical predicate the
    # builder's admission gate enforces (single source of truth for the
    # character classes); THIS gate still measures the corpus bytes
    # itself and never reads the builder's counts (rule 13: measures,
    # never derives).
    from build_stringparam_corpus import non_english_chars

    flagged_rows = [idx for idx, r in enumerate(silver_rows + gold_rows)
                    if non_english_chars(r.get("request") or "")]
    report["language"] = {
        "declared": ["en"], "violations": len(flagged_rows),
        "first_flagged_rows": flagged_rows[:10]}
    if flagged_rows:
        failures.append(
            f"language: {len(flagged_rows)} corpus row(s) carry CJK/Hangul "
            "characters outside the declared English-only register (spec "
            "data plan; first flagged row indices: "
            f"{flagged_rows[:10]})")

    # --- 4. composition shares (assembled train side) ----------------------
    n_frozen = frozen.get("train", 0)
    total = n_frozen + n_silver + n_gold
    share_frozen = n_frozen / total if total else 0.0
    share_silver = n_silver / total if total else 0.0
    report["composition"]["train_total"] = total
    report["composition"]["shares"] = {
        "frozen": round(share_frozen, 4),
        "bfcl-silver": round(share_silver, 4),
        "harvest-gold": round(n_gold / total, 4) if total else 0.0}
    lo, hi = STRINGPARAM_TARGETS["share_frozen"]
    if not (lo <= share_frozen <= hi):
        failures.append(f"composition: frozen share {share_frozen:.4f} "
                        f"outside declared [{lo}, {hi}]")
    lo, hi = STRINGPARAM_TARGETS["share_silver"]
    if not (lo <= share_silver <= hi):
        failures.append(f"composition: bfcl-silver share "
                        f"{share_silver:.4f} outside declared [{lo}, {hi}]")

    # --- 5. eval-window exclusion, re-verified independently ---------------
    missing = sorted(n for n, p in eval_files.items() if not Path(p).is_file())
    if missing:
        failures.append(f"disjointness: eval surface(s) missing — the "
                        f"exclusion contract cannot be verified: {missing}")
    else:
        ev = eval_request_norms(eval_files)
        hits = sum(1 for n in set(src_norms) if n in ev)
        report["disjointness"] = {
            "eval_requests_normed": len(ev),
            "surfaces": sorted(eval_files), "hits": hits}
        if hits:
            failures.append(f"disjointness: {hits} corpus request(s) "
                            "overlap the eval window "
                            f"({', '.join(sorted(eval_files))})")

    # --- verdict -----------------------------------------------------------
    passed = not failures
    stamp = {"passed": passed, "arm": ARM_STRINGPARAM, "report": report,
             "failures": failures, "extra_sha256": extra_sha}
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=1))
    if failures:
        for f in failures:
            print(f"GATE FAIL: {f}", file=sys.stderr)
        print(json.dumps(report, indent=1))
        print(f"arm stamp (passed=false) written -> {stamp_path}")
        return 2
    print(json.dumps(report, indent=1))
    print(f"GATE PASS -> {stamp_path}")
    return 0


# --- unified-span-spike arm (unified-span-extractor) ------
# Registered phase-1 architecture spike (2026-10-10, BEFORE any candidate
# measurement; single atomic addition by the spike agent). Declaration: the
# EXISTING frozen round-1 corpus ONLY — data/real/spanfill/{train,dev}.jsonl
# (read-only, sha-pinned below, the same pins the stringparam arm carries),
# bfcl-silver 0 rows this spike, harvest-gold 0 (dropped upstream), authored
# templates 0, language en. The gate MEASURES the frozen bytes (shas, row
# counts, slot/enum/multi-word composition, verbatim-presence, dedupe drift)
# and writes the fail-closed stamp the spike trainer requires.
ARM_UNIFIED_SPAN = "unified-span-spike"
UNIFIED_SPAN_SHA_PINS = {
    "train": "8bf69189eac6f0e11d8dc88d19af85940998227fb7a52554016bde909d7705b6",
    "dev": "3233c8cc5d988088b4b01e17a06d8d6881dd52bdaa58f47df74951e2f9080108",
}
UNIFIED_SPAN_TARGETS = {
    "frozen_train": 12_407,
    "frozen_dev": 1_402,
    "bfcl_silver": 0,
    "harvest_gold": 0,
    "templates": 0,
}


def audit_unified_span(frozen_dir: Path | None = None,
                       stamp_path: Path = STAMP) -> int:
    """Audit + stamp the unified-span-spike arm's frozen corpus (rule 13:
    measures the bytes on disk, never derives). Fail-closed: a failed audit
    leaves a failed stamp so the spike trainer's arm+passed check blocks."""
    frozen_dir = frozen_dir or (ROOT / "data/real/spanfill")
    failures: list[str] = []
    report: dict = {
        "declaration": {
            "registered": "2026-10-10 (unified-span-extractor/"
                          "spec.md; phase-1 architecture spike, dev-only "
                          "measurement)",
            "sources": {"frozen_corpus": "data/real/spanfill/"
                                         "{train,dev}.jsonl (read-only, "
                                         "sha-pinned)",
                        "bfcl-silver": 0, "harvest-gold": 0,
                        "templates": 0},
            "language": "en (declared; the corpus is the same frozen bytes "
                        "the stringparam arm audits)",
        },
        "frozen_integrity": {}, "composition": {}, "dupes": {},
        "disjointness": {"note": "spike introduces no new rows; dev is the "
                                 "only measurement surface and is part of "
                                 "the frozen corpus (round-1 gates hold for "
                                 "the held-out/wild surfaces)"},
    }
    extra_sha: dict[str, str] = {}
    rows: dict[str, list[dict]] = {}
    for part in ("train", "dev"):
        path = frozen_dir / f"{part}.jsonl"
        if not path.is_file():
            failures.append(f"frozen: {part} split missing: {path}")
            continue
        rows[part] = corpus_rows(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        extra_sha[_rel(path)] = sha
        report["frozen_integrity"][f"{part}_sha256"] = sha
        if sha != UNIFIED_SPAN_SHA_PINS[part]:
            failures.append(f"frozen: {part} sha drift vs the registered "
                            "2026-10-10 pin")
        want = UNIFIED_SPAN_TARGETS[f"frozen_{part}"]
        if len(rows[part]) != want:
            failures.append(f"frozen: {part} has {len(rows[part])} rows, "
                            f"declared {want}")

    # composition, measured from the bytes (slot/enum/multi-word shares;
    # verbatim presence — the honest-values precondition the spec declares)
    comp: dict = {}
    for part in ("train", "dev"):
        slots = enum_slots = int_slots = mw_slots = verbatim_absent = 0
        for r in rows.get(part, []):
            props = ((r.get("tool") or {}).get("parameters") or {}) \
                .get("properties") or {}
            req = r.get("request") or ""
            for name, value in (r.get("gold") or {}).items():
                slots += 1
                prop = props.get(name) if isinstance(props.get(name),
                                                     dict) else {}
                if prop.get("enum"):
                    enum_slots += 1
                if prop.get("type") == "integer":
                    int_slots += 1
                if isinstance(value, str) and " " in value.strip():
                    mw_slots += 1
                needle = value if isinstance(value, str) \
                    else json.dumps(value)
                if needle not in req:
                    verbatim_absent += 1
        comp[part] = {"rows": len(rows.get(part, [])), "gold_slots": slots,
                      "enum_slots": enum_slots, "integer_slots": int_slots,
                      "multiword_slots": mw_slots,
                      "enum_share": round(enum_slots / max(1, slots), 4),
                      "verbatim_absent_gold": verbatim_absent}
        if verbatim_absent:
            failures.append(f"composition: {part} has {verbatim_absent} "
                            "gold value(s) absent verbatim from the request "
                            "(the declaration is 0)")
    report["composition"] = comp

    # dedupe: REPORTED, non-binding for this arm. The frozen corpus is
    # template-generated (12 templates; rows repeat verbatim by
    # construction — measured 2026-10-10: request-level duplicate rate
    # ~0.978 on both splits), so the house 1% NEW-corpus ceiling does not
    # apply; the sha pins above are the drift guard (identical bytes ->
    # identical rate), and this block exists so the number is MEASURED and
    # on record, never assumed (rule 13).
    report["dupes"] = {"binding": False,
                       "note": "template-generated frozen corpus; the "
                               "sha pins are the drift guard"}
    for part in ("train", "dev"):
        norms = [norm(r.get("request") or "") for r in rows.get(part, [])]
        dupes = len(norms) - len(set(norms))
        rate = dupes / max(1, len(norms))
        report["dupes"][part] = {"n": dupes, "rate": round(rate, 4),
                                 "distinct": len(set(norms))}

    # no new-source files may appear (the declaration is frozen-only)
    for p in sorted(frozen_dir.glob("*.jsonl")):
        if p.name not in ("train.jsonl", "dev.jsonl"):
            failures.append(f"frozen: unexpected file in the frozen corpus "
                            f"dir: {p.name}")

    passed = not failures
    stamp = {"passed": passed, "arm": ARM_UNIFIED_SPAN, "report": report,
             "failures": failures, "extra_sha256": extra_sha}
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=1))
    if failures:
        for f in failures:
            print(f"GATE FAIL: {f}", file=sys.stderr)
        print(json.dumps(report, indent=1))
        print(f"arm stamp (passed=false) written -> {stamp_path}")
        return 2
    print(json.dumps(report, indent=1))
    print(f"GATE PASS -> {stamp_path}")
    return 0


# --- realworld-slices-r1 arm (realworld-slices-r1) --------
# Registered 2026-10-10 BEFORE any candidate measurement (owner order: ONE
# combined data round for the two stuck slices). Declaration encoded here:
# per-source admitted-row bands (source -> (min, max)), the increment text
# share bound with its declared math, the frozen lineage the assembled
# aux/extra files must extend byte-exactly, and the adjudication gate.
# Source files (all under data/real/realworld-r1/):
#   post-commit-r1.jsonl      P  Mastodon dev-announce register  -> text
#   slack-r1.jsonl            S  software_slacks fresh offsets    -> text
#   wrapper-paste-r1.jsonl    D  chat wrapper + fenced error paste -> log_event
#   review-counter-r1.jsonl   C  GitHub PR inline review comments  -> review_comment
#   ood-aux-train-v6-realworld.jsonl  = frozen v5 bytes ++ P ++ S (MIVIA_OOD_AUX)
#   extra-realworld-r1.jsonl          = frozen v11 bytes ++ D ++ C (MIVIA_EXTRA_INDOMAIN)
ARM_REALWORLD_R1 = "realworld-slices-r1"
REALWORLD_R1_DIR = ROOT / "data/real/realworld-r1"
REALWORLD_R1_SOURCES = {
    "post-commit-r1.jsonl": (3_000, 5_500),
    "slack-r1.jsonl": (4_000, 8_500),
    # band amended 2026-10-11 (orchestrator, conscious re-registration
    # recorded in the spec): screen admitted 145, adjudication dropped
    # 1 -> 144; wp is auxiliary texture, binding bars untouched
    "wrapper-paste-r1.jsonl": (100, 500),
    "review-counter-r1.jsonl": (1_200, 2_000),
}
REALWORLD_R1_CLASSES = {"post-commit-r1.jsonl": "text",
                        "slack-r1.jsonl": "text",
                        "wrapper-paste-r1.jsonl": "log_event",
                        "review-counter-r1.jsonl": "review_comment"}
REALWORLD_R1_AUX = REALWORLD_R1_DIR / "ood-aux-train-v6-realworld.jsonl"
REALWORLD_R1_EXTRA = REALWORLD_R1_DIR / "extra-realworld-r1.jsonl"
# the frozen bytes the assembled files claim to extend (the chatood4 pins)
REALWORLD_R1_LINEAGE = {
    "aux": (ROOT / "data/real/ood-aux/ood-aux-train-v5-chat.jsonl",
            17_173, "5b9968c053cede26a60451580c7a16a549509276e04f09bc16b899e7a66cebed"),
    "extra": (ROOT / "data/real/multilang-tickets/multilang-tickets-v11.jsonl",
              5_666, "3f39693efdd67f90d9eb17683b881835127702b66a38a6d70166103c9310e30c"),
}
REALWORLD_R1_MIX_BASE = 90_790   # the promoted chatood4 mix (audit-measured)
REALWORLD_R1_INCREMENT_TEXT_SHARE_MAX = 0.19
# the eval windows the corpus must be disjoint from, by replicated selection
REALWORLD_R1_EVAL = {
    "post-commit": ROOT / "data/wild/realworld/post-commit.jsonl",
    "slack-snippets": ROOT / "data/wild/realworld/slack-snippets.jsonl",
}
REALWORLD_R1_RAW_POOLS = [ROOT / "data/incoming/realworld/raw-commit.jsonl",
                          ROOT / "data/incoming/realworld/raw-slack.jsonl"]
REALWORLD_R1_KAPPA_MIN = 0.6
# the extra-class whitelist for the EXTRA file (v11 classes + this round's)
REALWORLD_R1_EXTRA_CLASSES = {"issue", "tool_call", "log_event",
                              "review_comment"}


def _cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa over paired categorical codings."""
    from collections import Counter

    n = len(a)
    if not n:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _replicate_eval_windows() -> tuple[dict[str, set[str]], dict[str, dict]]:
    """Independently replicate the eval selection rules from the committed
    raw pools (the same rules scripts/collect_realworld_slices.py applied,
    re-implemented HERE so the gate does not trust the builder's claim).
    Returns slice -> normalized-text set, plus diagnostics (replicated pool
    sizes and eval-file label agreement)."""
    import re as _re

    commit_content = _re.compile(
        r"(?i)\b(pushed|committed|merged|merge of|merged in|pull request|"
        r"pr #\d+|deployed to (main|master|prod(uction)?)|landed in)\b")
    release_content = _re.compile(
        r"(?i)\b(v?\d+\.\d+(\.\d+)?\s*(is out|is here|is live|released)|"
        r"released? v?\d|new (release|version)|just (shipped|released)|"
        r"changelog|release notes|shipped v?\d)\b")
    commit_negative = _re.compile(
        r"(?i)(\bcve[- ]?\d|vulnerab|patch (your|now)|threat|malware|"
        r"stars?\s*·|fastest-growing|tutorial|how to\b|guide:|course|"
        r"new face|top contributor|health\s*[:\d]|breach|extortion|"
        r"attackers|credential)")
    email = _re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
    stacky = _re.compile(
        r"(?i)(traceback|exception|uncaught|internal error|error:|failed:|"
        r"fatal\b|panic:|exit code|callstack|segmentation fault|aborted|"
        r"stack trace|compilation failed)")
    # the full override table (key, expected) in collector order — first
    # match wins; demotions precede promotions exactly like the eval's
    # OVERRIDES (the ambiguous flag is not label-bearing, not replicated)
    overrides = [
        ("new blog post “maybe a better way to track", "no_tool_call"),
        ("til about `nbstripout`", "no_tool_call"),
        ("tool called difit", "no_tool_call"),
        ("answer three questions — pushed or not", "no_tool_call"),
        ("qwen38flash", "no_tool_call"),
        ("was getting consistently a 500 error", "no_tool_call"),
        ("selected models in github copilot deprecated", "no_tool_call"),
        ("accessibility statements highlighted on repository overview",
         "no_tool_call"),
        ("cpanel's per-account pricing", "no_tool_call"),
        ("git knowledge check", "no_tool_call"),
        ("gitea v28.0.0", "commit"),
        ("localai v4.11.0", "commit"),
        ("يقدم الإصدار 2.56", "commit"),
        ("highlights from git 2.56", "commit"),
        ("git 2.56 lansat", "commit"),
        ("latest git release brings a dedicated", "commit"),
        ("gitea 28.1.0 is out", "commit")]

    def mask(t: str) -> str:
        return email.sub("<email>", t)

    def collector_norm(t: str) -> str:
        return " ".join(t.strip().lower().split())

    out: dict[str, set[str]] = {"post-commit": set(), "slack-snippets": set()}
    diag: dict[str, dict] = {}

    # post-commit: label every raw-pool post with the replicated rules
    pc_rows = []
    for line in REALWORLD_R1_RAW_POOLS[0].read_text(
            encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        text = mask(json.loads(line)["text"])
        ntext = collector_norm(text)
        override = next((o for o in overrides if o[0] in ntext), None)
        if override:
            pc_rows.append((text, override[1]))
        elif commit_negative.search(text) or text.count("http") > 3:
            pc_rows.append((text, "no_tool_call"))
        elif commit_content.search(text) or release_content.search(text):
            pc_rows.append((text, "commit"))
        else:
            pc_rows.append((text, "no_tool_call"))
    out["post-commit"] = {norm(t) for t, _ in pc_rows}
    ev_pc = [json.loads(ln) for ln in REALWORLD_R1_EVAL["post-commit"]
             .read_text(encoding="utf-8").splitlines() if ln.strip()]
    pc_by_norm = {}
    for t, lab in pc_rows:
        pc_by_norm.setdefault(norm(t), set()).add(lab)
    agree = sum(1 for r in ev_pc
                if r["expected"] in pc_by_norm.get(norm(r["text"]), set()))
    diag["post-commit"] = {"replicated_pool": len(pc_rows),
                           "eval_rows": len(ev_pc),
                           "eval_rows_matched_with_same_label": agree}

    # slack: replicate the filter pool; the eval took its head
    sl_rows = []
    for line in REALWORLD_R1_RAW_POOLS[1].read_text(
            encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        text = mask(html_unescape(json.loads(line)["text"]).strip())
        if not (25 <= len(text) <= 1500):
            continue
        if text.count("http") > len(text) / 60:
            continue
        err_in_fence = any(stacky.search(b)
                           for b in _re.findall(r"```(.*?)```", text, _re.S))
        err_anywhere = bool(stacky.search(text))
        if err_in_fence or (err_anywhere and text.count("\n") >= 3):
            sl_rows.append((text, "log_event"))
        elif err_anywhere:
            sl_rows.append((text, "no_tool_call"))
        else:
            sl_rows.append((text, "no_tool_call"))
    out["slack-snippets"] = {norm(t) for t, _ in sl_rows}
    ev_sl = [json.loads(ln) for ln in REALWORLD_R1_EVAL["slack-snippets"]
             .read_text(encoding="utf-8").splitlines() if ln.strip()]
    sl_by_norm = {}
    for t, lab in sl_rows:
        sl_by_norm.setdefault(norm(t), set()).add(lab)
    agree_sl = sum(1 for r in ev_sl
                   if r["expected"] in sl_by_norm.get(norm(r["text"]), set()))
    diag["slack-snippets"] = {"replicated_pool": len(sl_rows),
                              "eval_rows": len(ev_sl),
                              "eval_rows_matched_with_same_label": agree_sl}
    return out, diag


def html_unescape(text: str) -> str:
    from html import unescape as _u
    return _u(text)


def audit_realworld_r1(stamp_path: Path = STAMP) -> int:
    """Audit + stamp the realworld-slices-r1 corpus (rule 13: measures the
    bytes on disk, never derives). Fail-closed: a failed audit leaves a
    failed stamp so the trainer's arm+passed check blocks the GPU run."""
    failures: list[str] = []
    report: dict = {
        "declaration": {
            "registered": "2026-10-10 (realworld-slices-r1/"
                          "spec.md; BEFORE any candidate measurement)",
            "sources": {k: list(v)
                        for k, v in REALWORLD_R1_SOURCES.items()},
            "labels": dict(REALWORLD_R1_CLASSES),
            "increment_text_share_max":
                REALWORLD_R1_INCREMENT_TEXT_SHARE_MAX,
            "declared_math": "worst case 14000/(90790+14000+2000+500)=0.131"
                             " <= 0.19; brief example ~13k/103,790=0.125",
            "total_text_share": "REPORTED, non-binding (shipped state is"
                                " already 0.189); BFCL >= 82 is the binding"
                                " text-side gate",
            "templates": 0,
        },
        "composition": {}, "lineage": {}, "dupes": {}, "disjointness": {},
        "adjudication": {}, "provenance": {}}
    extra_sha: dict[str, str] = {}

    # --- 1. source corpora: declared files only, bands, sidecars -----------
    src_rows: dict[str, list[dict]] = {}
    if not REALWORLD_R1_DIR.is_dir():
        failures.append(f"corpus: directory missing: {REALWORLD_R1_DIR}")
    declared = dict(REALWORLD_R1_SOURCES)
    declared.update({REALWORLD_R1_AUX.name: None,
                     REALWORLD_R1_EXTRA.name: None,
                     "adjudication-A.jsonl": None,
                     "adjudication-B.jsonl": None,
                     # committed adjudication evidence (rule 5): the sample
                     # rows and the full pass-A labels, 2026-10-11
                     "adjudication-sample.jsonl": None,
                     "candidates-A.jsonl": None})
    for p in sorted(REALWORLD_R1_DIR.glob("*.jsonl")):
        if p.name not in declared:
            failures.append(f"corpus: unexpected file {p.name}")
    for name, band in REALWORLD_R1_SOURCES.items():
        p = REALWORLD_R1_DIR / name
        if not p.is_file():
            failures.append(f"corpus: declared source missing: {name}")
            continue
        rows = corpus_rows(p)
        src_rows[name] = rows
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        extra_sha[_rel(p)] = sha
        report["composition"][name] = len(rows)
        if not (band[0] <= len(rows) <= band[1]):
            failures.append(f"composition: {name} has {len(rows)} rows, "
                            f"declared band [{band[0]}, {band[1]}]")
        # label check, measured from the bytes
        want = REALWORLD_R1_CLASSES[name]
        bad = sum(1 for r in rows if r.get("label") != want)
        if bad:
            failures.append(f"composition: {name} has {bad} rows whose label "
                            f"!= {want}")
        sidecar = p.parent / (p.name + ".sha256")
        if not sidecar.is_file() or sidecar.read_text().strip() != sha:
            failures.append(f"corpus: sha256 sidecar missing/drift for "
                            f"{name}")

    # --- 2. lineage integrity: assembled = frozen bytes ++ increments -----
    for kind, path in (("aux", REALWORLD_R1_AUX),
                       ("extra", REALWORLD_R1_EXTRA)):
        if not path.is_file():
            failures.append(f"lineage: assembled {kind} file missing: {path}")
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        extra_sha[_rel(path)] = sha
        frozen_path, frozen_rows, frozen_sha = REALWORLD_R1_LINEAGE[kind]
        if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != frozen_sha:
            failures.append(f"lineage: frozen {kind} base sha drift vs the "
                            "chatood4 pin")
        inc_rows = corpus_rows(path)
        base_rows = corpus_rows(frozen_path)
        report["lineage"][f"{kind}_rows"] = len(inc_rows)
        report["lineage"][f"{kind}_frozen_rows"] = len(base_rows)
        if len(inc_rows) <= len(base_rows):
            failures.append(f"lineage: {kind} file did not grow "
                            f"({len(inc_rows)} <= {len(base_rows)})")
            continue
        if inc_rows[:len(base_rows)] != base_rows:
            failures.append(f"lineage: {kind} file does not start with the "
                            f"frozen {frozen_path.name} bytes (row order)")
        expected_total = len(base_rows) + sum(
            len(src_rows[n]) for n in REALWORLD_R1_SOURCES
            if (kind == "aux") == (REALWORLD_R1_CLASSES[n] == "text")
            and n in src_rows)
        if len(inc_rows) != expected_total:
            failures.append(f"lineage: {kind} has {len(inc_rows)} rows, "
                            f"frozen {len(base_rows)} + declared increments "
                            f"= {expected_total}")

    # --- 3. increment text share + dedupe ----------------------------------
    n_text = sum(len(src_rows.get(n, [])) for n in
                 ("post-commit-r1.jsonl", "slack-r1.jsonl"))
    n_all_new = sum(len(v) for v in src_rows.values())
    total_mix = REALWORLD_R1_MIX_BASE + n_all_new
    share = n_text / total_mix if total_mix else 1.0
    report["composition"]["new_rows"] = n_all_new
    report["composition"]["increment_text_share"] = round(share, 4)
    if share > REALWORLD_R1_INCREMENT_TEXT_SHARE_MAX:
        failures.append(f"balance: increment text share {share:.4f} > "
                        f"{REALWORLD_R1_INCREMENT_TEXT_SHARE_MAX} (declared "
                        "mix math)")
    src_norms = [norm(r.get("text") or "") for rows in src_rows.values()
                 for r in rows]
    dupes = len(src_norms) - len(set(src_norms))
    rate = dupes / max(1, len(src_norms))
    report["dupes"] = {"n": dupes, "rate": round(rate, 4)}
    if rate > DUPE_RATE_MAX:
        failures.append(f"dupes: {dupes} normalized duplicates "
                        f"(rate {rate:.4f})")

    # --- 4. eval-window exclusion, re-verified independently ---------------
    ev = eval_norms(full=True)
    windows, wdiag = _replicate_eval_windows()
    report["disjointness"]["replication"] = wdiag
    for slice_name, d in wdiag.items():
        # the replication must actually reproduce the committed eval file's
        # labels (>= 95%: email masking can flip a borderline rule) or the
        # gate's window claim is stale and needs conscious re-registration
        n_ev = d["eval_rows"]
        if d["eval_rows_matched_with_same_label"] < 0.95 * n_ev:
            failures.append(
                f"disjointness: {slice_name} replicated selection matches "
                f"{d['eval_rows_matched_with_same_label']}/{n_ev} eval rows "
                "with the same label — the replication is stale")
    blocked = set(ev)
    for texts in windows.values():
        blocked |= texts
    for pool in REALWORLD_R1_RAW_POOLS:
        if pool.is_file():
            for line in pool.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
                if line.strip():
                    blocked.add(norm(json.loads(line).get("text") or ""))
    # extra blocked surfaces specific to this arm: cleancodereview is
    # OFF-limits for training entirely (its benches shuffle the whole file)
    # and the review_comment/chat-ood corpora are the classes this round
    # must not poison
    for extra in ("data/incoming/external/cleancodereview/comments.jsonl",
                  "data/real/review_comment/train.jsonl",
                  "data/real/review_comment/dev.jsonl"):
        p = ROOT / extra
        if p.is_file():
            blocked |= {norm(t) for t in rows_of(p)}
    hits = sum(1 for t in set(src_norms) if t and t in blocked)
    report["disjointness"]["blocked_norms"] = len(blocked)
    report["disjointness"]["hits"] = hits
    if hits:
        failures.append(f"disjointness: {hits} corpus rows overlap the eval "
                        "windows / raw pools / frozen eval surfaces")

    # cross-source weight: corpus rows already inside the existing mix
    try:
        from mivia_small_json import model as _model

        bx, _by = _model.training_mix(seed=42)
        base_norms = {norm(t) for t in bx}
        in_base = sum(1 for t in set(src_norms) if t in base_norms)
        report["disjointness"]["rows_in_base_mix"] = in_base
        if in_base > 0.01 * max(1, len(set(src_norms))):
            failures.append(f"cross-source overlap: {in_base} corpus rows "
                            "are duplicates of the existing training mix "
                            "(hidden weight)")
    except Exception as exc:  # pragma: no cover - best-effort measurement
        report["disjointness"]["base_measure_error"] = repr(exc)

    # --- 5. adjudication: coverage + recomputed kappa ----------------------
    # Registered design (spec, amended pre-build): rater A full coverage
    # (mechanical pre-screen), rater B the declared stratified sample;
    # kappa is recomputed over the PAIRED SAMPLE (dropped rows included —
    # agreement on drops is signal), and no admitted row may carry a DROP
    # coding from either rater.
    la, lb = (REALWORLD_R1_DIR / "adjudication-A.jsonl",
              REALWORLD_R1_DIR / "adjudication-B.jsonl")
    if not (la.is_file() and lb.is_file()):
        failures.append("adjudication: label files missing (adjudication-A"
                        ".jsonl / adjudication-B.jsonl)")
    else:
        rows_by_src = {n: src_rows.get(n, []) for n in REALWORLD_R1_SOURCES}
        ids = [r.get("id") for rows in rows_by_src.values() for r in rows]
        mapa = {r.get("id"): r.get("label") for r in corpus_rows(la)}
        mapb = {r.get("id"): r.get("label") for r in corpus_rows(lb)}
        missing = [i for i in ids if i not in mapa]
        report["adjudication"]["admitted_rows"] = len(ids)
        report["adjudication"]["a_coverage_gaps"] = len(missing)
        if missing:
            failures.append(f"adjudication: {len(missing)} admitted row ids "
                            "missing from rater A (full-coverage file)")
        # B's declared per-source sample minima (id prefixes: pc/sl/wp/rc)
        b_minima = {"pc": 300, "sl": 400, "wp": 100, "rc": 150}
        b_prefix = {"post-commit-r1.jsonl": "pc", "slack-r1.jsonl": "sl",
                    "wrapper-paste-r1.jsonl": "wp",
                    "review-counter-r1.jsonl": "rc"}
        b_ids = [i for i in mapb if isinstance(i, str)]
        for name, pref in b_prefix.items():
            n_src = len(rows_by_src.get(name, []))
            want = min(b_minima[pref], n_src) or b_minima[pref]
            got = sum(1 for i in b_ids if i.startswith(pref))
            report["adjudication"][f"b_sample_{pref}"] = got
            if got < want:
                failures.append(f"adjudication: rater B covers {got} {pref} "
                                f"rows, declared minimum {want}")
        # kappa over the full paired sample (B's codings, A joined by id)
        paired = [(mapa[i], mapb[i]) for i in b_ids
                  if i in mapa and i in mapb]
        if len(paired) != len(b_ids):
            failures.append(f"adjudication: {len(b_ids) - len(paired)} "
                            "rater-B ids missing from rater A — kappa "
                            "cannot be computed over the full sample")
        ka = [str(a).split(":", 1)[0] for a, _ in paired]
        kb = [str(b).split(":", 1)[0] for _, b in paired]
        kappa = _cohen_kappa(ka, kb)
        report["adjudication"]["kappa_recomputed"] = round(kappa, 4)
        report["adjudication"]["paired_rows"] = len(paired)
        report["adjudication"]["kappa_min"] = REALWORLD_R1_KAPPA_MIN
        if kappa < REALWORLD_R1_KAPPA_MIN:
            failures.append(f"adjudication: recomputed kappa {kappa:.4f} < "
                            f"{REALWORLD_R1_KAPPA_MIN}")
        # the drop path is real: any DROP coding removes the row, so
        # admitted rows must carry keep-codings from BOTH raters
        bad_keep = [i for i in ids
                    if str(mapa.get(i, "keep:x")).split(":", 1)[0] == "drop"
                    or str(mapb.get(i, "keep:x")).split(":", 1)[0] == "drop"]
        if bad_keep:
            failures.append(f"adjudication: {len(bad_keep)} admitted rows "
                            "carry a DROP coding from a rater (the drop "
                            "path must remove rows, not admit them)")

    # --- 6. provenance ------------------------------------------------------
    prov = REALWORLD_R1_DIR / "PROVENANCE.md"
    if not prov.is_file():
        failures.append("provenance: no PROVENANCE.md")
    else:
        prov_text = prov.read_text(encoding="utf-8", errors="replace")
        for name in (*REALWORLD_R1_SOURCES, REALWORLD_R1_AUX.name,
                     REALWORLD_R1_EXTRA.name):
            if name in prov_text or Path(name).stem in prov_text:
                continue
            failures.append(f"provenance: no entry for {name}")

    # --- verdict -----------------------------------------------------------
    passed = not failures
    stamp = {"passed": passed, "arm": ARM_REALWORLD_R1, "report": report,
             "failures": failures, "extra_sha256": extra_sha}
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=1))
    if failures:
        for f in failures:
            print(f"GATE FAIL: {f}", file=sys.stderr)
        print(json.dumps(report, indent=1))
        print(f"arm stamp (passed=false) written -> {stamp_path}")
        return 2
    print(json.dumps(report, indent=1))
    print(f"GATE PASS -> {stamp_path}")
    return 0


# --- zero-shot-select-r1 arm (zero-shot-select-r1) --------
# Registered 2026-10-11 BEFORE any candidate measurement. The rank/select
# surface's first entailment-shaped, label-set-diverse corpus: S1 listwise
# sibling negatives (Banking77+CLINC150 train), S2 learned-null pairs,
# S3 synthetic label-set meta-training (blind double adjudication, kappa
# gate), S4 BFCL train split verbatim. The gate re-verifies the builder's
# eval-window exclusion INDEPENDENTLY (its own replication of the consumed
# s42..s46 bench windows and the full eval splits) and binds every corpus
# sha in the fail-closed stamp the select-side trainer requires.
ARM_ZS_SELECT_R1 = "zero-shot-select-r1"
ZS_SELECT_R1_DIR = ROOT / "data/real/zero-shot-select-r1"
ZS_SELECT_R1_BANDS = {
    "listwise-pairs.jsonl": (30_000, 60_000),
    "null-pairs.jsonl": (10_000, 15_000),
    "synthetic-labelsets.jsonl": (35_000, 60_000),
    "bfcl-train.jsonl": (6_700, 6_720),
}
ZS_SELECT_R1_RESERVE = "synthetic-labelsets-reserve.jsonl"
ZS_SELECT_R1_RESERVE_SETS = 8
ZS_SELECT_R1_KAPPA_MIN = 0.6
ZS_SELECT_R1_ADJ_B = 204


def _zs_b77_rows() -> list[tuple[str, str]]:
    import csv as _csv

    rows = []
    with (ROOT / "data/incoming/external/banking77/test.csv").open(
            encoding="utf-8", errors="replace", newline="") as fh:
        for r in _csv.DictReader(fh):
            text = (r.get("text") or "").strip()
            label = (r.get("category") or r.get("label") or "").strip()
            if text and label:
                rows.append((text, label))
    return rows


def _zs_clinc_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_zs_select_r1(corpus_dir: Path = ZS_SELECT_R1_DIR,
                       stamp_path: Path = STAMP) -> int:
    """Audit + stamp the zero-shot-select r1 corpus (fail-closed)."""
    import random as _random

    failures: list[str] = []
    report: dict = {"arm": ARM_ZS_SELECT_R1, "dir": _rel(corpus_dir)}

    corpora: dict[str, list[dict]] = {}
    for name, (lo, hi) in ZS_SELECT_R1_BANDS.items():
        p = corpus_dir / name
        if not p.is_file():
            failures.append(f"missing corpus file {name}")
            continue
        rows = corpus_rows(p)
        corpora[name] = rows
        report.setdefault("counts", {})[name] = len(rows)
        if not (lo <= len(rows) <= hi):
            failures.append(f"{name}: {len(rows)} rows outside declared "
                            f"band [{lo}, {hi}]")
            continue
        keys = [(norm(r.get("request") or ""), r.get("tool") or "")
                for r in rows]
        rate = 1 - len(set(keys)) / max(1, len(keys))
        report.setdefault("pair_dedupe_pct", {})[name] = round(100 * rate, 3)
        if rate > 0.01:
            failures.append(f"{name}: pair dedupe {rate:.4f} > 0.01")

    # --- independent eval-window replication + full-split disjointness ----
    excl: set[str] = set()
    b77_cache = _zs_b77_rows()
    excl |= {norm(t) for t, _ in b77_cache}
    for split in ("validation", "test"):
        excl |= {norm(r["text"]) for r in _zs_clinc_rows(
            ROOT / f"data/incoming/external/clinc150/{split}.jsonl")}
    windows: dict[str, set[str]] = {}
    seed_plan = {"banking77_test": ((42, 43, 44, 45, 46), b77_cache),
                 "clinc150_validation": ((42,),
                                         _zs_clinc_rows(
                                             ROOT / "data/incoming/external/"
                                             "clinc150/validation.jsonl")),
                 "clinc150_test": ((42, 44, 45, 46),
                                   _zs_clinc_rows(
                                       ROOT / "data/incoming/external/"
                                       "clinc150/test.jsonl"))}
    for wname, (seed_list, src) in seed_plan.items():
        for s in seed_list:
            sample = _random.Random(s).sample(src, min(600, len(src)))
            texts = [t for t, _ in sample] if wname == "banking77_test" \
                else [r["text"] for r in sample]
            windows[f"{wname}_s{s}"] = {norm(t) for t in texts}
    ood_valid = {norm(t.strip()) for t in
                 (ROOT / "data/incoming/external/banking77-oos/"
                         "ood-oos__valid__seq.in").read_text(
                     encoding="utf-8").splitlines() if t.strip()}
    id_valid = {norm(t.strip()) for t in
                (ROOT / "data/incoming/external/banking77-oos/"
                        "id-oos__valid__seq.in").read_text(
                    encoding="utf-8").splitlines() if t.strip()}
    excl |= ood_valid | id_valid
    excl |= set().union(*windows.values()) if windows else set()
    report["replicated_windows"] = {k: len(v) for k, v in windows.items()}
    report["exclusion_superset_norms"] = len(excl)
    for name, rows in corpora.items():
        leaked = {norm(r.get("request") or "") for r in rows} & excl
        report.setdefault("eval_leak", {})[name] = len(leaked)
        if leaked:
            failures.append(f"{name}: {len(leaked)} rows overlap the "
                            "replicated eval windows / full eval splits")

    # --- reserve: never trained, domains disjoint --------------------------
    res_path = corpus_dir / ZS_SELECT_R1_RESERVE
    if not res_path.is_file():
        failures.append("missing reserve file")
        reserve = []
    else:
        reserve = [json.loads(line) for line in res_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
    report["reserve_sets"] = len(reserve)
    if len(reserve) != ZS_SELECT_R1_RESERVE_SETS:
        failures.append(f"reserve: {len(reserve)} sets, expected "
                        f"{ZS_SELECT_R1_RESERVE_SETS}")
    res_norms = {norm(u) for r in reserve for lab in r["labels"]
                 for u in lab["utterances"]}
    train_norms = set()
    for rows in corpora.values():
        train_norms |= {norm(r.get("request") or "") for r in rows}
    overlap = res_norms & train_norms
    report["reserve_overlap"] = len(overlap)
    if overlap:
        failures.append(f"reserve: {len(overlap)} utterance norms also in "
                        "the training corpus")
    res_domains = {r["domain"] for r in reserve}
    trained_domains = {
        (r.get("source") or "").replace("synthetic_", "")
        for r in corpora.get("synthetic-labelsets.jsonl", [])}
    bad_domains = {d for d in res_domains
                   if any(td and (td == d or td.startswith(d + "-"))
                          for td in trained_domains)}
    if bad_domains:
        failures.append(f"reserve: domains {sorted(bad_domains)} also appear "
                        "as trained-set sources")

    # --- adjudication: coverage, kappa recomputed, drop path real ---------
    a_path = corpus_dir / "adjudication-A.jsonl"
    b_path = corpus_dir / "adjudication-B.jsonl"
    if not a_path.is_file() or not b_path.is_file():
        failures.append("adjudication: missing A or B evidence file")
    else:
        a_rows = [json.loads(line) for line in a_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        b_rows = [json.loads(line) for line in b_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        s3 = corpora.get("synthetic-labelsets.jsonl", [])
        a_norms = {(r["set_id"], norm(r["text"])): r["id"] for r in a_rows}
        # the adjudicated population = generated label-set utterances
        # (set-bearing positives); the bare yes/no polarity rows are
        # mechanical constructions from the authored fact pairs and are
        # reported, not adjudicated
        s3_pos = [r for r in s3 if r.get("label") == 1 and r.get("set_id")]
        n_polarity = sum(1 for r in s3
                         if r.get("label") == 1 and not r.get("set_id"))
        pos_norms = {(r.get("set_id"), norm(r.get("request") or ""))
                     for r in s3_pos}
        missing_a = sum(1 for k in pos_norms if k not in a_norms)
        report["adjudication"] = {
            "A_rows": len(a_rows), "B_rows": len(b_rows),
            "A_covers_S3_positives": missing_a == 0,
            "polarity_rows_reported_not_adjudicated": n_polarity,
        }
        if missing_a:
            failures.append(f"adjudication: A misses {missing_a} S3 "
                            "positives (full coverage required)")
        if len(b_rows) != ZS_SELECT_R1_ADJ_B:
            failures.append(f"adjudication: B has {len(b_rows)} rows, "
                            f"declared sample {ZS_SELECT_R1_ADJ_B}")
        kappa = _cohen_kappa([r["A_gold"] for r in b_rows],
                             [r["B_pick"] for r in b_rows])
        report["adjudication"]["kappa_recomputed"] = round(kappa, 4)
        report["adjudication"]["kappa_min"] = ZS_SELECT_R1_KAPPA_MIN
        if kappa < ZS_SELECT_R1_KAPPA_MIN:
            failures.append(f"adjudication: recomputed kappa {kappa:.4f} < "
                            f"{ZS_SELECT_R1_KAPPA_MIN}")
        dropped_ids = {r["id"] for r in b_rows if not r["agree"]}
        still_in = sum(
            1 for r in s3 if r.get("label") == 1
            and a_norms.get((r.get("set_id"),
                             norm(r.get("request") or ""))) in dropped_ids)
        if still_in:
            failures.append(f"adjudication: {still_in} B-dropped rows are "
                            "still in the corpus (drop path not applied)")

    # --- provenance --------------------------------------------------------
    prov = corpus_dir / "PROVENANCE.md"
    if not prov.is_file():
        failures.append("provenance: no PROVENANCE.md")
    else:
        prov_text = prov.read_text(encoding="utf-8", errors="replace")
        if "<!-- section:zero-shot-select-r1 -->" not in prov_text:
            failures.append("provenance: no zero-shot-select-r1 section")
        for name in (*ZS_SELECT_R1_BANDS, ZS_SELECT_R1_RESERVE,
                     "adjudication-A.jsonl", "adjudication-B.jsonl"):
            if name not in prov_text:
                failures.append(f"provenance: no entry for {name}")

    # --- stamp: bind EVERY corpus sha --------------------------------------
    extra_sha = {}
    for name in (*ZS_SELECT_R1_BANDS, ZS_SELECT_R1_RESERVE,
                 "adjudication-A.jsonl", "adjudication-B.jsonl"):
        p = corpus_dir / name
        if p.is_file():
            extra_sha[name] = _sha256_of(p)

    passed = not failures
    stamp = {"passed": passed, "arm": ARM_ZS_SELECT_R1, "report": report,
             "failures": failures, "extra_sha256": extra_sha}
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=1))
    if failures:
        for f in failures:
            print(f"GATE FAIL: {f}", file=sys.stderr)
        print(json.dumps(report, indent=1))
        print(f"arm stamp (passed=false) written -> {stamp_path}")
        return 2
    print(json.dumps(report, indent=1))
    print(f"GATE PASS -> {stamp_path}")
    return 0


# --- zero-shot-select-r2 arm (zero-shot-select-r2) --------
# Registered 2026-10-11 BEFORE any candidate measurement. Executes the r1
# falsification's next-round spec: NLI-converted entailment pairs
# (MNLI/SNLI/ANLI train splits -> select shape; the referee defect) plus
# the referee's exact construction over full mechanical pools (verdict
# pairs; the referee's own 500 scored texts EXCLUDED). The r1 corpus files
# are reused IN PLACE — this arm binds their committed sha256 sidecars
# (tamper check) and gates the two NEW files (bands, dedupe, declared NLI
# shares + polarity balance, referee disjointness, independent eval-window
# superset re-replication) into the fail-closed stamp the r2 trainers
# require.
ARM_ZS_SELECT_R2 = "zero-shot-select-r2"
ZS_SELECT_R2_DIR = ROOT / "data/real/zero-shot-select-r2"
ZS_SELECT_R2_BANDS = {
    "nli-converted.jsonl": (80_000, 100_000),
    "verdict-pairs.jsonl": (20_000, 60_000),
}
ZS_SELECT_R2_R1_FILES = ("listwise-pairs.jsonl", "null-pairs.jsonl",
                         "synthetic-labelsets.jsonl", "bfcl-train.jsonl")
# per-source FILE-row bands: sampled rows render 1 pair (dict) or 2 pairs
# (verdict); a handful of eval-overlap drops are inside the slack
ZS_SELECT_R2_NLI_SOURCE_BANDS = {
    "nli_dict_mnli": (13_000, 17_000), "nli_verdict_mnli": (27_000, 31_000),
    "nli_dict_snli": (8_000, 10_500), "nli_verdict_snli": (16_500, 19_500),
    "nli_dict_anli": (5_000, 7_500), "nli_verdict_anli": (11_000, 13_500),
}
ZS_SELECT_R2_BALANCE = (48.0, 52.0)   # positive share, nli file, percent


def audit_zs_select_r2(corpus_dir: Path = ZS_SELECT_R2_DIR,
                       stamp_path: Path = STAMP) -> int:
    """Audit + stamp the zero-shot-select r2 mix (fail-closed)."""
    import random as _random

    failures: list[str] = []
    report: dict = {"arm": ARM_ZS_SELECT_R2, "dir": _rel(corpus_dir)}

    # --- r1 file integrity (r2 trains on r1 bytes) -------------------------
    r1_dir = ROOT / "data/real/zero-shot-select-r1"
    for name in ZS_SELECT_R2_R1_FILES:
        p = r1_dir / name
        side = p.with_suffix(p.suffix + ".sha256")
        if not p.is_file() or not side.is_file():
            failures.append(f"r1 integrity: missing {name} or sidecar")
            continue
        want = side.read_text().strip().split()[0]
        got = _sha256_of(p)
        if got != want:
            failures.append(f"r1 integrity: {name} sha {got[:12]} != "
                            f"sidecar {want[:12]}")

    # --- new files: bands + pair dedupe (within and across) ----------------
    corpora: dict[str, list[dict]] = {}
    for name, (lo, hi) in ZS_SELECT_R2_BANDS.items():
        p = corpus_dir / name
        if not p.is_file():
            failures.append(f"missing corpus file {name}")
            continue
        rows = corpus_rows(p)
        corpora[name] = rows
        report.setdefault("counts", {})[name] = len(rows)
        if not (lo <= len(rows) <= hi):
            failures.append(f"{name}: {len(rows)} rows outside declared "
                            f"band [{lo}, {hi}]")
            continue
        keys = [(norm(r.get("request") or ""), r.get("tool") or "")
                for r in rows]
        rate = 1 - len(set(keys)) / max(1, len(keys))
        report.setdefault("pair_dedupe_pct", {})[name] = round(100 * rate, 3)
        if rate > 0.01:
            failures.append(f"{name}: pair dedupe {rate:.4f} > 0.01")
    if len(corpora) == len(ZS_SELECT_R2_BANDS):
        all_keys = [(norm(r.get("request") or ""), r.get("tool") or "")
                    for rows in corpora.values() for r in rows]
        cross = 1 - len(set(all_keys)) / max(1, len(all_keys))
        report["cross_file_pair_dedupe_pct"] = round(100 * cross, 3)
        if cross > 0.01:
            failures.append(f"cross-file pair dedupe {cross:.4f} > 0.01")

    # --- NLI declared shares + polarity balance (measured from bytes) ------
    nli = corpora.get("nli-converted.jsonl", [])
    src_counts: dict[str, int] = {}
    for r in nli:
        src_counts[r.get("source") or "?"] = \
            src_counts.get(r.get("source") or "?", 0) + 1
    report["nli_source_rows"] = src_counts
    for prefix, (lo, hi) in ZS_SELECT_R2_NLI_SOURCE_BANDS.items():
        n = sum(v for k, v in src_counts.items() if k.startswith(prefix))
        if not (lo <= n <= hi):
            failures.append(f"nli source {prefix}: {n} rows outside band "
                            f"[{lo}, {hi}]")
    if nli:
        pos = 100 * sum(r["label"] for r in nli) / len(nli)
        report["nli_positive_share_pct"] = round(pos, 2)
        lo, hi = ZS_SELECT_R2_BALANCE
        if not (lo <= pos <= hi):
            failures.append(f"nli polarity balance {pos:.2f}% outside "
                            f"[{lo}, {hi}]")
        verdict_rows = [r for r in nli if r["tool"] in ("yes", "no")]
        dict_rows = [r for r in nli if r["tool"] not in ("yes", "no")]
        report["nli_shape_rows"] = {"dict": len(dict_rows),
                                    "verdict": len(verdict_rows)}
        if not dict_rows or not verdict_rows:
            failures.append("nli: both renderings (dict + verdict) required")

    # --- verdict-vs-referee disjointness (= 0) ------------------------------
    referee = ROOT / "data/eval/binary-select-v1.jsonl"
    verdict = corpora.get("verdict-pairs.jsonl", [])
    if referee.is_file() and verdict:
        ref_norms = {norm(json.loads(line)["text"])
                     for line in referee.read_text().splitlines() if line.strip()}
        overlap = {norm(r.get("request") or "") for r in verdict} & ref_norms
        report["referee_overlap"] = len(overlap)
        if overlap:
            failures.append(f"verdict: {len(overlap)} rows overlap the "
                            "referee's scored texts (must be 0)")

    # --- independent eval-window superset re-replication -------------------
    excl: set[str] = set()
    b77_cache = _zs_b77_rows()
    excl |= {norm(t) for t, _ in b77_cache}
    for split in ("validation", "test"):
        excl |= {norm(r["text"]) for r in _zs_clinc_rows(
            ROOT / f"data/incoming/external/clinc150/{split}.jsonl")}
    for s in (42, 43, 44, 45, 46):
        excl |= {norm(t) for t, _ in _random.Random(s).sample(
            b77_cache, min(600, len(b77_cache)))}
    clinc_val = _zs_clinc_rows(
        ROOT / "data/incoming/external/clinc150/validation.jsonl")
    excl |= {norm(r["text"]) for r in _random.Random(42).sample(
        clinc_val, min(600, len(clinc_val)))}
    clinc_test = _zs_clinc_rows(
        ROOT / "data/incoming/external/clinc150/test.jsonl")
    for s in (42, 44, 45, 46):
        excl |= {norm(r["text"]) for r in _random.Random(s).sample(
            clinc_test, min(600, len(clinc_test)))}
    excl |= {norm(t.strip()) for t in
             (ROOT / "data/incoming/external/banking77-oos/"
                     "ood-oos__valid__seq.in").read_text(
                 encoding="utf-8").splitlines() if t.strip()}
    excl |= {norm(t.strip()) for t in
             (ROOT / "data/incoming/external/banking77-oos/"
                     "id-oos__valid__seq.in").read_text(
                 encoding="utf-8").splitlines() if t.strip()}
    report["exclusion_superset_norms"] = len(excl)
    for name, rows in corpora.items():
        leaked = {norm(r.get("request") or "") for r in rows} & excl
        report.setdefault("eval_leak", {})[name] = len(leaked)
        if leaked:
            failures.append(f"{name}: {len(leaked)} rows overlap the "
                            "replicated eval windows / full eval splits")

    # --- provenance (both arms' sections) -----------------------------------
    prov = corpus_dir / "PROVENANCE.md"
    if not prov.is_file():
        failures.append("provenance: no PROVENANCE.md (r2 dir)")
    else:
        prov_text = prov.read_text(encoding="utf-8", errors="replace")
        if "<!-- section:zero-shot-select-r2 -->" not in prov_text:
            failures.append("provenance: no zero-shot-select-r2 section")
        for name in ZS_SELECT_R2_BANDS:
            if name not in prov_text:
                failures.append(f"provenance: no entry for {name}")
    r1_prov = r1_dir / "PROVENANCE.md"
    if not r1_prov.is_file() or \
            "<!-- section:zero-shot-select-r1 -->" not in \
            r1_prov.read_text(encoding="utf-8", errors="replace"):
        failures.append("provenance: r1 dir missing its section")

    # --- stamp: bind EVERY sha the r2 trainers load -------------------------
    extra_sha: dict[str, str] = {}
    for name in ZS_SELECT_R2_R1_FILES:
        p = r1_dir / name
        if p.is_file():
            extra_sha[f"r1/{name}"] = _sha256_of(p)
    for name in ZS_SELECT_R2_BANDS:
        p = corpus_dir / name
        if p.is_file():
            extra_sha[name] = _sha256_of(p)

    passed = not failures
    stamp = {"passed": passed, "arm": ARM_ZS_SELECT_R2, "report": report,
             "failures": failures, "extra_sha256": extra_sha}
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=1))
    if failures:
        for f in failures:
            print(f"GATE FAIL: {f}", file=sys.stderr)
        print(json.dumps(report, indent=1))
        print(f"arm stamp (passed=false) written -> {stamp_path}")
        return 2
    print(json.dumps(report, indent=1))
    print(f"GATE PASS -> {stamp_path}")
    return 0


# ---------------------------------------------------------------------------
# Arm: zero-shot-select r3 (zero-shot-select-r3).
# Listwise-softmax rank round + near-domain oos supply. The r1 corpus files
# AND the r2 referee-recipe files are reused IN PLACE (both sides' sha256
# sidecars bound as tamper checks); the ONE new file is near-oos-pairs.jsonl
# (synthesized in-topic/out-of-intent questions, blind B-passed with an
# impostor-augmented kappa gate). The gate MEASURES the bytes: bands, pair
# dedupe (within + across r1/r2 keys), domain share, null/confusable
# rendering shape, adjudication coverage + kappa recomputed from the
# committed files, drop-path verification, near-oos disjointness vs the r1
# null oos supply and the referee, and the independent eval-window superset
# re-replication — into the fail-closed stamp the r3 trainers require.
ARM_ZS_SELECT_R3 = "zero-shot-select-r3"
ZS_SELECT_R3_DIR = ROOT / "data/real/zero-shot-select-r3"
ZS_SELECT_R3_BANDS = {"near-oos-pairs.jsonl": (10_000, 18_000)}
ZS_SELECT_R3_TEXT_BAND = (3_000, 4_200)
ZS_SELECT_R3_DOMAIN_SHARE = (0.40, 0.60)
ZS_SELECT_R3_KAPPA_MIN = 0.6
ZS_SELECT_R3_R1_FILES = ("listwise-pairs.jsonl", "null-pairs.jsonl",
                         "synthetic-labelsets.jsonl", "bfcl-train.jsonl")
ZS_SELECT_R3_R2_FILES = ("nli-converted.jsonl", "verdict-pairs.jsonl")
ZS_SELECT_R3_NULL_COMBOS = {
    f"{k}: {p}"
    for k in ("__null__", "none", "no_match")
    for p in ("none of these descriptions match",
              "the text is about none of these topics")
}
ZS_SELECT_R3_B_ROWS = 240          # 160 near-oos + 80 r1-train impostors

# --- zero-shot-select r4 (the completing run; zero-shot-select-r4/
# spec.md). r4 trains on the SAME committed bytes as r3 (the r3 gates are
# re-run verbatim) and adds ONE declared-composition gate: the rank-mix
# weighted train mass under the REGISTERED de-dilution weights (per-pair
# pointwise units), MEASURED from the stamped files (rule 13), never
# derived from a meta.
ARM_ZS_SELECT_R4 = "zero-shot-select-r4"
ZS_SELECT_R4_WEIGHTS = {
    "listwise-pairs.jsonl": 4.0,       # r2: 2.0 (the up-weight half)
    "null-pairs.jsonl": 1.0,
    "synthetic-labelsets.jsonl": 1.0,  # synthetic_polarity_* dropped at load
    "bfcl-train.jsonl": 1.0,           # near-miss sources x5.0 (below)
    "nli-converted.jsonl": 0.5,        # r2: 1.0 (the de-dilution half)
    "verdict-pairs.jsonl": 0.5,        # r2: 1.0
}
ZS_SELECT_R4_NEAR_MISS = frozenset({
    "bfcl_distractor_aligned", "bfcl_irrelevance_negatives",
    "synthetic_nearmiss_xlam", "synthetic_nearmiss_when2call"})
ZS_SELECT_R4_NEAR_MISS_WEIGHT = 5.0
ZS_SELECT_R4_LISTWISE_SHARE = (0.60, 0.68)      # measured 63.6% at registration
ZS_SELECT_R4_NLI_VERDICT_SHARE = (0.15, 0.25)   # measured 19.8%


def _zs_select_r4_mass_gate() -> tuple[list[str], dict]:
    """Measure the r4 rank-mix weighted train mass under the declared
    weights over the exact stamped files; gate the declared bands."""
    r1_dir = ROOT / "data/real/zero-shot-select-r1"
    r2_dir = ROOT / "data/real/zero-shot-select-r2"
    files = {
        "listwise-pairs.jsonl": r1_dir, "null-pairs.jsonl": r1_dir,
        "synthetic-labelsets.jsonl": r1_dir, "bfcl-train.jsonl": r1_dir,
        "nli-converted.jsonl": r2_dir, "verdict-pairs.jsonl": r2_dir,
    }
    mass: dict[str, float] = {}
    counts: dict[str, int] = {}
    dropped = 0
    for name, base in files.items():
        w = ZS_SELECT_R4_WEIGHTS[name]
        n = 0
        for r in corpus_rows(base / name):
            if r.get("split") != "train":
                continue
            if name == "synthetic-labelsets.jsonl" and \
                    (r.get("source") or "").startswith("synthetic_polarity_"):
                dropped += 1
                continue
            ww = w
            if name == "bfcl-train.jsonl" and \
                    r.get("source") in ZS_SELECT_R4_NEAR_MISS:
                ww *= ZS_SELECT_R4_NEAR_MISS_WEIGHT
            mass[name] = mass.get(name, 0.0) + ww
            n += 1
        counts[name] = n
    total = sum(mass.values())
    shares = {k: v / total for k, v in mass.items()} if total else {}
    report = {
        "weights": ZS_SELECT_R4_WEIGHTS,
        "train_pairs": counts,
        "polarity_dropped_at_load": dropped,
        "total_train_pairs": sum(counts.values()),
        "weighted_mass_share": {k: round(v, 4) for k, v in shares.items()},
    }
    failures: list[str] = []
    lo, hi = ZS_SELECT_R4_LISTWISE_SHARE
    if not (lo <= shares.get("listwise-pairs.jsonl", 0.0) <= hi):
        failures.append(
            f"r4 mass share: listwise {shares.get('listwise-pairs.jsonl', 0):.4f} "
            f"outside declared [{lo}, {hi}]")
    nv = shares.get("nli-converted.jsonl", 0.0) + \
        shares.get("verdict-pairs.jsonl", 0.0)
    lo, hi = ZS_SELECT_R4_NLI_VERDICT_SHARE
    if not (lo <= nv <= hi):
        failures.append(f"r4 mass share: nli+verdict {nv:.4f} outside "
                        f"declared [{lo}, {hi}]")
    return failures, report


def _zs_select_r3_kappa(a: list[str], b: list[str]) -> float:
    from collections import Counter as _C
    n = len(a)
    if not n:
        return 0.0
    ca, cb = _C(a), _C(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def audit_zs_select_r3(corpus_dir: Path = ZS_SELECT_R3_DIR,
                       stamp_path: Path = STAMP) -> int:
    """Audit + stamp the zero-shot-select r3 mix (fail-closed)."""
    import random as _random

    failures: list[str] = []
    report: dict = {"arm": ARM_ZS_SELECT_R3, "dir": _rel(corpus_dir)}

    # --- r1 + r2 file integrity (r3 trains on those bytes in place) --------
    r1_dir = ROOT / "data/real/zero-shot-select-r1"
    r2_dir = ROOT / "data/real/zero-shot-select-r2"
    for base, names in ((r1_dir, ZS_SELECT_R3_R1_FILES),
                        (r2_dir, ZS_SELECT_R3_R2_FILES)):
        for name in names:
            p = base / name
            side = p.with_suffix(p.suffix + ".sha256")
            if not p.is_file() or not side.is_file():
                failures.append(f"integrity: missing {base.name}/{name} "
                                "or sidecar")
                continue
            want = side.read_text().strip().split()[0]
            got = _sha256_of(p)
            if got != want:
                failures.append(f"integrity: {base.name}/{name} sha "
                                f"{got[:12]} != sidecar {want[:12]}")

    # --- near-oos file: bands + dedupe --------------------------------------
    near = corpus_dir / "near-oos-pairs.jsonl"
    rows: list[dict] = []
    if not near.is_file():
        failures.append("missing corpus file near-oos-pairs.jsonl")
    else:
        rows = corpus_rows(near)
        report["counts"] = {"near-oos-pairs.jsonl": len(rows)}
        lo, hi = ZS_SELECT_R3_BANDS["near-oos-pairs.jsonl"]
        if not (lo <= len(rows) <= hi):
            failures.append(f"near-oos-pairs: {len(rows)} pairs outside "
                            f"band [{lo}, {hi}]")
        keys = [(norm(r.get("request") or ""), r.get("tool") or "")
                for r in rows]
        rate = 1 - len(set(keys)) / max(1, len(keys))
        report["pair_dedupe_pct"] = round(100 * rate, 3)
        if rate > 0.01:
            failures.append(f"near-oos-pairs: pair dedupe {rate:.4f} > 0.01")
        text_norms = {n for n, _ in keys}
        report["distinct_texts"] = len(text_norms)
        tlo, thi = ZS_SELECT_R3_TEXT_BAND
        if not (tlo <= len(text_norms) <= thi):
            failures.append(f"near-oos-pairs: {len(text_norms)} texts "
                            f"outside band [{tlo}, {thi}]")
        # domain share (distinct texts, source prefix)
        dom: dict[str, set[str]] = {}
        for r in rows:
            dom.setdefault((r.get("source") or "?").rsplit("_", 1)[0],
                           set()).add(norm(r.get("request") or ""))
        b = len(dom.get("neardo_banking", set()))
        c = len(dom.get("neardo_clinc", set()))
        if b + c:
            share = b / (b + c)
            report["banking_text_share"] = round(share, 4)
            slo, shi = ZS_SELECT_R3_DOMAIN_SHARE
            if not (slo <= share <= shi):
                failures.append(f"near-oos-pairs: banking share {share:.2f} "
                                f"outside [{slo}, {shi}]")
        # rendering shape: label-1 tools are registered null combos; label-0
        # tools render a registered intent of the row's own domain
        intents = {
            "banking": set(json.loads(
                (ROOT / "data/incoming/external/banking77/categories.json")
                .read_text())),
            "clinc": {line for line in json.loads(
                (ROOT / "data/incoming/external/clinc150/labels.json")
                .read_text()) if line != "oos"},
        }
        bad_null = bad_conf = 0
        for r in rows:
            tool = r.get("tool") or ""
            if r.get("label") == 1:
                if tool not in ZS_SELECT_R3_NULL_COMBOS:
                    bad_null += 1
            else:
                name = tool.split(":", 1)[0].strip()
                d = ((r.get("source") or "").rsplit("_", 1)[0]
                     .replace("neardo_", ""))
                if name not in intents.get(d, set()):
                    bad_conf += 1
        report["rendering"] = {"bad_null_renderings": bad_null,
                               "bad_confusable_intents": bad_conf}
        if bad_null or bad_conf:
            failures.append(f"near-oos-pairs: {bad_null} bad null renderings "
                            f"+ {bad_conf} confusables naming a non-registered"
                            " intent")
        # pair-key dedupe vs r1 + r2 files
        prior_keys: set[tuple[str, str]] = set()
        for base, names in ((r1_dir, ZS_SELECT_R3_R1_FILES),
                            (r2_dir, ZS_SELECT_R3_R2_FILES)):
            for name in names:
                p = base / name
                if p.is_file():
                    for r in corpus_rows(p):
                        prior_keys.add((norm(r.get("request") or ""),
                                        r.get("tool") or ""))
        cross = len(set(keys) & prior_keys)
        report["cross_file_pair_overlap"] = cross
        if cross > max(1, int(0.01 * len(keys))):
            failures.append(f"near-oos-pairs: {cross} pair keys overlap the "
                            "r1/r2 files (> 1%)")

        # --- disjointness: r1 null oos supply + referee ----------------------
        null_oos = {norm(r.get("request") or "") for r in corpus_rows(
            r1_dir / "null-pairs.jsonl")
            if r.get("source") == "null_pos_oos"}
        ov1 = text_norms & null_oos
        report["overlap_r1_null_oos"] = len(ov1)
        if ov1:
            failures.append(f"near-oos: {len(ov1)} texts overlap the r1 null "
                            "oos supply (must be 0)")
        referee = ROOT / "data/eval/binary-select-v1.jsonl"
        ref_norms = {norm(json.loads(line)["text"])
                     for line in referee.read_text().splitlines() if line.strip()}
        ov2 = text_norms & ref_norms
        report["overlap_referee"] = len(ov2)
        if ov2:
            failures.append(f"near-oos: {len(ov2)} texts overlap the "
                            "referee's scored texts (must be 0)")

        # --- adjudication: coverage + kappa + drop path ----------------------
        a_rows = [json.loads(line) for line in
                  (corpus_dir / "adjudication-A.jsonl").read_text().splitlines()
                  if line.strip()]
        b_rows = [json.loads(line) for line in
                  (corpus_dir / "adjudication-B.jsonl").read_text().splitlines()
                  if line.strip()]
        a_near = {norm(m["text"]) for m in a_rows
                  if m.get("area") != "r1_train_impostor"}
        a_imp = [m for m in a_rows if m.get("area") == "r1_train_impostor"]
        missing = text_norms - a_near
        report["adjudication"] = {"A_near_rows":
                                  len(a_near), "A_impostor_rows": len(a_imp),
                                  "B_rows": len(b_rows)}
        if missing:
            failures.append(f"adjudication A: {len(missing)} corpus texts "
                            "have no A row (coverage must be 100%)")
        if len(a_imp) != 80:
            failures.append(f"adjudication A: {len(a_imp)} impostor rows "
                            "(must be 80)")
        if len(b_rows) != ZS_SELECT_R3_B_ROWS:
            failures.append(f"adjudication B: {len(b_rows)} rows (must be "
                            f"{ZS_SELECT_R3_B_ROWS})")
        # impostors must sit verbatim in the r1 listwise TRAIN split with the
        # labeled gold intent (evidence-only rows; never corpus members)
        r1_train: dict[str, str] = {}
        for r in corpus_rows(r1_dir / "listwise-pairs.jsonl"):
            if r.get("label") == 1 and r.get("split") == "train":
                r1_train[norm(r.get("request") or "")] = \
                    (r.get("tool") or "").split(":", 1)[0].strip()
        imp_bad = 0
        for m in a_imp:
            n = norm(m["text"])
            if r1_train.get(n) != m.get("gold"):
                imp_bad += 1
            if norm(m["text"]) in text_norms:
                imp_bad += 1   # an impostor leaked INTO the r3 corpus
        report["adjudication"]["impostor_mismatch_or_leak"] = imp_bad
        if imp_bad:
            failures.append(f"adjudication: {imp_bad} impostor rows not "
                            "verbatim r1-train rows with that gold, or "
                            "leaked into the corpus")
        # kappa recomputed from the committed B file (gold_code vs pick)
        kap = _zs_select_r3_kappa([m["gold_code"] for m in b_rows],
                                  [m["pick"] for m in b_rows])
        report["adjudication"]["kappa"] = round(kap, 4)
        if kap < ZS_SELECT_R3_KAPPA_MIN:
            failures.append(f"adjudication: kappa {kap:.4f} < "
                            f"{ZS_SELECT_R3_KAPPA_MIN}")
        # drop path: every B-rejected near-oos text is OUT of the corpus
        rejected = {norm(m["text"]) for m in b_rows
                    if not m["id"].startswith("impostor")
                    and m["pick"] != "Z"}
        leaked_drops = rejected & text_norms
        report["adjudication"]["b_rejected_texts"] = len(rejected)
        report["adjudication"]["b_rejected_still_in_corpus"] = \
            len(leaked_drops)
        if leaked_drops:
            failures.append(f"adjudication: {len(leaked_drops)} B-rejected "
                            "texts remain in the corpus (drop path violated)")

        # --- independent eval-window superset re-replication ------------------
        excl: set[str] = set()
        b77_cache = _zs_b77_rows()
        excl |= {norm(t) for t, _ in b77_cache}
        for split in ("validation", "test"):
            excl |= {norm(r["text"]) for r in _zs_clinc_rows(
                ROOT / f"data/incoming/external/clinc150/{split}.jsonl")}
        for s in (42, 43, 44, 45, 46):
            excl |= {norm(t) for t, _ in _random.Random(s).sample(
                b77_cache, min(600, len(b77_cache)))}
        clinc_val = _zs_clinc_rows(
            ROOT / "data/incoming/external/clinc150/validation.jsonl")
        excl |= {norm(r["text"]) for r in _random.Random(42).sample(
            clinc_val, min(600, len(clinc_val)))}
        clinc_test = _zs_clinc_rows(
            ROOT / "data/incoming/external/clinc150/test.jsonl")
        for s in (42, 44, 45, 46):
            excl |= {norm(r["text"]) for r in _random.Random(s).sample(
                clinc_test, min(600, len(clinc_test)))}
        excl |= {norm(t.strip()) for t in
                 (ROOT / "data/incoming/external/banking77-oos/"
                         "ood-oos__valid__seq.in").read_text(
                     encoding="utf-8").splitlines() if t.strip()}
        excl |= {norm(t.strip()) for t in
                 (ROOT / "data/incoming/external/banking77-oos/"
                         "id-oos__valid__seq.in").read_text(
                     encoding="utf-8").splitlines() if t.strip()}
        leak = text_norms & excl
        report["eval_leak"] = {"near-oos-pairs.jsonl": len(leak)}
        if leak:
            failures.append(f"near-oos-pairs: {len(leak)} texts overlap the "
                            "replicated eval windows / full eval splits")

    # --- provenance ----------------------------------------------------------
    prov = corpus_dir / "PROVENANCE.md"
    if not prov.is_file():
        failures.append("provenance: no PROVENANCE.md (r3 dir)")
    else:
        prov_text = prov.read_text(encoding="utf-8", errors="replace")
        if "<!-- section:zero-shot-select-r3 -->" not in prov_text:
            failures.append("provenance: no zero-shot-select-r3 section")
        if "near-oos-pairs.jsonl" not in prov_text:
            failures.append("provenance: no entry for near-oos-pairs.jsonl")

    # --- stamp: bind EVERY sha the r3 trainers load --------------------------
    extra_sha: dict[str, str] = {}
    for tag, base, names in (("r1", r1_dir, ZS_SELECT_R3_R1_FILES),
                             ("r2", r2_dir, ZS_SELECT_R3_R2_FILES)):
        for name in names:
            p = base / name
            if p.is_file():
                extra_sha[f"{tag}/{name}"] = _sha256_of(p)
    for name in ("near-oos-pairs.jsonl", "adjudication-A.jsonl",
                 "adjudication-B.jsonl"):
        p = corpus_dir / name
        if p.is_file():
            extra_sha[name] = _sha256_of(p)

    passed = not failures
    stamp = {"passed": passed, "arm": ARM_ZS_SELECT_R3, "report": report,
             "failures": failures, "extra_sha256": extra_sha}
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=1))
    if failures:
        for f in failures:
            print(f"GATE FAIL: {f}", file=sys.stderr)
        print(json.dumps(report, indent=1))
        print(f"arm stamp (passed=false) written -> {stamp_path}")
        return 2
    print(json.dumps(report, indent=1))
    print(f"GATE PASS -> {stamp_path}")
    return 0


def audit_zs_select_r4(corpus_dir: Path = ZS_SELECT_R3_DIR,
                       stamp_path: Path = STAMP) -> int:
    """Audit + stamp the zero-shot-select r4 mix (fail-closed).

    r4 (the completing run) trains on the SAME committed bytes as r3
    (4 x r1 + 2 x r2 + the r3 near-oos file + adjudication evidence), so
    the r3 byte gates apply verbatim and are RE-RUN here unchanged. This
    wrapper adds the one r4-specific gate — the declared-weight weighted
    train mass measured from the stamped files — and re-arms the passing
    stamp under the r4 arm name only if EVERY gate (r3's and the mass
    gate) passed. Any failure leaves a fail-closed stamp; the r4 trainers
    refuse any stamp whose arm is not zero-shot-select-r4.
    """
    mass_fail, mass_report = _zs_select_r4_mass_gate()
    rc = audit_zs_select_r3(corpus_dir, stamp_path)
    if rc != 0:
        # the r3 gates already wrote a fail-closed stamp; re-arm it under
        # the r4 name so the failure is attributable to THIS arm too
        stamp = json.loads(stamp_path.read_text())
        stamp["arm"] = ARM_ZS_SELECT_R4
        stamp["report"]["r4_mass_share"] = mass_report
        stamp_path.write_text(json.dumps(stamp, indent=1))
        return rc
    stamp = json.loads(stamp_path.read_text())
    stamp["arm"] = ARM_ZS_SELECT_R4
    stamp["report"]["r4_mass_share"] = mass_report
    if mass_fail:
        stamp["passed"] = False
        stamp["failures"] = list(stamp.get("failures", [])) + mass_fail
        stamp_path.write_text(json.dumps(stamp, indent=1))
        for f in mass_fail:
            print(f"GATE FAIL: {f}", file=sys.stderr)
        print(f"arm stamp (passed=false) written -> {stamp_path}")
        return 2
    stamp_path.write_text(json.dumps(stamp, indent=1))
    print(f"r4 mass-share gate PASS "
          f"(listwise {mass_report['weighted_mass_share']['listwise-pairs.jsonl']:.4f}, "
          f"nli+verdict "
          f"{(mass_report['weighted_mass_share']['nli-converted.jsonl']
               + mass_report['weighted_mass_share']['verdict-pairs.jsonl']):.4f})")
    print(f"GATE PASS -> {stamp_path} (arm {ARM_ZS_SELECT_R4})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="pre-training data audit gate (arm-aware)")
    ap.add_argument("--arm", choices=(ARM_CHATOOD, ARM_STRINGPARAM,
                                      ARM_UNIFIED_SPAN, ARM_REALWORLD_R1,
                                      ARM_ZS_SELECT_R1, ARM_ZS_SELECT_R2,
                                      ARM_ZS_SELECT_R3, ARM_ZS_SELECT_R4),
                    default=ARM_CHATOOD,
                    help="which training arm's mix to audit and stamp "
                         "(default: the chat-ood router rounds)")
    args = ap.parse_args(argv)
    if args.arm == ARM_STRINGPARAM:
        return audit_stringparam()
    if args.arm == ARM_UNIFIED_SPAN:
        return audit_unified_span()
    if args.arm == ARM_REALWORLD_R1:
        return audit_realworld_r1()
    if args.arm == ARM_ZS_SELECT_R1:
        return audit_zs_select_r1()
    if args.arm == ARM_ZS_SELECT_R2:
        return audit_zs_select_r2()
    if args.arm == ARM_ZS_SELECT_R3:
        return audit_zs_select_r3()
    if args.arm == ARM_ZS_SELECT_R4:
        return audit_zs_select_r4()
    return audit_chatood()


if __name__ == "__main__":
    raise SystemExit(main())
