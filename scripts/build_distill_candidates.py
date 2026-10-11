#!/usr/bin/env python3
"""Mine the four weak-quadrant candidate rows for the Jev distill corpus.

Spec: jev-teacher-guard-distillation §3.1 (phase P1,
ticket 03). Every row leaves this script with a *construction* label — a
generator whose correctness Jev and the admission rule then audit; Jev is
never the sole source of a label. Deterministic: digest-keyed choices and
seed 20261007 everywhere, no unseeded randomness; the mining walk iterates
the upstream BFCL question order (stable for a fixed snapshot) and cap
allocation therefore follows that order.

Quadrants (mirroring the guard-eval-v2 constructors over DISJOINT requests):

- ``distill_ask_followup``   When2Call ``request_for_info``/``cannot_answer``
  questions the eval did NOT consume (it used 468 unique requests — 600 rows
  over 473 uuids — of 2,299 unique bank questions), paired with their MCQ
  tool lists. The bank is ~65% verbatim BFCL questions, so the disjointness
  filter (not the cap) bounds the yield at 735.
- ``distill_infeasible``     contract-violation mutations of real BFCL gold
  pairs (impossible date / absurd scale / negative quantity): the request
  still fills the parameters (``validate_fillability`` passes — infeasible,
  not merely unfillable) but no listed tool can honestly serve it. ACEBench
  Special is exhausted by the eval (all 403 rows), so this quadrant has no
  untouched real source and is generated.
- ``distill_sibling_swap``   signature-matched sibling substitution on BFCL
  gold pairs — the eval constructor over held-out requests.
- ``distill_no_tool_needed`` the eval's argument-strip construction: only rows
  whose strip actually breaks fillability are kept, so the should-abstain
  label is justified.
- ``distill_positive_twin``  the unmutated (request, gold tool) pair behind
  every emitted mutation row (label 1, gold ground truth). Twins ship only
  where the ORIGINAL request is itself clean: most single-turn-derived
  mutations come from requests already in pairs.jsonl (only the mutated
  string passes the filter), and their positives are already carried by the
  base corpus the final merged corpus folds in unchanged.

Multi-turn gold pairs keep only requests whose gold tool is CONSTANT across
the MT splits that ask them (the pair-well-definedness rule,
reports/exp-pair-well-definedness.md: base/miss_func/miss_param re-ask the
same conversations with different gold trajectories, and 27 requests would
otherwise carry two conflicting gold tools).

The Glaive conversational variant of no_tool_needed (spec §3.1) is dropped:
the arg-strip constructor alone reaches the 600-candidate target, and
pairing chit-chat with "plausible" catalogues would need exactly the kind of
undeterministic plausibility judgment this pipeline avoids. Recorded in the
training report.

Disjointness (mirrors tests/test_bfcl_pairs.py, so the gates pass by
construction): a row is dropped when its request — exact OR
whitespace/case-normalized — appears in pairs.jsonl, guard-eval-v2.jsonl,
calls.jsonl or irrelevance_eval.jsonl, when its 5-gram Jaccard against any
guard-eval-v2 request reaches 0.8 (BFCL template reuse), or when the
(request, tool) pair already exists in an eval set.

Writes data/incoming/external/bfcl/distill-candidates.jsonl plus a meta
JSON (per-source counts, every drop reason) under the same directory.
Offline after the BFCL fetch cache (/tmp/bfcl_v4_*), same as the builders.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_guard_eval as gve  # noqa: E402
import build_tool_match_corpus as btc  # noqa: E402
from mivia_small_json.tool_match import validate_fillability  # noqa: E402

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
OUT = BFCL / "distill-candidates.jsonl"
META = BFCL / "distill-candidates-meta.json"

SEED = 20261007
VERSION = "v1-20261007"
CAPS = {
    "distill_ask_followup": 1200,
    "distill_infeasible": 600,
    "distill_sibling_swap": 600,
    "distill_no_tool_needed": 600,
    "distill_positive_twin": 1800,
}
BLOCKED_FILES = ("pairs.jsonl", "guard-eval-v2.jsonl", "calls.jsonl",
                 "irrelevance_eval.jsonl")
# Multi-turn gold turns are conversational continuations as often as
# self-contained requests; only self-contained ones become training rows.
_CONTINUATION = re.compile(
    r"^(what about|how about|and |also|now |next|then |same |continue|"
    r"that one|the same|again|ok|okay|great|thanks?|yes|no)\b", re.I)
_MIN_REQUEST_WORDS = 6


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _shingles(text: str, n: int = 5) -> set[str]:
    t = _norm(text)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


class Disjointness:
    """Exact + normalized request ban and the Jaccard-0.8 template guard."""

    def __init__(self) -> None:
        self.blocked: set[str] = set()
        for name in BLOCKED_FILES:
            path = BFCL / name
            for rec in gve._rows(path):
                request = rec.get("request")
                if request is None:
                    _tools, request = gve.parse_catalogue_raw(
                        rec.get("text") or rec.get("raw") or "")
                self.blocked.add(request)
                self.blocked.add(_norm(request))
                self.blocked.add(btc._clean(request))
        self.v2 = [r["request"] for r in gve._rows(BFCL / "guard-eval-v2.jsonl")]
        self.v2_shingles = [_shingles(r) for r in self.v2]
        self.index: dict[str, list[int]] = {}
        for i, sh in enumerate(self.v2_shingles):
            for gram in sh:
                self.index.setdefault(gram, []).append(i)
        self.stats = collections.Counter()

    def ok(self, request: str) -> bool:
        if not request:
            self.stats["dropped_empty"] += 1
            return False
        if request in self.blocked or _norm(request) in self.blocked:
            self.stats["dropped_request_overlap"] += 1
            return False
        sh = _shingles(request)
        shared: dict[int, int] = collections.Counter()
        for gram in sh:
            for i in self.index.get(gram, ()):
                shared[i] += 1
        for i, n_shared in shared.items():
            if n_shared < 0.8 * min(len(sh), len(self.v2_shingles[i])):
                continue
            union = len(sh) + len(self.v2_shingles[i]) - n_shared
            if union and n_shared / union >= 0.8:
                self.stats["dropped_near_duplicate"] += 1
                return False
        return True


class RowSink:
    """Dedup, caps, and the single-pair label contract."""

    def __init__(self, disjoint: Disjointness) -> None:
        self.disjoint = disjoint
        self.rows: list[dict] = []
        self.pair_labels: dict[tuple[str, str], int] = {}
        self.per_source: collections.Counter = collections.Counter()
        self.stats = collections.Counter()

    def add(self, request: str, tool: dict, label: int, source: str,
            source_id: str) -> bool:
        request = btc._clean(request)
        if not request or not gve._name(tool):
            self.stats["dropped_empty"] += 1
            return False
        pair = (request, gve._canon(tool))
        # Single-pair contract: one (request, tool) pair carries exactly one
        # construction label. The same request with a DIFFERENT tool under a
        # different label is the designed twin structure, not a conflict.
        if pair in self.pair_labels:
            if self.pair_labels[pair] != label:
                self.stats["dropped_label_conflict"] += 1
            else:
                self.stats["dropped_duplicate_pair"] += 1
            return False
        if not self.disjoint.ok(request):
            self.stats.update(self.disjoint.stats)
            self.disjoint.stats = collections.Counter()
            return False
        if self.per_source[source] >= CAPS[source]:
            self.stats[f"dropped_cap_{source}"] += 1
            return False
        self.pair_labels[pair] = label
        self.per_source[source] += 1
        self.rows.append({
            "request": request, "tool": tool, "label": label,
            "category": source.removeprefix("distill_"),
            "source": source, "source_id": source_id,
            "split": btc._split_of(request, gve._v1(tool)),
            "seed": SEED, "version": VERSION,
        })
        return True


def gold_pairs() -> list[tuple[str, dict, str]]:
    """Real (request, gold tool) pairs from BFCL single-turn gold answers.

    Every split question pairs its request with the tool its
    possible_answers entry names. Requests consumed by the corpus or any
    eval are dropped later by the Disjointness filter, so this walks the
    full pool and lets the filter do the work.
    """
    questions, answers = btc._sources()
    out: list[tuple[str, dict, str]] = []
    for rec in questions:
        by_name = {gve._name(t): t for t in rec.get("function") or []}
        raw = rec.get("question")
        if isinstance(raw, list):
            request = btc._clean(raw[0][0].get("content", ""))
        else:
            request = btc._clean(str(raw or ""))
        # v4 single-turn ground truth: [{'func_name': {param: [vals]}}]
        for entry in answers.get(rec["id"]) or []:
            if isinstance(entry, dict) and entry:
                func = next(iter(entry))
                if func in by_name:
                    out.append((request, by_name[func], str(rec["id"])))
                    break
    return out


def mt_gold_pairs() -> list[tuple[str, dict, str]]:
    """Self-contained gold turns of the multi-turn splits, merge-filtered.

    The corpus builders and the eval keep only the ABSTAINING turns; the
    gold turns (ground_truth non-empty) are untapped real pairs. Two extra
    rules keep the quadrant sound:

    - base/miss_func/miss_param re-ask the SAME conversations with
      different gold trajectories; a request whose gold tool differs
      across the splits that yield it is dropped wholesale (the
      pair-well-definedness rule, reports/exp-pair-well-definedness.md —
      27 such requests measured on 2026-10-07).
    - Turn requests that read as conversational continuations are dropped
      ("what about the second one?" trains the guard on nothing).
    """
    cats = btc._mt_catalogue()
    per_split: dict[str, list[tuple[str, dict, str]]] = {}
    for split in btc.MT_SPLITS:
        questions = btc._fetch(f"{btc.BASE}/BFCL_v4_{split}.json",
                               Path(f"/tmp/bfcl_v4_{split}.json"))
        answers = btc._mt_answers(split)
        out: list[tuple[str, dict, str]] = []
        for rec in questions:
            classes = [c for c in (rec.get("involved_classes") or [])
                       if c in cats]
            # Catalogue tool names carry the class prefix
            # ("GorillaFileSystem.cat"); gold call strings use the bare
            # function name ("cd(folder='document')"). Index by both.
            by_name: dict[str, dict] = {}
            for c in classes:
                for t in cats[c]:
                    full = gve._name(t)
                    by_name[full] = t
                    by_name.setdefault(full.split(".")[-1], t)
                    by_name.setdefault(full.split(".", 1)[-1], t)
            golds = answers.get(rec["id"]) or []
            index = 0
            for turn in rec.get("question") or []:
                if not turn:
                    continue
                request = btc._clean(turn[0].get("content", ""))
                this_turn_golds = (golds[index] if index < len(golds)
                                   else [])
                index += 1
                if not this_turn_golds or not request:
                    continue
                if _CONTINUATION.match(request):
                    continue
                if len(request.split()) < _MIN_REQUEST_WORDS:
                    continue
                # v4 MT ground truth: per turn, python-syntax call strings
                # like "cd(folder='document')"
                for call in this_turn_golds:
                    func = str(call).split("(", 1)[0]
                    if func in by_name:
                        out.append((request, by_name[func],
                                    f"{rec['id']}#t{index - 1}"))
                        break
        per_split[split] = out

    # Merge-filter: keep a request only when every split that yields it
    # agrees on the gold tool (canon comparison).
    by_request: dict[str, dict[str, tuple[dict, str]]] = {}
    for split, rows in per_split.items():
        for request, tool, sid in rows:
            by_request.setdefault(request, {})[split] = (tool, sid)
    merged: list[tuple[str, dict, str]] = []
    conflicts = 0
    for request, hits in by_request.items():
        canon_tools = {gve._canon(tool) for tool, _sid in hits.values()}
        if len(canon_tools) > 1:
            conflicts += 1
            continue
        tool, sid = next(iter(hits.values()))
        merged.append((request, tool, sid))
    print(f"mt merge filter: {len(merged)} kept, "
          f"{conflicts} cross-split gold conflicts dropped")
    return merged


def when2call_rows(used_questions: set[str]) -> list[tuple[str, list[dict], str]]:
    """When2Call ask-rows the eval did not consume, deduped by question."""
    out = []
    seen: set[str] = set()
    path = btc.WHEN2CALL_DIR / "when2call_test_mcq.jsonl"
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("correct_answer") not in ("request_for_info",
                                             "cannot_answer"):
            continue
        request = btc._clean(rec.get("question") or "")
        if not request or _norm(request) in used_questions:
            continue
        if _norm(request) in seen:
            continue  # one row per question keeps the pool wide
        seen.add(_norm(request))
        out.append((request, btc._load_tool_list(rec.get("tools")),
                    str(rec.get("uuid") or "")))
    return out


_DATE_HINT = re.compile(r"date|day|from|to|when|year|month", re.I)
_SCALE_HINT = re.compile(r"num|count|amount|quantity|qty|limit|size|"
                         r"max|min|budget|duration|people|guests?|adults?|"
                         r"children|rooms?|nights?|days?|hours?|miles|km",
                         re.I)
_ABSURD = ("10000000000", "-7")


def infeasible_mutation(request: str, tool: dict) -> str | None:
    """One contract-violating rewrite of a gold request, chosen by hash.

    The mutated request must still FILL the tool's parameters (the guard's
    fillability gate sees a number/date and passes — that is what makes the
    row infeasible rather than merely unfillable) while naming something no
    real service can honor. ``None`` when no template applies.
    """
    params = (tool.get("parameters") or {}).get("properties") or {}
    applicable = []
    if any(_DATE_HINT.search(k) for k in params):
        applicable.append(lambda r: f"{r}, on February 30th of the year 1800")
    if any(_SCALE_HINT.search(k) for k in params):
        applicable.append(lambda r: f"{r} — quantity {_ABSURD[0]} (ten billion)")
        applicable.append(lambda r: f"{r} for {_ABSURD[1]} (negative seven) units")
    if not applicable:
        return None
    pick = int(btc._digest(request + gve._name(tool)), 16) % len(applicable)
    return applicable[pick](request)


def main() -> int:
    disjoint = Disjointness()
    sink = RowSink(disjoint)

    # ---- eval-consumed When2Call questions leave the pool (eval requests
    # are stored _clean()ed; compare in the same space)
    v2 = gve._rows(BFCL / "guard-eval-v2.jsonl")
    used_w2c = {_norm(btc._clean(r["request"])) for r in v2
                if r["source"].startswith("when2call_")}

    # ---- real gold pairs: single-turn answers + self-contained MT turns
    single = gold_pairs()
    multi = mt_gold_pairs()
    base_pairs = single + multi
    print(f"gold pair pool: {len(base_pairs)} "
          f"(single-turn {len(single)} + mt {len(multi)})")

    sig_index: dict[tuple, list[dict]] = {}
    pool: list[dict] = []
    seen: set[str] = set()
    for _request, tool, _sid in base_pairs:
        key = gve._canon(tool)
        if key in seen:
            continue
        seen.add(key)
        pool.append(tool)
        sig = btc._signature(tool)
        if sig:
            sig_index.setdefault(sig, []).append(tool)
    print(f"tool pool: {len(pool)} distinct tools, "
          f"{len(sig_index)} signature groups")

    # ---- miners
    for request, tool, sid in base_pairs:
        mutated = None
        # sibling_swap: same required-parameter interface, other function
        gold_sig = btc._signature(tool)
        gold_name = gve._name(tool)
        sibling = gve._pick(sig_index.get(gold_sig, []),
                            lambda t: gve._name(t) != gold_name,
                            request + "|sibling")
        if sibling is not None and sink.add(request, sibling, 0,
                                            "distill_sibling_swap",
                                            sid + "|sib"):
            mutated = True
        # no_tool_needed: strip argument values from the request's own tool
        stripped = gve._strip_arguments(request)
        if (stripped is not None
                and not validate_fillability(tool, stripped)[0]
                and sink.add(stripped, tool, 0, "distill_no_tool_needed",
                             sid + "|strip")):
            mutated = True
        # infeasible: contract-violating rewrite of the request
        bad = infeasible_mutation(request, tool)
        if (bad is not None and validate_fillability(tool, bad)[0]
                and sink.add(bad, tool, 0, "distill_infeasible",
                             sid + "|inf")):
            mutated = True
        # the gold twin ships with any mutation (and only then, per spec:
        # the twin of every mutated row)
        if mutated:
            sink.add(request, tool, 1, "distill_positive_twin", sid)

    # ---- ask_followup from the unconsumed When2Call questions
    for request, tools, uuid in when2call_rows(used_w2c):
        for tool in tools[:2]:
            if sink.add(request, tool, 0, "distill_ask_followup", uuid):
                break  # one row per question keeps the pool wide

    counts = collections.Counter(r["source"] for r in sink.rows)
    print("candidates:", dict(counts))
    print("drops:", dict(sink.stats))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in sink.rows))
    META.write_text(json.dumps({
        "version": VERSION, "seed": SEED,
        "candidates": len(sink.rows), "per_source": dict(counts),
        "drops": dict(sink.stats), "caps": CAPS,
    }, indent=2) + "\n")
    print(f"wrote {OUT} ({len(sink.rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
