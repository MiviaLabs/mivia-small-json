#!/usr/bin/env python3
"""Build and measure the categorized tool-match guard evaluation set (v2).

The v1 guard evaluation is one number on one construction: 884 BFCL
``live_irrelevance`` rows and 183 real calls. Every ``live_irrelevance`` row is
a *single-turn near-miss* (BFCL builds it by stripping an argument the tool
needs or by swapping in a sibling tool), so the set cannot say *which* failure
mode a change moves, and it carries no session channel, no multi-turn turn, no
infeasible request and no synthetic near-miss. This script writes a richer set
where every row carries an explicit ``category``, then measures the shipped
guard per category so the residual false-call mass can be located.

Categories and sources
----------------------
``single_turn_irrelevance``  BFCL ``live_irrelevance`` (real, single-turn near-miss)
``single_turn_relevance``    BFCL ``calls.jsonl`` (real calls; the retention control)
``multi_turn_miss_func``     BFCL ``multi_turn_miss_func`` (real; the turn's needed
                             function is excluded, so the whole turn abstains)
``multi_turn_miss_param``    BFCL ``multi_turn_miss_param`` (real; the function is
                             listed but a required argument is missing)
``ask_followup``             When2Call ``request_for_info`` + ``cannot_answer``
                             (CC BY 4.0; the honest answer is to ask, not call)
``infeasible``               ACEBench ``Special`` irrelevant/incomplete/error_param
                             (MIT; no listed tool can serve the request)
``foreign_tool``             SYNTHETIC: a held-out request paired with a real tool
                             from another catalogue (different name prefix)
``sibling_swap``             SYNTHETIC: a held-out request paired with a real tool
                             that shares its gold tool's required-parameter signature
``no_tool_needed``           SYNTHETIC: a held-out request with its numeric argument
                             values stripped, paired with its own tool. Only rows
                             whose strip actually breaks fillability are kept.

The three synthetic categories are the guard's own suggestion: deterministic
(seeded by content hash), derived from real catalogues, and built only from
requests that are already held out of the training corpus (the committed
``calls.jsonl`` requests). Nothing here is trained on.

Row shape
---------
``{"request": str, "tool": <raw tool dict>, "label": int, "category": str,
"source": str, "split": "eval"}``. ``tool`` is the full tool definition, not a
pre-rendered span, so the shipped gate's fillability span
(``tool_match.render_tool_fillability``) can be rendered from the file alone;
``tool_match.render_tool`` recovers the shipped v1 span for dedup against the
committed pair sets. ``source`` is ``<tag>#<catalogue id>`` so the guard's
per-catalogue decision (``max`` over the listed tools) is reconstructed exactly
from the per-pair rows. Single-pair well-defined: no ``(request, tool)`` pair
carries two labels (a pair that would is dropped and counted).

Disjointness
------------
Every row's request is verified to be absent from the training corpus
(``pairs.jsonl`` requests) and the newly added rows are deduped against
``pairs.jsonl`` and every committed eval set. The two baseline categories
(``single_turn_irrelevance``, ``single_turn_relevance``) *are* the v1 eval sets,
so they are exempt from the eval-set dedup by design.

Subcommands
-----------
``build``    write ``guard-eval-v2.jsonl`` + refresh its provenance entry
``measure``  redirect to the canonical ``bench_tool_abstention.py --eval v2``

Deterministic, offline after the first fetch, stdlib + existing deps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json.tool_match import (  # noqa: E402
    parse_catalogue_raw,
    render_tool,
    validate_fillability,
)

import build_tool_match_corpus as btc  # noqa: E402
import bench_tool_abstention as _bench  # noqa: E402  canonical v2 measurement

BFCL = btc.BFCL
EXT = btc.EXT
OUT = BFCL / "guard-eval-v2.jsonl"
PROVENANCE = EXT / "HARDER_SETS_PROVENANCE.md"

# The category order is the report's row order; it is not a training label.
CATEGORIES = (
    "single_turn_irrelevance",
    "single_turn_relevance",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "ask_followup",
    "infeasible",
    "foreign_tool",
    "sibling_swap",
    "no_tool_needed",
)
BASELINE = frozenset({"single_turn_irrelevance", "single_turn_relevance"})
# Deterministic per-category caps: the raw sources are larger than the set
# needs, and a stable content-hash subsample keeps runs reproducible.
CAPS = {
    "single_turn_irrelevance": 0,   # all 884
    "single_turn_relevance": 0,     # all 183
    "multi_turn_miss_func": 600,
    "multi_turn_miss_param": 600,
    "ask_followup": 600,
    "infeasible": 0,                # all 403
    "foreign_tool": 300,
    "sibling_swap": 300,
    "no_tool_needed": 300,
}
_ROLE = re.compile(r"^(?:user|assistant|system)\s*:\s*", re.IGNORECASE)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_WS = re.compile(r"\s+")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canon(tool: dict) -> str:
    return json.dumps(tool, sort_keys=True, ensure_ascii=False)


def _name(tool: dict) -> str:
    return str(tool.get("name", ""))


def _v1(tool: dict) -> str:
    """The shipped v1 span, for dedup against the committed pair sets."""
    return btc._clean(render_tool(tool))


def _row(request: str, tool: dict, label: int, category: str,
         source: str, cid: str) -> dict:
    """One eval row. ``source`` carries the catalogue id after ``#``.

    The catalogue id lets the measurement reconstruct the guard's per-catalogue
    decision (the shipped decision is ``max`` over the listed tools) exactly
    from the per-pair rows, even when BFCL reuses a request across catalogues.
    """
    return {"request": btc._clean(request), "tool": tool, "label": label,
            "category": category, "source": f"{source}#{cid}",
            "split": "eval"}


# ---------------------------------------------------------------------------
# Real sources
# ---------------------------------------------------------------------------
def live_irrelevance_rows() -> list[dict]:
    """BFCL ``live_irrelevance``: 884 real single-turn near-miss catalogues."""
    out = []
    for rec in btc._fp_eval_rows():
        request = rec["question"][0][0]["content"]
        for tool in rec.get("function") or []:
            out.append(_row(request, tool, 0, "single_turn_irrelevance",
                            "bfcl_live_irrelevance", rec["id"]))
    return out


def calls_rows() -> list[dict]:
    """BFCL ``calls.jsonl``: the 183 real calls (retention control)."""
    out = []
    for rec in _rows(BFCL / "calls.jsonl"):
        tools, request = parse_catalogue_raw(rec.get("text") or "")
        cid = _digest(rec.get("text") or "")[:12]
        for tool in tools or []:
            out.append(_row(request, tool, 1, "single_turn_relevance",
                            "bfcl_calls", cid))
    return out


def _turn_golds(rec: dict, gt: list, index: int, classes: list[str],
                names: set[str]) -> set[str]:
    """The gold tool names for one turn, class-qualified."""
    golds: set[str] = set()
    if index < len(gt):
        for call in gt[index]:
            match = btc.MT_CALL.search(call)
            if not match:
                continue
            for cls in classes:
                qualified = f"{cls}.{match.group(1)}"
                if qualified in names:
                    golds.add(qualified)
    return golds


def multiturn_rows(split: str, category: str) -> list[dict]:
    """All-negative turns of one BFCL multi-turn split, as raw (request, tool).

    Replicates ``build_tool_match_corpus.multiturn_rows``' turn/gold walk but
    keeps the raw tool dict (the eval must be scorable by the shipped gate from
    the file alone) and keeps only turns whose ``ground_truth`` is empty -- the
    turns that must abstain. ``multi_turn_miss_func`` excludes the needed
    function; ``multi_turn_miss_param`` lists it but strips a required argument.
    Only one split is used per category, so no ``(request, tool)`` pair carries
    two labels.
    """
    questions = btc._fetch(f"{btc.BASE}/BFCL_v4_{split}.json",
                           Path(f"/tmp/bfcl_v4_{split}.json"))
    answers = btc._mt_answers(split)
    cats = btc._mt_catalogue()
    out: list[dict] = []
    for rec in questions:
        classes = [c for c in (rec.get("involved_classes") or []) if c in cats]
        tools = [t for c in classes for t in cats[c]]
        if not tools:
            continue
        names = {_name(t) for t in tools}
        gt = answers.get(rec["id"], [])
        index = 0
        for turn in rec.get("question") or []:
            if not turn:
                continue
            request = btc._clean(turn[0].get("content", ""))
            golds = _turn_golds(rec, gt, index, classes, names)
            index += 1
            if not request:
                continue
            if golds:
                continue  # a gold turn; only the abstaining turns are eval rows
            for tool in tools:
                out.append(_row(request, tool, 0, category, f"bfcl_{split}",
                                f"{rec['id']}#t{index - 1}"))
    return out


def when2call_rows() -> list[dict]:
    """When2Call MCQ ``request_for_info`` + ``cannot_answer`` (ask, not call)."""
    path = btc.WHEN2CALL_DIR / "when2call_test_mcq.jsonl"
    if not path.is_file():
        print("skipping when2call: raw file missing", file=sys.stderr)
        return []
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        answer = rec.get("correct_answer")
        if answer not in ("request_for_info", "cannot_answer"):
            continue
        request = btc._clean(rec.get("question") or "")
        if not request:
            continue
        for tool in btc._load_tool_list(rec.get("tools")):
            out.append(_row(request, tool, 0, "ask_followup",
                            f"when2call_{answer}", str(rec.get("uuid") or "")))
    return out


def acebench_rows() -> list[dict]:
    """ACEBench ``Special``: irrelevant / incomplete / error_param requests."""
    kinds = (("data_special_irrelevant.json", "irrelevant"),
             ("data_special_incomplete.json", "incomplete"),
             ("data_special_error_param.json", "error_param"))
    out = []
    for name, kind in kinds:
        path = EXT / "acebench" / name
        if not path.is_file():
            print(f"skipping {name}: raw file missing", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            request = btc._clean(_ROLE.sub("", rec.get("question") or ""))
            if not request:
                continue
            for tool in rec.get("function") or []:
                out.append(_row(request, tool, 0, "infeasible",
                                f"acebench_special_{kind}",
                                str(rec.get("id") or "")))
    return out


# ---------------------------------------------------------------------------
# Synthetic near-misses from real catalogues
# ---------------------------------------------------------------------------
def _tool_pool() -> list[dict]:
    """Every distinct tool in the BFCL catalogues, content-hash ordered."""
    questions, _answers = btc._sources()
    seen: set[str] = set()
    pool: list[dict] = []
    for rec in questions:
        for tool in rec.get("function") or []:
            key = _canon(tool)
            if key not in seen:
                seen.add(key)
                pool.append(tool)
    return sorted(pool, key=lambda t: _digest(_canon(t)))


def _pick(candidates: list[dict], predicate, key: str) -> dict | None:
    """The first candidate (in the given order) satisfying ``predicate``.

    The candidate list is content-hash ordered, and the start offset is derived
    from ``key`` so one request does not always land on the same tool.
    """
    if not candidates:
        return None
    offset = int(_digest(key), 16) % len(candidates)
    for step in range(len(candidates)):
        tool = candidates[(offset + step) % len(candidates)]
        if predicate(tool):
            return tool
    return None


def _strip_arguments(request: str) -> str | None:
    """Remove the argument values a tool would fill from the request text.

    The BFCL near-miss construction strips the argument the tool needs. The
    synthetic version removes the numeric argument tokens and collapses the
    whitespace, so the request is topically the same but no longer fills the
    listed tool. ``None`` when the request carries no argument to strip.
    """
    stripped = _WS.sub(" ", _NUM.sub(" ", request)).strip()
    return stripped if stripped and stripped != request else None


def synthetic_rows(pool: list[dict], base: list[tuple[str, dict]]
                   ) -> list[dict]:
    """foreign_tool / sibling_swap / no_tool_needed from real catalogues.

    ``base`` is the held-out ``(request, gold tool)`` pairs of the committed
    ``calls.jsonl``; every synthetic row reuses one of those requests, so the
    set stays disjoint from the training corpus. Tools come from the real BFCL
    pool.
    """
    by_sig: dict[tuple, list[dict]] = {}
    for tool in pool:
        sig = btc._signature(tool)
        if sig:
            by_sig.setdefault(sig, []).append(tool)
    out: list[dict] = []
    for request, gold in base:
        cid = _digest(request)[:12]
        gold_name = _name(gold)
        gold_sig = btc._signature(gold)
        prefix = gold_name.split(".")[0]
        # sibling_swap: same required-parameter interface, different function.
        if gold_sig:
            sibling = _pick(
                by_sig.get(gold_sig, []),
                lambda t: _name(t) != gold_name, request + "|sibling")
            if sibling is not None:
                out.append(_row(request, sibling, 0, "sibling_swap",
                                "synthetic_sibling_swap", cid))
        # foreign_tool: a real tool from another name family.
        foreign = _pick(
            pool,
            lambda t: (_name(t).split(".")[0] != prefix
                       and _name(t) != gold_name
                       and btc._signature(t) != gold_sig),
            request + "|foreign")
        if foreign is not None:
            out.append(_row(request, foreign, 0, "foreign_tool",
                            "synthetic_foreign_tool", cid))
        # no_tool_needed: the request's argument values are stripped from its
        # own tool. Only rows whose strip actually breaks fillability are kept,
        # so the should-abstain label is justified (a still-fillable pair would
        # be a real call, not a near-miss).
        stripped = _strip_arguments(request)
        if stripped is not None and not validate_fillability(gold, stripped)[0]:
            out.append(_row(stripped, gold, 0, "no_tool_needed",
                            "synthetic_param_strip", cid))
    return out


def _calls_base() -> list[tuple[str, dict]]:
    """Held-out ``(request, tool)`` pairs from the committed ``calls.jsonl``."""
    out: list[tuple[str, dict]] = []
    for rec in _rows(BFCL / "calls.jsonl"):
        tools, request = parse_catalogue_raw(rec.get("text") or "")
        for tool in tools or []:
            out.append((btc._clean(request), tool))
    return out


# ---------------------------------------------------------------------------
# Assembly: dedup, label-conflict, disjointness
# ---------------------------------------------------------------------------
def _pairs_index(path: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """(request, v1 span) pairs and the request set of a committed pair file."""
    pairs: set[tuple[str, str]] = set()
    requests: set[str] = set()
    if not path.is_file():
        return pairs, requests
    for rec in _rows(path):
        request = btc._clean(rec.get("request", ""))
        tool = rec.get("tool")
        if isinstance(tool, dict):
            span = _v1(tool)
        else:
            span = btc._clean(str(tool or ""))
        pairs.add((request, span))
        requests.add(request)
    return pairs, requests


def _catalogue_index(path: Path) -> set[tuple[str, str]]:
    """(request, raw-tool canonical) pairs of a committed catalogue file."""
    out: set[tuple[str, str]] = set()
    if not path.is_file():
        return out
    for rec in _rows(path):
        tools, request = parse_catalogue_raw(rec.get("text") or "")
        for tool in tools or []:
            out.add((btc._clean(request), _canon(tool)))
    return out


def build(caps: dict[str, int] | None = None) -> tuple[list[dict], dict]:
    caps = caps or CAPS
    stats: dict[str, int] = {c: 0 for c in CATEGORIES}
    for name in ("duplicate_rows_kept", "dropped_label_conflict",
                 "dropped_eval_overlap", "dropped_training_request",
                 "dropped_cap", "dropped_empty"):
        stats[name] = 0

    # Dedup / disjointness targets.
    pairs_pairs, pairs_requests = _pairs_index(btc.OUT)
    eval_pairs: set[tuple[str, str]] = set()
    for name in ("calls.jsonl", "irrelevance_eval.jsonl", "irrelevance.jsonl"):
        eval_pairs |= _catalogue_index(BFCL / name)
    for name in ("pairs-ace.jsonl", "pairs-nearmiss.jsonl"):
        pp, _rq = _pairs_index(BFCL / name)
        eval_pairs |= pp

    labels: dict[tuple[str, str], int] = {}
    rows: list[dict] = []
    per_cat: dict[str, int] = {c: 0 for c in CATEGORIES}

    def add(row: dict) -> None:
        request = btc._clean(row["request"])
        tool = row["tool"]
        if not request or not isinstance(tool, dict) or not _name(tool):
            stats["dropped_empty"] += 1
            return
        key = (request, _canon(tool))
        category = row["category"]
        # Zero request overlap with the training corpus is a hard requirement.
        if request in pairs_requests:
            stats["dropped_training_request"] += 1
            return
        # The single-pair contract: no (request, tool) pair may carry two
        # labels. A source that would create one is excluded (and counted).
        # Same-label repeats (one pair in several catalogues) are kept; they
        # are the same pair with the same answer.
        if key in labels and labels[key] != row["label"]:
            stats["dropped_label_conflict"] += 1
            return
        # New rows must not duplicate anything already in an eval set. The two
        # baseline categories *are* the v1 eval sets, so they are exempt.
        if category not in BASELINE:
            if (request, _canon(tool)) in eval_pairs:
                stats["dropped_eval_overlap"] += 1
                return
            if (request, _v1(tool)) in pairs_pairs:
                stats["dropped_eval_overlap"] += 1
                return
        cap = caps.get(category, 0)
        if cap and per_cat[category] >= cap:
            stats["dropped_cap"] += 1
            return
        if key in labels:
            stats["duplicate_rows_kept"] += 1
        labels.setdefault(key, row["label"])
        per_cat[category] += 1
        stats[category] += 1
        rows.append({"request": request, "tool": tool, "label": row["label"],
                     "category": category, "source": row["source"],
                     "split": "eval"})

    def add_capped(source_rows: list[dict]) -> None:
        # Content-hash order makes the cap a stable subsample.
        ordered = sorted(source_rows,
                         key=lambda r: _digest(r["request"] + "\x00"
                                               + _canon(r["tool"])))
        for row in ordered:
            add(row)

    try:
        add_capped(live_irrelevance_rows())
    except Exception as exc:  # network / shape drift
        print(f"skipping live_irrelevance: {type(exc).__name__}: {exc}",
              file=sys.stderr)
    try:
        add_capped(calls_rows())
    except Exception as exc:
        print(f"skipping calls: {type(exc).__name__}: {exc}", file=sys.stderr)
    for split, category in (("multi_turn_miss_func", "multi_turn_miss_func"),
                            ("multi_turn_miss_param", "multi_turn_miss_param")):
        try:
            add_capped(multiturn_rows(split, category))
        except Exception as exc:
            print(f"skipping {split}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    for loader in (when2call_rows, acebench_rows):
        try:
            add_capped(loader())
        except Exception as exc:
            print(f"skipping {loader.__name__}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    try:
        add_capped(synthetic_rows(_tool_pool(), _calls_base()))
    except Exception as exc:
        print(f"skipping synthetic: {type(exc).__name__}: {exc}",
              file=sys.stderr)
    return rows, stats


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
_START = "<!-- guard-eval-v2:start -->"
_END = "<!-- guard-eval-v2:end -->"


def _provenance_block(rows: list[dict], digest: str) -> str:
    counts = {c: sum(1 for r in rows if r["category"] == c) for c in CATEGORIES}
    counts = {c: n for c, n in counts.items() if n}
    lines = [
        _START,
        "",
        "## bfcl/guard-eval-v2.jsonl — categorized guard evaluation set",
        "",
        f"Built by `scripts/build_guard_eval.py` (subcommand `build`). "
        f"{len(rows)} rows, sha256 `{digest[:16]}…` (full "
        f"`{digest}`). Rows are `{{request, tool, label, category, source, "
        f"split}}`; `tool` is the full tool definition so the shipped gate's "
        f"fillability span renders from the file alone.",
        "",
        "| category | label | rows | construction |",
        "|---|---|---|---|",
    ]
    construction = {
        "single_turn_irrelevance":
            "BFCL `live_irrelevance` near-miss (Apache-2.0)",
        "single_turn_relevance":
            "BFCL `calls.jsonl` real call (retention control)",
        "multi_turn_miss_func":
            "BFCL `multi_turn_miss_func` all-negative turns (Apache-2.0)",
        "multi_turn_miss_param":
            "BFCL `multi_turn_miss_param` all-negative turns (Apache-2.0)",
        "ask_followup":
            "When2Call MCQ `request_for_info` + `cannot_answer` (CC BY 4.0)",
        "infeasible":
            "ACEBench `Special` irrelevant/incomplete/error_param (MIT)",
        "foreign_tool":
            "synthetic: held-out request + real tool from another family",
        "sibling_swap":
            "synthetic: held-out request + same-signature sibling tool",
        "no_tool_needed":
            "synthetic: held-out request with argument values stripped",
    }
    label = {"single_turn_relevance": 1}
    for category in CATEGORIES:
        if category in counts:
            lines.append(f"| {category} | {label.get(category, 0)} | "
                         f"{counts[category]} | {construction[category]} |")
    lines += [
        "",
        "Deterministic (seeded by content hash). Every row's request is "
        "verified absent from the training corpus (`bfcl/pairs.jsonl`); the "
        "newly added rows are deduped against `pairs.jsonl` and every "
        "committed eval set. No `(request, tool)` pair carries two labels. "
        "The two baseline categories are the v1 eval sets and are exempt "
        "from the eval-set dedup.",
        "",
        _END,
        "",
    ]
    return "\n".join(lines)


def write_provenance(rows: list[dict], digest: str) -> None:
    block = _provenance_block(rows, digest)
    if PROVENANCE.is_file():
        text = PROVENANCE.read_text(encoding="utf-8")
        if _START in text and _END in text:
            head = text[:text.index(_START)]
            tail = text[text.index(_END) + len(_END):].lstrip("\n")
            text = head + block + tail
        else:
            text = text.rstrip("\n") + "\n\n" + block
    else:
        text = "# Harder external eval sets\n\n" + block
    PROVENANCE.write_text(text, encoding="utf-8")


def cmd_build(args) -> int:
    rows, stats = build()
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                      for r in rows)
    OUT.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    write_provenance(rows, digest)
    for name, value in stats.items():
        print(f"{name}: {value}")
    print(f"total rows: {len(rows)}  labels: "
          f"{sorted({r['label'] for r in rows})}")
    print(f"sha256: {digest}")
    print(f"written to {OUT.relative_to(ROOT)}")
    print(f"provenance: {PROVENANCE.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# Measurement: the canonical path is bench_tool_abstention --eval v2.
# The names below are backwards-compatible aliases (tests/test_guard_eval_v2.py
# pins _operating_point and _catalogues); measure/cmd_measure redirect to the
# single canonical implementation.
# ---------------------------------------------------------------------------
_operating_point = _bench._operating_point
_rates = _bench._rates
_catalogues = _bench._catalogue_scores


# Redirect to the canonical v2 measurement (bench --eval v2).
measure = _bench.measure_v2


def cmd_measure(args) -> int:
    print("canonical guard-eval-v2 measurement is "
          "scripts/bench_tool_abstention.py --eval v2; redirecting ...",
          file=sys.stderr)
    if not OUT.is_file():
        print(f"missing {OUT}; run build first", file=sys.stderr)
        return 1
    return _bench.cmd_measure_v2(args.json, getattr(args, "limit", 0))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="write the categorized eval set")
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("measure", help="score the shipped guard per category")
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(func=cmd_measure)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
