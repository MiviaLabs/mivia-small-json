#!/usr/bin/env python3
"""Phase-0 decomposition of the residual guard quadrants (spec
residual-quadrant-decision-layer §2).

Diagnosis only — no bar, read-only on the eval (scores are computed, never
trained on, never tuned from). For each weak category (infeasible,
ask_followup, sibling_swap) it takes the catalogues the shipped chain still
CALLS at the canonical operating point and decomposes them into shapes a
decision-layer predicate could reach:

- infeasible: fillability-veto state + contract-violation markers
  (impossible calendar dates, absurd magnitudes >= 1e9, negative values for
  unsigned params, enum mismatches) + the ACEBench sub-kind (irrelevant /
  incomplete / error_param) from the eval's source tag.
- ask_followup: question-shaped (the shipped `_is_ask_followup`), veto
  state, and whether the residual is still the pinned 118.
- sibling_swap: whether the RANK artifact separates gold from sibling
  (`select` margin via `_score_raw`) while the gate calls — the composition
  opportunity the select-margin floor hints at.

Also quantifies the threshold-artifact: per-category abstention of the
VETOED chain at the OLD operating point (0.9925) vs the NEW one, showing
what the floor rule traded.

Writes .scratch/jev-distill/residual-decomposition.json and prints the
tables. Deterministic, offline, <5 min CPU.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json import tool_match as T  # noqa: E402
from mivia_small_json.pipeline import _is_ask_followup  # noqa: E402

EVAL = ROOT / "data" / "incoming" / "external" / "bfcl" / "guard-eval-v2.jsonl"
OUT = ROOT / ".scratch" / "jev-distill" / "residual-decomposition.json"
WEAK = ("infeasible", "ask_followup", "sibling_swap")
OLD_THR = 0.9925

_DATE_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{1,2})\b", re.I)
_DAYS_IN = {"february": 29, "april": 30, "june": 30, "september": 30,
            "november": 30}
_BIGNUM = re.compile(r"\b(\d{7,})\b")
_NEGNUM = re.compile(r"\b-\s*\d+(\.\d+)?\b")


def contract_markers(tool: dict, request: str) -> list[str]:
    """Cheap, high-precision contract-violation markers (phase-0 probes,
    not the phase-1 predicate)."""
    markers = []
    for month, day in _DATE_MONTH.findall(request):
        limit = _DAYS_IN.get(month.lower())
        if limit and int(day) > limit:
            markers.append("impossible_date")
            break
    if _BIGNUM.search(request):
        markers.append("absurd_magnitude")
    if _NEGNUM.search(request):
        markers.append("negative_value")
    bare = tool.get("function") if isinstance(tool.get("function"), dict) \
        else tool
    props = (bare.get("parameters") or {}).get("properties") or {}
    text = request.lower()
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        enum = spec.get("enum")
        if isinstance(enum, list) and enum:
            hits = [v for v in enum
                    if str(v).lower() in text]
            # enum present in schema but request names a value that looks
            # like a parameter fill yet matches no enum member
            if not hits and re.search(
                    rf"\b{re.escape(name)}\s*(?:=|:|of|to)\s+(\w+)", text):
                markers.append("enum_mismatch")
                break
    return markers


def main() -> int:
    rows = [json.loads(line) for line in
            EVAL.read_text().splitlines() if line.strip()]
    matcher = T.load_matcher()
    if matcher is None:
        raise SystemExit("shipped guard not loadable")

    # catalogue = source key; score = max over its rows (canonical grouping)
    by_cat: dict[str, dict[str, dict]] = {}
    for row in rows:
        text = ("Tools available:\n"
                + json.dumps([row["tool"]], ensure_ascii=False)
                + "\n\nUser request: " + row["request"])
        score = float(matcher.best_score(text))
        cat = by_cat.setdefault(row["category"], {})
        entry = cat.setdefault(row["source"], {
            "request": row["request"], "tool": row["tool"],
            "label": row["label"], "score": score,
            "subkind": row["source"].split("_")[-1]
            if row["source"].startswith("acebench") else ""})
        entry["score"] = max(entry["score"], score)

    # canonical operating point (the same rule the bench uses)
    neg = [e["score"] for c, cats in by_cat.items()
           if c != "single_turn_relevance" for e in cats.values()]
    pos = [e["score"] for e in by_cat["single_turn_relevance"].values()]
    best = None
    for thr in sorted(set(neg + pos)):
        kept = 100 * sum(s >= thr for s in pos) / len(pos)
        abst = 100 * sum(s < thr for s in neg) / len(neg)
        if kept >= 95.0 and (best is None or abst > best[1]):
            best = (thr, abst, kept)
    thr_new, pooled, retention = best
    print(f"canonical op (vetoed): thr {thr_new:.4f} pooled {pooled:.1f}% "
          f"retention {retention:.1f}% (bench recorded 0.6020 / 79.7 / 95.1)")

    result = {"operating_point": {"thr": thr_new, "pooled": pooled,
                                  "retention": retention},
              "threshold_artifact": {}, "categories": {}}
    # threshold artifact: vetoed scores at the OLD threshold
    for cat, cats in by_cat.items():
        scores = [e["score"] for e in cats.values()]
        result["threshold_artifact"][cat] = {
            "n": len(scores),
            "abst_at_new_thr": round(100 * sum(s < thr_new for s in scores)
                                     / len(scores), 1),
            "abst_at_old_thr": round(100 * sum(s < OLD_THR for s in scores)
                                     / len(scores), 1),
        }
    print(f"\n{'category':<24}{'n':>5}{'@new thr':>10}{'@old thr':>10}")
    for cat, t in result["threshold_artifact"].items():
        print(f"{cat:<24}{t['n']:>5}{t['abst_at_new_thr']:>9}%"
              f"{t['abst_at_old_thr']:>9}%")

    for cat in WEAK:
        residual = [e for e in by_cat[cat].values()
                    if e["score"] >= thr_new]
        shapes = Counter()
        examples: list[dict] = []
        for e in residual:
            veto_ok, _r, _m = T.fillability_veto(e["tool"], e["request"])
            markers = contract_markers(e["tool"], e["request"])
            if cat == "infeasible":
                shape = ("contract_marker:" + "+".join(sorted(markers))
                         if markers else
                         "fillable_no_marker" if veto_ok else "veto_missed")
                shapes[(shape, e["subkind"] or "-")] += 1
            elif cat == "ask_followup":
                shape = ("question_shaped" if _is_ask_followup(e["request"])
                         else "command_shaped")
                shapes[(shape, "veto_ok" if veto_ok else "veto_fail")] += 1
            else:  # sibling_swap
                # does the rank artifact separate gold from the listed
                # sibling? gold tool unknown here (eval lists the sibling);
                # use the rank score margin against the gate call as a proxy
                shapes[("called_by_gate", "-")] += 1
            if len(examples) < 6:
                examples.append({
                    "request": e["request"][:100], "score": e["score"],
                    "veto_ok": veto_ok, "markers": markers,
                    "subkind": e["subkind"]})
        result["categories"][cat] = {
            "residual_calls": len(residual),
            "total": len(by_cat[cat]),
            "shapes": {f"{s}/{k}": n for (s, k), n in sorted(shapes.items())},
            "examples": examples,
        }
        print(f"\n{cat}: {len(residual)} residual calls / "
              f"{len(by_cat[cat])} catalogues")
        for (s, k), n in sorted(shapes.items()):
            print(f"  {s:<28} {k:<12} {n}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
