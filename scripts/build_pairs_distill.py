#!/usr/bin/env python3
"""Merge admitted distill rows with the gate-span base corpus.

Spec §3.2 (phase P1, ticket 05). Writes:

- ``pairs-distill.jsonl``  the gate-span base corpus (build_gate_corpus,
  reproducing the shipped pairs.jsonl counts with
  ``render_tool_fillability`` spans — the corpus the shipped gate trains on)
  followed by the admitted distill rows, rendered to the same gate span.
  The committed pairs.jsonl is never rewritten.
- ``pairs-distill-nofilter.jsonl``  the same base plus EVERY candidate under
  its construction label (teacher probability recorded, not gating) — the
  §7.7 no-filter ablation corpus. The base already carries the positives
  for those mutated candidates whose original requests are corpus-blocked,
  so the ablation corpus is positive-complete.
- meta JSON: per-source counts, the negative-mass cap check (new negatives
  must stay ≤ 50% of total negative mass, spec §3.1), corpus sha256s.

Key conventions (documented, intentionally distinct): base rows carry the
base builder's ``key`` (sha256 of request + v1 span, kept for
reproducibility with the shipped corpus lineage); distill rows carry
sha256(request + gate span). No current consumer joins across the two
families (the trainer's twin join needs ``twin_key``, which only the
falsified contrastive arm used).

Rows carry the raw tool dict in ``tool_dict`` (re-renderable provenance)
alongside the rendered span the trainer reads. Deterministic; offline;
refuses to write when the cap check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_tool_match_corpus as btc  # noqa: E402
from mivia_small_json.tool_match import render_tool_fillability  # noqa: E402

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
ADMITTED = BFCL / "distill-admitted.jsonl"
CANDIDATES = BFCL / "distill-candidates.jsonl"
OUT = BFCL / "pairs-distill.jsonl"
OUT_NOFILTER = BFCL / "pairs-distill-nofilter.jsonl"
META = BFCL / "pairs-distill-meta.json"
NEG_MASS_CAP = 0.50


def _key(request: str, span: str) -> str:
    return hashlib.sha256((request + "\x00" + span).encode()).hexdigest()


def _read_rows(path: Path, what: str) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing {path} ({what}); run the upstream phase "
                         "first (tickets 03/05)")
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    if not rows:
        print(f"WARNING: {path} is empty — the merged corpus will be "
              "base-only", file=sys.stderr)
    return rows


def distill_rows(records: list[dict], provenance: str,
                 admitted_only: bool) -> list[dict]:
    out = []
    for rec in records:
        if admitted_only and not rec.get("admitted"):
            continue
        request = rec["request"]
        span = render_tool_fillability(rec["tool"], request)
        out.append({
            "key": _key(request, span),
            "label": rec["label"],
            "request": request,
            "source": rec["source"],
            "split": rec["split"],
            "tool": span,
            "tool_dict": rec["tool"],
            "label_source": provenance,
            "teacher": rec.get("teacher"),
            "teacher_p": rec.get("teacher_p"),
            "admitted": bool(rec.get("admitted")),
            "seed": rec.get("seed"),
            "version": rec.get("version"),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="build in memory and print the meta; write "
                             "nothing")
    args = parser.parse_args()

    base_rows, base_stats = btc.build_gate_corpus(contrast=False)
    n_base_neg = sum(1 for r in base_rows if r["label"] == 0)

    admitted = _read_rows(ADMITTED, "labeling output")
    candidates = _read_rows(CANDIDATES, "mining output")

    merged = base_rows + distill_rows(admitted,
                                      "construction+teacher_agree",
                                      admitted_only=True)
    nofilter = base_rows + distill_rows(candidates, "construction",
                                        admitted_only=False)

    n_new_neg = sum(1 for r in merged[len(base_rows):] if r["label"] == 0)
    neg_mass = n_new_neg / (n_base_neg + n_new_neg)
    payloads = {
        OUT: "".join(json.dumps(r, sort_keys=True) + "\n" for r in merged),
        OUT_NOFILTER: "".join(json.dumps(r, sort_keys=True) + "\n"
                              for r in nofilter),
    }
    meta = {
        "base_rows": len(base_rows),
        "base_stats": base_stats,
        "admitted_rows": len(admitted),
        "candidates": len(candidates),
        "merged_rows": len(merged),
        "nofilter_rows": len(nofilter),
        "new_negatives": n_new_neg,
        "negative_mass": round(neg_mass, 4),
        "negative_mass_cap": NEG_MASS_CAP,
        "cap_ok": neg_mass <= NEG_MASS_CAP,
        "split_rule": ("base rows carry the base builder's 80/20 "
                       "content-hash split; distill rows btc._split_of "
                       "(same rule) — spec §3.2's '~85/15' is this"),
        "key_convention": ("base keys hash request+v1 span (shipped "
                           "lineage); distill keys hash request+gate span"),
        "per_source": {}, "per_source_nofilter": {},
        "sha256": {},
    }
    for path, payload in payloads.items():
        meta["sha256"][path.name] = hashlib.sha256(
            payload.encode()).hexdigest()
    for tag, rows in (("per_source", merged),
                      ("per_source_nofilter", nofilter)):
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        meta[tag] = dict(sorted(counts.items()))

    if not meta["cap_ok"]:
        print(json.dumps(meta, indent=2))
        print(f"negative-mass cap violated: {neg_mass:.3f} > "
              f"{NEG_MASS_CAP}; refusing to write", file=sys.stderr)
        return 1
    if args.check_only:
        print(json.dumps(meta, indent=2))
        return 0
    for path, payload in payloads.items():
        path.write_text(payload, encoding="utf-8")
        print(f"wrote {path} ({payload.count(chr(10))} rows)")
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in
                      ("merged_rows", "nofilter_rows", "new_negatives",
                       "negative_mass", "cap_ok")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
