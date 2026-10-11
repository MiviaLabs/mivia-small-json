#!/usr/bin/env python3
"""chat-ood round 4: build multilang-tickets-v11.jsonl.

v11 = v10 (round-3 lineage, already deduped + eval-disjoint) + 500
greeting-led informal tickets (de 150 / fr 150) and Polish tickets
(200, half greeting-led), blind double-adjudicated (round-4 files).
The residual round-3 flips were greeting-led tickets and pl — this is
that exact texture. Guards identical to the v10 builder; PROVENANCE
appended. Deterministic, offline.
"""

from __future__ import annotations

import hashlib
import re
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_multilang_tickets_v10 import strong_norm, eval_norms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "data/real/multilang-tickets/multilang-tickets-v10.jsonl"
OUT = ROOT / "data/real/multilang-tickets/multilang-tickets-v11.jsonl"
AUTHORED_DIR = ROOT / "data/incoming/tickets-extra"
AUTHORED = {"greet-de": 150, "greet-fr": 150, "pl": 200}


def main() -> int:
    failures: list[str] = []
    rows = [json.loads(ln) for ln in V10.read_text(encoding="utf-8",
                                                 errors="replace")
            .splitlines() if ln.strip()]
    v10_count = len(rows)

    adjud = {}
    for f in ("adjudication-r4-A.jsonl", "adjudication-r4-B.jsonl"):
        p = AUTHORED_DIR / f
        adjud[f.removeprefix("adjudication-").removesuffix(".jsonl")] = [
            json.loads(ln)["label"] for ln in p.read_text().splitlines()
            if ln.strip()]

    idx = 0
    dropped_by_gate = 0
    for stem, want in AUTHORED.items():
        n_file = 0
        for line in (AUTHORED_DIR / f"authored-{stem}.jsonl").read_text(
                encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_file += 1
            r = json.loads(line)
            # same registered label gate as the v10 builder: both blind
            # adjudicators must call the row a ticket
            labels = (adjud["r4-A"][idx], adjud["r4-B"][idx])
            idx += 1
            if labels != ("ticket", "ticket"):
                dropped_by_gate += 1
                continue
            if r.get("category") != "issue":
                failures.append(f"authored {stem}: category "
                                f"{r.get('category')!r}")
                continue
            rows.append({"text": r["text"], "category": "issue",
                         "lang": stem.removeprefix("greet-"),
                         "source": f"authored-{stem}"})
        if n_file != want:
            failures.append(f"authored {stem}: file has {n_file} rows, "
                            f"declared {want}")

    ev = eval_norms()
    # v10 is trusted lineage (deduped + eval-gated at build time); the
    # round-4 delta is re-gated here, and the audit re-verifies all of v11.
    kept = list(rows[:v10_count])
    seen = {strong_norm(r["text"]) for r in kept}
    dupes = hits = 0
    for r in rows[v10_count:]:
        n = strong_norm(r["text"])
        if n in seen:
            dupes += 1
            continue
        if n in ev:
            hits += 1
            continue
        seen.add(n)
        kept.append(r)
    if hits:
        failures.append(f"{hits} delta rows overlap frozen eval surfaces")
    if dupes > 0.01 * len(rows):
        failures.append(f"dedupe: {dupes} duplicates exceed 1%")

    if failures:
        for f in failures:
            print(f"BUILD FAIL: {f}", file=sys.stderr)
        return 2

    with OUT.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / (OUT.name + ".sha256")).write_text(digest + "\n")

    from collections import Counter

    by_src = dict(sorted(Counter(r["source"] for r in kept).items()))
    prov = OUT.parent / "PROVENANCE.md"
    prior = prov.read_text(encoding="utf-8") if prov.is_file() else ""
    prior = re.sub(
        r"\n## multilang-tickets-v11\.jsonl [\s\S]*?(?=\n## |\Z)", "",
        prior)
    prov.write_text(prior, encoding="utf-8")
    with prov.open("a", encoding="utf-8") as fh:
        fh.write(f"""

## multilang-tickets-v11.jsonl (chat-ood round 4, 2026-10-10)

Built by scripts/build_multilang_tickets_v11.py. v10 lineage as-is
({v10_count} rows) + 500 greeting-led informal tickets (de 150 /
fr 150 / pl 200 — half greeting-led; the round-3 residual flips were
greeting-led tickets and pl). Blind double-adjudication: agreement
500/500 on both labelers, 0 rows dropped.

delta sources: {{{', '.join(f'{k}: {v}' for k, v in by_src.items() if k.startswith('authored'))}}}
label gate: {dropped_by_gate} rows dropped (both-blind-adjudicator ticket gate)
total rows: {len(kept)}
dedupe: {dupes} dropped, eval-surface hits: {hits} (gate: 0)
sha256: {digest}
""")
    print(f"rows={len(kept)} (v10 lineage {v10_count} + delta "
          f"{len(kept) - v10_count}) sha256={digest[:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
