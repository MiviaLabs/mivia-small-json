"""Build the English Bitext intent-pair corpus for the intent-rank head.

Source: the five priority Bitext per-domain English intent sets
(CDLA-Sharing-1.0, raw dumps in data/incoming/external/bitext/):
retail-banking, telco, insurance, retail-ecommerce, customer-support.

Every row becomes a positive intent pair ({label: 1, request, source,
split, tool}) in the guard trainer's pairs schema; `tool` is the
description string inference scores, byte-identical to the zero-shot
select recipe "<intent> (<intent words>)".

Determinism: domains in fixed order, rows in file order, cap of 300
rows per (domain, intent) — first N, no sampling. Re-runs are
idempotent.

Hygiene (same standard the disjointness gate holds new corpora to):
- skip empty/whitespace instructions or intents;
- global dedupe by weak and strong-normalized request (first wins);
- exclude every request that collides (weak or strong norm) with an
  eval surface (Banking77 TEST, CLINC150 validation/test), a guard eval
  (guard-eval-v2, irrelevance_eval), the router's dev splits, the
  frozen sets (data/heldout, data/wild, data/synthetic), or a request
  already in the sibling pair corpora (train.jsonl, multilang.jsonl).

Writes data/real/intent-pairs/bitext-en.jsonl and prints the summary
block for PROVENANCE.md.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/incoming/external/bitext"
OUT = ROOT / "data/real/intent-pairs/bitext-en.jsonl"

DOMAINS = (
    # domain -> raw dump csv (mission priority order)
    ("retail-banking", "retail-banking.csv"),
    ("telco", "telco.csv"),
    ("insurance", "insurance.csv"),
    ("retail-ecommerce", "retail-ecommerce.csv"),
    ("customer-support", "customer-support.csv"),
)

CAP = 300
SEED = "bitext-en intent pairs (CDLA-Sharing-1.0)"  # provenance note only


# --- normalizers: byte-identical to tests/test_intent_pairs_disjoint.py ---
_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def _weak(text: str) -> str:
    return " ".join((text or "").lower().split())


def _strong(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(_PUNCT.sub(" ", t).split())


def _norms(text: str) -> tuple[str, str]:
    return _weak(text), _strong(text)


# --- hygiene surfaces ------------------------------------------------------
def _jsonl_fields(path: Path, fields=("raw", "text", "request")) -> set[str]:
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for f in fields:
            v = row.get(f)
            if isinstance(v, str) and v:
                out.add(v)
    return out


def load_excluded() -> tuple[set[str], set[str]]:
    """(weak-norms, strong-norms) of every surface the corpus must avoid."""
    weak: set[str] = set()
    strong: set[str] = set()

    def add(texts: set[str]) -> None:
        for t in texts:
            w, s = _norms(t)
            weak.add(w)
            strong.add(s)

    # eval surfaces (recomputed from raw, same files as the gate)
    b77 = ROOT / "data/incoming/external/banking77/test.csv"
    if b77.is_file():
        with b77.open(encoding="utf-8", errors="replace", newline="") as fh:
            add({r["text"] for r in csv.DictReader(fh) if r.get("text")})
    clinc = ROOT / "data/incoming/external/clinc150"
    for name in ("validation.jsonl", "test.jsonl"):
        add(_jsonl_fields(clinc / name, ("text",)))

    # guard canonical eval instruments
    for name in ("guard-eval-v2.jsonl", "irrelevance_eval.jsonl"):
        add(_jsonl_fields(ROOT / "data/incoming/external/bfcl" / name))

    # router in-domain dev splits (measurement surfaces)
    for p in sorted((ROOT / "data/real").glob("*/dev.jsonl")):
        add(_jsonl_fields(p))

    # frozen sets — whole-record exclusion, any single line
    for base in ("data/heldout", "data/wild", "data/synthetic"):
        for p in sorted((ROOT / base).rglob("*.jsonl")):
            if p.name in ("manifest.json",):
                continue
            add(_jsonl_fields(p))

    # sibling pair corpora: dedupe, don't double-count a request
    for name in ("train.jsonl", "multilang.jsonl"):
        add(_jsonl_fields(ROOT / "data/real/intent-pairs" / name, ("request",)))

    return weak, strong


# --- build -----------------------------------------------------------------
def main() -> None:
    excluded_weak, excluded_strong = load_excluded()
    seen_weak: set[str] = set()
    seen_strong: set[str] = set()
    per_pair: Counter = Counter()  # (domain, intent) -> emitted
    per_domain: Counter = Counter()
    intents_by_domain: dict[str, set[str]] = {}
    dropped = Counter()
    rows_out: list[dict] = []

    for domain, fname in DOMAINS:
        path = RAW / fname
        if not path.is_file():
            raise SystemExit(f"missing raw dump: {path} — fetch it first")
        intents_by_domain[domain] = set()
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                request = (row.get("instruction") or "").strip()
                intent = (row.get("intent") or "").strip()
                if not request or not intent:
                    dropped[f"{domain}:empty"] += 1
                    continue
                if per_pair[(domain, intent)] >= CAP:
                    dropped[f"{domain}:cap"] += 1
                    continue
                w, s = _norms(request)
                if w in excluded_weak or s in excluded_strong:
                    dropped[f"{domain}:excluded-surface"] += 1
                    continue
                if w in seen_weak or s in seen_strong:
                    dropped[f"{domain}:duplicate"] += 1
                    continue
                seen_weak.add(w)
                seen_strong.add(s)
                per_pair[(domain, intent)] += 1
                per_domain[domain] += 1
                intents_by_domain[domain].add(intent)
                rows_out.append(
                    {
                        "label": 1,
                        "request": request,
                        "source": f"bitext_{domain}_pos",
                        "split": "train",
                        "tool": f"{intent} ({intent.replace('_', ' ')})",
                    }
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()

    print(f"wrote {OUT} rows={len(rows_out)} sha256={digest}")
    print("per-domain:", dict(per_domain))
    print("distinct-intents:", {d: len(v) for d, v in intents_by_domain.items()})
    print("pairs-under-cap:", sum(1 for v in per_pair.values() if v < CAP))
    print("dropped:", dict(dropped))
    all_intents = sorted(i for v in intents_by_domain.values() for i in v)
    print(f"total distinct intents (cross-domain union): {len(all_intents)}")


if __name__ == "__main__":
    main()
