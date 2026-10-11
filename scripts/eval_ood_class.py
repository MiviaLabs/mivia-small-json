#!/usr/bin/env python3
"""Phase 2 one-shot: evaluate a candidate router with the trained
text class (the non-workflow outcome).

    .venv/bin/python scripts/eval_ood_class.py <int8-candidate.onnx> \
        [--tokenizer <candidate-tokenizer.json>] \
        [--classes <candidate-classes.json>]

The tokenizer and the class map MUST be the candidate's own (an mm12-family
candidate uses the XLM-R 250k vocab — scoring it through the shipped MiniLM
tokenizer is silent garbage, never a fallback). Defaults resolve to the
candidate-dir siblings ``<stem>-tokenizer.json`` / ``<stem>-classes.json``
(the export script's naming contract).

Bars (ood-routing-band, phase 2 amendment):
  1. v2 OOD set minus ticket-non-en (frozen, 605 rows — the non-EN
     tickets are a RETENTION surface per the 2026-10-10 multilingual
     owner decision): confident-rejection >= 80% overall, free-chat
     >= 85%.
  2. Retention >= 99% per canonical surface — measured on EVAL-ONLY rows
     (dev splits / external eval files; never data/real/*/train.jsonl,
     which is in the training mix — that made retention partly trivial,
     found in the 2026-10-10 spec review).
  3. No regression on canonical routing: held-out (int8) / LogHub /
     tickets en+de / PRs / SWE-bench routed-accuracy vs the spec's pinned
     bands, with the shipped router measured on the same instrument for
     context. Guard bytes untouched (asserted via git status models/).

Loads the candidate directly (never the shipped bytes) and restores the
environment; purely read-only measurement. One-shot: run once per seed on
the frozen bars, record the verdict either way.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

V2 = ROOT / "data" / "incoming" / "ood" / "ood-set-v2.jsonl"

# Spec bar 2 surfaces -> (rows source, gold schema for bar 3). All EVAL-ONLY:
# dev splits and external caches, never the train.jsonl files the mix reads.
SURFACE_GOLD = {
    "heldout": None,      # gold carried per-row (data/heldout/test.jsonl)
    "loghub": "log_event",
    "tickets-en": "issue",
    "tickets-de": "issue",
    "prs": "pull_request",
    "swebench": "issue",
}
SURFACE_LIMITS = {"heldout": 350, "loghub": 2000, "tickets-en": 300,
                  "tickets-de": 300, "prs": 300, "swebench": 100}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def eval_surface_rows() -> list[dict]:
    """Bar-2/3 surfaces from eval-only sources (no training-mix rows)."""
    out: list[dict] = []

    def add_jsonl(surface: str, path: Path, n: int) -> None:
        taken = 0
        for line in path.read_text(encoding="utf-8", errors="replace") \
                .splitlines():
            if taken >= n or not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            text = r.get("raw") or r.get("text") or ""
            if not isinstance(text, str) or len(text.strip()) < 4:
                continue
            gold = r.get("schema") if surface == "heldout" \
                else SURFACE_GOLD[surface]
            out.append({"surface": surface, "text": text, "gold": gold})
            taken += 1

    add_jsonl("heldout", ROOT / "data/heldout/test.jsonl",
              SURFACE_LIMITS["heldout"])
    add_jsonl("loghub", ROOT / "data/real/log_event/dev.jsonl",
              SURFACE_LIMITS["loghub"])
    add_jsonl("tickets-en", ROOT / "data/real/issue/dev.jsonl",
              SURFACE_LIMITS["tickets-en"])
    add_jsonl("prs", ROOT / "data/real/pull_request/dev.jsonl",
              SURFACE_LIMITS["prs"])

    de_path = (ROOT / "data/incoming/external/incident-severity"
               / "tickets-multi-lang-20k.csv")
    if de_path.is_file():
        taken = 0
        with de_path.open(encoding="utf-8", errors="replace",
                          newline="") as fh:
            for row in csv.DictReader(fh):
                if taken >= SURFACE_LIMITS["tickets-de"]:
                    break
                if row.get("language") != "de":
                    continue
                text = (row.get("body") or "").strip()
                if len(text) >= 4:
                    out.append({"surface": "tickets-de", "text": text,
                                "gold": SURFACE_GOLD["tickets-de"]})
                    taken += 1

    sw_path = (ROOT / "data/incoming/external/swebench"
               / "swebench_lite_100.json")
    if sw_path.is_file():
        for entry in json.loads(sw_path.read_text())[:SURFACE_LIMITS["swebench"]]:
            text = (entry.get("problem_statement") or "").strip()
            if len(text) >= 4:
                out.append({"surface": "swebench", "text": text,
                            "gold": SURFACE_GOLD["swebench"]})
    return out


def _accuracy(router, rows: list[dict]) -> tuple[float, float]:
    """(retention, routed-accuracy) over eval rows through the pipeline."""
    from mivia_small_json import pipeline

    flagged = wrong = 0
    for r in rows:
        schema, _rb, _conf = pipeline._route_with_confidence(router,
                                                             r["text"])
        flagged += int(schema == "text")
        wrong += int(r["gold"] is not None and schema != r["gold"])
    n = len(rows)
    return (n - flagged) / n, (n - wrong) / n


# The REGISTERED bars (ood-routing-band): rejection
# >= 0.80, free-chat >= 0.85, retention >= 0.98 (owner re-registration —
# "Consequence: v9 PASSES all bars"). Never retune these against numbers.
def _bars(result: dict) -> dict:
    return {
        "rejection>=0.80": result["rejection_overall"] >= 0.80,
        "freechat>=0.85": (result["freechat_rejection"] or 0) >= 0.85,
        "retention>=0.98": result["min_retention"] >= 0.98,
    }


def _guard_untouched() -> bool:
    """Sha check of the shipped guard artifacts (not a git-state proxy:
    any router promotion dirties `git status models/` by construction)."""
    from mivia_small_json import integrity

    models = ROOT / "models"
    return all(integrity.verified(models / name) for name in (
        "tool-match-int8.onnx", "tool-match-tokenizer.json",
        "tool-match-gate-int8.onnx", "tool-match-gate-tokenizer.json",
    ))


def _sha256_of(path: Path) -> str | None:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() \
        else None


def _verdict(result: dict) -> str:
    """PASS needs every bar AND untouched guard bytes: recording the guard
    check without gating on it let `false` ship next to `PASS` once."""
    ok = (all(result["bars"].values())
          and result.get("guard_bytes_untouched") is True)
    return "PASS" if ok else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--classes", type=Path, default=None)
    args = ap.parse_args()

    cand = args.candidate
    if not cand.is_file():
        print(f"candidate not found: {cand}")
        return 2
    stem = cand.stem.replace("-int8", "")
    tok_path = args.tokenizer or cand.parent / f"{stem}-tokenizer.json"
    cls_path = args.classes or cand.parent / f"{stem}-classes.json"
    if not tok_path.is_file():
        print(f"candidate tokenizer not found: {tok_path} (pass --tokenizer; "
              "the shipped tokenizer is the wrong vocab for an mm12 "
              "candidate and is never a fallback)")
        return 2
    if not cls_path.is_file():
        print(f"candidate class map not found: {cls_path} (pass --classes)")
        return 2

    # The text emission branch in _route_with_confidence fires only when
    # the schema is registered; without this the measurement reads 0%.
    from mivia_small_json import bootstrap

    bootstrap.register_all_schemas()

    from mivia_small_json import encoder, pipeline

    router = encoder.load_encoder(
        path=cand,
        tokenizer_path=tok_path,
        classes=tuple(json.loads(cls_path.read_text(encoding="utf-8"))),
    )
    if router is None:
        print("candidate refused to load (integrity?)")
        return 2
    if "text" not in router.classes:
        print(f"candidate class map has no text: {router.classes}")
        return 2

    print("classes:", router.classes)

    ood = _rows(V2)
    # Multilingual owner decision (2026-10-10): tickets are in-domain in
    # ANY language — the v2 set's ticket-non-en rows flip from rejection
    # category to a RETENTION surface (pass = not classified text).
    non_en_tickets = [r for r in ood if r.get("category") == "ticket-non-en"]
    ood = [r for r in ood if r.get("category") != "ticket-non-en"]
    surf_rows = eval_surface_rows() + [
        {"surface": "tickets-non-en", "text": r["text"], "gold": None}
        for r in non_en_tickets]

    ood_hit = fc_hit = fc_n = 0
    ood_conf = 0
    for r in ood:
        schema, _rb, conf = pipeline._route_with_confidence(router,
                                                            r["text"])
        hit = schema == "text"
        ood_hit += hit
        if conf >= 0.85:
            ood_conf += 1
            fc_hit += int(hit and r.get("category") == "free-chat")
            fc_n += int(r.get("category") == "free-chat")

    retention: dict[str, float] = {}
    accuracy: dict[str, float] = {}
    for name in sorted({r["surface"] for r in surf_rows}):
        rows = [r for r in surf_rows if r["surface"] == name]
        retention[name], accuracy[name] = _accuracy(router, rows)

    # Bar 3 context: the shipped router on the same instrument, plus the
    # pinned bands from the spec amendment. Guard bytes must be untouched.
    shipped_ret = shipped_acc = None
    shipped = encoder.load_encoder()
    if shipped is not None:
        shipped_ret, shipped_acc = {}, {}
        for name in sorted({r["surface"] for r in surf_rows}):
            rows = [r for r in surf_rows if r["surface"] == name]
            shipped_ret[name], shipped_acc[name] = _accuracy(shipped, rows)

    result = {
        "candidate": cand.name,
        # GOVERNING RULE: instruments name the bytes they scored. After a
        # promotion copy lands in models/, "candidate == shipped" is
        # trivially true — the 2026-10-10 adjudication needed these two
        # hashes to prove a real candidate regression had been mislabeled
        # as instrument drift (both columns scored the same new bytes).
        "candidate_sha256": _sha256_of(cand),
        "shipped_sha256": _sha256_of(ROOT / "models/router-encoder-int8.onnx"),
        "tokenizer": tok_path.name,
        "n_ood": len(ood),
        "rejection_overall": round(ood_hit / len(ood), 4),
        "confident_ood": ood_conf,
        "freechat_rejection": round(fc_hit / fc_n, 4) if fc_n else None,
        "retention": {k: round(v, 4) for k, v in retention.items()},
        "min_retention": round(min(retention.values()), 4),
        "routed_accuracy": {k: round(v, 4) for k, v in accuracy.items()},
        "shipped_same_instrument": {
            "retention": ({k: round(v, 4) for k, v in shipped_ret.items()}
                          if shipped_ret else None),
            "routed_accuracy": ({k: round(v, 4)
                                 for k, v in shipped_acc.items()}
                                if shipped_acc else None),
        },
        "guard_bytes_untouched": _guard_untouched(),
    }
    result["bars"] = _bars(result)
    result["verdict"] = _verdict(result)
    print(json.dumps(result, indent=1))
    out = ROOT / "reports" / f"ood-class-oneshot-{cand.name}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"written {out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
