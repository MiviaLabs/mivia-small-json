#!/usr/bin/env python3
"""Merge the blind adjudication-B answers into the committed evidence file,
compute Cohen's kappa(A, B) over the paired sample, and apply the REAL drop
path: any sampled row B coded as a mismatch or "none of these" (code Z) is
DROPPED from synthetic-labelsets.jsonl (sidecar + build meta refreshed).

Fails loudly when: answers do not cover the declared sample exactly, an
answer names a code not in that row's candidate list, or kappa < the
registered 0.6 gate (the audit gate recomputes kappa from the files and
would refuse the stamp anyway).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_zs_select_r1_corpus import norm  # noqa: E402

OUT = ROOT / "data/real/zero-shot-select-r1"
SCRATCH = ROOT / ".scratch/zs-select-r1"
KAPPA_MIN = 0.6


def _cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa over paired categorical codings (the audit gate's own
    implementation, mirrored here so the round reports the same number)."""
    n = len(a)
    if not n:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def main() -> int:
    a_rows = [json.loads(ln) for ln in
              (OUT / "adjudication-A.jsonl").read_text(
                  encoding="utf-8").splitlines() if ln.strip()]
    prompts = [json.loads(ln) for ln in
               (SCRATCH / "adjudication-B-prompts.jsonl").read_text(
                   encoding="utf-8").splitlines() if ln.strip()]
    answers = {}
    for line in (SCRATCH / "adjudication-B-answers.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            answers[r["id_suffix"]] = r["pick"]

    if len(answers) != len(prompts):
        print(f"COVERAGE FAIL: {len(answers)} answers for "
              f"{len(prompts)} prompts", file=sys.stderr)
        return 2

    by_suffix: dict[str, dict] = {}
    for p in prompts:
        suf = p["id"][-6:]
        if suf in by_suffix:
            print(f"DUPLICATE SUFFIX {suf}", file=sys.stderr)
            return 2
        by_suffix[suf] = p

    a_by_id = {r["id"]: r for r in a_rows}
    b_rows: list[dict] = []
    gold: list[str] = []
    blind: list[str] = []
    drop_ids: list[str] = []
    for p in prompts:
        suf = p["id"][-6:]
        pick = answers[suf]
        codes = [c["code"] for c in p["candidates"]]
        if pick != "Z" and pick not in codes:
            print(f"BAD CODE {p['id']} -> {pick}", file=sys.stderr)
            return 2
        a = a_by_id[p["id"]]
        # map the blind pick back to a label name (Z = none-of-these)
        pick_label = ("NONE_OF_THESE" if pick == "Z" else
                      next(c["desc"].split(":")[0].strip()
                           for c in p["candidates"] if c["code"] == pick))
        agree = pick_label == a["gold_label"]
        gold.append(a["gold_label"])
        blind.append(pick_label)
        if not agree:
            drop_ids.append(p["id"])
        b_rows.append({"id": p["id"], "domain": p["domain"],
                       "text": p["text"], "A_gold": a["gold_label"],
                       "B_pick": pick_label, "agree": agree})
    kappa = _cohen_kappa(gold, blind)
    print(json.dumps({
        "n": len(prompts), "agree": len(prompts) - len(drop_ids),
        "kappa": round(kappa, 4), "gate": KAPPA_MIN,
        "kappa_pass": kappa >= KAPPA_MIN, "drop_count": len(drop_ids),
        "disagreements": sorted(
            f"{r['A_gold']}|{r['B_pick']}" for r in b_rows if not r["agree"]),
    }, indent=1))
    if kappa < KAPPA_MIN:
        print("KAPPA FAIL", file=sys.stderr)
        return 2

    # real drop path: rewrite the S3 file minus B-rejected rows
    dropped_norms = set()
    a_dropped = {r["id"] for r in b_rows if not r["agree"]}
    s3 = [json.loads(ln) for ln in
          (OUT / "synthetic-labelsets.jsonl").read_text(
              encoding="utf-8").splitlines() if ln.strip()]
    a_by_key = {}
    for r in a_rows:
        a_by_key[(r["set_id"], norm(r["text"]))] = r["id"]

    def row_id(r: dict) -> str | None:
        if r["source"].endswith("_sibneg"):
            return a_by_key.get((r.get("set_id"), norm(r["request"])))
        return next((rr["id"] for rr in a_rows
                     if rr["set_id"] == r.get("set_id")
                     and norm(rr["text"]) == norm(r["request"])), None)

    kept, dropped = [], 0
    for r in s3:
        rid = row_id(r)
        if rid in a_dropped:
            dropped += 1
            dropped_norms.add(norm(r["request"]))
            continue
        kept.append(r)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    (OUT / "synthetic-labelsets.jsonl").write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    (OUT / "synthetic-labelsets.jsonl.sha256").write_text(digest + "\n",
                                                          encoding="utf-8")
    write_jsonl_b(OUT / "adjudication-B.jsonl", b_rows)

    meta_path = OUT / "build-meta.json"
    meta = json.loads(meta_path.read_text())
    meta["files"]["synthetic-labelsets.jsonl"]["rows"] = len(kept)
    meta["files"]["synthetic-labelsets.jsonl"]["sha256"] = digest
    meta["adjudication"] = {
        **meta.get("adjudication", {}),
        "kappa": round(kappa, 4), "B_rows": len(b_rows),
        "dropped_rows": dropped, "dropped_texts": len(dropped_norms),
        "finalized": True,
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"dropped {dropped} rows ({len(dropped_norms)} texts); "
          f"S3 now {len(kept)} rows, sha {digest[:12]}")
    return 0


def write_jsonl_b(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n",
                                                         encoding="utf-8")
    return digest


if __name__ == "__main__":
    raise SystemExit(main())
