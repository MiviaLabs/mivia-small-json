#!/usr/bin/env python3
"""zero-shot-select r3 corpus builder (zero-shot-select-r3,
Source R `near-oos-pairs.jsonl` — synthesized near-domain oos questions).

Two phases (the approved teacher pattern with blind sample re-check):

  default        expand the authored nuclei (scripts/zs_select_r3_neardo.py)
                 through the 10 registered r1 frames + slot lexicons,
                 mechanically admit (dedupe, length, SUPERSET eval-window
                 exclusion, 0 overlap vs the r1 null oos supply and the
                 referee), emit 4 pairs per text (2 null renderings label 1,
                 2 lexical-hard confusables label 0), write
                 .scratch/zs-select-r3/near-oos-staging.jsonl +
                 adjudication-A.jsonl (full coverage) + the blind B prompts
                 (240 rows: 160 near-oos + 80 r1-train impostors).
  --finalize     merge the blind B answers, recompute Cohen's kappa(A,B)
                 (>= 0.6 gate), apply the REAL drop path (sampled near-oos
                 rows B codes as covered-by-an-intent are dropped with all
                 their pairs), write data/real/zero-shot-select-r3/
                 near-oos-pairs.jsonl + committed evidence + build meta +
                 idempotent PROVENANCE section.

Deterministic (fixed seeds). Fails loudly on any declared-band, dedupe
(<= 1%), supply-shortfall, kappa or leak breach — never shrinks silently.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_zs_select_r1_corpus import (  # noqa: E402
    build_exclusion, dedupe_rate, lexical_siblings, norm, split_of, write_jsonl,
)
from zs_select_r1_domains import FRAMES, SLOTS  # noqa: E402
from zs_select_r3_neardo import NUCLEI_B77, NUCLEI_CLINC  # noqa: E402

OUT = ROOT / "data/real/zero-shot-select-r3"
R1 = ROOT / "data/real/zero-shot-select-r1"
B77 = ROOT / "data/incoming/external/banking77"
CLINC = ROOT / "data/incoming/external/clinc150"
REFEREE = ROOT / "data/eval/binary-select-v1.jsonl"
SCRATCH = ROOT / ".scratch/zs-select-r3"

SEED = 42
B_SEED = 424242
TARGET_TEXTS = 3_800                 # in the registered band [3_000, 4_200]
B_SAMPLE_NEARDO_PER_DOMAIN = 80
B_SAMPLE_IMPOSTOR_PER_DOMAIN = 40
KAPPA_MIN = 0.6
DEDUPE_MAX = 0.01
BANDS = {"texts": (3_000, 4_200), "pairs": (10_000, 18_000)}
DOMAIN_SHARE = (0.40, 0.60)
NULL_KEYS = ["__null__", "none", "no_match"]
NULL_PHRASINGS = ["none of these descriptions match",
                  "the text is about none of these topics"]
NULL_COMBOS = [(k, p) for k in NULL_KEYS for p in NULL_PHRASINGS]  # 6


def render_desc(intent: str) -> str:
    return f"{intent}: {intent.replace('_', ' ')} ({intent.replace('_', ' ')})"


def domain_intents() -> dict[str, list[str]]:
    b77 = json.loads((B77 / "categories.json").read_text())
    clinc = [line for line in json.loads(
        (CLINC / "labels.json").read_text()) if line != "oos"]
    return {"banking": sorted(b77), "clinc": sorted(clinc)}


def r1_null_oos_norms() -> set[str]:
    out = set()
    for line in (R1 / "null-pairs.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["source"] == "null_pos_oos":
                out.add(norm(r["request"]))
    return out


def referee_norms() -> set[str]:
    return {norm(json.loads(line)["text"])
            for line in REFEREE.read_text().splitlines() if line.strip()}


def expand_texts() -> list[dict]:
    """Nuclei x frames x slots -> candidate texts (pre-admission)."""
    rng = random.Random(SEED)
    texts: list[dict] = []
    for domain, nuclei in (("banking", NUCLEI_B77),
                           ("clinc", NUCLEI_CLINC)):
        for ni, (nucleus, area) in enumerate(nuclei):
            markers = re.findall(r"\{(\w+)\}", nucleus)
            for fi, frame in enumerate(FRAMES):
                vals = {}
                for m in markers:
                    if m not in SLOTS:
                        raise SystemExit(f"unknown slot {m} in {nucleus!r}")
                    vals[m] = rng.choice(SLOTS[m])
                base = nucleus
                for m, v in vals.items():
                    base = base.replace(f"{{{m}}}", v)
                text = frame.format(n=base)
                text = re.sub(r"\s+", " ", text).strip()
                texts.append({"text": text, "domain": domain, "area": area,
                              "nucleus": ni})
    return texts


def admit(texts: list[dict]) -> list[dict]:
    excl, _rec = build_exclusion()
    excl |= referee_norms()
    excl |= r1_null_oos_norms()
    seen: set[str] = set()
    out: list[dict] = []
    stats = {"raw": len(texts), "excluded": 0, "dupes": 0, "len": 0}
    for t in texts:
        n = norm(t["text"])
        if not (8 <= len(t["text"]) <= 140):
            stats["len"] += 1
            continue
        if n in excl:
            stats["excluded"] += 1
            continue
        if n in seen:
            stats["dupes"] += 1
            continue
        seen.add(n)
        out.append(t)
    return out, stats


def build() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    intents = domain_intents()
    texts, stats = admit(expand_texts())
    rng = random.Random(SEED + 1)
    if len(texts) > TARGET_TEXTS:
        texts = rng.sample(texts, TARGET_TEXTS)
    share_b = sum(1 for t in texts if t["domain"] == "banking") / len(texts)
    if not (DOMAIN_SHARE[0] <= share_b <= DOMAIN_SHARE[1]):
        print(f"DOMAIN SHARE FAIL: banking share {share_b:.2f} outside "
              f"{DOMAIN_SHARE}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    admit_meta: list[dict] = []
    for i, t in enumerate(texts):
        rng_i = random.Random(SEED + 100_000 + i)   # per-text determinism
        combos = rng_i.sample(NULL_COMBOS, 2)
        dom_intents = intents[t["domain"]]
        conf = [it for _ov, it in lexical_siblings(t["text"], dom_intents)[:2]]
        sp = split_of(t["text"])
        tid = f"neardo-{i:05d}"
        for k, p in combos:
            rows.append({"label": 1, "request": t["text"],
                         "tool": f"{k}: {p}",
                         "source": f"neardo_{t['domain']}_pos", "split": sp})
        for it in conf:
            rows.append({"label": 0, "request": t["text"],
                         "tool": render_desc(it),
                         "source": f"neardo_{t['domain']}_conf", "split": sp})
        admit_meta.append({"id": tid, "text": t["text"],
                           "domain": t["domain"], "area": t["area"],
                           "gold": "Z", "confusables": conf})

    rate = dedupe_rate(rows)
    if rate > DEDUPE_MAX:
        print(f"DEDUPE FAIL: {rate:.4f} > {DEDUPE_MAX}", file=sys.stderr)
        return 2

    # -- blind B sample: 160 near-oos (area-stratified) + 80 impostors ------
    rng_b = random.Random(B_SEED)
    impostors: list[dict] = []
    pool_by_dom = {"banking": [], "clinc": []}
    for line in (R1 / "listwise-pairs.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["source"] == "banking77_pos" and r["split"] == "train":
                pool_by_dom["banking"].append((r["request"], r["tool"]))
            elif r["source"] == "clinc150_pos" and r["split"] == "train":
                pool_by_dom["clinc"].append((r["request"], r["tool"]))
    for domain in ("banking", "clinc"):
        for text, tool in rng_b.sample(
                pool_by_dom[domain], B_SAMPLE_IMPOSTOR_PER_DOMAIN):
            gold_name = tool.split(":", 1)[0].strip()
            if gold_name not in intents[domain]:
                print(f"IMPOSTOR GOLD FAIL: {gold_name}", file=sys.stderr)
                return 2
            impostors.append({"text": text, "domain": domain,
                              "gold": gold_name, "r1_row_tool": tool})

    prompts: list[dict] = []
    a_extra: list[dict] = []
    sampled: list[dict] = []
    near_by_dom = {"banking": [m for m in admit_meta
                               if m["domain"] == "banking"],
                   "clinc": [m for m in admit_meta if m["domain"] == "clinc"]}
    for domain in ("banking", "clinc"):
        by_area: dict[str, list[dict]] = {}
        for m in near_by_dom[domain]:
            by_area.setdefault(m["area"], []).append(m)
        per_area = B_SAMPLE_NEARDO_PER_DOMAIN // max(1, len(by_area))
        sample: list[dict] = []
        for area in sorted(by_area):
            sample += rng_b.sample(by_area[area],
                                   min(per_area, len(by_area[area])))
        # round-robin the remainder so the sample hits exactly 80
        areas = sorted(by_area)
        while len(sample) < B_SAMPLE_NEARDO_PER_DOMAIN:
            for area in areas:
                if len(sample) >= B_SAMPLE_NEARDO_PER_DOMAIN:
                    break
                rest = [m for m in by_area[area] if m not in sample]
                if rest:
                    sample.append(rng_b.choice(rest))
        sample = sample[:B_SAMPLE_NEARDO_PER_DOMAIN]
        sampled += sample
        for m in sample:
            m["sampled_for_B"] = True
        for m in admit_meta:
            m.setdefault("sampled_for_B", False)
        for m in sample:
            items = [(f"{ci:02d}", render_desc(it))
                     for ci, it in enumerate(intents[domain])]
            rng_row = random.Random(B_SEED + int(m["id"].split("-")[1]))
            rng_row.shuffle(items)
            prompts.append({
                "id": m["id"], "domain": domain,
                "task": ("Below is one user message and a numbered list of "
                         "the assistant's registered capabilities. Decide "
                         "which single capability covers the message, or Z "
                         "if none does. Answer with the code only."),
                "text": m["text"],
                "candidates": [f"{c}. {d}" for c, d in items] + [
                    "Z. none of these descriptions match"],
            })
    for imp in impostors:
        domain = imp["domain"]
        items = [(f"{ci:02d}", render_desc(it))
                 for ci, it in enumerate(intents[domain])]
        rng_row = random.Random(B_SEED + 7_000_000 + len(prompts))
        rng_row.shuffle(items)
        pid = f"impostor-{len(prompts):05d}"
        prompts.append({
            "id": pid, "domain": domain,
            "task": prompts[0]["task"],
            "text": imp["text"],
            "candidates": [f"{c}. {d}" for c, d in items] + [
                "Z. none of these descriptions match"],
        })
        a_extra.append({"id": pid, "text": imp["text"],
                        "domain": domain, "area": "r1_train_impostor",
                        "gold": imp["gold"], "r1_row_tool": imp["r1_row_tool"],
                        "sampled_for_B": True})
    for m in admit_meta:
        m.setdefault("sampled_for_B", False)

    write_jsonl(SCRATCH / "near-oos-staging.jsonl", rows)
    (SCRATCH / "admit-meta.json").write_text(
        json.dumps({"admitted": admit_meta, "admission_stats": stats,
                    "banking_share": round(share_b, 4),
                    "kappa_min": KAPPA_MIN,
                    "b_sample": {"neardo_per_domain":
                                 B_SAMPLE_NEARDO_PER_DOMAIN,
                                 "impostor_per_domain":
                                 B_SAMPLE_IMPOSTOR_PER_DOMAIN}},
                   indent=1), encoding="utf-8")
    with (SCRATCH / "adjudication-A-extra.jsonl").open(
            "w", encoding="utf-8") as fh:
        for m in a_extra:
            fh.write(json.dumps(m) + "\n")
    with (SCRATCH / "adjudication-B-prompts.jsonl").open(
            "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")

    print(f"admitted texts: {len(texts)} (banking share {share_b:.2f}); "
          f"pairs: {len(rows)}; dedupe {rate:.4f}")
    print(f"admission stats: {stats}")
    print(f"B prompts: {len(prompts)} "
          f"({sum(1 for p in prompts if not p['id'].startswith('impostor'))} "
          f"near-oos + {sum(1 for p in prompts if p['id'].startswith('impostor'))} "
          "impostors)")
    print("NEXT: rate every row in .scratch/zs-select-r3/"
          "adjudication-B-prompts.jsonl BLIND (gold hidden), commit the "
          "answers to .scratch/zs-select-r3/adjudication-B-answers.jsonl, "
          "then run this script --finalize")
    return 0


def finalize() -> int:
    admit_doc = json.loads((SCRATCH / "admit-meta.json").read_text())
    admit_meta = admit_doc["admitted"]
    prompts = [json.loads(ln) for ln in
               (SCRATCH / "adjudication-B-prompts.jsonl").read_text(
                   encoding="utf-8").splitlines() if ln.strip()]
    answers: dict[str, str] = {}
    for line in (SCRATCH / "adjudication-B-answers.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            answers[r["id_suffix"]] = r["pick"]
    if len(answers) != len(prompts) or set(answers) != {
            p["id"] for p in prompts}:
        print(f"COVERAGE FAIL: {len(answers)} answers for "
              f"{len(prompts)} prompts (ids must match exactly)",
              file=sys.stderr)
        return 2
    by_id = {p["id"]: p for p in prompts}
    for pid, pick in answers.items():
        cands = [c.split(".", 1)[0] for c in by_id[pid]["candidates"]]
        if pick not in cands:
            print(f"ANSWER FAIL: {pid} pick {pick} not a candidate code",
                  file=sys.stderr)
            return 2

    gold = {m["id"]: m["gold"] for m in admit_meta}
    a_extra_path = SCRATCH / "adjudication-A-extra.jsonl"
    extra = a_extra_rows(a_extra_path)
    gold.update({m["id"]: m["gold"] for m in extra})
    id2meta = {m["id"]: m for m in admit_meta}
    id2meta.update({m["id"]: m for m in extra})

    paired = []
    for pid in sorted(answers):
        g = gold[pid]
        if g != "Z":
            # resolve the impostor gold intent NAME into this row's shuffled
            # CODE space (mechanical; B judged and answered in code space)
            hits = [c.split(".", 1)[0] for c in by_id[pid]["candidates"]
                    if c.split(".", 1)[0] != "Z" and
                    c.split(". ", 1)[1].split(":")[0].strip() == g]
            assert len(hits) == 1, (pid, g, hits)
            g = hits[0]
        paired.append((g, answers[pid]))
    a, b = [x for x, _ in paired], [y for _, y in paired]
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    kappa = 1.0 if (po == 1.0 and pe == 1.0) else \
        (po - pe) / (1 - pe) if pe < 1.0 else 0.0
    drop_ids = {pid for pid in answers
                if not pid.startswith("impostor") and answers[pid] != "Z"}
    print(f"blind B: {n} rows, agreement {po:.4f}, kappa {kappa:.4f} "
          f"(gate {KAPPA_MIN}); near-oos rows B codes as covered: "
          f"{len(drop_ids)} (drop path)")

    # -- apply the drop path to the staged corpus ----------------------------
    drop_norms: set[str] = set()
    for did in drop_ids:
        m = next(x for x in admit_meta if x["id"] == did)
        drop_norms.add(norm(m["text"]))
    staged = [json.loads(ln) for ln in
              (SCRATCH / "near-oos-staging.jsonl").read_text().splitlines()
              if ln.strip()]
    rows = [r for r in staged if norm(r["request"]) not in drop_norms]
    kept_texts = {norm(r["request"]) for r in rows}
    n_texts = len(kept_texts)
    share_b = sum(1 for m in admit_meta
                  if m["domain"] == "banking"
                  and norm(m["text"]) in kept_texts) / max(1, n_texts)

    lo, hi = BANDS["texts"]
    plo, phi = BANDS["pairs"]
    if not (lo <= n_texts <= hi and plo <= len(rows) <= phi):
        print(f"BAND FAIL after drop path: {n_texts} texts / {len(rows)} "
              f"pairs vs bands {BANDS}", file=sys.stderr)
        return 2
    if not (DOMAIN_SHARE[0] <= share_b <= DOMAIN_SHARE[1]):
        print(f"DOMAIN SHARE FAIL after drop path: {share_b:.2f}",
              file=sys.stderr)
        return 2
    if kappa < KAPPA_MIN:
        print(f"KAPPA FAIL: {kappa:.4f} < {KAPPA_MIN} (fail-closed; no "
              "corpus is written)", file=sys.stderr)
        return 2
    rate = dedupe_rate(rows)
    if rate > DEDUPE_MAX:
        print(f"DEDUPE FAIL: {rate:.4f}", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    digest = write_jsonl(OUT / "near-oos-pairs.jsonl", rows)
    a_rows = [{"id": m["id"], "text": m["text"], "domain": m["domain"],
               "area": m["area"], "gold": m["gold"],
               "sampled_for_B": m.get("sampled_for_B", False)}
              for m in admit_meta] + extra
    write_jsonl(OUT / "adjudication-A.jsonl", a_rows)
    b_rows = []
    for pid in sorted(answers):
        m = id2meta[pid]
        pick = answers[pid]
        g = m["gold"]
        if g != "Z":
            hits = [c.split(".", 1)[0] for c in by_id[pid]["candidates"]
                    if c.split(".", 1)[0] != "Z" and
                    c.split(". ", 1)[1].split(":")[0].strip() == g]
            assert len(hits) == 1, (pid, g, hits)
            g_code = hits[0]
        else:
            g_code = "Z"
        b_rows.append({"id": pid, "text": m["text"], "domain": m["domain"],
                       "gold": m["gold"], "gold_code": g_code,
                       "pick": pick})
    write_jsonl(OUT / "adjudication-B.jsonl", b_rows)

    meta = {
        "arm": "zero-shot-select-r3",
        "seed": SEED, "b_seed": B_SEED,
        "nuclei": {"banking": len(NUCLEI_B77), "clinc": len(NUCLEI_CLINC)},
        "admission": admit_doc["admission_stats"],
        "target_texts": TARGET_TEXTS,
        "final": {"texts": n_texts, "pairs": len(rows),
                  "banking_share": round(share_b, 4), "dedupe": rate,
                  "sha256": digest},
        "adjudication": {"rows": n, "agreement": round(po, 4),
                         "kappa": round(kappa, 4), "gate": KAPPA_MIN,
                         "kappa_pass": kappa >= KAPPA_MIN,
                         "dropped_texts": len(drop_ids),
                         "impostors": B_SAMPLE_IMPOSTOR_PER_DOMAIN * 2},
        "bands": BANDS,
    }
    (OUT / "build-meta.json").write_text(json.dumps(meta, indent=1),
                                         encoding="utf-8")

    section = f"""<!-- section:zero-shot-select-r3 -->
## zero-shot-select-r3 (near-oos-pairs.jsonl + adjudication evidence)

- near-oos-pairs.jsonl — SYNTHESIZED in-topic/out-of-intent questions
  (original authored content, disclosed as such; guideline in
  scripts/zs_select_r3_neardo.py, committed before the build): nuclei x the
  10 registered r1 surface frames x slot lexicons (fictional values),
  mechanically admitted (strong-norm dedupe, length 8-140, SUPERSET
  eval-window exclusion, 0 overlap vs the r1 null oos supply and the
  referee's 500 scored texts). 4 pairs per admitted text: 2 null renderings
  (label 1, keys __null__/none/no_match x 2 phrasings, seeded rotation) +
  2 lexical-hard confusable in-domain descriptions (label 0, the r1
  B77-OOS move). No real user data; no eval-surface text.
- adjudication-A.jsonl — rater A gold, FULL coverage of admitted texts
  (gold Z = no registered intent covers it) + the {B_SAMPLE_IMPOSTOR_PER_DOMAIN * 2}
  evidence-only impostor rows (r1 listwise TRAIN-split texts with their
  true intent gold; they never enter the r3 corpus).
- adjudication-B.jsonl — the blind re-check (rater B answers on the
  240-row sample: {B_SAMPLE_NEARDO_PER_DOMAIN * 2} near-oos + {B_SAMPLE_IMPOSTOR_PER_DOMAIN * 2}
  impostors; full shuffled registered candidate list per row, gold hidden).
  Agreement {po:.4f}, kappa {kappa:.4f} (gate {KAPPA_MIN}), drop path applied to
  {len(drop_ids)} sampled texts B coded as covered-by-an-intent.
- same-model double-pass disclosure: A and B are the same session LLM
  (r1's registered precedent); the impostor rows exist because a
  constant-gold kappa is degenerate — registered in the spec BEFORE the
  build.
Rules: no fabricated values; deterministic build (seeds above); idempotent
section (replaced on rebuild, rule 12). Counts + sha256 in build-meta.json
and the *.sha256 sidecars.
"""
    prov = OUT / "PROVENANCE.md"
    text = (prov.read_text() if prov.exists()
            else "# data/real/zero-shot-select-r3 — provenance\n")
    marker = "<!-- section:zero-shot-select-r3 -->"
    if marker in text:
        text = text[:text.index(marker)].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    prov.write_text(text, encoding="utf-8")

    print(json.dumps({"texts": n_texts, "pairs": len(rows),
                      "kappa": round(kappa, 4), "dropped": len(drop_ids),
                      "sha256": digest[:12]}, indent=1))
    return 0


def a_extra_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    raise SystemExit(finalize() if args.finalize else build())
