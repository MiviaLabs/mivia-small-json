#!/usr/bin/env python3
"""zero-shot-select r1 corpus builder (zero-shot-select-r1).

Four sources, one corpus root (data/real/zero-shot-select-r1/):

  S1 listwise-pairs.jsonl      Banking77 train + CLINC150 train, every
                               admitted text -> 1 positive + sibling
                               negatives (lexical-hard + random; +2 on a
                               seeded 10% subset).
  S2 null-pairs.jsonl          text-vs-null: oos texts score the null
                               candidate label 1 (abstention LEARNED),
                               in-domain texts score it label 0, plus each
                               oos text vs its most-confusable real
                               description label 0.
  S3 synthetic-labelsets.jsonl 144 trained label sets over 12 authored
                               domains (scripts/zs_select_r1_domains.py),
                               1 positive + 2 same-set sibling negatives
                               per utterance + bare yes/no polarity rows.
  S4 bfcl-train.jsonl          data/incoming/external/bfcl/pairs.jsonl
                               train split VERBATIM (dev split excluded:
                               it feeds the select-calibrator instruments).

Eval-window exclusion (protocol rule 3, SUPERSET form): every training text
is checked against the strong-normalized union of the FULL official
banking77 test split, the FULL CLINC150 validation+test splits, and the
banking77-oos ood-oos valid window — so the consumed s42 window, the seeds
43-46 candidate ledgers, and every future fresh-reserve draw are disjoint
from training by construction. The s42/s43-46 windows are additionally
REPLICATED here and their sizes recorded (the audit gate re-verifies
independently).

Deterministic (fixed seeds). Fails loudly (exit 2) on any declared-band,
dedupe (<=1%) or pool-exhaustion breach — it never shrinks silently.
Writes sha256 sidecars, a build meta, the idempotent PROVENANCE section,
adjudication-A.jsonl (rater A = intended gold, full S3 coverage) and the
blind adjudication-B prompt file (shuffled, gold-hidden; answers are
produced separately and merged by
scripts/finalize_zs_select_adjudication.py).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from zs_select_r1_domains import (  # noqa: E402
    DOMAINS, FRAMES, RESERVE_DOMAINS, SLOTS, TRAINED_DOMAINS,
)

OUT = ROOT / "data/real/zero-shot-select-r1"
B77 = ROOT / "data/incoming/external/banking77"
CLINC = ROOT / "data/incoming/external/clinc150"
B77_OOS = ROOT / "data/incoming/external/banking77-oos"
BFCL_PAIRS = ROOT / "data/incoming/external/bfcl/pairs.jsonl"
SCRATCH = ROOT / ".scratch/zs-select-r1"

SEED = 42
SETS_PER_DOMAIN = 12
RESERVE_SETS_PER_DOMAIN = 1
UTTS_PER_LABEL_LO, UTTS_PER_LABEL_HI = 10, 16
NEG_SYNTH = 2                    # same-set sibling negatives per utterance
OOS_TEXTS_TARGET = 2500          # banking77-oos train sample for S2
INDOMAIN_NULL_NEG = 4000         # S2 (c): in-domain texts vs null -> label 0
ADJ_B_SAMPLE = 204               # blind re-check sample (17 per trained domain)
DEDUPE_MAX = 0.01

# Declared bands (spec; the audit gate enforces these against the bytes).
BANDS = {
    "listwise-pairs.jsonl": (30_000, 60_000),
    "null-pairs.jsonl": (10_000, 15_000),
    "synthetic-labelsets.jsonl": (35_000, 60_000),
    "bfcl-train.jsonl": (6_700, 6_720),
}

_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def norm(t: str) -> str:
    """Strong normalization: NFKC + casefold + punctuation-strip + collapse."""
    t = unicodedata.normalize("NFKC", t).casefold()
    return " ".join(_PUNCT.split(t)).strip()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def desc77(intent: str) -> str:
    """Serve-time rendered description (bench shape behind the key prefix)."""
    return f"{intent}: {intent.replace('_', ' ')} ({intent.replace('_', ' ')})"


def split_of(text: str) -> str:
    """Hash-stable 90/10 text split (the intent-pairs lineage convention)."""
    h = int(hashlib.sha256(norm(text).encode()).hexdigest(), 16)
    return "dev" if h % 10 == 0 else "train"


def lexical_siblings(text: str, intents: list[str]) -> list[tuple[str, int]]:
    """Siblings ranked by content-token overlap(text, intent words), desc.

    The diagnostic's miss structure is sibling clusters; lexical overlap of
    the intent's own words with the text is the deterministic hardness
    proxy (seeded random tie-break downstream)."""
    toks = set(norm(text).split())
    scored = []
    for it in intents:
        words = [w for w in it.replace("_", " ").split() if len(w) > 2]
        scored.append((sum(1 for w in words if w in toks), it))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored


def build_exclusion() -> tuple[set[str], dict]:
    """Superset exclusion norms + the replicated consumed-window record."""
    excl: set[str] = set()
    rec: dict = {}

    rows = []
    with (B77 / "test.csv").open(encoding="utf-8", errors="replace",
                                 newline="") as fh:
        for r in csv.DictReader(fh):
            text = (r.get("text") or "").strip()
            label = (r.get("category") or r.get("label") or "").strip()
            if text and label:
                rows.append((text, label))
    excl |= {norm(t) for t, _ in rows}
    rec["banking77_test_split"] = len(rows)
    # replicate the consumed windows (bench sampling logic) for the record
    consumed = {}
    for seed in (42, 43, 44, 45, 46):
        s = random.Random(seed).sample(rows, min(600, len(rows)))
        consumed[f"banking77_s{seed}"] = len(s)
        excl |= {norm(t) for t, _ in s}
    rec["banking77_consumed_windows"] = consumed

    for split in ("validation", "test"):
        crows = [json.loads(line) for line in
                 (CLINC / f"{split}.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
        excl |= {norm(r["text"]) for r in crows}
        rec[f"clinc150_{split}_split"] = len(crows)
        if split == "test":
            cons = {}
            for seed in (42, 44, 45, 46):
                s = random.Random(seed).sample(crows, min(600, len(crows)))
                cons[f"clinc150_test_s{seed}"] = len(s)
                excl |= {norm(r["text"]) for r in s}
            rec["clinc150_test_consumed_windows"] = cons
    vrows = [json.loads(line) for line in
             (CLINC / "validation.jsonl").read_text(
                 encoding="utf-8").splitlines() if line.strip()]
    s42 = random.Random(42).sample(vrows, min(600, len(vrows)))
    rec["clinc150_validation_s42"] = len(s42)
    excl |= {norm(r["text"]) for r in s42}

    ood_valid = [t.strip() for t in
                 (B77_OOS / "ood-oos__valid__seq.in").read_text(
                     encoding="utf-8").splitlines() if t.strip()]
    excl |= {norm(t) for t in ood_valid}
    rec["banking77_oos_valid_window"] = len(ood_valid)
    # the S2 null EVAL windows: ood-oos valid + the id-oos valid split stay
    # out of training (the id-oos TRAIN split is the null-positive supply)
    id_valid = [t.strip() for t in
                (B77_OOS / "id-oos__valid__seq.in").read_text(
                    encoding="utf-8").splitlines() if t.strip()]
    excl |= {norm(t) for t in id_valid}
    rec["banking77_oos_idvalid_window"] = len(id_valid)
    return excl, rec


def iter_source_texts() -> tuple[list[tuple[str, str, str]], dict]:
    """(text, gold_intent, dataset) for S1, exclusion + dedupe applied.

    Also records the replicated exclusion windows in the module-global
    _exclusion_record for the build meta."""
    global _exclusion_record
    excl, rec = build_exclusion()
    _exclusion_record = rec
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    stats = {"banking77": {"raw": 0, "excluded": 0, "dupes": 0},
             "clinc150": {"raw": 0, "excluded": 0, "dupes": 0}}

    def admit(text: str, gold: str, ds: str) -> None:
        stats[ds]["raw"] += 1
        n = norm(text)
        if n in excl:
            stats[ds]["excluded"] += 1
            return
        if n in seen:
            stats[ds]["dupes"] += 1
            return
        seen.add(n)
        out.append((text, gold, ds))

    with (B77 / "train.csv").open(encoding="utf-8", errors="replace",
                                  newline="") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            c = (r.get("category") or r.get("label") or "").strip()
            if t and c:
                admit(t, c, "banking77")
    for line in (CLINC / "train.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = (r.get("text") or "").strip()
        it = (r.get("intent") or "").strip()
        if t and it and it != "oos":
            admit(t, it, "clinc150")
    return out, {"exclusion_and_dupes": stats}


# ---------------------------------------------------------------------------
# S3 synthetic label sets
# ---------------------------------------------------------------------------

def label_variants(label_name: str, nuclei: list[str]) -> list[str]:
    """Rendered surface variants: nuclei x frames x slot fills (deduped)."""
    variants: list[str] = []
    off = int(hashlib.sha256(label_name.encode()).hexdigest(), 16) % 7
    for nucleus in nuclei:
        for fill_i in range(8):     # 8 slot fill cycles per frame
            fills = {k: SLOTS[k][(fill_i + off) % len(SLOTS[k])]
                     for k in SLOTS}
            base = nucleus
            for k, v in fills.items():
                base = base.replace("{" + k + "}", v)
            for frame in FRAMES:
                t = frame.format(n=base)
                t = t[0].upper() + t[1:] if t and t[0].isalpha() else t
                variants.append(t)
    seen: set[str] = set()
    return [v for v in variants if not (norm(v) in seen or seen.add(norm(v)))]


def polarity_rows(dom: str, facts: list[tuple[str, str]], rng: random.Random
                  ) -> list[dict]:
    """Bare yes/no rows (the referee's exact shape): question-wrapped + bare."""
    rows: list[dict] = []
    wraps = [
        ("Is the following claim true? Answer strictly from the facts "
         "stated in the text.\n{f}", True),
        ("{f}", False),
    ]
    for true_stmt, false_stmt in facts:
        for stmt, is_true in ((true_stmt, True), (false_stmt, False)):
            stmt = (stmt.replace("{CITY}", rng.choice(SLOTS["CITY"]))
                        .replace("{DATE}", rng.choice(SLOTS["DATE"]))
                        .replace("{PERSON}", rng.choice(SLOTS["PERSON"])))
            for tpl, _q in wraps:
                text = tpl.format(f=stmt)
                yes_label = 1 if is_true else 0
                rows.append({"label": yes_label, "request": text,
                             "tool": "yes", "source": f"synthetic_polarity_{dom}",
                             "split": split_of(text)})
                rows.append({"label": 1 - yes_label, "request": text,
                             "tool": "no",
                             "source": f"synthetic_polarity_{dom}",
                             "split": split_of(text)})
    return rows


def build_synthetic(used_norms: set[str]) -> tuple[list[dict], list[dict],
                                                   list[dict], dict]:
    """(S3 training pairs, A-adjudication rows, reserve sets, stats).

    Budgeted pools: each label's filtered variant pool is sliced EVENLY
    across its appearances (trained sets + the reserve draw), so a small
    pool degrades gracefully (fewer utterances per set) instead of
    exhausting — no silent text reuse, ever (global normalized dedupe)."""
    rng = random.Random(SEED + 7)
    pair_rows: list[dict] = []
    a_rows: list[dict] = []
    reserve: list[dict] = []
    used_norms = set(used_norms)    # do not mutate the caller's S1/S2 set
    variants_cache: dict[tuple[str, str], list[str]] = {}
    delivered = {"utterances": 0, "poor_pool_labels": 0}

    def variants_for(dom: str, name: str, nuclei: list[str]) -> list[str]:
        key = (dom, name)
        if key not in variants_cache:
            variants_cache[key] = label_variants(name, nuclei)
        return variants_cache[key]

    def fresh_pool(dom: str, name: str, nuclei: list[str]) -> list[str]:
        pool = [v for v in variants_for(dom, name, nuclei)
                if norm(v) not in used_norms]
        rng.shuffle(pool)
        return pool

    def emit(set_id: str, dom: str, name: str, desc: str, chunk: list[str],
             cap: int) -> list[str]:
        utts = chunk[:cap]
        del chunk[:cap]
        for utt in utts:
            used_norms.add(norm(utt))
            delivered["utterances"] += 1
            row = {"label": 1, "request": utt, "tool": desc,
                   "source": f"synthetic_{dom}", "set_id": set_id,
                   "gold_label": name, "split": split_of(utt)}
            pair_rows.append(row)
            a_rows.append({
                "id": (f"zs3-{hashlib.sha256(
                    (set_id + norm(utt)).encode()).hexdigest()[:16]}"),
                "set_id": set_id, "domain": dom, "text": utt,
                "gold_label": name, "gold_desc": desc,
            })
        return utts

    for dom in TRAINED_DOMAINS:
        spec = DOMAINS[dom]
        pool_names = [name for name, _, _ in spec["labels"]]
        desc_of = {name: f"{name}: {d}"
                   for name, d, _ in spec["labels"]}
        nuclei_of = {name: nuc for name, _, nuc in spec["labels"]}
        # phase 1: rosters + appearance counts
        rosters = []
        appearances: dict[str, int] = {n: 0 for n in pool_names}
        for set_i in range(SETS_PER_DOMAIN):
            k = 8 + (set_i % 4)                # 8..11 labels per set
            labels = rng.sample(pool_names, k)
            rosters.append(labels)
            for name in labels:
                appearances[name] += 1
        # phase 2: budgeted pools
        chunks: dict[str, list[str]] = {}
        for name in pool_names:
            pool = fresh_pool(dom, name, nuclei_of[name])
            if len(pool) < appearances[name] * 4:
                delivered["poor_pool_labels"] += 1
            chunks[name] = pool
        # phase 3: emit pairs
        for set_i, labels in enumerate(rosters):
            set_id = f"{dom}-s{set_i:02d}"
            label_descs = {name: desc_of[name] for name in labels}
            for name in labels:
                remaining_apps = appearances[name]
                budget = max(2, len(chunks[name]) // remaining_apps)
                appearances[name] -= 1
                n_want = rng.randint(UTTS_PER_LABEL_LO - 2,
                                     UTTS_PER_LABEL_HI - 4)
                utts = emit(set_id, dom, name, desc_of[name],
                            chunks[name], min(budget, n_want))
                siblings = [label_descs[o] for o in labels if o != name]
                negs = rng.sample(siblings, min(NEG_SYNTH, len(siblings)))
                for utt in utts:
                    for nd in negs:
                        pair_rows.append({
                            "label": 0, "request": utt, "tool": nd,
                            "source": f"synthetic_{dom}_sibneg",
                            "set_id": set_id, "gold_label": name,
                            "split": split_of(utt)})
        # polarity rows once per domain (bare yes/no, the referee shape)
        pair_rows.extend(polarity_rows(dom, spec["facts"], rng))

    for dom in RESERVE_DOMAINS:
        spec = DOMAINS[dom]
        pool_names = [name for name, _, _ in spec["labels"]]
        nuclei_of = {name: nuc for name, _, nuc in spec["labels"]}
        labels = rng.sample(pool_names, 12)
        entry = {"domain": dom, "set_id": f"{dom}-reserve",
                 "labels": []}
        for name in labels:
            pool = fresh_pool(dom, name, nuclei_of[name])
            entry["labels"].append({"name": name,
                                    "desc": f"{name}: " + next(
                                        d for n_, d, _ in spec["labels"]
                                        if n_ == name),
                                    "utterances": pool[:16]})
            delivered["utterances"] += min(16, len(pool))
        reserve.append(entry)

    stats = {"trained_sets": SETS_PER_DOMAIN * len(TRAINED_DOMAINS),
             "reserve_sets": len(reserve),
             "utterances_delivered": delivered["utterances"],
             "poor_pool_labels": delivered["poor_pool_labels"]}
    return pair_rows, a_rows, reserve, stats


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    digest = sha256_bytes(data.encode("utf-8"))
    (path.parent / (path.name + ".sha256")).write_text(digest + "\n",
                                                       encoding="utf-8")
    return digest


def dedupe_rate(rows: list[dict]) -> float:
    """Duplicate rate over PAIR keys (norm(request), rendered candidate).

    The corpus unit is the pair: one request legitimately appears with its
    positive AND several sibling negatives (that repetition IS the listwise
    structure). A duplicated PAIR would be hidden weight; that is what the
    gate measures."""
    keys = [(norm(r["request"]), r["tool"]) for r in rows]
    return 1 - len(set(keys)) / max(1, len(keys))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # ---------------- S1 listwise ----------------
    texts, stats1 = iter_source_texts()
    b77_intents = json.loads((B77 / "categories.json").read_text())
    clinc_labels = json.loads((CLINC / "labels.json").read_text())
    desc_of = {"banking77": {i: desc77(i) for i in b77_intents},
               "clinc150": {i: desc77(i) for i in clinc_labels}}
    s1: list[dict] = []
    used_norms: set[str] = {norm(t) for t, _, _ in texts}
    for text, gold, ds in texts:
        intents = b77_intents if ds == "banking77" else clinc_labels
        gold_desc = desc_of[ds][gold]
        s1.append({"label": 1, "request": text, "tool": gold_desc,
                   "source": f"{ds}_pos", "split": split_of(text)})
        cands = [it for it in intents if it != gold]
        ranked = lexical_siblings(text, cands)
        top = ranked[0][1]
        s1.append({"label": 0, "request": text, "tool": desc_of[ds][top],
                   "source": f"{ds}_sibneg_hard", "split": split_of(text)})
        others = [it for it in cands if it != top]
        s1.append({"label": 0, "request": text,
                   "tool": desc_of[ds][rng.choice(others)],
                   "source": f"{ds}_sibneg_rand", "split": split_of(text)})
        if rng.random() < 0.10:
            for it in rng.sample(others, 2):
                s1.append({"label": 0, "request": text,
                           "tool": desc_of[ds][it],
                           "source": f"{ds}_sibneg_extra",
                           "split": split_of(text)})

    # ---------------- S2 null ----------------
    excl, _ = build_exclusion()
    NULLS = [
        ("__null__", "none of these descriptions match"),
        ("none", "none of these descriptions match"),
        ("no_match", "the text is about none of these topics"),
        ("__null__", "the text is about none of these topics"),
    ]
    # S2 (a)+(b): oos texts from the CROSS-DOMAIN bitext verticals (telco +
    # insurance). MEASURED FINDING (pre-build data review, recorded in the
    # spec): the banking77-oos id-oos relabels contradict the eval surface's
    # own ground truth — 2,053/2,054 id-oos train texts sit in b77 train.csv
    # with IN-DOMAIN categories — so using them as null positives would label
    # the same text as both an S1 positive and a null positive. The
    # used_norms filter caught this (supply came out empty); the registered
    # supply is the topically-disjoint telco/insurance customer-support
    # utterances instead (real public data, question register matches the oos
    # eval windows).
    oos_supply = []
    seen_oos: set[str] = set()
    _ph = re.compile(r"\{\{[^}]*\}\}")
    for fname in ("telco.csv", "insurance.csv"):
        with (ROOT / f"data/incoming/external/bitext/{fname}").open(
                encoding="utf-8", errors="replace", newline="") as fh:
            for r in csv.DictReader(fh):
                t = (r.get("instruction") or "").strip()
                t = _ph.sub("the reference", t)
                t = re.sub(r"\s+", " ", t)
                n = norm(t)
                if (t and 15 <= len(t) <= 140 and n not in excl
                        and n not in used_norms and n not in seen_oos):
                    seen_oos.add(n)
                    oos_supply.append(t)
    rng.shuffle(oos_supply)
    oos_supply = oos_supply[:OOS_TEXTS_TARGET]
    s2: list[dict] = []
    for i, t in enumerate(oos_supply):
        key, phr = NULLS[i % len(NULLS)]
        s2.append({"label": 1, "request": t, "tool": f"{key}: {phr}",
                   "source": "null_pos_oos", "split": split_of(t)})
        key2, phr2 = NULLS[(i + 2) % len(NULLS)]
        if (key2, phr2) != (key, phr):
            s2.append({"label": 1, "request": t, "tool": f"{key2}: {phr2}",
                       "source": "null_pos_oos", "split": split_of(t)})
        ranked = lexical_siblings(t, b77_intents)
        top = ranked[0][1]
        s2.append({"label": 0, "request": t, "tool": desc77(top),
                   "source": "null_neg_oos_confusable",
                   "split": split_of(t)})
    # CLINC train oos supply (small; all in)
    clinc_oos_texts = []
    for line in (CLINC / "train.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("intent") == "oos" and norm(r["text"]) not in excl:
                clinc_oos_texts.append(r["text"].strip())
    for i, t in enumerate(clinc_oos_texts):
        key, phr = NULLS[i % len(NULLS)]
        s2.append({"label": 1, "request": t, "tool": f"{key}: {phr}",
                   "source": "null_pos_oos", "split": split_of(t)})
    # in-domain texts vs null -> label 0
    ind_texts = [t for t, _, _ in rng.sample(
        texts, min(INDOMAIN_NULL_NEG, len(texts)))]
    for i, t in enumerate(ind_texts):
        key, phr = NULLS[i % len(NULLS)]
        s2.append({"label": 0, "request": t, "tool": f"{key}: {phr}",
                   "source": "null_neg_indomain", "split": split_of(t)})

    # ---------------- S3 synthetic ----------------
    s3, a_rows, reserve, stats3 = build_synthetic(used_norms)

    # ---------------- S4 BFCL train verbatim ----------------
    s4 = [json.loads(ln) for ln in BFCL_PAIRS.read_text(
        encoding="utf-8").splitlines() if ln.strip()
        and json.loads(ln)["split"] == "train"]

    # ---------------- band + dedupe gates (fail loudly) ----------------
    files = {"listwise-pairs.jsonl": s1, "null-pairs.jsonl": s2,
             "synthetic-labelsets.jsonl": s3, "bfcl-train.jsonl": s4}
    for name, rows in files.items():
        lo, hi = BANDS[name]
        n = len(rows)
        rate = dedupe_rate(rows)
        if not (lo <= n <= hi):
            print(f"BAND FAIL {name}: {n} outside [{lo}, {hi}]",
                  file=sys.stderr)
            return 2
        if rate > DEDUPE_MAX:
            print(f"DEDUPE FAIL {name}: {rate:.4f} > {DEDUPE_MAX}",
                  file=sys.stderr)
            return 2
    if stats3["utterances_delivered"] < 12_000:
        print(f"POOL FAIL: only {stats3['utterances_delivered']} synthetic "
              "utterances delivered", file=sys.stderr)
        return 2

    # ---------------- write corpus ----------------
    digests = {}
    for name, rows in files.items():
        digests[name] = write_jsonl(OUT / name, rows)
    (OUT / "synthetic-labelsets-reserve.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reserve),
        encoding="utf-8")
    digests["synthetic-labelsets-reserve.jsonl"] = sha256_bytes(
        (OUT / "synthetic-labelsets-reserve.jsonl").read_bytes())
    (OUT / ("synthetic-labelsets-reserve.jsonl.sha256")).write_text(
        digests["synthetic-labelsets-reserve.jsonl"] + "\n",
        encoding="utf-8")

    # ---------------- adjudication A (full S3 coverage) + B prompts -----
    write_jsonl(OUT / "adjudication-A.jsonl", a_rows)
    # reserve utterances must NEVER overlap trained texts (spec)
    reserve_norms = {norm(u) for r in reserve for lab in r["labels"]
                     for u in lab["utterances"]}
    trained_norms = {norm(r["request"]) for r in s3}
    overlap = reserve_norms & trained_norms
    if overlap:
        print(f"RESERVE LEAK FAIL: {len(overlap)} reserve texts in training",
              file=sys.stderr)
        return 2

    by_dom: dict[str, list[dict]] = {}
    for r in a_rows:
        by_dom.setdefault(r["domain"], []).append(r)
    sample: list[dict] = []
    per_dom = ADJ_B_SAMPLE // len(TRAINED_DOMAINS)
    for dom in TRAINED_DOMAINS:
        sample.extend(rng.sample(by_dom[dom], per_dom))
    rng.shuffle(sample)
    # blind prompts: shuffled candidate codes, NO gold anywhere
    prompts = []
    for r in sample:
        dom = r["domain"]
        spec = DOMAINS[dom]
        pool_names = [name for name, _, _ in spec["labels"]]
        k = 9
        cand_names = rng.sample(pool_names, k)
        if r["gold_label"] not in cand_names:
            cand_names[rng.randrange(k)] = r["gold_label"]
        rng.shuffle(cand_names)
        codes = [chr(ord("A") + i) for i in range(len(cand_names))]
        cands = [{"code": c,
                  "desc": f"{n}: " + next(d for n_, d, _ in spec["labels"]
                                          if n_ == n)}
                 for c, n in zip(codes, cand_names)]
        prompts.append({"id": r["id"], "domain": dom, "text": r["text"],
                        "candidates": cands,
                        "instruction": (
                            "Pick the code of the description that best "
                            "matches the text for the task the text is "
                            "performing, or Z if none of them match. "
                            "Judge only from the text itself.")})
    write_jsonl(SCRATCH / "adjudication-B-prompts.jsonl", prompts)

    # ---------------- build meta + provenance ----------------
    meta = {
        "arm": "zero-shot-select-r1",
        "seed": SEED,
        "files": {k: {"rows": len(v), "sha256": digests[k]}
                  for k, v in files.items()},
        "reserve_file": {"sets": len(reserve),
                         "sha256": digests["synthetic-labelsets-reserve.jsonl"]},
        "s1_texts": stats1,
        "s2_nulls": {"oos_texts_b77": len(oos_supply),
                     "oos_texts_clinc_train": len(clinc_oos_texts),
                     "indomain_null_negs": len(ind_texts)},
        "s3_synthetic": stats3,
        "adjudication": {"A_rows": len(a_rows), "B_sample": len(prompts),
                         "B_prompts": str((SCRATCH /
                                           "adjudication-B-prompts.jsonl")
                                          .relative_to(ROOT))},
        "exclusion": _exclusion_record,
    }
    (OUT / "build-meta.json").write_text(json.dumps(meta, indent=1),
                                         encoding="utf-8")
    prov = OUT / "PROVENANCE.md"
    section = f"""<!-- section:zero-shot-select-r1 -->
## zero-shot-select-r1 (built {meta_seed_stamp()})

Sources (sha256 in build-meta.json; sidecars beside every jsonl):
- listwise-pairs.jsonl — Banking77 train + CLINC150 train (public research
  corpora, provenanced under data/incoming/external/*/PROVENANCE.md);
  gold = corpus labels (mechanical). Exclusion: FULL official test /
  validation+test splits + ood-oos valid window (superset of the consumed
  s42 + seeds 43-46 ledgers) by strong-normalized text; keep-first dedupe.
- null-pairs.jsonl — oos supply bitext telco+insurance cross-domain
  utterances (the banking77-oos id-oos relabels MEASURED-CONTRADICT the
  eval ground truth: 2,053/2,054 carry in-domain categories in b77
  train.csv — see the spec's Source 2 finding) + CLINC150 train oos
  survivors; null positives are mechanically labeled by oos provenance.
- synthetic-labelsets.jsonl — ORIGINAL authored content
  (scripts/zs_select_r1_domains.py), session-LLM generated per the approved
  teacher pattern; rater A = intended gold (adjudication-A.jsonl, full
  coverage), rater B = blind re-check of {len(prompts)} rows
  (.scratch/zs-select-r1/adjudication-B-prompts.jsonl rendered shuffled
  without A's labels; answers committed as evidence in
  adjudication-B.jsonl, merged by
  scripts/finalize_zs_select_adjudication.py); kappa(A,B) gate >= 0.6,
  mismatch rows DROPPED.
- bfcl-train.jsonl — verbatim copy of the BFCL pairs train split
  (dev split excluded: select-calibrator instruments).
- synthetic-labelsets-reserve.jsonl — 8 held-out sets, NEVER trained
  (zero normalized overlap verified in-builder and by the audit gate).
Rules: no fabricated values in real rows; synthetic entities are fictional;
deterministic build (seed 42), idempotent section (replaced on rebuild).
"""
    text = prov.read_text() if prov.exists() else \
        "# data/real/zero-shot-select-r1 — provenance\n"
    marker_begin = "<!-- section:zero-shot-select-r1 -->"
    if marker_begin in text:
        text = text[:text.index(marker_begin)].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    prov.write_text(text, encoding="utf-8")

    print(json.dumps({k: {"rows": len(v), "sha256": digests[k][:12]}
                      for k, v in files.items()} |
                     {"adjudication_B_prompts": len(prompts),
                      "reserve_sets": len(reserve)}, indent=1))
    return 0


def meta_seed_stamp() -> str:
    import time
    return time.strftime("%Y-%m-%d %H:%M")


_exclusion_record: dict = {}


if __name__ == "__main__":
    raise SystemExit(main())
