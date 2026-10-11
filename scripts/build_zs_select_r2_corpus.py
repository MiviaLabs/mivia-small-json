#!/usr/bin/env python3
"""zero-shot-select r2 corpus builder (zero-shot-select-r2).

Two NEW files under data/real/zero-shot-select-r2/ (the r1 corpus files are
reused IN PLACE — bytes untouched, sha-bound by the audit gate):

  nli-converted.jsonl   public NLI (MNLI/SNLI/ANLI train splits ONLY)
                        converted to the select shape:
                        entailment -> (premise, "{key}: {hypothesis}", 1),
                        contradiction -> (..., 0), NEUTRAL EXCLUDED.
                        Half the sampled rows render (a) dict-shape
                        (1 pair), half (b) verdict-shape (2 pairs, bare
                        yes/no over the registered question wrap or the
                        bare premise — the referee's exact surface).
  verdict-pairs.jsonl   the binary referee's exact construction
                        (scripts/build_binary_select_validation.py's four
                        families, shapes and committed mechanical label
                        rules) over the FULL pools with an independent
                        seed; the referee's own 500 scored texts are
                        EXCLUDED (verified 0 by the audit gate).

Deterministic (fixed seeds). Fails loudly (exit 2) on any declared-band,
dedupe (<=1%), supply-shortfall or referee-overlap breach — never shrinks
silently. Writes sha256 sidecars, a build meta and the idempotent
PROVENANCE section (rule 12).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_binary_select_validation import (  # noqa: E402
    DIAG, ISSUES, LOGS, MAX_TEXT_CHARS, Q_BUG, Q_LOG, Q_SEVERITY,
    _FAILURE_RE, _qtext, _REQUEST_RE, _YESNO_RE, load_jsonl,
)
from build_zs_select_r1_corpus import (  # noqa: E402
    build_exclusion, dedupe_rate, norm, split_of, write_jsonl,
)

OUT = ROOT / "data/real/zero-shot-select-r2"
NLI_RAW = ROOT / "data/incoming/external/nli/raw"
REFEREE = ROOT / "data/eval/binary-select-v1.jsonl"
SCRATCH = ROOT / ".scratch/zs-select-r2"

SEED = 42
VERDICT_SEED = 424242
NLI_SAMPLES = {"mnli": 30_000, "snli": 18_000, "anli": 12_000}
DICT_KEYS = ["description", "claim", "statement", "label", "topic"]
QUESTION_WRAP = ("Is the following claim true? Answer strictly from the "
                 "facts stated in the text.\n{p}\nClaim: {h}")
BARE_CLAIM = "{p}\nClaim: {h}"

# Declared bands (spec; the audit gate enforces these against the bytes).
BANDS = {
    "nli-converted.jsonl": (80_000, 100_000),
    "verdict-pairs.jsonl": (20_000, 60_000),
}
DEDUPE_MAX = 0.01


def _load_nli_source(src: str) -> tuple[list[tuple[str, str]],
                                        list[tuple[str, str]]]:
    """(entailment, contradiction) (premise, hypothesis) pairs, train only."""
    ent: list[tuple[str, str]] = []
    con: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def admit(p: str, h: str, lab: bool) -> None:
        p, h = re.sub(r"\s+", " ", p).strip(), re.sub(r"\s+", " ", h).strip()
        if not p or not h or len(p) > MAX_TEXT_CHARS:
            return
        k = (norm(p), norm(h))
        if k in seen:
            return
        seen.add(k)
        (ent if lab else con).append((p, h))

    if src == "mnli":
        path = NLI_RAW / "multinli_1.0/multinli_1.0_train.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            g = r.get("gold_label")
            if g in ("entailment", "contradiction"):
                admit(r["sentence1"], r["sentence2"], g == "entailment")
    elif src == "snli":
        path = NLI_RAW / "snli_1.0/snli_1.0_train.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            g = r.get("gold_label")
            if g in ("entailment", "contradiction"):
                admit(r["sentence1"], r["sentence2"], g == "entailment")
    elif src == "anli":
        for split in ("R1", "R2", "R3"):
            path = NLI_RAW / f"anli_v1.0/{split}/train.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("label") in ("e", "c"):
                    admit(r["context"], r["hypothesis"], r["label"] == "e")
    else:
        raise ValueError(src)
    return ent, con


def build_nli(excl: set[str]) -> tuple[list[dict], dict]:
    """Source N: the NLI-converted pairs (spec conversion rules)."""
    rng = random.Random(SEED)
    rows: list[dict] = []
    stats: dict = {}
    for src, target in NLI_SAMPLES.items():
        ent, con = _load_nli_source(src)
        half = target // 2
        if len(ent) < half or len(con) < half:
            raise SystemExit(
                f"SUPPLY FAIL nli/{src}: entail {len(ent)} / contra "
                f"{len(con)}, need {half} each (rule 4)")
        rng.shuffle(ent)
        rng.shuffle(con)
        sampled = [(p, h, 1) for p, h in ent[:half]] + \
                  [(p, h, 0) for p, h in con[:half]]
        rng.shuffle(sampled)
        kept = 0
        dropped_excl = 0
        for i, (p, h, lab) in enumerate(sampled):
            if norm(p) in excl:
                dropped_excl += 1
                continue
            sp = split_of(p)
            if i % 2 == 0:                       # (a) dict shape, 1 pair
                key = DICT_KEYS[i % len(DICT_KEYS)]
                rows.append({"label": lab, "request": p,
                             "tool": f"{key}: {h}",
                             "source": f"nli_dict_{src}", "split": sp})
                kept += 1
            else:                                # (b) verdict shape, 2 pairs
                text = (QUESTION_WRAP.format(p=p, h=h) if i % 4 < 2
                        else BARE_CLAIM.format(p=p, h=h))
                rows.append({"label": lab, "request": text, "tool": "yes",
                             "source": f"nli_verdict_{src}", "split": sp})
                rows.append({"label": 1 - lab, "request": text, "tool": "no",
                             "source": f"nli_verdict_{src}", "split": sp})
                kept += 1
        stats[src] = {"sampled": target, "pairs": kept,
                      "dropped_eval_overlap": dropped_excl,
                      "supply_entail": len(ent), "supply_contra": len(con)}
    return rows, stats


# ---------------------------------------------------------------------------
# Source V: the referee's exact construction over the full pools
# ---------------------------------------------------------------------------

def _verdict_pools(excl: set[str], referee_norms: set[str]) -> dict:
    """(family, shape) -> {"yes": [texts], "no": [texts]} — the referee
    builder's families/shapes/label rules, independent seed, FULL pools."""
    pools: dict[tuple[str, str], dict[str, list[str]]] = {}
    seen: set[str] = set()

    def admit(family: str, shape: str, raw: str, question: str | None,
              expected: str) -> None:
        raw = raw.strip()[:MAX_TEXT_CHARS]
        text = _qtext(question, raw) if question is not None else raw
        n = norm(text)
        if not n or n in seen or n in excl or n in referee_norms:
            return
        if shape == "state" and _YESNO_RE.search(raw):
            return
        seen.add(n)
        pools.setdefault((family, shape), {"yes": [], "no": []})[expected] \
            .append(text)

    for r in load_jsonl(DIAG):
        sev, tool, raw = r.get("severity"), r.get("tool"), r.get("raw") or ""
        if not raw.strip():
            continue
        if sev in ("error", "warning"):
            exp = "yes" if sev == "error" else "no"
            admit("diag-severity", "question", raw, Q_SEVERITY, exp)
            admit("diag-severity", "state", raw, None, exp)
    diag_rows = [r for r in load_jsonl(DIAG)
                 if r.get("tool") and (r.get("raw") or "").strip()]
    by_tool: dict[str, list[dict]] = {}
    for r in diag_rows:
        by_tool.setdefault(r["tool"], []).append(r)
    tools = sorted(by_tool)
    rng_tool = random.Random(VERDICT_SEED)
    for tool in tools:
        others = [t for t in tools if t != tool]
        for r in by_tool[tool]:
            raw = r["raw"]
            admit("diag-tool", "question", raw,
                  f"Was this diagnostic produced by {tool}?" +
                  " Answer strictly from the facts stated in the text.",
                  "yes")
            decoy = rng_tool.choice(others)
            admit("diag-tool", "question", raw,
                  f"Was this diagnostic produced by {decoy}?" +
                  " Answer strictly from the facts stated in the text.",
                  "no")
    for r in load_jsonl(LOGS):
        raw = r.get("raw") or ""
        level = str((r.get("gold") or {}).get("level") or "")
        if not raw.strip():
            continue
        if level in {"ERROR", "FATAL"}:
            admit("log-level", "question", raw, Q_LOG, "yes")
            admit("log-level", "state", raw, None, "yes")
        elif level == "INFO" and not _FAILURE_RE.search(raw):
            admit("log-level", "question", raw, Q_LOG, "no")
            admit("log-level", "state", raw, None, "no")
    for r in load_jsonl(ISSUES):
        raw = r.get("raw") or ""
        labels = " ".join((r.get("gold") or {}).get("labels") or []).lower()
        if not raw.strip():
            continue
        failing = bool(_FAILURE_RE.search(raw))
        if "bug" in labels and failing:
            admit("issue-kind", "question", raw, Q_BUG, "yes")
        elif (("feature" in labels or "enhancement" in labels)
              and not failing and _REQUEST_RE.search(raw)):
            admit("issue-kind", "question", raw, Q_BUG, "no")
    return pools


def build_verdict(excl: set[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    referee_norms = {norm(json.loads(line)["text"])
                     for line in REFEREE.read_text().splitlines() if line.strip()}
    pools = _verdict_pools(excl, referee_norms)
    stats: dict = {"referee_norms_excluded": len(referee_norms),
                   "pools": {}}
    for (family, shape), sides in sorted(pools.items()):
        stats["pools"][f"{family}/{shape}"] = {
            "yes": len(sides["yes"]), "no": len(sides["no"])}
        for expected, texts in sides.items():
            y = 1 if expected == "yes" else 0
            for text in texts:
                sp = split_of(text)
                rows.append({"label": y, "request": text, "tool": "yes",
                             "source": f"verdict_{family}_{shape}",
                             "split": sp})
                rows.append({"label": 1 - y, "request": text, "tool": "no",
                             "source": f"verdict_{family}_{shape}",
                             "split": sp})
    return rows, stats


# ---------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    excl, excl_rec = build_exclusion()

    nli_rows, nli_stats = build_nli(excl)
    verdict_rows, verdict_stats = build_verdict(excl)

    files = {"nli-converted.jsonl": nli_rows,
             "verdict-pairs.jsonl": verdict_rows}
    for name, rws in files.items():
        lo, hi = BANDS[name]
        n = len(rws)
        rate = dedupe_rate(rws)
        if not (lo <= n <= hi):
            print(f"BAND FAIL {name}: {n} outside [{lo}, {hi}]",
                  file=sys.stderr)
            return 2
        if rate > DEDUPE_MAX:
            print(f"DEDUPE FAIL {name}: {rate:.4f} > {DEDUPE_MAX}",
                  file=sys.stderr)
            return 2
    # across-file pair overlap
    keys = [(norm(r["request"]), r["tool"]) for rws in files.values()
            for r in rws]
    cross_rate = 1 - len(set(keys)) / len(keys)
    if cross_rate > DEDUPE_MAX:
        print(f"DEDUPE FAIL across files: {cross_rate:.4f} > {DEDUPE_MAX}",
              file=sys.stderr)
        return 2
    # referee disjointness (verdict file): 0 overlap, hard gate
    referee_norms = {norm(json.loads(line)["text"])
                     for line in REFEREE.read_text().splitlines() if line.strip()}
    overlap = {norm(r["request"]) for r in verdict_rows} & referee_norms
    if overlap:
        print(f"REFEREE LEAK FAIL: {len(overlap)} verdict texts overlap the "
              "referee", file=sys.stderr)
        return 2

    digests = {}
    for name, rws in files.items():
        digests[name] = write_jsonl(OUT / name, rws)

    # polarity balance (declared 50/50 +/- 2pp — by construction exact per
    # source; measured here from the bytes)
    pos = sum(r["label"] for r in nli_rows)
    bal = 100 * pos / max(1, len(nli_rows))
    if not (48.0 <= bal <= 52.0):
        print(f"BALANCE FAIL nli: {bal:.2f}% positives", file=sys.stderr)
        return 2

    meta = {
        "arm": "zero-shot-select-r2",
        "seed": SEED, "verdict_seed": VERDICT_SEED,
        "files": {k: {"rows": len(v), "sha256": digests[k]}
                  for k, v in files.items()},
        "nli": {**nli_stats, "positive_share_pct": round(bal, 2),
                "dict_rows": sum(1 for r in nli_rows
                                 if r["source"].startswith("nli_dict")),
                "verdict_rows": sum(1 for r in nli_rows
                                    if r["source"].startswith(
                                        "nli_verdict"))},
        "verdict": verdict_stats,
        "cross_file_pair_dedupe_pct": round(100 * cross_rate, 4),
        "referee_overlap": 0,
        "exclusion": excl_rec,
    }
    (OUT / "build-meta.json").write_text(json.dumps(meta, indent=1),
                                         encoding="utf-8")

    section = f"""<!-- section:zero-shot-select-r2 -->
## zero-shot-select-r2 (new files; r1 files reused in place)

- nli-converted.jsonl — MNLI/SNLI/ANLI TRAIN splits only (public research
  corpora; sources, urls, zip sha256 and license notes in
  data/incoming/external/nli/PROVENANCE.md; license verification deferred
  per governing rule 6). Conversion mechanical from gold labels:
  entailment -> match, contradiction -> no-match, neutral excluded.
  Declared sampling (seed 42): mnli 30,000 / snli 18,000 / anli 12,000
  premise-hypothesis rows; 50/50 dict-shape / verdict-shape rendering.
- verdict-pairs.jsonl — the binary referee's exact construction
  (build_binary_select_validation.py families/shapes/committed label
  rules) over the FULL pools, independent seed {VERDICT_SEED}; the
  referee's own 500 scored texts are excluded (gate-verified 0); labels
  mechanically derived (gold fields + committed lexicons — no model, no
  vibes).
- r1 files (listwise/null/synthetic/bfcl) are NOT duplicated here; the r2
  trainers read them from data/real/zero-shot-select-r1/ and the audit
  gate binds their committed sha256 sidecars.
Rules: no fabricated values; deterministic build (seeds above); idempotent
section (replaced on rebuild, rule 12). Counts + sha256 in build-meta.json
and the *.sha256 sidecars.
"""
    prov = OUT / "PROVENANCE.md"
    text = (prov.read_text() if prov.exists()
            else "# data/real/zero-shot-select-r2 — provenance\n")
    marker = "<!-- section:zero-shot-select-r2 -->"
    if marker in text:
        text = text[:text.index(marker)].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    prov.write_text(text, encoding="utf-8")

    print(json.dumps({k: {"rows": len(v), "sha256": digests[k][:12]}
                      for k, v in files.items()} |
                     {"nli_positive_share_pct": round(bal, 2),
                      "cross_file_dedupe_pct": round(100 * cross_rate, 4)},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
