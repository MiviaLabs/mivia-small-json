#!/usr/bin/env python3
"""Reality evals: extraction + routing on real data with real gold.

Spec: reality-evals Baselines, not bars — no training,
no artifact changes. Deterministic (seed 42 sampling). Every metric is
defined here once and printed with its n.

Surfaces:
- gh-prs (300 real PRs, title+body gold): routing distribution +
  title/body field quality vs gold.
- incident tickets (1,000 en + 1,000 de, subject+body gold): routing→issue,
  title/body quality, per language — the multilingual robustness slice.
- cleancodereview (n=1,000 real review comments, comment-text gold):
  routing→review_comment + comment containment (file/line have no gold
  here — presence reported, accuracy not claimed).
- SWE-bench Lite (100 real problem statements, body gold): body
  containment + title presence.

Writes reports/reality-evals.json and prints the tables.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json import bootstrap  # noqa: E402
from mivia_small_json.pipeline import classify, load_active_router  # noqa: E402

OUT_JSON = ROOT / "reports" / "reality-evals.json"
SEED = 42
_NORM = re.compile(r"[^a-z0-9äöüß ]+")


def _norm(text: str) -> str:
    return _NORM.sub(" ", (text or "").lower()).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9äöüß]{3,}", (text or "").lower())}


def _score_fields(obj: dict, gold_title: str, gold_body: str) -> dict:
    title = str(obj.get("title", "") or "")
    body = str(obj.get("body", "") or obj.get("description", "") or "")
    gn, tn = _norm(gold_title), _norm(title)
    gt, bt = _tokens(gold_body), _tokens(body)
    prefix = (tn[:60] == gn[:60]) if tn and gn else False
    return {
        "title_exact": tn == gn and bool(gn),
        "title_prefix60": prefix,
        "body_recall": (round(len(gt & bt) / len(gt), 3) if gt else None),
        "body_nonempty": bool(body.strip()),
    }


def _collect(rows: list[dict], expected: str | None,
             router) -> dict:
    types = Counter()
    agg = Counter()
    details = []
    for r in rows:
        obj, meta = classify(r["text"], router=router)
        got = obj.get("type")
        types[got] += 1
        if meta.get("valid") is False:
            agg["invalid_json"] += 1
        if expected is None or got == expected:
            f = _score_fields(obj, r["gold_title"], r["gold_body"])
            for k, v in f.items():
                if v is not None:
                    agg[k] += int(bool(v)) if isinstance(v, bool) else v
            agg["scored"] += 1
            if len(details) < 3:
                details.append({"gold_title": r["gold_title"][:70],
                                "got_type": got, **f})
    n = len(rows)
    out = {"n": n, "routing": dict(types.most_common()),
           "details": details}
    if expected is not None:
        out["routed_expected_pct"] = round(100 * types.get(expected, 0) / n, 1)
    scored = agg.get("scored", 0)
    for k in ("title_exact", "title_prefix60", "body_nonempty"):
        if scored:
            out[k + "_pct"] = round(100 * agg[k] / scored, 1)
    if scored:
        out["body_recall_mean_sampled"] = round(agg["body_recall"] / scored, 3)
    return out


def gh_prs(n: int = 300, router=None) -> dict:
    prs = json.loads((ROOT / "data/incoming/gh-prs.json").read_text())
    rng = random.Random(SEED)
    rng.shuffle(prs)
    rows = [{"text": f"{p['title']}\n\n{p.get('body') or ''}".strip(),
             "gold_title": p["title"], "gold_body": p.get("body") or ""}
            for p in prs[:n]]
    return _collect(rows, None, router)


def tickets(n_per_lang: int = 1000, router=None) -> dict:
    with (ROOT / "data/incoming/external/incident-severity/"
          "tickets-multi-lang-20k.csv").open(newline="", encoding="utf-8",
                                             errors="replace") as fh:
        all_rows = list(csv.DictReader(fh))
    out = {}
    for lang in ("en", "de"):
        pool = [r for r in all_rows if r.get("language") == lang]
        rng = random.Random(SEED)
        rng.shuffle(pool)
        rows = [{"text": f"{r['subject']}\n\n{r.get('body') or ''}".strip(),
                 "gold_title": r["subject"], "gold_body": r.get("body") or ""}
                for r in pool[:n_per_lang]]
        out[lang] = _collect(rows, "issue", router)
    out["de_minus_en_routed_pp"] = round(
        out["de"]["routed_expected_pct"] - out["en"]["routed_expected_pct"], 1)
    return out


def clean_code_review(n: int = 1000, router=None) -> dict:
    rows_all = [json.loads(line) for line in
                (ROOT / "data/incoming/external/cleancodereview/comments.jsonl")
                .read_text().splitlines() if line.strip()]
    rng = random.Random(SEED)
    rng.shuffle(rows_all)
    rows = [{"text": r["text"], "gold_title": r["text"][:80],
             "gold_body": r["text"]} for r in rows_all[:n]]
    res = _collect(rows, "review_comment", router)
    # comment containment on whatever type came back: gold text tokens found
    # in the extracted comment-ish fields
    hits = total = 0
    for r in rows[:200]:
        obj, _meta = classify(r["text"])
        blob = " ".join(str(v) for k, v in obj.items()
                        if k != "type" and isinstance(v, str))
        gt = _tokens(r["gold_body"])
        if gt:
            total += 1
            if len(gt & _tokens(blob)) / len(gt) >= 0.5:
                hits += 1
    res["comment_contained_50pct_pct"] = round(100 * hits / total, 1) if total else None
    return res


def swebench(router=None) -> dict:
    data = json.loads((ROOT / "data/incoming/external/swebench/"
                       "swebench_lite_100.json").read_text())
    rows = []
    for rec in data:
        body = rec.get("problem_statement") or ""
        title = (rec.get("title") or body.split("\n")[0])[:120]
        rows.append({"text": body, "gold_title": title, "gold_body": body})
    res = _collect(rows[:100], "issue", router)
    return res


def main() -> int:
    # THE shipped chain: full schema world + the active ONNX router.
    # classify(router=None) is the RULES-ONLY fallback — the first version
    # of this script omitted the router and measured the wrong system
    # (SWE-bench "54%" was rules; the true chain number is ~98%).
    bootstrap.register_all_schemas()
    router = load_active_router()
    if router is None:
        raise SystemExit("active ONNX router not loadable")
    result = {
        "spec": "reality-evals",
        "seed": SEED,
        "router": "models/router-encoder-int8.onnx via classify(router=...)",
        "gh_prs": gh_prs(router=router),
        "tickets": tickets(router=router),
        "cleancodereview": clean_code_review(router=router),
        "swebench_issue": swebench(router=router),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nwritten {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
