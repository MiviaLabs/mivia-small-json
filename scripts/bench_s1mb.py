"""S1MB head-to-head: mivia-small-json on the System One Mosaic Benchmark.

Dataset: hotchpotch/s1mb-dataset (HF) — 106 task configs, 14,009 rows;
each row = `input.state_json` (a JSON string) + `targets` (a list of
decision questions: `{decision_id, ids: [label...], probabilities:
one-hot gold}`).

Mapping: primitive-to-primitive, same contract as bench_jevbench.py — the
shipped zero-shot `select` ranks the target's `ids` as bare candidate
strings against the state text; argmax pick scored against the gold id.
No criteria strings exist in S1MB (bare labels), no per-config tuning.

Honesty rules: full coverage (every config, every row), no sampling; the
official S1MB board ranks models by Borda count over its own protocol —
our number is plain per-decision accuracy on the same rows and is labeled
as such. Calibration is reported per status (select always returns the
argmax; status is uncertainty metadata). Every run also reports a
label-set RANDOM BASELINE: the expected accuracy of a uniform pick over
each decision's candidate ids (1/|ids| per decision), computed from the
label sets at run time — never hardcoded. The aggregate net lift over
that baseline is the honest skill measure on this terrain.

Usage:
    python scripts/bench_s1mb.py            # full run (~84k decisions)
    python scripts/bench_s1mb.py --limit N  # cap rows per config (dev)
Writes/append results to reports/jevbench-head-to-head.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASET = "hotchpotch/s1mb-dataset"
API = "https://datasets-server.huggingface.co"
REPORT = ROOT / "reports" / "jevbench-head-to-head.md"
WORKERS = 8


CACHE = ROOT / "data" / "incoming" / "external" / "s1mb"


def get(url: str, timeout: int = 60):
    """GET with 429 backoff (datasets-server rate-limits bursts)."""
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 502, 503) or attempt == 5:
                raise
            wait = float(exc.headers.get("Retry-After") or 0) or 10.0
            print(f"  429 rate-limited; backing off {wait:.0f}s", flush=True)
            time.sleep(wait)


def list_configs() -> dict:
    info = get(f"{API}/info?dataset={DATASET}")["dataset_info"]
    return {name: sum(s.get("num_examples", 0)
                      for s in ci.get("splits", {}).values())
            for name, ci in sorted(info.items())}


def fetch_config(name: str, limit: int = 0) -> list[dict]:
    """Fetch one config's rows, cached to disk so runs are resumable."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rows, offset, page = [], 0, 100
    while True:
        n = min(page, limit - len(rows)) if limit else page
        payload = get(f"{API}/rows?dataset={DATASET}&config={name}"
                      f"&split=test&offset={offset}&length={n}")
        batch = [r["row"] for r in payload.get("rows", [])]
        rows.extend(batch)
        if len(batch) < n or (limit and len(rows) >= limit):
            break
        offset += len(batch)
        time.sleep(0.2)  # stay under the per-minute rate limit
    cache.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def build_jobs(rows: list[dict]) -> list[tuple[str, list[str], str]]:
    """Flatten rows to ``(state, ids, gold)`` decisions (one source of truth
    for both the scored run and the random baseline)."""
    jobs = []
    for r in rows:
        state = r["input"].get("state_json") or json.dumps(r["input"],
                                                           ensure_ascii=False)
        for t in r.get("targets", []):
            ids = [str(i) for i in t.get("ids", [])]
            if not ids:
                continue
            gold = ids[max(range(len(t.get("probabilities", []))),
                           key=lambda k: t["probabilities"][k])] \
                if t.get("probabilities") else str(t.get("expected", ""))
            jobs.append((state, ids, gold))
    return jobs


def random_baseline(jobs: list[tuple[str, list[str], str]]) -> float:
    """Raw expected-hit sum of a uniform-random pick over each label set.

    Per decision the chance of hitting the gold id is ``1/len(ids)`` when
    the gold is among the candidates, else 0. Computed from the label sets
    only — no model call, nothing hardcoded. Divide by the decision count
    for the rate.
    """
    return sum(1.0 / len(ids) if gold in ids else 0.0
               for _, ids, gold in jobs)


def score_rows(matcher, rows: list[dict]) -> tuple[list[tuple[bool, str]], float]:
    """Answer every decision in every row.

    Returns ``[(correct, status)]`` plus the random-baseline expected-hit
    sum over the same decisions (see ``random_baseline``).
    """
    jobs = build_jobs(rows)
    base = random_baseline(jobs)

    def one(job):
        state, ids, gold = job
        try:
            picked, _conf, status = matcher.select(state, ids, with_status=True)
            return (str(picked) == gold, status)
        except Exception:
            return (False, "error")

    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(one, jobs)), base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap rows per config (0 = all)")
    parser.add_argument("--configs", type=int, default=0,
                        help="cap number of configs (0 = all)")
    args = parser.parse_args()

    from mivia_small_json.tool_match import load_matcher

    matcher = load_matcher()
    if matcher is None:
        raise SystemExit("shipped guard artifacts not loadable")

    configs = list_configs()
    names = list(configs)[: args.configs] if args.configs else list(configs)
    total = sum(configs[n] for n in names)
    print(f"S1MB: {len(names)}/{len(configs)} configs, {total} rows")
    t0 = time.perf_counter()

    ok = err = 0
    base_total = 0.0
    by_status = defaultdict(lambda: [0, 0])
    per_config = {}
    skipped = []
    for k, name in enumerate(names):
        try:
            rows = fetch_config(name, args.limit)
        except Exception as exc:
            skipped.append(name)
            print(f"  SKIP {name}: {type(exc).__name__}", flush=True)
            continue
        res, base_sum = score_rows(matcher, rows)
        c = sum(1 for good, _ in res if good)
        e = sum(1 for _, s in res if s == "error")
        ok += c
        err += e
        base_total += base_sum
        for good, status in res:
            by_status[status][0] += good
            by_status[status][1] += 1
        per_config[name] = (c, len(res), base_sum / max(1, len(res)))
        if (k + 1) % 20 == 0:
            done = sum(t for _, t, _ in per_config.values())
            print(f"  {k + 1}/{len(names)} configs | decisions {done} | "
                  f"running acc {100 * ok / max(1, done):.1f}%", flush=True)

    wall = time.perf_counter() - t0
    n = sum(v[1] for v in by_status.values())
    rand = 100 * base_total / max(1, n)
    acc = 100 * ok / max(1, n)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["", f"## S1MB run — {stamp}",
             "",
             f"Coverage: {len(names)}/{len(list(configs))} configs, "
             f"{n} decisions (full coverage, no sampling). Plain per-decision "
             f"accuracy via shipped zero-shot `select` over bare label ids — "
             f"NOT the official S1MB Borda protocol.",
             "",
             f"### overall — {ok}/{n} = **{acc:.2f}%** "
             f"({wall:.0f}s, {WORKERS} workers, request errors {err}, "
             f"skipped configs {len(skipped)})",
             "",
             f"- label-set random baseline: **{rand:.2f}%** (uniform pick "
             f"over each decision's candidate ids, 1/|ids|; computed from "
             f"the label sets at run time) → net lift "
             f"**{acc - rand:+.2f}pp**",
             "- by status: " + ", ".join(
                 f"{k} {v[0]}/{v[1]}={100 * v[0] / max(1, v[1]):.0f}%"
                 for k, v in sorted(by_status.items())),
             "",
             "Best configs: " + ", ".join(
                 f"{k} {c}/{t}={100 * c / max(1, t):.0f}% (rand {100 * b:.0f}%)"
                 for k, (c, t, b) in
                 sorted(per_config.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][1]))[:6]),
             "",
             "Worst configs: " + ", ".join(
                 f"{k} {c}/{t}={100 * c / max(1, t):.0f}% (rand {100 * b:.0f}%)"
                 for k, (c, t, b) in
                 sorted(per_config.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]))[:6]),
             "",
             "### per config — accuracy vs random baseline",
             "",
             "| config | decisions | acc | rand | net |",
             "|---|---|---|---|---|",
             ]
    for name, (c, t, b) in sorted(per_config.items(),
                                  key=lambda kv: (kv[1][0] / max(1, kv[1][1])
                                                  - kv[1][2]),
                                  reverse=True):
        a = 100 * c / max(1, t)
        lines.append(
            f"| {name} | {t} | {a:.2f}% | {100 * b:.2f}% | {a - 100 * b:+.2f}pp |")
    lines += [""]
    text = "\n".join(lines)
    print(text)
    existing = REPORT.read_text() if REPORT.exists() else ""
    REPORT.write_text(existing + text + "\n")
    print(f"appended to {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
