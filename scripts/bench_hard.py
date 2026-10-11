#!/usr/bin/env python3
"""Harder adversarial evaluation: transform the frozen split and score it.

The plain held-out set is easy (the pipeline scores 99.7-100%). This builds
deterministic hard variants of the committed held-out rows and reports
routed-and-valid per transform plus an 8x8 confusion matrix, so distribution
and boundary failures become visible.

Transforms (seeded, reproducible):
- ``clean``   the row as committed
- ``typo``    character noise (swap/drop/insert) on ~5% of characters
- ``wrap``    ANSI codes, a zero-width space and emoji around the row
- ``confuse`` schema-boundary edits (issue made PR-like, PR stripped of its marker)
- ``truncate`` the first third of the row

Writes reports/benchmark-hard.md. Deterministic, offline, CPU-only.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json import bootstrap  # noqa: E402
from mivia_small_json.model import MODEL_PATH  # noqa: E402
from mivia_small_json.pipeline import classify, load_router  # noqa: E402
from mivia_small_json.validate import SCHEMAS  # noqa: E402

REPORT = ROOT / "reports" / "benchmark-hard.md"
SEED = 42
ANSI = "\x1b[31m"
ZW = "\u200b"
EMOJI = "\U0001f525"


def typo(text: str, schema: str, rng: random.Random) -> str:
    chars = list(text)
    for _ in range(max(1, len(chars) // 20)):
        i = rng.randrange(len(chars))
        op = rng.choice(("swap", "drop", "insert"))
        if op == "swap" and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop":
            chars[i] = ""
        else:
            chars.insert(i, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    return "".join(chars)


def wrap(text: str, rng: random.Random) -> str:
    return f"{ANSI}{text}{ZW} {EMOJI}"


def confuse(text: str, schema: str, rng: random.Random) -> str:
    if schema == "issue":
        return "PR #999 merged +10 -2 " + text
    if schema == "pull_request":
        return re.sub(r"(?i)\bPR\s*#\d+\b|\bpull request\b", "", text)
    return text


def truncate(text: str, schema: str, rng: random.Random) -> str:
    return text[: max(10, len(text) // 3)]


TRANSFORMS = {
    "clean": lambda t, s, r: t,
    "typo": typo,
    "wrap": lambda t, s, r: wrap(t, r),
    "confuse": confuse,
    "truncate": truncate,
}


def _fingerprint() -> str:
    import importlib.metadata as im
    import platform
    import subprocess

    def v(name):
        try:
            return im.version(name)
        except Exception:
            return "?"

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True
                             ).stdout.strip()[:12]
    except Exception:
        sha = "unknown"
    return (f"python {platform.python_version()} · scikit-learn {v('scikit-learn')} · "
            f"numpy {v('numpy')} · skops {v('skops')} · git {sha}")


def _confusion_block(title: str, conf: Counter) -> list[str]:
    lines = ["", f"## Confusion matrix (routed type vs true schema), {title}", "",
             "| true \\ pred | " + " | ".join(SCHEMAS) + " |",
             "|---|" + "---|" * len(SCHEMAS)]
    for true_schema in SCHEMAS:
        cells = [str(conf.get((true_schema, p), 0)) for p in SCHEMAS]
        lines.append(f"| {true_schema} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    bootstrap.register_example_schemas()
    router = load_router(MODEL_PATH)
    rows = [json.loads(line) for line in
            (ROOT / "data" / "heldout" / "test.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]

    results: dict[str, dict] = {}
    confusion: dict[str, Counter] = {}
    for name, fn in TRANSFORMS.items():
        rng = random.Random(SEED)
        strict = valid = 0
        conf: Counter = Counter()
        for row in rows:
            text = fn(row["raw"], row["schema"], rng)
            obj, meta = classify(text, router)
            pred = obj.get("type")
            valid += bool(meta["valid"])
            strict += bool(meta["valid"]) and pred == row["schema"]
            conf[(row["schema"], pred)] += 1
        results[name] = {"valid_pct": round(100 * valid / len(rows), 2),
                         "routed_and_valid_pct": round(100 * strict / len(rows), 2)}
        confusion[name] = conf

    lines = [
        "# Harder adversarial benchmark",
        "",
        "Deterministic hard variants of the frozen held-out split. Reveals",
        "distribution and schema-boundary failures the clean set cannot.",
        "",
        f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d')} · "
        "script: `scripts/bench_hard.py` · seed 42",
        "",
        f"- env: {_fingerprint()}",
        "",
        "| Transform | valid % | routed-and-valid % |",
        "|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(f"| {name} | {m['valid_pct']} | {m['routed_and_valid_pct']} |")
    lines += _confusion_block("clean", confusion["clean"])
    lines += _confusion_block("under `confuse`", confusion["confuse"])
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    for name, m in results.items():
        print(f"{name}: {m}")
    print(f"report written to {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
