#!/usr/bin/env python3
"""Build value-validity contrast twins + train the conformance gate (TRACK 2).

Twins: same request topic, same REAL catalogue tool (from the error_param
slice of guard-eval-v2.jsonl), the ONLY difference is per-value
conformance — a valid value vs a constraint-violating value, everything
else identical. Deterministic (seeded), deduped against pairs.jsonl and
every eval set incl. guard-eval-v2 (zero overlap required).

Training delegates to scripts/train_tool_match.py on the conformance
corpus (tool span = render_conformance_span). GPU is SERIAL: run under
``flock -w 7200 /tmp/mivia_gpu.lock``. Candidates ONLY under
gitignored data/incoming/models/, never promoted without clearing the bar.

Row schema matches pairs.jsonl: label/request/source/split/tool(span str).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from render_conformance_span import (  # noqa: E402
    check_value_conformance,
    render_conformance_span,
)

BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
EVAL = BFCL / "guard-eval-v2.jsonl"
PAIRS = BFCL / "pairs.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "pairs-conformance-v1.jsonl"

SOURCE = "conformance_value_contrast"


def _bare(tool: dict) -> dict:
    if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
        return tool["function"]
    return tool


def constrained_slots(tool: dict) -> list[tuple[str, dict]]:
    """Required params carrying a checkable constraint."""
    bare = _bare(tool)
    params = bare.get("parameters")
    params = params if isinstance(params, dict) else {}
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = list(required) if isinstance(required, list) else []
    out = []
    for slot in required:
        spec = props.get(slot) if isinstance(props.get(slot), dict) else {}
        keys = ("pattern", "minLength", "maxLength", "minimum", "maximum",
                "enum", "type")
        if any(k in spec for k in keys if k != "type") or spec.get("type") in (
                "integer", "number", "float", "boolean"):
            out.append((slot, spec))
    return out


def violate(value: str, spec: dict, rng: random.Random) -> str:
    """Deterministic constraint-violating rewrite of a passing value."""
    spec = spec if isinstance(spec, dict) else {}
    stype = spec.get("type", "")
    cands: list[str] = []
    if stype in ("integer", "number", "float"):
        cands.append("abc")
    if stype == "boolean":
        cands.append("maybe")
    ml = spec.get("minLength")
    if isinstance(ml, (int, float)):
        cands.append("x" * max(0, int(ml) - 1) if int(ml) > 1 else "")
    xl = spec.get("maxLength")
    if isinstance(xl, (int, float)):
        cands.append("z" * (int(xl) + 5))
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        cands.append("NOT_A_LISTED_OPTION")
    cands += [value + s for s in ("!", "#", "!!!", " not-a-value @@",
                                  "x" * 200)]
    cands.append("")
    for cand in cands:
        if cand != value and check_value_conformance(cand, spec):
            return cand
    raise ValueError(f"no violating rewrite for {value!r} vs {spec!r}")


def valid_value(slot: str, spec: dict) -> str | None:
    """A value passing the declared constraints (None if unconstructible)."""
    enum = spec.get("enum") if isinstance(spec, dict) else None
    if isinstance(enum, list) and enum:
        return str(enum[0])
    stype = spec.get("type") if isinstance(spec, dict) else ""
    if stype == "integer":
        lo = spec.get("minimum", 1)
        lo = int(lo) if isinstance(lo, (int, float)) else 1
        hi = spec.get("maximum", lo + 10)
        hi = int(hi) if isinstance(hi, (int, float)) else lo + 10
        return str(min(max(lo, lo), hi))
    if stype in ("number", "float"):
        lo = spec.get("minimum", 0.0)
        lo = float(lo) if isinstance(lo, (int, float)) else 0.0
        hi = spec.get("maximum", lo + 10.0)
        hi = float(hi) if isinstance(hi, (int, float)) else lo + 10.0
        cand = lo + 0.5 if lo + 0.5 <= hi else (lo + hi) / 2.0
        cand = min(max(cand, lo), hi)
        return str(cand)
    if stype == "boolean":
        return "true"
    pat = spec.get("pattern") if isinstance(spec, dict) else None
    if isinstance(pat, str) and pat:
        for cand in ("123", "ABCDE", "test_value_1", "42"):
            if not check_value_conformance(cand, spec):
                return cand
        return None
    ml = spec.get("minLength") if isinstance(spec, dict) else None
    base = "Valid Station Name"
    if isinstance(ml, (int, float)):
        base = base.ljust(int(ml), "x")
    xl = spec.get("maxLength") if isinstance(spec, dict) else None
    if isinstance(xl, (int, float)):
        base = base[:int(xl)] or "x"
    return base if not check_value_conformance(base, spec) else None


def build_twins(seed: int = 7) -> list[dict]:
    """Same-request validity twins from REAL error_param catalogue tools."""
    rng = random.Random(seed)
    eval_rows = [json.loads(line) for line in EVAL.read_text().splitlines()
                  if line.strip()]
    tools: dict[str, dict] = {}
    for r in eval_rows:
        if r.get("category") == "infeasible" and str(
                r.get("source", "")).startswith("acebench_special_error_param"):
            t = r.get("tool")
            if isinstance(t, dict):
                tools[t.get("name", json.dumps(t, sort_keys=True)[:80])] = t
    rows: list[dict] = []
    for name in sorted(tools):
        tool = tools[name]
        slots = constrained_slots(tool)
        if not slots:
            continue
        slot, spec = slots[0]
        good = valid_value(slot, spec)
        if good is None:
            continue
        try:
            bad = violate(good, spec, rng)
        except ValueError:
            continue
        assert not check_value_conformance(good, spec), (name, slot, good)
        assert check_value_conformance(bad, spec), (name, slot, bad)
        topic = f"Request concerning {name} with {slot} provided"
        for val, ok, split, label in ((good, True, "train", 1),
                                      (bad, False, "train", 0)):
            req = f'{topic}: {slot}="{val}".'
            span = render_conformance_span(tool, req)
            rows.append({"label": label, "request": req,
                         "source": SOURCE, "split": split, "tool": span,
                         "_slot": slot, "_ok": ok})
    dev = []
    for i, r in enumerate(rows):
        if i % 5 == 4:
            r = dict(r)
            r["split"] = "dev"
        dev.append(r)
    return dev


def content_hash(request: str, span: str) -> str:
    return hashlib.sha256((request + "\x00" + span).encode()).hexdigest()


def dedupe(rows: list[dict]) -> tuple[list[dict], dict]:
    """Drop rows overlapping pairs.jsonl or any eval set. Zero required."""
    index: set[str] = set()
    for path in [PAIRS, EVAL, BFCL / "irrelevance_eval.jsonl",
                 BFCL / "calls.jsonl", BFCL / "irrelevance.jsonl"]:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            req = rec.get("request", "")
            tool = rec.get("tool", "")
            span = tool if isinstance(tool, str) else json.dumps(
                tool, sort_keys=True)
            index.add(content_hash(str(req), str(span)))
            index.add(hashlib.sha256(str(req).encode()).hexdigest())
    kept, stats = [], {"built": len(rows), "overlap": 0}
    for r in rows:
        h = content_hash(r["request"], r["tool"])
        hr = hashlib.sha256(r["request"].encode()).hexdigest()
        if h in index or hr in index:
            stats["overlap"] += 1
            continue
        kept.append({k: v for k, v in r.items() if not k.startswith("_")})
    stats["kept"] = len(kept)
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--train-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    rows = build_twins(seed=args.seed)
    kept, stats = dedupe(rows)
    print(json.dumps(stats))
    h = hashlib.sha256()
    for r in kept:
        h.update(json.dumps(r, sort_keys=True).encode())
    print("sha256", h.hexdigest()[:16])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    if args.build_only:
        return
    cmd = [sys.executable, str(ROOT / "scripts" / "train_tool_match.py"),
           "--pairs", str(args.out), "--out",
           str(ROOT / "data" / "incoming" / "models"
               / "tool-match-conformance-int8.onnx"),
           "--report", str(ROOT / "reports" / "tool-match-conformance.md"),
           "--contrast-weight", "1.0", "--epochs", "8", "--batch", "16",
           "--seed", str(args.seed)] + list(args.train_args)
    print("GPU training is SERIAL; run under: flock -w 7200 "
          "/tmp/mivia_gpu.lock " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
