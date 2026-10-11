#!/usr/bin/env python3
"""SLM grammar tool-call PoC — registered arm ``slm-grammar-toolcall``.

Spec: slm-grammar-toolcall (registered 2026-10-10,
BEFORE any measurement; bars G1-G6 frozen there). This is the registered
fallback of the falsified unified-span spike (commits 47fcbe2, f97fcc4).

Scope: measure whether Qwen2.5-0.5B-Instruct (int4 GGUF Q4_K_M) with GBNF
grammar-constrained decoding can replace ONLY the string-slot argument-
filling stage: (tool schema + request) -> arguments JSON. Routing and the
tool-match guard stay mivia's shipped models and are NOT exercised here.

PoC-scoped: this script is NOT imported by src/, the dependency is NOT in
pyproject.toml, and nothing here promotes or writes outside .scratch/.

Subcommands:
  download   fetch the pinned GGUF into the HF cache, print its sha256
  calls      the 183-row BFCL calls set -> .scratch/slm-poc/calls-verdict.json
             (G1 complete-fill, G2 invented rate, G5 strict JSON, G6 enum;
             envelope VmRSS/VmHWM + per-fill latency -> G4)
  dev        the frozen 1,402-row spanfill dev split (REPORTED context vs
             the pairwise baseline 0.9165/0.0441; G3 = the 117-row
             multi-word slice) -> .scratch/slm-poc/dev-verdict.json

Every instrument names the model GGUF sha256 + the corpus sha256(s) it
scored (rule 10). No timing asserts anywhere; latency is measured, printed,
and recorded, never asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from mivia_small_json.tool_match import parse_catalogue_raw  # noqa: E402

CALLS = ROOT / "data/incoming/external/bfcl/calls.jsonl"
DEV = ROOT / "data/real/spanfill/dev.jsonl"
OUT = ROOT / ".scratch/slm-poc"

# Registered constants (spec: Implementation Decisions) --------------------
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEP_PIN = "llama-cpp-python==0.3.36"
SEED = 42
N_THREADS = 12          # physical cores of the 12C/24T AMD Ryzen 9 9900X
N_CTX = 1024            # prompts are ~150-250 tokens; outputs capped 256
MAX_TOKENS = 256
TEMPERATURE = 0.0       # greedy
WARMUP_FILLS = 5

SYSTEM_PROMPT = (
    "You fill tool-call arguments. Copy every argument value verbatim from "
    "the user request. Never invent, paraphrase, or reformat a value. "
    "Output only the JSON object."
)

# The G2 wrap-strip construct: the exact span_fill._wrap_strip construct
# (leading strip removes ONLY quotes - a leading dot is content, ".env" is
# a filename; trailing strip keeps the full punctuation set). Copied as the
# frozen construct, not imported, because span_fill.py is under parallel
# review and this PoC's checker bytes must be pinned by its own test.
_WRAP_TRAILING = " .,;:!?)]}\"'"


def _wrap_strip(raw: str) -> str:
    return raw.lstrip("\"'").rstrip(_WRAP_TRAILING)


# --- GBNF grammar builder (spec: the registered GBNF contract) ------------

_GBNF_HEADER = (
    'ws ::= [ \\t\\n]*',
    'str ::= "\\"" strchar* "\\""',
    'strchar ::= [^"\\\\\\x00-\\x1f] | "\\\\" (["\\\\/bfnrt] | "u" '
    '[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])',
    'int ::= "-"? ("0" | [1-9] [0-9]*)',
    'num ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? '
    '(("e" | "E") ("-" | "+")? [0-9]+)?',
)


def _rule_name(key: str) -> str:
    # llama.cpp GBNF rule names allow [a-zA-Z0-9-] only — no underscores
    safe = re.sub(r"[^A-Za-z0-9-]", lambda m: f"x{ord(m.group(0)):x}", key)
    return "val-" + safe


def _gbnf_lit(text: str) -> str:
    """A GBNF string literal matching `\"text\"` — the double-quoted JSON
    form, so the literal's quote characters are part of the matched bytes
    (GBNF-escaped as \\")."""
    inner = json.dumps(text, ensure_ascii=False)[1:-1]
    return '"\\"' + inner + '\\""'


def gbnf_for_tool(tool: dict) -> str:
    """A GBNF grammar forcing a JSON object with EXACTLY the required
    keys, in schema order, values typed per the declared schema (spec)."""
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    required = params.get("required") or []
    lines = list(_GBNF_HEADER)
    members = []
    for key in required:
        prop = props.get(key) or {"type": "string"}
        enum = prop.get("enum")
        t = str(prop.get("type", "string")).lower()
        key_lit = _gbnf_lit(key)  # "\"key\"" - quotes are part of the JSON
        if enum:
            alts = " | ".join(_gbnf_lit(str(v)) for v in enum)
            rname = _rule_name(key)
            lines.append(f"{rname} ::= {alts}")
            ref = rname
        elif t == "integer":
            ref = "int"
        elif t in ("number", "float"):
            ref = "num"
        elif t == "boolean":
            ref = _rule_name(key) + "-b"
            lines.append(f'{ref} ::= "true" | "false"')
        elif t == "array":
            ref = _rule_name(key) + "-a"
            lines.append(f'{ref} ::= "[" ws (str ("," ws str)*)? ws "]"')
        else:
            ref = "str"
        members.append(f"{key_lit} ws \":\" ws {ref}")
    seq = ' ws "," ws '.join(members)
    root = '"{" ws "}"' if not members else '"{" ws ' + seq + ' ws "}"'
    lines.append("root ::= " + root)
    return "\n".join(lines)


# --- prompt rendering (spec: the FIXED prompt construct) -------------------

def render_user_message(tool: dict, request: str) -> str:
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    required = params.get("required") or []
    lines = []
    for key in required:
        prop = props.get(key) or {}
        t = str(prop.get("type", "string"))
        line = f"- {key} ({t})"
        if prop.get("enum"):
            line += " one of: " + ", ".join(str(v) for v in prop["enum"])
        lines.append(line)
    return (
        f"Tool: {tool.get('name', '')}\n"
        f"Required parameters:\n" + ("\n".join(lines) if lines else "(none)")
        + f"\nRequest: {request}"
    )


# --- the G2 verbatim checker (mechanical; pinned by the focused test) ------

def _num_text(value) -> str:
    """JSON string form of a numeric/boolean emitted value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def value_is_verbatim(value, prop: dict, request: str) -> bool:
    """The G2 construct: a value is verbatim iff its string form, after the
    span arm's wrap-strip, is an exact substring of the request text.
    Arrays: every element must pass. (Spec: Implementation Decisions.)"""
    if isinstance(value, list):
        return all(value_is_verbatim(v, prop, request) for v in value)
    if value is None:
        return False
    if isinstance(value, str):
        stripped = _wrap_strip(value)
        if not stripped.strip():
            # an empty copy is not a copy: no verbatim bytes were carried,
            # so an empty (or whitespace-only) value counts as invented
            return False
        return stripped in request
    # numerics / booleans: JSON string form, case-insensitive substring
    return _num_text(value).lower() in request.lower()


def strict_parse(text: str):
    """G5 construct: json.loads refusing NaN/Infinity (parse_constant
    raises); returns (obj, None) or (None, reason)."""
    def _reject(x):
        raise ValueError(f"non-finite constant {x!r}")

    try:
        obj = json.loads(text, parse_constant=_reject)
    except Exception as exc:  # noqa: BLE001 - the reason is recorded
        return None, f"parse: {exc}"
    if not isinstance(obj, dict):
        return None, "top-level is not a JSON object"
    return obj, None


# --- enums (G6) --------------------------------------------------------------

def enum_violation(value, prop: dict) -> bool:
    """True when an enum slot's emitted value is not a declared member."""
    enum = prop.get("enum")
    if not enum:
        return False
    if isinstance(value, list):
        return any(enum_violation(v, prop) for v in value)
    return not any(str(value) == str(v) for v in enum)


def _value_eq(emitted, gold, prop: dict) -> bool:
    """Exact-equality construct for the dev gold comparison (strings after
    the same wrap-strip; enums exact member; integers parsed-equal)."""
    if isinstance(gold, str):
        if isinstance(emitted, str):
            return _wrap_strip(emitted) == _wrap_strip(gold)
        return False
    if isinstance(gold, bool):
        return isinstance(emitted, bool) and emitted == gold
    if isinstance(gold, int):
        if isinstance(emitted, bool):
            return False
        if isinstance(emitted, int):
            return emitted == gold
        if isinstance(emitted, float) and emitted.is_integer():
            return int(emitted) == gold
        if isinstance(emitted, str):
            d = "".join(ch for ch in emitted if ch.isdigit() or ch == "-")
            return d.lstrip("-").isdigit() and int(d) == gold
        return False
    return emitted == gold


# --- model handle -------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_model_path(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"model file not found: {p}")
        return p
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(MODEL_REPO, MODEL_FILE))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"model download failed ({exc}); run the download subcommand "
            "first or pass --model"
        )


class _VmRSS:
    def __init__(self) -> None:
        self.samples: list[int] = []

    def sample(self) -> int:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                self.samples.append(kb)
                return kb
        return 0

    def hwm_kb(self) -> int:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
        return 0

    def peak_mb(self) -> float:
        hwm = self.hwm_kb()
        sampled = max(self.samples) if self.samples else 0
        return round(max(hwm, sampled) / 1024.0, 1)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
    return ordered[idx]


def build_engine(model_path: Path):
    import llama_cpp

    return llama_cpp.Llama(
        model_path=str(model_path),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_threads_batch=N_THREADS,
        seed=SEED,
        verbose=False,
    )


def grammar_for(grammar_cache: dict, tool: dict, tool_key: str):
    import llama_cpp

    if tool_key not in grammar_cache:
        grammar_cache[tool_key] = llama_cpp.LlamaGrammar.from_string(
            gbnf_for_tool(tool), verbose=False
        )
    return grammar_cache[tool_key]


def fill(llm, grammar, tool: dict, request: str) -> tuple[str, float]:
    """One fill; returns (raw text, wall seconds)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_message(tool, request)},
    ]
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=messages,
        grammar=grammar,
        temperature=TEMPERATURE,
        seed=SEED,
        max_tokens=MAX_TOKENS,
    )
    dt = time.perf_counter() - t0
    text = out["choices"][0]["message"]["content"] or ""
    return text, dt


# --- instruments ----------------------------------------------------------------

def _common_meta(model_path: Path, corpora: dict[str, Path]) -> dict:
    import llama_cpp
    return {
        "arm": "slm-grammar-toolcall",
        "spec": "slm-grammar-toolcall",
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "model_sha256": sha256_file(model_path),
        "model_bytes": model_path.stat().st_size,
        "dep_pin": DEP_PIN,
        "dep_version": llama_cpp.__version__,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "n_threads": N_THREADS,
        "n_ctx": N_CTX,
        "max_tokens": MAX_TOKENS,
        "system_prompt": SYSTEM_PROMPT,
        "corpora": {k: {"path": str(v), "sha256": sha256_text_file(v)}
                    for k, v in corpora.items()},
    }


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_calls(model_path: Path) -> dict:
    rows = _rows(CALLS)
    llm = build_engine(model_path)
    grammar_cache: dict = {}
    vm = _VmRSS()
    latencies: list[float] = []

    # G4: 5 warmup fills (warm cache), their latency discarded, their RSS
    # samples discarded (model-load allocations are still covered by VmHWM,
    # which is the whole-process high-water mark and reported).
    first_tools, first_request = parse_catalogue_raw(rows[0]["text"])
    warm_tool = (first_tools or [{}])[0]
    warm_g = grammar_for(grammar_cache, warm_tool, "warmup")
    for _ in range(WARMUP_FILLS):
        fill(llm, warm_g, warm_tool, first_request)
    vm.samples.clear()

    fills: list[dict] = []
    for i, row in enumerate(rows):
        if row.get("label") != "tool_call":
            raise SystemExit(f"row {i}: unexpected label {row.get('label')!r}")
        tools, request = parse_catalogue_raw(row["text"])
        if not tools:
            raise SystemExit(
                f"row {i}: catalogue did not parse (window-logic drift? "
                "parse_catalogue_raw is the one split for every caller)")
        row_record = {"row": i, "n_tools": len(tools), "fills": [],
                      "complete_any": False}
        for t_idx, tool in enumerate(tools):
            params = tool.get("parameters") or {}
            props = params.get("properties") or {}
            required = params.get("required") or []
            g = grammar_for(grammar_cache, tool, f"r{i}t{t_idx}")
            text, dt = fill(llm, g, tool, request)
            latencies.append(dt)
            rss_kb = vm.sample()
            obj, err = strict_parse(text)
            frec = {"row": i, "tool": tool.get("name"),
                    "required": required, "latency_s": round(dt, 4),
                    "rss_mb": round(rss_kb / 1024.0, 1), "raw": text,
                    "parse_error": err, "keys_exact": False}
            if obj is not None:
                frec["keys_exact"] = set(obj) == set(required)
                values = []
                for k, v in obj.items():
                    prop = props.get(k) or {}
                    values.append({
                        "slot": k, "value": v,
                        "verbatim": value_is_verbatim(v, prop, request),
                        "enum_violation": enum_violation(v, prop),
                        "slot_is_enum": bool(prop.get("enum")),
                    })
                frec["values"] = values
                if frec["keys_exact"]:
                    row_record["complete_any"] = True
            row_record["fills"].append(frec)
        fills.append(row_record)

    # ---- aggregate the bars (read back from these records, rule 15) ------
    complete_rows = sum(1 for r in fills if r["complete_any"])
    single = [r for r in fills if r["n_tools"] == 1]
    single_complete = sum(1 for r in single if r["complete_any"])
    all_fills = [f for r in fills for f in r["fills"]]
    parsed = [f for f in all_fills if f["parse_error"] is None]
    keys_exact = [f for f in parsed if f["keys_exact"]]
    values = [v for f in parsed for v in f.get("values", [])]
    invented = [v for v in values if not v["verbatim"]]
    enum_touched = [v for v in values if v["slot_is_enum"]]
    enum_bad = [v for v in enum_touched if v["enum_violation"]]

    g1_rate = complete_rows / len(rows)
    g2_rate = len(invented) / len(values) if values else None
    g4_peak = vm.peak_mb()
    g4_p95 = _percentile(latencies, 0.95) * 1000
    payload = _common_meta(model_path, {"calls": CALLS})
    payload.update({
        "set": "bfcl_calls",
        "rows": len(rows),
        "fills": len(all_fills),
        "G1_routed_to_expected": {
            "construct": "complete fill: emitted key set == required set "
                         "exactly, any listed tool (union bound); "
                         "single-tool subset reported alongside",
            "complete_rows": complete_rows,
            "rate_pct": round(100.0 * g1_rate, 4),
            "floor_pct": 89.0,
            "pass": g1_rate >= 0.89,
            "single_tool_rows": len(single),
            "single_tool_complete": single_complete,
            "single_tool_rate_pct":
                round(100.0 * single_complete / len(single), 4)
                if single else None,
        },
        "G2_honest_values": {
            "construct": "verbatim checker: wrap-strip then exact "
                         "substring of the request (the "
                         "span_fill._wrap_strip construct); "
                         "invented = non-verbatim emitted",
            "emitted_values": len(values),
            "verbatim_values": len(values) - len(invented),
            "invented_values": len(invented),
            "invented_rate": round(g2_rate, 4) if g2_rate is not None
            else None,
            "ceiling": 0.10,
            "pass": g2_rate is not None and g2_rate <= 0.10,
            "invented_examples": [
                {"slot": v["slot"], "value": v["value"],
                 "row": f["row"], "tool": f["tool"]}
                for v, f in ((v, f) for f in parsed
                             for v in f.get("values", [])
                             if not v["verbatim"])][:20],
        },
        "G4_envelope": {
            "construct": "VmHWM/VmRSS from /proc/self/status (never "
                         f"ru_maxrss); warm cache after {WARMUP_FILLS} "
                         f"warmup fills; CPU only; n_threads={N_THREADS} "
                         "(physical cores)",
            "peak_rss_mb": g4_peak,
            "rss_ceiling_mb": 600.0,
            "p50_ms": round(_percentile(latencies, 0.50) * 1000, 2),
            "p95_ms": round(g4_p95, 2),
            "mean_ms": round(statistics.fmean(latencies) * 1000, 2)
            if latencies else None,
            "latency_ceiling_p95_ms": 400.0,
            "cpu": "AMD Ryzen 9 9900X 12-Core (12C/24T)",
            "pass": g4_peak <= 600.0 and g4_p95 <= 400.0,
        },
        "G5_schema_validity": {
            "construct": "strict json.loads (parse_constant refuses "
                         "NaN/Infinity) + top-level object + key set == "
                         "required exactly",
            "fills": len(all_fills),
            "strict_parse_ok": len(parsed),
            "keys_exact": len(keys_exact),
            "rate_pct": round(100.0 * len(keys_exact) / len(all_fills), 4)
            if all_fills else None,
            "floor_pct": 100.0,
            "pass": bool(all_fills) and len(keys_exact) == len(all_fills),
        },
        "G6_enum_strictness": {
            "construct": "emitted enum-slot value must equal a declared "
                         "member exactly (calls fills here; the dev "
                         "instrument carries the dev count)",
            "enum_slots_touched": len(enum_touched),
            "enum_violations": len(enum_bad),
            "ceiling": 0,
            "pass": len(enum_bad) == 0,
        },
        "misses_G1": [
            {"row": r["row"], "n_tools": r["n_tools"],
             "tool_names": [f["tool"] for f in r["fills"]],
             "parse_errors": [f["parse_error"] for f in r["fills"]],
             "keys_exact_per_fill": [f["keys_exact"] for f in r["fills"]]}
            for r in fills if not r["complete_any"]
        ],
        "latencies_s": [round(x, 4) for x in latencies],
    })
    dest = OUT / "calls-verdict.json"
    dest.write_text(json.dumps(payload, indent=1) + "\n")
    summary = {k: v for k, v in payload.items()
               if k.startswith(("G1", "G2", "G4", "G5", "G6"))}
    print(json.dumps(summary, indent=1))
    print(f"wrote {dest}")
    return payload


def run_dev(model_path: Path) -> dict:
    rows = _rows(DEV)
    llm = build_engine(model_path)
    grammar_cache: dict = {}

    warm = rows[0]
    warm_g = grammar_for(grammar_cache, warm["tool"], "warmup")
    for _ in range(WARMUP_FILLS):
        fill(llm, warm_g, warm["tool"], warm["request"])

    fills: list[dict] = []
    for i, row in enumerate(rows):
        tool = row["tool"]
        request = row["request"]
        gold = row["gold"]
        props = ((tool.get("parameters") or {})
                 .get("properties")) or {}
        g = grammar_for(grammar_cache, tool, f"d{i}")
        text, _dt = fill(llm, g, tool, request)
        obj, err = strict_parse(text)
        rec = {"row": i, "parse_error": err,
               "multi_word": any(isinstance(v, str) and " " in v.strip()
                                 for v in gold.values()),
               "enum_violation": False, "exact": False,
               "emitted_count": 0, "emitted_wrong": 0}
        if obj is not None:
            exact = set(obj) == set(gold)
            wrong = 0
            for k, v in obj.items():
                prop = props.get(k) or {}
                if enum_violation(v, prop):
                    rec["enum_violation"] = True
                if k not in gold or not _value_eq(v, gold[k], prop):
                    wrong += 1
            exact = exact and wrong == 0
            rec.update({"exact": exact, "emitted_count": len(obj),
                        "emitted_wrong": wrong})
        fills.append(rec)

    n = len(rows)
    exact_rows = sum(1 for r in fills if r["exact"])
    emitted_total = sum(r["emitted_count"] for r in fills)
    wrong_total = sum(r["emitted_wrong"] for r in fills)
    mw = [r for r in fills if r["multi_word"]]
    mw_exact = sum(1 for r in mw if r["exact"])
    enum_viol = sum(1 for r in fills if r["enum_violation"])

    payload = _common_meta(model_path, {"dev": DEV})
    payload.update({
        "set": "spanfill_dev_REPORTED_CONTEXT_NOT_A_BAR",
        "rows": n,
        "dev_exact_fill": round(exact_rows / n, 4),
        "pairwise_baseline_exact": 0.9165,
        "dev_invented_rate": round(wrong_total / emitted_total, 4)
        if emitted_total else None,
        "pairwise_baseline_invented": 0.0441,
        "G3_multiword": {
            "construct": "117-row dev slice (gold string values with "
                         "whitespace); row exact-fill",
            "n_rows": len(mw),
            "exact_rows": mw_exact,
            "exact_fill": round(mw_exact / len(mw), 4) if mw else None,
            "floor": 0.50,
            "pass": bool(mw) and mw_exact / len(mw) >= 0.50,
            "misses": [r["row"] for r in mw if not r["exact"]][:60],
        },
        "G6_enum_strictness_dev": {
            "enum_violations": enum_viol,
            "ceiling": 0,
            "pass": enum_viol == 0,
        },
    })
    dest = OUT / "dev-verdict.json"
    dest.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("dev_exact_fill", "dev_invented_rate", "G3_multiword",
                       "G6_enum_strictness_dev")}, indent=1))
    print(f"wrote {dest}")
    return payload


def run_download() -> int:
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(MODEL_REPO, MODEL_FILE))
    print(json.dumps({"path": str(path), "sha256": sha256_file(path),
                      "bytes": path.stat().st_size}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["download", "calls", "dev"])
    ap.add_argument("--model", default=None,
                    help="explicit GGUF path (default: HF cache)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "download":
        return run_download()
    model_path = resolve_model_path(args.model)
    if args.command == "calls":
        run_calls(model_path)
    else:
        run_dev(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
