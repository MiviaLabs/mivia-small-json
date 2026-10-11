# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Internet-facing serving API: REST quartet + MCP over streamable HTTP.

This is the deployable surface (Dockerfile in the repo root builds it). The
stdlib sidecar (``serve.py``) and the stdio MCP server (``mcp.py``) stay
untouched — they remain the zero-extra-dependency, air-gapped path. This
module adds the authenticated, networked deployment surface on the official
MCP SDK (``mcp`` extra, pinned in pyproject.toml).

Endpoints (auth: ``Authorization: Bearer $MIVIA_API_TOKEN`` on everything
except ``/health`` when the env var is set; open when unset):

- ``GET  /health``: ``{status, model, backend, guard, grade, rss_mb,
  version}`` — no inference; ``model`` is always ``mivia-small-json``,
  ``backend`` names the loaded router (``onnx-int8`` / ``skops`` / ``rules``),
  ``guard`` and ``grade`` report component availability.
- ``POST /v1/route``  ``{"text": ...}`` -> ``{schema, confidence, in_domain,
  model, latency_ms, routed_by}``. The router alone (~2.7 ms p95, int8 ONNX).
- ``POST /v1/extract`` ``{"text": ..., "dump": false}`` -> ``{records:
  [{result, meta}]}``. Single record, or the whole multi-line stream
  (``dump: true`` runs ``pipeline.classify_dump``).
- ``POST /v1/guard``  ``{"request": ..., "tools": [...]}`` ->
  ``{action: "CALL"|"CLARIFY"|"ABSTAIN", tool?, missing_parameters?,
  clarification_prompt?, record, meta}``. ``tools`` is a BFCL-style list of
  function specs, a ``{name: description}`` map, or a list of names. The
  prompt is rendered exactly like the measured surface (``Tools available:
  <json>`` + user request). CLARIFY is the shipped fillability veto: the
  guard asks the user for the missing mandatory parameters instead of
  guessing (owner decision 2026-10-07).
- ``POST /v1/grade``  ``{"text": ..., "anchors": [...]?}`` ->
  ``pipeline.triage_severity`` decision (canonical 4-tier severity scale by
  default: register gate -> promoted specialist; measured 46.8% exact /
  84.2% within-1 / QWK 0.675 on 600 real tickets).
- ``POST /v1/select`` ``{"text": ..., "candidates": [...]}`` ->
  ``{choice, confidence, status}`` (zero-shot multi-candidate pick).
- ``POST /v1/structure`` ``{"text": ..., "schema": ...}`` ->
  declared schema in, filled record out — no routing (universal-
  structuring 1d; ``tool_call`` refuses with a pointer to ``/v1/guard``;
  empty text yields the defaults record on REST — the MCP tool requires
  non-empty text).

Per-key metering + quota (2026-10-10, spec per-key-metering/
spec.md; opt-in via ``MIVIA_METERING_DB``, default ./data/metering.sqlite3,
gitignored): per-key ``mvk_live_…`` bearers (sha256-hashed at rest,
plaintext shown once), free tier 1,000 calls/month (``MIVIA_FREE_CALLS_LIMIT``
overrides), 429 with the reset date at the wall, ``X-Mivia-Count`` /
``X-Mivia-Limit`` / ``X-Mivia-Reset`` on metered routes,
``GET /v1/usage`` (a key sees only its own month), and
``POST /v1/admin/keys`` (create/list/revoke, admin-key guarded). The
store (``metering.py``, stdlib sqlite) holds ONLY aggregate counters and
key metadata — zero retention holds; M1 in tests/test_metering.py dumps
the whole store file and asserts no request marker survives. Only 2xx
consumes quota (a 400/422/429 does not count — the committed pricing
FAQ). MCP ``tools/call`` hits the same counter as REST.

Inference concurrency cap (2026-10-10 load test, reports/
load-test-2026-10-10.md): the seven POST /v1 routes are bounded in-process
(default 8; override ``MIVIA_MAX_CONCURRENT_INFERENCE``) because every
concurrently-serving request holds its own ONNX activation working set and
unbounded parallel traffic breached the < 2 GB envelope. Requests beyond
the cap WAIT in the app (arrival-order pacing) — no 429, no behavior
change per response; ``/health`` and the metering/admin surfaces stay
ungated so health answers during saturation. This is the in-process half
of the registered rate-limit lever (owner, 2026-10-08); edge rate-limit
and instance class remain deployment decisions.

MCP: the same capabilities are exposed as MCP tools (``mivia_route``,
``mivia_extract``, ``mivia_guard``, ``mivia_grade``, ``mivia_select``,
``mivia_repair_tool_call``, ``mivia_structure``) on the official SDK,
streamable HTTP, stateless JSON mode — mounted at ``/mcp`` in the same
process/port. Claude Code:
``claude mcp add --transport http mivia https://HOST/mcp --header
"Authorization: Bearer $TOKEN"``. Cursor: ``mcp.json`` URL + headers.

Honest scoping: every number above is a measurement on its named set (see
docs/MODEL-STATE.md); the guard's 79.7%-abstention operating point is
BFCL-catalogue-shaped, real agent-session catalogues are served by the
session-transfer rank (71.5% label-1, AUC 0.93). No OAuth discovery, no
TLS here — reverse-proxy territory (docs/USE_CASES.md); the monthly
per-key quota is in-process when metering is on (above), fine-grained
rate limiting still is not.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import metering
from . import pipeline

MAX_TEXT_CHARS = 262_144          # 256 KiB of text per request
MAX_TOOLS = 200                    # catalogue entries per guard request
MAX_CANDIDATES = 64                # select() candidates per request
MAX_ANCHORS = 200                  # grade() anchors per request (one
                                   # cross-encoder pass per anchor)
MAX_BODY_BYTES = 4 * 1024 * 1024   # whole-body ceiling, matching the MCP
                                   # SDK's 4 MiB streamable-HTTP cap
# Dict-form select candidates' per-candidate examples ("description
# enrichment", the measured 48.0% -> 63.8% Banking77 win): 2-3 examples is
# the measured sweet spot, 8 is the hard ceiling; beyond ~1024 chars the
# encoder's truncation window cannot read an example anyway, so accepting
# more would be silent waste.
MAX_SELECT_EXAMPLES = 8
MAX_EXAMPLE_CHARS = 1024


# --------------------------------------------------------------------------
# Eager warmup (spec guard-serve-performance): the served
# surfaces exercise every artifact they serve BEFORE reporting ready, so the
# first request pays decision latency only. Warmup loads nothing new — every
# artifact here is one the first request would load lazily anyway — and a
# failure is a loud warning naming the step, never fatal (a broken optional
# artifact degrades to today's lazy per-request semantics).
# --------------------------------------------------------------------------

# Fixed short benign text for the router warmup (log_event-shaped: no
# catalogue, no tool vocabulary).
_WARMUP_TEXT = "2026-03-01T12:00:00Z ERROR [auth]: token expired"
# Fixed benign two-tool catalogue for the guard warmup: loads both served
# cross-encoders (gate and rank) through the shipped sha-verified loader —
# the same singleton and construction path a real request hits — and the
# decision itself first-runs the gate on real forwards.
_WARMUP_TOOLS = (
    {"type": "function",
     "function": {"name": "get_weather",
                  "description": "Get the current weather for a city.",
                  "parameters": {"type": "object",
                                 "properties": {"city": {
                                     "type": "string",
                                     "description": "City name."}},
                                 "required": ["city"]}}},
    {"type": "function",
     "function": {"name": "list_files",
                  "description": "List the files in a directory.",
                  "parameters": {"type": "object",
                                 "properties": {"path": {
                                     "type": "string",
                                     "description": "Directory path."}},
                                 "required": ["path"]}}},
)
_WARMUP_REQUEST = "list the files in the directory"


def _warmup(router) -> None:
    """One classify + one guard decision, each step individually guarded.

    Runs synchronously inside the lifespan before it yields: uvicorn does
    not serve traffic until startup completes, so no request can observe
    the cold-start spike. Per-worker warm is automatic — the lifespan runs
    once per worker process.
    """
    from . import tool_match

    def _guard_warmup():
        # The eager load through the shipped sha-verified loader (the same
        # singleton a real request hits), then one real decision on the
        # fixed two-tool catalogue — first-runs the gate cross-encoder.
        tool_match._get_guard()
        return do_guard(_WARMUP_REQUEST, [dict(t) for t in _WARMUP_TOOLS],
                        router)

    steps = (
        ("router classify",
         lambda: pipeline.classify(_WARMUP_TEXT, router)),
        ("guard decision", _guard_warmup),
    )
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            print(f"[mivia-small-json] warmup step {name!r} failed "
                  f"({type(exc).__name__}: {exc}); the first request will "
                  "pay the lazy load instead", file=sys.stderr)


class ArtifactUnavailable(RuntimeError):
    """A required model artifact failed to load (maps to HTTP 503)."""


def _validated_text(payload: dict, key: str = "text", allow_empty: bool = False) -> str:
    text = payload.get(key)
    if not isinstance(text, str):
        raise ValueError(f"field {key!r} must be a string")
    if not allow_empty and not text.strip():
        raise ValueError(f"field {key!r} must be a non-empty string")
    if len(text) > MAX_TEXT_CHARS:
        raise OverflowError(f"field {key!r} exceeds {MAX_TEXT_CHARS} chars")
    return text


def _canonical_anchors() -> list[str]:
    from .severity_head import CANONICAL_SEVERITY_ANCHORS

    return list(CANONICAL_SEVERITY_ANCHORS)


# --------------------------------------------------------------------------
# Handler core — one source of truth, shared by REST and MCP. Each returns a
# JSON-serializable dict; ValueError/OverflowError map to 4xx, anything else
# is a 500 that never leaks a traceback.
# --------------------------------------------------------------------------


def do_route(text: str, router) -> dict:
    """Route one text: schema + confidence + in-domain flag (ONE encoder
    pass — pipeline._route_with_gate carries the gate verdict from the
    same forward pass the route already needed)."""
    t0 = time.perf_counter()
    schema, routed_by, conf, in_domain = pipeline._route_with_gate(router, text)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {"schema": schema, "confidence": round(float(conf), 4),
            "in_domain": in_domain, "model": "mivia-small-json",
            "latency_ms": latency_ms, "routed_by": routed_by}


def do_structure(text: str, schema: Any) -> dict:
    """Universal-structuring surface: declared schema in, filled record
    out (no routing). Unknown schema -> ValueError -> 422 upstream."""
    t0 = time.perf_counter()
    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("schema is required: pass a built-in or "
                         "registered schema name")
    obj, meta = pipeline.structure(text, schema.strip())
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {"schema": schema.strip(), "record": obj,
            "valid": bool(meta["valid"]),
            "retries": int(meta.get("retries", 0)),
            "routed_by": meta.get("routed_by", "declared"),
            "model": "mivia-small-json", "latency_ms": latency_ms}


def do_extract(text: str, router, dump: bool = False) -> dict:
    """Extract structured record(s): one text, or a whole multi-line dump."""
    pairs = (pipeline.classify_dump(text, router) if dump
             else [pipeline.classify(text, router)])
    return {"records": [{"result": obj, "meta": meta} for obj, meta in pairs]}


MAX_TOOLS_BYTES = MAX_TEXT_CHARS  # 256 KiB, the oversize law


def _normalize_tools(tools: Any) -> list:
    def _entry_name(entry: dict) -> str:
        """The tool's name wherever the spec carries it (inner or outer)."""
        inner = entry.get("function")
        inner = inner if isinstance(inner, dict) else entry
        name = inner.get("name")
        return str(name).strip() if name is not None else ""

    if isinstance(tools, dict):
        out = []
        for name, desc in tools.items():
            if not str(name).strip():
                # The same law do_select applies to blank candidates: a
                # nameless tool cannot be called, and gating it produced
                # {"action": "CALL", "tool": ""} (review N4) — an explicit
                # 4xx is the honest answer.
                raise ValueError("each tool must have a non-empty name")
            out.append({"type": "function",
                        "function": {"name": str(name),
                                     "description": str(desc)}})
        return out
    if isinstance(tools, list):
        out = []
        for entry in tools:
            if isinstance(entry, str):
                if not entry.strip():
                    raise ValueError("each tool must have a non-empty name")
                out.append({"type": "function", "function": {"name": entry}})
            elif isinstance(entry, dict):
                if "function" not in entry and "parameters" in entry:
                    # Flat OpenAI-style spec -> the nested BFCL shape the
                    # measured surface expects.
                    entry = {"type": "function", "function": entry}
                if not _entry_name(entry):
                    raise ValueError("each tool must have a non-empty name")
                out.append(entry)
            else:
                raise ValueError("each tool must be an object or a name string")
        return out
    raise ValueError("field 'tools' must be a list of tool specs or a "
                     "{name: description} map")


def _find_tool_entry(tools: list, name: str | None) -> dict | None:
    """The catalogue entry whose tool is named ``name`` (best effort)."""
    for entry in tools:
        inner = entry.get("function") if isinstance(entry, dict) else None
        if not isinstance(inner, dict):
            inner = entry if isinstance(entry, dict) else {}
        if str(inner.get("name", "")) == (name or ""):
            return entry
    return None


def _required_params(tools: list, name: str | None) -> list[str]:
    """Declared required parameters of tool ``name`` (best effort)."""
    entry = _find_tool_entry(tools, name)
    if entry is None:
        return []
    inner = entry.get("function") if isinstance(entry, dict) else None
    if not isinstance(inner, dict):
        inner = entry if isinstance(entry, dict) else {}
    params = inner.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("required"), list):
        return [str(p) for p in params["required"]]
    return []


def _guard_downgrade_check(out: dict, tools: list, request: str) -> dict:
    """Decision-layer composition: a CALL whose arguments are empty or miss
    declared required parameters is downgraded to CLARIFY — but ONLY when
    the ONE missing-params oracle (tool_match.fillability_veto) also says
    the request supplies nothing.

    The guard's fillability heuristic can accept a request whose values it
    can see ("restart nginx on staging") while the argument extractor fills
    nothing — measured 2026-10-08 (serving probe): that shape produced a
    false CALL with ``arguments: "{}"``. The veto remains the authority on
    which of those shapes is unfillable: when it confirms the request
    supplies nothing, the downgrade fires with the veto's own missing list
    (the arg-derived list is the fallback when the veto names none); when
    the veto says fillable, the CALL stays — it is the pipeline-vetted
    honest minimum, and the two surfaces agree by construction. The
    unconditional ``{}``-downgrade used here before 2026-10-10
    re-clarified those vetted calls — the served surface and the pipeline
    diverged in opposite directions (review P1), and the same unconditional
    re-clarification is the recorded BFCL routed-to-expected 89.1 -> 25.1
    failure mode. The veto verdict travels in ``veto_missing`` for the
    clarification; the veto reason is deterministic on its two unfillable
    outcomes (missing_arguments | underspecified_query) and both render the
    same ask. Deterministic check on the shipped bytes — no model change.
    """
    if out["action"] != "CALL":
        return out
    try:
        raw = out["record"].get("arguments")
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        args = None
    required = _required_params(tools, out["tool"])
    if not required:
        return out
    missing = required
    if isinstance(args, dict):
        missing = [p for p in required if p not in args]
    if not missing:
        return out
    from . import tool_match

    veto_ok, _reason, veto_missing = tool_match.fillability_veto(
        _find_tool_entry(tools, out["tool"]) or {}, request)
    if veto_ok:
        return out
    out["action"] = "CLARIFY"
    out["missing_parameters"] = veto_missing or missing
    out["clarification_prompt"] = (
        f"Please provide {', '.join(out['missing_parameters'])} "
        f"to use {out['tool']}.")
    return out


def do_guard(request: str, tools: Any, router) -> dict:
    """Honest tool gating: CALL / CLARIFY / ABSTAIN (+ the full record)."""
    if len(tools) > MAX_TOOLS:
        raise OverflowError(f"too many tools (>{MAX_TOOLS})")
    catalogue = json.dumps(tools, ensure_ascii=False)
    if len(catalogue) > MAX_TOOLS_BYTES:
        # Oversize law, same as MAX_TEXT_CHARS: the tokenizer would encode
        # the whole catalogue before head-tail truncation (a single 8 MB
        # description measured 6.7 s and multi-GB prompts were reachable).
        raise OverflowError(
            f"tools catalogue exceeds {MAX_TOOLS_BYTES} rendered bytes")
    if not tools:
        # Empty catalogue: no tool can ever be called, so the only honest
        # action is ABSTAIN. Classify the RAW request for the record — the
        # "Tools available: []" composite reads like a ticket and routed
        # plain chat to issue (confirmed P0, 2026-10-09). The composite
        # prompt below stays the measured BFCL surface (non-empty only).
        obj, meta = pipeline.classify(request, router)
        return {"action": "ABSTAIN", "tool": None,
                "missing_parameters": None, "clarification_prompt": None,
                "record": obj, "meta": meta}
    prompt = (f"Tools available: {catalogue}\n\n"
              f"User request: {request}")
    obj, meta = pipeline.classify(prompt, router)
    kind = obj.get("type")
    out: dict = {"action": "ABSTAIN", "tool": None,
                 "missing_parameters": None, "clarification_prompt": None,
                 "record": obj, "meta": meta}
    if kind == "tool_call":
        out["action"] = "CALL"
        out["tool"] = obj.get("name")
    elif kind == "tool_clarification":
        out["action"] = "CLARIFY"
        out["tool"] = obj.get("tool")
        out["missing_parameters"] = obj.get("missing_parameters")
        out["clarification_prompt"] = obj.get("clarification_prompt")
    # The RAW user request (never the composite prompt: the veto's
    # content check must not read the catalogue's own tool vocabulary).
    return _guard_downgrade_check(out, tools, request)


def do_grade(text: str, anchors: list[str] | None = None) -> dict:
    """Severity triage on the ordered anchors (canonical scale by default)."""
    if anchors is None:
        anchors = _canonical_anchors()
    if not isinstance(anchors, list) or not all(isinstance(a, str) for a in anchors):
        raise ValueError("field 'anchors' must be a list of strings")
    # Degenerate-anchor law (2026-10-09 serving probe): an empty list graded
    # to a 200 "ask ... Options: none listed" and a blank anchor ("  ")
    # graded to a 200 "act" on a tier the caller never defined — both
    # wrong-but-valid shapes. A scale with no usable rungs is a caller
    # programming error: explicit 4xx, never a guessed decision.
    if not anchors or not all(a.strip() for a in anchors):
        raise ValueError("field 'anchors' must be a non-empty list of "
                         "non-blank strings")
    if len(anchors) > MAX_ANCHORS:
        # One guard pass per anchor; an unbounded list would let N anchors
        # force N cross-encoder passes (same law as MAX_TOOLS).
        raise OverflowError(f"too many anchors (>{MAX_ANCHORS})")
    decision = pipeline.triage_severity(text, anchors)
    return {"result": decision, "anchors": anchors}


def _validated_select_candidates(candidates: Any) -> list:
    """The /v1/select candidates law, both accepted shapes.

    A non-empty list of non-blank strings (today's shape, unchanged) or of
    ``{"name", "description"?, "examples"?}`` dicts — the first-class
    description-enrichment form (name + description + example utterances,
    rendered with the measured separator convention). The strict serving
    law sits on top of the permissive library path: unknown keys are
    rejected (a typo'd "example" key would silently drop the measured
    lift), examples must be non-blank strings within the caps, and the
    intent head — which reads bare label strings only — refuses dicts.
    """
    for i, c in enumerate(candidates):
        if isinstance(c, str):
            if not c.strip():
                raise ValueError(
                    f"candidates[{i}] must be a non-blank string")
            continue
        if not isinstance(c, dict):
            raise ValueError(
                "field 'candidates' must be a non-empty list of strings "
                "or {name, description, examples} dicts")
        unknown = set(c) - {"name", "description", "examples"}
        if unknown:
            raise ValueError(
                f"candidates[{i}] carries unknown keys: {sorted(unknown)}; "
                "accepted: name, description, examples")
        name = c.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"candidates[{i}] must carry a non-blank 'name' string")
        desc = c.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ValueError(
                f"candidates[{i}].description must be a string")
        examples = c.get("examples")
        if examples is None:
            continue
        if not isinstance(examples, list) or not all(
                isinstance(e, str) for e in examples):
            raise ValueError(
                f"candidates[{i}].examples must be a list of strings")
        if any(not e.strip() for e in examples):
            raise ValueError(
                f"candidates[{i}].examples must be non-blank strings")
        if len(examples) > MAX_SELECT_EXAMPLES:
            raise OverflowError(
                f"candidates[{i}] carries too many examples "
                f"(>{MAX_SELECT_EXAMPLES})")
        for e in examples:
            if len(e) > MAX_EXAMPLE_CHARS:
                raise ValueError(
                    f"candidates[{i}].examples entries exceed "
                    f"{MAX_EXAMPLE_CHARS} chars (the encoder window "
                    "cannot read more)")
    return candidates


def do_select(text: str, candidates: Any, head: str = "tool") -> dict:
    """Zero-shot multi-candidate pick with an ambiguity status.

    ``head="tool"`` (default) serves the WIRED zero-shot select-rank head
    (OWNER WIRING DECREE 2026-10-11, zero-shot-select-r2/
    spec.md: Banking77 select 81.00 fresh-reserve top-1, binary referee
    99.6/99.6; the 85-parity and null-abstention bars are a registered
    accepted trade). Its own band-tau calibration gates
    select-when-confident. Degradation paths, in order: the head bytes
    missing/failing their sha sidecars, or ``MIVIA_SELECT_RANK_HEAD=
    legacy`` — the guard's rank artifact + the legacy select-calibration
    (the pre-decree surface, byte-identical behavior). ``head="intent"``
    serves the language-gated dual intent-rank head (tool_match
    .select_intent): EN -> intent-rank, non-EN -> the multilingual head —
    the universal-structuring selection surface.

    Candidates are non-blank strings, or ``{"name", "description"?,
    "examples"?}`` dicts (head="tool" only) — the first-class description
    enrichment form; see docs/USAGE.md §9.1 for the measured numbers."""
    from . import tool_match

    if head not in ("tool", "intent"):
        raise ValueError("field 'head' must be 'tool' or 'intent'")
    if (not isinstance(candidates, list) or not candidates):
        # Blank candidates are a caller programming error, not a pick to
        # lose: a "" candidate measured a 200 with choice "" (2026-10-09
        # serving probe) — an explicit 4xx is the honest answer.
        raise ValueError("field 'candidates' must be a non-empty list of "
                         "strings or {name, description, examples} dicts")
    _validated_select_candidates(candidates)
    if head == "intent":
        if any(not isinstance(c, str) for c in candidates):
            raise ValueError(
                "head 'tool' serves dict candidates; head 'intent' reads "
                "plain label strings")
    if len(candidates) > MAX_CANDIDATES:
        raise OverflowError(f"too many candidates (>{MAX_CANDIDATES})")
    if head == "intent":
        out = tool_match.select_intent(text, candidates)
        status = "ok" if out["confidence"] > tool_match.SELECT_CONF_FLOOR \
            else "ambiguous"
        return {"choice": out["choice"],
                "confidence": out["confidence"],
                "status": status, "head": out["head"], "lang": out["lang"]}
    matcher = tool_match._get_guard()
    if matcher is None:
        raise ArtifactUnavailable("tool-match guard artifact not loadable")
    choice, conf, status = matcher.select(text, candidates, with_status=True)
    return {"choice": choice, "confidence": round(float(conf), 4),
            "status": status}


def do_repair(raw_arguments: str) -> dict:
    """Repair broken LLM tool-call JSON arguments to strict JSON.

    ``lossy: true`` flags a repair that collapsed a substantive input to an
    empty object — the honest signal that the repair budget could not
    recover the arguments (measured 2026-10-08: single-quote payloads with
    mixed Python literals and XML-wrapped calls can collapse to ``{}``).
    """
    from .tool_call import _strip_fences, repair_arguments

    if len(raw_arguments) > MAX_TEXT_CHARS:
        raise OverflowError(f"raw_arguments exceeds {MAX_TEXT_CHARS} chars")

    repaired = repair_arguments(raw_arguments)
    lossy = False
    stripped = raw_arguments.strip()
    unfenced = _strip_fences(stripped).strip()
    is_empty_object = (
        unfenced.startswith("{")
        and unfenced.endswith("}")
        and not unfenced[1:-1].strip()
    )
    if repaired == "{}" and bool(stripped) and not is_empty_object:
        lossy = True
    return {"repaired": repaired, "lossy": lossy}


# --------------------------------------------------------------------------
# Inference concurrency cap (2026-10-10 load test, reports/
# load-test-2026-10-10.md): every concurrently-serving /v1 request holds its
# own ONNX activation working set — VmRSS peaked at 2389.6 MB with 24
# parallel 200-tool guard clients and 2866.2 MB in an 80-request burst,
# against a ~1.2 GB sequential baseline (~50 MB marginal per in-flight
# request). On a 2 GB-capped deploy that is an OOM-kill. The registered
# levers are edge rate-limit / instance class (owner, 2026-10-08); this is
# the in-process half of rate limiting, matching the stdlib sidecar's
# handler cap (serve.DEFAULT_MAX_CONCURRENT_HANDLERS = 8, commit 33609db).
# --------------------------------------------------------------------------

# The seven POST /v1 inference routes — today the exact set metering meters
# (metering.METERED_REST_PATHS), written as a literal here because the two
# sets answer different questions (billing vs the ONNX activation window)
# and must be free to diverge. /health, /v1/usage, /v1/admin/keys and /mcp
# are NOT gated: health must answer during saturation.
INFERENCE_ROUTES = frozenset({
    "/v1/route", "/v1/extract", "/v1/guard", "/v1/grade",
    "/v1/select", "/v1/repair", "/v1/structure",
})

# Default 8 vs the load data: ~1.2 GB sequential base + 8 x ~50 MB marginal
# per in-flight request ~= 1.6 GB < 2 GB with headroom for the interpreter,
# allocator arenas and response buffers — and the load test's own 8-client
# cell measured a 1586.9 MB peak, under the envelope. Override per
# deployment with MIVIA_MAX_CONCURRENT_INFERENCE.
DEFAULT_MAX_CONCURRENT_INFERENCE = 8


def _resolve_inference_cap(value: int | None = None) -> int:
    """Explicit argument > ``MIVIA_MAX_CONCURRENT_INFERENCE`` > module
    default, floored at 1 (garbage falls back to the default, never crashes
    boot) — serve._resolve_max_handlers' law, mirrored on purpose."""
    if value is None:
        raw = os.environ.get("MIVIA_MAX_CONCURRENT_INFERENCE", "").strip()
        try:
            value = int(raw) if raw else DEFAULT_MAX_CONCURRENT_INFERENCE
        except ValueError:
            value = DEFAULT_MAX_CONCURRENT_INFERENCE
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_CONCURRENT_INFERENCE
    return max(1, value)


class _InferenceGate:
    """Bound concurrent /v1/* inference with an asyncio semaphore — requests
    beyond the cap WAIT in the app (queueing), never rejected: no 429, no
    behavior change per response, only arrival-order pacing. The wait parks
    on the event loop (never blocking it — /health keeps answering), and
    the slot is held across the whole downstream call — metering admission,
    threadpool inference, response start — so at most ``cap`` inferences
    are in flight at once.

    Placement (create_app): the gate wraps the app OUTSIDE the metering
    middleware (Bearer -> Gate -> Metering -> FastAPI). Metering admits a
    call when the request REACHES it, so a request still queued at the gate
    is not yet admitted: a client that disconnects while queued costs
    nothing (exactly like a request that never arrived), and a
    queued-then-served request is admitted once, exactly like an unqueued
    one. Inside the gate, metering's admit/settle bracket is untouched.
    Unauthenticated traffic is rejected by the bearer gate before it can
    consume a slot.

    The semaphore binds to the running loop lazily (first contended
    acquire), so sequential requests through per-request test portals on
    different event loops never trip loop affinity; a deployment serves
    all traffic on one uvicorn loop.
    """

    def __init__(self, app, cap: int):
        self.app = app
        self.cap = cap
        self.slots = asyncio.Semaphore(max(1, int(cap)))

    @property
    def routes(self):
        """Transparent-middleware passthrough so the wrapped FastAPI app
        stays introspectable through the chain (route registration checks
        walk ``stack.app.routes``)."""
        return self.app.routes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" \
                or (scope.get("path") or "") not in INFERENCE_ROUTES:
            await self.app(scope, receive, send)
            return
        async with self.slots:  # full: further requests queue here
            await self.app(scope, receive, send)


# --------------------------------------------------------------------------
# Auth: one pure-ASGI middleware covers REST and the mounted MCP app alike.
# --------------------------------------------------------------------------


def _health_scope(scope: dict) -> dict | None:
    """Scope to forward for an open liveness probe, or None to authenticate.

    ``/health``, a trailing slash, and ``?probe=1`` (in ``query_string`` or
    glued onto ``path``) are the probe. The forwarded path is ``/health``
    so the route answers 200. ``/healthcare`` and ``/health/live`` are not
    the probe.
    """
    path = scope.get("path") or ""
    if not isinstance(path, str):
        return None
    bare, sep, embedded = path.partition("?")
    if len(bare) > 1:
        bare = bare.rstrip("/")
    if bare != "/health":
        return None
    if path == "/health":
        return scope
    rewritten = dict(scope)
    rewritten["path"] = "/health"
    if sep and embedded:
        extra = embedded.encode("ascii", "ignore")
        existing = rewritten.get("query_string") or b""
        if isinstance(existing, str):
            existing = existing.encode("ascii", "ignore")
        elif not isinstance(existing, (bytes, bytearray)):
            existing = b""
        rewritten["query_string"] = (
            extra if not existing else bytes(existing) + b"&" + extra)
    return rewritten


_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def _host_allowed(host: str | None) -> bool:
    """Loopback allowlist for the TOKENLESS gate (DNS-rebinding defense on
    the REST half, mirroring the MCP transport security). A port suffix is
    allowed; anything non-loopback is rejected — the remedy for proxies
    that rewrite Host is the same as for the MCP half: set
    MIVIA_API_TOKEN."""
    if not host:
        return False
    host = host.casefold()
    if host.startswith("["):  # bracketed IPv6: strip through ']'
        host = host.partition("]")[0] + "]"
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    return host in _LOOPBACK_HOSTS


class BearerMiddleware:
    """Static-bearer gate (constant-time compare).

    ``/health`` stays open, including a trailing slash and a probe query.
    Tokenless: the gate fails open by contract but still rejects
    non-loopback Host headers — DNS rebinding must not reach the REST
    half of a tokenless deployment (the MCP half is covered by the
    transport security settings).

    2026-10-10 (per-key metering, spec per-key-metering):
    two optional identity sources were added — ``admin_key`` (the
    ``MIVIA_ADMIN_KEY`` bootstrap credential) and ``key_lookup`` (per-key
    ``mvk_live_…`` resolution against the metering store). Both are None
    when metering is off, and then this gate is byte-identical to the
    pre-metering behavior. On success the resolved identity travels in
    ``scope["mivia.identity"]`` as ``{"kind": "static"|"admin"|"key"|"anonymous",
    ...}`` for the metering middleware inside. A presented-but-invalid
    ``mvk_live_`` credential is a 401 (a failing credential is an auth
    failure, not anonymous traffic), in tokenless mode too; any other
    bearer in tokenless mode stays open-anonymous.
    """

    def __init__(self, app, token: str, admin_key: str | None = None,
                 key_lookup=None):
        self.app = app
        self.token = token
        self.admin_key = admin_key
        self.key_lookup = key_lookup

    def _identity(self, provided: str):
        """Identity for a bearer that did NOT match the static token:
        an identity dict, the string ``"invalid"`` (a presented mvk
        credential failed), or None (anonymous)."""
        bare = provided[7:].strip() if provided.startswith("Bearer ") else ""
        if self.admin_key and bare and hmac.compare_digest(
                bare.encode("utf-8", "replace"),
                self.admin_key.encode("utf-8", "replace")):
            return {"kind": "admin"}
        if self.key_lookup is not None and bare.startswith(
                metering.KEY_PREFIX):
            row = self.key_lookup(bare)
            if row is not None:
                return row
            return "invalid"
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if not self.token:
            headers = {k: v for k, v in scope.get("headers") or []}
            if not _host_allowed(
                    headers.get(b"host", b"").decode("latin-1")):
                body = json.dumps({"error": "invalid host"}).encode()
                await send({"type": "http.response.start", "status": 421,
                            "headers": [(b"content-type",
                                         b"application/json"),
                                        (b"content-length",
                                         str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
            # 2026-10-10 (per-key metering): open mode stays open, but a
            # presented mvk credential is still resolved — metering works
            # tokenless; a bad one is 401, everything else anonymous.
            ident = None
            if self.admin_key or self.key_lookup is not None:
                provided = ""
                for name, value in scope.get("headers") or []:
                    if name == b"authorization":
                        provided = value.decode("latin-1")
                        break
                ident = self._identity(provided)
            if ident == "invalid":
                body = json.dumps({"error": "unauthorized"}).encode()
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type",
                                         b"application/json"),
                                        (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": body})
                return
            forwarded = dict(scope)
            forwarded["mivia.identity"] = ident or {"kind": "anonymous"}
            await self.app(forwarded, receive, send)
            return
        health = _health_scope(scope)
        if health is not None:
            await self.app(health, receive, send)
            return
        provided = ""
        for name, value in scope.get("headers") or []:
            if name == b"authorization":
                provided = value.decode("latin-1")
                break
        ok = provided.startswith("Bearer ") and hmac.compare_digest(
            provided[7:].strip().encode("utf-8", "replace"),
            self.token.encode("utf-8", "replace"))
        if not ok:
            ident = self._identity(provided)  # admin / per-key (2026-10-10)
            if ident is None or ident == "invalid":
                body = json.dumps({"error": "unauthorized"}).encode()
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type",
                                         b"application/json"),
                                        (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": body})
                return
            forwarded = dict(scope)
            forwarded["mivia.identity"] = ident
            await self.app(forwarded, receive, send)
            return
        forwarded = dict(scope)
        forwarded["mivia.identity"] = {"kind": "static"}
        await self.app(forwarded, receive, send)


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------


def _derive_origins(hosts: list[str]) -> list[str]:
    """Browser origins matching a host allowlist entry.

    Non-wildcard entries yield both schemes; wildcard entries
    (``host:*``) yield the wildcard pattern the SDK matches ports
    against. Host-only entries (``example.com``) yield the exact
    origins AND the port-wildcard variants — the browser sends the
    actual port in Origin, and the host is the security boundary here,
    not the port (review P2: ``https://example.com:8443`` otherwise
    403s). Explicit-port entries (``127.0.0.1:8443``) stay exact.
    """
    out: list[str] = []
    for h in hosts:
        if "://" in h:
            out.append(h)
            continue
        out.append(f"https://{h}")
        out.append(f"http://{h}")
        explicit_port = ("]:" in h) if h.startswith("[") else (":" in h)
        if not explicit_port and not h.endswith(":*"):
            out.append(f"https://{h}:*")
            out.append(f"http://{h}:*")
    return out


def _resolve_free_calls_limit(explicit: int | None) -> int:
    """Explicit argument > MIVIA_FREE_CALLS_LIMIT > 1,000 (garbage env
    falls back to the default, never crashes boot; explicit and env
    values clamp to >= 1 — docstring aligned with the clamp on
    2026-10-10, the previous text claimed a raise)."""
    if explicit is not None:
        return max(1, int(explicit))
    raw = os.environ.get("MIVIA_FREE_CALLS_LIMIT", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return metering.DEFAULT_FREE_CALLS


def create_app(token: str | None = None, allowed_hosts: list[str] | None = None,
               router=None, allowed_origins: list[str] | None = None,
               metering_db: str | None = None,
               metering_store: "metering.MeteringStore | None" = None,
               admin_key: str | None = None,
               free_calls_limit: int | None = None,
               max_concurrent_inference: int | None = None):
    """Build the FastAPI app (REST quartet + mounted MCP server).

    ``token``: bearer token; ``None`` reads ``MIVIA_API_TOKEN`` (empty/unset
    = open, local-dev default). ``allowed_hosts``: MCP DNS-rebinding allow
    list; ``None`` reads ``MIVIA_ALLOWED_HOSTS`` (unset = protection off).
    ``router``: preloaded router (tests); default loads the shipped bytes.
    ``max_concurrent_inference``: cap on concurrently-ACTIVE /v1 inference
    (beyond it requests queue, never 429 — see ``_InferenceGate``);
    ``None`` reads ``MIVIA_MAX_CONCURRENT_INFERENCE`` (default 8, floored
    at 1, garbage falls back to the default).

    Per-key metering + quota (spec per-key-metering,
    2026-10-10): OPT-IN — active iff ``MIVIA_METERING_DB`` (or
    ``metering_db=`` / a prebuilt ``metering_store=``) is set; the
    documented deployment default for the variable is
    ./data/metering.sqlite3 (gitignored). Auto-on would falsify the
    deployed zero-disk contract and make every existing create_app() call
    write a file. When on: ``MIVIA_ADMIN_KEY`` (or ``admin_key=``)
    bootstraps the admin credential, ``MIVIA_FREE_CALLS_LIMIT`` (or
    ``free_calls_limit=``) overrides the 1,000/month free cap, mvk keys
    authenticate, metered responses carry X-Mivia-* headers, and
    GET /v1/usage + POST /v1/admin/keys exist. Off: byte-identical to the
    pre-metering behavior.
    """
    from mcp.server import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings

    if token is None:
        token = os.environ.get("MIVIA_API_TOKEN", "") or ""
    if allowed_hosts is None:
        raw = os.environ.get("MIVIA_ALLOWED_HOSTS", "").strip()
        allowed_hosts = ([h.strip() for h in raw.split(",") if h.strip()]
                         or None)
    if not token:
        # The gate fails open by documented contract (local-dev default) —
        # say so once, loudly, so a public deployment cannot forget it.
        print(
            "[mivia-small-json] MIVIA_API_TOKEN is EMPTY: the REST and MCP "
            "surfaces are UNAUTHENTICATED. Set the token for anything "
            "reachable beyond loopback.", file=sys.stderr)

    if router is None:
        router = pipeline.load_active_router()

    # -- per-key metering + quota (opt-in; see the factory docstring) ----
    if metering_store is None and metering_db is None:
        raw_db = os.environ.get("MIVIA_METERING_DB", "").strip()
        metering_db = raw_db or None
    if metering_store is None and metering_db:
        metering_store = metering.MeteringStore(metering_db)
    if metering_store is not None and admin_key is None:
        admin_key = os.environ.get("MIVIA_ADMIN_KEY", "").strip() or None
    limit = _resolve_free_calls_limit(free_calls_limit)

    mcp = MCPServer(
        "mivia-small-json",
        version=_version(),
        instructions=("Structure unstructured software-workflow text "
                      "(issues, logs, diffs, tool calls, diagnostics, "
                      "reviews) into strictly valid JSON, and gate/grade "
                      "agent decisions on CPU."),
    )

    # -- MCP tools: same handler core as REST -----------------------------
    @mcp.tool()
    def mivia_route(text: str) -> dict:
        """Route text to its schema (issue, log_event, commit, pull_request,
        tool_call, diagnostic, diff, review_comment, no_tool_call)."""
        return do_route(_clean(text), router)

    @mcp.tool()
    def mivia_extract(text: str, dump: bool = False) -> dict:
        """Extract structured fields from text; dump=True parses a whole
        multi-line stream (stack traces grouped, chatter stripped)."""
        return do_extract(_clean(text), router, bool(dump))

    @mcp.tool()
    def mivia_guard(request: str, tools: list | dict) -> dict:
        """Honest tool gating against a tool catalogue: CALL when a listed
        tool matches with fillable parameters, CLARIFY with
        missing_parameters when mandatory arguments are absent, ABSTAIN
        (no_tool_call) when nothing matches. Pass BFCL-style function specs
        or a {name: description} map."""
        return do_guard(_clean(request), _normalize_tools(tools), router)

    @mcp.tool()
    def mivia_grade(text: str, anchors: list | None = None) -> dict:
        """Ordinal severity triage (act tier / ask). Default: the canonical
        4-tier severity scale; pass your own ordered anchor strings for
        arbitrary SLA scales."""
        return do_grade(_clean(text), anchors)

    @mcp.tool()
    def mivia_select(text: str, candidates: list) -> dict:
        """Zero-shot multi-candidate selection: pick the best candidate for
        the text (team routing, categorization) with a confidence and an
        ambiguity status."""
        return do_select(_clean(text), candidates)

    @mcp.tool()
    def mivia_repair_tool_call(raw_arguments: str) -> dict:
        """Repair broken LLM tool-call JSON arguments (single quotes, Python
        literals, trailing commas) into strict JSON."""
        return do_repair(raw_arguments)

    @mcp.tool()
    def mivia_structure(text: str, schema: str) -> dict:
        """Universal structuring: force a declared output schema (built-in
        or registered name) and extract its fields from the text — no
        routing; defaults instead of fabrication."""
        return do_structure(_clean(text), schema)

    # DNS-rebinding protection must be set EXPLICITLY: transport_security=None
    # makes the SDK auto-enable localhost-only protection (host defaults to
    # "127.0.0.1"), which rejects every public Host — and, behind proxies that
    # rewrite Host, even loopback. Local/dev (no MIVIA_ALLOWED_HOSTS): off —
    # /mcp is bearer-gated anyway. UNLESS the bearer gate is also open (no
    # token): then loopback-only protection stays ON, so a tokenless
    # deployment cannot be wide open on both layers at once. Public:
    # allowlist the serving domain AND set MIVIA_API_TOKEN.
    #
    # allowed_origins: with the allowlist set, the SDK checked Host but left
    # Origin empty — every browser cross-origin POST 403'd while no-Origin
    # requests reached the parser. The browser origins are derived from the
    # same allowlist (both schemes) unless overridden explicitly.
    if allowed_origins is None:
        allowed_origins = (
            _derive_origins(allowed_hosts) if allowed_hosts
            else (["https://localhost:*", "http://localhost:*",
                   "https://127.0.0.1:*", "http://127.0.0.1:*",
                   "https://[::1]:*", "http://[::1]:*"] if not token else []))
    transport_security = (
        TransportSecuritySettings(allowed_hosts=allowed_hosts,
                                  allowed_origins=allowed_origins)
        if allowed_hosts
        else TransportSecuritySettings(
            enable_dns_rebinding_protection=not token,
            allowed_hosts=(["localhost:*", "127.0.0.1:*", "[::1]:*"]
                           if not token else []),
            allowed_origins=allowed_origins,
        ))
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, json_response=True,
        transport_security=transport_security)

    @asynccontextmanager
    async def _lifespan(app):
        # Mounted sub-app lifespans never run; the session manager must be
        # entered by the host app (official SDK ASGI integration contract).
        async with mcp.session_manager.run():
            # Eager warmup AFTER the session manager is up and BEFORE the
            # app yields ready: one classify (router artifact) + one guard
            # decision (gate and rank cross-encoders). Each step is
            # individually guarded in _warmup — failures warn loudly and
            # startup continues with today's lazy semantics. Runs once per
            # worker process, so a multi-worker deployment warms per worker.
            _warmup(router)
            yield

    app = FastAPI(title="mivia-small-json", version=_version(),
                  lifespan=_lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)

    # -- REST: register routes BEFORE the mount ---------------------------
    def _error(status: int, message: str):
        return JSONResponse({"error": message}, status_code=status)

    async def _read_json_bounded(request: Request):
        """Read one JSON-object request body under the whole-body ceiling
        (the MCP SDK's 4 MiB law): Content-Length is checked up front, and
        the accumulated stream is checked in case the header lies or is
        absent. Returns (payload, None) on success, (None, error) on a cap
        or parse violation.

        Every JSON endpoint goes through this — including the admin key
        route, whose old request.json() was the one unbounded read left
        (review 2026-10-11: an admin body over the cap grew memory without
        bound, the exact hole the ceiling exists to close)."""
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > MAX_BODY_BYTES:
                    return None, _error(
                        413, f"request body exceeds {MAX_BODY_BYTES} bytes")
            except ValueError:
                pass  # a lying Content-Length falls through to streaming
        total = 0
        chunks = []
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                return None, _error(
                    413, f"request body exceeds {MAX_BODY_BYTES} bytes")
            chunks.append(chunk)
        raw = b"".join(chunks)
        # RecursionError (not a ValueError) escapes the 422 law below:
        # ~50k nested brackets blow the parser's C stack and would
        # surface as a 500 (review P1, 2026-10-10).
        try:
            payload = json.loads(raw or b"{}")
        except RecursionError:
            return None, _error(422, "request body nests too deeply")
        except ValueError as exc:
            return None, _error(422, str(exc))
        if not isinstance(payload, dict):
            return None, _error(422, "request body must be a JSON object")
        return payload, None

    async def _core(request: Request, fn):
        """Run one handler with the serve.py law: parse errors are 4xx,
        internal failures are JSON 500s without tracebacks. Sync handlers
        run in the worker threadpool (the event loop never blocks)."""
        try:
            payload, err = await _read_json_bounded(request)
            if err is not None:
                return err
            res = await run_in_threadpool(fn, payload)
            return JSONResponse(res)
        except UnicodeEncodeError:
            # A lone surrogate in the served record makes the JSON render's
            # .encode("utf-8") raise. That is a SERVER-side encoding
            # failure: UnicodeEncodeError is a ValueError subclass, so
            # without this branch it was served as a 422 client error with
            # the codec message in the body (validated 2026-10-10: a
            # \ud800-bearing line through /v1/extract). The pipeline
            # never-raise contract keeps classify itself alive; this is
            # purely about the served status code telling the truth.
            # UnicodeDecodeError (an unreadable REQUEST body) is a
            # different, client-side failure and stays on the 422 law.
            return _error(
                500,
                "response encoding failed: text contains unpaired surrogates")
        except (ValueError, TypeError) as exc:
            return _error(422, str(exc))
        except OverflowError as exc:
            return _error(413, str(exc))
        except ArtifactUnavailable as exc:
            return _error(503, str(exc))
        except Exception as exc:
            return _error(500, f"{type(exc).__name__}: {exc}")

    def _route_ep(payload: dict):
        return do_route(_validated_text(payload), router)

    def _extract_ep(payload: dict):
        return do_extract(_validated_text(payload), router,
                          bool(payload.get("dump", False)))

    def _guard_ep(payload: dict):
        return do_guard(_validated_text(payload, "request"),
                        _normalize_tools(payload.get("tools")), router)

    def _grade_ep(payload: dict):
        anchors = payload.get("anchors")
        return do_grade(_validated_text(payload),
                        anchors if anchors is not None else None)

    def _select_ep(payload: dict):
        return do_select(_validated_text(payload), payload.get("candidates"),
                         payload.get("head", "tool"))

    def _repair_ep(payload: dict):
        raw = _validated_text(payload, "raw_arguments", allow_empty=True)
        return do_repair(raw)

    def _structure_ep(payload: dict):
        return do_structure(_validated_text(payload, "text",
                                            allow_empty=True),
                            payload.get("schema"))

    async def route_ep(request: Request):
        return await _core(request, _route_ep)

    async def extract_ep(request: Request):
        return await _core(request, _extract_ep)

    async def guard_ep(request: Request):
        return await _core(request, _guard_ep)

    async def grade_ep(request: Request):
        return await _core(request, _grade_ep)

    async def select_ep(request: Request):
        return await _core(request, _select_ep)

    async def repair_ep(request: Request):
        return await _core(request, _repair_ep)

    async def structure_ep(request: Request):
        return await _core(request, _structure_ep)

    app.post("/v1/route", name="route")(route_ep)
    app.post("/v1/extract", name="extract")(extract_ep)
    app.post("/v1/guard", name="guard")(guard_ep)
    app.post("/v1/grade", name="grade")(grade_ep)
    app.post("/v1/select", name="select")(select_ep)
    app.post("/v1/repair", name="repair")(repair_ep)
    app.post("/v1/structure", name="structure")(structure_ep)

    # -- Metering plane (exists only when metering is on; spec
    #    per-key-metering sections 10-11) --------------
    if metering_store is not None:
        @app.get("/v1/usage", name="usage")
        async def usage_ep(request: Request):
            """The calling key's own month-to-date counts (never another
            key's); unmetered identities honestly report {"metered":
            false}."""
            ident = request.scope.get("mivia.identity") or {}
            if ident.get("kind") != "key":
                return JSONResponse({"metered": False})
            month = metering_store.month()
            c = metering_store.counts(ident["key_hash"], month)
            paid = ident.get("quota_override") is not None
            key_limit = None if paid else limit
            return JSONResponse({
                "key_id": ident["key_id"], "month": month,
                "ok": c["ok"], "error": c["error"], "reject": c["reject"],
                "limit": key_limit, "reset": metering_store.reset(),
                "quota_exceeded": (not paid and c["ok"] >= limit)})

        @app.post("/v1/admin/keys", name="admin_keys")
        async def admin_keys_ep(request: Request):
            """Key administration: create (plaintext shown ONCE) / list /
            revoke — admin-key guarded (403 for any other identity)."""
            ident = request.scope.get("mivia.identity") or {}
            if ident.get("kind") != "admin":
                return JSONResponse({"error": "admin key required"},
                                    status_code=403)
            payload, err = await _read_json_bounded(request)
            if err is not None:
                return err
            action = payload.get("action")
            if action == "create":
                override = payload.get("quota_override")
                # bool is an int subclass in Python — a JSON `true` must
                # NOT mint a paid key (review 2026-10-10, P3)
                if override is not None and (
                        not isinstance(override, int)
                        or isinstance(override, bool)):
                    return _error(422, "field 'quota_override' must be an "
                                       "integer or null")
                made = metering_store.create_key(quota_override=override)
                return JSONResponse({
                    "key_id": made["key_id"], "key": made["key"],
                    "created_at": made["created_at"],
                    "quota_override": made["quota_override"]})
            if action == "list":
                return JSONResponse({"keys": metering_store.list_keys()})
            if action == "revoke":
                key_id = payload.get("key_id")
                if not isinstance(key_id, str) or not key_id:
                    return _error(422, "field 'key_id' is required")
                res = metering_store.revoke_key(key_id)
                if res is None:
                    return _error(404, f"unknown key_id {key_id!r}")
                return JSONResponse(res)
            return _error(422, "field 'action' must be 'create', 'list' "
                               "or 'revoke'")

    @app.get("/health", name="health")
    async def health():
        from . import encoder, tool_match

        if isinstance(router, encoder.EncoderRouter):
            backend = "onnx-int8"
        elif router is not None:
            backend = "skops"
        else:
            backend = "rules"

        # File existence is not evidence (the bench.py attribution law):
        # a corrupt or tampered artifact would still report "onnx-int8" /
        # "ready" here while every guard/grade request silently degraded
        # to router-only decisions. Report what the process can actually
        # LOAD — the cached singletons; the first probe pays the one-time
        # sha-verified load, in the threadpool, never on the event loop.
        guard_loaded = await run_in_threadpool(
            lambda: tool_match._get_guard() is not None)
        grade_loaded = await run_in_threadpool(
            lambda: tool_match.load_register_gate()[0] is not None
            and tool_match.load_severity_specialist() is not None)
        guard_status = "onnx-int8" if guard_loaded else "unavailable"
        grade_status = "ready" if grade_loaded else "unavailable"
        return JSONResponse({"status": "ok", "model": "mivia-small-json",
                             "backend": backend, "guard": guard_status,
                             "grade": grade_status, "rss_mb": _rss_mb(),
                             "version": _version()})

    # The root mount goes last so REST routes match first; the MCP app
    # serves its own /mcp route exactly (no trailing-slash redirect).
    app.mount("/", mcp_app)

    # Auth wraps everything (REST + MCP); health stays open. The metering
    # middleware sits INSIDE the bearer gate (it only ever sees requests
    # the gate authenticated and identified) and wraps the whole app, so
    # REST and /mcp hit the same counter (spec sections 7 and 12). The
    # inference cap (2026-10-10 load test) sits BETWEEN the bearer gate and
    # the metering middleware — OUTSIDE metering admission, so a request
    # still queued on the cap is not yet admitted and a queued-then-
    # disconnected client is never charged, while a queued-then-served one
    # counts exactly like an unqueued call (see _InferenceGate). When
    # metering is off the gate wraps the bare app.
    inner = app
    if metering_store is not None:
        inner = metering.MeteringMiddleware(app, metering_store, limit)
    gate = _InferenceGate(
        inner, _resolve_inference_cap(max_concurrent_inference))
    return BearerMiddleware(
        gate, token,
        admin_key=(admin_key if metering_store is not None else None),
        key_lookup=(metering_store.lookup
                    if metering_store is not None else None))


def _clean(text) -> str:
    """MCP tool arguments arrive pre-validated by the SDK schema; this only
    enforces the type and the size cap."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > MAX_TEXT_CHARS:
        raise OverflowError(f"text exceeds {MAX_TEXT_CHARS} chars")
    return text


def _version() -> str:
    from . import __version__

    return __version__


def _rss_mb() -> float:
    """Current process RSS in MiB (best effort, -1.0 when unreadable).

    Delegates to serve.rss_mb — ONE reader for both serving surfaces. It
    reports the CURRENT resident set (/proc/self/status VmRSS on Linux;
    the ru_maxrss high-water only as the portable fallback), because
    /health must show memory coming back after a free (2026-10-10 serving
    finding: an 80 MB free left the old ru_maxrss reading at 99.1 MiB
    while VmRSS fell to ~19.5 MiB)."""
    from .serve import rss_mb

    return rss_mb()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mivia_small_json.api",
        description=("mivia-small-json serving API: REST quartet + MCP "
                     "(streamable HTTP) with bearer auth."),
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (use 0.0.0.0 in Docker)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port,
                # One worker: the models are loaded once per process and the
                # 2026-stateless MCP transport needs no sticky sessions; add
                # workers behind a proxy only with per-worker warmup budget.
                workers=1, access_log=False)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
