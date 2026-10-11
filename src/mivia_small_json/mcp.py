# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Native Model Context Protocol (MCP) server over stdio (JSON-RPC 2.0).

Stdlib only — no MCP SDK dependency. Any MCP-compatible coding assistant
(Claude Code, Cursor, Windsurf, ...) can use ``mivia-small-json`` as a
tool server::

    python -m mivia_small_json.mcp

Protocol: one JSON-RPC 2.0 message per line on stdin, one response per
line on stdout. Requests carry ``id``; notifications (no ``id``) get no
response. Tools exposed:

- ``mivia_classify``: ``{"text": ...}`` -> the routed ``agent_step`` /
  ``tool_call`` / ``log_event`` / ... record plus routing meta.
- ``mivia_classify_dump``: ``{"text": ...}`` -> multi-line stream records.
- ``mivia_repair_tool_call``: ``{"raw_arguments": ...}`` -> repaired JSON
  plus the ``lossy`` flag (same semantics as REST ``api.do_repair``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import pipeline
from .tool_call import _strip_fences, repair_arguments

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "mivia-small-json", "version": "0.1.0"}
MAX_TEXT_CHARS = 262_144   # per-string cap (api.MAX_TEXT_CHARS); api.py is
# fastapi-only and this stdio path is stdlib — mirrored here on purpose.

_TOOLS = (
    {
        "name": "mivia_classify",
        "description": ("Classify unstructured software engineering text "
                        "(issue, log, diff, tool call, agent step, ...) "
                        "into a strictly valid JSON record."),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "mivia_classify_dump",
        "description": ("Parse a multi-line stream: strip conversational "
                        "preambles, group stack-trace continuations, keep "
                        "atomic entities whole."),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "mivia_repair_tool_call",
        "description": ("Repair broken LLM tool-call JSON arguments "
                        "(single quotes, Python literals, trailing commas)."),
        "inputSchema": {
            "type": "object",
            "properties": {"raw_arguments": {"type": "string"}},
            "required": ["raw_arguments"],
        },
    },
)


def _text_block(payload: Any) -> dict:
    return {"type": "text",
            "text": json.dumps(payload, ensure_ascii=False)}


def _required_text(arguments: dict) -> str:
    """Non-blank ``text`` argument (the api.py law): a missing or blank
    text is a -32602 tool error, not a silent classification of ""."""
    text = arguments.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("field 'text' must be a non-empty string")
    if len(text) > MAX_TEXT_CHARS:
        # C3 (review 2026-10-10): REST caps text at 262,144 chars
        # (_validated_text, OverflowError -> 413); stdio maps the same cap
        # to -32602 so the caller resends something smaller.
        raise ValueError(f"field 'text' exceeds {MAX_TEXT_CHARS} chars")
    return text


def call_tool(name: str, arguments: dict, router) -> dict:
    """Run one tool call. Returns the MCP ``result`` object (or raises)."""
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    if name == "mivia_classify":
        text = _required_text(arguments)
        obj, meta = pipeline.classify(text, router)
        return {"content": [_text_block({"result": obj, "meta": meta})]}
    if name == "mivia_classify_dump":
        text = _required_text(arguments)
        records = [{"result": obj, "meta": meta}
                   for obj, meta in pipeline.classify_dump(text, router)]
        return {"content": [_text_block({"records": records})]}
    if name == "mivia_repair_tool_call":
        # C2 (review 2026-10-10): parity with REST api.do_repair — the
        # argument is required-and-string (no silent repair of a missing
        # value into "{}"), carries the 262,144-char cap, and the result
        # includes the lossy flag. api.py is fastapi-only, so the lossy
        # computation below is mirrored locally on purpose (reference:
        # api.do_repair) — keep the two in sync.
        if "raw_arguments" not in arguments:
            raise ValueError("field 'raw_arguments' is required")
        raw = arguments["raw_arguments"]
        if not isinstance(raw, str):
            raise ValueError("field 'raw_arguments' must be a string")
        if len(raw) > MAX_TEXT_CHARS:
            raise ValueError(
                f"raw_arguments exceeds {MAX_TEXT_CHARS} chars")
        repaired = repair_arguments(raw)
        lossy = False
        stripped = raw.strip()
        unfenced = _strip_fences(stripped).strip()
        is_empty_object = (
            unfenced.startswith("{")
            and unfenced.endswith("}")
            and not unfenced[1:-1].strip()
        )
        if repaired == "{}" and bool(stripped) and not is_empty_object:
            lossy = True
        return {"content": [_text_block(
            {"repaired": repaired, "lossy": lossy})]}
    raise KeyError(f"unknown tool {name!r}")


def dispatch(request: Any, router) -> dict | None:
    """Handle one decoded JSON-RPC message. ``None`` = no response."""
    if not isinstance(request, dict):
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "Invalid Request"}}
    method = request.get("method")
    req_id = request.get("id")
    # C4 (review 2026-10-10): a missing 'method' must be decided BEFORE any
    # method.startswith() use — the old code hit the AttributeError first
    # and the read loop mis-reported it as -32603 with id:null, while the
    # `elif method is None` branch further down was dead code. Any
    # non-string method (JSON-RPC requires a String) is Invalid Request by
    # the same token — otherwise `{"id": 7, "method": 42}` escaped as the
    # same mis-attributed -32603 id:null (review-fixes validation).
    if not isinstance(method, str):
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request"}}
    params = request.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    if method == "initialize":
        result = {"protocolVersion": PROTOCOL_VERSION,
                  "capabilities": {"tools": {}},
                  "serverInfo": SERVER_INFO}
    elif method == "ping":
        # JSON-RPC level liveness probe (MCP spec): an empty result, never
        # a "method not found" error.
        result = {}
    elif method.startswith("notifications/") or req_id is None:
        # A notification (no id) gets no response — an unknown notification
        # is dropped silently, not answered with a malformed id:null error.
        return None
    elif method in ("notifications/initialized", "notifications/cancelled"):
        return None
    elif method == "tools/list":
        result = {"tools": list(_TOOLS)}
    elif method == "tools/call":
        try:
            result = call_tool(params.get("name", ""),
                               params.get("arguments", {}), router)
        except KeyError as exc:
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32602, "message": str(exc)}}
        except (ValueError, TypeError) as exc:
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32602, "message": str(exc)}}
    else:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601,
                          "message": f"Method not found: {method}"}}
    if req_id is None:  # a notification that reached a result branch
        return None
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def handle_line(line: str, router) -> str | None:
    """Decode one stdin line, dispatch, and encode the response (if any)."""
    try:
        request = json.loads(line)
    except (ValueError, RecursionError):
        # RecursionError: pathologically deep nesting blows the interpreter
        # stack inside json.loads (it is a RuntimeError, not a ValueError).
        # REST maps the same input to 422 (a bad request), so stdio answers
        # the same law: -32700 Parse error — never a -32603 traceback name.
        return json.dumps({"jsonrpc": "2.0", "id": None,
                           "error": {"code": -32700,
                                     "message": "Parse error"}})
    response = dispatch(request, router)
    if response is None:
        return None
    return json.dumps(response, ensure_ascii=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mivia_small_json.mcp",
        description="mivia-small-json MCP server (stdio JSON-RPC 2.0).",
    )
    parser.parse_args(argv)
    from .bootstrap import register_all_schemas

    register_all_schemas()  # the service emits all schemas, not just built-ins
    try:
        router = pipeline.load_active_router()
    except Exception:
        router = None
    stdout = sys.stdout
    # The never-fail contract covers the ITERATION itself: stdin is read as
    # raw bytes and decoded leniently, so a lone bad byte degrades to U+FFFD
    # (or one -32603 reply) instead of killing the server mid-session.
    stdin = getattr(sys.stdin, "buffer", None) or sys.stdin
    while True:
        try:
            raw = stdin.readline()
            if not raw:
                break
            line = (raw.decode("utf-8", "replace")
                    if isinstance(raw, bytes) else raw)
            if not line.strip():
                continue
            out = handle_line(line, router)
        except Exception as exc:  # the stdio loop never dies on one message
            out = json.dumps({"jsonrpc": "2.0", "id": None,
                              "error": {"code": -32603,
                                        "message": f"{type(exc).__name__}: {exc}"}})
        if out is not None:
            # The write sits in its own guard: the never-fail contract above
            # covers reading/dispatch, this one covers the reply. Without it
            # a client that closes the pipe without reading the reply kills
            # the whole server with a BrokenPipeError traceback.
            try:
                stdout.write(out + "\n")
                stdout.flush()
            except BrokenPipeError:
                # The client closed the pipe without reading the reply; its
                # stdin is gone with it, so the session is over either way.
                # Exit gracefully (0, no traceback). Point stdout at
                # devnull so the interpreter's shutdown flush cannot emit a
                # spurious broken-pipe warning either.
                try:
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, stdout.fileno())
                    os.close(devnull)
                except Exception:
                    pass
                return 0
            except UnicodeEncodeError:
                # A lone surrogate (a legal \udXXX escape in the request)
                # renders into the reply because replies are serialized
                # with ensure_ascii=False, and a strict-UTF-8 pipe refuses
                # the raw surrogate. Re-emit the same reply with the
                # surrogate written as its JSON \udXXX escape — exactly
                # what ensure_ascii=True would have produced — instead of
                # silently dropping the response for that id (the blanket
                # guard below used to swallow it and the client waited
                # forever).
                stdout.write(
                    (out + "\n").encode("utf-8", "backslashreplace").decode())
                stdout.flush()
            except Exception:
                # Any other write error follows the same never-crash
                # contract: drop the response, keep reading stdin.
                continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
