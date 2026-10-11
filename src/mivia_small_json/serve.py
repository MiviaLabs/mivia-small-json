# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Local sidecar daemon: JSON classification over HTTP or a Unix socket.

Stdlib only (``http.server``) — no web framework, no heavy dependency —
so the daemon stays inside the < 2 GB RAM budget next to the int8
artifacts. The router (ONNX encoder when present, else skops) loads once
at startup and is shared by every request. Concurrently-active handler
dispatches are capped (``DEFAULT_MAX_CONCURRENT_HANDLERS``; override with
``MIVIA_SIDECAR_MAX_HANDLERS`` or the server's ``max_concurrent_handlers=``
constructor argument).

Usage::

    python -m mivia_small_json.serve --port 8000
    python -m mivia_small_json.serve --socket /tmp/mivia.sock

Endpoints:

- ``POST /classify``: ``{"text": "..."}`` ->
  ``{"result": {...}, "meta": {...}}``
- ``POST /classify_dump``: ``{"text": "..."}`` ->
  ``{"records": [{"result": ..., "meta": ...}, ...]}``
- ``POST /repair_arguments``: ``{"raw_arguments": "..."}`` ->
  ``{"repaired": "...", "lossy": bool}`` (same shape and lossy semantics
  as REST ``api.do_repair``)
- ``GET /health``: ``{"status": "ok", "rss_mb": ..., "model": ...}``
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as _ThreadingHTTPServer
from typing import Any

from . import pipeline
from .tool_call import _strip_fences, repair_arguments

_ROUTES = ("POST /classify", "POST /classify_dump",
           "POST /repair_arguments", "GET /health")

# Fixed short benign text for the startup warmup — one classify that loads
# and first-runs the router artifact before the accept loop, so no client
# ever observes the cold-start spike (spec
# guard-serve-performance). The sidecar serves classify
# surfaces only (no guard endpoint), so it deliberately does NOT warm the
# guard — that would load artifacts this process never uses.
_WARMUP_TEXT = "2026-03-01T12:00:00Z ERROR [auth]: token expired"


def warmup(router) -> None:
    """One classify before the accept loop; failure warns, never fatal.

    A broken artifact must degrade to the lazy path's per-request error
    semantics, never prevent startup (the health endpoint keeps answering).
    """
    try:
        pipeline.classify(_WARMUP_TEXT, router)
    except Exception as exc:
        print(f"[mivia-small-json] sidecar warmup step 'router classify' "
              f"failed ({type(exc).__name__}: {exc}); the first request "
              "will pay the lazy load instead", file=sys.stderr)


def build_router():
    """Load the shipped router once (ONNX encoder, else skops, else None)."""
    try:
        return pipeline.load_active_router()
    except Exception:
        return None


def model_name(router) -> str:
    """Short artifact label for the health probe."""
    from . import encoder

    if isinstance(router, encoder.EncoderRouter):
        return "onnx-int8"
    if router is not None:
        return "skops"
    return "rules"


MAX_BODY_BYTES = 4 * 1024 * 1024   # whole-body ceiling (matches api.py)
MAX_TEXT_CHARS = 262_144           # per-string-field cap (api.MAX_TEXT_CHARS;
# api.py is fastapi-only and this sidecar is stdlib — the number is mirrored
# here on purpose; keep the two in sync)

# Cap on concurrently-ACTIVE handler dispatches (2026-10-10 serving
# finding: ThreadingMixIn spawns one thread per connection with no bound —
# 30 concurrent 0.4 s requests reached 28 live handler threads). A
# semaphore around dispatch bounds the active set; connections beyond the
# cap stay accepted (kernel backlog) and their handlers queue on the
# semaphore instead of each pinning CPU + stack. Override per deployment
# with ``MIVIA_SIDECAR_MAX_HANDLERS`` or per server with the
# ``max_concurrent_handlers=`` constructor argument.
DEFAULT_MAX_CONCURRENT_HANDLERS = 8


def _resolve_max_handlers(value: int | None = None) -> int:
    """Explicit argument > ``MIVIA_SIDECAR_MAX_HANDLERS`` > module default,
    floored at 1 (garbage falls back to the default, never crashes boot)."""
    if value is None:
        raw = os.environ.get("MIVIA_SIDECAR_MAX_HANDLERS", "").strip()
        try:
            value = int(raw) if raw else DEFAULT_MAX_CONCURRENT_HANDLERS
        except ValueError:
            value = DEFAULT_MAX_CONCURRENT_HANDLERS
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_CONCURRENT_HANDLERS
    return max(1, value)


class _BodyTooLarge(Exception):
    """A parsed string field exceeds MAX_TEXT_CHARS (the handler maps this
    to HTTP 413 — distinct from a malformed body, which stays 400;
    review N12, 2026-10-10)."""


def _current_rss_kib() -> float | None:
    """The CURRENT resident set in KiB from /proc/self/status (Linux),
    ``None`` when unavailable (non-Linux, container without procfs)."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1])  # reported in kB == KiB
    except (OSError, ValueError):
        return None
    return None


def rss_mb() -> float:
    """Current process RSS in MiB (best effort, -1.0 when unreadable).

    The value must be the CURRENT resident set, never the ru_maxrss
    high-water: /health has to show memory coming back after a free
    (2026-10-10 serving finding — after freeing an 80 MB block the old
    ru_maxrss reading stayed at 99.1 MiB while /proc VmRSS fell from
    ~101,880 KiB to ~19,956 KiB). Linux reads /proc/self/status VmRSS;
    elsewhere the reader falls back to ru_maxrss (KiB on Linux, BYTES on
    macOS — the old unit law, kept for the fallback path)."""
    kib = _current_rss_kib()
    if kib is None:
        try:
            import resource
            import sys

            # ru_maxrss is KiB on Linux but BYTES on macOS (darwin).
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            kib = ru if sys.platform != "darwin" else ru / 1024.0
        except Exception:
            return -1.0
    return round(kib / 1024.0, 1)


def _required_text(payload: dict, field: str = "text") -> str:
    """Non-blank text field, present-and-typed (the api.py law).

    A missing key (typo'd field) or blank text must be a 400, not a silent
    classification of "" into a general-text record (2026-10-09 serving
    probe: ``{"txt": "..."}`` returned 200 general text)."""
    if field not in payload:
        raise ValueError(f"field {field!r} is required")
    text = payload[field]
    if not isinstance(text, str):
        raise ValueError(f"field {field!r} must be a string")
    if not text.strip():
        raise ValueError(f"field {field!r} must be a non-empty string")
    return text


def handle_classify(payload: dict, router) -> dict:
    """Classify one text: ``{"text": ...}`` -> ``{"result", "meta"}``."""
    obj, meta = pipeline.classify(_required_text(payload), router)
    return {"result": obj, "meta": meta}


def handle_classify_dump(payload: dict, router) -> dict:
    """Parse a multi-line dump into ``{"records": [...]}``."""
    records = [{"result": obj, "meta": meta}
               for obj, meta in
               pipeline.classify_dump(_required_text(payload), router)]
    return {"records": records}


def handle_repair(payload: dict) -> dict:
    """Repair pseudo-JSON tool arguments: ``{"repaired", "lossy"}``.

    Parity with REST ``api.do_repair`` (the reference; api.py is
    fastapi-only so the lossy computation below is mirrored here on
    purpose — keep the two in sync): a missing/non-string
    ``raw_arguments`` is a 400 (REST: 422), never a silent repair of ""
    into ``{}``, and ``lossy: true`` flags a repair that collapsed a
    substantive input to an empty object.
    """
    if "raw_arguments" not in payload:
        raise ValueError("field 'raw_arguments' is required")
    raw = payload["raw_arguments"]
    if not isinstance(raw, str):
        raise ValueError("field 'raw_arguments' must be a string")
    if len(raw) > MAX_TEXT_CHARS:
        raise OverflowError(f"raw_arguments exceeds {MAX_TEXT_CHARS} chars")

    # --- lossy computation mirrored from api.do_repair (reference) --------
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
    # ----------------------------------------------------------------------
    return {"repaired": repaired, "lossy": lossy}


def health_payload(router) -> dict:
    """Liveness probe body (no classification, no router inference)."""
    return {"status": "ok", "rss_mb": rss_mb(), "model": model_name(router)}


def dispatch(method: str, path: str, payload: dict, router) -> tuple[int, dict]:
    """Route one request to a handler. Returns (http_status, body)."""
    try:
        if method == "GET" and path == "/health":
            return 200, health_payload(router)
        if method == "POST" and path == "/classify":
            return 200, handle_classify(payload, router)
        if method == "POST" and path == "/classify_dump":
            return 200, handle_classify_dump(payload, router)
        if method == "POST" and path == "/repair_arguments":
            return 200, handle_repair(payload)
        return 404, {"error": f"unknown route {method} {path}",
                     "routes": list(_ROUTES)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except OverflowError as exc:
        # REST parity: an over-cap field is 413 with the size message
        # (api.do_repair / api._validated_text raise OverflowError too).
        return 413, {"error": str(exc)}
    except Exception as exc:  # never leak a traceback as a 500 HTML page
        return 500, {"error": f"{type(exc).__name__}: {exc}"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "mivia-small-json/0.1.0"
    timeout = 30  # never let a half-open socket pin a worker thread

    def _send(self, status: int, body: dict) -> None:
        blob = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _read_json(self) -> dict | None:
        """Parsed body object, ``None`` on any parse failure (the caller
        maps that to 400 — a bad body must not masquerade as an
        empty-but-valid request), or raises ``_BodyTooLarge`` when a string
        field exceeds MAX_TEXT_CHARS (the caller maps that to 413; review
        N12 — the size refusal must not be indistinguishable from a bad
        body)."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0:
            return None
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, RecursionError):
            # RecursionError: deeply-nested JSON must drop the request
            # (None -> 4xx path), never kill the sidecar (review P1).
            return None
        # C1 (review 2026-10-10): the dict check MUST come before any
        # attribute access on the parsed value — a JSON array or scalar
        # (``[1,2,3]``, ``42``, ``"x"``, ``null``) is a 400, never an
        # AttributeError escaping to the handler loop.
        if not isinstance(parsed, dict):
            return None
        for key, v in parsed.items():
            # Parity with api.MAX_TEXT_CHARS (mirrored above on purpose):
            # multi-MB text must not reach the tokenizer on the <2 GB path
            # (review P2, 2026-10-10).
            if isinstance(v, str) and len(v) > MAX_TEXT_CHARS:
                raise _BodyTooLarge(
                    f"field {key!r} exceeds {MAX_TEXT_CHARS} chars")
        return parsed

    def _serve(self) -> None:
        path = self.path.split("?", 1)[0]
        if self.command == "POST":
            if self.headers.get("Transfer-Encoding"):
                # Chunked bodies would be read as raw bytes against a
                # missing/lying Content-Length; refuse rather than guess.
                self._send(411, {"error": "chunked bodies not supported; "
                                          "send Content-Length"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0:
                # A POST without a body is not an empty-but-valid request.
                self._send(400, {"error": "request body must be a JSON "
                                          "object"})
                return
            if length > MAX_BODY_BYTES:
                # Drain (bounded, discarded) so the client can finish
                # writing before the 413 — an abrupt close would break its
                # pipe before it ever reads our status.
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                self._send(413, {"error": f"request body exceeds "
                                          f"{MAX_BODY_BYTES} bytes"})
                return
            try:
                payload = self._read_json()
            except _BodyTooLarge as exc:
                # N12 (review 2026-10-10): over-cap fields answer 413 with
                # an explicit size message — the misleading
                # "must be a JSON object" 400 hid "too big" from clients.
                self._send(413, {"error": str(exc)})
                return
            if payload is None:
                self._send(400, {"error": "request body must be a JSON "
                                          "object"})
                return
        else:
            payload = {}
        # Concurrency cap: at most server.max_concurrent_handlers dispatches
        # run at once; connections beyond it queue on the semaphore (a plain
        # server without handler_slots keeps the uncapped behavior).
        slots = getattr(self.server, "handler_slots", None)
        if slots is None:
            status, body = dispatch(self.command, path, payload,
                                    self.server.router)
        else:
            with slots:
                status, body = dispatch(self.command, path, payload,
                                        self.server.router)
        self._send(status, body)

    do_GET = _serve
    do_POST = _serve

    def log_message(self, *args: Any) -> None:  # keep stdout for MCP-style use
        pass


class _BoundedHandlersMixin:
    """Shared concurrency-cap plumbing for the threaded servers.

    ``handler_slots`` bounds how many handlers run ``dispatch`` at once
    (``_Handler`` acquires it around dispatch). The cap is the explicit
    ``max_concurrent_handlers=`` argument, else ``MIVIA_SIDECAR_MAX_HANDLERS``,
    else ``DEFAULT_MAX_CONCURRENT_HANDLERS``."""

    def __init__(self, *args, max_concurrent_handlers: int | None = None,
                 **kwargs):
        cap = _resolve_max_handlers(max_concurrent_handlers)
        super().__init__(*args, **kwargs)
        self.max_concurrent_handlers = cap
        self.handler_slots = threading.BoundedSemaphore(cap)


class ThreadingHTTPServer(_BoundedHandlersMixin, _ThreadingHTTPServer):
    """The stdlib threaded server with the dispatch cap applied by default.

    Deliberately bound to the module-level name ``ThreadingHTTPServer``:
    ``run()`` keeps constructing it exactly like before, so deployments and
    tests that substitute this name (positional ``(address, handler)``
    construction) keep working."""


class _UnixHTTPServer(_BoundedHandlersMixin,
                      socketserver.ThreadingMixIn,
                      socketserver.UnixStreamServer):
    """Threaded HTTP over a Unix-domain socket (same handler, no TCP port).

    ThreadingMixIn: one slow client must not stall the sidecar. The
    handler-dispatch concurrency cap applies here too (the mixin). The
    daemon socket is chmod'd 0600 after bind — it inherits umask
    otherwise, and a world-writable control socket is a local privilege
    boundary.
    """

    allow_reuse_address = True
    daemon_threads = True

    def server_bind(self):
        super().server_bind()
        try:
            os.chmod(self.server_address, 0o600)
        except OSError:
            pass

    def get_request(self):  # BaseHTTPRequestHandler needs (socket, address)
        conn, _ = self.socket.accept()
        return conn, ("local", 0)


def run(port: int = 8000, socket_path: str | None = None) -> int:
    """Serve forever on TCP ``port`` or Unix ``socket_path``."""
    from .bootstrap import register_all_schemas

    register_all_schemas()  # the service emits all schemas, not just built-ins
    router = build_router()
    # Warm between router load and the accept loop: no connection is ever
    # accepted cold (warmup warns and continues on failure, never fatal).
    warmup(router)
    if socket_path:
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        server = _UnixHTTPServer(socket_path, _Handler)
    else:
        server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.router = router
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mivia_small_json.serve",
        description="Local mivia-small-json sidecar daemon (HTTP/Unix socket).",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--socket", default=None,
                        help="Unix socket path (instead of TCP --port)")
    args = parser.parse_args(argv)
    return run(port=args.port, socket_path=args.socket)


if __name__ == "__main__":
    import sys

    sys.exit(main())
