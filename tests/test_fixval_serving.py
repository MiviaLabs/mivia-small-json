"""Regression tests for the serving findings C1-C4 + N12 of the
2026-10-10 code review (reports/code-review-validation-2026-10-10.md).

Surfaces under test: the stdlib sidecar (``serve.py``) and the stdio MCP
server (``mcp.py``). Parity reference is REST ``api.py`` (fastapi-only —
the two stdlib paths mirror its semantics locally on purpose).

The HTTP surface is driven end-to-end over a Unix-socket server bound to
a short-lived temp dir (no TCP ports, always stopped in teardown).
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import shutil
import socket
import tempfile
import threading

import pytest

from mivia_small_json import mcp, serve

MAX_TEXT_CHARS = 262_144  # api.MAX_TEXT_CHARS, mirrored (api is fastapi-only)


def _sock_dir() -> str:
    # AF_UNIX sun_path caps at 108 bytes; a CI checkout path exceeds it,
    # so the socket binds under TMPDIR (short) and the dir dies per test.
    return tempfile.mkdtemp(prefix="fixval-serving-")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_handler(body: bytes):
    """A serve._Handler stripped to the attributes _read_json touches."""
    h = serve._Handler.__new__(serve._Handler)
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body)
    return h


class _UnixHTTPConnection(http.client.HTTPConnection):
    """http.client over a Unix-domain socket (no TCP port anywhere)."""

    def __init__(self, sock_path: str):
        super().__init__("localhost")
        self._sock_path = sock_path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(15)
        s.connect(self._sock_path)
        self.sock = s


def _sock_path(prefix: str) -> tuple[str, str]:
    """A fresh temp dir and a pre-cleaned socket path inside it."""
    sock_dir = _sock_dir()
    sock_path = os.path.join(sock_dir, f"{prefix}-{os.getpid()}.sock")
    with contextlib.suppress(FileNotFoundError):
        os.unlink(sock_path)
    return sock_dir, sock_path


@pytest.fixture()
def sidecar():
    """A live stdlib sidecar on a temp-dir Unix socket (always stopped)."""
    sock_dir, sock_path = _sock_path("sidecar")
    server = serve._UnixHTTPServer(sock_path, serve._Handler)
    server.router = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = _UnixHTTPConnection(sock_path)

    def post(path: str, raw_body: bytes | None):
        body = raw_body if raw_body is not None else b""
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = json.loads(resp.read())
        return resp.status, payload

    yield post
    conn.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    try:
        os.unlink(sock_path)
    except OSError:
        pass
    shutil.rmtree(sock_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# C1: a JSON array / scalar body must be 400, never an AttributeError crash
# ---------------------------------------------------------------------------

def test_c1_read_json_array_returns_none():
    h = _bare_handler(b'[1, 2, 3]')
    assert serve._Handler._read_json(h) is None


@pytest.mark.parametrize("raw", [b'42', b'"x"', b'null', b'true'])
def test_c1_read_json_scalars_return_none(raw):
    h = _bare_handler(raw)
    assert serve._Handler._read_json(h) is None


def test_c1_read_json_still_parses_objects():
    h = _bare_handler(b'{"text": "hello"}')
    assert serve._Handler._read_json(h) == {"text": "hello"}


@pytest.mark.parametrize("raw", [b'[1, 2, 3]', b'42', b'"x"', b'null'])
def test_c1_http_array_and_scalar_bodies_are_400(sidecar, raw):
    status, body = sidecar("/classify", raw)
    assert status == 400, (status, body)
    assert "error" in body


def test_c1_http_array_body_repair_route_is_400(sidecar):
    status, body = sidecar("/repair_arguments", b'[{"raw_arguments": "x"}]')
    assert status == 400, (status, body)


# ---------------------------------------------------------------------------
# C2: repair-surface parity with REST api.do_repair
# ---------------------------------------------------------------------------

def test_c2_sidecar_repair_requires_raw_arguments():
    with pytest.raises(ValueError):
        serve.handle_repair({})
    # an explicit null is not a string either
    with pytest.raises(ValueError):
        serve.handle_repair({"raw_arguments": None})
    status, body = serve.dispatch("POST", "/repair_arguments", {}, None)
    assert status == 400, (status, body)


def test_c2_sidecar_repair_returns_lossy_flag():
    # substantive input collapses to {} -> the honest lossy signal
    out = serve.handle_repair({"raw_arguments": "garbage not json"})
    assert out == {"repaired": "{}", "lossy": True}
    # a repair that recovers fields is not lossy
    out = serve.handle_repair({"raw_arguments": "{'x': None}"})
    assert out == {"repaired": '{"x":null}', "lossy": False}
    # an already-empty object (raw or fenced) is not lossy
    assert serve.handle_repair({"raw_arguments": "{}"}) == {
        "repaired": "{}", "lossy": False}
    assert serve.handle_repair({"raw_arguments": "  {}  "}) == {
        "repaired": "{}", "lossy": False}
    assert serve.handle_repair({"raw_arguments": "```json\n{}\n```"}) == {
        "repaired": "{}", "lossy": False}
    # empty string repairs to {} without the lossy flag (REST parity)
    assert serve.handle_repair({"raw_arguments": ""}) == {
        "repaired": "{}", "lossy": False}


def test_c2_sidecar_repair_missing_is_400_not_200(sidecar):
    status, body = sidecar("/repair_arguments", b"{}")
    assert status == 400, (status, body)
    assert "raw_arguments" in body["error"]


def test_c2_sidecar_repair_http_shape(sidecar):
    status, body = sidecar(
        "/repair_arguments",
        json.dumps({"raw_arguments": "garbage not json"}).encode())
    assert status == 200, (status, body)
    assert body == {"repaired": "{}", "lossy": True}


def test_c2_stdio_repair_requires_raw_arguments():
    with pytest.raises(ValueError):
        mcp.call_tool("mivia_repair_tool_call", {}, None)
    with pytest.raises(ValueError):
        mcp.call_tool("mivia_repair_tool_call",
                      {"raw_arguments": None}, None)
    resp = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "mivia_repair_tool_call", "arguments": {}}},
        None)
    assert resp["error"]["code"] == -32602, resp
    assert "raw_arguments" in resp["error"]["message"]


def test_c2_stdio_repair_returns_lossy_flag():
    resp = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "mivia_repair_tool_call",
                    "arguments": {"raw_arguments": "garbage not json"}}},
        None)
    assert "error" not in resp
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload == {"repaired": "{}", "lossy": True}
    resp = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "mivia_repair_tool_call",
                    "arguments": {"raw_arguments": "{'x': None}"}}},
        None)
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload == {"repaired": '{"x":null}', "lossy": False}


def test_c2_all_three_surfaces_agree_with_api_do_repair():
    """The stdlib mirrors are semantically identical to the REST reference."""
    pytest.importorskip("fastapi")
    from mivia_small_json.api import do_repair

    samples = ["garbage not json", "{'x': None}", "{}", "  {}  ",
               "```json\n{}\n```", "", "{'a': True,}"]
    for raw in samples:
        rest = do_repair(raw)
        sidecar_out = serve.handle_repair({"raw_arguments": raw})
        resp = mcp.dispatch(
            {"jsonrpc": "2.0", "id": 0, "method": "tools/call",
             "params": {"name": "mivia_repair_tool_call",
                        "arguments": {"raw_arguments": raw}}}, None)
        stdio = json.loads(resp["result"]["content"][0]["text"])
        assert rest == sidecar_out, (raw, rest, sidecar_out)
        assert rest == stdio, (raw, rest, stdio)


# ---------------------------------------------------------------------------
# C3: the 262,144-char cap on the stdio surface (REST: OverflowError -> 413)
# ---------------------------------------------------------------------------

def test_c3_stdio_text_cap_raises_valueerror():
    with pytest.raises(ValueError):
        mcp._required_text({"text": "x" * (MAX_TEXT_CHARS + 1)})
    # at-cap is accepted
    assert mcp._required_text({"text": "x" * MAX_TEXT_CHARS}) \
        == "x" * MAX_TEXT_CHARS


def test_c3_stdio_oversized_text_is_minus_32602():
    resp = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "mivia_classify",
                    "arguments": {"text": "x" * (MAX_TEXT_CHARS + 1)}}},
        None)
    assert resp["error"]["code"] == -32602, resp
    assert "262144" in resp["error"]["message"]


def test_c3_stdio_raw_arguments_cap():
    with pytest.raises(ValueError):
        mcp.call_tool("mivia_repair_tool_call",
                      {"raw_arguments": "x" * (MAX_TEXT_CHARS + 1)}, None)
    resp = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "mivia_repair_tool_call",
                    "arguments": {"raw_arguments":
                                  "x" * (MAX_TEXT_CHARS + 1)}}},
        None)
    assert resp["error"]["code"] == -32602, resp
    assert "262144" in resp["error"]["message"]


def test_c3_sidecar_repair_oversized_is_413():
    status, body = serve.dispatch(
        "POST", "/repair_arguments",
        {"raw_arguments": "x" * (MAX_TEXT_CHARS + 1)}, None)
    assert status == 413, (status, body)
    assert "262144" in body["error"]


# ---------------------------------------------------------------------------
# C4: a JSON-RPC request without 'method' is -32600, not an AttributeError
# ---------------------------------------------------------------------------

def test_c4_missing_method_is_invalid_request():
    resp = mcp.dispatch({"jsonrpc": "2.0", "id": 9}, None)
    assert resp["error"]["code"] == -32600, resp
    assert resp["id"] == 9


def test_c4_missing_method_and_id_is_invalid_request_id_null():
    resp = mcp.dispatch({"jsonrpc": "2.0"}, None)
    assert resp["error"]["code"] == -32600, resp
    assert resp["id"] is None


def test_c4_handle_line_missing_method_is_32600_not_32603():
    out = mcp.handle_line('{"jsonrpc": "2.0", "id": 4}', None)
    assert out is not None
    resp = json.loads(out)
    assert resp["error"]["code"] == -32600, resp
    assert resp["id"] == 4


def test_c4_notification_semantics_unchanged():
    # notifications (no id) still get no response
    assert mcp.dispatch({"jsonrpc": "2.0",
                         "method": "notifications/whatever"}, None) is None
    assert mcp.dispatch({"jsonrpc": "2.0", "method": "anything"}, None) is None
    assert mcp.dispatch({"jsonrpc": "2.0",
                         "method": "notifications/initialized"}, None) is None
    # known notifications with an id still get no response
    assert mcp.dispatch({"jsonrpc": "2.0", "id": 1,
                         "method": "notifications/initialized"}, None) is None
    # ping still answers, unknown request still -32601, non-dict still -32600
    resp = mcp.dispatch({"jsonrpc": "2.0", "id": 7, "method": "ping"}, None)
    assert resp == {"jsonrpc": "2.0", "id": 7, "result": {}}
    assert mcp.dispatch({"jsonrpc": "2.0", "id": 8,
                         "method": "nope"}, None)["error"]["code"] == -32601
    assert mcp.dispatch([1, 2], None)["error"]["code"] == -32600


# ---------------------------------------------------------------------------
# N12: over-cap text is an explicit 413, bad bodies stay 400
# ---------------------------------------------------------------------------

def test_n12_read_json_raises_body_too_large():
    body = json.dumps({"text": "x" * (MAX_TEXT_CHARS + 1)}).encode()
    h = _bare_handler(body)
    with pytest.raises(serve._BodyTooLarge) as excinfo:
        serve._Handler._read_json(h)
    assert "262144" in str(excinfo.value)


def test_n12_oversized_text_is_413_with_size_message(sidecar):
    body = json.dumps({"text": "x" * (MAX_TEXT_CHARS + 1)}).encode()
    status, payload = sidecar("/classify", body)
    assert status == 413, (status, str(payload)[:120])
    assert "262144" in payload["error"]
    assert "JSON object" not in payload["error"]


def test_n12_oversized_dump_text_is_413(sidecar):
    body = json.dumps({"text": "x" * (MAX_TEXT_CHARS + 1)}).encode()
    status, payload = sidecar("/classify_dump", body)
    assert status == 413, (status, str(payload)[:120])


def test_n12_oversized_non_text_field_is_413(sidecar):
    body = json.dumps({"other": "x" * (MAX_TEXT_CHARS + 1)}).encode()
    status, payload = sidecar("/classify", body)
    assert status == 413, (status, str(payload)[:120])


def test_n12_malformed_body_stays_400_bad_body(sidecar):
    status, payload = sidecar("/classify", b"not json at all")
    assert status == 400, (status, payload)
    assert "JSON object" in payload["error"]


def test_n12_at_cap_text_still_classifies(sidecar):
    body = json.dumps({"text": "x" * MAX_TEXT_CHARS}).encode()
    status, payload = sidecar("/classify", body)
    assert status == 200, (status, str(payload)[:120])
    assert "result" in payload


# ---------------------------------------------------------------------------
# Concurrency cap (2026-10-10 serving finding): ThreadingMixIn spawned one
# thread per connection with no bound — 30 concurrent 0.4 s requests
# reached 28 live handler threads. The sidecar now bounds the
# concurrently-ACTIVE handler dispatches with a semaphore (module-level
# default; MIVIA_SIDECAR_MAX_HANDLERS or the server's
# ``max_concurrent_handlers=`` constructor argument override). Connections
# beyond the cap stay accepted; their handlers queue on the semaphore.
# ---------------------------------------------------------------------------

class TestConcurrencyCap:
    def test_env_and_constructor_override(self, monkeypatch):
        """Explicit argument > env var > module default, floored at 1."""
        monkeypatch.setenv("MIVIA_SIDECAR_MAX_HANDLERS", "3")
        assert serve._resolve_max_handlers(None) == 3
        monkeypatch.setenv("MIVIA_SIDECAR_MAX_HANDLERS", "not-a-number")
        assert serve._resolve_max_handlers(None) \
            == serve.DEFAULT_MAX_CONCURRENT_HANDLERS
        monkeypatch.setenv("MIVIA_SIDECAR_MAX_HANDLERS", "0")
        assert serve._resolve_max_handlers(None) == 1  # the floor is 1
        monkeypatch.setenv("MIVIA_SIDECAR_MAX_HANDLERS", "99")
        assert serve._resolve_max_handlers(5) == 5  # explicit beats env

    def test_bounded_server_carries_the_cap(self):
        """The server carries the resolved cap and that many slots."""
        sock_dir, sock_path = _sock_path("cap-attrs")
        server = serve._UnixHTTPServer(sock_path, serve._Handler,
                                       max_concurrent_handlers=2)
        try:
            assert server.max_concurrent_handlers == 2
            held = 0
            try:
                while server.handler_slots.acquire(blocking=False):
                    held += 1
                    if held > 2:
                        break
            finally:
                for _ in range(held):
                    server.handler_slots.release()
            assert held == 2, "the server must expose exactly cap slots"
        finally:
            server.server_close()
            try:
                os.unlink(sock_path)
            except OSError:
                pass
            shutil.rmtree(sock_dir, ignore_errors=True)

    def test_active_dispatches_never_exceed_the_cap(self, monkeypatch):
        """Deterministic cap mechanism: dispatch is instrumented with a
        counter of simultaneously-active handlers; the test handler parks
        on an event. The cap holders park inside dispatch; queued
        connections must wait at the semaphore. Any dispatch entry beyond
        the cap parks BEFORE decrementing, so k entries would trip
        ``all_entered`` — impossible under the semaphore, guaranteed under
        the uncapped ThreadingMixIn (red at HEAD). No timing asserts."""
        cap = 2
        k = cap + 3
        sock_dir, sock_path = _sock_path("cap-live")
        server = serve._UnixHTTPServer(sock_path, serve._Handler,
                                       max_concurrent_handlers=cap)
        server.router = None
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        lock = threading.Lock()
        state = {"active": 0, "max_active": 0}
        cap_reached = threading.Event()
        all_entered = threading.Event()
        release = threading.Event()
        real_dispatch = serve.dispatch

        def parked_dispatch(method, path, payload, router):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"],
                                          state["active"])
                if state["active"] >= cap:
                    cap_reached.set()
                if state["active"] >= k:
                    all_entered.set()
            release.wait(timeout=10)
            with lock:
                state["active"] -= 1
            return real_dispatch(method, path, payload, router)

        monkeypatch.setattr(serve, "dispatch", parked_dispatch)

        results = [None] * k

        def drive(i: int, conn) -> None:
            try:
                conn.request("GET", "/health")
                resp = conn.getresponse()
                results[i] = resp.status
                resp.read()
            except Exception as exc:  # surfaced by the final status assert
                results[i] = exc

        conns: list[_UnixHTTPConnection] = []
        workers = []
        try:
            for _ in range(k):
                conns.append(_UnixHTTPConnection(sock_path))
            for i in range(k):
                t = threading.Thread(target=drive, args=(i, conns[i]))
                t.start()
                workers.append(t)
            # The cap slots fill and park inside dispatch.
            assert cap_reached.wait(timeout=10), "cap slots never filled"
            # While the slot holders park, the queued connections must not
            # raise the active count above the cap.
            assert all_entered.wait(timeout=1.0) is False, (
                f"{state['max_active']} dispatches were active at once; "
                f"the cap ({cap}) is not enforced")
            assert state["max_active"] <= cap
            release.set()
            for t in workers:
                t.join(timeout=10)
            # Every queued request was eventually served, none dropped.
            assert all(r == 200 for r in results), results
            assert state["max_active"] <= cap
        finally:
            release.set()
            for conn in conns:
                try:
                    conn.close()
                except Exception:
                    pass
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            try:
                os.unlink(sock_path)
            except OSError:
                pass
            shutil.rmtree(sock_dir, ignore_errors=True)
