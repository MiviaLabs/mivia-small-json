"""Regression tests for the 2026-10-09 serving review (code-review §9).

One test group per finding F1-F8: Host-header normalization (F1), MCP
Origin allowlist derivation (F2), non-ASCII bearer crash (F3), anchor cap
(F4), REST body ceiling (F5), cached guard in do_select (F6), the stdlib
sidecar hardening (F7), and the stdio MCP loop fixes (F8).
"""

from __future__ import annotations

import io
import json
import socketserver
import stat
import types

import pytest


# ---------------------------------------------------------------------------
# F1: Host allowlist must accept bracketed IPv6 with port and any case
# ---------------------------------------------------------------------------

def test_f1_host_allowed_variants():
    from mivia_small_json.api import _host_allowed

    assert _host_allowed("localhost:8000")
    assert _host_allowed("127.0.0.1:8000")
    assert _host_allowed("[::1]:8000")
    assert _host_allowed("LOCALHOST:8000")
    assert _host_allowed("[::1]")
    assert _host_allowed("localhost")
    assert not _host_allowed("evil.com")
    assert not _host_allowed("evil.com:8000")
    assert not _host_allowed(None)


@pytest.mark.parametrize("host", ["localhost:8000", "127.0.0.1:8000",
                                  "[::1]:8000", "LOCALHOST:8000"])
def test_f1_health_open_on_loopback_variants(host):
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    client = TestClient(api.create_app(token="", router=object()))
    r = client.get("/health", headers={"Host": host})
    assert r.status_code == 200, (host, r.status_code)


def test_f1_health_still_421_on_foreign_host():
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    client = TestClient(api.create_app(token="", router=object()))
    r = client.get("/health", headers={"Host": "evil.com:8000"})
    assert r.status_code == 421


# ---------------------------------------------------------------------------
# F2: MCP Origin allowlist derived from allowed_hosts
# ---------------------------------------------------------------------------

def _post_mcp(client, origin):
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream",
               "Host": "127.0.0.1"}
    if origin:
        headers["Origin"] = origin
    return client.post(
        "/mcp", content=b'{"jsonrpc": "2.0", "id": 1, "method": "ping"}',
        headers=headers)


def test_f2_mcp_origin_from_allowed_hosts_not_403():
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    app = api.create_app(
        token="", router=object(),
        allowed_hosts=["127.0.0.1", "mivia.example.com"])
    with TestClient(app) as client:
        r = _post_mcp(client, "https://mivia.example.com")
        assert r.status_code != 403, r.status_code
        r = _post_mcp(client, "http://mivia.example.com")
        assert r.status_code != 403, r.status_code


def test_f2_mcp_foreign_origin_still_403():
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    app = api.create_app(
        token="", router=object(),
        allowed_hosts=["127.0.0.1", "mivia.example.com"])
    with TestClient(app) as client:
        r = _post_mcp(client, "https://evil.example.net")
        assert r.status_code == 403, r.status_code


def test_f2_mcp_no_origin_unaffected():
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    app = api.create_app(
        token="", router=object(),
        allowed_hosts=["127.0.0.1", "mivia.example.com"])
    with TestClient(app) as client:
        r = _post_mcp(client, None)
        assert r.status_code != 403, r.status_code


def test_f2_explicit_allowed_origins_kwarg_wins():
    pytest.importorskip("mcp")
    from mcp.server.mcpserver import MCPServer

    from mivia_small_json import api

    captured = {}

    def fake_streamable_http_app(self, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop")

    orig = MCPServer.streamable_http_app
    MCPServer.streamable_http_app = fake_streamable_http_app
    try:
        api.create_app(token="t", router=object(),
                       allowed_hosts=["mivia.example.com"],
                       allowed_origins=["https://custom.example.org"])
    except RuntimeError:
        pass
    finally:
        MCPServer.streamable_http_app = orig
    ts = captured["transport_security"]
    assert "https://custom.example.org" in ts.allowed_origins


# ---------------------------------------------------------------------------
# F3: non-ASCII bearer must be a plain 401, never a 500
# ---------------------------------------------------------------------------

def test_f3_non_ascii_bearer_is_401():
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    client = TestClient(api.create_app(token="t", router=object()))
    r = client.post("/v1/route", json={"text": "x"},
                    headers={"Authorization": "Bearer ŝecret✉token"
                             .encode("utf-8")})
    assert r.status_code == 401, (r.status_code, r.text[:200])


def test_f3_bearer_statuses():
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    client = TestClient(api.create_app(token="t", router=object()))
    r = client.post("/v1/route", json={"text": "x"},
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    r = client.post("/v1/route", json={"text": "x"},
                    headers={"Authorization": "Bearer t"})
    # Auth passed (dummy router) -> the handler itself errors, never 401/500
    # on the auth layer. 500 from the dummy router is expected and fine.
    assert r.status_code != 401


# ---------------------------------------------------------------------------
# F4: grade anchors are capped like tools/candidates
# ---------------------------------------------------------------------------

def test_f4_too_many_anchors_overflow():
    from mivia_small_json.api import MAX_ANCHORS, do_grade

    assert MAX_ANCHORS == 200
    with pytest.raises(OverflowError):
        do_grade("x", ["a"] * (MAX_ANCHORS + 1))


def test_f4_grade_endpoint_413():
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    client = TestClient(api.create_app(token="t", router=object()))
    auth = {"Authorization": "Bearer t"}
    r = client.post("/v1/grade", json={"text": "x", "anchors": ["a"] * 201},
                    headers=auth)
    assert r.status_code == 413, r.status_code


# ---------------------------------------------------------------------------
# F5: whole-body ceiling before parsing
# ---------------------------------------------------------------------------

def test_f5_oversize_body_413():
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    assert api.MAX_BODY_BYTES == 4 * 1024 * 1024
    client = TestClient(api.create_app(token="t", router=object()))
    auth = {"Authorization": "Bearer t", "Content-Type": "application/json"}
    body = b"x" * (api.MAX_BODY_BYTES + 1)
    r = client.post("/v1/route", content=body, headers=auth)
    assert r.status_code == 413, r.status_code


def test_f5_normal_body_still_parses():
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    client = TestClient(api.create_app(token="t", router=object()))
    r = client.post("/v1/route", content=b"not json",
                    headers={"Authorization": "Bearer t",
                             "Content-Type": "application/json"})
    assert r.status_code == 422, r.status_code


# ---------------------------------------------------------------------------
# F6: do_select must reuse the cached guard, not rebuild per request
# ---------------------------------------------------------------------------

def test_f6_select_uses_cached_guard(monkeypatch):
    from mivia_small_json import api, tool_match

    calls = []

    class _Dummy:
        def select(self, text, candidates, with_status=False):
            calls.append(1)
            return candidates[0], 0.9, "ok"

    matcher = _Dummy()
    builds = []
    monkeypatch.setattr(tool_match, "load_matcher",
                        lambda *a, **k: (builds.append(1), matcher)[1])
    monkeypatch.setattr(tool_match, "_GUARD", None)
    monkeypatch.setattr(tool_match, "_GUARD_LOADED", False)

    out1 = api.do_select("pick one", ["a", "b"])
    out2 = api.do_select("pick one", ["a", "b"])
    assert out1["choice"] == "a" and out2["choice"] == "a"
    assert len(calls) == 2          # the matcher was used twice...
    assert len(builds) == 1         # ...but built once


# ---------------------------------------------------------------------------
# F7: stdlib sidecar hardening
# ---------------------------------------------------------------------------

class _ServeServer:
    """One threaded stdlib server on an ephemeral loopback port."""

    def __init__(self):
        from mivia_small_json import serve

        from http.server import ThreadingHTTPServer

        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), serve._Handler)
        self.server.router = None
        import threading

        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.server.server_address[1]

    def post(self, body: bytes | None, headers: dict | None = None):
        import urllib.error
        import urllib.request

        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/classify", data=body,
            headers=h, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture()
def serve_server():
    s = _ServeServer()
    yield s
    s.close()


def test_f7a_bad_body_is_400(serve_server):
    status, body = serve_server.post(b"not json")
    assert status == 400, (status, body)


def test_f7a_empty_body_classify_is_400(serve_server):
    status, body = serve_server.post(None)
    assert status == 400, (status, body)


def test_f7b_oversize_body_413(serve_server):
    from mivia_small_json import serve

    status, body = serve_server.post(
        b"x" * (serve.MAX_BODY_BYTES + 1))
    assert status == 413, (status, str(body)[:100])


def test_f7b_handler_has_timeout():
    from mivia_small_json import serve

    assert serve._Handler.timeout and serve._Handler.timeout > 0


def test_f7c_chunked_is_411(serve_server):
    status, body = serve_server.post(
        b'{"text": "x"}',
        headers={"Transfer-Encoding": "chunked"})
    assert status == 411, (status, str(body)[:100])


def test_f7d_unix_server_threaded_and_private(tmp_path):
    import os

    from mivia_small_json import serve

    assert issubclass(serve._UnixHTTPServer, socketserver.ThreadingMixIn)
    sock_path = str(tmp_path / "mivia.sock")
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass
    server = serve._UnixHTTPServer(sock_path, serve._Handler)
    try:
        mode = stat.S_IMODE(os.stat(sock_path).st_mode)
        assert mode == 0o600, oct(mode)
    finally:
        server.server_close()


@pytest.mark.parametrize("darwin,maxrss,expected", [(False, 512, 0.5),
                                                    (True, 512 * 1024, 0.5)])
def test_f7e_rss_mb_platform(darwin, maxrss, expected, monkeypatch):
    """Linux ru_maxrss is KiB; macOS reports BYTES. Same 512 KiB process
    must read as 0.5 MiB on both.

    Re-pinned 2026-10-10 (serving finding, 33609db): /health must show
    memory COMING BACK, so the primary source is /proc/self/status VmRSS
    and ru_maxrss is the FALLBACK only. These tests pin the fallback unit
    law by forcing the VmRSS reader to fail; the primary path is pinned
    by test_f7e_rss_mb_reads_vmrss below."""
    import resource

    from mivia_small_json import api, serve

    monkeypatch.setattr(serve, "_current_rss_kib", lambda: None)
    fake = types.SimpleNamespace(
        getrusage=lambda who: types.SimpleNamespace(ru_maxrss=maxrss))
    monkeypatch.setattr(resource, "getrusage", fake.getrusage)
    monkeypatch.setattr("sys.platform", "darwin" if darwin else "linux")
    assert serve.rss_mb() == expected
    assert api._rss_mb() == expected


def test_f7e_rss_mb_reads_vmrss(monkeypatch):
    """Primary path (2026-10-10 serving finding): rss_mb reads CURRENT
    VmRSS, not the ru_maxrss high-water — after freeing an 80 MB block
    the old reading stayed at 99.1 MiB while VmRSS fell to ~19.5 MiB."""
    from mivia_small_json import api, serve

    monkeypatch.setattr(serve, "_current_rss_kib", lambda: 19956)
    assert serve.rss_mb() == 19.5
    assert api._rss_mb() == 19.5


# ---------------------------------------------------------------------------
# F8: stdio MCP protocol fixes
# ---------------------------------------------------------------------------

def test_f8a_ping_answers_empty_result():
    from mivia_small_json import mcp

    out = mcp.handle_line('{"jsonrpc": "2.0", "id": 7, "method": "ping"}',
                          router=None)
    assert out is not None
    resp = json.loads(out)
    assert resp == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_f8b_unknown_notification_is_silent():
    from mivia_small_json import mcp

    assert mcp.dispatch({"jsonrpc": "2.0", "method": "notifications/whatever"},
                        router=None) is None
    assert mcp.dispatch({"jsonrpc": "2.0", "method": "anything", "params": {}},
                        router=None) is None


def test_f8b_unknown_request_still_minus_32601():
    from mivia_small_json import mcp

    resp = mcp.dispatch({"jsonrpc": "2.0", "id": 3, "method": "nope"},
                        router=None)
    assert resp["error"]["code"] == -32601
    assert resp["id"] == 3


def test_f8c_bad_byte_does_not_kill_loop(monkeypatch, capsys):
    from mivia_small_json import bootstrap, mcp, pipeline

    monkeypatch.setattr(bootstrap, "register_all_schemas", lambda: None)
    monkeypatch.setattr(pipeline, "load_active_router", lambda: None)
    payload = (b"\xff\n"
               b'{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n')
    monkeypatch.setattr(
        "sys.stdin", io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8"))
    assert mcp.main([]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2          # parse error for the bad byte + ping
    first = json.loads(lines[0])
    # The bad byte degrades to U+FFFD via errors="replace" and surfaces as
    # a JSON-RPC parse error (-32700) or an internal error (-32603) — the
    # contract is that the loop SURVIVES and keeps answering either way.
    assert first["error"]["code"] in (-32603, -32700)
    second = json.loads(lines[1])
    assert second == {"jsonrpc": "2.0", "id": 1, "result": {}}
