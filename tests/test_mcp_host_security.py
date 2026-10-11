"""MCP host-header security: the env bridge must behave as documented.

Regression (found probing a dev deployment):
``create_app`` passed ``transport_security=None`` when no allowlist was
configured, which makes the SDK auto-enable localhost-only DNS-rebinding
protection — and through that stack every Host was rejected with 421,
loopback included. Local MCP clients could not connect at all. The
contract (docstring + serving spec) is: no ``MIVIA_ALLOWED_HOSTS`` ->
protection off (bearer already gates ``/mcp``); hosts configured ->
exactly those Host values are accepted, everything else 421.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mcp")

from mivia_small_json import api  # noqa: E402

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
               "clientInfo": {"name": "host-probe", "version": "0"}},
}


def _initialize(client, host):
    resp = client.post("/mcp", content=json.dumps(INITIALIZE).encode(),
                       headers={"Host": host, "Authorization": "Bearer t",
                                "Accept": "application/json, text/event-stream",
                                "Content-Type": "application/json"})
    return resp.status_code


def test_no_allowlist_protection_off_accepts_any_host(monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.delenv("MIVIA_ALLOWED_HOSTS", raising=False)
    app = api.create_app(token="t")
    with TestClient(app) as client:
        assert _initialize(client, "mivia.example.com") == 200
        assert _initialize(client, "localhost") == 200


def test_allowlist_accepts_listed_host_and_rejects_others(monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.setenv("MIVIA_ALLOWED_HOSTS",
                           "mivia.example.com, mivia.example.org")
    app = api.create_app(token="t")
    with TestClient(app) as client:
        assert _initialize(client, "mivia.example.com") == 200
        assert _initialize(client, "evil.example.net") == 421


def test_allowlist_via_argument_matches_env_bridge(monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.delenv("MIVIA_ALLOWED_HOSTS", raising=False)
    app = api.create_app(token="t", allowed_hosts=["mivia.example.com"])
    with TestClient(app) as client:
        assert _initialize(client, "mivia.example.com") == 200
        assert _initialize(client, "mivia.example.org") == 421


def _captured_transport_security(monkeypatch, token):
    """Build create_app with the SDK's streamable_http_app captured, so the
    transport-security DECISION is asserted without SDK internals."""
    from mivia_small_json import api

    captured = {}

    def fake_streamable_http_app(self, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before serving")

    from mcp.server.mcpserver import MCPServer

    monkeypatch.setattr(MCPServer, "streamable_http_app",
                        fake_streamable_http_app)
    try:
        # A dummy router: create_app(router=None) loads a full ONNX encoder
        # per call, and two of those breach the 2 GB sandbox law. The app
        # never serves here, so the dummy is safe.
        api.create_app(token=token, router=object())
    except RuntimeError:
        pass
    return captured.get("transport_security")


def test_tokenless_app_enables_rebinding_protection(monkeypatch):
    """A tokenless deployment must not be open on BOTH layers at once.

    The bearer gate fails open without MIVIA_API_TOKEN (documented local
    default); the MCP DNS-rebinding protection therefore stays ON in that
    case, and only a token (or an explicit MIVIA_ALLOWED_HOSTS) turns it
    off.
    """
    monkeypatch.delenv("MIVIA_API_TOKEN", raising=False)
    monkeypatch.delenv("MIVIA_ALLOWED_HOSTS", raising=False)
    ts = _captured_transport_security(monkeypatch, token="")
    assert ts is not None
    assert ts.enable_dns_rebinding_protection is True


def test_tokenized_app_keeps_protection_off(monkeypatch):
    monkeypatch.delenv("MIVIA_ALLOWED_HOSTS", raising=False)
    ts = _captured_transport_security(monkeypatch, token="t")
    assert ts is not None
    assert ts.enable_dns_rebinding_protection is False


def test_tokenless_rest_rejects_attacker_host():
    """DNS-rebinding defense covers the REST half of a tokenless app:
    non-loopback Host headers get 421 on every /v1 route; loopback with
    any port passes (2026-10-08 serving review P1)."""
    from starlette.testclient import TestClient

    from mivia_small_json import api

    client = TestClient(api.create_app(token="", router=object()))
    r = client.post("/v1/route", json={"text": "ERROR [auth]: token expired"},
                    headers={"Host": "evil.com"})
    assert r.status_code == 421, r.status_code
    r = client.post("/v1/route", json={"text": "ERROR [auth]: token expired"},
                    headers={"Host": "127.0.0.1:8123"})
    assert r.status_code == 200, (r.status_code, r.text[:200])
    r = client.get("/health", headers={"Host": "evil.com"})
    assert r.status_code == 421, "health must not be host-open either"


def test_tokened_rest_ignores_host_header():
    """A tokened deployment keeps its existing contract: any Host is fine,
    the bearer gate decides (proxies rewriting Host keep working)."""
    from starlette.testclient import TestClient

    from mivia_small_json import api

    client = TestClient(api.create_app(token="t", router=object()))
    r = client.post("/v1/route", json={"text": "x"},
                    headers={"Host": "evil.com"})
    assert r.status_code == 401, "bearer gate, not the host gate, decides"
