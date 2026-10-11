"""Integration tests: the model actually runs through the HTTP serve server.

``tests/test_serve_mcp.py`` unit-tests ``serve.handle_classify`` and
``serve.dispatch`` but always passes ``router=None``, so the model path and
``python -m mivia_small_json.serve`` (the real ``main()``) are never
exercised. These tests load the shipped router and drive both the in-process
dispatcher and a live HTTP daemon.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from mivia_small_json import bootstrap, pipeline, registry, serve
from mivia_small_json.validate import validate

REPO_ROOT = Path(__file__).resolve().parents[1]

# Empirically verified against the shipped int8 encoder: routes to log_event
# with ``routed_by == "model"``, not the rules fallback.
CLASSIFY_TEXT = "2026-03-01T12:00:00Z ERROR [auth]: token expired"
EXPECTED_TYPE = "log_event"


@pytest.fixture(scope="session")
def router():
    """Load the shipped router once (session scope; ~1-2 s to load)."""
    loaded = pipeline.load_active_router()
    if loaded is None:
        pytest.skip("no model artifact available; nothing to integrate")
    return loaded


def test_serve_dispatch_uses_the_encoder(router):
    with registry.temporary_schemas():
        bootstrap.register_all_schemas()
        status, body = serve.dispatch(
            "POST", "/classify", {"text": CLASSIFY_TEXT}, router)

        assert status == 200
        assert body["meta"]["routed_by"] == "model", (
            "serve fell back to the rules router; the model is not wired"
        )
        assert body["result"]["type"] == EXPECTED_TYPE
        assert validate(body["result"], body["result"]["type"]) == []


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"serve exited early: {proc.stderr.read()}")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # not listening yet
            last = exc
            time.sleep(0.2)
    raise AssertionError(f"serve never became ready: {last!r}")


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_serve_http_server_end_to_end():
    """Boot ``python -m mivia_small_json.serve`` and hit ``/classify``.

    Covers ``main()`` -> ``run()`` -> ``register_all_schemas`` -> live
    ``_Handler``; a broken entry point or router wiring fails here.
    """
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "mivia_small_json.serve", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO_ROOT,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        _wait_for_health(base_url, proc)
        body = _post_json(base_url + "/classify", {"text": CLASSIFY_TEXT})
        assert body["meta"]["routed_by"] == "model", (
            "the HTTP server fell back to the rules router"
        )
        assert body["result"]["type"] == EXPECTED_TYPE
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
