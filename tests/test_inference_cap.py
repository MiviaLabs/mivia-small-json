"""Inference concurrency cap for the served API (2026-10-10 load-test fix).

Red-first TDD for the committed load test (reports/load-test-2026-10-10.md):
every concurrently-serving /v1 request holds its own ONNX activation
working set — VmRSS peaked 2389.6 MB at 24 parallel 200-tool guard clients
and 2866.2 MB in an 80-request burst, against a ~1.2 GB sequential baseline
(~50 MB marginal per in-flight request) — an OOM-kill on a 2 GB-capped
deploy. The in-process half of the registered rate-limit lever (owner
2026-10-08; instance class is the owner's call) bounds concurrent /v1/*
inference with an asyncio semaphore: default 8 (mirroring serve.py's
DEFAULT_MAX_CONCURRENT_HANDLERS, commit 33609db), override
MIVIA_MAX_CONCURRENT_INFERENCE (clamped >= 1, garbage falls back to the
default). Requests beyond the cap WAIT in the app (arrival-order pacing) —
no 429, no behavior change per response. /health and the metering/admin
surfaces stay ungated: health must answer during saturation.

ORDER IS LOAD-BEARING: the gate wraps the app OUTSIDE the metering
middleware (Bearer -> Gate -> Metering -> FastAPI). Metering admits a call
when the request REACHES it, so a request still queued at the gate has not
been admitted — a client that disconnects while queued costs nothing
(exactly like a request that never arrived), and a queued-then-served
request counts exactly like an unqueued one.

Deterministic, no timing asserts: handlers park on events (the sidecar cap
test's event-parked pattern, tests/test_fixval_serving.py TestConcurrencyCap);
event waits with timeouts are the only clocks, and the "queue holds" probe
is a negative wait on a state that is IMPOSSIBLE under the semaphore and
guaranteed without it (red at HEAD).

Focused suite only (owner rule 14):
    python -m pytest tests/test_inference_cap.py tests/test_api.py \
        tests/test_metering.py tests/test_serve_mcp.py -q
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from mivia_small_json import api, metering  # noqa: E402

# router=False keeps create_app off the model-loading path (the rules
# fallback): these tests meter and pace HTTP, they do not measure the
# router, and the parked handlers below never run inference at all.

_CANNED_GUARD = {"action": "ABSTAIN", "tool": None,
                 "missing_parameters": None, "clarification_prompt": None,
                 "record": {"type": "no_tool_call"}, "meta": {}}

_ENV = "MIVIA_MAX_CONCURRENT_INFERENCE"


# --------------------------------------------------------------------------
# Env handling: explicit argument > MIVIA_MAX_CONCURRENT_INFERENCE > module
# default, floored at 1, garbage falls back — serve.py's cap law, mirrored.
# --------------------------------------------------------------------------


def test_env_and_explicit_override(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert api.DEFAULT_MAX_CONCURRENT_INFERENCE == 8
    assert api._resolve_inference_cap(None) == 8  # unset -> default
    monkeypatch.setenv(_ENV, "3")
    assert api._resolve_inference_cap(None) == 3  # "3" -> 3
    monkeypatch.setenv(_ENV, "not-a-number")
    assert api._resolve_inference_cap(None) == 8  # garbage -> default
    monkeypatch.setenv(_ENV, "")
    assert api._resolve_inference_cap(None) == 8  # empty -> default
    for bad in ("0", "-5"):
        monkeypatch.setenv(_ENV, bad)
        assert api._resolve_inference_cap(None) == 1  # the floor is 1
    monkeypatch.setenv(_ENV, "99")
    assert api._resolve_inference_cap(5) == 5  # explicit beats env
    assert api._resolve_inference_cap("x") == 8  # garbage arg -> default


# --------------------------------------------------------------------------
# Live-server plumbing: one uvicorn process in a thread per test (mirrors
# tests/test_api.py _Server), cap resolved from the env the TEST set before
# construction. lifespan="off": these tests hit /v1/* and /health only and
# never need the MCP session manager or the model warmup.
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CapServer:
    def __init__(self, metering_store=None):
        kwargs = {"token": "", "router": False}
        if metering_store is not None:
            kwargs["metering_store"] = metering_store
            kwargs["admin_key"] = "admin-test"
        self.port = _free_port()
        self.app = api.create_app(**kwargs)
        import uvicorn

        self.server = uvicorn.Server(uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="error",
            lifespan="off"))
        threading.Thread(target=self.server.run, daemon=True).start()
        for _ in range(240):
            if self.server.started:
                return
            time.sleep(0.25)
        raise RuntimeError("uvicorn did not start")

    def get(self, path: str, timeout: int = 2):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")

    def post(self, path: str, payload: dict, timeout: int = 30):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def stop(self):
        self.server.should_exit = True


def _parked_guard(cap, k, lock, state, cap_reached, all_entered, release):
    """Instrumented do_guard (the sidecar cap test's pattern): counts
    simultaneously-ACTIVE inference handlers. Any entry beyond the cap
    parks BEFORE decrementing, so k entries would trip ``all_entered`` —
    impossible under the semaphore, guaranteed without the gate (red)."""

    def parked(request, tools, router):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            if state["active"] >= cap:
                cap_reached.set()
            if state["active"] >= k:
                all_entered.set()
        release.wait(timeout=10)
        with lock:
            state["active"] -= 1
        return dict(_CANNED_GUARD)

    return parked


def _fire_guard_threads(srv, k, results):
    workers = []

    def drive(i: int) -> None:
        try:
            results[i] = srv.post("/v1/guard", {"request": "x", "tools": []})
        except Exception as exc:  # surfaced by the final status assert
            results[i] = exc

    for i in range(k):
        t = threading.Thread(target=drive, args=(i,))
        t.start()
        workers.append(t)
    return workers


# --------------------------------------------------------------------------
# Mechanism: in-flight inference never exceeds the cap; the rest queue and
# still complete 200 (queued, not rejected — no 429).
# --------------------------------------------------------------------------


def test_active_inference_never_exceeds_the_cap(monkeypatch):
    """MIVIA_MAX_CONCURRENT_INFERENCE=2: five concurrent /v1/guard requests
    whose handler parks on an event. The two cap holders park inside the
    inference entry; the rest must wait at the gate (never counted active)
    and still complete 200 with unchanged bodies."""
    monkeypatch.setenv(_ENV, "2")
    cap, k = 2, 5
    srv = _CapServer()
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}
    cap_reached = threading.Event()
    all_entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        api, "do_guard",
        _parked_guard(cap, k, lock, state, cap_reached, all_entered, release))
    results = [None] * k
    try:
        workers = _fire_guard_threads(srv, k, results)
        # The cap slots fill and park inside the inference entry point.
        assert cap_reached.wait(timeout=10), "cap slots never filled"
        # While the slot holders park, the queued requests must not raise
        # the in-flight count above the cap (impossible under the gate).
        assert all_entered.wait(timeout=1.0) is False, (
            f"{state['max_active']} inferences ran at once; the cap "
            f"({cap}) is not enforced")
        assert state["max_active"] <= cap
        release.set()
        for t in workers:
            t.join(timeout=10)
        # Every queued request was eventually served, none dropped, and
        # the response body is exactly the handler's (no 429, no change).
        assert all(isinstance(r, tuple) and r[0] == 200 for r in results), \
            results
        assert all(r[1]["action"] == "ABSTAIN" for r in results), results
        assert state["max_active"] <= cap
    finally:
        release.set()
        srv.stop()


# --------------------------------------------------------------------------
# Saturation surfaces stay ungated: /health and the metering/admin plane
# must answer while every inference slot is parked.
# --------------------------------------------------------------------------


def test_health_answers_while_the_cap_is_full(monkeypatch):
    """With the one slot parked by an in-flight guard and a second request
    queued at the gate, /health still answers (the timeout=2 request proves
    it is not queued behind the gate)."""
    monkeypatch.setenv(_ENV, "1")
    cap, k = 1, 2
    srv = _CapServer()
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}
    cap_reached = threading.Event()
    all_entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        api, "do_guard",
        _parked_guard(cap, k, lock, state, cap_reached, all_entered, release))
    results = [None] * k
    try:
        workers = _fire_guard_threads(srv, k, results)
        assert cap_reached.wait(timeout=10), "cap slot never filled"
        status, body = srv.get("/health")
        assert status == 200, body
        assert body["status"] == "ok"
        # The second guard is still queued, not dropped and not admitted.
        assert all_entered.wait(timeout=1.0) is False
        release.set()
        for t in workers:
            t.join(timeout=10)
        assert all(isinstance(r, tuple) and r[0] == 200 for r in results), \
            results
    finally:
        release.set()
        srv.stop()


def test_usage_surface_stays_ungated_while_saturated(monkeypatch, tmp_path):
    """The metering/admin surfaces stay outside the cap: GET /v1/usage
    answers while every inference slot is parked."""
    monkeypatch.setenv(_ENV, "1")
    store = metering.MeteringStore(str(tmp_path / "metering.sqlite3"))
    cap, k = 1, 2
    srv = _CapServer(metering_store=store)
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}
    cap_reached = threading.Event()
    all_entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        api, "do_guard",
        _parked_guard(cap, k, lock, state, cap_reached, all_entered, release))
    results = [None] * k
    try:
        workers = _fire_guard_threads(srv, k, results)
        assert cap_reached.wait(timeout=10), "cap slot never filled"
        status, body = srv.get("/v1/usage")
        assert status == 200, body
        assert body == {"metered": False}  # anonymous identity, ungated
        assert all_entered.wait(timeout=1.0) is False
        release.set()
        for t in workers:
            t.join(timeout=10)
        assert all(isinstance(r, tuple) and r[0] == 200 for r in results), \
            results
    finally:
        release.set()
        srv.stop()


# --------------------------------------------------------------------------
# Accounting identity: the cap paces arrivals, it never re-prices them.
# --------------------------------------------------------------------------


def test_metered_accounting_identity_cap_one_vs_uncapped(monkeypatch,
                                                         tmp_path):
    """With metering ON and the cap at 1, two sequential metered calls
    produce ok=2 with X-Mivia-Count 1,2 — exactly like the uncapped app."""
    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.setattr(
        api, "do_guard", lambda request, tools, router: dict(_CANNED_GUARD))
    from starlette.testclient import TestClient

    outcomes = {}
    for cap in (1, None):
        store = metering.MeteringStore(
            str(tmp_path / f"metering-{cap}.sqlite3"))
        kwargs = {"token": "", "router": False, "metering_store": store,
                  "admin_key": "admin-test"}
        if cap is not None:
            kwargs["max_concurrent_inference"] = cap
        client = TestClient(api.create_app(**kwargs),
                            base_url="http://127.0.0.1:7200")
        made = client.post("/v1/admin/keys", json={"action": "create"},
                           headers={"Authorization": "Bearer admin-test"})
        assert made.status_code == 200, made.text
        key = made.json()["key"]
        headers = {"Authorization": f"Bearer {key}"}
        for i in (1, 2):
            r = client.post("/v1/guard", json={"request": "x", "tools": []},
                            headers=headers)
            assert r.status_code == 200, (cap, r.text)
            assert r.headers["x-mivia-count"] == str(i)
        kh = metering.key_hash(key)
        outcomes[cap] = store.counts(kh, store.month())
    assert outcomes[1] == {"ok": 2, "error": 0, "reject": 0}
    assert outcomes[1] == outcomes[None], (
        "the cap changed metering accounting for identical outcomes")


def test_queued_then_disconnected_is_not_counted(monkeypatch, tmp_path):
    """The placement law, behaviorally: the gate sits OUTSIDE metering
    admission, so a request still WAITING on the cap is not yet admitted.
    A (metered, key identity) fills the one slot and parks; B queues at the
    gate; the client disconnects (the ASGI task cancels); B must cost
    nothing, and A must settle as exactly one call — identical to the
    uncapped bracket. If the gate sat inside the metering middleware, B
    would consume its reservation while queued and the disconnect could not
    un-charge it (ok=2 red)."""
    monkeypatch.delenv(_ENV, raising=False)
    store = metering.MeteringStore(str(tmp_path / "metering.sqlite3"))
    made = store.create_key()
    app = api.create_app(token="", router=False, metering_store=store,
                         admin_key="admin-test", max_concurrent_inference=1)

    entered = threading.Event()
    release = threading.Event()

    def parked_guard(request, tools, router):
        entered.set()
        release.wait(timeout=10)
        return dict(_CANNED_GUARD)

    monkeypatch.setattr(api, "do_guard", parked_guard)

    async def drive():
        def _recv(body):
            async def receive():
                return {"type": "http.request", "body": body,
                        "more_body": False}
            return receive

        def _collector(box):
            async def send(message):
                if message["type"] == "http.response.start":
                    box["status"] = message["status"]
            return send

        def _scope():
            return {
                "type": "http", "asgi": {"version": "3.0"},
                "http_version": "1.1", "method": "POST", "scheme": "http",
                "path": "/v1/guard", "raw_path": b"/v1/guard",
                "query_string": b"", "root_path": "",
                "headers": [
                    (b"host", b"127.0.0.1:7200"),
                    (b"content-type", b"application/json"),
                    (b"authorization",
                     f"Bearer {made['key']}".encode("ascii")),
                ],
                "client": ("127.0.0.1", 50000),
                "server": ("127.0.0.1", 7200),
            }

        kh = metering.key_hash(made["key"])
        month = store.month()
        body = json.dumps({"request": "x", "tools": []}).encode()
        status_a: dict = {}

        # A: fills the one slot, parks inside the handler — admitted once.
        task_a = asyncio.create_task(app(_scope(), _recv(body),
                                         _collector(status_a)))
        await asyncio.to_thread(entered.wait, 10)
        assert store.counts(kh, month) == {"ok": 1, "error": 0, "reject": 0}

        # B: queues at the gate, NOT yet admitted.
        task_b = asyncio.create_task(app(_scope(), _recv(body),
                                         _collector({})))
        for _ in range(100):  # scheduling ticks: B reaches the gate's queue
            await asyncio.sleep(0)
        assert store.counts(kh, month) == {"ok": 1, "error": 0, "reject": 0}

        # The client disconnects while still queued: nothing is consumed.
        task_b.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task_b
        assert task_b.cancelled()
        assert store.counts(kh, month) == {"ok": 1, "error": 0, "reject": 0}

        # A finishes served: one call, one charge — today's bracket exactly.
        release.set()
        await asyncio.wait_for(task_a, timeout=10)
        assert status_a["status"] == 200
        assert store.counts(kh, month) == {"ok": 1, "error": 0, "reject": 0}

    asyncio.run(drive())
