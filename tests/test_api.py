"""serving-api spec: the deployable REST quartet + MCP surface.

Bars are pre-registered in serving-api: a fixed corpus
through every endpoint (2xx + required keys, zero 500s), the bearer gate on
/v1/* and /mcp with /health always open, a live MCP client session, and the
< 2 GB RSS sandbox law. The stdlib serve.py/mcp.py surfaces are untouched —
their own tests keep covering them.
"""

from __future__ import annotations

import asyncio
import gc
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from mivia_small_json.api import MAX_TEXT_CHARS, BearerMiddleware, create_app  # noqa: E402

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

SUM_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_sum",
        "description": "Calculate the mathematical sum of two numbers.",
        "parameters": {
            "type": "object",
            "properties": {"num1": {"type": "number"},
                           "num2": {"type": "number"}},
            "required": ["num1", "num2"],
        },
    },
}
LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List the files in a directory.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """One uvicorn process in a thread, shared by the whole module."""

    def __init__(self, token: str | None):
        from mivia_small_json import pipeline

        self.port = _free_port()
        self.app = create_app(token=token or "",
                              router=pipeline.load_active_router())
        import uvicorn

        self.server = uvicorn.Server(uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="error"))
        threading.Thread(target=self.server.run, daemon=True).start()
        for _ in range(240):
            if self.server.started:
                return
            time.sleep(0.25)
        raise RuntimeError("uvicorn did not start")

    def post(self, path: str, payload: dict, token: str | None = None,
             auth_header: str | None = None):
        headers = {"Content-Type": "application/json"}
        if auth_header is not None:
            headers["Authorization"] = auth_header
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw)
            except ValueError:
                # Unknown paths fall through to the mounted MCP sub-app,
                # whose 404 body is Starlette's plaintext — status is the
                # contract here, not the body.
                return exc.code, {"raw": raw.decode("utf-8", "replace")}


@pytest.fixture(scope="module")
def open_server():
    srv = _Server(token=None)
    yield srv
    srv.server.should_exit = True


@pytest.fixture(scope="module")
def authed_server():
    srv = _Server(token="test-token-0123456789")
    yield srv
    srv.server.should_exit = True


# --------------------------------------------------------------------------
# RSS reader law (2026-10-10 serving finding): api._rss_mb must report the
# CURRENT resident set, not the ru_maxrss high-water that can never come
# down (an 80 MB free left the old reading at 99.1 MiB while /proc VmRSS
# fell to ~19.5 MiB). api._rss_mb delegates to the single serve.rss_mb
# reader: VmRSS on Linux, ru_maxrss fallback elsewhere.
# --------------------------------------------------------------------------

def _vmrss_mib() -> float | None:
    """The kernel's own current-RSS number, parsed independently (MiB)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def test_api_rss_mb_returns_a_positive_float():
    from mivia_small_json import api

    value = api._rss_mb()
    assert isinstance(value, float)
    assert value > 0


def test_api_rss_mb_matches_current_vmrss_within_a_sane_band():
    kernel = _vmrss_mib()
    if kernel is None:
        pytest.skip("no /proc/self/status (non-Linux); ru_maxrss fallback")
    from mivia_small_json import api

    value = api._rss_mb()
    assert abs(value - kernel) <= max(8.0, 0.25 * kernel), (value, kernel)


@pytest.mark.skipif(_vmrss_mib() is None,
                    reason="the free-drop law is observable on Linux only")
def test_api_rss_mb_comes_down_after_a_freed_block():
    """Current-RSS law: after a freed 64 MiB block the reading must come
    DOWN. ru_maxrss (the old reader) can never decrease, so under the bug
    second == first and this test is red at HEAD. No exact-number asserts,
    only a drop of at least a quarter of the freed block."""
    from mivia_small_json import api

    block = bytearray(64 * 1024 * 1024)
    for i in range(0, len(block), 4096):  # touch every page -> resident
        block[i] = 1
    first = api._rss_mb()
    del block
    gc.collect()
    second = api._rss_mb()
    assert second <= first - 16.0, (first, second)


# --------------------------------------------------------------------------
# Bar 1: the fixed corpus through every endpoint.
# --------------------------------------------------------------------------


def test_bare_log_fragment_routes_log_event(open_server):
    """A single timestamp-less log fragment routes log_event (~0.999 on
    the shipped router) — same as every timestamped/syslog form."""
    status, body = open_server.post(
        "/v1/route", {"text": "ERROR [auth]: token expired"})
    assert status == 200, body
    assert body["schema"] == "log_event"


class TestQuartetCorpus:
    def test_health_open_and_labelled(self, open_server):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{open_server.port}/health") as resp:
            body = json.loads(resp.read())
        assert resp.status == 200
        assert body["model"] == "mivia-small-json"
        if body["backend"] != "onnx-int8":
            pytest.skip(
                f"shipped ONNX encoder absent (backend={body['backend']!r}); "
                "the strict backend pin holds only on the shipped tree")
        assert body["guard"] in ("onnx-int8", "ready", "unavailable")
        assert body["grade"] in ("ready", "onnx-int8", "unavailable")
        # 2026-10-10 re-pin: rss_mb is CURRENT RSS (/proc VmRSS on Linux;
        # ru_maxrss only as the portable fallback) — the law bounds the
        # live process, not the lifetime high-water the old reader returned.
        assert body["rss_mb"] < 2048

    def test_health_reports_loadability_not_file_existence(
            self, open_server, monkeypatch):
        """File existence is not evidence (the bench.py attribution law):
        /health must ask the cached loaders, not Path.is_file(). A corrupt
        or tampered artifact is still on disk — is_file() reported
        "onnx-int8"/"ready" while every guard/grade request silently
        degraded to router-only decisions (review 2026-10-11).
        """
        from mivia_small_json import tool_match
        monkeypatch.setattr(tool_match, "_get_guard", lambda: None)
        monkeypatch.setattr(tool_match, "load_register_gate",
                            lambda: (None, None))
        monkeypatch.setattr(tool_match, "load_severity_specialist",
                            lambda: None)
        with urllib.request.urlopen(
                f"http://127.0.0.1:{open_server.port}/health") as resp:
            body = json.loads(resp.read())
        assert resp.status == 200
        assert body["guard"] == "unavailable"
        assert body["grade"] == "unavailable"

    def test_route_corpus(self, open_server):
        rows = [
            ("ERROR [auth]: token expired", "log_event"),
            ("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
             "@@ -1,1 +1,1 @@\n-old\n+new\n", "diff"),
            ("src/main.rs:12:18: error[E0308]: mismatched types", "diagnostic"),
        ]
        for text, expected in rows:
            status, body = open_server.post("/v1/route", {"text": text})
            assert status == 200, body
            assert body["schema"] == expected, (text, body)
            for key in ("schema", "confidence", "in_domain", "model", "latency_ms", "routed_by"):
                assert key in body, (key, body)
            assert body["model"] == "mivia-small-json"
            assert isinstance(body["latency_ms"], (int, float)) and body["latency_ms"] >= 0

    def test_extract_single_and_dump(self, open_server):
        status, body = open_server.post("/v1/extract", {
            "text": "2026-10-05 08:14:22,109 INFO "
                    "org.apache.hadoop.hdfs.server.datanode.DataNode: "
                    "BlockVerification completed for blk_1073741825"})
        assert status == 200, body
        rec = body["records"][0]["result"]
        assert rec["type"] == "log_event"
        assert rec["component"].endswith("DataNode")
        assert body["records"][0]["meta"]["valid"] is True

        status, body = open_server.post("/v1/extract", {
            "text": "hey what is this?\n"
                    "Jun 14 15:16:01 combo sshd[19939]: authentication "
                    "failure; rhost=1.2.3.4\n",
            "dump": True})
        assert status == 200, body
        # Re-pinned 2026-10-11 (realworld-r1 promotion + N2 precedence):
        # router-scored chunks are KEPT as valid records (2026-10-10 review
        # N2 — never silently drop user content); the promoted chain routes
        # the bare chat line to review_comment, the log stays log_event.
        assert len(body["records"]) == 2
        types = {r["result"]["type"] for r in body["records"]}
        assert types == {"review_comment", "log_event"}, types
        assert all(r["meta"]["valid"] for r in body["records"])

    def test_guard_tri_state(self, open_server):
        # CALL: full parameters present.
        status, body = open_server.post("/v1/guard", {
            "request": "calculate the sum of 15 and 30", "tools": [SUM_TOOL]})
        assert status == 200 and body["action"] == "CALL", body
        assert body["tool"] == "calculate_sum"
        assert json.loads(body["record"]["arguments"]) == {"num1": 15, "num2": 30}
        # CLARIFY: intent matches, mandatory arguments absent.
        status, body = open_server.post("/v1/guard", {
            "request": "calculate the sum", "tools": [SUM_TOOL]})
        assert status == 200 and body["action"] == "CLARIFY", body
        assert body["missing_parameters"] == ["num1", "num2"]
        assert body["clarification_prompt"]
        # ABSTAIN: irrelevant statement vs a veto-safe (no-required-args)
        # catalogue — the honest no_tool_call path.
        status, body = open_server.post("/v1/guard", {
            "request": "purple elephants dream of velvet", "tools": [LIST_TOOL]})
        assert status == 200, body
        assert body["action"] == "ABSTAIN", body
        assert body["record"]["type"] == "no_tool_call"
        # A {name: description} map is accepted too.
        status, body = open_server.post("/v1/guard", {
            "request": "calculate the sum of 4 and 7",
            "tools": {"calculate_sum": "Calculate the sum of two numbers."}})
        assert status == 200 and body["action"] in ("CALL", "CLARIFY"), body
        # Decision-layer composition (serving probe 2026-10-08): a CALL
        # whose arguments are empty while the tool declares required
        # parameters is downgraded to CLARIFY — executing an empty call is
        # the one unsafe outcome.
        ops_tool = {"name": "restart_service",
                    "description": "Restart a systemd service on a named host.",
                    "parameters": {"type": "object",
                                   "properties": {"service": {"type": "string"},
                                                  "host": {"type": "string"}},
                                   "required": ["service", "host"]}}
        status, body = open_server.post("/v1/guard", {
            "request": "restart nginx on staging", "tools": [ops_tool]})
        assert status == 200, body
        # 2026-10-10 survivor round (M1): the downgrade is VETO-GATED — the
        # single oracle is fillability_veto, so when the veto says the
        # request supplies the values ("nginx"/"staging" are content the
        # tool vocabulary cannot explain), the vetted honest-minimum CALL
        # stands and /v1/guard agrees with pipeline.classify. A CALL here
        # may carry the honest-minimum "{}"; what must never happen is the
        # two surfaces disagreeing on the same record.
        if body["action"] == "CALL":
            from mivia_small_json import pipeline as _pipeline

            obj, _ = _pipeline.classify(
                f"Tools available: {json.dumps([ops_tool])}\n\n"
                f"User request: restart nginx on staging",
                _pipeline.load_active_router())
            assert obj.get("type") == "tool_call", (body, obj)
            assert obj.get("name") == body["tool"], (body, obj)
        else:
            assert body["action"] == "CLARIFY", body
            assert body["missing_parameters"] == ["service", "host"]

    def test_grade_canonical_and_custom_anchors(self, open_server):
        status, body = open_server.post("/v1/grade", {
            "text": "Checkout hangs on 3D Secure with HTTP 504, customers "
                    "cannot pay at all, revenue impacted"})
        assert status == 200, body
        decision = body["result"]
        assert decision["action"] in ("act", "ask")
        assert isinstance(decision["tier_set"], list)
        # the measured canonical surface: 4 ordered anchors
        assert len(body["anchors"]) == 4

        status, body = open_server.post("/v1/grade", {
            "text": "typo in the footer text",
            "anchors": ["P4 low", "P3 medium", "P2 high", "P1 outage"]})
        assert status == 200, body
        assert body["result"]["action"] in ("act", "ask")

    def test_select_shape(self, open_server):
        status, body = open_server.post("/v1/select", {
            "text": "Users in the EU report checkout fails during 3D Secure "
                    "verification, gateway hangs then HTTP 504.",
            "candidates": ["Frontend UI", "Authentication", "Payments",
                           "DevOps"]})
        assert status == 200, body
        assert body["choice"] in ["Frontend UI", "Authentication",
                                  "Payments", "DevOps"]
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["status"] in ("ok", "ambiguous", "abstain")  # abstain: QW-C conformal rule

    def test_select_dict_form_enrichment(self, open_server):
        """First-class description enrichment (2026-10-11): candidates may
        carry {"name", "description", "examples"} — the measured Banking77
        win (48.0% bare -> 63.8% with per-label examples, 92.5% precision
        @ 20% coverage). Same response shape as the string form."""
        status, body = open_server.post("/v1/select", {
            "text": "Users in the EU report checkout fails during 3D Secure "
                    "verification, gateway hangs then HTTP 504.",
            "candidates": [
                {"name": "Frontend UI", "description": "UI rendering",
                 "examples": ["the page renders blank",
                              "button does nothing"]},
                {"name": "Payments", "description": "Payment processing",
                 "examples": ["card declined at checkout",
                              "3d secure verification hangs"]},
                {"name": "Authentication", "description": "Login flows"},
            ]})
        assert status == 200, body
        assert body["choice"] in ("Frontend UI", "Payments",
                                  "Authentication")
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["status"] in ("ok", "ambiguous", "abstain")

    def test_select_dict_form_error_law(self, open_server):
        """The strict serving shape: malformed dict candidates and cap
        violations are 4xx naming the field — never a guessed 200 pick
        (a typo'd 'example' key would silently drop the measured lift)."""
        base = {"text": "t"}
        for candidates, want in (
            ([{"name": "  "}], 422),                       # blank name
            ([{"name": "a", "examples": ["ok", 42]}], 422),  # non-string
            ([{"name": "a", "examples": "nope"}], 422),    # not a list
            ([{"name": "a", "example": ["x"]}], 422),      # unknown key
            ([{"name": "a", "examples": ["e"] * 9}], 413),  # > 8 examples
            ([{"name": "a", "examples": ["x" * 1025]}], 422),  # > 1024 ch
            ([{"name": "a"}, 42], 422),                    # mixed junk
            ([{"name": "a"}], 200),                        # desc/examples optional
        ):
            payload = dict(base, candidates=candidates)
            status, body = open_server.post("/v1/select", payload)
            assert status == want, (candidates, status, body)
            if want == 422 or want == 413:
                assert "error" in body

    def test_repair(self, open_server):
        status, body = open_server.post("/v1/repair", {
            "raw_arguments": "{'path': 'x.py', 'n': 1,}"})
        assert status == 200
        assert json.loads(body["repaired"]) == {"path": "x.py", "n": 1}
        assert body["lossy"] is False
        # A collapsed repair is flagged, never silent (serving probe finding).
        status, body = open_server.post("/v1/repair", {
            "raw_arguments": "{'path': 'x.py', 'overwrite': True, "
                             "'n': None, 'tags': ['a','b'],}"})
        assert status == 200
        # Valid whitespace-padded empty objects must not be marked lossy
        for empty_raw in ("{}", "{ }", "{\n}", "{\n  \t\r\n}", "```json\n{}\n```"):
            status, body = open_server.post("/v1/repair", {"raw_arguments": empty_raw})
            assert status == 200
            assert body["repaired"] == "{}"
            assert body["lossy"] is False, f"Empty object {empty_raw!r} marked lossy: true"

        # Truly collapsed repairs must be marked lossy: true
        status, body = open_server.post("/v1/repair", {
            "raw_arguments": "{broken input that collapses to empty}"})
        assert status == 200
        assert body["repaired"] == "{}", body
        assert body["lossy"] is True, body

    def test_error_law(self, open_server):
        # 4xx never becomes a 500; 500s (if any) never leak tracebacks.
        status, body = open_server.post("/v1/route", {"text": 42})
        assert status == 422 and "error" in body
        status, body = open_server.post("/v1/route", {})
        assert status == 422
        status, body = open_server.post("/v1/nope", {})
        assert status == 404  # body may be the MCP sub-app's plaintext 404
        status, body = open_server.post(
            "/v1/route", {"text": "x" * (MAX_TEXT_CHARS + 1)})
        assert status == 413
        status, body = open_server.post(
            "/v1/repair", {"raw_arguments": "x" * (MAX_TEXT_CHARS + 1)})
        assert status == 413
        status, body = open_server.post("/v1/select", {
            "text": "t", "candidates": []})
        assert status == 422

    def test_degenerate_inputs_are_4xx_not_guessed_200s(self, open_server):
        """2026-10-09 serving probe: empty/blank anchors and candidates
        produced wrong-but-valid 200s — ask "Options: none listed", an
        "act" on a whitespace tier, and choice "" — before the degenerate
        law made them explicit 4xx."""
        status, body = open_server.post("/v1/grade", {
            "text": "prod is down", "anchors": []})
        assert status == 422, body
        status, body = open_server.post("/v1/grade", {
            "text": "prod is down", "anchors": ["  "]})
        assert status == 422, body
        status, body = open_server.post("/v1/select", {
            "text": "hello", "candidates": [""]})
        assert status == 422, body
        status, body = open_server.post("/v1/select", {
            "text": "hello", "candidates": ["Payments", "  "], "head": "intent"})
        assert status == 422, body

    def test_sync_handlers_run_in_threadpool(self, monkeypatch):
        """Docstring contract of api._core:
        'Sync handlers run in the worker threadpool (the event loop never blocks).'
        Sync model inference must run in a worker thread, keeping the event loop
        free to serve health checks and concurrent I/O.
        """
        from starlette.testclient import TestClient
        from mivia_small_json import api

        app = api.create_app(token="", router=None)
        client = TestClient(base_url="http://127.0.0.1", app=app)

        handler_thread_info = {}
        orig_do_route = api.do_route

        def spy_do_route(text, router):
            try:
                asyncio.get_running_loop()
                handler_thread_info["on_loop"] = True
            except RuntimeError:
                handler_thread_info["on_loop"] = False
            return orig_do_route(text, router)

        monkeypatch.setattr(api, "do_route", spy_do_route)
        resp = client.post("/v1/route", json={"text": "ERROR [auth]: token expired"})
        assert resp.status_code == 200
        # A worker thread has no running loop. The name check is not
        # enough: "MainThread" contains "thread" and would pass while
        # the handler blocked the loop.
        assert handler_thread_info["on_loop"] is False

    def test_concurrent_requests_not_blocked_by_sync_inference(
            self, open_server, monkeypatch):
        """Concurrent requests (e.g. /health) must not be serialized or blocked
        behind slow/long sync model inference.

        Deterministic by construction: the slow handler parks on an event
        (``release``) once it has entered; /health must answer WHILE it is
        parked. If the event loop were blocked, /health could only answer
        after release — which happens after this assert — so the request
        hangs into its timeout and fails. No wall-clock assertion.
        """
        from mivia_small_json import api

        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        orig_do_route = api.do_route

        def parked_do_route(text, router):
            entered.set()
            release.wait(timeout=10)
            finished.set()
            return orig_do_route(text, router)

        monkeypatch.setattr(api, "do_route", parked_do_route)
        t = threading.Thread(
            target=open_server.post,
            args=("/v1/route", {"text": "ERROR [auth]: token expired"}))
        t.start()
        assert entered.wait(timeout=10), "handler never entered"
        with urllib.request.urlopen(
                f"http://127.0.0.1:{open_server.port}/health", timeout=2) as resp:
            assert resp.status == 200
        assert not finished.is_set(), (
            "/health answered only after the sync handler finished: "
            "the event loop was blocked")
        release.set()
        t.join(timeout=10)
        assert finished.is_set()


# --------------------------------------------------------------------------
# Bar 2: the bearer gate.
# --------------------------------------------------------------------------


class TestAuth:
    def test_open_when_unset(self, open_server):
        status, _ = open_server.post("/v1/route", {"text": "hello"})
        assert status == 200

    def test_401_without_and_with_wrong_token(self, authed_server):
        for kwargs in ({}, {"token": "wrong"}):
            status, body = authed_server.post("/v1/route",
                                              {"text": "hello"}, **kwargs)
            assert status == 401 and body == {"error": "unauthorized"}
        # every /v1/* route is gated, and /mcp is gated too
        for path, payload in (
                ("/v1/extract", {"text": "x"}),
                ("/v1/guard", {"request": "x", "tools": []}),
                ("/v1/grade", {"text": "x"}),
                ("/v1/select", {"text": "x", "candidates": ["a"]}),
                ("/v1/repair", {"raw_arguments": "x"}),
                ("/v1/structure", {"text": "x", "schema": "log_event"}),
                ("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})):
            status, _ = authed_server.post(path, payload)
            assert status == 401, path

    def test_correct_token_passes_and_health_stays_open(self, authed_server):
        status, body = authed_server.post("/v1/route",
                                          {"text": "ERROR [auth]: x"},
                                          token="test-token-0123456789")
        assert status == 200
        assert isinstance(body.get("schema"), str) and body["schema"]
        with urllib.request.urlopen(
                f"http://127.0.0.1:{authed_server.port}/health") as resp:
            assert resp.status == 200

    def _bearer_status(self, path: str, query: bytes = b"") -> tuple[int, dict]:
        """One request through the bearer gate. ``seen`` is what the
        downstream app received; it stays empty when the gate rejects."""
        seen = {}

        async def inner(scope, receive, send):
            seen["path"] = scope.get("path")
            seen["query"] = bytes(scope.get("query_string") or b"")
            await send({"type": "http.response.start", "status": 200,
                        "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def _call():
            sent = {}

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                if message["type"] == "http.response.start":
                    sent["status"] = message["status"]

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": query,
                "headers": [],
                "client": ("127.0.0.1", 9),
                "server": ("127.0.0.1", 80),
            }
            await BearerMiddleware(inner, "secret")(scope, receive, send)
            return sent["status"]

        return asyncio.run(_call()), seen

    @pytest.mark.parametrize("path,query,want_query", [
        ("/health", b"", b""),
        ("/health/", b"", b""),
        ("/health//", b"", b""),
        ("/health", b"probe=1", b"probe=1"),
        ("/health?probe=1", b"", b"probe=1"),
        ("/health/?probe=1", b"", b"probe=1"),
    ])
    def test_health_probe_shapes_stay_open(self, path, query, want_query):
        """Trailing slashes and a query glued into the path are the open
        liveness probe. The downstream app sees path /health, so the route
        answers 200 rather than a slash redirect."""
        status, seen = self._bearer_status(path, query)
        assert status == 200
        assert seen["path"] == "/health"
        assert seen["query"] == want_query

    @pytest.mark.parametrize("path", [
        "/healthcare", "/health/live", "/Health", "/v1/route",
    ])
    def test_health_neighbors_stay_gated(self, path):
        status, seen = self._bearer_status(path)
        assert status == 401
        assert seen == {}

    def test_each_v1_route_is_registered_once(self):
        """Each POST is one route. A second lambda registration shadows
        the named handler and makes the table lie about which code runs."""
        wrapped = create_app(token="t", router=object())
        paths = [getattr(route, "path", None) for route in wrapped.app.routes]
        for path in ("/v1/route", "/v1/extract", "/v1/guard",
                     "/v1/grade", "/v1/select", "/v1/repair", "/health"):
            assert paths.count(path) == 1, (path, paths)

    def test_bearer_prefix_required(self, authed_server):
        status, _ = authed_server.post(
            "/v1/route", {"text": "x"},
            auth_header="test-token-0123456789")  # no "Bearer " prefix
        assert status == 401


# --------------------------------------------------------------------------
# Bar 3: a live MCP client session over streamable HTTP.
# --------------------------------------------------------------------------


def test_mcp_client_roundtrip(open_server):
    anyio = pytest.importorskip("anyio")
    from mcp.client import Client

    url = f"http://127.0.0.1:{open_server.port}/mcp"

    def _payload(result) -> dict:
        """Tool results come back as JSON text content (bare-dict returns
        carry no structuredContent)."""
        assert result.is_error is False
        return json.loads(result.content[0].text)

    async def _run():
        async with Client(url, raise_exceptions=True) as client:
            listing = await client.list_tools()
            names = {t.name for t in listing.tools}
            assert {"mivia_route", "mivia_extract", "mivia_guard",
                    "mivia_grade", "mivia_select",
                    "mivia_repair_tool_call"} <= names

            out = await client.call_tool("mivia_route",
                                         {"text": "ERROR [auth]: token expired"})
            assert _payload(out)["schema"] == "log_event"

            out = await client.call_tool(
                "mivia_guard",
                {"request": "calculate the sum of 15 and 30",
                 "tools": [SUM_TOOL]})
            assert _payload(out)["action"] == "CALL"

            # A {name: description} map is a valid MCP tool argument too.
            out = await client.call_tool(
                "mivia_guard",
                {"request": "calculate the sum",
                 "tools": {"calculate_sum": "Calculate the sum of two numbers."}})
            assert _payload(out)["action"] in ("CALL", "CLARIFY")

            out = await client.call_tool(
                "mivia_extract",
                {"text": "hey what is this?\nJun 14 15:16:01 combo "
                         "sshd[9]: authentication failure; rhost=1.2.3.4",
                 "dump": True})
            records = _payload(out)["records"]
            # Re-pinned 2026-10-11 (N2 precedence): the chat line is kept
            # as a valid router-scored record; find the log by type.
            log_records = [r for r in records
                           if r["result"]["type"] == "log_event"]
            assert len(log_records) == 1
            assert any(r["result"]["type"] == "review_comment"
                       for r in records)

            out = await client.call_tool(
                "mivia_grade",
                {"text": "total outage, all customers down, data lost"})
            assert _payload(out)["result"]["action"] in ("act", "ask")

            out = await client.call_tool(
                "mivia_select",
                {"text": "checkout gateway 504",
                 "candidates": ["Payments", "Frontend"]})
            assert _payload(out)["choice"] in ("Payments", "Frontend")

            out = await client.call_tool("mivia_repair_tool_call",
                                         {"raw_arguments": "{'a': 1}"})
            assert _payload(out)["repaired"] == '{"a":1}'

    anyio.run(_run)


# --------------------------------------------------------------------------
# Bar 5: the stdlib surfaces stay untouched (regression sentinel).
# --------------------------------------------------------------------------


def test_stdlib_serving_modules_stay_stdlib_only():
    """The stdlib surfaces stay the zero-dependency path (regression
    sentinel, updated 2026-10-09).

    The original sentinel pinned a byte-identical serve.py/mcp.py, which
    blocked the reviewed serving hardening (F7 body caps / 411 / threaded
    unix socket, F8 stdio ping/notification/bad-byte fixes). The real
    invariant is dependency-freedom: neither module may import anything
    beyond the stdlib and this package.
    """
    import ast
    import sys
    from pathlib import Path

    from mivia_small_json import api as api_mod

    stdlib = set(sys.stdlib_module_names)
    local = {"mivia_small_json"}
    for name in ("serve.py", "mcp.py"):
        path = Path(api_mod.__file__).parent / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 \
                    and node.module:
                mods = [node.module.split(".")[0]]
            for mod in mods:
                assert mod in stdlib or mod in local, (
                    f"{name} imports {mod!r}: the stdlib sidecar must stay "
                    f"zero-extra-dependency")


class TestSingleForwardPass:
    def test_route_runs_one_encoder_pass(self, monkeypatch):
        """/v1/route must not re-encode the text for the in-domain flag.

        The route decision already computes the energy internally; under
        the shipped sentinel (-100.0, gate retired) the flag is decided
        without any second forward pass. A second pass doubles latency.
        """
        from starlette.testclient import TestClient

        from mivia_small_json import api
        from mivia_small_json.encoder import load_encoder

        router = load_encoder()
        if router is None:
            pytest.skip("shipped ONNX encoder absent; rules/skops-only checkout")

        calls = []
        real_run = router.sess.run

        def counting_run(*args, **kwargs):
            calls.append(1)
            return real_run(*args, **kwargs)

        monkeypatch.setattr(router.sess, "run", counting_run)
        client = TestClient(base_url="http://127.0.0.1",
                        app=api.create_app(token="", router=router))
        resp = client.post("/v1/route", json={"text": "ERROR [auth]: token expired"})
        assert resp.status_code == 200, resp.json()
        assert resp.json()["in_domain"] is True
        assert len(calls) <= 1, (
            f"/v1/route ran {len(calls)} encoder passes for one request")


class TestGuardOversize:
    def test_huge_tool_description_is_413(self, open_server):
        """The oversize law covers the TOOLS catalogue too, not just text:
        a multi-MB description must 413, not encode for seconds."""
        huge = {"name": "t.tool", "description": "x" * (MAX_TEXT_CHARS + 1)}
        status, body = open_server.post(
            "/v1/guard", {"request": "do the thing", "tools": [huge]})
        assert status == 413, (status, body)
