"""per-key-metering spec: M1-M6 bars (per-key-metering).

Bars are pre-registered in the spec; this file is written RED-first against
them. The retention contract is the product law under test: the metering
store holds ONLY aggregate counters and key metadata — M1 dumps the whole
store file bytes and asserts real request markers are absent.

Focused suite only (owner rule 14):
    python -m pytest tests/test_metering.py tests/test_api.py \
        tests/test_serve_mcp.py -q
"""

from __future__ import annotations

import datetime as dt
import glob
import secrets
import sqlite3
import threading

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from mivia_small_json.api import create_app  # noqa: E402
from mivia_small_json.metering import (  # noqa: E402
    DEFAULT_FREE_CALLS,
    KEY_PREFIX,
    MeteringStore,
)

# router=False keeps create_app off the model-loading path (classify falls
# back to the rules baseline): these tests meter HTTP, they do not measure
# the router, and no test here needs the shipped ONNX artifacts.
RULES_ROUTER = False

ADMIN = "admin-" + secrets.token_urlsafe(16)


class FakeClock:
    """Injected clock (M3): deterministic UTC time the tests can advance."""

    def __init__(self, start=(2026, 10, 15, 12, 0, 0)):
        self._dt = dt.datetime(*start, tzinfo=dt.timezone.utc)
        self._lock = threading.Lock()

    def __call__(self) -> dt.datetime:
        with self._lock:
            return self._dt

    def advance(self, **kwargs) -> None:
        with self._lock:
            self._dt = self._dt + dt.timedelta(**kwargs)


def _build(tmp_path, *, limit=None, clock=None, token="", admin_key=ADMIN,
           metering=True, env_db=False):
    """One app + client (no lifespan: metering tests never warm models).

    Returns (store_or_None, client). With ``env_db`` the store comes from
    MIVIA_METERING_DB (the deployment enablement path) instead of an
    explicit factory argument.
    """
    store = None
    kwargs = {}
    if metering:
        if env_db:
            kwargs["metering_db"] = str(tmp_path / "envdb.sqlite3")
        else:
            store = MeteringStore(str(tmp_path / "metering.sqlite3"),
                                  clock=clock)
            kwargs["metering_store"] = store
    if limit is not None:
        kwargs["free_calls_limit"] = limit
    app = create_app(token=token, router=RULES_ROUTER, admin_key=admin_key,
                     **kwargs)
    # base_url 127.0.0.1:7200 — the loopback Host gate (REST) and the MCP
    # transport's "127.0.0.1:*" pattern both need a loopback Host WITH a
    # port; the TestClient default host "testserver" is rightly rejected.
    return store, TestClient(app, base_url="http://127.0.0.1:7200")


def _mkkey(client, admin_key=ADMIN, quota_override=None) -> dict:
    r = client.post("/v1/admin/keys",
                    json={"action": "create",
                          "quota_override": quota_override},
                    headers={"Authorization": f"Bearer {admin_key}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key"].startswith(KEY_PREFIX)
    return body


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _call(client, key: str, text: str = "ERROR [auth]: token expired"):
    return client.post("/v1/route", json={"text": text}, headers=_auth(key))


def _store_rows(path: str, table: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(f"SELECT * FROM {table}").fetchall()
    finally:
        conn.close()


def _store_blob(store: MeteringStore) -> bytes:
    """EVERY byte of the store: db + wal + shm (M1's full-file dump)."""
    store.checkpoint()
    blob = b""
    for name in sorted(glob.glob(store.path + "*")):
        with open(name, "rb") as fh:
            blob += fh.read()
    return blob


# --------------------------------------------------------------------------
# M1 — zero-leak: the store carries no request markers, ever.
# --------------------------------------------------------------------------

def test_m1_store_bytes_carry_no_request_markers(tmp_path):
    admin = ADMIN
    store, client = _build(tmp_path, token="static-tok", admin_key=admin)
    made = _mkkey(client, admin)
    plain_key = made["key"]

    marker_text = "MIVIA-LEAK-CANARY-7f3a9 owner-secret-phrase <stderr>"
    marker_header = "canary-header-3f9a1-value"
    marker_query = "canary-query-9fa3b"
    marker_ua = "canary-ua-11aa-agent"
    with TestClient(create_app(
            token="static-tok", router=RULES_ROUTER, admin_key=admin,
            metering_store=store)) as live:
        # REST entry, carrying markers in text, header, query and UA.
        r = live.post(f"/v1/route?{marker_query}=1", json={"text": marker_text},
                      headers={"Authorization": f"Bearer {plain_key}",
                               "X-Canary": marker_header,
                               "User-Agent": marker_ua})
        assert r.status_code == 200, r.text
        # MCP entry: one tools/call whose ARGUMENTS carry the canary.
        call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "mivia_route",
                           "arguments": {"text": marker_text + " mcp-arm"}}}
        r = live.post("/mcp", json=call,
                      headers={"Authorization": f"Bearer {plain_key}",
                               "Accept": "application/json, text/event-stream",
                               "User-Agent": marker_ua})
        assert r.status_code == 200, r.text

    blob = _store_blob(store)
    for marker in (marker_text, marker_header, marker_query, marker_ua,
                   plain_key, marker_text + " mcp-arm"):
        assert marker.encode("utf-8") not in blob, \
            f"retention contract broken: {marker!r} found in store bytes"
    # The store still functions: hash lookup finds the key...
    row = store.lookup(plain_key)
    assert row is not None and row["key_id"] == made["key_id"]
    # ...and only the hash, never the plaintext, is stored.
    conn = sqlite3.connect(store.path)
    try:
        stored = conn.execute("SELECT key_hash FROM keys").fetchall()
    finally:
        conn.close()
    assert len(stored) == 1
    import hashlib
    assert stored[0][0] == hashlib.sha256(plain_key.encode()).hexdigest()


# --------------------------------------------------------------------------
# M2 — the free-tier wall is EXACT at the limit; paid keys skip it.
# --------------------------------------------------------------------------

def test_m2_free_wall_fires_exactly_at_limit(tmp_path):
    limit = 5
    store, client = _build(tmp_path, limit=limit)
    key = _mkkey(client)["key"]
    for i in range(limit):
        r = _call(client, key)
        assert r.status_code == 200, (i, r.text)
        assert r.headers["x-mivia-count"] == str(i + 1)
        assert r.headers["x-mivia-limit"] == str(limit)
        assert r.headers["x-mivia-reset"]  # a reset date is always named
    # Call limit+1 is refused, names the reset date, does not consume.
    r = _call(client, key)
    assert r.status_code == 429, r.text
    body = r.json()
    assert body["error"] == "quota exceeded"
    assert body["limit"] == limit
    assert "reset" in body and body["reset"]
    assert r.headers["x-mivia-count"] == str(limit)
    usage = client.get("/v1/usage", headers=_auth(key)).json()
    assert usage["ok"] == limit and usage["reject"] == 1
    assert usage["quota_exceeded"] is True


def test_m2_paid_key_skips_the_free_cap(tmp_path):
    store, client = _build(tmp_path, limit=3)
    key = _mkkey(client, quota_override=1)["key"]
    for i in range(7):  # well past the free cap of 3
        r = _call(client, key)
        assert r.status_code == 200, (i, r.text)
        assert r.headers["x-mivia-count"] == str(i + 1)
        assert "x-mivia-limit" not in r.headers  # no free limit applies
    usage = client.get("/v1/usage", headers=_auth(key)).json()
    assert usage["limit"] is None and usage["quota_exceeded"] is False


def test_m2_env_free_calls_limit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MIVIA_FREE_CALLS_LIMIT", "2")
    store, client = _build(tmp_path)  # no explicit limit: env governs
    key = _mkkey(client)["key"]
    assert _call(client, key).status_code == 200
    assert _call(client, key).status_code == 200
    r = _call(client, key)
    assert r.status_code == 429
    assert r.json()["limit"] == 2


def test_m2_default_free_limit_is_1000(tmp_path, monkeypatch):
    monkeypatch.delenv("MIVIA_FREE_CALLS_LIMIT", raising=False)
    store, client = _build(tmp_path)
    usage_zero = client.get(
        "/v1/usage", headers=_auth(_mkkey(client)["key"])).json()
    assert usage_zero["limit"] == DEFAULT_FREE_CALLS == 1000


# --------------------------------------------------------------------------
# M3 — month rollover resets the wall (injected clock).
# --------------------------------------------------------------------------

def test_m3_month_rollover_resets_counters(tmp_path):
    clock = FakeClock((2026, 10, 15, 12, 0, 0))
    limit = 2
    store, client = _build(tmp_path, limit=limit, clock=clock)
    key = _mkkey(client)["key"]
    made = client.get("/v1/usage", headers=_auth(key)).json()
    assert made["month"] == "2026-10"

    assert _call(client, key).status_code == 200
    assert _call(client, key).status_code == 200
    assert _call(client, key).status_code == 429

    clock.advance(days=20)  # 2026-10-15 -> 2026-11-04
    # The same key is unblocked: rollover addresses fresh counter rows.
    assert _call(client, key).status_code == 200
    assert _call(client, key).status_code == 200
    r = _call(client, key)
    assert r.status_code == 429
    usage = client.get("/v1/usage", headers=_auth(key)).json()
    assert usage["month"] == "2026-11"
    assert usage["ok"] == 2 and usage["reject"] == 1
    assert usage["reset"].startswith("2026-12-01")
    # History preserved, never carried over: month A keeps its own rows.
    old = store.counts(store.lookup(key)["key_hash"], "2026-10")
    assert old == {"ok": 2, "error": 0, "reject": 1}


# --------------------------------------------------------------------------
# M4 — outcome classes are declared AND tested (spec section 7).
# --------------------------------------------------------------------------

def test_m4_unauthenticated_requests_record_nothing(tmp_path):
    # A token must be set for "unauthenticated" to mean 401 (open mode has
    # no auth by contract — its anonymous traffic is covered below).
    store, client = _build(tmp_path, limit=5, token="static-tok")
    key_row = _mkkey(client)
    assert client.post("/v1/route", json={"text": "x"}).status_code == 401
    assert client.post("/v1/route", json={"text": "x"},
                       headers=_auth("mvk_live_not-a-real-key")).status_code \
        == 401
    # No usage row exists at all for the untouched key.
    path = store.path
    rows = _store_rows(path, "usage")
    assert rows == []
    # And the key still works afterwards (nothing was consumed).
    assert _call(client, key_row["key"]).status_code == 200


def test_m4_error_calls_are_recorded_but_never_charged(tmp_path):
    limit = 2
    store, client = _build(tmp_path, limit=limit)
    key = _mkkey(client)["key"]
    # A 422 (validation error) records `error` and consumes NO quota...
    r = client.post("/v1/route", json={"text": 123}, headers=_auth(key))
    assert r.status_code == 422
    usage = client.get("/v1/usage", headers=_auth(key)).json()
    assert usage["ok"] == 0 and usage["error"] == 1 and usage["reject"] == 0
    # ...proved by the wall: the next `limit` calls are all 200 with the
    # billable count starting at 1 (the 422 did not consume).
    for i in range(limit):
        r = _call(client, key)
        assert r.status_code == 200
        assert r.headers["x-mivia-count"] == str(i + 1)
    assert _call(client, key).status_code == 429
    usage = client.get("/v1/usage", headers=_auth(key)).json()
    assert usage == {"key_id": usage["key_id"], "month": usage["month"],
                     "ok": limit, "error": 1, "reject": 1, "limit": limit,
                     "reset": usage["reset"], "quota_exceeded": True}


def test_m4_static_token_and_admin_are_not_metered(tmp_path):
    store, client = _build(tmp_path, token="static-tok")
    for i in range(3):
        r = client.post("/v1/repair", json={"raw_arguments": "{'a': 1}"},
                        headers={"Authorization": "Bearer static-tok"})
        assert r.status_code == 200
        assert "x-mivia-count" not in r.headers
        r = client.post("/v1/repair", json={"raw_arguments": "{'a': 1}"},
                        headers={"Authorization": f"Bearer {ADMIN}"})
        assert r.status_code == 200
        assert "x-mivia-count" not in r.headers
    assert _store_rows(store.path, "usage") == []


def test_m4_admin_keys_body_under_the_same_ceiling(tmp_path):
    """The admin key route used to read request.json() unbounded — the one
    JSON endpoint outside the 4 MiB whole-body ceiling (review 2026-10-11:
    an over-cap admin body grew memory without bound). It shares the
    bounded reader now: an over-cap body is a 413."""
    _, client = _build(tmp_path, token="static-tok")
    r = client.post(
        "/v1/admin/keys",
        content=b"x" * (4 * 1024 * 1024 + 1),
        headers={"Authorization": f"Bearer {ADMIN}",
                 "Content-Type": "application/json"})
    assert r.status_code == 413, r.text


def test_m4_usage_reports_own_row_only(tmp_path):
    store, client = _build(tmp_path, limit=5)
    key_a = _mkkey(client)["key"]
    key_b = _mkkey(client)["key"]
    assert _call(client, key_a).status_code == 200
    usage_b = client.get("/v1/usage", headers=_auth(key_b)).json()
    assert usage_b["ok"] == 0 and usage_b["key_id"] != "mvk_live_"
    usage_a = client.get("/v1/usage", headers=_auth(key_a)).json()
    assert usage_a["ok"] == 1


def test_m4_unmetered_identity_gets_metered_false(tmp_path):
    _, client = _build(tmp_path, token="static-tok")
    r = client.get("/v1/usage", headers={"Authorization": "Bearer static-tok"})
    assert r.status_code == 200 and r.json() == {"metered": False}
    r = client.get("/v1/usage", headers={"Authorization": f"Bearer {ADMIN}"})
    assert r.status_code == 200 and r.json() == {"metered": False}


# --------------------------------------------------------------------------
# MCP entry: same counter, and non-tools/call traffic is not metered.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mcp_surface(tmp_path_factory):
    """One app WITH lifespan (the MCP session manager needs it)."""
    tmp = tmp_path_factory.mktemp("mcp")
    store = MeteringStore(str(tmp / "metering.sqlite3"), clock=FakeClock())
    app = create_app(token="", router=RULES_ROUTER, admin_key=ADMIN,
                     metering_store=store, free_calls_limit=5)
    with TestClient(app, base_url="http://127.0.0.1:7200") as client:
        yield store, client


def _mcp(client, payload, key=None):
    headers = {"Accept": "application/json, text/event-stream"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return client.post("/mcp", json=payload, headers=headers)


def test_mcp_tools_call_shares_the_counter(mcp_surface):
    store, client = mcp_surface
    key = _mkkey(client)["key"]
    # Handshake traffic (initialize / notifications / tools/list) is NOT a
    # tool invocation — unmetered (pricing FAQ definition).
    r = _mcp(client, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2025-06-18",
                                 "capabilities": {},
                                 "clientInfo": {"name": "t", "version": "0"}}},
             key=key)
    assert r.status_code == 200, r.text
    _mcp(client, {"jsonrpc": "2.0", "method": "notifications/initialized"},
         key=key)
    _mcp(client, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, key=key)
    assert store.counts(store.lookup(key)["key_hash"], "2026-10")["ok"] == 0
    # One tools/call = one call, counted on the same store REST uses.
    r = _mcp(client, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "mivia_route",
                                 "arguments": {"text": "ERROR [auth]: x"}}},
             key=key)
    assert r.status_code == 200, r.text
    counts = store.counts(store.lookup(key)["key_hash"], "2026-10")
    assert counts["ok"] == 1 and counts["error"] == 0


def test_mcp_tools_call_consumes_quota_and_hits_429(mcp_surface):
    store, client = mcp_surface
    key = _mkkey(client)["key"]
    call = {"jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {"name": "mivia_route",
                       "arguments": {"text": "ERROR [auth]: x"}}}
    for i in range(5):
        assert _mcp(client, call, key=key).status_code == 200
    r = _mcp(client, call, key=key)
    assert r.status_code == 429, r.text
    assert r.json()["error"] == "quota exceeded"
    counts = store.counts(store.lookup(key)["key_hash"], "2026-10")
    assert counts["ok"] == 5 and counts["reject"] == 1


# --------------------------------------------------------------------------
# Admin surface: create (plaintext once) / list / revoke; guard 401/403.
# --------------------------------------------------------------------------

def test_admin_create_list_revoke_lifecycle(tmp_path):
    _, client = _build(tmp_path)
    made = _mkkey(client)
    # Plaintext shown exactly once: create returns it, list never does.
    listed = client.post("/v1/admin/keys", json={"action": "list"},
                         headers=_auth(ADMIN)).json()["keys"]
    assert len(listed) == 1
    entry = listed[0]
    assert entry["key_id"] == made["key_id"]
    assert "key" not in entry and "key_hash" not in entry
    # The plaintext authenticates exactly one key's worth of traffic.
    assert _call(client, made["key"]).status_code == 200
    # Revoke, then the same plaintext is 401.
    r = client.post("/v1/admin/keys",
                    json={"action": "revoke", "key_id": made["key_id"]},
                    headers=_auth(ADMIN))
    assert r.status_code == 200 and r.json()["revoked"] is True
    assert _call(client, made["key"]).status_code == 401
    # Revoking is idempotent for a known key, 404 for an unknown one.
    r = client.post("/v1/admin/keys",
                    json={"action": "revoke", "key_id": made["key_id"]},
                    headers=_auth(ADMIN))
    assert r.status_code == 200
    r = client.post("/v1/admin/keys",
                    json={"action": "revoke", "key_id": "mvk_live_nope"},
                    headers=_auth(ADMIN))
    assert r.status_code == 404


def test_admin_rejects_boolean_quota_override(tmp_path):
    # bool is an int subclass in Python — a JSON `true` must not mint a
    # paid key (review 2026-10-10, P3).
    _, client = _build(tmp_path)
    for bad in (True, False):
        r = client.post("/v1/admin/keys",
                        json={"action": "create", "quota_override": bad},
                        headers=_auth(ADMIN))
        assert r.status_code == 422, bad
    listed = client.post("/v1/admin/keys", json={"action": "list"},
                         headers=_auth(ADMIN)).json()["keys"]
    assert listed == [], "a boolean quota_override must mint no key"


def test_admin_surface_guard(tmp_path):
    _, client = _build(tmp_path, token="static-tok")
    key = _mkkey(client)["key"]
    # Unauthenticated -> 401 (bearer gate); authenticated non-admin -> 403.
    assert client.post("/v1/admin/keys", json={"action": "list"}).status_code \
        == 401
    r = client.post("/v1/admin/keys", json={"action": "list"},
                    headers={"Authorization": "Bearer static-tok"})
    assert r.status_code == 403
    r = client.post("/v1/admin/keys", json={"action": "list"},
                    headers=_auth(key))
    assert r.status_code == 403
    # Unknown action is a caller error (422), not a crash.
    r = client.post("/v1/admin/keys", json={"action": "explode"},
                    headers=_auth(ADMIN))
    assert r.status_code == 422


# --------------------------------------------------------------------------
# Enablement + metering-off contract (spec decisions 4 and 5, M5 support).
# --------------------------------------------------------------------------

def test_metering_off_is_todays_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("MIVIA_METERING_DB", raising=False)
    monkeypatch.delenv("MIVIA_ADMIN_KEY", raising=False)
    _, client = _build(tmp_path, token="static-tok", metering=False,
                       admin_key=None)
    r = client.post("/v1/repair", json={"raw_arguments": "{'a': 1}"},
                    headers={"Authorization": "Bearer static-tok"})
    assert r.status_code == 200
    assert "x-mivia-count" not in r.headers
    # The metering plane does not exist when it is off.
    assert client.get("/v1/usage",
                      headers={"Authorization": "Bearer static-tok"}
                      ).status_code == 404
    assert client.post("/v1/admin/keys", json={"action": "list"},
                       headers={"Authorization": "Bearer static-tok"}
                       ).status_code == 404
    # Wrong bearer is still exactly today's 401 body.
    assert client.post("/v1/route", json={"text": "x"},
                       headers={"Authorization": "Bearer nope"}
                       ).json() == {"error": "unauthorized"}


def test_env_metering_db_enablement(tmp_path, monkeypatch):
    monkeypatch.setenv("MIVIA_METERING_DB", str(tmp_path / "envdb.sqlite3"))
    monkeypatch.setenv("MIVIA_ADMIN_KEY", ADMIN)
    _, client = _build(tmp_path, env_db=True)
    key = _mkkey(client)["key"]
    assert _call(client, key).status_code == 200
    rows = _store_rows(str(tmp_path / "envdb.sqlite3"), "usage")
    assert len(rows) == 1 and rows[0]["ok"] == 1


def test_tokenless_open_mode_still_resolves_keys(tmp_path, monkeypatch):
    # Open mode stays open (today's contract), and a valid mvk key is
    # metered while a BAD mvk credential is an auth failure, not anonymous.
    _, client = _build(tmp_path, token="")  # tokenless
    made = _mkkey(client)
    assert client.post("/v1/route", json={"text": "x"}).status_code == 200
    assert _call(client, made["key"]).status_code == 200
    assert client.post("/v1/route", json={"text": "x"},
                       headers=_auth("mvk_live_bogus")).status_code == 401
