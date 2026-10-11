# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Per-key metering + quota for the hosted API (spec
per-key-metering, registered 2026-10-10).

The monetization plane the pricing doc calls "the launch TODO": free tier
1,000 calls/month per key, paid keys skip the free cap (the prepaid-balance
decrement itself is billing-plane, out of scope). Stdlib only — ``sqlite3``,
``hashlib``, ``hmac``, ``secrets``, ``threading``, ``json``, ``datetime`` —
the serving image gains no dependency.

THE RETENTION CONTRACT (spec section 6; enforced by tests/test_metering.py
M1, which dumps the whole store file bytes): the store holds ONLY aggregate
counters and key metadata — key_hash, key_id, month, ok/error/reject
counts, quota_override, created/revoked/updated timestamps. NO request
text, NO response payloads, NO IPs, NO user agents, NO access logs. The
4 MiB body buffer used to detect an MCP ``tools/call`` lives only in RAM.

What counts as a call is the committed pricing FAQ's definition, verbatim:
one HTTP request to any /v1 endpoint, or one MCP tool invocation; a
4xx is not charged (only 2xx consumes quota).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import secrets
import sqlite3
import threading

KEY_PREFIX = "mvk_live_"
SCHEMA_VERSION = 1
DEFAULT_FREE_CALLS = 1000

# The MCP body-peek law: the same 4 MiB ceiling api.MAX_BODY_BYTES and the
# MCP SDK's streamable-HTTP cap already enforce (mirrored here on purpose —
# metering.py is stdlib-only and must not import the fastapi-facing api
# module; keep the two in sync). An over-cap /mcp body is refused 413
# UNMETERED (a 4xx is not charged), so the buffer is hard-bounded.
METER_MAX_BODY_BYTES = 4 * 1024 * 1024

_SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
    key_hash       TEXT PRIMARY KEY,
    key_id         TEXT NOT NULL UNIQUE,
    created_at     TEXT NOT NULL,
    revoked_at     TEXT,
    quota_override INTEGER
);
CREATE TABLE IF NOT EXISTS usage (
    key_hash   TEXT NOT NULL,
    month      TEXT NOT NULL,
    ok         INTEGER NOT NULL DEFAULT 0,
    error      INTEGER NOT NULL DEFAULT 0,
    reject     INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (key_hash, month)
);
"""

# The seven metered REST endpoints ("one HTTP request to any /v1 endpoint");
# exact-set, not a prefix, so future/unmounted /v1 paths are not silently
# metered. /v1/usage and /v1/admin/keys are never metered.
METERED_REST_PATHS = frozenset({
    "/v1/route", "/v1/extract", "/v1/guard", "/v1/grade",
    "/v1/select", "/v1/repair", "/v1/structure",
})


def key_hash(plaintext: str) -> str:
    """sha256 hex of the full plaintext key — the only stored form."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def month_of(when: dt.datetime) -> str:
    return f"{when.year:04d}-{when.month:02d}"


def reset_of(when: dt.datetime) -> str:
    """First day of next month UTC, ``2026-11-01T00:00:00Z`` shape."""
    year, month = (when.year + 1, 1) if when.month == 12 else \
        (when.year, when.month + 1)
    return f"{year:04d}-{month:02d}-01T00:00:00Z"


def seconds_until_reset(when: dt.datetime) -> int:
    year, month = (when.year + 1, 1) if when.month == 12 else \
        (when.year, when.month + 1)
    reset = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
    return max(0, int((reset - when).total_seconds()))


def default_clock() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class MeteringStore:
    """One WAL-mode sqlite file of aggregate counters + key metadata.

    A single connection guarded by an RLock: every admission decision plus
    its counter bump is one lock-held transaction, so the free cap is
    EXACT under sequential AND concurrent load (spec section 8). ``clock``
    is injectable for the M3 rollover test; ``:memory:`` is supported.
    """

    def __init__(self, path: str, clock=None):
        self.path = str(path)
        self.clock = clock or default_clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"metering store {self.path!r} has schema version "
                    f"{version}, newer than supported {SCHEMA_VERSION}")
            if version < SCHEMA_VERSION:
                self._conn.executescript(_SCHEMA)
                self._conn.execute(
                    f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    # -- keys -------------------------------------------------------------

    def create_key(self, quota_override: int | None = None) -> dict:
        """Generate a key; return it with its plaintext exactly once."""
        with self._lock:
            while True:
                token = secrets.token_urlsafe(24)
                plaintext = KEY_PREFIX + token
                kh = key_hash(plaintext)
                key_id = KEY_PREFIX + token[:8]  # display prefix
                created_at = self.clock().isoformat()
                try:
                    self._conn.execute(
                        "INSERT INTO keys (key_hash, key_id, created_at, "
                        "revoked_at, quota_override) VALUES (?,?,?,NULL,?)",
                        (kh, key_id, created_at, quota_override))
                    self._conn.commit()
                except sqlite3.IntegrityError:
                    continue  # 8-char prefix collision (probability ~0)
                return {"key": plaintext, "key_id": key_id, "key_hash": kh,
                        "created_at": created_at,
                        "quota_override": quota_override}

    def list_keys(self) -> list[dict]:
        """Key metadata only — never hashes, never plaintext."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key_id, created_at, revoked_at, quota_override "
                "FROM keys ORDER BY created_at, key_id").fetchall()
            return [dict(r) for r in rows]

    def revoke_key(self, key_id: str) -> dict | None:
        """Revoke by display id. ``None`` when unknown; idempotent when
        already revoked (returns the row's state)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT revoked_at FROM keys WHERE key_id = ?",
                (key_id,)).fetchone()
            if row is None:
                return None
            if row["revoked_at"] is None:
                self._conn.execute(
                    "UPDATE keys SET revoked_at = ? WHERE key_id = ?",
                    (self.clock().isoformat(), key_id))
                self._conn.commit()
            return {"key_id": key_id, "revoked": True}

    def lookup(self, plaintext: str) -> dict | None:
        """Resolve a presented plaintext key to an active-key identity, or
        ``None`` when unknown/revoked (auth failure, not anonymous)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT key_hash, key_id, quota_override FROM keys "
                "WHERE key_hash = ? AND revoked_at IS NULL",
                (key_hash(plaintext),)).fetchone()
        if row is None:
            return None
        return {"kind": "key", "key_hash": row["key_hash"],
                "key_id": row["key_id"],
                "quota_override": row["quota_override"]}

    # -- usage ------------------------------------------------------------

    def month(self) -> str:
        return month_of(self.clock())

    def reset(self) -> str:
        return reset_of(self.clock())

    def admit(self, kh: str, month: str, limit: int,
              paid: bool) -> tuple[bool, int]:
        """Atomically decide one metered call and bump its counter.

        Returns ``(admitted, ok_count_after)``. A free-tier key at/over
        the cap is refused (``reject`` +1, quota untouched — a 429 is not
        charged); otherwise ``ok`` is tentatively incremented as the
        reservation (a later non-2xx moves it to ``error`` — a 4xx/5xx is
        not charged either). Lock-serialized: exact under concurrency.
        """
        with self._lock:
            now = self.clock().isoformat()
            self._conn.execute(
                "INSERT INTO usage (key_hash, month, ok, error, reject, "
                "updated_at) VALUES (?,?,0,0,0,?) "
                "ON CONFLICT(key_hash, month) DO NOTHING", (kh, month, now))
            row = self._conn.execute(
                "SELECT ok FROM usage WHERE key_hash = ? AND month = ?",
                (kh, month)).fetchone()
            ok = int(row["ok"])
            if not paid and ok >= limit:
                self._conn.execute(
                    "UPDATE usage SET reject = reject + 1, updated_at = ? "
                    "WHERE key_hash = ? AND month = ?", (now, kh, month))
                self._conn.commit()
                return False, ok
            self._conn.execute(
                "UPDATE usage SET ok = ok + 1, updated_at = ? "
                "WHERE key_hash = ? AND month = ?", (now, kh, month))
            self._conn.commit()
            return True, ok + 1

    def move_to_error(self, kh: str, month: str) -> None:
        """Settle a reservation as ``error`` (4xx/5xx are not charged)."""
        with self._lock:
            self._conn.execute(
                "UPDATE usage SET ok = CASE WHEN ok > 0 THEN ok - 1 "
                "ELSE 0 END, error = error + 1, updated_at = ? "
                "WHERE key_hash = ? AND month = ?",
                (self.clock().isoformat(), kh, month))
            self._conn.commit()

    def counts(self, kh: str, month: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT ok, error, reject FROM usage "
                "WHERE key_hash = ? AND month = ?", (kh, month)).fetchone()
        if row is None:
            return {"ok": 0, "error": 0, "reject": 0}
        return {"ok": int(row["ok"]), "error": int(row["error"]),
                "reject": int(row["reject"])}

    def checkpoint(self) -> None:
        """Fold the WAL back into the main file (tests' full-bytes dump)."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


# --------------------------------------------------------------------------
# Metering middleware: pure ASGI, sits INSIDE the bearer gate (only ever
# sees authenticated requests carrying scope["mivia.identity"]) and wraps
# the whole FastAPI app, so REST and MCP hit the same counter.
# --------------------------------------------------------------------------


class MeteringMiddleware:
    """Per-key quota enforcement + usage headers (spec sections 7-10).

    Metered: POST to the seven /v1 endpoints (any outcome; only 2xx
    consumes quota) and POST /mcp whose JSON-RPC method is exactly
    ``tools/call`` (one tool invocation). Not metered: /health,
    /v1/usage, /v1/admin/keys, non-POSTs, MCP handshake/notification
    messages, unauthenticated traffic (no identity to attribute), and
    over-cap MCP bodies (413, refused unmetered — a 4xx is not charged).
    """

    def __init__(self, app, store: MeteringStore, free_limit: int):
        self.app = app
        self.store = store
        self.limit = free_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        ident = scope.get("mivia.identity") or {}
        metered, peek = self._metered_request(scope)
        if ident.get("kind") != "key" or not metered:
            await self.app(scope, receive, send)
            return
        if peek:
            body, over, recv = await _buffer_body(receive)
            if over:
                # The 4 MiB law (api._core / the MCP SDK cap), mirrored:
                # refuse unmetered — a 4xx is not charged, and buffering
                # stays hard-bounded at METER_MAX_BODY_BYTES.
                body_bytes = json.dumps(
                    {"error": "request body exceeds "
                              f"{METER_MAX_BODY_BYTES} bytes"}).encode()
                await send({"type": "http.response.start", "status": 413,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length",
                                 str(len(body_bytes)).encode())]})
                await send({"type": "http.response.body",
                            "body": body_bytes})
                return
            receive = recv
            if _jsonrpc_method(body) != "tools/call":
                # initialize / tools/list / notifications / unparseable:
                # not a tool invocation, not a call (pricing FAQ).
                await self.app(scope, receive, send)
                return
        await self._metered_call(scope, receive, send, ident)

    def _metered_request(self, scope) -> tuple[bool, bool]:
        """(needs_metering, needs_mcp_body_peek) for one HTTP request."""
        if (scope.get("method") or "").upper() != "POST":
            return False, False
        path = scope.get("path") or ""
        if path in METERED_REST_PATHS:
            return True, False
        if path == "/mcp":
            return True, True
        return False, False

    async def _metered_call(self, scope, receive, send, ident) -> None:
        kh = ident["key_hash"]
        paid = ident.get("quota_override") is not None
        month = self.store.month()
        admitted, ok_after = self.store.admit(kh, month, self.limit, paid)
        if not admitted:
            now = self.store.clock()
            await _quota_exceeded(send, limit=self.limit,
                                  reset=self.store.reset(),
                                  retry=seconds_until_reset(now),
                                  count=ok_after, paid=paid)
            return
        status_box: dict = {}

        async def send_wrap(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                status_box["status"] = status
                # Billable count AFTER this request's outcome settles:
                # a non-2xx moves its own reservation to `error` below.
                count = ok_after - 1 if status >= 400 else ok_after
                headers = list(message.get("headers") or [])
                headers.append((b"x-mivia-count", str(count).encode()))
                headers.append((b"x-mivia-reset",
                                self.store.reset().encode()))
                if not paid:
                    headers.append((b"x-mivia-limit",
                                    str(self.limit).encode()))
                message = dict(message, headers=headers)
            await send(message)

        try:
            await self.app(scope, receive, send_wrap)
        except Exception:
            self.store.move_to_error(kh, month)  # never a billed ok
            raise
        if status_box.get("status", 500) >= 400:
            self.store.move_to_error(kh, month)


async def _buffer_body(receive):
    """Buffer the request body (bounded by METER_MAX_BODY_BYTES) so the
    JSON-RPC method can be read; returns (body, over_cap, replay_receive).

    The buffer is RAM-only and never persisted (retention contract, M1).
    Over cap: the remaining stream is drained-DISCARDED (bounded memory,
    the api._core drain law) and ``over`` is True.
    """
    chunks = []
    total = 0
    over = False
    while True:
        message = await receive()
        mtype = message.get("type")
        if mtype == "http.request":
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > METER_MAX_BODY_BYTES:
                over = True
                chunks = []  # never keep what cannot be forwarded
            elif not over:
                chunks.append(chunk)
            if not message.get("more_body", False):
                break
        elif mtype == "http.disconnect":
            return b"", True, receive
    body = b"".join(chunks)

    sent = {"done": False}

    async def replay():
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return body, over, replay


def _jsonrpc_method(body: bytes) -> str | None:
    try:
        obj = json.loads(body or b"null")
    except ValueError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("method"), str):
        return obj["method"]
    return None


async def _quota_exceeded(send, *, limit: int, reset: str, retry: int,
                          count: int, paid: bool) -> None:
    """The 429: JSON body naming the reset date + the usage headers."""
    body = json.dumps({"error": "quota exceeded", "limit": limit,
                       "reset": reset}).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"retry-after", str(retry).encode()),
        (b"x-mivia-count", str(count).encode()),
        (b"x-mivia-reset", reset.encode()),
    ]
    if not paid:
        headers.append((b"x-mivia-limit", str(limit).encode()))
    await send({"type": "http.response.start", "status": 429,
                "headers": headers})
    await send({"type": "http.response.body", "body": body})
