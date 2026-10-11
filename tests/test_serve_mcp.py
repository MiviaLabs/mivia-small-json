"""Tests for the sidecar daemon and the MCP server (no sockets)."""

import gc
import json

import pytest

from mivia_small_json import mcp, registry, serve
from mivia_small_json.validate import validate


@pytest.fixture(autouse=True)
def _schemas():
    from mivia_small_json import bootstrap

    bootstrap.register_example_schemas()
    yield


def test_serve_classify_returns_result_and_meta():
    body = serve.handle_classify(
        {"text": "2026-03-01T12:00:00Z ERROR [auth]: token expired"}, None)
    assert body["result"]["type"] == "log_event"
    assert set(body["meta"]) == {"routed_by", "valid", "retries", "latency_ms"}
    assert body["meta"]["valid"] is True


def test_serve_classify_rejects_a_non_string_text():
    with pytest.raises(ValueError):
        serve.handle_classify({"text": 42}, None)


def test_serve_classify_rejects_missing_or_blank_text():
    """A missing key (typo'd field) or blank text must be a 400, not a
    silent classification of "" into a general-text record (2026-10-09
    serving probe)."""
    with pytest.raises(ValueError):
        serve.handle_classify({}, None)  # e.g. {"txt": "..."} typo
    with pytest.raises(ValueError):
        serve.handle_classify({"text": ""}, None)
    with pytest.raises(ValueError):
        serve.handle_classify({"text": "   "}, None)
    with pytest.raises(ValueError):
        serve.handle_classify_dump({}, None)
    with pytest.raises(ValueError):
        serve.handle_classify_dump({"text": ""}, None)
    status, body = serve.dispatch("POST", "/classify", {}, None)
    assert status == 400 and "text" in body["error"]


def test_mcp_classify_rejects_missing_or_blank_text():
    for bad in ({}, {"text": ""}, {"text": "  "}):
        with pytest.raises(ValueError):
            mcp.call_tool("mivia_classify", bad, None)
    # the JSON-RPC surface maps it to a tool error, not a wrong record
    resp = json.loads(mcp.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mivia_classify", "arguments": {}}}), None))
    assert resp["error"]["code"] == -32602


def test_serve_classify_dump_returns_records():
    body = serve.handle_classify_dump(
        {"text": "step 1: tool call execute_bash; return_code: 0"}, None)
    assert len(body["records"]) == 1
    assert body["records"][0]["result"]["type"] == "agent_step"


def test_serve_repair_arguments():
    body = serve.handle_repair({"raw_arguments": "{'a': True,}"})
    assert json.loads(body["repaired"]) == {"a": True}
    with pytest.raises(ValueError):
        serve.handle_repair({"raw_arguments": None})


def test_serve_health_payload():
    body = serve.health_payload(None)
    assert body["status"] == "ok"
    assert isinstance(body["rss_mb"], float)
    assert body["model"] in ("onnx-int8", "skops", "rules")


# ---------------------------------------------------------------------------
# RSS reader law (2026-10-10 serving finding): /health must report the
# CURRENT resident set. ru_maxrss is a high-water mark — after freeing an
# 80 MB block the old reader stayed at 99.1 MiB while /proc VmRSS fell from
# ~101,880 KiB to ~19,956 KiB — so serve.rss_mb reads /proc/self/status
# VmRSS on Linux and keeps ru_maxrss only as the portable fallback.
# ---------------------------------------------------------------------------

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


def test_rss_mb_returns_a_positive_float():
    value = serve.rss_mb()
    assert isinstance(value, float)
    assert value > 0


def test_rss_mb_matches_current_vmrss_within_a_sane_band():
    kernel = _vmrss_mib()
    if kernel is None:
        pytest.skip("no /proc/self/status (non-Linux); ru_maxrss fallback")
    value = serve.rss_mb()
    # Allocator caching makes exact equality meaningless; require the
    # reader to sit in a sane band around the kernel's current number.
    assert abs(value - kernel) <= max(8.0, 0.25 * kernel), (value, kernel)


@pytest.mark.skipif(_vmrss_mib() is None,
                    reason="the free-drop law is observable on Linux only")
def test_rss_mb_comes_down_after_a_freed_block():
    """Current-RSS law: after a freed 64 MiB block the reading must come
    DOWN. ru_maxrss (the old reader) can never decrease, so under the bug
    second == first and this test is red at HEAD. Deterministic: a 64 MiB
    bytearray is mmap-backed and munmap'd on free, so the kernel gets the
    pages back — no exact-number asserts, only a drop of at least a
    quarter of the freed block."""
    block = bytearray(64 * 1024 * 1024)
    for i in range(0, len(block), 4096):  # touch every page -> resident
        block[i] = 1
    first = serve.rss_mb()
    del block
    gc.collect()
    second = serve.rss_mb()
    assert second <= first - 16.0, (first, second)


def test_serve_dispatch_routes_and_errors():
    status, body = serve.dispatch("GET", "/health", {}, None)
    assert status == 200 and body["status"] == "ok"
    status, _ = serve.dispatch("GET", "/missing", {}, None)
    assert status == 404
    status, _ = serve.dispatch("POST", "/classify", {"text": 1}, None)
    assert status == 400


def test_mcp_initialize_and_tool_list():
    router = None
    resp = mcp.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {}}, router)
    assert resp["result"]["protocolVersion"] == mcp.PROTOCOL_VERSION
    assert resp["result"]["serverInfo"]["name"] == "mivia-small-json"
    resp = mcp.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                         "params": {}}, router)
    names = [t["name"] for t in resp["result"]["tools"]]
    assert names == ["mivia_classify", "mivia_classify_dump",
                     "mivia_repair_tool_call"]


def _call_text(resp):
    assert resp["jsonrpc"] == "2.0"
    assert len(resp["result"]["content"]) == 1
    return json.loads(resp["result"]["content"][0]["text"])


def test_mcp_classify_tool():
    resp = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "mivia_classify",
                    "arguments": {"text": "step 2: tool call ls; return_code: 0"}}},
        None)
    payload = _call_text(resp)
    assert payload["result"]["type"] == "agent_step"
    assert validate(payload["result"], "agent_step") == []


def test_mcp_classify_dump_tool():
    resp = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "mivia_classify_dump",
                    "arguments": {"text": "step 2: tool call ls"}}}, None)
    payload = _call_text(resp)
    assert len(payload["records"]) == 1


def test_mcp_repair_tool():
    resp = mcp.dispatch(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "mivia_repair_tool_call",
                    "arguments": {"raw_arguments": "{'x': None}"}}}, None)
    # 2026-10-10 C2 parity: stdio carries the same lossy flag as REST.
    assert _call_text(resp) == {"repaired": '{"x":null}', "lossy": False}


def test_register_all_schemas_boots_the_full_world():
    from mivia_small_json import bootstrap

    # Restore-on-exit is the suite-wide autouse hermetic fixture.
    bootstrap.register_all_schemas()
    bootstrap.register_all_schemas()
    assert {"issue", "log_event", "commit", "pull_request", "tool_call",
            "no_tool_call", "tool_clarification", "diagnostic", "diff",
            "review_comment", "agent_step"} <= set(registry.active())


def test_mcp_errors_and_notifications():
    assert mcp.dispatch(
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "nope", "arguments": {}}},
        None)["error"]["code"] == -32602
    assert mcp.dispatch({"jsonrpc": "2.0", "id": 7, "method": "nope"}, None
                        )["error"]["code"] == -32601
    assert mcp.dispatch([1, 2], None)["error"]["code"] == -32600
    assert mcp.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized",
                         "params": {}}, None) is None
    assert mcp.handle_line("not json", None) is not None
    assert json.loads(mcp.handle_line(
        "not json", None))["error"]["code"] == -32700
    assert "agent_step" in registry.active()
