"""Integration tests: the model actually runs through the MCP server.

``tests/test_serve_mcp.py`` unit-tests ``mcp.dispatch`` / ``mcp.handle_line``
but always passes ``router=None``, so the model path and the ``python -m``
stdio loop are never exercised. These tests load the shipped router and
drive the real entry points end to end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mivia_small_json import bootstrap, mcp, pipeline, registry
from mivia_small_json.validate import validate

REPO_ROOT = Path(__file__).resolve().parents[1]

# Empirically verified against the shipped int8 encoder: this routes to
# log_event with confidence 1.0 and ``routed_by == "model"``, not the rules
# fallback. The assertion on ``routed_by`` is what makes the test fail if
# the model is not wired; the type assertion alone would pass under rules.
CLASSIFY_TEXT = "2026-03-01T12:00:00Z ERROR [auth]: token expired"
EXPECTED_TYPE = "log_event"


@pytest.fixture(scope="session")
def router():
    """Load the shipped router once (session scope; ~1-2 s to load)."""
    loaded = pipeline.load_active_router()
    if loaded is None:
        pytest.skip("no model artifact available; nothing to integrate")
    return loaded


def test_mcp_call_tool_uses_the_encoder(router):
    with registry.temporary_schemas():
        bootstrap.register_all_schemas()
        result = mcp.call_tool("mivia_classify", {"text": CLASSIFY_TEXT}, router)

        assert len(result["content"]) == 1
        payload = json.loads(result["content"][0]["text"])
        assert payload["meta"]["routed_by"] == "model", (
            "the MCP tool fell back to the rules router; the model is not wired"
        )
        assert payload["result"]["type"] == EXPECTED_TYPE
        assert validate(payload["result"], payload["result"]["type"]) == []


def test_mcp_stdio_server_end_to_end():
    """The ``python -m`` stdio loop: framing, exit code, model routing.

    This is the only test that catches a broken ``main()`` loop or stdout
    framing — ``dispatch``/``handle_line`` never touch the process.
    """
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "mivia_classify",
                    "arguments": {"text": CLASSIFY_TEXT}}},
    ]
    stdin = "\n".join(json.dumps(r) for r in requests) + "\n"
    proc = subprocess.run(
        [sys.executable, "-m", "mivia_small_json.mcp"],
        input=stdin, capture_output=True, text=True, cwd=REPO_ROOT,
    )

    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == len(requests), (
        f"expected one response line per id-bearing request, got "
        f"{len(lines)}: {proc.stdout!r}"
    )
    responses = [json.loads(ln) for ln in lines]
    assert [r["id"] for r in responses] == [1, 2, 3]

    payload = json.loads(responses[2]["result"]["content"][0]["text"])
    assert payload["meta"]["routed_by"] == "model", (
        "the stdio server fell back to the rules router"
    )
    assert payload["result"]["type"] == EXPECTED_TYPE


def test_mcp_stdio_reply_survives_lone_surrogate():
    """A lone surrogate (a legal ``\\ud800`` escape in the request) must
    not silently swallow the reply.

    Replies render with ``ensure_ascii=False``, so the raw surrogate made
    the strict-UTF-8 pipe write raise ``UnicodeEncodeError`` — which the
    blanket write-guard swallowed, sending NOTHING for that id while a
    well-behaved client waited forever. The guard now re-emits the reply
    with the surrogate as its JSON ``\\udXXX`` escape (what
    ``ensure_ascii=True`` would have written), so exactly one reply line
    must come back. This is the stdio twin of api.py's 500
    "response encoding failed" law.
    """
    request = {
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "mivia_classify",
                   "arguments": {"text": CLASSIFY_TEXT + " \\ud800"}},
    }
    stdin = json.dumps(request) + "\n"
    proc = subprocess.run(
        [sys.executable, "-m", "mivia_small_json.mcp"],
        input=stdin, capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, (
        f"the surrogate-bearing request must still get exactly one reply "
        f"line, got {len(lines)}: {proc.stdout!r}"
    )
    reply = json.loads(lines[0])
    assert reply["id"] == 7
    # The wire line carries the surrogate as the six-char \udXXX escape,
    # never as a raw byte the pipe refused.
    assert "\\ud800" in lines[0]
    if "result" in reply:
        payload = json.loads(reply["result"]["content"][0]["text"])
        assert payload["result"]["type"] == EXPECTED_TYPE
