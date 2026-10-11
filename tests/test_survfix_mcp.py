"""Survivor fixes for the stdio MCP server and the CLI (survfix-mcp).

Execution-confirmed red states (validation evidence:
``.scratch/validate-a/v7b_mcp_pipe.py``, ``.scratch/validate-a/v7_mcp.py``,
``.scratch/validate-b/v4_infer_router.py``):

1. HIGH: the response ``write``/``flush`` in :func:`mcp.main` sat OUTSIDE
   the per-message ``try``/``except``, so one ``BrokenPipeError`` (client
   closed without reading the reply) killed the whole stdio server with a
   traceback. Contract: a broken pipe exits gracefully (exit 0, no
   traceback — the client's stdin is gone with it); any other write error
   follows the loop's never-crash contract (the session survives).

2. Parity: a deeply-nested JSON line raised ``RecursionError`` out of
   ``json.loads``; ``handle_line`` only caught ``ValueError``, so the loop
   answered ``-32603 "RecursionError..."`` with ``id:null``. REST maps the
   same input to 422 (a bad request), so stdio must answer the same law:
   ``-32700 Parse error``.

3. Divergence: ``infer.py`` loaded the skops fallback router directly, so
   the CLI disagreed with every served surface (``load_active_router`` ->
   ONNX encoder) on the same input — e.g. ``'سلام خوبی چطوری'`` routes
   ``review_comment`` on the CLI vs ``text`` served. Contract: the CLI
   classifies through ``pipeline.load_active_router`` exactly like
   serve.py / api.py / mcp.py.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mivia_small_json import infer, mcp, pipeline
from mivia_small_json.model import MODEL_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]

# Texts where the skops fallback and the shipped ONNX encoder disagree
# (confirmed by .scratch/validate-b/v4_infer_router.py on 2026-10-10).
DIVERGENT_TEXTS = [
    "سلام خوبی چطوری",
    "please write me a poem about the sea",
]


class FakeStdin:
    """Bytes stdin stand-in: ``main()`` reads ``stdin.readline()``."""

    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else b""


class RecorderStdout:
    def __init__(self):
        self.written = []

    def write(self, s):
        self.written.append(s)
        return len(s)

    def flush(self):
        pass


class BrokenPipeStdout(RecorderStdout):
    def write(self, s):
        raise BrokenPipeError(32, "Broken pipe")


class DyingStdout(RecorderStdout):
    """A write error that is NOT a broken pipe (closed-file style)."""

    def write(self, s):
        raise ValueError("I/O operation on closed file")


@pytest.fixture(autouse=True)
def _no_router_load():
    """The stdio loop must behave identically with or without a model;
    shimming the loader keeps these protocol tests off the ONNX disk hit."""
    real = pipeline.load_active_router
    pipeline.load_active_router = lambda: None
    yield
    pipeline.load_active_router = real


# --- bug 1: broken pipe on stdout must not kill the server ------------------


def test_broken_pipe_write_exits_gracefully(monkeypatch):
    """The v7b probe scenario in-process: the first reply hits a pipe the
    client already closed; main() must return 0, never raise."""
    monkeypatch.setattr(sys, "stdin", FakeStdin(
        [b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n']))
    monkeypatch.setattr(sys, "stdout", BrokenPipeStdout())
    rc = mcp.main([])  # red: BrokenPipeError escapes main()
    assert rc == 0


def test_broken_pipe_after_first_reply_subprocess_exits_cleanly():
    """Real process: one huge reply (larger than the 64 KiB pipe buffer)
    can never fit, so the child is necessarily still writing when the
    client closes the read end — the exact "closed without reading" event.
    The child must exit 0 with no traceback.

    raw_arguments stays under the 262,144-char cap so the reply is a real
    result, not an early -32602 error (which would be small enough to
    never hit a broken pipe)."""
    big = "a" * 250_000
    request = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mivia_repair_tool_call",
                   "arguments": {"raw_arguments": json.dumps({"x": big})}},
    }) + "\n"
    assert len(request) < mcp.MAX_TEXT_CHARS + 64  # under the tool's cap
    proc = subprocess.Popen(
        [sys.executable, "-m", "mivia_small_json.mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, cwd=REPO_ROOT,
    )
    proc.stdin.write(request.encode())
    proc.stdin.flush()
    proc.stdin.close()
    # Close the read end while the child still owes us >64 KiB of reply:
    # its blocked/in-flight write raises BrokenPipeError.
    proc.stdout.close()
    try:
        rc = proc.wait(timeout=120)
    finally:
        if proc.poll() is None:  # never hang the suite on a regression
            proc.kill()
            proc.wait()
    stderr = proc.stderr.read().decode("utf-8", "replace")
    proc.stderr.close()
    assert rc == 0, f"server died on the closed pipe (rc={rc})\n{stderr}"
    assert "Traceback" not in stderr, stderr
    assert "exception escaped mcp.main" not in stderr, stderr


def test_other_write_error_never_kills_the_session(monkeypatch):
    """A non-broken-pipe write failure follows the never-crash contract:
    the response is dropped, the loop keeps reading stdin."""
    out = DyingStdout()
    monkeypatch.setattr(sys, "stdin", FakeStdin(
        [b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n',
         b'{"jsonrpc":"2.0","id":2,"method":"ping"}\n']))
    monkeypatch.setattr(sys, "stdout", out)
    rc = mcp.main([])  # red: ValueError escapes main()
    assert rc == 0


# --- bug 2: RecursionError is a Parse error (-32700), not -32603 ------------

DEEP_LINE = "[" * 100_000 + "]" * 100_000


def test_deeply_nested_line_answers_parse_error():
    out = mcp.handle_line(DEEP_LINE, None)  # red: RecursionError escapes
    assert out is not None
    resp = json.loads(out)
    assert resp["id"] is None
    assert resp["error"]["code"] == -32700
    assert resp["error"]["message"] == "Parse error"


def test_deeply_nested_line_does_not_kill_the_loop(monkeypatch):
    """End-to-end through main(): the deep line answers -32700 and the
    next request is still served (the loop never dies on one message)."""
    out = RecorderStdout()
    monkeypatch.setattr(sys, "stdin", FakeStdin(
        [DEEP_LINE.encode() + b"\n",
         b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n']))
    monkeypatch.setattr(sys, "stdout", out)
    assert mcp.main([]) == 0
    assert len(out.written) == 2, out.written
    first, second = (json.loads(w) for w in out.written)
    assert first["error"]["code"] == -32700  # red: was -32603 RecursionError
    assert first["error"]["message"] == "Parse error"
    assert second == {"jsonrpc": "2.0", "id": 1, "result": {}}


# --- bug 3: the CLI routes through the served (active) router ---------------


@pytest.fixture(scope="module")
def active_router():
    loaded = pipeline.load_active_router()
    if loaded is None:
        pytest.skip("no model artifact available")
    return loaded


def test_infer_wiring_goes_through_load_active_router(monkeypatch):
    """infer.main must obtain its default router from
    pipeline.load_active_router (the served surface), never from a direct
    skops load. Red today: infer.py has no such hook at all."""
    calls = []
    real_load_router = infer.load_router

    def fake_active():
        calls.append(True)
        return real_load_router(MODEL_PATH)  # deterministic skops stand-in

    monkeypatch.setattr(infer, "load_active_router", fake_active)

    class _Stdin:
        buffer = iter([("salam\n").encode()])

    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdin", _Stdin())
    monkeypatch.setattr(sys, "stdout", sink)
    assert infer.main([]) == 0
    assert calls == [True], (
        "infer.main never consulted load_active_router; it disagrees with "
        "every served surface on the same input")


def test_infer_matches_the_active_router_on_divergent_texts(
        monkeypatch, active_router):
    """For texts where the skops fallback and the shipped encoder disagree,
    the CLI must agree with the served surface (the active router)."""
    lines = "".join(t + "\n" for t in DIVERGENT_TEXTS)

    class _Stdin:
        buffer = iter(lines.encode().splitlines(keepends=True))

    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdin", _Stdin())
    monkeypatch.setattr(sys, "stdout", sink)
    assert infer.main([]) == 0

    emitted = [json.loads(ln) for ln in sink.getvalue().splitlines() if ln]
    assert len(emitted) == len(DIVERGENT_TEXTS)
    for text, rec in zip(DIVERGENT_TEXTS, emitted):
        expected, expected_meta = pipeline.classify(text, active_router)
        assert rec["obj"]["type"] == expected["type"], (
            f"CLI/served divergence on {text!r}: infer routed "
            f"{rec['obj']['type']}, served surface routed {expected['type']}")
        assert rec["meta"]["routed_by"] == expected_meta["routed_by"]
