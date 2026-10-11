"""No-truncation contract (owner directive 2026-10-10).

"we can't truncate anything in the output for user — it is his data,
users will expect it back."

Every extracted content field carries the FULL matched content: no
length slices, no first-line-only captures, no fabricated digits.
Output is bounded by the request's 256 KiB input cap (api.MAX_TEXT_CHARS),
so a record can never exceed what the user sent.

Guarded surfaces (2026-10-10):
- log_event message   (was msg[:200])
- commit message      (was [:300])
- diagnostic message  (was first-line-only ``[^\\n\\r]+``)
- agent_step error_message (was first-line/semicolon-only ``[^\\n;]+``)
- normalize_sha       (was zero-padding < 7 hex chars: "abc12" -> "abc1200")
"""

import pytest

from mivia_small_json import rules
from mivia_small_json.agent_step import extract_agent_step
from mivia_small_json.diagnostic import extract_diagnostic, \
    register_diagnostic_schema
from mivia_small_json import pipeline
from mivia_small_json import validate as V


@pytest.fixture(autouse=True)
def _diagnostic_world():
    """The diagnostic schema must be active for routing + the atomic-take
    path (rolled back by the suite-wide hermetic-registry fixture)."""
    register_diagnostic_schema()
    yield


class _FixedRouter:
    """Deterministic stand-in for the encoder: routes every text to one
    schema. Carries the ``classes``/``_logits`` surface the dump's
    atomic-take fast path probes, so the per-line pass is one batch."""

    _logits = True

    def __init__(self, schema):
        self.schema = schema

    @property
    def classes(self):
        from mivia_small_json import registry
        return tuple(registry.active())

    def predict(self, texts):
        return [self.schema] * len(texts)

# A payload safely over the old 200/300-char caps; single line, single
# spaces (the commit extractor's longest-chunk split breaks on double
# spaces/pipes/newlines by design — that is chunk SELECTION, not slicing;
# this test pins that nothing is CUT within the selected chunk).
_LOG_PAYLOAD = (
    "upstream payment-gateway handshake timed out after 30 seconds while "
    "replaying batch 4711; the retry queue now holds 12480 deferred jobs "
    "and the oldest pending acknowledgment is 96 minutes old, which exceeds "
    "the 15-minute freshness floor by more than a factor of six and will "
    "force a full replay of the settlement window unless the downstream "
    "drain completes before the next reconciliation sweep begins"
)
_COMMIT_PAYLOAD = (
    "this change reworks the retry scheduler so that deferred settlement "
    "batches drain in submission order even when the downstream gateway "
    "throttles individual merchant endpoints, and it adds a per-batch "
    "audit trail entry for every deferred attempt together with a "
    "reconciliation sweep that runs at the top of each hour and reports "
    "the oldest pending acknowledgment across every region before the "
    "daily window closes"
)


# --- 1. log_event message: full carry past the old 200-char cap ------------


def test_log_message_over_200_chars_returned_in_full():
    text = f"2026-10-10T12:00:00Z ERR payment-api msg: {_LOG_PAYLOAD}"
    assert len(_LOG_PAYLOAD) > 200
    rec = rules.extract(text, "log_event")
    assert rec["message"] == _LOG_PAYLOAD


def test_log_message_over_200_chars_full_through_emit():
    text = f"2026-10-10T12:00:00Z ERR payment-api msg: {_LOG_PAYLOAD}"
    obj, _attempts, _schema, fallback = rules.emit(text)
    assert fallback is False
    assert obj["message"] == _LOG_PAYLOAD


# --- 2. commit message: full carry past the old 300-char cap ---------------


def test_commit_message_over_300_chars_returned_in_full():
    assert len(_COMMIT_PAYLOAD) > 300
    sha = "e7f3a1b2c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4"
    text = (f"commit {sha} 42 files changed by Ora Ivia <ora@mivia.dev> "
            f"{_COMMIT_PAYLOAD}")
    rec = rules.extract(text, "commit")
    assert rec["sha"] == sha
    assert rec["message"] == _COMMIT_PAYLOAD


def test_commit_message_over_300_chars_full_through_emit():
    sha = "e7f3a1b2c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4"
    text = (f"commit {sha} 42 files changed by Ora Ivia <ora@mivia.dev> "
            f"{_COMMIT_PAYLOAD}")
    obj, _attempts, _schema, fallback = rules.emit(text)
    assert fallback is False
    assert obj["message"] == _COMMIT_PAYLOAD


# --- 3. diagnostic message: every line, multi-line included ----------------

_RUSTC_BLOCK = (
    "error[E0308]: mismatched types\n"
    " --> src/main.rs:12:18\n"
    "  |\n"
    "12 |     let x: u32 = \"hello\";\n"
    "   |                 ^^^^^^^ expected `u32`, found `&str`\n"
    "For further information visit https://doc.rust-lang.org/error-codes.html"
)


def test_multiline_diagnostic_message_carries_every_line():
    d = extract_diagnostic(_RUSTC_BLOCK)
    expected = _RUSTC_BLOCK[len("error[E0308]: "):]
    assert d["message"] == expected
    assert "\n" in d["message"]
    # Every line after the severity/code marker is in the message.
    for line in _RUSTC_BLOCK.splitlines()[1:]:
        assert line in d["message"]


def test_multiline_diagnostic_full_through_atomic_take():
    """The classify_dump atomic-take path hands the whole block to the
    diagnostic schema; the record must carry the complete block, not its
    first line. (The stub router stands in for the encoder, which routes
    the canonical rustc block to diagnostic; the rules-keyword fallback
    ties log_event on this block and is not the shipped router.)"""
    records = pipeline.classify_dump(_RUSTC_BLOCK,
                                     router=_FixedRouter("diagnostic"))
    assert len(records) == 1
    obj, _meta = records[0]
    assert obj["type"] == "diagnostic"
    assert obj["message"] == _RUSTC_BLOCK[len("error[E0308]: "):]
    assert "expected `u32`, found `&str`" in obj["message"]


def test_pytest_failure_block_keeps_assertion_tail():
    text = ("AssertionError: assert 'issue' == 'commit'\n"
            " +  where 'issue' = route('commit 4f1a2b fix the flaky retry')")
    d = extract_diagnostic(text)
    assert d["message"] == text[len("AssertionError: "):]


# --- 4. agent_step error_message: semicolons and newlines are content ------


def test_agent_step_error_with_semicolon_carries_everything():
    text = ("step_index: 7; agent_role: tool; tool: query_db; "
            "error_message: assertion failed; expected 3 rows, got 1; "
            "return_code: 1; status: error")
    res = extract_agent_step(text)
    assert res["error_message"] == "assertion failed; expected 3 rows, got 1"


def test_agent_step_error_multiline_carries_every_line():
    text = ("step 7: tool call stream_shard\n"
            "error_message: connection reset while streaming shard 4\n"
            "retry budget exhausted after 5 attempts\n"
            "return_code: 1\n"
            "status: error")
    res = extract_agent_step(text)
    assert (res["error_message"] ==
            "connection reset while streaming shard 4\n"
            "retry budget exhausted after 5 attempts")


def test_agent_step_error_led_form_carries_following_lines():
    text = ("step 9: tool call run_tests\n"
            "ValueError: unexpected schema drift\n"
            "column billing_total vanished from the response\n"
            "return_code: 2")
    res = extract_agent_step(text)
    assert res["error_type"] == "ValueError"
    assert (res["error_message"] ==
            "unexpected schema drift\n"
            "column billing_total vanished from the response")


# --- 5. normalize_sha: below 7 hex chars is absent, never padded -----------


def test_short_hex_sha_is_dropped_not_padded():
    assert V.normalize_sha("abc12") == ""          # was "abc1200"
    assert V.normalize_sha("abcd") == ""
    assert V.normalize_sha("") == ""               # was "0000000"
    assert V.normalize_sha("!!") == ""


def test_valid_sha_lengths_pass_through_verbatim():
    assert V.normalize_sha("abcdef0") == "abcdef0"
    assert V.normalize_sha("A1B2C3D!") == "a1b2c3d"
    assert V.normalize_sha("a" * 45) == "a" * 40   # schema format bound: 40


def test_repair_drops_too_short_sha_instead_of_padding():
    fixed = V.repair({"type": "commit", "sha": "abc12"}, ["format:sha"],
                     "commit")
    assert "sha" not in fixed


def test_commit_with_5_hex_token_has_no_sha_at_all():
    """"commit abc12" — 5 hex chars is not a sha id; the record ships with
    the field absent, never with padded digits."""
    obj, _attempts, _schema, fallback = rules.emit("commit abc12 fix the thing")
    assert fallback is False
    assert "sha" not in obj
