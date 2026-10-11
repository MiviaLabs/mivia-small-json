"""Fix-validation tests for the survival-fix wave: agent_step + diagnostic.

Covers (every defect test here failed against the pre-fix tree):

- M2 (owner-mandated): multi-line captured stdout/stderr must survive —
  the B10 newline exclusion truncated genuinely multi-line values at the
  first line. A value now ends at a ``;`` field separator OR a newline
  followed by a field keyword with its ``:``/``=``, not at any newline.
- Status corruption: ``return_code`` is the primary status signal;
  markers inside captured ``arguments`` are never a signal, and markers
  inside captured output upgrade only when the rc is consistent with
  them (rc 0 wins over prose in captured output).
- ANSI leakage: ``extract_diagnostic`` now sanitizes like
  ``extract_agent_step`` does.
- Private-mode CSI (``ESC [ ? 2004 h``) survives ``sanitize_log`` and
  glues into extracted file paths.
"""

import pytest

from mivia_small_json import pipeline
from mivia_small_json.agent_step import (
    extract_agent_step,
    register_agent_step_schema,
)
from mivia_small_json.diagnostic import (
    extract_diagnostic,
    register_diagnostic_schema,
    sanitize_log,
)
from mivia_small_json.validate import validate


@pytest.fixture(autouse=True)
def _schemas():
    register_agent_step_schema()
    register_diagnostic_schema()
    yield


# --- M2: multi-line captured values survive ---------------------------------


def test_b10_repro_keyword_led_newline_still_terminates():
    """The original B10 repro stays green: `stdout` ends at the newline
    that leads into `stderr:`, `stderr` at the one before `return_code:`."""
    obj = extract_agent_step("stdout: hello\nstderr: boom\nreturn_code: 1")
    assert obj["stdout"] == "hello"
    assert obj["stderr"] == "boom"
    assert obj["return_code"] == 1
    assert obj["status"] == "error"


def test_multiline_captured_stdout_survives():
    """A genuinely multi-line value must not be truncated at line one."""
    obj = extract_agent_step(
        "stdout: line one\nline two\nline three; return_code: 0")
    assert obj["stdout"] == "line one\nline two\nline three"
    assert obj["return_code"] == 0
    assert obj["status"] == "success"
    assert validate(obj, "agent_step") == []


def test_multiline_captured_stderr_continuation_survives():
    obj = extract_agent_step(
        "stdout: ok\nstderr: trace one\ntrace two\nreturn_code: 1")
    assert obj["stdout"] == "ok"
    assert obj["stderr"] == "trace one\ntrace two"
    assert obj["return_code"] == 1
    assert obj["status"] == "error"


def test_newline_without_keyword_colon_does_not_terminate():
    """The terminator is a newline followed by a field keyword WITH its
    ':'/'=' — a bare keyword mention on the next line stays in the value."""
    obj = extract_agent_step("stdout: a\nstderr\nreturn_code: 0")
    assert obj["stdout"] == "a\nstderr"
    assert obj["return_code"] == 0


def test_empty_stdout_field_still_parses_both_fields():
    obj = extract_agent_step("stdout: ; stderr: boom")
    assert obj["stdout"] == ""
    assert obj["stderr"] == "boom"


def test_canonical_semicolon_layout_unchanged():
    obj = extract_agent_step(
        'tool: execute_bash; arguments: {"command": "pytest -q"}; '
        "stdout: 157 passed; stderr: ; return_code: 0; duration_ms: 120")
    assert obj["stdout"] == "157 passed"
    assert obj["stderr"] == ""
    assert obj["return_code"] == 0
    assert obj["duration_ms"] == 120
    assert obj["status"] == "success"
    assert validate(obj, "agent_step") == []


def test_shell_semicolon_inside_value_still_tolerated():
    obj = extract_agent_step("stdout: echo a; echo b; return_code: 0")
    assert obj["stdout"] == "echo a; echo b"
    assert obj["return_code"] == 0


# --- Status: rc is primary; captured text is not an outcome -----------------


def test_rc0_beats_prose_marker_in_captured_stdout():
    """`cancelled` in captured stdout is output content, not the outcome:
    return_code 0 wins."""
    obj = extract_agent_step(
        'tool: shell arguments: {"command": "echo done"}; '
        "stdout: job cancelled by user; return_code: 0")
    assert obj["stdout"] == "job cancelled by user"
    assert obj["return_code"] == 0
    assert obj["status"] == "success"


def test_timeouterror_literal_inside_arguments_is_not_outcome():
    """`TimeoutError` inside the command string is command content: the
    error_type scan must skip the arguments span and rc 1 means error."""
    obj = extract_agent_step(
        "tool: shell arguments: {\"command\": \"python -c 'raise TimeoutError'\"}; "
        "return_code: 1")
    assert obj["error_type"] == ""
    assert obj["return_code"] == 1
    assert obj["status"] == "error"


def test_real_timeout_control_still_times_out():
    obj = extract_agent_step(
        'tool: shell arguments: {"command": "sleep 100"}; '
        "stderr: execution timed out after 30000ms; return_code: 124")
    assert obj["status"] == "timeout"


def test_interrupted_marker_in_captured_stderr_with_nonzero_rc():
    obj = extract_agent_step(
        "step 6: tool call execute_bash; return_code: 130; "
        "stderr: interrupted by user")
    assert obj["status"] == "interrupted"


def test_timeout_marker_in_captured_output_with_rc0_is_success():
    obj = extract_agent_step(
        'tool: shell; arguments: {"command": "run-suite"}; '
        "stdout: one case timed out; return_code: 0")
    assert obj["status"] == "success"


def test_prose_interrupted_marker_outside_captures_still_wins():
    """A marker in the narrator's own prose (not captured output) keeps
    the old behavior: it is the narrator's verdict of the run."""
    obj = extract_agent_step("step 1; canceled by user; return_code: 0")
    assert obj["status"] == "interrupted"


def test_timeouterror_outside_arguments_with_consistent_rc_times_out():
    obj = extract_agent_step(
        'arguments: {"command": "run-suite"}; return_code: 124; '
        "TimeoutError: connection lost")
    assert obj["arguments"] == '{"command":"run-suite"}'
    assert obj["error_type"] == "TimeoutError"
    assert obj["status"] == "timeout"


def test_explicit_status_field_still_overrides_everything():
    obj = extract_agent_step(
        'tool: shell; arguments: {"command": "x"}; '
        "stdout: job cancelled by user; return_code: 0; status: interrupted")
    assert obj["status"] == "interrupted"


def test_args_pseudo_json_timeouterror_error_type_and_status():
    """The v6 case 2 shape through the pipeline stays a valid error."""
    obj, meta = pipeline.classify(
        "tool: shell arguments: {\"command\": \"python -c 'raise TimeoutError'\"}; "
        "return_code: 1", router=None)
    assert obj["type"] == "agent_step"
    assert obj["error_type"] == ""
    assert obj["status"] == "error"
    assert meta["valid"] is True


# --- Diagnostic: ANSI sanitized before extraction ---------------------------


def test_diagnostic_extraction_sanitizes_ansi():
    d = extract_diagnostic("\x1b[31msrc/app.py:10:2: error: boom\x1b[0m")
    assert d["file"] == "src/app.py"
    assert int(d["line"]) == 10
    assert int(d["column"]) == 2
    assert d["message"] == "boom"
    assert "\x1b" not in str(d)


def test_diagnostic_extraction_ansi_colored_warning_ladder():
    d = extract_diagnostic(
        "\x1b[1;34mtests/test_x.py:42:5: warning: unused import\x1b[0m")
    assert d["file"] == "tests/test_x.py"
    assert int(d["line"]) == 42
    assert int(d["column"]) == 5
    assert d["severity"] == "warning"
    assert d["message"] == "unused import"


def test_classify_diagnostic_with_ansi_is_clean():
    obj, meta = pipeline.classify(
        "\x1b[31msrc/app.py:10:2: error: boom\x1b[0m", router=None)
    assert obj["type"] == "diagnostic"
    assert obj["file"] == "src/app.py"
    assert "\x1b" not in str(obj)


def test_timestamp_guard_still_holds_after_sanitize():
    """A space-separated log timestamp must not read as a path (the
    line-start anchor; sanitize_log must not strip this timestamp form)."""
    d = extract_diagnostic(
        "2026-10-10 11:22:33,444 src/app.py:10:2: worker started")
    assert d.get("file", "") == ""


def test_clean_text_extractions_unchanged():
    d = extract_diagnostic("server.py:3:1: F401 `os` imported but unused")
    assert d["file"] == "server.py"
    assert d["code"] == "F401"
    assert d["message"] == "`os` imported but unused"
    d2 = extract_diagnostic(
        "error[E0308]: mismatched types\n  --> src/main.rs:12:18")
    assert d2["file"] == "src/main.rs"
    assert d2["code"] == "E0308"


# --- sanitize_log: private-mode CSI -----------------------------------------


@pytest.mark.parametrize("seq", ["\x1b[?2004h", "\x1b[?1004h", "\x1b[?25l"])
def test_private_mode_csi_stripped(seq):
    assert sanitize_log("x" + seq + "y") == "xy"


@pytest.mark.parametrize("seq", ["\x1b[31m", "\x1b[0m", "\x1b[2K", "\x1b[1;34m"])
def test_normal_csi_still_stripped(seq):
    assert sanitize_log("x" + seq + "y") == "xy"


def test_csi_private_and_intermediate_bytes_stripped():
    assert sanitize_log("x\x1b[0 qy") == "xy"
    assert sanitize_log("x\x1b[?1!py") == "xy"


def test_csi_glue_does_not_pollute_extracted_file_path():
    d = extract_diagnostic("\x1b[?2004hsrc/app.py:10:2: error: boom")
    assert d["file"] == "src/app.py"
    assert int(d["line"]) == 10
    assert d["message"] == "boom"
