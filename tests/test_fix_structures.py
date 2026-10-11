"""Regression tests for the 2026-10-09 structural-fix wave.

Covers: tool_call argument repair (NaN/Infinity, brace trim, balanced-brace
extraction, string-safe literal rewrite), agent_step field extraction
(empty stdout/stderr, option-spelled timeout, US "canceled"), agent_state
key collisions and tool_call summaries, diff_comment per-file change_type /
hunk counting / severity boundaries / quoted paths, diagnostic extraction
ladders and rustc normalization, sanitize_log ANSI ownership, integrity
empty-sidecar handling, and the encoder thread budget + warning accuracy.
"""

from pathlib import Path

import pytest

from mivia_small_json import agent_state, diagnostic, integrity, registry
from mivia_small_json.agent_step import extract_agent_step
from mivia_small_json.diff_comment import extract_diff, extract_review_comment
from mivia_small_json.tool_call import extract_tool_call, repair_arguments


@pytest.fixture(autouse=True)
def _register():
    diagnostic.register_diagnostic_schema()
    from mivia_small_json.agent_step import register_agent_step_schema
    from mivia_small_json.diff_comment import register_diff_and_review_schemas
    from mivia_small_json.tool_call import (
        register_tool_call_schema,
        register_tool_clarification_schema,
    )
    before = set(registry.active())
    register_agent_step_schema()
    register_tool_call_schema()
    register_tool_clarification_schema()
    register_diff_and_review_schemas()
    yield
    for name in set(registry.active()) - before:
        registry.unregister(name)


# --- F1: NaN/Infinity must never be emitted ---------------------------------

def test_f1_nan_degrades_not_emits():
    assert repair_arguments('{"a": NaN}') == "{}"


def test_f1_infinity_degrades():
    assert repair_arguments('{"a": Infinity}') == "{}"
    assert repair_arguments('{"a": -Infinity}') == "{}"


def test_f1_overflow_still_degrades():
    assert repair_arguments('{"a": 1e400}') == "{}"


def test_f1_valid_payload_untouched():
    assert repair_arguments('{"a": 1.5}') == '{"a":1.5}'


# --- F2: brace-trim loop strips every extra closer ---------------------------

def test_f2_two_extra_closers():
    assert repair_arguments('{"a":1}}}') == '{"a":1}'


def test_f2_agent_step_triple_closer():
    obj = extract_agent_step(
        'arguments: {"command": "pytest -q"}}}; return_code: 0')
    assert obj["arguments"] == '{"command":"pytest -q"}'


# --- F3: balanced-brace extraction (nested + pretty-printed) -----------------

def test_f3_nested_inline_object():
    obj = extract_tool_call(
        '{"name": "run", "arguments": {"command": "ls"}, "id": "1"}')
    assert obj["arguments"] == '{"command":"ls"}'


def test_f3_pretty_printed_object():
    obj = extract_tool_call(
        '{"name": "run",\n  "arguments": {\n    "command": "ls"\n  }\n}')
    assert obj["arguments"] == '{"command":"ls"}'


def test_f3_braces_inside_strings():
    obj = extract_tool_call(
        '{"name": "run", "arguments": {"cmd": "echo }{"}}')
    assert obj["arguments"] == '{"cmd":"echo }{"}'


# --- F4: literal rewrite must not corrupt string content ---------------------

def test_f4_true_inside_string_survives_truncation():
    assert repair_arguments('{"cmd": "echo True things", "n": 1') \
        == '{"cmd":"echo True things","n":1}'


def test_f4_python_literals_outside_strings_still_rewritten():
    assert repair_arguments("{'a': True, 'b': None}") \
        == '{"a":true,"b":null}'


# --- F5: agent_step ----------------------------------------------------------

def test_f5a_empty_stdout_does_not_swallow_next_field():
    obj = extract_agent_step(
        "step_index: 1; stdout: ; stderr: boom; return_code: 1")
    assert obj["stdout"] == ""
    assert obj["stderr"] == "boom"
    assert obj["return_code"] == 1


def test_f5b_option_spelled_timeout_is_not_a_status():
    obj = extract_agent_step(
        'tool: execute_bash; arguments: {"command": '
        '"curl --connect-timeout 30"}; return_code: 0')
    assert obj["status"] == "success"


def test_f5b_real_timeout_still_detected():
    obj = extract_agent_step("stderr: execution timed out after 30000ms")
    assert obj["status"] == "timeout"


def test_f5c_us_canceled_is_interrupted():
    obj = extract_agent_step("step 1; canceled by user; return_code: 0")
    assert obj["status"] == "interrupted"


# --- F6: agent_state ---------------------------------------------------------

def test_f6a_stepless_steps_do_not_collide():
    st = agent_state.merge_step(agent_state.new_state(), {
        "type": "agent_step", "tool": "t1", "arguments": "{}",
        "status": "success"})
    st = agent_state.merge_step(st, {
        "type": "agent_step", "tool": "t2", "arguments": "{}",
        "status": "success"})
    assert len(st["steps"]) == 2


def test_f6b_tool_call_name_maps_to_tool_in_summary():
    st = agent_state.merge_step(agent_state.new_state(), {
        "type": "tool_call", "name": "run_tests", "arguments": "{}",
        "result": "ok"})
    summary = list(st["steps"].values())[0]
    assert summary["tool"] == "run_tests"


def test_f6c_state_key_only_when_filed():
    state = agent_state.new_state()
    obj, meta = agent_state.classify_turn(
        "tool: run_tests needs clarification, missing parameters: number1",
        state, router=None)
    assert obj["type"] == "tool_clarification"
    assert meta.get("state_key") == ""


def test_f6c_state_key_present_when_filed():
    state = agent_state.new_state()
    obj, meta = agent_state.classify_turn(
        "step_index: 1; tool: execute_bash; arguments: {}; return_code: 0",
        state, router=None)
    assert obj["type"] == "agent_step"
    assert meta.get("state_key", "").startswith("step-")


# --- F7: diff_comment --------------------------------------------------------

def test_f7a_hunk_content_not_probed_for_change_type():
    obj = extract_diff(
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n+++ b/src/a.py\n"
        "@@ -1,1 +1,1 @@\n-old rename from here\n+new file mode text\n")
    assert obj["change_type"] == "modify"


def test_f7a_added_file_reported_not_the_modified_one():
    patch = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n+++ b/src/a.py\n"
        "@@ -1,1 +1,1 @@\n-x\n+y\n"
        "diff --git a/src/b.py b/src/b.py\n"
        "new file mode 100644\n--- /dev/null\n+++ b/src/b.py\n"
        "@@ -0,0 +1,1 @@\n+hi\n")
    obj = extract_diff(patch)
    assert obj["change_type"] == "add"
    assert obj["new_path"] == "src/b.py"
    assert obj["additions"] == 1


def test_f7b_four_plus_payload_counted_as_addition():
    obj = extract_diff(
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
        "@@ -0,0 +1,2 @@\n++++ payload\n++ok\n")
    assert obj["additions"] == 2


def test_f7c_blocking_needs_word_boundary():
    assert extract_review_comment("nonblocking: looks fine")["severity"] \
        != "blocking"
    assert extract_review_comment("unblocking the queue now")["severity"] \
        != "blocking"


def test_f7c_minority_is_not_a_nit():
    assert extract_review_comment("a minority of paths skip the check") \
        ["severity"] != "nit"
    assert extract_review_comment("minor: rename this")["severity"] == "nit"
    assert extract_review_comment("nit: rename this")["severity"] == "nit"


def test_f7d_quoted_git_paths():
    patch = (
        'diff --git a/"src/my file.py" b/"src/my file.py"\n'
        '--- "a/src/my file.py"\n+++ "b/src/my file.py"\n'
        "@@ -1,1 +1,1 @@\n-x\n+y\n")
    obj = extract_diff(patch)
    assert obj["old_path"] == "src/my file.py"
    assert obj["new_path"] == "src/my file.py"


def test_f7e_routing_marker_multiline():
    from mivia_small_json import rules
    text = "prose line\n\ndiff --git a/f.py b/f.py\n--- a/f.py\n"
    assert rules.route(text) == "diff"


# --- F8: diagnostic extraction ladder ---------------------------------------

def _extract(text):
    return registry.extractor_for("diagnostic")(text)


def test_f8a_gcc_line_col():
    obj = _extract("src/main.c:12:5: error: 'foo' undeclared")
    assert obj["file"] == "src/main.c"
    assert obj["line"] == "12"
    assert obj["column"] == "5"


def test_f8b_ruff_file_code_line_col():
    obj = _extract("server.py:3:1: F401 `os` imported but unused")
    assert obj["file"] == "server.py"
    assert obj["code"] == "F401"
    assert obj["line"] == "3"
    assert obj["column"] == "1"


def test_f8c_rustc_tool_normalized():
    assert _extract("error[E0308]: mismatched types")["tool"] == "rustc"
    obj = _extract("error[E0308]: mismatched types\n  --> src/main.rs:12:18")
    assert obj["tool"] == "rustc"
    assert obj["file"] == "src/main.rs"
    assert obj["line"] == "12"
    assert obj["column"] == "18"


# --- F9: sanitize_log ANSI ownership ----------------------------------------

def test_f9_charset_designator_eaten_following_char_kept():
    assert diagnostic.sanitize_log("foo\x1b(0bar") == "foobar"


def test_f9_osc_title_fully_stripped():
    assert diagnostic.sanitize_log("\x1b]0;title\x07ok") == "ok"


def test_f9_osc8_hyperlink_stripped():
    text = ("\x1b]8;;http://example.com\x1b\\link\x1b]8;;\x1b\\")
    assert diagnostic.sanitize_log(text) == "link"


def test_f9_sgr_still_stripped():
    assert diagnostic.sanitize_log("\x1b[31;1merr\x1b[0m") == "err"


# --- F10: integrity empty sidecar -------------------------------------------

def test_f10_empty_sidecar_is_malformed(tmp_path: Path):
    sidecar = tmp_path / "x.sha256"
    sidecar.write_text("", encoding="utf-8")
    expected, problem = integrity.expected_digest(sidecar)
    assert expected is None
    assert problem == "malformed"


def test_f10_verified_false_on_empty_sidecar(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"payload")
    sidecar = tmp_path / "artifact.bin.sha256"
    sidecar.write_text("", encoding="utf-8")
    assert integrity.verified(artifact, sidecar) is False


# --- F11: encoder session budget + warning accuracy --------------------------

def test_f11_ort_runtime_thread_budget():
    import os

    from mivia_small_json import ort_runtime
    import onnxruntime as ort

    opts = ort_runtime.apply_cpu_thread_budget(ort_runtime.session_options())
    assert opts.intra_op_num_threads == min(4, os.cpu_count() or 1)
    assert opts.inter_op_num_threads == 1
    assert opts.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL


def test_f11_tokenizer_warning_names_tokenizer(tmp_path: Path, capsys):
    import shutil

    from mivia_small_json import encoder

    root = Path(encoder.__file__).resolve().parents[2]
    good_model = root / "models" / "router-encoder-int8.onnx"
    good_tok = root / "models" / "router-tokenizer.json"
    if not (good_model.is_file() and good_tok.is_file()):
        pytest.skip("shipped encoder artifacts not present")
    # Tampered tokenizer (valid file, wrong bytes) -> warning must name it.
    tok = tmp_path / "router-tokenizer.json"
    tok.write_text("{}", encoding="utf-8")
    encoder.load_encoder(tokenizer_path=tok)
    err = capsys.readouterr().err
    assert "router-tokenizer.json" in err
    assert "failed its sha256 check" in err
    del shutil
