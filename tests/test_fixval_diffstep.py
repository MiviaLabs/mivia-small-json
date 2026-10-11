"""Fix-validation tests for the 2026-10-10 code-review findings.

Covers N6/B12/B8 (``diff_comment``) and B9/B10 (``agent_step``) from
``reports/code-review-validation-2026-10-10.md``. Every defect test here
failed against the reviewed tree before the fix.
"""

import pytest

from mivia_small_json.agent_step import (
    extract_agent_step,
    register_agent_step_schema,
)
from mivia_small_json.diff_comment import (
    extract_diff,
    extract_review_comment,
    register_diff_and_review_schemas,
)
from mivia_small_json.validate import validate


@pytest.fixture(autouse=True)
def _schemas():
    register_diff_and_review_schemas()
    register_agent_step_schema()
    yield


# --- N6: /dev/null is not a real path ---------------------------------------

ADD_FILE_DIFF = (
    "diff --git a/new.py b/new.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/new.py\n"
    "@@ -0,0 +1,1 @@\n"
    "+print('hi')\n"
)
DELETE_FILE_DIFF = (
    "diff --git a/old.py b/old.py\n"
    "deleted file mode 100644\n"
    "--- a/old.py\n"
    "+++ /dev/null\n"
    "@@ -1,1 +0,0 @@\n"
    "-print('bye')\n"
)


def test_add_file_diff_reports_a_real_old_path():
    obj = extract_diff(ADD_FILE_DIFF)
    assert obj["change_type"] == "add"
    assert obj["old_path"] == "new.py"
    assert obj["new_path"] == "new.py"
    assert validate(obj, "diff") == []


def test_delete_file_diff_reports_a_real_new_path():
    obj = extract_diff(DELETE_FILE_DIFF)
    assert obj["change_type"] == "delete"
    assert obj["old_path"] == "old.py"
    assert obj["new_path"] == "old.py"
    assert validate(obj, "diff") == []


def test_bare_add_diff_without_git_header_reports_real_paths():
    obj = extract_diff("--- /dev/null\n+++ b/late.py\n@@ -0,0 +1,1 @@\n+x\n")
    assert obj["change_type"] == "add"
    assert obj["new_path"] == "late.py"
    assert obj["old_path"] == "late.py"


def test_hunk_less_dev_null_diff_reports_real_paths_and_counts():
    obj = extract_diff("--- /dev/null\n+++ b/made.py\n+x\n-y\n")
    assert obj["old_path"] == "made.py"
    assert obj["new_path"] == "made.py"
    assert obj["additions"] == 1
    assert obj["deletions"] == 1


# --- B12: a diff without @@ hunk headers still counts +/- lines --------------


def test_hunk_less_diff_counts_additions_and_deletions():
    obj = extract_diff("--- a/app.py\n+++ b/app.py\n+import os\n-print sys\n")
    assert obj["additions"] == 1
    assert obj["deletions"] == 1


def test_git_header_hunk_less_diff_counts_changes():
    obj = extract_diff(
        "diff --git a/app.py b/app.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "+import os\n"
        "-print sys\n"
        "+import sys\n"
    )
    assert obj["additions"] == 2
    assert obj["deletions"] == 1


def test_hunk_less_diff_plus_plus_plus_payload_is_an_addition():
    obj = extract_diff("--- a/f.py\n+++ b/f.py\n++++ payload\n++ok\n")
    assert obj["additions"] == 2
    assert obj["deletions"] == 0


def test_hunk_less_diff_bare_rule_line_is_not_a_deletion():
    obj = extract_diff("--- a/f.py\n+++ b/f.py\n---\n+ok\n")
    assert obj["additions"] == 1
    assert obj["deletions"] == 0


def test_hunk_tracked_diffs_count_unchanged():
    obj = extract_diff(
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,1 +1,2 @@\n"
        "-old\n"
        "+new\n"
        "+added\n"
    )
    assert obj["additions"] == 2
    assert obj["deletions"] == 1


# --- B8: hyphen-prefixed severity words are not verdicts ---------------------


def test_non_blocking_is_not_blocking():
    assert extract_review_comment("non-blocking: looks fine")["severity"] \
        == "suggestion"
    assert extract_review_comment("non-nit: fine by me")["severity"] \
        == "suggestion"
    assert extract_review_comment("non-minor: fine by me")["severity"] \
        == "suggestion"


def test_unprefixed_severity_markers_still_match():
    assert extract_review_comment("blocking: SQL injection")["severity"] \
        == "blocking"
    assert extract_review_comment("this is a blocking issue")["severity"] \
        == "blocking"
    assert extract_review_comment("nit: rename this")["severity"] == "nit"
    assert extract_review_comment("minor: rename this")["severity"] == "nit"
    assert extract_review_comment("warning: deprecated call")["severity"] \
        == "warning"


# --- B9: return-code priority and argument-scoped status markers -------------


def test_specific_exit_code_phrase_beats_echoed_exit_code():
    obj = extract_agent_step(
        'arguments: {"command": "echo exit code 0"}; '
        "stdout: command finished with exit code 1")
    assert obj["return_code"] == 1
    assert obj["status"] == "error"


def test_return_code_field_beats_bare_exit_code_in_output():
    obj = extract_agent_step("tool: x; stdout: exit code 7; return_code: 0")
    assert obj["return_code"] == 0
    assert obj["status"] == "success"


def test_interrupted_marker_inside_arguments_is_not_a_status():
    obj = extract_agent_step(
        'tool: execute_bash; '
        'arguments: {"command": "git log --grep=cancelled"}; return_code: 0')
    assert obj["arguments"] == '{"command":"git log --grep=cancelled"}'
    assert obj["return_code"] == 0
    assert obj["status"] == "success"


def test_timeout_marker_inside_arguments_is_not_a_status():
    obj = extract_agent_step(
        'tool: execute_bash; arguments: {"command": "echo timed out"}; '
        "return_code: 0")
    assert obj["return_code"] == 0
    assert obj["status"] == "success"


def test_interrupted_marker_outside_arguments_still_wins():
    obj = extract_agent_step(
        "step 6: tool call execute_bash; return_code: 130; "
        "stderr: interrupted by user")
    assert obj["status"] == "interrupted"


def test_timeout_marker_outside_arguments_still_wins():
    obj = extract_agent_step(
        "step 5: action: execute_bash; arguments: {\"command\": \"pytest -q\"}; "
        "stdout: ; stderr: execution timed out after 30000ms; "
        "return_code: 124; duration_ms: 30000")
    assert obj["status"] == "timeout"


# --- B10: stdout/stderr values stop at the field-terminating newline ---------


def test_multiline_record_fields_stop_at_newline():
    obj = extract_agent_step("stdout: hello\nstderr: boom\nreturn_code: 1")
    assert obj["stdout"] == "hello"
    assert obj["stderr"] == "boom"
    assert obj["return_code"] == 1
    assert obj["status"] == "error"


def test_semicolon_layout_capture_unchanged():
    obj = extract_agent_step("stdout: 157 passed; stderr: ; return_code: 0")
    assert obj["stdout"] == "157 passed"
    assert obj["stderr"] == ""
    assert obj["return_code"] == 0


def test_empty_stdout_field_still_parses_both_fields():
    obj = extract_agent_step("stdout: ; stderr: boom")
    assert obj["stdout"] == ""
    assert obj["stderr"] == "boom"


def test_boundary_keyword_mention_inside_stdout_survives():
    obj = extract_agent_step(
        'tool: execute_bash; arguments: {"command": "grep return_code x"}; '
        "stdout: checked return_code and stderr handling; return_code: 0")
    assert obj["stdout"] == "checked return_code and stderr handling"
    assert obj["return_code"] == 0
