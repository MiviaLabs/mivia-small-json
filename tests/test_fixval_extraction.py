"""Failing-tests-first fixes for the 2026-10-10 review, sections B1-B7,
N7, N10, N11 (reports/code-review-validation-2026-10-10.md).

Each test reproduces a confirmed defect before the fix:
- B1  tool_call name: ``tool:``/``call:`` tried after the bare ``"name":``
- B2  arguments object limb dies at a newline / bare ``arguments:`` key
- B3  repair_arguments: balanced-object slice never retried after the
      rewrite ladder; raw NUL byte collapses the payload
- B4  diagnostic line/col ladders anchored at column 0
- B5  diagnostic message requires error:/warning:/note:
- B6/N10  span_fill comma-grouped numbers (wrong-slot assignment)
- B7  coerce emits inf -> arguments_json ValueError
- B11 diagnostic code pattern matches lowercase tokens
- N7  result capture keeps the sibling closer for non-string scalars
- N11 untyped numeric-family slots filled as strings
"""

from __future__ import annotations

import json
import re

import pytest

from mivia_small_json.diagnostic import extract_diagnostic
from mivia_small_json.span_fill import arguments_json, coerce, fill_arguments
from mivia_small_json.tool_call import (
    TOOL_CALL_EXTRACTION,
    _clean_result,
    extract_tool_call,
    repair_arguments,
)


def _num_tool(slots: dict) -> dict:
    """Tool whose required params are the given name -> declared-type map
    (a missing type stays undeclared)."""
    props = {
        name: ({"type": stype} if stype else {})
        for name, stype in slots.items()
    }
    return {
        "type": "function",
        "function": {
            "name": "t",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(slots),
            },
        },
    }


# --- B1: name priority ------------------------------------------------------

class TestB1NamePriority:
    def test_tool_prefix_wins_over_embedded_name_value(self):
        rec = extract_tool_call(
            'tool: get_weather arguments: {"name": "Paris"}; result: sunny')
        assert rec["name"] == "get_weather"
        assert rec["arguments"] == '{"name":"Paris"}'
        assert rec["result"] == "sunny"

    def test_call_prefix_wins_over_embedded_name_value(self):
        rec = extract_tool_call(
            'call: search arguments: {"name": "index"}; result: ok')
        assert rec["name"] == "search"

    def test_quoted_tool_key_inside_json_does_not_match_bare_ladder(self):
        # `"tool":` has a quote between the word and the colon; the bare
        # \btool:\s* alternative must not fire through it.
        rec = extract_tool_call(
            '{"tool": "web_search", "arguments": {"query": "x"}}')
        assert rec["name"] == ""
        assert json.loads(rec["arguments"]) == {"query": "x"}

    @pytest.mark.parametrize("tag", ["<functioncall>", "<tool_call>"])
    def test_tagged_forms_still_first(self, tag):
        rec = extract_tool_call(
            tag + ' {"name": "get_weather", "arguments": {"name": "Paris"}}')
        assert rec["name"] == "get_weather"


# --- B2: arguments across newlines / bare key ------------------------------

class TestB2ArgumentsCapture:
    def test_bare_key_multiline_object_captured_whole(self):
        rec = extract_tool_call(
            'call: search arguments: {"name": "index", "query": "drain"}\n'
            'result: 3 hits')
        assert rec["name"] == "search"
        # name must not be a VALUE from inside arguments
        assert rec["name"] != "index"
        assert json.loads(rec["arguments"]) == {
            "name": "index", "query": "drain"}
        assert rec["result"] == "3 hits"

    def test_bare_key_pretty_printed_object_captured_whole(self):
        rec = extract_tool_call(
            'call: search arguments: {\n  "query": "drain",\n  "limit": 5\n}'
            '\nresult: 5 hits')
        assert json.loads(rec["arguments"]) == {"query": "drain", "limit": 5}

    def test_quoted_key_object_limb_tolerates_newlines(self):
        # The registered ladder's object limb is the fallback when the
        # balanced scan cannot run (e.g. rules-baseline use); it must not
        # die at the first newline inside the braces.
        pat = TOOL_CALL_EXTRACTION["arguments"][0]
        m = re.search(pat, '"arguments": {"a": 1,\n"b": 2}')
        assert m is not None
        assert json.loads(m.group(1)) == {"a": 1, "b": 2}

    def test_inline_object_still_captured(self):
        rec = extract_tool_call(
            'tool: ls arguments: {"path": "src/"}; result: ok')
        assert json.loads(rec["arguments"]) == {"path": "src/"}


# --- B3: repair ladder ------------------------------------------------------

class TestB3RepairLadder:
    def test_python_bool_with_trailing_prose(self):
        assert json.loads(repair_arguments(
            '{"command": "ls", "ok": True}; return_code: 0')
        ) == {"command": "ls", "ok": True}

    def test_python_bool_trailing_comma_and_prose(self):
        assert json.loads(repair_arguments(
            '{"command": "ls", "ok": True,}; return_code: 0')
        ) == {"command": "ls", "ok": True}

    def test_raw_nul_byte_inside_string_value(self):
        assert json.loads(
            repair_arguments('{"command": "echo a\x00b"}')
        ) == {"command": "echo ab"}

    # must-not-break (pinned correct behavior)

    def test_mnb_pure_python_literal(self):
        assert repair_arguments(
            "{'command': 'echo', 'ok': True}"
        ) == '{"command":"echo","ok":true}'

    def test_mnb_hi_colon_string_with_trailing_prose(self):
        assert json.loads(repair_arguments(
            """{"command": "echo 'hi': ok"}; trailing prose""")
        ) == {"command": "echo 'hi': ok"}

    def test_mnb_shell_semicolons_inside_strings(self):
        assert repair_arguments(
            '{"command": "ls; rm -rf /; echo done"}'
        ) == '{"command":"ls; rm -rf /; echo done"}'

    def test_mnb_none_inside_string_value(self):
        assert json.loads(repair_arguments(
            '{"cmd": "None of them"}')) == {"cmd": "None of them"}

    def test_mnb_escaped_quotes_inside_strings(self):
        assert json.loads(repair_arguments(
            '{"command": "echo \\"hi\\" && ls"}')
        ) == {"command": 'echo "hi" && ls'}


# --- N7: result scalar capture ----------------------------------------------

class TestN7ResultScalar:
    @pytest.mark.parametrize("raw,expect", [
        ("42", "42"),
        ("3.14", "3.14"),
        ("true", "true"),
        ("false", "false"),
        ("null", "null"),
    ])
    def test_nonstring_scalars_drop_sibling_closer(self, raw, expect):
        rec = extract_tool_call(
            '{"name": "get_count", "arguments": {"q": "x"}, '
            '"result": %s}' % raw)
        assert rec["result"] == expect

    def test_nested_wrapper_closers_peeled(self):
        rec = extract_tool_call(
            '{"name": "f", "arguments": {}, "result": 42}')
        assert rec["result"] == "42"

    def test_trailing_comma_peeled(self):
        assert _clean_result("42,") == "42"

    def test_quoted_string_result_unchanged(self):
        rec = extract_tool_call(
            '{"name": "f", "arguments": {}, "result": "done"}')
        assert rec["result"] == "done"

    def test_object_result_not_mangled(self):
        # a dict result is not a scalar: the captured value keeps its own
        # braces (behavior unchanged for shapes the fix does not target)
        assert _clean_result('{"temperature": 68}') == '{"temperature": 68}'
        assert _clean_result("done}") == "done}"


# --- B4/B5/B11: diagnostic ladders ------------------------------------------

class TestB4LeadingIndent:
    def test_indented_gcc_ladder_line_col(self):
        d = extract_diagnostic("  src/main.c:12:5: error: 'foo' undeclared")
        assert d["file"] == "src/main.c"
        assert int(d["line"]) == 12
        assert int(d["column"]) == 5

    def test_unindented_ladder_unchanged(self):
        d = extract_diagnostic("src/main.c:12:5: error: 'foo' undeclared")
        assert int(d["line"]) == 12
        assert int(d["column"]) == 5


class TestB5BareCodeMessage:
    def test_ruff_code_line_message(self):
        d = extract_diagnostic("server.py:3:1: F401 `os` imported but unused")
        assert d["code"] == "F401"
        assert d["message"] == "`os` imported but unused"

    def test_severity_word_message_unchanged(self):
        d = extract_diagnostic("server.py:3:1: error: undefined name 'os'")
        assert d["message"] == "undefined name 'os'"


class TestB11CodeCase:
    def test_lowercase_token_is_not_a_code(self):
        d = extract_diagnostic("src/a.c:1:1: error: process pid1234 crashed")
        assert d.get("code", "") == ""

    @pytest.mark.parametrize("code", ["F401", "E0308", "TS1234"])
    def test_uppercase_codes_still_match(self, code):
        d = extract_diagnostic(
            "src/a.py:1:1: error: %s something broke" % code)
        assert d["code"] == code


# --- B6/N10: comma-grouped numbers ------------------------------------------

class TestB6CommaNumbers:
    def test_key_delimited_comma_amount(self):
        filled, missing = fill_arguments(
            _num_tool({"amount": "number", "fee": "number"}),
            "amount=12,345,678.90 fee=2.50")
        assert filled == {"amount": 12345678.9, "fee": 2.5}
        assert missing == []

    def test_window_comma_money_no_wrong_slot(self):
        filled, missing = fill_arguments(
            _num_tool({"amount": "number", "fee": "number"}),
            "transfer amount 1,000 fee 2")
        assert filled == {"amount": 1000, "fee": 2}
        assert missing == []

    def test_window_comma_float(self):
        filled, missing = fill_arguments(
            _num_tool({"amount": "number", "fee": "number"}),
            "transfer amount 1,234.56 fee 2.50")
        assert filled == {"amount": 1234.56, "fee": 2.5}
        assert missing == []

    def test_alignment_comma(self):
        filled, missing = fill_arguments(
            _num_tool({"amount": "number"}),
            "the amount is 1,000 today")
        assert filled == {"amount": 1000}
        assert missing == []

    def test_integer_comma(self):
        assert coerce("1,000", "integer") == 1000
        filled, missing = fill_arguments(
            _num_tool({"count": "integer"}), "count 1,000 please")
        assert filled == {"count": 1000}
        assert missing == []

    def test_coerce_comma_forms(self):
        assert coerce("1,000", "number") == 1000
        assert coerce("12,345,678.90", "number") == 12345678.9
        assert coerce("-1,234", "integer") == -1234
        assert coerce("1,000.50", "number") == 1000.5

    def test_plain_numbers_unchanged(self):
        assert coerce("42", "integer") == 42
        assert coerce("2.50", "number") == 2.5
        assert coerce("1e2", "number") == 100
        # malformed groups stay uncoercible
        assert coerce("1,23", "number") is None
        assert coerce("12,34,56", "integer") is None


# --- N11: untyped numeric-family slots --------------------------------------

class TestN11UntypedNumericSlots:
    def test_untyped_required_slots_coerce_to_number(self):
        tool = {
            "type": "function",
            "function": {
                "name": "calc",
                "parameters": {"type": "object",
                               "required": ["num1", "num2"]},
            },
        }
        filled, missing = fill_arguments(tool, "calc num1 3 num2 4")
        assert filled == {"num1": 3, "num2": 4}
        assert missing == []

    def test_untyped_slot_rejects_non_numeric_text(self):
        tool = {
            "type": "function",
            "function": {
                "name": "calc",
                "parameters": {"type": "object", "required": ["num1"]},
            },
        }
        filled, missing = fill_arguments(tool, "calc num1 abc")
        assert filled == {}
        assert missing == ["num1"]


# --- B7: non-finite floats ----------------------------------------------------

class TestB7NonFinite:
    def test_overflow_float_is_rejected_not_emitted(self):
        filled, missing = fill_arguments(
            _num_tool({"a": "number"}), "a=1e309")
        assert filled == {}
        assert missing == ["a"]
        # no path may raise in serialization
        assert json.loads(arguments_json(filled)) == {}

    @pytest.mark.parametrize("raw", ["NaN", "inf", "-inf", "1e309"])
    def test_nonfinite_tokens_leave_slot_missing(self, raw):
        filled, missing = fill_arguments(
            _num_tool({"a": "number"}), "a=%s" % raw)
        assert filled == {}
        assert missing == ["a"]

    def test_finite_scientific_still_coerces(self):
        assert coerce("1e2", "number") == 100
        assert coerce("-2.5e-1", "number") == -0.25
