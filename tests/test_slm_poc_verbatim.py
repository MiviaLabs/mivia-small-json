"""Contract pins for the slm-grammar-toolcall PoC's mechanical instruments.

Spec: slm-grammar-toolcall (registered 2026-10-10,
BEFORE any measurement). These tests pin the G2 verbatim checker — the
instrument that decides the honest-values bar — with SYNTHETIC rows only:
no model bytes are needed, nothing here loads or downloads a model, and
nothing here touches the network.

The construct under pin is the span arm's ``_wrap_strip`` (leading strip
removes ONLY quote characters — a leading dot is content, ``.env`` is a
filename, not noise; trailing strip keeps the full punctuation set), the
same construct the shipped spanfill decode uses (spanfix wave bugs 5/8).
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from slm_grammar_poc import (  # noqa: E402
    _wrap_strip,
    enum_violation,
    gbnf_for_tool,
    strict_parse,
    value_is_verbatim,
    _value_eq,
)

REQUEST = "please restart the cron-runner service on host app-07 in staging"


# --- the wrap-strip construct ----------------------------------------------

def test_wrap_strip_leading_strips_quotes_only():
    # a leading dot is CONTENT: ".env" is a filename, not wrapper noise
    assert _wrap_strip('".env"') == ".env"
    assert _wrap_strip("'app-07'") == "app-07"


def test_wrap_strip_trailing_keeps_full_punctuation_set():
    # the spanfix wave's bug-8 class: trailing punctuation is wrapper noise
    assert _wrap_strip("prod-eu-1!") == "prod-eu-1"
    assert _wrap_strip("app-07.") == "app-07"
    assert _wrap_strip('"staging"),') == "staging"


# --- the G2 verbatim checker ------------------------------------------------

def test_verbatim_plain_copy_is_verbatim():
    assert value_is_verbatim("cron-runner", {}, REQUEST)


def test_verbatim_after_quote_and_wrap_strip():
    assert value_is_verbatim('"cron-runner"', {}, REQUEST)
    assert value_is_verbatim("app-07!", {}, REQUEST)


def test_verbatim_leading_dot_is_content():
    # ".env" must NOT match a request that never carries it — the leading
    # dot is content and may not be stripped away into a match
    assert not value_is_verbatim(".env", {}, REQUEST)
    assert value_is_verbatim(".env", {}, "deploy the .env file to app-07")


def test_verbatim_paraphrase_is_invented():
    # "cron runner" (space) is not a copy of "cron-runner" (hyphen)
    assert not value_is_verbatim("cron runner", {}, REQUEST)


def test_verbatim_case_difference_is_invented():
    # verbatim = exact copy; the checker is case-sensitive for strings
    assert not value_is_verbatim("Cron-Runner", {}, REQUEST)


def test_verbatim_absent_value_is_invented():
    assert not value_is_verbatim("web-99", {}, REQUEST)


def test_verbatim_integer_and_float_json_forms():
    assert value_is_verbatim(4, {"type": "integer"},
                             "with 4 replicas")
    assert not value_is_verbatim(9, {"type": "integer"},
                                 "with 4 replicas")
    assert value_is_verbatim(0.5, {"type": "float"}, "at 0.5 ratio")
    assert not value_is_verbatim(0.7, {"type": "float"}, "at 0.5 ratio")


def test_verbatim_boolean_json_form_case_insensitive():
    assert value_is_verbatim(True, {"type": "boolean"}, "run dry true now")
    assert not value_is_verbatim(True, {"type": "boolean"}, REQUEST)


def test_verbatim_array_requires_every_element():
    prop = {"type": "array"}
    assert value_is_verbatim(["cron-runner", "app-07"], prop, REQUEST)
    assert not value_is_verbatim(["cron-runner", "web-99"], prop, REQUEST)


def test_verbatim_none_and_empty_are_invented():
    # a missing copy is not a copy: an emitted empty/None value carries no
    # verbatim bytes, so it counts as invented (conservative direction)
    assert value_is_verbatim(None, {}, REQUEST) is False
    assert value_is_verbatim("", {}, REQUEST) is False


# --- the G5 strict parse ------------------------------------------------------

def test_strict_parse_accepts_object_and_required_keys():
    obj, err = strict_parse('{"service": "cron-runner"}')
    assert err is None and obj == {"service": "cron-runner"}


def test_strict_parse_rejects_nan_and_infinity():
    # the hole the repo's validator closed: non-finite constants are
    # invalid even though Python's json accepts them by default
    obj, err = strict_parse('{"ratio": NaN}')
    assert obj is None and "NaN" in err
    obj, err = strict_parse('{"ratio": Infinity}')
    assert obj is None and "Infinity" in err
    obj, err = strict_parse('{"ratio": -Infinity}')
    assert obj is None and "Infinity" in err


def test_strict_parse_rejects_non_object_top_level():
    obj, err = strict_parse('["cron-runner"]')
    assert obj is None and "not a JSON object" in err
    obj, err = strict_parse('"cron-runner"')
    assert obj is None and "not a JSON object" in err


# --- the G6 enum checker -------------------------------------------------------

def test_enum_violation_declared_member_is_clean():
    prop = {"enum": ["staging", "production", "canary"]}
    assert not enum_violation("staging", prop)


def test_enum_violation_undeclared_member_flagged():
    prop = {"enum": ["staging", "production", "canary"]}
    assert enum_violation("prod", prop)


def test_enum_violation_no_enum_property_is_never_a_violation():
    assert not enum_violation("anything", {"type": "string"})


def test_enum_violation_arrays_recurse():
    prop = {"enum": ["web", "db"]}
    assert not enum_violation(["web", "db"], prop)
    assert enum_violation(["web", "cache"], prop)


# --- the dev exact-equality construct -------------------------------------------

def test_value_eq_string_wrap_strip_parity():
    assert _value_eq("app-07!", "app-07", {"type": "string"})
    assert not _value_eq("web-99", "app-07", {"type": "string"})


def test_value_eq_integer_parsed_from_string_not_bool_confused():
    assert _value_eq("4", 4, {"type": "integer"})
    assert _value_eq(4.0, 4, {"type": "integer"})
    assert not _value_eq(True, 1, {"type": "integer"})


def test_value_eq_boolean_exact():
    assert _value_eq(True, True, {"type": "boolean"})
    assert not _value_eq("true", True, {"type": "boolean"})


# --- grammar wellformedness (the structural half of G5/G6) ----------------------

def _tool(required, enum=None):
    props = {
        "service": {"type": "string"},
        "environment": {"type": "string"},
        "replicas": {"type": "integer"},
    }
    if enum:
        props["environment"]["enum"] = enum
    return {"name": "restart_service",
            "parameters": {"type": "object", "properties": props,
                           "required": required}}


def test_grammar_root_carries_exactly_the_required_key_literals():
    g = gbnf_for_tool(_tool(["service", "replicas"]))
    root = next(line for line in g.splitlines() if line.startswith("root"))
    assert '\\"service\\"' in root and '\\"replicas\\"' in root
    assert '\\"environment\\"' not in root  # optional slots are not emitted


def test_grammar_enum_rule_lists_only_declared_members():
    g = gbnf_for_tool(_tool(["environment"],
                            enum=["staging", "production"]))
    val = next(line for line in g.splitlines()
               if line.startswith("val-environment ::="))
    assert '"\\"staging\\""' in val and '"\\"production\\""' in val
    assert "canary" not in val
