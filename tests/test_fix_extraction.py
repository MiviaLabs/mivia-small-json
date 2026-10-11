"""Regression tests for the 2026-10-09 code-review fixes (F1-F9).

Covers: implicit-severity overwrite, rules word-boundary defects, message
key regex, component level-lookalike, timestamp offsets, validate helper
defaults, registry spec checks and mutable defaults, teach fallback
alternation, and learned-extraction hygiene.
"""

from __future__ import annotations

import numpy as np
import pytest

from mivia_small_json import extract_learned, registry, rules, teach
from mivia_small_json.validate import normalize_ts, repair, template, validate


@pytest.fixture(autouse=True)
def _no_scratch_schemas():
    """Tests here register scratch schemas into the global registry;
    remove them afterwards so registry.active()/spec_fingerprint stay
    order-independent for the rest of the suite (review P1)."""
    yield
    for name in ("scratch", "mut", "cmdref"):
        try:
            registry.unregister(name)
        except Exception:
            pass


# --- stubs ------------------------------------------------------------------


class StubImplicit:
    """Binary implicit-severity stub that always says FATAL."""

    classes_ = ["INFO", "FATAL"]

    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


class StubComponent:
    """Binary component stub; the token it 'finds' is fixed per instance."""

    classes_ = [0, 1]

    def __init__(self, token):
        self.token = token

    def predict_proba(self, X):
        return [[0.05, 0.95]]


@pytest.fixture
def component_stub(monkeypatch):
    def _make(token):
        stub = StubComponent(token)
        monkeypatch.setattr(extract_learned, "extract_component",
                            lambda text, model, threshold=0.20: token)
        return stub

    return _make


# --- F1: implicit severity must not overwrite an explicit rules level ------


@pytest.mark.parametrize("text,expected", [
    ("ERROR: disk full on /var", "ERROR"),
    ("WARNING: disk nearly full", "WARN"),
    ("severity: ERROR disk full", "ERROR"),
])
def test_f1_implicit_never_overwrites_explicit_level(text, expected):
    obj = extract_learned.extract_log_event(
        text, implicit_severity_extractor=StubImplicit())
    assert obj["level"] == expected


def test_f1_implicit_still_applies_without_rules_level():
    obj = extract_learned.extract_log_event(
        "authentication failure; logname= uid=0 euid=0",
        implicit_severity_extractor=StubImplicit())
    assert obj["level"] == "FATAL"


# --- F2: rules.py word boundaries ------------------------------------------


def test_f2a_author_requires_separator():
    obj = rules._extract_commit(
        "Fix author lookup for commits, 2 files changed, author: Jane Doe")
    assert obj["author"] == "Jane Doe"
    assert obj["message"].startswith("Fix author lookup")


def test_f2a_authored_by_is_not_author_key():
    obj = rules._extract_commit("authored by Jane Doe: nothing to see here")
    assert not obj["author"].startswith("ed by")
    assert obj["author"] == "Jane Doe"


def test_f2b_files_cleaner_word_boundary():
    obj = rules._extract_commit("Update the profile parser, 1 file changed")
    assert obj["files_changed"] == 1
    assert "profile parser" in obj["message"]


def test_f2c_sha_needs_a_digit():
    obj = rules._extract_commit("defaced the login banner, 2 files changed")
    assert obj["sha"] != "defaced"
    assert "defaced" in obj["message"]


def test_f2c_real_sha_still_extracts():
    sha = "abc1234"  # letters + digit
    obj = rules._extract_commit(f"commit {sha} fix the thing")
    assert obj["sha"] == sha


def test_f2d_project_id_beats_hash_id():
    obj = rules._extract_issue("PAY-2214: checkout failed (#99)")
    assert obj["id"] == "PAY-2214"


# --- F3: message key regex --------------------------------------------------


def test_f3_message_key():
    assert rules._extract_log("lvl=CRITICAL component=db message=cannot connect"
                              )["message"] == "cannot connect"
    assert rules._extract_log("component=db msg=cannot connect"
                              )["message"] == "cannot connect"


# --- F4: component key must not be a level word ----------------------------


def test_f4_severity_key_is_not_component():
    obj = rules._extract_log("severity: ERROR disk full")
    assert obj["component"] != "severity"
    assert obj["component"] != "ERROR"


# --- F5: space-separated timezone offsets ----------------------------------


def test_f5_ts_any_keeps_offset():
    text = "2026-03-01 12:00:00 +0000 ERROR disk full on /var"
    obj = rules._extract_log(text)
    assert obj["ts"] == "2026-03-01T12:00:00+00:00"


def test_f5_normalize_ts_space_offset():
    assert normalize_ts("2026-03-01 12:00:00 +0000") == \
        "2026-03-01T12:00:00+00:00"


# --- F6: validate helpers ---------------------------------------------------


def test_f6a_empty_enum_value_uses_declared_default():
    fixed = repair({"level": ""}, ["enum:level"], "log_event")
    assert fixed["level"] == "INFO"


def test_f6b_level_alias_logic_only_for_level_enums():
    spec = {"type": {"const": "scratch"},
            "level": {"type": str, "enum": ["junior", "mid", "senior"]}}
    registry.register_schema("scratch", spec,
                             {"strong": [r"\bscratch\b"], "weak": []}, {},
                             defaults={"level": "mid"})
    # Re-pinned 2026-10-10 (enum-aliases refusal contract): the OTel
    # level ladder (and the generic nearest/prefix probe with it) now
    # applies to BUILT-IN enums only. On a user schema, "SENOR" matches
    # no declared member and no declared alias, so the value is REFUSED:
    # the field is UNFILLED (dropped), never rewritten to a
    # nearest-looking member. Declared aliases are the only rewrite.
    fixed = repair({"level": "SENOR"}, ["enum:level"], "scratch")
    assert "level" not in fixed


def test_f6c_default_for_type_is_the_const():
    assert template("issue")["type"] == "issue"
    fixed = repair({}, ["missing:type"], "commit")
    assert fixed["type"] == "commit"
    assert "const:type" not in validate(fixed, "commit")


def test_f6d_non_str_unexpected_keys_are_dropped():
    fixed = repair({"type": "issue", "id": "ISS-42", "title": "t",
                    "labels": [], "priority": "low", "state": "open",
                    3: "junk"},
                   ["unexpected:3"], "issue")
    assert 3 not in fixed
    assert validate(fixed, "issue") == []


# --- F7: registry checks and defaults ---------------------------------------


def _ok_routing():
    return {"strong": [r"\bbadtags\b"], "weak": []}


def test_f7a_enum_on_list_field_rejected():
    with pytest.raises(ValueError, match="enum"):
        registry.register_schema(
            "badtags",
            {"type": {"const": "badtags"},
             "labels": {"type": list, "enum": ["a", "b", "c"]}},
            _ok_routing(), {})


def test_f7b_format_on_non_str_field_rejected():
    with pytest.raises(ValueError, match="format"):
        registry.register_schema(
            "badts",
            {"type": {"const": "badts"},
             "n": {"type": int, "format": "iso8601"}},
            _ok_routing(), {})


def test_f7c_defaults_for_deep_copies():
    spec = {"type": {"const": "mut"}, "tags": {"type": list, "optional": True}}
    registry.register_schema("mut", spec, _ok_routing(), {},
                             defaults={"tags": ["x"]})
    d1 = registry.defaults_for("mut")
    d1["tags"].append("y")
    assert registry.defaults_for("mut")["tags"] == ["x"]


# --- F8: teach fallback alternation -----------------------------------------


def test_f8_fallback_longest_first():
    # Mismatched contexts force the exact-alternation fallback; {a, ab} on
    # "ab" must extract "ab", not "a".
    pattern = teach._field_pattern([("q a", "a"), ("ab r", "ab")])
    import re as _re
    m = _re.search(pattern, "ab")
    assert m and m.group(1) == "ab"


def test_f8_teach_schema_extracts_longest_value():
    report = teach.teach_schema(
        "cmdref",
        {"type": {"const": "cmdref"}, "name": {"type": str}},
        [{"text": "q a", "json": {"type": "cmdref", "name": "a"}},
         {"text": "ab r", "json": {"type": "cmdref", "name": "ab"}}])
    assert report.learned
    from mivia_small_json import registry as reg
    obj = reg.extractor_for("cmdref")("ab r")
    assert obj["name"] == "ab"


# --- F9: extract_learned hygiene -------------------------------------------


def test_f9a_prose_title_span_and_body():
    text = "hey what is this about Login loop on Safari then lots of body text"
    class StubProse:
        classes_ = [0, 1]
        def predict_proba(self, feats):
            # tokens: hey what is this about Login(5) loop(6) on(7) ...
            out = []
            for i in range(len(feats)):
                p = 0.9 if 5 <= i <= 6 else 0.1
                out.append([1 - p, p])
            return out
    res = extract_learned.extract_prose_title(text, model=StubProse())
    assert res is not None
    title, body = res
    assert title == "Login loop"
    assert "hey what is this about" in body
    assert "body text" in body


def test_f9b_positive_column_refuses_multiclass():
    class Multi:
        classes_ = ["INFO", "WARN", "ERROR"]
    assert extract_learned._positive_column(Multi()) is None


def test_f9c_no_openbracket_feature():
    tokens = [("level", 0), ("=", 5)]
    feats = extract_learned.level_token_features("level=", tokens, 1)
    assert not any(f == "openbracket" for f in feats)
    assert not any(f.startswith("openbracket") for f in feats)


def test_f9d_component_merge_requires_word_boundary_prefix(monkeypatch):
    monkeypatch.setattr(extract_learned, "extract_component",
                        lambda text, model, threshold=0.20: "sys")
    # rules component "syslogd"; learned "sys" (contained, not boundary
    # prefix) must NOT replace it.
    obj = extract_learned.extract_log_event(
        "syslogd: disk ok", component_extractor="stub")
    assert obj["component"] == "syslogd"


def test_f9d_component_extension_keeps_longer(monkeypatch):
    monkeypatch.setattr(extract_learned, "extract_component",
                        lambda text, model, threshold=0.20: "sys.log")
    # rules component "sys"; learned "sys.log" extends it on a word
    # boundary -> learned wins.
    obj = extract_learned.extract_log_event(
        "sys: disk ok", component_extractor="stub-extended")
    assert obj["component"] == "sys.log"


@pytest.mark.parametrize("line,expected", [
    ("[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in", "ERROR"),
    ("[Sun Dec 04 04:47:44 2005] [notice] jk2_init() Found child 6", "INFO"),
    ("[Mon Jan 02 03:04:05 2006] [warn] child process exited", "WARN"),
])
def test_bracketed_level_alias_lookup(line, expected):
    """_LEVEL_ANY matches brackets; the alias lookup must strip them or the
    raw '[ERROR]' fails the level enum and repair defaults to INFO (the
    Apache LogHub 0/2000 regression, found 2026-10-10)."""
    assert rules._extract_log(line)["level"] == expected


# --- 2026-10-09 extraction bug hunt: same class as the "[ERROR]" and
# empty-catalogue regressions — extraction values that fail (or fake)
# validity while the repair path silently substitutes defaults. ---------


@pytest.mark.parametrize("text", [
    "hotfix release 4015551234 callers please update, 1 file changed",
    "Bug fix: updated parser for 20261009 format, 2 files changed",
])
def test_pure_digit_run_is_not_a_bare_sha(text):
    """A phone number / date in commit-shaped prose is NOT a sha: the bare
    fallback requires at least one hex letter, else it would emit a
    wrong-but-valid sha (masked as a real commit)."""
    obj = rules._extract_commit(text)
    assert obj["sha"] not in ("4015551234", "20261009")
    assert "4015551234" in obj["message"] or "20261009" in obj["message"]


def test_pure_digit_sha_still_extracts_via_commit_keyword():
    obj = rules._extract_commit("commit 1234567 fix the thing")
    assert obj["sha"] == "1234567"


def test_bare_sha_with_letter_and_digit_still_extracts():
    obj = rules._extract_commit("revert abc1234 because it broke the build")
    assert obj["sha"] == "abc1234"


def test_urgent_priority_maps_to_critical_not_default():
    """'urgent' is a plausible priority word outside the enum: it used to
    fall through empty and get repaired to the default 'medium'."""
    obj = rules._extract_issue("ISS-1234 title priority: urgent state: open")
    assert obj["priority"] == "critical"


def test_tool_call_result_strips_json_delimiters():
    """The `"result": "done"}` capture keeps quotes and the sibling closer;
    the honest value is the quoted string's content, not the wrapper."""
    from mivia_small_json import tool_call as _tc

    _tc.register_tool_call_schema()
    obj = rules.extract(
        '{"name": "fetch_url", "arguments": {"url": "https://x.com"}, '
        '"result": "done"}', "tool_call")
    assert obj["result"] == "done"


def test_tool_call_result_braced_object_kept_verbatim():
    from mivia_small_json import tool_call as _tc

    _tc.register_tool_call_schema()
    obj = rules.extract(
        'tool_call: name=f result={"found": 1}', "tool_call")
    assert obj["result"] == '{"found": 1}'
