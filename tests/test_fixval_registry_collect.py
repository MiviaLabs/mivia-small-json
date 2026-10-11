"""Fixval 2026-10-10: registry registration hygiene + pool banned-set integrity.

Regression tests for the confirmed findings in
reports/code-review-validation-2026-10-10.md — C6/N8/N13 (registry.py
registration) and C5/N9 (collect.py banned-set). Failing tests written
FIRST; each reproduces its bug before the fix.

- C6: register_schema must defensively copy the caller's spec and
  defaults — a caller mutating the enum list (or default list) it
  passed in must not change validate/template/repair afterwards.
  Stored enums become tuples (the built-in convention).
- N8: list defaults must be enforced as list-of-strings at
  registration; otherwise template() emits an object that fails strict
  validation (the last-resort escape emits valid=False records).
- N13: a non-callable extractor must be rejected at registration;
  otherwise rules.extract raises TypeError forever after and the
  pipeline silently degrades the schema to template records.
- C5: the banned-set hash chain must cover rows keyed by "request"
  (data/heldout/spanfill-reserve.jsonl), not only "raw"/"text" —
  held-out texts must be banned from the pool.
- N9: one malformed line (or a valid-JSON non-object row) in any
  banned jsonl must be SKIPPED, not crash the entire collect cycle;
  skipped lines are counted in collect stats under
  ``banned_read_errors``.

Registry tests run under ``registry.temporary_schemas()``; collect
tests build a TEMP tree (``tmp_path``) — never data/.
"""

import json
from pathlib import Path

import pytest

from mivia_small_json import collect, registry, rules
from mivia_small_json.validate import repair, template, validate

ROOT = Path(__file__).resolve().parents[1]


def _register(name, spec, *, defaults=None, extractor=None):
    registry.register_schema(
        name, spec, {"strong": [rf"\b{name}\b"], "weak": []}, {},
        defaults=defaults, extractor=extractor)


# --- C6: caller mutations of spec/defaults must not leak ---------------------


def test_c6_enum_list_mutation_cannot_change_validation():
    with registry.temporary_schemas():
        enum = ["low", "high"]
        _register("c6a", {"type": {"const": "c6a"},
                          "sev": {"type": str, "enum": enum}})
        assert validate({"type": "c6a", "sev": "high"}, "c6a") == []

        enum.append("catastrophic")  # caller mutates the list it passed in
        errors = validate({"type": "c6a", "sev": "catastrophic"}, "c6a")
        assert "enum:sev" in errors, (
            "caller's enum append changed validation — shallow copy (C6)")
        # And repair must not learn the leaked value either.
        fixed = repair({"type": "c6a", "sev": "catastrophic"}, errors, "c6a")
        # Re-pinned 2026-10-10 (enum-aliases refusal contract): a
        # found-but-unmatched enum value on a USER schema is UNFILLED
        # (dropped), never rewritten — the old pin (re-filled "high" via
        # the enum-middle fallback) asserted a silently fabricated value.
        # The C6 point is unchanged and stronger: the leaked value can
        # never re-enter the record.
        assert "sev" not in fixed
        # Stored convention matches the built-ins: enum is a tuple.
        assert registry.resolve("c6a")["sev"]["enum"] == ("low", "high")


def test_c6_default_list_mutation_cannot_change_template():
    with registry.temporary_schemas():
        tags = ["a"]
        # tags is REQUIRED so template() actually emits the default.
        _register("c6b", {"type": {"const": "c6b"},
                          "tags": {"type": list}},
                  defaults={"tags": tags})
        assert template("c6b")["tags"] == ["a"]

        tags.append("MUTATED")  # caller mutates the list it passed in
        assert template("c6b")["tags"] == ["a"], (
            "caller's default-list append leaked into template() (C6)")
        assert registry.defaults_for("c6b")["tags"] == ["a"]
        assert validate(template("c6b"), "c6b") == []


def test_c6_spec_rule_dict_mutation_cannot_change_validation():
    with registry.temporary_schemas():
        rule = {"type": str, "enum": ["low", "high"]}
        _register("c6c", {"type": {"const": "c6c"}, "sev": rule})
        rule["enum"] = ("low",)  # replace, not append
        assert registry.resolve("c6c")["sev"]["enum"] == ("low", "high")


# --- N8: list defaults are enforced as list-of-strings -----------------------


def test_n8_non_string_default_list_items_rejected():
    with registry.temporary_schemas():
        with pytest.raises(ValueError, match="expected a list of strings"):
            _register("n8", {"type": {"const": "n8"},
                             "tags": {"type": list}},
                      defaults={"tags": [1, 2]})
        assert "n8" not in registry.active()


def test_n8_string_list_default_still_accepted():
    with registry.temporary_schemas():
        _register("n8ok", {"type": {"const": "n8ok"},
                           "tags": {"type": list}},
                  defaults={"tags": ["x", "y"]})
        tmpl = template("n8ok")
        assert tmpl["tags"] == ["x", "y"]
        assert validate(tmpl, "n8ok") == []


# --- N13: extractor must be callable or None ---------------------------------


def test_n13_non_callable_extractor_rejected():
    with registry.temporary_schemas():
        with pytest.raises(ValueError, match="extractor"):
            _register("n13", {"type": {"const": "n13"},
                              "f": {"type": str}}, extractor=42)
        with pytest.raises(ValueError, match="extractor"):
            _register("n13", {"type": {"const": "n13"},
                              "f": {"type": str}}, extractor="nope")
        assert "n13" not in registry.active()


def test_n13_callable_extractor_still_accepted():
    with registry.temporary_schemas():
        _register("n13ok", {"type": {"const": "n13ok"},
                            "f": {"type": str}},
                  extractor=lambda text: {"type": "n13ok",
                                          "f": "from-callable"})
        obj = rules.extract("n13ok anything", "n13ok")
        assert obj == {"type": "n13ok", "f": "from-callable"}


# --- C5: request-keyed held-out rows are banned ------------------------------


def test_c5_spanfill_reserve_requests_are_banned():
    rows = [json.loads(line)
            for line in (ROOT / "data" / "heldout" / "spanfill-reserve.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "fixture drift: empty reserve panel"
    assert all("request" in r for r in rows), "fixture drift: key set changed"
    banned = collect._banned_hashes(ROOT)
    leaked = [r["request"] for r in rows
              if collect.text_hash(r["request"]) not in banned]
    assert not leaked, (
        f"{len(leaked)}/{len(rows)} reserve requests are NOT banned from "
        f"the pool (C5); first leak: {leaked[0]!r}")


# --- N9: bad banned-file lines are skipped, cycle survives -------------------


_FROZEN_RAW = "a frozen heldout record that is long enough to keep"
_FROZEN_REQ = "the service redis-cache on edge-fra-3 needs a restart"


def _temp_tree(tmp_path: Path) -> Path:
    held = tmp_path / "data" / "heldout"
    held.mkdir(parents=True)
    (held / "panel.jsonl").write_text("\n".join([
        json.dumps({"raw": _FROZEN_RAW}),
        "NOT JSON AT ALL",                 # malformed line -> skipped
        "[1, 2, 3]",                       # valid JSON, not an object
        json.dumps({"request": _FROZEN_REQ, "gold": {}}),
        "",                                # blank: ignored, not an error
    ]) + "\n", encoding="utf-8")
    inbox = tmp_path / "data" / "pool" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "drop.txt").write_text(
        f"{_FROZEN_RAW}\n\n{_FROZEN_REQ}\n\n"
        "a fresh inbox record long enough to be kept\n\n", encoding="utf-8")
    return tmp_path


def test_n9_bad_banned_lines_are_skipped_not_fatal(tmp_path):
    root = _temp_tree(tmp_path)
    records, stats = collect.collect_all(root)
    # The two bad lines are counted, the three good texts survive hashing.
    assert stats["banned_read_errors"] == 2, stats
    assert stats["frozen_collision"] == 2, stats  # raw text + request text
    assert stats["kept"] == 1, stats
    assert [r["raw"] for r in records] == \
        ["a fresh inbox record long enough to be kept"]


def test_n9_banned_hashes_still_returns_set_with_errors(tmp_path):
    root = _temp_tree(tmp_path)
    banned = collect._banned_hashes(root)
    assert isinstance(banned, set)  # public-ish contract kept for test_pool
    assert collect.text_hash(_FROZEN_RAW) in banned
    assert collect.text_hash(_FROZEN_REQ) in banned


def test_n9_all_bad_lines_in_a_banned_file_still_yield_empty_set(tmp_path):
    root = tmp_path
    held = root / "data" / "heldout"
    held.mkdir(parents=True)
    (held / "junk.jsonl").write_text(
        "garbage\n[1]\n{\"raw\": 42}\n", encoding="utf-8")
    records, stats = collect.collect_all(root)
    # "garbage" does not parse; "[1]" is not an object; {"raw": 42} has a
    # truthy NON-STRING text (would crash text_hash). All three are read
    # errors, skipped honestly — the cycle survives with an empty set.
    assert stats["banned_read_errors"] == 3, stats
    assert records == []
