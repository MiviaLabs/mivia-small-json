"""Registry tests: registration validation, merged view, pull_request end-to-end.

The documented example schema (``pull_request``) is registered
suite-wide by ``tests/conftest.py`` through the public API — these
tests prove a NEW output format works end to end with zero changes to
built-in code: synth -> train (committed artifact) -> classify.
"""

import hashlib
import json
from pathlib import Path

import pytest
from sklearn.metrics import f1_score

from mivia_small_json import pipeline, registry, rules, synth
from mivia_small_json.model import MODEL_PATH
from mivia_small_json.validate import SCHEMAS, repair, template, validate

ROOT = Path(__file__).resolve().parents[1]

GOOD_PR = {"type": "pull_request", "number": 1042, "title": "Fix login loop",
           "author": "Jane Doe", "state": "merged", "additions": 120,
           "deletions": 15}


@pytest.fixture
def clean_registry():
    """A test may register/unregister freely; the suite-wide autouse
    hermetic fixture (tests/conftest.py) restores the world on exit."""
    yield


# --- registration validation ------------------------------------------------


def _expect_rejected(spec, routing, extraction, *rest, match):
    with pytest.raises(ValueError, match=match):
        registry.register_schema("scratch", spec, routing, extraction, *rest)


def test_bad_specs_are_rejected(clean_registry):
    ok_routing = {"strong": [r"\bscratch\b"], "weak": []}
    _expect_rejected({"id": {"type": str}}, ok_routing, {},
                     match='missing const "type"')
    _expect_rejected({"type": {"const": "other"}}, ok_routing, {},
                     match='must be exactly')
    _expect_rejected({"type": {"const": "scratch"},
                      "f": {"type": float}}, ok_routing, {},
                     match="unknown type")
    _expect_rejected({"type": {"const": "scratch"},
                      "f": {"type": str, "enum": []}}, ok_routing, {},
                     match="non-empty list of strings")
    _expect_rejected({"type": {"const": "scratch"},
                      "f": {"type": str, "bogus": 1}}, ok_routing, {},
                     match="unknown rule key")
    _expect_rejected({"type": {"const": "scratch"},
                      "f": {"type": str, "format": "blob"}}, ok_routing, {},
                     match="unknown format")


def test_bad_routing_extraction_defaults_examples_rejected(clean_registry):
    spec = {"type": {"const": "scratch"}, "f": {"type": str}}
    with pytest.raises(ValueError, match="routing"):
        registry.register_schema("scratch", spec, {"strong": []}, {})
    with pytest.raises(ValueError, match="does not compile"):
        registry.register_schema("scratch", spec,
                                 {"strong": ["("], "weak": []}, {})
    with pytest.raises(ValueError, match="extraction.f"):
        registry.register_schema("scratch", spec,
                                 {"strong": [r"\bscratch\b"], "weak": []},
                                 {"f": ["nof groupe"]})
    with pytest.raises(ValueError, match="extraction"):
        registry.register_schema("scratch", spec,
                                 {"strong": [r"\bscratch\b"], "weak": []},
                                 {"ghost": ["(g)"]})
    with pytest.raises(ValueError, match="defaults.f"):
        registry.register_schema("scratch", spec,
                                 {"strong": [r"\bscratch\b"], "weak": []},
                                 {"f": [r"\b(f)oo\b"]}, {"f": 3})
    with pytest.raises(ValueError, match="examples"):
        registry.register_schema("scratch", spec,
                                 {"strong": [r"\bscratch\b"], "weak": []},
                                 {"f": [r"\b(f)oo\b"]}, None, [{"f": 1}])


def test_duplicate_and_builtin_names_rejected(clean_registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register_schema("issue", {"type": {"const": "issue"}},
                                 {"strong": [], "weak": []}, {})
    with pytest.raises(ValueError, match="already registered"):
        registry.register_schema("pull_request",
                                 {"type": {"const": "pull_request"}},
                                 {"strong": [], "weak": []}, {})


def test_unregister_rules(clean_registry):
    registry.unregister("pull_request")
    assert "pull_request" not in registry.active()
    with pytest.raises(ValueError, match="built-in"):
        registry.unregister("issue")
    with pytest.raises(ValueError, match="not registered"):
        registry.unregister("ghost")


# --- merged view semantics ----------------------------------------------------


def test_active_and_resolve_merged_view():
    # Hermetic: the suite-wide conftest registration (and any schema a
    # router boot may have added earlier in the run) must not leak in.
    with registry.temporary_schemas():
        registry._reset()
        synth.register_example_schemas()
        assert registry.BUILT_IN_SCHEMAS == ("issue", "log_event", "commit")
        assert registry.active() == ("issue", "log_event", "commit",
                                     "pull_request", "agent_step")
        from mivia_small_json import schemas

        assert registry.resolve("issue") is schemas.SPEC["issue"]
        assert registry.resolve("pull_request")["state"]["enum"] == \
            ("open", "merged", "closed", "draft")
        with pytest.raises(KeyError, match="unknown schema"):
            registry.resolve("ghost")
        assert list(SCHEMAS) == list(registry.active())


def test_validate_repair_template_work_for_registered_schema():
    assert validate(GOOD_PR, "pull_request") == []
    assert "enum:state" in validate(dict(GOOD_PR, state="banana"),
                                    "pull_request")
    fixed = repair(dict(GOOD_PR, state="cloze"),
                   validate(dict(GOOD_PR, state="cloze"), "pull_request"),
                   "pull_request")
    # Re-pinned 2026-10-10 (enum-aliases refusal contract): a found-but-
    # unmatched enum value on a USER schema is UNFILLED (dropped), never
    # engine-rewritten. The old pin (state == "closed" via _nearest's
    # prefix probe) encoded exactly the silent rewrite the honesty law
    # removed — only the schema author's declared aliases may coerce.
    assert "state" not in fixed
    broken = {"type": "pull_request", "number": "12", "additions": True}
    errors = validate(broken, "pull_request")
    assert errors
    fixed = repair(broken, errors, "pull_request")
    assert validate(fixed, "pull_request") == []
    assert fixed["number"] == 12 and fixed["deletions"] == 0
    tmpl = template("pull_request")
    assert validate(tmpl, "pull_request") == []
    assert validate(dict(GOOD_PR, extra=1), "pull_request") == ["unexpected:extra"]


def test_rules_route_and_extract_registered_schema():
    canonical = ("pull_request: number: 1042 :: title: Fix login loop :: "
                 "author: Jane Doe :: state: merged :: additions: 120 :: "
                 "deletions: 15")
    assert rules.route(canonical) == "pull_request"
    obj = rules.extract(canonical, "pull_request")
    assert obj["number"] == "1042" and obj["state"] == "merged"
    # one capture group per registered pattern, enforced at registration
    assert all(p.groups == 1
               for _, patterns in registry.extraction_for("pull_request")
               for p in patterns)
    for field, _ in registry.extraction_for("pull_request"):
        assert field in GOOD_PR and field != "type"


def test_built_in_routing_unchanged_by_registration():
    assert rules.route("commit " + "a" * 40 + " author: X - 2 files changed") \
        == "commit"
    assert rules.route("2026-03-01T12:00:00Z ERROR [auth]: token expired") \
        == "log_event"
    assert rules.route("[ISS-1042] Login times out (priority: high) "
                       "labels: api state: open") == "issue"


# --- synth degrade-from-examples path -----------------------------------------


def _digest(splits):
    payload = "".join(json.dumps(r, sort_keys=True) + "\n"
                      for split in splits.values() for r in split)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_registered_examples_drive_corpus_generation():
    assert len(registry.examples_for("pull_request")) >= 40
    splits = synth.build_splits(10, 42, n_registered=8)
    pr = [r for sp in splits.values() for r in sp if r["schema"] == "pull_request"]
    assert len(pr) == 8
    for r in pr:
        assert validate(r["gold"], "pull_request") == []
        assert set(r) == {"schema", "raw", "gold"}
    # built-in contract untouched: no registered rows unless asked
    assert not [r for sp in synth.build_splits(10, 42).values() for r in sp
                if r["schema"] == "pull_request"]
    assert _digest(synth.build_splits(10, 42, 8)) == \
        _digest(synth.build_splits(10, 42, 8))


# --- end-to-end proof: synth -> train -> classify ------------------------------


def _heldout_rows():
    return [json.loads(line) for line in
            (ROOT / "data" / "heldout" / "test.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]


def test_pull_request_end_to_end_valid_and_f1():
    rows = _heldout_rows()
    pr_rows = [r for r in rows if r["schema"] == "pull_request"]
    assert len(pr_rows) >= 40, "expected ~50 held-out pull_request records"
    router = pipeline.load_router(MODEL_PATH)
    assert router is not None
    pred, valid, pr_valid = [], 0, 0
    for r in rows:
        obj, meta = pipeline.classify(r["raw"], router)
        pred.append(obj.get("type"))
        routed_right = meta["valid"] and obj.get("type") == r["schema"]
        valid += routed_right
        if r["schema"] == "pull_request" and routed_right:
            pr_valid += 1
    labels = list(SCHEMAS)
    gold = [r["schema"] for r in rows]
    per_class = f1_score(gold, pred, average=None, labels=labels)
    f1_by_class = dict(zip(labels, (float(f) for f in per_class)))
    pr_rate = pr_valid / len(pr_rows)
    assert pr_rate >= 0.99, \
        f"pull_request held-out valid {pr_rate:.4f} < 0.99"
    assert f1_by_class["pull_request"] >= 0.90, \
        f"pull_request routing F1 {f1_by_class['pull_request']:.4f} < 0.90 " \
        f"(per class: {f1_by_class})"
    overall = valid / len(rows)
    assert overall >= 0.99, f"held-out valid {overall:.4f} < 0.99"


def test_classify_never_raises_on_registered_schema_garbage():
    router = pipeline.load_router(MODEL_PATH)
    garbage = [
        "pull_request: number: bananas :: state: banana",
        "PR #\x00 (+a -b) title \x1b[31m",
        "pull request " * 200,
    ]
    for s in garbage:
        for r in (None, router):
            obj, meta = pipeline.classify(s, r)
            tag = f"input={s[:60]!r} router={'model' if r is router else 'none'}"
            assert isinstance(obj, dict), tag
            assert set(meta) == {"routed_by", "valid", "retries", "latency_ms"}, tag
            assert 0 <= meta["retries"] <= 2, tag
