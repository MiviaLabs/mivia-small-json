"""Validator tests: accept / reject / repair cases per schema."""

from mivia_small_json import validate as V

GOOD_ISSUE = {"type": "issue", "id": "ISS-1042", "title": "Login times out",
              "labels": ["auth"], "priority": "high", "state": "open"}
GOOD_LOG = {"type": "log_event", "level": "ERROR", "component": "auth",
            "message": "token expired", "ts": "2026-03-01T12:00:00Z"}
GOOD_COMMIT = {"type": "commit", "sha": "a1b2c3d", "message": "fix: null session",
               "files_changed": 3, "author": "Jane Doe"}


def test_accepts_valid_objects():
    for schema, obj in (("issue", GOOD_ISSUE), ("log_event", GOOD_LOG),
                        ("commit", GOOD_COMMIT)):
        assert V.validate(obj, schema) == []


def test_accepts_iso_ts_variants():
    for ts in ("2026-03-01T12:00:00Z", "2026-03-01T12:00:00.123Z",
               "2026-03-01T12:00:00+02:00", "2026-12-31T23:59:59-05:00"):
        assert V.validate(dict(GOOD_LOG, ts=ts), "log_event") == []


def test_rejects_bad_enum_type_format():
    assert "enum:priority" in V.validate(dict(GOOD_ISSUE, priority="urgent"), "issue")
    assert "type:files_changed" in V.validate(dict(GOOD_COMMIT, files_changed="3"), "commit")
    assert "type:files_changed" in V.validate(dict(GOOD_COMMIT, files_changed=True), "commit")
    assert "type:labels" in V.validate(dict(GOOD_ISSUE, labels="auth"), "issue")
    assert "format:ts" in V.validate(dict(GOOD_LOG, ts="2026-03-01 12:00"), "log_event")
    assert "format:ts" in V.validate(dict(GOOD_LOG, ts="2026-02-30T00:00:00Z"), "log_event")
    assert "format:sha" in V.validate(dict(GOOD_COMMIT, sha="NOTHEX"), "commit")
    assert "format:sha" in V.validate(dict(GOOD_COMMIT, sha="a" * 41), "commit")
    assert "missing:author" in V.validate({k: v for k, v in GOOD_COMMIT.items()
                                           if k != "author"}, "commit")
    assert V.validate(dict(GOOD_ISSUE, extra=1), "issue") == ["unexpected:extra"]


def test_repair_fixes_missing_fields_per_schema():
    for schema, obj in (("issue", {"type": "issue"}),
                        ("log_event", {"type": "log_event"}),
                        ("commit", {"type": "commit"})):
        errors = V.validate(obj, schema)
        assert errors
        repaired = V.repair(obj, errors, schema)
        assert V.validate(repaired, schema) == []
        assert V.validate(V.repair(repaired, V.validate(repaired, schema), schema), schema) == []


def test_repair_coerces_types_and_nearest_enum():
    obj = {"type": "commit", "sha": "A1B2C3D!", "message": 42,
           "files_changed": "7 files", "author": None}
    fixed = V.repair(obj, V.validate(obj, "commit"), "commit")
    assert V.validate(fixed, "commit") == []
    assert fixed["files_changed"] == 7 and fixed["message"] == "42"
    assert fixed["sha"] == "a1b2c3d"
    obj2 = V.repair(dict(GOOD_ISSUE, priority="urgent"), ["enum:priority"], "issue")
    assert obj2["priority"] in V.PRIORITIES
    obj3 = V.repair(dict(GOOD_LOG, level="warning"), ["enum:level"], "log_event")
    assert obj3["level"] == "WARN"


def test_normalize_helpers_and_template():
    assert V.normalize_ts("2026-03-01 12:00:00z") == "2026-03-01T12:00:00Z"
    assert V.normalize_ts("2026-03-01T12:00:00+0200") == "2026-03-01T12:00:00+02:00"
    assert V.normalize_ts("garbage") == ""
    assert V.normalize_sha("A1B2C3D!") == "a1b2c3d"
    # No-truncation law (owner directive 2026-10-10): short hex is absent,
    # never zero-padded into a fabricated id ("aa" used to become 7 chars).
    assert V.normalize_sha("aa") == ""
    assert V.normalize_sha("a" * 45) == "a" * 40
    for schema in V.SCHEMAS:
        assert V.validate(V.template(schema), schema) == []


def test_otel_severity_levels_and_aliases():
    for level in ("TRACE", "FATAL"):
        assert V.validate(dict(GOOD_LOG, level=level), "log_event") == []
    assert V.repair(dict(GOOD_LOG, level="critical"), ["enum:level"],
                    "log_event")["level"] == "FATAL"
    assert V.repair(dict(GOOD_LOG, level="verbose"), ["enum:level"],
                    "log_event")["level"] == "TRACE"
    assert V.repair(dict(GOOD_LOG, level="warning"), ["enum:level"],
                    "log_event")["level"] == "WARN"


def test_optional_commit_type():
    minimal = {"type": "commit", "sha": "a1b2c3d", "message": "m",
               "files_changed": 1, "author": "a"}
    assert V.validate(minimal, "commit") == []          # absent is valid
    with_type = dict(minimal, commit_type="fix")
    assert V.validate(with_type, "commit") == []
    assert "enum:commit_type" in V.validate(dict(minimal, commit_type="banana"),
                                             "commit")
    fixed = V.repair(dict(minimal, commit_type="banana"),
                     V.validate(dict(minimal, commit_type="banana"), "commit"),
                     "commit")
    assert fixed["commit_type"] in V.COMMIT_TYPES      # bounded enum fallback
    probed = V.repair(dict(minimal, commit_type="fi"),
                      V.validate(dict(minimal, commit_type="fi"), "commit"),
                      "commit")
    assert probed["commit_type"] == "fix"               # prefix probe repair
    tmpl = V.template("commit")
    assert "commit_type" not in tmpl
    assert V.validate(tmpl, "commit") == []
