"""Rules-baseline tests on tiny fixtures."""

from mivia_small_json import rules
from mivia_small_json.validate import validate

ISSUE_RAW = ("[ISS-1042] Login times out after 30 minutes (priority: high) "
             "labels: auth, backend state: open")
LOG_RAW = "2026-03-01T12:00:00Z ERROR [auth]: token validation failed for user 42"
COMMIT_RAW = ("commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0 fix: handle null "
              "session (3 files changed) author: Jane Doe")


def test_routes_and_extracts_issue():
    assert rules.route(ISSUE_RAW) == "issue"
    obj = rules.extract(ISSUE_RAW, "issue")
    assert obj["id"] == "ISS-1042"
    assert obj["priority"] == "high" and obj["state"] == "open"
    assert obj["labels"] == ["auth", "backend"]
    assert obj["title"] == "Login times out after 30 minutes"


def test_routes_and_extracts_log():
    assert rules.route(LOG_RAW) == "log_event"
    obj = rules.extract(LOG_RAW, "log_event")
    assert obj["level"] == "ERROR" and obj["component"] == "auth"
    assert obj["ts"] == "2026-03-01T12:00:00Z"
    assert obj["message"] == "token validation failed for user 42"


def test_routes_and_extracts_commit():
    assert rules.route(COMMIT_RAW) == "commit"
    obj = rules.extract(COMMIT_RAW, "commit")
    assert obj["sha"] == "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
    assert obj["files_changed"] == 3 and obj["author"] == "Jane Doe"
    assert obj["message"] == "fix: handle null session"


def test_emit_is_strictly_valid_with_bounded_retries():
    for raw, schema in ((ISSUE_RAW, "issue"), (LOG_RAW, "log_event"),
                        (COMMIT_RAW, "commit")):
        obj, attempts, routed, escaped = rules.emit(raw)
        assert routed == schema and attempts <= 2
        assert validate(obj, schema) == [] and escaped is False


def test_baseline_never_reads_annotated_data():
    """The rules layer must not gain the tools to read annotated corpora.

    Structure, not source text: if rules imports json or pathlib it can
    reach a .jsonl corpus, and the baseline's honesty is gone.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(rules))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            else:
                imported.update(a.name.split(".")[0] for a in node.names)
    assert "json" not in imported, "rules must not parse annotated records"
    assert "pathlib" not in imported, "rules must not read corpus files"
