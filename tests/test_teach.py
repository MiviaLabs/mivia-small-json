"""Tests for teach_schema: examples -> patterns -> a registered schema."""

import pytest

from mivia_small_json import pipeline, registry, rules, teach

SPEC = {
    "type": {"const": "task"},
    "id": {"type": str},
    "owner": {"type": str},
    "points": {"type": int},
}
EXAMPLES = [
    {"text": "task task-101 assigned to alice 5 points",
     "json": {"type": "task", "id": "task-101", "owner": "alice", "points": 5}},
    {"text": "task task-102 assigned to bob 8 points",
     "json": {"type": "task", "id": "task-102", "owner": "bob", "points": 8}},
    {"text": "task task-103 assigned to cara 3 points",
     "json": {"type": "task", "id": "task-103", "owner": "cara", "points": 3}},
]


@pytest.fixture
def taught():
    report = teach.teach_schema("task", SPEC, EXAMPLES)
    yield report
    if "task" in registry.active():
        registry.unregister("task")


def _patterns():
    return {f: [p.pattern for p in ps] for f, ps in registry.extraction_for("task")}


def test_teach_extracts_from_an_unseen_example(taught):
    assert set(taught.learned) == {"id", "owner", "points"}
    obj, meta = pipeline.classify("task task-999 assigned to dave 7 points", None)
    assert obj["type"] == "task"
    assert obj["id"] == "task-999"
    assert obj["owner"] == "dave"
    assert obj["points"] == 7


def test_teach_generalises_numbers(taught):
    obj, _ = pipeline.classify("task task-42 assigned to erin 123 points", None)
    assert obj["points"] == 123


def test_teach_reports_uncovered_fields():
    spec = dict(SPEC, priority={"type": str})
    examples = [
        {"text": "task task-1 assigned to alice 5 points",
         "json": {"type": "task", "id": "task-1", "owner": "alice",
                  "points": 5, "priority": "high"}},
        {"text": "task task-2 assigned to bob 8 points",
         "json": {"type": "task", "id": "task-2", "owner": "bob",
                  "points": 8, "priority": "low"}},
    ]
    try:
        report = teach.teach_schema("task", spec, examples)
        assert "priority" in report.uncovered
        assert "priority" not in report.learned
    finally:
        if "task" in registry.active():
            registry.unregister("task")


def test_teach_is_deterministic():
    teach.teach_schema("task", SPEC, EXAMPLES)
    first = _patterns()
    registry.unregister("task")
    teach.teach_schema("task", SPEC, EXAMPLES)
    second = _patterns()
    registry.unregister("task")
    assert first == second


def test_teach_routing_override_wins():
    report = teach.teach_schema(
        "task", SPEC, EXAMPLES,
        routing={"strong": [r"(?i)\bTASKITEM\b"], "weak": []})
    try:
        assert report.routing["strong"] == [r"(?i)\bTASKITEM\b"]
        assert rules.route("TASKITEM 1") == "task"
    finally:
        registry.unregister("task")


def test_teach_rejects_unusable_examples():
    with pytest.raises(ValueError):
        teach.teach_schema("task", SPEC, [])
    with pytest.raises(ValueError):
        teach.teach_schema("task", SPEC, [
            {"text": "nothing matches here",
             "json": {"type": "task", "id": "zz", "owner": "yy", "points": 1}}])


def test_teach_trains_a_classifier_for_a_non_verbatim_field():
    """A field whose value never appears in the text is learned, not guessed."""
    spec = {"type": {"const": "note"}, "body": {"type": str},
            "mood": {"type": str}}
    examples = []
    for i in range(30):
        happy = i % 2 == 0
        text = (f"note entry {i}: " + ("great progress today" if happy
                                       else "blocked on the failing test"))
        examples.append({"text": text,
                         "json": {"type": "note", "body": text,
                                  "mood": "positive" if happy else "negative"}})
    try:
        report = teach.teach_schema("note", spec, examples)
        assert "mood" in report.classified
        assert "mood" not in report.uncovered
        obj, _ = pipeline.classify("note entry 99: great progress today", None)
        assert obj["mood"] == "positive"
        obj2, _ = pipeline.classify("note entry 98: blocked on the failing test",
                                    None)
        assert obj2["mood"] == "negative"
    finally:
        if "note" in registry.active():
            registry.unregister("note")


def test_teach_leaves_a_thin_classifier_field_uncovered():
    """Fewer than MIN_CLASSIFIER_EXAMPLES keeps a field uncovered."""
    spec = {"type": {"const": "note"}, "body": {"type": str},
            "mood": {"type": str}}
    examples = [{"text": f"note {i}: great progress",
                 "json": {"type": "note", "body": f"note {i}: great progress",
                          "mood": "positive"}} for i in range(5)]
    try:
        report = teach.teach_schema("note", spec, examples)
        assert "mood" in report.uncovered
        assert "mood" not in report.classified
    finally:
        if "note" in registry.active():
            registry.unregister("note")


def test_teach_uncovered_field_falls_back_to_default():
    """An uncovered field takes the taught default through classify."""
    spec = dict(SPEC, priority={"type": str})
    examples = [
        {"text": "task task-1 assigned to alice 5 points",
         "json": {"type": "task", "id": "task-1", "owner": "alice",
                  "points": 5, "priority": "high"}},
        {"text": "task task-2 assigned to bob 8 points",
         "json": {"type": "task", "id": "task-2", "owner": "bob",
                  "points": 8, "priority": "low"}},
    ]
    try:
        report = teach.teach_schema("task", spec, examples,
                                    defaults={"priority": "medium"})
        assert "priority" in report.uncovered
        obj, meta = pipeline.classify("task task-9 assigned to zed 7 points",
                                      None)
        assert obj["priority"] == "medium"
        assert meta["valid"] is True
    finally:
        if "task" in registry.active():
            registry.unregister("task")


def test_synthesize_routing_uses_tokens_common_to_half_the_examples():
    """When no token is shared by all, a token in >= half still routes."""
    texts = ["alpha widget one", "beta widget two",
             "gamma three", "delta four"]
    routing = teach._synthesize_routing("z", texts)
    assert any("widget" in marker for marker in routing["strong"]), routing


def test_synthesize_routing_falls_back_to_the_schema_name():
    """Disjoint vocabularies with no repeat leave only the name."""
    routing = teach._synthesize_routing("widget", ["alpha one", "beta two",
                                                   "gamma three"])
    assert routing["strong"] == [r"(?i)\bwidget\b"]


def test_synthesize_routing_handles_non_english_text():
    """Markers come from the examples in any language, no word list."""
    texts = ["ошибка подключения к базе", "ошибка записи в файл",
             "ошибка таймаута сети"]
    routing = teach._synthesize_routing("z", texts)
    assert any("ошибка" in marker for marker in routing["strong"]), routing
