"""Loading the shipped router boots the full shipped schema world.

Regression (found building the Jev head-to-head harness, 2026-10-07):
``pipeline.classify`` with the shipped router could only ever emit
``issue``/``log_event``/``commit`` — every other router class fell
through ``pred not in registry.active()`` to ``rules.route()`` and
silently emitted a WRONG builtin object (pull_request text -> commit).
``load_active_router`` now calls ``bootstrap.register_all_schemas``
first, so the 9-class router's predictions are all emittable. Custom
routers keep the old explicit-registration contract; per-name demand
registration is available as ``bootstrap.register_shipped_schema``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mivia_small_json import bootstrap, pipeline, registry  # noqa: E402

SHIPPED_NON_BUILTIN = frozenset({
    "pull_request", "agent_step", "diagnostic", "diff", "review_comment",
    "tool_call", "no_tool_call", "tool_clarification",
})


def test_load_active_router_boots_shipped_schema_world():
    with registry.temporary_schemas():
        registry._reset()  # conftest boots the 4-schema world; start clean
        assert not SHIPPED_NON_BUILTIN & set(registry.active())
        router = pipeline.load_active_router()
        assert router is not None
        assert SHIPPED_NON_BUILTIN <= set(registry.active())


def test_classify_emits_router_classes_end_to_end():
    router = pipeline.load_active_router()
    cases = [
        ("pull_request: number: 5636 :: title: Cache computed project "
         "quotas for 5 s :: author: Nadia Rahman :: state: draft",
         "pull_request"),
        ("src/main.py:41:5: error TS2322: Type 'string' is not assignable "
         "to type 'number'", "diagnostic"),
        ("2026-10-07T03:11:22Z ERROR db: connection pool exhausted (n=20)",
         "log_event"),
    ]
    for text, expected in cases:
        obj, meta = pipeline.classify(text, router=router)
        assert obj["type"] == expected, (meta["routed_by"], obj["type"])


def test_register_shipped_schema_idempotent_and_rejects_unknown():
    with registry.temporary_schemas():
        registry._reset()
        assert bootstrap.register_shipped_schema("diagnostic") is True
        assert "diagnostic" in registry.active()
        before = registry.active()
        assert bootstrap.register_shipped_schema("diagnostic") is True
        assert registry.active() == before
        assert bootstrap.register_shipped_schema("no_such_schema") is False


def test_register_shipped_schema_knows_text():
    """The shipped `text` schema must have a registrar like every other
    shipped schema: a router `text` prediction in a world without `text`
    active would silently degrade to the rules fallback."""
    with registry.temporary_schemas():
        registry._reset()
        assert bootstrap.register_shipped_schema("text") is True
        assert "text" in registry.active()
