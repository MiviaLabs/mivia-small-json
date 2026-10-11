"""Universal-structuring 1d: the /v1/structure surface.

``pipeline.structure(text, schema)`` forces a declared output schema —
no routing: extract -> validate -> repair (<= 2) -> template escape,
defaults instead of fabrication. The HTTP surface (POST /v1/structure
{text, schema}) exposes it; built-ins keep their dedicated endpoints.
"""
from __future__ import annotations

import pytest

from mivia_small_json import bootstrap, pipeline, registry


@pytest.fixture(autouse=True)
def _shipped_world():
    bootstrap.register_all_schemas()


def test_structure_forces_the_declared_schema():
    obj, meta = pipeline.structure(
        "ERROR [auth]: token expired", "log_event")
    assert obj["type"] == "log_event"
    assert obj["level"] == "ERROR"
    assert meta["routed_by"] == "declared"
    assert meta["valid"] is True


def test_structure_uses_defaults_not_fabrication():
    obj, meta = pipeline.structure("", "log_event")
    # Empty input: extraction finds nothing, the record still matches the
    # schema through defaults/repair and says so honestly.
    assert obj["type"] == "log_event"
    assert set(obj) <= {"type", "level", "component", "message", "ts"}
    assert isinstance(meta["valid"], bool)


def test_structure_never_raises_on_any_str():
    for text in ("", "%%%//////", "éè中文" * 500, "diff --git a/x b/x"):
        obj, meta = pipeline.structure(text, "issue")
        assert obj["type"] == "issue"
        assert "valid" in meta


def test_structure_unknown_schema_is_a_value_error():
    with pytest.raises(ValueError):
        pipeline.structure("hello", "no_such_schema")


def test_structure_refuses_tool_call():
    """`tool_call` extraction is special-cased through the guard ladder
    (veto/clarification, needs a tools catalogue) — the declared-schema
    contract would silently return no_tool_call records. Honest refusal:
    use /v1/guard (2026-10-08 review P2)."""
    with pytest.raises(ValueError, match="/v1/guard"):
        pipeline.structure("find my invoices", "tool_call")


def test_structure_works_for_user_registered_schemas():
    with registry.temporary_schemas():
        registry.register_schema(
            "ticket_urgency",
            {"type": {"const": "ticket_urgency"},
             "urgency": {"type": str, "enum": ("low", "high"),
                         "aliases": {"high": ["urgent", "asap",
                                              "immediately"]}}},
            {"strong": [], "weak": []},
            {"urgency": [r"(?i)\b(urgent|asap|immediately)\b"]},
            defaults={"urgency": "low"},
        )
        obj, meta = pipeline.structure(
            "this is urgent, we are blocked in production", "ticket_urgency")
        assert obj["type"] == "ticket_urgency"
        # The author-declared alias map is the ONLY rewrite: "urgent" is
        # not an enum member, so without the declaration above the
        # coercion contract REFUSES it (field unfilled -> default "low")
        # — the coercion-honesty law. Declared, it canonicalizes to
        # "high".
        assert obj["urgency"] == "high"
        obj2, _ = pipeline.structure("whenever someone gets a minute",
                                     "ticket_urgency")
        assert obj2["urgency"] == "low"

    def test_structure_refuses_undeclared_enum_surfaces(self):
        """Pin the refusal contract itself: an extracted surface the
        author never declared is UNFILLED (flows to the default), never
        nearest-matched — 'it is urgent' must never silently become
        'low' under an enum that only declared 'high'/'low'."""
        with registry.temporary_schemas():
            registry.register_schema(
                "ticket_urgency_bare",
                {"type": {"const": "ticket_urgency_bare"},
                 "urgency": {"type": str, "enum": ("low", "high")}},
                {"strong": [], "weak": []},
                {"urgency": [r"(?i)\b(urgent|asap|immediately)\b"]},
                defaults={"urgency": "low"},
            )
            obj, meta = pipeline.structure(
                "this is urgent, we are blocked in production",
                "ticket_urgency_bare")
            assert obj["type"] == "ticket_urgency_bare"
            assert obj["urgency"] == "low", (
                "an undeclared surface was rewritten — the coercion "
                "refusal contract regressed")


class TestStructureHttp:
    def test_structure_endpoint_round_trip(self):
        from starlette.testclient import TestClient

        from mivia_small_json import api

        client = TestClient(base_url="http://127.0.0.1",
                        app=api.create_app(token="", router=object()))
        resp = client.post("/v1/structure",
                           json={"text": "ERROR [auth]: token expired",
                                 "schema": "log_event"})
        assert resp.status_code == 200, resp.json()
        body = resp.json()
        assert body["schema"] == "log_event"
        assert body["record"]["type"] == "log_event"
        assert body["record"]["level"] == "ERROR"
        assert body["valid"] is True
        assert body["routed_by"] == "declared"
        assert body["latency_ms"] >= 0

    def test_structure_endpoint_unknown_schema_is_422(self):
        from starlette.testclient import TestClient

        from mivia_small_json import api

        client = TestClient(base_url="http://127.0.0.1",
                        app=api.create_app(token="", router=object()))
        resp = client.post("/v1/structure",
                           json={"text": "x", "schema": "no_such_schema"})
        assert resp.status_code == 422
        assert "no_such_schema" in resp.json()["error"]

    def test_structure_endpoint_refuses_tool_call(self):
        from starlette.testclient import TestClient

        from mivia_small_json import api

        client = TestClient(base_url="http://127.0.0.1",
                        app=api.create_app(token="", router=object()))
        resp = client.post("/v1/structure",
                           json={"text": "x", "schema": "tool_call"})
        assert resp.status_code == 422
        assert "/v1/guard" in resp.json()["error"]

    def test_structure_endpoint_requires_schema_field(self):
        from starlette.testclient import TestClient

        from mivia_small_json import api

        client = TestClient(base_url="http://127.0.0.1",
                        app=api.create_app(token="", router=object()))
        resp = client.post("/v1/structure", json={"text": "x"})
        assert resp.status_code == 422
