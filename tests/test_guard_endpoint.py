"""Empty-catalogue guard honesty: chat stays chat, never an issue.

`do_guard` used to classify the composite ``"Tools available: []\n\n
User request: hi"`` — with an EMPTY catalogue the composite reads like a
ticket and the router emitted ``issue`` for plain chat (confirmed P0,
2026-10-09). An empty catalogue means no tool can ever be called, so the
guard classifies the RAW request there; the composite prompt stays the
measured BFCL surface for non-empty catalogues.
"""
from __future__ import annotations

import pytest

from mivia_small_json import bootstrap, pipeline
from mivia_small_json.api import do_guard

CHAT_REQUESTS = ("hi", "what time is it now")


@pytest.fixture(scope="module")
def router():
    return pipeline.load_active_router()


@pytest.fixture(autouse=True)
def _shipped_world():
    bootstrap.register_all_schemas()


@pytest.mark.parametrize("text", CHAT_REQUESTS)
def test_empty_catalogue_chat_is_not_an_issue(text, router):
    """tools=[] + plain chat: ABSTAIN with a text-class record, never issue."""
    out = do_guard(text, [], router)
    assert out["action"] == "ABSTAIN"
    assert out["tool"] is None
    assert out["record"]["type"] != "issue"
    assert out["record"]["type"] in ("text", "no_tool_call")


def test_empty_catalogue_tool_shaped_request_still_abstains(router):
    """No tool exists to call: even a tool-shaped request must ABSTAIN —
    the CALL branch requires a listed tool, and there are none."""
    out = do_guard("calculate the sum of 15 and 30", [], router)
    assert out["action"] == "ABSTAIN"
    assert out["tool"] is None


def test_non_empty_catalogue_keeps_composite_surface(router):
    """The measured BFCL surface is unchanged: with >=1 tool the composite
    prompt still routes (irrelevant chat ABSTAINs, matching requests flow)."""
    tool = {"name": "list_files",
            "description": "List the files in a directory."}
    out = do_guard("hi", [tool], router)
    assert out["action"] == "ABSTAIN"
    out = do_guard("list the files in the directory", [tool], router)
    assert out["action"] in ("CALL", "CLARIFY")


class TestGuardHttpEmptyCatalogue:
    """POST /v1/guard with tools: [] — tokenless TestClient (Host gate)."""

    def test_empty_tools_chat_returns_text_not_issue(self, router):
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        from mivia_small_json import api

        client = TestClient(base_url="http://127.0.0.1",
                            app=api.create_app(token="", router=router))
        for request in CHAT_REQUESTS:
            resp = client.post("/v1/guard",
                               json={"request": request, "tools": []})
            assert resp.status_code == 200, resp.json()
            body = resp.json()
            assert body["action"] == "ABSTAIN", body
            assert body["tool"] is None
            assert body["record"]["type"] != "issue"
            assert body["record"]["type"] in ("text", "no_tool_call")
