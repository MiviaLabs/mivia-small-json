"""Residual-fix regression tests (residfix round, 2026-10-10).

Three small defects recorded by review agents across two fix rounds, one
section per bug:

1. ``tool_match.fillability_veto`` raised on a non-str request while its
   sibling ``validate_fillability`` was hardened the same day to coerce it
   to ``""`` — the asymmetry is a live trap; both predicates must degrade
   to the honest underspecified/missing path, never raise.
2. ``api._core`` served a SERVER-side JSON-encoding failure as a 422
   client error: a lone surrogate in the text makes starlette's render
   raise ``UnicodeEncodeError`` (a ``ValueError`` subclass), which fell
   into the ``ValueError -> 422`` branch with the codec message in the
   body. It must be an honest 500.
3. ``span_fill.fill_arguments`` donated a NEIGHBOR slot's number to a slot
   whose own key-delimited/alignment value failed coercion
   (``amount=1e309 fee=2`` emitted ``{'amount': 2}`` — fee's value as
   amount). A filled-but-invalid slot stays unfilled (missing) and must
   never receive a window or positional donation.
"""

from __future__ import annotations

import json

import pytest

from mivia_small_json import span_fill, tool_match


# ---------------------------------------------------------------------------
# Bug 1: fillability_veto non-str request must degrade, never raise
# ---------------------------------------------------------------------------

_THREE_NUM_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "first operand"},
                "b": {"type": "number", "description": "second operand"},
                "c": {"type": "number", "description": "third operand"},
            },
            "required": ["a", "b", "c"],
        },
    },
}

_STRING_SLOT_TOOL = {
    "name": "delete_database",
    "description": "Delete a named database.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}


class TestVetoNonStrRequest:
    @pytest.mark.parametrize("bad", [None, 42, ["compute a=3 b=2 c=1"]])
    def test_non_str_request_degrades_not_raises(self, bad):
        ok, reason, missing = tool_match.fillability_veto(
            _THREE_NUM_TOOL, bad)
        assert (ok, reason) == (False, "underspecified_query")
        assert missing == ["a", "b", "c"]

    def test_non_str_request_string_slot_degrades_not_raises(self):
        ok, reason, missing = tool_match.fillability_veto(
            _STRING_SLOT_TOOL, None)
        assert (ok, reason) == (False, "underspecified_query")
        assert missing == ["name"]

    def test_parity_with_validate_fillability(self):
        """The sibling was hardened first; both must agree on the outcome."""
        for bad in (None, 42):
            assert tool_match.fillability_veto(
                _THREE_NUM_TOOL, bad) == \
                tool_match.validate_fillability(_THREE_NUM_TOOL, bad)

    def test_str_request_decision_unchanged(self):
        assert tool_match.fillability_veto(
            _THREE_NUM_TOOL, "compute a=3 b=2 c=1") == (True, "", [])

    def test_is_veto_numeric_non_dict_props_degrades(self):
        """The props lookup must never raise on a malformed catalogue: a
        non-dict props behaves like an empty one (no type info — the name
        hint then decides, the pre-existing semantics)."""
        assert tool_match._is_veto_numeric("count", None) == \
            tool_match._is_veto_numeric("count", {}) is True
        assert tool_match._is_veto_numeric("country", None) is False
        assert tool_match._is_veto_numeric("country", 3) is False


# ---------------------------------------------------------------------------
# Bug 2: server-side codec failure must be 500, not a 422 client error
# ---------------------------------------------------------------------------


class TestServerSideCodecFailure:
    @staticmethod
    def _client():
        pytest.importorskip("fastapi")
        from starlette.testclient import TestClient

        from mivia_small_json import api

        app = api.create_app(token="", router=object())
        return TestClient(base_url="http://127.0.0.1", app=app)

    @staticmethod
    def _post(client, payload):
        # ensure_ascii=True is the safe wire form: the lone surrogate
        # travels as a \ud800 escape and is decoded server-side.
        body = json.dumps(payload).encode()
        return client.post("/v1/extract", content=body,
                           headers={"Content-Type": "application/json"})

    def test_lone_surrogate_is_500_not_422(self):
        client = self._client()
        resp = self._post(
            client,
            {"text": "ERROR [auth]: token expired " + chr(0xd800)})
        assert resp.status_code == 500, (resp.status_code, resp.text)
        payload = resp.json()
        assert "unpaired surrogates" in payload.get("error", ""), payload

    def test_client_side_errors_stay_422(self):
        """The new 500 branch must not swallow the 4xx client-error law
        (including UnicodeDecodeError from an unreadable request body)."""
        client = self._client()
        resp = self._post(client, {"text": 123})
        assert resp.status_code == 422, (resp.status_code, resp.text)
        resp = client.post(
            "/v1/extract", content=b'{"text": "\xff\xfe"}',
            headers={"Content-Type": "application/json"})
        assert resp.status_code == 422, (resp.status_code, resp.text)


# ---------------------------------------------------------------------------
# Bug 3: a filled-but-invalid slot receives no neighbor donation
# ---------------------------------------------------------------------------


def _two_num_tool():
    return {
        "name": "pay",
        "parameters": {
            "type": "object",
            "properties": {"amount": {"type": "number"},
                           "fee": {"type": "number"}},
            "required": ["amount", "fee"],
        },
    }


class TestFilledButInvalidNoDonation:
    def test_uncoercible_own_value_does_not_donate_neighbor(self):
        """The recorded repro: fee's 2 must not be emitted as amount."""
        filled, missing = span_fill.fill_arguments(
            _two_num_tool(), "amount=1e309 fee=2")
        assert filled == {"fee": 2}
        assert missing == ["amount"]

    def test_invalidated_slot_receives_no_window_or_positional_number(self):
        """The invalid own value (NaN) must not be papered over by the free
        numbers 7/9; the other slot still gets its own nearest number."""
        filled, missing = span_fill.fill_arguments(
            _two_num_tool(), "amount=NaN fee 7 9")
        assert filled == {"fee": 7}
        assert missing == ["amount"]

    def test_integer_own_value_failure_does_not_donate_neighbor(self):
        tool = {
            "name": "f",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer"},
                               "m": {"type": "number"}},
                "required": ["n", "m"],
            },
        }
        filled, missing = span_fill.fill_arguments(tool, "n=3.5 m=2")
        assert filled == {"m": 2}
        assert missing == ["n"]

    def test_second_own_mention_still_rescues_the_slot(self):
        """A later own mention with a coercible value fills the slot —
        invalidation only applies when NO own value coerces."""
        filled, missing = span_fill.fill_arguments(
            _two_num_tool(), "amount=1e309 then amount is 5 and fee=2")
        assert filled == {"amount": 5, "fee": 2}
        assert missing == []

    def test_alignment_own_value_invalid_stays_missing(self):
        tool = {
            "name": "c",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "number"}},
                "required": ["amount"],
            },
        }
        filled, missing = span_fill.fill_arguments(tool, "amount is 1e309")
        assert filled == {}
        assert missing == ["amount"]

    # -- preservation pins -------------------------------------------------

    def test_window_pin_transfer_amount_fee_unchanged(self):
        filled, missing = span_fill.fill_arguments(
            _two_num_tool(), "transfer amount 1,000 fee 2")
        assert filled == {"amount": 1000, "fee": 2}
        assert missing == []

    def test_window_assignment_without_own_value_unchanged(self):
        """No own key-delimited/alignment value: the window ladder still
        fills ('pay 100 to x' shapes)."""
        filled, missing = span_fill.fill_arguments(
            _two_num_tool(), "please pay 100 with a 2 fee")
        assert filled == {"amount": 100, "fee": 2}
        assert missing == []

    def test_positional_fill_without_own_values_unchanged(self):
        tool = {
            "name": "calc_sum",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"},
                               "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        }
        filled, missing = span_fill.fill_arguments(
            tool, "calculate sum of 10 and 20")
        assert filled == {"a": 10, "b": 20}
        assert missing == []

    def test_b7_nonfinite_rejection_intact(self):
        tool = {
            "name": "c",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}},
                "required": ["a"],
            },
        }
        filled, missing = span_fill.fill_arguments(tool, "a=1e309")
        assert filled == {}
        assert missing == ["a"]
        assert json.loads(span_fill.arguments_json(filled)) == {}
