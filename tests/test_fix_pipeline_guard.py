"""Regression tests for the confirmed pipeline/span_fill/logtemplate bugs.

One test class (or flat group) per finding, named F1..F11 as in the
finding list. All tests are deterministic; the guard is faked through
``tool_match._get_guard`` so no model artifact is needed.
"""

import inspect
import json

import pytest

from mivia_small_json import bootstrap, pipeline, rules, span_fill
from mivia_small_json.logtemplate import TemplateCache
from mivia_small_json.tool_call import TOOL_CLARIFICATION_SPEC
from mivia_small_json.validate import validate


@pytest.fixture
def shipped_world():
    bootstrap.register_all_schemas()
    yield


class _FakeGuard:
    """best_match returns the configured tool and score."""

    def __init__(self, tool, score):
        self._tool = tool
        self._score = score

    def best_match(self, text):
        return self._tool, self._score


def _catalogue(tools, request):
    return f"Tools available:\n[{json.dumps(tools)}]\n\nUser request: {request}"


def _fn_tool(name, props, required):
    return {"type": "function", "function": {
        "name": name,
        "description": f"{name} tool",
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required,
        },
    }}


# --- F1: guard acceptance must never fall through to rules.extract ----------

class TestF1GuardFallThrough:
    def test_wrong_tool_not_emitted_on_guard_acceptance(self, monkeypatch,
                                                        shipped_world):
        delete_db = _fn_tool("delete_database", {}, [])
        get_weather = _fn_tool(
            "get_weather", {"city": {"type": "string"}}, ["city"])
        text = _catalogue([delete_db, get_weather],
                          "Call get_weather for Paris.")
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard",
            lambda: _FakeGuard(get_weather, 0.9))
        # Forced-absent (loader's negative-cache seam): this pins the
        # NO-ARTIFACT world, where the string family admits the tool but
        # the deterministic ladders have no slot evidence — the row stays
        # on the veto path and emits the honest-minimum {} (the arm's
        # learned fill is pinned by the stringparam tests with the
        # artifact injected).
        from mivia_small_json import span_fill as _sf

        saved = (_sf._SPANFILL_CLF, _sf._SPANFILL_LOADED)
        _sf._SPANFILL_CLF, _sf._SPANFILL_LOADED = None, True
        try:
            obj, meta = pipeline._classify_known_schema(
                text, "tool_call", "model")
        finally:
            _sf._SPANFILL_CLF, _sf._SPANFILL_LOADED = saved
        # The fully-specified string-slot call emits the guard's tool with
        # the honest-minimum arguments (no fabricated values); the name is
        # never a leaked catalogue name (the original F1 finding).
        assert obj.get("type") == "tool_call"
        assert obj.get("name") == "get_weather"
        assert json.loads(obj["arguments"]) == {}
        assert meta["valid"] is True

    def test_required_slot_dropped_is_a_clarification(self, monkeypatch,
                                                      shipped_world):
        # A request that truly lacks a required numeric slot is still a
        # clarification: the decision-layer fillability veto fires before
        # the call is emitted (the digit-free stripped-request
        # construction).
        thermostat = _fn_tool(
            "set_temperature", {"temp": {"type": "number"}}, ["temp"])
        text = _catalogue([thermostat],
                          "Set the thermostat to a cozy warm temperature.")
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard",
            lambda: _FakeGuard(thermostat, 0.9))
        obj, meta = pipeline._classify_known_schema(text, "tool_call", "model")
        assert obj.get("type") == "tool_clarification"
        assert obj["tool"] == "set_temperature"
        assert obj["missing_parameters"] == ["temp"]
        assert meta["valid"] is True

    def test_fully_filled_numeric_tool_still_emits_tool_call(
            self, monkeypatch, shipped_world):
        calc = _fn_tool("calc_sum",
                        {"a": {"type": "number"}, "b": {"type": "number"}},
                        ["a", "b"])
        text = _catalogue([calc], "add a=5 and b=3 please")
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard",
            lambda: _FakeGuard(calc, 0.9))
        obj, meta = pipeline._classify_known_schema(text, "tool_call", "model")
        assert obj.get("type") == "tool_call"
        assert obj["name"] == "calc_sum"
        assert json.loads(obj["arguments"]) == {"a": 5, "b": 3}
        assert meta["valid"] is True

    def test_clarification_matches_spec(self, monkeypatch, shipped_world):
        thermostat = _fn_tool(
            "set_temperature", {"temp": {"type": "number"}}, ["temp"])
        text = _catalogue([thermostat],
                          "Set the thermostat to a cozy warm temperature.")
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard",
            lambda: _FakeGuard(thermostat, 0.9))
        obj, _ = pipeline._classify_known_schema(text, "tool_call", "model")
        for field in TOOL_CLARIFICATION_SPEC:
            assert field in obj
        assert obj["reason"] in (
            "missing_arguments", "ambiguous_intent", "underspecified_query")
        assert isinstance(obj["clarification_prompt"], str)
        assert obj["clarification_prompt"]
        assert validate(obj, "tool_clarification") == []

    def test_guard_rejection_still_abstains(self, monkeypatch, shipped_world):
        get_weather = _fn_tool(
            "get_weather", {"city": {"type": "string"}}, ["city"])
        text = _catalogue([get_weather], "What is the weather in Paris?")
        text = _catalogue([get_weather], "Run the weather lookup for Paris.")
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard",
            lambda: _FakeGuard(get_weather, 0.1))
        obj, _ = pipeline._classify_known_schema(text, "tool_call", "model")
        assert obj.get("type") == "no_tool_call"


# --- F2: _severity_clarification must satisfy its own schema ----------------

class TestF2SeverityClarification:
    def test_severity_clarification_is_spec_compliant(self, shipped_world):
        obj = pipeline._severity_clarification(["info", "error"])
        assert validate(obj, "tool_clarification") == []
        assert obj["reason"] == "ambiguous_intent"
        assert obj["clarification_prompt"]
        assert "options" not in obj
        assert "info" in obj["clarification_prompt"]
        assert "error" in obj["clarification_prompt"]


# --- F3: family_for must gate name hints on declared type -------------------

class TestF3FamilyTypeGate:
    def test_declared_string_is_never_numeric(self):
        # Deterministic-only family admission (2026-10-10): a declared
        # string slot admits the tool to the STRING family — never the
        # numeric one.
        tool = _fn_tool("book_flight",
                        {"country": {"type": "string"}}, ["country"])
        assert span_fill.family_for(tool) == "string"
        tool = _fn_tool("f",
                        {"festival": {"type": "string"}}, ["festival"])
        assert span_fill.family_for(tool) == "string"

    def test_declared_numeric_stays_numeric(self):
        tool = _fn_tool("calc_sum",
                        {"a": {"type": "number"}, "b": {"type": "integer"}},
                        ["a", "b"])
        assert span_fill.family_for(tool) == "numeric"

    def test_name_hints_apply_only_when_type_absent(self):
        tool = _fn_tool("f", {"count": {}}, ["count"])
        assert span_fill.family_for(tool) == "numeric"
        # Undeclared type WITHOUT a whole-token hint match admits no
        # family (the spanfix bug-2 contract: the "count" substring in
        # "country" is never a hint, and an unadmitted shape keeps the
        # caller's fillability path).
        tool = _fn_tool("f", {"country": {}}, ["country"])
        assert span_fill.family_for(tool) is None

    def test_string_country_no_longer_fills_from_a_number(self):
        # Forced-absent (loader's negative-cache seam): with no learned
        # artifact the string family admits the tool but the ladders have
        # no evidence for "country" here — the row comes back UNHANDLED
        # (never a partial), and the number 2 never lands in the slot.
        tool = _fn_tool("book_flight",
                        {"country": {"type": "string"}}, ["country"])
        saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = None, True
        try:
            filled, missing = span_fill.fill_arguments(
                tool, "book a flight to France for 2 people.")
        finally:
            span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = saved
        assert filled is None  # unhandled: the caller's veto path keeps it
        assert missing == []


# --- F4: exponent numbers must not be truncated -----------------------------

class TestF4Exponent:
    def test_num_re_matches_exponent(self):
        m = span_fill._NUM_RE.search("a=1e2")
        assert m is not None and m.group(0) == "1e2"

    def test_key_delimited_exponent_value(self):
        tool = _fn_tool("calc",
                        {"a": {"type": "number"}, "b": {"type": "number"}},
                        ["a", "b"])
        filled, missing = span_fill.fill_arguments(tool, "a=1e2 and b=2.5e1")
        assert missing == []
        assert filled["a"] == 100
        assert filled["b"] == 25

    def test_coerce_exponent(self):
        assert span_fill.coerce("1e2", "number") == 100
        assert span_fill.coerce("2.5E1", "number") == 25.0

    def test_arguments_json_strict(self):
        tool = _fn_tool("calc",
                        {"a": {"type": "number"}, "b": {"type": "number"}},
                        ["a", "b"])
        filled, _ = span_fill.fill_arguments(tool, "a=1e2 b=3e-1")
        text = span_fill.arguments_json(filled)
        parsed = json.loads(text)
        assert parsed == {"a": 100, "b": 0.3}


# --- F5: the template cache must not double-add chunks ----------------------

class _CountingDrainTemplateCache(TemplateCache):
    """TemplateCache that counts how many messages entered the Drain."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.adds = 0
        real_add = self._drain.add

        def counting_add(message):
            self.adds += 1
            return real_add(message)

        self._drain.add = counting_add


class TestF5CacheDoubleAdd:
    def _log_dump(self, n):
        return "\n".join(
            f"Jun 14 15:16:0{i % 10} combo sshd[100{i}]: "
            f"authentication failure; rhost=1.2.3.4 user user{i}"
            for i in range(n))

    def test_pipeline_adds_each_chunk_exactly_once(self, monkeypatch,
                                                   shipped_world):
        monkeypatch.setattr(
            pipeline, "_route_with_confidence",
            lambda router, text: ("log_event", "rules", 1.0))
        cache = _CountingDrainTemplateCache()
        dump = self._log_dump(6)
        records = pipeline.classify_dump(dump, None, template_cache=cache)
        assert len(records) == 6
        assert cache.adds == 6

    def test_probe_is_read_only(self):
        cache = TemplateCache()
        cache.probe("Jun 14 15:16:01 combo sshd[1000]: auth failure rhost=1")
        assert len(cache._drain.clusters) == 0
        assert len(cache) == 0

    def test_match_then_learn_still_hits(self, shipped_world):
        monkey = pytest.MonkeyPatch()
        monkey.setattr(
            pipeline, "_route_with_confidence",
            lambda router, text: ("log_event", "rules", 1.0))
        try:
            cache = TemplateCache()
            dump = (
                "Jun 14 15:16:01 combo sshd[1000]: auth failure rhost=1.2.3.4\n"
                "Jun 14 15:16:02 combo sshd[1001]: auth failure rhost=1.2.3.5\n"
            )
            first = pipeline.classify_dump(dump, None, template_cache=cache)
            assert all(o.get("type") == "log_event" for o, _ in first)
            second = pipeline.classify_dump(dump, None, template_cache=cache)
            assert [m["routed_by"] for _, m in second] == ["cache", "cache"]
        finally:
            monkey.undo()


# --- F6: continuation over-merging ------------------------------------------

class TestF6Continuation:
    def test_prose_at_line_not_merged(self):
        lines = [
            "2024-06-01 12:00:00 INFO service started",
            "2024-06-01 12:00:01 INFO at least one connection failed",
            "2024-06-01 12:00:02 INFO done",
        ]
        grouped = pipeline._group_lines(lines)
        assert len(grouped) == 3

    def test_java_frame_still_merges(self):
        lines = [
            "2024-06-01 12:00:00 ERROR boot failed",
            "at com.foo.Bar.run(Bar.java:10)",
            "Caused by: java.lang.Exception: inner",
        ]
        grouped = pipeline._group_lines(lines)
        assert len(grouped) == 1
        assert "at com.foo.Bar.run" in grouped[0]

    def test_ellipsis_merges_only_after_a_frame(self):
        merged = pipeline._group_lines([
            "2024-06-01 12:00:00 ERROR x",
            "at com.foo.Bar.run(Bar.java:10)",
            "... 12 more",
        ])
        assert len(merged) == 1
        not_merged = pipeline._group_lines([
            "2024-06-01 12:00:00 INFO finished batch",
            "... next steps pending",
        ])
        assert len(not_merged) == 2

    def test_tab_frames_still_merge(self):
        grouped = pipeline._group_lines([
            "2024-06-01 12:00:00 ERROR x",
            "\tat com.foo.Bar.run(Bar.java:10)",
        ])
        assert len(grouped) == 1

    def test_plain_multiline_log_dump_still_groups_correctly(self):
        dump = "\n".join(
            f"2024-06-01 12:00:0{i % 10} INFO worker{i} processed batch {i}"
            for i in range(10))
        assert pipeline._looks_like_log_dump(dump)
        grouped = pipeline._group_lines(dump.splitlines())
        assert len(grouped) == 10


# --- F7: trailing exception line dropped ------------------------------------

class TestF7TracebackTail:
    def test_trailing_exception_line_joins_traceback(self):
        lines = [
            "Traceback (most recent call last):",
            '  File "x.py", line 1, in <module>',
            "    boom()",
            "ValueError: bad",
        ]
        grouped = pipeline._group_lines(lines)
        assert len(grouped) == 1
        assert "ValueError: bad" in grouped[0]

    def test_during_handling_line_joins(self):
        lines = [
            "Traceback (most recent call last):",
            '  File "x.py", line 1, in <module>',
            "ValueError: bad",
            "During handling of the above exception, another occurred:",
        ]
        grouped = pipeline._group_lines(lines)
        assert len(grouped) == 1

    def test_standalone_exception_line_is_kept(self):
        grouped = pipeline._group_lines(["ValueError: bad"])
        assert len(grouped) == 1

    def test_trailing_exception_survives_classify_dump(self, monkeypatch,
                                                       shipped_world):
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard", lambda: None)
        dump = (
            "2024-06-01 12:00:00 ERROR job crashed\n"
            '  File "worker.py", line 9, in run\n'
            "    raise ValueError('bad')\n"
            "ValueError: bad\n"
        )
        records = pipeline.classify_dump(dump, None)
        # The tail is merged into the traceback chunk and emitted with it,
        # never dropped as chatter.
        assert len(records) == 1
        assert records[0][1]["valid"] is True
        assert "ValueError" in json.dumps(records[0][0])


# --- F8: timestamp line dialects --------------------------------------------

class TestF8TimestampLine:
    def test_iso_offset_match(self):
        line = "2024-06-01T12:00:00+00:00 INFO service started"
        assert pipeline._LOG_TS_LINE.search(line)

    def test_existing_dialects_still_match(self):
        for line in (
            "2024-06-01 12:00:00,963 INFO x",
            "2024-06-01 12:00:00, Info x",
            "2024-06-01T12:00:00.123+00:00 INFO x",
            "2024-06-01T12:00:00Z INFO x",
            "[Sat Jun 14 15:16:01 2025] x",
            "[1.2 15:16:01] x",
        ):
            assert pipeline._LOG_TS_LINE.search(line), line

    def test_bank_csv_still_rejected(self):
        assert not pipeline._LOG_TS_LINE.search(
            "12:30:00,-10.50,debit,acc1")

    def test_rfc3164_match(self):
        line = "Jun 14 15:16:01 combo sshd[1000]: Accepted password for root"
        assert pipeline._LOG_TS_LINE.search(line)

    def test_offset_dump_is_a_log_dump(self):
        dump = "\n".join(
            f"2024-06-01T12:00:0{i}+00:00 INFO worker did thing {i}"
            for i in range(10))
        assert pipeline._looks_like_log_dump(dump)


# --- F9: load_active_router must not silently ignore root -------------------

class TestF9LoadActiveRouterRoot:
    def test_no_silent_root_parameter(self):
        params = inspect.signature(pipeline.load_active_router).parameters
        assert "root" not in params

    def test_load_active_router_callable(self):
        router = pipeline.load_active_router()
        assert router is not None


# --- F10: _route_full in_domain consistency ---------------------------------

class _OODRouter:
    """Energy gate says out-of-domain; prediction is not an active schema."""

    in_domain_energy = 0.0
    classes = ["bogus_class"]

    def predict_proba_with_energy(self, texts):
        return [[1.0]], [-100.0]

    def predict_proba(self, texts):
        return [[1.0] for _ in texts]

    def predict(self, texts):
        return ["bogus_class" for _ in texts]


class TestF10InDomainConsistency:
    def test_rules_fallback_carries_gate_verdict(self, monkeypatch):
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard", lambda: None)
        schema, routed_by, conf, in_domain = pipeline._route_full(
            _OODRouter(), "plain words only")
        assert schema == rules.route("plain words only")
        assert in_domain is False


# --- F11: extractor cache and function-field guard --------------------------

class TestF11Misc:
    def test_none_loader_result_not_cached_forever(self):
        pipeline._DEFAULT_EXTRACTORS.pop("fix-f11", None)
        calls = []

        def loader():
            calls.append(1)
            return None if len(calls) == 1 else {"ok": True}

        first = pipeline._get_default_extractor("fix-f11", loader)
        assert first is None
        second = pipeline._get_default_extractor("fix-f11", loader)
        assert second == {"ok": True}
        assert len(calls) == 2

    def test_non_dict_function_field_does_not_crash(self, monkeypatch,
                                                    shipped_world):
        tool = {"name": "", "function": "not-a-dict"}
        text = _catalogue([tool], "use the tool now please")
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard",
            lambda: _FakeGuard(tool, 0.9))
        obj, meta = pipeline._classify_known_schema(text, "tool_call", "model")
        assert obj  # a record is emitted, never the emergency fallback
        assert meta["valid"] is True
