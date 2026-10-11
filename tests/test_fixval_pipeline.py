"""Fixval 2026-10-10: pipeline-side fixes for the confirmed review bugs.

One test group per bug this agent owns in
reports/code-review-validation-2026-10-10.md:

- A1 hardening (pipeline side): a ``tool_call``-routed text where the
  guard's ``best_match`` returns ``None`` must never fall through to
  ``rules.extract`` (which mints a tool name from catalogue JSON).
- A3: dump grouping must not drop or merge real lines (trailing ``...``,
  IP-shaped ``at`` lines, bare ``Exception:`` tails).
- N1: politeness openers (please/tell/show/help) must not turn no-tool
  imperative commands into clarifications.
- N2: ``classify_dump`` must keep router-confident chunks for ANY schema,
  not only ``log_event``.
- N5 (narrow): an atomic whole-text route carrying strong atomic markers
  keeps the atomic take regardless of the log-line ratio.

TDD: this file was written red against the buggy tree, then pipeline.py
was fixed. Deterministic: the guard is faked through
``tool_match._get_guard`` unless the shipped guard's score is itself the
fixture precondition (asserted, never assumed).
"""

import json

import pytest

from mivia_small_json import bootstrap, pipeline, tool_match as T
from mivia_small_json.validate import validate

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": ("Retrieves the weather forecast for a specified "
                        "city on a given date."),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string",
                         "description": "The name of the city."},
                "date": {"type": "string", "description": "The date."},
            },
            "required": ["city", "date"],
        },
    },
}

DROP_DB_TOOL = {
    "type": "function",
    "function": {
        "name": "drop_database",
        "description": "Drop the whole database. Destructive.",
        "parameters": {"type": "object", "properties": {},
                       "required": []},
    },
}

# Forced tool_call route: mirrors production, where the encoder routes a
# catalogue text to tool_call and the guard decides call/clarify/abstain
# (same construction as tests/test_clarification_routing.py).
CALL_ROUTER = lambda text: "tool_call"  # noqa: E731


def _catalogue(tools, request):
    return ("Tools available:\n" + json.dumps(tools)
            + "\n\nUser request: " + request)


class _FakeGuard:
    """best_match returns the configured (tool, score) verbatim."""

    def __init__(self, tool, score):
        self._tool = tool
        self._score = score

    def best_match(self, text):
        return self._tool, self._score


@pytest.fixture
def shipped_world():
    bootstrap.register_all_schemas()
    yield


# --- Bug 1 (A1 hardening): best_tool None must never mint a tool name -------

class TestA1BlankRequestNeverCalls:
    def test_blank_request_catalogue_abstains(self, shipped_world):
        """The A1 repro: wellformed catalogue, EMPTY request side.

        The outcome pin is branch-agnostic BY DESIGN (defense in depth):
        whatever best_match returns for the blank request — the guard's
        own (None, 0.0) abstention, or any (None, >= THRESHOLD) shape —
        the emitted record is no_tool_call and never a minted
        `drop_database` call.
        """
        text = _catalogue([DROP_DB_TOOL], "")
        best, _score = T._get_guard().best_match(text)
        assert best is None, "fixture: blank request must not match a tool"
        obj, meta = pipeline.classify(text, router=CALL_ROUTER)
        assert obj.get("type") == "no_tool_call"
        assert meta["valid"] is True

    def test_unparsed_catalogue_with_plain_text_abstains(self, shipped_world):
        """Not-a-catalogue short-circuit (None, 1.0) with a non-ask
        request: still no_tool_call, never a minted call."""
        text = ("Tools available:\n[ this is not json\n\n"
                "User request: just delete the database for me")
        best, score = T._get_guard().best_match(text)
        assert best is None and score >= T.THRESHOLD
        obj, _ = pipeline.classify(text, router=CALL_ROUTER)
        assert obj.get("type") == "no_tool_call"

    def test_genuine_ask_on_unparsed_catalogue_asks(self, shipped_world):
        """Defense in depth, ask limb: best_tool None + a genuine
        question -> tool_clarification (when registered), not a call."""
        question = "Can you find me a movie that features Timothy Bateson?"
        text = "Tools available:\n[ this is not json\n\nUser request: " + question
        best, score = T._get_guard().best_match(text)
        assert best is None and score >= T.THRESHOLD
        obj, meta = pipeline.classify(text, router=CALL_ROUTER)
        assert obj.get("type") == "tool_clarification"
        assert meta["valid"] is True
        assert validate(obj, "tool_clarification") == []
        assert obj["reason"] == "underspecified_query"

    def test_catalogue_name_never_minted_regardless_of_guard(
            self, monkeypatch, shipped_world):
        """Defense in depth: WHATEVER best_match returns — here
        (None, 0.95), a shape no current caller produces — a tool_call
        route with best_tool None never emits a catalogue name."""
        text = _catalogue([DROP_DB_TOOL], "hi")
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard",
            lambda: _FakeGuard(None, 0.95))
        obj, meta = pipeline._classify_known_schema(text, "tool_call", "model")
        assert obj.get("type") != "tool_call"
        assert obj.get("type") == "no_tool_call"
        assert meta["valid"] is True

    def test_fake_guard_none_with_ask_request_asks(
            self, monkeypatch, shipped_world):
        text = _catalogue([WEATHER_TOOL], "Can you find me a movie?")
        monkeypatch.setattr(
            "mivia_small_json.tool_match._get_guard",
            lambda: _FakeGuard(None, 0.95))
        obj, _ = pipeline._classify_known_schema(text, "tool_call", "model")
        assert obj.get("type") == "tool_clarification"
        assert obj["tool"] == ""
        assert validate(obj, "tool_clarification") == []

    def test_available_tools_label_first_variant_never_mints(
            self, shipped_world):
        """The label-first catalogue header ("Available tools:") with a
        single-newline separator parses as NOT-a-catalogue (no blank-line
        "User request:" separator) and scores the 1.0 pass-through —
        review-fixval 2026-10-10: it used to fall through to
        rules.extract and mint the first catalogue name with {} (real
        router: tool_call @ 0.9989; real guard: (None, 1.0))."""
        text = ("Available tools: " + json.dumps([DROP_DB_TOOL])
                + "\nUser request: delete the database")
        best, score = T._get_guard().best_match(text)
        assert best is None and score >= T.THRESHOLD, (best, score)
        obj, meta = pipeline.classify(text, router=CALL_ROUTER)
        assert obj.get("type") != "tool_call", obj
        assert meta["valid"] is True

    def test_available_tools_label_user_prompt_variant_never_mints(
            self, shipped_world):
        """Same label-first hole under the "User prompt:" label."""
        text = ("Available tools: " + json.dumps([DROP_DB_TOOL])
                + "\nUser prompt: delete the database")
        obj, meta = pipeline.classify(text, router=CALL_ROUTER)
        assert obj.get("type") != "tool_call", obj
        assert meta["valid"] is True


# --- Bug 2 (A3): dump grouping must not drop or merge real lines -------------

class TestA3GroupingRegexes:
    def test_trailing_ellipsis_is_not_traceback_structure(self):
        # (a) only a line BEGINNING with "..." is traceback structure.
        assert pipeline._TRACEBACK_MARK.search("INFO task finished...") is None
        assert pipeline._TRACEBACK_MARK.search(
            "ERROR boot failed\n... 12 more") is not None

    def test_anchored_bare_ellipsis_still_a_mark(self):
        assert pipeline._TRACEBACK_MARK.search("... 12 more") is not None

    def test_frame_line_requires_a_letter_in_the_token(self):
        # (b) "at 10.0.0.1 ..." is prose, "at com.foo.Bar.run(...)" a frame.
        assert pipeline._FRAME_LINE.match("at 10.0.0.1 connection refused") \
            is None
        assert pipeline._FRAME_LINE.match("at 192.168.0.1:8080/hello") is None
        assert pipeline._FRAME_LINE.match("at com.foo.Bar.run(Bar.java:10)") \
            is not None
        assert pipeline._FRAME_LINE.match("at org.apache.Foo.<init>(Foo.java:1)") \
            is not None

    def test_exception_tail_allows_the_bare_form(self):
        # (c) a bare "Exception: bad" tail line is traceback structure too.
        assert pipeline._EXCEPTION_TAIL.match("Exception: bad") is not None
        assert pipeline._EXCEPTION_TAIL.match("Error: disk on fire") \
            is not None
        # non-tails stay non-tails
        assert pipeline._EXCEPTION_TAIL.match("Errorbar: not a tail") is None
        assert pipeline._EXCEPTION_TAIL.match("ValueError: bad") is not None

    def test_info_ellipsis_and_connection_error_keep_both_lines(self):
        """Report repro 1: the trailing-ellipsis INFO line must not
        swallow the ConnectionError line."""
        grouped = pipeline._group_lines([
            "INFO task finished...",
            "ConnectionError: timeout talking to db",
        ])
        assert len(grouped) == 2
        assert "task finished" in grouped[0]
        assert "ConnectionError: timeout talking to db" in grouped[1]

    def test_bare_exception_tail_stays_grouped(self):
        """Report repro 2: a traceback ending in a bare `Exception: bad`
        keeps the tail grouped (it used to be dropped entirely)."""
        grouped = pipeline._group_lines([
            "Traceback (most recent call last):",
            '  File "x.py", line 1, in <module>',
            "    boom()",
            "Exception: bad",
        ])
        assert len(grouped) == 1
        assert "Exception: bad" in grouped[0]

    def test_ip_at_line_keeps_server_started(self):
        """Report repro 3: an IP-shaped `at` line is not a Java frame."""
        grouped = pipeline._group_lines([
            "INFO server started",
            "at 10.0.0.1 connection refused",
        ])
        assert len(grouped) == 2
        assert "server started" in grouped[0]
        assert "at 10.0.0.1 connection refused" in grouped[1]

    def test_real_frames_still_group_after_the_fix(self):
        grouped = pipeline._group_lines([
            "2024-06-01 12:00:00 ERROR boot failed",
            "at com.foo.Bar.run(Bar.java:10)",
            "Caused by: java.lang.Exception: inner",
            "... 12 more",
        ])
        assert len(grouped) == 1


class _StaticRouter:
    """_route_with_confidence stand-in: everything routes log_event."""

    def __call__(self, router, text):
        return ("log_event", "rules", 1.0)


class TestA3EndToEnd:
    @pytest.fixture(autouse=True)
    def _static_routing(self, monkeypatch, shipped_world):
        monkeypatch.setattr(pipeline, "_route_with_confidence",
                            _StaticRouter())
        yield

    def _records(self, dump):
        return pipeline.classify_dump(dump, None)

    def test_info_ellipsis_paste_keeps_task_finished(self):
        records = self._records(
            "INFO task finished...\n"
            "ConnectionError: timeout talking to db\n")
        assert len(records) == 2
        assert "task finished" in json.dumps(records)
        assert "timeout talking to db" in json.dumps(records)

    def test_bare_exception_traceback_keeps_tail(self):
        records = self._records(
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in <module>\n'
            "    boom()\n"
            "Exception: bad\n")
        # the tail is grouped with the traceback (one chunk, one record);
        # the bare exception survives the extraction as the component —
        # the same standard tests/test_fix_pipeline_guard.py F7 applies
        # to the "ValueError: bad" tail.
        assert len(records) == 1
        assert "Exception" in json.dumps(records)

    def test_ip_paste_keeps_server_started(self):
        records = self._records(
            "INFO server started\n"
            "at 10.0.0.1 connection refused\n")
        assert len(records) == 2
        assert "server started" in json.dumps(records)


# --- Bug 3 (N1): politeness openers must not clarify imperative commands -----

class TestN1PolitenessOpeners:
    @pytest.mark.parametrize(("text", "expected"), [
        ("please book a table for two at an italian restaurant tonight",
         False),
        ("tell me a joke about databases", False),
        ("show me your favourite pasta recipe", False),
        ("help me write a haiku about databases", False),
        # genuine question forms stay covered
        ("Can you find me a movie that features Timothy Bateson?", True),
        ("What is the weather Wednesday?", True),
        ("could you help me find a therapist?", True),
        ("Show me the list of projects?", True),  # trailing '?'
        ("How do I serve this?", True),
        ("Execute irreversible deletion immediately.", False),
        ("", False),
    ])
    def test_ask_followup_shape(self, text, expected):
        assert pipeline._is_ask_followup(text) is expected

    def test_please_command_abstains_like_the_bare_command(self,
                                                           shipped_world):
        """End-to-end N1 repro: the polite form must take the same path
        as the bare imperative — a silent abstention, not a CLARIFY that
        asks for an unmatched tool's parameters."""
        request = ("please book a table for two at an italian "
                   "restaurant tonight")
        text = _catalogue([WEATHER_TOOL], request)
        guard = T._get_guard()
        best, score = guard.best_match(text)
        assert score < T.THRESHOLD, "fixture must guard-abstain"
        veto_ok, _reason, _missing = T.fillability_veto(best, request)
        assert veto_ok, (
            "fixture must isolate the opener (a veto-driven clarification "
            "would not test N1)")
        obj, meta = pipeline.classify(text, router=CALL_ROUTER)
        assert obj.get("type") == "no_tool_call"
        assert meta["valid"] is True


# --- Bug 4 (N2): router-confident chunks are content for ANY schema ----------

def _routing_for(chunk_routes):
    """Build a _route_with_confidence stand-in from a predicate."""

    def _route(router, text):
        for marker, route in chunk_routes:
            if marker(text):
                return route
        return ("text", "model", 0.99)

    return _route


class TestN2RouterConfidentChunk:
    @pytest.fixture(autouse=True)
    def _shipped(self, shipped_world):
        yield

    def test_router_confident_issue_chunk_is_kept(self, monkeypatch):
        """The N2 repro: a non-English line the router confidently routes
        to `issue` with zero rules markers must survive the dump."""
        german = "Der Server stuerzt beim Start ab"
        routes = [(lambda t, _g=german: t == _g,
                   ("issue", "model", 0.981))]
        monkeypatch.setattr(pipeline, "_route_with_confidence",
                            _routing_for(routes))
        dump = "\n".join([german, "lol", "haha nice", "ok cool"])
        records = pipeline.classify_dump(dump, None)
        types = [obj.get("type") for obj, _ in records]
        assert types == ["issue"], types

    def test_router_text_no_marker_chunk_still_dropped(self, monkeypatch):
        """The actual chatter case keeps today's behavior: router-`text`
        with no rules marker is dropped (the dump falls back to one
        whole-text record)."""
        monkeypatch.setattr(pipeline, "_route_with_confidence",
                            _routing_for([]))
        records = pipeline.classify_dump("lol\nhaha nice\nok cool\n", None)
        assert len(records) == 1
        assert records[0][0].get("type") == "text"

    def test_router_text_with_rules_marker_is_kept(self, monkeypatch):
        """A chunk the router calls `text` but that carries a real log
        marker is still content (the rules OR-limb of the gate) — it is
        emitted under its routed schema, here `text`."""
        line = "Jun 14 15:16:01 combo sshd[1000]: authentication failure"
        routes = [(lambda t, _l=line: t == _l, ("text", "model", 0.99))]
        monkeypatch.setattr(pipeline, "_route_with_confidence",
                            _routing_for(routes))
        records = pipeline.classify_dump(line + "\nlol\n", None)
        # the marker chunk survives (emitted under its routed schema,
        # `text` — the text record does not echo the raw line), the
        # markerless chatter chunk is dropped
        assert [obj.get("type") for obj, _ in records] == ["text"]


# --- Bug 5 (N5, narrow): strong atomic markers keep the atomic take ----------

_BUILD_LOG_LINES = [
    "2024-06-01 12:00:00 INFO build started",
    "2024-06-01 12:00:01 INFO compile module a",
    "2024-06-01 12:00:02 INFO compile module b",
    "2024-06-01 12:00:03 INFO link objects",
    "2024-06-01 12:00:04 INFO build finished",
]


class TestN5AtomicCarveOut:
    @pytest.fixture(autouse=True)
    def _shipped(self, shipped_world):
        yield

    @staticmethod
    def _patch_routing(monkeypatch, whole, whole_route):
        def _route(router, text):
            if text == whole:
                return whole_route
            if text.startswith("2024-"):
                return ("log_event", "structure", 1.0)
            return ("diff", "model", 0.9)

        monkeypatch.setattr(pipeline, "_route_with_confidence", _route)

    def test_unified_diff_plus_build_logs_stays_atomic(self, monkeypatch):
        """The N5 repro: a unified diff (the @@ hunk is a strong atomic
        marker) plus 5 timestamped build-log lines (ratio 5/8 >= 0.35)
        must keep the atomic whole-text take. The @@ hunk is structural
        proof of a diff, so the take is the `diff` schema (a diff-plus-log
        paste routes `diagnostic` and the diagnostic extractor fills
        NOTHING for a diff — review-fixval 2026-10-10: the take used to
        be a content-free diagnostic record) and its patch field must
        carry the payload."""
        whole = "\n".join([
            "@@ -12,6 +12,7 @@ def main():",
            "+import os",
            " import sys",
            *_BUILD_LOG_LINES,
        ])
        self._patch_routing(monkeypatch, whole,
                            ("diagnostic", "model", 0.97))
        records = pipeline.classify_dump(whole, None)
        assert len(records) == 1, [o.get("type") for o, _ in records]
        obj = records[0][0]
        assert obj.get("type") == "diff", obj.get("type")
        blob = json.dumps(obj)
        for needle in ("-12,6", "+import os", "build started"):
            assert needle in blob, needle

    def test_traceback_frame_marker_keeps_atomic(self, monkeypatch):
        whole = "\n".join([
            "Traceback (most recent call last):",
            '  File "worker.py", line 9, in run',
            "    raise ValueError('bad')",
            *_BUILD_LOG_LINES,
        ])
        self._patch_routing(monkeypatch, whole,
                            ("diagnostic", "model", 0.97))
        records = pipeline.classify_dump(whole, None)
        assert len(records) == 1, [o.get("type") for o, _ in records]

    def test_compiler_diag_marker_keeps_atomic(self, monkeypatch):
        whole = "\n".join([
            "src/main.c:12:5: error: 'foo' undeclared (first use)",
            "src/main.c:40:1: error: expected ';' before 'return'",
            *_BUILD_LOG_LINES,
        ])
        self._patch_routing(monkeypatch, whole,
                            ("diagnostic", "model", 0.97))
        records = pipeline.classify_dump(whole, None)
        assert len(records) == 1, [o.get("type") for o, _ in records]

    def test_carve_out_semantics_unchanged_without_markers(self,
                                                           monkeypatch):
        """Narrow-fix pin: WITHOUT strong atomic markers, the 0.35
        carve-out still shreds (existing semantics, deliberately not
        touched)."""
        whole = "\n".join([
            "ISS-500: System crashes on boot when the machine boots",
            *_BUILD_LOG_LINES,
        ])

        def _route(router, text):
            if text == whole:
                return ("diagnostic", "model", 0.97)
            if text.startswith("2024-"):
                return ("log_event", "structure", 1.0)
            return ("issue", "model", 0.9)

        monkeypatch.setattr(pipeline, "_route_with_confidence", _route)
        records = pipeline.classify_dump(whole, None)
        types = [obj.get("type") for obj, _ in records]
        # shredded: one issue record + 5 log_event records, no atomic take
        assert len(records) == 6, types
        assert types.count("log_event") == 5, types
        assert types.count("issue") == 1, types
