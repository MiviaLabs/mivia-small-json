"""Survivor-fix regression tests for the tool-match fillability gate.

Two execution-confirmed survivors (2026-10-10 validation round):

1. ``required: [3]`` — an int entry, a real OpenAI-tool-schema shape — raised
   AttributeError inside :func:`validate_fillability` (``'int' has no
   .lower'``). The crash escaped through the gate span
   (:func:`render_tool_fillability`) into ``SplitToolMatcher.best_match``,
   hit the pipeline emergency catch-all, and the served ``/v1/guard`` minted
   a template ``tool_call`` with name ``""``. The decision-layer veto
   (:func:`fillability_veto`) already carried the degrade-to-no-veto
   str-filter; the gate-side predicate must agree.

2. The URL heuristic matched bare filenames (``report.pdf``, ``photo.jpg``)
   as URLs, so a required ``url`` slot looked satisfied and the guard emitted
   a CALL with ``{}`` where the honest answer is to ask for the URL. Real
   url shapes (``https://...``, ``www....``, ``example.com/path``,
   ``api.service.dev``) must keep matching — over-tightening this regex once
   dropped BFCL 89.1 -> 25.1 (tests/test_bfcl_pairs.py holds the bar).
"""

import json

import pytest

from mivia_small_json import tool_match


@pytest.fixture(scope="module")
def guard():
    """The shipped guard (split matcher), loaded once; skip when absent."""
    loaded = tool_match._get_guard()
    if loaded is None:
        pytest.skip("tool-match guard artifacts missing or unverified")
    return loaded


def _weather_tool(required):
    """The v1 repro tool: real OpenAI shape, one malformed required entry."""
    return {
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
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Survivor 1: non-string entries in "required" must degrade, never raise
# ---------------------------------------------------------------------------


def test_validate_fillability_int_required_degrades_to_no_gate():
    """`required: [3]` filters to empty -> no-gate (True, '', []), no raise."""
    fillable, reason, missing = tool_match.validate_fillability(
        _weather_tool([3]), "compute a=3 b=2 c=1")
    assert (fillable, reason, missing) == (True, "", [])


def test_validate_fillability_null_required_degrades_to_no_gate():
    tool = _weather_tool([3])
    tool["function"]["parameters"]["required"] = [None, 3]
    fillable, reason, missing = tool_match.validate_fillability(
        tool, "compute a=3 b=2 c=1")
    assert (fillable, reason, missing) == (True, "", [])


def test_validate_fillability_still_checks_the_string_entries():
    """The str-filter drops garbage entries, not real checks.

    A mixed ``[3, "url"]`` must still veto on the missing url slot — the
    filter narrows to strings, it must not widen to no-gate.
    """
    tool = {
        "name": "fetch_api",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "target endpoint"}
            },
            "required": [3, "url"],
        },
    }
    fillable, reason, missing = tool_match.validate_fillability(
        tool, "check the status of that server for me please")
    assert (fillable, reason) == (False, "missing_arguments")
    assert missing == ["url"]


def test_validate_fillability_agrees_with_the_veto_on_int_required():
    """Gate-side and decision-side predicates must degrade the same way."""
    tool = _weather_tool([3])
    request = "compute a=3 b=2 c=1"
    assert tool_match.validate_fillability(tool, request) == \
        tool_match.fillability_veto(tool, request) == (True, "", [])


def test_validate_fillability_non_string_request_degrades_honestly():
    """A non-str request supplies no values: underspecified_query, no raise."""
    tool = _weather_tool(["a", "b", "c"])
    fillable, reason, missing = tool_match.validate_fillability(tool, None)
    assert (fillable, reason) == (False, "underspecified_query")
    assert missing == ["a", "b", "c"]


def test_is_veto_numeric_non_string_name_is_false():
    """The helper the traceback pointed at: a non-str name is simply not a
    numeric hint — it must return False, never raise."""
    assert tool_match._is_veto_numeric(3, {}) is False
    assert tool_match._is_veto_numeric(None, {}) is False


def test_render_tool_fillability_int_required_returns_base_span():
    """The gate span for a fully-malformed required list is the plain span."""
    span = tool_match.render_tool_fillability(
        _weather_tool([3]), "compute a=3 b=2 c=1")
    assert span == "get_weather: Get current weather for a city."


def test_render_tool_fillability_mixed_required_renders_only_strings():
    tool = {
        "name": "fetch_api",
        "description": "Fetch.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "target endpoint"}
            },
            "required": [3, "url"],
        },
    }
    span = tool_match.render_tool_fillability(tool, "check the server")
    assert "url (string" in span
    assert "3 (" not in span


def test_guard_best_match_survives_int_required_entry(guard):
    """The crash chain that minted CALL tool:"": gate span -> best_match."""
    text = ("Tools available: " + json.dumps([_weather_tool([3])])
            + "\n\nUser request: compute a=3 b=2 c=1")
    best, score = guard.best_match(text)  # raised AttributeError before the fix
    assert best is None or best.get("function", best).get("name") == "get_weather"
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Survivor 2: a bare filename is not a URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", [
    "report.pdf", "manual.doc", "summary.docx", "photo.jpg", "icon.gif",
    "notes.txt", "export.csv", "readme.md", "archive.zip", "backup.tar",
    "dump.gz", "script.py", "notebook.ipynb", "config.json", "schema.yaml",
    "deploy.yml", "page.xml", "index.html",
])
def test_url_pattern_ignores_bare_document_filenames(filename):
    assert tool_match._URL_PATTERN.search(filename) is None, filename
    assert tool_match._URL_PATTERN.search("upload " + filename) is None, \
        filename


@pytest.mark.parametrize("url", [
    "https://example.com/api",
    "http://x.test",
    "https://cdn.example.com/report.pdf",
    "ftp://files.example.org/pub",
    "www.example.com",
    "example.com/path",
    "example.com/report.pdf",
    "api.service.dev",
    "example.com",
    "fetch https://a.io/f then summarize",
])
def test_url_pattern_still_matches_real_urls(url):
    assert tool_match._URL_PATTERN.search(url) is not None, url


def test_validate_fillability_filename_is_not_a_url():
    """The v11 repro: 'upload report.pdf' must ask for the URL, not call."""
    tool = {
        "type": "function",
        "function": {
            "name": "upload_file",
            "description": "Upload a file from a url to storage.",
            "parameters": {
                "properties": {
                    "url": {"type": "string",
                            "description": "The url of the file to upload"},
                },
                "required": ["url"],
            },
        },
    }
    assert tool_match.validate_fillability(tool, "upload report.pdf") == \
        (False, "missing_arguments", ["url"])
    # Real url shapes keep the slot satisfied (BFCL retention).
    assert tool_match.validate_fillability(
        tool, "upload from https://x.test/a.pdf")[0] is True
    assert tool_match.validate_fillability(
        tool, "grab example.com/report.pdf and send it over")[0] is True
    assert tool_match.validate_fillability(
        tool, "pull api.service.dev and deploy it today please")[0] is True


def test_fillability_veto_filename_is_not_a_url():
    """The decision layer reads the same pattern: the pair must veto."""
    tool = {
        "type": "function",
        "function": {
            "name": "upload_file",
            "description": "Upload a file from a url to storage.",
            "parameters": {
                "properties": {
                    "url": {"type": "string",
                            "description": "The url of the file to upload"},
                },
                "required": ["url"],
            },
        },
    }
    assert tool_match.fillability_veto(tool, "upload report.pdf") == \
        (False, "missing_arguments", ["url"])
    assert tool_match.fillability_veto(
        tool, "upload from https://x.test/a.pdf") == (True, "", [])


def test_guard_abstains_on_filename_only_url_request(guard):
    """End to end over the split matcher: the vetoed pair scores 0.0.

    Before the fix the guard scored this pair 0.999 and the pipeline minted
    a CALL with empty arguments.
    """
    catalogue = json.dumps([
        {"type": "function", "function": {
            "name": "upload_file",
            "description": "Upload a file from a url to cloud storage.",
            "parameters": {"properties": {
                "url": {"type": "string",
                        "description": "The url of the file to upload"}},
                "required": ["url"]}}},
        {"type": "function", "function": {
            "name": "list_files",
            "description": "List the files currently stored.",
            "parameters": {"properties": {}, "required": []}}},
    ])
    best, score = guard.best_match(
        catalogue + "\n\nUser request: upload report.pdf")
    assert best is None or score == 0.0, (best, score)
    # Control: a real url keeps the call (retention side of the veto).
    _best, control = guard.best_match(
        catalogue + "\n\nUser request: upload the file at "
                    "https://cdn.example.com/report.pdf")
    assert control > 0.0
