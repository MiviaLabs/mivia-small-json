"""Guard-fix wave regression tests (2026-10-10, five re-runner-confirmed bugs).

Written RED-first against the confirmed defects, then turned green:

1. The opt-in BM25 prune read one shared digit ("25") as lexical overlap,
   so the zero-overlap skip never fired and an arbitrary survivor set
   discarded the true tool for a Japanese request.
2. ``PruneConfig(k=float("inf"))`` escaped validation and exploded with
   ``OverflowError`` inside ``_prune_candidates`` (``int()`` of a
   non-finite), bypassing the fallback except.
3. The fillability veto's reason was discarded on the veto-abstain path:
   the clarification prompt said "none of the available tools match as
   stated" even while ``missing_parameters`` listed the actual gaps.
4. (confirm-or-refute) a blank candidate key matched ``\\b`` + "" + ``\\b``
   and could steal a tie inside ``SELECT_MARGIN_FLOOR``.
5. (confirm-or-refute) ``load_matcher(..., tier2=...)`` dropped the prune
   config on the cascade object that actually scans the catalogue.

Deterministic only: stub matchers instead of artifact scores, no timing.
"""

from __future__ import annotations

import json

import pytest

from mivia_small_json import bootstrap, pipeline, tool_match
from mivia_small_json.tool_match import (
    BaseToolMatcher,
    PruneConfig,
    SELECT_MARGIN_FLOOR,
    _prune_candidates,
)


def _catalogue_text(tools: list[dict], request: str) -> str:
    return ("Tools available: " + json.dumps(tools, ensure_ascii=False)
            + "\n\nUser request: " + request)


# ---------------------------------------------------------------------------
# Bug 1 — a shared digit must not read as lexical overlap in the prune skip
# ---------------------------------------------------------------------------

# The only ASCII token under `[a-z0-9]+` is the digit "25" (asserted below).
JP_DEGREE_REQUEST = "東京の天気を25度で教えて"
JP_PLAIN_REQUEST = "東京の天気を教えて"


def _bug1_catalogue() -> list[dict]:
    """27 tools; the true tool sits beyond the k=25 arbitrary-survivor set.

    ``search_docs`` (index 2) is the ONLY span whose rendered text carries
    the token "25"; ``get_weather`` (index 26) shares no token with the
    request — under the bug the digit counts as overlap, BM25 ranks
    search_docs (and ties) into the survivor set, and get_weather is never
    scored.
    """
    tools = []
    for i in range(27):
        if i == 2:
            tools.append({
                "name": "search_docs",
                "description": "Search indexed documents; returns at most "
                               "25 results per query.",
            })
        elif i == 26:
            tools.append({
                "name": "get_weather",
                "description": "Get the current weather forecast for a city.",
            })
        else:
            code = f"{chr(97 + i // 26)}{chr(97 + i % 26)}"
            tools.append({
                "name": f"filler_{code}",
                "description": "Generic filler operation for the catalogue.",
            })
    return tools


def test_bug1_premise_digit_is_the_only_query_token():
    assert tool_match._bm25_tokenize(JP_DEGREE_REQUEST) == ["25"]
    assert tool_match._bm25_tokenize(JP_PLAIN_REQUEST) == []
    tools = _bug1_catalogue()
    vocab = set()
    for tool in tools:
        vocab.update(tool_match._bm25_tokenize(tool_match.render_tool(tool)))
    assert "25" in vocab, "fixture: search_docs must carry the digit"
    assert not ({"weather", "get", "city", "current", "forecast"}
                & set(tool_match._bm25_tokenize(JP_DEGREE_REQUEST)))


def test_bug1_digit_only_token_takes_the_zero_overlap_skip():
    """The exact re-runner request must yield full-catalogue scoring: the
    prune returns None so the true tool (index 26, beyond k=25) is scored."""
    cfg = PruneConfig(enabled=True, k=25, backend="bm25")
    assert _prune_candidates(JP_DEGREE_REQUEST, _bug1_catalogue(), cfg) is None


def test_bug1_request_without_the_digit_still_skips():
    """The previously-correct sibling behavior stays: no ASCII token at all
    means no lexical overlap, full catalogue."""
    cfg = PruneConfig(enabled=True, k=25, backend="bm25")
    assert _prune_candidates(JP_PLAIN_REQUEST, _bug1_catalogue(), cfg) is None


def test_bug1_single_character_token_does_not_defeat_the_skip():
    """A lone "x" sits in almost any English span vocabulary; it must not
    count as lexical overlap either."""
    cfg = PruneConfig(enabled=True, k=1, backend="bm25")
    tools = [{"name": "x_ray", "description": "Inspect x items in detail."}]
    assert tool_match._bm25_tokenize("xについて教えて") == ["x"]
    assert _prune_candidates("xについて教えて", tools, cfg) is None


def test_bug1_real_overlap_still_prunes():
    """Control: a genuine multi-character lexical token still narrows."""
    cfg = PruneConfig(enabled=True, k=1, backend="bm25")
    tools = _bug1_catalogue()
    survivors = _prune_candidates("delete the production database", tools, cfg)
    assert survivors is not None, (
        "an overlapping latin request must still prune (skip stays narrow)")


class _ScanningStub(BaseToolMatcher):
    """Scores rendered spans from a table; records what it was asked."""

    def __init__(self, table: dict[str, float]):
        self.table = table
        self.scored: list[str] = []

    def _score(self, request: str, tool: str) -> float:
        self.scored.append(tool)
        return self.table.get(tool, 0.0)

    def _score_raw(self, request: str, tool: str) -> float:
        return self.table.get(tool, 0.0)


def test_bug1_decision_scores_the_true_tool_end_to_end():
    """Decision-level: with the bug the gate never sees get_weather's span;
    the fix must scan the full catalogue and find it."""
    tools = _bug1_catalogue()
    table = {tool_match.render_tool(t): (0.9 if t["name"] == "get_weather"
                                         else 0.1) for t in tools}
    matcher = _ScanningStub(table)
    matcher.prune = PruneConfig(enabled=True, k=25, backend="bm25")
    best, score = matcher.best_match(_catalogue_text(tools, JP_DEGREE_REQUEST))
    weather_span = tool_match.render_tool(tools[26])
    assert weather_span in matcher.scored, (
        "the pruned-away true tool must be scored (full-catalogue fallback)")
    assert best is not None and best["name"] == "get_weather"
    assert score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Bug 2 — PruneConfig validates k: non-finite or < 1 is a clear ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_k", [float("inf"), float("nan"), 0, -3])
def test_bug2_prune_config_rejects_bad_k(bad_k):
    with pytest.raises(ValueError, match="k"):
        PruneConfig(enabled=True, k=bad_k)


def test_bug2_prune_config_accepts_positive_integers():
    assert PruneConfig(enabled=True, k=1).k == 1
    assert PruneConfig(k=25).k == 25
    assert PruneConfig().k == tool_match.DEFAULT_PRUNE_K


def test_bug2_prune_candidates_never_explodes_on_a_duck_typed_cfg():
    """A library caller can bypass the dataclass; the seam must still
    degrade to the default budget instead of raising OverflowError."""

    class _DuckCfg:
        enabled = True
        k = float("inf")
        backend = "bm25"

    tools = _bug1_catalogue()
    survivors = _prune_candidates("delete the production database", tools,
                                  _DuckCfg())
    assert survivors is not None and len(survivors) <= len(tools)


# ---------------------------------------------------------------------------
# Bug 3 — the veto's reason reaches the clarification text
# ---------------------------------------------------------------------------

MATH_ADD = {
    "name": "math.add", "description": "Add two numbers",
    "parameters": {"type": "dict", "properties": {
        "a": {"type": "number", "description": "first addend"},
        "b": {"type": "number", "description": "second addend"}},
        "required": ["a", "b"]}}
UPLOAD_FILE = {
    "name": "upload_file",
    "description": "Upload a file from a url to cloud storage.",
    "parameters": {"properties": {
        "url": {"type": "string",
                "description": "The url of the file to upload"}},
        "required": ["url"]}}
DELETE_DATABASE = {
    "name": "delete_database",
    "description": "Delete the named database.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string",
                                "description": "the database name"}},
        "required": ["name"],
    },
}


class _FixedScoreGuard:
    """A guard stub served through the real pipeline singleton path.

    best_match always returns (tool, 0.0): below THRESHOLD with the argmax
    tool kept — exactly the veto-abstain construction the pipeline must
    clarify on. select/verify stay inert so no custom-schema ladder fires.
    """

    def __init__(self, tool):
        self.tool = tool

    def best_match(self, text):
        return self.tool, 0.0

    def select(self, text, candidates, *args, **kwargs):
        return "", 0.0

    def verify(self, text, hypothesis, threshold=0.5):
        return False, 0.0


@pytest.fixture()
def registered_schemas():
    bootstrap.register_all_schemas()
    yield


def _classify_tool_call(text: str, tool: dict) -> dict:
    from mivia_small_json import registry

    real = tool_match._get_guard
    tool_match._get_guard = lambda: _FixedScoreGuard(tool)
    try:
        obj, _meta = pipeline._classify_known_schema(
            text, "tool_call", routed_by="guardfix-wave")
    finally:
        tool_match._get_guard = real
    assert "tool_clarification" in registry.active()
    return obj


def test_bug3_underspecified_veto_names_the_missing_parameters(
        registered_schemas):
    """'sum' against math.add: the veto says underspecified_query with
    missing [a, b]; the clarification must carry THAT reason and list the
    parameters — not 'none of the available tools match as stated'."""
    obj = _classify_tool_call(_catalogue_text([MATH_ADD], "sum"), MATH_ADD)
    assert obj.get("type") == "tool_clarification", obj
    assert obj["tool"] == "math.add", obj
    assert obj["reason"] == "underspecified_query", obj
    assert obj["missing_parameters"] == ["a", "b"], obj
    prompt = obj["clarification_prompt"]
    assert "a, b" in prompt, obj
    assert "math.add" in prompt, obj
    assert "none of the available tools match" not in prompt, obj


def test_bug3_missing_argument_veto_names_the_parameter(registered_schemas):
    """'upload report.pdf' against upload_file: missing_arguments with
    missing [url]; the prompt must ask for the url of upload_file."""
    obj = _classify_tool_call(_catalogue_text([UPLOAD_FILE],
                                              "upload report.pdf"),
                              UPLOAD_FILE)
    assert obj.get("type") == "tool_clarification", obj
    assert obj["reason"] == "missing_arguments", obj
    assert obj["missing_parameters"] == ["url"], obj
    prompt = obj["clarification_prompt"]
    assert "url" in prompt, obj
    assert "upload_file" in prompt, obj
    assert "none of the available tools match" not in prompt, obj


def test_bug3_ask_followup_without_veto_keeps_the_generic_prompt(
        registered_schemas):
    """Control: a question no listed tool serves (veto passes, ask-followup
    fires) keeps the generic wording — the fix must not touch that path."""
    obj = _classify_tool_call(
        _catalogue_text([DELETE_DATABASE], "what time is it?"),
        DELETE_DATABASE)
    assert obj.get("type") == "tool_clarification", obj
    assert obj["reason"] == "underspecified_query", obj
    assert obj["missing_parameters"] == ["name"], obj
    assert "none of the available tools match" in obj["clarification_prompt"]


# ---------------------------------------------------------------------------
# Bug 4 — a blank candidate key must not steal the SELECT_MARGIN_FLOOR tie
# ---------------------------------------------------------------------------

class _TableStub(BaseToolMatcher):
    """Raw-score table keyed by the rendered candidate description."""

    def __init__(self, table: dict[str, float]):
        self.table = table

    def _score(self, request: str, tool: str) -> float:
        return self.table.get(tool, 0.0)

    def _score_raw(self, request: str, tool: str) -> float:
        return self.table.get(tool, 0.0)


def test_bug4_empty_key_cannot_steal_the_margin_tie():
    """select() tie-break: re.search(r'\\b\\b', text) matches every text,
    so a within-margin EMPTY key used to be preferred over the argmax."""
    matcher = _TableStub({
        "real: beta description": 0.501,   # the honest argmax
        ": alpha description": 0.495,      # within SELECT_MARGIN_FLOOR
    })
    key, conf = matcher.select(
        "beta things", {"": "alpha description", "real": "beta description"})
    assert key == "real", (
        "an empty candidate key must never win the margin tie-break")


def test_bug4_whitespace_key_cannot_steal_the_margin_tie():
    r"""Same law for a whitespace key: \b' '\b matches 'word word' text."""
    matcher = _TableStub({
        "real: beta description": 0.501,
        " : alpha description": 0.495,
    })
    key, _conf = matcher.select(
        "beta things", {" ": "alpha description", "real": "beta description"})
    assert key == "real"


def test_bug4_verbatim_key_tie_break_still_works():
    """Control: the measured case the tie-break exists for (a real key
    named verbatim in the text, tied with a saturated neighbor) keeps
    winning."""
    matcher = _TableStub({
        "devops: devops team tooling": 0.40,
        "other: unrelated platform work": 0.4001,
    })
    key, _conf = matcher.select(
        "devops", {"other": "unrelated platform work", "devops": "devops "
                                                                   "team "
                                                                   "tooling"})
    assert key == "devops"


def test_bug4_margin_floor_constant_untouched():
    """The tie window itself is a measured operating point — pinned so the
    fix cannot quietly retune it."""
    assert SELECT_MARGIN_FLOOR == 0.05


# ---------------------------------------------------------------------------
# Bug 5 — load_matcher(..., tier2=...) must keep the prune config reachable
# ---------------------------------------------------------------------------

def test_bug5_cascade_via_load_matcher_keeps_the_prune_config(monkeypatch):
    """Red: the cascade object that actually scans the catalogue dropped
    the config (class-default None), so an explicit tier2 silently turned
    the opt-in prune OFF."""
    cfg = PruneConfig(enabled=True, k=9)
    seen: list = []

    def fake_loader(path, tokenizer_path=None, max_len=None, span_fn=None,
                    prune=None):
        seen.append(prune)
        return object()  # a stand-in matcher; only identity is needed

    monkeypatch.setattr(tool_match, "load_single_matcher", fake_loader)
    matcher = tool_match.load_matcher(
        path="unused.onnx",
        tier2=tool_match.Tier2Spec(path="tier2.onnx"),
        prune=cfg)
    assert isinstance(matcher, tool_match.CascadedToolMatcher)
    assert matcher.prune is cfg, (
        "the prune config must survive onto the object that scores")
    # The tier-1 loader received the explicit config; the tier-2 loader
    # call takes none (the config is composition-layer, owned by the
    # cascade that scans the catalogue).
    assert seen and seen[0] is cfg


def test_bug5_split_over_cascade_also_carries_the_config(monkeypatch):
    """The shipped shape with an explicit tier2: the gate wraps the cascade
    in a split — both the split AND the underlying cascade must carry the
    config, so either object's catalogue scan honors it."""
    cfg = PruneConfig(enabled=True, k=9)

    def fake_loader(path, tokenizer_path=None, max_len=None, span_fn=None,
                    prune=None):
        return object()

    monkeypatch.setattr(tool_match, "load_single_matcher", fake_loader)
    matcher = tool_match.load_matcher(
        tier2=tool_match.Tier2Spec(path="tier2.onnx"), prune=cfg)
    assert isinstance(matcher, tool_match.SplitToolMatcher)
    assert matcher.prune is cfg
    assert isinstance(matcher.rank, tool_match.CascadedToolMatcher)
    assert matcher.rank.prune is cfg, (
        "the cascade under the split dropped the config")


def test_bug5_default_construction_stays_prune_free():
    """Direct cascade construction without a config keeps the OFF default."""
    cascade = tool_match.CascadedToolMatcher(object(), None)
    assert cascade.prune is None
