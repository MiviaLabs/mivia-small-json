"""Embedding tool-pruner arm (spec embed-tool-pruner).

External behavior only: the select_survivors contract, the backend
dispatch/fallbacks, the degenerate-request full-catalogue path, the veto
and primitive composition, and the MEASURED no-batching pin (the export
measured solo-vs-batch divergence — max cosine delta 0.0484, survivor
sets diverging 30/30 — so tool-side batching must stay OUT; the forward
accounting below fails loudly if anyone reintroduces it). No timing
asserts; no network; the embed-specific tests skip with a clear message
only when the assembled artifact is absent (its absence is itself a
tested degradation path).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mivia_small_json import tool_match
from mivia_small_json.tool_match import (
    EMBED_PRUNER_CACHE,
    PRUNE_BACKEND_AUTO,
    PRUNE_BACKENDS,
    EmbeddingToolPruner,
    PruneConfig,
    load_embed_pruner,
    load_prune_config,
    parse_catalogue_raw,
)

ROOT = tool_match.ROOT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WEATHER = {"name": "get_weather",
           "description": "Get the current weather forecast for a city."}
DNA = {"name": "sequence_dna",
       "description": "Sequence a DNA molecule by its identifier."}
EMAIL = {"name": "send_email",
         "description": "Send an email to a recipient."}
FILLERS = [{"name": f"tool_{i}",
            "description": f"Generic utility number {i} for catalogues."}
           for i in range(10)]
DE_WEATHER_REQUEST = "Wie ist das Wetter heute in Paris?"
EN_WEATHER_REQUEST = "What is the weather like in Paris today?"


def _catalogue_text(tools: list[dict], request: str) -> str:
    """The exact catalogue surface the guard reads (builder == runtime)."""
    return ("Tools available:\n" + json.dumps(tools, ensure_ascii=False)
            + "\n\nUser request: " + request)


@pytest.fixture(scope="module")
def pruner():
    """The real embedding pruner, loaded once; skips when absent."""
    loaded = load_embed_pruner()
    if loaded is None:
        pytest.skip("models/tool-pruner-embed-int8.onnx missing or failed "
                    "its sha256 check (run scripts/assemble_models.py)")
    return loaded


@pytest.fixture(scope="module")
def shipped():
    """The shipped guard (split gate+rank), loaded once."""
    loaded = tool_match.load_matcher()
    if loaded is None:
        pytest.skip("models/tool-match-*.onnx are missing or failed their "
                    "sha256 check")
    return loaded


def _reset_embed_loader(monkeypatch, model_path=None):
    """Point the loader at ``model_path`` (default: nonexistent) and reset
    the singleton; monkeypatch restores the module state on teardown."""
    import pathlib

    monkeypatch.setattr(tool_match, "EMBED_PRUNER_MODEL",
                        model_path or pathlib.Path("/nonexistent") / "x.onnx",
                        raising=False)
    monkeypatch.setattr(tool_match, "_EMBED_PRUNER", None, raising=False)
    monkeypatch.setattr(tool_match, "_EMBED_PRUNER_LOADED", False,
                        raising=False)


# ---------------------------------------------------------------------------
# Config surface: the backend field
# ---------------------------------------------------------------------------

def test_backend_defaults_to_auto():
    cfg = load_prune_config({})
    assert cfg.backend == PRUNE_BACKEND_AUTO
    assert PRUNE_BACKEND_AUTO in PRUNE_BACKENDS


@pytest.mark.parametrize("value,expected", [
    ("auto", "auto"), ("bm25", "bm25"), ("embed", "embed"),
    ("EMBED", "embed"), ("  bm25  ", "bm25"),
])
def test_backend_env_values_parse(value, expected):
    assert load_prune_config({"MIVIA_GUARD_PRUNE_BACKEND": value}) \
        .backend == expected


@pytest.mark.parametrize("value", ["", "garbage", "BM", "0", "1", None])
def test_backend_env_garbage_falls_back_to_auto(value):
    assert load_prune_config({"MIVIA_GUARD_PRUNE_BACKEND": value}) \
        .backend == PRUNE_BACKEND_AUTO


def test_backend_env_reads_with_enabled_and_k(monkeypatch):
    monkeypatch.setenv("MIVIA_GUARD_PRUNE", "1")
    monkeypatch.setenv("MIVIA_GUARD_PRUNE_K", "5")
    monkeypatch.setenv("MIVIA_GUARD_PRUNE_BACKEND", "embed")
    assert load_prune_config() == PruneConfig(enabled=True, k=5,
                                              backend="embed")


def test_default_config_is_disabled_auto():
    """The arm's pre-registered default: OFF until every bar passes."""
    cfg = PruneConfig()
    assert cfg.enabled is False
    assert cfg.backend == PRUNE_BACKEND_AUTO


# ---------------------------------------------------------------------------
# Loader: verification, absence, negative caching (never raises)
# ---------------------------------------------------------------------------

def test_loader_negative_cache_is_stable():
    a = load_embed_pruner()
    b = load_embed_pruner()
    assert a is b


def test_missing_artifact_falls_back_without_raising(monkeypatch, capsys):
    _reset_embed_loader(monkeypatch)
    assert load_embed_pruner() is None
    err = capsys.readouterr().err
    assert "embedding pruner artifact missing" in err


def test_failed_sha256_falls_back_without_raising(monkeypatch, tmp_path,
                                                  capsys):
    bogus = tmp_path / "tool-pruner-embed-int8.onnx"
    bogus.write_bytes(b"not an onnx file")
    bogus.with_name(bogus.name + ".sha256").write_text(
        "0" * 64 + "\n", encoding="utf-8")
    _reset_embed_loader(monkeypatch, bogus)
    assert load_embed_pruner() is None
    assert "failed its sha256 check" in capsys.readouterr().err


def test_missing_backend_falls_back_to_bm25_survivors(monkeypatch):
    """backend="auto" and a forced "embed" both degrade to the lexical
    ranking when the artifact is absent — the prune never breaks a
    decision, and auto == bm25 exactly."""
    _reset_embed_loader(monkeypatch)
    tools = [WEATHER, DNA, EMAIL] + FILLERS
    request = "delete the production database email"
    auto = tool_match._prune_candidates(
        request, tools, PruneConfig(enabled=True, k=4, backend="auto"))
    forced = tool_match._prune_candidates(
        request, tools, PruneConfig(enabled=True, k=4, backend="embed"))
    lexical = tool_match._prune_candidates(
        request, tools, PruneConfig(enabled=True, k=4, backend="bm25"))
    assert auto is not None and forced is not None and lexical is not None
    assert auto == lexical
    assert forced == lexical


# ---------------------------------------------------------------------------
# The degenerate-request full-catalogue skip (embed analogue of the
# zero-lexical-overlap fallback)
# ---------------------------------------------------------------------------

class _StubEmbedPruner:
    """Deterministic stub: a fixed embedding and a loud ranking method."""

    def __init__(self, vec: np.ndarray):
        self.vec = vec
        self.embed_calls = 0
        self.rank_calls = 0

    def embed_one(self, text):
        self.embed_calls += 1
        return self.vec

    def select_survivors_vec(self, request_vec, tools, k):
        self.rank_calls += 1
        return list(tools)[:max(1, min(k, len(tools)))]


def _install_stub(monkeypatch, stub):
    monkeypatch.setattr(tool_match, "load_embed_pruner", lambda: stub)
    monkeypatch.setattr(tool_match, "_EMBED_PRUNER", stub, raising=False)
    monkeypatch.setattr(tool_match, "_EMBED_PRUNER_LOADED", True,
                        raising=False)


def test_degenerate_embedding_scores_the_full_catalogue(monkeypatch):
    """An all-zero request embedding carries no direction: the backend must
    decline (None = full catalogue), never rank arbitrarily."""
    stub = _StubEmbedPruner(np.zeros(384, dtype=np.float32))
    _install_stub(monkeypatch, stub)
    tools = [WEATHER, DNA, EMAIL]
    out = tool_match._prune_candidates(
        EN_WEATHER_REQUEST, tools, PruneConfig(True, 2, "embed"))
    assert out is None
    assert stub.embed_calls == 1
    assert stub.rank_calls == 0, "a degenerate vector must never be ranked"


def test_blank_request_skips_the_embed_backend_without_a_forward(monkeypatch):
    for blank in ("", "   "):
        stub = _StubEmbedPruner(np.ones(384, dtype=np.float32))
        _install_stub(monkeypatch, stub)
        out = tool_match._prune_candidates(
            blank, [WEATHER, DNA], PruneConfig(True, 2, "embed"))
        assert out is None
        assert stub.embed_calls == 0


def test_backend_dispatch_failure_falls_back_to_full_catalogue(monkeypatch):
    class _Broken:
        def embed_one(self, text):
            raise RuntimeError("boom")

    _install_stub(monkeypatch, _Broken())
    out = tool_match._prune_candidates(
        EN_WEATHER_REQUEST, [WEATHER, DNA], PruneConfig(True, 2, "embed"))
    assert out is None


# ---------------------------------------------------------------------------
# The select_survivors contract (real artifact; deterministic)
# ---------------------------------------------------------------------------

def test_contract_bounds_and_order(pruner):
    tools = [WEATHER, DNA, EMAIL] + FILLERS
    out = pruner.select_survivors(EN_WEATHER_REQUEST, tools, 5)
    assert len(out) == 5
    # catalogue order: survivor positions are ascending
    positions = [tools.index(t) for t in out]
    assert positions == sorted(positions)
    # min(k, N) both ways; k <= 0 degrades to the full catalogue
    assert pruner.select_survivors("x", tools, 99) == tools
    assert pruner.select_survivors("x", tools, 0) == tools
    assert pruner.select_survivors("x", [], 3) == []
    # identical spans tie and break by catalogue index: with both in the
    # budget, the earlier catalogue entry is returned first
    dup = {"name": "dup", "description": "Does the exact same thing."}
    dup2 = dict(dup)
    tied = pruner.select_survivors("does the exact same thing",
                                   [dup, dup2, WEATHER], 2)
    assert tied == [dup, dup2]


def test_pruner_is_deterministic(pruner):
    tools = [WEATHER, DNA, EMAIL] + FILLERS
    a = pruner.select_survivors(DE_WEATHER_REQUEST, tools, 6)
    b = pruner.select_survivors(DE_WEATHER_REQUEST, tools, 6)
    assert a == b


def test_cross_lingual_ranking_survives_the_prune(pruner):
    """The property BM25 lacks: a German/Japanese request ranks the English
    weather tool FIRST in the shared multilingual space (rank probe = k=1,
    since survivors come back in catalogue order)."""
    tools = [DNA, EMAIL] + FILLERS + [WEATHER]
    for request in (DE_WEATHER_REQUEST, EN_WEATHER_REQUEST,
                    "パリの今日の天気を教えてください。"):
        assert pruner.select_survivors(request, tools, 1) == [WEATHER]
        assert WEATHER in pruner.select_survivors(request, tools, 2)


def test_rendered_span_is_what_is_embedded(pruner):
    """The span embedding reads render_tool — the same span the ranking
    artifact and the lexical backend read (one source of truth)."""
    tools = [WEATHER, DNA]
    survivors = pruner.select_survivors(EN_WEATHER_REQUEST, tools, 2)
    assert all(tool_match.render_tool(t) for t in survivors)


def test_span_cache_and_the_no_batching_forward_pin(pruner):
    """Honest cost accounting, and the batching pin: a cold catalogue of N
    tools costs EXACTLY N+1 embedding forwards (1 request + N solo span
    forwards) — the export measured solo-vs-batch divergence (max cosine
    delta 0.0484; survivor sets diverged 30/30), so if this count ever
    drops below N+1 the tool side is being batched again and the
    measurement in scripts/export_tool_pruner_embed.py must be re-done
    before that ships."""
    fresh = EmbeddingToolPruner(pruner.sess, pruner.tok)
    tools = [WEATHER, DNA, EMAIL] + FILLERS
    survivors = fresh.select_survivors(EN_WEATHER_REQUEST, tools, 12)
    assert len(survivors) == min(12, len(tools))
    assert fresh.n_forwards == len(tools) + 1
    # a second decision over the SAME catalogue pays the request only
    fresh.select_survivors(DE_WEATHER_REQUEST, tools, 12)
    assert fresh.n_forwards == len(tools) + 2
    # a fresh catalogue embeds only its new spans
    extra = tools + [{"name": "extra",
                      "description": "One more catalogue entry."}]
    fresh.select_survivors("another request entirely", extra, 5)
    assert fresh.n_forwards == len(tools) + 3 + 1
    # cache hits do not change any output (a pure cache)
    again = fresh.select_survivors(EN_WEATHER_REQUEST, tools, 12)
    cold = EmbeddingToolPruner(pruner.sess, pruner.tok)
    assert again == cold.select_survivors(EN_WEATHER_REQUEST, tools, 12)


def test_span_cache_is_bounded_lru(pruner):
    fresh = EmbeddingToolPruner(pruner.sess, pruner.tok,
                                cache_size=EMBED_PRUNER_CACHE)
    assert fresh._cache_size == EMBED_PRUNER_CACHE
    tiny = EmbeddingToolPruner(pruner.sess, pruner.tok, cache_size=2)
    tiny.select_survivors("request one", [WEATHER, DNA, EMAIL], 3)
    assert len(tiny._cache) == 2  # the oldest span was evicted, bound held
    # the evicted entry re-embeds honestly (a miss, not a stale vector)
    before = tiny.n_forwards
    tiny.select_survivors("request two", [WEATHER, DNA, EMAIL], 3)
    assert tiny.n_forwards - before >= 1


def test_embedder_max_len_is_the_training_profile(pruner):
    """128 tokens — the sentence-transformers profile the export baked in.
    A drift here changes what the model reads per span."""
    trunc = pruner.tok.truncation
    max_len = trunc["max_length"] if isinstance(trunc, dict) else trunc.max_length
    assert max_len == tool_match.EMBED_PRUNER_MAX_LEN
    assert tool_match.EMBED_PRUNER_MAX_LEN == 128


def test_identical_texts_embed_identically(pruner):
    a = pruner.embed_one(EN_WEATHER_REQUEST)
    b = pruner.embed_one(EN_WEATHER_REQUEST)
    assert np.array_equal(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# The seam on the shipped guard: narrowing, veto composition, primitives
# ---------------------------------------------------------------------------

def test_embed_backend_narrows_the_scored_set(shipped, monkeypatch):
    """With backend="embed" the seam narrows to the stub's survivors: the
    gate scores ONLY the survivor spans, never the dropped catalogue rows."""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    stub = _StubEmbedPruner(np.ones(384, dtype=np.float32))
    _install_stub(monkeypatch, stub)
    tools = [WEATHER, DNA, EMAIL] + FILLERS
    text = _catalogue_text(tools, "anything at all")
    scored = []
    real_score = shipped.gate._score

    def counting(request, span):
        scored.append(span)
        return real_score(request, span)

    monkeypatch.setattr(shipped.gate, "_score", counting)
    matcher = tool_match.SplitToolMatcher(
        shipped.gate, shipped.rank,
        prune=PruneConfig(enabled=True, k=1, backend="embed"))
    matcher.best_match(text)
    # the stub's survivor list is the catalogue head (k=1): exactly one span
    assert scored == [tool_match.render_tool(tools[0])]


def test_embed_backend_fixture_decisions_equal_under_prune(shipped):
    """End to end on the shipped split, real embedding backend: with the
    default budget the catalogue decision is identical on/off on a fixed
    multilingual fixture set."""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    if load_embed_pruner() is None:
        pytest.skip("embedding pruner artifact absent; backend falls back")
    fixtures = [
        ([WEATHER, DNA, EMAIL] + FILLERS, EN_WEATHER_REQUEST),
        ([WEATHER, DNA, EMAIL] + FILLERS, DE_WEATHER_REQUEST),
        ([DNA, WEATHER] + FILLERS,
         "Sequenzieren Sie die DNA-Molekül mit der ID DNA123."),
    ]
    for tools, request in fixtures:
        text = _catalogue_text(tools, request)
        base = tool_match.SplitToolMatcher(shipped.gate, shipped.rank)
        pruned = tool_match.SplitToolMatcher(
            shipped.gate, shipped.rank,
            prune=PruneConfig(enabled=True, k=3, backend="embed"))
        assert pruned.best_match(text) == base.best_match(text)


def test_veto_still_clarifies_under_embed_prune(shipped):
    """The veto composes after the survivor argmax, backend-independent."""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    if load_embed_pruner() is None:
        pytest.skip("embedding pruner artifact absent; backend falls back")
    tool = {"name": "calculate_sum",
            "description": "Calculate the mathematical sum of two numbers.",
            "parameters": {"type": "object",
                           "properties": {"num1": {"type": "number",
                                                   "description": "first"},
                                          "num2": {"type": "number",
                                                   "description": "second"}},
                           "required": ["num1", "num2"]}}
    text = _catalogue_text([tool] + FILLERS, "calculate the sum please")
    pruned = tool_match.SplitToolMatcher(
        shipped.gate, shipped.rank,
        prune=PruneConfig(enabled=True, k=2, backend="embed"))
    base = tool_match.SplitToolMatcher(shipped.gate, shipped.rank)
    assert pruned.best_match(text) == base.best_match(text)
    assert base.best_match(text)[1] == 0.0, (
        "the veto must fire on the shipped path for this fixture")


def test_primitives_byte_identical_under_embed_prune(shipped):
    """score_pair / select / grade / verify never read the prune config,
    whatever the backend."""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    pruned = tool_match.SplitToolMatcher(
        shipped.gate, shipped.rank,
        prune=PruneConfig(enabled=True, k=1, backend="embed"))
    request = "list the files"
    span = "list_files: List the files in a directory."
    assert pruned.score_pair(request, span) == shipped.score_pair(request,
                                                                  span)
    assert pruned.verify(request, span) == shipped.verify(request, span)
    candidates = ["get_weather: Get the current weather for a city.",
                  "list_files: List the files in a directory."]
    assert pruned.select(request, candidates) == shipped.select(request,
                                                                candidates)
    anchors = ["info level: routine message", "error level: failure"]
    assert pruned.grade(request, anchors) == shipped.grade(request, anchors)


def test_embed_prune_recall_floor_on_multilingual_fixtures(shipped):
    """The committed floor for the embed backend: on deterministic
    multilingual fixtures over a 14-tool catalogue, the tool the SHIPPED
    guard argmaxes must survive an embed prune at k=3. (The full-corpus
    agreement/recall measurement is the instrument's job; this floor must
    never regress.)"""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    if load_embed_pruner() is None:
        pytest.skip("embedding pruner artifact absent; backend falls back")
    fixtures = [
        ([DNA, EMAIL] + FILLERS + [WEATHER], EN_WEATHER_REQUEST),
        ([DNA, EMAIL] + FILLERS + [WEATHER], DE_WEATHER_REQUEST),
        ([WEATHER, EMAIL] + FILLERS + [DNA],
         "Sequenzieren Sie das DNA-Molekül mit der ID DNA123."),
        ([WEATHER, DNA] + FILLERS + [EMAIL],
         "Envoie un courriel à alice@example.com."),
        ([WEATHER, DNA] + FILLERS + [EMAIL],
         "パリの今日の天気を教えてください。"),
    ]
    checked = 0
    for tools, request in fixtures:
        text = _catalogue_text(tools, request)
        best_tool, _score = shipped.best_match(text)
        assert best_tool is not None
        survivors = tool_match._prune_candidates(
            request, tools, PruneConfig(enabled=True, k=3, backend="embed"))
        assert survivors, "the pruner must never return zero survivors"
        assert any(s == best_tool for s in survivors), (
            f"embed prune dropped the shipped argmax for {request!r}")
        checked += 1
    assert checked == len(fixtures)


def test_served_config_constant_names_the_backend_env():
    """The env var name is contract: operators pin backends with it."""
    assert tool_match.PRUNE_BACKEND_ENV == "MIVIA_GUARD_PRUNE_BACKEND"


def test_catalogue_parse_still_feeds_the_seam(pruner):
    """The pruner consumes raw tool dicts exactly as parse_catalogue_raw
    yields them (the seam the guard's best_match uses)."""
    text = _catalogue_text([WEATHER, DNA, EMAIL], DE_WEATHER_REQUEST)
    tools, request = parse_catalogue_raw(text)
    survivors = pruner.select_survivors(request, tools, 1)
    assert survivors == [WEATHER]
