"""Guard performance spec: first-stage prune (default OFF) + eager warmup.

Spec: guard-serve-performance External behavior only —
no test reads pruner internals beyond the public ``select_survivors``
contract, and there are no timing asserts anywhere (AGENTS.md: mechanical
gates are deterministic; the latency report carries the numbers).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mivia_small_json import tool_match
from mivia_small_json.tool_match import (
    DEFAULT_PRUNE_K,
    LexicalToolPruner,
    PruneConfig,
    load_prune_config,
    parse_catalogue_raw,
)

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"
CALLS = BFCL / "calls.jsonl"
FP_SET = BFCL / "irrelevance_eval.jsonl"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _catalogue_text(tools: list[dict], request: str) -> str:
    """The exact catalogue surface the guard reads (builder == runtime)."""
    return ("Tools available:\n" + json.dumps(tools, ensure_ascii=False)
            + "\n\nUser request: " + request)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


class _ScriptedMatcher(tool_match.BaseToolMatcher):
    """Scores a (request, rendered span) pair from a fixed table.

    No encoder: the table pins the shipped argmax independent of the BM25
    ranking, so the prune seam's effect is observable without the GPU-costly
    artifacts. _score stays the shipped one-pair-at-a-time contract.
    """

    def __init__(self, table: dict[str, float], prune=None):
        self.table = table
        self.prune = prune
        self.scored: list[str] = []

    def _score(self, request: str, tool: str) -> float:
        self.scored.append(tool)
        return self.table[tool]


FIXTURE_TOOLS = [
    {"name": "delete_database",
     "description": "Delete a database by name from the hosting account."},
    {"name": "get_weather",
     "description": "Get the current weather forecast for a city."},
    {"name": "sync_calendar",
     "description": "Sync calendar events across connected accounts."},
    {"name": "list_files",
     "description": "List the files in a directory."},
    {"name": "send_email",
     "description": "Send an email to a recipient."},
]
# The scripted argmax is sync_calendar; the request's vocabulary instead
# matches delete_database — so a prune with k=1 MUST change the decision,
# which is what proves the seam actually narrows the candidate list.
SCRIPTED_SCORES = {f"{t['name']}: {t['description']}": score
                   for t, score in zip(FIXTURE_TOOLS,
                                       [0.1, 0.2, 0.9, 0.3, 0.4])}
FIXTURE_REQUEST = "delete the production database"


@pytest.fixture(scope="module")
def shipped():
    """The shipped guard (split gate+rank), loaded once."""
    loaded = tool_match.load_matcher()
    assert loaded is not None, (
        "models/tool-match-*.onnx are missing or failed their sha256 "
        "check; the guardperf tests need the shipped guard")
    return loaded


# ---------------------------------------------------------------------------
# Config surface: env-driven, read once at guard construction. Default OFF
# (returned OFF 2026-10-10: decision-agreement on genuinely-matching large
# catalogues failed the binding >=99% bar — see the tool_match module
# comment for the full dated history). Only exact truthy tokens enable;
# garbage falls back to the default.
# ---------------------------------------------------------------------------

def test_prune_config_defaults_off():
    assert load_prune_config({}) == PruneConfig(enabled=False,
                                                k=DEFAULT_PRUNE_K)
    assert DEFAULT_PRUNE_K == 25


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "enabled", "",
                                   "garbage", " 1 "])
def test_prune_config_truthy_literals_only(value):
    enabled = load_prune_config({"MIVIA_GUARD_PRUNE": value}).enabled
    assert enabled is (value.strip().lower() in tool_match._PRUNE_TRUTHY)


def test_prune_config_survives_garbage_k():
    for bad in ("abc", "", "-4", "0", "12.5", None):
        cfg = load_prune_config({"MIVIA_GUARD_PRUNE_K": bad})
        assert cfg.k == DEFAULT_PRUNE_K


def test_prune_config_reads_the_process_environment(monkeypatch):
    monkeypatch.delenv("MIVIA_GUARD_PRUNE", raising=False)
    monkeypatch.delenv("MIVIA_GUARD_PRUNE_K", raising=False)
    assert load_prune_config().enabled is False
    monkeypatch.setenv("MIVIA_GUARD_PRUNE", "1")
    monkeypatch.setenv("MIVIA_GUARD_PRUNE_K", "7")
    assert load_prune_config() == PruneConfig(enabled=True, k=7)


def test_split_owns_the_prune_config():
    """The split stores the config it was given; the gate object stays
    untouched (composition-layer ownership, same seam as the veto)."""
    cfg = PruneConfig(enabled=True, k=3)
    split = tool_match.SplitToolMatcher(
        tool_match.ToolMatcher(None, None, set()),
        tool_match.ToolMatcher(None, None, set()), prune=cfg)
    assert split.prune is cfg
    assert split.gate.prune is None


def test_guard_construction_reads_the_environment(monkeypatch):
    """load_matcher with no explicit config reads the environment once at
    guard construction and hands the config to the loaders; an explicit
    config overrides the environment. (Recording fake loaders — the seam
    is the point, not the artifacts; keeps the suite's shared-process RSS
    inside the sandbox law.)"""
    monkeypatch.setenv("MIVIA_GUARD_PRUNE", "1")
    monkeypatch.setenv("MIVIA_GUARD_PRUNE_K", "9")
    seen: list = []

    def fake_loader(path, tokenizer_path=None, max_len=None, span_fn=None,
                    prune=None):
        seen.append(prune)
        return None

    monkeypatch.setattr(tool_match, "load_single_matcher", fake_loader)
    assert tool_match.load_matcher() is None  # loaders refused; warns
    assert seen and all(p == PruneConfig(enabled=True, k=9) for p in seen)
    seen.clear()
    assert tool_match.load_matcher(prune=PruneConfig(enabled=False)) is None
    assert seen and all(p is not None and p.enabled is False for p in seen)


# ---------------------------------------------------------------------------
# Catalogue decision: prune-on == prune-off (fixture-level recall), the seam
# really narrows, zero-survivor fallback, determinism.
# ---------------------------------------------------------------------------

def _scripted(text: str, prune=None) -> tuple[_ScriptedMatcher, dict, float]:
    matcher = _ScriptedMatcher(SCRIPTED_SCORES, prune=prune)
    best, score = matcher.best_match(text)
    return matcher, best, score


def test_prune_off_is_the_shipped_scan():
    text = _catalogue_text(FIXTURE_TOOLS, FIXTURE_REQUEST)
    matcher, best, score = _scripted(text)
    assert best["name"] == "sync_calendar"
    assert score == 0.9
    assert len(matcher.scored) == len(FIXTURE_TOOLS)


def test_prune_on_equals_prune_off_at_full_budget():
    """k = N survivors: the decision must be identical (catalogue recall)."""
    text = _catalogue_text(FIXTURE_TOOLS, FIXTURE_REQUEST)
    _, best, score = _scripted(
        text, prune=PruneConfig(enabled=True, k=len(FIXTURE_TOOLS)))
    assert best["name"] == "sync_calendar"
    assert score == 0.9


def test_prune_actually_narrows_the_scored_set():
    """k = 1 with a request matching the wrong tool: the BM25 winner is
    scored and the scripted argmax is never seen — the seam selects rows."""
    text = _catalogue_text(FIXTURE_TOOLS, FIXTURE_REQUEST)
    matcher, best, score = _scripted(text, prune=PruneConfig(True, 1))
    assert matcher.scored == ["delete_database: "
                              "Delete a database by name from the hosting "
                              "account."]
    assert best["name"] == "delete_database"
    assert score == 0.1


def test_prune_survivors_are_scored_with_the_shipped_span():
    """The scan must render the same span on/off (tool identity and span
    text are the argmax keys) — the pruner reads exactly render_tool."""
    pruner = LexicalToolPruner()
    survivors = pruner.select_survivors(FIXTURE_REQUEST, FIXTURE_TOOLS, 2)
    for tool in survivors:
        assert tool_match.render_tool(tool) in SCRIPTED_SCORES


def test_pruner_is_deterministic_and_breaks_ties_by_catalogue_index():
    pruner = LexicalToolPruner()
    first = pruner.select_survivors(FIXTURE_REQUEST, FIXTURE_TOOLS, 3)
    second = pruner.select_survivors(FIXTURE_REQUEST, FIXTURE_TOOLS, 3)
    assert first == second
    # A query with zero lexical overlap scores every span 0.0: the top-k
    # degenerates to the first k catalogue entries, in catalogue order.
    nohit = pruner.select_survivors("zzz qqq", FIXTURE_TOOLS, 3)
    assert nohit == FIXTURE_TOOLS[:3]
    # k >= N returns the whole catalogue in catalogue order; k <= 0 degrades
    # to the same safe full catalogue.
    assert pruner.select_survivors("x", FIXTURE_TOOLS, 99) == FIXTURE_TOOLS
    assert pruner.select_survivors("x", FIXTURE_TOOLS, 0) == FIXTURE_TOOLS
    assert pruner.select_survivors("x", [], 3) == []
    # min(k, N): a budget larger than the catalogue cannot invent tools.
    assert len(pruner.select_survivors(FIXTURE_REQUEST, FIXTURE_TOOLS, 12)) \
        == len(FIXTURE_TOOLS)


def test_zero_survivor_fallback_scores_the_full_catalogue(monkeypatch):
    """A pathological pruner returning nothing must never yield an empty
    argmax: the decision path falls back to the full catalogue. (Pinned on
    the bm25 backend — the embed backend has its own degenerate-request
    skip, tested in tests/test_embedpruner.py.)"""

    class _EmptyPruner:
        def select_survivors(self, request, tools, k):
            return []

    monkeypatch.setattr(tool_match, "_DEFAULT_PRUNER", _EmptyPruner())
    text = _catalogue_text(FIXTURE_TOOLS, FIXTURE_REQUEST)
    matcher = _ScriptedMatcher(
        SCRIPTED_SCORES, prune=PruneConfig(enabled=True, k=2, backend="bm25"))
    best, score = matcher.best_match(text)
    assert best["name"] == "sync_calendar"
    assert score == 0.9
    assert len(matcher.scored) == len(FIXTURE_TOOLS)


def test_pruner_failure_falls_back_to_the_full_catalogue(monkeypatch):
    """(bm25 backend pin; the embed backend's dispatch-failure fallback is
    tests/test_embedpruner.py::test_backend_dispatch_failure_falls_back_to_
    full_catalogue.)"""

    class _BrokenPruner:
        def select_survivors(self, request, tools, k):
            raise RuntimeError("boom")

    monkeypatch.setattr(tool_match, "_DEFAULT_PRUNER", _BrokenPruner())
    text = _catalogue_text(FIXTURE_TOOLS, FIXTURE_REQUEST)
    matcher = _ScriptedMatcher(
        SCRIPTED_SCORES, prune=PruneConfig(True, 2, "bm25"))
    best, score = matcher.best_match(text)
    assert best["name"] == "sync_calendar" and score == 0.9


# ---------------------------------------------------------------------------
# Zero-lexical-overlap fallback (multilingual prune-recall wave, spec
# guard-serve-performance Further Notes). A request whose
# tokens share NO vocabulary with any tool span — the CJK/non-Latin case —
# leaves BM25 without signal (every span scores 0.0, survivors degenerate to
# the first k entries): the seam must skip the stage and score the FULL
# catalogue, silently. Partial-overlap latin-script requests still prune.
# These are BM25-backend pins (backend="bm25"): the EMBED backend ranks
# cross-lingually by design and its behavior on these very requests is
# covered in tests/test_embedpruner.py.
# ---------------------------------------------------------------------------

# Zero ASCII tokens under the BM25 tokenizer `[a-z0-9]+` — the premise is
# asserted in the first test so a sentence edit cannot silently void it.
JAPANESE_REQUEST = "東京タワーへの行き方を教えてください"
# A realistic partial-overlap latin-script request: German politeness around
# English tool/identifier vocabulary (delete, database, production).
GERMAN_PARTIAL_REQUEST = "lösche bitte die production database"


def test_zero_ascii_request_premise():
    assert tool_match._bm25_tokenize(JAPANESE_REQUEST) == []
    assert tool_match._bm25_tokenize(GERMAN_PARTIAL_REQUEST)


def test_zero_ascii_request_takes_the_disabled_path():
    """k=1 would narrow the scan to the BM25 winner; the zero-overlap
    fallback must instead score the whole catalogue (decision == prune-off,
    scorer sees every span)."""
    text = _catalogue_text(FIXTURE_TOOLS, JAPANESE_REQUEST)
    matcher, best, score = _scripted(text,
                                     prune=PruneConfig(True, 1, "bm25"))
    assert len(matcher.scored) == len(FIXTURE_TOOLS)
    assert best["name"] == "sync_calendar"
    assert score == 0.9


def test_partial_overlap_latin_request_still_prunes():
    text = _catalogue_text(FIXTURE_TOOLS, GERMAN_PARTIAL_REQUEST)
    matcher, best, score = _scripted(text,
                                     prune=PruneConfig(True, 1, "bm25"))
    assert len(matcher.scored) == 1, (
        "a request with lexical overlap must still narrow the scan")
    assert best["name"] == "delete_database"
    assert score == 0.1


def test_zero_overlap_fallback_is_silent(capsys):
    """The fallback is a designed path, not a failure: it warns nothing."""
    text = _catalogue_text(FIXTURE_TOOLS, JAPANESE_REQUEST)
    _scripted(text, prune=PruneConfig(True, 1, "bm25"))
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_shipped_guard_zero_overlap_scores_full_catalogue(shipped, monkeypatch):
    """End to end on the shipped split: the zero-overlap request with k=1
    must produce the identical (tool, score) the prune-off path gives, and
    the gate must score every catalogue tool."""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    text = _catalogue_text(FIXTURE_TOOLS, JAPANESE_REQUEST)
    base_tool, base_score = shipped.best_match(text)
    calls = []
    real_score = shipped.gate._score

    def counting(request, span):
        calls.append(span)
        return real_score(request, span)

    monkeypatch.setattr(shipped.gate, "_score", counting)
    pruned = tool_match.SplitToolMatcher(
        shipped.gate, shipped.rank, prune=PruneConfig(True, 1, "bm25"))
    best_tool, score = pruned.best_match(text)
    assert len(calls) == len(FIXTURE_TOOLS), (
        "the zero-overlap fallback must scan the full catalogue")
    assert best_tool == base_tool
    assert score == pytest.approx(base_score, abs=1e-9)


# ---------------------------------------------------------------------------
# Shipped-guard behavior under prune: fixture-level recall, veto composition,
# primitives byte-identical.
# ---------------------------------------------------------------------------

def _fixture_catalogues(count: int, seed: int = 20261010) -> list[str]:
    """A deterministic bounded sample of committed multi-tool catalogues.

    Single-tool catalogues cannot miss (min(k, 1) = 1), so the sample draws
    only catalogues with >= 2 tools — the shapes the prune can actually hurt.
    """
    texts: list[str] = []
    seen_heads: set[str] = set()
    for path in (CALLS, FP_SET):
        if not path.is_file():
            pytest.skip(f"{path.name} not committed")
        rows = _rows(path)
        random.Random(seed).shuffle(rows)
        for rec in rows:
            text = rec.get("text") or rec.get("raw") or ""
            tools, request = parse_catalogue_raw(text)
            head = text.split("\n\n")[0]
            if (tools and len(tools) >= 2 and request
                    and head not in seen_heads):
                seen_heads.add(head)
                texts.append(text)
            if len(texts) >= count:
                return texts
    return texts


def test_shipped_guard_fixture_decisions_equal_under_prune(shipped):
    """Fixture-level recall: prune-on catalogue decisions equal prune-off on
    a deterministic sample of the committed corpora."""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    texts = _fixture_catalogues(4)
    assert len(texts) >= 4, "the committed corpora lost their multi-tool rows"
    pruned = tool_match.SplitToolMatcher(shipped.gate, shipped.rank,
                                         prune=PruneConfig(True, DEFAULT_PRUNE_K))
    for text in texts:
        base_tool, base_score = shipped.best_match(text)
        pruned_tool, pruned_score = pruned.best_match(text)
        assert pruned_tool == base_tool
        assert pruned_score == pytest.approx(base_score, abs=1e-9)


def test_veto_still_clarifies_under_prune(shipped):
    """The veto composes on the survivor argmax: a catalogue whose argmax
    tool needs unfilled parameters abstains identically on/off."""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    tool = {"name": "calculate_sum",
            "description": "Calculate the mathematical sum of two numbers.",
            "parameters": {"type": "object",
                           "properties": {"num1": {"type": "number",
                                                   "description": "first"},
                                          "num2": {"type": "number",
                                                   "description": "second"}},
                           "required": ["num1", "num2"]}}
    text = _catalogue_text([tool], "calculate the sum please")
    pruned = tool_match.SplitToolMatcher(shipped.gate, shipped.rank,
                                         prune=PruneConfig(True, 1))
    base_tool, base_score = shipped.best_match(text)
    pruned_tool, pruned_score = pruned.best_match(text)
    assert base_tool is not None and base_score == 0.0, \
        "the veto must fire on the shipped path for this fixture"
    assert pruned_tool == base_tool
    assert pruned_score == 0.0


def test_primitives_are_byte_identical_on_and_off(shipped):
    """score_pair / select / grade / verify never read the prune config."""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    pruned = tool_match.SplitToolMatcher(shipped.gate, shipped.rank,
                                         prune=PruneConfig(True, 1))
    request, span = "list the files", "list_files: List the files in a directory."
    assert pruned.score_pair(request, span) == shipped.score_pair(request, span)
    assert pruned.verify(request, span) == shipped.verify(request, span)
    candidates = ["get_weather: Get the current weather for a city.",
                  "list_files: List the files in a directory."]
    assert pruned.select(request, candidates) == shipped.select(request,
                                                                candidates)
    anchors = ["info level: routine message", "error level: failure"]
    assert pruned.grade(request, anchors) == shipped.grade(request, anchors)


def test_prune_recall_floor_on_bounded_committed_sample(shipped):
    """Pre-registered bar 1, bounded fixture form: the shipped argmax tool
    survives the default budget (k=12) on EVERY sampled catalogue. (The
    full-corpus measurement is the instrument's job and lands in the
    report; this is the deterministic floor that must never regress.)"""
    pruner = LexicalToolPruner()
    checked = 0
    for text in _fixture_catalogues(6, seed=42):
        tools, request = parse_catalogue_raw(text)
        best_tool, _score = shipped.best_match(text)
        assert best_tool is not None
        survivors = pruner.select_survivors(request, tools, DEFAULT_PRUNE_K)
        assert survivors, "the pruner must never return zero survivors"
        assert any(s == best_tool for s in survivors), (
            "prune dropped the shipped argmax — a false abstention on a "
            f"real call; catalogue head: {text[:120]!r}")
        checked += 1
    assert checked >= 6


# ---------------------------------------------------------------------------
# Eager warmup: REST lifespan, sidecar startup, failure semantics.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def router():
    from mivia_small_json import pipeline

    return pipeline.load_active_router()


def _fresh_guard_singleton(monkeypatch):
    """Reset the process-wide guard singleton so the lazy load is observable.

    Restored on teardown, so the suite's shared singleton state is intact.
    """
    monkeypatch.setattr(tool_match, "_GUARD", None, raising=False)
    monkeypatch.setattr(tool_match, "_GUARD_LOADED", False)


class _StubGuard:
    """A catalogue-decision stub served through the real singleton path.

    The warmup tests target the lifespan SEQUENCING (warm before ready; the
    spy must not be re-invoked), not the artifacts — the real gate/rank
    artifacts are exercised by the ``shipped`` fixture above. Keeping the
    REST warmup tests on a stub holds the suite's shared-process peak RSS
    inside the 2 GB sandbox law the serving tests assert."""

    def best_match(self, text: str):
        tools, request = tool_match.parse_catalogue_raw(text)
        if not tools or not request:
            return None, 0.0
        return tools[0], 0.9

    def select(self, text, candidates, *args, **kwargs):
        return "", 0.0  # inert: the custom-schema ladder must not fire

    def verify(self, text, hypothesis, threshold=0.5):
        return False, 0.0


def test_rest_lifespan_warms_and_first_guard_request_is_warm(
        router, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    from mivia_small_json import api

    _fresh_guard_singleton(monkeypatch)
    loads = []

    def spy(*args, **kwargs):
        loads.append(1)
        return _StubGuard()

    monkeypatch.setattr(tool_match, "load_matcher", spy)
    app = api.create_app(token="", router=router)
    with TestClient(base_url="http://127.0.0.1", app=app) as client:
        assert loads, "the lifespan must warm the guard before ready"
        n = len(loads)
        resp = client.post(
            "/v1/guard",
            json={"request": "list the files in the directory",
                  "tools": [{"name": "list_files",
                             "description": "List the files in a directory."}]})
        assert resp.status_code == 200, resp.json()
        assert resp.json()["action"] in ("CALL", "CLARIFY")
        assert len(loads) == n, "the first guard request must not lazy-load"


def test_rest_warmup_failure_warns_and_startup_continues(
        router, monkeypatch, capsys):
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    from mivia_small_json import api, pipeline

    def boom(*args, **kwargs):
        raise RuntimeError("artifact broken")

    monkeypatch.setattr(pipeline, "classify", boom)
    monkeypatch.setattr(tool_match, "load_matcher", lambda *a, **k: _StubGuard())
    _fresh_guard_singleton(monkeypatch)
    app = api.create_app(token="", router=router)
    with TestClient(base_url="http://127.0.0.1", app=app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200, "startup must survive warmup failure"
    err = capsys.readouterr().err
    assert "warmup step 'router classify' failed" in err
    assert "warmup step 'guard decision' failed" in err


def test_sidecar_warms_before_the_accept_loop(monkeypatch):
    from mivia_small_json import serve

    events = []

    class _FakeServer:
        def __init__(self, address, handler):
            pass

        def serve_forever(self):
            events.append("accept_loop")

    monkeypatch.setattr(serve, "build_router", lambda: None)
    monkeypatch.setattr(serve, "ThreadingHTTPServer", _FakeServer)
    real_classify = serve.pipeline.classify

    def spy_classify(text, router_):
        events.append("classify")
        return real_classify(text, router_)

    monkeypatch.setattr(serve.pipeline, "classify", spy_classify)
    assert serve.run(port=0) == 0
    assert events == ["classify", "accept_loop"], (
        "the sidecar must finish warming before it accepts a connection")


def test_sidecar_does_not_warm_the_guard(monkeypatch):
    """The sidecar serves classify surfaces only: warming the guard there
    would load artifacts the process never uses (deliberate scope cut)."""
    import mivia_small_json.tool_match as tm
    from mivia_small_json import serve

    events = []
    real_classify = serve.pipeline.classify

    def spy_classify(text, router_):
        events.append("classify")
        return real_classify(text, router_)

    real_guard = tm._get_guard

    def spy_guard():
        events.append("guard")
        return real_guard()

    monkeypatch.setattr(serve, "build_router", lambda: None)
    monkeypatch.setattr(serve.pipeline, "classify", spy_classify)
    monkeypatch.setattr(tm, "_get_guard", spy_guard)

    class _FakeServer:
        def __init__(self, address, handler):
            pass

        def serve_forever(self):
            pass

    monkeypatch.setattr(serve, "ThreadingHTTPServer", _FakeServer)
    serve.run(port=0)
    assert events == ["classify"], (
        "the sidecar must warm the router only, never the guard")


def test_sidecar_warmup_failure_warns_and_continues(monkeypatch, capsys):
    from mivia_small_json import serve

    def boom(*args, **kwargs):
        raise RuntimeError("router broken")

    monkeypatch.setattr(serve.pipeline, "classify", boom)
    serve.warmup(None)  # must not raise
    err = capsys.readouterr().err
    assert "sidecar warmup step 'router classify' failed" in err


# ---------------------------------------------------------------------------
# Embed prune backend (spec embed-tool-pruner) — additive
# backend tests; the deep embed coverage lives in tests/test_embedpruner.py.
# ---------------------------------------------------------------------------

def test_backend_env_contract():
    """auto|bm25|embed parse; anything else falls back to auto."""
    assert load_prune_config({"MIVIA_GUARD_PRUNE_BACKEND": "embed"}).backend \
        == "embed"
    assert load_prune_config({"MIVIA_GUARD_PRUNE_BACKEND": "bm25"}).backend \
        == "bm25"
    assert load_prune_config({"MIVIA_GUARD_PRUNE_BACKEND": "auto"}).backend \
        == "auto"
    assert load_prune_config({"MIVIA_GUARD_PRUNE_BACKEND": "garbage"}) \
        .backend == "auto"
    assert load_prune_config({}).backend == "auto"


def test_embed_backend_absent_degrades_to_the_lexical_seam(monkeypatch):
    """With the embed artifact unloadable, backend="embed" narrows exactly
    like bm25 — the decision path never breaks, never changes shape."""
    import pathlib

    monkeypatch.setattr(tool_match, "EMBED_PRUNER_MODEL",
                        pathlib.Path("/nonexistent") / "x.onnx", raising=False)
    monkeypatch.setattr(tool_match, "_EMBED_PRUNER", None, raising=False)
    monkeypatch.setattr(tool_match, "_EMBED_PRUNER_LOADED", False,
                        raising=False)
    text = _catalogue_text(FIXTURE_TOOLS, FIXTURE_REQUEST)
    matcher = _ScriptedMatcher(
        SCRIPTED_SCORES, prune=PruneConfig(True, 1, "embed"))
    best, score = matcher.best_match(text)
    assert matcher.scored == ["delete_database: "
                              "Delete a database by name from the hosting "
                              "account."]
    assert best["name"] == "delete_database" and score == 0.1


def test_primitives_byte_identical_under_embed_backend(shipped):
    """The embed prune config is invisible to the pair-level and ranking
    primitives (backend="embed" variant of the lexical pin)."""
    if not hasattr(shipped, "gate"):
        pytest.skip("shipped guard is not the split (gate artifact missing)")
    pruned = tool_match.SplitToolMatcher(
        shipped.gate, shipped.rank,
        prune=PruneConfig(enabled=True, k=1, backend="embed"))
    request, span = "list the files", "list_files: List the files in a directory."
    assert pruned.score_pair(request, span) == shipped.score_pair(request, span)
    assert pruned.verify(request, span) == shipped.verify(request, span)
