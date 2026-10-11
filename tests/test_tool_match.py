"""Tests for the cross-encoder tool-match guard (no-tool abstention)."""

import json

import numpy as np
import pytest

from mivia_small_json import tool_match


@pytest.fixture(scope="module")
def matcher():
    """The shipped guard, loaded once.

    Four tests below assert the shipped artifact's behaviour, so the
    dependency on ``models/tool-match-int8.onnx`` is declared here and a
    missing artifact fails naming the file instead of surfacing as an
    AttributeError on None.
    """
    loaded = tool_match.load_matcher()
    assert loaded is not None, (
        "models/tool-match-int8.onnx is missing or failed its sha256 check; "
        "regenerate it with scripts/train_tool_match.py")
    return loaded


def test_parse_catalogue_extracts_tools_and_request():
    text = ('Tools available:\n[{"name": "f", "description": "d"}]\n\n'
            "User request: do it")
    tools, request = tool_match.parse_catalogue(text)
    assert tools == ["f: d"]
    assert request == "do it"


def test_parse_catalogue_is_empty_for_plain_text():
    assert tool_match.parse_catalogue("just a log line") == ([], "")


def test_render_tool_carries_the_parameter_schema_required_first():
    """The parameter text is where a capability constraint often lives.

    The v2 span; no shipped model uses it yet, so it is exercised directly.
    """
    tool = {"name": "get_closest_prime", "description": "Closest prime.",
            "parameters": {"type": "object", "required": ["n"],
                           "properties": {
                               "note": {"type": "string",
                                        "description": "a note"},
                               "n": {"type": "integer",
                                     "description": "the number"}}}}
    span = tool_match.render_tool(tool, parameters=True)
    assert span.startswith("get_closest_prime: Closest prime. | parameters:")
    assert "n (integer, required): the number" in span
    assert "note (string, optional): a note" in span
    assert span.index("n (integer") < span.index("note (string")


def test_render_tool_defaults_to_the_shipped_span():
    """The shipped guard reads name and description only."""
    tool = {"name": "f", "description": "d",
            "parameters": {"type": "object", "properties": {
                "a": {"type": "integer", "description": "an a"}}}}
    assert tool_match.render_tool(tool) == "f: d"


def test_render_tool_unwraps_an_openai_function_wrapper():
    wrapped = {"type": "function",
               "function": {"name": "f", "description": "d"}}
    assert tool_match.render_tool(wrapped) == "f: d"


def test_parse_catalogue_uses_the_shipped_span():
    text = ('Tools available:\n[{"name": "f", "description": "d", '
            '"parameters": {"type": "object", "required": ["a"], "properties": '
            '{"a": {"type": "integer", "description": "an a"}}}}]\n\n'
            "User request: do it")
    tools, request = tool_match.parse_catalogue(text)
    assert request == "do it"
    assert tools == ["f: d"], "the runtime span must match the shipped model"


def test_best_score_abstains_on_an_empty_catalogue():
    """`Tools available: []` has nothing to call.

    Four BFCL live-irrelevance rows are exactly this and were counted as
    false positives before the branch existed. The scorer is never reached,
    so no session is needed.
    """
    matcher = tool_match.ToolMatcher(None, None, set())
    assert matcher.best_score("Tools available:\n[]\n\nUser request: hi") == 0.0
    assert matcher.best_score("just a log line") == 1.0


def test_best_score_leaves_a_malformed_catalogue_alone():
    """A catalogue the guard cannot read must not drop a real call.

    Abstaining here would turn a formatting fault into a lost tool call.
    """
    matcher = tool_match.ToolMatcher(None, None, set())
    text = "Tools available:\nnot json\n\nUser request: do it"
    assert matcher.best_score(text) == 1.0


def test_guard_refuses_an_unverified_artifact(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"junk")
    (tmp_path / "m.onnx.sha256").write_text("0" * 64 + "\n")
    assert tool_match.load_matcher(model) is None


def test_shipped_guard_loads():
    """The loader itself, separate from the fixture the behavioural tests use."""
    assert tool_match.load_matcher() is not None


def test_served_truncation_matches_the_training_context(matcher):
    """Every leaf artifact the guard serves must truncate at 256.

    The artifacts were fine-tuned with ``--max-len 256``
    (tool-abstention); a served truncation below that
    cuts long pairs differently at serve time than the model learned to
    read. The guard can be a single matcher, a cascade, or a gate/rank split,
    so walk to the leaves. Tier 2 is the cascade's 512-token ModernBERT, by
    its own spec, so it is excluded.
    """
    assert tool_match.MAX_LEN == 256

    def tier1_leaves(m):
        if isinstance(m, tool_match.SplitToolMatcher):
            return tier1_leaves(m.gate) + tier1_leaves(m.rank)
        if isinstance(m, tool_match.CascadedToolMatcher):
            return tier1_leaves(m.tier1)
        return [m]

    for leaf in tier1_leaves(matcher):
        assert leaf.tok.truncation["max_length"] == 256


def test_guard_keeps_a_real_matching_call(matcher):
    """A real BFCL call, held out of the cross-encoder's training split."""
    text = ('Tools available:\n[{"name": "sports_celebrity.get_major_'
            'achievements", "description": "Returns a list of major '
            'achievements of a particular sports celebrity."}]\n\n'
            "User request: Find all the major achievements of the footballer "
            "Lionel Messi.")
    best = matcher.best_score(text)
    assert best >= tool_match.THRESHOLD
    # Absolute band against co-drift: measured 0.9993 on the shipped
    # artifact c34c562c03561ecd at the serving context (MAX_LEN 256).
    # Against THRESHOLD alone, a regressed artifact plus a lowered
    # threshold would pass silently.
    assert best >= 0.9493


def test_guard_abstains_when_no_listed_tool_matches(matcher):
    """A catalogue whose request needs no tool must score below threshold.

    Regression: the scorer fed all-zero ``token_type_ids``, so the model
    could not tell the request span from the tool span. This real BFCL
    irrelevance pair scored 0.70 (a false positive) instead of 0.18.
    """
    text = ('Tools available:\n[{"name": "determine_body_mass_index", '
            '"description": "Calculate body mass index given weight and '
            'height."}]\n\n'
            "User request: Calculate the area of a triangle given the base is "
            "10 meters and height is 5 meters.")
    best = matcher.best_score(text)
    assert best < tool_match.THRESHOLD
    # Absolute band against co-drift: measured 0.0006 on the shipped
    # artifact c34c562c03561ecd at the serving context (MAX_LEN 256).
    # Against THRESHOLD alone, a regressed artifact plus a raised
    # threshold would pass silently.
    assert best <= 0.0506


def test_sub_scores_reads_both_artifacts_ungated(matcher):
    """The split accessor exposes (gate, rank) soft probabilities.

    The shipped decision (`best_score`) hard-zeroes any pair the gate's
    SelectiveNet head distrusts; `sub_scores` must return the ungated
    probabilities underneath. On a call the shipped guard keeps (gated
    score >= 0.9493) the selection head accepted the pair, so the gate
    sub-score is that same probability — and the rank artifact, reading
    the plain span, must separate the pair too.
    """
    text = ('Tools available:\n[{"name": "sports_celebrity.get_major_'
            'achievements", "description": "Returns a list of major '
            'achievements of a particular sports celebrity."}]\n\n'
            "User request: Find all the major achievements of the footballer "
            "Lionel Messi.")
    tools, request = tool_match.parse_catalogue_raw(text)
    gate_score, rank_score = matcher.sub_scores(request, tools[0])
    assert 0.0 <= gate_score <= 1.0 and 0.0 <= rank_score <= 1.0
    assert gate_score >= 0.9
    assert rank_score >= 0.9


class _StubGate:
    """Gate leaf: fixed ungated probability, records the span it read."""

    def __init__(self):
        self.span = None

    def _span(self, tool, request, catalogue):
        self.span = f"gate:{tool['name']}"
        return self.span

    def _score_raw(self, request, tool):
        assert tool == "gate:my_tool", "the gate must read its own span"
        return 0.4


class _StubRank:
    def _score_raw(self, request, tool):
        assert tool == "my_tool: does a thing", (
            "the rank artifact must read the plain name+description span")
        return 0.8


def test_sub_scores_splits_the_pair_across_both_artifacts():
    """The gate reads its fillability span, the rank the plain span."""
    split = tool_match.SplitToolMatcher(_StubGate(), _StubRank())
    pair = split.sub_scores("a request",
                            {"name": "my_tool", "description": "does a thing"})
    assert pair == (0.4, 0.8)


class _Encoded:
    """A pair encoding with distinct segment ids, as a real tokenizer emits."""

    ids = [101, 2054, 102, 3079, 102]
    attention_mask = [1, 1, 1, 1, 1]
    type_ids = [0, 0, 0, 1, 1]


class _FakeTokenizer:
    def encode_batch(self, pairs):
        return [_Encoded() for _ in pairs]


class _RecordingSession:
    """Stands in for the ONNX session, keeping the feed it was handed."""

    def __init__(self):
        self.feed = None

    def run(self, _outputs, feed):
        self.feed = feed
        return [np.tile(np.array([[0.0, 1.0]]), (len(feed["input_ids"]), 1))]


def test_scorer_hands_the_model_the_tokenizers_segment_ids():
    """The feed must carry the segment ids, not zeros.

    Zeroing them is the defect the abstention test above pins
    behaviourally; this one names the mechanism, so a retrain that happens
    to abstain on that pair anyway cannot hide a reintroduction.
    """
    session = _RecordingSession()
    matcher = tool_match.ToolMatcher(
        session, _FakeTokenizer(),
        ["input_ids", "attention_mask", "token_type_ids"])
    matcher._score("a request", "a tool: does a thing")
    assert session.feed["token_type_ids"].tolist() == [[0, 0, 0, 1, 1]]


def test_validate_fillability_catches_missing_numbers():
    tool = {
        "name": "calculate_sum",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    }
    # Ultra-short action query without numbers -> underspecified_query
    fillable, reason, missing = tool_match.validate_fillability(tool, "calculate sum")
    assert not fillable
    assert reason == "underspecified_query"
    assert "a" in missing and "b" in missing

    # Longer query without numbers -> missing_arguments
    fillable, reason, missing = tool_match.validate_fillability(
        tool, "please calculate the sum of those values"
    )
    assert not fillable
    assert reason == "missing_arguments"
    assert "a" in missing and "b" in missing

    # Numbers provided
    fillable, reason, missing = tool_match.validate_fillability(
        tool, "calculate sum of 10 and 20"
    )
    assert fillable
    assert missing == []



def test_validate_fillability_catches_missing_urls():
    tool = {
        "name": "fetch_api",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "target endpoint URL"}
            },
            "required": ["url"],
        },
    }
    fillable, reason, missing = tool_match.validate_fillability(tool, "check status of server")
    assert not fillable
    assert "url" in missing

    fillable, reason, missing = tool_match.validate_fillability(
        tool, "fetch https://example.com/api"
    )
    assert fillable



def test_pnorm_confidence_normalizes_logits_by_their_p_norm():
    """The p-norm confidence estimator (Cattelan & Silva, UAI 2024).

    ``max_i z_i / ||z||_p``: a dominant logit scores near 1, equal logits
    score 1/sqrt(k), and an all-zero vector has no signal.
    """
    assert tool_match.pnorm_confidence([10.0, 0.0]) == pytest.approx(1.0)
    assert tool_match.pnorm_confidence([0.0, 10.0]) == pytest.approx(1.0)
    assert tool_match.pnorm_confidence([1.0, 1.0]) == pytest.approx(2 ** -0.5)
    assert tool_match.pnorm_confidence([0.0, 0.0]) == 0.0
    assert tool_match.pnorm_confidence([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.5)


def test_pnorm_confidence_is_monotone_in_the_logit_margin():
    """Widening the margin between two non-zero logits must raise confidence."""
    vals = [tool_match.pnorm_confidence([m, 1.0]) for m in (1.0, 2.0, 5.0, 10.0)]
    assert vals == sorted(vals)
    assert vals[-1] > vals[0]


def test_risk_threshold_satisfies_the_conformal_bound():
    """CRC picks the smallest lambda whose corrected risk is <= alpha.

    Conformal risk control (Angelopoulos et al., ICLR 2024). ``scores`` accept
    when >= lambda; ``labels`` are 1 for a true call, 0 for a near-miss.
    """
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    labels = [1, 1, 0, 1, 0, 1]
    lam = tool_match.fit_risk_threshold(scores, labels, alpha=0.2)
    n = len(scores)
    risk = sum(1 for s, y in zip(scores, labels) if s >= lam and y == 0) / n
    assert (n / (n + 1)) * risk + 1 / (n + 1) <= 0.2 + 1e-12


def test_risk_threshold_tightens_as_alpha_shrinks():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    labels = [1, 1, 0, 1, 0, 1]
    loose = tool_match.fit_risk_threshold(scores, labels, alpha=0.5)
    tight = tool_match.fit_risk_threshold(scores, labels, alpha=0.05)
    assert tight >= loose


def test_risk_threshold_accepts_nothing_when_alpha_is_tiny():
    scores = [0.9, 0.8, 0.7]
    labels = [1, 1, 1]
    assert tool_match.fit_risk_threshold(scores, labels, alpha=1e-6) == 1.0


def test_loader_ignores_gitignored_builds_by_default():
    """A local build must not silently outrank a committed artifact.

    This machine ran a cascade from the gitignored staging directory while a
    clean clone ran the shipped artifact, and the live routing tests were
    calibrated to the cascade. The default is the committed path now; local
    builds are opt-in. The cascade probes are gone entirely (spec
    fillability-veto-guard §2.2): tier-1 is the committed
    rank artifact plus, under the opt-in, the staged rank artifact.
    """
    from mivia_small_json import tool_match

    assert tool_match._staged_allowed() is False
    for cand in tool_match._tier1_candidates():
        assert cand.parent != tool_match._GITIGNORED_MODELS, cand
        assert cand.is_file(), (
            f"tier-1 probes a nonexistent committed path: {cand}")


def test_staged_candidates_are_opt_in(monkeypatch):
    """`MIVIA_ALLOW_STAGED=1` offers the local builds, so the opt-in works."""
    from mivia_small_json import tool_match

    monkeypatch.setenv("MIVIA_ALLOW_STAGED", "1")
    staged = [
        cand for cand in tool_match._tier1_candidates()
        if cand.parent == tool_match._GITIGNORED_MODELS
    ]
    assert staged, "opt-in should offer the staged builds"


# ---------------------------------------------------------------------------
# Decision-layer fillability veto (fillability-veto-guard)
# ---------------------------------------------------------------------------

def _tool(required, types=None):
    props = {}
    for i, name in enumerate(required):
        spec = {"type": (types or {}).get(name, "string"), "description": name}
        props[name] = spec
    return {"name": "t.tool", "description": "does a thing",
            "parameters": {"type": "dict", "properties": props,
                           "required": required}}


def test_fillability_veto_catches_stripped_requests():
    """The no_tool_needed construction: the tool's own request with the
    argument values stripped. The veto must fire on all of them."""
    tool = _tool(["size", "weight"], {"size": "number", "weight": "number"})
    ok, _reason, missing = tool_match.fillability_veto(tool, "set the limits")
    assert not ok and set(missing) == {"size", "weight"}


def test_fillability_veto_does_not_fire_on_fulfilled_requests():
    tool = _tool(["size"], {"size": "number"})
    ok, _reason, missing = tool_match.fillability_veto(
        tool, "set the size limit to 100m")
    assert ok and not missing


def test_fillability_veto_ignores_numeric_names_on_string_params():
    """The 'count' inside 'country' / the 'val' inside 'festival' must not
    veto: the heuristic only applies when the declared type is numeric."""
    tool = _tool(["country"])
    ok, _reason, _missing = tool_match.fillability_veto(
        tool, "what is the capital of Italy")
    assert ok


def test_fillability_veto_reads_attached_numbers():
    """5m, 100g, 19th, 0.01m2: numbers without a trailing word boundary are
    still values (the shipped regex's trailing \\b misses them)."""
    tool = _tool(["radius"], {"radius": "number"})
    ok, _reason, missing = tool_match.fillability_veto(
        tool, "circle area for radius 5m")
    assert ok and not missing


def test_split_matcher_abstains_when_the_veto_fires():
    """The composition: an argmax tool whose required parameters are absent
    from the request abstains regardless of its score — inside the scorer
    load_matcher() returns, so the canonical metric moves (spec §2.1)."""
    matcher = tool_match.load_matcher()
    if matcher is None or not isinstance(matcher, tool_match.SplitToolMatcher):
        pytest.skip("shipped split guard not loadable")
    tool = {"name": "math.add", "description": "Add two numbers",
            "parameters": {"type": "dict", "properties": {
                "a": {"type": "number", "description": "first addend"},
                "b": {"type": "number", "description": "second addend"}},
                "required": ["a", "b"]}}
    text = ("Tools available:\n" + json.dumps([tool], ensure_ascii=False)
            + "\n\nUser request: do the addition for me please")
    best_tool, score = matcher.best_match(text)
    assert score == 0.0, "a stripped request must veto to abstention"
    assert best_tool is not None, "the argmax tool is reported for routing"


def test_split_matcher_keeps_real_calls():
    matcher = tool_match.load_matcher()
    if matcher is None or not isinstance(matcher, tool_match.SplitToolMatcher):
        pytest.skip("shipped split guard not loadable")
    tool = {"name": "math.add", "description": "Add two numbers",
            "parameters": {"type": "dict", "properties": {
                "a": {"type": "number", "description": "first addend"},
                "b": {"type": "number", "description": "second addend"}},
                "required": ["a", "b"]}}
    text = ("Tools available:\n" + json.dumps([tool], ensure_ascii=False)
            + "\n\nUser request: add 4 and 6 for me please")
    _best_tool, score = matcher.best_match(text)
    assert score > tool_match.THRESHOLD


# ---------------------------------------------------------------------------
# Honest split ship: the cascade auto-probe is gone (spec §2.2)
# ---------------------------------------------------------------------------

def test_loader_has_no_cascade_probe():
    """The shipped chain is gate+rank. Tier-2 loads only via an explicit
    Tier2Spec: no auto-probe list exists and no committed path names an
    artifact that was never shipped."""
    from mivia_small_json import tool_match

    assert not hasattr(tool_match, "_tier2_candidates"), (
        "the cascade auto-probe must be removed")
    for cand in tool_match._tier1_candidates():
        assert cand == tool_match.DEFAULT_MODEL or (
            cand.parent == tool_match._GITIGNORED_MODELS), (
            f"unexpected committed tier-1 candidate: {cand}")


def test_clean_clone_loads_the_pinned_split():
    """On the committed tree the loader returns the gate+rank split with no
    warnings, and both artifact shas equal the SHIPPED.json pins."""
    import json as _json

    from mivia_small_json import tool_match

    assert tool_match._staged_allowed() is False
    matcher = tool_match.load_matcher()
    assert isinstance(matcher, tool_match.SplitToolMatcher)
    shipped = _json.loads(
        (tool_match.ROOT / "models" / "SHIPPED.json").read_text())
    pinned = shipped.get("artifacts", {})
    for role, model_name in (("gate", "tool-match-gate-int8.onnx"),
                             ("rank", "tool-match-int8.onnx")):
        assert getattr(getattr(matcher, role), "sess", None) is not None
        # A wrong/modified artifact fails the loader's own sha verification
        # before here, so a live matcher pins the committed bytes already;
        # this asserts the SHIPPED.json manifest names what we loaded.
        assert model_name in pinned, f"{model_name} missing from SHIPPED.json"


# ---------------------------------------------------------------------------
# Integrity + policy gates (2026-10-08 review axes)
# ---------------------------------------------------------------------------

def test_single_matcher_refuses_artifact_without_sidecar(tmp_path, monkeypatch):
    """A missing sidecar is a refusal, not a bypass.

    `load_single_matcher` used to verify only `if sidecar.is_file()`, so a
    tampered artifact with its sidecar DELETED loaded and served — while the
    same tampered bytes with a stale sidecar were refused. Every sibling
    loader (encoder, extract_learned) refuses on a missing sidecar.
    """
    import shutil

    from mivia_small_json import tool_match

    art = tmp_path / "tool-match-int8.onnx"
    tok = tmp_path / "tool-match-tokenizer.json"
    shutil.copy2(tool_match.DEFAULT_MODEL, art)
    shutil.copy2(tool_match.DEFAULT_TOKENIZER, tok)
    assert not (tmp_path / "tool-match-int8.onnx.sha256").exists()
    assert not (tmp_path / "tool-match-tokenizer.json.sha256").exists()

    calls = []

    def _deny(_path):
        calls.append(_path)
        return False

    monkeypatch.setattr(tool_match.integrity, "verified", _deny)
    matcher = tool_match.load_single_matcher(art, tokenizer_path=tok)
    assert matcher is None, (
        "an unverifiable artifact without sidecars must be refused")
    assert calls, "integrity.verified was never consulted"


def test_grade_matcher_staging_fallback_is_opt_in(monkeypatch):
    """`load_grade_matcher` must honour the MIVIA_ALLOW_STAGED policy.

    It used to probe the gitignored staging dir unconditionally, so a
    committed-artifact failure silently served the local build — the exact
    divergence `_staged_allowed()` exists to prevent.
    """
    import threading

    from mivia_small_json import tool_match

    sentinel = object()
    seen = []

    def _fake_load(path, **_kw):
        seen.append(path)
        if path.parent == tool_match._GITIGNORED_MODELS:
            return sentinel if tool_match._staged_allowed() else None
        return None

    monkeypatch.setattr(tool_match, "load_single_matcher", _fake_load)
    monkeypatch.setattr(tool_match, "_GRADE_LOADED", False)
    monkeypatch.setattr(tool_match, "_GRADE_MATCHER", None)
    monkeypatch.setattr(tool_match, "_GRADE_LOCK", threading.Lock())

    monkeypatch.delenv("MIVIA_ALLOW_STAGED", raising=False)
    assert tool_match.load_grade_matcher() is None
    assert len(seen) == 1, "staging dir probed without the opt-in"

    monkeypatch.setenv("MIVIA_ALLOW_STAGED", "1")
    monkeypatch.setattr(tool_match, "_GRADE_LOADED", False)
    monkeypatch.setattr(tool_match, "_GRADE_MATCHER", None)
    assert tool_match.load_grade_matcher() is sentinel


def test_select_over_a_set_is_key_ordered_not_hash_ordered():
    """Tied scores over a `set` of candidates must not depend on
    PYTHONHASHSEED: the winner comes from a deterministic ordering."""
    from mivia_small_json.tool_match import ToolMatcher

    m = ToolMatcher.__new__(ToolMatcher)
    m._score_raw = lambda text, desc: 0.5  # every candidate ties

    candidates = {"monitoring", "devops", "security", "networking"}
    key, _score = m.select("deploy the service", candidates)
    assert key == sorted(candidates)[0], (key, sorted(candidates))


def test_severity_tokenizer_refuses_missing_sidecar(tmp_path):
    """`_build_severity_ordinal` must refuse a tokenizer whose sidecar is
    missing — the same bypass class fixed in load_single_matcher (a wrong
    vocab once degraded every canonical grade to ask)."""
    import shutil

    from mivia_small_json import tool_match

    art = tmp_path / "severity-ordinal-int8.onnx"
    tok = tmp_path / "severity-tokenizer.json"
    shutil.copy2(tool_match.SEVERITY_ORDINAL_MODEL, art)
    shutil.copy2(tool_match.SEVERITY_TOKENIZER, tok)
    calls = []

    def _deny_only_tokenizer(_path):
        calls.append(_path)
        return _path != tok  # the artifact verifies; the tokenizer must not

    orig = tool_match.integrity.verified
    tool_match.integrity.verified = _deny_only_tokenizer
    try:
        head = tool_match._build_severity_ordinal(art, tok, 256)
    finally:
        tool_match.integrity.verified = orig
    assert head is None, "unverifiable tokenizer must refuse"
    assert tok in calls, "the tokenizer itself was never verified"
