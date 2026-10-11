"""Unit tests for zero-shot discriminative primitives on ToolMatcher.

Tests select(), verify(), and grade() covering:
- Mocked dual-output and single-output sessions (SelectiveNet rejection/acceptance)
- Empty and single-candidate edge cases
- Selection head gating (SelectiveNet) rejection propagation
- Prompt-to-model routing pipelines (routing user asks to specialized models)
- Real ONNX model integration when artifact is available
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from mivia_small_json import tool_match
from mivia_small_json.tool_match import ToolMatcher, load_grade_matcher, load_matcher


class MockEncoding:
    def __init__(self):
        self.ids = [101, 102]
        self.attention_mask = [1, 1]
        self.type_ids = [0, 1]


class MockTokenizer:
    def encode_batch(self, pairs):
        return [MockEncoding() for _ in pairs]


class MockSession:
    """Deterministic ONNX session returning configurable logits and selection scores."""

    def __init__(self, logits: list[float], selection_score: float | None = None):
        self.logits = np.array([logits], dtype=np.float32)
        self.selection_score = (
            np.array([[selection_score]], dtype=np.float32)
            if selection_score is not None
            else None
        )

    def run(self, output_names, feed):
        if self.selection_score is not None:
            return [self.logits, self.selection_score]
        return [self.logits]


def test_tool_matcher_mock_session_accepted():
    # Logits: class 0 = -2.0, class 1 = 2.0 -> prob ~ 0.982. Selection 0.85 >= 0.50 -> accepted
    sess = MockSession(logits=[-2.0, 2.0], selection_score=0.85)
    tok = MockTokenizer()
    matcher = ToolMatcher(sess, tok, inputs=["input_ids", "attention_mask"])
    score = matcher._score("request", "tool: description")
    expected_prob = 1.0 / (1.0 + math.exp(-4.0))
    assert abs(score - expected_prob) < 1e-4
    assert score > 0.95


def test_tool_matcher_mock_session_rejected():
    # Logits: class 0 = -2.0, class 1 = 2.0, but selection 0.20 < 0.50 -> collapsed to 0.0
    sess = MockSession(logits=[-2.0, 2.0], selection_score=0.20)
    tok = MockTokenizer()
    matcher = ToolMatcher(sess, tok, inputs=["input_ids", "attention_mask"])
    score = matcher._score("request", "tool: description")
    assert score == 0.0


def test_tool_matcher_single_output_session():
    # Model without auxiliary selection head (standard cross-encoder)
    sess = MockSession(logits=[-1.0, 1.0], selection_score=None)
    tok = MockTokenizer()
    matcher = ToolMatcher(sess, tok, inputs=["input_ids", "attention_mask"])
    score = matcher._score("request", "tool: description")
    expected_prob = 1.0 / (1.0 + math.exp(-2.0))
    assert abs(score - expected_prob) < 1e-4


class ControlledToolMatcher(ToolMatcher):
    """ToolMatcher with deterministic pair-scoring for unit testing primitives."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def _score(self, request: str, tool: str) -> float:
        # Match by exact key, substring, or suffix
        if tool in self._scores:
            return self._scores[tool]
        for k, v in self._scores.items():
            if k in tool or tool in k:
                return v
        return 0.0

    def _score_raw(self, request: str, tool: str) -> float:
        return self._score(request, tool)


def test_select_dict_candidates():
    scores = {
        "Bug report with stack trace": 0.92,
        "Git commit message": 0.15,
        "System daemon log line": 0.05,
    }
    matcher = ControlledToolMatcher(scores)
    candidates = {
        "issue": "Bug report with stack trace",
        "commit": "Git commit message",
        "log": "System daemon log line",
    }
    # calibrated=False pins the pre-Platt softmax. The default return is
    # the Platt value; test_select_confidence_does_not_depend_on_with_status
    # owns that number.
    best_key, conf = matcher.select(
        "NullPointerException in main", candidates, calibrated=False)
    assert best_key == "issue"
    assert conf > 0.50
    raw_s = np.array([0.92, 0.15, 0.05])
    exp_s = np.exp(raw_s - np.max(raw_s))
    expected_softmax = float(exp_s[0] / exp_s.sum())
    assert abs(conf - expected_softmax) < 1e-5


def test_select_list_candidates():
    scores = {
        "calculate_sum: adds numbers": 0.88,
        "query_database: runs SQL": 0.12,
    }
    matcher = ControlledToolMatcher(scores)
    best_key, conf = matcher.select("add 5 and 10", list(scores.keys()))
    assert best_key == "calculate_sum: adds numbers"
    assert conf > 0.60


def test_select_empty_and_single_edge_cases():
    matcher = ControlledToolMatcher({})
    assert matcher.select("test", []) == ("", 0.0)
    assert matcher.select("test", {}) == ("", 0.0)

    matcher_single = ControlledToolMatcher({"only_one": 0.77})
    key, score = matcher_single.select("test", ["only_one"])
    assert key == "only_one"
    assert abs(score - 0.77) < 1e-5


def test_select_all_zero_scores():
    matcher = ControlledToolMatcher({"a": 0.0, "b": 0.0})
    key, conf = matcher.select("test", ["a", "b"])
    assert key in ("a", "b")
    assert conf == 0.0


class _ShiftCalibrator:
    """Stand-in Platt head: p -> 0.2 + 0.5*p. Not the shipped coefficients."""

    def predict_proba(self, xs):
        raw = np.asarray(xs, dtype=np.float64)[:, 0]
        p = 0.2 + 0.5 * raw
        return np.column_stack([1.0 - p, p])


def _install_shift_calibrator(monkeypatch, tau: float, calls: dict | None = None):
    def _load():
        if calls is not None:
            calls["n"] += 1
        return (_ShiftCalibrator(), tau)

    monkeypatch.setattr(tool_match, "load_select_calibrator", _load)


def test_select_confidence_does_not_depend_on_with_status(monkeypatch):
    """Asking for a status flag must not change the confidence number.

    Both shapes return the Platt value of the same softmax. Tau sits
    between the raw softmax and the calibrated value, so an abstain
    decision that compared the raw number would report ok.
    """
    _install_shift_calibrator(monkeypatch, tau=0.60)
    matcher = ControlledToolMatcher({"x": 0.95, "y": 0.10})
    raw = 1.0 / (1.0 + math.exp(0.10 - 0.95))
    want = 0.2 + 0.5 * raw

    key, conf = matcher.select("text", ["x", "y"])
    key2, conf2, status = matcher.select("text", ["x", "y"], with_status=True)

    assert key == key2 == "x"
    assert conf == pytest.approx(want)
    assert conf2 == pytest.approx(conf)
    assert conf != pytest.approx(raw)
    assert status == "abstain"


def test_select_with_status_loads_the_calibrator_once(monkeypatch):
    """Confidence and tau come from one load. A second load is the
    calibrate-then-reload path."""
    calls = {"n": 0}
    _install_shift_calibrator(monkeypatch, tau=0.60, calls=calls)
    matcher = ControlledToolMatcher({"x": 0.95, "y": 0.10})

    matcher.select("text", ["x", "y"], with_status=True)

    assert calls["n"] == 1


def test_select_calibrated_false_keeps_the_pre_platt_softmax(monkeypatch):
    """The fit reads the pre-Platt softmax. Calibrating that score again
    would train the map on its own output. The flag does not depend on
    with_status either."""
    _install_shift_calibrator(monkeypatch, tau=0.60)
    matcher = ControlledToolMatcher({"x": 0.95, "y": 0.10})
    raw = 1.0 / (1.0 + math.exp(0.10 - 0.95))

    _key, conf = matcher.select("text", ["x", "y"], calibrated=False)
    _key2, conf2, _status = matcher.select(
        "text", ["x", "y"], with_status=True, calibrated=False)

    assert conf == pytest.approx(raw)
    assert conf2 == pytest.approx(raw)


def test_select_all_zero_stays_zero_with_or_without_status(monkeypatch):
    """A collapsed score is confidence 0, not Platt(0), on both shapes.
    Status stays ambiguous: there is no positive softmax to abstain on."""
    _install_shift_calibrator(monkeypatch, tau=0.50)
    matcher = ControlledToolMatcher({"a": 0.0, "b": 0.0})

    _key, conf = matcher.select("test", ["a", "b"])
    _key2, conf2, status = matcher.select("test", ["a", "b"], with_status=True)

    assert conf == 0.0
    assert conf2 == 0.0
    assert status == "ambiguous"


def test_select_single_candidate_stays_on_the_raw_score(monkeypatch):
    """n=1 is outside the Platt contract. The head was fit on softmaxes."""
    _install_shift_calibrator(monkeypatch, tau=0.10)
    matcher = ControlledToolMatcher({"only": 0.77})

    assert matcher.select("t", ["only"])[1] == pytest.approx(0.77)
    key, score, status = matcher.select("t", ["only"], with_status=True)
    assert key == "only"
    assert score == pytest.approx(0.77)
    assert status == "ok"


def test_verify_acceptance_and_rejection():
    scores = {
        "valid hypothesis": 0.85,
        "invalid hypothesis": 0.20,
    }
    matcher = ControlledToolMatcher(scores)

    accepted, score = matcher.verify("text", "valid hypothesis")
    assert accepted is True
    assert abs(score - 0.85) < 1e-5

    accepted_bad, score_bad = matcher.verify("text", "invalid hypothesis")
    assert accepted_bad is False
    assert abs(score_bad - 0.20) < 1e-5


def test_verify_custom_threshold():
    matcher = ControlledToolMatcher({"borderline": 0.65})
    assert matcher.verify("text", "borderline", threshold=0.70)[0] is False
    assert matcher.verify("text", "borderline", threshold=0.60)[0] is True


def test_grade_ordinal_alignment():
    scores = {
        "debug: detailed diagnostic information": 0.10,
        "info: normal operational messages": 0.25,
        "warn: recoverable abnormal conditions": 0.70,
        "error: unrecoverable failure or crash": 0.95,
    }
    matcher = ControlledToolMatcher(scores)
    anchors = list(scores.keys())

    best_anchor, score = matcher.grade("Fatal disk failure detected", anchors)
    assert best_anchor == "error: unrecoverable failure or crash"
    assert abs(score - 0.95) < 1e-5


def test_grade_empty():
    matcher = ControlledToolMatcher({})
    assert matcher.grade("test", []) == ("", 0.0)


def test_score_raw_does_not_collapse_on_selection_rejection():
    """Ranking scores must ignore the selection head that gates _score."""
    sess = MockSession(logits=[-2.0, 2.0], selection_score=0.20)
    tok = MockTokenizer()
    matcher = ToolMatcher(sess, tok, inputs=["input_ids", "attention_mask"])
    assert matcher._score("request", "tool: description") == 0.0
    expected_prob = 1.0 / (1.0 + math.exp(-4.0))
    assert abs(matcher._score_raw("request", "tool: description") - expected_prob) < 1e-4


def test_score_raw_matches_score_when_selection_accepts():
    sess = MockSession(logits=[-2.0, 2.0], selection_score=0.85)
    tok = MockTokenizer()
    matcher = ToolMatcher(sess, tok, inputs=["input_ids", "attention_mask"])
    assert matcher._score_raw("request", "tool: description") == pytest.approx(
        matcher._score("request", "tool: description")
    )


def test_score_raw_single_output_session():
    sess = MockSession(logits=[-1.0, 1.0], selection_score=None)
    tok = MockTokenizer()
    matcher = ToolMatcher(sess, tok, inputs=["input_ids", "attention_mask"])
    expected_prob = 1.0 / (1.0 + math.exp(-2.0))
    assert abs(matcher._score_raw("request", "tool: description") - expected_prob) < 1e-4


def test_select_ranking_ignores_selection_collapse():
    """Out-of-domain routing: selection head rejects every candidate, raw
    probabilities still rank correctly. The gated select scored all-zero here."""
    sess = MockSession(logits=[2.0, -2.0], selection_score=0.05)
    tok = MockTokenizer()
    matcher = ToolMatcher(sess, tok, inputs=["input_ids", "attention_mask"])
    best_key, conf = matcher.select(
        "deploy the service", {"a": "a: alpha", "b": "b: beta"}, calibrated=False)
    assert best_key == "a"  # logits [2, -2] -> prob ~0.018 for both, tie -> first
    assert conf == pytest.approx(0.5)


def test_select_with_status_flags_tied_low_scores():
    matcher = ControlledToolMatcher({"x": 0.20, "y": 0.18})
    key, conf, status = matcher.select("text", ["x", "y"], with_status=True)
    assert key in ("x", "y")
    assert status == "ambiguous"


def test_select_with_status_ok_on_clear_winner():
    matcher = ControlledToolMatcher({"x": 0.95, "y": 0.10})
    key, conf, status = matcher.select("text", ["x", "y"], with_status=True)
    assert key == "x"
    assert status == "ok"


def test_select_default_return_is_pair():
    matcher = ControlledToolMatcher({"x": 0.95, "y": 0.10})
    result = matcher.select("text", ["x", "y"])
    assert len(result) == 2


def test_select_with_status_single_candidate_ok():
    matcher = ControlledToolMatcher({"only": 0.77})
    key, score, status = matcher.select("test", ["only"], with_status=True)
    assert key == "only"
    assert status == "ok"


def test_grade_uses_ungated_scores():
    """grade must rank on raw probabilities even when the gate rejects."""
    sess = MockSession(logits=[2.0, -2.0], selection_score=0.05)
    tok = MockTokenizer()
    matcher = ToolMatcher(sess, tok, inputs=["input_ids", "attention_mask"])
    anchor, score = matcher.grade("text", ["low: trivial", "high: severe"])
    assert anchor == "low: trivial"
    assert score > 0.0  # ungated: ~0.018, not the gated 0.0


def test_grade_non_monotone_mock_scores():
    matcher = ControlledToolMatcher({
        "p1": 0.10,
        "p2": 0.90,
        "p3": 0.40,
    })
    anchor, score = matcher.grade("text", ["p1", "p2", "p3"])
    assert anchor == "p2"
    assert abs(score - 0.90) < 1e-5


def test_select_tiebreak_prefers_key_named_in_text():
    """Two statistically tied candidates: the key the text itself names wins."""
    matcher = ControlledToolMatcher({"devops": 0.30, "python": 0.28})
    key, _conf, status = matcher.select("devops", list(matcher._scores), with_status=True)
    assert key == "devops"
    assert status == "ambiguous"


def test_select_tiebreak_no_key_in_text_keeps_argmax():
    matcher = ControlledToolMatcher({"alpha": 0.30, "beta": 0.28})
    key, _conf, status = matcher.select("deploy the service", ["alpha", "beta"], with_status=True)
    assert key == "alpha"  # argmax; nothing in the text names a key
    assert status == "ambiguous"


def test_select_clear_winner_not_flagged_ambiguous_by_tiebreak():
    matcher = ControlledToolMatcher({"devops": 0.95, "python": 0.10})
    key, conf, status = matcher.select(
        "restart nginx on the host", ["devops", "python"], with_status=True)
    assert key == "devops"
    assert status == "ok"


# --- Live adversarial suites (shipped artifacts; skip when absent) ---

ROUTING_DOMAINS_10 = {
    "math": (
        "math.solve: Solve mathematical equations and numeric computation: "
        "quadratic and polynomial roots, derivatives, integrals, matrix "
        "algebra, and arithmetic."
    ),
    "python": (
        "python.dev: Write, debug, and review Python code: functions, "
        "scripts, algorithms, libraries, and error fixes."
    ),
    "rust": (
        "rust.dev: Write and debug Rust programs: cargo builds, the borrow "
        "checker, lifetimes, and systems code."
    ),
    "sql": (
        "sql.query: Write and run SQL for relational databases: SELECT "
        "queries, joins, indexes, migrations, and stored procedures."
    ),
    "devops": (
        "devops.run: Operate servers and infrastructure: restart services, "
        "systemd and nginx, docker containers, kubernetes deployments, and "
        "checking logs and disk space."
    ),
    "git": (
        "git.ops: Version control with git: commits, branches, rebases, "
        "merges, stash, pull requests, and undoing changes."
    ),
    "docs": (
        "docs.write: Write technical documentation: README files, API "
        "references, setup guides, and usage examples."
    ),
    "security": (
        "security.audit: Find and fix security issues: vulnerabilities, "
        "injection, hardcoded secrets, CVEs, and unsafe code."
    ),
    "translate": (
        "translate.lang: Translate text between human languages, preserving "
        "names and technical terms."
    ),
    "creative": (
        "creative.write: Write creative fiction: stories, poems, scripts, "
        "and imaginative prose."
    ),
}

# Pinned on the SERVED select surface — re-measured 2026-10-11 on the
# wired select-rank head (owner wiring decree,
# zero-shot-select-r2: models/select-rank-int8.onnx +
# its own band-tau calibration). Bands are absolute with +/- 0.06
# headroom so a regressed artifact cannot pass silently. The pre-wiring
# bands (recorded 2026-10-06 on the guard rank split, re-verified
# 2026-10-07) pinned the legacy surface: MIVIA_SELECT_RANK_HEAD=legacy
# serves those bytes again and will NOT satisfy these bands — that is
# the honest rollback signature, not a regression.
LIVE_ROUTING_CASES = [
    ("Calculate the integral of 3x^2 from 0 to 2", "math", 0.3685, 0.4885),
    ("python", "python", 0.3116, 0.4316),
    ("$ cargo build --release\nerror[E0507]: cannot move out of `ctx.config` which is "
     "behind a shared reference\nhelp: consider borrowing it; the function should "
     "take ownership and clone first", "rust", 0.2259, 0.3459),
    ("Restart the nginx service on the staging host and verify it came back up",
     "devops", 0.2521, 0.3721),
    ("git", "git", 0.2647, 0.3847),
    ("```markdown\n## API\n\nPOST /v1/items\n```\nexpand this stub into full API "
     "reference documentation with request and response examples", "docs", 0.3233, 0.4433),
    ("User report (de): \"Beim Speichern der Datei ist ein Fehler aufgetreten.\"\n"
     "I need the German and French versions of every UI string in this table, "
     "translated", "translate", 0.2479, 0.3679),
    ("Write a detective story where a grumpy sysadmin hunts down a mysterious "
     "memory leak through the night", "creative", 0.2353, 0.3553),
    ("Scan this Express route for SQL injection and secrets leakage vulnerabilities",
     "security", 0.3615, 0.4815),
    ("Write a README section documenting the config file format and all options",
     "docs", 0.3802, 0.5002),
    ("devops", "devops", 0.2829, 0.4029),
    ("story", "creative", 0.2372, 0.3572),
]


def _select_abstain_tau() -> float:
    """The conformal threshold the SERVED select surface actually applies
    (review fix: the abstain assertion must check the real rule, not be
    tautological). Head-aware since the 2026-10-11 wiring: the active
    calibrator is the wired select-rank head's when it loads, the legacy
    guard-rank calibration otherwise."""
    from mivia_small_json import tool_match as _tm

    cal = _tm.load_select_calibrator()
    assert cal is not None, "no select calibrator served"
    tau = _tm._select_abstain_tau(cal, len(ROUTING_DOMAINS_10))
    assert tau is not None
    return float(tau)


def test_live_routing_10_domains_messy_and_terse():
    """Adversarial routing: clean, terse, messy (traces, fences, preambles).

    Measured on the SERVED select surface — the wired select-rank head
    (owner wiring decree 2026-10-11, models/select-rank-int8.onnx): 9 of
    the 12 cases win, and the three misses are named below so a *new*
    miss fails here. The wiring FIXED all three legacy-surface misses
    (the markdown-API stub routes docs, the de-report routes translate,
    the security scan routes security) and introduced three new ones at
    the terse/ambiguous end (the cargo build error abstains toward git,
    the bare "git" lands math, the bare "devops" lands git) — 9/12
    either way, different texture; every number measured on the final
    tree. The legacy surface (48c43bce bytes, pre-decree) reproduces its
    own recorded behavior under MIVIA_SELECT_RANK_HEAD=legacy.
    """
    matcher = load_matcher()
    if matcher is None:
        pytest.skip("tool-match model artifact not available")

    known_miss_prefixes = (
        "$ cargo build --release",
        "git",
        "devops",
    )
    wrong = []
    for prompt, expected, lo, hi in LIVE_ROUTING_CASES:
        key, conf, status = matcher.select(prompt, ROUTING_DOMAINS_10, with_status=True)
        assert lo <= conf <= hi, f"{prompt[:50]!r} confidence {conf:.4f} outside [{lo}, {hi}]"
        # QW-C (2026-10-09): the conformal abstain supersedes the ambiguity
        # flag (abstain = calibrated < tau); the regression signal now
        # lives in the per-case calibrated bands above.
        if status == "abstain":
            tau = _select_abstain_tau()
            assert conf < tau, (prompt[:50], conf, tau)
        else:
            assert status in ("ok", "ambiguous")
        if key != expected:
            wrong.append((prompt, expected, key))
    assert len(wrong) <= 3, f"routing regressed: {wrong}"
    for prompt, _expected, got in wrong:
        assert prompt.startswith(known_miss_prefixes), (
            f"a NEW miss appeared: {prompt[:60]!r} routed to {got!r}")


def test_live_routing_ambiguity_status_flags_close_calls():
    """A measured near-tie is flagged ambiguous, not passed as confident.

    Measured on the committed configuration: the Rust build-error prompt is the
    near-tie (top-2 margin 0.0012), so the flag has to fire there. An earlier
    version used the integral prompt, which was a near-tie only on a gitignored
    local cascade.
    """
    matcher = load_matcher()
    if matcher is None:
        pytest.skip("tool-match model artifact not available")

    prompt = ("$ cargo build --release\nerror[E0507]: cannot move out of "
              "`ctx.config` which is behind a shared reference")
    key, _conf, status = matcher.select(prompt, ROUTING_DOMAINS_10, with_status=True)
    # QW-C (2026-10-09): abstain supersedes ambiguous when the calibrated
    # confidence sits under tau; the near-tie must at least not read "ok".
    assert status in ("ambiguous", "abstain"), (
        f"top-2 margin no longer close for {key!r}")


SEVERITY_ANCHORS_LIVE = [
    "info level: routine informational messages about normal successful "
    "operation and expected events.",
    "warning level: recoverable abnormal conditions that may need attention "
    "but are not failures.",
    "error level: operation failures, faults, and errors that disrupted a "
    "task or transaction.",
    "critical level: severe crashes, fatal errors, data loss, or complete "
    "system failure requiring immediate action.",
]

# Pinned implicit-severity LogHub dev rows (two per tier; level token absent
# from the text). Measured on the shipped ordinal grade artifact 2026-10-06:
# all correct, scores 0.76-0.99.
LIVE_GRADE_SAMPLES = [
    ("[Mon Dec 05 18:20:51 2005] [notice] jk2_init() Found child 6670 in "
     "scoreboard slot 7", 0),
    ("[Sun Dec 04 20:16:10 2005] [notice] jk2_init() Found child 1914 in "
     "scoreboard slot 8", 0),
    ("2015-07-29 19:33:35,800 - WARN  [SendWorker:188978561024:"
     "QuorumCnxManager$SendWorker@679] - Interrupted while waiting for message "
     "on queue", 1),
    ("2015-07-29 19:34:48,617 - WARN  [SendWorker:188978561024:"
     "QuorumCnxManager$SendWorker@679] - Interrupted while waiting for message "
     "on queue", 1),
    ("Jul  7 11:48:06 calvisitor-10-105-162-178 com.apple.WebKit.WebContent"
     "[32778]: [11:48:06.869] FigAgglomeratorSetObjectForKey signalled "
     "err=-16020 (kFigStringConformerError_ParamErr) (NULL key) at "
     "/Library/Caches/com.apple.xbs/Sources/CoreMedia/"
     "CoreMedia-1731.15.207/Prototypes/LegibleOutput/FigAgglomerator.c "
     "line 92", 2),
    ("Jul  5 17:23:22 calvisitor-10-105-163-9 com.apple.AddressBook."
     "InternetAccountsBridge[36248]: dnssd_clientstub ConnectToServer: "
     "connect() failed path:/var/run/mDNSResponder Socket:4 Err:-1 Errno:1 "
     "Operation not permitted", 2),
    ("Jul 18 04:03:25 combo logrotate: ALERT exited abnormally with [1]", 3),
    ("Jul 12 04:03:42 combo logrotate: ALERT exited abnormally with [1]", 3),
]

_TIER_FROM_ANCHOR = {"info": 0, "warning": 1, "error": 2, "critical": 3}


def test_live_grade_ordinal_artifact_pinned_severity():
    """grade on the dedicated ordinal artifact: pinned implicit-severity rows."""
    grade = load_grade_matcher()
    if grade is None:
        pytest.skip("grade model artifact not available")

    for text, expected in LIVE_GRADE_SAMPLES:
        anchor, score = grade.grade(text, SEVERITY_ANCHORS_LIVE)
        pred = _TIER_FROM_ANCHOR[anchor.split(" ")[0]]
        assert pred == expected, f"{text[:50]!r} graded {pred}, expected {expected}"
        assert score >= 0.70, f"{text[:50]!r} grade score {score:.4f} below 0.70"


# --- Tier-set decode (aps_tier_set / grade with_set=True) ---


def test_aps_tier_set_confident_scores_are_singleton():
    from mivia_small_json.tool_match import aps_tier_set

    # p_top1 ~ 0.99 after softmax: sum-above(top2) exceeds the quantile.
    scores = [0.99, 0.0006, 0.0006, 0.0006]
    assert aps_tier_set(scores, 0.30) == [0]


def test_aps_tier_set_collapsed_scores_pair_adjacent_tiers():
    from mivia_small_json.tool_match import aps_tier_set

    # All-collapsed scores -> uniform distribution -> {top1, top2}.
    scores = [0.0006, 0.0006, 0.0006, 0.0006]
    assert aps_tier_set(scores, 0.30) == [0, 1]


def test_aps_tier_set_is_contiguous_and_contains_argmax():
    from mivia_small_json.tool_match import aps_tier_set

    # Mid tier dominates: set must be contiguous around the argmax.
    scores = [0.05, 0.60, 0.30, 0.05]
    s = aps_tier_set(scores, 0.60)
    assert 1 in s
    assert s == list(range(min(s), max(s) + 1))


def test_aps_tier_set_high_quantile_grows_set():
    from mivia_small_json.tool_match import aps_tier_set

    scores = [0.99, 0.0006, 0.0006, 0.0006]
    assert aps_tier_set(scores, 0.30) == [0]
    assert aps_tier_set(scores, 0.999) == [0, 1, 2, 3]


def test_aps_tier_set_edge_cases():
    from mivia_small_json.tool_match import aps_tier_set

    assert aps_tier_set([], 0.30) == []
    assert aps_tier_set([0.42], 0.30) == [0]


def test_grade_with_set_default_return_unchanged():
    matcher = ControlledToolMatcher({"low": 0.90, "high": 0.10})
    result = matcher.grade("text", ["low", "high"])
    assert len(result) == 2
    anchor, score, tier_set = matcher.grade("text", ["low", "high"], with_set=True)
    assert anchor == "low"
    assert tier_set == [0]


def test_grade_with_set_uses_live_quantile_constant():
    from mivia_small_json.tool_match import GRADE_SET_QUANTILE, aps_tier_set

    matcher = ControlledToolMatcher({"a": 0.0006, "b": 0.0006, "c": 0.0006, "d": 0.0006})
    _anchor, _score, tier_set = matcher.grade("text", ["a", "b", "c", "d"], with_set=True)
    assert tier_set == aps_tier_set([0.0006] * 4, GRADE_SET_QUANTILE)


def test_live_grade_tier_set_covers_gold_on_pinned_samples():
    """The tier-set decode must keep the gold tier in the set on the pinned
    LogHub samples (all confident rows -> singleton containing gold)."""
    grade = load_grade_matcher()
    if grade is None:
        pytest.skip("grade model artifact not available")

    for text, expected in LIVE_GRADE_SAMPLES:
        _anchor, _score, tier_set = grade.grade(
            text, SEVERITY_ANCHORS_LIVE, with_set=True)
        assert expected in tier_set, f"{text[:50]!r} set {tier_set} misses {expected}"
        assert len(tier_set) == 1  # in-domain rows decode as singletons


def test_live_grade_tier_set_novel_register_coverage_band():
    """Measured on the shipped v4 artifact at the pinned quantile 0.30:
    gold-in-set on the frozen novel-register set is 55.0% (22/40) at mean
    set size 1.55 (in-domain LogHub stays singletons). Band keeps a
    regressed decode from passing silently."""
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    from bench_zero_shot_primitives import NOVEL_REGISTER_GRADE_CASES

    from mivia_small_json.tool_match import GRADE_SET_QUANTILE

    grade = load_grade_matcher()
    if grade is None:
        pytest.skip("grade model artifact not available")

    covered = 0
    for text, gold in NOVEL_REGISTER_GRADE_CASES:
        _anchor, _score, tier_set = grade.grade(
            text, SEVERITY_ANCHORS_LIVE, with_set=True)
        assert tier_set == list(range(min(tier_set), max(tier_set) + 1))
        covered += gold in tier_set
    assert 0.45 <= covered / len(NOVEL_REGISTER_GRADE_CASES) <= 0.65, (
        f"novel-register set coverage {covered}/40 outside the pinned band"
    )
    assert GRADE_SET_QUANTILE == 0.30


def test_live_grade_falls_back_to_guard_matcher_shape():
    """Any matcher instance answers grade() calls (decode is matcher-agnostic)."""
    matcher = load_matcher()
    if matcher is None:
        pytest.skip("tool-match model artifact not available")

    anchor, score = matcher.grade("Add 12 and 38 together", SEVERITY_ANCHORS_LIVE)
    assert anchor in SEVERITY_ANCHORS_LIVE
    assert 0.0 <= score <= 1.0


# Near-miss verify: measured 2026-10-06 (then-cascade, now the shipped
# gate+rank split). The semantic
# gate accepts parameter-starved requests (0.9993) — that is the boundary
# verify() cannot see and the deterministic fillability gate's job. Bands
# +/- 0.05.
LIVE_NEAR_MISS_CASES = [
    ("Calculate the sum of 42",
     "calculate_sum: Adds two numbers a and b and returns the total.",
     True, 0.95, 1.0),
    ("Calculate the sum of 42 and 58",
     "calculate_sum: Adds two numbers a and b and returns the total.",
     True, 0.95, 1.0),
    ("Book me a table for dinner sometime soon",
     "find_restaurant: Books a table at a restaurant for a specific ISO date (YYYY-MM-DD).",
     True, 0.95, 1.0),
    ("Book me a table for dinner on 2026-10-15",
     "find_restaurant: Books a table at a restaurant for a specific ISO date (YYYY-MM-DD).",
     True, 0.95, 1.0),
    # Known recorded miss (Arm C rank promotion 48c43bce, 2026-10-07): the
    # session-shaped rank now accepts this vague, parameter-starved request
    # at 0.9065 — the composite fillability gate is what still catches it.
    ("I want to convert my US dollars to euros for my vacation",
     "get_currency_symbol: Returns the currency symbol string for a country code.",
     True, 0.85, 0.95),
    ("What is the currency symbol for Japan?",
     "get_currency_symbol: Returns the currency symbol string for a country code.",
     True, 0.95, 1.0),
    ("Transfer money from my savings account to my checking account",
     "get_exchange_rate: Returns the exchange rate between two currency codes.",
     False, 0.0, 0.05),
    ("What is the exchange rate from USD to EUR today?",
     "get_exchange_rate: Returns the exchange rate between two currency codes.",
     True, 0.95, 1.0),
    ("Play my favorite jazz playlist",
     "calculate_sum: Adds two numbers a and b and returns the total.",
     False, 0.0, 0.05),
]


def test_live_verify_near_miss_bands():
    """Near-miss boundaries: keyword-overlap negatives rejected, semantic
    matches (including parameter-starved ones) accepted — the measured
    frontier that motivates the composite rule."""
    from mivia_small_json.tool_match import validate_fillability

    matcher = load_matcher()
    if matcher is None:
        pytest.skip("tool-match model artifact not available")

    for request, tool, expect_verified, lo, hi in LIVE_NEAR_MISS_CASES:
        verified, score = matcher.verify(request, tool)
        assert verified is expect_verified, f"{request[:45]!r} -> {score:.4f}"
        assert lo <= score <= hi, f"{request[:45]!r} score {score:.4f} outside [{lo}, {hi}]"

    # Composite rule: parameter-starved semantic matches are caught by the
    # deterministic fillability gate ("Calculate the sum of 42" has one
    # number; calculate_sum requires two).
    tool_dict = {
        "name": "calculate_sum",
        "description": "Adds two numbers a and b and returns the total.",
        "parameters": {
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    }
    fillable, reason, _missing = validate_fillability(tool_dict, "Calculate the sum of 42")
    assert fillable is False
    assert reason == "missing_arguments"
    _verified, score = matcher.verify(
        "Calculate the sum of 42", tool_dict["name"] + ": " + tool_dict["description"])
    assert score >= 0.95  # semantics still accept; the composite rejects


def test_live_matcher_zero_shot_primitives():
    matcher = load_matcher()
    if matcher is None:
        pytest.skip("tool-match model artifact not available")

    # 1. select between math vs weather
    candidates = {
        "math": "solve_quadratic: Find the roots of a quadratic equation. Returns both roots.",
        "weather": "get_weather: Get local weather forecast.",
    }
    prompt = "Can you calculate the roots of a quadratic equation where a=2, b=5, and c=3?"
    choice, conf = matcher.select(prompt, candidates)
    assert choice == "math"
    assert conf > 0.50

    # 2. verify (positive and negative)
    verified, score = matcher.verify(
        "Can you calculate the roots of a quadratic equation?",
        "solve_quadratic: Find the roots of a quadratic equation. Returns both roots.",
    )
    assert verified is True
    assert score > 0.80

    neg_verified, neg_score = matcher.verify(
        "Can you calculate the roots of a quadratic equation?",
        "get_weather: Get local weather forecast.",
    )
    assert neg_verified is False
    assert neg_score < 0.20

    # 3. grade severity / complexity
    anchors = [
        "weather.forecast: Weather prediction.",
        "math.factorial: Calculate factorial of number.",
    ]
    level, score = matcher.grade("Calculate the factorial of 5 using math functions", anchors)
    assert level == "math.factorial: Calculate factorial of number."
    assert score > 0.80


# --- Prompt-to-Model Routing Pipeline Tests ---


def test_prompt_to_model_routing_pipeline_controlled():
    """Unit test: prompt routing across specialized candidate models using ControlledToolMatcher."""
    models = {
        "math_expert": "Solve mathematical equations, algebra, and calculus.",
        "code_expert": "Generate python code, algorithms, and debug functions.",
        "sql_expert": "Run SQL queries, joins, aggregations, and database mutations.",
        "sysadmin_expert": "Manage Linux servers, bash shell commands, and docker containers.",
    }
    mock_scores = {
        "math_expert": 0.95,
        "code_expert": 0.10,
        "sql_expert": 0.05,
        "sysadmin_expert": 0.02,
    }
    matcher = ControlledToolMatcher(mock_scores)
    prompt = "Find all real roots of 2x^2 + 5x - 3 = 0"
    choice, conf = matcher.select(prompt, models)
    assert choice == "math_expert"
    assert conf > 0.40


def test_prompt_to_model_routing_pipeline_live():
    """End-to-end test: route prompts to specialized target models via cross-encoder."""
    matcher = load_matcher()
    if matcher is None:
        pytest.skip("tool-match model artifact not available")

    models = {
        "math_model": (
            "math.solve: Solve complex mathematical equations, calculate roots of quadratic "
            "formulas, calculus, and algebra."
        ),
        "code_model": (
            "code.generate: Write Python code, implement algorithms, debug functions, "
            "and software programs."
        ),
        "sql_model": (
            "sql.query: Run SQL queries, database joins, schema mutations, and data aggregations."
        ),
        "sysadmin_model": (
            "system.manage: Linux bash commands, systemctl services, restart docker containers, "
            "and server infrastructure."
        ),
    }

    test_cases = [
        ("Calculate the roots of a quadratic equation where a=2, b=5, and c=3", "math_model"),
        (
            "Write a Python function to compute the Fibonacci sequence using memoization",
            "code_model",
        ),
        (
            "Run a SQL query to select department and average salary from employees table",
            "sql_model",
        ),
        ("Restart the docker container and check nginx service logs", "sysadmin_model"),
    ]

    for user_prompt, expected_model in test_cases:
        chosen_model, confidence = matcher.select(user_prompt, models)
        assert chosen_model == expected_model, (
            f"Prompt {user_prompt!r} routed to {chosen_model!r}, expected {expected_model!r}"
        )
        # Known recorded trade (Arm C rank promotion 48c43bce, 2026-10-07):
        # the sysadmin case routes correctly but at 0.2522 (was >0.40) —
        # the session-shaped rank compresses select confidences.
        assert confidence > 0.20, (
            f"Expected confident routing for {expected_model!r}, got {confidence:.4f}"
        )


def test_prompt_routing_fallback_low_confidence():
    """Verify that irrelevant prompts yield low scores across specialized models."""
    matcher = load_matcher()
    if matcher is None:
        pytest.skip("tool-match model artifact not available")

    specialized_models = {
        "math_model": "math.solve: Solve calculus, algebra, and numeric equations.",
        "code_model": "code.generate: Write Python algorithms and data structures.",
    }
    # An out-of-domain query (cooking recipe / travel)
    chitchat_prompt = "Can you give me a recipe for chocolate chip cookies?"
    choice, conf = matcher.select(chitchat_prompt, specialized_models)
    # Neither model should be strongly entailed
    accepted_math, score_math = matcher.verify(
        chitchat_prompt, specialized_models["math_model"], threshold=0.50
    )
    accepted_code, score_code = matcher.verify(
        chitchat_prompt, specialized_models["code_model"], threshold=0.50
    )
    assert accepted_math is False
    assert accepted_code is False
    assert score_math < 0.20
    assert score_code < 0.20


def test_prompt_model_capability_verification():
    """Verify semantic entailment between a user request and specific model capabilities."""
    matcher = load_matcher()
    if matcher is None:
        pytest.skip("tool-match model artifact not available")

    code_ask = "Write a Python function to compute the Fibonacci sequence using memoization"
    code_cap = (
        "code.generate: Write Python code, implement algorithms, debug functions, "
        "and software programs."
    )
    sql_cap = "sql.query: Run SQL queries, database joins, schema mutations, and data aggregations."

    # Code ask should verify against code capability
    acc_code, score_code = matcher.verify(code_ask, code_cap, threshold=0.50)
    assert acc_code is True
    assert score_code > 0.80

    # Code ask should fail verification against SQL capability
    acc_sql, score_sql = matcher.verify(code_ask, sql_cap, threshold=0.50)
    assert acc_sql is False
    assert score_sql < 0.20


# --- RPS ordinal conformal decode (rps_tier_set) ---


def test_rps_tier_set_confident_top_tier_is_singleton():
    from mivia_small_json.tool_match import rps_tier_set

    # A dominant top tier has a near-zero ranked probability score for tier 0.
    scores = [5.0, -5.0, -5.0, -5.0]
    assert rps_tier_set(scores, 0.05) == [0]


def test_rps_tier_set_is_contiguous_and_contains_argmax():
    from mivia_small_json.tool_match import rps_tier_set

    scores = [0.05, 0.60, 0.30, 0.05]
    s = rps_tier_set(scores, 0.5)
    assert 1 in s
    assert s == list(range(min(s), max(s) + 1))


def test_rps_tier_set_high_quantile_grows_set():
    from mivia_small_json.tool_match import rps_tier_set

    scores = [2.0, 0.0, 0.0, 0.0]
    small = rps_tier_set(scores, 0.05)
    large = rps_tier_set(scores, 5.0)
    assert len(large) >= len(small)
    assert 0 in small
