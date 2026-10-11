"""Regression tests for confirmed tool_match bug findings F1-F9.

Each test was written red against the buggy behavior and turned green by
the fix. Deterministic only: no artifact scores are pinned, no timing.
"""
from __future__ import annotations

import json

import pytest

from mivia_small_json import integrity, tool_match

ANCHORS = [
    "info level: routine messages.",
    "warning level: recoverable conditions.",
    "error level: failures.",
    "critical level: outages.",
]


def _catalogue_text(tools: list[dict], request: str) -> str:
    return json.dumps(tools) + "\n\nUser request:" + request


# ---------------------------------------------------------------- F1


class TestF1TierDecisionArgmax:
    def test_default_argmax_uses_first_tier(self):
        """No argmax_index -> the argmax position, not a TypeError->empty ask."""
        d = tool_match.tier_decision([2], ANCHORS)
        assert d["action"] == "act"
        assert d["tier_index"] == 2
        assert d["tier"] == ANCHORS[2]

    def test_explicit_argmax_index_unchanged(self):
        d = tool_match.tier_decision([2], ANCHORS, argmax_index=2)
        assert d["action"] == "act"
        assert d["tier_index"] == 2

    def test_string_argmax_index_coerced(self):
        d = tool_match.tier_decision([1, 2], ANCHORS, argmax_index="2")
        assert d["action"] == "act"
        assert d["tier_index"] == 2
        assert d["tier"] == ANCHORS[2]

    def test_garbage_index_does_not_discard_valid_tier_set(self):
        """A non-numeric argmax falls back to the set's first tier; the
        tier set itself is still valid and acted on."""
        d = tool_match.tier_decision([1, 2], ANCHORS, argmax_index="oops")
        assert d["action"] == "act"
        assert d["tier_index"] == 1
        assert d["tier_set"] == [1, 2]

    def test_garbage_tier_set_still_asks(self):
        assert tool_match.tier_decision("bad", ANCHORS,
                                        argmax_index=None)["action"] == "ask"
        assert tool_match.tier_decision(None, None)["action"] == "ask"
        assert tool_match.tier_decision([0, 5], ANCHORS,
                                        argmax_index=0)["action"] == "ask"
        assert tool_match.tier_decision([], ANCHORS,
                                        argmax_index=0)["action"] == "ask"


# ---------------------------------------------------------------- F2


class TestF2ApsTierSetDegenerate:
    def test_quantile_below_zero_falls_back_to_argmax(self):
        scores = [0.1, 3.0, 0.2]
        assert tool_match.aps_tier_set(scores, -1.0) == [1]

    def test_nan_quantile_falls_back_to_argmax(self):
        scores = [0.1, 3.0, 0.2]
        assert tool_match.aps_tier_set(scores, float("nan")) == [1]

    def test_q_zero_keeps_argmax_singleton(self):
        scores = [0.1, 3.0, 0.2]
        assert tool_match.aps_tier_set(scores, 0.0) == [1]

    def test_normal_quantile_behavior_unchanged(self):
        # A dominant argmax and a tiny neighbor pair up at a generous
        # quantile; ties/ordering contract stays.
        scores = [0.1, 3.0, 2.9]
        s = tool_match.aps_tier_set(scores, 0.9)
        assert s[0] <= 1 <= s[-1] and 1 in s


# ---------------------------------------------------------------- F3


class TestF3UrlWordBoundary:
    def _tool(self, desc):
        return {
            "name": "t",
            "parameters": {
                "properties": {"p": {"type": "string", "description": desc}},
                "required": ["p"],
            },
        }

    @pytest.mark.parametrize("fn", ["validate_fillability", "fillability_veto"])
    def test_curl_in_description_is_not_a_url_requirement(self, fn):
        tool = self._tool("runs curl to fetch the resource")
        ok, reason, missing = getattr(tool_match, fn)(
            tool, "fetch the resource for me please")
        # The F3 law: 'curl' in a parameter DESCRIPTION must never
        # phantom-miss as a URL requirement — the url limb stays silent
        # here and 'p' is never reported missing-as-a-url. The veto may
        # still honestly clarify the request on CONTENT grounds: since
        # 2026-10-10 (A4c) polite fillers are veto-stopwords, so a
        # description-echo request that names no value for 'p' fires
        # underspecified_query — the same law that stops "just delete
        # the database" from executing on a guessed argument.
        if fn == "validate_fillability":
            assert ok, f"{fn} phantom-missing on curl: {missing}"
            assert missing == []
        else:
            assert (ok and missing == []) or (
                reason == "underspecified_query" and missing == ["p"]), \
                f"{fn}: url phantom-miss or wrong reason: {reason} {missing}"

    @pytest.mark.parametrize("fn", ["validate_fillability", "fillability_veto"])
    def test_real_url_parameter_still_required(self, fn):
        tool = self._tool("the url to fetch")
        ok, _reason, missing = getattr(tool_match, fn)(
            tool, "fetch the resource for me please")
        assert not ok
        assert missing == ["p"]


# ---------------------------------------------------------------- F4


class TestF4ValidateTypeGate:
    def test_declared_string_param_never_numeric_in_validate(self):
        """'country' contains 'count'; a declared-string parameter must not
        be treated as numeric by validate_fillability (parity with the
        veto's type gate)."""
        tool = {
            "name": "lookup",
            "parameters": {
                "properties": {"country": {"type": "string"}},
                "required": ["country"],
            },
        }
        fillable, reason, missing = tool_match.validate_fillability(
            tool, "what is the capital of Italy please")
        assert fillable and reason == "" and missing == []

    def test_declared_numeric_still_checked(self):
        tool = {
            "name": "lookup",
            "parameters": {
                "properties": {"count": {"type": "number"}},
                "required": ["count"],
            },
        }
        fillable, reason, _missing = tool_match.validate_fillability(
            tool, "give me the count of things available")
        assert not fillable
        assert reason == "missing_arguments"


# ---------------------------------------------------------------- F5


class TestF5VetoCountsNumbersFirst:
    def _tool(self):
        return {
            "name": "calc",
            "parameters": {
                "properties": {"a": {"type": "number"},
                               "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        }

    def test_two_numbers_short_request_not_vetoed(self):
        ok, reason, _missing = tool_match.fillability_veto(
            self._tool(), "10 20")
        assert ok, f"veto fired ({reason}) despite two supplied numbers"

    def test_zero_numbers_short_request_still_underspecified(self):
        ok, reason, missing = tool_match.fillability_veto(
            self._tool(), "sum")
        assert not ok
        assert reason == "underspecified_query"
        assert set(missing) == {"a", "b"}

    def test_add_ten_and_twenty_still_passes(self):
        ok, _reason, _missing = tool_match.fillability_veto(
            self._tool(), "add 10 and 20")
        assert ok

    def test_one_number_two_required_still_underspecified(self):
        ok, reason, _missing = tool_match.fillability_veto(
            self._tool(), "10")
        assert not ok and reason == "underspecified_query"


# ---------------------------------------------------------------- F6


class TestF6GradeMatcherStickyFailure:
    def test_failed_explicit_path_does_not_poison_default(self, monkeypatch):
        from mivia_small_json import tool_match as tm

        fake = object()
        calls = []

        def fake_loader(path, *a, **k):
            calls.append(str(path))
            return fake if str(path) == str(tm.GRADE_MODEL) else None

        monkeypatch.setattr(tm, "load_single_matcher", fake_loader)
        monkeypatch.setattr(tm, "_GRADE_MATCHER", None)
        monkeypatch.setattr(tm, "_GRADE_LOADED", False)

        assert tm.load_grade_matcher(path="/nonexistent/x.onnx") is None
        assert tm.load_grade_matcher() is fake, (
            "a failed explicit-path attempt disabled the default path")

    def test_success_is_cached(self, monkeypatch):
        from mivia_small_json import tool_match as tm

        fake = object()
        monkeypatch.setattr(tm, "load_single_matcher", lambda *a, **k: fake)
        monkeypatch.setattr(tm, "_GRADE_MATCHER", None)
        monkeypatch.setattr(tm, "_GRADE_LOADED", False)
        assert tm.load_grade_matcher() is fake
        # Second call comes from the cache (loader swapped for a bomb).
        monkeypatch.setattr(
            tm, "load_single_matcher",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("reload")))
        assert tm.load_grade_matcher() is fake


# ---------------------------------------------------------------- F7


class _StubMatcher(tool_match.BaseToolMatcher):
    def __init__(self, score=0.9):
        self.score = score

    def _score(self, request, tool):
        return self.score

    def _score_raw(self, request, tool):
        return self.score


class TestF7CascadeBestMatch:
    TOOLS = [
        {"name": "get_weather", "description": "current weather"},
        {"name": "get_time", "description": "current time"},
    ]

    def test_cascade_best_match_does_not_raise(self):
        cascade = tool_match.CascadedToolMatcher(_StubMatcher(), _StubMatcher())
        tool, score = cascade.best_match(
            _catalogue_text(self.TOOLS, "what is the weather"))
        assert tool is not None and tool["name"] == "get_weather"
        assert score == 0.9

    def test_cascade_best_score_via_bench_construction(self):
        """The bench --cascaded-style construction: CascadedToolMatcher built
        directly, .best_score called (scripts/bench_tool_abstention.py)."""
        cascade = tool_match.CascadedToolMatcher(
            _StubMatcher(0.4), _StubMatcher(0.4))
        score = cascade.best_score(
            _catalogue_text(self.TOOLS, "what time is it"))
        assert score == 0.4

    def test_span_renders_like_toolmatcher(self):
        cascade = tool_match.CascadedToolMatcher(_StubMatcher(), _StubMatcher())
        assert cascade._span(self.TOOLS[0], "req", self.TOOLS) == \
            tool_match.render_tool(self.TOOLS[0])


# ---------------------------------------------------------------- F8


class TestF8ParseCatalogueTruncation:
    def _head(self, name_count, desc_texts):
        bits = []
        for i in range(name_count):
            obj = f'{{"name": "tool{i}", "description": "{desc_texts[i]}"}}' \
                if i < len(desc_texts) else \
                f'{{"name": "tool{i}", "description": "unterminated {i}'
            bits.append(obj)
        return "[" + ", ".join(bits)  # truncated: no closing bracket

    def test_misaligned_truncation_returns_no_pairs(self):
        text = (self._head(2, ["fully described one"])
                + "\n\nUser request: do the thing")
        tools, request = tool_match.parse_catalogue(text)
        assert request == "do the thing"
        # tool1's description was cut mid-string: no fabricated
        # tool1: <parameter-text> pair may appear.
        assert tools == []

    def test_aligned_truncation_still_pairs(self):
        text = (self._head(2, ["fully described zero",
                               "fully described one"])
                + "\n\nUser request: do the thing")
        tools, request = tool_match.parse_catalogue(text)
        assert tools == ["tool0: fully described zero",
                         "tool1: fully described one"]
        assert request == "do the thing"

    def test_no_parameter_text_leaks_into_pairs(self):
        head = ('[{"name": "get_weather", "description": "weather now",'
                ' "parameters": {"properties": {"url": '
                '{"type": "string", "description": "the url to query"'
                + "\n\nUser request: weather?")
        tools, _request = tool_match.parse_catalogue(head)
        joined = " ".join(tools)
        assert "the url to query" not in joined


# ---------------------------------------------------------------- F9


class TestF9MetaSidecarVerification:
    MODELS = None

    @pytest.fixture
    def meta_env(self, tmp_path, monkeypatch):
        """Copies a real meta + fresh sidecar into tmp_path and resets the
        loader caches."""
        src = tool_match.ROOT / "models" / "register-gate-meta.json"
        meta = tmp_path / "register-gate-meta.json"
        meta.write_bytes(src.read_bytes())
        sidecar = tmp_path / "register-gate-meta.json.sha256"
        sidecar.write_text(integrity.sha256_file(meta) + "\n")
        monkeypatch.setattr(tool_match, "REGISTER_GATE_MODEL",
                            tmp_path / "missing-gate.skops")
        monkeypatch.setattr(tool_match, "REGISTER_GATE_META", meta)
        monkeypatch.setattr(tool_match, "_REGISTER_GATE", None)
        monkeypatch.setattr(tool_match, "_REGISTER_GATE_TAU", None)
        monkeypatch.setattr(tool_match, "_REGISTER_GATE_LOADED", False)
        return meta, sidecar

    def test_meta_sidecars_exist_in_house_format(self):
        for name in ("select-calibration-meta.json",
                     "register-gate-meta.json",
                     "is-security-gate-meta.json"):
            sidecar = tool_match.ROOT / "models" / (name + ".sha256")
            digest, problem = integrity.expected_digest(sidecar)
            assert problem == "", f"{sidecar.name}: {problem}"
            assert digest == integrity.sha256_file(
                tool_match.ROOT / "models" / name), name

    def test_tampered_meta_is_refused(self, meta_env, monkeypatch):
        meta, sidecar = meta_env
        # Tamper AFTER the sidecar was written.
        data = json.loads(meta.read_text())
        data["tau"] = 0.01
        meta.write_text(json.dumps(data))
        gate, tau = tool_match.load_register_gate()
        assert gate is None and tau is None

    def test_untampered_meta_with_valid_sidecar_is_used(self, meta_env,
                                                        monkeypatch):
        meta, _sidecar = meta_env
        # The .skops body is missing in tmp_path, so the verified loader
        # returns None regardless; pin the meta-verification branch instead
        # via the helper every load site now routes through.
        assert tool_match._meta_verified(meta) is True

    def test_tampered_meta_fails_helper(self, meta_env):
        meta, sidecar = meta_env
        meta.write_text(meta.read_text() + " ")
        assert tool_match._meta_verified(meta) is False

    def test_missing_sidecar_is_graceful_for_dev_artifacts(self, meta_env):
        meta, sidecar = meta_env
        sidecar.unlink()
        assert tool_match._meta_verified(meta) is True

    def test_select_calibrator_tampered_meta_degrades(self, tmp_path,
                                                      monkeypatch):
        src = tool_match.ROOT / "models" / "select-calibration-meta.json"
        meta = tmp_path / "select-calibration-meta.json"
        meta.write_bytes(src.read_bytes())
        (tmp_path / "select-calibration-meta.json.sha256").write_text(
            integrity.sha256_file(meta) + "\n")
        meta.write_text(meta.read_text().replace("rank_sha256", "rank_sha"))
        # Owner wiring decree 2026-10-11: load_select_calibrator serves the
        # ACTIVE head's calibration, so this legacy-meta degrade check pins
        # the seam to the legacy surface (the tampered legacy meta is only
        # consulted there; on the wired tree the select-rank calibrator
        # serves and the legacy meta is dead code by construction).
        monkeypatch.setattr(tool_match, "select_rank_legacy", lambda: True)
        monkeypatch.setattr(tool_match, "_SELECT_CAL_META", meta)
        monkeypatch.setattr(tool_match, "_SELECT_CAL", None)
        monkeypatch.setattr(tool_match, "_SELECT_CAL_LOADED", False)
        assert tool_match.load_select_calibrator() is None

    # (the lang-gate tampered-meta degradation test retired with the
    # artifact: the public tree ships no lang-gate.skops — it was trained
    # on a CC-BY-NC corpus and is excluded from the open-source release.
    # The same meta-verification law stays covered by the register-gate
    # and select-calibrator tests above.)
