"""Gates for the canonical guard-eval-v2 measurement path.

The canonical per-category measurement lives in
``scripts/bench_tool_abstention.py`` (``--eval v2``);
``scripts/build_guard_eval.py`` ``measure`` redirects to it. These tests pin
the metric semantics on synthetic scores (no model runs) so the instrument
cannot silently change meaning.
"""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "guard-eval-v2.md"


def _load(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def bench():
    return _load("bench_tool_abstention")


def test_canonical_v2_helpers_exist(bench):
    assert bench.GUARD_EVAL_V2.name == "guard-eval-v2.jsonl"
    assert set(bench.V2_CATEGORIES) == {
        "single_turn_irrelevance", "single_turn_relevance",
        "multi_turn_miss_func", "multi_turn_miss_param", "ask_followup",
        "infeasible", "foreign_tool", "sibling_swap", "no_tool_needed",
    }
    for name in ("_catalogue_scores", "_operating_point", "measure_v2"):
        assert callable(getattr(bench, name)), f"missing {name}"


def test_operating_point_maximizes_abstention_at_the_floor(bench):
    neg = [0.1, 0.2, 0.9]
    pos = [0.9, 0.95, 0.99]
    thr, abst, kept = bench._operating_point(neg, pos, floor=100.0)
    assert kept >= 100.0
    assert thr == pytest.approx(0.9)
    assert abst == pytest.approx(100 * 2 / 3, abs=1e-6)


def test_catalogue_grouping_takes_max_over_listed_tools(bench):
    grouped = bench._catalogue_scores(
        [("cat#1", 0.2), ("cat#1", 0.8), ("cat#2", 0.3)])
    assert grouped == {"cat#1": 0.8, "cat#2": 0.3}


class _StubMatcher:
    """Scores by a fixed request -> score map (no model)."""

    def __init__(self, scores: dict):
        self._scores = scores
        self.calls = 0

    def best_score(self, text: str) -> float:
        self.calls += 1
        for request, score in self._scores.items():
            if request in text:
                return score
        raise AssertionError(f"unstubbed request in: {text[:80]}")


def _row(request: str, category: str, source: str, label: int = 0) -> dict:
    return {"request": request, "tool": {"name": "t.f", "description": "d"},
            "label": label, "category": category, "source": source,
            "split": "eval"}


def test_measure_v2_reports_per_category_abstention_retention(bench, capsys):
    rows = [
        _row("neg-a-one", "single_turn_irrelevance", "base#a"),
        _row("neg-a-two", "single_turn_irrelevance", "base#a"),
        _row("neg-b-one", "single_turn_irrelevance", "base#b"),
        _row("pos-a-call", "single_turn_relevance", "calls#c", label=1),
        _row("pos-b-call", "single_turn_relevance", "calls#d", label=1),
    ]
    stub = _StubMatcher({"neg-a-one": 0.10, "neg-a-two": 0.999,
                         "neg-b-one": 0.05,
                         "pos-a-call": 0.999, "pos-b-call": 0.998})
    payload = bench.measure_v2(rows, matcher=stub)
    # Catalogue max: base#a scores max(0.10, 0.999) = 0.999 (a false call).
    op = payload["operating_point"]
    assert op["retention"] >= 95.0
    by_cat = {c["category"]: c for c in payload["per_category"]}
    assert set(by_cat) == {"single_turn_irrelevance", "single_turn_relevance"}
    irrelevance = by_cat["single_turn_irrelevance"]
    assert irrelevance["catalogues"] == 2
    assert irrelevance["catalogue_abstention_at_op"] == pytest.approx(50.0)
    assert irrelevance["catalogue_retention_at_op"] == pytest.approx(50.0)
    relevance = by_cat["single_turn_relevance"]
    assert relevance["catalogue_retention_at_op"] == pytest.approx(100.0)
    printed = capsys.readouterr().out
    assert "single_turn_irrelevance" in printed
    assert stub.calls == len(rows)


def test_build_measure_redirects_to_the_canonical_path(bench):
    bge = _load("build_guard_eval")
    assert bge.measure is bench.measure_v2
    assert bge._operating_point is bench._operating_point
    assert bge._catalogues is bench._catalogue_scores
    assert set(bge.CATEGORIES) == set(bench.V2_CATEGORIES)


# (report-shape gate retired with reports/: the instrument record is
# not redistributed; the private tree keeps the gate.)
