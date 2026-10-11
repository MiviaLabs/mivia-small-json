"""Presence-contrast twins and the twin margin objective (Step 4).

Fast CPU-only unit tests for the fillability-contrast machinery:
- the numeric strip construction (`strip_required_values`),
- twin eligibility (`eligible_twin`: full must fill, stripped must miss),
- the eval-overlap contract (`overlap_decision`),
- twin-aware batching (`group_twins`) and the margin loss
  (`contrastive_margin_loss`).

No training, no network, no eval files: all fixtures are synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_tool_match_corpus as btc  # noqa: E402
import train_tool_match as ttm  # noqa: E402

TOOL = {
    "name": "weather.get",
    "description": "Get the weather for a place in a year.",
    "parameters": {
        "properties": {
            "location": {"type": "string", "description": "the place"},
            "year": {"type": "integer", "description": "the year"},
        },
        "required": ["location", "year"],
    },
}


def test_strip_removes_numerics():
    stripped = btc.strip_required_values("weather in Paris in 2021 please")
    assert stripped is not None
    assert "2021" not in stripped
    assert "Paris" in stripped


def test_strip_none_without_numerics():
    assert btc.strip_required_values("weather in Paris please") is None


def test_eligible_twin_needs_fillable_full_and_broken_strip():
    stripped = btc.eligible_twin("weather in Paris in 2021 please", TOOL)
    assert stripped is not None
    assert "2021" not in stripped
    # An already-unfillable request is not a contrast (label 1 dishonest).
    assert btc.eligible_twin("weather in Paris please", TOOL) is None
    # A tool with no required args can never break: no twin.
    no_req = {"name": "t.ping", "description": "ping",
              "parameters": {"properties": {}, "required": []}}
    assert btc.eligible_twin("ping 42 times", no_req) is None


def test_overlap_decision_catches_every_eval_shape():
    index = {
        "v2_requests": {"a stripped eval request"},
        "v2_pairs": {("other", "canon-tool")},
        "catalogue": {("third", "canon-tool")},
        "corpus": {("fourth", "stored span")},
    }
    assert btc.overlap_decision(
        "a stripped eval request", "canon-x", "span-x",
        index) == "eval_request_v2"
    assert btc.overlap_decision(
        "other", "canon-tool", "span-x", index) == "eval_pair_v2"
    assert btc.overlap_decision(
        "third", "canon-tool", "span-x", index) == "eval_pair_catalogue"
    assert btc.overlap_decision(
        "fourth", "canon-x", "stored span", index) == "training_pair_corpus"
    assert btc.overlap_decision(
        "clean request", "canon-x", "span-x", index) is None


def test_twin_split_matches_full_side_split():
    assert btc.twin_split("some request 42", "tool span") == \
        btc._split_of("some request 42", "tool span")


def test_group_twins_pairs_stripped_with_base_positive():
    rows = [
        {"request": "full 42", "tool": "span", "label": 1,
         "source": "bfcl_positives", "split": "train", "key": "K1"},
        {"request": "other 7", "tool": "span", "label": 1,
         "source": "bfcl_positives", "split": "train", "key": "K2"},
        {"request": "neg", "tool": "span", "label": 0,
         "source": "bfcl_foreign_negatives", "split": "train", "key": "K3"},
        {"request": "full", "tool": "span-missing",
         "label": 0, "source": "contrast_fillability_stripped",
         "split": "train", "key": "K4", "twin_key": "K1"},
        {"request": "orphan", "tool": "span-missing",
         "label": 0, "source": "contrast_fillability_stripped",
         "split": "train", "key": "K5", "twin_key": "MISSING"},
    ]
    groups, pair_of, role_of = ttm.group_twins(rows)
    paired = [g for g in groups if len(g) == 2]
    assert len(paired) == 1
    assert paired[0] == [0, 3]
    assert role_of[0] == "full" and role_of[3] == "stripped"
    assert pair_of[0] == pair_of[3]
    # The orphan twin (no base) and the plain rows stay singletons.
    assert sum(1 for g in groups if len(g) == 1) == 3


def test_contrastive_margin_loss_zero_past_margin():
    torch = pytest.importorskip("torch")
    full = torch.tensor([0.9, 0.8])
    stripped = torch.tensor([0.2, 0.4])
    assert float(ttm.contrastive_margin_loss(
        full, stripped, 0.3)) == pytest.approx(0.0)
    # Gaps 0.1 and -0.1 against margin 0.3: mean(0.2, 0.4) = 0.3.
    assert float(ttm.contrastive_margin_loss(
        torch.tensor([0.5, 0.3]), torch.tensor([0.4, 0.4]),
        0.3)) == pytest.approx(0.3)
    empty = torch.tensor([])
    assert float(ttm.contrastive_margin_loss(
        empty, empty, 0.3)) == pytest.approx(0.0)


def test_default_build_keeps_shipped_signature():
    import inspect

    assert inspect.signature(
        btc.build_gate_corpus).parameters["contrast"].default is False
