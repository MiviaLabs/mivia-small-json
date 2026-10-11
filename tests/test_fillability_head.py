"""TRACK 1: objective-level fillability term for the incomplete slice.

- Unit tests pin the new pure machinery (`request_concreteness`,
  `multitask_loss`): no training, no GPU, no eval files.
- test_train_eval_disjoint pins the zero-overlap invariant (no synthetic
  rows are added; the corpus is the shipped one).
- test_bar_not_cleared is a FALSIFICATION RECORD: it reads the measured
  verdict in reports/exp-fillability-head.md and asserts the joint bar
  is NOT cleared. If a future head clears it, this test FAILS on
  purpose — update the report and this record together.

No schema registration anywhere in this module.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "data" / "incoming" / "external" / "bfcl"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "train_fillability_head",
        ROOT / "scripts" / "train_fillability_head.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_concreteness_marks_values_not_topics():
    mod = _load_script()
    assert mod.request_concreteness("set cool mode with a temp of 24 oC") == 1
    assert mod.request_concreteness("plan a bus route from Beijing zone 321") == 1
    # Glued values ("Beijing321") have no word-boundary number under the
    # shipped gate's own pattern: that is the error_param slice (capability
    # blindness), not the incomplete slice this head targets.
    assert mod.request_concreteness("plan a bus route from Beijing321 now") == 0
    assert mod.request_concreteness("fetch https://api.example.com/data") == 1
    assert mod.request_concreteness("delete the 'red-team' please") == 1
    # Topical but value-free requests carry no fillable value.
    assert mod.request_concreteness("set cool mode please") == 0
    assert mod.request_concreteness("what is the weather like") == 0


def test_multitask_loss_reduces_to_shipped_at_zero_weight():
    torch = pytest.importorskip("torch")
    mod = _load_script()
    import torch.nn.functional as F

    match_logits = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    aux_logits = torch.tensor([[2.0], [-2.0]])
    match_labels = torch.tensor([1, 0])
    aux_labels = torch.tensor([1, 0])
    total, match_term, _ = mod.multitask_loss(
        match_logits, aux_logits, match_labels, aux_labels,
        weights=torch.tensor([5.0, 1.0]), aux_weight=0.0)
    ce = F.cross_entropy(match_logits, match_labels, reduction="none")
    expected = (ce * torch.tensor([5.0, 1.0])).sum() / 6.0
    assert float(total) == pytest.approx(float(expected))
    assert float(match_term) == pytest.approx(float(expected))


def test_multitask_loss_aux_term_is_bce():
    torch = pytest.importorskip("torch")
    mod = _load_script()
    import torch.nn.functional as F

    match_logits = torch.zeros(2, 2)
    aux_logits = torch.tensor([[10.0], [-10.0]])
    total, _, aux_term = mod.multitask_loss(
        match_logits, aux_logits, torch.tensor([0, 1]),
        torch.tensor([1, 0]), aux_weight=1.0)
    assert float(aux_term) == pytest.approx(
        float(F.binary_cross_entropy_with_logits(
            aux_logits.view(-1), torch.tensor([1.0, 0.0]))))
    assert float(total) > 0.0


# (test_train_eval_disjoint and test_bar_not_cleared retired on the
# public tree: the BFCL pair corpus / eval sets and the falsification
# report are not redistributed. The private tree keeps both gates.)
