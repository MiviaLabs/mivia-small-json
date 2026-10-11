"""Tests for SelectiveNet reject option (arm S1).

Verifies the mathematical formulation, coverage penalty behavior, gradient flow,
model topology, and runtime dual-output ONNX gating in ToolMatcher.
"""
from __future__ import annotations

import pytest

# torch is a training-extra dependency, not a runtime pin: a clean clone
# without `pip install -e ".[training]"` (or the CI CPU build) skips this
# file instead of erroring at collection.
torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402

from mivia_small_json.tool_match import ToolMatcher  # noqa: E402
from scripts.train_tool_match import (  # noqa: E402
    SelectiveCrossEncoder, selectivenet_loss)


def test_selectivenet_loss_penalty_zero_when_coverage_satisfied():
    logits = torch.zeros(4, 2, requires_grad=True)
    aux_logits = torch.zeros(4, 2, requires_grad=True)
    # empirical coverage is 0.85, target is 0.75
    selection = torch.tensor([[0.85], [0.85], [0.85], [0.85]], requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])

    tot, sel_risk, aux_risk, cov, cov_pen = selectivenet_loss(
        logits, selection, aux_logits, labels, target_coverage=0.75, coverage_weight=32.0
    )

    assert cov_pen.item() == 0.0
    assert abs(cov.item() - 0.85) < 1e-6
    # Total loss should be alpha * sel_risk + (1-alpha) * aux_risk
    assert abs(tot.item() - (0.5 * sel_risk.item() + 0.5 * aux_risk.item())) < 1e-6


def test_selectivenet_loss_quadratic_penalty_when_coverage_insufficient():
    logits = torch.zeros(4, 2)
    aux_logits = torch.zeros(4, 2)
    # empirical coverage is 0.50, target is 0.75
    selection = torch.tensor([[0.50], [0.50], [0.50], [0.50]], requires_grad=True)

    tot, sel_risk, aux_risk, cov, cov_pen = selectivenet_loss(
        logits, selection, aux_logits, torch.tensor([0, 1, 0, 1]),
        target_coverage=0.75, coverage_weight=32.0,
    )

    # Expected penalty: 32.0 * (0.75 - 0.50)^2 = 32.0 * 0.0625 = 2.0
    expected_pen = 32.0 * (0.25**2)
    assert abs(cov_pen.item() - expected_pen) < 1e-5

    # Gradient of coverage penalty wrt selection must be negative (increasing s reduces penalty)
    cov_pen.backward()
    assert (selection.grad < 0).all()


def test_selectivenet_loss_selective_risk_downweights_rejected_errors():
    # 2 samples: sample 0 is wrong, sample 1 is right
    # sample 0 has high CE loss, sample 1 has low CE loss
    logits = torch.tensor([[ -5.0,  5.0],  # predicts 1, label is 0 (wrong, high loss)
                           [ -5.0,  5.0]]) # predicts 1, label is 1 (right, low loss)
    aux_logits = torch.zeros(2, 2)
    labels = torch.tensor([0, 1])

    # Case A: both accepted (s = [1, 1])
    s_full = torch.tensor([[1.0], [1.0]])
    _, sel_risk_full, _, _, _ = selectivenet_loss(logits, s_full, aux_logits, labels)

    # Case B: wrong sample rejected (s = [0.01, 1.0])
    s_selective = torch.tensor([[0.01], [1.0]])
    _, sel_risk_selective, _, _, _ = selectivenet_loss(logits, s_selective, aux_logits, labels)

    # Rejecting the error must significantly reduce selective risk
    assert sel_risk_selective.item() < sel_risk_full.item()


def test_selectivenet_loss_backward_gradients_flow():
    logits = torch.randn(4, 2, requires_grad=True)
    aux_logits = torch.randn(4, 2, requires_grad=True)
    selection = torch.tensor([[0.6], [0.7], [0.8], [0.5]], requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    weights = torch.tensor([1.0, 5.0, 1.0, 5.0])

    tot, _, _, _, _ = selectivenet_loss(
        logits, selection, aux_logits, labels, weights=weights, target_coverage=0.75
    )
    tot.backward()

    assert logits.grad is not None and not torch.isnan(logits.grad).any()
    assert aux_logits.grad is not None and not torch.isnan(aux_logits.grad).any()
    assert selection.grad is not None and not torch.isnan(selection.grad).any()


def test_selective_cross_encoder_forward_and_shapes():
    class DummyBert(nn.Module):
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            return (None, torch.zeros(input_ids.shape[0], 384))

    class DummyBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = DummyBert()
            self.classifier = nn.Linear(384, 2)

    base = DummyBase()
    model = SelectiveCrossEncoder(base, hidden_size=384, num_labels=2)
    x = torch.ones(3, 16, dtype=torch.long)

    # Inference forward: logits + selection
    logits, selection = model(x)
    assert logits.shape == (3, 2)
    assert selection.shape == (3, 1)
    # Selection head bias is initialized to 1.1 -> sigmoid(1.1) ~= 0.75
    assert (selection >= 0.70).all()

    # Training forward with aux
    logits, selection, aux_logits = model(x, return_aux=True)
    assert logits.shape == (3, 2)
    assert selection.shape == (3, 1)
    assert aux_logits.shape == (3, 2)


class MockSessionSingleOutput:
    def __init__(self, logits):
        self._logits = np.array(logits, dtype=np.float32)

    def run(self, output_names, feed):
        return [self._logits]


class MockSessionDualOutput:
    def __init__(self, logits, selection):
        self._logits = np.array(logits, dtype=np.float32)
        self._selection = np.array(selection, dtype=np.float32)

    def run(self, output_names, feed):
        return [self._logits, self._selection]


class DummyTokenizer:
    def encode_batch(self, pairs):
        class Enc:
            ids = [101, 102]
            attention_mask = [1, 1]
            type_ids = [0, 1]
        return [Enc()]


def test_tool_matcher_single_output_backward_compatibility():
    # Logits [0, 2] -> softmax prob for class 1 is ~0.8808
    sess = MockSessionSingleOutput([[0.0, 2.0]])
    matcher = ToolMatcher(sess, DummyTokenizer(), ["input_ids", "attention_mask"])
    score = matcher._score("some request", "some tool")
    exp = np.exp([0.0, 2.0] - np.max([0.0, 2.0]))
    expected_prob = float((exp / exp.sum())[1])
    assert abs(score - expected_prob) < 1e-5


def test_tool_matcher_dual_output_selective_gating():
    # When selection >= 0.50, returns pred_prob
    sess_accepted = MockSessionDualOutput([[0.0, 2.0]], [[0.75]])
    matcher_acc = ToolMatcher(sess_accepted, DummyTokenizer(), ["input_ids", "attention_mask"])
    score_acc = matcher_acc._score("some request", "some tool")
    exp = np.exp([0.0, 2.0] - np.max([0.0, 2.0]))
    expected_prob = float((exp / exp.sum())[1])
    assert abs(score_acc - expected_prob) < 1e-5

    # When selection < 0.50, returns 0.0 (abstention)
    sess_rejected = MockSessionDualOutput([[0.0, 2.0]], [[0.35]])
    matcher_rej = ToolMatcher(sess_rejected, DummyTokenizer(), ["input_ids", "attention_mask"])
    score_rej = matcher_rej._score("some request", "some tool")
    assert score_rej == 0.0
