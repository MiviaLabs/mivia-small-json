"""The EREG pairwise error-regularization term (arm B1, ticket 01).

``scripts/train_tool_match.py:error_reg_loss`` implements Xin et al.'s
pairwise term ("The Art of Abstention", ACL 2021, Eqs. 9-11): the mean of
max(0, c_i - c_j)^2 over batch pairs where example i is currently wrong and
j is right. These tests pin its behavior with hand-computed values; the
training run only wires the flag.
"""

import importlib.util
import math
from pathlib import Path

import pytest

# torch is a training-extra dependency, not a runtime pin: a clean clone
# without `pip install -e ".[training]"` skips this file instead of
# erroring at collection.
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "train_tool_match", ROOT / "scripts" / "train_tool_match.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.error_reg_loss


@pytest.fixture(scope="module")
def error_reg_loss():
    return _load()


def test_zero_when_batch_all_correct(error_reg_loss):
    logits = torch.tensor([[2.0, 0.5]])
    assert error_reg_loss(logits, torch.tensor([0])).item() == 0.0


def test_positive_when_wrong_outconfidences_right(error_reg_loss):
    # Row 0 is wrong (pred 1, label 0) at confidence e^3/(1+e^3);
    # row 1 is right (pred 0, label 0) at confidence e^0.5/(1+e^0.5).
    logits = torch.tensor([[0.0, 3.0], [0.5, 0.0]])
    got = error_reg_loss(logits, torch.tensor([0, 0])).item()
    c_wrong = math.exp(3.0) / (1.0 + math.exp(3.0))
    c_right = math.exp(0.5) / (1.0 + math.exp(0.5))
    assert got == pytest.approx((c_wrong - c_right) ** 2, abs=1e-6)
    assert got > 0.05


def test_zero_when_wrong_already_least_confident(error_reg_loss):
    logits = torch.tensor([[0.0, 0.5], [3.0, 0.0]])
    assert error_reg_loss(logits, torch.tensor([0, 0])).item() == 0.0


def test_zero_when_batch_all_wrong(error_reg_loss):
    # No right example to pair with, so no ordered wrong/right pair exists.
    logits = torch.tensor([[0.0, 3.0], [0.0, 2.0]])
    assert error_reg_loss(logits, torch.tensor([0, 0])).item() == 0.0


def test_invariant_to_batch_order(error_reg_loss):
    logits = torch.tensor([[0.0, 3.0], [0.5, 0.0], [4.0, 0.0]])
    labels = torch.tensor([0, 0, 0])
    assert error_reg_loss(logits, labels).item() == pytest.approx(
        error_reg_loss(logits.flip(0), labels.flip(0)).item())
