"""Soft adjacent-tier CORAL targets (Phase 2 arm, Diaz & Marathe CVPR 2019).

Pure-python target builder used by train_severity_ordinal --soft-adjacent.
Pins: simplex correctness, rank consistency (targets monotone
non-increasing), boundary folding, hard-target equivalence at mass 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_severity_ordinal import soft_cumulative_targets  # noqa: E402


def test_simplex_and_rank_consistency():
    for gold in range(4):
        for targets in [soft_cumulative_targets([gold], 4, 0.3)]:
            assert all(0.0 <= t <= 1.0 for t in targets[0])
            assert targets[0] == sorted(targets[0], reverse=True)
    # interior gold: mass split evenly on both neighbors
    t1 = soft_cumulative_targets([1], 4, 0.2)[0]
    assert t1 == [0.9, 0.1, 0.0]


def test_boundary_folds_back_onto_gold():
    t0 = soft_cumulative_targets([0], 4, 0.2)[0]
    t3 = soft_cumulative_targets([3], 4, 0.2)[0]
    assert t0 == [0.1, 0.0, 0.0]
    assert t3 == [1.0, 1.0, 0.9]


def test_mass_zero_is_hard_targets():
    assert soft_cumulative_targets([1], 4, 0.0) == [[1.0, 0.0, 0.0]]
    assert soft_cumulative_targets([2], 4, 0.0) == [[1.0, 1.0, 0.0]]


def test_rejects_bad_mass_and_gold():
    import pytest
    with pytest.raises(ValueError):
        soft_cumulative_targets([1], 4, 1.0)
    with pytest.raises(ValueError):
        soft_cumulative_targets([4], 4, 0.2)
