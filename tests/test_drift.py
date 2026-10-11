"""Drift tripwires: hand-built inputs, pure read-only readings.

The monitor is a pure function over two class distributions, a feature
matrix and two vote lists. These tests pin the contract with hand-built
numbers: a stable distribution and a shifted one must flag differently,
identical inputs must read as no drift, and nothing may be mutated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from mivia_small_json import drift

REFERENCE = {"issue": 0.5, "commit": 0.5}
STABLE = {"issue": 0.52, "commit": 0.48}
MODERATE = {"issue": 0.7, "commit": 0.3}
SIGNIFICANT = {"issue": 0.95, "commit": 0.05}


def test_psi_of_identical_distributions_is_zero():
    assert drift.psi(REFERENCE, dict(REFERENCE)) == 0.0
    assert drift.psi({"a": 1.0}, {"a": 1.0}) == 0.0


def test_psi_unions_keys_and_clamps_the_missing_side():
    # Mass moved to a class absent from expected is real skew, not noise.
    plain = drift.psi({"issue": 0.9, "commit": 0.1}, REFERENCE)
    with_extra = drift.psi({"issue": 0.8, "commit": 0.1, "mystery": 0.1}, REFERENCE)
    assert with_extra > plain > 0
    assert drift.psi({}, {"issue": 1.0}) > 0  # empty observed is max skew


def test_psi_level_thresholds():
    assert drift.psi_level(0.0) == "stable"
    assert drift.psi_level(0.09) == "stable"
    assert drift.psi_level(0.1) == "moderate"
    assert drift.psi_level(0.249) == "moderate"
    assert drift.psi_level(0.25) == "significant"


def test_drift_report_flags_stable_and_shifted_differently():
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    stable = drift.drift_report(STABLE, REFERENCE, matrix, ["a", "b"], ["a", "b"])
    shifted = drift.drift_report(SIGNIFICANT, REFERENCE, matrix, ["a", "b"], ["a", "b"])
    assert stable["level"] == "stable"
    assert shifted["level"] == "significant"
    assert shifted["psi"] > stable["psi"]
    assert set(stable) == {"psi", "level", "feature_occupancy", "agreement_rate"}


def test_drift_report_moderate_bucket():
    report = drift.drift_report(MODERATE, REFERENCE, [[1.0]], ["x"], ["x"])
    assert report["level"] == "moderate"
    assert 0.1 <= report["psi"] < 0.25


def test_agreement_rate_is_one_for_identical_lists():
    assert drift.agreement_rate(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_agreement_rate_counts_matching_positions():
    assert drift.agreement_rate(["a", "b", "c", "d"], ["a", "b", "x", "d"]) == 0.75
    assert drift.agreement_rate(["a", "b"], ["c", "d"]) == 0.0


def test_agreement_rate_empty_lists_agree_vacuously():
    assert drift.agreement_rate([], []) == 1.0


def test_agreement_rate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        drift.agreement_rate(["a", "b"], ["a"])


def test_feature_occupancy_on_a_known_dense_matrix():
    # 2x3, two non-zero cells, row norms sqrt(5) and 0.
    occ = drift.feature_occupancy([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    assert occ["nonzero_rate"] == pytest.approx(2 / 6)
    assert occ["mean_norm"] == pytest.approx(math.sqrt(5) / 2)


def test_feature_occupancy_reads_sparse_and_dense_the_same():
    dense = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]
    sp = sparse.csr_matrix(dense)
    assert drift.feature_occupancy(sp) == drift.feature_occupancy(dense)


def test_feature_occupancy_of_an_empty_matrix_is_all_zero():
    assert drift.feature_occupancy(np.zeros((0, 4))) == {"nonzero_rate": 0.0, "mean_norm": 0.0}


def test_drift_report_does_not_mutate_its_inputs():
    pred = {"issue": 0.7, "commit": 0.3}
    ref = {"issue": 0.5, "commit": 0.5}
    matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
    votes_a = ["a", "b"]
    votes_b = ["a", "c"]

    drift.drift_report(pred, ref, matrix, votes_a, votes_b)

    assert pred == {"issue": 0.7, "commit": 0.3}
    assert ref == {"issue": 0.5, "commit": 0.5}
    assert np.array_equal(matrix, np.array([[1.0, 0.0], [0.0, 2.0]]))
    assert votes_a == ["a", "b"]
    assert votes_b == ["a", "c"]
