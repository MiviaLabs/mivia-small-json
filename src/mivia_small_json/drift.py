# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Read-only drift tripwires: PSI, feature occupancy, labeler agreement.

The flywheel trains and measures; this module only watches. Every
function here is a pure read over class distributions, a feature matrix
and two vote lists. Nothing retrains, nothing writes, nothing mutates
its arguments. The cycle report carries the readings and the promotion
decision stays a human one.

Three cheap signals:

- Population Stability Index on class shares (predicted vs reference) —
  a shift means the input world moved out from under the artifact.
- Feature occupancy: the non-zero rate and the mean row norm of the
  hashed feature matrix — a collapse toward all-zero rows means the
  encoder saw text it was not built for.
- Weak-labeler/model agreement rate — a concept-drift approximation
  that needs no labels.

Thresholds follow the standard PSI cut: below 0.1 stable, below 0.25
moderate, otherwise significant. ``psi`` is the canonical
implementation; ``flywheel`` imports it rather than keeping a second
copy (see ``from .drift import psi as _psi`` there).
"""

from __future__ import annotations

import math

import numpy as np

# A class present on only one side of a PSI comparison is real skew, not
# noise to hide. Clamp the missing side here so the score moves instead
# of exploding to infinity.
_EPSILON = 1e-6

# PSI tripwire cuts (the classic 0.1 / 0.25 thresholds).
STABLE_BELOW = 0.1
MODERATE_BELOW = 0.25


def psi(observed: dict[str, float], expected: dict[str, float]) -> float:
    """Population Stability Index over class shares.

    Key sets are unioned: a class present on only one side is real skew
    and must move the score, not hide (the missing side clamps to
    ``_EPSILON``). Identical dicts score 0.0.
    """
    score = 0.0
    for key in set(observed) | set(expected):
        obs = max(observed.get(key, 0.0), _EPSILON)
        exp = max(expected.get(key, 0.0), _EPSILON)
        score += (obs - exp) * math.log(obs / exp)
    return score


def psi_level(score: float) -> str:
    """Bucket a PSI score: ``stable`` / ``moderate`` / ``significant``."""
    if score < STABLE_BELOW:
        return "stable"
    if score < MODERATE_BELOW:
        return "moderate"
    return "significant"


def _row_norms(matrix) -> np.ndarray:
    """L2 norm of each row, for a scipy-sparse or dense matrix."""
    if hasattr(matrix, "nnz") and hasattr(matrix, "toarray"):
        # scipy.sparse: multiply-sum-square avoids densifying the matrix.
        squared = matrix.multiply(matrix).sum(axis=1)
        return np.sqrt(np.asarray(squared, dtype=float).ravel())
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.linalg.norm(arr, axis=1)


def feature_occupancy(matrix) -> dict:
    """Non-zero rate and mean row norm of a feature matrix.

    Accepts a scipy sparse matrix or a dense array/list of lists. The
    non-zero rate is over every cell (``nnz / rows*cols``); the mean
    norm is the mean of the per-row L2 norms. An empty matrix reads as
    all zeros rather than raising.
    """
    if hasattr(matrix, "nnz") and hasattr(matrix, "toarray"):
        rows, cols = matrix.shape
        nnz = int(matrix.nnz)
    else:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        rows, cols = arr.shape
        nnz = int(np.count_nonzero(arr))
    cells = rows * cols
    norms = _row_norms(matrix)
    return {
        "nonzero_rate": (nnz / cells) if cells else 0.0,
        "mean_norm": float(norms.mean()) if norms.size else 0.0,
    }


def agreement_rate(votes_a, votes_b) -> float:
    """Fraction of positions where two label lists agree.

    The lists must be the same length; a mismatch is a caller bug, not a
    number to average over. Two empty lists agree vacuously (1.0).
    """
    a, b = list(votes_a), list(votes_b)
    if len(a) != len(b):
        raise ValueError(f"vote lists differ in length: {len(a)} vs {len(b)}")
    if not a:
        return 1.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def drift_report(pred_dist, reference_dist, matrix, votes_a, votes_b) -> dict:
    """One read-only drift reading, ready for the cycle report.

    ``level`` is the PSI bucket of the predicted-class distribution
    against the reference. The feature occupancy and the agreement rate
    ride alongside as context. Nothing here retrains or mutates.
    """
    score = psi(pred_dist, reference_dist)
    return {
        "psi": score,
        "level": psi_level(score),
        "feature_occupancy": feature_occupancy(matrix),
        "agreement_rate": agreement_rate(votes_a, votes_b),
    }
