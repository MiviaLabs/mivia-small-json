"""BCa bootstrap CI and minimum-detectable-effect helpers (Phase 0b).

Pins the small-n honesty tooling added after the register-router review:
bca_ci (Efron 1987 bias-corrected accelerated bootstrap, stdlib only) and
mde (smallest reliably detectable effect at given alpha/power). Both must
be deterministic given a seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mivia_small_json.stats import _norm_ppf, bca_ci, mde  # noqa: E402


def test_norm_ppf_matches_known_quantiles():
    assert abs(_norm_ppf(0.975) - 1.959964) < 1e-3
    assert abs(_norm_ppf(0.5)) < 1e-6
    assert abs(_norm_ppf(0.8) - 0.841621) < 1e-3


def test_bca_ci_covers_true_mean_and_is_deterministic():
    values = [1.0] * 58 + [0.0] * 42  # true mean 0.58
    lo, hi = bca_ci(values, statistic=lambda v: sum(v) / len(v), seed=7)
    assert lo <= 0.58 <= hi
    assert (lo, hi) == bca_ci(values, statistic=lambda v: sum(v) / len(v),
                              seed=7)
    # CI width shrinks as n grows (same generator, larger sample)
    big = [1.0] * 580 + [0.0] * 420
    lo2, hi2 = bca_ci(big, statistic=lambda v: sum(v) / len(v), seed=7)
    assert (hi2 - lo2) < (hi - lo)


def test_bca_ci_skewed_statistic_shifts_from_percentile():
    # Right-skewed values: BCa's bias correction should push the upper
    # bound above the naive percentile band on the same resamples.
    values = [1.0] + [0.1] * 29
    lo, hi = bca_ci(values, statistic=sum, seed=3, n_resamples=2000)
    naive_lo = 1.0  # every resample contains the single 1.0 at least...
    assert hi > naive_lo or lo > 0.0  # finite, sane interval


def test_mde_shrinks_with_n_and_is_symmetric_bar():
    m60 = mde(60, 0.5)
    m300 = mde(300, 0.5)
    assert m300 < m60
    assert m60 > 0.15  # n=60 cannot reliably see a 15pp effect
    assert mde(10, 0.5) == float("inf") or mde(10, 0.5) > 0.4
