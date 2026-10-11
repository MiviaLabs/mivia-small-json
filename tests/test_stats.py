"""Statistics helpers for honest small-sample evaluation.

Ticket 04 of result-quality-improvement At n=40 a bare
percentage hides a ~8-point standard error, and rows that share a template are
not independent, so row-level intervals are too narrow. These helpers give
Wilson intervals, cluster bootstrap intervals, and an exact McNemar test.
"""

from mivia_small_json import stats


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = stats.wilson_interval(22, 40)
    assert lo < 0.55 < hi
    # The interval is wider than the normal approximation at small n.
    assert 0.35 < lo < 0.45
    assert 0.65 < hi < 0.75


def test_wilson_interval_is_tight_for_certain_outcomes():
    lo, hi = stats.wilson_interval(40, 40)
    assert hi == 1.0
    assert lo > 0.85
    assert lo < 1.0


def test_wilson_interval_handles_zero_n():
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_cluster_bootstrap_is_wider_than_row_bootstrap():
    """Clustered rows must not be treated as independent samples."""
    # 20 clusters of 5 identical outcomes each; row-level would be very tight.
    clusters = {f"c{i}": [1, 1, 1, 1, 1] if i % 2 == 0 else [0, 0, 0, 0, 0]
                for i in range(20)}
    lo_c, hi_c = stats.cluster_bootstrap_ci(clusters, seed=1)
    rows = [v for vs in clusters.values() for v in vs]
    lo_r, hi_r = stats.cluster_bootstrap_ci({"all": rows}, seed=1)
    assert (hi_c - lo_c) > (hi_r - lo_r)


def test_cluster_bootstrap_brackets_the_mean():
    clusters = {f"c{i}": [1, 0] for i in range(30)}
    lo, hi = stats.cluster_bootstrap_ci(clusters, seed=1)
    assert lo <= 0.5 <= hi


def test_mcnemar_exact_is_symmetric_and_uses_only_discordants():
    # No discordant pairs -> no evidence of a difference.
    assert stats.mcnemar_exact(0, 0) == 1.0
    # 10 vs 0 discordants is strong evidence.
    assert stats.mcnemar_exact(10, 0) < 0.01
    assert stats.mcnemar_exact(0, 10) < 0.01
    assert stats.mcnemar_exact(4, 6) > 0.5
