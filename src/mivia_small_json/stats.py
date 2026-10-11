# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Small-sample statistics for honest evaluation.

Ticket 04 of result-quality-improvement At n=40 a bare
percentage hides an ~8-point standard error, and rows that share a template
are not independent, so row-level intervals are too narrow. These helpers give
Wilson intervals, cluster-bootstrap intervals, and an exact McNemar test.

Stdlib only, deterministic given a seed.
"""
from __future__ import annotations

import math
import random

Z_95 = 1.959963985


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Behaves at small n and at the boundaries, where the normal approximation
    does not: 0/n and n/n both get a real interval.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ci(clusters, n_resamples: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for the mean, resampling whole clusters.

    ``clusters`` maps a cluster id (template, source, conversation) to its
    values. Rows inside a cluster are not independent, so resampling clusters
    is the honest interval; resampling rows understates it.
    """
    keys = sorted(clusters)
    if not keys:
        return (0.0, 1.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        picked = [clusters[rng.choice(keys)] for _ in keys]
        values = [v for vs in picked for v in vs]
        means.append(sum(values) / len(values) if values else 0.0)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    ``b`` counts rows where A is right and B wrong, ``c`` the reverse.
    Concordant pairs carry no information about the difference between A and
    B, so only ``b + c`` enters the test.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2 ** n))


def bca_ci(values: list[float], statistic=sum, n_resamples: int = 4000,
           seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """BCa (bias-corrected and accelerated) bootstrap CI for ``statistic``.

    Percentile bootstrap undercorrects when the statistic's distribution is
    skewed; BCa adds a bias-correction (z0, from the proportion of
    resamples below the observed statistic) and an acceleration term (a,
    from jackknife influence sums) — Efron 1987. Deterministic given seed.
    """
    if not values:
        return (0.0, 0.0)
    n = len(values)
    theta = statistic(values)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(statistic(sample))
    boots.sort()
    below = sum(1 for b in boots if b < theta) / n_resamples
    z0 = _norm_ppf(below) if 0.0 < below < 1.0 else (
        -8.0 if below == 0.0 else 8.0)
    jack = [statistic(values[:i] + values[i + 1:]) for i in range(n)]
    mean_j = sum(jack) / n
    num = sum((mean_j - j) ** 3 for j in jack)
    den = 6.0 * (sum((mean_j - j) ** 2 for j in jack) ** 1.5)
    a = num / den if den > 0 else 0.0
    out = []
    for ai in (alpha / 2, 1 - alpha / 2):
        z_alpha = _norm_ppf(ai)
        adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        out.append(_norm_cdf(adj))
    lo_i = min(max(int(out[0] * n_resamples), 0), n_resamples - 1)
    hi_i = min(max(int(out[1] * n_resamples), 0), n_resamples - 1)
    return (boots[lo_i], boots[hi_i])


def mde(n: int, baseline: float, alpha: float = 0.05,
        power: float = 0.80) -> float:
    """Minimum detectable effect (absolute proportion points) for a paired
    comparison on n items at the given alpha/power — the smallest true
    difference the test can reliably see. Normal approximation; for
    discordant-pair designs the effective n is the discordant count.
    """
    if n <= 1:
        return float("inf")
    z_a = _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    p = min(max(baseline, 0.05), 0.95)
    sigma = math.sqrt(2 * p * (1 - p))
    return (z_a + z_b) * sigma / math.sqrt(n)


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))
