"""Cumulative-threshold severity head (ticket 12).

A CORAL/CORN-style ordinal head replaces K independent tier sigmoids with K-1
ordered cutpoints on a single latent score. The tier is the number of
cutpoints the latent exceeds, so adjacent boundaries share one decision axis.
"""

import importlib.util
from pathlib import Path

from mivia_small_json.severity_head import (
    CANONICAL_SEVERITY_ANCHORS,
    coral_probability,
    cumulative_tier,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cumulative_tier_counts_exceeded_cutpoints():
    assert cumulative_tier([-1.0, -1.0, -1.0]) == 0
    assert cumulative_tier([1.0, -1.0, -1.0]) == 1
    assert cumulative_tier([1.0, 1.0, -1.0]) == 2
    assert cumulative_tier([1.0, 1.0, 1.0]) == 3


def test_cumulative_tier_is_monotone_in_the_latent():
    tiers = [cumulative_tier([z, z, z]) for z in (-2.0, -0.5, 0.5, 2.0)]
    assert tiers == [0, 0, 3, 3]


def test_cumulative_tier_handles_no_cutpoints():
    assert cumulative_tier([]) == 0


def test_canonical_severity_scale_is_four_ordered_anchors():
    assert len(CANONICAL_SEVERITY_ANCHORS) == 4
    assert [a.split(" ")[0] for a in CANONICAL_SEVERITY_ANCHORS] == [
        "info", "warning", "error", "critical"]


def test_canonical_scale_matches_the_bench_templates():
    """One scale: the package constant and the bench/training templates must
    not drift, or `triage_severity` would route the canonical anchors to the
    cross-encoder instead of the promoted ordinal head."""
    spec = importlib.util.spec_from_file_location(
        "bench_zero_shot_primitives",
        ROOT / "scripts" / "bench_zero_shot_primitives.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert list(CANONICAL_SEVERITY_ANCHORS) == list(mod.SEVERITY_ANCHORS_EXCLUSIVE)


def test_coral_probability_peaks_at_the_decoded_tier():
    # z_k > 0 for k < 1 and z_k < 0 for k >= 1 -> tier 1.
    logits = [2.0, -2.0, -2.0]
    assert cumulative_tier(logits) == 1
    probs = [coral_probability(logits, t) for t in range(4)]
    assert max(range(4), key=probs.__getitem__) == 1
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_coral_probability_is_bounded_and_handles_absent_cutpoints():
    assert coral_probability([10.0, 10.0, 10.0], 3) > 0.99
    assert coral_probability([-10.0, -10.0, -10.0], 0) > 0.99
    assert 0.0 <= coral_probability([], 0) <= 1.0
