# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Cumulative-threshold ordinal severity head.

Ticket 12 of result-quality-improvement The shipped grade
artifact scores a text against each ordered anchor with an independent sigmoid;
those sigmoids saturate near 0/1 and the residual error is adjacent-tier
boundary mass. A cumulative-threshold head instead learns a single latent
severity score and K-1 ordered cutpoints, so adjacent boundaries share one
decision axis (CORAL/CORN; Cao et al., arXiv:1901.07884; Shi et al.,
arXiv:2111.08851).

The head applies to the *fixed* severity scale only. Arbitrary user anchors
still go through the cross-encoder anchor-alignment path.
"""
from __future__ import annotations

import math

# The fixed severity scale the ordinal head is calibrated on: the exclusive
# 4-tier anchor templates (`scripts/bench_zero_shot_primitives.py` and
# `scripts/train_grade_anchor_alignment.py` hold the same strings). The
# cumulative head is trained to decode *this* ordered scale, so it is the
# only anchor set it is allowed to serve; arbitrary user anchors stay on the
# cross-encoder anchor-alignment path.
CANONICAL_SEVERITY_ANCHORS = (
    "info level: routine informational messages about normal successful "
    "operation and expected events.",
    "warning level: recoverable abnormal conditions that may need attention "
    "but are not failures.",
    "error level: operation failures, faults, and errors that disrupted a "
    "task or transaction.",
    "critical level: severe crashes, fatal errors, data loss, or complete "
    "system failure requiring immediate action.",
)


def cumulative_tier(cumulative_logits) -> int:
    """Tier index: how many ordered cutpoints the latent exceeds.

    ``cumulative_logits[k]`` is ``latent - cutpoint_k`` for the ordered
    cutpoints; the tier is the count of positive entries.
    """
    return sum(1 for z in cumulative_logits if z > 0.0)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(z)))))


def coral_probability(cumulative_logits, tier: int) -> float:
    """CORAL probability of ``tier`` under independent per-cutpoint sigmoids.

    CORAL treats ``sigmoid(cumulative_logits[k])`` as ``P(y > k)``, so
    ``P(y = tier) = P(y > tier-1) * P(y <= tier)``, with the absent
    cutpoints contributing 1. Bounded in [0, 1]; a ranking/confidence score,
    not a calibrated probability.
    """
    logits = list(cumulative_logits)
    k = len(logits)
    tier = max(0, min(int(tier), k))
    p_above_prev = 1.0 if tier == 0 else _sigmoid(logits[tier - 1])
    p_at_or_below = 1.0 if tier >= k else 1.0 - _sigmoid(logits[tier])
    return p_above_prev * p_at_or_below
