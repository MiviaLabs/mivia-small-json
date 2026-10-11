"""Increment 3 — per-slot top-K (slot,span) assignment
(string-param-span-fill, INCREMENT 3 section,
2026-10-10).

The registered residual texture of the increment-2 falsification
(reports/training-report.md, commit c4e521a decomposition;
.scratch/incr2/b2-miss-texture.json): greedy argmax collisions. Two
required slots' scorer argmax land on the SAME token; the greedy decode
gives the span to the higher-confidence slot and the loser abstains —
its second-best span is NEVER consulted ("country" loses `pandas` to
`species` while `China` sits verbatim in the request). The scorer fires
on every missing slot (no_fire = 0); the collision is decode-reachable.

The registered mechanism: per-slot top-K candidate seeds (K = 3),
extend each by the unchanged whitespace-bounded rule, then assign spans
to slots maximizing total confidence under each-span-used-once — greedy
over (slot,span) pairs by confidence descending. Floors do not move
(emit 0.3 / extend 0.15 are meta-bound; the assignment change is the
only lever).

RED-first: the collision test fails on the greedy argmax decode and
passes on top-K. Deterministic, CPU-only, no artifact needed (mock
scorers through the trainer's injection seam, always restored).
"""

from __future__ import annotations

import re

from mivia_small_json import span_fill

# The registered fixture (from the measured decomposition): two slots,
# one shared argmax token, disjoint second-best spans.
PANDA_TOOL = {"name": "identify_bird", "description": "Identify wildlife.",
              "parameters": {"properties": {
                  "species": {"type": "string"},
                  "country": {"type": "string"}},
                  "required": ["species", "country"]}}
PANDA_REQUEST = "Identify the panda and report its country China"


class _CollisionScorer:
    """species argmax = `panda` (0.99) with second-best `China` (0.85);
    country argmax = the SAME `panda` token (0.97) with second-best
    `China` (0.80) — the greedy loser's verbatim value is its own
    second-best, exactly the measured texture."""

    _P = {"species": {"panda": 0.99, "China": 0.85},
          "country": {"panda": 0.97, "China": 0.80}}

    def pair_probs(self, request, param, prop):
        return [self._P[param].get(m.group(0), 0.01)
                for m in re.finditer(r"\w+", request)]


class _Inject:
    """Injection through the trainer's seam, always restored."""

    def __init__(self, scorer, emit=0.6, extend=0.15):
        self._scorer = scorer
        self._floors = (emit, extend)

    def __enter__(self):
        self._saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
        self._saved_floors = (span_fill._SPANFILL_EMIT_FLOOR,
                              span_fill._SPANFILL_EXTEND_FLOOR)
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = \
            self._scorer, True
        span_fill.configure_spanfill_floors(*self._floors)
        return self._scorer

    def __exit__(self, *exc):
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = self._saved
        span_fill.configure_spanfill_floors(*self._saved_floors)
        return False


def test_collision_loser_falls_to_its_second_best_span():
    """THE registered red-first case: `country`'s argmax collides with
    `species`' on `panda`; top-K assignment consults country's
    second-best (`China`, verbatim in the request) and fills BOTH slots.
    The greedy argmax decode leaves country missing (RED on the old
    decode). The higher-confidence slot still wins the shared token."""
    with _Inject(_CollisionScorer()):
        filled, missing = span_fill.learned_fill(
            PANDA_TOOL, PANDA_REQUEST)
    assert filled == {"species": "panda", "country": "China"}, filled
    assert missing == [], missing


def test_collision_winner_is_the_higher_confident_slot():
    """Assignment order stays confidence-ordered: species (0.99) takes
    the shared `panda` token; country never steals it downward."""
    with _Inject(_CollisionScorer()):
        filled, _missing = span_fill.learned_fill(
            PANDA_TOOL, PANDA_REQUEST)
    assert filled.get("species") == "panda", filled


def test_topk_bound_k3_fourth_best_seed_is_never_consulted():
    """K = 3 (registered): a slot whose first THREE seeds are all taken
    by other slots falls back NO further — its 4th-best seed (free, and
    above the emit floor) is never consulted; the slot stays missing.
    A larger K would donate a weak 4th choice into the slot."""
    tool = {"name": "t", "parameters": {"properties": {
        s: {"type": "string"} for s in
        ("alpha", "gamma", "delta", "beta")},
        "required": ["alpha", "gamma", "delta", "beta"]}}

    class _KBoundScorer:
        _P = {"alpha": {"t1": 0.99},
              "gamma": {"t2": 0.98},
              "delta": {"t3": 0.97},
              # beta's three best seeds are exactly the other slots'
              # argmax spans; only its 4th seed is free.
              "beta": {"t1": 0.96, "t2": 0.95, "t3": 0.94, "t4": 0.90}}

        def pair_probs(self, request, param, prop):
            return [self._P[param].get(m.group(0), 0.01)
                    for m in re.finditer(r"\w+", request)]

    with _Inject(_KBoundScorer(), emit=0.6):
        filled, missing = span_fill.learned_fill(
            tool, "value t1 value t2 value t3 value t4")
    assert filled == {"alpha": "t1", "gamma": "t2", "delta": "t3"}, filled
    assert missing == ["beta"], missing


def test_second_best_below_emit_floor_is_never_consulted():
    """The emit floor gates EVERY candidate seed, not just the argmax:
    a colliding slot whose only other span sits below the floor stays
    missing — no weak fallback, abstain over fabricate."""
    tool = {"name": "t", "parameters": {"properties": {
        "alpha": {"type": "string"}, "beta": {"type": "string"}},
        "required": ["alpha", "beta"]}}

    class _LowSecondScorer:
        _P = {"alpha": {"t1": 0.99, "t2": 0.95},
              "beta": {"t1": 0.97, "t2": 0.50}}  # 0.50 < emit 0.6

        def pair_probs(self, request, param, prop):
            return [self._P[param].get(m.group(0), 0.01)
                    for m in re.finditer(r"\w+", request)]

    with _Inject(_LowSecondScorer(), emit=0.6):
        filled, missing = span_fill.learned_fill(
            tool, "span t1 span t2")
    assert filled == {"alpha": "t1"}, filled
    assert missing == ["beta"], missing


def test_ladder_claims_stay_reserved_against_topk_fallbacks():
    """`reserved` spans (the deterministic ladders' claims) are never
    re-claimed — not even by a colliding slot's second-best candidate."""
    tool = {"name": "t", "parameters": {"properties": {
        "alpha": {"type": "string"}, "beta": {"type": "string"}},
        "required": ["alpha", "beta"]}}

    class _SameEverywhereScorer:
        def pair_probs(self, request, param, prop):
            return [0.9 if m.group(0) == "claimed" else 0.7
                    for m in re.finditer(r"\w+", request)]

    with _Inject(_SameEverywhereScorer()):
        filled, missing = span_fill.learned_fill(
            tool, "claimed span free2 free1",
            reserved=[(0, len("claimed"))], only=["alpha", "beta"])
    # `claimed` is reserved; alpha/beta must land on the free tokens —
    # each span used once, confidence order: free1 (last token? no —
    # equal confidence 0.7, deterministic by (-conf, slot, seed)).
    assert set(filled) <= {"alpha", "beta"}
    assert "claimed" not in (filled or {}).values()
    assert sorted(list(filled) + list(missing)) == ["alpha", "beta"]
