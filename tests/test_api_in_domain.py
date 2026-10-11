"""Pin: /v1/route `in_domain` must match the pipeline's energy gate semantics.

Regression (audit 2026-10-08, plan Phase 0): `api.do_route` computed
``energy <= threshold`` while the pipeline gate is ``energy >= threshold``
(higher energy = closer to the training manifold). The API flag reported
the OPPOSITE of the pipeline's decision on every energy-gated row.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mivia_small_json import api  # noqa: E402
from mivia_small_json.pipeline import _route_with_confidence  # noqa: E402

THRESHOLD = 8.4


class EnergyRouter:
    """Duck router: fixed prediction, parametric energy score."""

    classes = ["issue"]

    def __init__(self, energy: float):
        self._energy = energy
        self.in_domain_energy = THRESHOLD

    def predict_proba(self, texts):
        return [[0.99] for _ in texts]

    def predict_proba_with_energy(self, texts):
        probs = self.predict_proba(texts)
        return probs, [self._energy] * len(texts)


@pytest.mark.parametrize("energy,expected", [
    (10.0, True),   # high energy = in-domain (pipeline: energy >= threshold)
    (5.0, False),   # low energy = OOD
])
def test_in_domain_matches_pipeline_gate(energy, expected):
    router = EnergyRouter(energy)
    text = "2026-10-08 03:11:22 ERROR db: pool exhausted"
    # ground truth: the pipeline gate's own decision for this router
    _, routed_by, _ = _route_with_confidence(router, text)
    obj = api.do_route(text, router)
    assert obj["in_domain"] is expected, (
        f"energy={energy}: api said {obj['in_domain']}, pipeline gate says "
        f"{expected} (routed_by={routed_by})")
    assert obj["in_domain"] == (energy >= THRESHOLD)
