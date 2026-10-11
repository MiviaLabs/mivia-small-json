"""Tests for the external comparison benchmark harness.

Deterministic, CPU-only, seconds. Exercises the evaluate seam on tiny
synthetic data so the real corpora are not required to run the suite.
"""

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_external", ROOT / "scripts" / "bench_external.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_separates_easy_classes():
    mod = _load()
    train = (["alpha alpha alpha"] * 30 + ["beta beta beta"] * 30,
             ["a"] * 30 + ["b"] * 30)
    test = (["alpha here"] * 5 + ["beta here"] * 5, ["a"] * 5 + ["b"] * 5)
    out = mod.evaluate(train, test)
    assert out["accuracy"] == 1.0
    assert out["f1_macro"] == 1.0
    assert out["n_train"] == 60 and out["n_test"] == 10
    assert out["n_classes"] == 2


def test_evaluate_reports_finite_metrics_on_noise():
    mod = _load()
    train = ([f"tok{i % 7}" for i in range(40)], [f"c{i % 2}" for i in range(40)])
    test = (["unseen"], ["c0"])
    out = mod.evaluate(train, test)
    # A single test row makes accuracy exactly 0 or 1; a value in between, or a
    # NaN metric, means the harness is broken rather than merely unlucky.
    assert out["n_test"] == 1
    assert out["accuracy"] in (0.0, 1.0)
    assert math.isfinite(out["f1_macro"])
