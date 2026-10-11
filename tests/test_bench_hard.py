"""Smoke tests for the adversarial hard-eval harness.

The transforms are deterministic and the report carries an env
fingerprint. The unit tests drive the transforms directly; the report
smoke test asserts the metric families are present, following the
bench-report prior art.
"""

import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_hard", ROOT / "scripts" / "bench_hard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_transforms_are_deterministic():
    mod = _load()
    text, schema = "ISS-1 crash on boot priority: high labels: api state: open", "issue"
    for name, fn in mod.TRANSFORMS.items():
        first = fn(text, schema, random.Random(42))
        second = fn(text, schema, random.Random(42))
        assert first == second, name


def test_confuse_makes_an_issue_look_like_a_pull_request():
    mod = _load()
    out = mod.confuse("ISS-1 crash", "issue", random.Random(0))
    assert out.startswith("PR #999")


def test_fingerprint_names_the_stack_and_git():
    fp = _load()._fingerprint()
    assert "scikit-learn" in fp and "git" in fp
