"""Gates for the one train-all entry point.

The step list is the contract: synthetic generation, the router, all
three extractors, then the bench verdict, in that order. The level
extractor must be present — the cycle's extractor retrain once omitted
it and let the artifact go stale. The script is loaded via importlib
(as tests/test_backfill.py does) so importing it never runs a step.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_STEPS = (
    ("-m", "mivia_small_json.synth"),
    ("-m", "mivia_small_json.train"),
    ("scripts/train_level_extractor.py",),
    ("scripts/train_component_extractor.py",),
    ("scripts/train_implicit_severity.py",),
    ("scripts/export_encoder_onnx.py",),
    ("-m", "mivia_small_json.bench"),
)


def _load_train_all():
    spec = importlib.util.spec_from_file_location(
        "train_all", ROOT / "scripts" / "train_all.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_steps_are_exactly_the_seven_in_order():
    """The step list is the seven artifacts, in the documented order."""
    mod = _load_train_all()
    assert tuple(tuple(step) for step in mod.STEPS) == EXPECTED_STEPS


def test_level_extractor_is_in_the_step_list():
    """The level extractor is retrained by the entry point."""
    mod = _load_train_all()
    assert any("train_level_extractor.py" in " ".join(step) for step in mod.STEPS)
