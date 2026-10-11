"""Gates for the learned agent_step status extractor.

The rule extractor already reads canonical markers; the learned model
covers free-form traces where those markers are absent (the ``stripped``
proxy: canonical ``return_code``/``status`` fields removed, labels kept).
It ships only if it beats the rules baseline on stripped dev without
regressing canonical dev in assist mode.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "train_agent_step_status",
        ROOT / "scripts" / "train_agent_step_status.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load()


def test_strip_removes_canonical_markers_only():
    raw = ("tool: Bash; arguments: pytest -q; stdout: all green; "
           "return_code: 0; status: success")
    stripped = T.strip_canonical(raw)
    assert "return_code" not in stripped and "status:" not in stripped
    assert "pytest" in stripped and "green" in stripped


def test_status_features_are_deterministic():
    text = "tool: Bash; stdout: FAILED 3 tests; return_code: 1"
    assert T.status_features(text) == T.status_features(text)
    assert len(T.status_features(text)) > 0


def test_learner_separates_toy_markers():
    train = [("job completed ok wall time 0.1 seconds", "success"),
             ("all 12 tests passed", "success"),
             ("FAILED tests/test_queue.py::test_drain", "error"),
             ("exit code 1 segmentation fault", "error"),
             ("execution timed out after 30 seconds", "timeout"),
             ("timed out waiting for lock", "timeout"),
             ("keyboardinterrupt received, cancelled by user", "interrupted"),
             ("cancelled mid-run by operator", "interrupted")]
    model = T.fit_status([t for t, _ in train], [s for _, s in train])
    pred = list(model.predict([T.status_features(t) for t, _ in train]))
    assert pred == [s for _, s in train]


def test_rules_baseline_helpers_agree_with_extractor():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from mivia_small_json.agent_step import extract_agent_step
    assert extract_agent_step(
        "tool: Bash; stdout: ok; return_code: 0; status: success"
    )["status"] == "success"
    assert T.strip_canonical("a status: success b") == "a b"
