"""Tests for the ordinal severity head trainer and its metrics.

Ticket: severity ordinal loss swap. The script is a candidate-only trainer
(`scripts/train_severity_ordinal.py`); these tests pin the pure pieces that
decide the verdict: the ordinal metrics, the CORAL/CORN decodes, and the
scope guards (candidate path, frozen panel, pre-registered bar).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROMOTED = ROOT / "models" / "severity-ordinal-int8.onnx"


def _load():
    spec = importlib.util.spec_from_file_location(
        "train_severity_ordinal", ROOT / "scripts" / "train_severity_ordinal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def test_quadratic_weighted_kappa_is_one_on_perfect_agreement():
    assert M.quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)


def test_quadratic_weighted_kappa_penalises_reversal():
    reversed_preds = [3, 2, 1, 0]
    assert M.quadratic_weighted_kappa([0, 1, 2, 3], reversed_preds) < 0.0


def test_quadratic_weighted_kappa_ignores_distance_scale():
    # A single off-by-one error costs less than the same count of far misses.
    near = M.quadratic_weighted_kappa([0, 0, 1, 1], [0, 1, 1, 2])
    far = M.quadratic_weighted_kappa([0, 0, 1, 1], [2, 3, 3, 3])
    assert near > far


def test_ordinal_metrics_counts_distance_and_within_one():
    m = M.ordinal_metrics([0, 1, 2, 3], [0, 1, 2, 2])
    assert m["n"] == 4
    assert m["exact"] == pytest.approx(75.0)
    assert m["aod"] == pytest.approx(0.25)
    assert m["within_one"] == pytest.approx(100.0)


def test_ordinal_metrics_penalises_full_reversal():
    # [3, 2, 1, 0] reverses the order: every row is wrong and the two middle
    # rows are only one tier off, so within-one is half.
    m = M.ordinal_metrics([0, 1, 2, 3], [3, 2, 1, 0])
    assert m["exact"] == pytest.approx(0.0)
    assert m["aod"] == pytest.approx(2.0)
    assert m["within_one"] == pytest.approx(50.0)


def test_ordinal_metrics_within_one_is_zero_when_all_misses_are_far():
    m = M.ordinal_metrics([0, 0, 1, 1], [3, 3, 3, 3])
    assert m["exact"] == pytest.approx(0.0)
    assert m["within_one"] == pytest.approx(0.0)


def test_ordinal_metrics_handles_empty_group():
    m = M.ordinal_metrics([], [])
    assert m["n"] == 0 and m["exact"] == 0.0 and m["qwk"] == 0.0


def test_coral_decode_counts_positive_cutpoints():
    assert M.coral_decode([1.0, 1.0, 1.0]) == 3
    assert M.coral_decode([1.0, -1.0, -1.0]) == 1
    assert M.coral_decode([-1.0, -1.0, -1.0]) == 0
    # Strictly greater than zero: an exact cutpoint hit stays below it.
    assert M.coral_decode([0.0, 0.0, 0.0]) == 0


def test_corn_decode_stays_in_range_and_matches_extremes():
    assert M.corn_decode([10.0, 10.0, 10.0]) == 3
    assert M.corn_decode([-10.0, -10.0, -10.0]) == 0
    for logits in ([0.3, -0.3, 5.0], [5.0, -5.0, 0.0], [1.0, 1.0, -1.0]):
        assert 0 <= M.corn_decode(logits) <= 3


def test_corn_decode_is_monotone_in_the_first_cutpoint():
    # A larger first logit cannot lower the decoded tier.
    low = M.corn_decode([-0.5, 1.0, 1.0])
    high = M.corn_decode([0.5, 1.0, 1.0])
    assert high >= low


def test_candidate_path_is_gitignored_never_models():
    out = M.DEFAULT_OUT.resolve()
    assert out.parent == (ROOT / "data" / "incoming" / "models").resolve()
    assert (ROOT / "models") not in out.parents


def test_frozen_panel_is_the_register_panel():
    assert M.PANEL.resolve() == (ROOT / "data" / "heldout" /
                                 "grade-register-panel.jsonl").resolve()


def test_pre_registered_bar_is_the_stated_one():
    assert M.BAR == {"overall_exact": 50.0, "overall_within_one": 80.0,
                     "delta_points": 10.0, "p": 0.05}


# (panel-strata gate retired on the public tree: the frozen register
# panel is not redistributed; the private tree keeps the gate.)

def test_promoted_artifact_exists_matches_sidecar_and_manifest():
    import hashlib
    import json

    assert PROMOTED.is_file(), "models/severity-ordinal-int8.onnx missing"
    digest = hashlib.sha256(PROMOTED.read_bytes()).hexdigest()
    sidecar = (ROOT / "models" / "severity-ordinal-int8.onnx.sha256").read_text()
    assert digest == sidecar.strip()
    manifest = json.loads(
        (ROOT / "models" / "SHIPPED.json").read_text())["artifacts"]
    assert manifest["severity-ordinal-int8.onnx"]["sha256"] == digest


def test_promoted_artifact_has_cumulative_logits_shape():
    import onnxruntime as ort

    sess = ort.InferenceSession(str(PROMOTED), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input_ids", "attention_mask"]
    outputs = sess.get_outputs()
    assert outputs[0].name == "cumulative_logits"
    assert outputs[0].shape == ["batch", M.K - 1]


def test_load_severity_ordinal_loads_and_decodes():
    from mivia_small_json.severity_head import CANONICAL_SEVERITY_ANCHORS
    from mivia_small_json.tool_match import load_severity_ordinal

    head = load_severity_ordinal()
    assert head is not None, "promoted ordinal head failed to load"
    logits = head.logits("ERROR: disk failure on /dev/sda, unrecoverable")
    assert len(logits) == M.K - 1
    anchor, score = head.grade(
        "ERROR: disk failure on /dev/sda, unrecoverable",
        list(CANONICAL_SEVERITY_ANCHORS))
    assert anchor in CANONICAL_SEVERITY_ANCHORS
    assert 0.0 <= score <= 1.0


def test_ordinal_grade_with_set_is_a_singleton_tier():
    from mivia_small_json.severity_head import (
        CANONICAL_SEVERITY_ANCHORS, cumulative_tier)
    from mivia_small_json.tool_match import load_severity_ordinal

    head = load_severity_ordinal()
    assert head is not None
    text = "The backup restore failed and the primary ledger database is unrecoverable"
    anchor, score, tier_set = head.grade(text, list(CANONICAL_SEVERITY_ANCHORS),
                                         with_set=True)
    assert tier_set == [cumulative_tier(head.logits(text))]
    assert CANONICAL_SEVERITY_ANCHORS[tier_set[0]] == anchor


def test_load_severity_ordinal_never_raises_on_bad_input():
    from mivia_small_json.tool_match import load_severity_ordinal

    # Missing path, a directory, and a garbage file all degrade to None.
    assert load_severity_ordinal(path=ROOT / "models" / "nope.onnx") is None
    assert load_severity_ordinal(path=ROOT / "models") is None


def test_load_severity_ordinal_sha_verifies_the_artifact(tmp_path):
    """A tampered copy must fail the sha256 check and load as None."""
    from mivia_small_json.tool_match import load_severity_ordinal

    tampered = tmp_path / "severity-ordinal-int8.onnx"
    tampered.write_bytes(PROMOTED.read_bytes() + b"\x00")
    (tmp_path / "severity-ordinal-int8.onnx.sha256").write_text("0" * 64 + "\n")
    assert load_severity_ordinal(path=tampered) is None


def test_ordinal_head_grade_never_raises_on_empty_text():
    from mivia_small_json.severity_head import CANONICAL_SEVERITY_ANCHORS
    from mivia_small_json.tool_match import load_severity_ordinal

    head = load_severity_ordinal()
    assert head is not None
    anchor, score = head.grade(None, list(CANONICAL_SEVERITY_ANCHORS))
    assert anchor in CANONICAL_SEVERITY_ANCHORS
