# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the label-head learner
(``mivia_small_json.fewshot_classifier``, ticket 01 of
``.scratch/fewshot-internal-endpoints/``).

Pure module tests: no HTTP, no file persistence (artifacts stay bytes).
Validation tests run without the encoder (validation precedes embedding);
every test that trains or predicts follows the house skip convention
(test_fewshot_onboarding.py).
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mivia_small_json import fewshot_classifier as fc
from mivia_small_json.fewshot import ENUM_CLF


def _labels(n=2, per=6, start=0):
    """n labels x per distinct short examples, no cross-label overlaps."""
    return [
        {
            "name": f"label{i + start}",
            "examples": [
                f"{i + start} example {j} text about topic {i + start}"
                for j in range(per)
            ],
        }
        for i in range(n)
    ]


def _require_encoder():
    if fc.encoder() is None:
        pytest.skip("shipped sentence encoder not available")


def _artifact_bytes(out: dict) -> bytes:
    return base64.b64decode(out["head_artifact_b64"])


def _train_default(**kw) -> dict:
    return fc.train_label_head(kw.pop("labels", _labels()), **kw)


# ---------------------------------------------------------------------------
# Caps: over-limit is refused, never truncated (message names the cap).
# Validation only -> no encoder needed.
# ---------------------------------------------------------------------------


def test_cap_max_fs_labels():
    with pytest.raises(ValueError) as exc:
        fc.train_label_head(_labels(51, per=4), seed=1)
    assert "MAX_FS_LABELS" in str(exc.value)


def test_cap_examples_per_label():
    with pytest.raises(ValueError) as exc:
        fc.train_label_head(_labels(2, per=51), seed=1)
    assert "MAX_FS_EXAMPLES_PER_LABEL" in str(exc.value)


def test_cap_example_chars():
    labels = _labels(2, per=4)
    labels[0]["examples"][0] = "x" * 2001
    with pytest.raises(ValueError) as exc:
        fc.train_label_head(labels, seed=1)
    assert "MAX_FS_EXAMPLE_CHARS" in str(exc.value)


def test_cap_total_examples():
    labels = _labels(20, per=38)  # 760
    labels.append({"name": "extra", "examples": [f"extra {j}" for j in range(41)]})
    assert sum(len(lab["examples"]) for lab in labels) == 801
    with pytest.raises(ValueError) as exc:
        fc.train_label_head(labels, seed=1)
    assert "MAX_FS_TOTAL_EXAMPLES" in str(exc.value)


def test_cap_total_chars_is_overflowerror():
    labels = _labels(9, per=45)  # 405 examples of exactly 2000 chars
    for lab in labels:
        lab["examples"] = [
            f"{lab['name']}-{j} ".ljust(2000, "x") for j in range(45)
        ]
    assert sum(len(e) for lab in labels for e in lab["examples"]) > 800_000
    with pytest.raises(OverflowError) as exc:
        fc.train_label_head(labels, seed=1)
    assert "MAX_FS_TOTAL_CHARS" in str(exc.value)


# ---------------------------------------------------------------------------
# Shape floors.
# ---------------------------------------------------------------------------


def test_single_label_refused():
    with pytest.raises(ValueError):
        fc.train_label_head(_labels(1, per=6), seed=1)


def test_label_with_three_examples_refused():
    labels = _labels(2, per=6)
    labels[1]["examples"] = labels[1]["examples"][:3]
    with pytest.raises(ValueError):
        fc.train_label_head(labels, seed=1)


def test_duplicate_label_name_after_normalize_refused():
    labels = _labels(2, per=6)
    labels[1]["name"] = "  Label0 "  # collect.normalize collides with label0
    with pytest.raises(ValueError):
        fc.train_label_head(labels, seed=1)


def test_cross_label_duplicate_text_is_contradiction():
    labels = _labels(2, per=6)
    labels[1]["examples"][0] = "  " + labels[0]["examples"][0].upper() + " "
    with pytest.raises(ValueError) as exc:
        fc.train_label_head(labels, seed=1)
    assert "contradiction" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Happy path: the frozen train response shape.
# ---------------------------------------------------------------------------


def test_train_response_shape():
    _require_encoder()
    out = _train_default(seed=7)
    assert set(out) == {"head_artifact_b64", "head_sha256", "metrics",
                        "backbone", "learner"}
    raw = _artifact_bytes(out)
    assert out["head_sha256"] == hashlib.sha256(raw).hexdigest()

    metrics = out["metrics"]
    assert set(metrics) == {"schema", "per_label", "eval_mode", "eval_n",
                            "mean_accuracy", "adjacency_warnings",
                            "thresholds"}
    assert metrics["schema"] == 2
    # 2x6 = 12 rows sits under the LOO gate: every row is an eval point.
    assert metrics["eval_mode"] == "loo"
    assert metrics["eval_n"] == 12
    assert set(metrics["per_label"]) == {"label0", "label1"}
    for entry in metrics["per_label"].values():
        assert set(entry) == {"held_out_n", "eval_n", "accuracy", "confusions"}
        assert entry["held_out_n"] == 0  # loo mode: no holdout pool
        assert entry["eval_n"] == 6
        assert isinstance(entry["accuracy"], float)
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert all(isinstance(c, str) for c in entry["confusions"])
    assert isinstance(metrics["mean_accuracy"], float)
    assert metrics["thresholds"] == {"low_confidence": 0.80}
    assert all(len(pair) == 2 for pair in metrics["adjacency_warnings"])
    head = fc.load_label_head(_artifact_bytes(out), out["head_sha256"])
    assert head.meta["version"] == 2
    assert head.meta["eval_mode"] == "loo"
    assert head.meta["seed"] == 7
    assert out["backbone"]["artifact"] == "tool-pruner-embed-int8.onnx"
    assert re.fullmatch(r"[0-9a-f]{64}", out["backbone"]["sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", out["learner"]["fewshot_py_sha256"])
    assert out["learner"]["fewshot_py_sha256"] == hashlib.sha256(
        Path(fc.__file__).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Determinism: content-level gate (skops bytes are NOT reproducible).
# ---------------------------------------------------------------------------


def test_refit_twice_is_content_identical():
    _require_encoder()
    a = _train_default(labels=_labels(3, per=6), seed=11)
    b = _train_default(labels=_labels(3, per=6), seed=11)
    assert a["metrics"] == b["metrics"]
    ha = fc.load_label_head(_artifact_bytes(a), a["head_sha256"])
    hb = fc.load_label_head(_artifact_bytes(b), b["head_sha256"])
    assert np.array_equal(ha.clf.coef_, hb.clf.coef_)
    assert ha.clf.classes_.tolist() == hb.clf.classes_.tolist()
    assert ha.classes == hb.classes


# The seed gate is the fitted coef_, not the artifact sha and not the
# per-label metrics: skops bytes are never reproducible (same-seed refits
# already differ), and accuracy lives in a coarse value space where two
# honest different splits can collide. The seed test runs OVER the LOO
# gate (3x40 = 120 rows, holdout): there the seed still picks the split,
# so different held rows mean different training rows and the fitted
# coef_ cannot match.
def test_different_seed_shifts_the_split():
    _require_encoder()
    a = _train_default(labels=_labels(3, per=40), seed=1)
    b = _train_default(labels=_labels(3, per=40), seed=2)
    assert a["metrics"]["eval_mode"] == "holdout"
    ha = fc.load_label_head(_artifact_bytes(a), a["head_sha256"])
    hb = fc.load_label_head(_artifact_bytes(b), b["head_sha256"])
    assert not np.array_equal(ha.clf.coef_, hb.clf.coef_)


# LOO leaves the seed nothing to split: the shipped head fits all rows in
# the canonical order (label input order, then example text), so coef_ is
# bitwise seed-stable, metrics identical, only meta.seed moves the sha.
def test_loo_seed_stability_bitwise_coef_and_metrics():
    _require_encoder()
    a = _train_default(labels=_labels(3, per=6), seed=1)
    b = _train_default(labels=_labels(3, per=6), seed=2)
    assert a["metrics"] == b["metrics"]
    ha = fc.load_label_head(_artifact_bytes(a), a["head_sha256"])
    hb = fc.load_label_head(_artifact_bytes(b), b["head_sha256"])
    assert np.array_equal(ha.clf.coef_, hb.clf.coef_)
    assert a["head_sha256"] != b["head_sha256"]


# ---------------------------------------------------------------------------
# The 4-examples-per-label edge: the holdout pool would be empty, so the
# hybrid flips to LOO and every row is scored by a refit on the others.
# ---------------------------------------------------------------------------


def test_four_examples_per_label_uses_loo():
    _require_encoder()
    out = _train_default(labels=_labels(3, per=4), seed=3)
    m = out["metrics"]
    assert m["eval_mode"] == "loo"
    assert m["eval_n"] == 12
    per = m["per_label"]
    assert all(e["held_out_n"] == 0 for e in per.values())
    assert all(e["eval_n"] == 4 for e in per.values())
    assert all(isinstance(e["accuracy"], float) for e in per.values())
    assert all(0.0 <= e["accuracy"] <= 1.0 for e in per.values())
    assert all(isinstance(e["confusions"], list) for e in per.values())
    assert isinstance(m["mean_accuracy"], float)


# ---------------------------------------------------------------------------
# The LOO gate (spec section 2): <= 100 total rows is loo, over is holdout.
# ---------------------------------------------------------------------------


# 2 labels cannot exceed the gate: the per-label cap is 50 examples.
def test_gate_boundary_100_rows_is_loo():
    _require_encoder()
    out = _train_default(labels=_labels(2, per=50), seed=1)
    m = out["metrics"]
    assert m["eval_mode"] == "loo"
    assert m["eval_n"] == 100


def test_gate_boundary_101_rows_is_holdout():
    _require_encoder()
    # First row count over the gate: two labels cap at 100, so 101 needs
    # a 34/34/33 split. One row past 100 must already be holdout.
    labels = _labels(3, per=34)
    labels[2]["examples"] = labels[2]["examples"][:33]
    out = _train_default(labels=labels, seed=1)
    m = out["metrics"]
    assert m["eval_mode"] == "holdout"
    assert m["eval_mode"] != "loo"


def test_gate_boundary_102_rows_is_holdout():
    _require_encoder()
    out = _train_default(labels=_labels(3, per=34), seed=1)
    m = out["metrics"]
    assert m["eval_mode"] == "holdout"
    assert m["eval_n"] == 21  # three held pools of min(8, ceil(0.2*34)) = 7
    assert all(e["accuracy"] is not None for e in m["per_label"].values())


# The wizard-minimum large shape the 100 gate exists for (spec section 2).
def test_wizard_minimum_16x4_is_loo():
    _require_encoder()
    out = _train_default(labels=_labels(16, per=4), seed=1)
    m = out["metrics"]
    assert m["eval_mode"] == "loo"
    assert m["eval_n"] == 64


# ---------------------------------------------------------------------------
# Holdout honesty over the gate: an empty pool is null, never a fake 0.0.
# ---------------------------------------------------------------------------


def test_holdout_all_empty_pools_is_null_everywhere():
    _require_encoder()
    out = _train_default(labels=_labels(26, per=4), seed=1)
    m = out["metrics"]
    assert m["eval_mode"] == "holdout"
    per = m["per_label"]
    assert len(per) == 26
    assert all(e["accuracy"] is None for e in per.values())
    assert all(e["held_out_n"] == 0 and e["eval_n"] == 0 for e in per.values())
    assert m["mean_accuracy"] is None
    assert m["eval_n"] == 0
    assert m["adjacency_warnings"] == []


def test_holdout_mixed_null_label_mean_over_contributing():
    _require_encoder()
    labels = _labels(2, per=49)
    labels.append({"name": "tiny",
                   "examples": ["tiny alpha one", "tiny alpha two",
                                "tiny alpha three", "tiny alpha four"]})
    out = _train_default(labels=labels, seed=1)
    m = out["metrics"]
    assert m["eval_mode"] == "holdout"  # 49 + 49 + 4 = 102 rows
    assert m["per_label"]["tiny"]["accuracy"] is None
    assert m["per_label"]["tiny"]["eval_n"] == 0
    accs = [e["accuracy"] for e in m["per_label"].values()
            if e["accuracy"] is not None]
    assert len(accs) == 2
    assert m["mean_accuracy"] == pytest.approx(sum(accs) / len(accs))
    assert m["eval_n"] == 16  # two held pools of min(8, ceil(0.2*49)) = 8


# ---------------------------------------------------------------------------
# LOO internals: the shipped head fits ALL rows canonically; refits only
# score. Adjacency stays computable at 4 examples per label.
# ---------------------------------------------------------------------------


def test_loo_head_fits_all_rows_canonical_order():
    _require_encoder()
    labels = _labels(3, per=5)
    out = _train_default(labels=labels, seed=9)
    head = fc.load_label_head(_artifact_bytes(out), out["head_sha256"])
    enc = fc.encoder()
    texts = [t for lab in labels for t in lab["examples"]]
    ys = [lab["name"] for lab in labels for _ in lab["examples"]]
    X = np.stack([enc.embed_one(t) for t in texts])
    refit = LogisticRegression(**ENUM_CLF)
    refit.fit(X, ys)
    assert np.array_equal(head.clf.coef_, refit.coef_)


# Two LOO mispredictions of alpha as beta clear both adjacency bars
# (>= 2 count and >= 25% of the 4-row eval pool).
def test_adjacency_from_loo_mispredictions_at_four_per_label():
    _require_encoder()
    labels = [
        {"name": "alpha",
         "examples": ["alpha ledger entry one", "alpha ledger entry two",
                      "beta router setup four", "beta router setup five"]},
        {"name": "beta",
         "examples": ["beta router config one", "beta router config two",
                      "beta router config three", "beta router config six"]},
    ]
    out = _train_default(labels=labels, seed=1)
    m = out["metrics"]
    assert m["eval_mode"] == "loo"
    assert ["alpha", "beta"] in m["adjacency_warnings"]
    assert m["per_label"]["alpha"]["confusions"][0] == "beta"
    assert m["per_label"]["alpha"]["accuracy"] == 0.5


# ---------------------------------------------------------------------------
# Infer roundtrip (in-memory, no HTTP).
# ---------------------------------------------------------------------------


def test_predict_roundtrip():
    _require_encoder()
    out = _train_default(seed=5)
    head = fc.load_label_head(_artifact_bytes(out), out["head_sha256"])
    label, confidence, candidates = head.predict(
        "example 9 text about topic 0")
    assert label in {"label0", "label1"}
    assert 0.0 <= confidence <= 1.0
    assert 1 <= len(candidates) <= 2
    assert set(candidates) <= {"label0", "label1"}


# ---------------------------------------------------------------------------
# Load gates: tamper, sha, truncation, oversize.
# ---------------------------------------------------------------------------


def test_tampered_byte_refused():
    _require_encoder()
    out = _train_default(seed=5)
    raw = bytearray(_artifact_bytes(out))
    raw[len(raw) // 2] ^= 0x01
    with pytest.raises(ValueError):
        fc.load_label_head(bytes(raw), out["head_sha256"])


def test_wrong_sha_refused():
    _require_encoder()
    out = _train_default(seed=5)
    with pytest.raises(ValueError):
        fc.load_label_head(_artifact_bytes(out), "0" * 64)


def test_truncated_skops_refused():
    _require_encoder()
    out = _train_default(seed=5)
    raw = _artifact_bytes(out)
    cut = raw[: len(raw) // 2]
    # correct sha for the truncated bytes: the skops LOAD gate must fire
    with pytest.raises(ValueError):
        fc.load_label_head(cut, hashlib.sha256(cut).hexdigest())


def test_oversize_artifact_refused():
    blob = b"0" * (256 * 1024 + 1)
    with pytest.raises(ValueError) as exc:
        fc.load_label_head(blob, hashlib.sha256(blob).hexdigest())
    assert "256" in str(exc.value) or "artifact" in str(exc.value)
