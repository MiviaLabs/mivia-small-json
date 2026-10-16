# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Label-head few-shot classification for the internal endpoints
(spec ``docs/specs/fewshot-internal-endpoints/spec.md`` sections 2.1/3.1).

A label head turns ``[{name, examples}, ...]`` plus a seed into a trained
single ``LogisticRegression`` over the SHIPPED sentence encoder's
embeddings (``fewshot.encoder`` — the sha-verified tool-pruner ONNX
export), persisted as skops BYTES. No file is ever written: the caller
(the platform, via ticket 02's routes) owns the artifact bytes.

Calibration is the genuinely new part: a hybrid eval mode per train call
(honest-fewshot-metrics spec). At or under ``MAX_LOO_ROWS`` total rows,
leave-one-out scores every row (self-consistency, siblings still in each
fold); over the gate, the per-label held-out split with ``accuracy: null``
when a pool is empty - never a fabricated zero. Both modes ship accuracy,
top-3 confusions, and adjacency warnings (label ``a`` confused as ``b`` by
enough eval rows to matter). The platform overlays its own confidence bar
per version; this module only ships the default threshold.

Determinism carries the registered correction (few-shot-onboarding spec,
2026-10-10): skops BYTES are not reproducible (``__id__`` memory
addresses), so the gate is content-level — refitting twice on the same
inputs + seed yields bit-identical ``coef_`` / ``classes_`` and identical
metrics, pinned by a test. Per-label shuffles derive their streams from
``f"{seed}:{label}"`` (``hash()`` is forbidden: PYTHONHASHSEED-random per
process).
"""

from __future__ import annotations

import base64
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import integrity
from .collect import normalize
from .fewshot import ENUM_CLF, encoder

# Caps (spec section 2.1): over-limit is an explicit error the routes map
# to 4xx, never a silent truncation. OverflowError (total chars) maps to 413.
MAX_FS_LABELS = 50
MAX_FS_EXAMPLES_PER_LABEL = 50
MAX_FS_EXAMPLE_CHARS = 2000
MAX_FS_TOTAL_EXAMPLES = 800
MAX_FS_TOTAL_CHARS = 800_000
MIN_FS_LABELS = 2
MIN_FS_EXAMPLES_PER_LABEL = 4
LOW_CONFIDENCE_DEFAULT = 0.80
MAX_ARTIFACT_BYTES = 256 * 1024
# Adjacency warning bars: a confusion must clear BOTH to be reported.
ADJACENCY_MIN_COUNT = 2
ADJACENCY_MIN_SHARE = 0.25
MAX_CONFUSIONS = 3
# Held-out split: min(8, ceil(0.2 n)), 0 when training would keep < 4 rows.
MAX_HELD_OUT = 8
# LOO gate (honest-fewshot-metrics spec section 2): <= this many total rows
# scores every row by refit; over it, holdout. 100 covers 25 labels x 4.
MAX_LOO_ROWS = 100


@dataclass
class LabelHead:
    """A loaded head: declared labels, the fitted classifier, its classes."""

    labels: list[str]
    clf: object
    classes: list[str]
    meta: dict

    def predict(self, text: str) -> tuple[str, float, list[str]]:
        """(top label, confidence, top-2 candidates) for one text."""
        enc = encoder()
        if enc is None:
            raise RuntimeError("encoder unavailable")
        vec = np.asarray(enc.embed_one(text), dtype=np.float32).reshape(1, -1)
        probs = self.clf.predict_proba(vec)[0]
        order = np.argsort(-probs, kind="stable")  # stable: deterministic ties
        top = int(order[0])
        label = self.classes[top]
        confidence = float(probs[top])
        candidates = [self.classes[int(i)] for i in order[:2]]
        return label, confidence, candidates


def _validate(labels: list[dict]) -> list[tuple[str, list[str]]]:
    """Shape, cap, and contradiction checks; (name, examples) pairs out.

    ValueError maps to 422 at the route; total chars over the cap raises
    OverflowError, which maps to 413 (the platform's own limits allow
    800 x 2000 = 1.6M chars, so legal traffic can breach it).
    """
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels must be a non-empty list of {name, examples}")
    names: list[str] = []
    seen_names: set[str] = set()
    for i, entry in enumerate(labels):
        if not isinstance(entry, dict):
            raise ValueError(f"labels[{i}] must be an object with name and examples")
        name = entry.get("name")
        examples = entry.get("examples")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"labels[{i}].name must be a non-empty string")
        if not isinstance(examples, list) or not all(
                isinstance(e, str) for e in examples):
            raise ValueError(f"labels[{i}].examples must be a list of strings")
        key = normalize(name)
        if key in seen_names:
            raise ValueError(f"duplicate label name: {name!r}")
        seen_names.add(key)
        names.append(name)
    if len(labels) < MIN_FS_LABELS:
        raise ValueError(f"at least {MIN_FS_LABELS} labels are required")
    if len(labels) > MAX_FS_LABELS:
        raise ValueError(f"labels exceed MAX_FS_LABELS={MAX_FS_LABELS}")
    total_examples = 0
    total_chars = 0
    for name, entry in zip(names, labels):
        examples: list[str] = entry["examples"]
        n = len(examples)
        if n < MIN_FS_EXAMPLES_PER_LABEL:
            raise ValueError(f"label {name!r} has {n} examples; "
                             f"minimum is {MIN_FS_EXAMPLES_PER_LABEL}")
        if n > MAX_FS_EXAMPLES_PER_LABEL:
            raise ValueError(f"label {name!r} exceeds "
                             f"MAX_FS_EXAMPLES_PER_LABEL={MAX_FS_EXAMPLES_PER_LABEL}")
        if any(len(e) > MAX_FS_EXAMPLE_CHARS for e in examples):
            raise ValueError(f"an example in label {name!r} exceeds "
                             f"MAX_FS_EXAMPLE_CHARS={MAX_FS_EXAMPLE_CHARS}")
        total_examples += n
        total_chars += sum(len(e) for e in examples)
    if total_examples > MAX_FS_TOTAL_EXAMPLES:
        raise ValueError(f"examples exceed MAX_FS_TOTAL_EXAMPLES="
                         f"{MAX_FS_TOTAL_EXAMPLES}")
    if total_chars > MAX_FS_TOTAL_CHARS:
        raise OverflowError(f"examples exceed MAX_FS_TOTAL_CHARS="
                            f"{MAX_FS_TOTAL_CHARS}")
    owner: dict[str, str] = {}
    for name, entry in zip(names, labels):
        for text in entry["examples"]:
            key = normalize(text)
            seen = owner.get(key)
            if seen is not None and seen != name:
                raise ValueError(f"contradiction: the same example text "
                                 f"appears under labels {seen!r} and {name!r}")
            owner[key] = name
    return [(name, list(entry["examples"])) for name, entry in zip(names, labels)]


def _held_out_n(n: int) -> int:
    hold = min(MAX_HELD_OUT, math.ceil(0.2 * n))
    if n - hold < MIN_FS_EXAMPLES_PER_LABEL:
        return 0
    return hold


def _ranked(counts: dict[str, int]) -> list[str]:
    """Keys by descending count, ties by name (deterministic)."""
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _backbone() -> dict:
    from .tool_match import EMBED_PRUNER_MODEL

    return {"artifact": EMBED_PRUNER_MODEL.name,
            "sha256": integrity.sha256_file(EMBED_PRUNER_MODEL)}


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def train_label_head(labels: list[dict], seed: int) -> dict:
    """Train one label head; the frozen train response (section 2.1)."""
    rows = _validate(labels)
    enc = encoder()
    if enc is None:
        raise RuntimeError("encoder unavailable")
    prepared: list[tuple[str, list[tuple[str, int]], list[tuple[str, int]]]] = []
    canon_texts: list[str] = []
    canon_labels: list[str] = []
    for name, examples in rows:
        # Derived stream per label; NEVER hash(label) (PYTHONHASHSEED-random).
        # Pairs carry the canonical row index through the shuffle.
        pairs = [(t, i) for i, t in enumerate(examples, start=len(canon_texts))]
        random.Random(f"{seed}:{name}").shuffle(pairs)
        n_hold = _held_out_n(len(pairs))
        prepared.append((name, pairs[:n_hold], pairs[n_hold:]))
        canon_texts.extend(examples)
        canon_labels.extend([name] * len(examples))
    X = np.stack([enc.embed_one(t) for t in canon_texts])
    y = np.array(canon_labels)
    total_rows = len(canon_texts)
    eval_mode = "loo" if total_rows <= MAX_LOO_ROWS else "holdout"
    clf = LogisticRegression(**ENUM_CLF)
    if eval_mode == "loo":
        # Canonical fit order: the seed has nothing to split, coef_ is stable.
        clf.fit(X, canon_labels)
        per_label, adjacency, accuracy_sum, contributing = _loo_scores(
            prepared, X, y)
        eval_n = total_rows
        n_trained = total_rows
    else:
        train_idx = [i for _, _held, train in prepared for _t, i in train]
        clf.fit(X[train_idx], y[train_idx])
        (per_label, adjacency, accuracy_sum, contributing,
         eval_n) = _holdout_scores(prepared, clf, X)
        n_trained = len(train_idx)
    classes = [str(c) for c in clf.classes_]
    meta = {
        "version": 2,
        "encoder_sha256_16": fewshot_encoder_sha16(),
        "n": n_trained,
        "seed": seed,
        "eval_mode": eval_mode,
        "labels": [name for name, _, _ in prepared],
        "per_label": per_label,
    }
    import skops.io as sio

    artifact = sio.dumps({"meta": meta, "clf": clf, "classes": classes})
    return {
        "head_artifact_b64": base64.b64encode(artifact).decode("ascii"),
        "head_sha256": hashlib.sha256(artifact).hexdigest(),
        "metrics": {
            # Schema marker on the metrics blob itself: stored metrics_json
            # is frozen (never rewritten), so frozen rows must stay
            # self-describing without guessing from key presence.
            "schema": 2,
            "per_label": per_label,
            "eval_mode": eval_mode,
            "eval_n": eval_n,
            "mean_accuracy": accuracy_sum / contributing if contributing
            else None,
            "adjacency_warnings": adjacency,
            "thresholds": {"low_confidence": LOW_CONFIDENCE_DEFAULT},
        },
        "backbone": _backbone(),
        "learner": {"fewshot_py_sha256": _module_sha256()},
    }


def _adjacency_others(wrong: dict[str, int], pool_n: int) -> list[str]:
    """Wrong labels clearing BOTH adjacency bars (ranked, deterministic)."""
    return [other for other in _ranked(wrong)
            if wrong[other] >= ADJACENCY_MIN_COUNT
            and wrong[other] >= ADJACENCY_MIN_SHARE * pool_n]


def _loo_scores(
    prepared: list[tuple[str, list[tuple[str, int]], list[tuple[str, int]]]],
    X: np.ndarray, y: np.ndarray,
) -> tuple[dict, list[list[str]], float, int]:
    """Leave-one-out scoring: refit on all OTHER rows, predict the row.

    Refits visit the per-label shuffled pools; each refit's training
    matrix keeps the canonical row order, so scores cannot drift with the
    seed even in float low bits. Siblings stay in every fold, so LOO is a
    self-consistency check, not a production accuracy estimate.
    """
    per_label: dict[str, dict] = {}
    adjacency: list[list[str]] = []
    accuracy_sum = 0.0
    contributing = 0
    for name, held, train in prepared:
        pool = held + train  # the full shuffled pool
        wrong: dict[str, int] = {}
        hits = 0
        for _text, idx in pool:
            keep = np.arange(len(y)) != idx
            refit = LogisticRegression(**ENUM_CLF)
            refit.fit(X[keep], y[keep])
            pred = str(refit.predict(X[idx:idx + 1])[0])
            if pred == name:
                hits += 1
            else:
                wrong[pred] = wrong.get(pred, 0) + 1
        accuracy = hits / len(pool)
        per_label[name] = {
            "held_out_n": 0,
            "eval_n": len(pool),
            "accuracy": accuracy,
            "confusions": _ranked(wrong)[:MAX_CONFUSIONS],
        }
        accuracy_sum += accuracy
        contributing += 1
        adjacency.extend([name, other]
                         for other in _adjacency_others(wrong, len(pool)))
    return per_label, adjacency, accuracy_sum, contributing


def _holdout_scores(
    prepared: list[tuple[str, list[tuple[str, int]], list[tuple[str, int]]]],
    clf: object, X: np.ndarray,
) -> tuple[dict, list[list[str]], float, int, int]:
    """Today's held-out scoring; an empty pool is null, never a fake 0.0."""
    per_label: dict[str, dict] = {}
    adjacency: list[list[str]] = []
    accuracy_sum = 0.0
    contributing = 0
    eval_n = 0
    for name, held, _train in prepared:
        if not held:  # pool too small to split: not measurable
            per_label[name] = {"held_out_n": 0, "eval_n": 0,
                               "accuracy": None, "confusions": []}
            continue
        idxs = [i for _t, i in held]
        preds = [str(p) for p in clf.predict(X[idxs])]
        wrong: dict[str, int] = {}
        hits = 0
        for pred in preds:
            if pred == name:
                hits += 1
            else:
                wrong[pred] = wrong.get(pred, 0) + 1
        accuracy = hits / len(held)
        per_label[name] = {
            "held_out_n": len(held),
            "eval_n": len(held),
            "accuracy": accuracy,
            "confusions": _ranked(wrong)[:MAX_CONFUSIONS],
        }
        accuracy_sum += accuracy
        contributing += 1
        eval_n += len(held)
        adjacency.extend([name, other]
                         for other in _adjacency_others(wrong, len(held)))
    return per_label, adjacency, accuracy_sum, contributing, eval_n


def fewshot_encoder_sha16() -> str:
    """First 16 hex of the shipped encoder's sha256 ("unknown" when absent)."""
    try:
        from .tool_match import EMBED_PRUNER_MODEL

        return integrity.sha256_file(EMBED_PRUNER_MODEL)[:16]
    except Exception:
        return "unknown"


def load_label_head(artifact: bytes, expected_sha: str) -> LabelHead:
    """Verify + load a head artifact (bytes in, never a file)."""
    artifact = bytes(artifact)
    if len(artifact) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"head artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    actual = hashlib.sha256(artifact).hexdigest()
    expected = expected_sha if isinstance(expected_sha, str) else ""
    if not integrity.digests_match(actual, expected.lower()):
        raise ValueError("head_sha256 does not match the artifact bytes")
    try:
        import skops.io as sio

        container = sio.loads(artifact,
                              trusted=sio.get_untrusted_types(data=artifact))
    except Exception as exc:
        raise ValueError(
            f"head artifact failed to load ({type(exc).__name__})") from exc
    if not isinstance(container, dict) or container.get("clf") is None:
        raise ValueError("head artifact is missing its classifier")
    meta = dict(container.get("meta") or {})
    classes = [str(c) for c in container.get("classes") or []]
    if not classes:
        raise ValueError("head artifact is missing its classes")
    return LabelHead(labels=[str(x) for x in meta.get("labels") or []],
                     clf=container["clf"], classes=classes, meta=meta)
