# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Logistic schema router: hashed char+word TF-IDF -> LogisticRegression.

Used ONLY for routing where keyword routing is unreliable (AGENTS.md
rule 1: rules first). Trained on the train split, picked on dev,
persisted with skops — never pickle (AGENTS.md rule 4) — to
models/router.skops with a sha256 sidecar for deterministic reload.

HashingVectorizer (murmurhash, fixed seed) instead of a fitted
TfidfVectorizer vocabulary: skops stores a fitted vocab as one JSON node
per term (~13 MB); the hashing trick needs no vocab at all, so the
artifact stays in the sub-1 MB cost target and reload is deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from . import registry

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def artifact_path(root: Path | str | None = None) -> Path:
    """The router artifact inside ``root`` (default: this repository).

    The ONE answer to "where is the artifact". Every construction of
    the path goes through here; the repository constant ``MODEL_PATH``
    is defined through this helper too.
    """
    return (Path(root) if root is not None else ROOT) / "models" / "router.skops"


def sidecar_path(root: Path | str | None = None) -> Path:
    """The sha256 sidecar beside the artifact inside ``root``.

    Same convention the loader enforces: ``<artifact-stem>.sha256``
    next to the artifact (models/router.skops <-> models/router.sha256).
    """
    artifact = artifact_path(root)
    return artifact.parent / (artifact.stem + ".sha256")


def trainmeta_path(root: Path | str | None = None) -> Path:
    """The lineage record beside the artifact inside ``root``."""
    artifact = artifact_path(root)
    return artifact.parent / (artifact.stem + ".trainmeta.json")


# Iteration 3: real-data training sets (data/real). Rows carry
# {source, raw, gold}; the gold routing label is the dataset name.
# Single source of truth for BOTH the loader and the lineage manifest
# (TRAIN_SOURCES): a set added here is trained on AND recorded.
REAL_SETS = (
    ("issue", "train.jsonl"),
    ("issue", "train-prneg.jsonl"),
    ("commit", "train.jsonl"),
    ("log_event", "train.jsonl"),
    ("pull_request", "train.jsonl"),
    ("tool_call", "train.jsonl"),
    ("no_tool_call", "train.jsonl"),
    ("diagnostic", "train.jsonl"),
    ("diff", "train.jsonl"),
    ("review_comment", "train.jsonl"),
    ("agent_step", "train.jsonl"),
)

# Every corpus that feeds the router training mix.
TRAIN_SOURCES = ("data/synthetic/train.jsonl",) + tuple(
    f"data/real/{schema}/{split}" for schema, split in REAL_SETS
) + ("data/pool/confident.jsonl",)

MODEL_PATH = artifact_path()


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True,
                              timeout=30
                              ).stdout.strip()
    except Exception:
        return "unknown"


def _environment() -> dict:
    import platform

    import numpy
    import scipy
    import sklearn
    import skops

    return {"python": platform.python_version(),
            "scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__, "skops": skops.__version__}


def _data_manifest(root: Path) -> list[dict]:
    """sha256 + row count for every training corpus that exists."""
    out = []
    sources = list(TRAIN_SOURCES)
    gh = root / "data" / "incoming" / "gh-user" / "curated"
    if gh.is_dir():
        sources += [f"data/incoming/gh-user/curated/{p.name}"
                    for p in sorted(gh.glob("*.jsonl"))]
    for rel in sources:
        path = root / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        rows = sum(1 for line in data.decode("utf-8", "replace").splitlines()
                   if line.strip())
        out.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(),
                    "rows": rows})
    return out


def build() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            HashingVectorizer(
                                ngram_range=(1, 2),
                                n_features=2**11,
                                alternate_sign=False,
                                norm=None,
                            ),
                        ),
                        (
                            "char",
                            HashingVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 4),
                                n_features=2**12,
                                alternate_sign=False,
                                norm=None,
                            ),
                        ),
                    ]
                ),
            ),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=SEED)),
        ]
    )


def _load_jsonl(path: Path):
    rows = [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [r["raw"] for r in rows], [r["schema"] for r in rows]


def _guard_frozen_path(raw: str, root: Path) -> Path:
    """Refuse an env-supplied corpus path that resolves into the frozen
    tree (``data/wild`` or ``data/heldout``), matching training_mix's
    documented guarantee that it never touches them."""
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    for frozen in ("data/wild", "data/heldout"):
        boundary = (root / frozen).resolve()
        try:
            resolved.relative_to(boundary)
        except ValueError:
            continue
        raise ValueError(
            f"refusing frozen corpus path for training: {path} resolves "
            f"under {frozen} — frozen audit data never enters the "
            "training mix")
    return path


REAL_DEV_SETS = (
    ("issue", "dev.jsonl"),
    ("commit", "dev.jsonl"),
    ("log_event", "dev.jsonl"),
    ("pull_request", "dev.jsonl"),
    ("tool_call", "dev.jsonl"),
    ("no_tool_call", "dev.jsonl"),
    ("diagnostic", "dev.jsonl"),
    ("diff", "dev.jsonl"),
    ("review_comment", "dev.jsonl"),
    ("agent_step", "dev.jsonl"),
)


def _load_real(pairs, root: Path | None = None) -> tuple[list, list]:
    root = Path(root) if root is not None else ROOT
    raws, labels = [], []
    for schema, split in pairs:
        path = root / "data" / "real" / schema / split
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                raws.append(json.loads(line)["raw"])
                labels.append(schema)
    return raws, labels


def _load_gh(root: Path) -> tuple[list, list]:
    """Curated owner-repo rows, if present. Gitignored, opt-in only.

    ``MIVIA_GH_DIR`` overrides the curated directory, so curation variants
    (e.g. a repo-quality filter) can be measured without moving files.
    """
    import os

    if os.environ.get("MIVIA_GH_DIR"):
        # Same frozen-path guard as the sibling env corpus vars: the env
        # override must never point at data/wild or data/heldout
        # (review P2, 2026-10-10 — _load_gh was the one ungated path).
        base = _guard_frozen_path(os.environ["MIVIA_GH_DIR"], root)
    else:
        base = root / "data" / "incoming" / "gh-user" / "curated"
    raws, labels = [], []
    if not base.is_dir():
        return raws, labels
    # dict.fromkeys dedupes while keeping order: REAL_SETS carries issue
    # twice (train.jsonl + train-prneg.jsonl), and iterating it raw read
    # the curated issue file twice, duplicating every row.
    for schema in dict.fromkeys(schema for schema, _ in REAL_SETS):
        path = base / f"{schema}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                raws.append(json.loads(line)["raw"])
                labels.append(schema)
    return raws, labels


def _augment(X: list, y: list, rng, rate: float = 0.3) -> tuple[list, list]:
    """Truncated variants so routing survives short or partial text.

    The adversarial benchmark showed a ~10pp drop on inputs cut to a third.
    Adding half- and third-length copies of a 30% sample of rows lifts the
    truncate score from 90.6% to 99.4% with no loss on the clean set (a
    higher rate traded 0.3pp of clean accuracy for the last 0.3pp).
    """
    out_x: list = []
    out_y: list = []
    for text, label in zip(X, y):
        if len(text) > 30 and rng.random() < rate:
            out_x.append(text[: max(10, len(text) // 3)])
            out_y.append(label)
            out_x.append(text[: max(10, len(text) // 2)])
            out_y.append(label)
    return out_x, out_y


def training_mix(
    root: Path | None = None,
    seed: int = SEED,
    pool_path: Path | None = None,
    pool_cap: int = 600,
    real_cap: int = 200_000,
    include_gh: bool = False,
    augment: bool = False,
) -> tuple[list, list]:
    """Iteration 3 training mix (deterministic, seed 42) + pool rows.

    All of synthetic corpus v3 train (2800 rows, 4 classes incl.
    pull_request) + all of the real issue/commit/log_event train pools
    (curated by scripts/extend_real_data.py from data/incoming/ dumps)
    + confident weak-labeled rows from the improvement pool
    (data/pool/confident.jsonl, capped at pool_cap, appended last).
    Real pools are capped at real_cap (owner decision 2026-10-05: train
    on all available real data — training cost is fine, and inference
    cost is fixed by the hashed feature space, not the row count), and
    the combined list is shuffled with the same seeded RNG. Never
    touches data/wild or data/heldout.
    """
    import random

    root = Path(root) if root is not None else ROOT
    X_s, y_s = _load_jsonl(root / "data" / "synthetic" / "train.jsonl")
    X_r, y_r = _load_real(REAL_SETS, root=root)
    pool = Path(pool_path) if pool_path else root / "data" / "pool" / "confident.jsonl"
    rng = random.Random(seed)
    X_p: list = []
    y_p: list = []
    if pool.exists():
        # _load_jsonl returns (raws, labels) — unpack FIRST, cap both lists
        # after. Indexing the tuple directly pairs raw[i] with label[i+1]
        # and trains a raw commit message as its own class label.
        pool_raws, pool_labels = _load_jsonl(pool)
        if len(pool_raws) > pool_cap:
            # Seed-sample, never truncate: growth past the cap must still
            # be represented instead of silently dropped.
            idx = sorted(rng.sample(range(len(pool_raws)), pool_cap))
            X_p = [pool_raws[i] for i in idx]
            y_p = [pool_labels[i] for i in idx]
        else:
            X_p, y_p = pool_raws, pool_labels
    if len(X_r) > real_cap:
        idx = sorted(rng.sample(range(len(X_r)), real_cap))
        X_r = [X_r[i] for i in idx]
        y_r = [y_r[i] for i in idx]
    X, y = X_s + X_r + X_p, y_s + y_r + y_p
    if include_gh:
        X_g, y_g = _load_gh(root)
        X, y = X + X_g, y + y_g
    # Phase 2 (ood-routing-band; owner product decision
    # 2026-10-10: the class is plain ``text`` — no rejection wording):
    # the auxiliary non-workflow training rows (label: text), gated
    # so every existing consumer (skops fallback training, gates, fixtures)
    # keeps the 10-class mix unless the phase-2 arm explicitly opts in.
    # Extra in-domain rows (multilingual owner decision 2026-10-10): real
    # non-EN tickets as their canonical class, {text, category} shape.
    extra_indomain = os.environ.get("MIVIA_EXTRA_INDOMAIN", "").strip()
    if extra_indomain:
        extra_path = _guard_frozen_path(extra_indomain, root)
        ex_x: list = []
        ex_y: list = []
        for line in extra_path.read_text(encoding="utf-8",
                                         errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            text = r.get("text")
            cat = r.get("category")
            if (isinstance(text, str) and text.strip()
                    and isinstance(cat, str) and cat):
                ex_x.append(text)
                ex_y.append(cat)
        X, y = X + ex_x, y + ex_y
    ood_aux = os.environ.get("MIVIA_OOD_AUX", "").strip()
    if ood_aux:
        aux_path = _guard_frozen_path(ood_aux, root)
        # OOD rows use the {text, category} shape (build script output)
        aux_rows = []
        for line in aux_path.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            text = r.get("text")
            if isinstance(text, str) and text.strip():
                aux_rows.append(text)
        X, y = X + aux_rows, y + ["text"] * len(aux_rows)
    if augment:
        ax, ay = _augment(X, y, rng)
        X, y = X + ax, y + ay
    order = list(range(len(X)))
    rng.shuffle(order)
    return [X[i] for i in order], [y[i] for i in order]


def main(root: Path | None = None, out_path: Path | None = None) -> dict:
    """Train and write the artifact under ``root`` (default: this repo).

    A cycle run against a scratch root must read its training data from
    that root and write the artifact and sidecar back into it — never
    into this repository's own ``models/``. The caller's root wins for
    every read and every write.

    ``out_path`` writes the artifact (and its sidecar and lineage record)
    to that path instead of the root's default, so a cycle can stage a
    candidate before promoting it.
    """
    from skops.io import dump

    import datetime as dt

    from . import bootstrap
    from .agent_step import register_agent_step_schema
    from .diagnostic import register_diagnostic_schema
    from .diff_comment import register_diff_and_review_schemas
    from .tool_call import register_no_tool_call_schema, register_tool_call_schema

    prev_active = set(registry.active())
    # The eval world boots pull_request through
    # bootstrap.register_all_schemas / register_example_schemas; the
    # recorded fingerprint must cover the class set the model is
    # actually trained AND evaluated on, so register the example world
    # here too — before spec_fingerprint() is read below.
    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()
    register_agent_step_schema()

    try:
        root = Path(root) if root is not None else ROOT
        import os

        X_tr, y_tr = training_mix(
            root=root,
            include_gh=os.environ.get("MIVIA_GH_DATA") == "1",
            augment=os.environ.get("MIVIA_AUGMENT") == "1")
        X_dv, y_dv = _load_jsonl(root / "data" / "synthetic" / "dev.jsonl")
        pipe = build().fit(X_tr, y_tr)
        pred = pipe.predict(X_dv)
        acc = float(accuracy_score(y_dv, pred))
        f1m = float(f1_score(y_dv, pred, average="macro"))
        X_rd, y_rd = _load_real(REAL_DEV_SETS, root=root)
        acc_r = float(accuracy_score(y_rd, pipe.predict(X_rd)))
        f1_r = float(f1_score(y_rd, pipe.predict(X_rd), average="macro"))
        # Size recipe (iteration 2): store coefficients as float32 before the
        # dump — predictions are unchanged on this scale, the artifact shrinks
        # ~3x. build() itself stays float64 (tests may fit it directly).
        pipe.named_steps["clf"].coef_ = pipe.named_steps["clf"].coef_.astype("float32")
        model_path = Path(out_path) if out_path is not None else artifact_path(root)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(pipe, model_path)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        (model_path.parent / (model_path.stem + ".sha256")).write_text(
            digest + "\n", encoding="utf-8")
        size = model_path.stat().st_size
        meta = {
            "created": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "artifact_sha256": digest,
            "seed": SEED,
            "n_train_rows": len(X_tr),
            "classes": sorted(set(y_tr)),
            "spec_fingerprint": registry.spec_fingerprint(),
            "data": _data_manifest(root),
            "environment": _environment(),
            "metrics": {"dev_accuracy": acc, "dev_macro_f1": f1m,
                        "real_dev_accuracy": acc_r, "real_dev_macro_f1": f1_r},
        }
        (model_path.parent / (model_path.stem + ".trainmeta.json")).write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"synthetic dev accuracy={acc:.4f} macro_f1={f1m:.4f}")
        print(f"real dev accuracy={acc_r:.4f} macro_f1={f1_r:.4f}")
        print(f"saved {model_path.name}: {size} bytes sha256={digest[:12]}...")
        return {
            "dev_accuracy": acc,
            "dev_macro_f1": f1m,
            "real_dev_accuracy": acc_r,
            "real_dev_macro_f1": f1_r,
            "size_bytes": size,
        }
    finally:
        for s in ("pull_request", "tool_call", "no_tool_call", "diagnostic",
                  "diff", "review_comment", "agent_step"):
            if s not in prev_active and s in registry.active():
                registry.unregister(s)


if __name__ == "__main__":
    main()
