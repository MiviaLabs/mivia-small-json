# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Few-shot schema onboarding: a per-schema learned head from ~8-16
labeled examples (spec ``few-shot-onboarding``).

A user registers a custom schema WITH text-labeled examples
(``register_schema(..., examples=[{"_text": ..., **gold}], fewshot=True)``)
and gets a trained per-schema extractor head in CPU-minutes:

- **enum fields** (str + ``enum``): a message-level
  ``LogisticRegression`` over the SHIPPED sentence encoder's embedding
  (``tool_match.load_embed_pruner`` — the sha-verified zero-shot
  paraphrase-multilingual-MiniLM int8 ONNX export; single-input
  forwards only, the int8 per-tensor requantization makes batching
  non-inert). The head may only emit a DECLARED enum member — the
  author's closed vocabulary, the same trust level as the enum/alias
  coercion. Predicted members still pass through
  ``registry.canonical_enum_value`` (belt and braces).
- **str / int fields**: a token-level tagger, the house
  ``extract_learned`` pattern (``FeatureHasher`` +
  ``LogisticRegression`` over token features). Decode = the longest
  contiguous run of positive tokens at p >= 0.5, taken as the ORIGINAL
  text slice — verbatim by construction and VERIFIED verbatim anyway:
  the honesty gate is explicit, never by-construction. A predicted
  value that does not appear verbatim in the text is refused (the
  field stays unfilled, the normal missing-field path).
- **list fields**: the learned head abstains (deterministic ladder
  only).

Composition order (wired in ``registry.extractor_for``): the
deterministic pattern engine runs FIRST (its enum-alias coercion is
untouched); the head fills only fields the engine left UNFILLED;
neither fires -> the field stays absent. A user-supplied ``extractor``
callable is authoritative and never composed under.

Persistence: skops beside the registration (``MIVIA_FEWSHOT_DIR``,
default ``<MIVIA_ROOT or .>/data/fewshot/`` — a gitignored runtime
dir), sha256 sidecar, verified on ``attach``. Auto-wiring uses the
IN-MEMORY head only; disk artifacts are evidence and restart-resume
(``fewshot.attach``), so a schema re-registered without examples can
never silently resurrect a stale head. ``unregister`` cleans everything.

Determinism: no wall clock, no RNG beyond the declared
``random_state=42``; refitting twice yields bit-identical model
content and predictions. (Artifact BYTES are not stable across dumps —
skops ``schema.json`` carries ``__id__`` memory addresses, demonstrated
2026-10-10 — so the determinism gate is content-level, not bytes.)
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression

from . import integrity

ARTIFACT_VERSION = 1
TAG_THRESHOLD = 0.5
# Frozen in the spec before any measurement (few-shot-onboarding).
ENUM_CLF = dict(C=10.0, max_iter=2000, solver="lbfgs")
TAG_CLF = dict(C=2.0, max_iter=1000, random_state=42)
_MIN_EXAMPLES_PER_FIELD = 2

# House token machinery (extract_learned): the generic \w+ tokenizer and
# the token feature template the component/level extractors train on.
from .extract_learned import (  # noqa: E402  (deliberate reuse, aliased)
    level_token_features as _token_features,
    level_tokenize as _tokenize,
)


def _warn(msg: str) -> None:
    print(f"[mivia-small-json] {msg}", file=sys.stderr)


def fewshot_dir() -> Path:
    """Runtime artifact dir (``MIVIA_FEWSHOT_DIR`` wins; gitignored)."""
    env = os.environ.get("MIVIA_FEWSHOT_DIR")
    if env:
        return Path(env)
    root = os.environ.get("MIVIA_ROOT", ".")
    return Path(root) / "data" / "fewshot"


def encoder():
    """The shipped sentence encoder, sha-verified (None when absent).

    ``tool_match.load_embed_pruner`` is the cached process singleton;
    a missing artifact is a stable property, so its negative caching is
    exactly the degrade-once contract onboarding needs.
    """
    try:
        from . import tool_match

        return tool_match.load_embed_pruner()
    except Exception:  # never raise out of onboarding
        return None


def _encoder_sha() -> str:
    try:
        from . import tool_match

        return integrity.sha256_file(tool_match.EMBED_PRUNER_MODEL)[:16]
    except Exception:
        return "unknown"


# --- the head ---------------------------------------------------------------


class FewShotHead:
    """Per-schema learned field heads: enum members + verbatim spans."""

    def __init__(self, schema: str, spec: dict, meta: dict,
                 enum_heads: dict, taggers: dict):
        self.schema = schema
        self.spec = spec  # frozen copy of the registration spec
        self.meta = meta
        self.enum_heads = enum_heads  # field -> {"clf", "classes"}
        self.taggers = taggers        # field -> {"hasher", "clf"}

    # -- prediction seams (tests override these to probe the honesty gate)

    def predict_member(self, text: str, field: str):
        """Declared enum member for ``text``, or None (no head / no text)."""
        entry = self.enum_heads.get(field)
        if entry is None or not isinstance(text, str) or not text.strip():
            return None
        try:
            vec = np.asarray(_embed_one(text), dtype=np.float32).reshape(1, -1)
            member = str(entry["clf"].predict(vec)[0])
            # Declared-members-only, enforced at the seam: the head can
            # never emit a value the schema author did not declare.
            return member if member in set(entry["classes"]) else None
        except Exception as exc:
            _warn(f"fewshot predict_member failed ({type(exc).__name__}: {exc})")
            return None

    def predict_span(self, text: str, field: str):
        """Longest positive token run as an ORIGINAL text slice, else None.

        Verbatim by construction (a slice of the input) and re-verified
        by the caller's gate — the honesty check is explicit.
        """
        entry = self.taggers.get(field)
        if entry is None or not isinstance(text, str) or not text.strip():
            return None
        tokens = _tokenize(text)
        if not tokens:
            return None
        feats = [_token_features(text, tokens, i) for i in range(len(tokens))]
        X = entry["hasher"].transform(feats)
        probs = entry["clf"].predict_proba(X)
        classes = list(entry["clf"].classes_)
        pos = classes.index(1) if 1 in classes else 1
        flags = probs[:, pos] >= TAG_THRESHOLD
        best: tuple[int, int] | None = None
        run_start = None
        for i, hit in enumerate(flags):
            if hit:
                if run_start is None:
                    run_start = i
                continue
            if run_start is not None:
                if best is None or (i - run_start) > (best[1] - best[0]):
                    best = (run_start, i)
                run_start = None
        if run_start is not None:
            end = len(flags)
            if best is None or (end - run_start) > (best[1] - best[0]):
                best = (run_start, end)
        if best is None:
            return None
        start = tokens[best[0]][1]
        last = tokens[best[1] - 1]
        span = text[start:last[1] + len(last[0])]
        # Edge wrapper noise (the spanfix convention): trim quotes and the
        # punctuation that legitimately terminates a captured span. The
        # trimmed span is still a contiguous substring, hence verbatim.
        span = span.strip(" \"'.,;:!?)]}")
        if span and span in text:
            return span
        return None

    # -- the composition seam -----------------------------------------------

    def fill_missing(self, text: str, obj: dict) -> dict:
        """Fill the fields the deterministic engine left UNFILLED.

        Honesty law: enum heads emit declared members only (re-checked
        through ``canonical_enum_value``); str/int heads emit verbatim
        spans of ``text`` or refuse. Never raises — a head failure
        degrades to the deterministic object, never a crash.
        """
        if not isinstance(text, str) or not text.strip():
            return obj
        from . import registry as _registry

        for field, rule in self.spec.items():
            if field == "type" or field in obj:
                continue
            try:
                if rule.get("type") is str and "enum" in rule:
                    member = self.predict_member(text, field)
                    if member is None:
                        continue
                    canonical = _registry.canonical_enum_value(
                        self.schema, field, member)
                    if canonical is not None:
                        obj[field] = canonical
                elif rule.get("type") is str:
                    span = self.predict_span(text, field)
                    if span and span in text:  # the explicit verbatim gate
                        obj[field] = span
                elif rule.get("type") is int:
                    span = self.predict_span(text, field)
                    if span and span in text:
                        try:
                            obj[field] = int(span)
                        except ValueError:
                            pass  # refuse: not a number, stay absent
                # list-typed fields: the learned head abstains by contract
            except Exception as exc:
                _warn(f"fewshot fill {self.schema}.{field} failed "
                      f"({type(exc).__name__}: {exc})")
        return obj


def _embed_one(text: str) -> np.ndarray:
    enc = encoder()
    if enc is None:
        raise RuntimeError("encoder unavailable")
    return enc.embed_one(text)


# --- training ----------------------------------------------------------------


def _split_pairs(schema: str, pairs: list[dict], spec: dict) -> tuple[dict, dict, int]:
    """Per-field training rows from (text, fields) pairs.

    Returns (enum_rows, span_rows, skipped) where enum_rows/span_rows map
    field -> [(text, gold_value)] and ``skipped`` counts pairs whose gold
    string/int value never appears verbatim in its own text (honest
    exclusion, reported in the artifact meta).
    """
    enum_rows: dict[str, list] = {}
    span_rows: dict[str, list] = {}
    skipped = 0
    for pair in pairs:
        text, fields = pair["text"], pair.get("fields") or {}
        for field, rule in spec.items():
            if field == "type" or field not in fields:
                continue
            value = fields[field]
            if rule.get("type") is str and "enum" in rule:
                if isinstance(value, str) and value in rule["enum"]:
                    enum_rows.setdefault(field, []).append((text, value))
            elif rule.get("type") is str:
                if isinstance(value, str) and value and value in text:
                    span_rows.setdefault(field, []).append((text, value))
                else:
                    skipped += 1  # not verbatim: refused as training signal
            elif rule.get("type") is int:
                if isinstance(value, int) and not isinstance(value, bool) \
                        and str(value) in text:
                    span_rows.setdefault(field, []).append((text, value))
                else:
                    skipped += 1
            # list-typed fields: never learned (abstain by contract)
    return enum_rows, span_rows, skipped


def _fit_enum_head(encoder_obj, rows: list) -> dict | None:
    """Message-level LR over shipped-encoder embeddings (declared members)."""
    members = sorted({value for _, value in rows})
    if len(members) < 2:
        return None  # one member is not a classification task
    X = np.stack([encoder_obj.embed_one(text) for text, _ in rows])
    y = [value for _, value in rows]
    clf = LogisticRegression(**ENUM_CLF)
    clf.fit(X, y)
    return {"clf": clf, "classes": [str(c) for c in clf.classes_]}


def _fit_tagger(rows: list) -> dict | None:
    """Token-level tagger, the house FeatureHasher + LR pattern."""
    X: list[list[str]] = []
    y: list[int] = []
    for text, value in rows:
        needle = str(value)
        idx = text.find(needle)
        if idx < 0:
            continue
        end = idx + len(needle)
        tokens = _tokenize(text)
        for i, (_tok, start) in enumerate(tokens):
            stop = start + len(_tok)
            X.append(_token_features(text, tokens, i))
            y.append(1 if (start < end and stop > idx) else 0)
    if not X or set(y) != {0, 1}:
        return None  # no value tokens or no background: not learnable
    hasher = FeatureHasher(n_features=2 ** 18, input_type="string")
    clf = LogisticRegression(**TAG_CLF)
    clf.fit(hasher.transform(X), y)
    return {"hasher": hasher, "clf": clf}


# --- persistence --------------------------------------------------------------


def _artifact_paths(schema: str) -> tuple[Path, Path]:
    base = fewshot_dir()
    return base / f"{schema}.skops", base / f"{schema}.skops.sha256"


def _persist(schema: str, head: FewShotHead) -> Path:
    import skops.io as sio

    path, sidecar = _artifact_paths(schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = {
        "meta": dict(head.meta),
        "spec": {
            field: {k: (v.__name__ if k == "type" else v)
                    for k, v in rule.items()}
            for field, rule in head.spec.items()
        },
        "enum": {field: dict(entry) for field, entry in head.enum_heads.items()},
        "tag": {field: dict(entry) for field, entry in head.taggers.items()},
    }
    sio.dump(container, path)
    digest = integrity.sha256_file(path)
    sidecar.write_text(digest + "\n", encoding="ascii")
    return path


def _load_verified(schema: str):
    """Sha-verified skops load of one head artifact (None when refused)."""
    path, sidecar = _artifact_paths(schema)
    try:
        if not path.is_file() or not sidecar.is_file():
            return None
        expected, problem = integrity.expected_digest(sidecar)
        if problem:
            _warn(f"fewshot sidecar problem for {schema}: {problem}")
            return None
        if not integrity.digests_match(integrity.sha256_file(path), expected):
            _warn(f"fewshot artifact sha256 mismatch for {schema}; refusing")
            return None
        import skops.io as sio

        return sio.load(path, trusted=sio.get_untrusted_types(file=path))
    except Exception as exc:
        _warn(f"fewshot load failed for {schema} "
              f"({type(exc).__name__}: {exc})")
        return None


def _head_from_container(schema: str, container: dict) -> FewShotHead:
    spec = {}
    for field, rule in (container.get("spec") or {}).items():
        entry = dict(rule)
        if "type" in entry:
            entry["type"] = {"str": str, "int": int, "list": list}[
                entry["type"]]
        spec[field] = entry
    return FewShotHead(schema, spec, dict(container.get("meta") or {}),
                       dict(container.get("enum") or {}),
                       dict(container.get("tag") or {}))


# --- public API ----------------------------------------------------------------

_HEADS: dict[str, FewShotHead] = {}
_HEADS_LOCK = threading.Lock()


def onboard(schema: str, pairs: list[dict] | None = None) -> dict:
    """Train + persist + wire the per-schema head. Returns the report.

    ``pairs`` items are ``{"text": str, "fields": {field: value}}``;
    ``None`` reads the registration's inline pairs (the ``"_text"``
    examples). The report is honest on every path: ``trained`` False
    always carries a ``reason``. Never raises for a registered schema.
    """
    from . import registry as _registry

    report: dict = {"schema": schema, "trained": False,
                    "n_examples": 0, "per_field": {}}
    try:
        spec = {f: dict(r) for f, r in _registry.resolve(schema).items()}
        if pairs is None:
            pairs = [dict(p) for p in _registry.pairs_for(schema)]
        pairs = [dict(p) for p in (pairs or [])]
        report["n_examples"] = len(pairs)
        enc = encoder()
        if enc is None:
            report["reason"] = "encoder unavailable"
            return report
        if not pairs:
            report["reason"] = "no text pairs supplied"
            return report
        bad = [i for i, p in enumerate(pairs)
               if not isinstance(p, dict)
               or not isinstance(p.get("text"), str) or not p["text"].strip()
               or not isinstance(p.get("fields"), dict)]
        if bad:
            report["reason"] = f"malformed pairs at {bad[:5]}"
            return report

        enum_rows, span_rows, skipped = _split_pairs(schema, pairs, spec)
        enum_heads: dict[str, dict] = {}
        taggers: dict[str, dict] = {}
        per_field: dict[str, dict] = {}
        for field, rows in enum_rows.items():
            trained = _fit_enum_head(enc, rows) if len(rows) >= _MIN_EXAMPLES_PER_FIELD else None
            if trained is not None:
                enum_heads[field] = trained
            per_field[field] = {"kind": "enum", "examples": len(rows),
                                "trained": trained is not None}
        for field, rows in span_rows.items():
            trained = _fit_tagger(rows) if len(rows) >= _MIN_EXAMPLES_PER_FIELD else None
            if trained is not None:
                taggers[field] = trained
            per_field[field] = {"kind": "span", "examples": len(rows),
                                "trained": trained is not None}
        abstained = [f for f, r in spec.items()
                     if f != "type" and r.get("type") is list]
        if abstained:
            per_field.update({f: {"kind": "list", "trained": False,
                                  "reason": "list fields abstain by contract"}
                              for f in abstained})
        meta = {"version": ARTIFACT_VERSION, "schema": schema,
                "encoder_sha256_16": _encoder_sha(),
                "n_examples": len(pairs), "skipped_nonverbatim": skipped,
                "per_field": per_field}
        head = FewShotHead(schema, spec, meta, enum_heads, taggers)
        path = _persist(schema, head)
        with _HEADS_LOCK:
            _HEADS[schema] = head
        report.update({"trained": True, "per_field": per_field,
                       "artifact": path.name,
                       "skipped_nonverbatim": skipped})
        return report
    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        return report


def head_for(schema: str) -> FewShotHead | None:
    """The wired in-memory head, or None (auto-wiring consults ONLY this)."""
    return _HEADS.get(schema)


def attach(schema: str) -> bool:
    """Restart-resume: load the persisted head (sha-verified) into the cache."""
    if schema in _HEADS:
        return True
    container = _load_verified(schema)
    if container is None:
        return False
    with _HEADS_LOCK:
        _HEADS[schema] = _head_from_container(schema, container)
    return True


def forget(schema: str) -> None:
    """Drop the cached head and delete the persisted artifacts. Never raises."""
    with _HEADS_LOCK:
        _HEADS.pop(schema, None)
    try:
        path, sidecar = _artifact_paths(schema)
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
    except Exception as exc:
        _warn(f"fewshot cleanup failed for {schema} "
              f"({type(exc).__name__}: {exc})")


def status(schema: str) -> dict | None:
    """Onboarding status for one schema (None = no wired head)."""
    head = _HEADS.get(schema)
    if head is None:
        return None
    return {"schema": schema, "trained": True, "meta": dict(head.meta)}


def extract_via_schema(text: str, schema: str) -> tuple[dict, dict]:
    """The wired extraction for ``schema`` (deterministic first, head fills).

    The exact surface classify/structure reach through
    ``rules.extract``; exposed here so instruments and tests measure the
    same path production takes.
    """
    from . import rules

    return rules.extract(text, schema), {}
