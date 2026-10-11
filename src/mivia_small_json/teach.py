# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Teach a schema from examples: (text, json) pairs -> a registered schema.

The north star is "dump any text, define an output schema, get filled
JSON". A user supplies a spec and a few examples; this module aligns each
field's value inside its text, synthesises a conservative per-field
pattern from the aligned spans, derives routing markers from tokens the
examples share, and registers the schema through the existing
``registry.register_schema``. Everything downstream (validate, repair,
template, classify) then works unchanged.

It reports what it learned and what it could not: a field whose value
never appears verbatim in the text is named as uncovered, never guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import registry

_SHAPE_RUNS = re.compile(r"[0-9]+|[a-z]+|[A-Z]+|[^A-Za-z0-9]+")
# A "word" for routing: any script's letter, then two or more word chars.
# Unicode-aware so non-English text is not silently dropped; no hardcoded
# stopword list, the marker choice is driven by the examples themselves.
_TOKEN = re.compile(r"[^\W\d_]\w{2,}", re.UNICODE)
_CONTEXT_TOKEN = re.compile(r"[A-Za-z0-9_.:-]+")
_SEED = 42
# A field with no verbatim value needs this many examples before a learned
# classifier is worth trusting; below it, the field stays uncovered.
MIN_CLASSIFIER_EXAMPLES = 20


def _train_field_classifier(texts: list[str], values: list[str]):
    """A small CPU classifier that predicts a field's value from the text."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("features", HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 14,
                                       alternate_sign=False, norm=None)),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=_SEED)),
    ])
    return pipe.fit(texts, values)


def _shape_pattern(value: str) -> str:
    """Generalise a literal value into a regex with exactly one capture group.

    Digit runs become ``\\d+``, lowercase runs ``[a-z]+``, uppercase runs
    ``[A-Z]+``; punctuation stays literal. Word boundaries are added when
    the value starts or ends alphanumeric, so the pattern does not match a
    substring of a longer token.
    """
    body = "".join(
        r"\d+" if run.isdigit()
        else "[a-z]+" if run.islower()
        else "[A-Z]+" if run.isupper()
        else re.escape(run)
        for run in _SHAPE_RUNS.findall(value)
    )
    if value[:1].isalnum():
        body = r"\b" + body
    if value[-1:].isalnum():
        body = body + r"\b"
    return "(" + body + ")"


def _context_candidates(text: str, value: str, shape: str):
    """Patterns anchored on the value's neighbours, longest context first."""
    idx = text.find(value)
    if idx < 0:
        return
    before = _CONTEXT_TOKEN.findall(text[:idx])
    after = _CONTEXT_TOKEN.findall(text[idx + len(value):])
    for total in (3, 2, 1):
        for pn in range(total, -1, -1):
            sn = total - pn
            prefix = " ".join(before[-pn:]) if pn else ""
            suffix = " ".join(after[:sn]) if sn else ""
            yield (
                (re.escape(prefix) + r"\s+" if prefix else "")
                + shape
                + (r"\s+" + re.escape(suffix) if suffix else "")
            )


def _field_pattern(pairs: list[tuple[str, str]]) -> str:
    """The first synthesised pattern that reproduces every aligned value."""
    text0, val0 = pairs[0]
    shape = _shape_pattern(val0)
    for pattern in _context_candidates(text0, val0, shape):
        try:
            rx = re.compile(pattern)
        except re.error:
            continue
        if all((m := rx.search(text)) and m.group(1) == val for text, val in pairs):
            return pattern
    # Fallback: an exact alternation of the observed values, longest first
    # so "ab" wins over its prefix "a", with word boundaries when a value
    # starts/ends alphanumeric (overfits, but only ever matches taught
    # values in full).
    def _alt(value: str) -> str:
        body = re.escape(value)
        if value[:1].isalnum():
            body = r"\b" + body
        if value[-1:].isalnum():
            body = body + r"\b"
        return body

    values = sorted({v for _, v in pairs}, key=len, reverse=True)
    return "(" + "|".join(_alt(v) for v in values) + ")"


def _synthesize_routing(name: str, texts: list[str]) -> dict:
    """Routing markers derived from the examples, data-driven.

    Strong markers are tokens every example shares; when the examples share
    no token, the tokens appearing in at least half of them are used. When
    even that is empty, the schema name is the last resort. No hardcoded
    word list: the markers come from the examples, in any language.
    """
    token_sets = [set(_TOKEN.findall(text.lower())) for text in texts]
    if not token_sets:
        return {"strong": [rf"(?i)\b{re.escape(name)}\b"], "weak": []}
    shared = set.intersection(*token_sets)
    markers = sorted(shared)[:5]
    if not markers:
        counts: dict[str, int] = {}
        for tokens in token_sets:
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
        threshold = max(2, (len(token_sets) + 1) // 2)
        markers = sorted(t for t, c in counts.items() if c >= threshold)[:5]
    strong = [rf"(?i)\b{re.escape(tok)}\b" for tok in markers]
    return {"strong": strong or [rf"(?i)\b{re.escape(name)}\b"], "weak": []}


def _gold(example: dict) -> dict:
    return example.get("json") or {}


@dataclass
class TeachReport:
    name: str
    learned: dict[str, str] = field(default_factory=dict)
    classified: dict[str, str] = field(default_factory=dict)
    uncovered: list[str] = field(default_factory=list)
    routing: dict = field(default_factory=dict)


def teach_schema(name: str, spec: dict, examples: list[dict],
                 routing: dict | None = None,
                 classifiers: bool = True,
                 defaults: dict | None = None) -> TeachReport:
    """Register ``name`` from examples; return what was learned.

    ``examples`` is a list of ``{"text": str, "json": dict}``. Fields whose
    value appears verbatim in the text get a synthesised pattern. A field
    with no verbatim value gets a small learned classifier when it has at
    least ``MIN_CLASSIFIER_EXAMPLES`` examples and two or more distinct
    values; otherwise it is reported uncovered and falls to ``defaults``
    through the normal validate/repair ladder. Raises ``ValueError`` when
    nothing is learned.
    """
    if not examples:
        raise ValueError("teach_schema needs at least one example")
    fields = [f for f in spec if f != "type"]
    pairs: dict[str, list[tuple[str, str]]] = {f: [] for f in fields}
    texts: list[str] = []
    for example in examples:
        text = example.get("text") or ""
        gold = _gold(example)
        texts.append(text)
        for f in fields:
            value = gold.get(f)
            if value is None:
                continue
            sv = str(value)
            if re.search(re.escape(sv), text):
                pairs[f].append((text, sv))

    extraction: dict[str, list[str]] = {}
    learned: dict[str, str] = {}
    classified: dict[str, str] = {}
    uncovered: list[str] = []
    _classifiers: dict[str, object] = {}
    for f in fields:
        if pairs[f]:
            pattern = _field_pattern(pairs[f])
            extraction[f] = [pattern]
            learned[f] = pattern
            continue
        observed = [(ex.get("text") or "", str(_gold(ex).get(f)))
                    for ex in examples if _gold(ex).get(f) is not None]
        if (classifiers and len(observed) >= MIN_CLASSIFIER_EXAMPLES
                and len({v for _, v in observed}) >= 2):
            clf = _train_field_classifier([t for t, _ in observed],
                                          [v for _, v in observed])
            classified[f] = (f"logreg({len({v for _, v in observed})} classes, "
                             f"{len(observed)} rows)")
            _classifiers[f] = clf
        else:
            uncovered.append(f)

    if not extraction and not _classifiers:
        raise ValueError(
            "teach_schema could not learn any field: no example value "
            "appears verbatim in its text and no field had enough examples "
            "for a classifier")

    def _extract(text: str) -> dict:
        obj: dict = {"type": name}
        for f in fields:
            if f in extraction:
                for pattern in extraction[f]:
                    m = re.search(pattern, text)
                    if m:
                        obj[f] = m.group(1)
                        break
            elif f in _classifiers:
                obj[f] = str(_classifiers[f].predict([text])[0])
        return obj

    routing = routing or _synthesize_routing(name, texts)
    registry.register_schema(name, spec, routing, extraction, defaults,
                             examples=[ex.get("json") for ex in examples],
                             extractor=_extract)
    return TeachReport(name=name, learned=learned, classified=classified,
                       uncovered=uncovered, routing=routing)
