# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Cross-encoder tool-match guard: the no-tool abstention decision.

The router sends a tool-catalogue text to ``tool_call``; this guard asks
whether the request actually matches one of the listed tools. Below the
threshold the honest answer is ``no_tool_call``.

A cross-encoder attends across the request and the tool together, which is
the relation a bag-of-ngrams router cannot express. Measured end to end on
eval sets disjoint from its training corpus at the *request* level
(scripts/bench_tool_abstention.py): 97.3% of real calls stay `tool_call`, and
69.2% of no-tool catalogues abstain. Without the guard both were 100%, so the
guard helps, and abstention is still the open problem.

Those abstention rows are near-misses, not unrelated text: the tool is
topically adjacent but inapplicable (`find_roots` for a quadratic equation
against a question about a linear one). Data and representation changes move
the operating point along a frontier (abstention and retention trade roughly
one for one) but do not separate the classes, so the next lever is a trained
reject option or a capability head, not a bigger corpus. See the training
report.

The scorer must pass the tokenizer's segment ids to the model. A cross-encoder
needs them to tell the request span from the tool span; zeroing them costs
about 55 points of false positives.

The load is sha256-verified and never raises; a missing guard simply leaves
the router's decision untouched.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import integrity
from .severity_head import (
    CANONICAL_SEVERITY_ANCHORS,
    coral_probability,
    cumulative_tier,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "tool-match-int8.onnx"
# The bert-base-uncased vocab family default for the cross-encoder heads.
# Historically this was the router's tokenizer (byte-identical bert-base);
# since the router moved to the multilingual XLM-R vocab (2026-10-10) the
# router file is a DIFFERENT vocab family and must never feed these heads
# (token ids out of embedding bounds). tool-match-tokenizer.json is the
# committed bert-base copy these artifacts were trained with.
DEFAULT_TOKENIZER = ROOT / "models" / "tool-match-tokenizer.json"
# Phase-0 decoupling (modernbert-rebuild plan): the severity heads and the
# guard carry their OWN tokenizer copies so a future backbone swap that
# replaces the router tokenizer cannot silently feed the old heads a wrong
# vocab (byte-identical today; the files, not the sharing, are the contract).
SEVERITY_TOKENIZER = ROOT / "models" / "severity-tokenizer.json"
# Best (request, tool) score below this abstains to `no_tool_call`.
#
# Measured end to end on the shipped int8 artifact, all rows
# (scripts/bench_tool_abstention.py):
#
# | threshold | no-tool false positives % | real calls kept % |
# |---|---|---|
# | 0.4 | 33.4 | 97.3 |
# | 0.5 | 32.0 | 97.3 |
# | 0.6 | 30.8 | 97.3 |
# | 0.7 | 28.8 | 97.3 |
#
# The curve is still flat, so the threshold is not the lever: the guard is
# confident on both classes. 0.6 sits mid-curve. Data and representation
# changes move the point along a frontier rather than off it; the lever is a
# trained reject option, not a bigger corpus.
THRESHOLD = 0.6
# The served truncation budget. Must equal the context the shipped artifact
# was fine-tuned at (--max-len 256; tool-abstention): a
# drift here truncates long pairs differently at serve time than the model
# was trained to read.
MAX_LEN = 256
# Description head scanned for a "<key>: ..." colon separator in select():
# a candidate description that already carries its key (colon this close to
# the head) is kept as-is instead of having the key prepended.
_DESC_COLON_SCAN = 30
_CATALOGUE = re.compile(r"\[.*\]", re.S)
_SPLIT = "\n\nUser request:"
# Ranking primitives (`select`, `grade`) read the raw prediction probability,
# not the SelectiveNet-gated score. Measured on the Tier 1 artifact: outside
# the BFCL function-calling domain the selection head rejects nearly every
# pair (selection 0.02-0.23), which collapses all candidates to 0.0 and turns
# argmax into tie-breaking, while the raw probabilities still rank correctly
# (12/12 prompt-routing and 99/100 candidate selection ungated vs 9/12 and
# 93/100 gated). The gate is the acceptance mechanism (`verify`, the guard);
# it is noise for relative ranking.
#
# A `select` win is only meaningful when the candidates are not statistically
# tied. Measured full-precision margins: in-domain wins are >= 0.9; wins where
# every raw score collapses run from 1e-6 to 1e-4. When top1 - top2 <=
# SELECT_MARGIN_FLOOR and top1 <= SELECT_CONF_FLOOR the winner rides on a
# collapsed score and `select(..., with_status=True)` reports "ambiguous".
SELECT_MARGIN_FLOOR = 0.05
SELECT_CONF_FLOOR = 0.30
# Conformal abstain-tau band boundary (candidate-count shape key, 2026-10-11
# fix of the shipped abstain-all-at-large-K bug): the shipped tau 0.7030 was
# fitted on the K=4 selection population (BFCL-dev + prose rows, gold + 3
# distractors). Measured top1-softmax support halves with each doubling of K
# (Banking77 train, .scratch/zs-diag/kband_measure.py): K=2 tops at 0.73,
# K=4 at 0.48, K=8 at 0.28 — so from K=8 up the small-band tau already
# abstained on 0% -> every row (600/600 on the diagnostic's Banking77 test
# sample). K <= 4 keeps the shipped tau (the K=2 binary referee and the K=4
# fit population are unchanged); K >= 5 is served from the many-candidate
# band registered in models/select-calibration-meta.json (tau_by_k fit
# points, each a conformal refit on its own measured K population).
SELECT_SMALL_BAND_K_MAX = 4
# `grade(..., with_set=True)` decodes to an adaptive tier set over the
# per-tier scores, filled to be contiguous — an ordinal set {INFO, ERROR} is
# meaningless. GRADE_SET_QUANTILE is a MEASURED operating point, not a
# conformal coverage guarantee: the per-tier sigmoid scores saturate near 0/1,
# which breaks the exchangeability argument a conformal quantile would need.
# The sweep (scripts/calibrate_grade_sets.py, shipped v4 artifact) shows the
# frontier: at 0.30 in-domain rows decode as singletons (LogHub dev 100%
# coverage, mean size 1.00 — confident rows act without annotation noise)
# while ambiguous novel-register rows pair up (gold-in-set 55% at mean size
# 1.55, vs 35% for the bare argmax). Higher quantiles buy shifted-domain
# coverage at the cost of pairing confident rows (0.50: novel 75% but
# LogHub size 2.00) or spanning 3+ tiers (0.60+: an escalate-to-human
# signal, not a set). Callers can pass a different quantile; the sweep
# table is in reports/training-report.md.
GRADE_SET_QUANTILE = 0.30
# Candidate fallback roots that are gitignored: artifacts there are local
# builds, so a result loaded from one may not reproduce from a clean clone.
_GITIGNORED_MODELS = ROOT / "data" / "incoming" / "models"


def _warn(msg: str) -> None:
    print(f"[mivia-small-json] {msg}", file=sys.stderr)


# The measured enrichment separator convention (.scratch/zs-diag/
# run_enriched600.py — Banking77 zero-shot 48.0% bare -> 63.8% with
# per-label examples, 92.5% precision @ 20% coverage): examples join as
# double-quoted, space-separated utterances appended to the description
# after ". Examples: ". The standard "key: " colon law then applies on
# top (both the diagnostic's dict shape and this one prepend it).
_ENRICHED_SEP = ". Examples: "


def render_select_candidate(name, description=None, examples=None) -> str:
    """The description span one dict-form select candidate renders to.

    ``"<name>: <description>. Examples: \"<e1>\" \"<e2>\" ..."``
    (description optional). The key-prefix law runs on the description
    BEFORE the examples are appended — the "Examples:" colon sits well
    inside the _DESC_COLON_SCAN window for a short description and must
    never be mistaken for the description already carrying its key (the
    diagnostic's bench-shape descriptions were long enough to dodge this;
    a first-class form has to get it right for every caller). The result
    is idempotent under select()'s own colon law.
    """
    desc = str(description).strip() if description is not None else ""
    if examples:
        if isinstance(examples, str):
            examples = [examples]
        quoted = " ".join(f'"{str(e)}"' for e in examples)
        if not desc:
            return f"{name}: Examples: {quoted}"
    else:
        quoted = ""
    if not (desc.startswith(f"{name}:") or (":" in desc[:_DESC_COLON_SCAN])):
        desc = f"{name}: {desc}"
    if quoted:
        desc = f"{desc}{_ENRICHED_SEP}{quoted}"
    return desc


def _select_item_from_spec(spec: dict) -> tuple[str, str]:
    """(key, rendered description) from one dict-form candidate spec.

    The library path is permissive (str-casts, missing fields degrade);
    the serving endpoint enforces the strict shape/caps law on top. A
    spec without a usable name is a caller programming error even here:
    an unnamed candidate cannot be picked back.
    """
    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "dict candidates must carry a non-blank 'name' string")
    return name, render_select_candidate(
        name, spec.get("description"), spec.get("examples"))


def render_tool(tool: dict, parameters: bool = False) -> str:
    """The string the cross-encoder reads for one tool.

    One source of truth: the corpus builder imports this, so the training
    span and the runtime span cannot drift.

    ``parameters=True`` appends the parameter schema. In near-miss catalogues
    the capability constraint often lives only there: a generic
    ``requests.get`` whose ``url`` parameter is pinned to the Date Nager
    holiday API cannot serve a request about stock prices, and the tool's own
    description does not say so. Required parameters come first, because the
    tokenizer truncates the tail.

    The shipped guard is trained on the name-and-description span, so the
    default is ``False``. The parameter-aware span was measured and did raise
    abstention, but it cost retention, so no model trained with it ships; see
    the training report.
    """
    if isinstance(tool.get("function"), dict):
        tool = tool["function"]
    text = f"{tool.get('name', '')}: {tool.get('description', '')}".strip()
    if not parameters:
        return text
    params = tool.get("parameters")
    params = params if isinstance(params, dict) else {}
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = set(required) if isinstance(required, list) else set()
    ordered = sorted(props.items(), key=lambda item: item[0] not in required)
    bits = []
    for name, spec in ordered:
        spec = spec if isinstance(spec, dict) else {}
        flag = "required" if name in required else "optional"
        desc = str(spec.get("description", "")).strip()
        bits.append(f"{name} ({spec.get('type', 'any')}, {flag}): {desc}")
    return f"{text} | parameters: " + "; ".join(bits) if bits else text


def render_tool_fillability(tool: dict, request: str, catalogue=None) -> str:
    """The gate span: the shipped span plus what the request actually supplies.

    BFCL builds a near-miss by stripping the arguments a tool needs
    (``set cool mode`` from ``set cool mode with a temp of 24 oC``) or by
    swapping in a sibling tool. The shipped span is ``name: description``,
    which shows neither, so the guard is asked to refuse on evidence it cannot
    see. This renders the required parameters, the values the request supplies,
    and what is missing, so a missing required argument becomes visible text.

    Measured on the 884 near-misses / 183 calls: abstention 80.0% -> 83.5% at
    matched 95.1% retention, exact McNemar p=0.0001. It costs `select`, which
    reads plain strings, so the gate and the ranking primitives run on
    different artifacts (``SplitToolMatcher``).
    """
    from . import span_fill

    base = render_tool(tool)
    bare = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    params = bare.get("parameters") if isinstance(bare, dict) else None
    params = params if isinstance(params, dict) else {}
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = list(required) if isinstance(required, list) else []
    # Same str-filter as the decision-layer veto: a malformed catalogue
    # ("required": [3]) renders the plain span — never crashes the gate
    # span and never leaks non-string slots into the model's text.
    required = [p for p in required if isinstance(p, str)]
    if not required:
        return base

    def _type(name: str) -> str:
        spec = props.get(name)
        return str(spec.get("type", "any")) if isinstance(spec, dict) else "any"

    parts = [base, "requires: "
             + ", ".join(f"{p} ({_type(p)})" for p in required)]
    filled, missing_fill = span_fill.fill_arguments(tool, request)
    _ok, _reason, missing = validate_fillability(tool, request)
    if filled is not None:
        values = ", ".join(f"{k}={v}" for k, v in filled.items()) or "(none)"
        parts.append(f"provides: {values}")
        parts.append(f"missing: {', '.join(missing_fill) or '(none)'}")
    elif missing:
        parts.append(f"missing: {', '.join(missing)}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# First-stage lexical prune (spec guard-serve-performance).
#
# A deterministic BM25 over the catalogue's rendered spans ranks the tools
# and the cross-encoder then scores only the top-k survivors — one
# session.run per pair, exactly as shipped (the pruner decides WHICH rows
# get scored, never HOW; batching stays forbidden).
#
# Default history (all dated, honest): shipped OFF; the owner removed the
# 1.5x latency bar and flipped ON (2026-10-10); the same day,
# decision-agreement measurement on genuinely-matching large catalogues
# FAILED the binding >=99% agreement bar (EN N=200 98.74% CALL-conditioned;
# non-English 67-89%; true matches pruned away — every disagreement scored
# lower than shipped) and multilingual argmax recall measured 0.39-0.87, so
# the default returned to OFF. The zero-lexical-overlap fallback stays (it
# fixes the arbitrary-survivor non-Latin case for opt-in users).
# Opt-in: MIVIA_GUARD_PRUNE=1|true|yes|on. Re-earning a default flip = the
# Tier-2 embedding pruner arm named in the spec, with pre-registered
# agreement bars on large catalogues.
#
# k=25 (measured 2026-10-10, EN 183-row agreement protocol): the two k=12
# N=200 misses sat at BM25 rank 18/20 — k=25 restores 100% overall and
# 100% CALL-conditioned agreement at N=200 (and passes at N=50) while
# keeping ~1/8 of shipped forwards (26 vs 201 per decision). No true tool
# ranked beyond 37 anywhere in the corpus. k does NOT fix non-English
# (lexical coverage, not budget) — that is the embed pruner's job.
#
# The embed BACKEND (embed-tool-pruner,
# MIVIA_GUARD_PRUNE_BACKEND=auto|bm25|embed) shares this seam, contract and
# config; "auto" (the default) prefers it when its artifact loads. Its own
# bars and default verdict live in that spec and
# reports/embed-pruner-2026-10-10.md.
# ---------------------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_PRUNE_K = 25
PRUNE_ENV = "MIVIA_GUARD_PRUNE"
PRUNE_K_ENV = "MIVIA_GUARD_PRUNE_K"
# Prune backend selection (spec embed-tool-pruner):
# "auto" = the multilingual embedding backend when its artifact loads,
# the lexical BM25 otherwise; "bm25"/"embed" force one.
PRUNE_BACKEND_ENV = "MIVIA_GUARD_PRUNE_BACKEND"
PRUNE_BACKEND_AUTO = "auto"
PRUNE_BACKENDS = ("auto", "bm25", "embed")
_PRUNE_TRUTHY = ("1", "true", "yes", "on")
_BM25_TOKEN = re.compile(r"[a-z0-9]+")


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    The pattern's split on every non-alphanumeric also breaks compound tool
    names ("delete_database" -> delete, database), so a request's plain
    words meet the tool's snake_case vocabulary; consistent with the veto's
    own compound-splitting asymmetry guard.
    """
    return _BM25_TOKEN.findall(text.lower() if isinstance(text, str) else "")


@dataclasses.dataclass(frozen=True)
class PruneConfig:
    """First-stage prune config, read once at guard construction.

    ``enabled`` ships OFF: decision-agreement measurement on
    genuinely-matching large catalogues failed the binding >=99% bar
    (EN N=200 98.74%; non-English 67-89% CALL-conditioned) and multilingual
    argmax recall measured 0.39-0.87 — the prune changes real decisions on
    exactly the surface (N >> k) its latency win is for. Opt-in via
    ``MIVIA_GUARD_PRUNE``; the zero-lexical-overlap skip in
    ``_prune_candidates`` removes the arbitrary-survivor non-Latin case for
    opt-in users. ``k`` is the survivor budget, applied as ``min(k, N)``.

    ``backend`` picks the pruner: ``"auto"`` (default) uses the
    multilingual embedding backend when its artifact loads and falls back
    to the lexical BM25 when it does not; ``"bm25"``/``"embed"`` force one
    (an explicit ``"embed"`` whose artifact fails to load still falls back
    to bm25 — a broken first stage must never break a decision; the loader
    warns once). Env: MIVIA_GUARD_PRUNE_BACKEND."""

    enabled: bool = False
    k: int = DEFAULT_PRUNE_K
    backend: str = PRUNE_BACKEND_AUTO

    def __post_init__(self):
        # Construction-time validation (guardfix wave, 2026-10-10):
        # load_prune_config cannot produce a bad k, but a library caller
        # can — PruneConfig(k=float("inf")) used to survive until
        # _prune_candidates's int() raised OverflowError OUTSIDE its
        # fallback except and broke the decision. Non-finite or < 1 is a
        # clear ValueError naming the field, at the construction site.
        k = self.k
        try:
            valid = (isinstance(k, (int, float))
                     and not isinstance(k, bool)
                     and math.isfinite(k) and k >= 1)
        except TypeError:  # exotic non-numeric k (e.g. a list)
            valid = False
        if not valid:
            raise ValueError(
                f"PruneConfig.k must be a finite number >= 1, got {k!r}")


def load_prune_config(env=None) -> PruneConfig:
    """PruneConfig from the environment; every garbage value falls back safe.

    The prune ships DISABLED (default returned OFF 2026-10-10 after the
    agreement-bar failure; see the module comment for the full dated
    history): only the exact truthy tokens ("1", "true", "yes", "on")
    enable it. A ``k`` that is not a positive integer parses as
    DEFAULT_PRUNE_K. A ``backend`` that is not one of PRUNE_BACKENDS
    parses as PRUNE_BACKEND_AUTO."""
    env = os.environ if env is None else env
    try:
        raw = str(env.get(PRUNE_ENV, "")).strip().lower()
    except Exception:
        raw = ""
    enabled = raw in _PRUNE_TRUTHY
    k = DEFAULT_PRUNE_K
    raw = env.get(PRUNE_K_ENV, "")
    raw = str(raw).strip() if raw is not None else ""
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed >= 1:
            k = parsed
    try:
        backend = str(env.get(PRUNE_BACKEND_ENV, "")).strip().lower()
    except Exception:
        backend = ""
    if backend not in PRUNE_BACKENDS:
        backend = PRUNE_BACKEND_AUTO
    return PruneConfig(enabled=enabled, k=k, backend=backend)


class LexicalToolPruner:
    """Deterministic BM25 first-stage pruner over rendered tool spans.

    Scores ``render_tool(tool)`` — the same span the ranking artifact was
    trained on — against the request with in-module BM25 (stdlib tokenizer,
    no new dependency) and returns the top-k tool dicts in catalogue order;
    ties break by catalogue index. A pure function of its inputs: no
    randomness, no wall clock, identical outputs on identical inputs.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b

    def select_survivors(self, request: str, tools: list[dict],
                         k: int) -> list[dict]:
        """Top-k tools by BM25(request, render_tool(tool)), catalogue order.

        Never returns more than ``min(k, N)`` tools, and never fewer than
        one while ``tools`` is non-empty (an all-zero score set degrades to
        the first k catalogue entries, so the scan can never argmax over
        nothing)."""
        n = len(tools)
        if n == 0:
            return []
        if k <= 0:
            k = n
        k = min(k, n)
        query = _bm25_tokenize(request)
        docs = [_bm25_tokenize(render_tool(t)) for t in tools]
        avgdl = sum(len(d) for d in docs) / n
        df: dict[str, int] = {}
        for d in docs:
            for term in set(d):
                df[term] = df.get(term, 0) + 1
        idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5))
               for t, c in df.items()}
        scores: list[float] = []
        for d in docs:
            tf: dict[str, int] = {}
            for term in d:
                tf[term] = tf.get(term, 0) + 1
            length_norm = (self.k1 * (1.0 - self.b + self.b * (len(d) / avgdl))
                           if avgdl else 0.0)
            s = 0.0
            for term in query:
                c = tf.get(term)
                if c:
                    s += idf[term] * (c * (self.k1 + 1.0)) / (c + length_norm)
            scores.append(s)
        top = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
        return [tools[i] for i in sorted(top)]


_DEFAULT_PRUNER = LexicalToolPruner()


# ---------------------------------------------------------------------------
# Tier-2 embedding prune backend (spec embed-tool-pruner).
#
# The lexical backend failed its binding agreement bars on large catalogues
# (EN N=200 98.74% CALL-conditioned; non-English 67-89%) because BM25 is
# lexical: a German/Japanese/Arabic request shares no surface tokens with an
# English tool span, so the true tool ranks arbitrarily. The embedding
# backend embeds the request and each rendered span with a multilingual
# sentence encoder (paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, int8
# ONNX models/tool-pruner-embed-int8.onnx) and selects survivors by cosine
# in one shared cross-lingual space — the relation a bag-of-ngrams cannot
# express. Zero-shot (no fine-tuning); export + verification recipe:
# scripts/export_tool_pruner_embed.py.
#
# MEASURED (export-time, pinned in tests/test_embedpruner.py): batching is
# FORBIDDEN for this graph too — the int8 dynamic quantization requantizes
# activations per-tensor over the batch (the exact mechanism that bit the
# router and the guard graphs), solo-vs-batch max cosine delta measured
# 0.0484 and survivor sets diverged on 30/30 seeded catalogues. Every
# embedding — request AND span — is one single-input forward; the span
# cache below is what makes that affordable.
#
# ZERO-SHOT VERDICT (2026-10-10, reports/embed-pruner-2026-10-10.md):
# FALSIFIED on the pre-registered bars. EN agreement 100%/100% at N=50/200
# (beats BM25's 98.74%), but ALL 8 non-English slices FAILED the >= 99%
# bar (CALL-conditioned-strict 33.3-90.5%; argmax recall 34-76% at k=12,
# 63-87% at k=25) — on failing pairs the model ranks the true tool in the
# 30s-40s of 50 even in English. The prune default stays OFF (both
# backends opt-in); the backend ships for opt-in use (it is the better
# cross-lingual RANKER than raw BM25 on 7/8 languages, just nowhere near
# the bar) and fine-tuning on the guard corpus is the registered
# follow-up arm.
# ---------------------------------------------------------------------------
EMBED_PRUNER_MODEL = ROOT / "models" / "tool-pruner-embed-int8.onnx"
EMBED_PRUNER_TOKENIZER = ROOT / "models" / "tool-pruner-embed-tokenizer.json"
# The embedder's sentence-transformers training profile (the export bakes
# the same truncation into the tokenizer copy).
EMBED_PRUNER_MAX_LEN = 128
# Span-embedding LRU bound: real agent catalogues repeat across turns, so a
# per-process bounded cache turns the per-span cold forward into a one-time
# cost per span. 4,096 spans x 384 float32 ~= 6 MB — inside the 2 GB budget.
EMBED_PRUNER_CACHE = 4096


class EmbeddingToolPruner:
    """Multilingual cosine first-stage pruner over rendered tool spans.

    Same ``select_survivors(request, tools, k)`` contract as
    :class:`LexicalToolPruner`: top-k by cosine(request, render_tool(tool)),
    returned in catalogue order, ties by catalogue index, never more than
    ``min(k, N)`` and never fewer than one while ``tools`` is non-empty (an
    all-zero similarity set degrades to the first k catalogue entries, so
    the scan can never argmax over nothing — the degenerate-request
    FULL-CATALOGUE skip lives in ``_prune_candidates``, the same seam as
    the lexical zero-overlap skip). A pure function of its inputs: no
    randomness, no wall clock, identical outputs on identical inputs.
    Every embedding is one single-input forward (batching measured and
    forbidden — see the backend comment above); rendered span vectors cache
    in a bounded LRU keyed by the rendered span text.
    """

    def __init__(self, session, tokenizer,
                 cache_size: int = EMBED_PRUNER_CACHE):
        self.sess = session
        self.tok = tokenizer
        self._pad_id = tokenizer.token_to_id("<pad>")
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self._lock = threading.Lock()
        # Honest cost accounting for the measurement instruments (never
        # affects an output): embedding forwards actually paid.
        self.n_forwards = 0

    def embed_one(self, text: str) -> np.ndarray:
        """L2-normalized embedding for one text — one single-input forward.

        A non-str encodes as "" (the tokenizer's special tokens only), so a
        malformed request degrades, never raises, out of the pruner.
        """
        enc = self.tok.encode(text if isinstance(text, str) else "")
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        if self._pad_id is not None:
            ids = np.where(mask == 1, ids, self._pad_id)
        vec = self.sess.run(["embedding"], {
            "input_ids": ids, "attention_mask": mask})[0][0]
        with self._lock:
            self.n_forwards += 1
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return np.asarray(vec, dtype=np.float32)

    def _span_vec(self, span: str) -> np.ndarray:
        """Cached span embedding (bounded LRU keyed by the rendered span)."""
        with self._lock:
            hit = self._cache.get(span)
            if hit is not None:
                self._cache.move_to_end(span)
                return hit
        vec = self.embed_one(span)
        with self._lock:
            self._cache[span] = vec
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vec

    def select_survivors(self, request: str, tools: list[dict],
                         k: int) -> list[dict]:
        """Top-k tools by cosine(request, render_tool(tool)), catalogue order."""
        return self.select_survivors_vec(self.embed_one(request), tools, k)

    def select_survivors_vec(self, request_vec: np.ndarray, tools: list[dict],
                             k: int) -> list[dict]:
        """The ranking over a precomputed request embedding (one forward saved)."""
        n = len(tools)
        if n == 0:
            return []
        if k <= 0:
            k = n
        k = min(k, n)
        scores = np.empty(n, dtype=np.float64)
        for i, tool in enumerate(tools):
            scores[i] = float(np.dot(self._span_vec(render_tool(tool)),
                                     request_vec))
        top = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
        return [tools[i] for i in sorted(top)]


def _build_embed_pruner() -> "EmbeddingToolPruner | None":
    """Load the embedding pruner from its verified artifact (never raises).

    Both the ONNX artifact AND its tokenizer must pass their sha256
    sidecars (integrity.verified refuses a missing/malformed sidecar too —
    the house loader semantics); any failure warns once and returns None,
    and the backend falls back to the lexical pruner.
    """
    try:
        if not EMBED_PRUNER_MODEL.is_file():
            _warn(f"embedding pruner artifact missing: "
                  f"{EMBED_PRUNER_MODEL.name}; embed backend falls back "
                  "to bm25")
            return None
        if not integrity.verified(EMBED_PRUNER_MODEL):
            _warn(f"embedding pruner failed its sha256 check: "
                  f"{EMBED_PRUNER_MODEL.name}")
            return None
        if not integrity.verified(EMBED_PRUNER_TOKENIZER):
            _warn(f"tokenizer failed its sha256 check: "
                  f"{EMBED_PRUNER_TOKENIZER.name}")
            return None
        from tokenizers import Tokenizer

        session = _cpu_session(EMBED_PRUNER_MODEL)
        tokenizer = Tokenizer.from_file(str(EMBED_PRUNER_TOKENIZER))
        tokenizer.enable_truncation(max_length=EMBED_PRUNER_MAX_LEN)
        tokenizer.no_padding()
        return EmbeddingToolPruner(session, tokenizer)
    except Exception as exc:
        _warn(f"embedding pruner load failed ({type(exc).__name__}: {exc})")
        return None


_EMBED_PRUNER: "EmbeddingToolPruner | None" = None
_EMBED_PRUNER_LOADED = False
_EMBED_PRUNER_LOCK = threading.Lock()


def load_embed_pruner() -> "EmbeddingToolPruner | None":
    """The cached embedding pruner, or None (never raises; negative-cached).

    One load per process like the guard singleton — a missing artifact is
    a stable property, so the None result is cached too and the fallback
    to the lexical backend is decided once. Tests reset the module global
    pair to re-load.
    """
    global _EMBED_PRUNER, _EMBED_PRUNER_LOADED
    if not _EMBED_PRUNER_LOADED:
        with _EMBED_PRUNER_LOCK:
            if not _EMBED_PRUNER_LOADED:
                _EMBED_PRUNER = _build_embed_pruner()
                # Flag LAST (sibling-singleton order): a concurrent reader
                # checks the flag outside the lock and must never observe
                # loaded=True with _EMBED_PRUNER still unset.
                _EMBED_PRUNER_LOADED = True
    return _EMBED_PRUNER


def _prune_overlap(query: list[str], vocab: set[str]) -> bool:
    """True when a query token that can carry lexical signal is in vocab.

    Single-character and digit-only tokens must not count as lexical
    overlap: "a"/"i" sit in nearly every English span and a lone digit
    ("東京の天気を25度で教えて" tokenizes to just ["25"], which happened to
    sit in a search_docs span) is a coincidence, not a match — either one
    used to defeat the zero-overlap skip and let an arbitrary survivor set
    discard the true tool (guardfix wave, 2026-10-10). Ranking itself is
    untouched; this gates only the skip decision.
    """
    return any(tok in vocab and len(tok) > 1 and not tok.isdigit()
               for tok in query)


def _prune_candidates(request: str, tools: list[dict],
                      cfg: "PruneConfig | None") -> "list[dict] | None":
    """Survivors under ``cfg``, or None when the full catalogue must be scored.

    None covers every safe path: the config absent, the prune disabled, a
    malformed budget, a pruner failure, or a pathological empty survivor
    set — the caller then scans the full catalogue (never an empty argmax).

    Zero-lexical-overlap is one more None path for the BM25 backend
    (multilingual prune-recall measurement, spec
    guard-serve-performance Further Notes): a request
    whose tokens share NO vocabulary with any tool span — the CJK/non-Latin
    case — leaves the lexical stage without signal (every span scores 0.0
    and the survivors degenerate to the first k catalogue entries, an
    arbitrary prune). The stage is skipped and the full catalogue is
    scored, silently like every other fallback.

    The embedding backend (spec embed-tool-pruner) has
    the analogous degenerate-embedding skip: a request whose embedding is
    all-zero carries no direction, so its cosine ranking would be
    arbitrary. ``"auto"`` resolves to embed-when-loadable/bm25-otherwise;
    an explicit backend that cannot load degrades to bm25 (the loader
    warned once at negative-cache fill).
    """
    if cfg is None or not getattr(cfg, "enabled", False) or not tools:
        return None
    try:
        k = int(getattr(cfg, "k", DEFAULT_PRUNE_K))
    except (TypeError, ValueError, OverflowError):
        # OverflowError: int() of a non-finite float from a duck-typed cfg
        # that bypassed PruneConfig's own validation.
        k = DEFAULT_PRUNE_K
    backend = getattr(cfg, "backend", PRUNE_BACKEND_AUTO)
    if backend not in PRUNE_BACKENDS:
        backend = PRUNE_BACKEND_AUTO
    try:
        if backend in (PRUNE_BACKEND_AUTO, "embed"):
            pruner = load_embed_pruner()
            if pruner is not None:
                if not isinstance(request, str) or not request.strip():
                    return None
                request_vec = pruner.embed_one(request)
                if not float(np.abs(request_vec).sum()) > 0.0:
                    return None  # degenerate: rank would be arbitrary
                return pruner.select_survivors_vec(request_vec, tools,
                                                   k) or None
        query = _bm25_tokenize(request)
        if not query:
            return None
        vocab: set[str] = set()
        for tool in tools:
            vocab.update(_bm25_tokenize(render_tool(tool)))
        if not _prune_overlap(query, vocab):
            return None
        return _DEFAULT_PRUNER.select_survivors(request, tools, k) or None
    except Exception as exc:  # a broken first stage must never break a decision
        _warn(f"prune backend {backend!r} failed "
              f"({type(exc).__name__}: {exc}); scoring the full catalogue")
        return None


def parse_catalogue_raw(text: str) -> tuple[list[dict] | None, str]:
    """(raw tool dicts, request) from a catalogue text, before rendering.

    ``None`` tools means the text is a catalogue but its tool array is missing
    or unparseable (a truncated catalogue). An empty list means there was no
    catalogue at all, or an explicitly empty one. One split/regex for every
    caller, so the builder's request-level filter and the runtime parser
    cannot drift apart.
    """
    head, sep, request = text.partition(_SPLIT)
    if not sep:
        return [], ""
    match = _CATALOGUE.search(head)
    if not match:
        return None, request.strip()
    try:
        tools = json.loads(match.group(0))
    except ValueError:
        return None, request.strip()
    return [t for t in tools if isinstance(t, dict)], request.strip()


def parse_catalogue(text: str) -> tuple[list[str], str]:
    """(tool descriptions, request) from a tool-catalogue text.

    Returns empty tools when the text is not a catalogue, so the guard is a
    no-op on ordinary text.
    """
    tools, request = parse_catalogue_raw(text)
    if tools is None:
        # A catalogue truncated by the corpus length cap: pull the
        # name/description pairs straight from the text. Positional pairing
        # is only safe when every extracted name has its own description —
        # an unterminated description string (or a missing field) shifts
        # the alignment and would fabricate pairs like a tool name glued to
        # a *parameter* description, so counts must agree.
        head = text.partition(_SPLIT)[0]
        names = re.findall(r'"name"\s*:\s*"([^"]*)"', head)
        descs = re.findall(r'"description"\s*:\s*"([^"]*)"', head)
        if not names or len(names) != len(descs):
            return [], request
        return [f"{n}: {d}".strip() for n, d in zip(names, descs)], request
    return [render_tool(t) for t in tools], request


_NUM_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million",
})
# Document extensions that mark a bare name.ext token as a FILENAME, not a
# host: "report.pdf" is a file to upload, not a URL, so a required url slot
# fed by a filename stays missing (the guard then asks for the URL instead of
# minting a CALL with {}). A token escapes the exclusion when it carries a
# scheme (https?://, ftp://), a www. prefix, or a path — the bare-domain
# shapes real BFCL requests use ("example.com", "api.service.dev") keep
# matching. Over-tightening this pattern once dropped BFCL 89.1 -> 25.1;
# tests/test_bfcl_pairs.py holds the bar.
_DOC_EXTS = (
    "pdf", "doc", "docx", "jpg", "png", "gif", "txt", "csv", "md", "zip",
    "tar", "gz", "py", "ipynb", "json", "yaml", "yml", "xml", "html",
)
_URL_PATTERN = re.compile(
    r"(?i)\b(?:(?:https?|ftp)://|www\.)\S+"
    r"|\b[a-z0-9_\-]+\.[a-z]{2,6}/[^\s]*"
    r"|\b[a-z0-9_\-]+\.(?!(?:" + "|".join(_DOC_EXTS) + r")\b)[a-z]{2,6}"
)
_NUM_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


def validate_fillability(tool: dict, request: str) -> tuple[bool, str, list[str]]:
    """Determine whether a request provides required primitive arguments for a tool.

    Deterministic CPU-only gate (< 0.05 ms). Filters ~33% of BFCL semantic
    near-miss false positives (e.g. 'sum' matching calculate_sum without numbers,
    or requests.get matching an API query without a URL) before invoking tools.

    Returns (fillable: bool, reason: str, missing_parameters: list[str]).
    """
    if not isinstance(tool, dict):
        return True, "", []
    if isinstance(tool.get("function"), dict):
        tool = tool["function"]
    params = tool.get("parameters")
    if not isinstance(params, dict):
        return True, "", []
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = list(required) if isinstance(required, list) else []
    # Same degrade-to-no-gate str-filter as the decision-layer veto: a
    # malformed catalogue ("required": [3] — a real OpenAI-tool-schema
    # shape) must not raise ('int' has no .lower') — the crash used to
    # escape through the gate span into the pipeline emergency catch-all,
    # which minted a template tool_call with name "" on the served surface.
    required = [p for p in required if isinstance(p, str)]
    if not required:
        return True, "", []
    # A non-str request supplies no values; treat it as empty so the
    # underspecified_query outcome (an honest ask) is what degrades out.
    request = request if isinstance(request, str) else ""

    req_words = [w for w in re.findall(r"\w+", request.lower()) if w]

    # Ultra-short action query without parameters (e.g. "sum", "turn on", "help me")
    if len(req_words) <= 2 and required:
        return False, "underspecified_query", required

    missing = []

    # Count numeric requirements
    num_required = 0
    num_params = []
    for p in required:
        # Same type-gated predicate as the decision-layer veto: a
        # declared-string parameter ("country") is never numeric, so both
        # paths agree on name hints like "count"/"val".
        if _is_veto_numeric(p, props):
            num_required += 1
            num_params.append(p)

    if num_required > 0:
        found_nums = len(_NUM_PATTERN.findall(request))
        found_num_words = sum(1 for w in req_words if w in _NUM_WORDS)
        total_nums = found_nums + found_num_words
        if total_nums < num_required:
            for p in num_params[total_nums:]:
                if p not in missing:
                    missing.append(p)

    # Check URL / endpoint requirements
    for p in required:
        p_spec = props.get(p, {})
        p_name = p.lower()
        p_desc = str(p_spec.get("description", "")).lower() if isinstance(p_spec, dict) else ""
        # Word-bounded "url": "curl" inside a description must not become a
        # phantom missing-url requirement.
        if p_name in ("url", "uri", "endpoint", "link") or re.search(
                r"\burl\b", p_desc):
            if not _URL_PATTERN.search(request):
                if p not in missing:
                    missing.append(p)

    if missing:
        return False, "missing_arguments", missing
    return True, "", []


# The decision-layer veto's numeric-parameter heuristic (spec
# fillability-veto-guard §2.1): name hints count only when
# the declared type is not string/boolean/object/array — "count" inside
# "country" and "val" inside "festival" must not veto a string parameter.
_VETO_NUM_NAME = ("number", "num", "count", "amount", "price", "val",
                  "index", "coord")
_VETO_NON_NUMERIC_TYPES = ("string", "boolean", "object", "array")
_VETO_NUM = re.compile(r"\b\d+(?:\.\d+)?")  # no trailing \b: 5m, 100g, 19th
# Word/underscore/camel segments of a parameter name, digits trimmed:
# "account_ids" -> ["account", "ids"], "itemCount" -> ["item", "count"],
# "num1" -> ["num"]. Hint matching is SEGMENT-EXACT, so the "count" inside
# "account" can never read as a numeric hint again (review A5) while the
# "num" of a BFCL-style "num1" slot keeps working.
_VETO_NAME_TOKEN = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def _veto_name_tokens(name: str) -> list[str]:
    return [t.strip("0123456789")
            for t in _VETO_NAME_TOKEN.findall(name or "")]


def _is_veto_numeric(name: str, props: dict) -> bool:
    if not isinstance(name, str):
        # A malformed "required" entry (an int, None) is not a numeric
        # hint; it must degrade, never raise (.lower on a non-str was the
        # v1 survivor crash inside this function).
        return False
    # Same degrade pattern for a malformed properties map: the caller
    # paths build a dict, but this predicate is tested directly and must
    # degrade, never raise (.get on a non-dict).
    props = props if isinstance(props, dict) else {}
    spec = props.get(name, {})
    spec = spec if isinstance(spec, dict) else {}
    p_type = str(spec.get("type", ""))
    if p_type in _VETO_NON_NUMERIC_TYPES:
        return False
    if p_type in ("number", "integer", "float"):
        return True
    tokens = _veto_name_tokens(name.lower())
    return any(k in tokens for k in _VETO_NUM_NAME)


def fillability_veto(tool: dict, request: str) -> tuple[bool, str, list[str]]:
    """Tightened fillability predicate for the shipped guard's decision.

    Abstains when the argmax tool's required arguments are absent from the
    request text — the ``no_tool_needed`` construction (a stripped request
    beside its own tool), where the cross-encoder is near-blind (6.9%
    abstention; the rendered ``missing:`` clause it reads moves its score by
    ~1e-05). Measured on guard-eval-v2: fires on 130/130 no_tool_needed
    pairs, false-fires on 4/183 real calls (digit-free relative dates —
    indistinguishable from a stripped request), lifting the category to
    97.7% catalogue abstention at the canonical operating point.

    Two deliberate tightenings over :func:`validate_fillability` (which is
    load-bearing in the gate span, the eval builder and corpus building; its
    decision logic on well-formed inputs is unchanged — both predicates now
    share only the malformed-``required`` str-filter that degrades garbage
    catalogue entries to no-gate instead of raising): the numeric-name
    heuristic is type-gated (see ``_VETO_NUM_NAME``), and the number regex
    keeps values attached to units (``5m``, ``100g``, ``19th``).
    Decision-layer only: composed in
    ``SplitToolMatcher.best_match``; pair-level primitives (``score_pair``,
    ``verify``) and ranking primitives (``select``, ``grade``) are untouched.

    Returns (veto_safe: bool, reason: str, missing_parameters: list[str]).
    """
    if not isinstance(tool, dict):
        return True, "", []
    if isinstance(tool.get("function"), dict):
        tool = tool["function"]
    params = tool.get("parameters")
    if not isinstance(params, dict):
        return True, "", []
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = list(required) if isinstance(required, list) else []
    if not required:
        return True, "", []

    # A malformed catalogue ("required": [null, 3]) must degrade to
    # no-veto, not raise: the pipeline's never-raise ladder catches it, but
    # the measurement instruments (bench --eval v2) score raw catalogues.
    required = [p for p in required if isinstance(p, str)]
    if not required:
        return True, "", []

    # Same non-str-request coercion as the sibling validate_fillability:
    # a non-str request supplies no values, so the honest
    # underspecified/missing path degrades out instead of raising on
    # .lower()/.findall (the asymmetry with the hardened sibling was a
    # live trap — residfix, 2026-10-10).
    request = request if isinstance(request, str) else ""

    req_words = [w for w in re.findall(r"\w+", request.lower()) if w]
    # Count the supplied numbers BEFORE the ultra-short query
    # short-circuit: "10 20" against a tool requiring two numbers is
    # fully specified even though it is two words long. The
    # underspecified_query outcome is reserved for short requests that
    # supply fewer numbers than the tool has NUMERIC slots — a string
    # slot ("delete production" -> delete_database(name)) is a value the
    # request can carry in words, so it must not trip the short-circuit
    # (review A5).
    num_params = [p for p in required if _is_veto_numeric(p, props)]
    non_num_params = [p for p in required if not _is_veto_numeric(p, props)]
    supplied_nums = len(_VETO_NUM.findall(request))
    supplied_nums += sum(1 for w in req_words if w in _NUM_WORDS)
    if len(req_words) <= 2 and supplied_nums < len(num_params):
        return False, "underspecified_query", required

    missing: list[str] = []
    if num_params:
        missing.extend(num_params[supplied_nums:])
    for p in required:
        spec = props.get(p, {})
        p_name = p.lower()
        p_desc = str(spec.get("description", "")).lower() \
            if isinstance(spec, dict) else ""
        # Word-bounded "url" (see validate_fillability): "curl" in a
        # parameter description is not a URL requirement.
        if p_name in ("url", "uri", "endpoint", "link") or re.search(
                r"\burl\b", p_desc):
            if not _URL_PATTERN.search(request) and p not in missing:
                missing.append(p)

    # Underspecified-content veto: if the request carries NOTHING beyond
    # the tool's own vocabulary (name, tool description, param
    # names/descriptions) and no numbers, it supplies no slot values —
    # "delete the database" against delete_database(name) must clarify,
    # never execute on a guessed argument (review P1, 2026-10-10).
    # Fully-specified natural calls keep content words the tool
    # vocabulary cannot explain ("find the lowest price for a Yamaha
    # guitar in Austin" -> yamaha/austin/...), so BFCL-first stays intact.
    #
    # It runs whenever ANY required slot is non-numeric (a satisfied
    # number never fills a string slot — review A4) or the numeric slots
    # are unsatisfied, and it has NO length cap: the old <= 8-word cap
    # exempted "could you please help me to delete the database" for
    # being one word too long.
    if non_num_params or supplied_nums < len(num_params):
        tool_words: set[str] = set()
        tname = tool.get("name")
        if isinstance(tname, str):
            tool_words |= {w for w in re.findall(r"[a-z]{3,}", tname.lower())}
        tdesc = tool.get("description")
        if isinstance(tdesc, str):
            tool_words |= {w for w in re.findall(r"[a-z]{3,}", tdesc.lower())}
        for p in required:
            tool_words |= {w for w in re.findall(r"[a-z]{3,}", p.lower())}
            s = props.get(p, {})
            if isinstance(s, dict):
                tool_words |= {w for w in re.findall(
                    r"[a-z]{3,}", str(s.get("description", "")).lower())}
        content = [w for w in req_words
                   if not _veto_word_explains_nothing(w, tool_words)]
        if not content:
            return False, "underspecified_query", required
    if missing:
        return False, "missing_arguments", missing
    return True, "", []


def aps_tier_set(scores: list[float], quantile: float) -> list[int]:
    """Contiguous tier set from per-tier scores (adaptive-set decode).

    Tiers are included in descending-probability order while the sum of the
    strictly-higher tiers' probabilities stays at or below ``quantile`` (the
    measured operating point — see the GRADE_SET_QUANTILE comment); the
    argmax always stays in, and the set is filled to be contiguous. Returns
    ascending indices.

    Calibration must use the same transform (scripts/calibrate_grade_sets.py
    imports this path). Escalation policy for callers: a singleton set is a
    confident tier; an adjacent pair means "between tier i and i+1"; three
    or more tiers means the text is not resolved on this ordinal scale and a
    human (or a re-phrased anchor set) is the honest next step.
    """
    if not scores:
        return []
    k = len(scores)
    if k == 1:
        return [0]
    arr = np.asarray(scores, dtype=np.float64)
    e = np.exp(arr - arr.max())
    p = e / e.sum()
    order = sorted(range(k), key=lambda i: (-p[i], i))
    chosen: list[int] = []
    cum = 0.0
    for rank, idx in enumerate(order):
        if rank > 0:
            cum += float(p[order[rank - 1]])
        if cum <= quantile:
            chosen.append(idx)
        else:
            break
    if not chosen:
        # A quantile below 0 (or NaN) admits nothing; the argmax singleton
        # is always the honest floor — an empty set cannot name a tier.
        chosen = [order[0]]
    lo, hi = min(chosen), max(chosen)
    return list(range(lo, hi + 1))


def _rps_nonconformity(probs: np.ndarray, tier: int) -> float:
    """Ranked probability score of the cumulative distribution for ``tier``.

    RPS = sum_k (F_k - 1{tier <= k}) ** 2, the proper scoring rule over the
    cumulative predictive distribution (Epstein 1969; Gneiting & Raftery
    2007). It is minimised near the argmax and grows with ordinal distance,
    so its sublevel sets are contiguous.
    """
    cumulative = np.cumsum(probs)
    target = np.arange(len(probs)) >= tier
    return float(np.sum((cumulative - target) ** 2))


def rps_tier_set(scores, quantile: float) -> list[int]:
    """Contiguous ordinal conformal set from RPS nonconformity.

    Haas, Killmaier, Javanmardi, Hüllermeier, UAI 2026 (PMLR v337,
    arXiv:2606.24959): using the ranked probability score as the
    nonconformity function yields median-centred contiguous prediction sets by
    construction. ``quantile`` is a measured calibration quantile over RPS
    values (scripts/calibrate_grade_sets.py), not a probability.
    """
    if not scores:
        return []
    k = len(scores)
    if k == 1:
        return [0]
    arr = np.asarray(scores, dtype=np.float64)
    e = np.exp(arr - arr.max())
    probs = e / e.sum()
    rps = [_rps_nonconformity(probs, i) for i in range(k)]
    included = [i for i in range(k) if rps[i] <= quantile]
    # RPS sets are median-centred, so they can exclude the argmax; the decode
    # contract (and `tier_decision`) requires the argmax to stay in.
    included.append(int(np.argmax(probs)))
    lo, hi = min(included), max(included)
    return list(range(lo, hi + 1))


def tier_decision(tier_set, anchors, argmax_index=None) -> dict:
    """Escalation policy for a grade tier set (never raises).

    Maps `grade(with_set=True)`'s tier set to the agent-loop decision:
    a singleton acts on the tier; an adjacent pair acts on the argmax with
    an explicit "between i and i+1" annotation; three or more tiers means
    the text is not resolved on this ordinal scale and the honest next step
    is to ask. Garbage input degrades to "ask".

    Returns a dict with "action" in {"act", "ask"}; "act" carries
    tier_index/tier/tier_set (plus "annotation" for an adjacent pair), "ask"
    carries tier_set/options.
    """
    try:
        idxs = [int(i) for i in (tier_set or [])]
    except (TypeError, ValueError):
        return {"action": "ask", "tier_set": [], "options": []}
    if (not anchors or not idxs or len(idxs) > len(anchors)
            or min(idxs) < 0 or max(idxs) >= len(anchors)):
        return {"action": "ask", "tier_set": [], "options": []}
    try:
        argmax_index = int(argmax_index)
    except (TypeError, ValueError):
        argmax_index = idxs[0]
    if not (0 <= argmax_index < len(anchors)):
        argmax_index = idxs[0]
    if len(idxs) == 1:
        i = idxs[0]
        return {"action": "act", "tier_index": i, "tier": anchors[i],
                "tier_set": [i]}
    lo, hi = min(idxs), max(idxs)
    contiguous = list(range(lo, hi + 1))
    if len(contiguous) <= 2:
        return {
            "action": "act",
            "tier_index": int(argmax_index),
            "tier": anchors[argmax_index],
            "annotation": f"between {anchors[lo]!r} and {anchors[hi]!r}",
            "tier_set": contiguous,
        }
    return {
        "action": "ask",
        "tier_set": contiguous,
        "options": [anchors[i] for i in contiguous],
    }


def pnorm_confidence(logits, p: float = 2.0) -> float:
    """Confidence from raw logits via p-norm normalization.

    Cattelan & Silva (UAI 2024, arXiv:2305.15508) show that
    ``max_i z_i / ||z||_p`` fixes the pathological flat-confidence behaviour
    of many classifiers, consistently under distribution shift, with no
    retraining. It is a ranking/threshold score, not a probability: equal
    logits score ``k ** (-1/p)`` and an all-zero vector scores 0.
    """
    arr = np.asarray(logits, dtype=np.float64)
    norm = float(np.linalg.norm(arr, ord=p))
    if norm == 0.0:
        return 0.0
    return float(arr.max() / norm)


def fit_risk_threshold(scores, labels, alpha: float) -> float:
    """Smallest accept threshold whose conformal risk bound is <= ``alpha``.

    Conformal risk control (Angelopoulos, Bates, Fisch, Lei, Schuster,
    ICLR 2024, arXiv:2208.02814) controls the expected value of a bounded
    monotone loss. Here the loss of accepting a pair is 1 when the pair is a
    near-miss (``labels[i] == 0``) and 0 otherwise, so the bound is a
    false-fire budget over accepted calls.

    ``scores[i] >= lambda`` accepts. The finite-sample-corrected empirical
    risk is ``n/(n+1) * mean(loss) + 1/(n+1)``; the loss falls as ``lambda``
    rises, so the smallest ``lambda`` that meets the budget keeps the most
    coverage. Returns 1.0 when no threshold meets it.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = s.size
    if n == 0:
        return 1.0
    for lam in np.unique(np.concatenate([s, np.array([1.0])])):
        accepted = s >= lam
        risk = float(np.mean(accepted & (y == 0)))
        if (n / (n + 1)) * risk + 1 / (n + 1) <= alpha:
            return float(lam)
    return 1.0


class BaseToolMatcher:
    """Base interface for single-model and cascaded tool matchers."""

    # First-stage prune config (see PruneConfig). None -> OFF: subclasses
    # and stubs that never set it keep byte-identical catalogue behavior.
    prune: "PruneConfig | None" = None

    def _score(self, request: str, tool: str) -> float:
        raise NotImplementedError

    def _score_raw(self, request: str, tool: str) -> float:
        """Match probability without the SelectiveNet acceptance collapse.

        The ranking primitives (`select`, `grade`) compare candidates against
        each other, so a per-pair reject decision is noise here: applied to
        out-of-domain prompts it zeroes every candidate and argmax degenerates
        to tie-breaking. See the SELECT_MARGIN_FLOOR comment for the numbers.
        """
        raise NotImplementedError

    def _span(self, tool: dict, request: str, catalogue) -> str:
        """Default tool span: the shipped name-and-description rendering.

        Subclasses override (ToolMatcher's fillability gate span). The base
        implementation keeps best_match/best_score usable on matchers that
        have no span policy of their own (CascadedToolMatcher built directly
        by scripts/bench_tool_abstention.py --cascaded).
        """
        return render_tool(tool)

    def _catalogue_candidates(self, request: str,
                              tools: list[dict]) -> list[dict]:
        """The tools the catalogue scan scores: the prune's survivors when a
        config is set ON, else the full catalogue. The seam cannot reach the
        scoring primitives — it only decides WHICH rows the shipped per-pair
        loop sees (one session.run per pair, unchanged)."""
        survivors = _prune_candidates(request, tools,
                                      getattr(self, "prune", None))
        return tools if survivors is None else survivors

    def _argmax_over(self, request: str, tools: list[dict],
                     candidates: list[dict]) -> tuple[dict | None, float]:
        """The shipped scan over an explicit candidate list.

        ``tools`` (the full catalogue) stays the span-rendering context so a
        tool's span is byte-identical whether or not a prune narrowed the
        candidate list. SplitToolMatcher.best_match reuses this on the gate
        with its own candidate list."""
        best_tool = None
        best_score = -1.0
        for tool in candidates:
            score = self._score(request, self._span(tool, request, tools))
            if score > best_score:
                best_score = score
                best_tool = tool
        return best_tool, max(best_score, 0.0)

    def score_pair(self, request: str, tool: str) -> float:
        """Public interface for evaluating a (request, tool) pair."""
        return self._score(request, tool)

    def best_match(self, text: str) -> tuple[dict | None, float]:
        """Highest scoring tool dict and its match probability.

        Returns (best_tool_dict, score). When not a catalogue, returns (None, 1.0).
        When an empty catalogue — or a catalogue whose request side parsed to
        empty — returns (None, 0.0).
        """
        tools, request = parse_catalogue_raw(text)
        if not request:
            # A catalogue whose request side is empty/whitespace can never
            # call anything: score 0.0 so the guard abstains (it used to
            # return 1.0 here, clearing THRESHOLD, and the pipeline's
            # rules.extract fallback minted the first catalogue tool name
            # with {} — the blank "User request:" CALL, review A1; every
            # catalogue malformation that hides the request fired the same
            # way). Not a catalogue at all (no request separator) keeps the
            # 1.0 pass-through the measurement scripts rely on, and a
            # malformed catalogue with a REAL request keeps it too (a
            # formatting fault must never drop a call).
            return None, (0.0 if _SPLIT in text else 1.0)
        if tools is None:
            return None, 1.0
        if not tools:
            return None, 0.0
        return self._argmax_over(request, tools,
                                 self._catalogue_candidates(request, tools))

    def best_score(self, text: str) -> float:
        """Highest (request, tool) match score, or 1.0 when not a catalogue."""
        _, score = self.best_match(text)
        return score

    def select(
        self,
        text: str,
        candidates: "list[str | dict] | dict[str, str]",
        with_status: bool = False,
        *,
        calibrated: bool = True,
    ) -> tuple[str, float] | tuple[str, float, str]:
        """Ranks text against multiple candidate descriptions by raw match
        probability (the SelectiveNet gate does not apply; see _score_raw).

        Returns (best_key, confidence), or (best_key, confidence, status)
        when ``with_status`` is set. Accepted candidate shapes:

        - a dict: keys are candidate identifiers, values are descriptions;
        - a list of strings: items are both identifier and description
          (sorted for deterministic tie-breaks — a set rides the same path);
        - a list of ``{"name": ..., "description": ..., "examples": [...]}``
          dicts (first-class description enrichment, 2026-10-11): the span
          is rendered by :func:`render_select_candidate` — the exact
          separator convention the measured win used ("…description….
          Examples: \"e1\" \"e2\" \"e3\"", then the standard ``"key: "``
          prefix law). Measured on Banking77 zero-shot: per-label examples
          took top-1 from 48.0% to 63.8% on the 600-row sample, and 92.5%
          precision at 20% coverage when gated by confidence. Input order
          is preserved on this shape (the caller's list order is the
          deterministic tie-break).

        If all scores are <= 0.0 or candidates is empty, returns
        ("", 0.0) or (first_key, 0.0).

        For two or more candidates the confidence is the Platt-calibrated
        softmax of the raw scores, or the softmax itself when the calibrator
        is unavailable. ``with_status`` adds ``status`` and does not change
        the confidence. ``status`` is ``"abstain"`` when that calibrated
        confidence falls under the fitted conformal threshold tau for the
        candidate count it was served (band selection — see
        ``_select_abstain_tau``; a softmax over 77 candidates tops out far
        below one over 2-4, so one global tau abstained on every large-K
        row).
        Otherwise it is ``"ambiguous"`` when the top two raw scores are
        within SELECT_MARGIN_FLOOR and the winner is at or below
        SELECT_CONF_FLOOR (the argmax then rides on a collapsed score).
        A non-positive softmax stays 0 and does not take the abstain branch.

        ``calibrated=False`` returns the pre-Platt softmax on both shapes.
        The fit script uses it so the map is not trained on its own output.
        Serving callers leave the default.

        A single candidate returns the raw score unchanged. The Platt head
        was fit on multi-candidate softmaxes; an n=1 head is a separate
        owner decision and is not applied here.
        """
        if not candidates:
            if with_status:
                return "", 0.0, "ambiguous"
            return "", 0.0
        if isinstance(candidates, dict):
            items = []
            for k, v in candidates.items():
                desc = str(v).strip()
                if not (desc.startswith(f"{k}:") or (":" in desc[:_DESC_COLON_SCAN])):
                    desc = f"{k}: {desc}"
                items.append((k, desc))
        elif isinstance(candidates, (list, tuple, set)):
            if isinstance(candidates, set) or not any(
                    isinstance(c, dict) for c in candidates):
                # sorted(): a set's iteration order is PYTHONHASHSEED-
                # dependent, and tied scores break by position —
                # deterministic key order. Pure-string lists keep the
                # shipped sorted() rendering byte-for-byte.
                items = sorted((c, c) for c in candidates)
            else:
                # Dict-form specs: preserve the caller's order (the same
                # insertion-order semantics the dict shape has; the
                # measured enrichment run ordered candidates this way).
                items = [
                    _select_item_from_spec(c) if isinstance(c, dict)
                    else (c, c)
                    for c in candidates
                ]
        else:
            items = [(str(candidates), str(candidates))]
        if not items:
            if with_status:
                return "", 0.0, "ambiguous"
            return "", 0.0
        if len(items) == 1:
            key, desc = items[0]
            score = self._score_raw(text, desc)
            if with_status:
                return key, score, "ambiguous" if score <= SELECT_CONF_FLOOR else "ok"
            return key, score

        scores = [self._score_raw(text, desc) for _, desc in items]
        best_idx = max(range(len(scores)), key=scores.__getitem__)
        best_key = items[best_idx][0]
        max_score = scores[best_idx]
        ordered = sorted(scores, reverse=True)
        tied_top2 = len(scores) > 1 and max_score - ordered[1] <= SELECT_MARGIN_FLOOR

        # Tie-break on key overlap: when the top two candidates are
        # statistically tied (margin <= SELECT_MARGIN_FLOOR), the argmax
        # rides on quantization noise. If one of the tied keys appears
        # verbatim in the text (word-bounded), it is the caller's own name
        # for what they want and is preferred. Measured on the 10-domain
        # routing set: single-word prompts like "devops" otherwise lose to
        # a saturated neighbor by a 0.0002 margin.
        #
        # Blank keys (empty/whitespace) are skipped: \b around an empty or
        # single-space key matches almost any text (\b + "" + \b matches
        # every word boundary; \b + " " + \b matches "word word"), so a
        # blank key inside the margin used to steal the win from the
        # honest argmax (guardfix wave, 2026-10-10).
        if max_score > 0.0 and tied_top2:
            text_lower = text.lower()
            tied = [
                i for i, s in enumerate(scores)
                if max_score - s <= SELECT_MARGIN_FLOOR
                and str(items[i][0]).strip()
                and re.search(rf"(?i)\b{re.escape(str(items[i][0]))}\b", text_lower)
            ]
            if tied:
                best_idx = max(tied, key=lambda i: scores[i])
                best_key = items[best_idx][0]
                max_score = scores[best_idx]

        status = "ok"
        if max_score <= 0.0 or (
            tied_top2 and max_score <= SELECT_CONF_FLOOR
        ):
            status = "ambiguous"
        confidence = 0.0
        if max_score > 0.0:
            confidence = float(1.0 / sum(math.exp(s - max_score) for s in scores))
            # One load covers the returned confidence and tau.
            # with_status does not rescale the number.
            if calibrated or with_status:
                cal = load_select_calibrator()
                judged = _apply_select_calibration(confidence, cal)
                if calibrated:
                    confidence = judged
                if with_status and cal is not None:
                    tau = _select_abstain_tau(cal, len(items))
                    if tau is not None and judged < tau:
                        status = "abstain"
        if with_status:
            return best_key, confidence, status
        return best_key, confidence

    def verify(
        self,
        text: str,
        hypothesis: str,
        threshold: float = 0.50,
    ) -> tuple[bool, float]:
        """Evaluates binary semantic entailment between text and hypothesis.

        Returns (accepted: bool, score: float).
        If the underlying model has a SelectiveNet selection head, rejection (selection < 0.5)
        collapses score to 0.0, which naturally fails verification.
        """
        score = self._score(text, hypothesis)
        accepted = bool(score >= threshold)
        return accepted, score

    def grade(
        self,
        text: str,
        anchors: list[str],
        with_set: bool = False,
        set_quantile: float = GRADE_SET_QUANTILE,
    ) -> tuple[str, float] | tuple[str, float, list[int]]:
        """Evaluates ordinal alignment across ordered anchor strings (e.g. severity levels).

        Anchors are ranked by raw match probability (see _score_raw); the
        caller supplies ordered anchor definitions and reads the argmax as
        the level. Returns (best_anchor: str, score: float).

        ``with_set=True`` returns ``(best_anchor, score, tier_indices)``
        where ``tier_indices`` is a contiguous ascending list of anchor
        indices forming the tier set (see aps_tier_set). A singleton means a
        confident tier; an adjacent pair means "between tier i and i+1";
        three or more means the text is not resolved on this scale.
        ``set_quantile`` trades coverage against set size (measured sweep in
        reports/training-report.md); the default is the pinned operating
        point.
        """
        if not anchors:
            if with_set:
                return "", 0.0, []
            return "", 0.0
        scores = [self._score_raw(text, a) for a in anchors]
        best_idx = max(range(len(scores)), key=scores.__getitem__)
        if with_set:
            tier_set = aps_tier_set(scores, set_quantile)
            lo = min(min(tier_set), best_idx)
            hi = max(max(tier_set), best_idx)
            return anchors[best_idx], float(scores[best_idx]), list(range(lo, hi + 1))
        return anchors[best_idx], float(scores[best_idx])


class ToolMatcher(BaseToolMatcher):
    """Scores (request, tool) pairs with the fine-tuned cross-encoder."""

    def __init__(self, session, tokenizer, inputs, span_fn=None, prune=None):
        self.sess = session
        self.tok = tokenizer
        self.inputs = set(inputs)
        # The gate span. ``None`` is the shipped ``name: description``;
        # ``render_tool_fillability`` is the fillability-aware gate span.
        self._span_fn = span_fn
        # First-stage prune config (catalogue scan only; every pair-level
        # and ranking primitive is untouched). ``None`` = OFF.
        self.prune = prune

    def _span(self, tool: dict, request: str, catalogue) -> str:
        if self._span_fn is None:
            return render_tool(tool)
        return self._span_fn(tool, request, catalogue)

    def _feed(self, request: str, tool: str) -> dict:
        """Model inputs for one (request, tool) pair.

        A cross-encoder reads the two spans in one sequence, so the
        tokenizer's segment ids are what mark where the request ends and
        the tool begins. Zeroing them costs about 55 points of false
        positives; see scripts/diagnose_tool_match.py.
        """
        enc = self.tok.encode_batch([(request, tool)])[0]
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        return feed

    def _predict(self, request: str, tool: str) -> tuple[float, float | None]:
        """(match probability, selection score) for one pair; selection is None
        when the graph has no SelectiveNet head.

        One pair per model call, deliberately. The int8 graph is not
        batch-invariant: this same pair scores 0.527 at batch 1 and 0.489
        at batch 2 (the fp32 graph gives 0.467 both ways, so it is the
        dynamic quantization). Scoring pairs together would let one tool's
        length and presence move another tool's score, and the guard must
        give a pair the same answer whatever else the catalogue holds.
        """
        outputs = self.sess.run(None, self._feed(request, tool))
        logits = outputs[0][0]
        diff = float(logits[0] - logits[1])
        pred_prob = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, diff))))
        if len(outputs) > 1:
            return pred_prob, float(outputs[1].flat[0])
        return pred_prob, None

    def _score(self, request: str, tool: str) -> float:
        """Match probability for one pair, gated by the SelectiveNet reject
        option: a selection score < 0.50 collapses the pair to 0.0. This is
        the acceptance decision the guard and `verify` need; ranking callers
        use `_score_raw`."""
        pred_prob, selection = self._predict(request, tool)
        if selection is not None and selection < 0.50:
            return 0.0
        return pred_prob

    def _score_raw(self, request: str, tool: str) -> float:
        return self._predict(request, tool)[0]

    def _logits(self, request: str, tool: str) -> list[float]:
        """Raw output logits for one pair, one forward pass.

        Callers that want a second confidence estimator (e.g.
        ``pnorm_confidence``) take it from these, so no extra inference runs.
        """
        outputs = self.sess.run(None, self._feed(request, tool))
        return [float(x) for x in outputs[0][0]]


class SplitToolMatcher(BaseToolMatcher):
    """Gate on one artifact, ranking on another.

    The fillability-aware gate span lifts guard abstention (80.0% -> 83.5% at
    matched 95.1% retention, exact McNemar p=0.0001) but costs ``select``,
    which reads plain candidate strings rather than tool spans. One artifact
    cannot serve both, so the gate runs the artifact trained on the gate span
    and the primitives that take free text (``select``, ``grade``, ``verify``)
    run the artifact trained on the shipped span. Both are ~23 MB.

    Every method that takes a *catalogue* goes to the gate; every method that
    takes a candidate or a hypothesis goes to the ranking artifact.

    The catalogue decision additionally applies the decision-layer fillability
    veto (:func:`fillability_veto`): a catalogue whose argmax tool's required
    parameters are absent from the request abstains regardless of score —
    the ``no_tool_needed`` quadrant the cross-encoder is near-blind to
    (6.9% -> 97.7% catalogue abstention, spec
    fillability-veto-guard). Pair-level and ranking
    primitives are untouched.

    An optional first-stage prune (:class:`PruneConfig`, default off)
    narrows the catalogue the gate scans BEFORE scoring — the survivors
    only; the veto composes after the survivor argmax exactly as shipped
    (spec guard-serve-performance).
    """

    def __init__(self, gate: "ToolMatcher", rank: BaseToolMatcher,
                 prune: "PruneConfig | None" = None):
        self.gate = gate
        self.rank = rank
        # The split owns the composition: the config lives here (never on
        # the gate), so the gate object stays byte-identical on/off.
        self.prune = prune

    def _score(self, request: str, tool: str) -> float:
        return self.gate._score(request, tool)

    def _score_raw(self, request: str, tool: str) -> float:
        return self.rank._score_raw(request, tool)

    def best_match(self, text: str):
        tools, request = parse_catalogue_raw(text)
        if not request:
            # Same early-return matrix as the base scan (the blank-request
            # abstention and the not-a-catalogue pass-through).
            return None, (0.0 if _SPLIT in text else 1.0)
        if tools is None:
            return None, 1.0
        if not tools:
            return None, 0.0
        # The prune narrows the CANDIDATES before scoring; the gate's own
        # span policy and per-pair scoring are reused verbatim via
        # _argmax_over (one session.run per survivor, never co-batched).
        best_tool, score = self.gate._argmax_over(
            request, tools, self._catalogue_candidates(request, tools))
        if best_tool is not None and score > 0.0:
            if not fillability_veto(best_tool, request)[0]:
                # Veto to abstention but keep the argmax tool: the pipeline's
                # clarify companion reads it to ask for the missing
                # parameters instead of silently emitting no_tool_call.
                return best_tool, 0.0
        return best_tool, score

    def best_score(self, text: str) -> float:
        # Through best_match, not the gate's own best_score: the veto is
        # part of the catalogue decision and the canonical measurement reads
        # best_score (bypassing it here silently disables the veto — that
        # exact bug measured as a no-op on the first canonical run).
        _best_tool, score = self.best_match(text)
        return score

    def score_pair(self, request: str, tool: str) -> float:
        return self.gate._score(request, tool)

    def sub_scores(self, request: str, tool: dict) -> tuple[float, float]:
        """(gate, rank) probabilities for one (request, tool) pair, ungated.

        ``tool`` is the raw tool dict: the gate reads its fillability span,
        the rank artifact the plain name+description span it was trained on.
        Neither sub-score passes the gate's SelectiveNet collapse (a
        selection < 0.50 hard-zeroes the pair); this exposes the soft
        probabilities underneath. The composition arm built on it was
        FALSIFIED — no composition beats the shipped collapsed gate at the
        95% retention floor — see reports/tool-match-composition.md. No
        shipped decision changes: ``best_match``/``best_score``/
        ``score_pair`` keep reading the collapsed gate score.
        """
        gate_span = self.gate._span(tool, request, [tool])
        return (self.gate._score_raw(request, gate_span),
                self.rank._score_raw(request, render_tool(tool)))

    def select(self, *args, **kwargs):
        # OWNER WIRING DECREE 2026-10-11 (zero-shot-select-r2/
        # spec.md): the wired select-rank head serves the selection surface.
        # The guard's rank head is the degrade path — missing/failed-sha
        # bytes, or the MIVIA_SELECT_RANK_HEAD=legacy rollback flag. Only
        # select moves: best_match/best_score/score_pair (gate chain),
        # grade and verify stay on the guard's own artifacts.
        head = load_select_rank_head()
        if head is not None:
            return head.select(*args, **kwargs)
        return self.rank.select(*args, **kwargs)

    def grade(self, *args, **kwargs):
        return self.rank.grade(*args, **kwargs)

    def verify(self, *args, **kwargs):
        return self.rank.verify(*args, **kwargs)


class CascadedToolMatcher(BaseToolMatcher):
    """Cascaded two-tier tool matcher: fast Tier 1 gate to deep Tier 2 escalation.

    - Tier 1 runs first (< 10 ms).
    - If Tier 1 score is outside ambiguity_band (default [0.35, 0.65])
      and context <= length_threshold: returns immediately without running Tier 2.
    - If Tier 1 falls in the ambiguity band or context exceeds length_threshold:
      escalates to Tier 2 for high-precision deep evaluation.
    - Graceful degradation: if Tier 2 is None, runs 100% functional on Tier 1 alone.
    """

    def __init__(
        self,
        tier1: BaseToolMatcher,
        tier2: BaseToolMatcher | None = None,
        ambiguity_band: tuple[float, float] = (0.35, 0.65),
        length_threshold: int = 512,
        prune: "PruneConfig | None" = None,
    ):
        self.tier1 = tier1
        self.tier2 = tier2
        self.ambiguity_band = ambiguity_band
        self.length_threshold = length_threshold
        # First-stage prune config for the catalogue scan. The cascade is
        # the object whose best_match -> _catalogue_candidates reads
        # ``self.prune`` (the BaseToolMatcher seam), so load_matcher threads
        # the config here — a cascade built with a tier2 used to drop it and
        # silently turn an explicit opt-in prune OFF (guardfix wave,
        # 2026-10-10).
        self.prune = prune

    @property
    def tok(self):
        return getattr(self.tier1, "tok", None)

    @property
    def sess(self):
        return getattr(self.tier1, "sess", None)

    @property
    def inputs(self):
        return getattr(self.tier1, "inputs", set())

    def should_escalate(self, score: float, request: str, tool: str) -> bool:
        if self.tier2 is None:
            return False
        low, high = self.ambiguity_band
        if low <= score <= high:
            return True
        if (len(request) + len(tool)) // 4 > self.length_threshold:
            return True
        return False

    def _score(self, request: str, tool: str) -> float:
        score1 = self.tier1._score(request, tool)
        if not self.should_escalate(score1, request, tool):
            return score1
        if self.tier2 is None:
            return score1
        return self.tier2._score(request, tool)

    def _score_raw(self, request: str, tool: str) -> float:
        """Ungated probability, same escalation policy as `_score`.

        Escalation keys off the Tier 1 raw score and hands the pair to Tier 2
        raw: the ambiguity band and length threshold decide *who* reads the
        pair, never the pair's accept/reject fate.
        """
        score1 = self.tier1._score_raw(request, tool)
        if not self.should_escalate(score1, request, tool):
            return score1
        if self.tier2 is None:
            return score1
        return self.tier2._score_raw(request, tool)



_GUARD = None
_GUARD_LOADED = False
_GUARD_LOCK = threading.Lock()

# QW-C (2026-10-09): Platt calibrator + conformal abstain threshold for the
# select() confidence. Loaded lazily, bound to the rank artifact sha; a
# missing/stale artifact degrades to the raw score with no abstain rule.
_SELECT_CAL = None
_SELECT_CAL_LOADED = False
_SELECT_CAL_LOCK = threading.Lock()

# Owner re-registration 2026-10-09 (QW-C: measured 11.43% on the reserve).
SELECT_ABSTAIN_ALPHA = 0.12
_SELECT_CAL_MODEL = ROOT / "models" / "select-calibration.skops"
_SELECT_CAL_META = ROOT / "models" / "select-calibration-meta.json"


def _meta_verified(meta_path: Path) -> bool:
    """True when a decision meta json passes its sha256 sidecar.

    A present sidecar must match (a tampered/stale meta must never feed a
    decision threshold); a MISSING sidecar is tolerated so non-shipped/dev
    artifacts keep loading — matching the house verified() refusal
    semantics for everything else (malformed sidecar -> refuse).
    """
    sidecar = meta_path.parent / (meta_path.name + ".sha256")
    expected, problem = integrity.expected_digest(sidecar)
    if problem == "missing":
        return True
    if problem:
        return False
    if not meta_path.is_file():
        return False
    return integrity.digests_match(integrity.sha256_file(meta_path), expected)


def load_select_calibrator():
    """(platt_lr, tau_small, tau_by_k) for the select confidence, or None.

    ``tau_small`` is the shipped conformal threshold for the small
    candidate-count band (K <= SELECT_SMALL_BAND_K_MAX — the shape the
    original fit was measured on). ``tau_by_k`` is the many-candidate
    band: a tuple of (k_fit, tau) pairs, each a conformal refit (same
    alpha, same called-risk rule, same shared Platt map) on a measured
    K-population — Banking77/CLINC150 train-split selections. None when
    the meta carries no band registration (legacy meta degrades to
    today's pre-band behavior).

    The meta pins the rank artifact sha the calibration was fitted against;
    a mismatch means the calibrator describes different bytes — degrade to
    the raw confidence rather than serve a stale calibration. The skops
    bytes themselves are sha256-sidecar-verified like every other head.
    Loading negative results are cached for the process lifetime (a
    calibrator installed after first use needs a restart) — the same
    one-shot load trade the guard singleton makes.

    OWNER WIRING DECREE 2026-10-11: the calibration served is the ACTIVE
    select head's. When the wired select-rank head loads, its OWN
    calibration is served (a Platt map fitted on a different head's
    softmax distribution is a stale calibration by this loader's own
    sha-pin contract); the legacy calibration below keeps serving the
    guard's rank head — the rollback surface and the degrade path.
    """
    global _SELECT_CAL, _SELECT_CAL_LOADED
    if not select_rank_legacy():
        head = load_select_rank_head()
        if head is not None:
            return load_select_rank_calibrator()
    if not _SELECT_CAL_LOADED:
        with _SELECT_CAL_LOCK:
            if not _SELECT_CAL_LOADED:
                try:
                    import json

                    # The meta pins tau, a decision threshold: verify its
                    # digest like the .skops body before reading it.
                    if not _meta_verified(_SELECT_CAL_META):
                        _warn("select calibrator meta failed its sha256 "
                              "check; serving raw confidence")
                    else:
                        meta = json.loads(_SELECT_CAL_META.read_text())
                        if meta.get("rank_sha256") != integrity.sha256_file(
                                DEFAULT_MODEL):
                            _warn("select calibrator stale (rank sha mismatch); "
                                  "serving raw confidence")
                        else:
                            from .extract_learned import _load_skops_verified

                            lr = _load_skops_verified(_SELECT_CAL_MODEL,
                                                      "select calibrator")
                            if lr is not None:
                                tau_by_k = None
                                many = (meta.get("bands") or {}).get("many") \
                                    or {}
                                points = many.get("tau_by_k")
                                if isinstance(points, list) and points:
                                    parsed = []
                                    for pt in points:
                                        try:
                                            parsed.append(
                                                (int(pt["k"]),
                                                 float(pt["tau"])))
                                        except (TypeError, ValueError,
                                                KeyError):
                                            parsed = None
                                            break
                                    if parsed:
                                        tau_by_k = tuple(
                                            sorted(parsed))
                                _SELECT_CAL = (lr, float(meta["tau"]),
                                               tau_by_k)
                except Exception as exc:
                    _warn(f"select calibrator unavailable "
                          f"({type(exc).__name__}: {exc})")
                # Flag LAST (sibling-singleton order): a concurrent reader
                # checks the flag outside the lock and must never observe
                # loaded=True with _SELECT_CAL still unset.
                _SELECT_CAL_LOADED = True
    return _SELECT_CAL


def _select_abstain_tau(cal, k: int) -> "float | None":
    """The conformal abstain threshold for a selection of size ``k``.

    Band selection by the registered shape key (candidate count):

    - ``k <= SELECT_SMALL_BAND_K_MAX`` (the binary referee and the K=4
      fit population): the shipped small-band tau — byte-identical
      behavior to the pre-band surface;
    - ``k >= 5``: the nearest registered many-band fit point by
      ``|log(k / k_fit)|`` (ties to the larger K — the lower, more
      conservative support). A softmax top-1 over 77 candidates tops out
      near 0.28 calibrated where the K=4 fit saw 0.88, which is why the
      shipped single tau abstained on every large-K row.

    A legacy/foreign calibrator shaped ``(lr, tau)`` — the fit script and
    the unit-test stand-ins — serves ``tau`` at every K (the pre-band
    contract). A missing many-band registration does the same: today's
    behavior, never a crash.
    """
    if cal is None:
        return None
    if len(cal) >= 3 and cal[2] and k > SELECT_SMALL_BAND_K_MAX:
        points = cal[2]
        try:
            k_fit, tau = min(
                points,
                key=lambda p: abs(math.log(k / p[0]) if k > 0 else abs(k - p[0])))
        except (TypeError, ValueError, ZeroDivisionError):
            return cal[1]
        return float(tau)
    return cal[1]


def _apply_select_calibration(raw_confidence: float, cal) -> float:
    """Monotone Platt map. Identity when ``cal`` is missing or the head fails."""
    if cal is None:
        return float(raw_confidence)
    lr = cal[0]
    try:
        import numpy as np

        p = lr.predict_proba(np.asarray([[raw_confidence]], dtype=np.float64))
        return float(p[0, 1])
    except Exception:
        return float(raw_confidence)


def calibrate_select_confidence(raw_confidence: float) -> float:
    """Monotone Platt map raw softmax -> calibrated confidence (identity
    when the calibrator is unavailable). Ranking order is unchanged."""
    return _apply_select_calibration(raw_confidence, load_select_calibrator())


# ---------------------------------------------------------------------------
# The wired zero-shot select-rank head (OWNER WIRING DECREE 2026-10-11,
# zero-shot-select-r2). The r2 candidate rank head
# (sha 2db0d0ae...) ships BESIDE the guard as models/select-rank-int8.onnx
# and serves the SELECTION surface only — SplitToolMatcher.select (hence
# matcher.select() and POST /v1/select). The guard chain is byte-identical
# to the pre-decree tree: best_match/best_score/score_pair keep reading the
# gate (8d758cff...) + rank (48c43bce...) artifacts, grade/verify keep the
# guard rank. Registered accepted trade (not silently passed): Banking77
# 81.00 vs the 85 parity bar, CLINC150 63.67 vs 78, the dedicated nomatch
# head's null abstention non-functional (confidence-gating on the wired
# head's own band-tau calibration serves abstention instead); bought: the
# binary referee fix (expected-yes 26.8 -> 99.6).
# ---------------------------------------------------------------------------
SELECT_RANK_MODEL = ROOT / "models" / "select-rank-int8.onnx"
SELECT_RANK_TOKENIZER = ROOT / "models" / "select-rank-tokenizer.json"
_SELECT_RANK_CAL_MODEL = ROOT / "models" / "select-rank-calibration.skops"
_SELECT_RANK_CAL_META = ROOT / "models" / "select-rank-calibration-meta.json"

_SELECT_RANK_HEAD: "ToolMatcher | None" = None
_SELECT_RANK_HEAD_LOADED = False
_SELECT_RANK_HEAD_LOCK = threading.Lock()
_SELECT_RANK_CAL = None
_SELECT_RANK_CAL_LOADED = False
_SELECT_RANK_CAL_LOCK = threading.Lock()


def select_rank_legacy() -> bool:
    """The rollback flag: ``MIVIA_SELECT_RANK_HEAD=legacy`` restores the
    pre-decree select() surface (guard rank head + legacy
    select-calibration-meta). Read at load time — the head singleton is
    one-shot like every sibling, so a rollback is an env flip plus a
    process restart, never a byte move."""
    return os.environ.get("MIVIA_SELECT_RANK_HEAD", "").strip().lower() \
        == "legacy"


def load_select_rank_head() -> "ToolMatcher | None":
    """The wired zero-shot select-rank head, sha-verified, or None.

    NEVER raises and NEVER touches the guard: a missing artifact, a failed
    sha256 sidecar, or the legacy rollback flag returns None and the
    selection surface degrades to the guard's rank head (the pre-decree
    behavior). Cached for the process lifetime including negative results
    (a head installed after first use needs a restart) — the same one-shot
    load trade every sibling singleton makes.
    """
    global _SELECT_RANK_HEAD, _SELECT_RANK_HEAD_LOADED
    if not select_rank_legacy() and not _SELECT_RANK_HEAD_LOADED:
        with _SELECT_RANK_HEAD_LOCK:
            if not select_rank_legacy() and not _SELECT_RANK_HEAD_LOADED:
                m = load_single_matcher(
                    SELECT_RANK_MODEL, tokenizer_path=SELECT_RANK_TOKENIZER)
                if m is None:
                    _warn("select-rank head not loadable "
                          f"({SELECT_RANK_MODEL.name}); select() serves the "
                          "guard rank head (pre-decree surface)")
                # Flag LAST (sibling-singleton order — a concurrent reader
                # checks the flag outside the lock).
                _SELECT_RANK_HEAD = m
                _SELECT_RANK_HEAD_LOADED = True
    return _SELECT_RANK_HEAD


def load_select_rank_calibrator():
    """(platt_lr, tau_small, tau_by_k) for the WIRED head, or None.

    Same shape and degrade contract as :func:`load_select_calibrator`, but
    the meta pins the select-rank head sha: a mismatch means the map
    describes different bytes — serve the raw confidence rather than a
    stale calibration. It deliberately never falls back to the LEGACY
    calibration: that map was fitted on the guard rank's softmax
    distribution, and serving it on this head would be exactly the stale
    -calibration failure the sha pin exists to prevent.
    """
    global _SELECT_RANK_CAL, _SELECT_RANK_CAL_LOADED
    if not _SELECT_RANK_CAL_LOADED:
        with _SELECT_RANK_CAL_LOCK:
            if not _SELECT_RANK_CAL_LOADED:
                try:
                    import json

                    if not _meta_verified(_SELECT_RANK_CAL_META):
                        _warn("select-rank calibrator meta failed its sha256 "
                              "check; serving raw confidence")
                    else:
                        meta = json.loads(_SELECT_RANK_CAL_META.read_text())
                        if meta.get("rank_sha256") != integrity.sha256_file(
                                SELECT_RANK_MODEL):
                            _warn("select-rank calibrator stale (head sha "
                                  "mismatch); serving raw confidence")
                        else:
                            from .extract_learned import _load_skops_verified

                            lr = _load_skops_verified(
                                _SELECT_RANK_CAL_MODEL,
                                "select-rank calibrator")
                            if lr is not None:
                                tau_by_k = None
                                many = (meta.get("bands") or {}).get("many") \
                                    or {}
                                points = many.get("tau_by_k")
                                if isinstance(points, list) and points:
                                    parsed = []
                                    for pt in points:
                                        try:
                                            parsed.append(
                                                (int(pt["k"]),
                                                 float(pt["tau"])))
                                        except (TypeError, ValueError,
                                                KeyError):
                                            parsed = None
                                            break
                                    if parsed:
                                        tau_by_k = tuple(
                                            sorted(parsed))
                                _SELECT_RANK_CAL = (lr, float(meta["tau"]),
                                                    tau_by_k)
                except Exception as exc:
                    _warn(f"select-rank calibrator unavailable "
                          f"({type(exc).__name__}: {exc})")
                # Flag LAST (sibling-singleton order).
                _SELECT_RANK_CAL_LOADED = True
    return _SELECT_RANK_CAL


@dataclasses.dataclass(frozen=True)
class Tier2Spec:
    """Tier-2 cross-encoder of a cascaded guard, loaded as one unit.

    Path, tokenizer and truncation budget always travel together: they
    describe one artifact and how the shipped guard was fine-tuned to read
    it. With ``path`` unset the auto-probe (when no explicit ``path`` is
    given to ``load_matcher``) still honors ``tokenizer_path``/``max_len``.
    """

    path: Path | str | None = None
    tokenizer_path: Path | str | None = None
    max_len: int | None = None


def _get_guard():
    """The cached guard, loaded once. None when the artifact is absent.

    Two threads can reach this on the first routing decision, and building
    the guard means a 23 MB sha256 plus a session init. The lock keeps the
    loser from building a second one.
    """
    global _GUARD, _GUARD_LOADED
    if not _GUARD_LOADED:
        with _GUARD_LOCK:
            if not _GUARD_LOADED:
                _GUARD = load_matcher()
                _GUARD_LOADED = True
    return _GUARD


def _cpu_session(path: Path):
    """int8 ONNX session with the shipped CPU thread budget.

    One place for the session options so every artifact (guard, grade,
    ordinal severity) reads with the same thread budget. The CPU memory
    arena gate is shared with the router (``ort_runtime.session_options``):
    by default the arena stays on, matching the ORT default.
    """
    import onnxruntime as ort

    from .ort_runtime import session_options

    opts = session_options()
    opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def load_single_matcher(
    path: Path | str,
    tokenizer_path: Path | str | None = None,
    max_len: int | None = None,
    span_fn=None,
    prune: "PruneConfig | None" = None,
) -> ToolMatcher | None:
    """Sha-verified int8 tool-match cross-encoder, or None (never raises)."""
    p = Path(path)
    if not p.is_file():
        return None
    if tokenizer_path is None:
        cand_tok = p.parent / (p.name.replace("-int8.onnx", "-tokenizer.json"))
        tok_p = cand_tok if cand_tok.is_file() else DEFAULT_TOKENIZER
    else:
        tok_p = Path(tokenizer_path)

    try:
        # integrity.verified refuses on a missing/malformed sidecar too: a
        # deleted sidecar must never become a load bypass (every sibling
        # loader refuses the same way).
        if not integrity.verified(p):
            _warn(f"tool-match guard failed its sha256 check: {p.name}")
            return None
        if not integrity.verified(tok_p):
            _warn(f"tokenizer failed its sha256 check: {tok_p.name}")
            return None
        from tokenizers import Tokenizer

        session = _cpu_session(p)
        tokenizer = Tokenizer.from_file(str(tok_p))
        tokenizer.enable_truncation(max_length=max_len or MAX_LEN)
        tokenizer.enable_padding(length=None)
        inputs = [i.name for i in session.get_inputs()]
        return ToolMatcher(session, tokenizer, inputs, span_fn=span_fn,
                           prune=prune)
    except Exception as exc:
        _warn(f"tool-match guard load failed ({type(exc).__name__}: {exc})")
        return None


def _staged_allowed() -> bool:
    """Whether local (gitignored) builds may be loaded. Off by default.

    A gitignored staging file must not silently outrank a committed artifact,
    or the same call returns different answers on a clean clone and on a
    machine with local builds lying around. That happened: this machine ran a
    cascade while a clean clone ran the shipped single artifact, and the live
    routing tests were calibrated to the local cascade. Local builds are
    opt-in, for development only: ``MIVIA_ALLOW_STAGED=1``.
    """
    return os.environ.get("MIVIA_ALLOW_STAGED") == "1"


def _tier1_candidates() -> list[Path]:
    """Guard candidates: the committed rank artifact, plus the staged dev
    fallback under ``MIVIA_ALLOW_STAGED=1``.

    The shipped chain is the split (gate ``tool-match-gate-int8.onnx`` + rank
    ``tool-match-int8.onnx``). The former tier-1/tier-2 cascade probes were
    removed (spec fillability-veto-guard §2.2): the
    cascade artifacts were never shipped, the cascade is strictly dominated
    by the split (79.5% @ 92.9% vs 83.5% @ 95.1% — it breaches the retention
    floor), and probing committed paths for nonexistent artifacts made
    clean-clone behavior a matter of what happened to sit in gitignored
    staging. A cascade remains available as an explicit experiment via
    ``tier2=Tier2Spec(path=...)``.
    """
    out = [DEFAULT_MODEL]
    if _staged_allowed():
        out.append(_GITIGNORED_MODELS / "tool-match-int8.onnx")
    return out


def load_matcher(
    path: Path | str | None = None,
    tokenizer_path: Path | str | None = None,
    max_len: int | None = None,
    tier2: Tier2Spec | None = None,
    ambiguity_band: tuple[float, float] = (0.35, 0.65),
    prune: "PruneConfig | None" = None,
) -> BaseToolMatcher | None:
    """Sha-verified int8 tool-match cross-encoder (split, single, or explicit cascade).

    Never raises; returns None when unverified or missing. The shipped chain
    is the split: gate ``tool-match-gate-int8.onnx`` (fillability span,
    decision + veto) over rank ``tool-match-int8.onnx``. A tier-2 deep
    matcher attaches only when the caller passes ``tier2=Tier2Spec(path=…)``
    explicitly — there is no auto-probe (spec
    fillability-veto-guard §2.2).

    ``prune``: first-stage prune config for the catalogue decision. ``None``
    reads the environment ONCE here (guard construction: MIVIA_GUARD_PRUNE /
    MIVIA_GUARD_PRUNE_K, default OFF — load_prune_config); an explicit
    config overrides the environment (tests, the latency instrument).
    """
    if prune is None:
        prune = load_prune_config()
    spec = tier2 if tier2 is not None else Tier2Spec()
    if path is not None:
        t1 = load_single_matcher(path, tokenizer_path, max_len, prune=prune)
    else:
        t1 = None
        for cand in _tier1_candidates():
            t1 = load_single_matcher(cand, tokenizer_path, max_len,
                                     prune=prune)
            if t1 is not None:
                if cand.parent == _GITIGNORED_MODELS:
                    _warn(
                        f"tool-match guard loaded from gitignored path: {cand}; "
                        "results may not reproduce from a clean clone"
                    )
                break

    if t1 is None:
        if path is not None:
            _warn(f"tool-match guard missing: {Path(path).name}; abstention disabled")
        else:
            _warn(f"tool-match guard missing: {DEFAULT_MODEL.name}; abstention disabled")
        return None

    # Tier 2: explicit spec only — never probed from committed or staging
    # paths (the staged probe shipped clean-clone-divergent behavior once).
    t2 = None
    if spec.path is not None:
        t2 = load_single_matcher(spec.path, spec.tokenizer_path, spec.max_len or 512)

    base = (CascadedToolMatcher(t1, t2, ambiguity_band=ambiguity_band,
                                prune=prune)
            if t2 is not None else t1)

    # Gate artifact: the fillability-aware span, when it is shipped. It lifts
    # abstention but costs `select`, which reads plain candidate strings, so
    # the primitives that take free text stay on `base`.
    if path is None:
        gate_path = ROOT / "models" / "tool-match-gate-int8.onnx"
        if gate_path.is_file():
            gate = load_single_matcher(gate_path, tokenizer_path, max_len,
                                       span_fn=render_tool_fillability)
            if gate is not None:
                # The split owns the prune config (composition-layer, same
                # seam as the veto); the gate object stays untouched by it.
                return SplitToolMatcher(gate, base, prune=prune)
    return base


# The ordinal `grade` artifact. Co-trained on the guard corpus plus an
# ordinal anchor-alignment task (listwise softmax over anchor descriptions
# per text; scripts/train_grade_anchor_alignment.py). The guard matcher must
# keep its measured abstention/retention profile, and the co-training measurably
# trades guard retention for grade accuracy, so grade runs on its own artifact:
# shipped guard stays untouched, grade gets the ordinal training. The anchors
# it was calibrated on are the severity templates in the training script; new
# anchor strings are scored zero-shot but read best when phrased in the same
# family ("info level: ...", "warning level: ...").
# ---------------------------------------------------------------------------
# Language-gated dual intent-rank head (2026-10-08, owner decision).
# Two promoted heads behind one dispatch: EN text -> intent-rank-int8
# (ms-marco MiniLM-L6, EN-only; 82.8-85 on English fresh reserve),
# non-EN -> intent-rank-mul-int8 (Multilingual-MiniLM-L12 + the
# kappa-0.959 multilingual corpus; 62.2 on the multilang dev split where
# the EN head scores 10.3). The shipped guard and its select() contract
# are untouched; this head serves intent-shaped candidate sets
# (/v1/select "head": "intent").
# ---------------------------------------------------------------------------
INTENT_RANK_MODEL = ROOT / "models" / "intent-rank-int8.onnx"
INTENT_RANK_MUL_MODEL = ROOT / "models" / "intent-rank-mul-int8.onnx"
INTENT_RANK_TOKENIZER = ROOT / "models" / "intent-rank-tokenizer.json"
INTENT_RANK_MUL_TOKENIZER = ROOT / "models" / "intent-rank-mul-tokenizer.json"

_INTENT_RANK: ToolMatcher | None = None
_INTENT_RANK_LOADED = False
_INTENT_RANK_LOCK = threading.Lock()
_INTENT_RANK_MUL: ToolMatcher | None = None
_INTENT_RANK_MUL_LOADED = False
_INTENT_RANK_MUL_LOCK = threading.Lock()

# English-profile dispatch, deterministic and heuristic on purpose
# (a trained gate can replace it without touching the contract). The
# non-ASCII letter share catches umlauts/accents/diacritics/CJK; the
# stopword share catches ASCII-written other languages ("danke schon,
# wann kommt meine karte"). Precision-first for the MULTILINGUAL side:
# non-English text hitting the EN head loses ~5x (10.3 vs 62.2) what
# English text hitting the MUL head loses (-7.5pp), so ambiguous text
# routes multilingual.
_EN_STOPWORDS = frozenset(
    "the and is are was were be been can could how why what when where who "
    "my me i you your we do does did not no yes with for to of on in it its "
    "this that these those have has had help please get got make made want "
    "need account card payment transfer money bank refund charge show tell "
    "know think still there here about after before from will would should".split()
)


# The decision-layer content veto's stopword set (review A4): the shared EN
# list plus the politeness fillers a request wraps its command in ("just
# delete the database"). Veto-local on purpose: _EN_STOPWORDS also feeds the
# EN/MUL language dispatch above and must not move, and validate_fillability
# (load-bearing in the gate span and the corpus builder) keeps its own list.
_VETO_STOPWORDS = _EN_STOPWORDS | frozenset(
    "just please simply can could would".split())


def _veto_word_explains_nothing(word: str, tool_words: set[str]) -> bool:
    """True when a request word can be a politeness filler, a number, or
    the tool's own vocabulary — i.e. it supplies no slot value.

    The comparison runs BOTH raw and underscore/hyphen-split: the request
    tokenizer keeps compound tokens whole ("can you delete_database") while
    the tool vocabulary splits them ([a-z]{3,} over "delete_database"),
    and that asymmetry used to smuggle a stripped request past the veto
    (review A4).
    """
    if (word in tool_words or word in _VETO_STOPWORDS
            or word in _NUM_WORDS or _VETO_NUM.search(word)):
        return True
    parts = [p for p in re.split(r"[_\-]+", word) if p]
    if not parts:
        return True  # no letters at all: nothing lexical to supply
    return all(p in tool_words or p in _VETO_STOPWORDS
               or p in _NUM_WORDS or _VETO_NUM.search(p)
               for p in parts)


def _prefers_multilingual(text: str) -> bool:
    """Heuristic EN-vs-other dispatch (deterministic; ambiguous -> True:
    a wrong EN route costs ~5x a wrong MUL route)."""
    letters = [c for c in text if c.isalpha()]
    if letters:
        non_ascii = sum(1 for c in letters if ord(c) > 127)
        if non_ascii / len(letters) > 0.10:
            return True
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in _EN_STOPWORDS)
    return hits / len(words) < 0.20


def load_intent_rank() -> ToolMatcher | None:
    """Sha-verified English intent-rank head, or None (never raises)."""
    global _INTENT_RANK, _INTENT_RANK_LOADED
    if not _INTENT_RANK_LOADED:
        with _INTENT_RANK_LOCK:
            if not _INTENT_RANK_LOADED:
                m = load_single_matcher(
                    INTENT_RANK_MODEL, tokenizer_path=INTENT_RANK_TOKENIZER)
                # Flag LAST (sibling-singleton order — a concurrent reader
                # checks the flag outside the lock).
                _INTENT_RANK = m
                _INTENT_RANK_LOADED = True
    return _INTENT_RANK


def load_intent_rank_mul() -> ToolMatcher | None:
    """Sha-verified multilingual intent-rank head, or None (never raises)."""
    global _INTENT_RANK_MUL, _INTENT_RANK_MUL_LOADED
    if not _INTENT_RANK_MUL_LOADED:
        with _INTENT_RANK_MUL_LOCK:
            if not _INTENT_RANK_MUL_LOADED:
                m = load_single_matcher(
                    INTENT_RANK_MUL_MODEL,
                    tokenizer_path=INTENT_RANK_MUL_TOKENIZER)
                _INTENT_RANK_MUL = m
                _INTENT_RANK_MUL_LOADED = True
    return _INTENT_RANK_MUL


def select_intent(text: str, candidates) -> dict:
    """Language-gated intent selection over the dual head.

    Returns {"choice", "confidence", "head" ("en"|"mul"), "lang"
    ("en"|"non-en")}. Never raises: a missing head falls through to the
    other, and a total failure returns choice "" with the honest head
    marker. Candidates follow the guard's select() shapes (list, set, or
    name->description dict).

    The confidence is the head's RAW softmax (``calibrated=False``): the
    Platt calibrator was fitted — and its meta sha-pinned — to the guard's
    rank artifact (``tool-match-int8.onnx``), not to these intent heads,
    and serving it here reported 0.74-0.93 where the head's own softmax is
    0.33-0.57 (review N3). The calibrator stays bound to the guard's rank
    surface, where its sha pin matches."""
    mul = _prefers_multilingual(text or "")
    head = load_intent_rank_mul() if mul else load_intent_rank()
    which = "mul" if mul else "en"
    if head is None:
        head = load_intent_rank() if mul else load_intent_rank_mul()
        which = "en" if mul else "mul"
    if head is None:
        return {"choice": "", "confidence": 0.0, "head": which,
                "lang": "non-en" if mul else "en"}
    choice, conf = head.select(text, candidates, calibrated=False)
    return {"choice": choice, "confidence": round(float(conf), 4),
            "head": which, "lang": "non-en" if mul else "en"}


GRADE_MODEL = ROOT / "models" / "tool-match-grade-int8.onnx"
_GRADE_MATCHER: ToolMatcher | None = None
_GRADE_LOADED = False
_GRADE_LOCK = threading.Lock()


def load_grade_matcher(path: Path | str | None = None) -> ToolMatcher | None:
    """Sha-verified ordinal grade cross-encoder, or None (never raises).

    Cached like the guard. Falls back to the gitignored staging path so a
    local candidate can be measured before promotion; that fallback warns.
    """
    global _GRADE_MATCHER, _GRADE_LOADED
    if not _GRADE_LOADED:
        with _GRADE_LOCK:
            if not _GRADE_LOADED:
                candidates = (
                    [Path(path)] if path is not None
                    else [GRADE_MODEL]
                    + ([_GITIGNORED_MODELS / GRADE_MODEL.name]
                       if _staged_allowed() else [])
                )
                for cand in candidates:
                    m = load_single_matcher(cand)
                    if m is not None:
                        if cand.parent == _GITIGNORED_MODELS:
                            _warn(
                                "grade matcher loaded from gitignored path: "
                                f"{cand}; results may not reproduce from a clean clone"
                            )
                        _GRADE_MATCHER = m
                        break
                # Cache only a SUCCESS: a failed load (explicit dev path,
                # missing artifact) must not permanently disable the grade
                # path — the next call retries, including the default path.
                if _GRADE_MATCHER is not None:
                    _GRADE_LOADED = True
    return _GRADE_MATCHER


# The promoted ordinal severity head. A single-tower text encoder (not a
# cross-encoder) decoded by a cumulative-tier head: it encodes the text alone
# and reads K-1 ordered cutpoints (`severity_head.cumulative_tier`). Measured
# through this shipped path on the frozen register panel: 70.0% exact / 93.1%
# within-one (shipped cross-encoder: 39.2% / 73.7%); reports/exp-severity-ordinal.md.
#
# It serves the fixed severity scale (`severity_head.CANONICAL_SEVERITY_ANCHORS`)
# only. The tokenizer is the DEDICATED shipped copy
# (`models/severity-tokenizer.json`, SEVERITY_TOKENIZER) — phase 0 of the
# backbone swap deliberately decoupled the heads' vocabularies from the
# router's (a future backbone swap must never feed the old heads a wrong
# vocab; scripts/train_severity_ordinal.py --max-len 256).
SEVERITY_ORDINAL_MODEL = ROOT / "models" / "severity-ordinal-int8.onnx"
SEVERITY_ORDINAL_MAX_LEN = 256
_SEVERITY_ORDINAL: "SeverityOrdinalHead | None" = None
_SEVERITY_ORDINAL_LOADED = False
_SEVERITY_ORDINAL_LOCK = threading.Lock()


class SeverityOrdinalHead:
    """Fixed-severity-scale ordinal head: one encoder + ordered cutpoints.

    ``grade`` decodes the K-1 cumulative logits to one tier with
    ``severity_head.cumulative_tier`` and returns the matching anchor plus a
    CORAL confidence. It answers only the canonical 4-tier severity scale; it
    has no notion of arbitrary anchor strings, so callers must route those to
    the cross-encoder (``load_grade_matcher``).
    """

    def __init__(self, session, tokenizer, inputs, anchors=None,
                 max_len=SEVERITY_ORDINAL_MAX_LEN):
        self.sess = session
        self.tok = tokenizer
        self.inputs = set(inputs)
        self.anchors = list(anchors if anchors is not None
                            else CANONICAL_SEVERITY_ANCHORS)
        self.max_len = max_len

    def logits(self, text: str) -> list[float]:
        """The K-1 cumulative logits for ``text`` (non-strings encode as "")."""
        enc = self.tok.encode(text if isinstance(text, str) else "")
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        return [float(z) for z in
                self.sess.run(["cumulative_logits"], feed)[0][0]]

    def grade(self, text, anchors=None, with_set=False):
        """(anchor, score) or (anchor, score, [tier]) for the fixed scale.

        The tier comes from the cumulative decode; the caller's own anchor
        strings are echoed back by index, never a canonical string the caller
        did not pass. ``with_set=True`` returns a singleton tier set, so the
        ``tier_decision`` policy acts on the decoded tier.
        """
        scale = list(anchors) if anchors else list(self.anchors)
        if not scale:
            return ("", 0.0, []) if with_set else ("", 0.0)
        logits = self.logits(text)
        tier = max(0, min(cumulative_tier(logits), len(scale) - 1))
        anchor = scale[tier]
        score = coral_probability(logits, tier)
        if with_set:
            return anchor, score, [tier]
        return anchor, score


def _build_severity_ordinal(path, tokenizer_path, max_len):
    """Load one ordinal head from an explicit path (never raises)."""
    p = Path(path)
    if not p.is_file():
        return None
    tok_p = (Path(tokenizer_path) if tokenizer_path is not None
             else SEVERITY_TOKENIZER)
    try:
        if not integrity.verified(p):
            _warn(f"ordinal severity head failed its sha256 check: {p.name}")
            return None
        # integrity.verified refuses a missing/malformed tokenizer sidecar
        # too — a deleted sidecar must never become a load bypass (the
        # phase-0 vocab decoupling exists because a wrong-vocab tokenizer
        # once degraded every grade to ask).
        if not integrity.verified(tok_p):
            _warn(f"tokenizer failed its sha256 check: {tok_p.name}")
            return None
        from tokenizers import Tokenizer

        session = _cpu_session(p)
        tokenizer = Tokenizer.from_file(str(tok_p))
        tokenizer.enable_truncation(
            max_length=max_len or SEVERITY_ORDINAL_MAX_LEN)
        tokenizer.enable_padding(length=None)
        inputs = [i.name for i in session.get_inputs()]
        return SeverityOrdinalHead(session, tokenizer, inputs)
    except Exception as exc:
        _warn(f"ordinal severity head load failed ({type(exc).__name__}: {exc})")
        return None


def load_severity_ordinal(
    path: Path | str | None = None,
    tokenizer_path: Path | str | None = None,
    max_len: int | None = None,
) -> SeverityOrdinalHead | None:
    """Sha-verified ordinal severity head, or None (never raises).

    Cached like the guard. The tokenizer default is the dedicated
    ``models/severity-tokenizer.json`` copy (SEVERITY_TOKENIZER), NOT the
    router's vocab — the phase-0 decoupling keeps every head's vocabulary
    independent of the router backbone. An explicit ``path`` bypasses the
    cache (for measurement and tests).
    """
    global _SEVERITY_ORDINAL, _SEVERITY_ORDINAL_LOADED
    if path is not None:
        return _build_severity_ordinal(path, tokenizer_path, max_len)
    if not _SEVERITY_ORDINAL_LOADED:
        with _SEVERITY_ORDINAL_LOCK:
            if not _SEVERITY_ORDINAL_LOADED:
                _SEVERITY_ORDINAL = _build_severity_ordinal(
                    SEVERITY_ORDINAL_MODEL, tokenizer_path, max_len)
                _SEVERITY_ORDINAL_LOADED = True
    return _SEVERITY_ORDINAL


# The register router (2026-10-07, reports/grade-register-router.md): the
# shipped ordinal head keeps the canonical register; a business-register
# specialist (same CORAL shape, fine-tuned with the teacher pools) serves
# only rows a cheap TF-IDF+LogReg gate routes to it. Both candidates cleared
# the pre-registered bar on the frozen slice AND did not regress the frozen
# panel. The gate is skops-persisted (extract_learned house pattern) with its
# dev-chosen tau in the meta sidecar; tau is NEVER retuned on panel or slice.
REGISTER_GATE_MODEL = ROOT / "models" / "register-gate.skops"
REGISTER_GATE_META = ROOT / "models" / "register-gate-meta.json"
SEVERITY_SPECIALIST_MODEL = ROOT / "models" / "severity-ordinal-specialist-int8.onnx"
# Arm 3 de sub-arm promotion (owner decision 2026-10-08): bilingual
# (de+en) ordinal head + the en/de language gate that routes German text
# to it inside triage_severity.
SEVERITY_DE_MODEL = ROOT / "models" / "severity-de-mixed-int8.onnx"
LANG_GATE_MODEL = ROOT / "models" / "lang-gate.skops"
LANG_GATE_META = ROOT / "models" / "lang-gate-meta.json"

_REGISTER_GATE = None
_REGISTER_GATE_TAU: float | None = None
_REGISTER_GATE_LOADED = False
_REGISTER_GATE_LOCK = threading.Lock()
_SEVERITY_SPECIALIST: "SeverityOrdinalHead | None" = None
_SEVERITY_SPECIALIST_LOADED = False
_SEVERITY_SPECIALIST_LOCK = threading.Lock()
_SEVERITY_DE = None
_SEVERITY_DE_LOADED = False
_SEVERITY_DE_LOCK = threading.Lock()
_LANG_GATE = None
_LANG_GATE_TAU = None
_LANG_GATE_LOADED = False
_LANG_GATE_LOCK = threading.Lock()


def load_register_gate() -> tuple["object | None", float | None]:
    """(register-gate pipeline, tau) sha-verified, or (None, None).

    Never raises; callers degrade to the shipped ordinal head.
    """
    global _REGISTER_GATE, _REGISTER_GATE_TAU, _REGISTER_GATE_LOADED
    if not _REGISTER_GATE_LOADED:
        with _REGISTER_GATE_LOCK:
            if not _REGISTER_GATE_LOADED:
                from .extract_learned import _load_skops_verified

                gate = _load_skops_verified(REGISTER_GATE_MODEL,
                                            "register gate")
                tau: float | None = None
                if gate is not None and REGISTER_GATE_META.is_file() \
                        and _meta_verified(REGISTER_GATE_META):
                    try:
                        meta = json.loads(REGISTER_GATE_META.read_text())
                        # type() guard: a valid-JSON non-dict (null, [], "…")
                        # makes ["tau"] raise TypeError, which must degrade
                        # to (None, None) like every other failure.
                        if type(meta) is dict:
                            tau = float(meta["tau"])
                    except (ValueError, KeyError, TypeError, OSError):
                        tau = None
                if tau is None:
                    gate = None
                _REGISTER_GATE, _REGISTER_GATE_TAU = gate, tau
                _REGISTER_GATE_LOADED = True
    return _REGISTER_GATE, _REGISTER_GATE_TAU


def load_severity_de() -> "SeverityOrdinalHead | None":
    """Bilingual de+en ordinal head (de-mixed promotion), or None."""
    global _SEVERITY_DE, _SEVERITY_DE_LOADED
    if not _SEVERITY_DE_LOADED:
        with _SEVERITY_DE_LOCK:
            if not _SEVERITY_DE_LOADED:
                _SEVERITY_DE = _build_severity_ordinal(
                    SEVERITY_DE_MODEL, None, SEVERITY_ORDINAL_MAX_LEN)
                _SEVERITY_DE_LOADED = True
    return _SEVERITY_DE


def load_lang_gate() -> tuple["object | None", float | None]:
    """(en/de language gate, tau) sha-verified, or (None, None).

    tau is recall-prioritized (de-missed tickets keep the old en-only
    grading failure; en over-routed to the bilingual head is harmless).
    Never raises; callers degrade to the shipped routing.
    """
    global _LANG_GATE, _LANG_GATE_TAU, _LANG_GATE_LOADED
    if not _LANG_GATE_LOADED:
        with _LANG_GATE_LOCK:
            if not _LANG_GATE_LOADED:
                from .extract_learned import _load_skops_verified

                gate = _load_skops_verified(LANG_GATE_MODEL, "lang gate")
                tau: float | None = None
                if gate is not None and LANG_GATE_META.is_file() \
                        and _meta_verified(LANG_GATE_META):
                    try:
                        meta = json.loads(LANG_GATE_META.read_text())
                        # type() + isinstance guard: a non-numeric tau would
                        # otherwise make lang_is_de raise OUTSIDE its try and
                        # turn every canonical triage call into an ask.
                        if type(meta) is dict and isinstance(
                                meta.get("tau"), (int, float)):
                            tau = float(meta["tau"])
                    except Exception:
                        tau = None
                if gate is not None and tau is None:
                    gate = None
                _LANG_GATE, _LANG_GATE_TAU = gate, tau
                _LANG_GATE_LOADED = True
    return _LANG_GATE, _LANG_GATE_TAU


def lang_is_de(text: str) -> bool:
    """True when the language gate says German (False on any failure)."""
    gate, tau = load_lang_gate()
    if gate is None or tau is None:
        return False
    try:
        prob = float(gate.predict_proba([text])[0, list(
            gate.classes_).index(1)])
    except Exception:
        return False
    return prob >= tau


def load_severity_specialist() -> "SeverityOrdinalHead | None":
    """Business-register specialist ordinal head, or None (never raises).

    Served only behind the register gate (see load_register_gate); it is
    never the default path for the canonical scale.
    """
    global _SEVERITY_SPECIALIST, _SEVERITY_SPECIALIST_LOADED
    if not _SEVERITY_SPECIALIST_LOADED:
        with _SEVERITY_SPECIALIST_LOCK:
            if not _SEVERITY_SPECIALIST_LOADED:
                _SEVERITY_SPECIALIST = _build_severity_ordinal(
                    SEVERITY_SPECIALIST_MODEL, None, SEVERITY_ORDINAL_MAX_LEN)
                _SEVERITY_SPECIALIST_LOADED = True
    return _SEVERITY_SPECIALIST
