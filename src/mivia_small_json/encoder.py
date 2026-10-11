# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""ONNX sentence-encoder router: the shipped routing model.

A fine-tuned MiniLM encoder, exported to int8 ONNX. It replaces the hashed
linear router for ROUTING only; extraction, validation, repair, and the
never-raise contract are unchanged. ``predict``/``predict_proba`` match the
router interface ``pipeline.classify`` already expects.

The load is sha256-verified from a sidecar, like the skops loader, and never
raises: any integrity problem prints a warning and returns None so callers
degrade to the skops router.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import integrity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "router-encoder-int8.onnx"
DEFAULT_TOKENIZER = ROOT / "models" / "router-tokenizer.json"
DEFAULT_CLASSES = ROOT / "models" / "router-encoder-classes.json"
# The shipped MiniLM encoder carries 512 positional embeddings natively,
# so the context window is 512 tokens: tool catalogues, stack traces and
# multi-file diffs no longer lose their tail (the user query, the terminal
# exception, the bottom patch lines) to head-only truncation.
MAX_LEN = 512
# Head-tail truncation budgets: the head keeps the global context (tool
# catalogue, issue header, diff metadata), the tail keeps the intent (user
# request, final exception). 128 + 382 content tokens + [CLS]/[SEP] = 512.
HEAD_LEN = 128
TAIL_LEN = MAX_LEN - HEAD_LEN - 2  # 382

# Energy-based out-of-domain gate (Liu et al., "Energy-based Out-of-
# distribution Detection", arXiv 2010.03759). The in-domain score is the
# logsumexp of the router logits (the negative energy, T = 1); a low score
# means the input sits off the training manifold even when the softmax is
# peaked. It costs nothing: the same logits ``predict_proba`` already
# computes. Max-softmax is a poor OOD signal for this encoder (an invoice
# scores ``issue`` at 0.997); the gate keeps the 0.85 fast path for
# in-domain traffic and falls through to the zero-shot fallback otherwise.
#
# Calibration: retuned on a larger, more diverse OOD set (355 rows: 260 real
# non-developer rows from ag_news / wikitext-2 / SetFit emotion / SetFit sst2
# fetched via the HF datasets-server API, plus 95 hand-written rows across
# invoices, receipts, business email, recipes, weather, legal and product
# descriptions) against the nine ``data/real`` dev splits. The threshold is
# the largest value that keeps in-domain fast-path retention (P(energy >= t |
# max-softmax >= 0.85), pooled over the dev splits) at or above 0.90 while
# still rejecting the confident-invoice anchor (energy 8.18): 8.4. Retention
# 0.944 pooled (0.832 on a balanced per-class view; the gate's cost sits in
# the low-energy ``commit`` and ``no_tool_call`` classes), OOD rejection
# 0.885. OOD AUROC 0.9324 (max-softmax) -> 0.9571 (energy) pooled, 0.8248 ->
# 0.8610 balanced; FPR@95TPR 0.2423 -> 0.2930 pooled, 0.9380 -> 0.8028
# balanced. The syslog anchor (logsumexp 10.02) clears the gate; both
# invoices (5.72 short, 8.18 long) fall through. Full curve and verdict:
# reports/exp-energy-gate.md.
IN_DOMAIN_ENERGY = 8.4


def _head_tail_truncate(ids: list[int], mask: list[int]) -> tuple[list, list]:
    """Truncate one encoding to ``MAX_LEN`` keeping head and tail content.

    Short sequences pass through untouched. Over-long ones become
    ``[CLS] + head + tail + [SEP]`` (``HEAD_LEN`` + ``TAIL_LEN`` content
    tokens): the single-``[SEP]`` layout keeps the exact 128/382 budgets
    inside the 512 positional embeddings.
    """
    if len(ids) <= MAX_LEN:
        return ids, mask
    head_ids = ids[1:1 + HEAD_LEN]
    head_mask = mask[1:1 + HEAD_LEN]
    # Last TAIL_LEN content tokens: skip the trailing [SEP] carried over
    # from the untruncated encoding (it is re-appended below).
    tail_ids = ids[-(TAIL_LEN + 1):-1]
    tail_mask = mask[-(TAIL_LEN + 1):-1]
    return ([ids[0]] + head_ids + tail_ids + [ids[-1]],
            [mask[0]] + head_mask + tail_mask + [mask[-1]])


def _warn(msg: str) -> None:
    print(f"[mivia-small-json] {msg}", file=sys.stderr)


# Pad-token spellings across the encoder families we ship: the MiniLM
# vocabulary uses the BERT-style "[PAD]"; converted XLM-R-style vocabs
# (mMiniLM) spell it "<pad>". A missed lookup must never silently pad
# with token id 0 — in XLM-R-style vocabs id 0 is <s>, and padding with
# a real content token is only harmless while every graph path honours
# the attention mask.
_PAD_SPELLINGS = ("[PAD]", "<pad>", "<PAD>", "[pad]")
_UNK_SPELLINGS = ("<unk>", "[UNK]")


def _resolve_pad_id(tokenizer) -> int:
    """Batch-padding token id for THIS vocab (never 0 by accident).

    Prefers the vocab's real pad token under either spelling; a vocab
    with none (rare) falls back to unk, then 0 with a warning.
    """
    for spelling in _PAD_SPELLINGS:
        pid = tokenizer.token_to_id(spelling)
        if isinstance(pid, int):
            return pid
    for spelling in _UNK_SPELLINGS:
        uid = tokenizer.token_to_id(spelling)
        if isinstance(uid, int):
            _warn(f"tokenizer has no pad token; padding with unk id {uid}")
            return uid
    _warn("tokenizer has neither a pad nor an unk token; padding with id 0")
    return 0


class EncoderRouter:
    """Router interface (``predict`` / ``predict_proba``) over an ONNX encoder."""

    def __init__(self, session, tokenizer, inputs, classes,
                 in_domain_energy: float | None = None):
        self.sess = session
        self.tok = tokenizer
        self.inputs = set(inputs)
        self.classes = list(classes)
        # Resolved once: the pad id is a property of the vocab, not of the
        # call. _logits no longer pads at all (one row per run — any
        # padding is NOT inert on the int8 graph, review A2), but the id
        # stays a resolved property of the router: a silently wrong one
        # (id 0 = <s> on XLM-R-style vocabs) is a trap for any future graph
        # path that does pad, and an accidentally real content token is
        # only harmless while every graph path honours the attention mask.
        if hasattr(tokenizer, "token_to_id"):
            self.pad_id = _resolve_pad_id(tokenizer)
        else:
            # A duck-typed stand-in without token_to_id can't be probed;
            # 0 here is explicit, not the silent fallback the resolver
            # exists to prevent.
            _warn("tokenizer not probeable for a pad token; padding "
                  "with id 0")
            self.pad_id = 0
        # logsumexp(logits) below this is judged out of domain; the pipeline
        # gates the 0.85 fast path on it. Carried on the router so the
        # decision travels with the artifact it was calibrated against.
        self.in_domain_energy = (IN_DOMAIN_ENERGY if in_domain_energy is None
                                 else float(in_domain_energy))

    def _logits(self, texts):
        import numpy as np

        # Tokenize once, truncate once; the loop below only forwards.
        encs = self.tok.encode_batch(list(texts))
        trunc = [_head_tail_truncate(list(e.ids), list(e.attention_mask))
                 for e in encs]
        if not trunc:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        # The int8 graph requantizes activations per run with a PER-TENSOR
        # scale: the min/max — and therefore every row's logits — depend on
        # ALL rows in the tensor. Review A2 (2026-10-10) caught the padding
        # variant: pad embeddings shift the scale, so "fix: null pointer in
        # parser" routes 'commit' solo but 'log_event' as row 0 next to a
        # longer line, and classify_dump silently loses a line to the flip.
        # Measured on the shipped artifact, padding is only one instance of
        # the hazard: co-batching ANY row with different content — even at
        # identical tokenized length, with no padding at all — shifts a
        # row's logits by whole points (pairing it with "fix: null deref in
        # parser" moves logits by up to 6.9 and flips the argmax), while a
        # byte-identical batch is bitwise stable. Grouping by length cannot
        # restore solo equality; the only co-batching that preserves a row's
        # logits is identical content. Correctness over microseconds (the
        # owner accepted milliseconds): run ONE session.run PER ROW, never
        # padded — neither up to the longest row nor out to MAX_LEN. Every
        # row's forward pass is then bit-for-bit what a solo call computes,
        # so predict/predict_proba keep their per-row contracts at any
        # batch size. A 32-line dump costs 32 small runs; head-tail
        # truncation above already caps each at MAX_LEN.
        outs = []
        for ids, mask in trunc:
            feed = {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.array([mask], dtype=np.int64),
            }
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            outs.append(self.sess.run(None, feed)[0][0])
        return np.stack(outs)

    def predict(self, texts):
        return [self.classes[i]
                for i in self._logits(list(texts)).argmax(-1).tolist()]

    def predict_proba(self, texts):
        import numpy as np

        logits = self._logits(list(texts))
        exp = np.exp(logits - logits.max(-1, keepdims=True))
        return exp / exp.sum(-1, keepdims=True)

    def predict_proba_with_energy(self, texts):
        """One forward pass -> (softmax probs, in-domain energy per row).

        ``energy`` is ``logsumexp(logits)``, the energy in-domain score
        (higher = closer to the training manifold). The softmax and the
        energy come from the same logits, so the OOD gate adds no forward
        pass over ``predict_proba``.
        """
        import numpy as np

        logits = self._logits(list(texts))
        shifted = logits - logits.max(-1, keepdims=True)
        exp = np.exp(shifted)
        denom = exp.sum(-1)
        probs = exp / denom[:, None]
        energy = logits.max(-1) + np.log(denom)
        return probs, energy

    def in_domain_scores(self, texts):
        """In-domain energy score per text (standalone; one forward pass).

        The routing hot path should call ``predict_proba_with_energy``
        instead so the score rides along with the probabilities.
        """
        import numpy as np

        logits = self._logits(list(texts))
        m = logits.max(-1, keepdims=True)
        return m[:, 0] + np.log(np.exp(logits - m).sum(-1))



def load_encoder(path: Path | str | None = None,
                 tokenizer_path: Path | str | None = None,
                 classes: tuple | None = None):
    """Sha-verified int8 ONNX router, or None (never raises)."""
    path = Path(path) if path is not None else DEFAULT_MODEL
    tokenizer_path = (Path(tokenizer_path) if tokenizer_path is not None
                      else DEFAULT_TOKENIZER)
    try:
        if not path.is_file():
            _warn(f"encoder artifact missing: {path}; using the skops router")
            return None
        if not integrity.verified(path):
            _warn(f"encoder artifact failed its sha256 check: {path.name}; "
                  "refusing unverified load")
            return None
        if not integrity.verified(tokenizer_path):
            _warn(f"tokenizer failed its sha256 check: "
                  f"{tokenizer_path.name}; refusing unverified load")
            return None
        import onnxruntime as ort
        from tokenizers import Tokenizer

        from .ort_runtime import apply_cpu_thread_budget, session_options

        # Same thread budget as the guard (tool_match._cpu_session): the
        # router used to run on ORT's default pool and could oversubscribe
        # the cores the guard's session already holds (review §11).
        session = ort.InferenceSession(
            str(path),
            sess_options=apply_cpu_thread_budget(session_options()),
            providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # No tokenizer-side truncation or padding: _logits applies
        # head-tail truncation per sequence (keeping both ends inside the
        # 512 positional embeddings) and forwards one row per session.run —
        # no row is ever padded or co-batched with different content (the
        # per-tensor activation requantization makes both non-inert).
        tokenizer.no_truncation()
        tokenizer.no_padding()
        if classes is None:
            # The output index -> schema map is fixed at training time
            # (sorted(set(y))), so it travels beside the artifact. It is
            # verified too: a wrong map mislabels every route silently.
            if not integrity.verified(DEFAULT_CLASSES):
                _warn(f"class map failed its sha256 check: "
                      f"{DEFAULT_CLASSES.name}; refusing unverified load")
                return None
            classes = tuple(json.loads(
                DEFAULT_CLASSES.read_text(encoding="utf-8")))
        inputs = [i.name for i in session.get_inputs()]
        # Scale-bound calibration constants travel in a sha-verified meta
        # sidecar (rebuild plan Phase 0), not as code defaults: the energy
        # threshold is calibrated against THIS artifact's logit scale and
        # must be re-registered whenever the artifact changes.
        energy = IN_DOMAIN_ENERGY
        meta_path = path.parent / (path.stem.replace("-int8", "")
                                   + "-meta.json")
        if meta_path.is_file():
            if not integrity.verified(meta_path):
                _warn(f"encoder meta failed its sha256 check: "
                      f"{meta_path.name}; using the code default "
                      f"({IN_DOMAIN_ENERGY})")
            else:
                try:
                    energy = float(json.loads(
                        meta_path.read_text(encoding="utf-8"))
                        .get("in_domain_energy", IN_DOMAIN_ENERGY))
                except (ValueError, TypeError):
                    _warn(f"encoder meta unreadable: {meta_path.name}; "
                          "using the code default")
        return EncoderRouter(session, tokenizer, inputs, classes,
                             in_domain_energy=energy)
    except Exception as exc:  # corrupt file, missing runtime, bad graph
        _warn(f"encoder load failed ({type(exc).__name__}: {exc}); "
              "using the skops router")
        return None
