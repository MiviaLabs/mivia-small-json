"""Fix validation (review A2, 2026-10-10): batching must not change routes.

The shipped int8 ONNX graph requantizes activations per run with a
per-tensor scale, so a row's logits depend on every row in its tensor.
The old ``_logits`` right-padded rows to the batch's longest sequence
and co-batched them in one run: padding (and any co-batched row with
different content) shifted whole logits and flipped argmaxes —
"fix: null pointer in parser" routed 'commit' solo and 'log_event' in a
batch, and classify_dump silently lost a line to the flip.

``_logits`` now forwards ONE row per ``session.run``, never padded, so
every row's forward pass is bit-for-bit what a solo call computes.
These tests pin that property.
"""

import math

import numpy as np
import pytest

from mivia_small_json.encoder import load_encoder

pytestmark = pytest.mark.skipif(
    load_encoder() is None,
    reason="shipped ONNX encoder absent: the batch-invariance property "
           "pins the encoder chain")

# Mixed-length corpus: short 'fix: ...' lines, long multi-clause lines,
# the report's flip pair, and the near-tie rows that flipped under the
# old padded batching. >= 20 texts, 19 distinct tokenized lengths.
CORPUS = [
    "fix: null pointer in parser",
    "When user boots machine, system panics with exit code 1 and the "
    "watchdog restarts the node twice.",
    "ISS-900: service crashes on boot",
    "fix: null deref in parser",
    "INFO started",
    "2026-03-01T12:00:00Z ERROR [auth]: token validation failed for user "
    "admin from 10.0.0.7 after three consecutive retries",
    "commit 9f1788fabc123def4567890abcdef1234567890 author: X - 2 files "
    "changed",
    "feat(api): add bearer auth to the REST quartet and the MCP streamable "
    "HTTP surface with a sha-verified artifact loader",
    "hey can you tell me a joke about databases",
    "Traceback (most recent call last):\n  File \"parser.py\", line 42, in "
    "parse\n    node.consume()\nValueError: null pointer",
    "src/main.c:12:5: error: 'foo' undeclared (first use in this function)",
    "could you please help me to delete the database",
    "Jun 14 15:16:01 combo sshd[19939]: authentication failure rhost="
    "10.0.0.1 user=root",
    "the server is down again and nobody seems to know why this keeps "
    "happening every single time we deploy on friday afternoons",
    "review: the retry loop should be a fallback, not the primary "
    "mechanism; please add a bounded backoff before the second attempt",
    "WARN disk usage at 91 percent on /var, rotation did not run last "
    "night",
    "tool: get_weather arguments: {\"city\": \"Paris\"}",
    "fix: null pointer in parser when the tokenizer returns an empty "
    "encoding for a whitespace-only input string",
    "ERROR",
    "panic: runtime error: invalid memory address or nil pointer "
    "dereference goroutine 1 [running]: main.main()",
    "the invoice totals 1,234.56 EUR including VAT, please pay by Friday",
    "When the batch contains rows of different tokenized lengths the "
    "current implementation right-pads every row to the longest sequence "
    "in the batch, and on the shipped int8 graph that padding is not "
    "inert: the same line that routes to commit solo routes to log_event "
    "in a batch, which is the exact defect under repair here today",
]

# The A2 paste: its batched routes fed classify_dump's atomicity probe,
# and the flip silently dropped the watchdog line from the dump.
PASTE_LINES = [
    "ISS-900: service crashes on boot",
    "When user boots machine, system panics with exit code 1 and the "
    "watchdog restarts the node twice.",
    "fix: null pointer in parser",
    "fix: null deref in parser",
    "INFO started",
]


def test_batched_predict_matches_solo_on_mixed_lengths():
    router = load_encoder()
    solo = [router.predict([t])[0] for t in CORPUS]
    batched = router.predict(CORPUS)
    assert solo == batched, [
        (i, t[:50], a, b) for i, (t, a, b)
        in enumerate(zip(CORPUS, solo, batched)) if a != b]


def test_paste_lines_batched_match_solo():
    router = load_encoder()
    solo = [router.predict([t])[0] for t in PASTE_LINES]
    assert router.predict(PASTE_LINES) == solo


def test_predict_proba_with_energy_rows_unchanged_by_batching():
    router = load_encoder()
    solo_probs, solo_energy = zip(
        *(router.predict_proba_with_energy([t]) for t in CORPUS))
    probs, energy = router.predict_proba_with_energy(CORPUS)
    assert probs.shape == (len(CORPUS), len(router.classes))
    assert energy.shape == (len(CORPUS),)
    assert all(math.isfinite(float(e)) for e in energy)
    assert all(abs(float(p.sum()) - 1.0) < 1e-4 for p in probs)
    np.testing.assert_allclose(probs, np.concatenate(solo_probs), rtol=1e-5,
                               atol=1e-6)
    np.testing.assert_allclose(energy, np.concatenate(solo_energy),
                               rtol=1e-5, atol=1e-6)


def test_logits_never_co_batches_or_pads_rows():
    """White-box: the invariant behind batch-invariance on this graph.

    Per-tensor activation requantization makes any co-batched row with
    different content (padded or not) shift a row's logits. The guard is
    one row per session.run, no padding: every feed is (1, L) with an
    all-ones mask and L == the row's own tokenized length.
    """
    router = load_encoder()
    shapes = []
    orig = router.sess.run

    def spy(names, feed):
        ids = np.asarray(feed["input_ids"])
        mask = np.asarray(feed["attention_mask"])
        shapes.append((int(ids.shape[0]), int(ids.shape[1]),
                       bool((mask == 1).all())))
        return orig(names, feed)

    router.sess.run = spy
    try:
        router.predict(CORPUS)
    finally:
        router.sess.run = orig
    assert len(shapes) == len(CORPUS)
    for text, (batch, width, mask_ones) in zip(CORPUS, shapes):
        assert batch == 1, f"co-batched: {text[:40]!r}"
        assert mask_ones, f"padded mask: {text[:40]!r}"
        own = len(router.tok.encode(text).ids)
        assert width == min(own, 512), (
            f"padded to {width}, own length {own}: {text[:40]!r}")


def test_empty_batch_returns_empty():
    router = load_encoder()
    assert router.predict([]) == []
    probs, energy = router.predict_proba_with_energy([])
    assert probs.shape == (0, len(router.classes))
    assert energy.shape == (0,)


# (equal-length co-batching measurement pin retired: the promoted
# router artifact makes the co-batch effect inert — dev reproduces
# the same failure, a stale pin no full suite caught. The served
# contract does not move: _logits forwards ONE row per session.run,
# pinned by the no-padding tests above. Re-measure the A2 rationale
# before EVER re-introducing batching; the private tree keeps this
# pin until then.)
