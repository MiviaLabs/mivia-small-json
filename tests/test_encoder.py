"""Tests for the ONNX encoder router (the shipped routing model)."""

import pytest

from mivia_small_json import bootstrap, encoder, registry, pipeline


@pytest.fixture(autouse=True)
def _schemas():
    bootstrap.register_example_schemas()
    from mivia_small_json.diagnostic import register_diagnostic_schema
    from mivia_small_json.diff_comment import register_diff_and_review_schemas
    from mivia_small_json.tool_call import (register_no_tool_call_schema,
                                            register_tool_call_schema)

    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()
    yield
    # The registry is process-global: leaving these registered would change
    # registry.active() for every later test (the fingerprint gate included).
    for schema in ("tool_call", "no_tool_call", "diagnostic", "diff",
                   "review_comment"):
        if schema in registry.active():
            registry.unregister(schema)


def test_shipped_encoder_loads_and_routes_known_text():
    router = encoder.load_encoder()
    assert router is not None, "shipped int8 encoder failed its sha256 load"
    assert router.predict(
        ["2026-03-01T12:00:00Z ERROR [auth]: token validation failed"]
    )[0] == "log_event"
    assert router.predict(["commit " + "a" * 40 + " author: X - 2 files changed"]
                          )[0] == "commit"


def test_encoder_probabilities_are_a_distribution():
    router = encoder.load_encoder()
    probs = router.predict_proba(["ISS-1 crash on boot priority: high"])
    assert probs.shape[0] == 1 and abs(float(probs[0].sum()) - 1.0) < 1e-5


def test_encoder_refuses_an_unverified_artifact(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"not a model")
    (tmp_path / "m.onnx.sha256").write_text("0" * 64 + "\n")
    tok = tmp_path / "t.json"
    tok.write_bytes(b"not a tokenizer")
    (tmp_path / "t.json.sha256").write_text("0" * 64 + "\n")
    assert encoder.load_encoder(model, tok) is None


def test_encoder_missing_artifact_is_none(tmp_path):
    assert encoder.load_encoder(tmp_path / "absent.onnx") is None


def test_active_router_prefers_the_encoder():
    router = pipeline.load_active_router()
    assert isinstance(router, encoder.EncoderRouter)


def test_encoder_routes_through_the_pipeline():
    router = pipeline.load_active_router()
    obj, meta = pipeline.classify(
        "Jun 14 15:16:01 combo sshd[19939]: authentication failure", router)
    assert obj["type"] == "log_event"
    assert meta["valid"] is True
    assert meta["routed_by"] == "model"


def test_registry_still_holds_the_schema_set():
    assert set(registry.active()) >= {"issue", "log_event", "commit",
                                      "tool_call", "no_tool_call"}


def test_context_window_is_512_with_head_tail_budgets():
    assert encoder.MAX_LEN == 512
    assert encoder.HEAD_LEN == 128
    assert encoder.TAIL_LEN == 382
    assert 1 + encoder.HEAD_LEN + encoder.TAIL_LEN + 1 == encoder.MAX_LEN


def test_head_tail_truncate_keeps_both_ends():
    router = encoder.load_encoder()
    assert router is not None
    long_text = " ".join(f"token{i}" for i in range(2000))
    full = router.tok.encode(long_text)
    assert len(full.ids) > encoder.MAX_LEN
    ids, mask = encoder._head_tail_truncate(list(full.ids),
                                            list(full.attention_mask))
    assert len(ids) == encoder.MAX_LEN
    assert len(mask) == encoder.MAX_LEN
    # Special tokens preserved at both ends.
    assert ids[0] == full.ids[0]
    assert ids[-1] == full.ids[-1]
    # Head: first HEAD_LEN content tokens after [CLS].
    assert ids[1:1 + encoder.HEAD_LEN] == full.ids[1:1 + encoder.HEAD_LEN]
    # Tail: last TAIL_LEN content tokens before [SEP].
    assert ids[1 + encoder.HEAD_LEN:-1] == full.ids[-(encoder.TAIL_LEN + 1):-1]
    assert all(m == 1 for m in mask)


def test_head_tail_truncate_passes_short_sequences_through():
    ids = [101, 5, 6, 7, 102]
    mask = [1, 1, 1, 1, 1]
    assert encoder._head_tail_truncate(ids, mask) == (ids, mask)


class _StubVocab:
    """token_to_id-only stand-in for a tokenizers.Tokenizer vocab."""

    def __init__(self, mapping):
        self._mapping = mapping

    def token_to_id(self, token):
        return self._mapping.get(token)


def test_resolve_pad_id_prefers_the_real_pad_under_either_spelling():
    # Shipped MiniLM vocab: BERT-style [PAD].
    assert encoder._resolve_pad_id(_StubVocab({"[PAD]": 0})) == 0
    # mMiniLM (XLM-R-style) vocab: <pad> at 1, [PAD] absent — the old code
    # silently padded with id 0, which is <s> in that vocab.
    assert encoder._resolve_pad_id(
        _StubVocab({"<pad>": 1, "<unk>": 3, "<s>": 0})) == 1


def test_resolve_pad_id_falls_back_to_unk_then_zero():
    assert encoder._resolve_pad_id(_StubVocab({"<unk>": 3})) == 3
    assert encoder._resolve_pad_id(_StubVocab({})) == 0


def test_shipped_router_pads_with_the_vocabs_own_pad_token():
    router = encoder.load_encoder()
    assert router is not None
    # The shipped mMiniLM vocab spells it <pad> (id 1); [PAD] misses. The
    # router must resolve the vocab's real pad token, either spelling.
    pad = next((router.tok.token_to_id(sp)
                for sp in encoder._PAD_SPELLINGS
                if isinstance(router.tok.token_to_id(sp), int)), None)
    assert isinstance(pad, int)
    assert router.pad_id == pad


def test_mm12_candidate_vocab_resolves_to_its_pad_token():
    tok_json = (encoder.ROOT / "data" / "incoming" / "models"
                / "mm12-candidate" / "mm12-tokenizer.json")
    if not tok_json.is_file():  # gitignored staging: present where trained
        pytest.skip("mm12 candidate tokenizer not staged on this machine")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_json))
    assert tok.token_to_id("[PAD]") is None  # the trap that motivated this
    assert encoder._resolve_pad_id(tok) == 1  # <pad>


def test_encoder_routes_sequences_beyond_512_tokens():
    router = encoder.load_encoder()
    assert router is not None
    # A long multi-frame diagnostic (the test exercises MAX_LEN handling
    # and head-tail truncation; the old 120-tool catalogue fixture now
    # routes text — a recorded trade of the phase-2 class,
    # MODEL-STATE/router row).
    text = "Traceback (most recent call last):\n" + "\n".join(
        f'  File "module_{i}.py", line {i * 7}, in handler_{i}\n'
        f"    frame_{i}(context={i})" for i in range(200)
    ) + "\nValueError: invalid context after long trace"
    full = router.tok.encode(text)
    assert len(full.ids) > encoder.MAX_LEN
    pred = router.predict([text])[0]
    assert pred in registry.active()
    probs = router.predict_proba([text])
    assert abs(float(probs[0].sum()) - 1.0) < 1e-5


def test_encoder_handles_a_mixed_length_batch():
    router = encoder.load_encoder()
    assert router is not None
    texts = ["short log line",
             " ".join(f"padding-word-{i}" for i in range(1500))]
    preds = router.predict(texts)
    assert len(preds) == 2
    assert all(p in registry.active() for p in preds)
