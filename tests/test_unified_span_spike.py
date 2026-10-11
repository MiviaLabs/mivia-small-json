"""Focused gates for the unified-span-spike arm (phase-1 spike).

The decode-contract tests are pure CPU + fabricated logits — they always
run and pin the registered decode for phase 2 regardless of whether a
candidate artifact exists. Artifact-gated tests (sha sidecar, exported
graph input contract) SKIP LOUDLY when no candidate is staged in
.scratch/unified-span/ — a bar must not silently vanish (the spike is
dev-only and never promotes; models/ is never touched).

Registered arm: unified-span-extractor
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "scripts")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

import train_unified_span_spike as T  # noqa: E402

SPIKE_DIR = ROOT / ".scratch/unified-span"

REQUEST = "restart nginx on host web-01 staging now"
# token char spans (trimmed, whitespace-free — the encode_pair contract)
TOKS = [(0, 7), (8, 13), (14, 16), (17, 21), (22, 28), (29, 36)]
#               restart   nginx     on     host    web-01  staging


def _enc(req_positions=(0, 1, 2, 3, 4, 5)):
    return {"ids": [0] * len(TOKS), "mask": [1] * len(TOKS),
            "offsets": list(TOKS), "schema_pos": [],
            "req_pos": list(req_positions)}


def _logits(F, hits, T_len=len(TOKS), value=4.0, other=-4.0):
    """hits: dict field -> (start_tok, end_tok)."""
    sl = np.full((F, T_len), other, dtype=np.float32)
    el = np.full((F, T_len), other, dtype=np.float32)
    for f, (s, e) in hits.items():
        sl[f, s] = value
        el[f, e] = value
    return sl, el


NAMES = ["service", "host"]
PROPS = [{"type": "string", "description": "service"}, {"type": "string"}]
FTOKS = [[0], [0]]  # schema tokens; unused by fill()


def _decode():
    return T.Decode(sess=None)  # fill() never touches the session


def test_enum_coercion_emits_declared_member():
    d = _decode()
    props = [{"type": "string", "enum": ["staging", "production"]},
             {"type": "string"}]
    logits = _logits(2, {0: (5, 5), 1: (4, 4)})
    filled, missing, _ = d.fill(REQUEST, ["environment", "host"], props,
                                _enc(), FTOKS, logits, 0.5, "D")
    assert filled.get("environment") == "staging"   # declared member, not span
    assert filled.get("host") == "web-01"
    assert missing == []


def test_enum_mismatch_abstains_never_fabricates():
    d = _decode()
    # 'staging' is verbatim in the request but NOT a declared member
    props = [{"type": "string", "enum": ["production"]}, {"type": "string"}]
    logits = _logits(2, {0: (5, 5), 1: (4, 4)})
    filled, missing, _ = d.fill(REQUEST, ["environment", "host"], props,
                                _enc(), FTOKS, logits, 0.5, "D")
    assert "environment" in missing          # abstain over fabricate
    assert filled == {"host": "web-01"}


def test_enum_constrained_variant_decodes_member_span():
    d = _decode()
    props = [{"type": "string", "enum": ["staging", "production"]},
             {"type": "string"}]
    # argmax drifts to 'restart' (sigmoid(6) ~ 0.998), but the member span
    # 'staging' still scores sigmoid(3) ~ 0.95 — above the floor, so
    # decode-C's member-constrained pick fires; decode-D would emit the
    # drifted span.
    logits = _logits(2, {0: (0, 0), 1: (4, 4)}, value=6.0)
    logits[0][0, 5] = 3.0                    # member-span start (field 0)
    logits[1][0, 5] = 3.0                    # member-span end   (field 0)
    filled, missing, _ = d.fill(REQUEST, ["environment", "host"], props,
                                _enc(), FTOKS, logits, 0.5, "C")
    assert filled.get("environment") == "staging"  # constrained to member
    assert missing == []


def test_multiword_span_emitted_verbatim():
    d = _decode()
    props = [{"type": "string"}, {"type": "string"}]
    # a request whose value is multi-word ('web-01 staging')
    req = "restart nginx on web-01 staging now"
    enc = {"ids": [0] * 5, "mask": [1] * 5,
           "offsets": [(0, 7), (8, 13), (14, 16), (17, 31), (32, 35)],
           "schema_pos": [], "req_pos": [0, 1, 2, 3, 4]}
    logits = _logits(2, {0: (1, 1), 1: (3, 3)}, T_len=5)
    filled, missing, _ = d.fill(req, ["service", "target"], props, enc,
                                [[0], [0]], logits, 0.5, "D")
    assert filled == {"service": "nginx", "target": "web-01 staging"}
    assert missing == []


def test_greedy_exclusivity_higher_score_wins_loser_abstains():
    d = _decode()
    # both fields point at the same 'nginx' token; field 0 strictly more
    # confident (strong start AND end), field 1 has a weak end boundary
    logits = _logits(2, {0: (1, 1), 1: (1, 1)}, value=4.0, other=-4.0)
    logits[0][0, 1] = 8.0                    # service start: 0.9997
    logits[0][1, 1] = 8.0                    # service end:   0.9997
    logits[1][0, 1] = 8.0                    # host start:    0.9997
    logits[1][1, 1] = 2.0                    # host end: 0.881 -> 0.940
    filled, missing, dbg = d.fill(REQUEST, NAMES, PROPS, _enc(), FTOKS,
                                  logits, 0.3, "D")
    assert filled == {"service": "nginx"}
    assert missing == ["host"]
    assert dbg["collisions"] == 1


def test_below_emit_floor_abstains():
    d = _decode()
    # sigmoid(-2) ~ 0.119: every span scores ~ 0.119 < the 0.5 floor
    logits = _logits(2, {0: (1, 1), 1: (4, 4)}, value=-2.0, other=-8.0)
    filled, missing, _ = d.fill(REQUEST, NAMES, PROPS, _enc(), FTOKS,
                                logits, 0.5, "D")
    assert filled == {}                       # nothing confident: all abstain
    assert missing == NAMES


def test_integer_slot_coerces_to_int():
    d = _decode()
    req = "set the pool size to 500"
    enc = {"ids": [0] * 6, "mask": [1] * 6,
           "offsets": [(0, 3), (4, 7), (8, 12), (13, 17), (18, 20), (21, 24)],
           "schema_pos": [], "req_pos": [0, 1, 2, 3, 4, 5]}
    logits = _logits(1, {0: (5, 5)}, T_len=6)
    filled, missing, _ = d.fill(req, ["size"], [{"type": "integer"}], enc,
                                [[0]], logits, 0.5, "D")
    assert filled == {"size": 500}            # int, not "500"
    assert missing == []


def test_span_length_cap_bounds_candidates():
    d = _decode()
    # a span of 10 request tokens must be unreachable (SPAN_MAX_TOKENS = 8):
    # the only high-scoring boundary pair is (start 0, end 9); every legal
    # partial span scores ~0.5, below the 0.6 floor
    n = 10
    enc = {"ids": [0] * n, "mask": [1] * n,
           "offsets": [(2 * i, 2 * i + 1) for i in range(n)],
           "schema_pos": [], "req_pos": list(range(n))}
    logits = _logits(1, {0: (0, n - 1)}, T_len=n, value=8.0, other=-8.0)
    filled, missing, _ = d.fill("a b c d e f g h i j", ["x"],
                                [{"type": "string"}], enc, [[0]], logits,
                                0.6, "D")
    assert filled == {}                       # cap forbids the 10-token span
    assert missing == ["x"]


def test_row_exact_mirrors_shipped_construct():
    assert T._row_exact({"a": "x"}, [], {"a": "x"})
    assert not T._row_exact({}, ["a"], {"a": "x"})          # empty abstention
    assert not T._row_exact({"a": "x", "b": 1}, [], {"a": "x"})
    assert not T._row_exact({"a": "y"}, [], {"a": "x"})


# --- artifact-gated gates (skip loudly when no candidate is staged) ---------

def _staged_candidates():
    out = []
    if SPIKE_DIR.is_dir():
        for int8 in sorted(SPIKE_DIR.glob("cand-*/unified-span-*-int8.onnx")):
            tokj = int8.parent / (int8.name.replace("-int8.onnx",
                                                    "-tokenizer.json"))
            if tokj.is_file():
                out.append((int8, tokj))
    return out


@pytest.mark.skipif(not _staged_candidates(),
                    reason="no unified-span candidate staged in "
                           ".scratch/unified-span/ (spike artifacts are "
                           "gitignored; the contract gates above always run)")
def test_staged_candidate_sidecars_and_graph_contract():
    for int8, tokj in _staged_candidates():
        sidecar = Path(str(int8) + ".sha256")
        assert sidecar.is_file(), f"missing sha sidecar for {int8.name}"
        assert sidecar.read_text().split()[0] == \
            hashlib.sha256(int8.read_bytes()).hexdigest()
        import onnxruntime as ort

        sess = ort.InferenceSession(str(int8),
                                    providers=["CPUExecutionProvider"])
        names = {i.name for i in sess.get_inputs()}
        # the designed-out skew pin: NO token_type_ids input exists
        assert names == {"input_ids", "attention_mask", "field_mask"}, \
            f"{int8.name}: unexpected graph inputs {sorted(names)}"
        outs = {o.name for o in sess.get_outputs()}
        assert outs == {"start_logits", "end_logits"}
        feed = {"input_ids": np.ones((1, 8), dtype=np.int64),
                "attention_mask": np.ones((1, 8), dtype=np.int64),
                "field_mask": np.zeros((1, 1, 8), dtype=np.float32)}
        sl, el = sess.run(["start_logits", "end_logits"], feed)
        assert sl.shape == (1, 1, 8) and el.shape == (1, 1, 8)


@pytest.mark.skipif(not (SPIKE_DIR / "spike-verdict-s42.json").is_file(),
                    reason="no s42 verdict recorded (spike not yet run or "
                           "artifacts cleaned)")
def test_verdict_json_names_bytes_and_bars():
    v = json.loads((SPIKE_DIR / "spike-verdict-s42.json").read_text())
    assert v["arm"] == "unified-span-spike"
    for key, cand in v["candidates"].items():
        assert len(cand["int8_sha256"]) == 64
        for bar in ("S1_exact_fill", "S2_invented_rate",
                    "S3_multiword_exact", "S4_artifact",
                    "S5_enum_violations"):
            assert bar in cand["bars"]
        assert isinstance(cand["bars"]["PASS"], bool)


# --- HYBRID DECODE INCREMENT (2026-10-10, decode-E) --------------------------
# Registered in unified-span-extractor BEFORE any decode-E
# measurement: word-overlap attribution (never strict containment) + glue-only
# end-boundary snapping (extend floor 0.15, shipped span_fill parity;
# whitespace is a hard snapping limit BOTH ways). decode-D must stay
# byte-identical — the same fixtures pin its truncation for the A/B delta.

def _hybrid_glue_fixture():
    """`prod-eu-1` fragmented into subword tokens; the head's end argmax on
    the LAST '-' (the measured truncation texture) while every word of the
    glued run still carries strong end evidence (the probe-measured
    saturation: the region evidence survives, the argmax boundary is what
    lands on the artifact)."""
    req = "please restart the worker-queue service on host prod-eu-1"
    offs = [(0, 6), (7, 14), (15, 18), (19, 25), (25, 26), (26, 31),
            (32, 39), (40, 42), (43, 47), (48, 52), (52, 53), (53, 55),
            (55, 56), (56, 57)]
    enc = {"ids": [0] * len(offs), "mask": [1] * len(offs),
           "offsets": list(offs), "schema_pos": [],
           "req_pos": list(range(len(offs)))}
    sl = np.full((1, len(offs)), 3.0, dtype=np.float32)
    el = np.full((1, len(offs)), 4.0, dtype=np.float32)
    sl[0, 9] = 6.0            # start argmax on 'prod'
    el[0, 12] = 5.0           # end argmax on the LAST '-'  -> 'prod-eu-'
    return req, enc, (sl, el)


def test_decode_e_recovers_glue_truncated_value_decode_d_unchanged():
    req, enc, logits = _hybrid_glue_fixture()
    d = _decode()
    # decode-D: the registered parity pin — still truncates on the artifact
    filled_d, _, _ = d.fill(req, ["host"], [{"type": "string"}], enc,
                            [[0]], logits, 0.3, "D")
    assert filled_d == {"host": "prod-eu-"}
    # decode-E: overlap attribution + glue snapping decodes the FULL value
    filled_e, missing_e, _ = d.fill(req, ["host"], [{"type": "string"}],
                                    enc, [[0]], logits, 0.3, "E")
    assert filled_e == {"host": "prod-eu-1"}
    assert missing_e == []


def test_decode_e_overlap_attribution_repairs_midword_end():
    req = "rotate the api key for alerts@svc.mivia.example.com in staging"
    offs = [(0, 6), (7, 10), (11, 14), (15, 18), (19, 22), (23, 29),
            (29, 30), (30, 32), (32, 33), (33, 34), (34, 35), (35, 39),
            (39, 40), (40, 42), (43, 45), (46, 53)]
    enc = {"ids": [0] * len(offs), "mask": [1] * len(offs),
           "offsets": list(offs), "schema_pos": [],
           "req_pos": list(range(len(offs)))}
    sl = np.full((1, len(offs)), 2.0, dtype=np.float32)
    el = np.full((1, len(offs)), 4.0, dtype=np.float32)
    sl[0, 5] = 6.0            # start argmax on 'alerts'
    el[0, 10] = 5.0           # end argmax on 'm' INSIDE the word 'mivia'
    d = _decode()
    filled_d, _, _ = d.fill(req, ["account"], [{"type": "string"}], enc,
                            [[0]], logits := (sl, el), 0.3, "D")
    assert filled_d == {"account": "alerts@svc.m"}      # the measured texture
    filled_e, _, _ = d.fill(req, ["account"], [{"type": "string"}], enc,
                            [[0]], logits, 0.3, "E")
    # 'm' attributes to the whole word 'mivia' by overlap, then the glue snap
    # crosses '.' while the run's end evidence holds: the full email.
    assert filled_e == {"account": "alerts@svc.mivia.example.com"}


def test_decode_e_whitespace_is_hard_limit_for_snapping():
    req = "migrate web-01 staging now please"
    offs = [(0, 7), (8, 11), (11, 12), (12, 14), (15, 22), (23, 26),
            (27, 33)]
    enc = {"ids": [0] * len(offs), "mask": [1] * len(offs),
           "offsets": list(offs), "schema_pos": [],
           "req_pos": list(range(len(offs)))}
    sl = np.full((1, len(offs)), 2.0, dtype=np.float32)
    el = np.full((1, len(offs)), 2.0, dtype=np.float32)
    sl[0, 1] = 6.0            # start argmax on 'web'
    el[0, 4] = 6.0            # end argmax on 'staging' (multi-word seed)
    el[0, 5] = 5.0            # decoy: HIGH end evidence on 'now'
    d = _decode()
    filled, missing, _ = d.fill(req, ["target"], [{"type": "string"}], enc,
                                [[0]], (sl, el), 0.3, "E")
    # the seed's own boundary evidence crosses whitespace (multi-word native,
    # preserved), but snapping NEVER follows: 'now' is whitespace-separated
    assert filled == {"target": "web-01 staging"}
    assert missing == []


def test_decode_e_snap_never_absorbs_whitespace_neighbor_from_short_seed():
    req = "migrate web-01 staging now please"
    offs = [(0, 7), (8, 11), (11, 12), (12, 14), (15, 22), (23, 26),
            (27, 33)]
    enc = {"ids": [0] * len(offs), "mask": [1] * len(offs),
           "offsets": list(offs), "schema_pos": [],
           "req_pos": list(range(len(offs)))}
    sl = np.full((1, len(offs)), 2.0, dtype=np.float32)
    el = np.full((1, len(offs)), 2.0, dtype=np.float32)
    sl[0, 1] = 6.0            # start argmax on 'web'
    el[0, 3] = 6.0            # seed end on '01' (single-word seed)
    el[0, 5] = 5.0            # high end evidence on 'now', ACROSS whitespace
    d = _decode()
    filled, _, _ = d.fill(req, ["target"], [{"type": "string"}], enc,
                          [[0]], (sl, el), 0.3, "E")
    # the head's own boundary evidence is the ONLY whitespace-crossing
    # authority: a short seed stays short even when a whitespace neighbor
    # carries stronger end evidence (the registered honesty trade)
    assert filled == {"target": "web-01"}


def test_decode_e_enum_extension_coerces_to_abstain_never_fabricates():
    req = "deploy the api to production-ready now"
    offs = [(0, 6), (7, 10), (11, 14), (15, 17), (18, 28), (28, 29),
            (29, 34), (35, 38)]
    enc = {"ids": [0] * len(offs), "mask": [1] * len(offs),
           "offsets": list(offs), "schema_pos": [],
           "req_pos": list(range(len(offs)))}
    sl = np.full((1, len(offs)), 2.0, dtype=np.float32)
    el = np.full((1, len(offs)), 2.0, dtype=np.float32)
    sl[0, 4] = 6.0            # start argmax on 'production'
    el[0, 6] = 6.0            # end argmax on 'ready' (glued suffix)
    el[0, 4] = 5.0
    d = _decode()
    for variant in ("D", "E"):
        filled, missing, _ = d.fill(req, ["environment"],
                                    [{"type": "string",
                                      "enum": ["production"]}],
                                    enc, [[0]], (sl, el), 0.3, variant)
        # 'production-ready' is not a declared member: abstain over fabricate
        # (S5 stays structural under decode-E's snap)
        assert filled == {}, variant
        assert missing == ["environment"], variant


def test_decode_e_below_emit_floor_abstains_before_snapping():
    req, enc, _ = _hybrid_glue_fixture()
    offs = enc["offsets"]
    sl = np.full((1, len(offs)), -8.0, dtype=np.float32)
    el = np.full((1, len(offs)), -8.0, dtype=np.float32)
    sl[0, 9] = -2.0           # sigmoid ~ 0.119: seed score below any floor
    d = _decode()
    filled, missing, _ = d.fill(req, ["host"], [{"type": "string"}], enc,
                                [[0]], (sl, el), 0.5, "E")
    assert filled == {}       # no extension can rescue a below-floor seed
    assert missing == ["host"]


def test_decode_e_extend_floor_is_shipped_spanfill_parity():
    from mivia_small_json import span_fill as _sf

    assert T.EXTEND_FLOOR == _sf._SPANFILL_EXTEND_FLOOR == 0.15


def test_multiword_native_capability_survives_decode_e():
    # the S3 texture: a whitespace value decoded verbatim by BOTH variants
    req = "restart nginx on web-01 staging now"
    offs = [(0, 7), (8, 13), (14, 16), (17, 23), (24, 31), (32, 35)]
    enc = {"ids": [0] * len(offs), "mask": [1] * len(offs),
           "offsets": list(offs), "schema_pos": [],
           "req_pos": list(range(len(offs)))}
    logits = _logits(1, {0: (3, 4)}, T_len=len(offs))
    d = _decode()
    filled, _, _ = d.fill(req, ["target"], [{"type": "string"}], enc,
                          [[0]], logits, 0.5, "E")
    assert filled == {"target": "web-01 staging"}
    filled_d, _, _ = d.fill(req, ["target"], [{"type": "string"}], enc,
                            [[0]], logits, 0.5, "D")
    assert filled_d == {"target": "web-01 staging"}


@pytest.mark.skipif(not (SPIKE_DIR / "hybrid-verdict.json").is_file(),
                    reason="no hybrid-verdict.json recorded (hybrid "
                           "measurement not yet run; the contract gates "
                           "above always run)")
def test_hybrid_verdict_json_names_bytes_and_bars():
    v = json.loads((SPIKE_DIR / "hybrid-verdict.json").read_text())
    assert v["arm"] == "unified-span-spike"
    assert "HYBRID DECODE INCREMENT" in v["increment"]
    assert v["verdict"] in ("HYBRID PASSES", "SECOND FALSIFICATION")
    assert v["corpus_sha256"]["dev"] == T.FROZEN_SHA["dev"]
    assert v["corpus_sha256"]["train"] == T.FROZEN_SHA["train"]
    for seed_key, cand in v["seeds"].items():
        assert len(cand["int8_sha256"]) == 64
        assert len(cand["tokenizer_sha256"]) == 64
        for variant in ("D", "E"):
            bars = cand[f"bars_{variant}"]
            for bar in ("S1_exact_fill", "S2_invented_rate",
                        "S3_multiword_exact", "S4_artifact",
                        "S5_enum_violations"):
                assert bar in bars
            assert isinstance(bars["PASS"], bool)
