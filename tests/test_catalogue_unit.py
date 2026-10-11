"""Catalogue-as-unit arm: rendering, set-head features, merge-rule counters."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exp_catalogue_unit as ecu  # noqa: E402

MT_CACHE = [Path(f"/tmp/bfcl_v4_multi_turn_{s}.json") for s in
            ("base", "miss_func", "miss_param")]
needs_mt_cache = pytest.mark.skipif(
    not all(p.is_file() for p in MT_CACHE),
    reason="needs cached BFCL multi-turn dumps")


def _tool(name, desc="does things"):
    return {"name": name, "description": desc,
            "parameters": {"properties": {}, "required": []}}


def test_render_catalogue_joins_and_caps():
    tools = [_tool(f"t{i}", "d" * 300) for i in range(8)]
    text = ecu.render_catalogue(tools, "req", mode="v1")
    assert len(text.split(" || ")) == ecu.MAX_TOOLS
    assert all(len(span) <= ecu.PER_TOOL_CHARS
               for span in text.split(" || "))


def test_render_catalogue_empty():
    assert ecu.render_catalogue([], "req") == ""


def test_sethead_features_known_values():
    feats = ecu.sethead_features([0.9, 0.7, 0.5])
    n, mx, mean, mn, _std, margin, f9, f5 = feats
    assert (n, mx, mn, margin) == (3, 0.9, 0.5, pytest.approx(0.2))
    assert mean == pytest.approx(0.7)
    assert f9 == pytest.approx(1 / 3)
    assert f5 == pytest.approx(1.0)
    assert len(feats) == len(ecu.SETHEAD_FEATURES)


def test_mcnemar_edges():
    assert ecu.mcnemar_exact(0, 0) == 1.0
    assert ecu.mcnemar_exact(5, 5) == 1.0
    assert ecu.mcnemar_exact(0, 10) < 0.01
    assert ecu.mcnemar_exact(10, 0) == ecu.mcnemar_exact(0, 10)


def test_split_is_deterministic():
    assert ecu._split_of("hello") == ecu._split_of("hello")
    assert ecu._split_of("hello") in ("train", "dev")


@needs_mt_cache
def test_mt_corpus_matches_welldefinedness_note():
    rows, stats = ecu.build_mt_rows()
    assert stats["mt_merged_rows_in"] == 66489
    assert stats["mt_merged_rows_kept"] == 62885
    assert stats["mt_merged_rows_dropped_conflict"] == 3604
    assert stats["mt_merged_pairs_conflicting"] == 1345
    assert stats["cat_positive"] + stats["cat_negative"] == len(rows)
    assert {r["label"] for r in rows} <= {0, 1}


@needs_mt_cache
def test_mt_corpus_has_zero_guard_eval_v2_request_overlap():
    rows, _stats = ecu.build_mt_rows()
    corpus_reqs = {r["request"] for r in rows}
    gev2 = ROOT / "data" / "incoming" / "external" / "bfcl" / "guard-eval-v2.jsonl"
    eval_reqs = {json.loads(line).get("request") or ""
                 for line in gev2.read_text(encoding="utf-8").split("\n")
                 if line.strip()}
    assert not (corpus_reqs & eval_reqs)
