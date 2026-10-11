"""TRACK 4: ask_followup residual new-feature research.

- test_feature_helpers pins the F5 entailment feature arithmetic on a toy
  catalogue (no guard, no schemas, milliseconds).
- test_family_separation pins the measured separation: desc_cov
  max(AUC, 1-AUC) >= 0.75 on residual-118 vs relevance keeps.
- test_bar_not_cleared is a FALSIFICATION RECORD: the pre-registered bar
  (ask drop >= 10pp, retention flat, no > 1pp regression, McNemar p < 0.05)
  was measured 2026-10-06 at -5.5pp, so the 1-feature head does not clear
  it. If a future head clears the bar, this test FAILS on purpose — update
  reports/exp-askfeat.md, do not weaken the test.

Metrics come from gitignored data/incoming/models/askfeat-metrics.json
(regenerate with scripts/askfeat_research.py). No schema registration here.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "data" / "incoming" / "models" / "askfeat-metrics.json"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "askfeat_research", ROOT / "scripts" / "askfeat_research.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_feature_helpers():
    mod = _load_script()
    tool = {
        "name": "Weather_1_GetWeather",
        "description": "Retrieves the weather forecast for a specified city.",
        "parameters": {"properties": {"city": {"description": "city name"}},
                       "required": ["city"]},
    }
    cats = mod.catalogue_table(
        [{"category": "ask_followup", "source": "s1",
          "request": "What is the weather like?", "tool": tool}],
        {"0": 0.999},
    )
    c = cats["s1"]
    assert c["n_tools"] == 1
    assert c["margin"] == 0.0  # singleton: top1 == top2
    assert 0.0 <= c["desc_cov"] <= 1.0
    assert 0.0 <= c["jaccard"] <= 1.0
    # "weather" overlaps the description: nonzero coverage both ways.
    assert c["desc_cov"] > 0.0
    assert c["req_cov"] > 0.0
    # Empty tool text degrades gracefully, never NaN/div-zero.
    c2 = mod.catalogue_table(
        [{"category": "ask_followup", "source": "s2",
          "request": "hi?", "tool": {"description": ""}}],
        {"0": 0.5},
    )["s2"]
    assert c2["desc_cov"] == 0.0


def test_auc_helper_direction():
    mod = _load_script()
    assert mod.auc_for([3.0, 2.0, 1.0], [1, 1, 0]) > 0.5
    assert mod.auc_for([1.0, 2.0, 3.0], [1, 1, 0]) < 0.5


def _metrics():
    if not METRICS.is_file():
        # Gitignored research output (clean clones never carry it): skip,
        # do not fail — the falsification record is research evidence,
        # not shipped behavior. On a machine that ran the research the
        # file exists and the record still pins.
        pytest.skip(
            f"missing {METRICS} (gitignored candidate output); regenerate "
            "with scripts/askfeat_research.py to re-verify this record")
    return json.loads(METRICS.read_text())


def test_residual_reproduced():
    m = _metrics()
    assert m["residual"]["n_catalogues"] == 473
    assert m["residual"]["n_residual"] == 118
    assert m["n_residual"] == 118
    assert m["n_keeps"] == 174


def test_family_separation():
    """F5 desc_cov clears the 0.75 diagnosis bar; F1/F2-all/F4 do not."""
    m = _metrics()
    fams = m["families"]
    assert fams["F5_entail"]["best_sep"] >= 0.75
    assert fams["F1_size"]["best_sep"] < 0.75
    assert fams["F2_margin"]["best_sep"] < 0.75
    assert fams["F4_fillgap"]["best_sep"] < 0.75


def test_bar_not_cleared():
    """Falsification record: the desc_cov head misses the 10pp bar (-5.5pp)."""
    m = _metrics()
    for key in ("seed_21", "seed_22"):
        ev = m[key]
        ask = ev["table"]["ask_followup"]
        rel = ev["table"]["single_turn_relevance"]
        drop = ask["call_before"] - ask["call_after"]
        worst_other = max(
            v["call_after"] - v["call_before"]
            for k, v in ev["table"].items() if k != "ask_followup"
        )
        assert drop < 10.0, (
            f"{key}: ask drop {drop:.1f}pp now clears the bar — "
            "update reports/exp-askfeat.md")
        assert rel["call_after"] == rel["call_before"], key
        assert worst_other <= 1.0, key
        assert ev["mcnemar"]["p"] < 0.05, key


def test_seed_spread_zero():
    m = _metrics()
    a = m["seed_21"]["table"]["ask_followup"]
    b = m["seed_22"]["table"]["ask_followup"]
    assert a == b
