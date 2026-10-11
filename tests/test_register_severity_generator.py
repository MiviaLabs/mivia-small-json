"""Tests for the register severity generator and grade corpus builder.

The generator and builder are the register-expansion machinery
(grade-data-expansion). The properties that matter:
deterministic output under the seed, template-level splits that never
split a family across train/dev, and the builder's eval-collision check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from bench_zero_shot_primitives import NOVEL_REGISTER_GRADE_CASES  # noqa: E402
from register_severity_generator import (  # noqa: E402
    FAMILIES,
    REGISTERS,
    build_rows,
    template_split,
)


def test_generator_is_deterministic_under_seed():
    assert build_rows(42) == build_rows(42)
    assert build_rows(42) != build_rows(43)


def test_generator_covers_all_registers_and_tiers():
    rows = build_rows(42)
    assert {r["register"] for r in rows} == set(REGISTERS)
    for register in REGISTERS:
        tiers = {r["gold"] for r in rows if r["register"] == register}
        assert tiers == {0, 1, 2, 3}, register


def test_generator_row_count_matches_families():
    rows = build_rows(42, per_family=7)
    assert len(rows) == 7 * len(FAMILIES)


def test_template_split_never_splits_a_family():
    train, dev = template_split(build_rows(42))
    train_fams = {r["template"] for r in train}
    dev_fams = {r["template"] for r in dev}
    assert not train_fams & dev_fams
    assert train_fams | dev_fams == {f[0] for f in FAMILIES}


def test_builder_rejects_frozen_eval_rows_in_corpus():
    """A frozen novel-register eval row inside the corpus must be fatal."""
    import build_grade_corpus as builder

    frozen_text, _tier = NOVEL_REGISTER_GRADE_CASES[0]
    rows = [{"text": frozen_text, "family": "severity", "gold": 0}]
    with pytest.raises(SystemExit):
        builder.eval_collision_check(rows)


def test_builder_accepts_clean_corpus():
    import build_grade_corpus as builder

    rows = [{"text": f"ordinary training row {i}", "family": "severity",
             "gold": i % 4} for i in range(10)]
    builder.eval_collision_check(rows)
