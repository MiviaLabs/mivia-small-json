"""QW-A/QW-B (quickwins-serving spec): repair-ladder corpus bar, prefixed-id
ladder, prose-title override guard. Rules-layer tests only — no model bytes
involved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mivia_small_json import pipeline, rules
from mivia_small_json.pipeline import _learned_title_wins
from mivia_small_json.tool_call import repair_arguments

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "data" / "malformed-json-corpus.jsonl"


class TestRepairCorpusBar:
    """The pre-registered QW-A bar, enforced by a deterministic test:
    collapse rate < 10% on the committed 30-case corruption corpus, every
    output a JSON dict, per-case expected keys where recovery is required."""

    @pytest.mark.parametrize(
        "row", [json.loads(line) for line in
                CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()],
        ids=lambda r: r["case"],
    )
    def test_corpus_row(self, row):
        out = repair_arguments(row["raw"])
        parsed = json.loads(out)  # every output must parse
        assert isinstance(parsed, dict)
        if row["expect_mode"] == "keys":
            assert out != "{}" or not row["expect_keys"]
            assert set(row["expect_keys"]) <= set(parsed), (row["case"], out)

    def test_collapse_budget(self):
        rows = [json.loads(line) for line in
                CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
        collapse = sum(
            1 for r in rows if repair_arguments(r["raw"]) == "{}")
        assert collapse / len(rows) < 0.10, (collapse, len(rows))


class TestPrefixedIdLadder:
    def test_generic_project_ids(self):
        for raw, want in [
            ("PAY-2214: Checkout fails for 3DS cards in EU", "PAY-2214"),
            ("GH-4711 login broken on safari", "GH-4711"),
            ("see ABC123-99 for context", "ABC123-99"),
            ("ISS-4091: pool exhaustion", "ISS-4091"),  # ISS- ladder first
            ("fixed by #12345 yesterday", "ISS-12345"),  # #-ladder keeps name
        ]:
            obj = rules.extract(raw, "issue")
            assert obj["id"] == want, (raw, obj["id"])

    def test_no_false_ids(self):
        for raw in [
            "Cannot login since version 42-12 shipped",   # not PREFIX-nnn
            "the diff touched a/b-123.txt",                # path segment
            "plain prose without any identifiers",
            # technical tokens that match the shape (review P1-3)
            "Update SHA-256 hashing to UTF-16 output",
            "Fix CVE-2024-31337 in the parser",
            "UTF-16 encoded exports (ISO-8859 fallback)",
            "JWT-256 tokens rejected",
            "USB-320 hub reset during boot",
        ]:
            obj = rules.extract(raw, "issue")
            assert obj["id"] == "", (raw, obj["id"])

    def test_tech_token_stoplist_lists_each_token_once(self):
        """USB was listed twice. Membership still worked; the stoplist
        itself must not carry a repeated entry."""
        tokens = rules._ID_TECH_TOKENS
        assert len(tokens) == len(set(tokens))
        assert "USB" in tokens

    def test_real_id_wins_over_technical_noise(self):
        # first-match-wins must skip technical tokens and find the ticket id
        obj = rules.extract(
            "Fix CVE-2024-31337 found by SHA-256 audit; tracked as PAY-2214",
            "issue")
        assert obj["id"] == "PAY-2214"


class TestLearnedTitleGuard:
    def test_truncations_lose(self):
        # the greedy-span collapses seen in the serving probe
        assert _learned_title_wins(
            "PAY-2214: Checkout fails for 3DS cards in EU", "PAY") is False
        assert _learned_title_wins(
            "Connection pool exhaustion during batch sync", "Connection") is False
        # rules strips the id from its title; the raw-first-line reference
        # must still catch the truncation
        assert _learned_title_wins(
            "Connection pool exhaustion during batch sync", "ISS",
            "ISS-4091: Connection pool exhaustion during batch sync") is False

    def test_degenerate_rules_titles_lose(self):
        assert _learned_title_wins("", "Real title from prose") is True
        assert _learned_title_wins("PAY-2214", "Checkout fails for EU users") is True
        assert _learned_title_wins("!!!", "Checkout fails for EU users") is True

    def test_substantive_different_titles(self):
        # a learned capture that is NOT a truncation keeps the Arm A
        # behavior (the learned extractor is the improvement; the guard
        # only blocks the greedy-span collapse shapes)
        assert _learned_title_wins(
            "App crashes on login since update", "Login screen crash") is True

    def test_pipeline_end_to_end(self):
        obj, meta = pipeline.classify(
            "PAY-2214: Checkout fails for 3DS cards in EU\nPriority: high\n"
            "Labels: payments\nState: open")
        assert obj["type"] == "issue"
        assert obj["id"] == "PAY-2214"
        # the greedy-span collapse shape must never become the title
        # (asserting the defect's absence, not the current winner — the
        # learned extractor may legitimately improve past the rules title)
        assert obj["title"] != "PAY"
        assert len(obj["title"]) > 10
