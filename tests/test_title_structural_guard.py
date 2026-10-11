"""Structural leading-line title guard (2026-10-10, red-first).

The ticket (EN/DE) and gh-prs extraction surfaces present the instrument
shape ``subject<THB-SEP>body`` — the gold title IS the leading subject
line — yet the chain still lets body sentences through: the rules title
is the LONGEST chunk (usually a body sentence) and the learned span
decoder truncates the subject or picks a body sentence, which the QW-B
prefix guard then cannot always veto. Reality-extract on the shipped
bytes: title-exact EN 72.5 / DE 66.4 / gh-prs 23.0 while the bare
leading-line rule scores 91.7 / 94.6 / ~95 (German lags only by
construction and inverts under the rule — the owner's multilingual MUST).

The guard: when the text is one short leading line, a blank-line
separator (\\n\\n, \\r\\n\\r\\n, whitespace-only lines), then prose, the
title is the leading line. The learned extractor stays authoritative
for genuinely line-free prose and for an implausibly long leading line.
"""

from __future__ import annotations

import pytest

from mivia_small_json.pipeline import (
    _classify_known_schema,
    _structural_leading_line_title,
)

DE_SUBJECT = ("Probleme bei der Aktualisierung des Dashboards "
              "für digitales Marketing-Analyse")
DE_BODY = ("Das digitale Marketing-Analyse-Dashboard wurde korrekt "
           "aktualisiert. Obwohl das Cache-Clearing und das "
           "Wiederherstellen der Anwendung durchgeführt wurde, behält "
           "das Problem seinen Verlauf bei.")

EN_SUBJECT = "Unable to connect to the payment gateway since this morning"
EN_BODY = ("Our checkout is completely down for every customer in the EU "
           "region. Requests to the gateway time out after thirty seconds "
           "and the order is left pending until the maintenance window.")

PR_SUBJECT = "Fix segfault in worker pool shutdown"
PR_BODY = ("Stopping the worker pool under load segfaults the supervisor "
           "process and leaves orphaned connections open.")


class PhraseTitleStub:
    """Learned-extractor stub whose title span is ALWAYS the given
    phrase, wherever the phrase occurs in the text — the body-sentence
    failure mode of the real span decoder, under test control."""

    classes_ = [0, 1]

    def __init__(self, phrase: str):
        self.words = set(phrase.lower().split())

    def predict_proba(self, feats):
        out = []
        for row in feats:
            tok = next(f[4:] for f in row if f.startswith("tok="))
            out.append([0.1, 0.9] if tok in self.words else [0.9, 0.1])
        return out


def _classify(schema, text, prose_extractor=None):
    obj, _meta = _classify_known_schema(
        text, schema, "test", prose_extractor=prose_extractor)
    return obj


# --- the diagnostic shapes (real shipped prose extractor) -------------------


class TestDiagnosticShapes:
    def test_de_body_first_pick_must_not_happen(self):
        """The exact failing shape from the 2026-10-10 diagnostic: the
        gold title is the leading subject line; the chain must not pick
        a body sentence."""
        obj = _classify("issue", f"{DE_SUBJECT}\n\n{DE_BODY}")
        assert obj["title"] == DE_SUBJECT

    def test_en_subject_line_wins(self):
        obj = _classify("issue", f"{EN_SUBJECT}\n\n{EN_BODY}")
        assert obj["title"] == EN_SUBJECT

    def test_gh_prs_shape_leading_line_wins(self):
        """gh-prs rows are ``title\\n\\nbody`` with the PR title gold —
        same instrument, same rule."""
        obj = _classify("pull_request", f"{PR_SUBJECT}\n\n{PR_BODY}")
        assert obj["title"] == PR_SUBJECT

    def test_body_field_still_extracted(self):
        obj = _classify("issue", f"{EN_SUBJECT}\n\n{EN_BODY}")
        assert obj.get("body", "").strip()


# --- precedence under a controlled learned extractor ------------------------


class TestStructuralPrecedence:
    def test_structural_beats_body_first_learned_pick(self):
        """Even when the learned extractor picks a body phrase, the
        structural shape carries the title; the learned body still
        applies (title-only override)."""
        stub = PhraseTitleStub("maintenance window")
        obj = _classify("issue", f"{EN_SUBJECT}\n\n{EN_BODY}",
                        prose_extractor=stub)
        assert obj["title"] == EN_SUBJECT
        # the learned body still applies (rules extraction for issue
        # never emits a body field — any body here is the learned one;
        # the stub's title span itself is absent from it)
        assert "EU region" in obj.get("body", "")
        assert "maintenance window" not in obj.get("body", "")

    def test_single_block_prose_keeps_learned_extractor(self):
        """Line-free prose (no blank-line separator) is untouched: the
        learned extractor stays authoritative."""
        stub = PhraseTitleStub("maintenance window")
        obj = _classify("issue", EN_BODY, prose_extractor=stub)
        assert obj["title"] == "maintenance window"

    def test_single_newline_is_not_a_separator(self):
        """A soft wrap (single \\n) is not the blank-line separator:
        the learned extractor keeps the title."""
        stub = PhraseTitleStub("maintenance window")
        obj = _classify("issue", f"{EN_SUBJECT}\n{EN_BODY}",
                        prose_extractor=stub)
        assert obj["title"] == "maintenance window"


# --- separator forms and boundaries -----------------------------------------


class TestSeparatorForms:
    @pytest.mark.parametrize("sep", ["\n\n", "\r\n\r\n", "\n \n", "\n\n\n",
                                     "\r\n\n", "\n\r\n"])
    def test_blank_line_separators_fire(self, sep):
        stub = PhraseTitleStub("maintenance window")
        obj = _classify("issue", EN_SUBJECT + sep + EN_BODY,
                        prose_extractor=stub)
        assert obj["title"] == EN_SUBJECT, sep

    def test_leading_blank_lines_are_skipped(self):
        stub = PhraseTitleStub("maintenance window")
        obj = _classify("issue", "\n\n" + EN_SUBJECT + "\n\n" + EN_BODY,
                        prose_extractor=stub)
        assert obj["title"] == EN_SUBJECT

    def test_whitespace_only_lead_is_skipped(self):
        stub = PhraseTitleStub("maintenance window")
        text = "  \n\t\n" + EN_SUBJECT + "\n\n" + EN_BODY
        obj = _classify("issue", text, prose_extractor=stub)
        assert obj["title"] == EN_SUBJECT

    def test_title_is_trimmed(self):
        stub = PhraseTitleStub("maintenance window")
        text = "  " + EN_SUBJECT + "  \n\n" + EN_BODY
        obj = _classify("issue", text, prose_extractor=stub)
        assert obj["title"] == EN_SUBJECT

    def test_no_prose_after_separator_does_not_fire(self):
        assert _structural_leading_line_title(f"{EN_SUBJECT}\n\n") is None

    def test_single_line_does_not_fire(self):
        assert _structural_leading_line_title(EN_SUBJECT) is None

    def test_empty_text_does_not_fire(self):
        assert _structural_leading_line_title("") is None


class TestLengthBound:
    """The leading line must be a plausible title: at most 200 chars
    (registered bound in pipeline._LEADING_LINE_TITLE_MAX_CHARS)."""

    @pytest.mark.parametrize("n", [200])
    def test_at_bound_fires(self, n):
        line = ("Boundary " + "x" * (n - len("Boundary ")))[:n]
        assert len(line) == n
        assert _structural_leading_line_title(
            line + "\n\n" + EN_BODY) == line

    @pytest.mark.parametrize("n", [201, 400])
    def test_over_bound_falls_back_to_learned(self, n):
        line = ("Boundary " + "x" * (n - len("Boundary ")))[:n]
        stub = PhraseTitleStub("maintenance window")
        obj = _classify("issue", line + "\n\n" + EN_BODY,
                        prose_extractor=stub)
        assert obj["title"] == "maintenance window"


class TestPlausibility:
    @pytest.mark.parametrize("lead", ["!!!", "12345", "---"])
    def test_letterless_leading_line_does_not_fire(self, lead):
        stub = PhraseTitleStub("maintenance window")
        obj = _classify("issue", f"{lead}\n\n{EN_BODY}",
                        prose_extractor=stub)
        assert obj["title"] == "maintenance window"

    def test_dash_and_colon_edges_trimmed(self):
        assert _structural_leading_line_title(
            "— Payment gateway down\n\n" + EN_BODY) == "Payment gateway down"
        assert _structural_leading_line_title(
            "Payment gateway down:\n\n" + EN_BODY) == "Payment gateway down"
