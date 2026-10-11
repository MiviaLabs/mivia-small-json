"""Severity taxonomies: the deterministic crosswalk to canonical tiers.

FEATURE (2026-10-10): ten production severity vocabularies (PagerDuty,
Atlassian P1-P4, Bugzilla, Jira priorities, GitHub label conventions,
syslog RFC 5424, OTel SeverityNumber, Log4j, CVSS, ITIL) mapped onto
the canonical 4-tier scale the trained grade head measures
(``severity_head.CANONICAL_SEVERITY_ANCHORS``). Data-only, stdlib-only:
no model bytes, no network, no training.

Polarity is verified FROM THE CODE, not the task description: the
model's ``cumulative_tier``/``tier_index`` is 0-based (0 = info ...
3 = critical); the crosswalk speaks the human-facing 1..4 convention
(tier 4 = most severe), so ``tier = tier_index + 1``. Per-taxonomy
numeric polarity is the point of a crosswalk: PagerDuty counts DOWN
(SEV-1 worst), syslog counts UP (0 worst), OTel counts UP (24 worst)
— every external placement below was verified against the primary
source on 2026-10-10 (PagerDuty incident-response docs;
OTel logs data-model Appendix B; Atlassian incident-severity guidance).

The alias packs compose with the enum-aliases feature (5eccd69): they
are generated FROM these tables (so they cannot drift), register under
the registry's collision law, and coerce real tokens to canonical
members end-to-end. The committed docs/USAGE.md §7.2 block is guarded
here byte-for-byte against drift from the module.
"""

import re
from pathlib import Path

import pytest

from mivia_small_json import pipeline, registry
from mivia_small_json.severity_taxonomies import (
    TIER_NAMES,
    TAXONOMIES,
    USAGE_BEGIN_MARK,
    USAGE_END_MARK,
    alias_pack,
    canonical_anchor,
    merge_packs,
    normalize_token,
    render_usage_markdown,
    resolve,
    resolve_note,
    taxonomy_names,
)
from mivia_small_json.severity_head import CANONICAL_SEVERITY_ANCHORS

REPO_ROOT = Path(__file__).resolve().parents[1]
USAGE_PATH = REPO_ROOT / "docs" / "USAGE.md"
MODEL_STATE_PATH = REPO_ROOT / "docs" / "MODEL-STATE.md"


# --- polarity: the inverted ops vocabularies ---------------------------------

def test_polarity_pagerduty_inverted_lower_is_worse():
    # PagerDuty counts DOWNWARD (verified: response.pagerduty.com
    # "Severity Levels" — "lower numbered severities being more urgent").
    assert resolve("pagerduty", "SEV-1") == 4  # critical state, data-exposing vuln
    assert resolve("pagerduty", "SEV-5") == 1  # cosmetic
    assert resolve("pagerduty", "sev3") == 3   # partial loss of functionality
    assert resolve("pagerduty", "SEV-4") == 2  # minor, no customer-facing loss
    assert resolve("pagerduty", "SEV-0") == 4  # informal "worse than SEV-1"


def test_polarity_atlassian_p_inverted():
    assert resolve("atlassian_p", "P1") == 4   # critical incident, very high impact
    assert resolve("atlassian_p", "P2") == 3   # major incident, significant impact
    assert resolve("atlassian_p", "P3") == 2   # minor incident, low impact
    assert resolve("atlassian_p", "P4") == 1   # minor usability / cosmetic


def test_polarity_cvss_word_scale():
    # CVSS words order none < low < medium < high < critical; the vuln-
    # management convention (Critical->P1 ... Low->P4) pins the collapse.
    assert resolve("cvss", "critical") == 4
    assert resolve("cvss", "high") == 3
    assert resolve("cvss", "medium") == 2
    assert resolve("cvss", "low") == 1
    assert resolve("cvss", "none") == 1


def test_polarity_syslog_verified_against_otel_appendix_b():
    # RFC 5424: numeric 0 is the WORST. Verified level-by-level against
    # the OTel logs data-model Appendix B table (2026-10-10):
    #   Emergency(0)->FATAL(21), Alert(1)->ERROR3(19),
    #   Critical(2)->ERROR2(18), Error(3)->ERROR(17), Warning(4)->WARN(13),
    #   Notice(5)->INFO2(10), Informational(6)->INFO(9), Debug(7)->DEBUG(5)
    # -> canonical tiers 4,3,3,3,2,1,1,1.
    assert [resolve("syslog", d) for d in "01234567"] == [4, 3, 3, 3, 2, 1, 1, 1]
    assert resolve("syslog", "emergency") == 4
    assert resolve("syslog", "panic") == 4      # legacy syslog(3) name
    assert resolve("syslog", "debug") == 1


def test_syslog_critical_name_collision_is_deliberate():
    # Canonical tier 4 is NAMED "critical", but syslog's level-2
    # "critical conditions" is documented by OTel INSIDE the ERROR band
    # (SeverityNumber 18) — the crosswalk follows the documented mapping,
    # not the English word. The Emergency-vs-FATAL debate (RFC 5424
    # Emergency = "system is unusable", often emitted by a surviving
    # monitor, vs application-fatal = the emitter is dying) is settled
    # the same way: OTel's table puts Emergency alone in the FATAL band.
    assert resolve("syslog", "critical") == 3
    assert resolve("syslog", "alert") == 3      # ERROR3 (19), not FATAL
    note = resolve_note("syslog", "critical")
    assert note is not None and "NAME COLLISION" in note
    assert "Emergency-vs-FATAL" in resolve_note("syslog", "emergency")


def test_polarity_otel_severitynumber_higher_is_worse():
    # OTel counts UPWARD: SeverityNumber 24 (FATAL band) is the worst.
    assert resolve("otel", "24") == 4
    assert resolve("otel", "21") == 4
    assert resolve("otel", "17") == 3
    assert resolve("otel", "13") == 2
    assert resolve("otel", "12") == 1
    assert resolve("otel", "1") == 1
    for band, tier in (("TRACE", 1), ("DEBUG", 1), ("INFO", 1),
                       ("WARN", 2), ("WARNING", 2), ("ERROR", 3),
                       ("FATAL", 4)):
        assert resolve("otel", band) == tier, band
    assert resolve("otel", "0") is None    # 0 is not an OTel SeverityNumber
    assert resolve("otel", "25") is None   # out of range: honest None


def test_log4j_levels_and_off_is_not_a_severity():
    assert resolve("log4j", "FATAL") == 4
    assert resolve("log4j", "ERROR") == 3
    assert resolve("log4j", "WARN") == 2
    assert resolve("log4j", "INFO") == 1
    assert resolve("log4j", "DEBUG") == 1
    assert resolve("log4j", "TRACE") == 1
    # OFF/ALL are logger controls, not severities: deliberately unmapped.
    assert resolve("log4j", "OFF") is None
    assert resolve("log4j", "ALL") is None


# --- normalization, unknowns, table integrity --------------------------------

def test_prefix_stripping_and_case_insensitivity():
    for surface in ("SEV-1", "sev1", "Sev 1", "sev_1"):
        assert resolve("pagerduty", surface) == 4, surface
    for surface in ("P1", "p1", "P-1", "p 1"):
        assert resolve("atlassian_p", surface) == 4, surface
    assert resolve("github_labels", "severity:high") == 3
    assert resolve("github_labels", "priority:critical") == 4
    # Only numeric-prefixed forms strip: the word 'severe' is untouched.
    assert resolve("pagerduty", "severe") is None
    assert normalize_token("severe") == "severe"
    assert normalize_token("panic") == "panic"


def test_unknown_token_and_unknown_taxonomy_return_none():
    assert resolve("pagerduty", "bogus") is None
    assert resolve("pagerduty", "") is None
    assert resolve("no-such-taxonomy", "SEV-1") is None
    assert resolve_note("no-such-taxonomy", "SEV-1") is None


def test_every_documented_row_resolves_to_its_documented_tier():
    names = taxonomy_names()
    assert names == ("pagerduty", "atlassian_p", "bugzilla", "jira_priority",
                     "github_labels", "syslog", "otel", "log4j", "cvss",
                     "itil")
    assert set(names) == set(TAXONOMIES)
    for name in names:
        for row in TAXONOMIES[name].rows:
            assert 1 <= row.tier <= 4
            assert row.note, (name, row.tokens)  # every mapping names its source
            for token in row.tokens:
                assert resolve(name, token) == row.tier, (name, token)


def test_canonical_anchor_and_tier_index_relation():
    # tier (1-based here) indexes the anchors the trained head decodes;
    # the model's 0-based tier_index is tier - 1.
    for tier in (1, 2, 3, 4):
        assert canonical_anchor(tier) == CANONICAL_SEVERITY_ANCHORS[tier - 1]
    assert TIER_NAMES == ("info", "warning", "error", "critical")


def test_render_is_deterministic():
    assert render_usage_markdown() == render_usage_markdown()


# --- alias packs: composition with the enum-aliases feature ------------------

_SPEC = {
    "type": {"const": "probe"},
    "severity": {"type": str, "enum": list(TIER_NAMES)},
}


def test_alias_packs_register_and_coerce_for_every_taxonomy():
    for name in taxonomy_names():
        schema = f"probe_{name}"
        spec = dict(_SPEC, type={"const": schema})
        spec["severity"]["aliases"] = alias_pack(name)
        registry.register_schema(schema, spec,
                                 routing={"strong": [], "weak": []},
                                 extraction={"severity": [r"(.*)"]})
        pack = alias_pack(name)
        for member, surfaces in pack.items():
            tier = TIER_NAMES.index(member) + 1
            for surface in surfaces:
                got = registry.canonical_enum_value(schema, "severity",
                                                    surface)
                assert got == member, (name, surface, got)
                # the pack cannot disagree with the crosswalk table
                assert resolve(name, surface) == tier, (name, surface)
                # casing is the registry's casefold, not a second mapping
                assert (registry.canonical_enum_value(
                    schema, "severity", surface.upper()) == member)


def test_alias_pack_omits_surfaces_that_equal_a_member_name():
    # Registry law: a surface equal to a DIFFERENT member's name would be
    # rejected at registration; equal to the TARGET member it is a no-op
    # (members coerce first). Either way the pack must not declare it.
    pack = alias_pack("syslog")
    for surfaces in pack.values():
        for surface in surfaces:
            assert surface.casefold() not in TIER_NAMES, surface
    assert "critical" not in pack["error"]   # syslog level-2 'critical'
    assert "crit" in pack["error"]           # ...but its short form is there
    assert "error" not in alias_pack("log4j")  # 'error' IS the member name


def test_syslog_critical_resolves_to_tier3_with_distinct_member_names():
    # The documented escape hatch for the name collision: serve syslog
    # with neutral member names and the crosswalk tier lands exactly.
    members = ("tier1", "tier2", "tier3", "tier4")
    pack = alias_pack("syslog", members=members)
    assert "critical" in pack["tier3"]
    schema = "probe_syslog_neutral"
    spec = {"type": {"const": schema},
            "severity": {"type": str, "enum": list(members),
                         "aliases": pack}}
    registry.register_schema(schema, spec,
                             routing={"strong": [], "weak": []},
                             extraction={"severity": [r"(.*)"]})
    assert (registry.canonical_enum_value(schema, "severity", "CRITICAL")
            == "tier3")
    assert (TIER_NAMES.index("error") + 1
            == resolve("syslog", "critical") == 3)


def test_merge_packs_is_the_safe_composition():
    # Regression pin: a naive {**a, **b} merge REPLACES each member's
    # list — the doc example silently lost the whole pagerduty pack
    # (bugzilla overwrote it) and "SEV-1" stopped coercing. Found by
    # this feature's own round-trip probe, 2026-10-10.
    merged = merge_packs("pagerduty", "bugzilla")
    naive = {**alias_pack("pagerduty"), **alias_pack("bugzilla")}
    assert "SEV-1" in merged["critical"]
    assert "blocker" in merged["critical"]
    assert "SEV-1" not in naive["critical"]  # exactly the footgun
    # order-preserving and deduped (bugzilla's row keeps short form
    # 'crit'; the word 'critical' itself is the member, omitted)
    assert (merged["critical"]
            == ["SEV-0", "SEV-1", "blocker", "block", "crit"])
    twice = merge_packs("pagerduty", "pagerduty")
    assert twice == alias_pack("pagerduty")


def test_merge_packs_refuses_cross_pack_tier_conflicts():
    # jira 'low' is tier 2; CVSS/GitHub/ITIL 'low' is tier 1. A surface
    # must coerce to exactly one member — refused here (and the registry
    # would refuse it at registration too), never silently reconciled.
    with pytest.raises(ValueError, match="reconcile"):
        merge_packs("jira_priority", "cvss")
    # packs that agree on the shared words merge fine ('P1' is tier 4 in
    # both; itil's word 'critical' is omitted as a member name)
    ok = merge_packs("atlassian_p", "itil")
    assert ok["critical"] == ["P1"]
    assert ok["error"] == ["P2", "high"]


def test_doc_example_round_trip_end_to_end():
    # The EXACT §7.2 example (kept in sync by test_usage_section_*):
    # a customer schema whose aliases come from the generated packs
    # coerces real tokens to canonical members through the pipeline.
    registry.register_schema(
        name="incident",
        spec={
            "type": {"const": "incident"},
            "title": {"type": str},
            "severity": {
                "type": str,
                "enum": ["info", "warning", "error", "critical"],
                "aliases": merge_packs("pagerduty", "bugzilla"),
                "optional": True,
            },
        },
        routing={"strong": [r"(?i)\bincident\b"], "weak": []},
        extraction={
            "title": [r"(?i)\bincident:\s*(.+?)(?:;\s*severity\b|$)"],
            "severity": [r"(?i)\bseverity[\s:]+([A-Za-z0-9_-]+)"],
        },
    )
    obj, meta = pipeline.classify(
        "incident: checkout down for all users; severity: SEV-1",
        router=None)
    assert obj["severity"] == "critical"      # SEV-1 -> tier 4 member
    assert obj["title"] == "checkout down for all users"
    assert meta["valid"] is True
    obj2, _ = pipeline.classify(
        "incident: typo on the pricing page; severity: trivial",
        router=None)
    assert obj2["severity"] == "info"         # bugzilla trivial -> tier 1
    obj3, _ = pipeline.classify(
        "incident: badge colors; severity: SEV-4", router=None)
    assert obj3["severity"] == "warning"      # tier 2 member
    # refusal contract (§7.1): an undeclared surface is ABSENT, never
    # repaired to a nearest-looking member.
    obj4, _ = pipeline.classify(
        "incident: mystery; severity: katastrophal", router=None)
    assert "severity" not in obj4


# --- the committed USAGE section cannot drift from the module ----------------

def _usage_region(text):
    begins = [m.start() for m in
              re.finditer(re.escape(USAGE_BEGIN_MARK), text)]
    ends = [m.start() for m in
            re.finditer(re.escape(USAGE_END_MARK), text)]
    return begins, ends, text


def test_usage_section_matches_module_byte_for_byte():
    text = USAGE_PATH.read_text(encoding="utf-8")
    begins, ends, _ = _usage_region(text)
    assert len(begins) == 1 and len(ends) == 1 and begins[0] < ends[0]
    committed = text[begins[0]:ends[0] + len(USAGE_END_MARK)]
    rendered = (USAGE_BEGIN_MARK + "\n" + render_usage_markdown()
                + "\n" + USAGE_END_MARK)
    assert committed == rendered


def test_usage_section_states_the_honesty_boundary():
    text = USAGE_PATH.read_text(encoding="utf-8")
    region = text[text.find(USAGE_BEGIN_MARK):text.find(USAGE_END_MARK)]
    # the trained head's measured numbers, and the plain statement that
    # the crosswalk is deterministic mapping, not a model surface
    assert "46.8% exact / 84.2% within-1" in region
    assert "n=600" in region
    assert "NOT a measured model surface" in region


def test_quoted_grade_numbers_exist_in_the_committed_sources():
    # Rule 15: numbers quoted in customer-facing docs must be readable
    # back from the committed instruments/sources they cite.
    model_state = MODEL_STATE_PATH.read_text(encoding="utf-8")
    assert "46.8%" in model_state
    assert "84.2% within-1" in model_state
    assert "n=600" in model_state
    # (the training-report leg retired with reports/: the record is not
    # redistributed; docs/MODEL-STATE.md is the committed public source.)


def test_generator_script_exists_and_is_the_documented_writer():
    script = REPO_ROOT / "scripts" / "gen_severity_alias_packs.py"
    assert script.exists()
    source = script.read_text(encoding="utf-8")
    assert "render_usage_markdown" in source
    assert "USAGE_BEGIN_MARK" in source
