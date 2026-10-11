# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Severity-vocabulary crosswalk: production taxonomies -> canonical tiers.

Deterministic, data-only, stdlib-only. NO model bytes, NO network, NO
training: every mapping below is a fixed table entry with a named source.

Why this exists: the trained grade head measures ONE ordinal scale — the
exclusive 4-tier anchors in ``severity_head.CANONICAL_SEVERITY_ANCHORS``
(info -> warning -> error -> critical). Customer tools speak PagerDuty,
syslog, CVSS and friends. This module maps each of those vocabularies
onto the canonical tiers so a canonical grade can be RENDERED in the
customer's own vocabulary, and so their tokens can be coerced onto the
canonical enum via the enum-aliases feature (docs/USAGE.md §7.1).

Polarity (verified from the code, 2026-10-10): the model's tier encoding
is 0-BASED — ``severity_head.cumulative_tier`` and
``triage_severity()["tier_index"]`` return 0 = info .. 3 = critical.
This module uses the human-facing 1..4 convention the docs speak:

    tier 1 = info      (routine, normal operation)
    tier 2 = warning   (recoverable abnormal, not a failure)
    tier 3 = error     (operation failures that disrupted a task)
    tier 4 = critical  (severe crashes, data loss, complete outage)

so ``tier = tier_index + 1`` and ``CANONICAL_SEVERITY_ANCHORS[tier - 1]``
is the tier's anchor (see :func:`canonical_anchor`).

Per-taxonomy polarity is THE point of a crosswalk: PagerDuty counts
DOWNWARD (SEV-1 is the worst), syslog counts upward (0 is the worst),
OTel SeverityNumber counts upward (24 is the worst), CVSS words order
none < low < medium < high < critical. Mappings follow each taxonomy's
own documented semantics — verified against the primary sources
2026-10-10 (PagerDuty incident-response docs; OTel logs data-model
Appendix B; Atlassian incident-severity guidance) — never against the
English words alone. Where a source's wording forces a judgment call
(e.g. where PagerDuty's 5 levels collapse onto 4 tiers), the call is
recorded in the row's note.
"""
from __future__ import annotations

import re
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

__all__ = [
    "TIER_NAMES",
    "TAXONOMIES",
    "Row",
    "Taxonomy",
    "resolve",
    "resolve_note",
    "taxonomy_names",
    "normalize_token",
    "canonical_anchor",
    "alias_pack",
    "merge_packs",
    "render_usage_markdown",
    "USAGE_BEGIN_MARK",
    "USAGE_END_MARK",
]

# --- the canonical scale (the grade head's anchors, 1-based here) -----------

#: Canonical tier names, index ``tier - 1``. These are also the enum member
#: names the generated alias packs coerce onto (docs/USAGE.md §7.2).
TIER_NAMES: Tuple[str, str, str, str] = ("info", "warning", "error",
                                         "critical")


def canonical_anchor(tier: int) -> str:
    """The ``severity_head.CANONICAL_SEVERITY_ANCHORS`` string for ``tier``.

    ``tier`` is 1-based (4 = most severe), matching this module's tables;
    the anchor index is 0-based, hence the ``- 1``.
    """
    from .severity_head import CANONICAL_SEVERITY_ANCHORS

    return CANONICAL_SEVERITY_ANCHORS[max(1, min(4, int(tier))) - 1]


# --- table shape -------------------------------------------------------------

#: One ordered mapping: surface tokens (display form) -> canonical tier
#: (1..4) with the named source note.
Row = namedtuple("Row", "tokens tier note")

#: A vocabulary: display title, family, primary source, ordered rows.
Taxonomy = namedtuple("Taxonomy", "title family source rows")

# --- the crosswalk tables ----------------------------------------------------
#
# Conventions for the notes: each row names its source and, where the
# collapse is a judgment call, says so. Tiers are 1-based (4 = worst).

_TAXONOMIES: Dict[str, Taxonomy] = {}

_TAXONOMIES["pagerduty"] = Taxonomy(
    title="PagerDuty SEV (incident/ops)",
    family="incident/ops",
    source=(
        "PagerDuty Incident Response Docs, 'Severity Levels' "
        "(response.pagerduty.com, repo PagerDuty/incident-response-docs): "
        "lower number = more severe; 'Anything above a SEV-3 is "
        "automatically considered a \"major incident\"'; 'All SEV-2's are "
        "major incidents'."
    ),
    rows=(
        Row(("SEV-0",), 4,
            "informal extension: orgs that use SEV-0 mean 'worse than "
            "SEV-1'; there is no slot above canonical tier 4"),
        Row(("SEV-1",), 4,
            "PagerDuty SEV-1 = critical issue: system in a critical state, "
            "actively impacting a large number of customers, "
            "customer-data-exposing vulnerability -> tier 4 (severe "
            "failure / data loss requiring immediate action)"),
        Row(("SEV-2",), 3,
            "PagerDuty SEV-2 = critical system issue actively impacting "
            "many customers (a major incident, short of SEV-1) -> tier 3 "
            "(failures that disrupted service). Judgment call: PD's own "
            "SEV-2 wording can describe an app unavailable for most/all "
            "users; tier 4 is reserved for SEV-1-grade complete outage / "
            "data loss — re-pin SEV-2 -> tier 4 if your SEV-2 IS a full "
            "outage"),
        Row(("SEV-3",), 3,
            "PagerDuty SEV-3 = partial loss of functionality, not "
            "affecting the majority of customers -> tier 3 (a failure "
            "that disrupted the task, for part of the user base)"),
        Row(("SEV-4",), 2,
            "PagerDuty SEV-4 = minor issues requiring action but not "
            "affecting customer ability to use the product (performance "
            "delays, single-node failure, cron failure) -> tier 2 "
            "(recoverable abnormal conditions, not failures)"),
        Row(("SEV-5",), 1,
            "PagerDuty SEV-5 = cosmetic issues or bugs -> tier 1 "
            "(routine)"),
    ),
)

_TAXONOMIES["atlassian_p"] = Taxonomy(
    title="Atlassian / Jira P1-P4 (incident/ops)",
    family="incident/ops",
    source=(
        "Atlassian incident-severity guidance "
        "(atlassian.com, 'Understanding incident severity levels'): SEV 1 "
        "'a critical incident with very high impact', SEV 2 'a major "
        "incident with significant impact', SEV 3 'a minor incident with "
        "low impact'; the extended model adds minor-usability / cosmetic "
        "tiers. Rendered here in the common P1-P4 incident-priority "
        "notation (lower = more severe)."
    ),
    rows=(
        Row(("P1",), 4,
            "Atlassian P1 = critical incident, very high impact, all "
            "hands -> tier 4"),
        Row(("P2",), 3,
            "Atlassian P2 = major incident with significant impact "
            "(short of P1) -> tier 3"),
        Row(("P3",), 2,
            "Atlassian P3 = minor incident with low impact -> tier 2"),
        Row(("P4",), 1,
            "Atlassian extended model P4 = minor usability / cosmetic "
            "impact -> tier 1"),
    ),
)

_TAXONOMIES["bugzilla"] = Taxonomy(
    title="Bugzilla severities (bug tracking)",
    family="bug tracking",
    source=(
        "Bugzilla severity field values (bugzilla.mozilla.org / "
        "bugzilla.org documentation): blocker, critical, major, normal, "
        "minor, trivial, enhancement; the legacy short forms "
        "block/crit/maj/nor/min/triv/enh are the same values."
    ),
    rows=(
        Row(("blocker", "block"), 4,
            "blocks development and/or testing or the release -> tier 4 "
            "(immediate-action blocker)"),
        Row(("critical", "crit"), 4,
            "Bugzilla critical = crashes, loss of data, severe memory "
            "leak -> tier 4 (the canonical critical anchor's own words: "
            "severe crashes, data loss)"),
        Row(("major", "maj"), 3,
            "major loss of function -> tier 3 (failures that disrupted a "
            "task or transaction)"),
        Row(("normal", "nor"), 2,
            "the ordinary default defect severity -> tier 2 (an abnormal "
            "condition needing attention, not an operational failure)"),
        Row(("minor", "min"), 2,
            "minor loss of function, easy workaround exists -> tier 2"),
        Row(("trivial", "triv"), 1,
            "cosmetic problems (misspellings, alignment) -> tier 1"),
        Row(("enhancement", "enh"), 1,
            "request for enhancement — not a defect at all -> tier 1 "
            "(routine)"),
    ),
)

_TAXONOMIES["jira_priority"] = Taxonomy(
    title="Jira priorities (bug tracking)",
    family="bug tracking",
    source=(
        "Jira / Jira Service Management default priority scheme: Highest "
        "(address immediately), High (address soon), Medium (in the "
        "normal course), Low (when time permits), Lowest (at the "
        "assignee's discretion)."
    ),
    rows=(
        Row(("highest",), 4,
            "address immediately -> tier 4"),
        Row(("high",), 3,
            "address soon to mitigate real damage -> tier 3"),
        Row(("medium",), 2,
            "in the normal course of work -> tier 2 (needs attention, "
            "not a failure)"),
        Row(("low",), 2,
            "when time permits — still a tracked abnormality, just not "
            "urgent -> tier 2"),
        Row(("lowest",), 1,
            "at the assignee's discretion -> tier 1 (routine)"),
    ),
)

_TAXONOMIES["github_labels"] = Taxonomy(
    title="GitHub label conventions (bug tracking)",
    family="bug tracking",
    source=(
        "GitHub community label conventions: priority:/severity: word "
        "labels (critical/high/medium/low) and the P0-P3 convention "
        "(P0 = drop-everything critical ... P3 = low)."
    ),
    rows=(
        Row(("critical",), 4, "label severity critical -> tier 4"),
        Row(("P0",), 4,
            "P0 convention: drop everything, site-level emergency -> "
            "tier 4"),
        Row(("high",), 3, "label severity high -> tier 3"),
        Row(("P1",), 3, "P1 convention: urgent, important -> tier 3"),
        Row(("medium",), 2, "label severity medium -> tier 2"),
        Row(("P2",), 2, "P2 convention: normal priority -> tier 2"),
        Row(("low",), 1, "label severity low -> tier 1"),
        Row(("P3",), 1, "P3 convention: low priority -> tier 1"),
    ),
)

_TAXONOMIES["syslog"] = Taxonomy(
    title="syslog RFC 5424 (logs)",
    family="logs",
    source=(
        "RFC 5424 §6.2.1 Severity (0 = Emergency ... 7 = Debug; LOWER "
        "numeric value = MORE severe), mapped level-by-level per the "
        "OTel logs data-model Appendix B table: "
        "Emergency->FATAL(21), Alert->ERROR3(19), Critical->ERROR2(18), "
        "Error->ERROR(17), Warning->WARN(13), Notice->INFO2(10), "
        "Informational->INFO(9), Debug->DEBUG(5)."
    ),
    rows=(
        Row(("0", "emerg", "emergency", "panic"), 4,
            "syslog 0 Emergency -> OTel FATAL band (SeverityNumber 21) -> "
            "tier 4. Emergency-vs-FATAL debate, for the record: RFC 5424 "
            "Emergency means 'system is unusable' and is often emitted by "
            "a monitor that survives, while application-fatal means the "
            "emitting process is dying — some argue Emergency should "
            "therefore NOT be fatal-tier. OTel's table nonetheless puts "
            "Emergency in the FATAL band and we follow the documented "
            "mapping. 'panic' is the legacy syslog(3) name for the same "
            "level"),
        Row(("1", "alert"), 3,
            "syslog 1 Alert -> OTel ERROR3 (19) -> tier 3: OTel "
            "deliberately documents Alert inside the ERROR band, not "
            "FATAL — tier 4 is Emergency-only on this taxonomy"),
        Row(("2", "crit", "critical"), 3,
            "syslog 2 Critical ('critical conditions') -> OTel ERROR2 "
            "(18) -> tier 3. NAME COLLISION, deliberate: canonical tier 4 "
            "is *named* 'critical', but the crosswalk follows the "
            "documented OTel placement (ERROR band), not the English "
            "word — see the alias-pack caveat in docs/USAGE.md §7.2"),
        Row(("3", "err", "error"), 3,
            "syslog 3 Error -> OTel ERROR (17) -> tier 3 (operation "
            "failures)"),
        Row(("4", "warn", "warning"), 2,
            "syslog 4 Warning -> OTel WARN (13) -> tier 2"),
        Row(("5", "notice"), 1,
            "syslog 5 Notice ('normal but significant') -> OTel INFO2 "
            "(10) -> tier 1: OTel documents Notice inside the INFO band "
            "(a stricter reading would put it in tier 2; we follow the "
            "documented mapping)"),
        Row(("6", "informational", "info"), 1,
            "syslog 6 Informational -> OTel INFO (9) -> tier 1"),
        Row(("7", "debug"), 1,
            "syslog 7 Debug -> OTel DEBUG (5) -> tier 1"),
    ),
)

_TAXONOMIES["otel"] = Taxonomy(
    title="OTel SeverityNumber 1-24 (logs)",
    family="logs",
    source=(
        "OpenTelemetry logs data-model, SeverityNumber: bands TRACE(1-4), "
        "DEBUG(5-8), INFO(9-12), WARN(13-16), ERROR(17-20), FATAL(21-24); "
        "HIGHER number = MORE severe (opposite of syslog's numeric "
        "direction)."
    ),
    rows=(
        Row(("TRACE", "1", "2", "3", "4"), 1,
            "SeverityNumber 1-4, TRACE band -> tier 1"),
        Row(("DEBUG", "5", "6", "7", "8"), 1,
            "SeverityNumber 5-8, DEBUG band -> tier 1"),
        Row(("INFO", "9", "10", "11", "12"), 1,
            "SeverityNumber 9-12, INFO band -> tier 1"),
        Row(("WARN", "WARNING", "13", "14", "15", "16"), 2,
            "SeverityNumber 13-16, WARN band -> tier 2"),
        Row(("ERROR", "17", "18", "19", "20"), 3,
            "SeverityNumber 17-20, ERROR band -> tier 3"),
        Row(("FATAL", "21", "22", "23", "24"), 4,
            "SeverityNumber 21-24, FATAL band -> tier 4"),
    ),
)

_TAXONOMIES["log4j"] = Taxonomy(
    title="Log4j levels (logs)",
    family="logs",
    source=(
        "Log4j built-in levels; the OTel Appendix B table maps TRACE->1, "
        "DEBUG->5, INFO->9, WARN->13, ERROR->17, FATAL->21. OFF and ALL "
        "are logger CONTROLS, not severities: deliberately unmapped "
        "(resolve() returns None for them)."
    ),
    rows=(
        Row(("TRACE",), 1, "Log4j TRACE -> tier 1"),
        Row(("DEBUG",), 1, "Log4j DEBUG -> tier 1"),
        Row(("INFO",), 1, "Log4j INFO -> tier 1"),
        Row(("WARN", "WARNING"), 2, "Log4j WARN -> tier 2"),
        Row(("ERROR",), 3, "Log4j ERROR -> tier 3"),
        Row(("FATAL",), 4, "Log4j FATAL -> tier 4"),
    ),
)

_TAXONOMIES["cvss"] = Taxonomy(
    title="CVSS qualitative severity (security)",
    family="security",
    source=(
        "CVSS v3.1/v4.0 Qualitative Severity Rating Scale (None, Low, "
        "Medium, High, Critical) plus the standard vulnerability-"
        "management CVSS->priority mapping (Critical->P1, High->P2, "
        "Medium->P3, Low->P4)."
    ),
    rows=(
        Row(("critical",), 4,
            "CVSS Critical (9.0-10.0) -> P1 -> tier 4"),
        Row(("high",), 3, "CVSS High (7.0-8.9) -> P2 -> tier 3"),
        Row(("medium",), 2, "CVSS Medium (4.0-6.9) -> P3 -> tier 2"),
        Row(("low",), 1,
            "CVSS Low (0.1-3.9) -> P4 -> tier 1 (the vuln-management "
            "convention puts Low in the lowest ops tier)"),
        Row(("none",), 1,
            "CVSS None (0.0): no security impact -> tier 1 (routine)"),
    ),
)

_TAXONOMIES["itil"] = Taxonomy(
    title="ITIL priority = impact x urgency (ITSM)",
    family="ITSM",
    source=(
        "ITIL incident-management priority matrix: priority = impact x "
        "urgency over the 3x3 grid -> P1 critical (high impact x high "
        "urgency), P2 high (high x med or med x high), P3 moderate "
        "(high x low, med x med, low x high), P4 low (med x low, low x "
        "med), P5 very low (low x low)."
    ),
    rows=(
        Row(("P1", "critical"), 4,
            "P1 = high impact x high urgency -> tier 4"),
        Row(("P2", "high"), 3, "P2 = high x medium or medium x high -> tier 3"),
        Row(("P3", "moderate", "medium"), 2,
            "P3 = moderate (high x low, medium x medium, low x high) -> "
            "tier 2"),
        Row(("P4", "low"), 1,
            "P4 = low (medium x low, low x medium) -> tier 1"),
        Row(("P5", "very low"), 1, "P5 = low x low -> tier 1"),
    ),
)

#: Family-ordered names (the display/documentation order).
_ORDER = ("pagerduty", "atlassian_p", "bugzilla", "jira_priority",
          "github_labels", "syslog", "otel", "log4j", "cvss", "itil")

#: The raw crosswalk tables, exported for schema authors and tooling:
#: ``TAXONOMIES[name].rows`` is the ordered ``(tokens, tier, note)`` list
#: exactly as documented (docs/USAGE.md §7.2).
TAXONOMIES: Dict[str, Taxonomy] = _TAXONOMIES


# --- normalization + resolve --------------------------------------------------

_SEV_NUM = re.compile(r"^sev[-_ :]?(\d+)$")
_P_NUM = re.compile(r"^p[-_ :]?(\d+)$")
_ECHO_PREFIX = re.compile(r"^(?:severity|priority)[-_ :](.+)$")


def normalize_token(token) -> str:
    """Case-insensitive surface normalization for crosswalk lookups.

    Trim + casefold + wrap-strip, then drop the vocabulary's own echo
    prefixes: ``severity:``/``priority-`` labels, ``SEV-1``/``sev_1``/
    ``sev 1``/``sev1`` spellings (a digit must follow — the word
    'severe' is untouched), and the P-notation ``P1``/``p-2`` (again
    only before a digit, so 'panic' survives). Deterministic, stdlib.
    """
    s = str(token).strip().casefold()
    s = s.strip("\"'`.,;:!?()[]{}")
    m = _ECHO_PREFIX.match(s)
    if m:
        s = m.group(1)
    m = _SEV_NUM.match(s)
    if m:
        return m.group(1)
    m = _P_NUM.match(s)
    if m:
        return m.group(1)
    return s


def _build_tables() -> Dict[str, Dict[str, Tuple[int, str]]]:
    """Normalized-key lookup tables, built once at import.

    Import-time integrity gate (deterministic, rule-12 spirit): a token
    that normalized to two different tiers inside one taxonomy, or a
    tier outside 1..4, is a build error — never a silent wrong mapping.
    """
    tables: Dict[str, Dict[str, Tuple[int, str]]] = {}
    for name in _ORDER:
        tax = _TAXONOMIES[name]
        table: Dict[str, Tuple[int, str]] = {}
        for row in tax.rows:
            if not (isinstance(row.tier, int) and 1 <= row.tier <= 4):
                raise ValueError(
                    f"severity_taxonomies[{name!r}]: tier {row.tier!r} "
                    f"outside 1..4 for tokens {row.tokens}")
            for token in row.tokens:
                key = normalize_token(token)
                if not key:
                    raise ValueError(
                        f"severity_taxonomies[{name!r}]: token {token!r} "
                        "normalizes to an empty key")
                other = table.get(key)
                if other is not None and other[0] != row.tier:
                    raise ValueError(
                        f"severity_taxonomies[{name!r}]: token {token!r} "
                        f"collides: {other[0]} vs {row.tier}")
                table.setdefault(key, (row.tier, row.note))
        tables[name] = table
    return tables


_TABLES = _build_tables()


def taxonomy_names() -> Tuple[str, ...]:
    """The crosswalk vocabularies in display (family) order."""
    return _ORDER


def resolve(taxonomy: str, token: str) -> Optional[int]:
    """Map one surface token to its canonical tier (1..4); ``None`` = no.

    Case-insensitive; strips the ``sev-``/``p``-style numeric prefixes
    (see :func:`normalize_token`). Unknown taxonomy, unknown token, or a
    deliberately-unmapped control token (log4j ``OFF``) all return
    ``None`` — never raise, never guess (honest-values law).
    """
    table = _TABLES.get(str(taxonomy).strip().casefold())
    if table is None:
        return None
    hit = table.get(normalize_token(token))
    return hit[0] if hit is not None else None


def resolve_note(taxonomy: str, token: str) -> Optional[str]:
    """The named source note behind :func:`resolve`'s verdict (or None)."""
    table = _TABLES.get(str(taxonomy).strip().casefold())
    if table is None:
        return None
    hit = table.get(normalize_token(token))
    return hit[1] if hit is not None else None


# --- alias packs (compose with the enum-aliases feature) ----------------------


def alias_pack(taxonomy: str,
               members: Tuple[str, str, str, str] = TIER_NAMES,
               ) -> Dict[str, List[str]]:
    """A paste-ready ``aliases`` map for one taxonomy (docs/USAGE.md §7.1).

    ``members`` are the schema author's four enum members in canonical
    order (least -> most severe); the default is the canonical tier
    names. The returned dict is ``{member: [surface, ...]}`` built FROM
    the crosswalk rows, so packs cannot drift from the tables.

    Registry-law-aware: a surface whose registry-normalized form equals
    ANY enum member's normalized form is omitted — equal to the target
    member it is a no-op (members coerce first anyway), and equal to a
    DIFFERENT member registration would reject it. This is why syslog's
    ``critical`` is absent from the default pack: with the default
    member names the word itself is the tier-4 member. Serve that
    taxonomy with distinct member names (e.g. ``("tier1", "tier2",
    "tier3", "tier4")``) and the word lands in the pack, targeting the
    tier-3 member, exactly as the crosswalk table says.
    """
    from .registry import _normalize_surface  # the one normalization law

    tax = _TAXONOMIES.get(str(taxonomy).strip().casefold())
    if tax is None:
        return {}
    member_norm = {_normalize_surface(m): m for m in members}
    pack: Dict[str, List[str]] = {m: [] for m in members}
    for row in tax.rows:
        target = members[row.tier - 1]
        for token in row.tokens:
            norm = _normalize_surface(token)
            if not norm or norm in member_norm:
                continue  # member names coerce first; never re-declare them
            if norm not in {_normalize_surface(s) for s in pack[target]}:
                pack[target].append(token)
    return {m: surfaces for m, surfaces in pack.items() if surfaces}


def merge_packs(*taxonomies: str,
                members: Tuple[str, str, str, str] = TIER_NAMES,
                ) -> Dict[str, List[str]]:
    """Union of several alias packs — the SAFE composition (§7.2 example).

    A naive ``{**pack_a, **pack_b}`` dict merge REPLACES each member's
    list wholesale (the second pack silently erases the first pack's
    surfaces for the same member — found by the doc example's own
    round-trip: bugzilla overwrote the whole pagerduty pack and
    "SEV-1" stopped coercing). This helper extends per member, keeps
    declaration order, dedupes surfaces, and refuses a surface that
    means two DIFFERENT members across packs (e.g. jira ``low`` = tier
    2 vs CVSS ``low`` = tier 1) — the same declared-vocabulary law the
    registry enforces at registration, raised earlier with a clearer
    message. Reconcile deliberately instead.
    """
    from .registry import _normalize_surface  # the one normalization law

    out: Dict[str, List[str]] = {}
    seen: Dict[str, str] = {}
    for name in taxonomies:
        for member, surfaces in alias_pack(name, members=members).items():
            bucket = out.setdefault(member, [])
            for surface in surfaces:
                norm = _normalize_surface(surface)
                other = seen.get(norm)
                if other is not None:
                    if other != member:
                        raise ValueError(
                            f"merge_packs: surface {surface!r} maps to "
                            f"{other!r} in one pack and {member!r} in "
                            "another; reconcile the packs before "
                            "declaring (a surface must coerce to "
                            "exactly one member)")
                    continue
                seen[norm] = member
                bucket.append(surface)
    return {m: s for m, s in out.items() if s}


# --- the generated markdown (rendered into docs/USAGE.md §7.2) ----------------

USAGE_BEGIN_MARK = ("<!-- BEGIN severity-taxonomies: generated by "
                    "scripts/gen_severity_alias_packs.py from "
                    "src/mivia_small_json/severity_taxonomies.py — "
                    "regenerate, never hand-edit -->")
USAGE_END_MARK = "<!-- END severity-taxonomies -->"

HONESTY_NOTE = (
    "The trained grade head measures the CANONICAL scale only — 46.8% "
    "exact / 84.2% within-1 on the n=600 real-ticket v3 set "
    "(docs/MODEL-STATE.md). The "
    "crosswalk is NOT a measured model surface: it is deterministic "
    "token mapping, stdlib-only, no model bytes."
)


def _tier_cells(tax: Taxonomy) -> List[List[str]]:
    cells: List[List[str]] = [[] for _ in range(4)]
    for row in tax.rows:
        cells[row.tier - 1].extend(row.tokens)
    return cells


def _compress(tokens: List[str]) -> str:
    """Display tokens; compress numeric runs ('1', '2', '3' -> '1-3')."""
    nums = sorted(int(t) for t in tokens if str(t).isdigit())
    words = [t for t in tokens if not str(t).isdigit()]
    parts: List[str] = []
    if nums:
        runs: List[List[int]] = [[nums[0]]]
        for n in nums[1:]:
            if n == runs[-1][-1] + 1:
                runs[-1].append(n)
            else:
                runs.append([n])
        parts.extend(str(r[0]) if len(r) == 1 else f"{r[0]}-{r[-1]}"
                     for r in runs)
    parts.extend(words)
    return " · ".join(parts)


def render_usage_markdown() -> str:
    """Render the §7.2 generated block (crosswalk table + alias packs).

    Deterministic: no wall-clock, no environment, no network — the
    committed output in docs/USAGE.md is byte-reproducible from the
    module alone (tests/test_severity_taxonomies.py guards the drift).
    """
    lines: List[str] = []

    lines.append(HONESTY_NOTE)
    lines.append("")
    lines.append("Tier names below are the canonical anchor names "
                 "(``severity_head.CANONICAL_SEVERITY_ANCHORS``); "
                 "**tier 4 is most severe**. The model's 0-based "
                 "``tier_index`` (``triage_severity``, "
                 "``severity_head.cumulative_tier``) is ``tier - 1``.")
    lines.append("")
    lines.append("| Taxonomy | Tier 1 `info` | Tier 2 `warning` | "
                 "Tier 3 `error` | Tier 4 `critical` |")
    lines.append("|---|---|---|---|---|")
    for name in _ORDER:
        tax = _TAXONOMIES[name]
        cells = _tier_cells(tax)
        row = (f"| `{name}` — {tax.title} | "
               + " | ".join(_compress(c) if c else "—" for c in cells)
               + " |")
        lines.append(row)
    lines.append("")
    lines.append("**Sources and judgment calls** (named, per the repo's "
                 "honesty rules):")
    lines.append("")
    for name in _ORDER:
        tax = _TAXONOMIES[name]
        lines.append(f"- `{name}` — {tax.source}")
        for row in tax.rows:
            note = row.note.replace("\n", " ")
            lines.append(f"  - {' / '.join(row.tokens)} -> tier "
                         f"{row.tier}: {note}")
    lines.append("")
    lines.append("#### Alias packs (generated from the tables above)")
    lines.append("")
    lines.append("Paste into your enum field's `aliases` (§7.1). The "
                 "canonical members are the four tier names; every "
                 "surface coerces EXACTLY (no fuzzy matching). Surfaces "
                 "that equal a canonical member name (syslog "
                 "`critical`, log4j `error`, `warning`, `info`) are "
                 "omitted — they already coerce as members; serve such "
                 "taxonomies with distinct member names and pass them "
                 "as `members=` to `alias_pack()`. To combine packs, "
                 "use `merge_packs()` — a naive `{**a, **b}` merge "
                 "REPLACES each member's list and silently drops the "
                 "first pack. A surface that means two tiers across "
                 "packs (jira `low` = tier 2 vs CVSS/GitHub/ITIL `low` "
                 "= tier 1) is refused by `merge_packs()` (and by the "
                 "registry at registration) BY DESIGN — pick the pack "
                 "that matches your tool, or reconcile deliberately.")
    lines.append("")
    lines.append("```python")
    for name in _ORDER:
        pack = alias_pack(name)
        tax = _TAXONOMIES[name]
        lines.append(f"# {tax.title}")
        lines.append(f"{name.upper()}_SEVERITY_ALIASES = {{")
        for member in TIER_NAMES:
            surfaces = pack.get(member)
            if not surfaces:
                continue
            joined = ", ".join(f'"{s}"' for s in surfaces)
            lines.append(f'    "{member}": [{joined}],')
        lines.append("}")
        lines.append("")
    lines.append("```")
    lines.append("")
    lines.append("Round-trip (works end-to-end with §7.1 coercion):")
    lines.append("")
    lines.append("```python")
    lines.append("from mivia_small_json import pipeline, registry")
    lines.append("from mivia_small_json.severity_taxonomies import "
                 "merge_packs")
    lines.append("")
    lines.append("registry.register_schema(")
    lines.append('    name="incident",')
    lines.append("    spec={")
    lines.append('        "type": {"const": "incident"},')
    lines.append('        "title": {"type": str},')
    lines.append('        "severity": {')
    lines.append("            \"type\": str,")
    lines.append('            "enum": ["info", "warning", "error", '
                 '"critical"],')
    lines.append('            "aliases": merge_packs("pagerduty", '
                 '"bugzilla"),')
    lines.append('            "optional": True,  # honest refusal-to-absent '
                 'choice (§7.1)')
    lines.append("        },")
    lines.append("    },")
    lines.append('    routing={"strong": [r"(?i)\\bincident\\b"], '
                 '"weak": []},')
    lines.append("    extraction={")
    lines.append('        "title": [r"(?i)\\bincident:\\s*(.+?)'
                 '(?:;\\s*severity\\b|$)"],')
    lines.append('        "severity": [r"(?i)\\bseverity[\\s:]+([A-Za-z0-9_-]+)"'
                 "],")
    lines.append("    },")
    lines.append(")")
    lines.append("")
    lines.append("obj, _ = pipeline.classify(")
    lines.append('    "incident: checkout down for all users; severity: '
                 'SEV-1", router=None)')
    lines.append('# obj["severity"] == "critical"  (SEV-1 -> tier 4 member)')
    lines.append("obj2, _ = pipeline.classify(")
    lines.append('    "incident: typo on the pricing page; severity: '
                 'trivial", router=None)')
    lines.append('# obj2["severity"] == "info"      (bugzilla trivial -> '
                 'tier 1 member)')
    lines.append("```")
    return "\n".join(lines)
