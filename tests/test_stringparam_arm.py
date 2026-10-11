"""The string-param span-fill arm (string-param-span-fill).

Scaffold-phase gates, in the spec's own order of importance:

- B6 byte-parity: with NO promoted extractor artifact the serving path is
  byte-identical to the pre-arm bytes, pinned over the committed fixture
  set (the routing golden's inputs, replayed through classify) and over a
  committed catalogue case set through the served guard path. The digests
  below were computed at the arm's scaffold commit with the shipped
  artifacts; a re-promotion of the ROUTER or GUARD bytes legitimately
  moves them (recompute then, never edit silently).
- Family gate: string admitted ONLY with a loadable extractor; mixed,
  boolean, and untyped-unhinted tools refused; numeric untouched (with
  and without an artifact).
- Decode: ladder-first claim, span exclusivity (the loser stays missing),
  strict enum coercion (declared form emitted; mismatch refused),
  abstain-on-absence (the served path emits the missing-arguments
  clarification, never a fabricated value).
- Data gates as failing checks: builder minimum-contribution failure,
  eval-window exclusion by normalized-set comparison, alignment-evidence
  presence, provenance idempotency, dedupe keep-first, the mechanical
  harvest admission filter, and the committed DATA-BLOCKED verdict.

Deterministic, CPU-only, no timing asserts. The mock scorer is injected
through the trainer's own injection seam and ALWAYS restored, so the
no-artifact byte-parity tests see the shipped state regardless of order.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from mivia_small_json import pipeline, registry, span_fill
from mivia_small_json.tool_call import (
    register_no_tool_call_schema,
    register_tool_call_schema,
    register_tool_clarification_schema,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # noqa: E402

FIXTURE_DIGEST = "e82de4ef269780477c6c18b11cfd3611c90f202da8329a6f50aabcf705956cba"
# Public-release re-pin, 2026-10-11: the fixture itself was regenerated
# over the PUBLIC sources only (synthetic + heldout; the third-party wild
# rows left the tree with their sources — see the generator's source
# contract), so the replayed input set changed and the digest moved with
# it. Recomputed from the measured forced-absent replay over the new
# fixture; dev's private-tree pin (2a2c2342...) is not comparable.
# Per this file's own law: recompute, never edit blindly.
# Deterministic-only family admission re-pin, 2026-10-10: the catalogue
# digest moved by exactly ONE row (service=nginx host=web-01 now CALLs
# with verbatim arguments instead of the veto's {} minimum; every other
# row — including "restart nginx on staging"'s {} honest minimum — is
# byte-unchanged; measured HEAD vs candidate under the forced-absent
# loader, .scratch/ladderfill/b6-noartifact-{head,candidate}.json). The
# fixture digest is UNCHANGED by the admission.
CATALOGUE_DIGEST = "6be73d692a82776327ad348b1fc3f4f3d0659c228cb177216f0b4f116c0bb584"

STR_TOOL = {"name": "restart_service", "description":
            "Restart a systemd service on a named host.",
            "parameters": {"properties": {
                "service": {"type": "string",
                            "description": "The systemd service unit."},
                "host": {"type": "string",
                         "description": "The target host."}},
                "required": ["service", "host"]}}
NUM_TOOL = {"name": "calculate_sum", "description":
            "Calculate the sum of two numbers a and b and return the result.",
            "parameters": {"properties": {
                "a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"]}}
ENUM_TOOL = {"name": "deploy_release", "description":
             "Ship a release to a target environment.",
             "parameters": {"properties": {
                 "release": {"type": "string"},
                 "environment": {"type": "string",
                                 "enum": ["staging", "production",
                                          "canary"]}},
                 "required": ["release", "environment"]}}
MIX_TOOL = {"name": "scale_pool", "description":
            "Resize a database connection pool.",
            "parameters": {"properties": {
                "pool": {"type": "string"}, "size": {"type": "integer"}},
                "required": ["pool", "size"]}}
BOOL_TOOL = {"name": "toggle", "description": "Flip a flag.",
             "parameters": {"properties": {
                 "flag": {"type": "boolean"}}, "required": ["flag"]}}


class MockScorer:
    """Deterministic scorer for the injected-artifact tests: scores the
    LAST \\w+ token 1.0 and everything else 0.1 (so the ladders' claims
    win and the learned decode has to respect exclusivity)."""

    def pair_probs(self, request, param, prop):
        n = len(re.findall(r"\w+", request))
        return [1.0 if i == n - 1 else 0.1 for i in range(n)]


@pytest.fixture()
def with_mock_artifact():
    saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
    saved_floors = (span_fill._SPANFILL_EMIT_FLOOR,
                    span_fill._SPANFILL_EXTEND_FLOOR)
    span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = MockScorer(), True
    span_fill.configure_spanfill_floors(0.6, 0.15)
    yield
    span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = saved
    span_fill.configure_spanfill_floors(*saved_floors)


# --- B6: byte parity without an artifact ------------------------------------

def _scrub(o):
    if isinstance(o, dict):
        return {k: ("<ts>" if k == "ts" else _scrub(v))
                for k, v in o.items() if k != "latency_ms"}
    if isinstance(o, list):
        return [_scrub(v) for v in o]
    return o


def _digest_classify(router, texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        obj, meta = pipeline.classify(t, router=router)
        h.update(json.dumps({"obj": _scrub(obj), "meta": _scrub(meta)},
                            sort_keys=True).encode())
    return h.hexdigest()


def _active_router():
    router = pipeline.load_active_router()
    if router is None:
        pytest.skip("committed router artifacts failed to load; assemble "
                    "models/parts first (scripts/assemble_models.py)")
    return router


def test_b6_no_artifact_byte_parity():
    """Bar B6: every classify output byte-identical to the shipped bytes
    on the committed fixture set, with the extractor FORCED ABSENT through
    the loader's negative-cache seam (runs on any checkout, staged
    candidate or not). Skips LOUDLY when the committed router artifact
    cannot load (the bar must not silently vanish), never on a digest
    mismatch — a mismatch here IS the regression."""
    router = _active_router()
    rows = [json.loads(line)
            for line in (ROOT / "tests/fixtures/routing-golden.jsonl")
            .read_text().splitlines() if line.strip()]
    assert rows, "golden fixture is empty"
    saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
    span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = None, True
    try:
        assert _digest_classify(router, [r["text"] for r in rows]) \
            == FIXTURE_DIGEST, (
            "B6 byte-parity broke on the committed fixture set: classify "
            "outputs moved without a promoted artifact")
    finally:
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = saved


def test_b6_catalogue_parity_without_artifact():
    """B6 over the committed catalogue case set (extractor FORCED ABSENT):
    numeric fills, the {} honest minimum for rows the deterministic-only
    admission cannot complete ("restart nginx on staging" stays on the
    veto's {} path), and the one legitimately moved row — the
    ladder-complete "service=nginx host=web-01" catalogue now CALLs with
    verbatim arguments (deterministic-only family admission, 2026-10-10;
    digest re-pinned from the measured candidate, never hand-edited)."""
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()

    def catalogue(tool, request):
        return (f"Tools available:\n[{json.dumps(tool)}]"
                f"\n\nUser request: {request}")

    int_tool = {"name": "scale_pool", "description":
                "Resize a database connection pool to the requested size.",
                "parameters": {"properties": {
                    "pool": {"type": "string"},
                    "size": {"type": "integer"}},
                    "required": ["pool", "size"]}}
    cases = [
        catalogue(NUM_TOOL, "calculate sum of 10 and 20"),
        catalogue(NUM_TOOL, "calculate the sum"),
        catalogue(NUM_TOOL, 'please sum {"a": 3, "b": 4} now'),
        catalogue(NUM_TOOL, "add b=7 and a=5 please"),
        catalogue(int_tool, "set size=8 for pool orders-primary"),
        catalogue(STR_TOOL, "restart nginx on staging"),
        catalogue(STR_TOOL, "service=nginx host=web-01 restart now"),
    ]
    saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
    span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = None, True
    try:
        assert _digest_classify(_active_router(), cases) \
            == CATALOGUE_DIGEST, (
            "B6 byte-parity broke on the catalogue case set")
    finally:
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = saved


# --- family gate -------------------------------------------------------------

def test_family_gate_admits_strings_without_artifact():
    """Deterministic-only family admission (2026-10-10, owner-approved):
    all-string-required tools are admitted with the extractor FORCED
    ABSENT through the loader's negative-cache seam. The ladders fill
    exact-when-they-fire; a row they cannot complete comes back UNHANDLED
    (the caller's fillability-veto path keeps it) — never a partial,
    never a new clarification."""
    saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
    span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = None, True
    try:
        assert span_fill.load_spanfill_extractor() is None
        assert span_fill.family_for(STR_TOOL) == "string"
        assert span_fill.family_for(ENUM_TOOL) == "string"
        # No ladder evidence for either slot: unhandled, not a partial.
        assert span_fill.fill_arguments(
            STR_TOOL, "restart nginx on staging") == (None, [])
        # Ladder-complete rows emit every required slot verbatim.
        assert span_fill.fill_arguments(
            STR_TOOL, "service=nginx host=staging") == (
            {"service": "nginx", "host": "staging"}, [])
    finally:
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = saved


def test_family_gate_admits_strings_with_artifact(with_mock_artifact):
    assert span_fill.family_for(STR_TOOL) == "string"
    assert span_fill.family_for(ENUM_TOOL) == "string"


def test_family_gate_mixed_boolean_numeric_untouched(with_mock_artifact):
    """Mixed required sets and boolean-required stay UNADMITTED even with
    a loadable extractor; the numeric family answer is identical with and
    without one."""
    assert span_fill.family_for(MIX_TOOL) is None
    assert span_fill.family_for(BOOL_TOOL) is None
    assert span_fill.family_for(NUM_TOOL) == "numeric"
    assert span_fill.fill_arguments(NUM_TOOL, "calculate sum of 10 and 20") \
        == ({"a": 10, "b": 20}, [])


# --- decode: ladders first, exclusivity, strict coercion ---------------------

def test_ladders_keep_first_claim_learned_fills_the_rest(with_mock_artifact):
    """service=nginx is a key-delimited claim; the learned scorer must
    fill only what it leaves open (host), never re-claim the ladder's
    span."""
    filled, missing = span_fill.fill_arguments(
        STR_TOOL, "service=nginx restart on the staging box")
    assert filled["service"] == "nginx"
    assert "service" not in missing


def test_span_exclusivity_loser_stays_missing(with_mock_artifact):
    """One verbatim span, two slots: the higher-confidence claim wins, the
    loser stays missing — never duplicated, never fabricated."""
    tool = {"name": "t", "parameters": {"properties": {
        "alpha": {"type": "string"}, "beta": {"type": "string"}},
        "required": ["alpha", "beta"]}}
    # MockScorer scores only the LAST token: "target" goes to whichever
    # slot sorts first by (confidence, insertion) — the other stays
    # missing, and the two slots never share a value span.
    filled, missing = span_fill.fill_arguments(
        tool, "run alpha and beta on target")
    assert len(filled) <= 1
    assert sorted(set(missing) | set(filled)) == ["alpha", "beta"]


def test_enum_match_emits_declared_form(with_mock_artifact):
    filled, missing = span_fill.fill_arguments(
        ENUM_TOOL, "ship release=2026.40 to environment=staging")
    assert filled == {"release": "2026.40", "environment": "staging"}
    assert missing == []


def test_enum_mismatch_refused_slot_stays_missing(with_mock_artifact):
    """The FILLED-BUT-INVALID rule: `environment=prod` (not a declared
    enum member) is present but unusable — the slot stays missing instead
    of the learned scorer donating a different span into it."""
    filled, missing = span_fill.fill_arguments(
        ENUM_TOOL, "release=2026.40 environment=prod")
    assert filled == {"release": "2026.40"}
    assert missing == ["environment"]


def test_filled_but_invalid_not_donated_even_when_a_valid_span_exists(
        with_mock_artifact):
    """The FILLED-BUT-INVALID rule holds when the request DOES carry
    another enum-valid span: the ladder said `environment=prod` (invalid);
    a learned span elsewhere (`staging`) must not be donated into the
    slot — it stays missing (review 2026-10-10: the donation case)."""
    class _StagingScorer:
        def pair_probs(self, request, param, prop):
            return [1.0 if m.group(0) == "staging" else 0.0
                    for m in re.finditer(r"\w+", request)]

    saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
    saved_floors = (span_fill._SPANFILL_EMIT_FLOOR,
                    span_fill._SPANFILL_EXTEND_FLOOR)
    span_fill._SPANFILL_CLF = _StagingScorer()
    span_fill._SPANFILL_LOADED = True
    span_fill.configure_spanfill_floors(0.6, 0.15)
    try:
        filled, missing = span_fill.fill_arguments(
            ENUM_TOOL, "release=2026.40 environment=prod deploy to staging")
    finally:
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = saved
        span_fill.configure_spanfill_floors(*saved_floors)
    assert filled == {"release": "2026.40"}
    assert missing == ["environment"]


def test_abstain_on_absence_emits_clarification():
    """A genuinely missing string value abstains: the served path emits
    the existing missing-arguments clarification naming the slot. A null
    scorer (nothing above the emit floor) stands in for a candidate that
    correctly refuses to invent."""
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()
    assert "tool_clarification" in registry.active()

    class _NullScorer:
        def pair_probs(self, request, param, prop):
            return [0.0] * len(re.findall(r"\w+", request))

    saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
    saved_floors = (span_fill._SPANFILL_EMIT_FLOOR,
                    span_fill._SPANFILL_EXTEND_FLOOR)
    span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = _NullScorer(), True
    span_fill.configure_spanfill_floors(0.6, 0.15)
    try:
        catalogue = (f"Tools available:\n[{json.dumps(STR_TOOL)}]"
                     "\n\nUser request: service=nginx restart it please")
        obj, meta = pipeline.classify(catalogue)
    finally:
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = saved
        span_fill.configure_spanfill_floors(*saved_floors)
    assert obj["type"] == "tool_clarification", obj
    assert obj["reason"] == "missing_arguments"
    assert obj["missing_parameters"] == ["host"]
    assert meta["valid"] is True


def test_served_call_carries_verbatim_arguments(with_mock_artifact):
    """User story 1/28: the restart-service probe emits a CALL with
    verbatim string arguments — compact strict JSON, no repair needed."""
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()
    catalogue = (f"Tools available:\n[{json.dumps(STR_TOOL)}]"
                 "\n\nUser request: service=nginx host=web-01 restart now")
    obj, meta = pipeline.classify(catalogue)
    assert obj["type"] == "tool_call", obj
    assert json.loads(obj["arguments"]) == {"service": "nginx",
                                            "host": "web-01"}
    assert " " not in obj["arguments"]
    assert meta["valid"] is True


# --- builder gates (a data gate that only prints is vacuous) -----------------

def _builder():
    import build_stringparam_corpus as b

    return b


# (test_builder_minimum_contribution_failure retired with the public-data
# cut: even with the build fns patched, the builder's main() composes the
# audit over inputs the open-source tree does not redistribute. The
# volume-gate law stays in the builder; the private tree keeps the gate.)


# (test_builder_eval_window_exclusion_by_normalized_set retired with the
# public-data cut: it replayed BFCL eval windows from data/incoming,
# which the open-source tree does not redistribute. The exclusion law
# lives in the builder — scripts/build_stringparam_corpus.py — which
# ships as the recipe, and the private tree keeps the gate.)

def test_builder_silver_carries_alignment_evidence():
    """Every ANSWERABLE silver row records per-slot evidence: slot,
    pattern, span offsets, raw text and the gold value, with the exact
    verbatim span equality as the mechanical re-verification.

    2026-10-10 (tiebreak extension, wave B2): the committed corpus is the
    blind double-adjudicated silver, so evidence pattern is "adjudicated"
    (the labelers' verbatim minimal span) — the scaffold's
    ladder-pattern pin (key_delimited/alignment + ladder re-derivation)
    described the pre-adjudication builder and no longer describes the
    committed rows. Unanswerable rows carry no evidence by construction:
    their gold is empty and the row teaches abstention."""
    path = ROOT / "data/real/spanfill-stringparam/bfcl-silver.jsonl"
    if not path.is_file():
        pytest.skip("corpus not emitted (builder gate failed); run "
                    "scripts/build_stringparam_corpus.py --build "
                    "--emit-despite-shortfall for the scaffold artifact")
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    assert rows, "silver file exists but is empty"
    answerable = 0
    for row in rows:
        if row.get("unanswerable"):
            assert row["evidence"] == [] and row["gold"] == {}, \
                "an unanswerable row must carry no evidence and no gold"
            continue
        assert row["evidence"], "answerable silver row without evidence"
        assert set(row["evidence"][0]) >= {
            "slot", "pattern", "span", "raw", "gold"}
        assert row["evidence"][0]["pattern"] == "adjudicated"
        s, e = row["evidence"][0]["span"]
        # verbatim minimal span: the request slice IS the gold (exact —
        # gold values that themselves contain quotes must survive intact)
        assert row["request"][s:e] == row["evidence"][0]["gold"]
        answerable += 1
    assert answerable > 0, "no answerable silver rows"


def test_builder_provenance_idempotent(tmp_path, monkeypatch):
    """The provenance writer replaces its own sections — a re-run never
    stacks duplicate entries (AGENTS.md rule 12 corollary)."""
    b = _builder()
    prov = tmp_path / "PROVENANCE.md"
    monkeypatch.setattr(b, "OUT", tmp_path)
    b.write_provenance({"src": "- alpha\n"}, "# header\n")
    first = prov.read_text()
    b.write_provenance({"src": "- alpha\n"}, "# header\n")
    b.write_provenance({"other": "- beta\n"}, "# header\n")
    second = prov.read_text()
    assert second.count("<!-- section:src -->") == 1
    assert second.count("<!-- section:other -->") == 1
    assert first.count("<!-- section:src -->") == 1
    assert "<!-- section:src -->\n- alpha\n" in second


def test_builder_dedupe_keeps_first_and_reports_frozen_overlap():
    b = _builder()
    rows = [{"request": "restart nginx on staging"},
            {"request": "Restart, NGINX  on STAGING!"},  # strong-normalized dup
            {"request": "restart nginx on the staging host"}]
    kept, stats = b.dedupe_check(rows, {"restart nginx on staging"})
    assert [r["request"] for r in kept] == ["restart nginx on staging",
                                            "restart nginx on the staging host"]
    assert stats["duplicates"] == 1 and stats["kept"] == 2
    assert stats["in_frozen_mix"] == 1


def test_harvest_admission_filter():
    """The mechanical filter: an invocation JSON is NOT a request (the
    committed harvest slice carries no user turns, so such rows refuse);
    a natural-language request admitting every verbatim value passes."""
    b = _builder()
    invocation = json.dumps({"name": "read_file",
                             "arguments": "{\"path\": \"x.md\"}"})
    row, reason = b._admit_harvest_row(invocation, {
        "type": "tool_call", "name": "read_file",
        "arguments": "{\"path\": \"x.md\"}"})
    assert row is None and reason == "no_request_turn"
    row, reason = b._admit_harvest_row(
        "please read the file x.md now", {
            "type": "tool_call", "name": "read_file",
            "arguments": "{\"path\": \"x.md\"}"})
    assert row is not None and reason == ""
    assert row["gold"] == {"path": "x.md"}
    assert row["tool"]["parameters"]["required"] == ["path"]
    row, reason = b._admit_harvest_row(
        "please read some other file", {
            "type": "tool_call", "name": "read_file",
            "arguments": "{\"path\": \"x.md\"}"})
    assert row is None and reason == "value_absent_from_request"


def test_committed_corpus_records_the_unblocked_adjudicated_verdict():
    """The committed composition report records the UNBLOCKED, adjudicated
    corpus: 2026-10-10 (tiebreak extension, wave B2) the 1,043 admitted
    silver rows meet the declared 1,000 minimum (never lowered) via the
    registered third-pass blind adjudication, and the harvest-gold source
    is DROPPED at 0 admissible rows (measured: the committed harvest
    slice carries no user turns) — a resurrected harvest-gold file would
    be an audit failure, never silent weight. The scaffold's
    data_blocked=true + shortfall-wording pin described the wave-A corpus
    and no longer describes reality; the frozen split counts and the
    reserve sha pin are unchanged and still asserted."""
    report_path = ROOT / "data/real/spanfill-stringparam/" \
                        "composition-report.json"
    if not report_path.is_file():
        pytest.skip("corpus not emitted yet")
    report = json.loads(report_path.read_text())
    assert report["data_blocked"] is False
    assert report["shortfalls"] == []
    silver = report["bfcl-silver"]
    assert silver["dedupe"]["kept"] >= 1000, \
        "admitted silver below the declared minimum (never lowered)"
    assert report["composition"]["per_source"]["bfcl-silver"] \
        == silver["dedupe"]["kept"]
    assert report["harvest-gold"]["admitted"] == 0
    assert report["composition"]["per_source"]["harvest-gold"] == 0
    assert report["frozen"]["counts"] == {"train": 12407, "dev": 1402,
                                          "reserve": 3191}
    # the frozen reserve sha is pinned inside the emitted report itself
    assert re.fullmatch(r"[0-9a-f]{64}", report["frozen"]["reserve_sha256"])


def test_committed_corpus_files_match_their_sidecars():
    out = ROOT / "data/real/spanfill-stringparam"
    if not out.is_file() and not (out / "bfcl-silver.jsonl").is_file():
        pytest.skip("corpus not emitted yet")
    for name in ("bfcl-silver.jsonl", "harvest-gold.jsonl"):
        path = out / name
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert (out / (name + ".sha256")).read_text().strip() == sha


def test_norm_matches_the_strong_normalization_contract():
    b = _builder()
    assert b.norm("Restart, NGINX  on STAGING!") \
        == b.norm("restart nginx on staging")


# --- whitespace-bounded span extension (registered increment, 2026-10-10) ----
#
# The wave-B3 falsification (reports/training-report.md, B1 one-shot):
# 680/680 reserve inventions were partial-span truncations of glued values
# (`prod-eu` emitted for `prod-eu-1`) — the decode extended across adjacent
# `\w+` tokens only. Measured mechanism (.scratch/incr2/probe-dev-glued.json,
# dev split): the per-token max-pool dropped every subtoken straddling a
# glue boundary (XLM-R subtoken `-1` spans the hyphen, fails the strict
# containment test) so the trailing segment read 0.0 and the extend floor
# stopped the span — 137/137 dev truncations. The registered increment:
# pool subtokens by OVERLAP (glue characters belong to the value when
# glued) and bound the extension by whitespace instead of token adjacency.

GLUED_TOOL = {"name": "migrate_stack", "description": "Migrate a stack.",
              "parameters": {"properties": {
                  "host": {"type": "string"},
                  "order": {"type": "string"}},
                  "required": ["host", "order"]}}


class _GluedScorer:
    r"""Scores the `\w+` segments of a slot's glued value 0.9 and every
    other token 0.01 — the trained model's measured texture on glued
    values (value segments high, context ~0.01)."""

    def __init__(self, values):
        self._segments = {param: set(re.findall(r"\w+", v))
                          for param, v in values.items()}

    def pair_probs(self, request, param, prop):
        seg = self._segments.get(param, set())
        return [0.9 if m.group(0) in seg else 0.01
                for m in re.finditer(r"\w+", request)]


class _Inject:
    """Injection through the trainer's seam, always restored."""

    def __init__(self, scorer, emit=0.6, extend=0.15):
        self._scorer = scorer
        self._floors = (emit, extend)

    def __enter__(self):
        self._saved = (span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED)
        self._saved_floors = (span_fill._SPANFILL_EMIT_FLOOR,
                              span_fill._SPANFILL_EXTEND_FLOOR)
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = \
            self._scorer, True
        span_fill.configure_spanfill_floors(*self._floors)
        return self._scorer

    def __exit__(self, *exc):
        span_fill._SPANFILL_CLF, span_fill._SPANFILL_LOADED = self._saved
        span_fill.configure_spanfill_floors(*self._saved_floors)
        return False


def test_glued_values_decode_full_span_not_wplus_prefix(with_mock_artifact):
    r"""The registered fixture: hyphen/dot values decode as the FULL glued
    span (`prod-eu-1`, `A-100.B`), never the `\w+` prefix (`prod-eu`)."""
    with _Inject(_GluedScorer({"host": "prod-eu-1", "order": "A-100.B"})):
        filled, missing = span_fill.fill_arguments(
            GLUED_TOOL,
            "migrate the stack to host prod-eu-1 tracking order A-100.B")
    assert filled == {"host": "prod-eu-1", "order": "A-100.B"}, filled
    assert missing == []


def test_whitespace_terminates_learned_span(with_mock_artifact):
    """Whitespace is a HARD span boundary in both directions: the decode
    never crosses a space to absorb a neighboring token, however
    confident. (The old adjacency loop extended across whitespace and
    absorbed the whole clause — the over-extension half of the same
    defect the falsified candidate exhibited.)"""
    tool = {"name": "search", "parameters": {"properties": {
        "query": {"type": "string"}}, "required": ["query"]}}

    class _AllHigh:
        def pair_probs(self, request, param, prop):
            return [0.9] * len(re.findall(r"\w+", request))

    with _Inject(_AllHigh()):
        filled, missing = span_fill.fill_arguments(
            tool, "nginx staging restart")
    assert filled == {"query": "nginx"}, filled
    assert missing == []


def test_extend_floor_still_gates_inside_glue(with_mock_artifact):
    """Emit/extend floor semantics are UNCHANGED: a glued segment below
    the extend floor still terminates the span (abstain over fabricate —
    the low-confidence `eu` is not silently absorbed)."""
    tool = {"name": "deploy", "parameters": {"properties": {
        "host": {"type": "string"}}, "required": ["host"]}}

    class _DipScorer:
        def pair_probs(self, request, param, prop):
            return [0.9 if m.group(0) in ("prod", "1") else
                    0.05 if m.group(0) == "eu" else 0.01
                    for m in re.finditer(r"\w+", request)]

    with _Inject(_DipScorer(), extend=0.15):
        filled, missing = span_fill.fill_arguments(
            tool, "ship it to host prod-eu-1 now")
    # The floor gated the extension at the dip: the span stays `prod` —
    # the falsification's partial-span texture (`prod-eu` for
    # `prod-eu-1`), never a fabricated continuation.
    assert filled == {"host": "prod"}, filled
    assert missing == []


def test_emit_floor_semantics_unchanged_on_glued_values(with_mock_artifact):
    """A glued value whose BEST segment sits below the emit floor is left
    missing entirely — no partial claims, no fabrication."""
    tool = {"name": "deploy", "parameters": {"properties": {
        "host": {"type": "string"}}, "required": ["host"]}}

    class _LowScorer:
        def pair_probs(self, request, param, prop):
            return [0.5 if m.group(0) in ("prod", "eu", "1") else 0.01
                    for m in re.finditer(r"\w+", request)]

    with _Inject(_LowScorer(), emit=0.6):
        filled, missing = span_fill.fill_arguments(
            tool, "ship it to host prod-eu-1 now")
    assert filled == {} and missing == ["host"]


def test_span_exclusivity_on_glued_multi_token_values(with_mock_artifact):
    """Span exclusivity holds on glued spans: two slots seeded in ONE
    glued value — the higher-confidence claim wins the WHOLE span, the
    loser stays missing (never a split, never a duplicate)."""
    tool = {"name": "t", "parameters": {"properties": {
        "alpha": {"type": "string"}, "beta": {"type": "string"}},
        "required": ["alpha", "beta"]}}
    with _Inject(_GluedScorer({"alpha": "web-01", "beta": "web-01"})):
        filled, missing = span_fill.fill_arguments(
            tool, "restart the web-01 probe")
    # One winner takes the whole glued span (confidence order decides
    # which), the loser stays missing — never a split, never a duplicate.
    assert len(filled) == 1, filled
    assert "web-01" in filled.values(), filled
    assert sorted(list(filled) + list(missing)) == ["alpha", "beta"]


def test_glued_subtoken_pools_onto_word_token():
    """The decode's per-token pooling attributes a subtoken straddling a
    glue boundary to the word token it overlaps: the measured mechanism
    of the falsification. XLM-R renders `app-07` as `app` + `-07`; the
    old strict-containment test (`s >= m.start()`) dropped `-07` from the
    token `07`, which read 0.0 and stopped the extend at the hyphen."""
    import math
    from types import SimpleNamespace

    request = "connect to app-07 now"
    # \w+ tokens: connect(0,7) to(8,10) app(11,14) 07(15,17) now(18,21)
    # subtokens: the model's evidence for the glued digit rides on `-07`
    # (14,17) — a glue-spanning subtoken.
    offsets = [(0, 7), (8, 10), (11, 14), (14, 17), (18, 21)]
    probs = [0.01, 0.01, 0.99, 0.98, 0.01]
    logits = [math.log(p / (1.0 - p)) for p in probs]

    class _FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name=n) for n in
                    ("input_ids", "attention_mask", "token_type_ids")]

        def run(self, _none, feed):
            return [[logits]]

    class _FakeTokenizer:
        def encode(self, text, pair=None):
            # sequence_ids added 2026-10-10 (bug-9 review): membership now
            # comes from sequence_ids, not type_ids; all five fake tokens
            # are request-side (sequence 0). Assertions unchanged.
            return SimpleNamespace(ids=[1, 2, 3, 4, 5], type_ids=[0] * 5,
                                   sequence_ids=[0] * 5,
                                   attention_mask=[1] * 5,
                                   offsets=offsets)

    scorer = span_fill._EncoderPairScorer(_FakeSession(), _FakeTokenizer())
    tokens = [m.group(0) for m in re.finditer(r"\w+", request)]
    out = scorer.pair_probs(request, "host", {"type": "string"})
    got = dict(zip(tokens, out))
    assert got["07"] >= 0.9, (
        "the glue-spanning subtoken `-07` was dropped from token `07` — "
        "the falsified candidate's truncation mechanism")
    assert got["app"] >= 0.9
    assert got["connect"] <= 0.1 and got["now"] <= 0.1


def test_gold_span_requires_verbatim_presence():
    """The trainer's mechanical honesty: a gold value the request does not
    carry verbatim is dropped, never fabricated."""
    train = _builder  # noqa: F841 (naming the module for readers)
    import importlib

    tr = importlib.import_module("train_stringparam_scorer")
    assert tr.gold_char_span("restart nginx on staging", "nginx") == (8, 13)
    assert tr.gold_char_span("restart the service", "nginx") is None
