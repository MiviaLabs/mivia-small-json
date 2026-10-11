"""Survivor low-severity fixes (survfix 2026-10-10, validate-b round).

Three confirmed survivors, one failing test each before the fix:

1. LOW  — ``diff_comment._count_patch_changes`` hunk-less branch: any
   ``--- ``/``+++ `` line flipped ``in_content`` and was skipped, so a
   deleted line whose ORIGINAL content renders ``--- old text`` was
   miscounted (v13 D1) and content before an email-style ``--- ``
   separator was dropped (v13 D2).
2. MINOR — ``logtemplate.TemplateCache.match`` runs ``Drain.add`` which
   can WIDEN the cluster template; the widened string was never in
   ``_known``, so a known cluster read as a miss (v14) until a re-learn
   healed it, and the pre-widening key stayed orphaned forever.
3. HYGIENE — ``registry.register_schema(description=None)`` stored the
   literal description ``"None"`` with ``description_explicit=True``
   (v10): the schema opted into zero-shot selection scored against
   ``"name: None"``.

The hunk-less semantics pinned here are a CHOICE, not a truth: a patch
with no ``@@`` markers is intrinsically ambiguous. Pinned: only the
FIRST ``--- ``-shaped and FIRST ``+++ ``-shaped lines are the header
block; every later line follows the +/- body rules (a bare ``---`` rule
line is still never a deletion); hunk-tracked (@@) counting is
untouched.
"""


from mivia_small_json import registry
from mivia_small_json.diff_comment import extract_diff
from mivia_small_json.logtemplate import TemplateCache


# --- 1. hunk-less diff counting (v13 D1 / D2) -------------------------------


def test_hunk_less_deleted_line_rendering_as_dashes_counts():
    """D1: a deleted line whose original content was '-- old text'
    renders as '--- old text' -- inside the body it is a deletion, not a
    second header."""
    obj = extract_diff(
        "--- a/notes.md\n"
        "+++ b/notes.md\n"
        "--- old text\n"
        "+new text\n"
    )
    assert obj["additions"] == 1, obj["additions"]
    assert obj["deletions"] == 1, obj["deletions"]


def test_hunk_less_content_before_sig_separator_counts():
    """D2: a bare +/- patch with an email-style '--- ' separator: the
    +/- lines BEFORE the separator count; the separator itself counts as
    neither an addition nor a deletion."""
    obj = extract_diff(
        "-removed line one\n"
        "+added line one\n"
        "--- \n"
        "+added line two\n"
    )
    assert obj["additions"] == 2, obj["additions"]
    assert obj["deletions"] == 1, obj["deletions"]


def test_hunk_less_git_header_section_same_pair_semantics():
    """Same pinned pair semantics when the section carries the usual
    'diff --git' / 'index' preamble lines before the header block."""
    obj = extract_diff(
        "diff --git a/notes.md b/notes.md\n"
        "index 1111111..2222222 100644\n"
        "--- a/notes.md\n"
        "+++ b/notes.md\n"
        "--- old text\n"
        "+new text\n"
    )
    assert obj["additions"] == 1
    assert obj["deletions"] == 1


def test_hunk_less_second_plus_plus_line_follows_body_rules():
    """Pinned ambiguity: '+++ b/g' AFTER the header block is governed by
    the addition rules (it starts with '+'), not skipped as a header."""
    obj = extract_diff("--- a/f\n+++ b/f\n+++ b/g\n+x\n")
    assert obj["additions"] == 2
    assert obj["deletions"] == 0


def test_hunk_less_bare_rule_line_still_not_a_deletion():
    """Preserved from the pre-fix branch: a bare '---' rule line is
    neither a header nor a deletion."""
    obj = extract_diff("--- a/f.py\n+++ b/f.py\n---\n+ok\n")
    assert obj["additions"] == 1
    assert obj["deletions"] == 0


def test_hunk_tracked_dashes_inside_hunk_still_a_deletion():
    """Hunk-tracked (@@) counting is byte-for-byte unchanged: inside a
    hunk, '--- old' is a deleted line (original content '-- old')."""
    obj = extract_diff(
        "--- a/f\n+++ b/f\n@@ -1,1 +1,1 @@\n--- old\n+new\n"
    )
    assert obj["additions"] == 1
    assert obj["deletions"] == 1


def test_hunk_less_dev_null_add_still_counts_and_paths():
    """Preserved: the /dev/null header path and its counting."""
    obj = extract_diff("--- /dev/null\n+++ b/made.py\n+x\n-y\n")
    assert obj["change_type"] == "add"
    assert obj["new_path"] == "made.py"
    assert obj["additions"] == 1
    assert obj["deletions"] == 1


# --- 2. TemplateCache cluster-widening key orphaning (v14) ------------------

_T1 = "INFO auth: user alice logged in"
_T2 = "INFO auth: user bob logged in"
_T3 = "INFO auth: user carol logged in"


def test_widened_cluster_match_hits_without_relearn():
    """THE defect: Drain widens 'alice' to '<*>' on the second line; the
    widened template must resolve to the learned schema immediately, not
    read as a miss until a re-learn heals the key."""
    cache = TemplateCache()
    t1, _ = cache.match(_T1)
    cache.learn(t1, "log_event")

    t2, schema = cache.match(_T2)
    assert t2 != t1  # sanity: the cluster really widened
    assert schema == "log_event", (
        "known cluster read as a miss after widening")


def test_widened_cluster_probe_hits_read_only():
    """The read-only probe hits the known cluster too (probe-hit routing
    keeps working). ``peek`` never merges, so it returns the cluster's
    current — still pre-widening — template, resolved from ``_known``;
    it must not add Drain clusters or cache entries."""
    cache = TemplateCache()
    t1, _ = cache.match(_T1)
    cache.learn(t1, "log_event")
    clusters_after_learn = len(cache._drain.clusters)

    probe_t, probe_schema = cache.probe(_T2)
    assert probe_t == t1  # peek does not widen
    assert probe_schema == "log_event"
    assert len(cache._drain.clusters) == clusters_after_learn
    assert len(cache) == 1  # a probe never adds a known entry

    # After a real match widened the cluster, the probe resolves the
    # widened string through the cluster alias as well.
    widened, _ = cache.match(_T2)
    probe_t2, probe_schema2 = cache.probe(_T2)
    assert probe_t2 == widened != t1
    assert probe_schema2 == "log_event"


def test_relearning_widened_template_does_not_orphan_the_old_key():
    """Re-key: after the widened string is learned, the cache holds ONE
    resolvable entry for the cluster -- the pre-widening key is folded
    into the same cluster id, not left orphaned forever."""
    cache = TemplateCache()
    t1, _ = cache.match(_T1)
    cache.learn(t1, "log_event")
    t2, _ = cache.match(_T2)
    cache.learn(t2, "log_event")

    assert len(cache) == 1, sorted(cache._known)
    assert cache.match(_T1) == (t2, "log_event")
    assert cache.match(_T2) == (t2, "log_event")
    assert cache.match(_T3) == (t2, "log_event")


def test_cache_baseline_contracts_unchanged():
    """Public contracts kept: miss on an unknown cluster, learn-only
    admission of truthy pairs, hit routing after learn."""
    cache = TemplateCache()
    assert cache.match(_T1) == (
        "INFO auth: user alice logged in", None)
    cache.learn("", "log_event")  # falsy template dropped
    cache.learn(_T1, "")          # falsy schema dropped
    assert len(cache) == 0
    t1, _ = cache.match(_T1)
    cache.learn(t1, "log_event")
    assert cache.match(_T1) == (t1, "log_event")
    assert cache.probe(_T1) == (t1, "log_event")


# --- 3. register_schema description=None (v10) ------------------------------

_SPEC = {"type": {"const": "scratch_desc"}, "name": {"type": str}}
_ROUTING = {"strong": [r"\bscratch desc\b"], "weak": []}


def _register_scratch(description):
    registry.register_schema(
        "scratch_desc",
        spec=dict(_SPEC),
        routing={"strong": list(_ROUTING["strong"]), "weak": []},
        extraction={},
        defaults={"name": ""},
        description=description,
    )
    return registry._REGISTERED["scratch_desc"]


def test_description_none_gets_field_list_and_stays_out_of_zero_shot():
    entry = _register_scratch(None)
    assert entry["description"] == "scratch_desc: name"
    assert entry["description_explicit"] is False
    assert "scratch_desc" not in registry.zero_shot_registered()
    assert registry.description_for("scratch_desc") == "scratch_desc: name"


def test_description_non_string_never_stringified():
    """int / list descriptions must not become '42'-style strings with
    description_explicit=True; they are treated as not supplied."""
    for bad in (42, ["widget thing"], 3.14):
        entry = _register_scratch(bad)
        assert entry["description"] == "scratch_desc: name", bad
        assert entry["description_explicit"] is False, bad
        registry.unregister("scratch_desc")
    assert "scratch_desc" not in registry.zero_shot_registered()


def test_description_empty_or_whitespace_gets_field_list():
    for empty in ("", "   "):
        entry = _register_scratch(empty)
        assert entry["description"] == "scratch_desc: name"
        assert entry["description_explicit"] is False
        registry.unregister("scratch_desc")


def test_description_valid_string_byte_identical():
    entry = _register_scratch("  a scratch widget for tests  ")
    assert entry["description"] == "a scratch widget for tests"
    assert entry["description_explicit"] is True
    assert "scratch_desc" in registry.zero_shot_registered()
