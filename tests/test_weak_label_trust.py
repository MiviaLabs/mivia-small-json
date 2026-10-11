"""Weak-label trust measured on development data only (SMAL-13).

Frozen audit data may measure final performance. It must never set a
threshold, a label, an admission decision, or a gate input.

Deterministic, offline, CPU-only, seconds.
"""

import hashlib
import json
from pathlib import Path

import pytest

from mivia_small_json import bootstrap, flywheel, label
from mivia_small_json.model import MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]
SYNTH_DEV = ROOT / "data" / "synthetic" / "dev.jsonl"

# The suite's committed shape: every collected node id, one per line.
# Regenerate deliberately with scripts/refresh-test-inventory.py.
TEST_INVENTORY = Path(__file__).parent / "test-inventory.txt"

LOG_LINE = ("2026-10-02T14:54:42.685056420Z [err]  \u2a2f TypeError: Unable "
            "to connect.")


def _rows(rel):
    return [json.loads(line) for line in
            (ROOT / rel).read_text(encoding="utf-8").splitlines() if line.strip()]


def _voted_texts(vote, n, exclude_gold=None):
    """Development texts the rules layer votes ``vote`` for."""
    out = []
    for row in _rows("data/synthetic/dev.jsonl"):
        if exclude_gold is not None and row["schema"] == exclude_gold:
            continue
        if label.rules_vote(row["raw"])[0] == vote:
            out.append(row["raw"])
        if len(out) == n:
            break
    assert len(out) == n, f"fixture drift: fewer than {n} texts voted {vote}"
    return out


def _write_dev(root, rows):
    """Write the frozen development reference a scratch root reads.

    ``dev_rows`` reads :data:`label.DEV_REFERENCE`, the committed frozen
    snapshot, not the growing dev files, so a test that wants a scratch
    trust basis records one here.
    """
    path = Path(root) / label.DEV_REFERENCE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in rows), encoding="utf-8")
    return path


# --- complete confusion matrix ------------------------------------------------


def test_false_positives_from_other_classes_count_against_the_voted_class(tmp_path):
    """One pooled matrix scores every class.

    Gold-issue rows that the rules vote log_event are false positives for
    log_event. A per-file score over log_event rows alone would miss them.
    """
    bootstrap.register_example_schemas()
    log_voted = _voted_texts("log_event", 40)
    issue_voted = _voted_texts("issue", 25)
    commit_voted = _voted_texts("commit", 25)
    rows = []
    # 20 gold log_event, voted log_event -> true positives.
    rows += [{"raw": t, "schema": "log_event"} for t in log_voted[:20]]
    # 20 gold issue, still voted log_event -> log_event false positives.
    rows += [{"raw": t, "schema": "issue"} for t in log_voted[20:40]]
    # 25 gold log_event, voted commit -> log_event false negatives.
    rows += [{"raw": t, "schema": "log_event"} for t in commit_voted]
    rows += [{"raw": t, "schema": "issue"} for t in issue_voted]
    _write_dev(tmp_path, rows)

    measure = label.measure_dev_rules(tmp_path)
    row = measure["per_class"]["log_event"]

    assert measure["confusion"]["log_event"]["log_event"] == 20
    assert row["support"] == 45
    assert row["false_positives"] == 20, "gold-issue rows voted log_event"
    assert row["abstentions"] == 0
    assert row["precision"] == pytest.approx(20 / 40, abs=1e-4)
    assert row["recall"] == pytest.approx(20 / 45, abs=1e-4)
    assert row["f1"] == pytest.approx(40 / 85, abs=1e-4)
    # Dropping the 20 false positives (scoring log_event rows alone) would
    # have reported precision 1.0 and F1 2*20/(2*20+25) = 0.6154.
    assert row["precision"] < 1.0
    assert row["f1"] < 2 * 20 / (2 * 20 + 25)
    assert row["status"] == "untrusted"

    issue = measure["per_class"]["issue"]
    assert issue["support"] == 45
    assert issue["false_positives"] == 0


def test_committed_dev_sets_produce_one_supported_trust_table(corpus_world):
    """Every class the artifact knows has labeled development support."""
    measure = label.measure_dev_rules(ROOT)
    assert measure["n"] == 2800
    assert sorted(measure["per_class"]) == ["agent_step", "commit", "issue",
                                             "log_event", "pull_request"]
    # agent_step ships with no labeled development rows yet, so its trust
    # is unknown (never admitted) while the four dev-covered classes keep
    # their measured verdicts.
    assert measure["unknown_classes"] == ["agent_step"]
    # Only log_event clears the bar on development data.
    assert measure["trusted_classes"] == ["log_event"]
    # Pin the counts the report prints, so the two cannot drift apart.
    assert {cls: row["false_positives"] for cls, row
            in measure["per_class"].items()} == {
        "agent_step": 0,
        "commit": 425, "issue": 222, "log_event": 13, "pull_request": 16}
    assert {cls: row["abstentions"] for cls, row
            in measure["per_class"].items()} == {
        "agent_step": 0,
        "commit": 13, "issue": 176, "log_event": 275, "pull_request": 0}
    for row in measure["per_class"].values():
        # 2026-10-09: the honest abstention-inclusive recall/F1 variants
        # are reported beside the gating ones (F10a), so the record grew
        # four reported-only fields.
        assert set(row) == {"precision", "recall", "f1", "support",
                            "recall_excl_abstentions", "f1_excl_abstentions",
                            "recall_incl_abstention_misses",
                            "f1_incl_abstention_misses",
                            "false_positives", "abstentions", "status"}


# --- frozen audit data cannot steer admission --------------------------------


def test_trust_path_never_opens_frozen_audit_files(monkeypatch):
    """No read of a frozen AUDIT file happens while trust is set.

    The frozen development reference is development data, so it is read.
    The frozen audit sets (data/heldout/test.jsonl and the data/wild
    slices) are not.

    ``pathlib.Path.read_text`` and ``.read_bytes`` both delegate to
    ``.open()``. Patching ``.open()`` alone therefore covers every read
    entry point this code base uses.
    """
    import pathlib

    real_open = pathlib.Path.open

    def guarded(self, *args, **kwargs):
        name = str(self).replace("\\", "/")
        # The frozen development reference is development data and is
        # read. The frozen AUDIT sets (data/heldout/test.jsonl, the
        # data/wild slices) are not.
        if name.endswith(label.DEV_REFERENCE):
            return real_open(self, *args, **kwargs)
        if "data/wild" in name or "data/heldout" in name:
            raise AssertionError(f"frozen audit file was read: {name}")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", guarded)

    measure = label.measure_dev_rules(ROOT)
    trust = label._trust(measure)
    records = [{"id": "r1", "source": "t", "raw": LOG_LINE},
               {"id": "r2", "source": "t", "raw": "zzz nothing to vote on"}]
    confident, hard, stats = label.build_pools(records, router=None,
                                               trust=trust, measure=measure)
    assert stats["trust"]["source"] == "development"
    assert stats["confident"] == 1 and stats["hard"] == 1


def test_frozen_audit_labels_cannot_alter_admission(tmp_path):
    """Relabel or rewrite the audit sets. Trust and pools must not move."""
    bootstrap.register_example_schemas()
    dev_rows = _rows("data/synthetic/dev.jsonl")
    _write_dev(tmp_path, dev_rows)
    baseline = label.measure_dev_rules(tmp_path)
    baseline_trust = label._trust(baseline)

    # Hostile frozen audit sets: every record claims a gold class that
    # would flip a trust verdict if it were ever read.
    hostile = [
        {"raw": LOG_LINE, "schema": "commit", "gold": {"type": "commit"}},
        {"raw": LOG_LINE, "schema": "issue", "gold": {"type": "issue"}},
        {"raw": "zzz nothing here", "schema": "pull_request",
         "gold": {"type": "pull_request"}},
    ] * 40
    for rel in ("data/wild/issue.jsonl", "data/wild/commit.jsonl",
                "data/wild/platform-logs.jsonl", "data/heldout/test.jsonl"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in hostile),
                        encoding="utf-8")

    after = label.measure_dev_rules(tmp_path)
    assert after == baseline, "frozen audit content changed the trust table"
    assert label._trust(after) == baseline_trust

    records = [{"id": "r1", "source": "t", "raw": LOG_LINE},
               {"id": "r2", "source": "t", "raw": "zzz nothing to vote on"}]
    # Measure against the tmp root, so the hostile audit files are in
    # scope of the pools we compare. Calling build_pools twice with no
    # varying input would compare the function with itself.
    hostile_trust = label._trust(label.measure_dev_rules(tmp_path))
    after_conf, after_hard, after_stats = label.build_pools(
        records, router=None, trust=hostile_trust,
        measure=label.measure_dev_rules(tmp_path))
    assert after_conf and after_conf[0]["schema"] == "log_event"
    assert after_stats["trust"] == label.trust_disclosure(baseline)
    assert len(after_hard) == 1


def test_empty_trust_map_still_trusts_nothing():
    """An explicit empty trust map is a policy, not a missing argument."""
    bootstrap.register_example_schemas()
    lab = label.label_record(LOG_LINE, router=None, trust={})
    assert lab["tier"] == "hard"
    assert lab["schema"] is None
    confident, _hard, stats = label.build_pools(
        [{"id": "r1", "source": "t", "raw": LOG_LINE}], router=None,
        trust={}, measure=label.measure_dev_rules(ROOT))
    assert confident == []


# --- insufficient support -----------------------------------------------------


def test_insufficient_development_support_is_unknown_and_rejects_tier2(tmp_path, corpus_world):
    """Too few labeled rows means unknown trust, and unknown admits nothing."""
    bootstrap.register_example_schemas()
    thin = _voted_texts("log_event", 4)
    _write_dev(tmp_path, [{"raw": t, "schema": "log_event"} for t in thin])
    measure = label.measure_dev_rules(tmp_path)

    row = measure["per_class"]["log_event"]
    assert row["support"] == 4 < label.MIN_DEV_SUPPORT
    assert row["status"] == "unknown"
    assert "log_event" in measure["unknown_classes"]
    assert label._trust(measure)["log_event"] is False

    lab = label.label_record(LOG_LINE, router=None, trust=label._trust(measure))
    assert lab["tier"] == "hard", lab
    assert lab["schema"] is None
    assert "tier2" not in lab["reason"]

    confident, _hard, stats = label.build_pools(
        [{"id": "r1", "source": "t", "raw": LOG_LINE}], router=None,
        trust=label._trust(measure), measure=measure)
    assert confident == []
    # Only log_event has any rows at all here; every class is unknown.
    assert sorted(stats["trust"]["unknown_classes"]) == [
        "agent_step", "commit", "issue", "log_event", "pull_request"]
    assert stats["trust"]["trusted_classes"] == []


def test_missing_development_data_leaves_every_class_unknown(tmp_path):
    """No labeled development data means no trust at all."""
    measure = label.measure_dev_rules(tmp_path)
    assert measure["n"] == 0
    assert measure["files"] == []
    assert all(row["status"] == "unknown"
               for row in measure["per_class"].values())
    assert not any(label._trust(measure).values())
    assert measure["trusted_classes"] == []


# --- probabilities are uncalibrated by default -------------------------------


def test_automatic_tier_is_disabled_without_a_development_calibration(monkeypatch):
    """A bare model probability must never admit a record."""
    from mivia_small_json.pipeline import load_router

    bootstrap.register_example_schemas()
    router = load_router(MODEL_PATH)
    # Without a calibration the model must not admit — the untrusted
    # issue class cannot fall back to tier 2 either. (Found before the
    # rules_vote stub below, which pins every vote to log_event.)
    # Keep tier 2 (rules-dominant) out of the way so this test is purely
    # about tier 1: a trusted log_event vote with a sub-dominant margin.
    monkeypatch.setattr(label, "rules_vote",
                        lambda t: ("log_event", 0.25))
    # 2026-10-09 (F1): tier-1 agreement is gated on the class's rule
    # trust, and issue is measured untrusted on the committed reference,
    # so the tier-1 leg of this test uses the trusted log_event class.
    text = "2026-02-24T14:19:00Z INFO [cache]: token validation failed for user 1516"

    default = label.label_record(text, router=router)
    assert default["tier1_enabled"] is False
    assert not default["reason"].startswith("tier1:")
    assert default["tier"] == "hard"
    assert default["proba"] is not None, "the vote is still reported"

    frozen = label.DevCalibration(source="data/wild/platform-logs.jsonl", n=13)
    assert label._tier1_enabled(frozen) is False
    assert label.label_record(text, router=router,
                              calibration=frozen)["tier1_enabled"] is False

    dev = label.DevCalibration(source="data/synthetic/dev.jsonl", n=350,
                               max_abs_error=0.02)
    assert label._tier1_enabled(dev) is True
    calibrated = label.label_record(text, router=router, calibration=dev)
    assert calibrated["reason"].startswith("tier1:"), calibrated["reason"]
    assert calibrated["tier"] == "confident"


@pytest.mark.parametrize("bad", [
    None,
    label.DevCalibration(source="data/wild/platform-logs.jsonl", n=13),
    label.DevCalibration(source="data/heldout/test.jsonl", n=350),
    label.DevCalibration(source="frozen/audit.jsonl", n=10),
    label.DevCalibration(source="data/synthetic/dev.jsonl", n=0),
    label.DevCalibration(source="data/eval.jsonl", n=10),
    label.DevCalibration(source="scored.jsonl", n=10),
])
def test_invalid_calibration_is_refused(bad):
    with pytest.raises((TypeError, ValueError)):
        label.check_calibration(bad)


def test_build_pools_refuses_a_non_development_calibration():
    with pytest.raises(ValueError):
        label.build_pools([], router=None,
                          calibration=label.DevCalibration(
                              source="data/wild/issue.jsonl", n=150))


def test_committed_calibration_matches_the_router_in_the_tree():
    """A retrain must not leave a calibration pointing at an old artifact."""
    bootstrap.register_example_schemas()
    record = json.loads((ROOT / "data" / "calibration.json")
                        .read_text(encoding="utf-8"))
    digest = (ROOT / "models" / "router.sha256").read_text(encoding="utf-8").strip()
    assert record["router_sha256"] == digest, (
        "router.skops changed without re-measuring the calibration")
    assert record["max_significant_error"] <= label.CALIBRATION_MAX_ERROR
    calibration = label.calibration_from_disk(ROOT)
    assert calibration is not None
    assert label._tier1_enabled(calibration) is True


def test_a_retrain_switches_tier_one_off(tmp_path):
    """A stored calibration is ignored once the artifact hash moves."""
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "router.sha256").write_text("deadbeef\n",
                                                      encoding="utf-8")
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "calibration.json").write_text(json.dumps({
        "source": "data/synthetic/dev.jsonl", "n": 750,
        "max_significant_error": 0.02, "min_probability": 0.90,
        "router_sha256": "cafebabe"}),
        encoding="utf-8")
    assert label.calibration_from_disk(tmp_path) is None

    (tmp_path / "models" / "router.sha256").write_text("cafebabe\n",
                                                      encoding="utf-8")
    restored = label.calibration_from_disk(tmp_path)
    assert restored is not None
    assert restored.n == 750


def test_measurement_needs_enough_rows_and_a_good_enough_router(tmp_path):
    from mivia_small_json.pipeline import load_router

    bootstrap.register_example_schemas()
    router = load_router(MODEL_PATH)
    with pytest.raises(ValueError, match="needs the router"):
        label.measure_dev_calibration(tmp_path, None)

    # An empty development root leaves too few rows to judge a router.
    with pytest.raises(ValueError, match="need"):
        label.measure_dev_calibration(tmp_path, router)

    record = label.measure_dev_calibration(ROOT, router)
    assert record["n"] >= label.CALIBRATION_MIN_ROWS
    assert record["min_probability"] == label.AGREE_MIN_PROBA
    assert "dev" in record["source"]
    # The measurement covers only the range tier 1 admits, so it must
    # exclude development rows below the admission probability. The
    # exact count belongs to the measured artifact; the invariant is
    # the range.
    total = len(label.dev_rows(ROOT)[0])
    assert label.CALIBRATION_MIN_ROWS <= record["n"] < total, (
        "sub-threshold rows leaked into the measurement")


def test_error_above_the_threshold_is_refused():
    with pytest.raises(ValueError, match="exceeds"):
        label.check_calibration(label.DevCalibration(
            source="data/synthetic/dev.jsonl", n=750, max_abs_error=0.5))


def _cycle_root(tmp_path, digest=None):
    """A temp root holding the development data, the frozen reference,
    and a real router.

    The artifact is copied in rather than faked, so a test can assert
    that the record is stamped with the hash of the file that was
    actually measured.
    """
    import shutil

    # model.main still reads the growing dev files; the trust and the
    # calibration read the frozen reference below.
    for rel, _gold in label.DEV_SOURCES:
        if not (ROOT / rel).is_file():
            # Public-release tree: only the public dev sources ship; the
            # record laws under test do not depend on which rows exist.
            continue
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, target)
    reference = tmp_path / label.DEV_REFERENCE
    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / label.DEV_REFERENCE, reference)
    (tmp_path / "models").mkdir(parents=True)
    shutil.copy(ROOT / "models" / "router.skops",
                tmp_path / "models" / "router.skops")
    (tmp_path / "models" / "router.sha256").write_text(
        (digest or (ROOT / "models" / "router.sha256")
         .read_text(encoding="utf-8").strip()) + "\n", encoding="utf-8")
    return tmp_path


def _digest_of(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_training_cycle_actually_calls_the_refresh(monkeypatch, tmp_path):
    """The refresh is wired into the accepted retrain path.

    The retrain branch is the one place the refresh matters, so a
    deleted call must fail a test. ``model.main`` is stubbed, so this
    never rewrites the real artifact.
    """
    import shutil

    import mivia_small_json.model as model_mod
    import mivia_small_json.pipeline as pipeline_mod
    from mivia_small_json.pipeline import load_router as real_load

    bootstrap.register_example_schemas()
    root = _cycle_root(tmp_path)
    for rel in ("data/heldout/test.jsonl", "data/wild/issue.jsonl",
                "data/wild/commit.jsonl", "data/wild/platform-logs.jsonl"):
        if not (ROOT / rel).is_file():
            continue  # public-release tree: this source does not ship
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, target)
    # The refresh only runs when the cycle accepts the artifact. Stub the
    # check so this test is about the refresh, not about the gate.
    monkeypatch.setattr(flywheel, "gates_pass", lambda *a, **k: (True, "stubbed"))

    router = real_load(MODEL_PATH)
    calls = []
    seen = {}
    # The stub must hand back a DIFFERENT object after training, or a
    # stale router is indistinguishable from a fresh one and the test
    # proves nothing about which model was measured.
    class _Proxy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

    loaded = []

    def fake_load(*a, **k):
        value = _Proxy(router) if loaded else router
        loaded.append(value)
        return value

    monkeypatch.setattr(model_mod, "main",
                        lambda *a, **k: calls.append("train"))
    monkeypatch.setattr(pipeline_mod, "load_router", fake_load)
    original = flywheel._refresh_calibration

    def spy(r, ro, report):
        seen["router"] = ro
        calls.append("refresh")
        return original(r, ro, report)

    monkeypatch.setattr(flywheel, "_refresh_calibration", spy)

    report = flywheel.run_cycle(root=root, do_train=True)

    assert "train" in calls, "the stubbed retrain did not run"
    assert "refresh" in calls, "the retrain cycle never refreshed the calibration"
    # The refresh must come after the artifact changes, not before.
    assert calls.index("refresh") > calls.index("train")
    # And it must measure the router the reload produced, not the one
    # loaded before training. Ordering alone does not prove this.
    assert len(loaded) >= 2, "the retrain cycle did not reload the router"
    assert seen["router"] is loaded[-1], (
        "the refresh measured a stale router")
    assert report["calibration_refreshed"]["status"] == "measured"
    # The record it wrote is valid for the artifact in this root.
    assert label.calibration_from_disk(root) is not None
    written = json.loads((root / label.CALIBRATION_PATH).read_text(encoding="utf-8"))
    assert written["router_sha256"] == _digest_of(
        root / "models" / "router.skops")
    # The two calibration fields describe different moments and must not
    # overwrite one another.
    assert "calibration" in report
    assert report["calibration"] is not report["calibration_refreshed"]
    assert report["calibration_refreshed"]["status"] == "measured"


def test_the_report_carries_the_gating_worst_band_itself(monkeypatch, tmp_path):
    """An accepted cycle replaces the record file, so the report must
    carry the gating worst band itself.

    A note that points at data/calibration.json names a file the
    refresh overwrites before anyone reads the report. The number that
    gated the cycle would then be nowhere at all.
    """
    import mivia_small_json.model as model_mod

    bootstrap.register_example_schemas()
    root = _cycle_root(tmp_path)
    _write_record(root, max_abs_error=0.0943)
    assert label.calibration_from_disk(root) is not None
    monkeypatch.setattr(model_mod, "main", lambda *a, **k: None)
    monkeypatch.setattr(flywheel, "gates_pass", lambda *a, **k: (True, "stubbed"))

    report = flywheel.run_cycle(root=root, do_train=True)

    block = report["calibration"]
    assert block["max_abs_error"] == pytest.approx(0.0943), (
        "the gating record's worst band must travel in the report")
    assert block["max_significant_error"] == pytest.approx(0.02)
    # The stale claim is that the number LIVES in the file; naming the
    # file to explain why the number travels here is fine.
    assert "sits in data/calibration.json" not in block["note"]


def test_a_reader_that_would_be_refused_returns_nothing(tmp_path):
    """The disk reader must agree with the strict check.

    Returning a record that `check_calibration` later refuses turns a
    readable file into a crash inside `build_pools`, which aborts a
    cycle rather than switching the tier off.
    """
    root = _cycle_root(tmp_path)
    # Only the error is wrong. The hash matches, so the refusal has to
    # come from the strict check and nowhere else. The raises branch is
    # covered by the parametrised refusal test.
    _write_record(root, max_significant_error=0.93)
    assert label.calibration_from_disk(root) is None


def _write_record(root, **over):
    # The hash must MATCH the sidecar, so the only reason for the reader
    # to refuse is the field under test. A mismatched hash would be
    # refused earlier, and the assertion would pass for the wrong reason.
    body = {"source": "data/synthetic/dev.jsonl", "n": 750,
            "max_significant_error": 0.02, "min_probability": 0.90,
            "router_sha256": (root / "models" / "router.sha256")
            .read_text(encoding="utf-8").strip()}
    body.update(over)
    path = root / label.CALIBRATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_a_rejected_cycle_does_not_arm_tier_one(monkeypatch, tmp_path):
    """A cycle that refuses the artifact must not leave it armed."""
    import shutil

    bootstrap.register_example_schemas()
    root = _cycle_root(tmp_path)
    for rel in ("data/heldout/test.jsonl", "data/wild/issue.jsonl",
                "data/wild/commit.jsonl", "data/wild/platform-logs.jsonl"):
        if not (ROOT / rel).is_file():
            continue  # public-release tree: this source does not ship
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, target)
    (root / label.CALIBRATION_PATH).write_text(json.dumps({
        "source": "data/synthetic/dev.jsonl", "n": 750,
        "max_significant_error": 0.02, "min_probability": 0.90,
        "router_sha256": (ROOT / "models" / "router.sha256")
        .read_text(encoding="utf-8").strip()}), encoding="utf-8")

    import mivia_small_json.model as model_mod
    monkeypatch.setattr(model_mod, "main", lambda *a, **k: None)
    monkeypatch.setattr(flywheel, "gates_pass", lambda *a, **k: (False, "refused"))

    report = flywheel.run_cycle(root=root, do_train=True)

    assert report["decision"] == "report-only"
    assert report["calibration_refreshed"]["status"] == "not-written"
    assert label.calibration_from_disk(root) is not None, (
        "the pre-existing record should be left untouched, not replaced")


# (test_a_refused_retrain_restores_the_previous_artifact retired on
# the public tree: it runs a REAL retrain over the full training mix,
# whose private corpora the open-source tree does not redistribute.
# The private tree keeps the rollback gate.)


def test_a_failed_refresh_is_recorded_and_switches_tier_one_off(tmp_path):
    """A cycle that cannot measure must report it, not raise it.

    Uses a real router against an empty development root, because that
    is the failure a cycle actually hits. Passing a missing router would
    exercise a branch production never reaches.
    """
    from mivia_small_json.pipeline import load_router

    bootstrap.register_example_schemas()
    report = {}
    result = flywheel._refresh_calibration(tmp_path, load_router(MODEL_PATH), report)
    assert result is None
    assert report["calibration_refreshed"]["status"] == "not-measured"
    assert "reach" in report["calibration_refreshed"]["reason"]
    assert not (tmp_path / label.CALIBRATION_PATH).exists()


def test_a_refresh_survives_an_unexpected_error(tmp_path):
    """Nothing may escape once the artifact has been rewritten.

    The artifact is already on disk by the time the refresh runs, so an
    exception here would abort the cycle mid-way.
    """
    class _Exploding:
        @property
        def named_steps(self):
            raise KeyError("clf")

    report = {}
    assert flywheel._refresh_calibration(tmp_path, _Exploding(), report) is None
    assert report["calibration_refreshed"]["status"] == "not-measured"
    assert "clf" in report["calibration_refreshed"]["reason"]


def test_the_record_hashes_the_cycle_root_not_the_repository(tmp_path):
    """A cycle against a scratch root must not stamp the repo artifact.

    The refresh measures the router the cycle loaded, which lives in the
    cycle's root. Hashing the repository's own copy instead writes a
    digest for an artifact that was never measured. Every other test was
    blind to that, not because it repointed the model path, but because
    its root artifact was a byte-identical copy of the repository's.
    """
    import io

    from mivia_small_json.pipeline import load_router
    from skops.io import dump as skops_dump

    bootstrap.register_example_schemas()
    root = _cycle_root(tmp_path)
    # Nothing repoints model.MODEL_PATH here: the repository's bytes
    # differ from what the cycle wrote, so a regression to reading that
    # constant stamps the digest of an artifact that was never measured.
    repo_digest = _digest_of(ROOT / "models" / "router.skops")

    router = load_router(MODEL_PATH)
    buffer = io.BytesIO()
    skops_dump(router, buffer)
    (root / "models" / "router.skops").write_bytes(buffer.getvalue())
    root_digest = _digest_of(root / "models" / "router.skops")
    (root / "models" / "router.sha256").write_text(root_digest + "\n",
                                                    encoding="utf-8")
    assert root_digest != repo_digest, "the two artifacts must differ"

    report = {}
    flywheel._refresh_calibration(root, router, report)

    assert report["calibration_refreshed"]["status"] == "measured"
    written = json.loads((root / label.CALIBRATION_PATH).read_text(encoding="utf-8"))
    assert written["router_sha256"] == root_digest, (
        "the record was stamped with an artifact that was never measured")
    assert written["router_sha256"] != repo_digest


def test_the_record_hashes_the_artifact_not_a_sidecar(tmp_path):
    """The record must describe the artifact, not what a file claims.

    Reading the sidecar would trust a text file. Hashing the artifact
    cannot be lied to, so a stale or wrong sidecar must not change the
    recorded digest.
    """
    from mivia_small_json.pipeline import load_router

    bootstrap.register_example_schemas()
    root = _cycle_root(tmp_path, digest="not-the-real-digest")
    real_digest = _digest_of(root / "models" / "router.skops")
    report = {}

    flywheel._refresh_calibration(root, load_router(MODEL_PATH), report)

    written = json.loads((root / label.CALIBRATION_PATH).read_text(encoding="utf-8"))
    assert written["router_sha256"] == real_digest
    assert written["router_sha256"] != "not-the-real-digest"


def test_a_missing_artifact_is_reported_not_raised(tmp_path):
    from mivia_small_json.pipeline import load_router

    bootstrap.register_example_schemas()
    root = _cycle_root(tmp_path)
    (root / "models" / "router.skops").unlink()
    report = {}

    assert flywheel._refresh_calibration(root, load_router(MODEL_PATH), report) is None
    assert report["calibration_refreshed"]["status"] == "not-measured"
    assert "router.skops" in report["calibration_refreshed"]["reason"]


def test_the_record_pins_the_worst_band_not_the_pooled_mean():
    """Recompute the bands and pin the worst one.

    A weaker assertion (`max >= mean`) passes when the field has
    regressed to the pooled mean, which is what this guards.
    """
    from mivia_small_json.pipeline import load_router

    bootstrap.register_example_schemas()
    router = load_router(MODEL_PATH)
    classes = list(router.named_steps["clf"].classes_)
    record = label.measure_dev_calibration(ROOT, router)
    floor = record["min_probability"]
    width = (1.0 - floor) / label.CALIBRATION_BINS
    buckets = {}
    for text, gold in label.dev_rows(ROOT)[0]:
        proba = router.predict_proba([text])[0]
        top = int(proba.argmax())
        confidence = float(proba[top])
        if confidence < floor:
            continue
        index = min(label.CALIBRATION_BINS - 1, int((confidence - floor) / width))
        bucket = buckets.setdefault(index, [0.0, 0, 0])
        bucket[0] += confidence
        bucket[1] += int(classes[top] == gold)
        bucket[2] += 1
    worst = max(abs(b[0] / b[2] - b[1] / b[2]) for b in buckets.values())
    assert record["max_abs_error"] == pytest.approx(round(worst, 4), abs=1e-4)
    assert record["max_abs_error_band"] in buckets
    # Every admitted row is inside a scored band.
    assert sum(b[2] for b in buckets.values()) == record["n"]
    assert record["bands_scored"] == len(buckets)


def test_a_thin_band_inside_its_interval_does_not_fail_the_record(tmp_path):
    """Twelve rows at p=0.90 with nine correct are a rate of 0.75.

    That is a gap of 0.15, over the limit. But 9/12 supports a Wilson
    interval reaching 0.91, so 0.90 sits inside it and the band is not
    distinguishably miscalibrated. Without the interval this fails the
    record on a band that cannot even be measured to that precision.
    """
    from types import SimpleNamespace

    import numpy as np

    bootstrap.register_example_schemas()
    rows = []
    for i in range(12):
        # Three of the twelve are mislabelled, so the band rates 0.75.
        schema = "issue" if i % 4 == 3 else "log_event"
        rows.append({"raw": f"THIN {i}", "schema": schema})
    rows += [{"raw": f"HIGH {i}", "schema": "log_event"} for i in range(400)]
    _write_dev(tmp_path, rows)

    class _Fake:
        classes_ = ["issue", "log_event"]

        @property
        def named_steps(self):
            return {"clf": SimpleNamespace(classes_=_Fake.classes_)}

        def predict_proba(self, texts):
            out = []
            for text in texts:
                confidence = 0.90 if text.startswith("THIN") else 1.00
                out.append(np.array([1.0 - confidence, confidence]))
            return out

    record = label.measure_dev_calibration(tmp_path, _Fake())
    assert record["n"] == 412
    assert record["max_abs_error"] == pytest.approx(0.15, abs=1e-3)
    assert record["max_significant_error"] == pytest.approx(0.0, abs=1e-6)


def test_the_record_scores_bands_not_a_pooled_mean(tmp_path):
    """The error must be a bound, not a pooled mean.

    A naive pooled difference scores this router at 0.08, inside the
    0.10 limit, even though its lowest band is wrong 40% of the time.
    """
    import numpy as np

    bootstrap.register_example_schemas()
    rows = [{"raw": f"HALF {i}", "schema": "log_event" if i % 2 else "issue"}
            for i in range(100)]
    rows += [{"raw": f"HIGH {i}", "schema": "log_event"} for i in range(400)]
    _write_dev(tmp_path, rows)

    class _Fake:
        classes_ = ["issue", "log_event"]

        @property
        def named_steps(self):
            return {"clf": _Fake()}

        def predict_proba(self, texts):
            out = []
            for text in texts:
                confidence = 0.90 if text.startswith("HALF") else 1.00
                out.append(np.array([1.0 - confidence, confidence]))
            return out

    with pytest.raises(ValueError, match="above"):
        label.measure_dev_calibration(tmp_path, _Fake())


def test_the_record_reports_a_bound_and_a_mean_separately():
    """Both numbers are kept, and the bound is the larger one."""
    from mivia_small_json.pipeline import load_router

    bootstrap.register_example_schemas()
    record = label.measure_dev_calibration(ROOT, load_router(MODEL_PATH))
    assert record["max_abs_error"] >= record["mean_abs_error"]
    assert record["bands_scored"] >= 1
    assert record["max_significant_error"] <= label.CALIBRATION_MAX_ERROR
    assert 0.0 <= record["min_probability"] <= record["predicted"] <= 1.0


def test_a_record_cut_over_a_different_range_is_refused(tmp_path):
    """The hash pins the artifact. It does not pin the admitted range.

    Moving `AGREE_MIN_PROBA` without a re-measure would leave a record
    whose bands were cut over a range the tier no longer admits.
    """
    root = _cycle_root(tmp_path)
    _write_record(root, min_probability=0.97)
    assert label.calibration_from_disk(root) is None


# --- calibrated confidence as an abstention policy ---------------------------


def test_precision_at_coverage_maps_coverage_to_a_threshold_and_precision():
    """A target coverage yields the threshold that admits it, and precision there."""
    import numpy as np

    pairs = [("T0", "log_event"), ("T1", "log_event"),
             ("F0", "log_event"), ("F1", "log_event")]

    class _Fake:
        classes_ = ["issue", "log_event"]

        @property
        def named_steps(self):
            return {"clf": self}

        def predict_proba(self, texts):
            out = []
            for text in texts:
                # The "T" rows are correct at 0.95; the "F" rows are wrong
                # (top class issue) at 0.85.
                out.append(np.array([0.05, 0.95]) if text.startswith("T")
                           else np.array([0.85, 0.15]))
            return out

    # The two most confident rows are exactly the two correct ones.
    threshold, precision = label.precision_at_coverage(_Fake(), pairs, 0.5)
    assert threshold == pytest.approx(0.95)
    assert precision == pytest.approx(1.0)

    # Full coverage drags the threshold down to the wrong rows.
    threshold, precision = label.precision_at_coverage(_Fake(), pairs, 1.0)
    assert threshold == pytest.approx(0.85)
    assert precision == pytest.approx(0.5)


def test_precision_at_coverage_rejects_a_missing_router_and_bad_coverage():
    import numpy as np

    pairs = [("T0", "log_event")]

    class _Fake:
        classes_ = ["log_event"]

        @property
        def named_steps(self):
            return {"clf": self}

        def predict_proba(self, texts):
            return [np.array([1.0]) for _ in texts]

    with pytest.raises(ValueError, match="needs the router"):
        label.precision_at_coverage(None, pairs, 0.5)
    for coverage in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="coverage"):
            label.precision_at_coverage(_Fake(), pairs, coverage)
    with pytest.raises(ValueError, match="development pairs"):
        label.precision_at_coverage(_Fake(), [], 0.5)


def test_calibration_record_carries_reliability_and_admission_threshold():
    """The record reports Brier/ECE and a precision-at-coverage threshold."""
    from mivia_small_json.pipeline import load_router

    bootstrap.register_example_schemas()
    router = load_router(MODEL_PATH)
    record = label.measure_dev_calibration(ROOT, router)

    assert 0.0 <= record["brier"] <= 1.0
    assert 0.0 <= record["ece"] <= 1.0
    # ECE is the band gaps averaged by band size; the bound is the worst
    # band, so the average cannot sit above it.
    assert record["ece"] <= record["max_abs_error"]
    # The calibrated threshold may tighten the fixed floor, never loosen it.
    assert record["min_probability"] <= record["admission_threshold"] <= 1.0
    assert 0.0 < record["admission_coverage"] <= 1.0
    assert 0.0 <= record["admission_precision"] <= 1.0

    # The threshold is exactly what the coverage helper picks for the target.
    threshold, _precision = label.precision_at_coverage(
        router, label.dev_rows(ROOT)[0], label.ADMISSION_TARGET_COVERAGE)
    assert record["admission_threshold"] == pytest.approx(
        max(record["min_probability"], round(threshold, 4)), abs=1e-4)


def test_tier1_uses_the_calibrated_admission_threshold(monkeypatch):
    """The calibration's threshold decides admission, not the fixed floor."""
    import numpy as np

    bootstrap.register_example_schemas()
    text = _voted_texts("issue", 1)[0]

    class _Fake:
        classes_ = ["issue", "log_event"]

        @property
        def named_steps(self):
            return {"clf": self}

        def predict_proba(self, texts):
            # Top class issue at 0.93, which the rules layer also votes.
            return [np.array([0.93, 0.07]) for _ in texts]

    router = _Fake()
    # 2026-10-09 (F1): the trust gate must not decide this test, and
    # tier 2 (rules-dominant, margin 1.0) must stay out of the way too,
    # so pin a trusted class with a sub-dominant rules margin.
    monkeypatch.setattr(label, "rules_vote", lambda t: ("issue", 0.10))
    at_floor = label.DevCalibration(source="data/synthetic/dev.jsonl", n=350,
                                    max_abs_error=0.02,
                                    admission_threshold=label.AGREE_MIN_PROBA)
    # 2026-10-09 (F1): agreement now also needs the class's rule trust,
    # so grant issue explicitly — this test is about the threshold.
    assert label.label_record(text, router=router, trust={"issue": True},
                              calibration=at_floor)["tier"] == "confident"

    tightened = label.DevCalibration(source="data/synthetic/dev.jsonl", n=350,
                                     max_abs_error=0.02,
                                     admission_threshold=0.95)
    lab = label.label_record(text, router=router, trust={"issue": True},
                             calibration=tightened)
    assert lab["tier"] == "hard", lab
    assert lab["proba"] == pytest.approx(0.93)
    assert not lab["reason"].startswith("tier1:")


def test_the_stored_admission_threshold_travels_through_the_disk_reader(tmp_path):
    root = _cycle_root(tmp_path)
    _write_record(root, admission_threshold=0.97)
    restored = label.calibration_from_disk(root)
    assert restored is not None
    assert restored.admission_threshold == pytest.approx(0.97)

    # A threshold under the measured floor would admit rows no band scored.
    _write_record(root, admission_threshold=0.50)
    assert label.calibration_from_disk(root) is None


def test_a_record_without_an_admission_threshold_keeps_the_fixed_floor(tmp_path):
    """A record written before the field existed reads back at the floor."""
    root = _cycle_root(tmp_path)
    _write_record(root)
    restored = label.calibration_from_disk(root)
    assert restored is not None
    assert restored.admission_threshold == label.AGREE_MIN_PROBA


def _count_test_definitions():
    """Count test definitions per file across the suite, one mechanism.

    Both suite-shape guards share this counter, so they cannot drift
    apart on what counts as a test definition.
    """
    import ast
    from collections import Counter

    seen: Counter = Counter()
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith("test_"):
                seen[f"{path.name}::{node.name}"] += 1
    return seen


def test_no_test_function_is_defined_twice():
    """A duplicate test name silently shadows the first definition.

    This shipped once: two definitions of the same test left the suite
    green with one test missing from collection, and the second carried
    a third test's body inside it.
    """
    seen = _count_test_definitions()
    duplicates = sorted(name for name, count in seen.items() if count > 1)
    assert not duplicates, f"duplicate test definitions: {duplicates}"


def test_the_suite_matches_its_committed_inventory(pytestconfig):
    """The committed inventory pins the suite's collected shape.

    A pinned integer floor decayed once already: a rebase brought
    dev-side tests in, so absorbed tests kept passing under the floor.
    This gate instead compares the sha256 of the sorted collected node
    ids against tests/test-inventory.txt. A test deleted, renamed,
    added, or re-parameterized changes the id list and fails here until
    the inventory is regenerated — deliberately — with
    scripts/refresh-test-inventory.py. The AST duplicate-name guard
    above still catches shadowing, which collection cannot see.
    """
    live = getattr(pytestconfig, "_collected_test_ids", None)
    assert live, "no collected ids recorded; run the full suite"
    live_sha = hashlib.sha256(("\n".join(live) + "\n").encode()).hexdigest()
    committed = TEST_INVENTORY.read_bytes()
    committed_sha = hashlib.sha256(committed).hexdigest()
    committed_count = len([line for line in committed.decode().splitlines() if line])
    partial = len(live) < committed_count
    assert live_sha == committed_sha, (
        f"collected test ids sha256 {live_sha} != committed inventory "
        f"{committed_sha} ({TEST_INVENTORY.name}: {committed_count} ids "
        f"committed, {len(live)} collected here). "
        + ("This looks like a PARTIAL collection (a node-id or -k "
           "selection); run the full suite. "
           if partial else "")
        + "If the suite's shape really changed, regenerate with: "
        "scripts/refresh-test-inventory.py")


def test_dev_pairs_reuse_the_measurement_already_in_hand():
    """The cycle reuses the trust measurement instead of re-reading it."""
    measure = label.measure_dev_rules(ROOT)
    pairs = flywheel._dev_pairs(measure)
    assert len(pairs) == measure["n"]
    assert all(row["raw"] and gold for row, gold in pairs)
    # Every class with labeled rows contributes pairs; agent_step has no
    # development rows yet, so it contributes none and stays unknown.
    assert {gold for _row, gold in pairs} == {
        cls for cls, row in measure["per_class"].items() if row["support"] > 0}


def test_each_development_file_is_read_once(monkeypatch):
    """Hashing and parsing share a single read of the frozen reference."""
    import pathlib

    real_open = pathlib.Path.open
    seen: list[str] = []

    def counting(self, *args, **kwargs):
        if "dev-reference.jsonl" in str(self).replace("\\", "/"):
            seen.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", counting)
    measure = label.measure_dev_rules(ROOT)
    flywheel._dev_pairs(measure)

    assert seen == [str(ROOT / label.DEV_REFERENCE)]
    assert len(seen) == 1, "the frozen reference was read twice"


def test_flywheel_report_discloses_the_trust_source(tmp_path, corpus_world):
    """The cycle report names the trust source and unknown classes."""
    inbox = tmp_path / "data" / "pool" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "sample.log").write_text(LOG_LINE + "\n", encoding="utf-8")
    report = flywheel.run_cycle(root=tmp_path, do_train=False)

    trust = report["trust"]
    assert trust["source"] == "development"
    assert trust["files"] == [], "no development data inside the tmp root"
    assert trust["n"] == 0
    assert sorted(trust["unknown_classes"]) == ["agent_step", "commit", "issue",
                                                "log_event", "pull_request"]
    assert trust["trusted_classes"] == []
    assert report["label"]["tier1_enabled"] is False
    manifest = json.loads((tmp_path / "data" / "pool" / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["cycles"][-1]["trust"]["source"] == "development"
