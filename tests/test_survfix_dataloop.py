"""Focused gates for the 9 execution-confirmed data-loop survivors.

Every test here pins one honest-measurement failure that was confirmed by
execution against HEAD 6d6e7d9 (validation evidence:
``.scratch/validate-c/``). No test runs a training or measurement cycle;
each imports the module under test and drives the exact function with
synthetic inputs (temp dirs for ``scripts/promote_hard.py``).

Survivors covered:
1  flywheel per-class drop gate ignores NEW classes        (V1)
2  flywheel post-train None router scored as candidate     (V4)
3  flywheel PSI reads the share-capped admit slice         (V6)
4  label.check_calibration has no CALIBRATION_MIN_ROWS     (V5)
5  bench attributes numbers by file existence              (V2)
6  promote_hard overwrites earlier promotions              (V3)
7  promote_hard one malformed line kills the batch         (V8)
8  promote_hard accepted-but-unpromotable rows vanish      (V9)
9  _retrain_extractors keeps a failed run's partial bytes  (V10)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from mivia_small_json import bench, drift, flywheel, label, model

ROOT = Path(__file__).resolve().parents[1]

# The four-class baseline world the gates were migrated with.
BASELINES = {
    "prev_dev_macro": 0.9992,  # headroom saturated -> macro gate skipped
    "prev_dev_per_class": {"issue": 0.9987, "log_event": 1.0,
                           "commit": 0.998, "pull_request": 1.0},
}
MIX = {"issue": 0.27, "log_event": 0.25, "commit": 0.27, "pull_request": 0.21}


def _gate_report(per_class, **over):
    """A cycle report whose other numbers pass every gate."""
    base = {
        "new_confident": 500,
        "dev": {
            "routed_and_valid": 0.995,
            "f1_macro": 0.80,  # dragged down; the macro gate is saturated
            "f1_per_class": per_class,
        },
        "heldout": None,
        "psi": 0.01,
        "artifact_bytes": 500_000,
    }
    base.update(over)
    return base


# --- Survivor 1: the per-class drop gate must measure NEW classes (V1) ------


def test_per_class_gate_fails_a_new_class_at_zero_f1():
    """A brand-new class measured at F1=0.0 must fail the gate.

    Before the fix the gate iterated the BASELINE keys only, so tool_call
    at 0.0 escaped measurement and gates_pass returned True.
    """
    per_class = {"issue": 0.999, "log_event": 1.0, "commit": 0.999,
                 "pull_request": 1.0, "tool_call": 0.0}
    ok, reason = flywheel.gates_pass(_gate_report(per_class), BASELINES, MIX)
    assert ok is False, "a new class at F1 0.0 passed the PR gate"
    assert "tool_call" in reason, reason


def test_per_class_gate_passes_a_strong_new_class():
    """A new class at the near-perfect bar the baselines carry passes."""
    per_class = {"issue": 0.999, "log_event": 1.0, "commit": 0.999,
                 "pull_request": 1.0, "tool_call": 0.995}
    ok, reason = flywheel.gates_pass(_gate_report(per_class), BASELINES, MIX)
    assert ok is True, reason


def test_per_class_gate_fails_when_a_baseline_class_vanishes():
    """A baseline class missing from the candidate's per-class map is a
    total drop, not an escape."""
    per_class = {"issue": 0.999, "log_event": 1.0, "pull_request": 1.0}
    ok, reason = flywheel.gates_pass(_gate_report(per_class), BASELINES, MIX)
    assert ok is False, "a vanished baseline class passed the PR gate"
    assert "commit" in reason, reason


def test_per_class_gate_still_fails_a_baseline_drop():
    """Control: the pre-existing drop detection keeps working."""
    per_class = {"issue": 0.94, "log_event": 1.0, "commit": 0.999,
                 "pull_request": 1.0}
    ok, _ = flywheel.gates_pass(_gate_report(per_class), BASELINES, MIX)
    assert ok is False


# --- Survivor 2: a failed candidate load must not become a measurement (V4) --


def test_eval_set_refuses_a_none_router():
    """_eval_set(pairs, None) must raise, not score the rules fallback.

    Before the fix it produced a normal-looking candidate metrics dict
    from rules-only routing.
    """
    pairs = [({"raw": "Steps to reproduce: app crashes on save."}, "issue")]
    with pytest.raises(ValueError, match="router"):
        flywheel._eval_set(pairs, None)


def test_candidate_router_reports_a_failed_load_loudly(tmp_path):
    """A sha-verified load failure is an explicit failure, not a router."""
    missing = tmp_path / "models" / "not-there.skops"
    router, failure = flywheel._candidate_router(missing)
    assert router is None
    assert failure is not None
    assert "no candidate measurement" in failure
    assert str(missing) in failure


def test_candidate_router_loads_the_real_artifact():
    """Control: the shipped artifact loads with no failure string."""
    router, failure = flywheel._candidate_router(model.MODEL_PATH)
    assert failure is None
    assert router is not None


# --- Survivor 3: PSI must read the pool, not the share-capped slice (V6) -----


def test_build_pools_counts_the_preadmission_pool_and_cap_drops(monkeypatch):
    """The cycle stats must expose the PRE-CAP label pool and the rows the
    caps dropped from both pools."""
    monkeypatch.setattr(
        label, "rules_vote",
        lambda text: (("log_event", 0.9) if text.startswith("LOG")
                      else ("issue", 0.9)))
    trust = {"log_event": True, "issue": True}
    measure = {"source": "development", "rule": "r", "files": [],
               "min_support": 20, "trust_f1": 0.85, "n": 0,
               "unknown_classes": [], "trusted_classes": [], "per_class": {}}
    records = []
    for i in range(10):
        for k in range(9):
            records.append({"id": f"log-{i}-{k}", "source": "s",
                            "raw": f"LOG:{i}:{k}"})
        records.append({"id": f"iss-{i}", "source": "s", "raw": f"ISS:{i}"})

    confident, hard, stats = label.build_pools(records, None, trust=trust,
                                               calibration=None,
                                               measure=measure)
    assert stats["pool_per_class"] == {"log_event": 90, "issue": 10}, (
        "the pre-admission pool must count every confident label, "
        "before the caps thin it")
    assert stats["dropped_over_cap"]["confident"] > 0, (
        "rows the share cap rejected must be counted, not vanish")
    assert stats["confident"] < 90, "the log_event share cap must have bit"


def test_pool_psi_blocks_what_the_admit_slice_hides(monkeypatch):
    """The gated PSI must be computed on the pool: PSI(pool) blocks while
    PSI(admit slice) passes (measured 0.953 vs 0.164)."""
    monkeypatch.setattr(
        label, "rules_vote",
        lambda text: (("log_event", 0.9) if text.startswith("LOG")
                      else ("issue", 0.9)))
    trust = {"log_event": True, "issue": True}
    measure = {"source": "development", "rule": "r", "files": [],
               "min_support": 20, "trust_f1": 0.85, "n": 0,
               "unknown_classes": [], "trusted_classes": [], "per_class": {}}
    records = []
    for i in range(10):
        for k in range(9):
            records.append({"id": f"log-{i}-{k}", "source": "s",
                            "raw": f"LOG:{i}:{k}"})
        records.append({"id": f"iss-{i}", "source": "s", "raw": f"ISS:{i}"})
    _confident, _hard, stats = label.build_pools(records, None, trust=trust,
                                                 calibration=None,
                                                 measure=measure)
    mix = {"issue": 0.27 / 0.52, "log_event": 0.25 / 0.52}
    psi_pool = flywheel._pool_psi(stats["pool_per_class"], mix)
    admit_shares = {s: stats["per_class"].get(s, 0) / max(1, stats["confident"])
                    for s in mix}
    psi_admit = drift.psi(admit_shares, mix)
    assert psi_pool >= flywheel.PSI_BLOCK, (
        f"pool PSI {psi_pool} must block; the shift is real")
    assert psi_admit < flywheel.PSI_BLOCK, (
        "control: the admit slice alone hides the shift (the old bug)")


def test_pool_psi_is_stable_for_a_balanced_pool():
    """Control: a pool shaped like the training mix scores ~0."""
    balanced = {k: int(v * 100) for k, v in MIX.items()}
    assert flywheel._pool_psi(balanced, MIX) < 0.05


# --- Survivor 4: the calibration gate enforces its own floor (V5) ------------


def test_check_calibration_refuses_records_below_the_row_floor():
    """n=3 must not enable tier 1: CALIBRATION_MIN_ROWS is the module's own
    statement of how many rows the measurement needs."""
    tiny = label.DevCalibration(source="data/synthetic/dev.jsonl", n=3,
                                max_abs_error=0.0, admission_threshold=0.9)
    with pytest.raises(ValueError, match=str(label.CALIBRATION_MIN_ROWS)):
        label.check_calibration(tiny)
    assert label._tier1_enabled(tiny) is False


def test_check_calibration_accepts_the_floor_boundary():
    """Exactly CALIBRATION_MIN_ROWS rows is usable."""
    cal = label.DevCalibration(source="data/synthetic/dev.jsonl",
                               n=label.CALIBRATION_MIN_ROWS,
                               max_abs_error=0.0, admission_threshold=0.9)
    label.check_calibration(cal)  # no raise
    assert label._tier1_enabled(cal) is True


def test_calibration_from_disk_refuses_a_small_n_record(tmp_path):
    """A stale/hand-edited calibration.json with n=3 must not enable
    tier 1 through the disk path either."""
    art = tmp_path / "models" / "router.skops"
    art.parent.mkdir(parents=True)
    art.write_bytes(b"fake-but-hashed")
    digest = hashlib.sha256(art.read_bytes()).hexdigest()
    (tmp_path / "models" / "router.sha256").write_text(digest + "\n")
    record = {"router_sha256": digest, "source": "data/synthetic/dev.jsonl",
              "n": 3, "max_significant_error": 0.0, "max_abs_error": 0.0,
              "min_probability": 0.9, "admission_threshold": 0.9}
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "calibration.json").write_text(json.dumps(record))
    assert label.calibration_from_disk(tmp_path) is None


# --- Survivor 5: bench attributes numbers to what actually loaded (V2) -------


def test_backend_of_names_what_actually_loaded():
    """The backend comes from the loaded object, never file existence."""
    from sklearn.pipeline import Pipeline

    from mivia_small_json.encoder import EncoderRouter

    assert bench._backend_of(None) == "rules"
    assert bench._backend_of(Pipeline(steps=[])) == "skops"
    assert bench._backend_of(object.__new__(EncoderRouter)) == "encoder"


def test_backend_note_follows_the_loaded_router_not_the_file(monkeypatch):
    """V2 red state: the ONNX file exists but its load fails and skops
    serves. The note must say skops, because skops answered."""
    from mivia_small_json import encoder, pipeline

    monkeypatch.setattr(encoder, "load_encoder", lambda *a, **k: None)
    pipe = pipeline.load_active_router()
    backend = bench._backend_of(pipe)
    assert backend == "skops", "the failed ONNX load fell back to skops"
    assert encoder.DEFAULT_MODEL.is_file(), (
        "fixture: the report's old source of truth (file existence) is a lie")
    note = bench._method_note(backend)
    assert "ONNX" not in note
    assert "LogisticRegression" in note
    attribution = bench._artifact_attribution(backend, model.MODEL_PATH.stat().st_size)
    assert "router.skops" in attribution
    assert "router-encoder-int8.onnx" not in attribution


def test_bench_rules_backend_is_named_honestly():
    """When nothing loads, the full column IS rules.route — the report must
    not claim a skops artifact served it."""
    note = bench._method_note("rules")
    assert "rules" in note.lower()
    assert "no model artifact" in note.lower()
    assert "LogisticRegression" not in note
    attribution = bench._artifact_attribution("rules", 0)
    assert "no model artifact" in attribution.lower()


def test_bench_helpers_keep_the_historical_bool_flags():
    """Compat: the existing helper callers pass True/False bools."""
    enc = bench._artifact_attribution(True, 135_000_000)
    skops = bench._artifact_attribution(False, 900_000)
    assert "router-encoder-int8.onnx" in enc and "tokenizer" in enc
    assert "router.skops" in skops
    assert "ONNX" in bench._method_note(True)
    assert "LogisticRegression" in bench._method_note(False)


# --- Survivors 6-8: scripts/promote_hard.py ----------------------------------

@pytest.fixture()
def promote_env(tmp_path, monkeypatch):
    """promote_hard with its pools pointed at a temp dir."""
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import promote_hard

    hard = tmp_path / "hard.jsonl"
    promoted = tmp_path / "promoted"
    monkeypatch.setattr(promote_hard, "HARD", hard)
    monkeypatch.setattr(promote_hard, "PROMOTED", promoted)
    return promote_hard, hard, promoted


def _write_batch(hard: Path, rows: list) -> None:
    hard.write_text("".join(
        (r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows),
        encoding="utf-8")


def _read_promoted(promoted: Path, schema: str) -> list[dict]:
    path = promoted / f"{schema}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_promote_merges_with_earlier_promotions(promote_env):
    """A second promotion must not wipe the first batch (V3)."""
    mod, hard, promoted = promote_env
    _write_batch(hard, [
        {"id": "h1", "raw": "batch-one row A", "schema": "issue", "review": "accept"},
        {"id": "h2", "raw": "batch-one row B", "schema": "issue", "review": "accept"},
    ])
    first = mod.promote(detailed=True)
    assert first["schemas"]["issue"]["added"] == 2
    _write_batch(hard, [
        {"id": "h3", "raw": "batch-two row C", "schema": "issue", "review": "accept"},
    ])
    second = mod.promote(detailed=True)
    assert second["schemas"]["issue"]["added"] == 1
    rows = [r["raw"] for r in _read_promoted(promoted, "issue")]
    assert rows == ["batch-one row A", "batch-one row B", "batch-two row C"], (
        f"earlier promotions were lost: {rows}")
    assert second["schemas"]["issue"]["total"] == 3


def test_promote_default_shape_stays_schema_to_added(promote_env):
    """Compat: the historical flat return contract is preserved."""
    mod, hard, _promoted = promote_env
    _write_batch(hard, [
        {"id": "h1", "raw": "kept", "schema": "issue", "review": "accept"},
        {"id": "b", "schema": "issue", "raw": "no review"},
    ])
    assert mod.promote() == {"issue": 1}


def test_promote_dedupes_by_text_hash_keep_first(promote_env):
    """A re-reviewed duplicate keeps the earlier promoted row."""
    mod, hard, promoted = promote_env
    _write_batch(hard, [
        {"id": "h1", "raw": "same text", "schema": "issue", "review": "accept"},
    ])
    mod.promote()
    _write_batch(hard, [
        {"id": "h2", "raw": "same text", "schema": "issue", "review": "accept"},
        {"id": "h3", "raw": "new text", "schema": "issue", "review": "accept"},
    ])
    result = mod.promote(detailed=True)
    assert result["schemas"]["issue"]["added"] == 1
    assert result["schemas"]["issue"]["already_promoted"] == 1
    assert result["schemas"]["issue"]["duplicates_in_batch"] == 1
    rows = [r["raw"] for r in _read_promoted(promoted, "issue")]
    assert rows == ["same text", "new text"]


def test_promote_survives_a_malformed_line(promote_env):
    """One bad line must not crash the promotion or lose the batch (V8)."""
    mod, hard, promoted = promote_env
    _write_batch(hard, [
        {"id": "h1", "raw": "good", "schema": "issue", "review": "accept"},
        "[1, 2]",
        "{not json at all",
        {"id": "h2", "raw": "after the bad lines", "schema": "issue",
         "review": "accept"},
    ])
    result = mod.promote(detailed=True)  # must not raise
    assert result["read_errors"] == 2
    rows = [r["raw"] for r in _read_promoted(promoted, "issue")]
    assert rows == ["good", "after the bad lines"], (
        "rows after a malformed line were lost")


def test_promote_counts_accepted_but_unpromotable_rows(promote_env):
    """schema=null accepts are counted and reported, never silent (V9)."""
    mod, hard, _promoted = promote_env
    _write_batch(hard, [
        {"id": f"n{i}", "raw": f"row {i}", "schema": None, "review": "accept"}
        for i in range(40)
    ])
    result = mod.promote(detailed=True)
    assert result["schemas"] == {}
    assert result["accepted_unpromotable"] == 40
    # The flat shape stays honest too: it does not pretend rows landed.
    assert mod.promote() == {}


def test_promote_main_names_unpromotable_accepts(promote_env, capsys):
    """The summary must not claim 'no rows marked review=accept' when 40
    accepts were dropped, and must not exit 0 silently (V9)."""
    mod, hard, _promoted = promote_env
    _write_batch(hard, [
        {"id": f"n{i}", "raw": f"row {i}", "schema": None, "review": "accept"}
        for i in range(40)
    ])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc != 0, "accepted rows existed but nothing could be promoted"
    assert "40" in out and "unpromotable" in out.lower()
    assert "no rows marked review=accept" not in out


def test_promote_main_prints_mixed_outcome_honestly(promote_env, capsys):
    """40 null-schema accepts beside 1 promotable commit: both facts print."""
    mod, hard, promoted = promote_env
    rows = [{"id": f"n{i}", "raw": f"null row {i}", "schema": None,
             "review": "accept"} for i in range(40)]
    rows.append({"id": "ok", "raw": "good row", "schema": "commit",
                 "review": "accept"})
    _write_batch(hard, rows)
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "40" in out, "the dropped accepts must be named in the summary"
    assert _read_promoted(promoted, "commit")


def test_promote_main_reports_merge_and_read_error_counts(promote_env, capsys):
    """The operator sees merge counts and the malformed-line count."""
    mod, hard, promoted = promote_env
    _write_batch(hard, [
        {"id": "h1", "raw": "round one", "schema": "issue", "review": "accept"}])
    mod.main()
    capsys.readouterr()
    _write_batch(hard, [
        {"id": "h2", "raw": "round two", "schema": "issue", "review": "accept"},
        "junk[",
    ])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "1" in out  # the new row added on top of the existing one
    assert "read" in out.lower(), "the malformed line must be counted aloud"
    assert len(_read_promoted(promoted, "issue")) == 2


# --- Survivor 9: a failed retrain restores the shipped extractors (V10) ------


def _write_extractor_script(scripts: Path, name: str, body: str) -> None:
    (scripts / name).write_text(body, encoding="utf-8")


def test_failed_extractor_retrain_restores_shipped_bytes(tmp_path):
    """A trainer that writes a partial artifact then exits non-zero must
    leave the shipped bytes intact (V10 measured: partial bytes survived)."""
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "models").mkdir()
    live = root / "models" / "level_extractor.skops"
    live.write_bytes(b"GOOD-SHIPPED-BYTES")
    _write_extractor_script(
        root / "scripts", "train_level_extractor.py",
        "import sys\n"
        "from pathlib import Path\n"
        "p = Path(__file__).resolve().parents[1] / 'models'"
        " / 'level_extractor.skops'\n"
        "p.write_bytes(b'PARTIAL-ARTIFACT-FROM-FAILED-RUN')\n"
        "print('wrote partial artifact')\n"
        "sys.exit(1)\n")

    results = flywheel._retrain_extractors(root)

    assert results["train_level_extractor.py"]["ok"] is False
    assert "restored" in results["train_level_extractor.py"]["summary"]
    assert live.read_bytes() == b"GOOD-SHIPPED-BYTES", (
        "the failed run's partial artifact replaced the shipped bytes")


def test_successful_extractor_retrain_keeps_new_bytes(tmp_path):
    """Control: a successful retrain's fresh artifact is not rolled back."""
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "models").mkdir()
    live = root / "models" / "level_extractor.skops"
    live.write_bytes(b"GOOD-SHIPPED-BYTES")
    _write_extractor_script(
        root / "scripts", "train_level_extractor.py",
        "from pathlib import Path\n"
        "p = Path(__file__).resolve().parents[1] / 'models'"
        " / 'level_extractor.skops'\n"
        "p.write_bytes(b'NEW-TRAINED-BYTES')\n"
        "print('done')\n")

    results = flywheel._retrain_extractors(root)

    assert results["train_level_extractor.py"]["ok"] is True
    assert live.read_bytes() == b"NEW-TRAINED-BYTES"


def test_missing_extractor_scripts_are_skipped_quietly(tmp_path):
    """A root without the trainer scripts reports ok-less entries, no crash."""
    root = tmp_path
    (root / "scripts").mkdir()
    results = flywheel._retrain_extractors(root)
    assert results == {}
