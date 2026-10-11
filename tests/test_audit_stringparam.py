"""The audit gate's string-param-span-fill arm (AGENTS.md rule 13: the
gate MEASURES, never derives).

Fixture-corpus gates over the arm-aware extension of
scripts/audit_training_mix.py — nothing here touches data/:

- the gate stamps the arm (passed=true, arm, extra sha binding) when a
  complete corpus layout is present (the round-1 frozen counts 12,407 /
  1,402 / 3,191, silver at its 1,000-row declared minimum, the dropped
  harvest-gold source at 0 rows, clean eval windows);
- it FAILS on a row shortfall, on an eval-window overlap (normalized-set
  re-verification against guard-eval-v2 / calls.jsonl / irrelevance_eval
  request windows), on resurrected harvest-gold rows, and on a frozen
  sha/pin drift — and still writes the stamp, fail-closed (passed=false)
  so the trainer's arm+passed check blocks the GPU;
- the chat-ood path is untouched by the extension (default --arm
  dispatches to the unchanged chat-ood audit), and the gate's eval
  patterns now include the 183-row BFCL calls set.

Deterministic, CPU-only, no timing asserts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # noqa: E402

import audit_training_mix as gate  # noqa: E402

# the round-1 frozen counts the re-registered declaration pins
N_TRAIN, N_DEV, N_RESERVE = 12_407, 1_402, 3_191


def sha_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in rows), encoding="utf-8")


FROZEN_TOOL = {"name": "restart_service",
               "description": "Restart a systemd service on a named host.",
               "parameters": {"type": "object", "properties": {
                   "service": {"type": "string",
                               "description": "The systemd service unit."},
                   "host": {"type": "string",
                            "description": "The target host."}},
                   "required": ["service", "host"]}}
ENUM_PROP_TOOL = {"name": "deploy_release",
                  "description": "Ship a release to an environment.",
                  "parameters": {"type": "object", "properties": {
                      "release": {"type": "string"},
                      "environment": {"type": "string",
                                      "enum": ["staging", "production"]}},
                      "required": ["release", "environment"]}}


def frozen_row(i: int) -> dict:
    return {"tool": FROZEN_TOOL,
            "request": f"restart unit svc-{i} on host-{i}",
            "gold": {"service": f"svc-{i}", "host": f"host-{i}"},
            "template": "restart unit {service} on host",
            "reserve_only": i >= 20_000}


def silver_row(i: int, tool: dict | None = None, request: str | None = None
               ) -> dict:
    tool = tool or FROZEN_TOOL
    request = request or f"provision bucket-{i} for tenant-{i}"
    return {"source": "bfcl-silver:live_simple", "tool": tool,
            "request": request,
            "gold": {"service": f"bucket-{i}", "host": f"tenant-{i}"},
            "evidence": [{"slot": "service", "pattern": "key_delimited",
                          "span": [10, 18], "raw": f"bucket-{i}",
                          "gold": f"bucket-{i}"}],
            "split": "train"}


def make_frozen(base: Path) -> Path:
    """A round-1-shaped frozen corpus with the exact registered counts and
    its own provenance reserve pin (the gate measures the pin)."""
    base.mkdir(parents=True, exist_ok=True)
    _write_jsonl(base / "train.jsonl",
                 [frozen_row(i) for i in range(N_TRAIN)])
    _write_jsonl(base / "dev.jsonl",
                 [frozen_row(10_000 + i) for i in range(N_DEV)])
    reserve = base / "reserve.jsonl"
    _write_jsonl(reserve, [frozen_row(20_000 + i) for i in range(N_RESERVE)])
    (base / "PROVENANCE.md").write_text(
        "# spanfill corpus provenance\n\n"
        "- seed: 20261009; rows: train 12407 / dev 1402 / reserve 3191\n"
        f"- reserve sha256: `{sha_of(reserve)}` (frozen before training)\n",
        encoding="utf-8")
    return base


def make_corpus(base: Path, silver: list[dict],
                gold: list[dict] | None = None) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    for name, rows in (("bfcl-silver.jsonl", silver),
                       ("harvest-gold.jsonl", gold or [])):
        p = base / name
        _write_jsonl(p, rows)
        (base / (name + ".sha256")).write_text(sha_of(p) + "\n",
                                               encoding="utf-8")
    # the house builder's provenance convention: idempotent sections
    # keyed by the source STEM (no literal filename) — the gate accepts
    # the convention, the sidecar sha check binds the bytes
    (base / "PROVENANCE.md").write_text(
        "# spanfill-stringparam corpus provenance\n\n"
        "<!-- section:bfcl-silver -->\n- adjudicated alignment silver\n"
        "<!-- /section:bfcl-silver -->\n\n"
        "<!-- section:harvest-gold -->\n- source dropped 2026-10-10, "
        "0 rows\n<!-- /section:harvest-gold -->\n",
        encoding="utf-8")
    return base


def catalogue_text(request: str) -> str:
    tool = json.dumps({"name": "math.hypot",
                       "description": "Calculate the Euclidean norm.",
                       "parameters": {"type": "dict", "properties": {
                           "x": {"type": "integer"},
                           "y": {"type": "integer"}},
                           "required": ["x", "y"]}})
    return ("Tools available:\n[" + tool + "]\n\nUser request: "
            + request)


def make_eval(base: Path, v2_requests: list[str],
              catalogue_requests: list[str]) -> dict:
    """The three eval surfaces in their real row shapes (bare request for
    guard-eval-v2; catalogue text for calls.jsonl / irrelevance_eval)."""
    base.mkdir(parents=True, exist_ok=True)
    _write_jsonl(base / "guard-eval-v2.jsonl",
                 [{"category": "single_turn_irrelevance", "label": 0,
                   "request": r, "source": "fixture", "split": "eval",
                   "tool": {}} for r in v2_requests])
    _write_jsonl(base / "calls.jsonl",
                 [{"source": "bfcl", "label": 1,
                   "text": catalogue_text(r)} for r in catalogue_requests])
    _write_jsonl(base / "irrelevance_eval.jsonl",
                 [{"source": "bfcl-live-irrelevance",
                   "text": catalogue_text(r)} for r in catalogue_requests])
    return {"calls.jsonl": base / "calls.jsonl",
            "guard-eval-v2.jsonl": base / "guard-eval-v2.jsonl",
            "irrelevance_eval.jsonl": base / "irrelevance_eval.jsonl"}


EVAL_REQUEST = "totally unrelated eval request about zeta widgets"


@pytest.fixture(scope="module")
def clean_eval(tmp_path_factory) -> dict:
    return make_eval(tmp_path_factory.mktemp("eval"),
                     v2_requests=[EVAL_REQUEST],
                     catalogue_requests=[
                         "unrelated catalogue probe for the eval window"])


@pytest.fixture(scope="module")
def frozen_dir(tmp_path_factory) -> Path:
    return make_frozen(tmp_path_factory.mktemp("frozen"))


def full_silver(n: int = 1_000) -> list[dict]:
    return [silver_row(i) for i in range(n)]


def run_gate(tmp_path: Path, frozen: Path, corpus: Path, eval_files: dict,
             ) -> tuple[int, dict]:
    """Run the arm audit against the fixture world: the fixture's OWN
    train/dev sha pins stand in for the registered constants (which pin
    the real committed bytes and are checked against them separately)."""
    stamp_path = tmp_path / "stamp.json"
    rc = gate.audit_stringparam(
        corpus_dir=corpus, frozen_dir=frozen,
        reserve_path=frozen / "reserve.jsonl",
        eval_files=eval_files,
        frozen_sha_pins={"train": sha_of(frozen / "train.jsonl"),
                         "dev": sha_of(frozen / "dev.jsonl")},
        stamp_path=stamp_path)
    return rc, json.loads(stamp_path.read_text())


def test_gate_stamps_arm_when_layout_present(tmp_path, frozen_dir,
                                             clean_eval):
    corpus = make_corpus(tmp_path / "corpus", full_silver())
    rc, stamp = run_gate(tmp_path, frozen_dir, corpus, clean_eval)
    assert rc == 0
    # the stamp carries the arm identity, the pass, the measured numbers,
    # and the corpus files' sha binding (the trainer's fail-closed check)
    assert stamp["passed"] is True
    assert stamp["arm"] == "string-param-span-fill"
    comp = stamp["report"]["composition"]
    assert comp["frozen_train"] == N_TRAIN
    assert comp["frozen_dev"] == N_DEV
    assert comp["frozen_reserve"] == N_RESERVE
    assert comp["bfcl_silver"] == 1_000
    assert comp["harvest_gold"] == 0
    assert comp["templates"] == 0
    # 12,407 / 13,407 = 0.9254 frozen; 1,000 / 13,407 = 0.0746 silver —
    # the amended 60-95% / 5-40% declared bands
    assert comp["shares"]["frozen"] == 0.9254
    assert comp["shares"]["bfcl-silver"] == 0.0746
    assert stamp["report"]["disjointness"]["hits"] == 0
    # enum-slot share is REPORTED, never binding
    assert stamp["report"]["enum_slots"]["binding"] is False
    # extra sha256 covers every corpus file the trainer will load
    bound = set(stamp["extra_sha256"].values())
    for f in (corpus / "bfcl-silver.jsonl",
              corpus / "harvest-gold.jsonl",
              frozen_dir / "train.jsonl",
              frozen_dir / "dev.jsonl",
              frozen_dir / "reserve.jsonl"):
        assert sha_of(f) in bound
    assert len(stamp["extra_sha256"]) == 5
    assert stamp["failures"] == []


def test_gate_fails_on_row_shortfall(tmp_path, frozen_dir, clean_eval):
    corpus = make_corpus(tmp_path / "corpus", full_silver(500))
    rc, stamp = run_gate(tmp_path, frozen_dir, corpus, clean_eval)
    assert rc == 2
    assert stamp["passed"] is False
    assert stamp["arm"] == "string-param-span-fill"
    assert any("bfcl-silver has 500" in f and "minimum 1000" in f
               for f in stamp["failures"])


def test_gate_fails_on_eval_window_overlap(tmp_path, frozen_dir,
                                           clean_eval):
    # one silver request IS a guard-eval-v2 eval request (the normalized
    # set catches it; the builder's exclusion contract must hold)
    rows = full_silver()
    rows[0] = silver_row(0, request=EVAL_REQUEST.upper())
    corpus = make_corpus(tmp_path / "corpus", rows)
    rc, stamp = run_gate(tmp_path, frozen_dir, corpus, clean_eval)
    assert rc == 2
    assert stamp["passed"] is False
    assert any("disjointness" in f for f in stamp["failures"])
    assert stamp["report"]["disjointness"]["hits"] >= 1


def test_gate_fails_on_eval_window_overlap_via_catalogue(tmp_path,
                                                         frozen_dir,
                                                         clean_eval):
    # the calls.jsonl surface stores catalogue renderings; the gate's
    # re-verification parses the BARE request out of them (the same
    # window the builder replicates), so a silver row carrying the
    # request text alone is caught
    rows = full_silver()
    rows[5] = silver_row(5, request="unrelated catalogue probe for the "
                                   "eval window")
    corpus = make_corpus(tmp_path / "corpus", rows)
    rc, stamp = run_gate(tmp_path, frozen_dir, corpus, clean_eval)
    assert rc == 2
    assert any("disjointness" in f for f in stamp["failures"])


def test_gate_fails_on_resurrected_harvest_gold(tmp_path, frozen_dir,
                                                clean_eval):
    corpus = make_corpus(
        tmp_path / "corpus", full_silver(),
        gold=[{"source": "harvest-gold", "tool": FROZEN_TOOL,
               "request": "restart unit svc-x on host-y",
               "gold": {"service": "svc-x", "host": "host-y"},
               "evidence": [], "split": "train"}])
    rc, stamp = run_gate(tmp_path, frozen_dir, corpus, clean_eval)
    assert rc == 2
    assert stamp["passed"] is False
    assert any("harvest-gold has 1 rows" in f and "DROPPED" in f
               for f in stamp["failures"])


def test_gate_fails_on_frozen_pin_drift(tmp_path, frozen_dir,
                                        clean_eval):
    drifted = tmp_path / "frozen-drift"
    shutil.copytree(frozen_dir, drifted)
    prov = drifted / "PROVENANCE.md"
    prov.write_text(prov.read_text(encoding="utf-8").replace(
        sha_of(drifted / "reserve.jsonl"), "0" * 64), encoding="utf-8")
    corpus = make_corpus(tmp_path / "corpus", full_silver())
    rc, stamp = run_gate(tmp_path, drifted, corpus, clean_eval)
    assert rc == 2
    assert any("reserve sha drift" in f for f in stamp["failures"])


def test_gate_fails_on_frozen_count_drift(tmp_path, frozen_dir,
                                          clean_eval):
    drifted = tmp_path / "frozen-counts"
    shutil.copytree(frozen_dir, drifted)
    lines = (drifted / "train.jsonl").read_text(
        encoding="utf-8").splitlines(keepends=True)
    (drifted / "train.jsonl").write_text("".join(lines[:-1]),
                                         encoding="utf-8")
    corpus = make_corpus(tmp_path / "corpus", full_silver())
    rc, stamp = run_gate(tmp_path, drifted, corpus, clean_eval)
    assert rc == 2
    assert any(f"frozen: train has {N_TRAIN - 1} rows" in f
               for f in stamp["failures"])


def test_gate_fails_closed_when_corpus_missing(tmp_path, frozen_dir,
                                               clean_eval):
    stamp_path = tmp_path / "stamp.json"
    rc = gate.audit_stringparam(
        corpus_dir=tmp_path / "absent-corpus", frozen_dir=frozen_dir,
        reserve_path=frozen_dir / "reserve.jsonl",
        eval_files=clean_eval, stamp_path=stamp_path)
    assert rc == 2
    # fail-closed: the failed audit still writes the stamp, so the
    # trainer's arm+passed check blocks the GPU run
    stamp = json.loads(stamp_path.read_text())
    assert stamp["passed"] is False
    assert stamp["arm"] == "string-param-span-fill"
    assert any("directory missing" in f for f in stamp["failures"])


def test_gate_fails_closed_on_missing_eval_surface(tmp_path, frozen_dir,
                                                   clean_eval):
    corpus = make_corpus(tmp_path / "corpus", full_silver())
    eval_files = dict(clean_eval)
    eval_files["calls.jsonl"] = tmp_path / "no-such-calls.jsonl"
    rc, stamp = run_gate(tmp_path, frozen_dir, corpus, eval_files)
    assert rc == 2
    assert any("eval surface(s) missing" in f for f in stamp["failures"])


def test_gate_fails_on_cjk_hangul_row_outside_declared_language(
        tmp_path, frozen_dir, clean_eval):
    # the spec declares this arm "deliberately English-only" (data plan:
    # declared so the audit compares against the declaration) — the gate
    # re-verifies the declaration against the corpus bytes; one CJK
    # silver row is enough to fail it
    rows = full_silver()
    rows[7] = silver_row(7, request="获取阿迪达斯的p4工单")
    corpus = make_corpus(tmp_path / "corpus", rows)
    rc, stamp = run_gate(tmp_path, frozen_dir, corpus, clean_eval)
    assert rc == 2
    assert stamp["passed"] is False
    assert stamp["report"]["language"]["violations"] == 1
    assert any(f.startswith("language:") for f in stamp["failures"])


def test_enum_slot_share_reported_not_binding(tmp_path, frozen_dir,
                                              clean_eval):
    # every silver row enum-constrained: the share is MEASURED and
    # reported, and — per the 2026-10-10 re-registration — never fails
    # the gate even at a value the old 10% bound would have refused
    rows = [{"source": "bfcl-silver:live_multiple", "tool": ENUM_PROP_TOOL,
             "request": f"ship tag-{i} to the staging site",
             "gold": {"release": f"tag-{i}", "environment": "staging"},
             "evidence": [], "split": "train"} for i in range(1_000)]
    corpus = make_corpus(tmp_path / "corpus", rows)
    rc, stamp = run_gate(tmp_path, frozen_dir, corpus, clean_eval)
    assert rc == 0
    slots = stamp["report"]["enum_slots"]
    assert slots["binding"] is False
    assert slots["positive"] == 2_000
    # only the environment slot is enum-constrained: 1,000 of 2,000
    assert slots["enum"] == 1_000
    assert slots["share"] == 0.5


def test_default_arm_is_chatood(monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "audit_chatood", lambda: calls.append("c")
                        or 0)

    def _boom(**_k):
        raise AssertionError("stringparam audit ran without --arm")

    monkeypatch.setattr(gate, "audit_stringparam", _boom)
    assert gate.main([]) == 0
    assert calls == ["c"]


def test_arm_flag_dispatches_to_stringparam(monkeypatch):
    monkeypatch.setattr(gate, "audit_stringparam",
                        lambda **_k: 7)

    def _boom():
        raise AssertionError("--arm string-param-span-fill ran the "
                             "chat-ood audit")

    monkeypatch.setattr(gate, "audit_chatood", _boom)
    assert gate.main(["--arm", "string-param-span-fill"]) == 7


@pytest.mark.skipif(
    not (ROOT / "data/incoming/external/bfcl/calls.jsonl").is_file(),
    reason="committed calls.jsonl absent")
def test_calls_jsonl_now_in_chatood_eval_patterns():
    # gap fix 2026-10-10: the 183-row calls set was the one BFCL eval
    # surface the gate's disjointness patterns did not scan
    row = json.loads((ROOT / "data/incoming/external/bfcl/calls.jsonl")
                     .read_text(encoding="utf-8").splitlines()[0])
    assert gate.norm(row["text"]) in gate.eval_norms()


@pytest.mark.skipif(
    not (ROOT / "data/real/spanfill/train.jsonl").is_file(),
    reason="committed frozen spanfill corpus absent")
def test_registered_frozen_pins_match_committed_bytes():
    # the 2026-10-10 registration created train/dev sha pins (round 1
    # pinned only the reserve); this holds the constants to the actual
    # committed round-1 bytes so the pin cannot silently rot
    assert gate.STRINGPARAM_FROZEN_SHA_PINS["train"] == sha_of(
        ROOT / "data/real/spanfill/train.jsonl")
    assert gate.STRINGPARAM_FROZEN_SHA_PINS["dev"] == sha_of(
        ROOT / "data/real/spanfill/dev.jsonl")
