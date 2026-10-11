"""Data-gate fixes for the string-param-span-fill arm (2026-10-10).

Three confirmed gate bugs, red-first, each a rule-12 gate (it ACTS):

- share breaches are ENFORCED: a composition bound breach (per-source
  share, template cap) fails the build with exit 2, naming each breach —
  recorded-but-ignored breaches are a vacuous gate;
- the declared English-only register (spec data plan: "this arm is
  deliberately English-only") is ENFORCED mechanically at admission: a
  request carrying CJK/Hangul characters is refused and COUNTED, named in
  the composition report, even when it would pass every other mechanical
  filter;
- a missing BFCL eval surface FAILS the build — the eval-window exclusion
  contract cannot be verified against an absent file — instead of the old
  silent skip.

The committed corpus bytes are the falsified round's pinned evidence
(models/spanfill-extractor-trainmeta.json binds their sha) and are NEVER
rewritten here: the committed-file tests are READ-ONLY demonstrations
that the new detector flags exactly the 15 CJK/Hangul rows recorded in
reports/training-report.md, and that the audit gate fails (honestly) on
those bytes.

Deterministic, CPU-only, no network, no timing asserts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # noqa: E402

import build_stringparam_corpus as b  # noqa: E402

COMMITTED_SILVER = ROOT / "data/real/spanfill-stringparam/bfcl-silver.jsonl"

# measured 2026-10-10 over the committed 1,043-row silver file (0-based,
# file order); pinned by tests against the bytes, never hand-copied
EXPECTED_CJK_ROWS = [30, 88, 90, 434, 437, 537, 579, 580, 581, 728, 765,
                     766, 929, 931, 994]

SILVER_TOOL = {"name": "search_tickets",
               "description": "Search support tickets by query.",
               "parameters": {"type": "object", "properties": {
                   "query": {"type": "string",
                             "description": "The search query."}},
                   "required": ["query"]}}


def _builder_argv(monkeypatch, *flags: str) -> None:
    monkeypatch.setattr(sys, "argv",
                        ["build_stringparam_corpus.py", *flags])


def _silver_stats(n_admitted: int) -> dict:
    return {"questions": 0, "string_family_calls": 0, "eval_blocked": 0,
            "aligned": n_admitted, "evidence": {}, "enum_refused": 0,
            "language_violation": 0}


def _gold_stats() -> dict:
    return {"scanned": 0, "refusals": {}, "admitted": 0}


def _probe(rows: list) -> dict:
    return {"n": len(rows), "easy_mass": 0, "signal": 0,
            "guard_rejected": 0, "abstain_target": 0}


def _silver_rows(n: int) -> list[dict]:
    return [{"source": "bfcl-silver-adjudicated:live_simple",
             "tool": SILVER_TOOL,
             "request": f"provision bucket-{i} for tenant-{i}",
             "gold": {"query": f"bucket-{i}"}, "evidence": [],
             "split": "train", "unanswerable": False}
            for i in range(n)]


def _stub_builds(monkeypatch, silver_rows: list[dict]) -> None:
    monkeypatch.setattr(b, "check_frozen", lambda: {
        "counts": dict(b.MIN_FROZEN), "reserve_sha256": "0" * 64,
        "paths": {}, "problems": []})
    monkeypatch.setattr(b, "build_silver",
                        lambda: (silver_rows, _silver_stats(len(silver_rows))))
    monkeypatch.setattr(b, "build_gold", lambda: ([], _gold_stats()))
    monkeypatch.setattr(b, "run_weakness_probe", _probe)


def _tiny_frozen(tmp_path: Path, monkeypatch) -> Path:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "train.jsonl").write_text(
        json.dumps({"request": "restart unit one"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(b, "FROZEN_DIR", frozen)
    return frozen


def _fixture_eval_dir(tmp_path: Path, monkeypatch) -> Path:
    """The three declared eval surfaces in miniature (hermetic: the tests
    never depend on the committed eval files' presence)."""
    d = tmp_path / "bfcl-eval"
    d.mkdir()
    (d / "guard-eval-v2.jsonl").write_text(
        json.dumps({"request": "eval window request alpha"}) + "\n",
        encoding="utf-8")
    (d / "calls.jsonl").write_text(
        json.dumps({"text": "Tools available:\n[]\n\nUser request: "
                            "eval catalogue request beta"}) + "\n",
        encoding="utf-8")
    (d / "irrelevance_eval.jsonl").write_text(
        json.dumps({"text": "Tools available:\n[]\n\nUser request: "
                            "eval catalogue request gamma"}) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(b, "BFCL_EVAL_DIR", d)
    return d


# --- bug 1: composition bound breaches must FAIL the build -------------------

def test_share_breach_fails_the_build(tmp_path, monkeypatch, capsys):
    """9,000 silver rows against the registered 12,407 frozen rows push
    BOTH declared share bands out while every row minimum holds — the
    build must fail (exit 2) naming each breach, never emit."""
    monkeypatch.setattr(b, "OUT", tmp_path / "out")
    _tiny_frozen(tmp_path, monkeypatch)
    _stub_builds(monkeypatch, _silver_rows(9_000))
    _builder_argv(monkeypatch, "--build")
    # main() RETURNS the gate's exit code (the __main__ wrapper turns it
    # into the process exit); the old SystemExit-style assertion on this
    # path was satisfiable by argparse accidents, not by the gate
    assert b.main() == 2
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # frozen share 12,407/21,407 = 0.5796 < 0.60; silver 0.4204 > 0.40
    assert "frozen share" in combined
    assert "bfcl-silver share" in combined
    assert "composition bound breach" in combined
    assert not (tmp_path / "out").exists(), \
        "a failing builder must write nothing"


def test_composition_within_bounds_still_builds(tmp_path, monkeypatch):
    """The passing path stays green: 1,000 silver rows inside every
    declared bound build and write to OUT."""
    monkeypatch.setattr(b, "OUT", tmp_path / "out")
    _tiny_frozen(tmp_path, monkeypatch)
    _stub_builds(monkeypatch, _silver_rows(1_000))
    _builder_argv(monkeypatch, "--build")
    assert b.main() == 0
    assert (tmp_path / "out" / "bfcl-silver.jsonl").is_file()
    assert (tmp_path / "out" / "composition-report.json").is_file()
    report = json.loads(
        (tmp_path / "out" / "composition-report.json").read_text())
    assert report["composition"]["breaches"] == []
    assert report["shortfalls"] == []


# --- bug 2a: the declared English-only register, enforced at admission -------

def test_language_detector_cjk_hangul_ranges():
    """The mechanical detector: Han ideographs, Kana, Hangul syllables and
    jamo (compat + extended) flag; plain English — including digits,
    punctuation and NFKC-foldable fullwidth Latin — does not."""
    assert b.non_english_chars("다용도실 통돌이 중지") != ""
    assert b.non_english_chars("我想知道上海目前的天气状况") != ""
    assert b.non_english_chars(
        "Listen to the phrase '我爱学习' in a male voice") != ""
    assert b.non_english_chars("\u3131") != ""     # Hangul compat jamo
    assert b.non_english_chars("\uac00") != ""     # first Hangul syllable
    assert b.non_english_chars("\ud7a3") != ""     # last Hangul syllable
    assert b.non_english_chars("\u1100") != ""     # Hangul jamo block
    assert b.non_english_chars("\u4e00") != ""     # first CJK ideograph
    assert b.non_english_chars("\u9fff") != ""     # last CJK basic block
    assert b.non_english_chars("\u3042") != ""     # Hiragana
    assert b.non_english_chars("provision bucket-7 for tenant-7!") == ""
    assert b.non_english_chars("ＡＢＣ １２３ — em dash") == ""
    assert b.non_english_chars("") == ""
    assert b.non_english_chars(None) == ""


def test_non_english_row_refused_at_admission(tmp_path, monkeypatch):
    """A CJK/Hangul request is refused and COUNTED at admission, even when
    it would pass every other mechanical filter (eval window, enum,
    verbatim presence). The English row is admitted."""
    _fixture_eval_dir(tmp_path, monkeypatch)
    merged = tmp_path / "merged-agreements.jsonl"
    en = {"request": "search tickets for bucket-7", "tool": SILVER_TOOL,
          "agreed": "answerable", "slots": {"query": "bucket-7"},
          "split": "live_simple"}
    zh = {"request": "获取阿迪达斯的p4工单", "tool": SILVER_TOOL,
          "agreed": "answerable", "slots": {"query": "p4"},
          "split": "live_simple"}
    ko = {"request": "hey do 거실 에어컨 실행", "tool": SILVER_TOOL,
          "agreed": "answerable", "slots": {"query": "실행"},
          "split": "live_simple"}
    merged.write_text(
        "".join(json.dumps(r) + "\n" for r in (en, zh, ko)),
        encoding="utf-8")
    rows, stats = b.build_silver_adjudicated(merged)
    assert [r["request"] for r in rows] == [en["request"]]
    assert stats["admitted"] == 1
    assert stats["language_violation"] == 2


def test_ladder_silver_path_refuses_non_english(tmp_path, monkeypatch):
    """The mechanical ladder path (build_silver) enforces the same
    declared register: the CJK question is refused, counted, named."""
    _fixture_eval_dir(tmp_path, monkeypatch)
    question = {"id": "q-1", "question": [
        [{"content": "获取阿迪达斯的p4工单"}]],
        "function": [SILVER_TOOL]}
    answer = {"id": "q-1", "ground_truth": [{"search_tickets":
                                             {"query": ["p4"]}}]}
    fetches = {b.BFCL_BASE + "/BFCL_v4_simple_python.json": [question],
               b.BFCL_BASE + "/possible_answer/BFCL_v4_simple_python.json":
               [answer]}
    monkeypatch.setattr(
        b, "_fetch", lambda url, cached: fetches.get(url, []))
    rows, stats = b.build_silver()
    assert rows == []
    assert stats["language_violation"] == 1


# --- bug 3: a missing eval surface FAILS, never skips ------------------------

def test_eval_request_norms_raise_when_surface_missing(tmp_path, monkeypatch):
    empty = tmp_path / "bfcl-eval-none"
    empty.mkdir()
    monkeypatch.setattr(b, "BFCL_EVAL_DIR", empty)
    with pytest.raises(FileNotFoundError) as exc:
        b.eval_request_norms()
    assert "guard-eval-v2.jsonl" in str(exc.value)


def test_missing_eval_surface_fails_the_build(tmp_path, monkeypatch, capsys):
    """The build fails (exit 2) naming the missing surface — BEFORE any
    fetch — and writes nothing. The old skip silently built with an empty
    blocked set, weakening the exclusion contract."""
    monkeypatch.setattr(b, "OUT", tmp_path / "out")
    _tiny_frozen(tmp_path, monkeypatch)
    _fixture_eval_dir(tmp_path, monkeypatch)
    for name in ("calls.jsonl", "guard-eval-v2.jsonl",
                 "irrelevance_eval.jsonl"):
        (tmp_path / "bfcl-eval" / name).unlink()
    # the frozen probe is not under test here — only the eval-surface
    # gate firing before any fetch
    monkeypatch.setattr(b, "check_frozen", lambda: {
        "counts": dict(b.MIN_FROZEN), "reserve_sha256": "0" * 64,
        "paths": {}, "problems": []})
    monkeypatch.setattr(b, "_fetch",
                        lambda url, cached: (_ for _ in ()).throw(
                            AssertionError("network reached before the "
                                           "eval-surface gate")))
    _builder_argv(monkeypatch, "--build")
    assert b.main() == 2
    err = capsys.readouterr().err
    assert "eval surface" in err
    assert "guard-eval-v2.jsonl" in err
    assert not (tmp_path / "out").exists()


# --- bug 2b: read-only demonstrations against the COMMITTED bytes ------------

@pytest.mark.skipif(not COMMITTED_SILVER.is_file(),
                    reason="committed bfcl-silver.jsonl absent")
def test_committed_silver_cjk_rows_found_by_the_same_detector():
    """The NEW admission detector, run READ-ONLY over the committed
    (falsified-round, sha-pinned, never rewritten) silver file, flags
    exactly the 15 CJK/Hangul rows recorded in the training report."""
    rows = [json.loads(line)
            for line in COMMITTED_SILVER.read_text(
                encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1_043
    flagged = [i for i, r in enumerate(rows)
               if b.non_english_chars(r.get("request") or "")]
    assert flagged == EXPECTED_CJK_ROWS


@pytest.mark.skipif(
    not COMMITTED_SILVER.is_file()
    or not (ROOT / "data/real/spanfill/train.jsonl").is_file()
    or not (ROOT / "data/real/spanfill/dev.jsonl").is_file()
    or not (ROOT / "data/heldout/spanfill-reserve.jsonl").is_file()
    or not (ROOT / "data/real/spanfill/PROVENANCE.md").is_file()
    or not (ROOT / "data/incoming/external/bfcl/guard-eval-v2.jsonl").is_file()
    or not (ROOT / "data/incoming/external/bfcl/calls.jsonl").is_file()
    or not (ROOT / "data/incoming/external/bfcl/irrelevance_eval.jsonl")
    .is_file(),
    reason="committed corpus/eval bytes absent")
def test_committed_corpus_trips_the_audit_language_gate(tmp_path):
    """The audit gate, over the COMMITTED bytes (the stamp is redirected
    to tmp — data/ and .scratch are never written), fails on exactly the
    declared-language breach: 15 CJK/Hangul rows outside the declared
    English-only register. Every other gate measures clean — the language
    breach is the ONLY failure, the honest red state of a falsified
    round."""
    import audit_training_mix as gate

    rc = gate.audit_stringparam(stamp_path=tmp_path / "stamp.json")
    assert rc == 2
    stamp = json.loads((tmp_path / "stamp.json").read_text())
    assert stamp["passed"] is False
    language = [f for f in stamp["failures"] if f.startswith("language:")]
    assert len(language) == 1
    assert "15" in language[0]
    assert stamp["report"]["language"]["violations"] == 15
    others = [f for f in stamp["failures"]
              if not f.startswith("language:")]
    assert others == [], f"unexpected unrelated failures: {others}"
