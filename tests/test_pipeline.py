"""Phase 3 integration tests: sha-verified load, never-raise classify, CLI."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from mivia_small_json import pipeline
from mivia_small_json.model import MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]

FIXTURES = [
    "ISS-4021 (high) Labels: api, blocker State: open - Pagination breaks on page>2",
    "[auth] : ERROR token refresh failed msg=grant rejected 2026-10-03T10:15:00Z",
    "commit 9f31c0ba2e5d1a7f4c8e6b2d0a9f31c0ba2e5d1a author: Dana Lee - "
    "4 files changed, fix flaky retry test",
    "",
]

_BAD_SIDECAR = "0" * 64


def _artifact_copy(tmp_path: Path) -> Path:
    art = tmp_path / "router.skops"
    shutil.copyfile(MODEL_PATH, art)
    return art


# --- load_router: integrity gate -------------------------------------------


def test_load_router_sha_ok():
    assert pipeline.load_router(MODEL_PATH) is not None


def test_load_router_missing_artifact(tmp_path):
    assert pipeline.load_router(tmp_path / "absent.skops") is None


def test_load_router_missing_sidecar(tmp_path):
    assert pipeline.load_router(_artifact_copy(tmp_path)) is None


def test_load_router_malformed_sidecar(tmp_path):
    art = _artifact_copy(tmp_path)
    (tmp_path / "router.sha256").write_text("xyz\n", encoding="utf-8")
    assert pipeline.load_router(art) is None


def test_load_router_sha_mismatch(tmp_path):
    art = _artifact_copy(tmp_path)
    (tmp_path / "router.sha256").write_text(_BAD_SIDECAR + "\n", encoding="utf-8")
    assert pipeline.load_router(art) is None
    # degraded load => rules fallback, per the integration contract
    obj, meta = pipeline.classify(FIXTURES[0], pipeline.load_router(art))
    assert meta["routed_by"] == "rules"
    assert meta["valid"] is True


def test_load_router_corrupt_artifact(tmp_path):
    art = tmp_path / "router.skops"
    art.write_bytes(b"definitely not a skops artifact")
    digest = hashlib.sha256(art.read_bytes()).hexdigest()
    (tmp_path / "router.sha256").write_text(digest + "\n", encoding="utf-8")
    assert pipeline.load_router(art) is None  # sha passes, skops load fails


# --- classify: never-raise contract ----------------------------------------


def _hostile_strings():
    base = [
        "",
        " ",
        "\n",
        "\t\t\r\n",
        "\x00",
        "\x00\x01\x02\x1b[31m",
        "a" * 100_000,
        "null\x00in\x00string\x00",
        "\ufffd" * 1000,
        "é" * 100 + "你好世界" + "🚀🔥✅",
        "([{}])" * 500,
        "^(?:a+)+$" * 100,
        "-" * 5000,
        "#123 #124 #125 " * 400,
        "ISS-" + "9" * 6 + " priority: critical state: closed labels: api",
        "ERROR [x]: " + "msg " * 2000,
        "commit " + "f" * 40 + " author: X - 0 files changed",
        "2026-13-45T99:99:99Z not-a-timestamp {broken json",
        "\x07\x08\x0b\x0c" * 250,
        "🚀" * 5000,
        # pull_request-shaped garbage (registered-schema world)
        "pull_request: number: bananas :: title: \x00 :: state: banana :: "
        "additions: -",
        "PR #" + "9" * 500,
        "PR #42: " + "x" * 5000 + " (+999999999999 -1)",
        "pull request " * 3000,
        "merged +1 -1 state: MERGED author: None #undefined 🚀",
        "pull_request: number: \x071 :: author: " + "é" * 2000,
    ]
    generated = [
        f"line {i} \x07 {'x' * (i % 97)} ISS-{i:04d} priority: high"
        for i in range(180)
    ]
    return base + generated


def test_classify_never_raises_fuzz():
    router = pipeline.load_router(MODEL_PATH)
    assert router is not None
    for s in _hostile_strings():
        for r in (None, router):
            obj, meta = pipeline.classify(s, r)
            tag = f"input={s[:60]!r} router={'model' if r is router else 'none'}"
            assert isinstance(obj, dict), tag
            assert set(meta) == {"routed_by", "valid", "retries", "latency_ms"}, tag
            assert meta["routed_by"] in {"model", "rules", "template"}, tag
            assert isinstance(meta["valid"], bool), tag
            assert isinstance(meta["retries"], int) and 0 <= meta["retries"] <= 2, tag
            assert meta["latency_ms"] >= 0.0, tag


def test_classify_rules_only_when_router_none():
    obj, meta = pipeline.classify(FIXTURES[0], None)
    assert meta["routed_by"] == "rules"
    assert obj["type"] == "issue"
    assert meta["valid"] is True and meta["retries"] <= 2


class _RaisingRouter:
    def predict_proba(self, texts):
        raise RuntimeError("boom")

    def predict(self, texts):
        raise RuntimeError("boom")


def test_classify_warns_when_model_route_fails(capsys):
    """A broken router degrades to rules loudly, not silently."""
    obj, meta = pipeline.classify(FIXTURES[0], _RaisingRouter())
    assert meta["routed_by"] == "rules"
    assert meta["valid"] is True
    assert obj["type"] == "issue"
    err = capsys.readouterr().err
    assert "router failed" in err and "boom" in err, err


class _LowConfidenceRouter:
    """Always predicts 'commit' with confidence below the override threshold."""

    def __init__(self, confidence=0.35):
        self._confidence = confidence

    def predict_proba(self, texts):
        import numpy as np

        rest = (1.0 - self._confidence) / 2.0
        return np.array([[self._confidence, rest, rest]] * len(texts))

    def predict(self, texts):
        return ["commit"] * len(texts)


def test_low_confidence_keeps_the_model_without_a_rules_signal():
    """No rules signal must not silently reroute a low-confidence model."""
    obj, meta = pipeline.classify("zzz qqq www nothing here at all",
                                  _LowConfidenceRouter())
    assert meta["routed_by"] == "model"
    assert obj["type"] == "commit"


def test_low_confidence_defers_to_a_rules_signal():
    text = "2026-03-01T12:00:00Z ERROR [auth]: token validation failed"
    obj, meta = pipeline.classify(text, _LowConfidenceRouter())
    assert meta["routed_by"] == "rules"
    assert obj["type"] == "log_event"


# --- determinism across loads -----------------------------------------------


def test_two_loads_identical_predictions():
    r1 = pipeline.load_router(MODEL_PATH)
    r2 = pipeline.load_router(MODEL_PATH)
    assert r1 is not None and r2 is not None
    for text in FIXTURES:
        o1, m1 = pipeline.classify(text, r1)
        o2, m2 = pipeline.classify(text, r2)
        assert o1 == o2
        assert (m1["routed_by"], m1["valid"], m1["retries"]) == (
            m2["routed_by"], m2["valid"], m2["retries"],
        )
    obj, meta = pipeline.classify(FIXTURES[0], r1)
    assert meta["routed_by"] == "model"  # sha-verified load routes via model
    assert obj["type"] == "issue" and meta["valid"] is True


# --- CLI end-to-end ----------------------------------------------------------


def _run_cli(args, payload, timeout=180):
    import os

    import mivia_small_json

    # The subprocess must find the package where THIS process found it
    # (an installed package normally; the mutant sandbox's src/ tree
    # under the mutation scan).
    pkg_root = str(Path(mivia_small_json.__file__).resolve().parent.parent)
    env = {**os.environ,
           "PYTHONPATH": pkg_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(
        [sys.executable, "-m", "mivia_small_json.infer", *args],
        input=payload,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=timeout,
        env=env,
    )


def test_cli_end_to_end():
    payload = "\n".join([
        "ISS-1001 (low) Labels: docs State: open - Typo in README",
        "",
        "garbage \x00 line 🚀 (((broken",
        "x" * 100_000,
        "commit " + "a" * 40 + " author: Bo Li - 2 files changed, fix tests",
    ]) + "\n"
    proc = _run_cli([], payload)
    assert proc.returncode == 0
    lines = [line for line in proc.stdout.splitlines() if line]
    assert len(lines) == 4  # blank line skipped
    records = [json.loads(line) for line in lines]
    assert all(set(r) == {"input", "obj", "meta"} for r in records)
    assert all(set(r["meta"]) == {"routed_by", "valid", "retries", "latency_ms"}
               for r in records)
    assert records[0]["obj"]["type"] == "issue" and records[0]["meta"]["valid"] is True
    assert records[1]["input"].startswith("garbage")  # hostile line still emitted


def test_cli_bad_model_still_exit_zero(tmp_path):
    art = tmp_path / "router.skops"
    art.write_bytes(b"junk")
    (tmp_path / "router.sha256").write_text(
        hashlib.sha256(b"junk").hexdigest() + "\n", encoding="utf-8"
    )
    proc = _run_cli(["--model", str(art)], "ISS-2002 (high) state: closed\n")
    assert proc.returncode == 0
    record = json.loads(proc.stdout.splitlines()[0])
    assert record["meta"]["routed_by"] == "rules"  # corrupt artifact => fallback


def test_load_level_extractor_sha_ok():
    ext = pipeline.load_level_extractor()
    assert ext is not None


def test_load_level_extractor_missing_or_corrupt(tmp_path):
    assert pipeline.load_level_extractor(tmp_path / "nonexistent.skops") is None
    corrupt = tmp_path / "corrupt.skops"
    corrupt.write_bytes(b"corrupt")
    (tmp_path / "corrupt.skops.sha256").write_text("0" * 64 + "\n")
    assert pipeline.load_level_extractor(corrupt) is None


def test_classify_learned_level_extraction():
    # Android LogCat line: single letter level 'D' extracts as 'DEBUG'
    router = pipeline.load_router(MODEL_PATH)
    line = "03-17 16:13:38.811  1702  2395 D WindowManager: printFreezingDisplayLogs"
    obj, meta = pipeline.classify(line, router)
    assert obj["type"] == "log_event"
    assert obj["level"] == "DEBUG"
    assert meta["valid"] is True


def test_classify_dump_multi_logs_with_chat():
    router = pipeline.load_router(MODEL_PATH)
    dump = """hey what is this error?
Jun 14 15:16:01 combo sshd[19939]: authentication failure; rhost=1.2.3.4
Jun 14 15:16:02 combo sshd[19937]: check pass; user unknown
Jun 15 02:04:59 combo sshd[20882]: session closed for user root
"""
    records = pipeline.classify_dump(dump, router)
    # 2026-10-10 review N2: the chat preamble carries no rules markers and
    # used to be silently dropped by the is_content gate; the fix keeps
    # router-scored chunks for ANY schema, so it is kept as a valid non-log
    # record ahead of the three log events. Which non-log schema it gets
    # depends on the ACTIVE schema world, not on this surface (conftest-
    # minimal: issue/rules degraded; shipped world booted: the skops
    # verdict review_comment/model) — the N2 property under test is KEPT.
    assert len(records) == 4
    assert records[0][0]["type"] in {"issue", "review_comment"}
    assert records[0][1]["valid"] is True
    for obj, meta in records[1:]:
        assert obj["type"] == "log_event"
        assert meta["valid"] is True
    assert records[1][0]["component"] == "sshd"
    assert records[1][0]["level"] == "ERROR"
    assert records[2][0]["level"] == "WARN"
    assert records[3][0]["level"] == "INFO"


def test_classify_dump_preserves_single_entity():
    router = pipeline.load_router(MODEL_PATH)
    issue_text = """ISS-500: System crashes on boot
When user boots machine, system panics with exit code 1.
"""
    records = pipeline.classify_dump(issue_text, router)
    assert len(records) == 1
    assert records[0][0]["type"] == "issue"
    assert records[0][0]["id"] == "ISS-500"


def test_classify_dump_keeps_atomic_entity_beside_logs():
    """A multi-line entity must survive a stream that also has log lines."""
    router = pipeline.load_router(MODEL_PATH)
    dump = (
        "ISS-500: System crashes on boot\n"
        "When user boots machine, system panics with exit code 1.\n"
        "Jun 14 15:16:01 combo sshd[19939]: authentication failure; rhost=1.2.3.4\n"
        "Jun 14 15:16:02 combo sshd[19937]: check pass; user unknown\n"
    )
    records = pipeline.classify_dump(dump, router)
    types = [obj["type"] for obj, _ in records]
    # 2026-10-10 review N2: the issue's second line routes issue/model
    # with no rules markers and used to be silently dropped by the dump
    # is_content gate; the fix keeps router-scored chunks for ANY schema,
    # so the entity survives as two issue records instead of one.
    assert types.count("issue") == 2, types
    assert types.count("log_event") == 2, types


def test_classify_dump_preserves_issue_with_traceback():
    """An issue with traceback steps must remain an atomic single entity."""
    # The shipped router (ONNX, issue @ 0.998 on this chunk). The skops
    # fallback prefers diagnostic here (0.64); the product path is what
    # the atomicity contract is pinned against.
    router = pipeline.load_active_router()
    dump = (
        "ISS-600: URLField clean fails\n"
        "Description:\n"
        "forms.URLField().clean('bad')\n"
        "Traceback (most recent call last):\n"
        "  File \"fields.py\", line 151, in clean\n"
        "ValueError: Invalid URL\n"
    )
    records = pipeline.classify_dump(dump, router)
    assert len(records) == 1
    assert records[0][0]["type"] == "issue"



