"""Smoke test for the artifact bench.

The bench is the measurement instrument behind every before/after number, so
a broken instrument would report a number rather than fail. This exercises it
end to end on the shipped artifact, at a row limit that keeps the suite fast.

The v2 tests pin the candidate measurement path (spec
jev-teacher-guard-distillation §7.1): ``--eval v2 --model
X`` must score X — never the shipped gate — and the payload must carry the
scored artifact's identity, because a v2 run that silently measures the
shipped artifact turns every candidate verdict into a false PASS.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mivia_small_json.tool_match import THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_tool_abstention.py"
MODEL = ROOT / "models" / "tool-match-int8.onnx"
GATE = ROOT / "models" / "tool-match-gate-int8.onnx"

# Every test here drives the bench end to end over the guard-eval-v2 set,
# which is not redistributed (docs/DATA.md). Skip, never fail; the private
# tree runs them for real against the built eval.
if not (ROOT / "data" / "incoming" / "external" / "bfcl"
        / "guard-eval-v2.jsonl").exists():
    pytest.skip("guard-eval-v2 absent (public tree; fetch per docs/DATA.md)",
                allow_module_level=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_v2(*extra: str, tmp_out: Path) -> subprocess.CompletedProcess:
    out = tmp_out / "v2.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--eval", "v2", "--json", str(out),
         *extra],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    return proc, out


def test_bench_reports_both_rates_on_the_shipped_artifact(tmp_path):
    assert MODEL.is_file(), "the shipped guard is missing"
    out = tmp_path / "bench.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--limit", "8", "--json", str(out)],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["abstention_rows"] == 8
    assert payload["retention_rows"] == 8
    assert payload["threshold"] == THRESHOLD
    assert len(payload["curve"]) == 7
    assert payload["peak_rss_mb"] > 0
    # The curve must behave like a threshold sweep, and the headline rates must
    # agree with it. A flat or inverted curve, or a headline outside the swept
    # range, means the measurement is broken.
    curve = payload["curve"]
    thr = [c["thr"] for c in curve]
    abstain = [c["abstain"] for c in curve]
    kept = [c["kept"] for c in curve]
    assert thr == sorted(thr), "curve thresholds are not ordered"
    assert all(a <= b for a, b in zip(abstain, abstain[1:])), (
        "abstention must not fall as the threshold rises")
    assert all(a >= b for a, b in zip(kept, kept[1:])), (
        "retention must not rise as the threshold rises")
    assert min(abstain) <= payload["abstention_pct"] <= max(abstain)
    assert min(kept) <= payload["retention_pct"] <= max(kept)


def test_v2_scores_the_requested_candidate_not_the_shipped_gate(tmp_path):
    """--eval v2 --model X stamps X's identity: the known bug measured the
    shipped gate no matter what --model said. The stamp is derived from the
    artifact measure_v2 actually loaded (the default path re-checks the
    SplitToolMatcher type), so a stamp/score divergence cannot recur.
    """
    proc, out = _run_v2("--model", str(MODEL), "--limit", "4",
                        tmp_out=tmp_path)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact"] == str(MODEL)
    assert payload["artifact_sha256"] == _sha(MODEL)
    assert payload["artifact_sha256"] != _sha(GATE)


def test_v2_decisions_dump_matches_the_payload_operating_point(tmp_path):
    """The paired-statistics feed must describe the same run as the
    payload: same artifact identity, threshold equal to the payload's
    operating point, and every v2 category present with catalogue keys.
    """
    out = tmp_path / "v2.json"
    dump_path = tmp_path / "decisions.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--eval", "v2", "--json", str(out),
         "--decisions", str(dump_path), "--limit", "4"],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    dump = json.loads(dump_path.read_text(encoding="utf-8"))
    assert dump["artifact"] == payload["artifact"]
    assert dump["artifact_sha256"] == payload["artifact_sha256"]
    # The payload prints the threshold at 4dp; the dump keeps full float
    # precision (decision-safe re-derivation).
    assert round(dump["threshold"], 4) == payload["operating_point"]["thr"]
    assert "single_turn_relevance" in dump["categories"]
    assert len(dump["categories"]) == 9
    assert all(isinstance(s, float)
               for scores in dump["categories"].values()
               for s in scores.values())


def test_v2_default_run_stamps_the_shipped_gate(tmp_path):
    proc, out = _run_v2("--limit", "4", tmp_out=tmp_path)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact"] == str(GATE)
    assert payload["artifact_sha256"] == _sha(GATE)


def test_v2_refuses_the_shipped_gate_as_candidate(tmp_path):
    proc, _ = _run_v2("--model", str(GATE), "--limit", "4", tmp_out=tmp_path)
    assert proc.returncode != 0
    assert "shipped" in (proc.stderr + proc.stdout).lower()


def test_v2_refuses_an_unloadable_candidate(tmp_path):
    corrupt = tmp_path / "broken-int8.onnx"
    corrupt.write_bytes(GATE.read_bytes() + b"\x00trailing garbage")
    proc, _ = _run_v2("--model", str(corrupt), "--limit", "4",
                      tmp_out=tmp_path)
    assert proc.returncode != 0
    assert "load" in (proc.stderr + proc.stdout).lower()
