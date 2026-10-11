"""Mechanical gates — deterministic checks, sized for the 2 GB CPU sandbox.

Rules for this file:
- Every test must run in seconds, CPU-only, no network.
- No timing asserts: latency is flaky on shared runners and lives in the
  bench reports (reports/*.md), not in the test suite.
- A gate failure must name the budget that broke.
"""

import hashlib
import json
from pathlib import Path

import pytest

from mivia_small_json import pipeline
from mivia_small_json.model import MODEL_PATH
from mivia_small_json import validate as V
from mivia_small_json.schemas import LEVELS, SPEC

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "router.skops"
SIZE_BUDGET = 1_000_000  # bytes; ROI criterion cost gate
VALID_BUDGET = 0.99      # stop-rule floor from docs/plans/PLAN.md


def test_artifact_exists_and_within_budget():
    assert MODEL.exists(), "models/router.skops missing — run `train` first"
    size = MODEL.stat().st_size
    assert size <= SIZE_BUDGET, f"artifact {size} B exceeds {SIZE_BUDGET} B budget"


def test_sidecar_matches_artifact():
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    sidecar = (ROOT / "models" / "router.sha256").read_text().strip()
    assert digest == sidecar, "artifact does not match its sha256 sidecar"


ONNX_ARTIFACTS = (
    "router-encoder-int8.onnx",
    "router-tokenizer.json",
    "router-encoder-classes.json",
    "tool-match-int8.onnx",
    "tool-match-gate-int8.onnx",
    "tool-match-grade-int8.onnx",
    "severity-ordinal-int8.onnx",
    "intent-rank-int8.onnx",
    "intent-rank-mul-int8.onnx",
    "intent-rank-tokenizer.json",
    "intent-rank-mul-tokenizer.json",
)
ONNX_SIZE_BUDGET = 64_000_000  # bytes; the int8 guard/severity heads are ~23 MB each
# The router is the multilingual 12-layer encoder (owner-accepted 2026-10-10
# promotion): its int8 artifact is ~118 MB. Distinct budget, same purpose —
# catch an accidental fp32/oversized drop, not the accepted backbone.
ROUTER_ONNX_SIZE_BUDGET = 128_000_000


def test_shipped_onnx_artifacts_exist_and_match_their_sidecars():
    """Gate the runtime ONNX artifacts like the skops router.

    The loaders return None on a missing or unverified file, so without this
    gate a deleted guard or encoder surfaces as an AttributeError in the
    behavioural tests instead of a named budget failure.
    """
    for name in ONNX_ARTIFACTS:
        path = ROOT / "models" / name
        assert path.exists(), f"models/{name} missing — see the Commands table"
        size = path.stat().st_size
        budget = (ROUTER_ONNX_SIZE_BUDGET
                  if name.startswith("router-")
                  # the multilingual intent head shares the router's
                  # accepted 12-layer multilingual backbone (2026-10-08
                  # dual-head promotion): same 128 MB class
                  else (ROUTER_ONNX_SIZE_BUDGET
                        if name.startswith("intent-rank-mul")
                        else ONNX_SIZE_BUDGET))
        assert size <= budget, (
            f"models/{name} {size} B exceeds {budget} B budget")
        sidecar = (ROOT / "models" / f"{name}.sha256").read_text().strip()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sidecar.split()[0], (
            f"models/{name} does not match its sha256 sidecar")


def test_schema_registry_is_sane():
    for schema, spec in SPEC.items():
        assert spec["type"]["const"] == schema
        for field, rule in spec.items():
            if "enum" in rule:
                assert rule["enum"], f"{schema}.{field}: empty enum"
                assert all(isinstance(e, str) for e in rule["enum"])
            if "format" in rule:
                assert rule["format"] in ("iso8601", "sha")
            if field != "type":
                assert rule["type"] in (str, int, list)
    assert "TRACE" in LEVELS and "FATAL" in LEVELS  # OTel severity alignment


def test_heldout_valid_rate_meets_stop_rule():
    """Strict stop rule: the emitted object must be schema-valid AND the
    pipeline must have routed to the record's true schema."""
    rows = [json.loads(line) for line in
            (ROOT / "data" / "heldout" / "test.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    router = pipeline.load_router(MODEL_PATH)
    strict = sum(1 for row in rows
                 if (lambda o, m: m["valid"] and o.get("type") == row["schema"])(
                     *pipeline.classify(row["raw"], router)))
    rate = strict / len(rows)
    assert rate >= VALID_BUDGET, f"held-out routed-and-valid rate {rate:.4f} below {VALID_BUDGET}"


def test_validator_rejects_unknown_schema_members():
    assert "enum:level" in V.validate(
        {"type": "log_event", "level": "BANANA", "component": "c",
         "message": "m", "ts": "2026-03-01T12:00:00Z"}, "log_event")
    assert "enum:commit_type" in V.validate(
        {"type": "commit", "sha": "a1b2c3d", "message": "m", "files_changed": 1,
         "author": "a", "commit_type": "banana"}, "commit")


def test_spec_fingerprint_matches_manifest(corpus_world):
    """The active schema set must match the frozen corpus fingerprint."""
    from mivia_small_json import registry

    manifest = json.loads((ROOT / "data" / "heldout" / "manifest.json").read_text())
    assert registry.spec_fingerprint() == manifest["spec_sha256"], (
        "active schema set drifted from the frozen corpus fingerprint")


def test_pool_manifest_hashes_match_the_pool_files():
    """The pool manifest's recorded sha256s must match the committed files.

    A backfill or cycle that rewrites confident.jsonl / hard.jsonl without
    updating data/pool/manifest.json must fail here, not drift silently.
    """
    pool = ROOT / "data" / "pool"
    if not (pool / "manifest.json").is_file():
        # The public-release tree ships no improvement pool (runtime
        # material, not part of the model); the drift gate only means
        # something where the pool exists.
        pytest.skip("data/pool absent (public-release tree)")
    manifest = json.loads((pool / "manifest.json").read_text())
    for name in ("confident.jsonl", "hard.jsonl"):
        digest = "sha256:" + hashlib.sha256(
            (pool / name).read_bytes()).hexdigest()
        assert manifest["hashes"][name] == digest, (
            f"data/pool/manifest.json hash for {name} drifted from the file on disk")


SHIPPED_MANIFEST = ROOT / "models" / "SHIPPED.json"
_ARTIFACT_SUFFIXES = (".sha256", ".trainmeta.json")


def _manifest() -> dict:
    assert SHIPPED_MANIFEST.is_file(), (
        "models/SHIPPED.json missing — run scripts/refresh_shipped_manifest.py")
    return json.loads(SHIPPED_MANIFEST.read_text(encoding="utf-8"))["artifacts"]


def _artifacts_on_disk() -> set[str]:
    return {
        p.name for p in (ROOT / "models").iterdir()
        if p.is_file() and p.name not in {"SHIPPED.json", ".gitkeep"}
        and not p.name.endswith(_ARTIFACT_SUFFIXES)
    }


def test_every_shipped_artifact_is_in_the_manifest():
    """A new artifact cannot be added without listing it.

    Without this, an artifact could be dropped into `models/` and used while
    the manifest stayed silent about it.
    """
    listed = set(_manifest())
    on_disk = _artifacts_on_disk()
    assert on_disk == listed, (
        f"manifest drift: {sorted(on_disk ^ listed)} — re-run "
        "scripts/refresh_shipped_manifest.py")


def test_shipped_artifact_hashes_match_the_manifest():
    """A shipped artifact cannot change without a visible manifest update.

    The per-file `.sha256` sidecars prove a file matches itself, so whoever
    swaps a file can regenerate its sidecar and keep every other gate green.
    This manifest is the second key: a swap that does not update it fails here.
    """
    for name, meta in _manifest().items():
        path = ROOT / "models" / name
        assert path.is_file(), f"models/{name} is listed but missing"
        assert path.stat().st_size == meta["bytes"], (
            f"models/{name} size changed from {meta['bytes']} to "
            f"{path.stat().st_size}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == meta["sha256"], (
            f"models/{name} changed (sha256 {digest[:16]} vs manifest "
            f"{meta['sha256'][:16]}); re-run scripts/refresh_shipped_manifest.py "
            "in the same commit if this was deliberate")
