"""Synth tests: determinism, stratified splits, gold validity, frozen corpus."""

import hashlib
import json
from pathlib import Path

import pytest

from mivia_small_json import synth

ROOT = Path(__file__).resolve().parents[1]


def _digest(splits):
    payload = "".join(
        json.dumps(r, sort_keys=True) + "\n"
        for split in splits.values() for r in split
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_generation_is_deterministic():
    assert _digest(synth.build_splits(15, 42)) == _digest(synth.build_splits(15, 42))
    assert _digest(synth.build_splits(15, 42)) != _digest(synth.build_splits(15, 7))


def test_split_fractions_are_stratified():
    splits = synth.build_splits(30, 42)
    expected = {"train": 24, "dev": 3, "test": 3}
    for split, count in expected.items():
        per = {s: 0 for s in synth.SCHEMA_ORDER}
        for r in splits[split]:
            per[r["schema"]] += 1
        assert set(per.values()) == {count}, f"{split} not stratified: {per}"


def test_gold_records_are_strictly_valid_and_raw_degraded():
    from mivia_small_json.validate import validate
    splits = synth.build_splits(10, 42)
    for records in splits.values():
        for r in records:
            assert validate(r["gold"], r["schema"]) == []
            assert set(r) == {"schema", "raw", "gold"}
            assert isinstance(r["raw"], str) and r["raw"].strip()


def test_committed_corpus_matches_manifest():
    held = ROOT / "data" / "heldout"
    manifest_path = held / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("corpus not generated yet")
    manifest = json.loads(manifest_path.read_text())
    test_bytes = (held / "test.jsonl").read_bytes()
    assert hashlib.sha256(test_bytes).hexdigest() == manifest["test_sha256"]
    rows = [json.loads(line) for line in test_bytes.decode().splitlines() if line]
    assert len(rows) == manifest["records"]["test"]
    assert manifest["records"]["train"] == 8 * manifest["records"]["test"]  # 80/10/10
    assert (held / "test.jsonl").stat().st_size < 5 * 1024 * 1024
