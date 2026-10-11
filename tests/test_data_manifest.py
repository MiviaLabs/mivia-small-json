"""Gates for the committed dataset-hash manifest and its drift check.

Deterministic, offline, CPU-only, seconds. The manifest logic is driven
over a scratch tree under ``tmp_path`` so the real corpora are not
required to run the suite.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_dataset_manifest", ROOT / "scripts" / "check_dataset_manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_tree(root: Path) -> None:
    """A miniature committed-corpus layout plus a file that must be ignored."""
    for sub in ("synthetic", "real/issue", "heldout", "wild", "pool", "incoming"):
        (root / "data" / sub).mkdir(parents=True, exist_ok=True)
    files = {
        "data/synthetic/train.jsonl": '{"a": 1}\n{"a": 2}\n',
        "data/real/issue/train.jsonl": '{"b": 1}\n{"b": 2}\n{"b": 3}\n',
        "data/heldout/test.jsonl": '{"c": 1}\n',
        "data/wild/issue.jsonl": '{"d": 1}\n',
        "data/pool/confident.jsonl": '{"e": 1}\n',
        # Not matched by any corpus glob; must never enter the manifest.
        "data/incoming/huge.jsonl": '{"f": 1}\n',
    }
    for rel, body in files.items():
        (root / rel).write_text(body, encoding="utf-8")


def test_manifest_records_hash_and_rows_for_committed_corpora(tmp_path):
    mod = _load()
    _make_tree(tmp_path)

    manifest = mod.build_manifest(tmp_path)

    assert list(manifest) == sorted(manifest), "ordering must be deterministic"
    assert "data/synthetic/train.jsonl" in manifest
    assert "data/real/issue/train.jsonl" in manifest
    assert "data/heldout/test.jsonl" in manifest
    assert "data/wild/issue.jsonl" in manifest
    assert "data/pool/confident.jsonl" in manifest
    assert "data/incoming/huge.jsonl" not in manifest
    entry = manifest["data/real/issue/train.jsonl"]
    assert entry["rows"] == 3
    assert len(entry["sha256"]) == 64


def test_verify_passes_then_names_the_mutated_file(tmp_path):
    mod = _load()
    _make_tree(tmp_path)
    mod.write_manifest(tmp_path)

    assert mod.verify(tmp_path) == []

    target = tmp_path / "data" / "real" / "issue" / "train.jsonl"
    target.write_text('{"b": 1}\n{"b": 999}\n{"b": 3}\n', encoding="utf-8")

    drift = mod.verify(tmp_path)
    assert drift, "a changed file must fail verification"
    assert any("data/real/issue/train.jsonl" in message for message in drift), drift


def test_verify_flags_added_and_missing_files(tmp_path):
    mod = _load()
    _make_tree(tmp_path)
    mod.write_manifest(tmp_path)

    (tmp_path / "data" / "synthetic" / "dev.jsonl").write_text('{"g": 1}\n')
    (tmp_path / "data" / "wild" / "issue.jsonl").unlink()

    drift = mod.verify(tmp_path)
    assert any("added: data/synthetic/dev.jsonl" in m for m in drift), drift
    assert any("missing: data/wild/issue.jsonl" in m for m in drift), drift


def test_cli_write_then_verify_exit_codes(tmp_path):
    mod = _load()
    _make_tree(tmp_path)

    assert mod.main(["--write", "--root", str(tmp_path)]) == 0
    assert mod.main(["--root", str(tmp_path)]) == 0

    (tmp_path / "data" / "heldout" / "test.jsonl").write_text('{"c": 2}\n')
    assert mod.main(["--root", str(tmp_path)]) == 1
