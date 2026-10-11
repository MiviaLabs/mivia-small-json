"""Smoke tests for the out-of-distribution routing benchmark."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_external_hard", ROOT / "scripts" / "bench_external_hard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sets_cover_the_expected_schemas():
    sets = _load()._sets()
    names = {name.split(":")[0] for name, _path, _expected in sets}
    assert {"nlbse", "cleancodereview", "bfcl"} <= names
    expected = {exp for _name, _path, exp in sets}
    assert {"issue", "review_comment", "tool_call"} <= expected


def test_rows_reads_text_and_skips_blank(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"text": "a"}\n\n{"text": "b"}\n{"nope": 1}\n',
                    encoding="utf-8")
    assert _load()._rows(path) == ["a", "b"]


def test_rows_of_a_missing_file_is_empty(tmp_path):
    assert _load()._rows(tmp_path / "absent.jsonl") == []


# (provenance-file gate retired: third-party data and its per-set
# provenance docs are not redistributed; see docs/DATA.md)
