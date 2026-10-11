"""The frozen development reference is the reproducible trust basis.

The trust (:func:`label.measure_dev_rules`) and the router calibration
(:func:`label.measure_dev_calibration`) are measured on a committed
snapshot of the labeled development rows, not on the growing
``data/real/*/dev.jsonl`` files. Admitting rows there once re-baselined
the trust (n 2800 -> 5093, ``trusted_classes`` -> ``[]``) and switched
the weak-label flywheel off; the reference pins the basis instead.

Deterministic, offline, CPU-only, seconds.
"""

import hashlib
import json
from pathlib import Path

from mivia_small_json import bootstrap, label
from mivia_small_json.model import MODEL_PATH
from mivia_small_json.pipeline import load_router

ROOT = Path(__file__).resolve().parents[1]

# The growing dev files the trust no longer reads.
GROWING_DEV_FILES = (
    "data/synthetic/dev.jsonl",
    "data/real/issue/dev.jsonl",
    "data/real/commit/dev.jsonl",
    "data/real/log_event/dev.jsonl",
)


def _reference_rows():
    return [json.loads(line) for line in
            (ROOT / label.DEV_REFERENCE).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_frozen_reference_reproduces_the_trust_basis(corpus_world):
    """The committed reference is the exact basis the trust is pinned to."""
    measure = label.measure_dev_rules(ROOT)
    assert measure["n"] == 2800
    assert measure["trusted_classes"] == ["log_event"]
    assert measure["unknown_classes"] == ["agent_step"]
    # The trust reads the reference and nothing else.
    assert [f["path"] for f in measure["files"]] == [label.DEV_REFERENCE]
    digest = hashlib.sha256((ROOT / label.DEV_REFERENCE).read_bytes()).hexdigest()
    assert measure["files"][0]["sha256"] == digest
    # One row per dev record, in order, carrying raw + schema.
    rows = _reference_rows()
    assert len(rows) == 2800
    assert all(set(row) == {"raw", "schema"} for row in rows)
    assert [f["rows"] for f in measure["files"]] == [2800]


def test_dev_file_edit_does_not_change_the_trust(tmp_path):
    """A change to the growing dev files must not move the trust table."""
    reference = tmp_path / label.DEV_REFERENCE
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes((ROOT / label.DEV_REFERENCE).read_bytes())

    before = label.measure_dev_rules(tmp_path)
    assert before["trusted_classes"] == ["log_event"]

    # A hostile row appended to every growing dev file: a class the trust
    # would lose if the trust were still measured on those files.
    for rel in GROWING_DEV_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"raw": "zzz hostile row",
                                 "schema": "commit"}) + "\n")

    after = label.measure_dev_rules(tmp_path)
    assert after == before, "growing dev files re-baselined the trust"
    assert after["trusted_classes"] == ["log_event"]
    assert label._trust(after) == label._trust(before)


def test_calibration_gate_passes_on_the_frozen_reference():
    """The committed calibration still passes its gate on the reference."""
    bootstrap.register_example_schemas()
    calibration = label.calibration_from_disk(ROOT)
    assert calibration is not None
    label.check_calibration(calibration)
    assert label._tier1_enabled(calibration) is True
    # The record names the frozen reference, and its file hash is bound
    # to the committed bytes.
    assert calibration.source == label.DEV_REFERENCE
    record = json.loads((ROOT / "data" / "calibration.json")
                        .read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / label.DEV_REFERENCE).read_bytes()).hexdigest()
    assert record["files"] == [digest]
    assert record["router_sha256"] == (
        ROOT / "models" / "router.sha256").read_text(encoding="utf-8").strip()


def test_calibration_is_stable_when_the_dev_files_change(tmp_path):
    """The calibration measurement does not read the growing dev files."""
    bootstrap.register_example_schemas()
    reference = tmp_path / label.DEV_REFERENCE
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes((ROOT / label.DEV_REFERENCE).read_bytes())
    router = load_router(MODEL_PATH)

    before = label.measure_dev_calibration(tmp_path, router)

    for rel in GROWING_DEV_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"raw": "zzz hostile row",
                                 "schema": "commit"}) + "\n")

    after = label.measure_dev_calibration(tmp_path, router)
    assert after == before, "growing dev files re-baselined the calibration"


def test_record_dev_reference_round_trips_over_the_public_dev_sources(tmp_path):
    """A deliberate re-record over the PUBLIC dev sources reproduces the
    synthetic share of the reference.

    The committed record was baselined over sources that include private
    corpora (data/real/*/dev.jsonl) which the open-source tree does not
    redistribute, so the full-record round-trip above is a private-tree
    check. What the public tree CAN pin is the law this test exists for:
    re-recording over the sources it ships (synthetic dev) reproduces
    exactly those pairs, byte for byte, in order.
    """
    public = [rel for rel in GROWING_DEV_FILES
              if (ROOT / rel).is_file()]
    assert public, "no public dev sources shipped; fixture contract broken"
    for rel in public:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / rel).read_bytes())

    written = label.record_dev_reference(tmp_path)
    assert written == tmp_path / label.DEV_REFERENCE

    pairs, files = label.dev_rows(tmp_path)
    expected: list[tuple[str, str]] = []
    for rel, fixed_gold in label.DEV_SOURCES:
        if rel not in public:
            continue
        for line in (tmp_path / rel).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gold = fixed_gold if fixed_gold is not None else row.get("schema")
            if gold and row.get("raw"):
                expected.append((row["raw"], str(gold)))
    assert pairs == expected
    assert files == [{"path": label.DEV_REFERENCE, "rows": len(expected),
                      "sha256": hashlib.sha256(written.read_bytes()).hexdigest()}]
