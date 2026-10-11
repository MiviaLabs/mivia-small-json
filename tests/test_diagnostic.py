"""Tests for the diagnostic schema, routing, extraction, and sanitization."""

from pathlib import Path
import pytest

from mivia_small_json import pipeline, registry, rules
from mivia_small_json.diagnostic import (
    register_diagnostic_schema,
    sanitize_log,
)
from mivia_small_json.model import MODEL_PATH
from mivia_small_json.validate import validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def setup_diagnostic():
    register_diagnostic_schema()
    yield
    if "diagnostic" in registry.active():
        registry.unregister("diagnostic")


def test_diagnostic_registration():
    assert "diagnostic" in registry.active()
    spec = registry.resolve("diagnostic")
    assert spec["type"]["const"] == "diagnostic"
    assert "file" in spec
    assert "line" in spec
    assert "severity" in spec


def test_sanitize_log():
    # ANSI escape codes
    raw = "\x1b[31;1merror[E0308]: mismatched types\x1b[0m"
    clean = sanitize_log(raw)
    assert clean == "error[E0308]: mismatched types"

    # Carriage returns overwrite (progress bar)
    raw = "downloading [====>    ] 50%\rdownloading [========>] 100%\r\nfinished"
    clean = sanitize_log(raw)
    assert "50%" not in clean
    assert "finished" in clean

    # CI metadata wrapper
    raw = "2026-10-05T03:14:07.123Z ##[error]cargo build failed"
    clean = sanitize_log(raw)
    assert clean == "cargo build failed"


def test_diagnostic_routing_rules():
    rustc_sample = "error[E0308]: mismatched types\n  --> src/main.rs:12:18"
    assert rules.route(rustc_sample) == "diagnostic"

    tsc_sample = (
        "src/api/client.ts(45,7): error TS2322: Type 'string' is not assignable to type 'number'."
    )
    assert rules.route(tsc_sample) == "diagnostic"

    ruff_sample = "backend/server.py:3:1: F401 `os` imported but unused"
    assert rules.route(ruff_sample) == "diagnostic"

    pytest_sample = (
        "Traceback (most recent call last):\n"
        '  File "tests/test_foo.py", line 10\n'
        "AssertionError: assert 1 == 2"
    )
    assert rules.route(pytest_sample) == "diagnostic"


def test_diagnostic_routing_model():
    router = pipeline.load_router(MODEL_PATH)
    assert router is not None

    samples = [
        "error[E0425]: cannot find value `foo` in this scope\n  --> src/lib.rs:5:10",
        (
            "src/auth.ts(10,2): error TS2345: "
            "Argument of type 'number' is not assignable to parameter of type 'string'."
        ),
        "tests/test_model.py:25:1: E501 Line too long (105 > 100)",
        "FAILED tests/test_api.py::test_login - AssertionError: assert False",
    ]
    preds = router.predict(samples)
    for sample, pred in zip(samples, preds):
        assert pred == "diagnostic", f"Expected diagnostic for: {sample}, got {pred}"


def test_diagnostic_extraction_and_classify():
    router = pipeline.load_router(MODEL_PATH)
    sample = (
        "error[E0308]: mismatched types\n  --> src/main.rs:12:18\n   | expected `u32`, found `&str`"
    )
    obj, meta = pipeline.classify(sample, router=router)

    assert meta["valid"] is True
    assert obj["type"] == "diagnostic"
    assert obj["file"] == "src/main.rs"
    assert obj["line"] == 12
    assert obj["column"] == 18
    assert obj["code"] == "E0308"
    assert obj["severity"] in ("error", "warning", "note")
    assert validate(obj, "diagnostic") == []
