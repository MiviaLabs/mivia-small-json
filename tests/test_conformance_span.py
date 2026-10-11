"""TRACK 2: per-value conformance span + validity contrasts.

Unit tests pin the new renderer (CPU-only, deterministic).
test_conformance_bar_error_param_lift is a passing FALSIFICATION RECORD:
the standalone fine-tune arm was measured and failed the pre-registered
bar (reports/exp-conformance-gate.md); it flips on purpose only if a
future candidate clears it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from render_conformance_span import (  # noqa: E402
    check_value_conformance,
    extract_candidate_values,
    render_conformance_span,
)
from mivia_small_json.tool_match import (  # noqa: E402
    render_tool_fillability,
)

TOOL = {
    "name": "NextBusPrediction",
    "description": "Predict the arrival time of the next bus.",
    "parameters": {
        "properties": {
            "bus_service": {"type": "string", "pattern": r"^\d{1,4}$"},
            "location_name": {"type": "string", "minLength": 5,
                              "maxLength": 100,
                              "pattern": r"^[A-Za-z][A-Za-z ]{4,99}$"},
        },
        "required": ["location_name"],
        "type": "dict",
    },
}


def test_pattern_violation_detected():
    assert check_value_conformance("Beijing321",
                                   {"type": "string",
                                    "pattern": r"^\d{1,4}$"})
    assert not check_value_conformance("321",
                                       {"type": "string",
                                        "pattern": r"^\d{1,4}$"})


def test_type_and_length_constraints():
    assert check_value_conformance("abc", {"type": "integer"})
    assert not check_value_conformance("-42", {"type": "integer"})
    assert check_value_conformance("x", {"type": "string", "minLength": 5})
    assert check_value_conformance("z" * 101,
                                   {"type": "string", "maxLength": 100})
    assert not check_value_conformance("Beijing Central",
                                       {"type": "string", "minLength": 5,
                                        "maxLength": 100})


def test_enum_and_range():
    assert check_value_conformance("maybe", {"type": "string",
                                             "enum": ["true", "false"]})
    assert not check_value_conformance("true", {"type": "string",
                                                "enum": ["true", "false"]})
    assert check_value_conformance("99", {"type": "integer",
                                          "maximum": 4})
    assert not check_value_conformance("3", {"type": "integer",
                                             "minimum": 1, "maximum": 4})


def test_span_appends_conformance_tail():
    req = 'Next bus at location_name="Beijing Central Station"?'
    span = render_conformance_span(TOOL, req)
    assert span.startswith(render_tool_fillability(TOOL, req))
    assert "conformance:" in span
    assert "location_name=Beijing Central Station:pass" in span


def test_span_marks_violation():
    req = 'Bus route from location_name="Beijing321" now?'
    span = render_conformance_span(TOOL, req)
    assert "fail(" in span and "location_name=Beijing321" in span


def test_span_marks_missing_required():
    span = render_conformance_span(TOOL, "When is the next bus?")
    assert "location_name=missing:required" in span


def test_extraction_is_deterministic():
    req = 'Bus route from location_name="Beijing321" now?'
    assert (extract_candidate_values(TOOL, req)
            == extract_candidate_values(TOOL, req))
    assert extract_candidate_values(TOOL, req)["location_name"] == \
        "Beijing321"


def test_shipped_span_lacks_conformance_falsification_record():
    """Pins the diagnosed gap: the shipped gate span renders no
    value-conformance, so no head could learn error_param refusal."""
    req = 'Bus route from location_name="Beijing321" now?'
    assert "conformance:" not in render_tool_fillability(TOOL, req)


# (falsification-record gate retired with reports/: the experiment
# record is not redistributed; the private tree keeps the gate.)
