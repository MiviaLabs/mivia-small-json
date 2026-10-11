"""Structural decision layer: log-shaped text routes log_event.

The 2026-10-08 realworld-slices re-measure exposed a decision-layer
regression: long log dumps (HDFS, Apache, Proxifier, Windows CBS, HPC
RAS) route to ``issue`` / ``text`` with high encoder confidence even
though their LINE STRUCTURE is a log by construction. The fix is
deterministic and lives in the decision layer (measured before any
training arm): uniform timestamp-prefixed line coverage routes
``log_event`` with ``routed_by == "structure"``, ahead of both the
confident fast path and the confident ``text`` emission.

A fenced-error-paste sibling rule was measured and FALSIFIED the same
day (real issue bodies embed fenced tracebacks); the test below pins
that scope decision. The wild slices themselves are gated separately
(tests/test_realworld_bars.py).
"""
from __future__ import annotations

import pytest

from mivia_small_json import bootstrap, pipeline, registry
from mivia_small_json.encoder import load_encoder

pytestmark = pytest.mark.skipif(
    load_encoder() is None, reason="shipped ONNX encoder absent (rules-only checkout)"
)


def _route_type(text: str) -> tuple[str, str]:
    bootstrap.register_example_schemas()
    router = pipeline.load_active_router()
    obj, meta = pipeline.classify(text, router)
    return obj["type"], meta["routed_by"]


def _hdfs_dump(lines: int = 10) -> str:
    return "\n".join(
        f"2015-10-18 18:01:{s:02d},963 INFO [main] org.apache.hadoop.mapred"
        f".TaskRunner Thread-{s}: Writing output for task attempt_local{s:04d}"
        for s in range(1, lines + 1)
    )


def _apache_dump(lines: int = 9) -> str:
    return "\n".join(
        f"[Sun Dec 04 04:47:{s:02d} 2005] [notice] jk2_init() Found child {s} "
        "in scoreboard slot 7"
        for s in range(1, lines + 1)
    )


def _proxifier_dump(lines: int = 9) -> str:
    return "\n".join(
        f"[10.30 16:49:{s:02d}] chrome.exe - proxy.cse.cuhk.edu.hk:5070 open "
        "through proxy proxy.cse.cuhk.edu.hk:5070 HTTPS"
        for s in range(1, lines + 1)
    )


def _hpc_dump(lines: int = 9) -> str:
    # Epoch-counter RAS dialect: no clock timestamp, no level word.
    return "\n".join(
        f"{218505 + s} node-D3 clusterfilesystem fdmn.full {1117130918 + s} 1 "
        f"ServerFileSystem: ServerFileSystem domain root22_local"
        for s in range(1, lines + 1)
    )


def test_timestamped_dump_routes_log_event_not_issue():
    kind, routed_by = _route_type(_hdfs_dump())
    assert kind == "log_event", f"HDFS-style dump routed {kind}"
    assert routed_by == "structure", routed_by


def test_apache_bracket_dump_routes_log_event():
    kind, _ = _route_type(_apache_dump())
    assert kind == "log_event", f"Apache-style dump routed {kind}"


def test_short_bracket_time_dump_routes_log_event():
    kind, _ = _route_type(_proxifier_dump())
    assert kind == "log_event", f"Proxifier-style dump routed {kind}"


def test_epoch_counter_dump_routes_log_event():
    kind, _ = _route_type(_hpc_dump())
    assert kind == "log_event", f"HPC-style dump routed {kind}"


def test_fenced_error_paste_inside_prose_does_not_flip():
    # Measured falsification (2026-10-08): a fence signal is NOT shipped.
    # Real issue bodies routinely embed fenced tracebacks (issue-realworld
    # rows 42/128/135/253 — three of them correctly routed issue before any
    # override), so no fence rule may override the router here. The slack
    # paste surface belongs to the training improvement arm.
    text = (
        "[BUG] `HIMPORT HELP` does not exist\n"
        "\n"
        "**Describe the bug**\n"
        "\n"
        "An unknown subcommand gets the generic error from `server.c`:\n"
        "\n"
        "```\n"
        "127.0.0.1:6390> HIMPORT HELP\n"
        "Error: unknown subcommand: HELP\n"
        "Try HIMPORT HELP\n"
        "```\n"
        "\n"
        "The help subcommand should exist for container commands.\n"
    )
    kind, routed_by = _route_type(text)
    assert (kind, routed_by) != ("log_event", "structure"), (kind, routed_by)


class _CommitRouter:
    """Duck router: confident in-domain commit on anything."""

    classes = ["commit"]

    def predict_proba(self, texts):
        import numpy as np

        return np.array([[0.97]] * len(texts))

    def predict_proba_with_energy(self, texts):
        import numpy as np

        return np.array([[0.97]] * len(texts)), [12.0] * len(texts)

    in_domain_energy = 8.4


def test_confident_commit_outranks_the_dump_rule():
    """The rule must not preempt a confident in-domain commit route on a
    timestamp-per-line paste (a `git log --pretty` dump, review P1):
    deterministic via a stub router, no encoder dependence."""
    bootstrap.register_example_schemas()  # commit is a built-in: active
    text = _hdfs_dump(10)
    schema, routed_by, conf = pipeline._route_with_confidence(
        _CommitRouter(), text)
    assert (schema, routed_by) == ("commit", "model"), (schema, routed_by)
    assert conf >= 0.85


def test_csv_paste_with_timestamp_column_stays_text():
    """Bank-statement/sensor CSV rows starting `YYYY-MM-DD HH:MM:SS,-amount`
    are NOT logs: the ISO dialect requires a millis/space/Z continuation,
    never a comma-minus (2026-10-08 review P1 — regression vs the
    confident-text behavior)."""
    text = "\n".join(
        f"2024-03-01 12:{m:02d}:00,-10.50,GROCERY STORE PURCHASE #{m}"
        for m in range(10, 20)
    )
    kind, routed_by = _route_type(text)
    assert (kind, routed_by) != ("log_event", "structure"), (kind, routed_by)


def test_git_log_paste_is_never_structure_routed():
    """A `git log --pretty` paste is never structure-routed. On the shipped
    encoder it routes log_event@~0.67 by MODEL call (unconfident, honest
    ladder exit) — the record quality is an improvement-arm surface, not
    a decision-layer bug; the pin is: no `structure` override here."""
    text = "\n".join(
        f"2026-09-0{d} 10:{m:02d}:22 +0200 fix(auth): rotate the {m} session token"
        for d, m in ((1, 1), (1, 15), (2, 2), (2, 31), (3, 3), (3, 44), (4, 4), (4, 59))
    )
    kind, routed_by = _route_type(text)
    assert routed_by != "structure", (kind, routed_by)


def test_git_log_paste_keeps_its_model_route():
    """A `git log --pretty="%ad %s"` paste is never STRUCTURE-routed; on
    the shipped encoder it routes log_event@~0.67 by honest model call —
    the record quality is an improvement-arm surface (review P1)."""
    text = "\n".join(
        f"2026-09-0{d} 10:{m:02d}:22 +0200 fix(auth): rotate the {m} session token"
        for d, m in ((1, 1), (1, 15), (2, 2), (2, 31), (3, 3), (3, 44), (4, 4), (4, 59))
    )
    kind, routed_by = _route_type(text)
    assert (kind, routed_by) != ("log_event", "structure"), (kind, routed_by)


def test_short_prose_does_not_flip():
    text = "the deploy failed again, see ticket PAY-2214 for the timeline"
    kind, routed_by = _route_type(text)
    assert (kind, routed_by) != ("log_event", "structure"), (kind, routed_by)


def test_single_log_line_does_not_go_through_the_dump_rule():
    text = "2026-03-01T12:00:00Z ERROR [auth]: token expired"
    _, routed_by = _route_type(text)
    assert routed_by != "structure", routed_by


def test_dated_changelog_without_time_of_day_does_not_flip():
    text = (
        "commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0\n"
        "release checklist merged:\n"
        "2024-01-01 shipped the parser rework\n"
        "2024-01-02 shipped the validator pins\n"
        "2024-01-03 shipped the bench harness\n"
        "2024-01-04 shipped the docs sweep\n"
        "2024-01-05 shipped the router eval\n"
        "2024-01-06 shipped the pool manifest\n"
        "2024-01-07 shipped the CI cache\n"
        "2024-01-08 shipped the release notes\n"
        "files_changed: 3 author: mac"
    )
    kind, routed_by = _route_type(text)
    assert (kind, routed_by) != ("log_event", "structure"), (kind, routed_by)


def test_user_claimed_schema_outranks_the_dump_rule():
    registry.register_schema(
        "invoice_lite",
        {"type": {"const": "invoice_lite"}, "total": {"type": str}},
        {"strong": [r"\bInvoice\s*#\s*INV-\d{4}-\d{4}\b"], "weak": []},
        {"total": [r"(?i)total\s*[:=]\s*(\$?\s*\d[\d.,]*)"]},
        description="invoice_lite: a billing invoice with an invoice number",
    )
    try:
        text = "\n".join([_hdfs_dump(), "", "Invoice #INV-2026-0042 total due"])
        kind, routed_by = _route_type(text)
        assert routed_by != "structure", (kind, routed_by)
        assert kind == "invoice_lite", (kind, routed_by)
    finally:
        registry.unregister("invoice_lite")
