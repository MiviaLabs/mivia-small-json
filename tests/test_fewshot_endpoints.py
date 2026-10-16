# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""HTTP contract tests for the internal few-shot endpoints
(ticket 02 of ``.scratch/fewshot-internal-endpoints/``).

The wire shapes are frozen by the platform's shipped client code (spec
``docs/specs/fewshot-internal-endpoints/spec.md`` section 2): 4xx bodies
carry ``{"detail": ...}`` — never ``_core``'s ``{"error": ...}`` — the
infer 200 body has all five keys assembled by the route, auth is the
bearer gate PLUS ``X-Internal-Token`` when ``MIVIA_INTERNAL_TOKEN`` is
set (empty env = check skipped), and the embed session warms at boot.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import math

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from mivia_small_json import fewshot_classifier as fc  # noqa: E402
from mivia_small_json.api import create_app  # noqa: E402

BEARER = "test-bearer-0123456789"
INTERNAL = "test-internal-9876543210"


def _labels(n=2, per=6, start=0):
    """n labels x per distinct short examples, no cross-label overlaps."""
    return [
        {
            "name": f"label{i + start}",
            "examples": [
                f"{i + start} example {j} text about topic {i + start}"
                for j in range(per)
            ],
        }
        for i in range(n)
    ]


def _fat_labels(n, per, example_chars):
    """n labels x per examples, each ``example_chars`` long (distinct).

    Sized so n x per x example_chars crosses MAX_FS_TOTAL_CHARS while
    every per-example and per-label cap still holds: only the total-chars
    cap may fire (the 413 case).
    """
    return [
        {
            "name": f"label{i}",
            "examples": [
                f"label{i} example {j} " + "x" * (example_chars - 20)
                for j in range(per)
            ],
        }
        for i in range(n)
    ]


def _post(client, path, payload, bearer=BEARER, internal=None):
    headers = {}
    if bearer is not None:
        headers["Authorization"] = f"Bearer {bearer}"
    if internal is not None:
        headers["X-Internal-Token"] = internal
    return client.post(path, json=payload, headers=headers)


@pytest.fixture(scope="module")
def client():
    """One authed app + TestClient with the lifespan entered once.

    The embed singleton is reset BEFORE startup so the warmup seam test
    can only pass if ``_warmup`` loaded it; the process-wide cache then
    serves every later encoder use in this module. A developer-exported
    MIVIA_INTERNAL_TOKEN must not leak into the env-unset case.

    The warmup steps this module does NOT test are disabled cheaply: a
    dummy router keeps create_app off load_active_router and a raising
    _get_guard keeps the router/guard cross-encoders out of this process.
    Native ONNX arena memory never fully returns, and a same-process
    follow-on module (test_api's 2 GB RSS health bar) must measure its
    own footprint, not this module's leftover. The fewshot embed step —
    the seam under test — stays real.
    """
    from starlette.testclient import TestClient
    from mivia_small_json import tool_match

    def _no_guard(*_a, **_k):
        raise RuntimeError("guard disabled by test_fewshot_endpoints")

    mp = pytest.MonkeyPatch()
    mp.delenv("MIVIA_INTERNAL_TOKEN", raising=False)
    mp.setattr(tool_match, "_get_guard", _no_guard, raising=False)
    mp.setattr(tool_match, "_EMBED_PRUNER", None, raising=False)
    mp.setattr(tool_match, "_EMBED_PRUNER_LOADED", False, raising=False)
    app = create_app(token=BEARER, router=object())
    with TestClient(app) as c:
        yield c
    mp.undo()
    del app, c
    gc.collect()


@pytest.fixture(scope="module")
def trained(client):
    """One real train response shared by the infer-shape tests."""
    resp = _post(client, "/internal/fewshot/train",
                 {"labels": _labels(), "seed": 7})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Train: happy path with the exact frozen response shape.
# ---------------------------------------------------------------------------


def test_train_seed_validation(client):
    """seed must be a real int: bool, string, and missing all 422."""
    labels = [{"name": "a", "examples": ["a1", "a2", "a3", "a4"]},
              {"name": "b", "examples": ["b1", "b2", "b3", "b4"]}]
    for bad in (True, "7"):
        r = _post(client, "/internal/fewshot/train",
                  {"labels": labels, "seed": bad})
        assert r.status_code == 422, (bad, r.status_code, r.text)
    r = _post(client, "/internal/fewshot/train", {"labels": labels})
    assert r.status_code == 422, r.text


def test_train_happy_path_full_shape(trained):
    assert set(trained) == {"head_artifact_b64", "head_sha256", "metrics",
                            "backbone", "learner"}
    raw = base64.b64decode(trained["head_artifact_b64"], validate=True)
    assert trained["head_sha256"] == hashlib.sha256(raw).hexdigest()
    assert set(trained["metrics"]) == {"schema", "per_label", "eval_mode",
                                       "eval_n", "mean_accuracy",
                                       "adjacency_warnings", "thresholds"}
    # the 2x6 fixture sits under the LOO gate: every row is an eval point
    assert trained["metrics"]["eval_mode"] == "loo"
    assert trained["metrics"]["eval_n"] == 12
    for entry in trained["metrics"]["per_label"].values():
        assert set(entry) == {"held_out_n", "eval_n", "accuracy", "confusions"}
        assert entry["held_out_n"] == 0
        assert entry["eval_n"] == 6
    assert set(trained["backbone"]) == {"artifact", "sha256"}
    assert set(trained["learner"]) == {"fewshot_py_sha256"}
    assert set(trained["metrics"]["thresholds"]) == {"low_confidence"}


# ---------------------------------------------------------------------------
# Infer: the FULL five-key 200 body, label in declared labels.
# ---------------------------------------------------------------------------


def test_infer_roundtrip_full_five_key_body(client, trained):
    resp = _post(client, "/internal/fewshot/infer",
                 {"head_artifact_b64": trained["head_artifact_b64"],
                  "head_sha256": trained["head_sha256"],
                  "text": "example 1 text about topic 0"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"label", "confidence", "status", "candidates",
                         "head_sha256_matches"}
    assert body["status"] == "ok"
    # the loo head trained on every row, so the topic-0 probe lands label0
    assert body["label"] == "label0"
    assert isinstance(body["confidence"], float)
    assert math.isfinite(body["confidence"])
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["candidates"], list)
    assert 1 <= len(body["candidates"]) <= 2
    assert body["label"] in body["candidates"]
    assert body["head_sha256_matches"] is True


# ---------------------------------------------------------------------------
# Caps: over-limit is an explicit 4xx, never a truncation.
# Validation-only cases need no encoder; the route must not 500.
# ---------------------------------------------------------------------------


def test_cap_max_fs_labels_names_the_cap(client):
    resp = _post(client, "/internal/fewshot/train",
                 {"labels": _labels(51, per=4), "seed": 1})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "MAX_FS_LABELS" in detail


def test_total_chars_over_cap_is_413(client):
    # 10 x 50 x 1700 = 850,000 chars: per-example, per-label and total
    # count caps all hold, only MAX_FS_TOTAL_CHARS fires.
    resp = _post(client, "/internal/fewshot/train",
                 {"labels": _fat_labels(10, 50, 1700), "seed": 1})
    assert resp.status_code == 413, resp.text
    assert "MAX_FS_TOTAL_CHARS" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Infer refusals: 422 with a reason, never a 500.
# ---------------------------------------------------------------------------


def test_infer_wrong_declared_sha_422(client, trained):
    resp = _post(client, "/internal/fewshot/infer",
                 {"head_artifact_b64": trained["head_artifact_b64"],
                  "head_sha256": "0" * 64,
                  "text": "example 0 text about topic 0"})
    assert resp.status_code == 422, resp.text
    assert "head_sha256" in resp.json()["detail"]


def test_infer_tampered_bytes_422(client, trained):
    raw = bytearray(base64.b64decode(trained["head_artifact_b64"]))
    raw[len(raw) // 2] ^= 0xFF
    resp = _post(client, "/internal/fewshot/infer",
                 {"head_artifact_b64": base64.b64encode(bytes(raw)).decode(),
                  "head_sha256": trained["head_sha256"],
                  "text": "example 0 text about topic 0"})
    assert resp.status_code == 422, resp.text
    assert "head_sha256" in resp.json()["detail"]


def test_infer_truncated_skops_422(client, trained):
    raw = base64.b64decode(trained["head_artifact_b64"])
    cut = raw[: len(raw) // 2]
    # correct sha for the cut bytes: the skops load gate must fire, not the
    # sha gate
    resp = _post(client, "/internal/fewshot/infer",
                 {"head_artifact_b64": base64.b64encode(cut).decode(),
                  "head_sha256": hashlib.sha256(cut).hexdigest(),
                  "text": "example 0 text about topic 0"})
    assert resp.status_code == 422, resp.text
    assert "load" in resp.json()["detail"]


def test_infer_empty_text_422(client, trained):
    for text in ("", "   "):
        resp = _post(client, "/internal/fewshot/infer",
                     {"head_artifact_b64": trained["head_artifact_b64"],
                      "head_sha256": trained["head_sha256"], "text": text})
        assert resp.status_code == 422, resp.text
        assert "text" in resp.json()["detail"]


def test_infer_oversize_artifact_422(client):
    blob = b"0" * (fc.MAX_ARTIFACT_BYTES + 1)
    resp = _post(client, "/internal/fewshot/infer",
                 {"head_artifact_b64": base64.b64encode(blob).decode(),
                  "head_sha256": hashlib.sha256(blob).hexdigest(),
                  "text": "x"})
    assert resp.status_code == 422, resp.text
    assert "exceeds" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Auth: bearer gate plus the internal token when its env is set.
# ---------------------------------------------------------------------------


def test_no_bearer_401(client):
    resp = client.post("/internal/fewshot/train",
                       json={"labels": _labels(), "seed": 1})
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}


def test_wrong_or_missing_internal_token_401_when_env_set(
        client, monkeypatch):
    monkeypatch.setenv("MIVIA_INTERNAL_TOKEN", INTERNAL)
    for internal in (None, "wrong-token"):
        resp = _post(client, "/internal/fewshot/train",
                     {"labels": _labels(), "seed": 1}, internal=internal)
        assert resp.status_code == 401, (internal, resp.text)
        assert resp.json() == {"error": "unauthorized"}


def test_env_unset_bearer_alone_passes_auth(client, monkeypatch):
    # The asymmetry contract (spec section 3.2): an empty env skips the
    # X-Internal-Token check. A 422 on an invalid body proves the request
    # crossed auth into the handler without paying for a train.
    monkeypatch.delenv("MIVIA_INTERNAL_TOKEN", raising=False)
    resp = _post(client, "/internal/fewshot/infer",
                 {"head_artifact_b64": "AAAA", "head_sha256": "0" * 64,
                  "text": ""})
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# The internal surface stays out of every catalogue.
# ---------------------------------------------------------------------------


def test_openapi_json_is_404(client):
    resp = client.get("/openapi.json",
                      headers={"Authorization": f"Bearer {BEARER}"})
    assert resp.status_code == 404


def test_no_fewshot_mcp_tool():
    """Spec section 5: the MCP tool list gains no fewshot entry."""
    from pathlib import Path

    from mivia_small_json import api

    src = Path(api.__file__).read_text()
    assert "mivia_fewshot" not in src
    assert "/internal/fewshot/train" in src  # the REST route exists


# ---------------------------------------------------------------------------
# Warmup: the embed session is loaded at boot, not on the first classify.
# The fixture reset the singleton before the lifespan ran, so only a
# warmup step can have loaded it. (Ticket 03 checks the boot log line;
# this test pins the loader seam itself.)
# ---------------------------------------------------------------------------


def test_warmup_loads_embed_session(client):
    from mivia_small_json import tool_match

    assert tool_match._EMBED_PRUNER is not None
