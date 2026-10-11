"""Encoder calibration constants travel in sha-verified meta (Phase 0).

`IN_DOMAIN_ENERGY = 8.4` was a code default calibrated against one
artifact's logit scale; a backbone swap inheriting it silently misroutes
every energy-gated row (rebuild-plan audit, silent-risk #2). The value now
loads from `models/router-encoder-meta.json` (sha-verified), with the code
default kept only as a last-resort fallback.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from mivia_small_json import encoder  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"
META = MODELS / "router-encoder-meta.json"


def test_meta_exists_and_sha_verifies():
    assert META.is_file(), "router-encoder-meta.json must ship"
    from mivia_small_json import integrity

    assert integrity.verified(META), "meta fails its sha256 sidecar"
    value = float(json.loads(META.read_text())["in_domain_energy"])
    assert value == -100.0  # superseded-gate sentinel, pinned


def test_load_encoder_honors_meta_value():
    router = encoder.load_encoder()
    assert router is not None
    assert router.in_domain_energy == -100.0  # the superseded-gate sentinel


def test_meta_override_changes_router_threshold(tmp_path, monkeypatch):
    """A recalibrated meta value must reach the router, not the code default."""
    alt = tmp_path / "alt-meta.json"
    alt.write_text(json.dumps({"in_domain_energy": 7.25}))
    import hashlib

    (tmp_path / "alt-meta.json.sha256").write_text(
        hashlib.sha256(alt.read_bytes()).hexdigest() + "\n")
    router = encoder.load_encoder(
        tokenizer_path=MODELS / "router-tokenizer.json")
    assert router is not None
    # swap the meta beside the real artifact, load, restore
    backup = META.read_bytes()
    meta_sha = MODELS / "router-encoder-meta.json.sha256"
    sha_backup = meta_sha.read_bytes()
    try:
        META.write_bytes(alt.read_bytes())
        meta_sha.write_text((tmp_path / "alt-meta.json.sha256").read_text())
        reloaded = encoder.load_encoder()
        assert reloaded is not None
        assert reloaded.in_domain_energy == pytest.approx(7.25)
    finally:
        META.write_bytes(backup)
        meta_sha.write_bytes(sha_backup)


def test_unverifiable_meta_falls_back_to_code_default(tmp_path):

    backup = META.read_bytes()
    meta_sha = MODELS / "router-encoder-meta.json.sha256"
    sha_backup = meta_sha.read_bytes()
    try:
        META.write_bytes(b'{"in_domain_energy": 1.0}\n')  # sidecar mismatch
        reloaded = encoder.load_encoder()
        assert reloaded is not None
        assert reloaded.in_domain_energy == 8.4  # the CODE default, pinned
    finally:
        META.write_bytes(backup)
        meta_sha.write_bytes(sha_backup)
