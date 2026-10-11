"""Per-artifact tokenizer decoupling (rebuild plan Phase 0).

Before this test existed, the severity heads and the guard silently shared
`models/router-tokenizer.json` with the router: replacing that file during
the backbone rebuild would have fed the OLD ordinal/guard heads a wrong
vocab with no sha failure (the sidecar travels with the replaced file).
Each artifact family now carries its own byte-identical copy; the files —
not the sharing — are the contract.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mivia_small_json import tool_match  # noqa: E402
from mivia_small_json.integrity import expected_digest  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"


def _verified(name: str) -> bool:
    path = MODELS / name
    digest, _ = expected_digest(MODELS / (name + ".sha256"))
    return path.is_file() and digest is not None and \
        hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_each_family_has_its_own_verified_tokenizer():
    families = {
        "router (encoder)": MODELS / "router-tokenizer.json",
        "severity heads": tool_match.SEVERITY_TOKENIZER,
        "guard rank": MODELS / "tool-match-tokenizer.json",
        "guard gate": MODELS / "tool-match-gate-tokenizer.json",
    }
    seen_paths = set()
    for label, path in families.items():
        assert path.is_file(), f"{label}: {path.name} missing"
        name = path.name
        assert _verified(name), f"{name} fails its sha256 sidecar"
        seen_paths.add(path.resolve())
    # decoupled = distinct files per family (no silent sharing)
    assert len(seen_paths) == len(families)


def test_guard_loader_prefers_its_sibling_tokenizer():
    """load_single_matcher derives <artifact>-tokenizer.json next to the onnx."""
    onnx = MODELS / "tool-match-int8.onnx"
    sibling = MODELS / "tool-match-tokenizer.json"
    assert sibling.is_file(), "guard sibling tokenizer must ship"
    derived = onnx.parent / onnx.name.replace("-int8.onnx", "-tokenizer.json")
    assert derived == sibling


def test_copies_are_byte_identical_today():
    """The bert-base families stay byte-identical to EACH OTHER; the router
    tokenizer is deliberately a different vocab family (multilingual XLM-R,
    2026-10-10 promotion) — pinned so a future swap is a visible DIFF,
    never a silent vocab mismatch. tool_match.DEFAULT_TOKENIZER must point
    at the bert-base copy, never the router's."""
    trio = {name: (MODELS / name).read_bytes() for name in (
        "severity-tokenizer.json", "tool-match-tokenizer.json",
        "tool-match-gate-tokenizer.json")}
    assert len(set(trio.values())) == 1, "bert-base families diverged"
    router_vocab = json.loads((MODELS / "router-tokenizer.json").read_text())
    assert len(router_vocab["model"]["vocab"]) > 200_000, (
        "router tokenizer must be the multilingual vocab")
    bert_vocab = json.loads(
        (MODELS / "tool-match-tokenizer.json").read_text())["model"]["vocab"]
    assert len(bert_vocab) == 30_522
    assert Path(tool_match.DEFAULT_TOKENIZER).name == "tool-match-tokenizer.json"
