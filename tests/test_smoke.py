"""Smoke tests: package importability and pin consistency (single source of truth)."""

import tomllib
from pathlib import Path

import mivia_small_json

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    assert mivia_small_json.__version__


def test_version_matches_pyproject():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert mivia_small_json.__version__ == declared


def _pins(path):
    return {line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def test_requirements_mirror_pyproject_pins():
    """requirements.txt and pyproject.toml pins must match BOTH ways."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    # dev + serve live in requirements.txt (a dev env runs the API tests);
    # training stays out (GPU-only, see the CI torch comment).
    declared = (set(pyproject["project"]["dependencies"])
                | set(extras["dev"]) | set(extras.get("serve", [])))
    req = _pins(ROOT / "requirements.txt")
    assert not (req - declared), f"{req - declared} not pinned in pyproject.toml"
    assert not (declared - req), f"{declared - req} missing from requirements.txt"


def test_lock_is_a_complete_strict_freeze():
    """requirements-lock.txt must be the real freeze: every direct pin
    present verbatim, every entry an exact == pin (no ranges)."""
    import re

    lock = _pins(ROOT / "requirements-lock.txt")
    req = _pins(ROOT / "requirements.txt")
    assert req <= lock, f"direct pins missing from the lock: {sorted(req - lock)}"
    for line in lock:
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==[^\s=]+", line), (
            f"lock line {line!r} is not a strict == pin")
