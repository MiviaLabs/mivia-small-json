"""The ood-class one-shot harness must grade on the REGISTERED bars.

Two audited divergences (2026-10-08 spec axis):

1. The retention bar is re-registered at >= 0.98 (owner decision recorded
   in ood-routing-band — "Consequence: v9 PASSES all
   bars"); the harness compared >= 0.99, so the spec's own recorded PASS
   computed FAIL.
2. `guard_bytes_untouched` was recorded but never gated the verdict: the
   shipped one-shot JSON carries `false` next to `verdict: "PASS"`.
   Guard cleanliness is a sha check of the guard artifacts, not a
   `git status` proxy (any router promotion dirties that by construction).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eval_ood_class", ROOT / "scripts" / "eval_ood_class.py")
MOD = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("eval_ood_class", MOD)
SPEC.loader.exec_module(MOD)


def _result(min_retention, guard=True):
    return {
        "rejection_overall": 0.8793,
        "freechat_rejection": 0.8958,
        "min_retention": min_retention,
        "guard_bytes_untouched": guard,
    }


def test_retention_bar_is_the_registered_0_98():
    bars = MOD._bars(_result(0.981))
    assert bars["retention>=0.98"], (
        "0.981 must PASS the re-registered 0.98 bar")
    assert not MOD._bars(_result(0.979))["retention>=0.98"]


def test_guard_bytes_gate_the_verdict():
    ok = MOD._bars(_result(0.99, guard=True))
    assert MOD._verdict({**_result(0.99, guard=True), "bars": ok}) == "PASS"
    assert MOD._verdict({**_result(0.99, guard=False), "bars": ok}) == "FAIL", (
        "untouched-guard=false with passing bars must not read PASS")


def test_guard_untouched_checks_sha_not_git_state():
    # The real shipped tree: the guard artifacts verify -> untouched.
    assert MOD._guard_untouched() is True
