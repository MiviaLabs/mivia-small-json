"""MIVIA_DISABLE_CPU_MEM_ARENA: env-gated ORT arena switch.

Default (env unset) must keep the arena ON — the ORT default and shipped
behavior since the serving layer landed — so nothing changes for existing
deployments. Setting the flag disables the arena: RSS returns to the OS
after bursts instead of plateauing at the high-water mark, trading a few
ms of per-run allocation. Found desirable after the 3,000-request live
hammer showed RSS climbing 223 -> 768 MB under 12-way concurrent load and
never returning (reports/serving-bench-live-2026-10-08.md).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("onnxruntime")

from mivia_small_json import ort_runtime  # noqa: E402

TRUTHY = ["1", "true", "TRUE", "yes", "on"]
FALSY = ["", "0", "false", "no", "bogus"]


@pytest.mark.parametrize("value", TRUTHY)
def test_env_set_disables_arena(monkeypatch, value):
    monkeypatch.setenv("MIVIA_DISABLE_CPU_MEM_ARENA", value)
    assert ort_runtime.disable_cpu_mem_arena() is True
    opts = ort_runtime.session_options()
    assert opts.enable_cpu_mem_arena is False


@pytest.mark.parametrize("value", FALSY)
def test_env_unset_or_falsy_keeps_arena_on(monkeypatch, value):
    if value:
        monkeypatch.setenv("MIVIA_DISABLE_CPU_MEM_ARENA", value)
    else:
        monkeypatch.delenv("MIVIA_DISABLE_CPU_MEM_ARENA", raising=False)
    assert ort_runtime.disable_cpu_mem_arena() is False
    opts = ort_runtime.session_options()
    assert opts.enable_cpu_mem_arena is True


def test_guard_session_thread_budget_untouched():
    """The shared options path must not disturb _cpu_session's budgets."""
    opts = ort_runtime.session_options()
    assert opts.execution_mode == 0  # ORT_SEQUENTIAL, the ORT default


def test_encoder_loads_and_routes_with_arena_disabled(monkeypatch):
    """Integration: a full session comes up and routes with the arena off."""
    monkeypatch.setenv("MIVIA_DISABLE_CPU_MEM_ARENA", "1")
    from mivia_small_json.encoder import load_encoder

    router = load_encoder()
    assert router is not None
    probs = router.predict_proba(["2026-10-08 03:11:22 ERROR db: pool exhausted"])
    assert len(probs[0]) == len(router.classes)
    assert abs(float(sum(probs[0])) - 1.0) < 1e-3
    assert str(router.classes[int(probs[0].argmax())]) == "log_event"
