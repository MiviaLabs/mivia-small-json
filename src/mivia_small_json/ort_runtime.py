# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared ONNXRuntime CPU session options — one place for the arena gate.

``MIVIA_DISABLE_CPU_MEM_ARENA``: set to ``1``/``true``/``yes``/``on`` to
disable the ORT CPU memory arena on every shipped artifact (router, guard,
grade, severity). The arena normally grows to the high-water mark of
activation sizes and never returns those pages to the OS while the process
lives — RSS reflects the historical peak, not live load. Disabling trades a
few ms of per-run allocation for RSS that falls back after bursts. The
default (env unset or any other value) keeps the arena ON, which is the
ORT default and the shipped behavior since the serving layer landed.
"""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def disable_cpu_mem_arena() -> bool:
    """The ``MIVIA_DISABLE_CPU_MEM_ARENA`` gate (default: arena stays on)."""
    return os.environ.get("MIVIA_DISABLE_CPU_MEM_ARENA", "").strip().lower() \
        in _TRUTHY


def apply_cpu_thread_budget(opts):
    """The shipped CPU thread budget on a ``SessionOptions`` (shared with
    ``tool_match._cpu_session``): intra = min(4, cpu_count), inter = 1,
    sequential execution. Every CPU ONNX artifact reads through this so
    the router cannot oversubscribe threads the guard already owns.

    Returns the same object for chaining.
    """
    import onnxruntime as ort

    opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


def session_options():
    """Fresh ``SessionOptions`` for a CPU ONNX artifact.

    Matches the ORT defaults unless the arena gate is set, so callers that
    need thread budgets set them on the returned object themselves (see
    ``tool_match._cpu_session``).
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if disable_cpu_mem_arena():
        opts.enable_cpu_mem_arena = False
    return opts
