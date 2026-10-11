# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic training setup.

The shipped artifacts are pinned by sha256 in `models/SHIPPED.json`, but a hash
pins a *file*, not a build. Re-running a GPU fine-tune produced a different
artifact from an identical loss, so "regenerate and check the hash" was not a
reproducibility gate. The ONNX int8 exporter is deterministic (verified: two
runs on the same fp32 parent give identical bytes); the weights are not.

`seed_everything` makes a run reproducible: same seed, same kernels, same
result. Training-time only — the shipped runtime installs no torch.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42, strict: bool | None = None) -> int:
    """Seed python, numpy and torch, and request deterministic kernels.

    Returns ``seed`` so callers can log it. Never raises: this is a
    training-time helper and the runtime environment has no torch.

    ``torch.use_deterministic_algorithms`` runs in warn-only mode by default,
    because some ops have no deterministic kernel and raising would make a
    training script unusable for a reason unrelated to the change. Set
    ``MIVIA_STRICT_DETERMINISM=1`` (or pass ``strict=True``) to raise instead.
    """
    # Required for deterministic cuBLAS matmuls; must be set before the first
    # CUDA context.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if strict is None:
            strict = os.environ.get("MIVIA_STRICT_DETERMINISM") == "1"
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    except Exception:
        pass
    return seed
