#!/usr/bin/env python3
"""Retrain every shipped artifact in order — one command, fail fast.

Runs synthetic generation, the router, the three learned extractors, and
the bench verdict, in that order, stopping at the first non-zero exit.
Each step is announced before it runs, so a failure names the step.

``STEPS`` is the ordered step list (module-level, importable): a tuple of
argv tuples, each run as ``[sys.executable, *step]`` from the repository
root. The level extractor belongs in it — the cycle's extractor retrain
once omitted it and let the artifact go stale.

Usage (repository root, project venv):

    python scripts/train_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS: tuple[tuple[str, ...], ...] = (
    ("-m", "mivia_small_json.synth"),
    ("-m", "mivia_small_json.train"),
    ("scripts/train_level_extractor.py",),
    ("scripts/train_component_extractor.py",),
    ("scripts/train_implicit_severity.py",),
    # The shipped router: fine-tune the encoder on the full mix and export it
    # to int8 ONNX under data/incoming/models/ (copy into models/ to ship).
    ("scripts/export_encoder_onnx.py",),
    ("-m", "mivia_small_json.bench"),
)


def run_step(step: tuple[str, ...]) -> int:
    """Run one step from the repository root and return its exit code."""
    argv = [sys.executable, *step]
    print(f"==> {' '.join(argv)}", flush=True)
    return subprocess.run(argv, cwd=ROOT, check=False).returncode


def main() -> int:
    for step in STEPS:
        code = run_step(step)
        if code != 0:
            print(f"step failed (exit {code}): {' '.join(step)}", file=sys.stderr)
            return code
    print("all steps complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
