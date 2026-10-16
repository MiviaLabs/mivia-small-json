#!/usr/bin/env python3
"""Internal few-shot endpoint latency bench (ticket 03, spec section 4).

A REPORT, never a CI gate: no timing asserts anywhere (house rule); the
printed numbers are the record, whatever they are. The spec statement this
informs is < 50 ms infer p95 inside a 2 GB CPU envelope.

Protocol (one box, one process, in-process app):

- App: ``create_app(token=...)`` with the same cheap stubs as
  ``tests/test_fewshot_endpoints.py`` — a dummy router keeps create_app
  off ``load_active_router`` and a raising ``_get_guard`` keeps the guard
  cross-encoder out of this process. The fewshot embed session (the
  encoder under bench) stays real and warms in the lifespan.
- Train: one 5-label head, 10 examples per label, deterministic seed, via
  POST /internal/fewshot/train. Train wall time is printed once,
  informational (spec scopes the bench to infer).
- Infer: 200 POST /internal/fewshot/infer calls with seeded varied texts;
  p50/p95 over the per-call wall times.
- Fingerprint: python version, git SHA.

Encoder unavailable (fresh checkout)? Assemble first:
``uv run scripts/assemble_models.py``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED = 20261015
N_LABELS = 5
PER_LABEL = 10
N_INFER = 200
BEARER = "bench-bearer-fewshot-endpoints"
ASSEMBLE_CMD = "uv run scripts/assemble_models.py"

TOPICS = ("billing", "shipping", "refunds", "accounts", "feedback")


def _labels(n: int, per: int) -> list[dict]:
    return [
        {
            "name": TOPICS[i % len(TOPICS)],
            "examples": [
                f"{TOPICS[i]} request {j} about {TOPICS[i]} item {j}"
                for j in range(per)
            ],
        }
        for i in range(n)
    ]


def _infer_texts(n: int, seed: int) -> list[str]:
    """Seeded varied probe texts cycling the label topics."""
    import random

    rng = random.Random(seed)
    texts = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        k = rng.randrange(10_000)
        texts.append(f"{topic} request {k} about {topic} item {k}")
    return texts


def _pctl(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _git_sha() -> str:
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return git.stdout.strip()[:12] if git.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _build_client():
    """One authed app + TestClient; the test fixture's cheap stubs."""
    from starlette.testclient import TestClient

    from mivia_small_json import tool_match
    from mivia_small_json.api import create_app

    def _no_guard(*_a, **_k):
        raise RuntimeError("guard disabled by bench_fewshot_endpoints")

    if not tool_match.EMBED_PRUNER_MODEL.is_file():
        print(f"encoder artifact missing: {tool_match.EMBED_PRUNER_MODEL}\n"
              f"assemble it first: {ASSEMBLE_CMD}", file=sys.stderr)
        raise SystemExit(2)

    # Script-only singleton resets (tests monkeypatch): a raising guard
    # loader and a fresh embed session, so the lifespan warmup is what
    # loads the encoder — never the router/guard cross-encoders.
    tool_match._get_guard = _no_guard
    tool_match._EMBED_PRUNER = None
    tool_match._EMBED_PRUNER_LOADED = False
    app = create_app(token=BEARER, router=object())
    return TestClient(app)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--labels", type=int, default=N_LABELS)
    parser.add_argument("--per-label", type=int, default=PER_LABEL)
    parser.add_argument("--infer-calls", type=int, default=N_INFER)
    args = parser.parse_args()

    print(f"fingerprint: python {sys.version.split()[0]} · "
          f"git {_git_sha()} · seed {args.seed}")

    auth = {"Authorization": f"Bearer {BEARER}"}
    with _build_client() as client:
        t0 = time.perf_counter()
        resp = client.post("/internal/fewshot/train",
                           json={"labels": _labels(args.labels,
                                                   args.per_label),
                                 "seed": args.seed},
                           headers=auth)
        train_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            print(f"train failed: HTTP {resp.status_code} "
                  f"{resp.text[:300]}", file=sys.stderr)
            print(f"if the encoder bytes are broken, run: {ASSEMBLE_CMD}",
                  file=sys.stderr)
            return 2
        trained = resp.json()

        samples: list[float] = []
        non200 = 0
        for text in _infer_texts(args.infer_calls, args.seed):
            t0 = time.perf_counter()
            r = client.post("/internal/fewshot/infer",
                            json={"head_artifact_b64":
                                  trained["head_artifact_b64"],
                                  "head_sha256": trained["head_sha256"],
                                  "text": text},
                            headers=auth)
            dt = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                non200 += 1
                continue
            samples.append(dt)

    print(f"train: {args.labels} labels x {args.per_label} examples · "
          f"wall {train_ms:.1f} ms (informational, head sha "
          f"{trained['head_sha256'][:16]}…)")
    print(f"infer: {args.infer_calls} calls ({non200} non-200) · "
          f"p50 {_pctl(samples, 0.50):.2f} ms · "
          f"p95 {_pctl(samples, 0.95):.2f} ms "
          f"(n={len(samples)} timed ok calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
