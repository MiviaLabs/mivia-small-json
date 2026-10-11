"""Regenerate tests/fixtures/routing-golden.jsonl.

Run from the repository root with the project venv:

    python tests/fixtures/make_routing_golden.py

The golden file pins the observable behavior of the rules layer
(route, scores, extract, emit) and of ``pipeline.classify`` with the
committed router artifact, for a seeded sample of the committed
corpora. The golden test replays the same inputs and requires
byte-identical JSON lines, so behavior changes cannot pass unnoticed.

Determinism:
- Fixed source order, seeded sample per source, ascending row index.
- Wall-clock values are scrubbed: every ``ts`` field becomes ``<ts>``
  and classify meta's ``latency_ms`` is dropped. Everything else is
  exact.
"""

import json
import random
import re
from pathlib import Path

from mivia_small_json import pipeline, rules
from mivia_small_json.model import MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "routing-golden.jsonl"

# (relative path, sample cap). Order is part of the fixture contract.
# Public-release tree: the fixture samples only the corpora that ship
# (synthetic + heldout). The third-party wild sources were removed from
# the public tree, so their rows left the fixture with them.
SOURCES = (
    ("data/synthetic/train.jsonl", 120),
    ("data/heldout/test.jsonl", 120),
)
SEED = 20261003
EXPECTED = {
    "data/synthetic/train.jsonl": 2800,
    "data/heldout/test.jsonl": 350,
}

_TS_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def scrub(obj):
    """Replace wall-clock ``ts`` values with a stable marker."""
    if isinstance(obj, dict):
        return {
            k: ("<ts>" if k == "ts" and isinstance(v, str)
                and _TS_LIKE.match(v) else scrub(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def keep(meta):
    """Meta without the wall-clock latency (everything behavioral)."""
    return {k: v for k, v in meta.items() if k != "latency_ms"}


def main() -> None:
    # The canonical shipped world: load_active_router boots all shipped
    # schemas, so capture and verification agree no matter the caller.
    from mivia_small_json import bootstrap

    bootstrap.register_all_schemas()
    router = pipeline.load_router(MODEL_PATH)
    assert router is not None, "committed router artifact failed to load"
    rng = random.Random(SEED)
    out = []
    for rel, cap in SOURCES:
        rows = [json.loads(line) for line
                in (ROOT / rel).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        assert len(rows) == EXPECTED[rel], (
            f"{rel}: {len(rows)} rows, expected {EXPECTED[rel]}")
        for i in sorted(rng.sample(range(len(rows)), min(cap, len(rows)))):
            text = rows[i]["raw"]
            route = rules.route(text)
            obj, attempts, emit_schema, escaped = rules.emit(text)
            cls_obj, cls_meta = pipeline.classify(text, router)
            out.append({
                "src": rel,
                "i": i,
                "text": text,
                "route": route,
                "scores": rules.route_scores(text),
                "extract": scrub(rules.extract(text, route)),
                "emit": {"obj": scrub(obj), "attempts": attempts,
                         "schema": emit_schema,
                         "template_fallback": escaped},
                "classify": {"obj": scrub(cls_obj), "meta": keep(cls_meta)},
            })
    OUT.write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=True,
                           separators=(",", ":")) + "\n" for r in out),
        encoding="utf-8")
    print(f"wrote {len(out)} rows -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
