# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic corpus generator (seed 42, stdlib ``random``).

Why ``random.Random(42)`` and not ``numpy.default_rng(42)``: the corpus
is pure scalar/string data, so the stdlib generator suffices and its
stream is stable across numpy versions — one fewer variable in the
reproducibility chain. Seed and pools are fixed; byte-identical reruns
are asserted by tests/test_synth.py.

Per record: build the GOLD structured object first, then degrade it into
noisy raw text (typos, cross-schema lookalike prefixes, truncation,
whitespace damage). Degradation only ever touches ``raw`` — the gold
object stays clean for scoring. Splits: 80/10/10 stratified by schema,
assigned by generation index; the held-out test slice is frozen.

Corpus v2 (iteration 2): alias severity spellings in raw log text
(WARNING/VERBOSE/CRITICAL/ERR — gold keeps the canonical OTel level),
``commit_type`` present only for the prefix-carrying commit subset, and
hard cross-schema lookalikes (ERROR-prefixed commit lines, ISS ids in
log lines, "[auth] ERROR …" in issue text).

Registered schemas (``registry.register_schema``) are generated from
their gold ``examples``: each record mutates one example, renders it in
the canonical ``schema: field: value :: …`` layout and reuses the same
degradation (typos, truncation, cross-schema lookalike prefixes). The
documented example schema ``pull_request`` lives in ``bootstrap.py``;
:func:`register_example_schemas` below forwards there for
compatibility, so callers that only need the registered view (bench,
label, flywheel) import the bootstrap instead of this generator.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from pathlib import Path

from . import registry
from .schemas import LEVELS

SEED = 42
N_PER_SCHEMA = 1000
N_REGISTERED = 500  # per registered-with-examples schema (~500 pull_request)
SCHEMA_ORDER = ("issue", "log_event", "commit")  # fixed generation order
ROOT = Path(__file__).resolve().parents[2]

ISSUE_TITLES = (
    "Login times out after {n} minutes of inactivity",
    "Crash when saving a file larger than {n} MB",
    "Search returns duplicated results on page {n}",
    "Cannot invite users with plus-addressed emails",
    "Dashboard widgets lose layout after refresh",
    "API returns 500 when uploading empty attachment",
    "Dark mode contrast too low on settings page",
    "Webhook retries exhaust queue workers",
    "Pagination skips last page on filtered lists",
    "SSO redirect loop on expired session",
    "Export to CSV drops columns with unicode names",
    "Mobile viewport overflows on invoice table",
    "Rate limiter blocks internal health checks",
    "Cache invalidation misses nested project pages",
)
LABELS = ("auth", "backend", "frontend", "api", "database", "ci", "docs",
          "performance", "security", "ux", "billing", "search", "notifications")
LOG_COMPONENTS = ("auth", "api", "scheduler", "database", "worker", "cache",
                  "gateway", "mailer", "webhook")
LOG_MESSAGES = (
    "token validation failed for user {n}",
    "query timeout after {n} ms on table orders",
    "retry {n} of 5 for job sync_users",
    "connection pool exhausted, {n} waiters",
    "cache miss ratio {n}% above threshold",
    "payment webhook signature mismatch",
    "session expired for device {n}",
    "failed to publish event to topic orders",
    "disk usage at {n}% on volume data",
    "background job finished in {n} ms",
    "unexpected status {n} from upstream",
    "email queue drained, {n} messages",
)
# (type, message): the prefix is part of the raw text and the gold
# carries the matching commit_type.
PREFIXED_COMMIT_MESSAGES = (
    ("fix", "fix: retry webhook delivery on 5xx responses"),
    ("feat", "feat(search): add typo-tolerant query parsing"),
    ("fix", "fix(auth): refresh session before redirect loop"),
    ("chore", "chore: bump dependency floor for security patch"),
    ("refactor", "refactor(db): split migration runner from CLI"),
    ("docs", "docs: document the retry budget for integrators"),
    ("fix", "fix: handle null avatar in profile renderer"),
    ("perf", "perf(api): cache computed project quotas"),
    ("test", "test: cover pagination edge on filtered lists"),
    ("feat", "feat: export invoices as CSV with unicode columns"),
    ("fix", "fix(worker): drain poison messages to dead-letter"),
    ("chore", "chore(ci): run pytest with random ordering"),
)
# No conventional prefix -> gold omits commit_type (optional, absent when
# unknown).
PLAIN_COMMIT_MESSAGES = (
    "retry webhook delivery on 5xx responses",
    "add typo-tolerant query parsing",
    "refresh session before redirect loop",
    "bump dependency floor for security patch",
    "split migration runner from CLI",
    "document the retry budget for integrators",
    "handle null avatar in profile renderer",
    "cache computed project quotas",
    "cover pagination edge on filtered lists",
    "export invoices as CSV with unicode columns",
    "drain poison messages to dead-letter",
    "run pytest with random ordering",
)
AUTHORS = ("Jane Doe", "Ravi Patel", "Mia Chen", "Leo Marques", "Sam Ortiz",
           "Nadia Rahman")
PREFIXES = {
    "issue": ("fix: ", "ERROR [auth]: ", "WARN [api]: ", "feat(ui): "),
    "log_event": ("fix: ", "issue #4242: ", "commit "),
    "commit": ("ERROR [db]: ", "priority: high ", "issue #7: "),
}
# Cross-schema lookalikes for registered schemas (never keyed per schema).
GENERIC_PREFIXES = ("ERROR [api]: ", "ISS-1042: ", "commit ", "fix: ")

def register_example_schemas() -> None:
    """Compatibility shim: the example schema moved to ``bootstrap.py``.

    The generator still wants the four-schema world for its corpus;
    everything else (bench, label, flywheel) calls
    ``bootstrap.register_example_schemas`` directly and never imports
    this module.
    """
    from .bootstrap import register_example_schemas as _register

    _register()


def _ts(rng: random.Random) -> str:
    t = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        seconds=rng.randrange(5_184_000)
    )
    style = rng.random()
    if style < 0.6:
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")
    if style < 0.8:
        return t.strftime("%Y-%m-%dT%H:%M:%S") + f".{rng.randrange(100, 1000)}Z"
    sign, hours = rng.choice((("+", 2), ("-", 5), ("+", 0)))
    return t.strftime("%Y-%m-%dT%H:%M:%S") + f"{sign}{hours:02d}:00"


def _gold_issue(i: int, rng: random.Random) -> dict:
    return {
        "type": "issue",
        "id": f"ISS-{1000 + i * 7 + rng.randrange(0, 7)}",
        "title": rng.choice(ISSUE_TITLES).format(
            n=rng.choice([5, 12, 30, 64, 120, 512])
        ),
        "labels": rng.sample(LABELS, rng.choice([1, 1, 2, 2, 3])),
        "priority": rng.choices(("low", "medium", "high", "critical"),
                                weights=[30, 40, 20, 10])[0],
        "state": rng.choices(("open", "closed"), weights=[55, 45])[0],
    }


def _raw_issue(g: dict, rng: random.Random) -> str:
    labels_s = ", ".join(g["labels"])
    variant = rng.randrange(3)
    if variant == 0:
        return (f"[{g['id']}] {g['title']} (priority: {g['priority']}) "
                f"labels: {labels_s} state: {g['state']}")
    if variant == 1:
        return (f"issue #{g['id'].split('-')[1]}: {g['title']} — "
                f"priority {g['priority']}, labels {labels_s}, state {g['state']}")
    return (f"{g['id']} :: {g['title']} :: p={g['priority']} :: "
            f"labels={','.join(g['labels'])} :: state={g['state']}")


def _gold_log(i: int, rng: random.Random) -> dict:
    return {
        "type": "log_event",
        "level": rng.choices(LEVELS,
                             weights=[10, 15, 40, 20, 10, 5])[0],
        "component": rng.choice(LOG_COMPONENTS),
        "message": rng.choice(LOG_MESSAGES).format(n=rng.randrange(1, 5000)),
        "ts": _ts(rng),
    }


def _raw_log(g: dict, rng: random.Random) -> str:
    # Alias spellings appear in raw text; extraction maps them back to the
    # canonical level via schemas.LEVEL_ALIASES.
    level = g["level"]
    if rng.random() < 0.4:
        level = {"TRACE": "VERBOSE", "WARN": "WARNING", "FATAL": "CRITICAL",
                 "ERROR": "ERR"}.get(level, level)
    variant = rng.choices((0, 1, 2), weights=(40, 40, 20))[0]
    if variant == 0:
        return f"{g['ts']} {level} [{g['component']}]: {g['message']}"
    if variant == 1:
        return (f"level={level} comp={g['component']} ts={g['ts']} "
                f"msg={g['message']}")
    return f"{g['component']}: {g['message']}"


def _gold_commit(i: int, rng: random.Random) -> dict:
    sha = "".join(rng.choices("0123456789abcdef", k=40))
    if rng.random() < 0.5:  # subset carries a Conventional-Commits prefix
        ctype, message = rng.choice(PREFIXED_COMMIT_MESSAGES)
    else:
        ctype, message = None, rng.choice(PLAIN_COMMIT_MESSAGES)
    out = {
        "type": "commit",
        "sha": sha if rng.random() < 0.6 else sha[:7],
        "message": message,
        "files_changed": rng.randrange(1, 25),
        "author": rng.choice(AUTHORS),
    }
    if ctype:
        out["commit_type"] = ctype
    return out


def _raw_commit(g: dict, rng: random.Random) -> str:
    variant = rng.randrange(3)
    if variant == 0:
        return (f"commit {g['sha']} {g['message']} ({g['files_changed']} files "
                f"changed) author: {g['author']}")
    if variant == 1:
        return f"{g['sha']} {g['message']} — {g['author']} — {g['files_changed']} files changed"
    return f"{g['message']} ({g['files_changed']} files changed, author {g['author']})"


def _typo(s: str, rng: random.Random) -> str:
    idxs = [i for i in range(len(s) - 1) if s[i].isalpha() and s[i + 1].isalpha()]
    if not idxs:
        return s
    i = rng.choice(idxs)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _degrade(raw: str, rng: random.Random, schema: str) -> str:
    for _ in range(rng.choices((0, 1, 2), weights=(45, 40, 15))[0]):
        raw = _typo(raw, rng)
    if rng.random() < 0.15:  # cross-schema lookalike prefix
        raw = rng.choice(PREFIXES.get(schema, GENERIC_PREFIXES)) + raw
    if rng.random() < 0.10:  # hard lookalike: deliberate cross-schema bait
        hard = {
            "issue": lambda: f"[auth] ERROR {raw}",
            "log_event": lambda: f"{raw} (tracking: ISS-{rng.randrange(100, 10000)})",
            "commit": lambda: f"ERROR {raw}",
        }.get(schema)
        if hard is None:  # registered schema: generic bait
            def hard():
                return f"ISS-{rng.randrange(100, 10000)} {raw}"
        raw = hard()
    if rng.random() < 0.12:  # truncation (log rotation, clipped reports)
        raw = raw[: max(16, int(len(raw) * rng.uniform(0.55, 0.85)))].rstrip()
    if rng.random() < 0.08:
        raw = raw.replace(" ", "  ", 1)
    return raw


def _mutate_registered(schema: str, rng: random.Random) -> dict:
    """Gold record for a registered schema: one example, values redrawn."""
    examples = registry.examples_for(schema)
    spec = registry.resolve(schema)
    gold = dict(rng.choice(examples))
    for field, rule in spec.items():
        if field == "type":
            continue
        if "enum" in rule:
            gold[field] = rng.choice(list(rule["enum"]))
        elif rule.get("type") is int:
            gold[field] = rng.randrange(1, 10000)
        elif rule.get("type") is list:
            gold[field] = list(rng.choice(examples).get(field, []))
        else:
            pool = [e[field] for e in examples if field in e]
            gold[field] = rng.choice(pool) if pool else ""
    return gold


def _raw_registered(gold: dict) -> str:
    """Canonical raw layout for registered schemas: extractable by
    construction — the schema's own ``field:`` extraction patterns."""
    pairs = " :: ".join(f"{f}: {gold[f]}" for f in gold if f != "type")
    return f"{gold['type']}: {pairs}"


def build_splits(n_per_schema: int = N_PER_SCHEMA, seed: int = SEED,
                 n_registered: int = 0) -> dict:
    """Return {"train"|"dev"|"test": [record, ...]}, record = {schema, raw, gold}.

    Built-ins generate exactly as before (byte-stable). Registered
    schemas that ship gold examples generate ``n_registered`` records
    each (0 by default; the corpus writer passes N_REGISTERED),
    stratified by the same 80/10/10.
    """
    register_example_schemas()
    rng = random.Random(seed)
    builders = {"issue": _gold_issue, "log_event": _gold_log, "commit": _gold_commit}
    rawers = {"issue": _raw_issue, "log_event": _raw_log, "commit": _raw_commit}
    splits: dict[str, list] = {"train": [], "dev": [], "test": []}
    for schema in SCHEMA_ORDER:
        for i in range(n_per_schema):
            gold = builders[schema](i, rng)
            raw = _degrade(rawers[schema](gold, rng), rng, schema)
            frac = i / n_per_schema
            split = "train" if frac < 0.8 else ("dev" if frac < 0.9 else "test")
            splits[split].append({"schema": schema, "raw": raw, "gold": gold})
    for schema in registry.active():
        if schema in SCHEMA_ORDER or not registry.examples_for(schema):
            continue
        for i in range(n_registered):
            gold = _mutate_registered(schema, rng)
            raw = _degrade(_raw_registered(gold), rng, schema)
            frac = i / n_registered
            split = "train" if frac < 0.8 else ("dev" if frac < 0.9 else "test")
            splits[split].append({"schema": schema, "raw": raw, "gold": gold})
    return splits


def write_all(root: Path = ROOT, n_per_schema: int = N_PER_SCHEMA,
              seed: int = SEED, n_registered: int = N_REGISTERED) -> dict:
    """Write data/synthetic/{train,dev}.jsonl, data/heldout/test.jsonl + manifest."""
    splits = build_splits(n_per_schema, seed, n_registered)
    syn, held = root / "data" / "synthetic", root / "data" / "heldout"
    for d in (syn, held):
        d.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, path in (("train", syn / "train.jsonl"), ("dev", syn / "dev.jsonl"),
                        ("test", held / "test.jsonl")):
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in splits[split]),
            encoding="utf-8",
        )
        counts[split] = len(splits[split])
    test_sha = hashlib.sha256((held / "test.jsonl").read_bytes()).hexdigest()
    present: list[str] = []
    for records in splits.values():
        for r in records:
            if r["schema"] not in present:
                present.append(r["schema"])
    per_schema = {
        s: {k: sum(1 for r in v if r["schema"] == s) for k, v in splits.items()}
        for s in present
    }
    manifest = {"seed": seed, "n_per_schema": n_per_schema,
                "n_registered": n_registered, "records": counts,
                "per_schema": per_schema, "test_sha256": test_sha}
    (held / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    return manifest


def main() -> None:
    manifest = write_all()
    print(json.dumps(manifest, indent=2))
    for rel in ("data/synthetic/train.jsonl", "data/synthetic/dev.jsonl",
                "data/heldout/test.jsonl"):
        print(f"{rel}: {(ROOT / rel).stat().st_size} bytes")


if __name__ == "__main__":
    main()
