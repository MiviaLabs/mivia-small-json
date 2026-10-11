# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Strict, hand-rolled validators driven by the schema registry.

Design choices (documented per AGENTS.md):

- No ``jsonschema`` dependency: the schemas are small closed field sets,
  so explicit type/enum/format checks are easier to audit and keep the
  dependency count at zero. This is the rules-first choice (AGENTS.md
  rules 1 and 5), not an accident.
- The checked table lives in ``schemas.py`` (single source of truth);
  this module holds the checking, coercion and bounded-repair mechanics.
  User-registered schemas (``registry.register_schema``) resolve through
  the same accessors, so validation/repair/template work unchanged for
  them.
- ``validate(obj, schema)`` returns structured error strings
  ``"<kind>:<field>"`` so ``repair`` acts on exactly the reported errors
  (error-fed repair, never a blanket overwrite).
- The pipeline calls ``repair`` at most twice (retry budget <= 2,
  AGENTS.md rule 2); validity is restored by bounded repair, and the
  minimal valid ``template`` is the last-resort constrained emission.
- Fields marked ``optional`` in the registry may be absent: no ``missing``
  error, no default injection, and ``template`` omits them.
"""

from __future__ import annotations

import datetime as dt
import json
import re

from . import registry
from .schemas import (
    COMMIT_TYPES,
    ISO_TS_RE,
    LEVELS,
    LEVEL_ALIASES,
    PRIORITIES,
    SHA_RE,
    STATES,
)


class _ActiveSchemas:
    """Read-only live view over ``registry.active()``.

    Iterating or materializing always reflects the registrations made so
    far (built-ins first, then registration order), so consumers that
    imported ``SCHEMAS`` once — e.g. ``bench`` — stay schema-agnostic.
    """

    def __iter__(self):
        return iter(registry.active())

    def __len__(self):
        return len(registry.active())

    def __getitem__(self, i):
        return registry.active()[i]

    def __contains__(self, schema):
        return schema in registry.active()

    def index(self, schema):
        return registry.active().index(schema)

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __repr__(self):
        return repr(tuple(self))


SCHEMAS = _ActiveSchemas()
__all__ = [
    "SCHEMAS", "PRIORITIES", "LEVELS", "STATES", "COMMIT_TYPES",
    "validate", "repair", "template", "now_iso", "normalize_ts", "normalize_sha",
]


def validate(obj, schema: str) -> list[str]:
    """Return error strings ("kind:field"); an empty list means valid."""
    spec = registry.resolve(schema)
    if not isinstance(obj, dict):
        return ["type:<root>"]
    errors: list[str] = []
    for field, rule in spec.items():
        if field not in obj:
            if not rule.get("optional"):
                errors.append(f"missing:{field}")
            continue
        value = obj[field]
        if "const" in rule:
            if value != rule["const"]:
                errors.append(f"const:{field}")
            continue
        want = rule["type"]
        if want is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif want is list:
            ok = isinstance(value, list) and all(isinstance(x, str) for x in value)
        else:
            ok = isinstance(value, str)
        if not ok:
            errors.append(f"type:{field}")
            continue
        if "enum" in rule and value not in rule["enum"]:
            errors.append(f"enum:{field}")
        if rule.get("format") == "iso8601" and not _valid_ts(value):
            errors.append(f"format:{field}")
        if rule.get("format") == "sha" and not SHA_RE.match(value):
            errors.append(f"format:{field}")
        if rule.get("format") == "json" and not _valid_json_object(value):
            errors.append(f"format:{field}")
    for field in obj:
        if field not in spec:
            errors.append(f"unexpected:{field}")
    return errors


def _valid_ts(value: str) -> bool:
    if not ISO_TS_RE.match(value):
        return False
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _reject_nan_constant(token: str):
    """NaN/Infinity are python-json extensions, not JSON (review 2026-10-10)."""
    raise ValueError(f"non-JSON constant: {token}")


def _valid_json_object(value: str) -> bool:
    """``format: "json"`` fields carry a JSON OBJECT as a string.

    Free prose ("run the tests now") and JSON scalars/arrays are invalid:
    the field's contract is a serialized argument object (spanfix wave
    bug 10, 2026-10-10 — validate used to accept any str here).
    """
    try:
        return isinstance(json.loads(value, parse_constant=_reject_nan_constant), dict)
    except (ValueError, TypeError):
        return False


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_ts(value: str) -> str:
    """Best-effort normalization of a timestamp to strict ISO-8601 (Z).

    Nanosecond fractions (7-9 digits, e.g. platform logs) are truncated
    to the 6 digits ``datetime`` keeps, so extraction normalizes the
    original reading instead of discarding it to wall-clock ``now()``.
    """
    s = (value or "").strip().upper()
    # Only the date-time separator space becomes "T"; a space before a
    # timezone offset ("... 12:00:00 +0000") is dropped, not turned into
    # a "T" (review 2026-10-09, F5).
    s = re.sub(r"\s+(Z|[+-]\d{2}:?\d{2})$", r"\1", s)
    s = re.sub(r"^(\d{4}-\d{2}-\d{2})[T ]", r"\1T", s, count=1)
    if re.search(r"[+-]\d{4}$", s):
        s = s[:-2] + ":" + s[-2:]
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    return s if _valid_ts(s) else ""


def normalize_sha(value: str) -> str:
    """Keep hex chars only; below 7 hex chars return "".

    An input too short to be a sha id is ABSENT, never padded: fabricating
    digits ("abc12" -> "abc1200") is the placeholder class the honesty law
    bans (owner no-truncation/honesty directive 2026-10-10). The empty or
    short result then drops through repair's optional-field absent path.
    40 is the schema's sha format bound (SHA_RE), not a content cap; the
    7-40 extraction regexes never deliver more than 40 hex in the first
    place, and output stays bounded by the request's 256 KiB input cap.
    """
    s = re.sub(r"[^0-9a-f]", "", (value or "").lower())
    if len(s) < 7:
        return ""
    return s[:40]


def _default(schema: str, field: str):
    """Repair/template fallback for ``field``.

    Every schema — built-in or registered — resolves through one
    registry view: declared defaults first (``schemas.BUILTIN_DEFAULTS``
    or the registered defaults), then rule shape (format/enum/type).
    """
    defaults = registry.defaults_for(schema)
    if field in defaults:
        return defaults[field]
    rule = registry.resolve(schema).get(field, {})
    if "const" in rule:
        return rule["const"]  # e.g. the "type" field (review 2026-10-09, F6c)
    if rule.get("format") == "iso8601":
        return now_iso()
    if rule.get("format") == "sha":
        # An absent/too-short sha has no honest value: "" keeps the record
        # invalid-honest instead of fabricating "0000000" digits that would
        # pass the format check (no-truncation/honesty law 2026-10-10).
        # Optional sha fields never reach template/repair-defaults; they
        # drop through repair's absent path.
        return normalize_sha("")
    if rule.get("format") == "json":
        return "{}"  # the honest-minimum arguments object (review 2026-10-10)
    if "enum" in rule:
        return rule["enum"][len(rule["enum"]) // 2]
    want = rule.get("type")
    if want is int:
        return 0
    if want is list:
        return []
    return ""


def _coerce(schema: str, field: str, value):
    want = registry.resolve(schema)[field]["type"]
    if want is int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, str):
            m = re.search(r"-?\d+", value)
            return int(m.group(0)) if m else 0
        if isinstance(value, float):
            return int(value)
        return 0
    if want is list:
        if isinstance(value, str):
            return [t.strip() for t in re.split(r"[,;/]+", value) if t.strip()]
        return []
    if value is None:
        return ""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _nearest(schema: str, field: str, value):
    enum = registry.resolve(schema)[field]["enum"]
    raw = str(value).strip()
    # The OTel level-alias ladder applies only when the field's enum
    # actually IS the log levels — not to any field merely named "level"
    # (review 2026-10-09, F6b).
    is_levels = list(enum) == list(LEVELS)
    v = raw.upper() if is_levels else raw.lower()
    if is_levels:
        if v in LEVEL_ALIASES:
            return LEVEL_ALIASES[v]
        probe, fallback = v[:4], LEVELS[2]          # INFO, the OTel default
    else:
        probe, fallback = v[:3], enum[len(enum) // 2]
    if not probe:
        # An empty value has no prefix signal: fall back to the schema's
        # declared default when it is in the enum, never to enum[0]
        # (review 2026-10-09, F6a).
        declared = registry.defaults_for(schema).get(field)
        if declared in enum:
            return declared
        return fallback
    if probe in enum:
        return probe
    for e in enum:
        if e.startswith(probe):
            return e
    return fallback


def repair(obj, errors, schema: str) -> dict:
    """Error-fed repair: fix exactly the fields named in ``errors``."""
    spec = registry.resolve(schema)
    optional = frozenset(f for f, r in spec.items() if r.get("optional"))
    fixed = dict(obj) if isinstance(obj, dict) else {}
    for err in errors:
        kind, _, field = err.partition(":")
        if kind == "unexpected":
            fixed.pop(field, None)
            # Non-string dict keys report via str() in the error but pop()
            # by name would miss them; drop the matching raw key too
            # (review 2026-10-09, F6d).
            for key in [k for k in fixed
                        if not isinstance(k, str) and str(k) == field]:
                del fixed[key]
            continue
        if kind == "missing":
            if field in optional:
                continue
            fixed[field] = _default(schema, field)
        elif fixed.get(field) is None and field not in optional:
            fixed[field] = _default(schema, field)
        elif kind == "const":
            fixed[field] = schema
        elif kind == "type":
            fixed[field] = _coerce(schema, field, fixed.get(field))
        elif kind == "enum":
            if schema in registry.BUILT_IN_SCHEMAS:
                # Built-ins keep the tuned nearest ladder (OTel level
                # aliases etc.) byte-identical — the golden fixture and
                # the pre-registered bars are measured on it.
                fixed[field] = _nearest(schema, field, fixed.get(field))
            else:
                # Refusal contract (enum-aliases feature, 2026-10-10):
                # for user schemas a found-but-unmatched enum value is
                # UNFILLED, never rewritten. The ONLY rewrite is the
                # schema author's declared alias map; anything else is
                # dropped here so the field flows through the
                # missing-field path (absent for optional fields) —
                # "it is urgent" must never silently become "normal".
                canonical = registry.canonical_enum_value(
                    schema, field, fixed.get(field))
                if canonical is None:
                    fixed.pop(field, None)
                else:
                    fixed[field] = canonical
        elif kind == "format":
            fmt = spec.get(field, {}).get("format")
            if fmt == "iso8601":
                fixed[field] = normalize_ts(str(fixed.get(field) or ""))
                if not fixed[field]:
                    # An optional field the source never carried is DROPPED,
                    # never fabricated with the wall clock (a 2019 log must
                    # not learn today's timestamp; same honesty rule as
                    # sha's short-hex drop below).
                    if field in optional:
                        fixed.pop(field, None)
                    else:
                        fixed[field] = now_iso()
            elif fmt == "sha":
                raw = str(fixed.get(field) or "")
                fixed[field] = normalize_sha(raw)
                if field in optional and (not raw or not fixed[field]):
                    # An optional sha the source never carried (or whose
                    # content has fewer than 7 hex chars) is absent, not
                    # zero-padded — the fabricated-placeholder class
                    # (owner sweep 2026-10-10; short-hex drop is the
                    # no-truncation law, 2026-10-10).
                    fixed.pop(field, None)
            elif fmt == "json":
                # A non-JSON / pseudo-JSON payload repairs through the
                # owning module's ladder (tool_call.repair_arguments:
                # never raises, always terminates at a valid JSON
                # object — "{}" at worst, so the repaired field validates)
                # (spanfix wave bug 10, 2026-10-10). Lazy import: the
                # schema module imports this package's registry.
                from .tool_call import repair_arguments

                fixed[field] = repair_arguments(
                    str(fixed.get(field) or ""))
    return fixed


def template(schema: str) -> dict:
    """Minimal, strictly valid object for ``schema`` (last-resort emission).

    Optional fields are omitted: absence is schema-valid by definition.
    """
    spec = registry.resolve(schema)
    optional = {f for f, r in spec.items() if r.get("optional")}
    return {f: (schema if f == "type" else _default(schema, f))
            for f in spec if f not in optional}
