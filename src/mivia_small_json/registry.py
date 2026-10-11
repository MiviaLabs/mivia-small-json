# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Runtime registry for user-defined output schemas and pluggable routers.

A project adds a NEW output format by calling :func:`register_schema`
once; validation, repair, rules routing/extraction, the never-raise
pipeline, the CLI and the bench then all work for it with zero changes
to built-in code. Registration is additive at runtime: the built-in
tuple (issue, log_event, commit) never changes and built-in extractors
keep their tuned behavior.

Contract (see ``schemas.SPEC`` for the entry shape):

- ``spec``: field rules dict — required ``"type"`` entry whose
  ``"const"`` equals the schema name, then ``{"type": str|int|list}``
  rules with optional ``"enum"`` / ``"format"`` ("iso8601"|"sha"|"json") /
  ``"optional"`` flags. An enum rule may declare ``"aliases"``: a
  ``{canonical member: [surface, ...]}`` map of declared surface words
  (any language, any vocabulary) that coerce to that member. Invalid
  specs raise ``ValueError``.
- ``routing``: ``{"strong": [regex, ...], "weak": [regex, ...]}`` marker
  patterns for the 2-points-strong / 1-point-weak keyword router.
- ``extraction``: ``{field: [regex, ...]}`` — patterns are tried in
  order, first match wins, each must carry exactly one capture group.
  A field no pattern matches stays absent and falls into the normal
  validate/repair ladder.
- ``defaults``: per-field fallback values used by repair/template
  (optional fields may be omitted).
- ``examples``: gold records (must pass the strict validator) used by
  ``synth`` to generate a degradation corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager

from .schemas import (
    BUILTIN_DEFAULTS,
    BUILTIN_ROUTING,
    SCHEMAS as BUILT_IN_SCHEMAS,  # single source of truth
    SHA_RE,
)


def _ok_iso8601(value) -> bool:
    """Format check for registered defaults (reuses validate's timestamp check)."""
    from .validate import _valid_ts  # lazy: validate imports this module

    return isinstance(value, str) and _valid_ts(value)


def _ok_json_object(value) -> bool:
    """Format check for registered defaults (reuses validate's object check)."""
    from .validate import _valid_json_object  # lazy: validate imports this module

    return isinstance(value, str) and _valid_json_object(value)

_ALLOWED_RULE_KEYS = frozenset(
    {"const", "type", "enum", "format", "optional", "aliases"})
_ALLOWED_TYPES = (str, int, list)
# "json": the field is a string that must parse as a JSON OBJECT
# (tool_call.arguments, spanfix wave bug 10, 2026-10-10).
_ALLOWED_FORMATS = ("iso8601", "sha", "json")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _normalize_surface(value) -> str:
    """Spanfix surface normalization for enum/alias matching.

    The existing spanfix conventions (span_fill._wrap_strip + the
    enum-comparison casefold, spanfix wave bugs 5 and 8, 2026-10-10):
    trim, casefold, then wrap-strip — leading quote characters and the
    trailing punctuation that legitimately terminates a captured span
    are wrapper noise, not content. Matching after this normalization is
    EXACT: no fuzzy matching, no engine-invented synonyms — a surface
    coerces only when the schema author declared it (honest-values law).
    """
    s = str(value).strip().casefold()
    return s.lstrip("\"'").rstrip(" .,;:!?)]}\"'")


def _enum_table(rule: dict) -> dict[str, str]:
    """Normalized-surface -> canonical-member lookup for one enum rule.

    Declared enum members first, then the rule's ``aliases`` (the schema
    author's declaration — the ONLY mapping that ever rewrites a value).
    Registration validation guarantees a surface resolves to at most one
    member, so ``setdefault`` order never masks a distinct target.
    """
    table: dict[str, str] = {}
    for member in rule["enum"]:
        table.setdefault(_normalize_surface(member), member)
    for target, surfaces in (rule.get("aliases") or {}).items():
        for surface in surfaces:
            table.setdefault(_normalize_surface(surface), target)
    return table


def canonical_enum_value(schema: str, field: str, value):
    """Author-declared enum coercion for one value; ``None`` = refused.

    The shared coercion contract (enum-aliases feature, 2026-10-10):
    extracted span -> trim + casefold + wrap-strip -> exact enum member
    first -> else declared alias lookup -> canonical member. Anything
    else is REFUSED (``None``): the caller treats the field as UNFILLED
    (the honest missing-field path), never a default, never a
    nearest-looking rewrite. Only used for enum fields.
    """
    rule = resolve(schema).get(field, {})
    if "enum" not in rule:
        return value
    if not isinstance(value, str):
        return None
    return _enum_table(rule).get(_normalize_surface(value))

# name -> {"spec", "routing" (compiled), "extraction" (compiled),
#          "defaults", "examples"} in registration order.
_REGISTERED: dict[str, dict] = {}


def _fail(what: str, detail: str) -> ValueError:
    return ValueError(f"register_schema: {what}: {detail}")


def _compile_many(patterns, what: str, groups: int | None = None) -> tuple:
    if isinstance(patterns, str) or not isinstance(patterns, (list, tuple)):
        raise _fail(what, "expected a list of regex strings")
    out = []
    for p in patterns:
        if not isinstance(p, str):
            raise _fail(what, f"pattern {p!r} is not a string")
        try:
            compiled = re.compile(p)
        except re.error as exc:
            raise _fail(what, f"pattern {p!r} does not compile ({exc})") from exc
        if groups is not None and compiled.groups != groups:
            raise _fail(what, f"pattern {p!r} must have exactly "
                              f"{groups} capture group(s), has {compiled.groups}")
        out.append(compiled)
    return tuple(out)


# Built-in routing markers, compiled once at import from the declarative
# table in ``schemas.py`` (same shape as a registered entry).
_BUILTIN_COMPILED_ROUTING = {
    name: (_compile_many(markers["strong"], f"BUILTIN_ROUTING.{name}.strong"),
           _compile_many(markers["weak"], f"BUILTIN_ROUTING.{name}.weak"))
    for name, markers in BUILTIN_ROUTING.items()
}


def _check_spec(name: str, spec) -> None:
    if not isinstance(spec, dict) or not spec:
        raise _fail("spec", "expected a non-empty dict of field rules")
    if "type" not in spec:
        raise _fail("spec", 'missing const "type" entry: add '
                            f'{{"type": {{"const": "{name}"}}}}')
    type_rule = spec["type"]
    if (not isinstance(type_rule, dict) or set(type_rule) != {"const"}
            or type_rule["const"] != name):
        raise _fail("spec", 'the "type" entry must be exactly '
                            f'{{"const": "{name}"}}')
    for field, rule in spec.items():
        if not isinstance(field, str) or not field:
            raise _fail("spec", f"invalid field name {field!r}")
        if not isinstance(rule, dict):
            raise _fail(f"spec.{field}", "rule must be a dict")
        unknown = set(rule) - _ALLOWED_RULE_KEYS
        if unknown:
            raise _fail(f"spec.{field}",
                        f"unknown rule key(s) {sorted(unknown)}; allowed: "
                        f"{sorted(_ALLOWED_RULE_KEYS)}")
        if field == "type":
            continue  # const-only rule, already validated above
        if "const" in rule:
            raise _fail(f"spec.{field}",
                        '"const" is only allowed on the "type" entry')
        if "type" not in rule:
            raise _fail(f"spec.{field}", 'missing "type" (str, int or list)')
        if rule["type"] not in _ALLOWED_TYPES:
            raise _fail(f"spec.{field}",
                        f"unknown type {rule['type']!r}; allowed: "
                        f"{[_t.__name__ for _t in _ALLOWED_TYPES]}")
        if "enum" in rule:
            enum = rule["enum"]
            if (not isinstance(enum, (list, tuple)) or not enum
                    or not all(isinstance(e, str) for e in enum)):
                raise _fail(f"spec.{field}",
                            "enum must be a non-empty list of strings")
            if rule["type"] is list:
                raise _fail(f"spec.{field}",
                            '"enum" is not supported on list-typed fields; '
                            "validate/repair check list items by type only")
        if "aliases" in rule:
            # Enum aliases (2026-10-10): declared surface vocabulary that
            # coerces to a canonical member. Keys must be declared enum
            # members; values are lists of surface strings; every surface
            # must resolve to exactly one member after normalization.
            if "enum" not in rule:
                raise _fail(f"spec.{field}",
                            '"aliases" requires "enum" on the same rule')
            if rule["type"] is not str:
                raise _fail(f"spec.{field}",
                            '"aliases" is only allowed on str-typed '
                            'enum fields')
            aliases = rule["aliases"]
            if not isinstance(aliases, dict) or not aliases:
                raise _fail(f"spec.{field}",
                            "aliases must be a non-empty dict "
                            "{canonical member: [surface, ...]}")
            claimed: dict[str, str] = {}
            for target, surfaces in aliases.items():
                if target not in rule["enum"]:
                    raise _fail(f"spec.{field}",
                                f"alias target {target!r} is not a declared "
                                "enum member")
                if (not isinstance(surfaces, (list, tuple)) or not surfaces
                        or not all(isinstance(s, str) for s in surfaces)):
                    raise _fail(f"spec.{field}",
                                f"aliases[{target!r}] must be a non-empty "
                                "list of strings")
                for surface in surfaces:
                    norm = _normalize_surface(surface)
                    if not norm:
                        raise _fail(f"spec.{field}",
                                    f"alias {surface!r} normalizes to an "
                                    "empty surface")
                    other = claimed.get(norm)
                    if other is not None and other != target:
                        raise _fail(f"spec.{field}",
                                    f"alias {surface!r} is declared for "
                                    f"both {other!r} and {target!r}; a "
                                    "surface must coerce to exactly one "
                                    "member")
                    claimed[norm] = target
                    for member in rule["enum"]:
                        if member != target and _normalize_surface(member) == norm:
                            raise _fail(f"spec.{field}",
                                        f"alias {surface!r} collides with "
                                        f"enum member {member!r}")
        if "format" in rule and rule["type"] is not str:
            raise _fail(f"spec.{field}",
                        '"format" is only allowed on str-typed fields')
        if "format" in rule and rule["format"] not in _ALLOWED_FORMATS:
            raise _fail(f"spec.{field}",
                        f"unknown format {rule['format']!r}; allowed: "
                        f"{list(_ALLOWED_FORMATS)}")
        if "optional" in rule and not isinstance(rule["optional"], bool):
            raise _fail(f"spec.{field}", '"optional" must be a bool')


def _freeze_spec(spec: dict) -> dict:
    """Defensive copy of a caller's spec (review 2026-10-10, C6).

    Rule dicts are copied and enum lists converted to tuples: a caller
    who appends to the enum list it passed in must never change
    validation/repair/template after registration. The ``aliases`` map
    is copied the same way (member -> tuple of surfaces), so mutating
    the caller's alias lists after registration cannot change coercion.
    Membership checks are unaffected, and the built-in specs already
    store enums as tuples (json canonicalization is identical, so
    spec_fingerprint does not move).
    """
    frozen: dict[str, dict] = {}
    for field, rule in spec.items():
        stored = dict(rule)
        enum = stored.get("enum")
        if isinstance(enum, list):
            stored["enum"] = tuple(enum)
        aliases = stored.get("aliases")
        if isinstance(aliases, dict):
            stored["aliases"] = {
                target: (tuple(surfaces) if isinstance(surfaces, list)
                         else surfaces)
                for target, surfaces in aliases.items()
            }
        frozen[field] = stored
    return frozen


def register_schema(name: str, spec: dict, routing: dict, extraction: dict,
                    defaults: dict | None = None,
                    examples: list[dict] | None = None,
                    extractor=None,
                    description: str = "",
                    fewshot: bool = False) -> dict | None:
    """Register a new output schema; every consumer sees it immediately.

    ``extractor`` is an optional callable ``(text) -> dict`` that replaces
    the compiled pattern engine for this schema, so a schema with
    cross-field logic can ship its own extractor.
    ``description`` is an optional semantic description string used by
    zero-shot dynamic schema selection; ``None`` (or any non-string) is
    treated as not supplied, never stringified (survfix 2026-10-10).
    The spec rules, defaults and examples are defensively copied at
    registration: later mutations of the caller's objects never change
    the registered schema (review 2026-10-10, C6).

    Few-shot onboarding (spec ``few-shot-onboarding``):
    ``fewshot=True`` opts a registration into training a per-schema
    learned head from the ``examples`` — an example item may carry a
    reserved ``"_text"`` key (the source text); it is stripped before
    strict validation and stored as a (text, fields) training pair,
    keeping the stored gold records clean for ``synth``. OPT-IN on
    purpose: the shipped boot path registers ``pull_request`` WITH
    examples on every router boot, and auto-onboarding there would train
    at every boot and change shipped behavior (spec bar F5). Returns the
    onboarding report when ``fewshot=True`` (``trained`` False carries a
    ``reason``), ``None`` otherwise — today's contract, byte-identical.
    A spec field named ``"_text"`` is rejected under ``fewshot=True``
    (it would make the reserved key ambiguous).

    Raises ``ValueError`` with a clear message on any invalid input.
    Raises ``ValueError`` when ``name`` is a built-in or already
    registered schema (unregister first to replace).
    """
    if not isinstance(name, str) or not _NAME_RE.match(name or ""):
        raise _fail("name", f"{name!r} must match {_NAME_RE.pattern}")
    if name in _REGISTERED or name in BUILT_IN_SCHEMAS:
        raise _fail("name", f"{name!r} is already registered")
    if fewshot and "_text" in spec:
        raise _fail("spec", '"_text" is the reserved few-shot text key '
                            "and cannot be a spec field under fewshot=True")
    _check_spec(name, spec)
    # N13 (review 2026-10-10): a non-callable extractor would raise
    # TypeError inside rules.extract forever after, silently degrading
    # the schema to template records — fail here, at the registration
    # site, not at first extraction.
    if extractor is not None and not callable(extractor):
        raise _fail("extractor", f"{extractor!r} is not callable")

    if (not isinstance(routing, dict)
            or not isinstance(routing.get("strong"), (list, tuple))
            or not isinstance(routing.get("weak"), (list, tuple))):
        raise _fail("routing",
                    'expected {"strong": [regex...], "weak": [regex...]}')
    strong = _compile_many(routing["strong"], "routing.strong")
    weak = _compile_many(routing["weak"], "routing.weak")

    if not isinstance(extraction, dict):
        raise _fail("extraction", "expected {field: [regex...]}")
    compiled_extraction: list[tuple[str, tuple]] = []
    for field, patterns in extraction.items():
        if field not in spec or field == "type":
            raise _fail("extraction",
                        f"{field!r} is not a spec field (or is \"type\")")
        compiled_extraction.append(
            (field, _compile_many(patterns, f"extraction.{field}", groups=1)))
    compiled_extraction.sort(key=lambda item: list(spec).index(item[0]))

    defaults = dict(defaults or {})
    for field, value in defaults.items():
        if field not in spec or field == "type":
            raise _fail("defaults", f"{field!r} is not a spec field")
        rule = spec[field]
        want = rule["type"]
        if want is int and (not isinstance(value, int)
                            or isinstance(value, bool)):
            raise _fail(f"defaults.{field}", "expected an int")
        if want is list and (not isinstance(value, list)
                             or not all(isinstance(item, str)
                                        for item in value)):
            # N8 (review 2026-10-10): the message always said "list of
            # strings"; now the check enforces it — a [1, 2] default made
            # template() emit an object that fails strict validation.
            raise _fail(f"defaults.{field}", "expected a list of strings")
        if want is str and not isinstance(value, str):
            raise _fail(f"defaults.{field}", "expected a string")
        if "enum" in rule and value not in rule["enum"]:
            raise _fail(f"defaults.{field}",
                        f"{value!r} is not in the field's enum")
        if rule.get("format") == "iso8601" and not _ok_iso8601(value):
            raise _fail(f"defaults.{field}",
                        "expected an ISO-8601 timestamp for a format field")
        if rule.get("format") == "sha" and not (
                isinstance(value, str) and SHA_RE.match(value)):
            raise _fail(f"defaults.{field}",
                        "expected a hex sha (7-40 chars) for a format field")
        if rule.get("format") == "json" and not _ok_json_object(value):
            raise _fail(f"defaults.{field}",
                        "expected a JSON object string for a format field")

    # Not-supplied means not-supplied (survfix 2026-10-10, v10): None or
    # a type-mismatched description must not become the literal string
    # "None"/"42" with description_explicit=True — that would opt the
    # schema into zero-shot selection scored against "name: None". Only
    # a real (non-empty after strip) string is a semantic description.
    explicit_desc = description.strip() if isinstance(description, str) \
        else ""
    desc = explicit_desc
    if not desc:
        fields = [f for f in spec if f != "type"]
        desc = f"{name}: " + ", ".join(fields)

    entry = {
        # Defensive copies (review 2026-10-10, C6): enums frozen to
        # tuples, list defaults copied, so caller-side mutations of the
        # objects passed in cannot change validate/template/repair.
        "spec": _freeze_spec(spec),
        "strong": strong,
        "weak": weak,
        "extraction": tuple(compiled_extraction),
        "defaults": {f: (list(v) if isinstance(v, list) else v)
                     for f, v in defaults.items()},
        "examples": (),
        "extractor": extractor,
        "description": desc,
        # Whether the caller supplied a semantic description. Only those
        # schemas are zero-shot selectable; the auto-generated field list
        # is a display fallback, not a semantic description.
        "description_explicit": bool(explicit_desc),
    }
    # Store first so example validation can resolve() the new schema;
    # roll back atomically if any example is not strictly valid.
    _REGISTERED[name] = entry
    examples = list(examples or [])
    pairs: list[dict] = []
    if examples:
        from .validate import validate as _validate  # lazy: avoids a cycle

        try:
            cleaned: list[dict] = []
            for i, item in enumerate(examples):
                gold = item
                # Reserved few-shot text key: stripped before validation so
                # the stored gold record stays clean for synth (the pair is
                # what onboarding trains on).
                if isinstance(item, dict) and "_text" in item:
                    text = item["_text"]
                    if not isinstance(text, str) or not text.strip():
                        raise _fail(f"examples[{i}]",
                                    '"_text" must be a non-empty string')
                    gold = {k: v for k, v in item.items() if k != "_text"}
                    pairs.append({
                        "text": text,
                        "fields": {k: v for k, v in gold.items()
                                   if k != "type"},
                    })
                errors = _validate(gold, name) if isinstance(gold, dict) \
                    else ["type:<root>"]
                if errors:
                    raise _fail(f"examples[{i}]",
                                f"gold record is not strictly valid: {errors}")
                cleaned.append(gold)
            entry["examples"] = tuple(dict(g) for g in cleaned)
        except Exception:
            del _REGISTERED[name]
            raise
    entry["fewshot_pairs"] = tuple(dict(p) for p in pairs)

    # Few-shot onboarding (opt-in): train + persist + wire the per-schema
    # head. Registration survives an onboarding failure (never-raise law);
    # the honest report says so instead.
    report = None
    if fewshot:
        from . import fewshot as _fewshot

        try:
            report = _fewshot.onboard(name)
        except Exception as exc:
            report = {"schema": name, "trained": False,
                      "reason": f"{type(exc).__name__}: {exc}"}
        entry["fewshot_report"] = report
    return report


def unregister(name: str) -> None:
    """Remove a registered schema. Built-ins cannot be removed.

    A few-shot-onboarded schema's learned artifact (``data/fewshot/``)
    and in-memory head are cleaned with it (spec: unregister cleans the
    artifact)."""
    if name in BUILT_IN_SCHEMAS:
        raise _fail("unregister", f"{name!r} is built-in and cannot be removed")
    if _REGISTERED.pop(name, None) is None:
        raise _fail("unregister", f"{name!r} is not registered")
    from . import fewshot as _fewshot  # lazy: fewshot imports this module

    _fewshot.forget(name)


def _reset() -> None:
    """Drop every registration (test isolation only)."""
    _REGISTERED.clear()


@contextmanager
def temporary_schemas() -> None:
    """Snapshot registration state; restore it on exit (test isolation).

    Schemas registered inside the ``with`` block are removed on exit and
    any schema unregistered inside is restored. Built-ins never change.
    """
    saved = dict(_REGISTERED)
    try:
        yield
    finally:
        _REGISTERED.clear()
        _REGISTERED.update(saved)


def active() -> tuple:
    """All usable schemas: built-ins first, then registration order."""
    return BUILT_IN_SCHEMAS + tuple(_REGISTERED)


def custom_registered() -> tuple[str, ...]:
    """Tuple of dynamically registered schema names (excluding built-ins).

    Retained as public API. ``pipeline`` no longer reads this: its
    zero-shot fallback keys off :func:`zero_shot_registered` (only
    schemas that opted in with an explicit description), because
    re-scoring a schema the router already emits is wasted work. The
    accessor stays exported for callers that need the full set of
    user-registered names.
    """
    return tuple(_REGISTERED.keys())


def zero_shot_registered() -> tuple[str, ...]:
    """Registered schemas that opted into zero-shot selection.

    Zero-shot ``select`` scores text against a schema's *semantic*
    description. A schema registered without one carries only the
    auto-generated field list, which is a display fallback, not something
    to select on — so it is not a candidate. The shipped schemas are all
    in this group, which keeps the fallback from re-scoring inputs the
    router already covers.
    """
    return tuple(name for name, entry in _REGISTERED.items()
                 if entry.get("description_explicit"))


def description_for(schema: str) -> str:
    """Semantic description for a registered schema, or fallback string."""
    entry = _REGISTERED.get(schema)
    if entry is not None and entry.get("description"):
        return entry["description"]
    return schema


def resolve(schema: str) -> dict:
    """Merged spec entry for ``schema`` (built-in or registered).

    Raises ``KeyError`` naming the active schemas when unknown.
    """
    if schema in BUILT_IN_SCHEMAS:
        from . import schemas

        return schemas.SPEC[schema]
    entry = _REGISTERED.get(schema)
    if entry is None:
        raise KeyError(f"unknown schema {schema!r}; active schemas: "
                       f"{list(active())}")
    return entry["spec"]


def routing_for(schema: str) -> tuple[tuple, tuple]:
    """Compiled (strong, weak) marker patterns for any active schema."""
    compiled = _BUILTIN_COMPILED_ROUTING.get(schema)
    if compiled is not None:
        return compiled
    entry = _REGISTERED.get(schema)
    if entry is None:
        raise KeyError(f"no registered routing for {schema!r}")
    return entry["strong"], entry["weak"]


def extraction_for(schema: str) -> tuple:
    """Compiled ((field, patterns), ...) for a registered schema."""
    entry = _REGISTERED.get(schema)
    if entry is None:
        raise KeyError(f"no registered extraction for {schema!r}")
    return entry["extraction"]


def defaults_for(schema: str) -> dict:
    """Per-field repair/template defaults for any active schema.

    Mutable default values (lists) are deep-copied: handing out the
    module-level list would let one caller's append corrupt every later
    template and repair (review 2026-10-09, F7c).
    """
    import copy

    if schema in BUILT_IN_SCHEMAS:
        return copy.deepcopy(BUILTIN_DEFAULTS)
    entry = _REGISTERED.get(schema)
    if entry is None:
        raise KeyError(f"no registered defaults for {schema!r}")
    return copy.deepcopy(entry["defaults"])


def extractor_for(schema: str):
    """Extractor callable for any active schema.

    Built-ins return their tuned hand-written extractor: the cross-field
    logic stays code, not pattern tables. Registered schemas return the
    compiled pattern engine — patterns tried in order per field, first
    match wins, fields no pattern matches stay absent and fall into the
    normal validate/repair ladder.

    Enum fields (2026-10-10) are coerced through the author-declared
    vocabulary: captured span -> trim + casefold + wrap-strip -> exact
    enum member -> declared alias -> canonical member. A span that
    neither matches nor has a declared alias is REFUSED: the field stays
    unset and flows through the missing-field path (absent for optional
    fields) — never a default, never a nearest-looking rewrite. The
    first matching pattern's verdict stands whether coerced or refused;
    later patterns never second-guess it.

    Few-shot composition (2026-10-10, opt-in via
    ``register_schema(..., fewshot=True)``): when a trained head is
    wired for the schema, the pattern engine runs FIRST and the head
    fills only the fields it left UNFILLED (enum heads emit declared
    members; str/int heads emit verbatim spans or refuse — the honesty
    law lives in ``fewshot.FewShotHead.fill_missing``). A schema-supplied
    ``extractor`` callable is authoritative and never composed under.
    """
    if schema in BUILT_IN_SCHEMAS:
        from . import rules  # lazy: rules imports this module

        return rules.BUILTIN_EXTRACTORS[schema]
    entry = _REGISTERED.get(schema)
    if entry is None:
        raise KeyError(f"no registered extraction for {schema!r}")
    if entry.get("extractor") is not None:
        return entry["extractor"]
    extraction = entry["extraction"]
    # Enum coercion tables, built once per extractor from the frozen
    # spec (the author-declared members + aliases; nothing else).
    _enum_tables = {field: _enum_table(entry["spec"][field])
                    for field, _ in extraction
                    if "enum" in entry["spec"].get(field, {})}

    def _extract(text: str, _schema: str = schema,
                 _extraction: tuple = extraction,
                 _tables: dict = _enum_tables) -> dict:
        obj = {"type": _schema}
        for field, patterns in _extraction:
            table = _tables.get(field)
            for pattern in patterns:
                m = pattern.search(text)
                if m:
                    raw = m.group(1)
                    if table is None:
                        obj[field] = raw
                    else:
                        canonical = (table.get(_normalize_surface(raw))
                                     if isinstance(raw, str) else None)
                        if canonical is not None:
                            obj[field] = canonical
                        # else: refused — the field stays UNFILLED
                        # (missing-field path), never defaulted here.
                    break
        return obj

    # The few-shot head composes UNDER the pattern engine (in-memory
    # wiring only — a stale disk artifact can never resurrect silently).
    from . import fewshot as _fewshot

    _head = _fewshot.head_for(schema)
    if _head is not None:
        def _composed(text: str, _extract=_extract, _head=_head) -> dict:
            return _head.fill_missing(text, _extract(text))

        return _composed
    return _extract


def examples_for(schema: str) -> tuple:
    """Gold example records for a registered schema (copies)."""
    entry = _REGISTERED.get(schema)
    if entry is None:
        raise KeyError(f"no registered examples for {schema!r}")
    return entry["examples"]


def pairs_for(schema: str) -> tuple:
    """Few-shot (text, fields) training pairs for a registered schema.

    Built from the reserved ``"_text"`` example keys at registration;
    empty for schemas registered without them."""
    entry = _REGISTERED.get(schema)
    if entry is None:
        raise KeyError(f"no registered pairs for {schema!r}")
    return entry.get("fewshot_pairs", ())


def fewshot_report(schema: str) -> dict | None:
    """The onboarding report recorded at registration, or None."""
    entry = _REGISTERED.get(schema)
    if entry is None:
        raise KeyError(f"no registration for {schema!r}")
    return entry.get("fewshot_report")


def spec_fingerprint() -> str:
    """sha256 over the canonical JSON of the ACTIVE spec set.

    Stable for the same schema set: fields sorted, type objects named,
    registered schemas included. Recorded in data/heldout/manifest.json
    and enforced by a gate — silent schema drift fails CI, not
    production.
    """
    canonical = {}
    for name in active():
        spec = resolve(name)
        canonical[name] = {
            field: ({**rule, "type": rule["type"].__name__}
                    if "type" in rule else dict(rule))
            for field, rule in spec.items()
        }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
