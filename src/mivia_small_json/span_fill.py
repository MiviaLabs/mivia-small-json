# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Span-selection arg filling for one tool family: numeric tools, plus the
learned string-scalar family (string-param-span-fill).

Fills a matched tool's required arguments with spans selected from the
user request (selection, never generation), coerced to each slot's schema
type. Two families are handled:

- the numeric family — every required parameter is number-typed, or
  number-named when its declared type is absent (a declared string
  parameter is never numeric regardless of its name, e.g.
  ``calc_sum(a, b)``) — deterministic ladders only;
- the string-scalar family — every required parameter declared string,
  admitted UNCONDITIONALLY (deterministic-only family admission,
  2026-10-10, owner-approved): the deterministic key-delimited and
  alignment ladders fill exact-when-they-fire, and a row they cannot
  complete is handed back to the caller unhandled when no learned
  artifact is loaded — never a partial, never a new clarification (the
  caller's fillability veto keeps the row). Emission still requires
  EVERY required slot to get a verbatim copied span.

Every other tool returns ``(None, [])``
so the caller keeps the pre-existing fillability path.

Candidate spans per slot, in scorer order:

1. quoted/key-delimited patterns (``a=5``, ``"a": 3``, ``a: 5``) — 1.0
2. verbatim alignment (``a is 5``, ``5 for a``) — 0.9
3. bounded windows (a number within ``WINDOW`` chars of a slot mention,
   scored down with distance) — 0.7 .. 0.5
4. positional fallback (k-th number in the request for the k-th
   still-open slot) — 0.5

Each candidate is coerced to the slot's schema type; the first coercible
candidate wins and its span is consumed so two slots never share one
number. CPU-only, stdlib-only (``re``/``json``).

The guard stays upstream: callers run this only after the tool-match
guard has picked the tool. Unfillable slots come back in ``missing`` so
the caller emits the existing missing-arguments clarification.
"""

from __future__ import annotations

import json
import math
import re

# A number counts as "near" a slot mention inside this character radius.
WINDOW = 40

_NUMERIC_TYPES = frozenset({"number", "integer", "float"})
_NON_NUMERIC_TYPES = frozenset({"string", "boolean", "object"})
_NUMERIC_NAME_HINTS = (
    "number", "num", "count", "amount", "price", "val", "index", "coord",
)
# Alphanumeric runs of a slot name: "num_items" -> ["num", "items"],
# "num1" -> ["num1"] -> digit-trimmed ["num"] (the tool_match veto's
# measured pattern). Hint matching is WHOLE-TOKEN: the "count" inside
# "country" or "account" is a substring, never a hint (spanfix wave
# bug 2, 2026-10-10 — "Look up the country France in 2024" used to fill
# {"country": 2024}).
_SLOT_NAME_TOKEN = re.compile(r"[^0-9a-z]+")


def _slot_name_tokens(name: str) -> list[str]:
    return [t.rstrip("0123456789")
            for t in _SLOT_NAME_TOKEN.split(str(name).lower()) if t]


# Leading/trailing currency markers stripped from a captured numeric
# token before coercion ("$12" -> 12, "US$1,000" -> 1000, "12," -> 12);
# the mid-token commas of a comma-grouped number are never touched
# (spanfix wave bug 1, 2026-10-10). Longest marks first so multi-char
# codes ("us$") strip before their symbol suffix ("$").
_CURRENCY_MARKS = (
    "us$", "ca$", "a$", "nz$", "hk$",
    "usd", "eur", "gbp", "jpy", "chf", "cad", "aud", "inr", "cny",
    "$", "€", "£", "¥", "₹", ",",
)


def _strip_currency_marks(text: str) -> str:
    low = text.lower()
    changed = True
    while changed:
        changed = False
        for mark in _CURRENCY_MARKS:
            if low.startswith(mark):
                text = text[len(mark):]
                low = low[len(mark):]
                changed = True
                break
            if low.endswith(mark):
                text = text[:len(text) - len(mark)]
                low = low[:len(low) - len(mark)]
                changed = True
                break
    return text
# Thousands groups stay one number ("1,000" is not 1 and 000);
# a minus is a sign only when the token itself starts with it
# ("item-3" yields 3, not -3 — the char before the digits decides;
# review P1, 2026-10-10).
_NUM_RE = re.compile(r"(?<![\w.,])(?:-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
                     r"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?!(?:[.,]\d))")
# Number token for the span ladders: comma-grouped first so `1,000` is
# captured whole instead of stopping at the comma (B6/N10, 2026-10-10).
_NUM_TOKEN = (r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?"
              r"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
# Coercible comma-grouped forms: `1,000` -> 1000, `12,345,678.90`
# -> 12345678.9; malformed groups (`1,23`, `12,34,56`) stay uncoercible.
_COMMA_INT_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+")
_COMMA_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_BOOL_TRUE = frozenset({"true", "yes", "1"})
_BOOL_FALSE = frozenset({"false", "no", "0"})


def _bare(tool: dict) -> dict:
    if isinstance(tool.get("function"), dict):
        return tool["function"]
    return tool


def _slot_type(tool: dict, slot: str) -> str:
    props = _bare(tool).get("parameters", {})
    props = props.get("properties", {}) if isinstance(props, dict) else {}
    spec = props.get(slot, {})
    return str(spec.get("type", "")) if isinstance(spec, dict) else ""


def _slot_stype(tool: dict, slot: str) -> str:
    """Declared slot type, or ``number`` when undeclared.

    ``fill_arguments`` only runs inside the numeric family, where
    ``family_for`` admits an undeclared slot by name hint — a
    name-hinted slot is a number, not a JSON string (N11, 2026-10-10:
    `calc num1 3 num2 4` filled ``"3"``/``"4"`` as strings).
    """
    stype = _slot_type(tool, slot)
    return "number" if stype == "" else stype


def family_for(tool: dict) -> str | None:
    """``"numeric"`` when every required parameter is numeric, else None.

    The string-scalar family (string-param-span-fill):
    when EVERY required parameter is declared ``string`` (enum-constrained
    strings included), the family is admitted UNCONDITIONALLY — the
    deterministic-only family admission (2026-10-10, owner-approved; see
    the spec's amendment section). The ladders fill exact-when-they-fire
    and ``_fill_string_family`` hands any row they cannot complete back
    to the caller unhandled when no learned artifact is loaded, so the
    no-artifact serving path keeps today's decision-layer behavior
    (fillability veto → clarification or the ``{}`` honest minimum) and
    emission still requires every required slot to carry a verbatim
    copied span. Mixed required sets (string plus numeric/boolean/object)
    and boolean-required tools stay UNADMITTED: they keep today's
    fillability-veto path.
    """
    if not isinstance(tool, dict):
        return None
    params = _bare(tool).get("parameters")
    if not isinstance(params, dict):
        return None
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = list(required) if isinstance(required, list) else []
    if not required:
        return None
    all_string = True
    saw_string = False
    for slot in required:
        spec = props.get(slot, {})
        stype = str(spec.get("type", "")) if isinstance(spec, dict) else ""
        if stype in _NUMERIC_TYPES:
            all_string = False
            continue
        if stype in _NON_NUMERIC_TYPES:
            # Declared string/boolean/object: name hints never apply
            # ("country" is not "count", "festival" is not "val").
            if stype != "string":
                return None
            saw_string = True
            continue
        # Undeclared type: the numeric name-hint rule. A hint hit makes the
        # slot numeric-shaped (and the family mixed), so the string family
        # cannot admit the tool either. Hints match WHOLE tokens of the
        # name (spanfix wave bug 2, 2026-10-10): the "count" substring
        # inside "country"/"account" never made them numeric.
        name = str(slot).lower()
        if any(hint in _slot_name_tokens(name)
               for hint in _NUMERIC_NAME_HINTS):
            all_string = False
            continue
        return None
    if saw_string and all_string:
        # String-scalar family (every required slot declared string):
        # admitted unconditionally (deterministic-only family admission,
        # 2026-10-10, owner-approved). The ladders are exact when they
        # fire; _fill_string_family refuses to emit anything unless every
        # required slot carries a verbatim copied span, and hands rows it
        # cannot complete back to the caller's decision-layer path when
        # the learned artifact is absent.
        return "string"
    if saw_string:
        # Mixed required set (string plus numeric/hinted): unadmitted this
        # arm — today's veto path owns it.
        return None
    return "numeric"


# A quoted value terminates ONLY at a boundary (whitespace, end, or
# punctuation): `'O'Brien'` is not the value `O` — the closing quote at
# the inner apostrophe is followed by a letter, so the quoted alternative
# refuses and the unquoted run takes the whole `'O'Brien'` (spanfix wave
# bug 7, 2026-10-10).
_QUOTED_VAL = (r"\"[^\"]*\"(?=$|[\s.,;:!?)\]}])"
               r"|'[^']*'(?=$|[\s.,;:!?)\]}])")
# The unquoted run stops at a comma IMMEDIATELY followed by another
# slot-assignment: `service=nginx,host=web-01` is two assignments, not
# one value `nginx,host=web-01` (spanfix wave bug 6, 2026-10-10).
_ASSIGN_COMMA = re.compile(r",\s*[A-Za-z_][A-Za-z0-9_\-]*\s*[:=]")
_RUN_VAL = r"(?:(?!" + _ASSIGN_COMMA.pattern + r")\S)+"


def _key_delimited_spans(request: str, slot: str) -> list[tuple[int, int, str]]:
    """``a=5``, ``a: 5``, ``"a": 3`` — (start, end, raw value text)."""
    name = re.escape(slot)
    pat = re.compile(
        r"(?:[\"']" + name + r"[\"']|\b" + name + r"\b)\s*[:=]\s*"
        r"(?P<val>" + _QUOTED_VAL + r"|" + _NUM_TOKEN + r"|" + _RUN_VAL + r")",
        re.IGNORECASE,
    )
    return [
        (m.start("val"), m.end("val"), m.group("val"))
        for m in pat.finditer(request)
    ]


def _alignment_spans(request: str, slot: str) -> list[tuple[int, int, str]]:
    """``a is 5``, ``5 for a`` — (start, end, raw value text)."""
    name = re.escape(slot)
    spans = []
    for m in re.finditer(
        r"\b" + name + r"\b\s+(?:is|=)\s*"
        r"(?P<val>" + _NUM_TOKEN + r")",
        request, re.IGNORECASE,
    ):
        spans.append((m.start("val"), m.end("val"), m.group("val")))
    for m in re.finditer(
        r"(?P<val>" + _NUM_TOKEN + r")\s+(?:for|of)\s+\b"
        + name + r"\b",
        request, re.IGNORECASE,
    ):
        spans.append((m.start("val"), m.end("val"), m.group("val")))
    return spans


def _window_spans(
    request: str, slot: str, numbers: list[re.Match]
) -> list[tuple[float, int, int, str]]:
    """Numbers near a slot mention: (score, start, end, text)."""
    mentions = [
        m.start()
        for m in re.finditer(r"\b" + re.escape(slot) + r"\b",
                             request, re.IGNORECASE)
    ]
    out = []
    for nm in numbers:
        dist = min((abs(nm.start() - pos) for pos in mentions), default=None)
        if dist is not None and dist <= WINDOW:
            out.append((0.7 - 0.2 * dist / WINDOW,
                        nm.start(), nm.end(), nm.group(0)))
    return out


def coerce(raw: str, stype: str):
    """Schema-type coercion of a verbatim span; None when not coercible.

    Comma-grouped numbers coerce (`1,000` -> 1000, `12,345,678.90`
    -> 12345678.9 — B6/N10, 2026-10-10) and non-finite floats
    (`1e309` -> inf, `NaN`, `inf`) are refused so no argument object can
    carry a value ``arguments_json`` cannot serialize (B7, 2026-10-10).
    Currency markers glued to the digits (`$12`, `US$1,000`, `12,`)
    strip before the numeric ladders; only a token that STILL fails
    coercion counts as present-but-invalid (spanfix wave bug 1,
    2026-10-10)."""
    text = raw.strip().strip("\"'")
    if stype == "integer":
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if _COMMA_INT_RE.fullmatch(text):
            return int(text.replace(",", ""))
        stripped = _strip_currency_marks(text)
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        if _COMMA_INT_RE.fullmatch(stripped):
            return int(stripped.replace(",", ""))
        return None
    if stype in ("number", "float"):
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if _COMMA_INT_RE.fullmatch(text):
            return int(text.replace(",", ""))
        if _COMMA_NUM_RE.fullmatch(text):
            return float(text.replace(",", ""))
        try:
            value = float(text)
        except ValueError:
            text = _strip_currency_marks(text)
            if re.fullmatch(r"-?\d+", text):
                return int(text)
            if _COMMA_INT_RE.fullmatch(text):
                return int(text.replace(",", ""))
            if _COMMA_NUM_RE.fullmatch(text):
                return float(text.replace(",", ""))
            try:
                value = float(text)
            except ValueError:
                return None
        return value if math.isfinite(value) else None
    if stype == "boolean":
        low = text.lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        return None
    return text


def fill_arguments(
    tool: dict, request: str
) -> tuple[dict | None, list[str]]:
    """Fill required slots with verbatim request spans.

    Returns ``(filled, missing)``; ``(None, [])`` when the tool is not in
    a handled family (or its shape is malformed). Each filled value is
    selected from a verbatim request span and then coerced to its schema
    type, so values are JSON-native (int/float/str/bool) and the character
    form may differ after coercion (``"05"`` -> ``5``).

    The numeric family keeps the deterministic ladders as the sole
    authority (bar B6: byte-identical to the pre-arm bytes). The
    string-scalar family — admitted unconditionally (deterministic-only
    family admission, 2026-10-10, owner-approved) — runs the same
    exact-when-it-fires ladder pass first and hands what they leave open
    to the learned scorer (``_fill_string_family``); with no extractor a
    row the ladders cannot complete comes back unhandled so the caller
    keeps today's decision-layer path.
    """
    family = family_for(tool)
    if family == "string":
        return _fill_string_family(tool, request)
    return _fill_numeric(tool, request)


def _fill_numeric(
    tool: dict, request: str
) -> tuple[dict | None, list[str]]:
    """The numeric family's deterministic ladder fill (verbatim move of the
    pre-arm ``fill_arguments`` body; bar B6 keeps it byte-identical)."""
    if family_for(tool) != "numeric":
        return None, []
    try:
        bare = _bare(tool)
        required = list(bare["parameters"]["required"])
        request = request if isinstance(request, str) else ""
        numbers = list(_NUM_RE.finditer(request))
    except Exception:
        # Malformed tool dict past the family gate (e.g. mutated between
        # calls): unhandled, so the caller keeps its fillability path.
        return None, []

    filled: dict = {}
    consumed: list[tuple[int, int]] = []
    open_slots = list(required)

    def taken(s: int, e: int) -> bool:
        return any(s < ce and cs < e for cs, ce in consumed)

    # Pass 1: slot-specific evidence (key-delimited, alignment, windows).
    deferred: list[str] = []
    for slot in open_slots:
        stype = _slot_stype(tool, slot)
        # Own evidence first (key-delimited 1.0, alignment 0.9), then the
        # window ladder (0.7..0.5) — the same score order the combined
        # sort produced, split so an unusable OWN value can be told apart
        # from a missing one.
        own: list[tuple[float, int, int, str]] = (
            [(1.0, s, e, t) for s, e, t in _key_delimited_spans(request, slot)]
            + [(0.9, s, e, t) for s, e, t in _alignment_spans(request, slot)]
        )
        own.sort(key=lambda c: -c[0])
        hit = False
        for _, s, e, raw in own:
            if taken(s, e):
                continue
            val = coerce(raw, stype)
            if val is None:
                continue
            filled[slot] = val
            consumed.append((s, e))
            hit = True
            break
        if not hit:
            # FILLED-BUT-INVALID (residfix, 2026-10-10): an unconsumed own
            # key-delimited/alignment value that fails coercion (e.g.
            # `amount=1e309` -> inf, B7) means the slot's value is PRESENT
            # but unusable. The slot must not fall through to the window
            # ladder or the positional fallback — either would donate a
            # NEIGHBOR slot's number into it (fee's 2 emitted as amount).
            # It stays unfilled -> missing -> the clarification path; a
            # later own mention that coerces still rescues the slot.
            if any(not taken(s, e) and coerce(raw, stype) is None
                   for _, s, e, raw in own):
                continue
            windows = sorted(_window_spans(request, slot, numbers),
                             key=lambda c: -c[0])
            for _, s, e, raw in windows:
                if taken(s, e):
                    continue
                val = coerce(raw, stype)
                if val is None:
                    continue
                filled[slot] = val
                consumed.append((s, e))
                hit = True
                break
        if not hit:
            deferred.append(slot)

    # Pass 2: positional fallback over the still-open slots in order.
    # It only runs when every open slot can take one: with fewer numbers
    # than slots the assignment would be a guess, not a selection, so the
    # shortfall stays missing (the clarification path).
    free_numbers = [nm for nm in numbers
                    if not taken(nm.start(), nm.end())]
    if len(free_numbers) >= len(deferred):
        for slot, nm in zip(deferred, free_numbers):
            val = coerce(nm.group(0), _slot_stype(tool, slot))
            if val is None:
                continue
            filled[slot] = val
            consumed.append((nm.start(), nm.end()))

    missing = [s for s in required if s not in filled]
    return filled, missing


def _alignment_spans_str(request: str, slot: str) -> list[tuple[int, int, str]]:
    """``slot is <value>`` / ``<value> for slot`` with quoted or bareword
    values — the string analogue of the numeric alignment ladder.
    (start, end, raw value text).

    The value of ``for|of <slot>`` FOLLOWS the slot name ("show files for
    path /tmp" means path=/tmp); the before-pattern is the FALLBACK for
    when the after-side yields nothing, and it is appended after the
    after-side spans so the stable score sort prefers the after-side
    (spanfix wave bug 4, 2026-10-10)."""
    name = re.escape(slot)
    spans = []
    for m in re.finditer(
        r"\b" + name + r"\b\s+(?:is|=)\s*"
        r"(?P<val>" + _QUOTED_VAL + r"|[^\s,;!?]+)",
        request, re.IGNORECASE,
    ):
        spans.append((m.start("val"), m.end("val"), m.group("val")))
    for m in re.finditer(
        r"\b(?:for|of)\s+" + name + r"\s*"
        r"(?P<val>" + _QUOTED_VAL + r"|[^\s,;!?]+)",
        request, re.IGNORECASE,
    ):
        spans.append((m.start("val"), m.end("val"), m.group("val")))
    for m in re.finditer(
        r"(?P<val>\"[^\"]*\"(?=$|[\s.,;:!?)\]}])"
        r"|'[^']*'(?=$|[\s.,;:!?)\]}])|[^\s,;!?]+)\s+(?:for|of)\s+\b"
        + name + r"\b",
        request, re.IGNORECASE,
    ):
        spans.append((m.start("val"), m.end("val"), m.group("val")))
    return spans


def _strict_string_coerce(raw: str, prop: dict):
    """Strict coercion for a string-scalar slot: an enum-constrained slot
    must match a DECLARED enum member (the declared form is emitted, never
    the request's casing); any other string passes through as the verbatim
    span trimmed of wrapping quotes/punctuation only. Absent (empty) and
    non-string outcomes are refused — absent stays absent."""
    value = _learned_coerce(raw, prop)
    if not isinstance(value, str) or not value:
        return None
    return value


def _fill_string_family(
    tool: dict, request: str
) -> tuple[dict | None, list[str]]:
    """The string-scalar family's fill: deterministic ladders keep first
    claim (they are exact when they fire), the learned scorer fills only
    what they leave open. Admitted unconditionally since the
    deterministic-only family admission (2026-10-10, owner-approved), so
    the no-artifact serving path DOES enter here — but a row the ladders
    cannot complete is handed back UNHANDLED ((None, [])) when no
    extractor loads: never a partial, never a new clarification; the
    caller's fillability-veto path keeps the row exactly as before the
    admission.

    The numeric window and positional ladders are deliberately absent:
    a number near a slot mention is not a string value, and donating one
    would fabricate. The FILLED-BUT-INVALID rule carries over: an own
    key-delimited/alignment value that fails strict coercion (enum
    mismatch, empty trim) leaves the slot missing rather than letting the
    learned scorer donate a different span into it.
    """
    try:
        bare = _bare(tool)
        params = bare["parameters"]
        required = list(params["required"])
        props = params.get("properties")
        props = props if isinstance(props, dict) else {}
        request = request if isinstance(request, str) else ""
    except Exception:
        # Malformed tool dict past the family gate (e.g. mutated between
        # calls): unhandled, so the caller keeps its fillability path.
        return None, []

    filled: dict = {}
    consumed: list[tuple[int, int]] = []
    # FILLED-BUT-INVALID carryover: a slot whose OWN key-delimited/alignment
    # evidence was present but failed strict coercion (enum mismatch, empty
    # trim) is present-but-unusable — the learned scorer must not donate a
    # DIFFERENT span into it (the numeric family's rule; review 2026-10-10).
    blocked: list[str] = []

    def taken(s: int, e: int) -> bool:
        return any(s < ce and cs < e for cs, ce in consumed)

    for slot in required:
        prop = props.get(slot, {})
        if not isinstance(prop, dict):
            prop = {}
        own: list[tuple[float, int, int, str]] = (
            [(1.0, s, e, t) for s, e, t in _key_delimited_spans(request, slot)]
            + [(0.9, s, e, t)
               for s, e, t in _alignment_spans_str(request, slot)]
        )
        own.sort(key=lambda c: -c[0])
        hit = False
        for _, s, e, raw in own:
            if taken(s, e):
                continue
            val = _strict_string_coerce(raw, prop)
            if val is None:
                continue
            filled[slot] = val
            consumed.append((s, e))
            hit = True
            break
        if not hit and any(
                not taken(s, e) and _strict_string_coerce(raw, prop) is None
                for _, s, e, raw in own):
            blocked.append(slot)
            continue

    open_slots = [s for s in required
                  if s not in filled and s not in blocked]
    if [s for s in required if s not in filled] \
            and load_spanfill_extractor() is None:
        # Deterministic-only admission (2026-10-10, owner-approved): the
        # ladders left required slots open (open or present-but-invalid)
        # and there is no learned leg. NOT a partial — the row is handed
        # back unhandled so the caller keeps today's decision-layer path
        # (fillability veto -> clarification or the {} honest minimum),
        # byte-identical to the pre-admission no-artifact world. Emission
        # still requires EVERY required slot to carry a verbatim copied
        # span.
        return None, []
    if open_slots:
        lfilled, _ = learned_fill(tool, request, reserved=consumed,
                                  only=open_slots)
        if lfilled:
            for slot, val in lfilled.items():
                # Span exclusivity against the ladders (the learned decode
                # already respects ``reserved``; re-check for safety).
                if slot in open_slots and slot not in filled:
                    filled[slot] = val

    missing = [s for s in required if s not in filled]
    return filled, missing


def arguments_json(filled: dict) -> str:
    """Compact JSON string for a filled slot dict (strict-validation ready).

    ``allow_nan=False``: a scanned float like 1e309 must not serialize to
    bare ``Infinity`` — invalid JSON that the strict validator would
    reject downstream, but only after the call executed (review, 2026-10-10).
    """
    return json.dumps(filled, separators=(",", ":"), allow_nan=False)


# ---------------------------------------------------------------------------
# Arm: learned value-span extraction for string-scalar parameters
# (string-param-span-fill). The numeric family above
# stays authoritative for numeric tools. The round-1 hashed-feature pair
# scorer was TRAINED (scripts/train_spanfill.py) and measured FAIL on its
# frozen reserve; its decode scaffolding (sidecar-refusing loader,
# meta-bound floors, greedy span decode, strict coercion) is this arm's
# decode path and stays INERT without a promoted artifact:
# load_spanfill_extractor() returns None and learned_fill() returns
# (None, []), so serving is byte-identical to the pre-arm bytes (bar B6).
# ---------------------------------------------------------------------------

_SPANFILL_TOKEN_RE = re.compile(r"\w+")
_SPANFILL_EMIT_FLOOR = 0.6
_SPANFILL_EXTEND_FLOOR = 0.15
# Per-slot top-K candidate seeds in the decode (INCREMENT 3, registered
# 2026-10-10, string-param-span-fill): how many
# second choices a colliding slot may fall back to. K = 3 covers the
# measured greedy-argmax collision texture (the loser's correct value is
# its 1st or 2nd candidate in every dumped case, .scratch/incr2/
# b2-miss-texture.json) with one spare; larger K approaches donating
# weak third/fourth choices into slots — a fabrication risk the B1
# invented ceiling bounds — for no measured return. Registered BEFORE
# any increment-3 measurement; never tuned against any eval. NOTE: this
# is a decode constant, NOT a floor — emit/extend stay meta-bound and
# travel with the artifact.
_SPANFILL_TOP_K = 3

_SPANFILL_CLF = None
_SPANFILL_LOADED = False
_SPANFILL_LOCK = None


def configure_spanfill_floors(emit: float, extend: float) -> None:
    """Override the decode thresholds (the trainer's dev tuning and tests);
    production uses the committed constants."""
    global _SPANFILL_EMIT_FLOOR, _SPANFILL_EXTEND_FLOOR
    _SPANFILL_EMIT_FLOOR = float(emit)
    _SPANFILL_EXTEND_FLOOR = float(extend)


def _spanfill_lock():
    global _SPANFILL_LOCK
    if _SPANFILL_LOCK is None:
        import threading

        _SPANFILL_LOCK = threading.Lock()
    return _SPANFILL_LOCK


def load_spanfill_extractor():
    """The trained pair scorer, or None (sidecar-verified, never raises).

    Two artifact lineages, tried in order:
    1. THIS arm (string-param-span-fill): the int8 ONNX
       span-pair scorer ``models/spanfill-extractor-int8.onnx`` with its
       sha256 sidecar, tokenizer sidecar and decode-meta trainmeta — the
       meta-bound decode floors travel with the artifact (the
       register-gate-meta pattern), and the meta's ``artifact_sha256`` must
       name the exact bytes loaded (rule 10: instruments/artifacts name
       their bytes).
    2. The round-1 legacy skops pair classifier (``spanfill-value``),
       which was demoted and never promoted — kept so a legacy promotion
       path still loads honestly.

    A negative load is cached for the process (same trade as the guard):
    with NO artifact the serving path is byte-identical to the pre-arm
    bytes (bar B6)."""
    global _SPANFILL_CLF, _SPANFILL_LOADED
    if not _SPANFILL_LOADED:
        with _spanfill_lock():
            if not _SPANFILL_LOADED:
                _SPANFILL_LOADED = True
                clf = _load_encoder_extractor()
                if clf is None:
                    clf = _load_skops_extractor()
                _SPANFILL_CLF = clf
    return _SPANFILL_CLF


def _load_skops_extractor():
    """The round-1 legacy skops loader (staged verbatim; never raises)."""
    try:
        import json
        from pathlib import Path

        from .extract_learned import _load_skops_verified

        root = Path(__file__).resolve().parents[2]
        artifact = root / "models" / "spanfill-extractor.skops"
        clf = _load_skops_verified(artifact,
                                   "spanfill extractor")
        if clf is not None:
            meta = json.loads(
                (root / "models"
                 / "spanfill-extractor-trainmeta.json")
                .read_text())
            configure_spanfill_floors(meta["emit_floor"],
                                      meta["extend_floor"])
            return clf
    except Exception:
        pass
    return None


def _load_encoder_extractor():
    """The int8 ONNX span-pair scorer, or None. Sidecar-verified
    (artifact sha256 + the meta's own artifact_sha256 pin), never raises;
    any problem degrades to None so the caller keeps today's path."""
    try:
        import json
        from pathlib import Path

        from . import integrity

        root = Path(__file__).resolve().parents[2]
        artifact = root / "models" / "spanfill-extractor-int8.onnx"
        tokenizer = root / "models" / "spanfill-extractor-tokenizer.json"
        meta_path = root / "models" / "spanfill-extractor-trainmeta.json"
        if not (artifact.is_file() and tokenizer.is_file()
                and meta_path.is_file()):
            return None
        sidecar = artifact.parent / (artifact.name + ".sha256")
        expected, problem = integrity.expected_digest(sidecar)
        # problem is "" on a VALID sidecar (integrity.expected_digest's
        # contract; the sibling `verified()` gates on truthiness too).
        # Gating on `is not None` refused EVERY sidecar, valid ones
        # included — the loader could never load any artifact (masked by
        # the scaffold tests, which inject a mock and never exercise this
        # path). Fixed 2026-10-10, string-param-span-fill wave B3; the
        # refusal contract for missing/malformed sidecars is unchanged.
        if problem:  # "missing" or "malformed"
            return None
        digest = integrity.sha256_file(artifact)
        if not integrity.digests_match(digest, expected):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("artifact_sha256", "")) != digest:
            # The meta must name the exact bytes it tunes (rule 10); a
            # floors/bytes mismatch refuses the load rather than serving
            # an untuned operating point.
            return None
        import importlib

        ort = importlib.import_module("onnxruntime")  # stdlib-only module
        from tokenizers import Tokenizer  # lazy: only when bytes exist

        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.intra_op_num_threads = 2
        sess = ort.InferenceSession(
            str(artifact), so, providers=["CPUExecutionProvider"])
        tok = Tokenizer.from_file(str(tokenizer))
        scorer = _EncoderPairScorer(
            sess, tok,
            max_len=int(meta.get("max_len", 256)))
        configure_spanfill_floors(float(meta["emit_floor"]),
                                  float(meta["extend_floor"]))
        return scorer
    except Exception:
        return None


def _encoding_sequence_ids(enc, limit: int) -> list:
    """Per-token sequence membership from the pair Encoding, truncated.

    The pair encodes the request as sequence 0 and the slot rendering as
    sequence 1; special tokens carry ``None``. The tokenizers library
    exposes ``sequence_ids`` as a method on the ``Encoding`` but some
    wrappers (and the test fakes) carry a plain list — both forms are
    accepted. ``type_ids`` is NOT membership: on the XLM-R-family
    tokenizer it is ALL ZERO, so gating on it pooled sequence-1 (slot)
    logits onto request tokens whose restarted offsets overlap them —
    the train/serve skew of spanfix wave bug 9 (2026-10-10). The trainer
    reads the same membership (``scripts/train_stringparam_scorer.py``,
    ``CandidateScorer.pair_probs`` → ``enc.sequence_ids(0)`` and the
    ``sequence_ids()[i] == 0`` label filter); this runtime helper is the
    mirror. An Encoding with no ``sequence_ids`` attribute at all degrades
    to ``[]`` — an honest abstain — never to ``type_ids``: on XLM-R-family
    pair encodings type_ids are all-zero, and gating on them is exactly the
    bug-9 skew this helper exists to close (review 2026-10-10)."""
    seq = getattr(enc, "sequence_ids", None)
    if seq is None:
        return []
    if callable(seq):
        try:
            seq = seq()
        except TypeError:
            return []
    try:
        return list(seq)[:limit]
    except TypeError:
        return []


class _EncoderPairScorer:
    """ONNX span-pair scorer: one forward pass per (request, slot) pair
    yields per-request-token span logits; ``pair_probs`` aligns them to
    the request's ``\\w+`` tokens for the greedy decode in
    ``learned_fill``. CPU-only; the exported graph's contract is
    (input_ids, attention_mask, token_type_ids) -> per-token logits."""

    def __init__(self, sess, tokenizer, max_len: int = 256):
        self.sess = sess
        self.tok = tokenizer
        self.max_len = max_len
        self.inputs = {i.name for i in sess.get_inputs()}

    @staticmethod
    def slot_text(param: str, prop: dict) -> str:
        """The pair's sentence A: parameter identity, description, enum."""
        parts = [str(param)]
        desc = str(prop.get("description", "")).strip()
        if desc:
            parts.append(desc)
        enum = prop.get("enum") or []
        if enum:
            parts.append("one of: " + ", ".join(str(v) for v in enum))
        typ = str(prop.get("type", "string"))
        parts.append(f"type: {typ}")
        return " | ".join(parts)

    def pair_probs(self, request: str, param: str, prop: dict):
        """Sigmoid span probability per ``\\w+`` token of ``request``."""
        import math

        enc = self.tok.encode(request, pair=self.slot_text(param, prop))
        ids = enc.ids[:self.max_len]
        types = enc.type_ids[:self.max_len]
        mask = enc.attention_mask[:self.max_len]
        offsets = enc.offsets[:self.max_len]
        feed = {"input_ids": [ids],
                "attention_mask": [mask],
                "token_type_ids": [types]}
        feed = {k: v for k, v in feed.items() if k in self.inputs}
        logits = self.sess.run(None, feed)[0][0]  # (T,) or (T,1)
        probs = []
        # Sequence membership, not type_ids: the pair carries the request
        # as sequence 0 and the slot rendering as sequence 1, and only
        # sequence-0 tokens are span candidates — the trainer filters
        # ``sequence_ids == 0`` at both its label build and its candidate
        # scorer; the runtime used to gate on ``type_ids != 0``, which is
        # all-zero on this tokenizer, so slot tokens pooled onto request
        # tokens (offsets restart at 0 and overlap) — spanfix wave bug 9
        # (2026-10-10), the train/serve skew.
        seq_ids = _encoding_sequence_ids(enc, self.max_len)
        for i, (s, e) in enumerate(offsets):
            if i >= len(logits):
                break
            lg = logits[i]
            try:
                lg = lg[0]  # (T,1) head shape
            except (TypeError, IndexError):
                pass
            # a token counts as value-candidate when ANY of its aligned
            # subtokens does — sequence 0 only (see seq_ids above).
            sid = seq_ids[i] if i < len(seq_ids) else None
            if sid != 0 or e <= s:
                continue
            probs.append((s, e, 1.0 / (1.0 + math.exp(-float(lg)))))
        # max-pool subtoken probs onto the request's \w+ tokens, by
        # OVERLAP (registered increment 2026-10-10, whitespace-bounded
        # span extension): a subtoken straddling a glue boundary (XLM-R
        # renders `app-07` as `app` + `-07`) IS the glued token's
        # evidence. The old strict-containment test (`s >= m.start()`)
        # dropped it, the token read 0.0, and the extend floor stopped
        # every glued value at the split — the measured mechanism of the
        # wave-B3 falsification (137/137 dev truncations;
        # .scratch/incr2/probe-dev-glued.json).
        out: list[float] = []
        matches = list(_SPANFILL_TOKEN_RE.finditer(request))
        for m in matches:
            best = 0.0
            for s, e, p in probs:
                if s < m.end() and e > m.start() and p > best:
                    best = p
            out.append(best)
        return out


def _pair_features(tokens: list[str], idx: int, param: str, prop: dict):
    """Token x parameter features — MUST mirror the trainer exactly
    (scripts/train_spanfill.py:pair_features)."""
    t = tokens[idx]
    low = t.lower()
    n = len(tokens)
    name_low = param.lower()
    desc = str(prop.get("description", "")).lower()
    enum = [str(v).lower() for v in prop.get("enum", [])]
    sh = "".join("X" if c.isupper() else "x" if c.islower()
                 else "d" if c.isdigit() else c for c in t)[:8]
    feats = {
        "tok=" + low,
        "shape=" + sh,
        f"len={min(len(t), 12)}",
        "posbucket=" + str(idx * 10 // max(n, 1)),
        "param=" + name_low,
        "ptype=" + str(prop.get("type", "string")),
    }
    if prop.get("enum"):
        feats.add("has_enum")
        if low in enum:
            feats.add("tok_in_enum")
    if low == name_low or name_low.endswith(low) or low.endswith(name_low):
        feats.add("tok_matches_param_name")
    for dtok in re.findall(r"\w+", desc)[:10]:
        if dtok == low:
            feats.add("tok_in_param_desc")
            break
    if any(low == e for e in enum):
        feats.add("tok_equals_enum_value")
    if any(c.isdigit() for c in t) and prop.get("type") in ("integer", "number"):
        feats.add("digit_for_numeric_param")
    if t[:1].isupper() and prop.get("type") == "string":
        feats.add("capital_for_string_param")
    for off, tag in ((1, "next"), (2, "next2"), (1, "prev"), (2, "prev2")):
        j = idx + off if tag.startswith("next") else idx - off
        if 0 <= j < n:
            feats.add(f"{tag}=" + tokens[j].lower()[:16])
    name_positions = [j for j, tk in enumerate(tokens)
                      if tk.lower() == name_low]
    if name_positions:
        near = min(abs(idx - j) for j in name_positions)
        feats.add(f"near_param_name={min(near, 5)}")
    for i in range(len(low) - 2):
        feats.add("c=" + low[i:i + 3])
    return sorted(feats)


def _wrap_strip(raw: str) -> str:
    """Wrap-strip of a captured string value: LEADING strip removes only
    quote characters (a leading dot is content — ``.env`` is a filename,
    not noise); trailing strip keeps the full punctuation set. The single
    wrap-strip for string values AND the enum comparison (spanfix wave
    bugs 5 and 8, 2026-10-10). The trailing set mirrors ``_QUOTED_VAL``'s
    boundary class ``[\\s.,;:!?)\\}]``: the punctuation that legitimately
    terminates a value in this module's grammar is wrapper noise on the
    captured span, so ``prod-eu-1!`` matches the enum member ``prod-eu-1``
    (spanfix wave bug 8's trailing-punctuation case)."""
    return raw.lstrip("\"'").rstrip(" .,;:!?)]}\"'")


def _learned_coerce(raw: str, prop: dict):
    """Strict coercion: enum values must match (no fabrication), numerics
    must parse; strings pass through stripped."""
    t = prop.get("type", "string")
    if prop.get("enum"):
        stripped = _wrap_strip(raw)
        for v in prop["enum"]:
            if str(v).lower() == stripped.lower():
                return v
        return None
    if t == "integer":
        d = re.sub(r"[^\d-]", "", raw)
        return int(d) if d.lstrip("-").isdigit() else None
    if t == "number":
        d = re.sub(r"[^\d.-]", "", raw)
        try:
            return float(d)
        except ValueError:
            return None
    if t == "boolean":
        if raw.lower() in ("true", "yes"):
            return True
        if raw.lower() in ("false", "no"):
            return False
        return None
    return _wrap_strip(raw)


def _slot_token_probs(clf, request: str, tokens: list[str],
                      param: str, prop: dict) -> list[float] | None:
    """Per-request-token span probabilities for one parameter slot, from the
    loaded extractor. The encoder-backed scorer (this arm's artifact) is
    queried through ``pair_probs``; the legacy round-1 skops pair
    classifier takes the staged hashed-feature path. Returns None for an
    unsupported extractor shape (never raises)."""
    pair_probs = getattr(clf, "pair_probs", None)
    if callable(pair_probs):
        try:
            probs = pair_probs(request, param, prop)
            if probs is not None and len(probs) == len(tokens):
                return [float(p) for p in probs]
        except Exception:
            return None
        return None
    try:
        X = [_pair_features(tokens, i, param, prop) for i in range(len(tokens))]
        return list(clf.predict_proba(X)[:, 1])
    except Exception:
        return None


def _glued(request: str, left, right) -> bool:
    """True when NO whitespace sits between two adjacent token spans —
    the gap is contiguous non-whitespace glue (hyphen/dot/slash/
    underscore/...), so both tokens belong to one value (registered
    increment 2026-10-10, whitespace-bounded span extension)."""
    gap = request[left.end():right.start()]
    return bool(gap) and not any(c.isspace() for c in gap)


def learned_fill(
    tool: dict, request: str,
    reserved: list[tuple[int, int]] | None = None,
    only: list[str] | None = None,
):
    """Learned value-span filling for required parameters (the learned
    twin of ``fill_arguments``). Global (slot, span) assignment, registered
    increment 3 (2026-10-10): every missing slot contributes its top-K
    candidate seed spans (K = ``_SPANFILL_TOP_K``, seeds at or above the
    emit floor), each extended by the unchanged whitespace-bounded rule;
    the (slot, span) pairs are then assigned greedily by confidence
    descending under each-span-used-once. A span claimed by two parameters
    goes to the higher-confidence one and the loser falls back to its own
    next candidate — the increment-2 falsification's residual texture
    (greedy argmax collisions: two slots' argmax on one token, the loser
    abstained and its second-best was never consulted, e.g. `country`
    losing `pandas` to `species` while `China` sat verbatim in the
    request). A slot with no usable candidate left stays missing (abstain
    over fabricate). ``reserved`` spans (claimed by the deterministic
    ladders) are never re-claimed; ``only`` restricts scoring to the
    still-open slots. Returns (filled, missing) — (None, []) when the
    extractor is unavailable so the caller keeps its existing path."""
    clf = load_spanfill_extractor()
    if clf is None:
        return None, []
    spans = list(_SPANFILL_TOKEN_RE.finditer(request))
    tokens = [m.group(0) for m in spans]
    if not tokens:
        return None, []
    # The OpenAI envelope is unwrapped through the SAME helper family_for
    # uses: learning to read {"type": "function", "function": {...}} past
    # the envelope left the learned leg parameter-blind on wrapped tools —
    # it scored an empty required set and returned ({}, []) while the
    # ladders (which unwrap) filled nothing either (spanfix wave bug 3,
    # 2026-10-10).
    params = _bare(tool).get("parameters", {}) \
        if isinstance(tool, dict) else {}
    params = params if isinstance(params, dict) else {}
    props = params.get("properties", {})
    props = props if isinstance(props, dict) else {}
    required = params.get("required", [])
    required = list(required) if isinstance(required, list) else []
    if only is not None:
        wanted = set(only)
        required = [p for p in required if p in wanted]
    if not required:
        return {}, []
    reserved = [(int(s), int(e)) for s, e in (reserved or [])]
    per_param = {}
    for param in required:
        prop = props.get(param, {})
        if not isinstance(prop, dict):
            prop = {}
        probs = _slot_token_probs(clf, request, tokens, param, prop)
        if probs is None:
            return None, []
        per_param[param] = probs
    claimed: list[tuple[int, int]] = []
    filled: dict = {}
    # Per-slot top-K seeds (INCREMENT 3, registered 2026-10-10): the K
    # highest-confidence tokens at or above the emit floor — not just the
    # argmax. K = 3 covers the measured pairwise collision texture
    # (b2-miss-texture.json: the loser's correct value is its 1st or 2nd
    # candidate) with one spare; larger K approaches donating weak
    # third/fourth choices — a fabrication risk the B1 invented ceiling
    # bounds. The emit floor gates EVERY seed, so weak fallbacks are
    # never consulted.
    candidates = []
    for param, probs in per_param.items():
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        for i in order[:_SPANFILL_TOP_K]:
            if probs[i] >= _SPANFILL_EMIT_FLOOR:
                candidates.append((float(probs[i]), param, i))
    # Greedy (slot, span) assignment by confidence descending —
    # deterministic tie-break (slot name, then seed index).
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    for conf, param, seed in candidates:
        if param in filled:
            continue  # this slot already took its best usable span
        prop = props.get(param, {})
        lo = hi = seed
        # Whitespace-bounded span extension (registered increment
        # 2026-10-10): the value is the contiguous NON-WHITESPACE run
        # around the seed token. A GLUED neighbor token (hyphen/dot/
        # slash/underscore glue) joins while its own confidence stays at
        # or above the extend floor; whitespace terminates the span in
        # BOTH directions. The old loop extended across adjacent `\w+`
        # tokens through whitespace and stopped at glue splits — the two
        # halves of the wave-B3 falsification (truncation + over-extension).
        while lo - 1 >= 0 \
                and per_param[param][lo - 1] >= _SPANFILL_EXTEND_FLOOR \
                and _glued(request, spans[lo - 1], spans[lo]):
            lo -= 1
        while hi + 1 < len(tokens) \
                and per_param[param][hi + 1] >= _SPANFILL_EXTEND_FLOOR \
                and _glued(request, spans[hi], spans[hi + 1]):
            hi += 1
        s, e = spans[lo].start(), spans[hi].end()
        # each-span-used-once: a taken/reserved span skips THIS candidate
        # and the slot falls back to its next one — it no longer abstains
        # outright (the registered collision fix); when no candidate is
        # left the slot lands in missing below.
        if any(not (e <= cs or s >= ce) for cs, ce in claimed) or any(
                not (e <= cs or s >= ce) for cs, ce in reserved):
            continue
        value = _learned_coerce(request[s:e], prop)
        if value is None or (isinstance(value, str) and not value):
            continue  # unusable span; try the slot's next candidate
        claimed.append((s, e))
        filled[param] = value
    missing = [p for p in required if p not in filled]
    return filled, missing
