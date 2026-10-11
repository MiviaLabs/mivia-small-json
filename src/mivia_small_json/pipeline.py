# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 integration: the stable inference path (load -> classify).

Production shape:

- ``load_router`` verifies the artifact's sha256 sidecar BEFORE the
  skops load; any integrity problem (missing artifact, missing or
  malformed sidecar, digest mismatch, corrupt file) prints a warning on
  stderr and returns None. It never raises, so callers degrade to
  rules-only routing instead of crashing.
- ``classify`` NEVER raises for any str input: extraction/repair reuse
  rules.py + validate.py (repair <= 2 attempts, template escape as last
  resort), a failed model route falls back to the rules router, and an
  unexpected error still yields a best-effort object plus honest meta.
  The failure mode is a rules/template fallback, never a crash.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from . import extract_learned, integrity, rules
from .validate import repair, template, validate

load_level_extractor = extract_learned.load_level_extractor
load_component_extractor = extract_learned.load_component_extractor
load_implicit_severity_extractor = extract_learned.load_implicit_severity_extractor
_MAX_REPAIRS = 2  # retry budget (AGENTS.md rule 2), same as rules.emit

# Below this model confidence a rules signal may outrank the model.
_LOW_CONFIDENCE = 0.40
# A rules score this high means at least two strong markers agreed.
_STRONG_RULE_SCORE = 4

# Schemas whose multi-line text is ONE atomic entity, not a stream of
# separate events. Named here rather than inlined at the use site.
_ATOMIC_SCHEMAS = frozenset({
    "diff", "diagnostic", "review_comment", "tool_call",
    "issue", "pull_request", "commit", "agent_step",
})

# When2Call ask_followup: an information request (`request_for_info`) or an
# unanswerable question (`cannot_answer`). The honest outcome is a clarifying
# question, not a call and not a silent abstention. Rule-shaped on purpose:
# a question mark anywhere, or a genuine question opener. Imperative tool
# commands (`find`, `search`, `get`, bare greetings) are NOT covered — a
# command no tool serves stays `no_tool_call`. The polite-imperative heads
# (`please|tell|show|help`) were REMOVED (2026-10-10 review N1): they are
# imperative command openers, not question forms, and turned no-tool
# commands ("please book a table...") into clarifications naming an
# unmatched tool's parameter while the bare command honestly abstained.
_ASK_FOLLOWUP_OPENER = re.compile(
    r"^(can|could|would|shall|should|is|are|was|were|do|does|did"
    r"|have|has|what|which|when|where|who|whom|whose|why|how)\b",
    re.IGNORECASE,
)


def _is_ask_followup(request: str) -> bool:
    """Whether a catalogue request reads as a follow-up question.

    Narrow on purpose: only a catalogue request (never a bare tool trace),
    and only question-shaped text. Returns False for empty input.
    """
    q = (request or "").strip().strip("\"'")
    if not q:
        return False
    return "?" in q or bool(_ASK_FOLLOWUP_OPENER.match(q.strip()))

# Lazily-loaded default extractors, keyed by name (one entry per extractor).
_DEFAULT_EXTRACTORS: dict[str, object] = {}
_EXTRACTOR_LOCK = threading.Lock()


def _get_default_extractor(name: str, loader):
    """Load and cache a default extractor by name. Never raises.

    Only a successful (truthy) load is cached: a transient load
    failure's ``None`` must not poison the cache for the process
    lifetime — the next call retries the loader.
    """
    if not _DEFAULT_EXTRACTORS.get(name):
        with _EXTRACTOR_LOCK:
            if not _DEFAULT_EXTRACTORS.get(name):
                _DEFAULT_EXTRACTORS[name] = loader()
    return _DEFAULT_EXTRACTORS.get(name)


def _warn(msg: str) -> None:
    print(f"[mivia-small-json] {msg}", file=sys.stderr)


def load_router(path: Path | str):
    """Sha-verified skops load. Returns the pipeline, or None (never raises).

    The path is REQUIRED: a default bound at import time made the
    artifact location an import-order fact instead of a caller decision.
    Callers pass ``model.artifact_path(root)`` (repository constant:
    ``model.MODEL_PATH``).

    Sidecar convention: ``<artifact-stem>.sha256`` next to the artifact
    (models/router.skops <-> models/router.sha256), one hex digest line
    as written by model.main().
    """
    path = Path(path)
    sidecar = path.parent / (path.stem + ".sha256")
    try:
        if not path.is_file():
            _warn(f"model artifact missing: {path}; using rules-only routing")
            return None
        expected, problem = integrity.expected_digest(sidecar)
        if problem == "missing":
            _warn(f"sha256 sidecar missing: {sidecar}; refusing unverified load")
            return None
        if problem == "malformed":
            _warn(f"malformed sha256 sidecar: {sidecar}; refusing unverified load")
            return None
        digest = integrity.sha256_file(path)
        if not integrity.digests_match(digest, expected):
            _warn(
                f"sha256 mismatch for {path.name} "
                f"(sidecar {expected[:12]}..., actual {digest[:12]}...); "
                "using rules-only routing"
            )
            return None
        import skops.io as sio

        return sio.load(path, trusted=sio.get_untrusted_types(file=path))
    except Exception as exc:  # corrupt artifact, unreadable file, skops error
        _warn(f"router load failed ({type(exc).__name__}: {exc}); using rules-only routing")
        return None


def load_active_router():
    """The shipped router: the ONNX encoder when present, else skops.

    Routing only. Extraction, validation and repair are unchanged, so a
    caller that swaps the router gets the same never-raise contract.

    Boots the full shipped schema world first: the router is trained on
    all 9 classes, and without their registrations every non-built-in
    prediction silently degraded to a wrong builtin through the rules
    fallback. Idempotent and additive; user registrations are untouched.
    """
    from . import bootstrap

    bootstrap.register_all_schemas()

    from . import encoder

    loaded = encoder.load_encoder()
    if loaded is not None:
        return loaded
    from .model import MODEL_PATH

    return load_router(MODEL_PATH)


# --- structural decision layer ----------------------------------------------
# Long log-shaped dumps are logs by construction, but the encoder
# confidently mislabels them (2026-10-08 realworld-slices regression:
# HDFS/Apache/Proxifier/Windows/HPC dumps routed issue/text at high
# confidence). The rule is deterministic and measured before any training
# arm; the slice floors it backs are pre-registered in
# tests/test_realworld_bars.py. The fenced-error-paste variant was ALSO
# measured and FALSIFIED: real issue bodies routinely embed fenced
# tracebacks (4 issue-realworld rows match, 3 of them correctly routed
# issue before), so no fence signal separates a paste-as-message from a
# bug report containing a paste — that surface belongs to the training
# improvement arm, not the decision layer.
_LOG_DUMP_MIN_LINES = 8
_LOG_DUMP_LINE_COVERAGE = 0.9

# A line "starts with a timestamp" in any of the dialects seen in real
# dumps, or is an epoch-counter RAS line (counter host comp event
# <9+ digit epoch> seq) which has structure but no clock. The ISO
# alternative requires a LOG continuation after the seconds — millis
# (",963 "), a comma-space (Windows CBS ", Info"), whitespace, an offset
# ("+00:00"), or a Z — never a comma/period followed by a non-digit-minus
# run: a bank-statement CSV ("12:30:00,-10.50,...") is tabular text, not
# a log (2026-10-08 review P1). The RFC 3164 alternative covers syslog
# ("Jun 14 15:16:01 combo sshd[1000]: ...").
_LOG_TS_LINE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?(?:\s|[.,]\s|$)"
    r"|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s"
    r"|\[\w{3} \w{3}\s+\d{1,2} \d{2}:\d{2}:\d{2} \d{4}\]"
    r"|\[\d{1,2}\.\d{1,2} \d{2}:\d{2}:\d{2}\]"
    r"|\d+\s+\S+\s+\S+\s+\S+\s+\d{9,}\s+\d+\s"
    # BGL raw: "- <epoch> <YYYY.MM.DD-HH.MM.SS.micro> <node> ..." — the
    # leading dash + epoch + dotted-date triple is a log timestamp by any
    # reading; the stronger text class captured these dumps when the
    # router's own confidence stopped covering them (chat-ood r2, 2026-10-10).
    r"|-\s\d{9,11}\s\d{4}\.\d{2}\.\d{2}\s+\S+\s+\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+\s)"
)


def _looks_like_log_dump(text: str) -> bool:
    """Whether >= 8 non-empty lines uniformly start with a timestamp."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < _LOG_DUMP_MIN_LINES:
        return False
    hits = sum(1 for line in lines if _LOG_TS_LINE.search(line))
    return hits / len(lines) >= _LOG_DUMP_LINE_COVERAGE


_GIT_LOG_LINE = re.compile(
    r"^.{0,44}?\s+\+\d{2}:?\d{2}\s+\w+\([^)]{0,24}\):\s+\S.*$")


def _looks_like_git_log_dump(text: str) -> bool:
    """>= 6 lines that are each a timestamped conventional-commit subject.

    EXEMPTS the text from the log-structure rule (a `git log --pretty`
    paste is never structure-routed — pinned contract; its record
    quality is an improvement-arm surface). Chat-ood r2 needed the
    signature exemption because the stronger text class stopped the
    commit-confident clause from firing (2026-10-10)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    hits = sum(1 for ln in lines if _GIT_LOG_LINE.match(ln))
    return hits / len(lines) >= 0.9


def _route(router, text: str) -> tuple[str, str]:
    schema, routed_by, _ = _route_with_confidence(router, text)
    return schema, routed_by


def _router_classes(router):
    """The classifier's class labels, when the router exposes them.

    ``None`` means the router exposes no labels — including an empty set —
    so callers treat its coverage as unknown rather than as "covers
    nothing". An empty label set would otherwise drop every zero-shot
    fallback candidate and index an empty class list.
    """
    if hasattr(router, "classes"):
        classes = router.classes
    else:
        steps = getattr(router, "named_steps", None)
        if steps and "clf" in steps:
            classes = list(getattr(steps["clf"], "classes_", []))
        else:
            classes = getattr(router, "classes_", None)
    if classes is None or len(classes) == 0:
        return None
    return classes


def _route_with_confidence(router, text: str) -> tuple[str, str, float]:
    """Return (schema, routed_by, confidence); failed route falls back to rules."""
    schema, routed_by, conf, _in_domain = _route_full(router, text)
    return schema, routed_by, conf


def _route_with_gate(router, text: str) -> tuple[str, str, float, bool]:
    """_route_with_confidence plus the energy-gate verdict, from the SAME
    forward pass. Serving surfaces use this so the in-domain flag never
    costs a second encoder run (the 2026-10-08 audit measured do_route at
    2 passes; one pass is the contract)."""
    return _route_full(router, text)


def _route_full(router, text: str) -> tuple[str, str, float, bool]:
    """The routing ladder, one forward pass: (schema, routed_by, conf,
    in_domain). Failed route falls back to rules."""
    if router is None:
        # No model, no gate: in_domain is true by construction (the
        # rules verdict is the only signal and no OOD score exists).
        return rules.route(text), "rules", 1.0, True
    # ONE forward pass up front: softmax + energy gate verdict together.
    # Computed before the ladder so a ladder-side failure (never-raise
    # rules fallback) still carries the gate's verdict — the separate
    # api-side energy call this replaces used to keep it.
    probs_batch = None
    in_domain = True
    if hasattr(router, "predict_proba"):
        try:
            if hasattr(router, "predict_proba_with_energy"):
                # One forward pass yields both the softmax and the energy
                # OOD score; no second transform.
                probs_batch, energy = router.predict_proba_with_energy([text])
                threshold = getattr(router, "in_domain_energy", None)
                if threshold is not None:
                    in_domain = float(energy[0]) >= float(threshold)
            else:
                probs_batch = router.predict_proba([text])
        except Exception as exc:
            _warn(f"router failed ({type(exc).__name__}: {exc}); using rules-only routing")
            # The forward pass died before the gate produced a score:
            # in_domain is true by construction (unknown, reported as
            # in-domain rather than fabricated).
            return rules.route(text), "rules", 1.0, True
    try:
        conf = 1.0
        if hasattr(router, "predict_proba"):
            probs = probs_batch[0]
            conf = float(max(probs))
            # predict() is the argmax of predict_proba(); reuse it instead
            # of a second full feature transform.
            classes = _router_classes(router)
            if classes is not None:
                pred = str(classes[int(probs.argmax())])
            elif hasattr(router, "predict"):
                pred = str(router.predict([text])[0])
            else:
                pred = str(router(text))
        elif hasattr(router, "predict"):  # loaded skops pipeline
            pred = str(router.predict([text])[0])
        else:
            pred = str(router(text))  # plain callable
        from . import registry

        # Explicit user intent outranks both the OOD emission and the
        # fast path: a registered schema the router cannot emit whose
        # STRONG marker matches the text is what the user installed it
        # for — a confident `issue` on an invoice (or a text
        # call on chat-shaped text) must not bury it. The zero-shot
        # ladder below makes the actual decision.
        user_claimed = False
        custom_all = registry.zero_shot_registered()
        if custom_all:
            covered_all = _router_classes(router)
            if covered_all is not None:
                covered_set = set(covered_all)
                custom_all = tuple(name for name in custom_all
                                   if name not in covered_set)
            for name in custom_all:
                strong, _weak = registry.routing_for(name)
                if any(p.search(text) for p in strong):
                    user_claimed = True
                    break

        # Structural decision layer, ahead of BOTH the confident text
        # emission and the confident fast path: a uniformly-timestamped
        # dump is a log regardless of what the encoder believes — EXCEPT
        # a confident in-domain commit route (a `git log --pretty` paste
        # is timestamp-per-line commit material, 2026-10-08 review P1).
        # User-claimed schemas already outranked this via ``user_claimed``
        # above. Known recorded edge: a timestamped changelog/release list
        # with no commit markers still routes log_event here.
        if (not user_claimed and pred != "log_event"
                and "log_event" in registry.active()
                and _looks_like_log_dump(text)
                and not _looks_like_git_log_dump(text)
                and not (pred == "commit" and conf >= 0.85 and in_domain)):
            return "log_event", "structure", 1.0, in_domain

        # The router's trained non-workflow class (owner product decision
        # 2026-10-10: no "out of domain" wording — the outcome is plain
        # ``text``): a CONFIDENT non-workflow prediction emits the text
        # record before the fast path. Unconfident keeps the fallback/
        # rules path as today.
        if (not user_claimed and pred == "text"
                and "text" in registry.active()):
            if conf >= 0.85:
                return "text", "model", conf, in_domain

        # Fast path: built-in model confidence >= 0.85 AND the energy gate
        # says the text is in domain. Max-softmax alone is overconfident out
        # of domain (an invoice scored 0.997); an out-of-domain score falls
        # through to the zero-shot fallback below instead of short-circuiting.
        # the text class itself never fast-paths: the emission branch above
        # is its only intended exit (so a user-claimed schema still gets
        # the ladder).
        if (not user_claimed and pred != "text"
                and pred in registry.active()
                and conf >= 0.85 and in_domain):
            return pred, "model", conf, in_domain

        # Dynamic schema evaluation for custom registered schemas. Two
        # filters keep the fallback to schemas it can actually decide:
        # - only schemas with an explicit semantic description are
        #   zero-shot selectable (the auto-generated field list is not a
        #   description to select on);
        # - only schemas the router cannot already emit are candidates.
        #   Re-scoring a router class costs an encoder run per candidate
        #   plus a verify, and the fine-tuned router is strictly better
        #   informed about its own classes.
        # When the router exposes no class labels its coverage is unknown,
        # so every described schema stays a candidate.
        custom = registry.zero_shot_registered()
        covered = _router_classes(router)
        if custom and covered is not None:
            covered = set(covered)
            custom = tuple(name for name in custom if name not in covered)
        if custom:
            from . import tool_match

            matcher = tool_match._get_guard()
            if matcher is not None:
                custom_descs = {}
                for name in custom:
                    d = registry.description_for(name)
                    if not d.startswith(f"{name}:"):
                        d = f"{name}: {d}"
                    custom_descs[name] = d
                best_cand, norm_conf = matcher.select(text, custom_descs)
                if best_cand:
                    verified, match_score = matcher.verify(text, custom_descs[best_cand])
                    # Normally the ladder must out-score the router's own
                    # confidence to override it. For user-claimed text (a
                    # strong-marker schema matched) the router's prediction
                    # — often a confident text call — is exactly what
                    # the user overruled, so a verified match suffices.
                    if verified and (user_claimed or match_score > conf):
                        return best_cand, "model", match_score, in_domain

        if pred not in registry.active():
            # The gate ran, so its verdict travels with the fallback.
            return rules.route(text), "rules", conf, in_domain
        if conf < _LOW_CONFIDENCE:
            scores = rules.route_scores(text)
            best = max(scores.values(), default=0)
            if best >= _STRONG_RULE_SCORE:
                return max(scores, key=lambda s: scores[s]), "rules", 1.0, in_domain
            if best > 0:
                rule_pred = rules.route(text)
                if rule_pred in registry.active():
                    return rule_pred, "rules", 1.0, in_domain
        return pred, "model", conf, in_domain
    except Exception as exc:
        _warn(f"router failed ({type(exc).__name__}: {exc}); using rules-only routing")
        return rules.route(text), "rules", 1.0, in_domain


def _log_to_pool(
    text: str,
    reason: str,
    pool_inbox: Path | str | None = None,
) -> None:
    """Log ambiguous or repair-requiring record to improvement pool inbox."""
    try:
        import os
        from datetime import datetime, timezone

        if pool_inbox is None:
            root = Path(os.environ.get("MIVIA_ROOT", "."))
            inbox = root / "data" / "pool" / "inbox"
            if not inbox.is_dir():
                return
            pool_inbox = inbox / "live_stream.jsonl"
        else:
            pool_inbox = Path(pool_inbox)

        pool_inbox.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "raw": text,
            "reason": reason,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with open(pool_inbox, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Never crash production classification due to logging failure


def _cache_lookup(cache, chunk: str) -> tuple[str, str | None]:
    """Cache probe that never raises; a broken cache reads as a miss."""
    try:
        return cache.match(chunk)
    except Exception as exc:
        _warn(f"template cache match failed ({type(exc).__name__}: {exc})")
        return "", None


def _cache_probe(cache, chunk: str) -> tuple[str, str | None]:
    """READ-ONLY cache probe (``TemplateCache.probe``): never adds to the
    Drain, so a probed chunk is still added exactly once — by the slow
    path's ``match`` — instead of twice (the 2026-10-09 double-add)."""
    try:
        return cache.probe(chunk)
    except Exception as exc:
        _warn(f"template cache probe failed ({type(exc).__name__}: {exc})")
        return "", None


def _cache_learn(cache, template: str, schema: str) -> None:
    """Cache write that never raises; a broken cache drops the entry."""
    try:
        cache.learn(template, schema)
    except Exception as exc:
        _warn(f"template cache learn failed ({type(exc).__name__}: {exc})")


_ID_LIKE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,9}-\d{2,6}\b")
_DEGENERATE_RE = re.compile(r"^[^A-Za-z]*$|[.!?]{2,}$")

# Structural leading-line title (2026-10-10, reality-extract diagnostic).
# The ticket (EN/DE) and gh-prs surfaces present the instrument shape
# ``subject<blank line>body`` — the gold title IS the leading subject
# line — yet the chain let body sentences through: the rules title is
# the longest chunk (usually a body sentence) and the learned span
# decoder truncated the subject or picked a body sentence the QW-B
# prefix guard could not always veto (title-exact EN 72.5 / DE 66.4 /
# gh-prs 23.0 vs 91.7 / 94.6 / ~95 for the bare leading-line rule; the
# German lag inverts under the rule by construction — the multilingual
# MUST). When the text is one short leading line, then a blank-line
# separator, then prose, the leading line is the title. The learned
# extractor stays authoritative for genuinely line-free prose (single
# block) and for an implausibly long leading line. The 200-char bound
# is registered here on purpose.
_LEADING_LINE_TITLE_MAX_CHARS = 200
# One blank line = optional horizontal whitespace, then a line break.
# The separator between leading line and prose must contain at least
# one, so ``\n\n``, ``\r\n\r\n``, ``\n \n`` and longer runs all fire
# while a single soft-wrap ``\n`` does not.
_BLANK_LINE_RE = re.compile(r"[ \t]*\r?\n")
# The existing title-normalization contract (rules._extract_issue):
# separators and quote ticks do not belong on a title's edge.
_TITLE_EDGE_RE = " \t\r\n-:—–\"'"


def _structural_leading_line_title(text: str) -> str | None:
    """Leading line of a ``subject<blank line>body`` text, else None.

    Fires only when, after skipping leading blank lines, the first line
    is non-empty, at most ``_LEADING_LINE_TITLE_MAX_CHARS`` chars after
    trimming, carries at least one Unicode letter, and a blank-line
    separator is followed by more prose. Title-only: callers keep the
    learned body. Never raises.
    """
    try:
        if not text:
            return None
        head, sep, rest = text.lstrip().partition("\n")
        if not sep:
            return None
        title = head.rstrip("\r").strip(_TITLE_EDGE_RE)
        if not title or len(title) > _LEADING_LINE_TITLE_MAX_CHARS:
            return None
        if not re.search(r"[^\W\d_]", title, re.UNICODE):
            return None
        m = _BLANK_LINE_RE.match(rest)
        if m is None or not rest[m.end():].strip():
            return None
        return title
    except Exception:
        return None


def _learned_title_wins(rules_title: str, learned_title: str,
                        text: str = "") -> bool:
    """QW-B guard: when may the learned prose title replace the rules one?

    The rules title wins when it is substantive and the learned title is a
    strict prefix-truncation of it — or of the raw first line (the rules
    ladder strips a leading `PAY-2214:` id, so the truncation check must
    see the id too; the greedy span decoder collapses `ISS-4091: …` to
    `ISS`). Degenerate rules titles (empty, punctuation/id-only) always
    lose to the learned span. Case-insensitive on purpose.
    """
    rules_title = (rules_title or "").strip()
    learned_title = (learned_title or "").strip()
    if not learned_title:
        return False
    if not rules_title or _DEGENERATE_RE.search(rules_title) \
            or not re.search(r"[A-Za-z].*[A-Za-z]", rules_title):
        return True
    low_learned = learned_title.lower()
    for reference in (rules_title,
                      (text or "").splitlines()[0] if (text or "").strip() else ""):
        low_ref = (reference or "").strip().lower()
        if low_ref and len(low_learned) < len(low_ref) \
                and low_ref.startswith(low_learned):
            return False
    if _ID_LIKE_RE.match(learned_title):
        return False
    return True


def _classify_known_schema(
    text: str,
    schema: str,
    routed_by: str,
    conf: float = 1.0,
    level_extractor=None,
    component_extractor=None,
    implicit_severity_extractor=None,
    prose_extractor=None,
    log_pool: bool | None = None,
    pool_inbox: Path | str | None = None,
) -> tuple[dict, dict]:
    """Extract -> validate -> repair (<= 2) -> template escape, no routing.

    The schema is already decided (by the router or a template-cache
    hit); only the validation, repair, and template-fallback ladder runs.
    Never raises for any str input. Returns (obj, meta) with the same
    honest ``valid`` contract as ``classify``.
    """
    t0 = time.perf_counter()
    retries = 0
    obj: dict = {}
    try:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if schema == "tool_call":
            # A tool catalogue is only a real call when the request matches a
            # listed tool. The cross-encoder guard is the abstention decision;
            # without it the router's answer stands.
            from . import tool_match

            guard = tool_match._get_guard()
            if guard is not None:
                best_tool, score = guard.best_match(text)
                if score < tool_match.THRESHOLD:
                    _, req = tool_match.parse_catalogue_raw(text)
                    from . import registry as _registry

                    # The veto-abstain path: the guard's argmax tool matched,
                    # but its required arguments are absent from the request
                    # (the no_tool_needed construction). Ask for the missing
                    # parameters instead of silently abstaining — same 3-state
                    # contract as the ask-followup branch below.
                    veto_ok, veto_reason, veto_missing = (True, "", [])
                    if req and isinstance(best_tool, dict):
                        veto_ok, veto_reason, veto_missing = (
                            tool_match.fillability_veto(best_tool, req))
                    if (req and "tool_clarification" in _registry.active()
                            and (_is_ask_followup(req) or not veto_ok)):
                        # Ask_followup on the abstain path: a question no
                        # listed tool can serve needs a clarifying question,
                        # not a silent abstention. Same construction shape
                        # as the unfillable-catalogue path below.
                        schema = "tool_clarification"
                        tool_name = ""
                        required: list = []
                        if isinstance(best_tool, dict):
                            inner = best_tool.get("function") \
                                if isinstance(best_tool.get("function"), dict) \
                                else best_tool
                            tool_name = str(inner.get("name", ""))
                            params = inner.get("parameters")
                            if isinstance(params, dict):
                                rq = params.get("required")
                                required = list(rq) if isinstance(rq, list) else []
                        if veto_missing:
                            required = veto_missing
                        if veto_ok:
                            # Ask_followup only: a question no listed tool
                            # serves — the honest ask stays generic.
                            reason = "underspecified_query"
                            prompt = (
                                "Please clarify your request so a tool can "
                                "serve it: none of the available tools match "
                                "as stated."
                            )
                        else:
                            # Veto-abstain (guardfix wave, 2026-10-10): the
                            # argmax tool DID match and the veto named what
                            # the request is missing — serve that reason and
                            # list the parameters instead of the false
                            # "none of the available tools match as stated".
                            reason = veto_reason or (
                                "missing_arguments" if veto_missing
                                else "underspecified_query")
                            param_str = (
                                ", ".join(str(p) for p in required)
                                if required else "required parameters")
                            prompt = (f"Please provide {param_str} to use "
                                      f"{tool_name}.")
                        obj = {
                            "type": "tool_clarification",
                            "tool": tool_name,
                            "reason": reason,
                            "missing_parameters": required,
                            "clarification_prompt": prompt,
                        }
                    else:
                        schema = "no_tool_call"
                elif best_tool is None:
                    # The last leg: score >= THRESHOLD but NO tool matched.
                    # A catalogue-SHAPED text must never fall through to
                    # rules.extract: it would mint a tool_call named after
                    # the first "name" key found in the catalogue JSON (the
                    # A1 CALL, 2026-10-10). Catalogue-shaped = the "User
                    # request:" separator OR a "Tools available:"-style
                    # label in EITHER word order ("Tools available:" /
                    # "Available tools:" — the label-first header with a
                    # single-newline separator parses as not-a-catalogue
                    # and minted the CALL before the review-fixval probe,
                    # 2026-10-10) — together these cover every wellformed
                    # and malformed shape the review's Part 2 CALLed on
                    # (blank request, User prompt: label, trailing comma,
                    # [ before the array). A text with NEITHER marker is a
                    # bare <functioncall>-style payload — the historical
                    # rules.extract path — and must keep taking it.
                    _, sep, req = text.partition(tool_match._SPLIT)
                    from . import registry as _registry

                    if sep or re.search(
                            r"(?i)\b(?:tools\s+available:|available\s+tools\s*:)",
                            text):
                        if ("tool_clarification" in _registry.active()
                                and _is_ask_followup(req)):
                            schema = "tool_clarification"
                            obj = {
                                "type": "tool_clarification",
                                "tool": "",
                                "reason": "underspecified_query",
                                "missing_parameters": [],
                                "clarification_prompt": (
                                    "Please clarify your request so a tool "
                                    "can serve it: none of the available "
                                    "tools match as stated."
                                ),
                            }
                        else:
                            schema = "no_tool_call"
                            obj = {
                                "type": "no_tool_call",
                                "reason": "no matching tool",
                            }
                elif best_tool is not None:
                    _, req = tool_match.parse_catalogue_raw(text)
                    from . import registry, span_fill

                    # The function field may be malformed (a non-dict past
                    # the guard): degrade to the outer shape, never raise.
                    inner = (best_tool.get("function")
                             if isinstance(best_tool.get("function"), dict)
                             else best_tool)
                    tool_name = str(
                        best_tool.get("name") or inner.get("name", "")
                    )
                    filled, unfilled = span_fill.fill_arguments(best_tool, req)
                    if filled is not None and not unfilled:
                        # Span selection filled every required slot: emit
                        # the tool_call with verbatim arguments. Strict
                        # validation + repair below still apply.
                        obj = {
                            "type": "tool_call",
                            "name": tool_name,
                            "arguments": span_fill.arguments_json(filled),
                            "result": "",
                        }
                    elif filled is not None:
                        # Handled family, slots left over: the
                        # missing-arguments clarification.
                        if "tool_clarification" in registry.active():
                            schema = "tool_clarification"
                            param_str = (
                                ", ".join(unfilled) if unfilled
                                else "required parameters"
                            )
                            obj = {
                                "type": "tool_clarification",
                                "tool": tool_name,
                                "reason": "missing_arguments",
                                "missing_parameters": unfilled,
                                "clarification_prompt": (
                                    f"Please provide {param_str} to use {tool_name}."
                                ),
                            }
                        else:
                            schema = "no_tool_call"
                    else:
                        # Tool outside the span-fill families: the
                        # decision-layer fillability veto decides. The veto
                        # (not validate_fillability, which is load-bearing in
                        # the gate span) is the decision predicate: its
                        # number regex keeps unit-attached values ("100g",
                        # "19th") that the word-bounded gate regex drops,
                        # so a fully-specified request is not asked back for
                        # a number it already carries.
                        fillable, reason, missing = tool_match.fillability_veto(
                            best_tool, req)
                        if not fillable:
                            if "tool_clarification" in registry.active():
                                schema = "tool_clarification"
                                param_str = (
                                    ", ".join(missing) if missing
                                    else "required parameters"
                                )
                                obj = {
                                    "type": "tool_clarification",
                                    "tool": tool_name,
                                    "reason": reason,
                                    "missing_parameters": missing,
                                    "clarification_prompt": (
                                        f"Please provide {param_str} to use {tool_name}."
                                    ),
                                }
                            else:
                                schema = "no_tool_call"
                        else:
                            # Guard accepted; span_fill selected nothing and
                            # the deterministic fillability gate says the
                            # request supplies the required arguments. NEVER
                            # fall through to rules.extract here — it would
                            # emit the first tool NAME found anywhere in the
                            # catalogue text with empty arguments (the
                            # delete_database/get_weather P0). Honest
                            # outcomes only: emit the call with the slots
                            # span_fill found — here none, so `{}` is the
                            # honest minimum (values are selected from the
                            # request, never fabricated). The genuine
                            # clarification cases fail the gate above
                            # (missing numbers/URLs) or the veto on the
                            # abstain path; re-clarifying on `required`
                            # alone collapsed the BFCL calls set 89.1% ->
                            # 25.1% (119/183 fully-specified natural-language
                            # calls asked back for slots they had already
                            # supplied).
                            obj = {
                                "type": "tool_call",
                                "name": tool_name,
                                "arguments": "{}",
                                "result": "",
                            }
        if not obj:
            if schema == "log_event" and level_extractor is not False:
                ext_level = (level_extractor if level_extractor is not None
                             else _get_default_extractor("level", load_level_extractor))
                ext_comp = (component_extractor if component_extractor is not None
                            else _get_default_extractor("component",
                                                        load_component_extractor))
                ext_impl = (implicit_severity_extractor
                            if implicit_severity_extractor is not None
                            else _get_default_extractor(
                                "implicit", load_implicit_severity_extractor))
                obj = extract_learned.extract_log_event(
                    text,
                    level_extractor=ext_level,
                    component_extractor=ext_comp,
                    implicit_severity_extractor=ext_impl,
                )
            else:
                obj = rules.extract(text, schema)
                # Arm A (2026-10-07): learned prose title/body for the
                # conversational surfaces. The learned title overrides the
                # rules heuristic when it fires; body fills the optional
                # schema field. Never-raise: a failed load or extraction
                # keeps the rules result verbatim.
                if schema in ("issue", "pull_request"):
                    prose = extract_learned.extract_prose_title(
                        text,
                        model=(prose_extractor if prose_extractor is not None
                               else _get_default_extractor(
                                   "prose_title",
                                   extract_learned.load_prose_title_extractor)))
                    if prose is not None:
                        learned_title, learned_body = prose
                        # QW-B (2026-10-09): the learned title only wins
                        # when the rules title is degenerate or the learned
                        # one is not a prefix-truncation of it (the greedy
                        # span decoder collapses "PAY-2214: Checkout…"
                        # to "PAY", destroying a good rules title; the
                        # serving probe caught the shape). The learned
                        # body always applies.
                        if learned_title and _learned_title_wins(
                                obj.get("title", ""), learned_title, text):
                            obj["title"] = learned_title
                        if learned_body:
                            obj["body"] = learned_body
                    # Structural leading-line title (2026-10-10): when the
                    # text is shaped "short subject line, blank line, prose"
                    # the leading line IS the title — the shape carries the
                    # answer (the tickets/gh-prs instrument), so it outranks
                    # the rules chunk and the learned span alike. The
                    # learned body above still applies.
                    structural = _structural_leading_line_title(text)
                    if structural:
                        obj["title"] = structural

        errors = validate(obj, schema)
        while errors and retries < _MAX_REPAIRS:
            obj = repair(obj, errors, schema)
            errors = validate(obj, schema)
            retries += 1
        if errors:  # last resort: minimal strictly-valid template
            obj = template(schema)
            routed_by = "template"
        valid = not errors
    except Exception as exc:  # contract: never raise, degrade honestly
        _warn(f"classify emergency fallback ({type(exc).__name__}: {exc})")
        conf = 0.0
        try:
            schema = rules.route(text if isinstance(text, str) else str(text))
        except Exception:
            schema = ""
        if schema in ("tool_call", "tool_clarification"):
            # Architecture-amplifier law (2026-10-10): an internal error
            # must never become a POSITIVE action. template("tool_call")
            # is {"name": "", ...} — a served CALL for a nonexistent tool
            # (the malformed "required": [3] catalogue AttributeError used
            # to surface on /v1/guard as action CALL, tool ""). Degrade to
            # the honest abstention instead; tool_clarification is the
            # alternative when no_tool_call is not registered. Non-tool
            # schemas keep the template escape below.
            from . import registry as _registry

            if "no_tool_call" in _registry.active():
                schema = "no_tool_call"
                obj = {"type": "no_tool_call", "reason": "internal error"}
            elif "tool_clarification" in _registry.active():
                schema = "tool_clarification"
                obj = {
                    "type": "tool_clarification",
                    "tool": "",
                    "reason": "underspecified_query",
                    "missing_parameters": [],
                    "clarification_prompt": (
                        "Please clarify your request so a tool can serve "
                        "it: none of the available tools match as stated."
                    ),
                }
            else:
                obj = {}
        else:
            try:
                obj = template(schema)
            except Exception:
                obj = {}
        routed_by, valid = "template", False

    # Active learning pool logging hook (if enabled via arg or MIVIA_POOL_LOG=1)
    if log_pool or (log_pool is None and os.environ.get("MIVIA_POOL_LOG") == "1"):
        reason = None
        if conf < 0.70:
            reason = f"uncertain_route:{conf:.2f}"
        elif retries > 0 or not valid or routed_by == "template":
            reason = f"repair_needed:{retries}:valid={valid}"
        if reason:
            _log_to_pool(text, reason, pool_inbox=pool_inbox)
    meta = {
        "routed_by": routed_by,
        "valid": bool(valid),
        "retries": retries,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }
    return obj, meta


def structure(
    text: str,
    schema: str,
    level_extractor=None,
    component_extractor=None,
    implicit_severity_extractor=None,
) -> tuple[dict, dict]:
    """Force a declared output schema: extract -> validate -> repair (<= 2)
    -> template escape, NO routing (universal-structuring 1d).

    The caller declares the shape; the model assigns values. Unknown or
    inactive schema names raise ValueError (a caller programming error,
    not a classification input — the never-raise law covers text, not
    the schema contract). Extraction falls back to schema defaults and
    the repair ladder: prediction degrades to defaults, never
    fabrication.
    """
    from . import registry

    if schema not in registry.active():
        raise ValueError(
            f"unknown or inactive schema {schema!r}: register it first "
            "(registry.register_schema) or use a built-in")
    if schema == "tool_call":
        # tool_call extraction is special-cased through the guard ladder
        # (fillability veto / clarification, needs a tools catalogue): a
        # declared-schema call would silently emit no_tool_call records.
        # The honest surface for tool calls is /v1/guard.
        raise ValueError(
            "structure() does not declare tool_call: its extraction runs "
            "the guard ladder and needs a tools catalogue — use /v1/guard")
    return _classify_known_schema(
        text,
        schema,
        "declared",
        1.0,
        level_extractor=level_extractor,
        component_extractor=component_extractor,
        implicit_severity_extractor=implicit_severity_extractor,
    )


def classify(
    text: str,
    router=None,
    level_extractor=None,
    component_extractor=None,
    implicit_severity_extractor=None,
    log_pool: bool | None = None,
    pool_inbox: Path | str | None = None,
) -> tuple[dict, dict]:
    """Route -> extract -> validate -> repair (<= 2) -> template escape.

    Never raises for any str input. Returns (obj, meta) with
    meta = {routed_by: "model"|"rules"|"template", valid, retries,
    latency_ms}. ``valid`` is honest: True only when the emitted object
    passes the strict validator without the template escape (a template
    or emergency emission is a fallback, not earned validity).
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    schema, routed_by, conf = _route_with_confidence(router, text)
    return _classify_known_schema(
        text,
        schema,
        routed_by,
        conf,
        level_extractor=level_extractor,
        component_extractor=component_extractor,
        implicit_severity_extractor=implicit_severity_extractor,
        log_pool=log_pool,
        pool_inbox=pool_inbox,
    )


def triage_severity(text, anchors: list[str], matcher=None) -> dict:
    """Ordinal severity triage with the escalation policy (never raises).

    Grades ``text`` against the ordered ``anchors`` with the grade artifact's
    tier-set decode and resolves the set through
    ``tool_match.tier_decision``: a singleton acts on the tier, an adjacent
    pair acts with a "between i and i+1" annotation, and three or more tiers
    asks. The ask outcome carries a ``clarification`` record shaped like the
    agent loop's ``tool_clarification`` — merging it with
    ``agent_state.merge_step`` leaves a pending clarification a later turn
    can resolve.

    For the canonical 4-tier severity scale
    (``severity_head.CANONICAL_SEVERITY_ANCHORS``) the default path is the
    promoted ordinal head (``tool_match.load_severity_ordinal``), which decodes
    a single latent score through K-1 ordered cutpoints. A register router
    (``tool_match.load_register_gate``) may route business-register texts to
    the specialist head (``load_severity_specialist``); it degrades to the
    shipped ordinal head whenever the gate or specialist is unavailable.
    Arbitrary user anchors
    keep the cross-encoder anchor-alignment path (``load_grade_matcher``); an
    explicit ``matcher`` always wins, so callers can inject a scorer.

    ``matcher=None`` (default) loads the shipped artifact for the chosen path;
    when it is unavailable the honest outcome is an ask, never a guessed tier
    or an exception.
    """
    decision: dict = {"action": "ask", "tier_set": [], "options": [],
                      "clarification": None}
    try:
        if matcher is None:
            from . import tool_match
            from .severity_head import CANONICAL_SEVERITY_ANCHORS

            if list(anchors or []) == list(CANONICAL_SEVERITY_ANCHORS):
                matcher = None
                # Arm 3 de promotion (owner decision 2026-10-08): German
                # text routes to the bilingual de-mixed head BEFORE the
                # register gate — the en-only heads grade severe German
                # tickets as info (the serving-probe finding). The de head
                # is bilingual (en rows in its training mix), so a gate
                # over-route is harmless; a miss keeps the old failure.
                if tool_match.lang_is_de(text):
                    matcher = tool_match.load_severity_de()
                if matcher is None:
                    # Register router (reports/grade-register-router.md): the
                    # gate decides business vs canonical register; the
                    # specialist serves only routed rows and is never the
                    # default path.
                    gate, tau = tool_match.load_register_gate()
                    if gate is not None and tau is not None:
                        specialist = tool_match.load_severity_specialist()
                        if specialist is not None:
                            try:
                                # Resolve the positive column from classes_
                                # (house pattern) instead of assuming col 1.
                                classes = list(gate.classes_)
                                pos_col = (classes.index(1)
                                           if 1 in classes else None)
                                business = pos_col is not None and bool(
                                    gate.predict_proba([text])[0, pos_col]
                                    >= tau)
                            except Exception as exc:
                                _warn(f"register gate degraded to shipped "
                                      f"head ({type(exc).__name__}: {exc})")
                                business = False
                            matcher = specialist if business else None
                if matcher is None:
                    matcher = tool_match.load_severity_ordinal()
                if matcher is None:
                    # The fixed scale has no ordinal head shipped; degrade to
                    # the cross-encoder rather than guess on the wrong path.
                    matcher = tool_match.load_grade_matcher()
            else:
                matcher = tool_match.load_grade_matcher()
        if matcher is None:
            decision["clarification"] = _severity_clarification(anchors)
            return decision
        anchor, score, tier_set = matcher.grade(text, anchors, with_set=True)
        from . import tool_match

        best_idx = anchors.index(anchor) if anchor in anchors else None
        resolved = tool_match.tier_decision(tier_set, anchors, argmax_index=best_idx)
        resolved["score"] = float(score) if isinstance(score, (int, float)) else None
        resolved["clarification"] = None
        if resolved["action"] == "ask":
            resolved["clarification"] = _severity_clarification(
                resolved.get("options") or anchors)
        return resolved
    except Exception as exc:
        _warn(f"triage_severity degraded to ask ({type(exc).__name__}: {exc})")
        decision["clarification"] = _severity_clarification(anchors)
        return decision


def _severity_clarification(anchors: list[str]) -> dict:
    """agent_state-compatible ask record for an unresolved severity.

    Spec-compliant (TOOL_CLARIFICATION_SPEC): reason, missing_parameters
    and a real clarification_prompt; the anchor options are folded into
    the prompt text instead of a non-spec key.
    """
    options = [str(a) for a in (anchors or []) if str(a)]
    return {
        "type": "tool_clarification",
        "tool": "severity_triage",
        "reason": "ambiguous_intent",
        "missing_parameters": ["severity"],
        "clarification_prompt": (
            "Which severity fits this text? Options: "
            + (", ".join(options) if options else "none listed") + "."
        ),
    }


# Frame-shaped "at ..." continuation: the token after "at " must look
# like a frame — a dotted identifier or call whose pre-bracket part
# CONTAINS A LETTER (Java/Python frames always do, numeric-dotted IPs
# never), so prose like "at least one connection failed" stays its own
# line (2026-10-09 over-merge finding) and "at 10.0.0.1 connection
# refused" does not merge via \S+ backtracking onto the IP's dots
# (2026-10-10 A3b).
_FRAME_LINE = re.compile(r"^at (?=\S*[A-Za-z]\S*[.(])\S+[.(]")

# A previous chunk that shows traceback structure: bare "..." continuations
# and trailing exception lines ("ValueError: bad", "During handling ...")
# only merge into such a chunk, never into ordinary log lines. The
# ellipsis limb is ANCHORED to line start (2026-10-10 A3a): a trailing
# "INFO task finished..." is prose, only a line BEGINNING with "..."
# ("... 12 more") is traceback structure. Real Python frames indent the
# File " line, so the frame limb allows leading whitespace.
_TRACEBACK_MARK = re.compile(
    r"^Traceback|" + _FRAME_LINE.pattern + r"|^Caused by:|^[ \t]*File \"|^\.\.\.",
    re.M)
# The tail may also be BARE ("Exception: bad", "Error: x" with no
# identifier before it — 2026-10-10 A3c); "Errorbar:" is not a tail.
_EXCEPTION_TAIL = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b"
    r"|(?:Error|Exception):"
    r"|During handling of the|The above exception)")

# Strong atomic markers (2026-10-10 N5, narrow): when the WHOLE text
# routed to an atomic schema, any of these pins the text as one atomic
# entity regardless of how many of its lines individually look like log
# lines — a unified diff's @@ hunk header, a Python traceback's File "
# frame, a compiler path:line:col diagnostic. Only consulted AFTER the
# router's atomic verdict, never instead of it; the 0.35 carve-out
# itself is unchanged for marker-less text.
_STRONG_ATOMIC_MARK = re.compile(
    r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@"
    r'|^[ \t]*File "'
    r"|^\S+\.[A-Za-z0-9]+:\d+:\d+:",
    re.M,
)
# The @@ limb alone is structural PROOF of a unified diff (git diffs always
# carry hunk headers). A diff pasted beside build-log lines routes
# `diagnostic` whole-text, and the diagnostic extractor fills NOTHING for a
# diff — the N5 atomic take would emit a content-free record (measured
# 2026-10-10, review-fixval: file/line/message all empty where the pre-fix
# per-line path at least carried every line's text). The hunk mark upgrades
# the take to the `diff` schema, whose patch field preserves the payload.
_STRONG_DIFF_MARK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.M)


def _group_lines(raw_lines: list[str]) -> list[str]:
    """Group continuation lines (stack traces, tabs, 'at ...', 'Caused by:')
    with their parent log line."""
    grouped: list[str] = []
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        prev = grouped[-1] if grouped else ""
        if grouped and (
            raw.startswith((" ", "\t"))
            or line.startswith("Caused by:")
            or _FRAME_LINE.match(line)
            or (line.startswith("...") or _EXCEPTION_TAIL.match(line))
            and _TRACEBACK_MARK.search(prev)
        ):
            grouped[-1] += "\n" + line
        else:
            grouped.append(line)
    return grouped


def classify_dump(
    text: str,
    router=None,
    level_extractor=None,
    component_extractor=None,
    implicit_severity_extractor=None,
    log_pool: bool | None = None,
    pool_inbox: Path | str | None = None,
    template_cache=None,
) -> list[tuple[dict, dict]]:
    """Parse a multi-line text dump or stream containing mixed chat and logs.

    Handles conversational preambles (e.g. 'hey what is this...'), groups
    multi-line stack traces with their parent log line, and separates
    individual log events while preserving atomic multi-line entities
    (diffs, compiler diagnostics, multi-line issues).

    When ``template_cache`` (a ``logtemplate.TemplateCache``) is given,
    each grouped chunk probes the cache first: a hit maps the known
    template directly to its record and bypasses the encoder
    (``routed_by == "cache"``); a miss classifies normally and feeds the
    new template back into the cache. Without a cache the path is
    unchanged. The never-raise contract holds either way.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    raw_lines = text.strip().splitlines()
    if len(raw_lines) <= 1:
        return [
            classify(
                text,
                router=router,
                level_extractor=level_extractor,
                component_extractor=component_extractor,
                implicit_severity_extractor=implicit_severity_extractor,
                log_pool=log_pool,
                pool_inbox=pool_inbox,
            )
        ]

    if template_cache is not None:
        # All-hit fast path: every grouped chunk resolves from a known
        # template, so the whole dump bypasses the encoder, including the
        # dump-level atomicity probe below. This agrees with the slow path
        # whenever it applies: only log-event templates are ever learned,
        # and a stream of all-log chunks takes the per-chunk loop there
        # too (the atomic-take branch needs mostly non-log lines, which
        # can never all hit). A single grouped chunk always takes the
        # slow path so atomic entities keep their whole-text verdict.
        grouped_probe = _group_lines(raw_lines)
        if len(grouped_probe) > 1:
            probed = [_cache_probe(template_cache, chunk)
                      for chunk in grouped_probe]
            # The cache serves log lines only: a non-log schema (possible
            # only from a hand-populated cache) is a miss, so atomic
            # entities and chatter always keep the slow-path verdict.
            if all(hit_schema == "log_event" for _, hit_schema in probed):
                return [
                    _classify_known_schema(
                        chunk,
                        hit_schema,
                        "cache",
                        level_extractor=level_extractor,
                        component_extractor=component_extractor,
                        implicit_severity_extractor=implicit_severity_extractor,
                        log_pool=log_pool,
                        pool_inbox=pool_inbox,
                    )
                    for chunk, (_, hit_schema) in zip(grouped_probe, probed)
                ]

    # Check if this is an atomic multi-line entity rather than a dump of separate logs
    overall_schema, _, _ = _route_with_confidence(router, text)
    if overall_schema in _ATOMIC_SCHEMAS:
        non_empty = [ln.strip() for ln in raw_lines if ln.strip()]
        # One forward pass per line (the ONNX router forwards one row per
        # session.run — review A2: co-batching is NOT inert on the int8
        # graph); the skops fallback loops exactly as before. Semantics
        # identical — same per-line argmax, same early-exits below.
        if hasattr(router, "classes") and hasattr(router, "_logits"):
            batch_preds = router.predict(non_empty)
            log_line_count = sum(1 for p in batch_preds if p == "log_event")
        else:
            log_line_count = sum(
                1 for ln in non_empty
                if _route_with_confidence(router, ln)[0] == "log_event")
        log_line_ratio = log_line_count / max(1, len(non_empty))
        strong_atomic = _STRONG_ATOMIC_MARK.search(text)
        if (log_line_count < 2 or log_line_ratio < 0.35
                or strong_atomic):
            from . import registry as _registry
            if (strong_atomic and _STRONG_DIFF_MARK.search(text)
                    and "diff" in _registry.active()):
                # The @@ hunk proves a unified diff; the router's whole-text
                # verdict for a diff-plus-log paste is diagnostic and the
                # diagnostic extractor fills nothing for a diff — take the
                # diff schema so the atomic record carries the patch.
                return [
                    _classify_known_schema(
                        text,
                        "diff",
                        "structure",
                        level_extractor=level_extractor,
                        component_extractor=component_extractor,
                        implicit_severity_extractor=implicit_severity_extractor,
                        log_pool=log_pool,
                        pool_inbox=pool_inbox,
                    )
                ]
            return [
                classify(
                    text,
                    router=router,
                    level_extractor=level_extractor,
                    component_extractor=component_extractor,
                    implicit_severity_extractor=implicit_severity_extractor,
                    log_pool=log_pool,
                    pool_inbox=pool_inbox,
                )
            ]


    # Group log lines with continuation lines (stack traces, tabs, 'at ...', 'Caused by:')
    grouped = _group_lines(raw_lines)

    results: list[tuple[dict, dict]] = []
    for chunk in grouped:
        if template_cache is not None:
            cached_template, cached_schema = _cache_lookup(template_cache, chunk)
            if cached_schema == "log_event":
                # Hit: the template is known, so the encoder is bypassed
                # and the record is built straight from the cached schema.
                results.append(
                    _classify_known_schema(
                        chunk,
                        cached_schema,
                        "cache",
                        level_extractor=level_extractor,
                        component_extractor=component_extractor,
                        implicit_severity_extractor=implicit_severity_extractor,
                        log_pool=log_pool,
                        pool_inbox=pool_inbox,
                    )
                )
                continue
        else:
            cached_template = ""
        schema, routed_by, conf = _route_with_confidence(router, chunk)
        # A chunk the router scores to ANY schema is content and is never
        # dropped (2026-10-10 N2: the old gate honored the router verdict
        # for log_event only, silently dropping e.g. a German line the
        # router confidently routed issue at 0.981 with zero rules
        # markers). Only router-`text` with no rules marker is chatter —
        # that is the actual conversational case.
        is_content = schema != "text" or max(
            rules.route_scores(chunk).values(), default=0
        ) > 0
        if is_content or len(grouped) == 1:
            record = _classify_known_schema(
                chunk,
                schema,
                routed_by,
                conf,
                level_extractor=level_extractor,
                component_extractor=component_extractor,
                implicit_severity_extractor=implicit_severity_extractor,
                log_pool=log_pool,
                pool_inbox=pool_inbox,
            )
            results.append(record)
            if template_cache is not None and record[0].get("type") == "log_event":
                # Miss: feed the new template back into the cache. Only
                # log-event templates are learned, so the cache only ever
                # serves log lines and bypass stays in scope.
                _cache_learn(template_cache, cached_template, "log_event")

    if not results:
        results.append(
            classify(
                text,
                router=router,
                level_extractor=level_extractor,
                component_extractor=component_extractor,
                implicit_severity_extractor=implicit_severity_extractor,
                log_pool=log_pool,
                pool_inbox=pool_inbox,
            )
        )
    return results

