# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Learned log field extraction (token classification over hashed features).

Complement to ``rules.py``: deterministic regex extraction cannot learn new
log formats, while this module extracts structured fields using small,
learned scikit-learn models (FeatureHasher + LogisticRegression, skops
persistence, CPU-only).

Primary entrypoints:
- ``load_level_extractor(path)``: sha-verified loader for explicit levels.
- ``load_component_extractor(path)``: sha-verified loader for components.
- ``load_implicit_severity_extractor(path)``: sha-verified loader for implicit severities.
- ``extract_level(text, model)``: extracts the explicit severity level token.
- ``extract_component(text, model)``: extracts the log component name.
- ``extract_implicit_severity(text, model)``: infers implicit syslog severity.
- ``extract_log_event(text, ...)``: full extraction for log_event with learned
  models and rules fallback.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import integrity, rules
from .schemas import LEVEL_ALIASES, LEVELS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEVEL_PATH = ROOT / "models" / "level_extractor.skops"
DEFAULT_COMPONENT_PATH = ROOT / "models" / "component_extractor.skops"
DEFAULT_IMPLICIT_PATH = ROOT / "models" / "implicit_severity.skops"


def _warn(msg: str) -> None:
    print(f"[mivia-small-json] {msg}", file=sys.stderr)


def canon(level: str) -> str:
    """Canonicalize a severity string to its OTel enum spelling."""
    lv = (level or "").strip().upper()
    return LEVEL_ALIASES.get(lv, lv)


def shape(tok: str) -> str:
    """Character shape representation (X for upper, x for lower, d for digit)."""
    return re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", re.sub(r"\d", "d", tok)))


# --- Token-level explicit level extraction ---------------------------------

LEVEL_TOKEN_RE = re.compile(r"\w+")


def level_tokenize(line: str) -> list[tuple[str, int]]:
    return [(m.group(0), m.start()) for m in LEVEL_TOKEN_RE.finditer(line)]


def level_token_features(line: str, tokens: list[tuple[str, int]], i: int) -> list[str]:
    tok = tokens[i][0]
    t = tok.lower()
    feats = {"tok=" + t, "shape=" + shape(tok), "len=" + str(min(len(t), 8))}
    start, end = tokens[i][1], tokens[i][1] + len(tok)
    if start > 0:
        feats.add("before=" + line[start - 1])
    if end < len(line):
        feats.add("after=" + line[end])
    feats.update("c=" + t[j : j + 3] for j in range(max(0, len(t) - 2)))
    feats.update("c4=" + t[j : j + 4] for j in range(max(0, len(t) - 3)))
    if i > 0:
        feats.add("prev=" + tokens[i - 1][0].lower())
        if i > 1:
            feats.add("prev2=" + tokens[i - 2][0].lower())
    if i + 1 < len(tokens):
        feats.add("next=" + tokens[i + 1][0].lower())
        if i + 2 < len(tokens):
            feats.add("next2=" + tokens[i + 2][0].lower())
    return sorted(feats)


# --- Token-level component extraction --------------------------------------

COMP_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")


def comp_tokenize(line: str) -> list[tuple[str, int]]:
    return [(m.group(0), m.start()) for m in COMP_TOKEN_RE.finditer(line)]


def comp_token_features(line: str, tokens: list[tuple[str, int]], i: int) -> list[str]:
    tok = tokens[i][0]
    t = tok.lower()
    feats = {"tok=" + t, "shape=" + shape(tok), "len=" + str(min(len(t), 12))}
    start, end = tokens[i][1], tokens[i][1] + len(tok)
    if start > 0:
        feats.add("before=" + line[start - 1])
    if end < len(line):
        feats.add("after=" + line[end])
    feats.update("c=" + t[j : j + 3] for j in range(max(0, len(t) - 2)))
    if i > 0:
        feats.add("prev=" + tokens[i - 1][0].lower())
        if i > 1:
            feats.add("prev2=" + tokens[i - 2][0].lower())
    if i + 1 < len(tokens):
        feats.add("next=" + tokens[i + 1][0].lower())
        if i + 2 < len(tokens):
            feats.add("next2=" + tokens[i + 2][0].lower())
    return sorted(feats)


# --- Message-level implicit severity extraction ----------------------------

IMPLICIT_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")


def implicit_message_features(text: str) -> list[str]:
    low = text.lower()
    tokens = IMPLICIT_TOKEN_RE.findall(low)
    feats = {"w=" + t for t in tokens if len(t) > 1}
    for i in range(len(tokens) - 1):
        feats.add(f"bi={tokens[i]}_{tokens[i+1]}")
    for i in range(len(tokens) - 2):
        feats.add(f"tri={tokens[i]}_{tokens[i+1]}_{tokens[i+2]}")
    for i in range(len(low) - 3):
        feats.add("c4=" + low[i : i + 4])
    return sorted(feats)


# --- Model loaders (sha256-verified skops) ---------------------------------

def _load_skops_verified(p: Path, name: str):
    sidecar = p.parent / (p.name + ".sha256")
    if not sidecar.is_file():
        sidecar = p.parent / (p.stem + ".sha256")
    try:
        if not p.is_file():
            return None
        if not sidecar.is_file():
            _warn(f"sha256 sidecar missing: {sidecar}; refusing unverified load")
            return None
        expected, problem = integrity.expected_digest(sidecar)
        if problem == "malformed":
            _warn(f"malformed sha256 sidecar: {sidecar}; refusing unverified load")
            return None
        digest = integrity.sha256_file(p)
        if not integrity.digests_match(digest, expected):
            _warn(
                f"sha256 mismatch for {p.name} "
                f"(sidecar {expected[:12]}..., actual {digest[:12]}...); "
                "refusing unverified load"
            )
            return None
        import skops.io as sio

        return sio.load(p, trusted=sio.get_untrusted_types(file=p))
    except Exception as exc:
        _warn(f"{name} load failed ({type(exc).__name__}: {exc})")
        return None


def load_level_extractor(path: Path | str | None = None):
    """Sha256-verified loader for the learned level extractor (never raises)."""
    p = Path(path) if path is not None else DEFAULT_LEVEL_PATH
    return _load_skops_verified(p, "level extractor")


def load_component_extractor(path: Path | str | None = None):
    """Sha256-verified loader for the learned component extractor (never raises)."""
    p = Path(path) if path is not None else DEFAULT_COMPONENT_PATH
    return _load_skops_verified(p, "component extractor")


def load_implicit_severity_extractor(path: Path | str | None = None):
    """Sha256-verified loader for implicit syslog severity (never raises)."""
    p = Path(path) if path is not None else DEFAULT_IMPLICIT_PATH
    return _load_skops_verified(p, "implicit severity extractor")


# --- Extractors ------------------------------------------------------------


def _positive_column(model) -> int | None:
    """Column of the positive class in a binary ``predict_proba`` output.

    Resolved from the model's own ``classes_``, so a model whose positive
    class is not at index 1 — or that is not binary at all — is refused
    instead of silently mis-read.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    classes = list(classes)
    if len(classes) != 2:
        return None  # multi-class models have no single positive column
    return classes.index(1) if 1 in classes else 1


def extract_level(text: str, model=None, threshold: float = 0.35) -> str:
    """Extract and canonicalize explicit log level using token classifier."""
    if model is None or not text:
        return ""
    try:
        tokens = level_tokenize(text)
        if not tokens:
            return ""
        feats = [level_token_features(text, tokens, i) for i in range(len(tokens))]
        if hasattr(model, "predict_proba"):
            col = _positive_column(model)
            if col is None:
                _warn(f"extract_level: model classes "
                      f"{getattr(model, 'classes_', None)!r} are not a binary "
                      "positive/negative pair; skipping")
                return ""
            proba = model.predict_proba(feats)[:, col]
            best = max(range(len(tokens)), key=lambda i: proba[i])
            if proba[best] >= threshold:
                c = canon(tokens[best][0])
                return c if c in LEVELS else ""
        return ""
    except Exception as exc:
        _warn(f"extract_level failed ({type(exc).__name__}: {exc})")
        return ""


def extract_component(text: str, model=None, threshold: float = 0.20) -> str:
    """Extract log component name using the learned token classifier.

    Rejects PIDs (pure digits) and timestamps.
    """
    if model is None or not text:
        return ""
    try:
        tokens = comp_tokenize(text)
        if not tokens:
            return ""
        feats = [comp_token_features(text, tokens, i) for i in range(len(tokens))]
        if hasattr(model, "predict_proba"):
            col = _positive_column(model)
            if col is None:
                _warn(f"extract_component: model classes "
                      f"{getattr(model, 'classes_', None)!r} are not a binary "
                      "positive/negative pair; skipping")
                return ""
            proba = model.predict_proba(feats)[:, col]
            best = max(range(len(tokens)), key=lambda i: proba[i])
            if proba[best] >= threshold:
                cand = tokens[best][0]
                if cand.isdigit():
                    return ""
                if re.match(r"^\d{2,4}[-./]\d{2}[-./]\d{2,4}$|^\d{2}:\d{2}:\d{2}", cand):
                    return ""
                return cand
        return ""
    except Exception as exc:
        _warn(f"extract_component failed ({type(exc).__name__}: {exc})")
        return ""


def extract_implicit_severity(text: str, model=None, threshold: float = 0.40) -> str:
    """Infer implicit severity for syslog/daemon messages without explicit tags."""
    if model is None or not text:
        return ""
    try:
        feats = [implicit_message_features(text)]
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(feats)[0]
            best_idx = probas.argmax()
            if probas[best_idx] >= threshold:
                pred_cls = model.classes_[best_idx]
                c = canon(pred_cls)
                return c if c in LEVELS else ""
        return ""
    except Exception as exc:
        _warn(f"extract_implicit_severity failed ({type(exc).__name__}: {exc})")
        return ""


def _extends_component(curr: str, pred: str) -> bool:
    """True when ``pred`` may replace the rules component ``curr``.

    The learned component must extend the rules component at a word
    boundary ("sys" -> "sys.log" or "sys" -> "syslog" is an extension of
    "sys" only when the next char is not a word char). A merely contained
    candidate ("sys" inside "syslogd") must not shrink a more specific
    rules extraction (review 2026-10-09, F9d).
    """
    cl, pl = curr.lower(), pred.lower()
    if pl == cl:
        return False
    if cl.startswith(pl):
        # pred is a boundary prefix of curr: curr is the more specific
        # form ("syslogd" vs contained "sys"), keep curr.
        return False
    if pl.startswith(cl) and (len(pl) == len(cl)
                              or not pl[len(cl)].isalnum()):
        return True  # pred extends curr on a word boundary
    return False


def extract_log_event(
    text: str,
    level_extractor=None,
    component_extractor=None,
    implicit_severity_extractor=None,
) -> dict:
    """Extract log_event fields with learned models and rules fallback."""
    res = rules._extract_log(text)

    # 1. Level extraction
    pred_level = extract_level(text, level_extractor) if level_extractor is not None else ""
    if not pred_level and implicit_severity_extractor is not None:
        msg_text = res.get("message") or text
        pred_level = extract_implicit_severity(msg_text, implicit_severity_extractor)
    # An implicit-severity guess must never overwrite an explicit level the
    # rules layer already extracted (review 2026-10-09, F1).
    if pred_level and not res.get("level"):
        res["level"] = pred_level

    # 2. Component extraction
    if component_extractor is not None:
        pred_comp = extract_component(text, component_extractor)
        if pred_comp:
            curr = res.get("component", "")
            if not curr or curr.isdigit() or _extends_component(curr, pred_comp):
                res["component"] = pred_comp

    # Hygiene: component should never be pure digits (a PID)
    if res.get("component") and res["component"].isdigit():
        res["component"] = ""

    # Clean up message if component prefix is duplicated
    comp = res.get("component", "")
    if comp and res.get("message", "").lower().startswith(comp.lower()):
        res["message"] = re.sub(
            rf"^{re.escape(comp)}\s*[:=]\s*", "", res["message"], flags=re.I
        ).strip()

    return res


# --- Prose title/body extraction (issue & pull_request, Arm A 2026-10-07) ---

# Same token vocabulary as the level extractor: \w+ keeps German umlauts
# and internal digits, which is what the ticket/PR surfaces are made of.
PROSE_TOKEN_RE = re.compile(r"\w+")
DEFAULT_PROSE_TITLE_PATH = ROOT / "models" / "prose-title-extractor.skops"
PROSE_TITLE_THRESHOLD = 0.5
# A title span longer than this fraction of the text is not a title.
_PROSE_TITLE_MAX_FRACTION = 0.6


def prose_token_features(tokens: list[str], idx: int) -> list[str]:
    """Features for token ``idx`` of a prose text (ticket subject+body).

    House pattern (level_token_features) plus prose-position features: the
    model must learn where the title stops, so position bucket, line start
    and paragraph boundary are the load-bearing additions.
    """
    t = tokens[idx]
    low = t.lower()
    n = len(tokens)
    feats = {
        "tok=" + low,
        "shape=" + shape(t)[:8],
        f"len={min(len(t), 12)}",
        "posbucket=" + str(idx * 10 // max(n, 1)),
        "posinv=" + str((n - 1 - idx) * 10 // max(n, 1)),
    }
    if idx == 0:
        feats.add("is_first")
    if idx == n - 1:
        feats.add("is_last")
    for off, name in ((1, "next"), (2, "next2")):
        if idx + off < n:
            feats.add(f"{name}=" + tokens[idx + off].lower())
    for off, name in ((1, "prev"), (2, "prev2")):
        if idx - off >= 0:
            feats.add(f"{name}=" + tokens[idx - off].lower())
    for i in range(len(low) - 2):
        feats.add("c=" + low[i : i + 3])
    return sorted(feats)


def prose_title_features(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """(per-token feature lists, token char spans) for a prose text."""
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    feats: list[list[str]] = []
    for m in PROSE_TOKEN_RE.finditer(text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    for i in range(len(tokens)):
        feats.append(prose_token_features(tokens, i))
    return feats, spans


def load_prose_title_extractor(path: Path | str | None = None):
    """Sha256-verified loader for the learned prose title extractor (never raises)."""
    p = Path(path) if path is not None else DEFAULT_PROSE_TITLE_PATH
    return _load_skops_verified(p, "prose title extractor")


def extract_prose_title(
    text: str, model=None, threshold: float = PROSE_TITLE_THRESHOLD
) -> tuple[str, str] | None:
    """(title, body) from conversational prose, or None (never raises).

    Token-level binary classifier decides title membership; the title is
    the longest contiguous above-threshold run (started at the first
    positive token), the body the remainder after it. Falls back to None
    so the caller keeps the rules heuristic untouched.
    """
    try:
        if model is None:
            model = load_prose_title_extractor()
        if model is None:
            return None
        feats, spans = prose_title_features(text)
        if not feats or len(spans) < 4:
            return None
        proba = model.predict_proba(feats)
        col = _positive_column(model)
        if col is None:
            return None
        pos = [p[col] for p in proba]
        # Longest contiguous run above threshold, anchored at the best token.
        best = max(range(len(pos)), key=lambda i: pos[i])
        if pos[best] < threshold:
            return None
        lo = best
        while lo > 0 and pos[lo - 1] >= threshold:
            lo -= 1
        hi = best
        while hi + 1 < len(pos) and pos[hi + 1] >= threshold:
            hi += 1
        # Extend over short below-threshold gaps? No: the longest clean run
        # is the conservative title; the rules fallback still covers the rest.
        span_chars = spans[hi][1] - spans[lo][0]
        if span_chars > _PROSE_TITLE_MAX_FRACTION * len(text):
            return None
        title = text[spans[lo][0] : spans[hi][1]].strip()
        if not title:
            return None
        preamble = text[: spans[lo][0]].strip()
        body = (preamble + " " + text[spans[hi][1] :]).strip()
        return title, body
    except Exception as exc:
        _warn(f"prose title extraction failed ({type(exc).__name__}: {exc})")
        return None
